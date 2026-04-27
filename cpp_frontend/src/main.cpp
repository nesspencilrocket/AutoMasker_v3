// AutoMasker C++ / Dear ImGui フロントエンド.
// Python 側の HTTP 推論サーバ (automasker.server) と通信してプレビューを表示する.
// - 画像ロード, プロンプト入力, スライダーで閾値調整
// - 推論サーバを叩いて返ってきた 2値マスクを半透明オーバーレイ
// - ImGui + GLFW + OpenGL 3 のみ使う最小構成
//
// コンパイル方法:
//   cd cpp_frontend
//   git clone https://github.com/ocornut/imgui external/imgui
//   wget https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h -P external/
//   # stb_image も同様に external/stb に入れる
//   cmake -B build -S . && cmake --build build

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include "httplib.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Base64 デコーダ (標準ライブラリに無いので最小実装)
// ---------------------------------------------------------------------------
static std::vector<uint8_t> base64_decode(const std::string& in) {
    static const int T[256] = { /* init lazily */ 0 };
    static bool init = false;
    static int Tbl[256];
    if (!init) {
        for (int i = 0; i < 256; ++i) Tbl[i] = -1;
        const char* A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) Tbl[(unsigned char)A[i]] = i;
        init = true;
    }
    std::vector<uint8_t> out;
    int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (Tbl[c] < 0) continue;
        val = (val << 6) | Tbl[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(uint8_t((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// テクスチャ管理
// ---------------------------------------------------------------------------
struct Texture {
    GLuint id = 0;
    int w = 0, h = 0;
    bool valid() const { return id != 0; }
};

static void upload_rgb_texture(Texture& tex, const uint8_t* rgb, int w, int h) {
    if (!tex.id) glGenTextures(1, &tex.id);
    glBindTexture(GL_TEXTURE_2D, tex.id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgb);
    tex.w = w; tex.h = h;
}

// RGB 画像とマスクをオーバーレイして RGBA テクスチャを作る
static std::vector<uint8_t> overlay_rgba(const std::vector<uint8_t>& rgb,
                                          const std::vector<uint8_t>& mask,
                                          int w, int h, float alpha = 0.5f) {
    std::vector<uint8_t> out(w * h * 4);
    for (int i = 0; i < w * h; ++i) {
        uint8_t r = rgb[i * 3 + 0];
        uint8_t g = rgb[i * 3 + 1];
        uint8_t b = rgb[i * 3 + 2];
        uint8_t m = mask.empty() ? 0 : mask[i];
        if (m > 127) {
            r = uint8_t(r * (1 - alpha) + 255 * alpha);
            g = uint8_t(g * (1 - alpha) + 64 * alpha);
            b = uint8_t(b * (1 - alpha) + 64 * alpha);
        }
        out[i * 4 + 0] = r;
        out[i * 4 + 1] = g;
        out[i * 4 + 2] = b;
        out[i * 4 + 3] = 255;
    }
    return out;
}

// ---------------------------------------------------------------------------
// HTTP クライアントのラッパ (別スレッドで実行)
// ---------------------------------------------------------------------------
struct InferResult {
    std::vector<uint8_t> mask;   // width*height バイトのグレースケール
    int box_count = 0;
    double elapsed_ms = 0.0;
    std::string error;
};

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

// レスポンス JSON から mask_png_base64 の中身を雑に取り出す (小さな依存対策)
static std::string extract_field(const std::string& json, const std::string& key) {
    auto p = json.find("\"" + key + "\"");
    if (p == std::string::npos) return "";
    p = json.find(':', p);
    if (p == std::string::npos) return "";
    p = json.find('"', p);
    if (p == std::string::npos) return "";
    auto q = json.find('"', p + 1);
    if (q == std::string::npos) return "";
    return json.substr(p + 1, q - p - 1);
}

static int count_occurrences(const std::string& s, const std::string& sub) {
    int c = 0;
    for (size_t p = 0; (p = s.find(sub, p)) != std::string::npos; p += sub.size())
        ++c;
    return c;
}

static InferResult call_infer(const std::string& host, int port,
                              const std::string& token,
                              const std::string& image_path,
                              const std::string& prompt,
                              float box_th, float text_th,
                              int* out_w, int* out_h) {
    InferResult r;
    auto t0 = std::chrono::steady_clock::now();

    httplib::Client cli(host, port);
    cli.set_read_timeout(120, 0);
    if (!token.empty()) {
        cli.set_default_headers({{"Authorization", "Bearer " + token}});
    }

    auto bytes = read_file(image_path);
    if (bytes.empty()) {
        r.error = "画像ファイルが読めません: " + image_path;
        return r;
    }

    httplib::MultipartFormDataItems items = {
        {"image", std::string((char*)bytes.data(), bytes.size()),
         "input.png", "image/png"},
        {"prompt", prompt, "", ""},
        {"box_threshold", std::to_string(box_th), "", ""},
        {"text_threshold", std::to_string(text_th), "", ""},
    };

    auto res = cli.Post("/infer", items);
    if (!res) {
        r.error = "HTTP request failed";
        return r;
    }
    if (res->status != 200) {
        r.error = "HTTP " + std::to_string(res->status) + ": " + res->body;
        return r;
    }

    const std::string& body = res->body;
    std::string b64 = extract_field(body, "mask_png_base64");
    r.box_count = count_occurrences(body, "\"x1\"");

    if (!b64.empty()) {
        auto png = base64_decode(b64);
        int w, h, ch;
        uint8_t* decoded = stbi_load_from_memory(png.data(), (int)png.size(),
                                                 &w, &h, &ch, 1);
        if (decoded) {
            r.mask.assign(decoded, decoded + (w * h));
            stbi_image_free(decoded);
            if (out_w) *out_w = w;
            if (out_h) *out_h = h;
        } else {
            r.error = "mask PNG decode failed";
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

// ---------------------------------------------------------------------------
// アプリ状態
// ---------------------------------------------------------------------------
struct App {
    std::string host = "127.0.0.1";
    int port = 8765;
    std::string token;   // Bearer トークン. 環境変数 AUTOMASKER_TOKEN から読む.

    std::string image_path;
    std::vector<uint8_t> image_rgb;
    int image_w = 0, image_h = 0;
    Texture tex;

    std::vector<uint8_t> mask;
    std::atomic<bool> inferring{false};
    std::string last_error;
    double last_ms = 0.0;
    int last_box_count = 0;

    char prompt[512] = "person . tripod";
    float box_threshold = 0.30f;
    float text_threshold = 0.25f;
    float overlay_alpha = 0.5f;
};

static void load_image_into(App& app, const std::string& path) {
    int w, h, ch;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &ch, 3);
    if (!data) {
        app.last_error = "画像を読み込めませんでした: " + path;
        return;
    }
    app.image_path = path;
    app.image_w = w; app.image_h = h;
    app.image_rgb.assign(data, data + w * h * 3);
    stbi_image_free(data);
    app.mask.clear();
    auto rgba = overlay_rgba(app.image_rgb, app.mask, w, h, 0.0f);
    upload_rgb_texture(app.tex, rgba.data(), w, h);
}

static void do_inference(App& app) {
    if (app.image_path.empty() || app.inferring.load()) return;
    app.inferring = true;
    std::thread([&app] {
        int mw, mh;
        InferResult r = call_infer(app.host, app.port, app.token,
                                   app.image_path,
                                   app.prompt, app.box_threshold, app.text_threshold,
                                   &mw, &mh);
        app.last_ms = r.elapsed_ms;
        app.last_box_count = r.box_count;
        if (!r.error.empty()) {
            app.last_error = r.error;
        } else {
            app.last_error.clear();
            app.mask = r.mask;
            // メインスレッドでのテクスチャ更新が望ましいが、OpenGL context は
            // この例では share されていないので、ポインタだけ差し替えて次のフレームで再アップロード.
        }
        app.inferring = false;
    }).detach();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (!glfwInit()) return 1;
    GLFWwindow* win = glfwCreateWindow(1400, 900, "AutoMasker (C++ frontend)", nullptr, nullptr);
    if (!win) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    App app;
    if (const char* tok = std::getenv("AUTOMASKER_TOKEN")) {
        app.token = tok;
    }
    if (argc > 1) load_image_into(app, argv[1]);
    bool need_reupload = false;

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        // マスクが更新されたら overlay を再構成
        if (!app.inferring && !app.mask.empty() &&
            (int)app.mask.size() == app.image_w * app.image_h) {
            auto rgba = overlay_rgba(app.image_rgb, app.mask,
                                     app.image_w, app.image_h, app.overlay_alpha);
            upload_rgb_texture(app.tex, rgba.data(), app.image_w, app.image_h);
            need_reupload = false;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // --- 左ペイン: コントロール ---------------------------------
        ImGui::SetNextWindowPos({0, 0}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({400, 900}, ImGuiCond_FirstUseEver);
        ImGui::Begin("AutoMasker");
        ImGui::TextDisabled("server: %s:%d", app.host.c_str(), app.port);
        if (ImGui::Button("Load image…")) {
            // file dialog は実装せず、argv[1] で指定する運用. ここは placeholder.
            ImGui::OpenPopup("LoadInfo");
        }
        if (ImGui::BeginPopup("LoadInfo")) {
            ImGui::Text("画像は起動時に argv[1] で指定してください");
            ImGui::EndPopup();
        }
        if (!app.image_path.empty()) {
            ImGui::TextWrapped("image: %s", app.image_path.c_str());
            ImGui::Text("size: %d x %d", app.image_w, app.image_h);
        }
        ImGui::Separator();
        ImGui::InputText("Prompt", app.prompt, sizeof(app.prompt));
        ImGui::SliderFloat("Box threshold",  &app.box_threshold,  0.05f, 0.95f);
        ImGui::SliderFloat("Text threshold", &app.text_threshold, 0.05f, 0.95f);
        ImGui::SliderFloat("Overlay alpha",  &app.overlay_alpha,  0.0f, 1.0f);

        if (ImGui::Button(app.inferring.load() ? "Running..." : "Run inference",
                          ImVec2(-1, 40))
            && !app.inferring.load()) {
            do_inference(app);
        }
        if (app.last_ms > 0) {
            ImGui::Text("last inference: %.1f ms (%d boxes)",
                        app.last_ms, app.last_box_count);
        }
        if (!app.last_error.empty()) {
            ImGui::TextColored({1, 0.3f, 0.3f, 1}, "error: %s", app.last_error.c_str());
        }
        ImGui::End();

        // --- 右ペイン: プレビュー -----------------------------------
        ImGui::SetNextWindowPos({400, 0}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({1000, 900}, ImGuiCond_FirstUseEver);
        ImGui::Begin("Preview");
        if (app.tex.valid()) {
            ImVec2 avail = ImGui::GetContentRegionAvail();
            float aspect = float(app.tex.w) / float(app.tex.h);
            float w = avail.x, h = w / aspect;
            if (h > avail.y) { h = avail.y; w = h * aspect; }
            ImGui::Image((ImTextureID)(intptr_t)app.tex.id, ImVec2(w, h));
        } else {
            ImGui::TextDisabled("No image. Launch with: automasker_frontend <image.jpg>");
        }
        ImGui::End();

        ImGui::Render();
        int dw, dh;
        glfwGetFramebufferSize(win, &dw, &dh);
        glViewport(0, 0, dw, dh);
        glClearColor(0.10f, 0.11f, 0.13f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
