cmake_minimum_required(VERSION 3.20)
project(automasker_frontend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# 外部依存
#   - GLFW: ウィンドウ/入力
#   - OpenGL: テクスチャ表示
#   - Dear ImGui: UI (external/imgui/ にソース一式が入っていることを前提)
#   - cpp-httplib: シングルヘッダHTTPクライアント (external/httplib.h)
#   - stb_image, stb_image_write: 画像ロード/保存 (external/stb/)
# いずれも fetch せず、ユーザーが事前に external/ へ配置する想定.
# ---------------------------------------------------------------------------

find_package(OpenGL REQUIRED)

# GLFW (システム / サブモジュール を想定)
find_package(glfw3 REQUIRED)

# ImGui
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
if(NOT EXISTS ${IMGUI_DIR}/imgui.cpp)
    message(FATAL_ERROR "Dear ImGui が見つかりません.\n"
                        "    git clone https://github.com/ocornut/imgui external/imgui")
endif()
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

add_executable(automasker_frontend
    src/main.cpp
    ${IMGUI_SOURCES}
)
target_include_directories(automasker_frontend PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CURRENT_SOURCE_DIR}/external/stb
)
target_link_libraries(automasker_frontend PRIVATE
    glfw
    OpenGL::GL
)

if(WIN32)
    target_link_libraries(automasker_frontend PRIVATE ws2_32)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(automasker_frontend PRIVATE pthread)
endif()

message(STATUS "automasker_frontend configured.")
