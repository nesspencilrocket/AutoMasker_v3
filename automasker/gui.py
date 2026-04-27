"""
PySide6 GUI.

- 左ペイン: 画像サムネイル一覧 (フォルダ読み込み)
- 中央: プレビュー (オリジナル / マスクオーバーレイ切替)
- 右ペイン: プロンプト入力 + 閾値スライダー + 実行ボタン

プレビューは 1枚選択して "Preview" を押すと推論 → 表示。
バッチは "Run Batch" で全画像を処理し masks/ に保存。
重い推論は QThread で実行し、UI をブロックしない。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel, QPushButton,
    QLineEdit, QCheckBox, QVBoxLayout, QSplitter, QListWidget,
    QListWidgetItem, QDoubleSpinBox, QSpinBox, QProgressBar,
    QStatusBar, QFormLayout, QMessageBox, QComboBox,
)

from .config import Config
from . import io_utils


# ---------------------------------------------------------------------------
# ヘルパ: numpy → QPixmap
# ---------------------------------------------------------------------------
def ndarray_to_qpixmap(img: np.ndarray) -> QPixmap:
    """RGB ndarray → QPixmap."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w, c = img.shape
    qimg = QImage(img.tobytes(), w, h, c * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray,
                 color=(255, 64, 64), alpha: float = 0.5) -> np.ndarray:
    """対象領域に半透明の色を重ねてプレビュー用にする."""
    out = image_rgb.copy()
    m = (mask > 0)
    color_arr = np.array(color, dtype=np.float32)
    out[m] = (out[m].astype(np.float32) * (1 - alpha)
              + color_arr * alpha).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# ワーカースレッド
# ---------------------------------------------------------------------------
class PreviewWorker(QThread):
    """単一画像のプレビュー推論."""
    finished_with = Signal(object, object)  # (image_rgb, mask_uint8)
    error = Signal(str)

    def __init__(self, cfg: Config, image_path: Path, prompt: str):
        super().__init__()
        self.cfg = cfg
        self.image_path = image_path
        self.prompt = prompt

    def run(self):
        try:
            from .pipeline import ImagePipeline
            pipe = ImagePipeline(self.cfg)
            res = pipe.run_single(self.image_path, self.prompt)
            img = io_utils.read_image(self.image_path)
            self.finished_with.emit(img, res.mask)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


class BatchWorker(QThread):
    """フォルダ / 動画の一括処理."""
    progress = Signal(int, int, str)  # (done, total, current)
    done = Signal(int)
    error = Signal(str)

    def __init__(self, cfg: Config, input_path: Path, output_dir: Path,
                 prompt: str, is_video: bool):
        super().__init__()
        self.cfg = cfg
        self.input_path = input_path
        self.output_dir = output_dir
        self.prompt = prompt
        self.is_video = is_video

    def run(self):
        try:
            def report(i, total, p):
                self.progress.emit(i, total, str(p.name))

            if self.is_video:
                from .pipeline import VideoPipeline
                pipe = VideoPipeline(self.cfg)
                if self.input_path.is_file():
                    import tempfile
                    frames_cache = Path(tempfile.gettempdir()) / "automasker_frames"
                    n = pipe.run_video(
                        self.input_path, self.output_dir, self.prompt,
                        frames_cache=frames_cache,
                        progress=report,
                    )
                else:
                    n = pipe.run_image_sequence(
                        self.input_path, self.output_dir, self.prompt,
                        progress=report,
                    )
            else:
                from .pipeline import ImagePipeline
                pipe = ImagePipeline(self.cfg)
                n = pipe.run_folder(
                    self.input_path, self.output_dir, self.prompt,
                    progress=report,
                )
            self.done.emit(n)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# メインウィンドウ
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoMasker — OSS Edition")
        self.resize(1400, 850)

        self.cfg = Config()
        self.input_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.current_image_path: Optional[Path] = None
        self._current_image_rgb: Optional[np.ndarray] = None
        self._current_mask: Optional[np.ndarray] = None

        self._preview_worker: Optional[PreviewWorker] = None
        self._batch_worker: Optional[BatchWorker] = None

        self._build_menu()
        self._build_ui()
        self._build_statusbar()

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------
    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")

        a_open = QAction("Open input folder / video…", self)
        a_open.setShortcut("Ctrl+O")
        a_open.triggered.connect(self.on_open_input)
        fm.addAction(a_open)

        a_out = QAction("Set output folder…", self)
        a_out.triggered.connect(self.on_set_output)
        fm.addAction(a_out)

        fm.addSeparator()
        a_quit = QAction("Quit", self)
        a_quit.setShortcut("Ctrl+Q")
        a_quit.triggered.connect(self.close)
        fm.addAction(a_quit)

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # ---- 左: ファイルリスト ------------------------------------
        self.file_list = QListWidget()
        self.file_list.setIconSize(QSize(96, 72))
        self.file_list.itemClicked.connect(self.on_file_clicked)
        splitter.addWidget(self.file_list)

        # ---- 中央: プレビュー --------------------------------------
        center = QWidget()
        c_layout = QVBoxLayout(center)
        self.preview_label = QLabel("画像を選んで Preview を押してください")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet(
            "background:#1e1e1e;color:#bbb;border:1px solid #333;"
        )
        c_layout.addWidget(self.preview_label, stretch=1)

        self.view_mode = QComboBox()
        self.view_mode.addItems(["Overlay", "Mask only", "Original"])
        self.view_mode.currentIndexChanged.connect(self._refresh_preview_image)
        c_layout.addWidget(self.view_mode)
        splitter.addWidget(center)

        # ---- 右: コントロール --------------------------------------
        right = QWidget()
        r = QVBoxLayout(right)

        # プロンプト
        r.addWidget(QLabel("<b>Prompt</b> (periods separate classes)"))
        self.prompt_edit = QLineEdit("person . tripod . camera")
        r.addWidget(self.prompt_edit)

        # 閾値
        form = QFormLayout()
        self.spin_box_thresh = QDoubleSpinBox()
        self.spin_box_thresh.setRange(0.05, 0.95)
        self.spin_box_thresh.setSingleStep(0.01)
        self.spin_box_thresh.setValue(self.cfg.box_threshold)
        form.addRow("Box threshold", self.spin_box_thresh)

        self.spin_text_thresh = QDoubleSpinBox()
        self.spin_text_thresh.setRange(0.05, 0.95)
        self.spin_text_thresh.setSingleStep(0.01)
        self.spin_text_thresh.setValue(self.cfg.text_threshold)
        form.addRow("Text threshold", self.spin_text_thresh)

        self.spin_dilate = QSpinBox()
        self.spin_dilate.setRange(0, 50)
        self.spin_dilate.setValue(self.cfg.mask_dilate)
        form.addRow("Dilate (px)", self.spin_dilate)

        self.spin_erode = QSpinBox()
        self.spin_erode.setRange(0, 50)
        self.spin_erode.setValue(self.cfg.mask_erode)
        form.addRow("Erode (px)", self.spin_erode)

        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(0, 100000)
        self.spin_min_area.setValue(self.cfg.mask_min_area)
        form.addRow("Min area (px²)", self.spin_min_area)
        r.addLayout(form)

        # 反転チェック
        self.chk_invert = QCheckBox("Invert mask (keep target, remove rest)")
        r.addWidget(self.chk_invert)

        self.chk_video = QCheckBox("Video / sequence mode (temporal propagation)")
        r.addWidget(self.chk_video)

        # ボタン
        self.btn_preview = QPushButton("Preview (selected image)")
        self.btn_preview.clicked.connect(self.on_preview)
        r.addWidget(self.btn_preview)

        self.btn_batch = QPushButton("Run Batch → masks/")
        self.btn_batch.setStyleSheet("font-weight:bold;padding:8px;")
        self.btn_batch.clicked.connect(self.on_run_batch)
        r.addWidget(self.btn_batch)

        # 進捗バー
        self.progress = QProgressBar()
        self.progress.setValue(0)
        r.addWidget(self.progress)

        r.addStretch(1)

        # 情報
        self.info_label = QLabel("No input loaded.")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color:#888;")
        r.addWidget(self.info_label)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        self.setCentralWidget(splitter)

    def _build_statusbar(self):
        sb = QStatusBar()
        self.setStatusBar(sb)
        sb.showMessage(f"device: {self.cfg.device}")

    # ------------------------------------------------------------------
    # 設定の同期
    # ------------------------------------------------------------------
    def _sync_cfg(self):
        self.cfg.box_threshold = float(self.spin_box_thresh.value())
        self.cfg.text_threshold = float(self.spin_text_thresh.value())
        self.cfg.mask_dilate = int(self.spin_dilate.value())
        self.cfg.mask_erode = int(self.spin_erode.value())
        self.cfg.mask_min_area = int(self.spin_min_area.value())
        self.cfg.invert = self.chk_invert.isChecked()

    # ------------------------------------------------------------------
    # スロット
    # ------------------------------------------------------------------
    def on_open_input(self):
        # フォルダ or 動画ファイルを受け付ける
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video file", "",
            "Videos (*.mp4 *.mov *.avi *.mkv);;All files (*.*)",
        )
        if not path:
            folder = QFileDialog.getExistingDirectory(self, "Open image folder")
            if not folder:
                return
            self.input_dir = Path(folder)
            self._populate_file_list(self.input_dir)
            if self.output_dir is None:
                # デフォルト出力: <parent>/masks
                self.output_dir = self.input_dir.parent / "masks"
        else:
            self.input_dir = Path(path)
            self.file_list.clear()
            self.file_list.addItem(QListWidgetItem(self.input_dir.name))
            self.chk_video.setChecked(True)
            if self.output_dir is None:
                self.output_dir = self.input_dir.parent / "masks"

        self.info_label.setText(
            f"<b>Input:</b> {self.input_dir}<br><b>Output:</b> {self.output_dir}"
        )

    def on_set_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Set output folder")
        if folder:
            self.output_dir = Path(folder)
            self.info_label.setText(
                f"<b>Input:</b> {self.input_dir}<br><b>Output:</b> {self.output_dir}"
            )

    def _populate_file_list(self, folder: Path):
        self.file_list.clear()
        for p in io_utils.list_images(folder, self.cfg):
            item = QListWidgetItem(p.name)
            item.setData(Qt.UserRole, str(p))
            self.file_list.addItem(item)

    def on_file_clicked(self, item: QListWidgetItem):
        data = item.data(Qt.UserRole)
        if not data:
            return
        self.current_image_path = Path(data)
        img = io_utils.read_image(self.current_image_path)
        self._current_image_rgb = img
        self._current_mask = None
        self._refresh_preview_image()

    def _refresh_preview_image(self):
        if self._current_image_rgb is None:
            return
        mode = self.view_mode.currentText()
        if mode == "Original" or self._current_mask is None:
            disp = self._current_image_rgb
        elif mode == "Mask only":
            disp = cv2.cvtColor(self._current_mask, cv2.COLOR_GRAY2RGB)
        else:
            disp = overlay_mask(self._current_image_rgb, self._current_mask)

        pix = ndarray_to_qpixmap(disp)
        pix = pix.scaled(
            self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)

    def on_preview(self):
        if self.current_image_path is None:
            QMessageBox.information(self, "AutoMasker", "先に画像を選択してください。")
            return
        self._sync_cfg()
        self.btn_preview.setEnabled(False)
        self.statusBar().showMessage("Running preview…")

        self._preview_worker = PreviewWorker(
            self.cfg, self.current_image_path, self.prompt_edit.text(),
        )
        self._preview_worker.finished_with.connect(self._on_preview_finished)
        self._preview_worker.error.connect(self._on_worker_error)
        self._preview_worker.start()

    def _on_preview_finished(self, img_rgb, mask):
        self._current_image_rgb = img_rgb
        self._current_mask = mask
        self._refresh_preview_image()
        self.btn_preview.setEnabled(True)
        self.statusBar().showMessage("Preview done.")

    def _on_worker_error(self, msg: str):
        self.btn_preview.setEnabled(True)
        self.btn_batch.setEnabled(True)
        self.statusBar().showMessage("Error.")
        QMessageBox.critical(self, "AutoMasker error", msg)

    def on_run_batch(self):
        if self.input_dir is None:
            QMessageBox.information(self, "AutoMasker", "入力フォルダ/動画を選択してください。")
            return
        if self.output_dir is None:
            QMessageBox.information(self, "AutoMasker", "出力フォルダを指定してください。")
            return
        self._sync_cfg()
        self.btn_batch.setEnabled(False)
        self.progress.setValue(0)
        self.statusBar().showMessage("Batch running…")

        is_video = (
            self.chk_video.isChecked()
            or self.input_dir.suffix.lower() in self.cfg.video_exts
        )
        self._batch_worker = BatchWorker(
            self.cfg, self.input_dir, self.output_dir,
            self.prompt_edit.text(), is_video,
        )
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.done.connect(self._on_batch_done)
        self._batch_worker.error.connect(self._on_worker_error)
        self._batch_worker.start()

    def _on_batch_progress(self, done: int, total: int, name: str):
        pct = int(done * 100 / max(1, total))
        self.progress.setValue(pct)
        self.statusBar().showMessage(f"[{done}/{total}] {name}")

    def _on_batch_done(self, n: int):
        self.btn_batch.setEnabled(True)
        self.progress.setValue(100)
        self.statusBar().showMessage(f"Done. {n} mask(s) written.")
        QMessageBox.information(
            self, "AutoMasker",
            f"{n} 枚のマスクを {self.output_dir} に出力しました。",
        )


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
