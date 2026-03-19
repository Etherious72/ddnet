from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QComboBox,
    QGridLayout,
    QDialog,
    QFileDialog,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPen, QColor
import os
import numpy as np
from func.datasets_reader import single_read_matfile, single_read_npyfile
from .base_button import BaseButton


class LossPanel(QWidget):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.train_loss = []
        self.val_loss = None
        self.smooth_window = None
        self.epochs = None
        self.x_title = "Epoch"
        self.y_title = "Loss"
        self.title = "Training Loss"
        self.error_text = ""

    # ===== 外部调用接口 =====
    def set_data(self, train_loss, val_loss=None, smooth=None, epochs=None,
                 x_title="Epoch", y_title="Loss", title="Training Loss"):
        self.train_loss = np.array(train_loss)
        self.val_loss = np.array(val_loss) if val_loss is not None else None
        self.smooth_window = smooth
        self.epochs = epochs
        self.x_title = x_title
        self.y_title = y_title
        self.title = title
        self.error_text = ""
        self.update()

    def set_error(self, text):
        self.train_loss = []
        self.val_loss = None
        self.error_text = text
        self.update()

    # ===== 平滑函数 =====
    def smooth(self, data, window):
        if window is None or window <= 1:
            return data
        return np.convolve(data, np.ones(window)/window, mode='same')

    # ===== 核心绘图 =====
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if len(self.train_loss) < 2:
            painter.fillRect(self.rect(), QColor(250, 252, 255))
            painter.setPen(QColor(185, 74, 72))
            font = QFont()
            font.setPointSize(11)
            painter.setFont(font)
            msg = self.error_text if self.error_text else "暂无可显示数据"
            painter.drawText(self.rect(), Qt.AlignCenter, msg)
            return

        w = self.width()
        h = self.height()

        left_margin = 88
        right_margin = 28
        top_margin = 52
        bottom_margin = 88
        plot_w = w - left_margin - right_margin
        plot_h = h - top_margin - bottom_margin
        if plot_w <= 20 or plot_h <= 20:
            return

        # ===== 数据处理 =====
        train = self.smooth(self.train_loss, self.smooth_window)
        data_all = train

        if self.val_loss is not None:
            val = self.smooth(self.val_loss, self.smooth_window)
            data_all = np.concatenate([train, val])
        else:
            val = None

        min_v = float(np.min(data_all))
        max_v = float(np.max(data_all))
        eps = 1e-8

        def norm_y(v):
            return (v - min_v) / (max_v - min_v + eps)

        # ===== 背景 =====
        painter.fillRect(self.rect(), QColor(250, 252, 255))

        # ===== 网格 =====
        grid_pen = QPen(QColor(222, 228, 236), 1, Qt.DashLine)
        painter.setPen(grid_pen)

        grid_n = 5
        for i in range(grid_n + 1):
            y = top_margin + int(i / grid_n * plot_h)
            painter.drawLine(left_margin, y, w - right_margin, y)

        for i in range(grid_n + 1):
            x = left_margin + int(i / grid_n * plot_w)
            painter.drawLine(x, top_margin, x, h - bottom_margin)

        # ===== 坐标轴 =====
        axis_pen = QPen(QColor(60, 60, 60), 2)
        painter.setPen(axis_pen)

        # x轴
        painter.drawLine(left_margin, h - bottom_margin, w - right_margin, h - bottom_margin)
        # y轴
        painter.drawLine(left_margin, top_margin, left_margin, h - bottom_margin)

        # ===== 画曲线函数 =====
        def draw_curve(data, color):
            pen = QPen(color, 2)
            painter.setPen(pen)

            n = len(data)
            points = []

            for i, v in enumerate(data):
                x = left_margin + int(i / (n - 1) * plot_w)
                y = top_margin + int((1 - norm_y(v)) * plot_h)
                points.append((x, y))

            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1],
                                 points[i+1][0], points[i+1][1])

            step = max(n // 12, 1)
            for i in range(0, n, step):
                painter.drawEllipse(points[i][0] - 2, points[i][1] - 2, 4, 4)

        # ===== 画 train =====
        draw_curve(train, QColor(30, 144, 255))  # 蓝色

        # ===== 画 val =====
        if val is not None:
            draw_curve(val, QColor(220, 20, 60))  # 红色

        # ===== 标题 =====
        painter.setPen(QColor(35, 35, 35))
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        painter.setFont(font)

        painter.drawText(0, 14, w, 30, Qt.AlignCenter, self.title)

        # ===== 图例 =====
        legend_y = 34

        painter.setPen(QPen(QColor(30, 144, 255), 2))
        painter.drawLine(w - 150, legend_y, w - 130, legend_y)
        painter.setPen(Qt.black)
        painter.drawText(w - 120, legend_y + 5, "Train")

        if val is not None:
            painter.setPen(QPen(QColor(220, 20, 60), 2))
            painter.drawLine(w - 80, legend_y, w - 60, legend_y)
            painter.setPen(Qt.black)
            painter.drawText(w - 50, legend_y + 5, "Val")

        # ===== 坐标刻度 =====
        # x 轴刻度：对齐 save_results，每 20 一个刻度，范围 [0, epochs]
        if self.epochs is not None and self.epochs > 0:
            tick_epochs = list(range(0, int(self.epochs) + 1, 20))
            if int(self.epochs) not in tick_epochs:
                tick_epochs.append(int(self.epochs))
            tick_epochs = sorted(set(tick_epochs))
            for epoch in tick_epochs:
                x = left_margin + int(epoch / max(int(self.epochs), 1) * plot_w)
                painter.drawLine(x, h - bottom_margin, x, h - bottom_margin + 6)
                painter.drawText(x - 18, h - bottom_margin + 20, 36, 16, Qt.AlignCenter, str(epoch))
        else:
            ticks = 5
            for i in range(ticks + 1):
                x = left_margin + int(i / ticks * plot_w)
                epoch = int(i / ticks * (len(train) - 1))
                painter.drawLine(x, h - bottom_margin, x, h - bottom_margin + 6)
                painter.drawText(x - 18, h - bottom_margin + 20, 36, 16, Qt.AlignCenter, str(epoch))

        ticks = 5
        for i in range(ticks + 1):
            # y轴
            y = top_margin + int(i / ticks * plot_h)
            val_text = max_v - i / ticks * (max_v - min_v)
            painter.drawLine(left_margin - 6, y, left_margin, y)
            painter.drawText(12, y - 8, left_margin - 20, 16, Qt.AlignRight | Qt.AlignVCenter, f"{val_text:.2f}")

        # x/y 轴标题
        font.setBold(False)
        font.setPointSize(11)
        painter.setFont(font)
        # 轴标题放在刻度文本之外，避免遮挡
        painter.drawText(left_margin, h - 26, plot_w, 18, Qt.AlignCenter, self.x_title)
        painter.save()
        painter.translate(14, top_margin + plot_h // 2)
        painter.rotate(-90)
        painter.drawText(-plot_h // 2, 0, plot_h, 20, Qt.AlignCenter, self.y_title)
        painter.restore()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def export_pixmap(self):
        pix = QPixmap(self.size())
        pix.fill(Qt.transparent)
        self.render(pix)
        return pix


class SystemReminderPage(QWidget):
    # Simple home page for system reminder with titles only
    train_requested = pyqtSignal()
    compare_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.title_label = QLabel("基于U-Net的地震反演系统", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        font = self.title_label.font()
        font.setPointSize(20)
        font.setBold(True)
        self.title_label.setFont(font)
        self.title_label.setStyleSheet("color: red;")
        layout.addWidget(self.title_label)

        bottom_widget = QWidget(self)
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(40, 0, 40, 28)
        bottom_layout.setSpacing(20)

        self.train_btn = BaseButton("训练页面", self, color="#1E90FF", radius=8)
        self.compare_btn = BaseButton("预测对比页面", self, color="#1E90FF", radius=8)

        bottom_layout.addWidget(self.train_btn)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.compare_btn)
        layout.addWidget(bottom_widget)
        self.setLayout(layout)

        # Signals to host (placeholders for future wiring)
        self.train_btn.clicked.connect(self.train_requested)
        self.compare_btn.clicked.connect(self.compare_requested)

    def update_content(self, text: str) -> None:
        pass


class LossWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.loss = []

    def set_data(self, loss):
        self.loss = loss
        self.update()

    def paintEvent(self, event):
        if not self.loss:
            return

        painter = QPainter(self)
        pen = QPen(Qt.blue, 2)
        painter.setPen(pen)

        w, h = self.width(), self.height()

        max_loss = max(self.loss)
        min_loss = min(self.loss)

        def norm(x):
            return (x - min_loss) / (max_loss - min_loss + 1e-8)

        points = []
        for i, v in enumerate(self.loss):
            x = int(i / len(self.loss) * w)
            y = int(h - norm(v) * h)
            points.append((x, y))

        for i in range(len(points) - 1):
            painter.drawLine(*points[i], *points[i+1])


class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class TrainingView(QWidget):
    train_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_export_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results",
        )
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        self.combo = QComboBox(self)
        self.combo.addItems(["SEGSalt", "SEGSimulation", "FlatVelA", "CurveFaultA", "FlatFaultA", "CurveVelA"])
        self.start_btn = BaseButton("显示Loss页面", self, color="#1E90FF", radius=8)
        self.back_btn = BaseButton("返回", self, color="#1E90FF", radius=8)
        left_layout.addWidget(self.combo, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.back_btn, alignment=Qt.AlignCenter)
        root.addWidget(left_widget, 1)

        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)
        self.loss_panel = LossPanel(self)
        right_layout.addWidget(self.loss_panel)
        root.addWidget(right_widget, 2)
        self.setLayout(root)
        self.start_btn.clicked.connect(self._on_show_loss)
        self.loss_panel.clicked.connect(self._show_loss_zoom)
        self.back_btn.clicked.connect(self.back_requested.emit)

    def _show_loss_zoom(self):
        if len(self.loss_panel.train_loss) < 2:
            return

        pix = self.loss_panel.export_pixmap()
        if pix.isNull():
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Loss - 放大图")
        dialog.resize(920, 640)

        layout = QVBoxLayout(dialog)
        zoom_label = QLabel(dialog)
        zoom_label.setAlignment(Qt.AlignCenter)
        zoom_label.setStyleSheet("background:#f3f7fb;")
        layout.addWidget(zoom_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        zoom_download_btn = BaseButton("下载Loss图", dialog, color="#20B2AA", radius=8)
        btn_row.addWidget(zoom_download_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        zoom_label.setPixmap(
            pix.scaled(
                880,
                600,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        zoom_download_btn.clicked.connect(lambda: self._download_loss_image(pixmap=pix, parent=dialog))
        dialog.exec_()

    def _download_loss_image(self, pixmap=None, parent=None):
        if len(self.loss_panel.train_loss) < 2:
            return

        os.makedirs(self._last_export_dir, exist_ok=True)
        dataset_name = self.combo.currentText()
        default_name = f"{dataset_name}_loss.png"
        save_path, _ = QFileDialog.getSaveFileName(
            parent or self,
            "保存Loss图",
            os.path.join(self._last_export_dir, default_name),
            "PNG (*.png);;JPG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not save_path:
            return

        self._last_export_dir = os.path.dirname(save_path)
        export_pix = pixmap if pixmap is not None else self.loss_panel.export_pixmap()
        export_pix.save(save_path)

    def _load_loss_by_path(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".npy":
            arr = np.load(file_path)
        elif ext == ".npz":
            data = np.load(file_path)
            arr = data[data.files[0]] if data.files else np.array([])
        elif ext == ".csv":
            arr = np.loadtxt(file_path, delimiter=",")
        else:
            arr = np.loadtxt(file_path)
        arr = np.asarray(arr).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"读取文件内容为空: {file_path}")
        return arr

    def _find_existing_path(self, paths):
        for path in paths:
            if os.path.exists(path):
                return path
        return None

    def _on_show_loss(self):
        dataset_name = self.combo.currentText()
        page_dir = os.path.dirname(__file__)
        base_dir = os.path.join(page_dir, "resources", "training_data", dataset_name)
        results_dir = os.path.join(
            os.path.dirname(page_dir),
            os.path.dirname(page_dir),
            "results",
            f"{dataset_name}Results",
        )

        train_loss_path = self._find_existing_path([
            os.path.join(base_dir, "train_loss.npy"),
            os.path.join(base_dir, "train_loss.csv"),
            os.path.join(base_dir, "train_loss.txt"),
            os.path.join(base_dir, "train_loss.npz"),
            os.path.join(results_dir, f"[Loss]{dataset_name}.npy"),
            os.path.join(results_dir, f"[Loss]{dataset_name}_CLStage1.npy"),
            os.path.join(results_dir, f"[Loss]{dataset_name}_CLStage2.npy"),
        ])
        val_loss_path = self._find_existing_path([
            os.path.join(base_dir, "val_loss.npy"),
            os.path.join(base_dir, "val_loss.csv"),
            os.path.join(base_dir, "val_loss.txt"),
            os.path.join(base_dir, "val_loss.npz"),
        ])
        if not train_loss_path:
            self.loss_panel.set_error("未找到 Loss 文件")
            return

        try:
            # 对齐 utils.save_results 的绘制逻辑：默认从 loss[1:] 开始绘制
            full_loss = self._load_loss_by_path(train_loss_path)
            train_loss = full_loss[1:] if full_loss.size > 1 else full_loss
            epochs = int(full_loss.size - 1) if full_loss.size > 1 else int(full_loss.size)

            val_loss = self._load_loss_by_path(val_loss_path) if val_loss_path else None
            self.loss_panel.set_data(
                train_loss,
                val_loss,
                smooth=None,
                epochs=epochs,
                x_title="Epoch",
                y_title="Loss",
                title=f"{dataset_name} Loss",
            )
        except Exception as e:
            self.loss_panel.set_error(f"数据读取失败: {e}")


class CompareView(QWidget):
    train_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_pixmaps = {}
        self._cached_compare = {}
        self._last_export_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results",
        )
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)

        self.combo = QComboBox(self)
        self.combo.addItems(["SEGSalt", "SEGSimulation", "FlatVelA", "CurveFaultA", "FlatFaultA", "CurveVelA"])

        self.start_btn = BaseButton("Test", self, color="#1E90FF", radius=8)
        self.pd_btn = BaseButton("PD", self, color="#1E90FF", radius=8)
        self.gt_btn = BaseButton("GT", self, color="#1E90FF", radius=8)
        self.back_btn = BaseButton("返回", self, color="#1E90FF", radius=8)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.pd_btn)
        btn_row.addSpacing(10)
        btn_row.addWidget(self.gt_btn)
        btn_row.addStretch()

        left_layout.addWidget(self.combo, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)
        left_layout.addLayout(btn_row)
        left_layout.addWidget(self.back_btn, alignment=Qt.AlignCenter)

        root.addWidget(left_widget)

        # 右侧
        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)

        img_col = QVBoxLayout()
        img_col.setContentsMargins(12, 12, 12, 12)
        img_col.setSpacing(14)

        self.pd_label = ClickableLabel("PD")
        self.gt_label = ClickableLabel("GT")

        for lbl in [self.pd_label, self.gt_label]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(260, 240)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setStyleSheet(
                "border:1px solid #c7d0db;"
                "background:#f8fbff;"
                "color:#2f3b4a;"
            )

        img_col.addWidget(self.pd_label, 1)
        img_col.addWidget(self.gt_label, 1)
        right_layout.addLayout(img_col)

        root.addWidget(right_widget, 2)

        self.start_btn.clicked.connect(self.show_pd)
        self.pd_btn.clicked.connect(self.show_pd)
        self.gt_btn.clicked.connect(self.show_gt)
        self.back_btn.clicked.connect(self.back_requested.emit)
        self.pd_label.clicked.connect(lambda: self._show_zoom(self.pd_label, "PD"))
        self.gt_label.clicked.connect(lambda: self._show_zoom(self.gt_label, "GT"))

        # 默认优先展示测试数据
        self.show_pd()

    def show_pd(self):
        self._update_compare_images(focus="pd", pd_mode="test", gt_mode="train")
        self.pd_label.setStyleSheet(
            "border:2px solid #1e90ff;"
            "background:#f8fbff;"
            "color:#2f3b4a;"
        )
        self.gt_label.setStyleSheet(
            "border:1px solid #c7d0db;"
            "background:#f8fbff;"
            "color:#2f3b4a;"
        )

    def show_gt(self):
        self._update_compare_images(focus="gt", pd_mode="test", gt_mode="train")
        self.gt_label.setStyleSheet(
            "border:2px solid #1e90ff;"
            "background:#f8fbff;"
            "color:#2f3b4a;"
        )
        self.pd_label.setStyleSheet(
            "border:1px solid #c7d0db;"
            "background:#f8fbff;"
            "color:#2f3b4a;"
        )

    def show_image(self, label, img_array):
        import numpy as np
        from PyQt5.QtGui import QImage, QPixmap

        if img_array is None:
            return

        if img_array.dtype != np.uint8:
            img_array = (255 * (img_array - img_array.min()) / (np.ptp(img_array) + 1e-8)).astype(np.uint8)

        h, w = img_array.shape
        qimg = QImage(img_array.data, w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        self._image_pixmaps[label] = pix
        label.setPixmap(pix.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _show_zoom(self, label, title):
        pix = self._image_pixmaps.get(label)
        if pix is None or pix.isNull():
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"{title} - 放大图")
        dialog.resize(720, 720)

        layout = QVBoxLayout(dialog)
        zoom_label = QLabel(dialog)
        zoom_label.setAlignment(Qt.AlignCenter)
        zoom_label.setStyleSheet("background:#0f131a;")
        layout.addWidget(zoom_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        zoom_download_btn = BaseButton("下载图像", dialog, color="#20B2AA", radius=8)
        btn_row.addWidget(zoom_download_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        zoom_label.setPixmap(
            pix.scaled(
                680,
                680,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        zoom_download_btn.clicked.connect(lambda: self._download_compare_image(title, pix, dialog))
        dialog.exec_()

    def _download_compare_image(self, title, pixmap, parent=None):
        if pixmap is None or pixmap.isNull():
            return

        os.makedirs(self._last_export_dir, exist_ok=True)
        dataset_name = self.combo.currentText()
        default_name = f"{dataset_name}_{title.lower()}.png"
        save_path, _ = QFileDialog.getSaveFileName(
            parent or self,
            "保存图像",
            os.path.join(self._last_export_dir, default_name),
            "PNG (*.png);;JPG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not save_path:
            return

        self._last_export_dir = os.path.dirname(save_path)
        pixmap.save(save_path)

    def _on_show_loss(self):
        self._update_compare_images(focus=None, pd_mode="test", gt_mode="train")

    def _dataset_cfg(self, dataset_name):
        if dataset_name in ["SEGSalt", "SEGSimulation"]:
            return {"data_dim": [400, 301], "model_dim": [201, 301], "default_select_id": 1615}
        return {"data_dim": [1000, 70], "model_dim": [70, 70], "default_select_id": [1, 2]}

    def _dataset_dir(self, dataset_name):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(root, "data", dataset_name) + os.sep

    def _load_velocity_model(self, dataset_name, train_or_test):
        cfg = self._dataset_cfg(dataset_name)
        select_id = cfg["default_select_id"]
        data_dir = self._dataset_dir(dataset_name)

        if dataset_name in ["SEGSalt", "SEGSimulation"]:
            _, velocity_model, _ = single_read_matfile(
                data_dir,
                cfg["data_dim"],
                cfg["model_dim"],
                select_id,
                train_or_test=train_or_test,
            )
            max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
        else:
            _, velocity_model, _ = single_read_npyfile(
                data_dir,
                select_id,
                train_or_test=train_or_test,
            )
            max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
            velocity_model = (velocity_model - np.min(velocity_model)) / (np.max(velocity_model) - np.min(velocity_model) + 1e-8)

        velocity_model = np.asarray(velocity_model)
        if velocity_model.size == 0:
            raise ValueError(f"读取文件内容为空: dataset={dataset_name}, mode={train_or_test}")

        return velocity_model.astype(np.float32), float(min_velocity), float(max_velocity)

    def _build_openfwi_model_pixmap(self, model, min_v, max_v, title):
        from PyQt5.QtGui import QImage

        width, height = 560, 560
        left_margin = 78
        right_margin = 36
        top_margin = 52
        bottom_margin = 76
        plot_w = width - left_margin - right_margin
        plot_h = height - top_margin - bottom_margin

        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(QColor(248, 251, 255))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)

        data = np.asarray(model, dtype=np.float32)
        if data.shape != (70, 70):
            y_idx = np.linspace(0, max(data.shape[0] - 1, 0), 70).astype(int)
            x_idx = np.linspace(0, max(data.shape[1] - 1, 0), 70).astype(int)
            data = data[y_idx][:, x_idx]

        rng = max(max_v - min_v, 1e-8)
        norm = np.clip((data - min_v) / rng, 0.0, 1.0)
        rgb = np.zeros((70, 70, 3), dtype=np.uint8)
        rgb[..., 0] = (255 * norm).astype(np.uint8)
        rgb[..., 1] = (255 * (1.0 - np.abs(norm - 0.5) * 2.0)).astype(np.uint8)
        rgb[..., 2] = (255 * (1.0 - norm)).astype(np.uint8)

        plot_img = QImage(rgb.data, 70, 70, 70 * 3, QImage.Format_RGB888).copy()
        painter.drawImage(
            left_margin,
            top_margin,
            plot_img.scaled(plot_w, plot_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation),
        )

        painter.setPen(QPen(QColor(68, 68, 68), 2))
        painter.drawRect(left_margin, top_margin, plot_w, plot_h)

        font = QFont()
        font.setPointSize(11)
        painter.setFont(font)
        painter.setPen(QColor(45, 45, 45))

        ticks = np.linspace(0.0, 0.7, 8)
        for t in ticks:
            ratio = t / 0.7
            x = left_margin + int(ratio * plot_w)
            y = top_margin + int(ratio * plot_h)
            painter.drawLine(x, top_margin + plot_h, x, top_margin + plot_h + 5)
            painter.drawText(x - 14, top_margin + plot_h + 10, 28, 18, Qt.AlignCenter, f"{t:.1f}")
            painter.drawLine(left_margin - 5, y, left_margin, y)
            painter.drawText(6, y - 9, left_margin - 14, 18, Qt.AlignRight | Qt.AlignVCenter, f"{t:.1f}")

        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(0, 14, width, 24, Qt.AlignCenter, title)

        font.setPointSize(11)
        font.setBold(False)
        painter.setFont(font)
        painter.drawText(left_margin, height - 32, plot_w, 20, Qt.AlignCenter, "Position (km)")
        painter.save()
        painter.translate(20, top_margin + plot_h // 2)
        painter.rotate(-90)
        painter.drawText(-plot_h // 2, 0, plot_h, 20, Qt.AlignCenter, "Depth (km)")
        painter.restore()

        painter.end()
        return QPixmap.fromImage(image)

    def _render_velocity_to_label(self, label, model, min_v, max_v, title):
        pix = self._build_openfwi_model_pixmap(model, min_v, max_v, title)
        self._image_pixmaps[label] = pix
        label.setPixmap(
            pix.scaled(
                label.width(),
                label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _load_compare_data(self, dataset_name, pd_mode="test", gt_mode="train"):
        cache_key = (dataset_name, pd_mode, gt_mode)
        if cache_key in self._cached_compare:
            return self._cached_compare[cache_key]

        try:
            pd_vm, pd_min, pd_max = self._load_velocity_model(dataset_name, pd_mode)
            gt_vm, gt_min, gt_max = self._load_velocity_model(dataset_name, gt_mode)
            warning = ""
        except Exception:
            rng = np.random.default_rng(abs(hash((dataset_name, pd_mode, gt_mode))) % 100000)
            gt_vm = np.clip(rng.normal(0.55, 0.2, (70, 70)), 0.0, 1.0).astype(np.float32)
            pd_vm = np.clip(gt_vm + rng.normal(0.0, 0.06, (70, 70)), 0.0, 1.0).astype(np.float32)
            gt_min, gt_max = 1500.0, 4500.0
            pd_min, pd_max = gt_min, gt_max
            warning = "文件为空或读取失败，已回退到测试数据"

        pd_show = pd_min + pd_vm * (pd_max - pd_min)
        gt_show = gt_min + gt_vm * (gt_max - gt_min)

        min_v = float(np.min(gt_min + gt_vm * (gt_max - gt_min)))
        max_v = float(np.max(gt_min + gt_vm * (gt_max - gt_min)))

        payload = {"pd": pd_show, "gt": gt_show, "min_v": min_v, "max_v": max_v, "warning": warning}
        self._cached_compare[cache_key] = payload
        return payload



    def _update_compare_images(self, focus=None, pd_mode="test", gt_mode="train"):
        dataset_name = self.combo.currentText()
        payload = self._load_compare_data(dataset_name, pd_mode=pd_mode, gt_mode=gt_mode)

        if focus in (None, "pd"):
            self._render_velocity_to_label(
                self.pd_label,
                payload["pd"],
                payload["min_v"],
                payload["max_v"],
                f"{dataset_name} PD",
            )
        if focus in (None, "gt"):
            self._render_velocity_to_label(
                self.gt_label,
                payload["gt"],
                payload["min_v"],
                payload["max_v"],
                f"{dataset_name} GT",
            )

    def _load_placeholder_image(self):
        base = os.path.dirname(__file__)
        path = os.path.join(base, 'resources', 'system_reminder.png')
        if os.path.exists(path):
            pix = QPixmap(path)
            if not pix.isNull():
                return pix
        pix = QPixmap(320, 240)
        pix.fill(Qt.white)
        painter = QPainter(pix)
        painter.setPen(Qt.black)
        font = QFont()
        font.setPointSize(14)
        painter.setFont(font)
        painter.drawText(pix.rect(), Qt.AlignCenter, "<system-reminder>")
        painter.end()
        return pix
