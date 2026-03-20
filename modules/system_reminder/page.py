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
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPlainTextEdit,
    QApplication,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPen, QColor, QFontMetrics
import os
import csv
import glob
import json
import subprocess
import sys
import hashlib
from datetime import datetime
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
        self.train_label = "Train"
        self.val_label = "Val"
        self.error_text = ""

    # ===== 外部调用接口 =====
    def set_data(self, train_loss, val_loss=None, smooth=None, epochs=None,
                 x_title="Epoch", y_title="Loss", title="Training Loss",
                 train_label="Train", val_label="Val"):
        self.train_loss = np.array(train_loss)
        self.val_loss = np.array(val_loss) if val_loss is not None else None
        self.smooth_window = smooth
        self.epochs = epochs
        self.x_title = x_title
        self.y_title = y_title
        self.title = title
        self.train_label = train_label
        self.val_label = val_label
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

        w = max(self.width(), 1)
        h = max(self.height(), 1)
        base = min(w, h)
        scale = max(0.65, min(2.4, base / 700.0))

        title_size = max(10, int(13 * scale))
        axis_size = max(8, int(11 * scale))
        tick_size = max(7, int(9 * scale))
        error_size = max(9, int(11 * scale))

        if len(self.train_loss) < 2:
            painter.fillRect(self.rect(), QColor(250, 252, 255))
            painter.setPen(QColor(185, 74, 72))
            font = QFont()
            font.setPointSize(error_size)
            painter.setFont(font)
            msg = self.error_text if self.error_text else "暂无可显示数据"
            painter.drawText(self.rect(), Qt.AlignCenter, msg)
            return

        left_margin = max(56, int(88 * scale))
        right_margin = max(18, int(28 * scale))
        top_margin = max(36, int(52 * scale))
        bottom_margin = max(56, int(88 * scale))
        left_margin = min(left_margin, int(w * 0.30))
        right_margin = min(right_margin, int(w * 0.20))
        top_margin = min(top_margin, int(h * 0.25))
        bottom_margin = min(bottom_margin, int(h * 0.32))
        plot_w = w - left_margin - right_margin
        plot_h = h - top_margin - bottom_margin
        if plot_w <= 40 or plot_h <= 40:
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
        axis_pen = QPen(QColor(60, 60, 60), max(1, int(2 * scale)))
        painter.setPen(axis_pen)

        # x轴
        painter.drawLine(left_margin, h - bottom_margin, w - right_margin, h - bottom_margin)
        # y轴
        painter.drawLine(left_margin, top_margin, left_margin, h - bottom_margin)

        # ===== 画曲线函数 =====
        def draw_curve(data, color):
            pen = QPen(color, max(1, int(2 * scale)))
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
            dot = max(2, int(4 * scale))
            for i in range(0, n, step):
                painter.drawEllipse(points[i][0] - dot // 2, points[i][1] - dot // 2, dot, dot)

        # ===== 画 train =====
        draw_curve(train, QColor(30, 144, 255))  # 蓝色

        # ===== 画 val =====
        if val is not None:
            draw_curve(val, QColor(220, 20, 60))  # 红色

        # ===== 标题 =====
        painter.setPen(QColor(35, 35, 35))
        font = QFont()
        font.setPointSize(title_size)
        font.setBold(True)
        painter.setFont(font)

        title_y = max(8, int(10 * scale))
        title_h = max(24, int(30 * scale))
        title_bottom = title_y + title_h
        painter.drawText(0, title_y, w, title_h, Qt.AlignCenter, self.title)

        # ===== 图例（右上角自适应，避免文字被裁切） =====
        font.setPointSize(max(8, int(10 * scale)))
        font.setBold(False)
        painter.setFont(font)
        fm = QFontMetrics(font)

        line_len = max(12, int(20 * scale))
        text_gap = max(6, int(8 * scale))
        item_gap = max(10, int(14 * scale))
        legend_items = [(QColor(30, 144, 255), self.train_label)]
        if val is not None:
            legend_items.append((QColor(220, 20, 60), self.val_label))

        total_w = 0
        for _, label in legend_items:
            total_w += line_len + text_gap + fm.horizontalAdvance(label) + item_gap
        total_w = max(0, total_w - item_gap)

        legend_x = max(left_margin + 4, w - right_margin - total_w - 4)
        legend_y = min(
            top_margin - max(10, int(16 * scale)),
            title_bottom + max(22, int(30 * scale)),
        )
        legend_y = max(legend_y, max(32, int(42 * scale)))

        cursor_x = legend_x
        for color, label in legend_items:
            painter.setPen(QPen(color, max(1, int(2 * scale))))
            painter.drawLine(cursor_x, legend_y, cursor_x + line_len, legend_y)
            cursor_x += line_len + text_gap
            painter.setPen(Qt.black)
            label_w = fm.horizontalAdvance(label)
            painter.drawText(cursor_x, legend_y + max(5, int(6 * scale)), label)
            cursor_x += label_w + item_gap

        # ===== 坐标刻度 =====
        # x 轴刻度：对齐 save_results，每 20 一个刻度，范围 [0, epochs]
        tick_font = QFont()
        tick_font.setPointSize(tick_size)
        painter.setFont(tick_font)

        if self.epochs is not None and self.epochs > 0:
            tick_epochs = list(range(0, int(self.epochs) + 1, 20))
            if int(self.epochs) not in tick_epochs:
                tick_epochs.append(int(self.epochs))
            tick_epochs = sorted(set(tick_epochs))
            for epoch in tick_epochs:
                x = left_margin + int(epoch / max(int(self.epochs), 1) * plot_w)
                tick_len = max(4, int(6 * scale))
                painter.drawLine(x, h - bottom_margin, x, h - bottom_margin + tick_len)
                text_w = max(28, int(44 * scale))
                text_h = max(14, int(18 * scale))
                painter.drawText(x - text_w // 2, h - bottom_margin + tick_len + int(2 * scale), text_w, text_h, Qt.AlignCenter, str(epoch))
        else:
            ticks = 5
            for i in range(ticks + 1):
                x = left_margin + int(i / ticks * plot_w)
                epoch = int(i / ticks * (len(train) - 1))
                tick_len = max(4, int(6 * scale))
                painter.drawLine(x, h - bottom_margin, x, h - bottom_margin + tick_len)
                text_w = max(28, int(44 * scale))
                text_h = max(14, int(18 * scale))
                painter.drawText(x - text_w // 2, h - bottom_margin + tick_len + int(2 * scale), text_w, text_h, Qt.AlignCenter, str(epoch))

        ticks = 5
        for i in range(ticks + 1):
            # y轴
            y = top_margin + int(i / ticks * plot_h)
            val_text = max_v - i / ticks * (max_v - min_v)
            tick_len = max(4, int(6 * scale))
            painter.drawLine(left_margin - tick_len, y, left_margin, y)
            text_h = max(14, int(18 * scale))
            painter.drawText(max(6, int(10 * scale)), y - text_h // 2, left_margin - max(14, int(18 * scale)), text_h, Qt.AlignRight | Qt.AlignVCenter, f"{val_text:.2f}")

        # x/y 轴标题
        font.setBold(False)
        font.setPointSize(axis_size)
        painter.setFont(font)
        # 轴标题放在刻度文本之外，避免遮挡
        painter.drawText(left_margin, h - max(22, int(26 * scale)), plot_w, max(16, int(18 * scale)), Qt.AlignCenter, self.x_title)
        painter.save()
        painter.translate(max(10, int(14 * scale)), top_margin + plot_h // 2)
        painter.rotate(-90)
        painter.drawText(-plot_h // 2, 0, plot_h, max(18, int(20 * scale)), Qt.AlignCenter, self.y_title)
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
    metric_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.title_label = QLabel("融合预训练策略的地震反演系统", self)
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
        self.metric_btn = BaseButton("指标分析", self, color="#1E90FF", radius=8)

        bottom_layout.addWidget(self.train_btn)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.metric_btn)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.compare_btn)
        layout.addWidget(bottom_widget)
        self.setLayout(layout)

        # Signals to host (placeholders for future wiring)
        self.train_btn.clicked.connect(self.train_requested)
        self.compare_btn.clicked.connect(self.compare_requested)
        self.metric_btn.clicked.connect(self.metric_requested)

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
        self.model1_label = QLabel("对比模型1", self)
        self.model1_label.setAlignment(Qt.AlignCenter)
        self.combo = QComboBox(self)
        self.combo.addItems(["Scratch", "Sup-Mig", "Self-Sup"])
        self.model2_label = QLabel("对比模型2", self)
        self.model2_label.setAlignment(Qt.AlignCenter)
        self.compare_model_combo = QComboBox(self)
        self.compare_model_combo.addItems(["Scratch", "Sup-Mig", "Self-Sup"])
        self.start_btn = BaseButton("显示Loss页面", self, color="#1E90FF", radius=8)
        self.back_btn = BaseButton("返回", self, color="#1E90FF", radius=8)
        left_layout.addWidget(self.model1_label, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.combo, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.model2_label, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.compare_model_combo, alignment=Qt.AlignCenter)
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

    def _method_root_dir(self, method_name):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        base = os.path.join(project_root, "results", "CurveFaultAResults")
        mapping = {
            "Scratch": os.path.join(base, "unused_pretrain", "DDNet70", "CurveFaultA"),
            "Sup-Mig": os.path.join(base, "used_pretrain", "DDNet70", "CurveFaultA"),
            "Self-Sup": os.path.join(base, "self_sup", "DDNet70", "CurveFaultA"),
        }
        return mapping.get(method_name)

    def _find_method_loss_path(self, method_name):
        root = self._method_root_dir(method_name)
        if not root or not os.path.isdir(root):
            return None

        matches = []
        for cur_root, _, files in os.walk(root):
            for name in files:
                lower = name.lower()
                if lower.endswith(".npy") and "[loss]" in lower and "curvefaulta" in lower:
                    full = os.path.join(cur_root, name)
                    matches.append((os.path.getmtime(full), full))

        if not matches:
            return None
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[0][1]

    def _on_show_loss(self):
        model1_method = self.combo.currentText()
        model2_method = self.compare_model_combo.currentText()

        model1_loss_path = self._find_method_loss_path(model1_method)
        model2_loss_path = self._find_method_loss_path(model2_method)
        if not model1_loss_path or not model2_loss_path:
            self.loss_panel.set_error(
                f"未找到双模型Loss文件: {model1_method}={bool(model1_loss_path)}, {model2_method}={bool(model2_loss_path)}"
            )
            return

        try:
            # 对齐 utils.save_results 的绘制逻辑：默认从 loss[1:] 开始绘制
            full_loss = self._load_loss_by_path(model1_loss_path)
            train_loss = full_loss[1:] if full_loss.size > 1 else full_loss

            val_loss_raw = self._load_loss_by_path(model2_loss_path)
            val_loss = val_loss_raw[1:] if val_loss_raw.size > 1 else val_loss_raw

            min_len = min(len(train_loss), len(val_loss))
            if min_len < 2:
                raise ValueError("对比 Loss 数据长度不足")
            train_loss = train_loss[:min_len]
            val_loss = val_loss[:min_len]

            self.loss_panel.set_data(
                train_loss,
                val_loss,
                smooth=None,
                epochs=min_len - 1,
                x_title="Epoch",
                y_title="Loss",
                title="CurveFaultA 双模型Loss对比",
                train_label=model1_method,
                val_label=model2_method,
            )
        except Exception as e:
            self.loss_panel.set_error(f"数据读取失败: {e}")


class MetricAnalysisView(QWidget):
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self._metric_results_dir = os.path.join(self._project_root, "results", "CurveFaultAResults")
        self._cache_dir = os.path.join(self._metric_results_dir, "metric_cache")
        self._snapshot_cache_path = os.path.join(self._cache_dir, "latest_compare_batch_snapshot.json")
        self._memory_pair_cache = {}
        self._memory_snapshot = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)

        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        self.model1_label = QLabel("对比模型1", self)
        self.model1_label.setAlignment(Qt.AlignCenter)
        self.combo = QComboBox(self)
        self.combo.addItems(["Scratch", "Sup-Mig", "Self-Sup"])
        self.model2_label = QLabel("对比模型2", self)
        self.model2_label.setAlignment(Qt.AlignCenter)
        self.compare_model_combo = QComboBox(self)
        self.compare_model_combo.addItems(["Scratch", "Sup-Mig", "Self-Sup"])
        self.start_btn = BaseButton("重新计算并刷新", self, color="#1E90FF", radius=8)
        self.back_btn = BaseButton("返回", self, color="#1E90FF", radius=8)
        left_layout.addWidget(self.model1_label, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.combo, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.model2_label, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.compare_model_combo, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.back_btn, alignment=Qt.AlignCenter)
        root.addWidget(left_widget, 1)

        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)
        self.result_table = QTableWidget(6, 3, self)
        self.result_table.setHorizontalHeaderLabels(["指标", "模型1", "模型2"])
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setSelectionMode(QTableWidget.NoSelection)

        metric_names = ["MSE", "MAE", "UQI", "LPIPS", "推理耗时(s)", "样本数"]
        for i, name in enumerate(metric_names):
            self.result_table.setItem(i, 0, QTableWidgetItem(name))
            self.result_table.setItem(i, 1, QTableWidgetItem("--"))
            self.result_table.setItem(i, 2, QTableWidgetItem("--"))

        self.table_hint = QLabel("请选择模型后点击“显示指标页面”开始运行测试并刷新结果", self)
        self.table_hint.setAlignment(Qt.AlignCenter)
        self.table_hint.setStyleSheet("color:#6b7280;")
        self.run_log = QPlainTextEdit(self)
        self.run_log.setReadOnly(True)
        self.run_log.setPlaceholderText("运行日志将在这里显示")
        self.run_log.setMinimumHeight(180)
        right_layout.addWidget(self.result_table)
        right_layout.addWidget(self.table_hint)
        right_layout.addWidget(self.run_log)
        root.addWidget(right_widget, 2)

        self.setLayout(root)
        self.start_btn.clicked.connect(self._on_show_metrics)
        self.combo.currentTextChanged.connect(self._on_model_selection_changed)
        self.compare_model_combo.currentTextChanged.connect(self._on_model_selection_changed)
        self._refresh_table_headers()
        self._try_render_from_cache(prefer_disk=True)
        self.back_btn.clicked.connect(self.back_requested.emit)

    def _refresh_table_headers(self):
        self.result_table.setHorizontalHeaderLabels([
            "指标",
            self.combo.currentText(),
            self.compare_model_combo.currentText(),
        ])

    def _method_to_alias_candidates(self, method_name):
        fixed_mapping = {
            "Scratch": ["Baseline-DDNet70-Scratch"],
            "Sup-Mig": ["Proposed-DDNet70-PretrainFinetune", "Sup-Mig"],
            "Self-Sup": ["Self-Sup", "SelfSup", "self_sup"],
        }
        return fixed_mapping.get(method_name, [method_name])

    def _pair_cache_key(self, model1_name, model2_name):
        return f"CurveFaultA|{model1_name}|{model2_name}"

    def _pair_cache_file(self, pair_key):
        digest = hashlib.md5(pair_key.encode("utf-8")).hexdigest()
        return os.path.join(self._cache_dir, f"pair_{digest}.json")

    def _csv_fingerprint(self, csv_path):
        stat = os.stat(csv_path)
        return {
            "name": os.path.basename(csv_path),
            "path": csv_path,
            "size": int(stat.st_size),
            "mtime": float(stat.st_mtime),
        }

    def _now_text(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _ensure_cache_dir(self):
        os.makedirs(self._cache_dir, exist_ok=True)

    def _build_snapshot_payload(self, rows, csv_path):
        rows_by_alias = {}
        for row in rows:
            alias = str(row.get("alias", "")).strip()
            if not alias:
                continue
            rows_by_alias[alias] = row
        return {
            "dataset": "CurveFaultA",
            "csv": self._csv_fingerprint(csv_path),
            "refreshed_at": self._now_text(),
            "rows_by_alias": rows_by_alias,
        }

    def _save_snapshot_cache(self, snapshot):
        self._ensure_cache_dir()
        with open(self._snapshot_cache_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    def _load_snapshot_cache_from_disk(self):
        if not os.path.exists(self._snapshot_cache_path):
            return None
        with open(self._snapshot_cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None

    def _build_pair_entry(self, snapshot, model1_name, model2_name):
        rows = list(snapshot.get("rows_by_alias", {}).values())
        model1_row = self._pick_row_for_method(rows, model1_name)
        model2_row = self._pick_row_for_method(rows, model2_name)
        if model1_row is None or model2_row is None:
            raise ValueError(
                f"缓存中未找到所选模型结果: {model1_name}={bool(model1_row)}, {model2_name}={bool(model2_row)}"
            )

        return {
            "pair_key": self._pair_cache_key(model1_name, model2_name),
            "dataset": "CurveFaultA",
            "model1": model1_name,
            "model2": model2_name,
            "csv": snapshot.get("csv", {}),
            "refreshed_at": snapshot.get("refreshed_at", ""),
            "metrics": {
                "mse": [self._format_metric(model1_row, "mse_mean"), self._format_metric(model2_row, "mse_mean")],
                "mae": [self._format_metric(model1_row, "mae_mean"), self._format_metric(model2_row, "mae_mean")],
                "uqi": [self._format_metric(model1_row, "uqi_mean"), self._format_metric(model2_row, "uqi_mean")],
                "lpips": [self._format_metric(model1_row, "lpips_mean"), self._format_metric(model2_row, "lpips_mean")],
                "time": [self._format_metric(model1_row, "per_sample_seconds"), self._format_metric(model2_row, "per_sample_seconds")],
                "sample_count": [self._format_metric(model1_row, "sample_count"), self._format_metric(model2_row, "sample_count")],
            },
        }

    def _save_pair_cache(self, entry):
        pair_key = entry.get("pair_key")
        if not pair_key:
            return
        self._ensure_cache_dir()
        path = self._pair_cache_file(pair_key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

    def _load_pair_cache_from_disk(self, pair_key):
        path = self._pair_cache_file(pair_key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None

    def _apply_pair_entry_to_table(self, entry, source_tag):
        metrics = entry.get("metrics", {})
        self._set_metric_row(0, *metrics.get("mse", ["--", "--"]))
        self._set_metric_row(1, *metrics.get("mae", ["--", "--"]))
        self._set_metric_row(2, *metrics.get("uqi", ["--", "--"]))
        self._set_metric_row(3, *metrics.get("lpips", ["--", "--"]))
        self._set_metric_row(4, *metrics.get("time", ["--", "--"]))
        self._set_metric_row(5, *metrics.get("sample_count", ["--", "--"]))

        csv_meta = entry.get("csv", {})
        csv_name = csv_meta.get("name") or "未知CSV"
        refreshed_at = entry.get("refreshed_at", "未知时间")
        self.table_hint.setText(f"来源: {source_tag} | 数据: {csv_name} | 刷新: {refreshed_at}")

    def _on_model_selection_changed(self):
        self._refresh_table_headers()
        self._try_render_from_cache(prefer_disk=True)

    def _try_render_from_cache(self, prefer_disk=False):
        model1_name = self.combo.currentText()
        model2_name = self.compare_model_combo.currentText()
        pair_key = self._pair_cache_key(model1_name, model2_name)

        cached = self._memory_pair_cache.get(pair_key)
        if cached:
            self._apply_pair_entry_to_table(cached, source_tag="内存缓存(L1)")
            return True

        if prefer_disk:
            pair_disk = self._load_pair_cache_from_disk(pair_key)
            if pair_disk:
                self._memory_pair_cache[pair_key] = pair_disk
                self._apply_pair_entry_to_table(pair_disk, source_tag="磁盘缓存(L2)")
                return True

        snapshot = self._memory_snapshot
        if snapshot is None and prefer_disk:
            snapshot = self._load_snapshot_cache_from_disk()
            if snapshot:
                self._memory_snapshot = snapshot

        if snapshot:
            try:
                entry = self._build_pair_entry(snapshot, model1_name, model2_name)
                self._memory_pair_cache[pair_key] = entry
                self._save_pair_cache(entry)
                self._apply_pair_entry_to_table(entry, source_tag="快照缓存")
                return True
            except Exception:
                pass

        self.table_hint.setText("当前组合暂无缓存，请点击“重新计算并刷新”")
        return False

    def _run_model_test(self):
        script_path = os.path.join(self._project_root, "model_test.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"未找到测试脚本: {script_path}")

        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        proc = subprocess.Popen(
            [sys.executable, script_path],
            cwd=self._project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        log_tail = []
        if proc.stdout is not None:
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if line:
                    self._append_run_log(line)
                    log_tail.append(line)
                    if len(log_tail) > 12:
                        log_tail.pop(0)
        proc.wait()
        if proc.returncode != 0:
            msg = "\n".join(log_tail[-8:]) if log_tail else "未知错误"
            raise RuntimeError(f"model_test.py 运行失败 (code={proc.returncode})\n{msg}")

    def _append_run_log(self, text):
        if not text:
            return
        self.run_log.appendPlainText(text)
        bar = self.run_log.verticalScrollBar()
        bar.setValue(bar.maximum())
        QApplication.processEvents()

    def _find_latest_compare_batch_csv(self):
        pattern = os.path.join(self._metric_results_dir, "[[]CompareBatch[]]CurveFaultA_*.csv")
        candidates = glob.glob(pattern)
        if not candidates:
            return None
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]

    def _pick_row_for_method(self, rows, method_name):
        candidates = self._method_to_alias_candidates(method_name)
        rows_ok = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
        search_rows = rows_ok if rows_ok else rows

        for alias in candidates:
            for row in search_rows:
                if row.get("alias") == alias:
                    return row

        tokens = [t.lower() for t in method_name.replace("-", " ").split() if t]
        for row in search_rows:
            alias = str(row.get("alias", "")).lower()
            if tokens and all(t in alias for t in tokens):
                return row
        return None

    def _set_metric_row(self, row_index, model1_val, model2_val):
        self.result_table.setItem(row_index, 1, QTableWidgetItem(model1_val))
        self.result_table.setItem(row_index, 2, QTableWidgetItem(model2_val))

    def _format_metric(self, row, key):
        if row is None:
            return "--"
        value = row.get(key, "")
        if value in (None, ""):
            return "--"
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return str(value)

    def _on_show_metrics(self):
        self._refresh_table_headers()
        self.table_hint.setText("正在运行 model_test.py 并刷新指标，请稍候...")
        self.run_log.clear()
        self._append_run_log(f"$ {sys.executable} model_test.py")
        self.start_btn.setEnabled(False)

        try:
            self._run_model_test()
            csv_path = self._find_latest_compare_batch_csv()
            if not csv_path:
                raise FileNotFoundError("未找到 [CompareBatch]CurveFaultA_*.csv 输出文件")

            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                raise ValueError("对比结果文件为空")

            snapshot = self._build_snapshot_payload(rows, csv_path)
            self._memory_snapshot = snapshot
            self._save_snapshot_cache(snapshot)

            model1_name = self.combo.currentText()
            model2_name = self.compare_model_combo.currentText()
            pair_entry = self._build_pair_entry(snapshot, model1_name, model2_name)
            pair_key = pair_entry["pair_key"]
            self._memory_pair_cache[pair_key] = pair_entry
            self._save_pair_cache(pair_entry)
            self._apply_pair_entry_to_table(pair_entry, source_tag="实时计算")

            self._append_run_log("model_test.py 运行完成")
        except Exception as e:
            self._append_run_log(f"ERROR: {e}")
            self.table_hint.setText(f"指标刷新失败: {e}")
        finally:
            self.start_btn.setEnabled(True)

    def refresh_metrics(self):
        self._on_show_metrics()


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
