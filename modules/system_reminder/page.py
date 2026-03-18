from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QPainter
import os
from .base_button import BaseButton
from .ui_styles import button_style
from func.utils import plot_loss_from_npy


class LossPanel(QWidget):
    """Panel that displays a single dataset loss curve.
    It loads a loss.npy corresponding to the selected dataset and renders
    the curve using the shared plotting utility. The image is shown in a QLabel.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.title_label = QLabel("Loss Curve", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        font = self.title_label.font()
        font.setPointSize(12)
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        # Placeholder until a real loss curve is loaded
        placeholder = self._placeholder_pix()
        self.image_label.setPixmap(placeholder)
        layout.addWidget(self.image_label)

        self.status_label = QLabel(self)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def load_loss(self, dataset_name: str) -> None:

        self.status_label.setText("加载中…")
        # TODO loss折线图的路径
        npy_path = f"/<system-reminder>/{dataset_name}_loss.npy"
        try:
            # Use the shared utility to generate the loss plot
            plot_loss_from_npy(npy_path, save_path=None, show=False)
            # Try to load the generated image from a common location
            img_path = self._resolve_latest_image_path(dataset_name)
            if img_path and os.path.exists(img_path):
                self.image_label.setPixmap(QPixmap(img_path))
                self.status_label.setText("加载完成")
            else:
                self.image_label.setPixmap(self._placeholder_pix())
                self.status_label.setText("曲线图未生成，或未找到图像文件")
        except Exception as e:
            self.image_label.setPixmap(self._placeholder_pix())
            self.status_label.setText(f"加载失败: {str(e)}")

    def _resolve_latest_image_path(self, dataset_name: str) -> str:
        # Try a few common save locations produced by plot_loss_from_npy.
        base_dir = os.path.join(os.getcwd(), "system_reminder")
        # 1) direct expected path
        direct = os.path.join(base_dir, f"{dataset_name}_loss.npy_loss_curve.png")
        if os.path.exists(direct):
            return direct
        # 2) any file containing dataset name and ending with _loss_curve.png
        if os.path.isdir(base_dir):
            for fname in os.listdir(base_dir):
                if dataset_name.lower() in fname.lower() and fname.endswith("_loss_curve.png"):
                    return os.path.join(base_dir, fname)
        # 3) any loss curve image in the base dir
        if os.path.isdir(base_dir):
            for fname in os.listdir(base_dir):
                if fname.endswith("_loss_curve.png"):
                    return os.path.join(base_dir, fname)
        return ""

    def _placeholder_pix(self) -> QPixmap:
        pix = QPixmap(640, 360)
        pix.fill(Qt.lightGray)
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


class TrainingView(QWidget):
    train_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
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
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        # pix = self._load_placeholder_image()
        # self.image_label.setPixmap(pix)
        right_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        root.addWidget(right_widget, 2)
        self.setLayout(root)
        self.start_btn.clicked.connect(self._on_show_loss)
        self.back_btn.clicked.connect(self.back_requested.emit)

    def _on_show_loss(self):
        dataset = self.combo.currentText()
        if dataset:
            self.loss_panel.load_loss(dataset)

    # def _load_placeholder_image(self):
    #     base = os.path.dirname(__file__)
    #     path = os.path.join(base, 'resources', 'system_reminder.png')
    #     if os.path.exists(path):
    #         pix = QPixmap(path)
    #         if not pix.isNull():
    #             return pix
    #     pix = QPixmap(320, 240)
    #     pix.fill(Qt.white)
    #     painter = QPainter(pix)
    #     painter.setPen(Qt.black)
    #     font = QFont()
    #     font.setPointSize(14)
    #     painter.setFont(font)
    #     painter.drawText(pix.rect(), Qt.AlignCenter, "<system-reminder>")
    #     painter.end()
    #     return pix


class CompareView(QWidget):
    train_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(150, 150)  # 控制图片区域大小
        self.image_label.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap("placeholder.png")
        self.image_label.setPixmap(
            pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

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

        left_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.combo, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)
        left_layout.addLayout(btn_row)
        left_layout.addWidget(self.back_btn, alignment=Qt.AlignCenter)

        root.addWidget(left_widget)

        right_widget = QWidget(self)
        right_layout = QVBoxLayout(right_widget)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        pix = self._load_placeholder_image()
        self.image_label.setPixmap(pix)
        right_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        root.addWidget(right_widget, 2)
        self.setLayout(root)
        self.start_btn.clicked.connect(self.train_requested.emit)
        self.back_btn.clicked.connect(self.back_requested.emit)


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
