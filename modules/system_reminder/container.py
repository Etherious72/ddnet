from PyQt5.QtWidgets import QWidget, QVBoxLayout, QStackedWidget
from .page import TrainingView, CompareView
from PyQt5.QtCore import pyqtSignal
from .page import SystemReminderPage


class SystemReminderContainer(QWidget):
    """
    对外提供接口
    Embeddable container for the System Reminder page.
    Exposes a minimal, future-proof API for host applications.
    """
    # Signals for future integration
    content_updated = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    # Placeholder signals to propagate button actions to host (future wiring)
    train_requested = pyqtSignal()
    compare_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._context = {}
        self._page = SystemReminderPage(self)
        self._training_view = TrainingView(self)
        self._compare_view = CompareView(self)
        self.stack = QStackedWidget(self)
        self.stack.addWidget(self._page)
        self.stack.addWidget(self._training_view)
        self.stack.addWidget(self._compare_view)
        layout = QVBoxLayout(self)
        layout.addWidget(self.stack)
        self.setLayout(layout)
        # Internal navigation wiring
        self._page.train_requested.connect(self._show_train)
        self._page.compare_requested.connect(self._show_compare)
        self._training_view.back_requested.connect(self._show_home)
        self._compare_view.back_requested.connect(self._show_home)

    def _show_home(self):
        self.stack.setCurrentWidget(self._page)

    def set_context(self, context: dict) -> None:
        """
        Provide environment/context information to the module.
        """
        self._context = context or {}
        # In the current simple design, context is not used to render the static page.
        # This method is reserved for future enhancements.
        return None

    def update_content(self, text: str) -> None:
        """
        Update the displayed content. Future use for dynamic model comparison results.
        """
        if text is None:
            text = ""
        self._page.update_content(text)
        self.content_updated.emit({"text": text})
    # Internal navigation helpers
    def _show_train(self):
        self.stack.setCurrentWidget(self._training_view)
        self.train_requested.emit()

    def _show_compare(self):
        self.stack.setCurrentWidget(self._compare_view)
        self.compare_requested.emit()
