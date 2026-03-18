"""
Minimal demo host app to showcase embedding SystemReminderContainer.
Run with: python ddnet-main/demo/qt_host_demo.py
"""
import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer

# Add the modules directory to import system_reminder as a package
modules_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'modules'))
if modules_dir not in sys.path:
    sys.path.insert(0, modules_dir)

try:
    from modules.system_reminder import SystemReminderContainer
except Exception:
    print("Failed to import SystemReminderContainer from system_reminder. Ensure PYTHONPATH is set correctly.")
    raise

def main():
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("欢迎使用")
    container = SystemReminderContainer()
    window.setCentralWidget(container)

    # Provide some context (future use)
    container.set_context({"host": "demo", "theme": "light"})
    # Initial content
    container.update_content("基于U-Net的地震反演系统")

    window.resize(600, 400)
    window.show()

    # Optional: simulate a content update after 3 seconds
    def update_later():
        container.update_content("更新：系统正在准备切换到新对比模型，界面将保持当前布局。")

    QTimer.singleShot(3000, update_later)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
