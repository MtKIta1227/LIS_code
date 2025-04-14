import sys
import subprocess
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel

class Launcher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GUI Launcher")
        self.setGeometry(300, 300, 300, 150)

        layout = QVBoxLayout()
        label = QLabel("起動したいGUIを選択してください")
        layout.addWidget(label)

        self.analysis_btn = QPushButton("📋 Cycle Analysis GUI")
        self.analysis_btn.clicked.connect(self.launch_analysis_gui)
        layout.addWidget(self.analysis_btn)

        self.visualizer_btn = QPushButton("📊 Dis/Chg Visualizer GUI")
        self.visualizer_btn.clicked.connect(self.launch_visualizer_gui)
        layout.addWidget(self.visualizer_btn)

        self.setLayout(layout)

        # 実行ファイルのディレクトリ（現在のフォルダ）
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

    def launch_analysis_gui(self):
        target = os.path.join(self.base_dir, "CycleAnalysisl.py")
        subprocess.Popen([sys.executable, target])

    def launch_visualizer_gui(self):
        target = os.path.join(self.base_dir, "DisChg_Chg_visualizer_ver1.71.py")
        subprocess.Popen([sys.executable, target])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    launcher = Launcher()
    launcher.show()
    sys.exit(app.exec_())
