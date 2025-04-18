#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BatterySuite 1.0
３機能統合版 GUI（PyQt5）
"""

import sys, os, importlib.util, traceback
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QMessageBox

# ---------- 動的インポートで既存クラスを取得 ----------
def load_class(file_name, class_name):
    try:
        path = os.path.join(os.path.dirname(__file__), file_name)
        spec = importlib.util.spec_from_file_location(class_name, path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, class_name)
    except Exception as e:
        error_msg = f"{file_name} の読み込みに失敗しました。\n\nエラー内容:\n{str(e)}\n\n詳細:\n{traceback.format_exc()}"
        QMessageBox.critical(None, "モジュール読み込みエラー", error_msg)
        raise  # 起動を止める

# ---------- メインウィンドウ ----------
class BatterySuite(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Battery Analysis Suite")
        self.resize(1280, 820)

        tabs = QTabWidget()
        try:
            CyclePlotterWidget = load_class("DisChg_Chg_visualizer_ver1.79.py", "CyclePlotterWidget")
            CycleAnalysisGUI   = load_class("CycleAnalysisl.py", "CycleAnalysisGUI")
            CellSpecEditor     = load_class("INFO_ver2.0.py", "CellSpecEditor")

            tabs.addTab(self.wrap(CyclePlotterWidget()), "Plot & Export")
            tabs.addTab(self.wrap(CycleAnalysisGUI ()), "Cycle Analyzer")
            tabs.addTab(self.wrap(CellSpecEditor   ()), "Cell Spec Editor")
        except Exception:
            sys.exit(1)  # 起動失敗したら終了

        self.setCentralWidget(tabs)

    @staticmethod
    def wrap(widget: QWidget):
        w = QWidget()
        lay = QVBoxLayout(w); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(widget)
        return w

# ---------- 起動 ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = app.font(); font.setPointSize(14); app.setFont(font)
    wnd = BatterySuite(); wnd.show()
    sys.exit(app.exec_())
