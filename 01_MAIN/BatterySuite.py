#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BatterySuite 1.0
３機能統合版 GUI（PyQt5）
  Tab 1 : Cycle Plotter & Excel Export
  Tab 2 : Cycle Analyzer Table
  Tab 3 : Cell‑Spec Editor
------------------------------------------------
既存ファイル : DisChg_Chg_visualizer_ver1.79.py
               CycleAnalysisl.py
               INFO_ver2.0.py
同じフォルダに置いたまま、本スクリプトを実行してください。
"""

import sys, os, importlib.util
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout

# ---------- 動的インポートで既存クラスを取得 ----------
def load_class(file_name, class_name):
    path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(class_name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)

CyclePlotterWidget = load_class("DisChg_Chg_visualizer_ver1.79.py", "CyclePlotterWidget")
CycleAnalysisGUI   = load_class("CycleAnalysisl.py",                    "CycleAnalysisGUI")
CellSpecEditor     = load_class("INFO_ver2.0.py",                       "CellSpecEditor")

# ---------- メインウィンドウ ----------
class BatterySuite(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Battery Analysis Suite")
        self.resize(1280, 820)

        tabs = QTabWidget()
        tabs.addTab(self.wrap(CyclePlotterWidget()), "Plot & Export")
        tabs.addTab(self.wrap(CycleAnalysisGUI ()), "Cycle Analyzer")
        tabs.addTab(self.wrap(CellSpecEditor   ()), "Cell Spec Editor")

        self.setCentralWidget(tabs)

    @staticmethod
    def wrap(widget: QWidget):
        """与えられた QWidget をレイアウト付きで包む."""
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
