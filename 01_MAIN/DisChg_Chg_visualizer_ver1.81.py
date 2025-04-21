#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
複数サンプル比較ツール

【主な機能】
- 「Load Samples」ボタンで、選択フォルダ内にサブフォルダがあれば複数サンプル、
  なければ1サンプルとして読み込み
- 左ペイン上部にフィルター（サンプル名／Cycle／選択状態）を配置し、サンプル一覧を絞り込み可能
  ※ マルチサンプルの場合、リスト項目は「SampleName:Cycle」となり、フィルターは各部分のみ対象
- 左ペイン下部にINFOテキストボックスを配置し、リストで選択中のサンプルに対応するINFOファイル（INFOフォルダ内のSampleName.txt）の内容を表示
- 右ペインはグラフ領域（Qtツールバー付き）
- 下部パネルに各種操作ボタン（全選択、個別プロット、範囲指定プロット、効率プロット、Excel出力、MONOモード切替、INFO保存）を配置
"""

import sys
import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from PyQt5.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QFileDialog, QListWidget,
    QListWidgetItem, QVBoxLayout, QHBoxLayout, QInputDialog, QMessageBox, QProgressBar,
    QTextEdit, QLineEdit, QLabel, QSplitter, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics

# Excel用の列番号→列記号変換関数
def col_idx_to_excel_col(idx):
    result = ""
    while idx >= 0:
        result = chr(ord('A') + (idx % 26)) + result
        idx = idx // 26 - 1
    return result

class CyclePlotterWidget(QWidget):
    def __init__(self):
        super().__init__()
        # データ保持用: { sample_name: { cycle: DataFrame } }
        self.data_by_sample = {}
        self.sample_paths = {}   # 各サンプルのフォルダパス
        self.multi_sample_mode = False  # False: 1サンプル, True: 複数サンプル
        self.last_plotted_cycles = []   # プロット対象の識別子（各リスト項目に保存した (sample, cycle) タプル）
        self.sample_colors = {}         # 複数サンプル時に各サンプルへ割り当てる色
        
        # デフォルト設定
        self.current_cmap = "viridis"   # 内部利用用
        self.mono_mode = False

        self.initUI()

    def initUI(self):
        # ---------------------
        # 上部ツールバー：Load Samplesボタン
        # ---------------------
        self.load_samples_btn = QPushButton("Load Samples")
        self.add_sample_btn = QPushButton("Add Sample")
        self.delete_sample_btn = QPushButton("Delete Sample")
        top_toolbar = QHBoxLayout()
        top_toolbar.addWidget(self.load_samples_btn)
        top_toolbar.addWidget(self.add_sample_btn)
        top_toolbar.addWidget(self.delete_sample_btn)
        top_toolbar.addStretch()

        # ---------------------
        # 左ペイン：フィルター領域、サンプル一覧（リスト）、INFOテキストボックス
        # ---------------------
        self.sample_filter_edit = QLineEdit()
        self.sample_filter_edit.setPlaceholderText("Sample")
        self.sample_filter_edit.setMinimumWidth(180)
        self.cycle_filter_edit = QLineEdit()
        self.cycle_filter_edit.setPlaceholderText("Cycle")
        self.cycle_filter_edit.setMinimumWidth(180)
        self.selection_filter_combo = QComboBox()
        self.selection_filter_combo.addItems(["All", "Selected", "Not Selected"])
        self.selection_filter_combo.setFixedWidth(180)

        from PyQt5.QtWidgets import QGridLayout
        filter_layout = QGridLayout()
        filter_layout.addWidget(QLabel("Sample:"), 0, 0)
        filter_layout.addWidget(self.sample_filter_edit, 0, 1, 1, 3)
        filter_layout.addWidget(QLabel("Cycle:"), 1, 0)
        filter_layout.addWidget(self.cycle_filter_edit, 1, 1, 1, 3)
        filter_layout.addWidget(QLabel("Selection:"), 2, 0)
        filter_layout.addWidget(self.selection_filter_combo, 2, 1, 1, 3)

        self.list_widget = QListWidget()
        self.info_textbox = QTextEdit()
        self.info_textbox.setReadOnly(False)
        self.info_textbox.setPlaceholderText("INFO text for selected sample will appear here.")

        left_layout = QVBoxLayout()
        left_layout.addLayout(filter_layout)
        left_layout.addWidget(self.list_widget)
        left_layout.addWidget(QLabel("INFO"))
        left_layout.addWidget(self.info_textbox)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setMaximumWidth(300)  # 最大幅を制限

        # ---------------------
        # 右ペイン：グラフ領域のみ（QTツールバー付き）
        # ---------------------
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 8)))
        toolbar = NavigationToolbar(self.canvas, self)
        graph_widget = QWidget()
        graph_layout = QVBoxLayout()
        graph_layout.addWidget(toolbar)
        graph_layout.addWidget(self.canvas)
        graph_widget.setLayout(graph_layout)
        right_widget = graph_widget

        # ---------------------
        # 中央領域：左右ペインをSplitterで配置
        # ---------------------
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        # ---------------------
        # 下部パネル：各種操作ボタン
        # ---------------------
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        self.all_plot_btn = QPushButton("Plot All")
        self.select_plot_btn = QPushButton("Plot Selected")
        self.range_plot_btn = QPushButton("Plot Range")
        self.dis_cap_btn = QPushButton("Show Efficiency")
        self.to_excel_btn = QPushButton("Export Excel")
        self.monoqlo_btn = QPushButton("MONOQLO OFF")
        self.info_save_btn = QPushButton("Save INFO")

        bottom_panel = QHBoxLayout()
        bottom_panel.addWidget(self.select_all_btn)
        bottom_panel.addWidget(self.deselect_all_btn)
        bottom_panel.addWidget(self.all_plot_btn)
        bottom_panel.addWidget(self.select_plot_btn)
        bottom_panel.addWidget(self.range_plot_btn)
        bottom_panel.addWidget(self.dis_cap_btn)
        bottom_panel.addWidget(self.to_excel_btn)
        bottom_panel.addWidget(self.monoqlo_btn)
        bottom_panel.addWidget(self.info_save_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_toolbar)
        main_layout.addWidget(main_splitter)
        main_layout.addLayout(bottom_panel)
        self.setLayout(main_layout)

        # ---------------------
        # シグナル接続
        # ---------------------
        self.load_samples_btn.clicked.connect(self.load_samples)
        self.select_all_btn.clicked.connect(self.select_all_items)
        self.deselect_all_btn.clicked.connect(self.deselect_all_items)
        self.all_plot_btn.clicked.connect(self.plot_all)
        self.select_plot_btn.clicked.connect(self.plot_selected)
        self.range_plot_btn.clicked.connect(self.plot_range_cycles)
        self.dis_cap_btn.clicked.connect(self.plot_dis_cap_efficiency)
        self.to_excel_btn.clicked.connect(self.export_to_excel)
        self.monoqlo_btn.clicked.connect(self.toggle_mono_mode)
        self.add_sample_btn.clicked.connect(self.add_sample)
        self.delete_sample_btn.clicked.connect(self.delete_sample)
        self.info_save_btn.clicked.connect(self.save_info_text)
        self.sample_filter_edit.textChanged.connect(self.filter_list)
        self.cycle_filter_edit.textChanged.connect(self.filter_list)
        self.selection_filter_combo.currentIndexChanged.connect(self.filter_list)
        self.list_widget.itemSelectionChanged.connect(self.update_info_area)

    def shorten_text(self, text, max_width):
        font_metrics = QFontMetrics(self.font())
        return font_metrics.elidedText(text, Qt.ElideRight, max_width)

    def shorten_label(self, text, first_chars=10, last_chars=5):
        # ラベルが長い場合、先頭first_chars文字＋"..."＋後ろlast_chars文字にする
        if len(text) > (first_chars + last_chars):
            return text[:first_chars] + "..." + text[-last_chars:]
        return text

    def select_all_items(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked)

    def deselect_all_items(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Unchecked)

    def update_status(self, message=""):
        main_window = self.window()
        if isinstance(main_window, QMainWindow):
            parts = []
            if self.multi_sample_mode:
                parts.append("Multiple Samples Loaded")
            else:
                if self.data_by_sample:
                    sample_name = list(self.data_by_sample.keys())[0]
                    parts.append(sample_name)
            if self.mono_mode:
                parts.append("MONOQLO Mode")
            if message:
                parts.append(message)
            main_window.statusBar().showMessage(" | ".join(parts))
        else:
            print(message)

    def filter_list(self):
        sample_filter = self.sample_filter_edit.text().lower()
        cycle_filter = self.cycle_filter_edit.text().lower()
        selection_filter = self.selection_filter_combo.currentText()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            data = item.data(Qt.UserRole)
            if data:
                sample, cycle = data
                text = f"{sample}:{cycle}"
            else:
                text = item.text()
            if self.multi_sample_mode:
                try:
                    sample_part, cycle_part = text.split(":", 1)
                except ValueError:
                    sample_part = text
                    cycle_part = ""
            else:
                sample_part = ""
                cycle_part = text
            sample_match = (sample_filter in sample_part.lower().strip()) if self.multi_sample_mode else True
            cycle_match = cycle_filter in cycle_part
            if selection_filter == "Selected":
                sel_match = (item.checkState() == Qt.Checked)
            elif selection_filter == "Not Selected":
                sel_match = (item.checkState() != Qt.Checked)
            else:
                sel_match = True
            item.setHidden(not (sample_match and cycle_match and sel_match))

    def update_info_area(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            self.info_textbox.clear()
            return
        data = selected_items[0].data(Qt.UserRole)
        if data:
            sample, _ = data
        else:
            sample = list(self.data_by_sample.keys())[0] if self.data_by_sample else ""
        info_dir = os.path.join(os.getcwd(), "INFO")
        info_path = os.path.join(info_dir, f"{sample}.txt")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info_str = f.read()
        else:
            info_str = f"No INFO file found for sample: {sample}"
        self.info_textbox.setPlainText(info_str)

    def load_sample_from_folder(self, sample_name, folder):
        self.data_by_sample[sample_name] = {}
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.upper().endswith(".CSV")]
        main_window = self.window()
        progress_bar = None
        if isinstance(main_window, QMainWindow):
            for widget in main_window.statusBar().findChildren(QProgressBar):
                progress_bar = widget
                break
        if progress_bar:
            progress_bar.setMaximum(len(files))
            progress_bar.setValue(0)
            progress_bar.setVisible(True)
        for file in sorted(files):
            try:
                cycle = str(int(os.path.splitext(os.path.basename(file))[0]))
                df = pd.read_csv(file, encoding="shift_jis", skiprows=3)
                df.columns = ["Mode", "Voltage(V)", "Capacity(mAh/g)", "dV(V)", "dQ(mAh/g)", "dQ/dV"]
                df = df[df["Mode"].isin(["DIS", "CHG"])]
                df["Cycle"] = cycle
                self.data_by_sample[sample_name][cycle] = df
                
                item = QListWidgetItem()
                item.setData(Qt.UserRole, (sample_name, cycle))
                display_sample = self.shorten_text(sample_name, 150)
                if self.multi_sample_mode or (len(self.data_by_sample) > 1):
                    display_text = f"{display_sample}:{cycle}"
                else:
                    display_text = f"{cycle}"
                item.setText(display_text)
                item.setToolTip(f"{sample_name} - Cycle {cycle}")
                item.setCheckState(Qt.Unchecked)
                self.list_widget.addItem(item)
            except Exception as e:
                print(f"Error loading {file} in {sample_name}: {e}")
            if progress_bar:
                progress_bar.setValue(progress_bar.value() + 1)
        if progress_bar:
            progress_bar.setVisible(False)

    def update_list_display(self):
        if self.multi_sample_mode:
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                data = item.data(Qt.UserRole)
                if data:
                    sample, cycle = data
                    item.setText(f"{sample}:{cycle}")

    def load_samples(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Sample(s)")
        if not folder:
            return
        self.data_by_sample.clear()
        self.list_widget.clear()
        self.sample_paths.clear()
        subdirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        multi = False
        for d in subdirs:
            subfolder = os.path.join(folder, d)
            if any(f.upper().endswith(".CSV") for f in os.listdir(subfolder)):
                multi = True
                break
        self.multi_sample_mode = multi
        if multi:
            sample_color_list = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'magenta', 'cyan']
            sample_index = 0
            for d in sorted(subdirs):
                subfolder = os.path.join(folder, d)
                if not any(f.upper().endswith(".CSV") for f in os.listdir(subfolder)):
                    continue
                self.sample_paths[d] = subfolder
                self.sample_colors[d] = sample_color_list[sample_index % len(sample_color_list)]
                sample_index += 1
                self.load_sample_from_folder(d, subfolder)
            self.update_list_display()
        else:
            sample_name = os.path.basename(folder)
            self.sample_paths[sample_name] = folder
            self.load_sample_from_folder(sample_name, folder)
        self.update_status("Samples loaded.")
        self.update_info_area()

    def plot_all(self):
        self.select_all_items()
        all_ids = [self.list_widget.item(i).data(Qt.UserRole) for i in range(self.list_widget.count())]
        self.plot_data(all_ids)

    def plot_selected(self):
        selected_ids = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected_ids.append(item.data(Qt.UserRole))
        if not selected_ids:
            QMessageBox.warning(self, "No selection", "Please check at least one cycle/sample item.")
            return
        self.plot_data(selected_ids)

    def plot_range_cycles(self):
        reply = QMessageBox.question(
            self, "Confirm", "Deselect all before range selection?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.deselect_all_items()
        else:
            return

        text, ok = QInputDialog.getText(self, "Input Cycle Range", "Enter cycle numbers (e.g. 1-3,6,8):")
        if not ok or not text:
            return
        pattern = re.compile(r'(\d+)(?:-(\d+))?')
        selected_set = set()
        for part in text.split(','):
            match = pattern.fullmatch(part.strip())
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else start
                selected_set.update(str(i) for i in range(start, end+1))
        valid_ids = []
        if self.multi_sample_mode:
            for sample in self.data_by_sample:
                for cycle in self.data_by_sample[sample]:
                    if cycle in selected_set:
                        valid_ids.append((sample, cycle))
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                data = item.data(Qt.UserRole)
                if data:
                    _, c = data
                    if c in selected_set:
                        item.setCheckState(Qt.Checked)
        else:
            sample = list(self.data_by_sample.keys())[0]
            for cycle in self.data_by_sample[sample]:
                if cycle in selected_set:
                    valid_ids.append((sample, cycle))
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                if item.text() in selected_set:
                    item.setCheckState(Qt.Checked)
        if not valid_ids:
            QMessageBox.warning(self, "Error", "No matching cycles found")
            return
        self.plot_data(valid_ids)

    def plot_data(self, ids):
        axis_label_size = 13
        tick_label_size = 11
        title_size = 14
        legend_size = 10
        self.canvas.figure.clf()
        self.ax = self.canvas.figure.add_subplot(111)
        if not self.multi_sample_mode:
            cmap_used = plt.cm.get_cmap(self.current_cmap)
        for id_item in ids:
            if isinstance(id_item, (tuple, list)) and len(id_item) == 2:
                sample, cycle = id_item
                df = self.data_by_sample.get(sample, {}).get(cycle)
                if df is None:
                    continue
                try:
                    cycle_num = int(cycle)
                except:
                    cycle_num = 1
                if self.multi_sample_mode:
                    base_color = self.sample_colors.get(sample, "gray")
                    try:
                        alpha = 1.0 if self.mono_mode else 0.3 + 0.07 * (cycle_num % 10)
                    except:
                        alpha = 1.0
                    color = "black" if self.mono_mode else base_color
                    label = f"{sample} Cycle {cycle}"
                else:
                    group_index = (cycle_num - 1) // 10
                    step_within_group = (cycle_num - 1) % 10
                    cmap_local = matplotlib.cm.get_cmap("tab20", 20)
                    color = "black" if self.mono_mode else cmap_local(group_index % cmap_local.N)
                    alpha = 1.0 if self.mono_mode else 1.0 - 0.07 * step_within_group
                    label = f"Cycle {cycle}"
            else:
                sample = list(self.data_by_sample.keys())[0]
                cycle = id_item
                df = self.data_by_sample[sample].get(cycle)
                if df is None:
                    continue
                try:
                    cycle_num = int(cycle)
                except:
                    cycle_num = 1
                alpha = 1.0 if self.mono_mode else 1.0 - 0.1 * ((cycle_num - 1) % 10)
                color = "black" if self.mono_mode else cmap_used(((cycle_num - 1) % 10) % cmap_used.N)
                label = f"Cycle {cycle}"
            # 短縮版ラベルを生成（前10文字＋"..."＋後5文字）
            short_label = self.shorten_label(label, 10, 5)
            first = True
            for mode in ["DIS", "CHG"]:
                df_mode = df[df["Mode"] == mode]
                self.ax.plot(
                    df_mode["Capacity(mAh/g)"],
                    df_mode["Voltage(V)"],
                    color=color,
                    alpha=alpha,
                    label=short_label if first and mode == "DIS" else None
                )
                first = False
        self.ax.set_xlabel("Capacity (mAh/g)", fontname="Arial", fontsize=axis_label_size)
        self.ax.set_ylabel("Voltage(V)", fontname="Arial", fontsize=axis_label_size)
        self.ax.set_title("Discharge-Charge Curves", fontname="Arial", fontsize=title_size)
        self.ax.tick_params(axis='both', direction='in', labelsize=tick_label_size)
        self.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        handles, labels = self.ax.get_legend_handles_labels()
        ncol = 2 if len(labels) > 20 else 1
        self.ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                       fontsize=legend_size, frameon=True, ncol=ncol)
        self.canvas.figure.subplots_adjust(right=0.8)
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.last_plotted_cycles = ids
        self.update_status(f"Plotted {len(ids)} cycles/samples.")

    def plot_dis_cap_efficiency(self):
        axis_label_size = 13
        tick_label_size = 11
        title_size = 14
        legend_size = 10
        self.canvas.figure.clf()
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax2 = self.ax.twinx()
        if self.multi_sample_mode:
            for sample in sorted(self.data_by_sample.keys()):
                cycles, max_dis_caps, efficiencies = [], [], []
                for cycle in sorted(self.data_by_sample[sample].keys(), key=lambda x: int(x)):
                    df = self.data_by_sample[sample][cycle]
                    dis_df = df[df["Mode"]=="DIS"]
                    chg_df = df[df["Mode"]=="CHG"]
                    if not dis_df.empty and not chg_df.empty:
                        cycles.append(int(cycle))
                        max_dis_caps.append(dis_df["Capacity(mAh/g)"].max())
                        eff = (chg_df["Capacity(mAh/g)"].max() / dis_df["Capacity(mAh/g)"].max()) if dis_df["Capacity(mAh/g)"].max() != 0 else None
                        efficiencies.append(eff)
                if cycles:
                    color = self.sample_colors.get(sample, "black")
                    self.ax.plot(cycles, max_dis_caps, color=color, marker='o', label=f"{sample} DIS")
                    self.ax2.plot(cycles, [(e * 100 if e is not None else None) for e in efficiencies], color=color, marker='x', label=f"{sample} Eff")
            self.ax.set_ylim(0, None)
            self.ax2.set_ylim(0, 110)
        else:
            sample = list(self.data_by_sample.keys())[0]
            cycles, max_dis_caps, efficiencies = [], [], []
            for cycle in sorted(self.data_by_sample[sample].keys(), key=lambda x: int(x)):
                df = self.data_by_sample[sample][cycle]
                dis_df = df[df["Mode"]=="DIS"]
                chg_df = df[df["Mode"]=="CHG"]
                if not dis_df.empty and not chg_df.empty:
                    cycles.append(int(cycle))
                    max_dis_caps.append(dis_df["Capacity(mAh/g)"].max())
                    eff = (chg_df["Capacity(mAh/g)"].max() / dis_df["Capacity(mAh/g)"].max()) if dis_df["Capacity(mAh/g)"].max() != 0 else None
                    efficiencies.append(eff)
            cmap_used = plt.cm.get_cmap(self.current_cmap)
            for i, cycle in enumerate(cycles):
                alpha = 1.0 if self.mono_mode else 1.0 - 0.1 * ((cycle - 1) % 10)
                color = "black" if self.mono_mode else cmap_used(((cycle - 1) % 10) % cmap_used.N)
                self.ax.plot([cycle], [max_dis_caps[i]], color=color, marker='o', label="DIS capacity" if i==0 else "")
                self.ax2.plot([cycle], [efficiencies[i] * 100], color=color, marker='x', label="Coulombic eff." if i==0 else "")
            self.ax.set_ylim(0, max(max_dis_caps) * 1.1)
            self.ax2.set_ylim(0, 110)
        self.ax.set_xlabel("Cycle", fontname="Arial", fontsize=axis_label_size)
        self.ax.set_ylabel("Capacity (mAh/g)", fontname="Arial", fontsize=axis_label_size, color="dodgerblue")
        self.ax2.set_ylabel("Coulombic Efficiency (%)", fontname="Arial", fontsize=axis_label_size, color="darkorange")
        self.ax.tick_params(axis='both', direction='in', labelsize=tick_label_size)
        self.ax2.tick_params(axis='both', direction='in', labelsize=tick_label_size)
        self.ax.set_title("Discharge Capacity and Coulombic Efficiency", fontname="Arial", fontsize=title_size)
        handles1, labels1 = self.ax.get_legend_handles_labels()
        handles2, labels2 = self.ax2.get_legend_handles_labels()
        combined_handles = handles1 + handles2
        combined_labels = labels1 + labels2
        ncol = 2 if len(combined_labels) > 20 else 1
        self.ax.legend(combined_handles, combined_labels, loc="upper center", bbox_to_anchor=(0.5, -0.15),
                       fontsize=legend_size, frameon=True, ncol=ncol)
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.update_status("Displayed capacity and efficiency.")

    def toggle_mono_mode(self):
        self.mono_mode = not self.mono_mode
        if self.mono_mode:
            self.monoqlo_btn.setText("MONOQLO ON")
            self.monoqlo_btn.setStyleSheet("background-color: black; color: white;")
        else:
            self.monoqlo_btn.setText("MONOQLO OFF")
            self.monoqlo_btn.setStyleSheet("")
        self.update_status()

    def export_to_excel(self):
        if not self.last_plotted_cycles:
            QMessageBox.warning(self, "No Data", "No plotted data to export")
            return
    
        # --- Determine default filename ---
        selected_samples = set(
            id_item[0] if isinstance(id_item, (tuple, list)) and len(id_item) == 2
            else list(self.data_by_sample.keys())[0]
            for id_item in self.last_plotted_cycles
        )
        if len(selected_samples) == 1:
            default_name = list(selected_samples)[0]
        else:
            default_name = "MultiSample_Output"
        default_path = os.path.join(os.getcwd(), default_name + ".xlsx")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Excel File", default_path, filter="Excel Files (*.xlsx)"
        )
        if not path:
            return
    
        import xlsxwriter
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            workbook = writer.book
    
            # --- 1) GraphData sheet as before ---
            ws1 = workbook.add_worksheet("GraphData")
            writer.sheets["GraphData"] = ws1
            chart1 = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
            col = 0
            for id_item in self.last_plotted_cycles:
                if isinstance(id_item, (tuple, list)) and len(id_item) == 2:
                    sample, cycle = id_item
                    df = self.data_by_sample[sample][cycle]
                    header = f"{sample} Cycle {cycle}"
                else:
                    sample = list(self.data_by_sample.keys())[0]
                    cycle = id_item
                    df = self.data_by_sample[sample][cycle]
                    header = f"Cycle {cycle}"
    
                df = df[df["Mode"].isin(["DIS", "CHG"])]
                cap, vol = [], []
                for mode in ["DIS", "CHG"]:
                    df_mode = df[df["Mode"] == mode].sort_values("Capacity(mAh/g)")
                    cap += df_mode["Capacity(mAh/g)"].tolist() + [None]
                    vol += df_mode["Voltage(V)"].tolist() + [None]
    
                ws1.write(0, col, header)
                ws1.write(1, col, "Capacity")
                ws1.write(1, col+1, "Voltage")
                for row in range(len(cap)):
                    ws1.write(row+2, col,   cap[row])
                    ws1.write(row+2, col+1, vol[row])
                col_letter = col_idx_to_excel_col(col)
                chart1.add_series({
                    'name':      f"=GraphData!${col_letter}$1",
                    'categories': ["GraphData", 2, col,     len(cap)+1, col],
                    'values':     ["GraphData", 2, col+1, len(vol)+1, col+1],
                    'marker':     {'type': 'none'},
                    'line':       {'width': 1.5},
                })
                col += 2
    
            chart1.set_title({'name': 'Capacity-Voltage Scatter Plot'})
            chart1.set_x_axis({'name': 'Capacity (mAh/g)', 'min': 0})
            chart1.set_y_axis({'name': 'Voltage (V)', 'min': 0.5, 'max': 3.5})
            chart1.set_style(11)
            ws1.insert_chart("K2", chart1)
    
            
            # --- 2) EfficiencyData sheet ---
            ws2 = workbook.add_worksheet("EfficiencyData")
            writer.sheets["EfficiencyData"] = ws2
            
            # 最も多くのサイクル数を持つサンプルを取得
            max_len = 0
            max_sample = None
            for sample, cycles in self.data_by_sample.items():
                if len(cycles) > max_len:
                    max_len = len(cycles)
                    max_sample = sample
            
            # Cycle列（左端）を作成（最大サイクル数に基づく）
            cycles_sorted = sorted(self.data_by_sample[max_sample].keys(), key=lambda x: int(x))
            ws2.write(0, 0, "Cycle")
            for i, cycle in enumerate(cycles_sorted):
                ws2.write(i + 1, 0, int(cycle))
            
            # 各サンプルのDIS容量とEFF(%)をそれぞれ2列ずつ出力
            col = 1
            for sample in sorted(self.data_by_sample.keys()):
                ws2.write(0, col,     f"{sample} DIS")
                ws2.write(0, col + 1, f"{sample} EFF (%)")
                sample_cycles = self.data_by_sample[sample]
                for i, cycle in enumerate(cycles_sorted):
                    df = sample_cycles.get(cycle)
                    if df is not None:
                        dis_df = df[df["Mode"] == "DIS"]
                        chg_df = df[df["Mode"] == "CHG"]
                        dis_max = dis_df["Capacity(mAh/g)"].max() if not dis_df.empty else None
                        chg_max = chg_df["Capacity(mAh/g)"].max() if not chg_df.empty else None
                        eff_pct = (chg_max / dis_max * 100) if dis_max and chg_max is not None else None
                        ws2.write(i + 1, col,     dis_max)
                        ws2.write(i + 1, col + 1, eff_pct)
                col += 2
            
            
            # --- 3) Kaleida sheet ---
            ws_kaleida = workbook.add_worksheet("Kaleida")
            col = 0
            for id_item in self.last_plotted_cycles:
                if isinstance(id_item, (tuple, list)) and len(id_item) == 2:
                    sample, cycle = id_item
                else:
                    sample = list(self.data_by_sample.keys())[0]
                    cycle = id_item

                df = self.data_by_sample[sample][cycle]

                # --- ヘッダ名の生成 ---
                if self.multi_sample_mode or (len(self.data_by_sample) > 1):
                    header = f"{sample[:10]}Cycle{cycle}"  # サンプル名(先頭10)＋Cycle数
                else:
                    header = f"Cycle{cycle}"               # Cycle数のみ

                # DIS と CHG をまとめ、Voltage と Capacity を分けてリスト化
                df = df[df["Mode"].isin(["DIS", "CHG"])]
                vol, cap = [], []
                for mode in ["DIS", "CHG"]:
                    df_m = df[df["Mode"] == mode].sort_values("Capacity(mAh/g)")
                    vol += df_m["Voltage(V)"].tolist() + [None]
                    cap += df_m["Capacity(mAh/g)"].tolist() + [None]

                ws_kaleida.write(0, col,     "Voltage")
                ws_kaleida.write(0, col + 1, header)
                for row in range(len(vol)):
                    ws_kaleida.write(row + 1, col,     vol[row])
                    ws_kaleida.write(row + 1, col + 1, cap[row])
                col += 2
# --- 4) ElectrodeInfo sheet as before ---
            ws_info = workbook.add_worksheet("ElectrodeInfo")
            writer.sheets["ElectrodeInfo"] = ws_info
            ws_info.write(0, 0, "Sample")
            ws_info.write(0, 1, "INFO")
            info_row = 1
            info_dir = os.path.join(os.getcwd(), "INFO")
            for sample in sorted(self.data_by_sample.keys()):
                info_path = os.path.join(info_dir, f"{sample}.txt")
                if os.path.exists(info_path):
                    lines_info = open(info_path, "r", encoding="utf-8").read().splitlines()
                else:
                    lines_info = [f"No INFO file found for sample: {sample}"]
                ws_info.write(info_row, 0, sample)
                for i, line in enumerate(lines_info):
                    ws_info.write(info_row + i, 1, line)
                info_row += max(1, len(lines_info)) + 1
    
        self.update_status(f"Excel saved: {os.path.basename(path)}")

    def save_info_text(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a sample to save INFO.")
            return
        if len(selected_items) > 1:
            QMessageBox.warning(self, "Multiple Selection", "Please select only one sample for saving INFO.")
            return
        data = selected_items[0].data(Qt.UserRole)
        if data:
            sample, _ = data
        else:
            sample = list(self.data_by_sample.keys())[0]
        info_dir = os.path.join(os.getcwd(), "INFO")
        if not os.path.exists(info_dir):
            os.makedirs(info_dir)
        file_path, _ = QFileDialog.getSaveFileName(self, "Save INFO File", os.path.join(info_dir, f"{sample}.txt"), "Text Files (*.txt)")
        if not file_path:
            return
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.info_textbox.toPlainText())
        self.update_status(f"INFO saved for sample: {sample}")

    def assign_new_color(self):
        cmap = matplotlib.colormaps["tab10"]
        idx = len(self.sample_colors)
        return cmap(idx % cmap.N)

    def add_sample(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder for One Sample")
        if not folder:
            return
        sample_name = os.path.basename(folder)
        if sample_name in self.data_by_sample:
            QMessageBox.warning(self, "Already Exists", f"Sample '{sample_name}' is already loaded.")
            return
        self.sample_paths[sample_name] = folder
        self.sample_colors[sample_name] = self.assign_new_color()
        self.data_by_sample[sample_name] = {}
        self.load_sample_from_folder(sample_name, folder)
        self.multi_sample_mode = True
        self.update_list_display()
        self.update_status(f"Sample '{sample_name}' added.")

    def delete_sample(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a sample to delete.")
            return
        data = selected_items[0].data(Qt.UserRole)
        if data:
            sample_name, _ = data
        else:
            sample_name = selected_items[0].text().split(":")[0]
        reply = QMessageBox.question(self, "Confirm Delete", f"Delete sample '{sample_name}'?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.data_by_sample.pop(sample_name, None)
            self.sample_paths.pop(sample_name, None)
            self.sample_colors.pop(sample_name, None)
            for i in reversed(range(self.list_widget.count())):
                item = self.list_widget.item(i)
                data = item.data(Qt.UserRole)
                if data and data[0] == sample_name:
                    self.list_widget.takeItem(i)
            self.info_textbox.clear()
            self.update_status(f"Deleted sample '{sample_name}'.")

class CyclePlotterMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cycle Plotter")
        self.resize(1200, 800)
        self.central_widget = CyclePlotterWidget()
        self.setCentralWidget(self.central_widget)
        self.statusBar().showMessage("Ready")
        self.progressBar = QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(1)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)
        self.statusBar().addPermanentWidget(self.progressBar)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = app.font()
    font.setPointSize(15)
    app.setFont(font)
    main_window = CyclePlotterMainWindow()
    main_window.show()
    sys.exit(app.exec_())
