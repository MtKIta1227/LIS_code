import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# グローバルでフォントを Arial に設定
plt.rcParams['font.family'] = 'Arial'

from PyQt5.QtWidgets import (
    QInputDialog, QComboBox,
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem, QHBoxLayout, QMessageBox, QSplitter, QFrame,
    QSizePolicy, QProgressBar, QTextEdit
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import re

def col_idx_to_excel_col(col):
    result = ""
    while col >= 0:
        result = chr((col % 26) + 65) + result
        col = col // 26 - 1
    return result


from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit, QDialogButtonBox, QFormLayout

class CycleInfoDialog(QDialog):
    def __init__(self, parent=None, data_by_cycle=None):
        super().__init__(parent)
        self.setWindowTitle("Cycle Info")
        self.setMinimumWidth(300)
        self.data_by_cycle = data_by_cycle or {}

        self.layout = QFormLayout(self)

        self.cycle_input = QLineEdit()
        self.layout.addRow("Cycle Number:", self.cycle_input)

        self.result_label = QLabel("")
        self.layout.addRow("Result:", self.result_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.calculate)
        self.button_box.rejected.connect(self.reject)

        self.layout.addWidget(self.button_box)

    def calculate(self):
        cycle = self.cycle_input.text().strip()
        if not cycle.isdigit() or cycle not in self.data_by_cycle:
            self.result_label.setText("❌ 無効なサイクル番号")
            return
        df = self.data_by_cycle[cycle]
        dis_df = df[df["Mode"] == "DIS"]
        chg_df = df[df["Mode"] == "CHG"]
        if dis_df.empty or chg_df.empty:
            self.result_label.setText("❌ DIS または CHG データなし")
            return
        dis_max = dis_df["Capacity(mAh/g)"].max()
        chg_max = chg_df["Capacity(mAh/g)"].max()
        eff = chg_max / dis_max if dis_max != 0 else 0
        # ここで計算した結果をラベルに表示
        self.result_label.setText(f"DIS Cap.: {dis_max:.1f} mAh/g\n Coulmb.Eff.: {eff * 100:.1f} %")

class CyclePlotterWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.current_folder = ""
        self.last_plotted_cycles = []
        self.efficiency_data = []
        self.data_by_cycle = {}
        self.mono_mode = False
        self.current_cmap = "tab10"

        main_layout = QVBoxLayout(self)

        # ==== Top Bar ====
        top_bar = QHBoxLayout()
        top_bar.setSpacing(15)
        top_bar.setContentsMargins(20, 10, 20, 5)

        self.load_btn = QPushButton("LOAD")
        self.monoqlo_btn = QPushButton("MONOQLO OFF")
        self.all_plot_btn = QPushButton("All Plot")
        self.select_plot_btn = QPushButton("Select Plot")
        self.dis_cap_btn = QPushButton("CyclePlot")

        for btn in [self.load_btn, self.monoqlo_btn, self.all_plot_btn, self.select_plot_btn, self.dis_cap_btn]:
            btn.setMinimumWidth(180)
            btn.setMinimumHeight(32)
            top_bar.addWidget(btn)

        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # ==== Splitter ====
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)

        # ==== Left Panel ====
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.StyledPanel)
        left_frame.setStyleSheet("background-color: #f9f9f9; border: 1px solid #ccc;")
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.list_widget = QListWidget()
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Clear All")
        btn_box = QHBoxLayout()
        btn_box.addWidget(self.select_all_btn)
        btn_box.addWidget(self.deselect_all_btn)
        left_layout.addWidget(self.list_widget)
        left_layout.addLayout(btn_box)
        left_frame.setMaximumWidth(150)
        splitter.addWidget(left_frame)

        # ==== Right Panel ====
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(8, 8, 8, 8)
        self.canvas = FigureCanvas(plt.Figure())
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        # right_layout.addWidget(self.canvas)

        # ==== INFO Box ====
        info_layout = QVBoxLayout()
        self.info_textbox = QTextEdit()
        #あらかじめテキストボックスに[[INFO]]と表示
        self.info_textbox.setPlaceholderText("ここにINFOを入力してください")
        self.info_save_btn = QPushButton("INFO Save")
        self.info_save_btn.setMinimumWidth(120)
        info_layout.addWidget(self.info_textbox)
        info_layout.addWidget(self.info_save_btn)

        # グラフとINFOを横並びに
        right_side_layout = QHBoxLayout()
        right_side_layout.addWidget(self.canvas, 4)
        right_side_layout.addLayout(info_layout, 1)

        right_layout.addWidget(self.toolbar)
        right_layout.addLayout(right_side_layout)

        self.ax = self.canvas.figure.add_subplot(111)
        splitter.addWidget(right_frame)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([150, 900])

        # ==== Bottom Buttons ====
        bottom_buttons = QHBoxLayout()
        self.color_map_btn = QPushButton("ColorMap")
        self.range_plot_btn = QPushButton("Cycle Range Plot")
        self.to_excel_btn = QPushButton("toEXCEL")
        self.to_excel_btn.setMinimumHeight(30)
        self.to_excel_btn.setMinimumWidth(120)
        bottom_buttons.addStretch()
        bottom_buttons.addWidget(self.monoqlo_btn)
        bottom_buttons.addWidget(self.color_map_btn)
        bottom_buttons.addWidget(self.range_plot_btn)
        bottom_buttons.addWidget(self.to_excel_btn)
        main_layout.addLayout(bottom_buttons)

        # ==== Connections ====
        self.load_btn.clicked.connect(self.load_files)
        self.all_plot_btn.clicked.connect(self.plot_all)
        self.select_plot_btn.clicked.connect(self.plot_selected)
        self.dis_cap_btn.clicked.connect(self.open_cycle_info_dialog)
        self.to_excel_btn.clicked.connect(self.export_to_excel)
        self.monoqlo_btn.clicked.connect(self.toggle_mono_mode)
        self.color_map_btn.clicked.connect(self.select_color_map)
        self.range_plot_btn.clicked.connect(self.plot_range_cycles)
        self.select_all_btn.clicked.connect(self.select_all_items)
        self.deselect_all_btn.clicked.connect(self.deselect_all_items)
        self.info_save_btn.clicked.connect(self.save_info_text)

    def select_all_items(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Checked)

    def deselect_all_items(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Unchecked)

# --- 以下の部分は、コードが長いため分割して次で保存 ---

    def update_status(self, message=""):
        main_window = self.window()
        if isinstance(main_window, QMainWindow):
            status_parts = []
            if self.current_folder:
                status_parts.append(f"📁 {os.path.basename(self.current_folder)}")
            if self.mono_mode:
                status_parts.append("🎨 MONOQLOモード")
            if message:
                status_parts.append(message)
            main_window.statusBar().showMessage(" | ".join(status_parts))
        else:
            print(message)
    def clear_right_axis(self):
        if hasattr(self, 'ax2') and self.ax2 in self.canvas.figure.axes:
            self.canvas.figure.delaxes(self.ax2)
            self.ax2 = None

    def load_files(self):
        self.data_by_cycle.clear()
        self.list_widget.clear()
        self.current_folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.current_folder:
            self.window().setWindowTitle(self.current_folder)
            files = [os.path.join(self.current_folder, f) for f in os.listdir(self.current_folder) if f.endswith(".CSV")]
            total_files = len(files)
            main_window = self.window()
            progress_bar = None
            if isinstance(main_window, QMainWindow):
                for widget in main_window.statusBar().findChildren(QProgressBar):
                    progress_bar = widget
                    break
            if progress_bar:
                progress_bar.setMaximum(total_files)
                progress_bar.setValue(0)
                progress_bar.setVisible(True)
            loaded_files = 0
            for file in sorted(files):
                cycle = str(int(os.path.splitext(os.path.basename(file))[0]))  # ゼロ埋め除去
                try:
                    df = pd.read_csv(file, encoding="shift_jis", skiprows=3)
                    df.columns = ["Mode", "Voltage(V)", "Capacity(mAh/g)", "dV(V)", "dQ(mAh/g)", "dQ/dV"]
                    df = df[df["Mode"].isin(["DIS", "CHG"])]
                    df["Cycle"] = cycle
                    self.data_by_cycle[cycle] = df
                    item = QListWidgetItem(f"Cycle {cycle}")
                    item.setCheckState(Qt.Unchecked)
                    self.list_widget.addItem(item)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                loaded_files += 1
                if progress_bar:
                    progress_bar.setValue(loaded_files)
            if progress_bar:
                progress_bar.setVisible(False)
            file_count = len(files)
            cycle_count = len(self.data_by_cycle)
            self.update_status(f"Loaded Folder: {self.current_folder} | ファイル: {file_count}個, サイクル: {cycle_count}個")

        # [INFO]フォルダから情報を読み込み
        info_dir = os.path.join(os.getcwd(), "INFO")
        folder_name = os.path.basename(self.current_folder)
        info_path = os.path.join(info_dir, f"{folder_name}.txt")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as file:
                self.info_textbox.setPlainText(file.read())
        else:
            self.info_textbox.clear()


    def plot_all(self):
        self.plot_data(list(self.data_by_cycle.keys()))

    def plot_selected(self):
        selected_cycles = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState():
                selected_cycles.append(item.text().split()[-1])
        if not selected_cycles:
            QMessageBox.warning(self, "No selection", "Please check at least one cycle.")
            return
        self.plot_data(selected_cycles)

    # plot_data, plot_dis_cap_efficiency, export_to_excel はこの後さらに追記


    def plot_data(self, cycles):
        axis_label_size = 13
        tick_label_size = 11
        title_size = 14
        legend_size = 10
        self.canvas.figure.clf()
        self.ax = self.canvas.figure.add_subplot(111)
        cmap = plt.cm.get_cmap(self.current_cmap)
        for cycle in cycles:
            cycle_num = int(cycle)
            group_index = (cycle_num - 1) // 10
            within_group = (cycle_num - 1) % 10
            alpha = 1.0 if self.mono_mode else 1.0 - 0.1 * within_group
            color = "black" if self.mono_mode else cmap(group_index % cmap.N)
            label = f"Cycle {cycle}"
            first = True
            df = self.data_by_cycle[cycle]
            for mode in ["DIS", "CHG"]:
                df_mode = df[df["Mode"] == mode]
                self.ax.plot(
                    df_mode["Capacity(mAh/g)"],
                    df_mode["Voltage(V)"],
                    color=color,
                    alpha=alpha,
                    label=label if first and mode == "DIS" else None
                )
                first = False

        self.ax.set_xlabel("Cycle", fontname="Arial", fontsize=axis_label_size)
        self.ax.set_ylabel("Capacity (mAh/g)", fontname="Arial", fontsize=axis_label_size)
        self.ax.set_title("Discharge-charge curves", fontname="Arial", fontsize=title_size)
        self.ax.tick_params(axis='both', direction='in', colors='black', labelsize=tick_label_size)
        self.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if len(cycles) <= 20:
            legend_fontsize = 11
        elif len(cycles) <= 40:
            legend_fontsize = 10
        else:
            legend_fontsize = 9

        if len(cycles) < 31:
            self.ax.legend(loc="upper right", fontsize=legend_size, frameon=True, ncol=1)
        else:
            self.ax.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=legend_size,
                frameon=True,
                ncol=2,
                borderaxespad=0.
            )

        self.canvas.figure.subplots_adjust(right=0.8)
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.last_plotted_cycles = cycles
        self.update_status(f"プロット完了: {len(cycles)} サイクル表示中")

    def plot_dis_cap_efficiency(self):
        axis_label_size = 13
        tick_label_size = 11
        title_size = 14
        legend_size = 10
        self.canvas.figure.clf()
        self.ax = self.canvas.figure.add_subplot(111)

        self.efficiency_data = []
        cycles = []
        max_dis_caps = []
        efficiencies = []

        for cycle in sorted(self.data_by_cycle.keys(), key=lambda x: int(x)):
            df = self.data_by_cycle[cycle]
            dis_df = df[df["Mode"] == "DIS"]
            chg_df = df[df["Mode"] == "CHG"]
            if not dis_df.empty and not chg_df.empty:
                dis_max = dis_df["Capacity(mAh/g)"].max()
                chg_max = chg_df["Capacity(mAh/g)"].max()
                eff = chg_max / dis_max if dis_max != 0 else None
                cycles.append(int(cycle))
                max_dis_caps.append(dis_max)
                efficiencies.append(eff)
                self.efficiency_data.append((int(cycle), dis_max, eff))

        self.ax.set_title("Discharge-charge curves", fontname="Arial", fontsize=title_size)
        self.ax.set_xlabel("Cycle", fontname="Arial", fontsize=axis_label_size)
        self.ax.set_ylabel("Capacity (mAh/g)", fontsize=axis_label_size, color="dodgerblue")
        self.ax.set_ylim(0, max(max_dis_caps) * 1.1)
        ln1 = self.ax.plot(cycles, max_dis_caps, color="dodgerblue", label="Discharge capacity", marker='o')[0]
        self.ax.tick_params(axis='both', direction='in', colors="k", labelsize=tick_label_size)
        self.ax.tick_params(axis='y', direction='in', colors="dodgerblue", labelsize=tick_label_size)
        self.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel("Coulombic efficiency (%)", fontname="Arial", fontsize=axis_label_size, color="darkorange")
        ln2 = self.ax2.plot(
            cycles, [e * 100 for e in efficiencies],
            color="darkorange", label="Coulombic efficiency", marker='x')[0]
        self.ax2.tick_params(axis='both', direction='in', colors="darkorange", labelsize=tick_label_size)
        self.ax2.set_ylim(0, 110)
        self.ax2.yaxis.set_major_formatter(PercentFormatter())

        lines = [ln1, ln2]
        labels = [line.get_label() for line in lines]

        if len(cycles) <= 20:
            legend_fontsize = 11
        elif len(cycles) <= 40:
            legend_fontsize = 10
        else:
            legend_fontsize = 9

        self.ax.legend(
            lines, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fontsize=legend_size,
            frameon=True, ncol=2
        )

        self.canvas.figure.subplots_adjust(bottom=0.25)
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.update_status("放電容量とクーロン効率を表示しました")


    def toggle_mono_mode(self):
        self.mono_mode = not self.mono_mode
        if self.mono_mode:
            self.monoqlo_btn.setText("MONOQLO ON")
            self.monoqlo_btn.setStyleSheet("background-color: black; color: white;")
        else:
            self.monoqlo_btn.setText("MONOQLO OFF")
            self.monoqlo_btn.setStyleSheet("")
        self.update_status()

    def select_color_map(self):
        maps = ["tab10", "tab20", "Set1", "Set2", "Paired", "Pastel1", "Dark2"]
        cmap, ok = QInputDialog.getItem(self, "カラーマップ選択", "カラーマップを選んでください：", maps, 0, False)
        if ok and cmap:
            self.current_cmap = cmap
            self.update_status(f"カラーマップ: {cmap}")

    def plot_range_cycles(self):
        text, ok = QInputDialog.getText(self, "サイクル範囲入力", "プロットしたいサイクル番号をカンマ区切りで入力（例：1-3,6,8）:")
        if not ok or not text:
            return
        import re
        pattern = re.compile(r'(\d+)(?:-(\d+))?')
        selected = set()
        for part in text.split(','):
            match = pattern.fullmatch(part.strip())
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else start
                selected.update(str(i) for i in range(start, end+1))
        # チェック状態に関係なく、入力された範囲のサイクルを強制的にチェックON
        valid_cycles = sorted([c for c in self.data_by_cycle.keys() if c in selected], key=lambda x: int(x))
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            cycle = item.text().split()[-1]
            if cycle in valid_cycles:
                item.setCheckState(Qt.Checked)
        if not valid_cycles:
            QMessageBox.warning(self, "エラー", "該当するサイクルが存在しません")
            return
        self.plot_data(valid_cycles)
    def export_to_excel(self):
        if not self.last_plotted_cycles:
            QMessageBox.warning(self, "No Data", "プロットされたデータがありません")
            return
        default_name = os.path.basename(self.current_folder) if self.current_folder else "output"
        default_path = os.path.join(os.path.dirname(self.current_folder), default_name + ".xlsx")
        path, _ = QFileDialog.getSaveFileName(self, "Save Excel File", default_path,
                                              filter="Excel Files (*.xlsx)")
        if not path:
            return
        import xlsxwriter
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            workbook = writer.book
            ws1 = workbook.add_worksheet("GraphData")
            writer.sheets["GraphData"] = ws1
            chart1 = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
            col = 0
            for cycle in self.last_plotted_cycles:
                df = self.data_by_cycle[cycle]
                df = df[df["Mode"].isin(["DIS", "CHG"])]
                cap, vol = [], []
                for mode in ["DIS", "CHG"]:
                    df_mode = df[df["Mode"] == mode].sort_values("Capacity(mAh/g)")
                    cap += df_mode["Capacity(mAh/g)"].tolist() + [None]
                    vol += df_mode["Voltage(V)"].tolist() + [None]
                ws1.write(0, col, f"Cycle {cycle}")
                ws1.write(1, col, "Capacity")
                ws1.write(1, col + 1, "Voltage")
                for row in range(len(cap)):
                    ws1.write(row + 2, col, cap[row])
                    ws1.write(row + 2, col + 1, vol[row])
                col_letter = col_idx_to_excel_col(col)
                chart1.add_series({
                    'name': f"=GraphData!${col_letter}$1",
                    'categories': ["GraphData", 2, col, len(cap) + 1, col],
                    'values': ["GraphData", 2, col + 1, len(vol) + 1, col + 1],
                    'marker': {'type': 'none'},
                    'line': {'width': 1.5},
                })
                col += 2
            chart1.set_title({'name': 'Capacity-Voltage Scatter Plot'})
            chart1.set_x_axis({'name': 'Capacity (mAh/g)', 'min': 0})
            chart1.set_y_axis({'name': 'Voltage (V)', 'min': 0.5, 'max': 3.5})
            chart1.set_style(11)
            ws1.insert_chart("K2", chart1)

            if not self.efficiency_data:
                self.efficiency_data = []
                for cycle in sorted(self.data_by_cycle.keys(), key=lambda x: int(x)):
                    df = self.data_by_cycle[cycle]
                    dis_df = df[df["Mode"] == "DIS"]
                    chg_df = df[df["Mode"] == "CHG"]
                    if not dis_df.empty and not chg_df.empty:
                        dis_max = dis_df["Capacity(mAh/g)"].max()
                        chg_max = chg_df["Capacity(mAh/g)"].max()
                        eff = chg_max / dis_max if dis_max != 0 else None
                        self.efficiency_data.append((int(cycle), dis_max, eff))
            ws2 = workbook.add_worksheet("EfficiencyData")
            writer.sheets["EfficiencyData"] = ws2
            if self.efficiency_data:
                ws2.write(0, 0, "Cycle")
                ws2.write(0, 1, "Max DIS Capacity")
                ws2.write(0, 2, "Coulombic Efficiency (%)")
                for i, (cyc, dcap, eff) in enumerate(self.efficiency_data, start=1):
                    ws2.write(i, 0, cyc)
                    ws2.write(i, 1, dcap)
                    ws2.write(i, 2, eff * 100 if eff is not None else None)
                chart2 = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
                num_rows = len(self.efficiency_data)
                chart2.add_series({
                    'name': 'Max DIS Capacity',
                    'categories': ['EfficiencyData', 1, 0, num_rows, 0],
                    'values':     ['EfficiencyData', 1, 1, num_rows, 1],
                    'marker': {'type': 'circle'},
                })
                chart2.add_series({
                    'name': 'Coulombic Efficiency (%)',
                    'categories': ['EfficiencyData', 1, 0, num_rows, 0],
                    'values':     ['EfficiencyData', 1, 2, num_rows, 2],
                    'marker': {'type': 'square'},
                    'y2_axis': 1,
                })
                chart2.set_title({'name': 'Max DIS Capacity & Coulombic Efficiency'})
                chart2.set_x_axis({'name': 'Cycle Number'})
                chart2.set_y_axis({'name': 'Max DIS Capacity (mAh/g)'})
                chart2.set_y2_axis({'name': 'Coulombic Efficiency (%)', 'min': 0, 'max': 110})
                chart2.set_style(11)
                ws2.insert_chart("E2", chart2)
            else:
                ws2.write(0, 0, "No efficiency data available.")
        self.update_status(f"Excelに保存しました: {os.path.basename(path)}")


    
    def open_cycle_info_dialog(self):
        self.plot_dis_cap_efficiency()
        dialog = CycleInfoDialog(self, data_by_cycle=self.data_by_cycle)
        dialog.exec_()

    def save_info_text(self):
        if not self.current_folder:
            QMessageBox.warning(self, "Warning", "先にフォルダを読み込んでください。")
            return
        info_dir = os.path.join(os.getcwd(), "INFO")
        if not os.path.exists(info_dir):
            os.makedirs(info_dir)
        folder_name = os.path.basename(self.current_folder)
        file_path = os.path.join(info_dir, f"{folder_name}.txt")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(self.info_textbox.toPlainText())
        self.update_status(f"INFOを保存しました: {folder_name}.txt")

class CyclePlotterMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cycle Plotter")
        self.resize(1000, 700)
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

    # 📌 GUIのフォントサイズを統一設定（例: 15pt）
    font = app.font()
    font.setPointSize(15)
    app.setFont(font)

    main_window = CyclePlotterMainWindow()
    main_window.show()
    sys.exit(app.exec_())


    def toggle_mono_mode(self):
        self.mono_mode = not self.mono_mode
        if self.mono_mode:
            self.monoqlo_btn.setText("MONOQLO ON")
            self.monoqlo_btn.setStyleSheet("background-color: black; color: white;")
        else:
            self.monoqlo_btn.setText("MONOQLO OFF")
            self.monoqlo_btn.setStyleSheet("")
        self.update_status()


    def select_color_map(self):
        maps = ["tab10", "tab20", "Set1", "Set2", "Paired", "Pastel1", "Dark2"]
        cmap, ok = QInputDialog.getItem(self, "カラーマップ選択", "カラーマップを選んでください：", maps, 0, False)
        if ok and cmap:
            self.current_cmap = cmap
            self.update_status(f"カラーマップ: {cmap}")

    def plot_range_cycles(self):
        text, ok = QInputDialog.getText(self, "サイクル範囲入力", "プロットしたいサイクル番号をカンマ区切りで入力（例：1-3,6,8）:")
        if not ok or not text:
            return
        import re
        pattern = re.compile(r'(\d+)(?:-(\d+))?')
        selected = set()
        for part in text.split(','):
            match = pattern.fullmatch(part.strip())
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else start
                selected.update(str(i) for i in range(start, end+1))
        # チェック状態に関係なく、入力された範囲のサイクルを強制的にチェックON
        valid_cycles = sorted([c for c in self.data_by_cycle.keys() if c in selected], key=lambda x: int(x))
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            cycle = item.text().split()[-1]
            if cycle in valid_cycles:
                item.setCheckState(Qt.Checked)
        if not valid_cycles:
            QMessageBox.warning(self, "エラー", "該当するサイクルが存在しません")
            return
        self.plot_data(valid_cycles)


from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit, QDialogButtonBox, QFormLayout

class CycleInfoDialog(QDialog):
    def __init__(self, parent=None, data_by_cycle=None):
        super().__init__(parent)
        self.setWindowTitle("Cycle Info")
        # ウィンドウサイズを指定
        self.setMinimumWidth(500)

        self.data_by_cycle = data_by_cycle or {}

        self.layout = QFormLayout(self)

        self.cycle_input = QLineEdit()
        self.layout.addRow("Cycle Number:", self.cycle_input)

        self.result_label = QLabel("")
        self.layout.addRow("Result:", self.result_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.calculate)
        self.button_box.rejected.connect(self.reject)

        self.layout.addWidget(self.button_box)

    def calculate(self):
        cycle = self.cycle_input.text().strip()
        if not cycle.isdigit() or cycle not in self.data_by_cycle:
            self.result_label.setText("❌ 無効なサイクル番号")
            return
        df = self.data_by_cycle[cycle]
        dis_df = df[df["Mode"] == "DIS"]
        chg_df = df[df["Mode"] == "CHG"]
        if dis_df.empty or chg_df.empty:
            self.result_label.setText("❌ DIS または CHG データなし")
            return
        dis_max = dis_df["Capacity(mAh/g)"].max()
        chg_max = chg_df["Capacity(mAh/g)"].max()
        eff = chg_max / dis_max if dis_max != 0 else 0
        self.result_label.setText(f"DIS Cap.: {dis_max:.1f} mAh/g\n CE.: {eff * 100:.1f} %")
