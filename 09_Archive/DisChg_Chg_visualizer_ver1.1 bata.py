import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# グローバルでフォントを Arial に設定
plt.rcParams['font.family'] = 'Arial'

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem, QHBoxLayout, QMessageBox, QSplitter, QFrame,
    QSizePolicy, QProgressBar
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.ticker import FormatStrFormatter, PercentFormatter

def col_idx_to_excel_col(col):
    result = ""
    while col >= 0:
        result = chr((col % 26) + 65) + result
        col = col // 26 - 1
    return result

class CyclePlotterWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.current_folder = ""
        self.last_plotted_cycles = []
        self.efficiency_data = []
        self.data_by_cycle = {}

        main_layout = QVBoxLayout(self)

        # ==== Top Bar ====
        top_bar = QHBoxLayout()
        top_bar.setSpacing(15)
        top_bar.setContentsMargins(20, 10, 20, 5)

        self.load_btn = QPushButton("LOAD")
        self.all_plot_btn = QPushButton("All Plot")
        self.select_plot_btn = QPushButton("Select Plot")
        self.dis_cap_btn = QPushButton("DIS-Cap/CLN EF")

        for btn in [self.load_btn, self.all_plot_btn, self.select_plot_btn, self.dis_cap_btn]:
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
        self.deselect_all_btn = QPushButton("Deselect All")
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
        right_layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.add_subplot(111)
        splitter.addWidget(right_frame)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([150, 900])

        # ==== Bottom Buttons ====
        bottom_buttons = QHBoxLayout()
        self.to_excel_btn = QPushButton("toEXCEL")
        self.to_excel_btn.setMinimumHeight(30)
        self.to_excel_btn.setMinimumWidth(120)
        bottom_buttons.addStretch()
        bottom_buttons.addWidget(self.to_excel_btn)
        main_layout.addLayout(bottom_buttons)

        # ==== Connections ====
        self.load_btn.clicked.connect(self.load_files)
        self.all_plot_btn.clicked.connect(self.plot_all)
        self.select_plot_btn.clicked.connect(self.plot_selected)
        self.dis_cap_btn.clicked.connect(self.plot_dis_cap_efficiency)
        self.to_excel_btn.clicked.connect(self.export_to_excel)
        self.select_all_btn.clicked.connect(self.select_all_items)
        self.deselect_all_btn.clicked.connect(self.deselect_all_items)

    def select_all_items(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Checked)

    def deselect_all_items(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Unchecked)

# --- 以下の部分は、コードが長いため分割して次で保存 ---

    def update_status(self, message):
        main_window = self.window()
        if isinstance(main_window, QMainWindow):
            main_window.statusBar().showMessage(message)
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
                cycle = os.path.splitext(os.path.basename(file))[0]
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
        self.canvas.figure.clf()
        self.ax = self.canvas.figure.add_subplot(111)
        cmap = plt.cm.get_cmap("tab10")
        for cycle in cycles:
            cycle_num = int(cycle)
            group_index = (cycle_num - 1) // 10
            within_group = (cycle_num - 1) % 10
            alpha = 1.0 - 0.1 * within_group
            color = cmap(group_index % cmap.N)
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

        self.ax.set_xlabel("Capacity (mAh/g)", fontname="Arial")
        self.ax.set_ylabel("Voltage (V)", fontname="Arial")
        self.ax.set_title("Capacity-Voltage by Cycle", fontname="Arial")
        self.ax.tick_params(axis='both', direction='in', colors='black')
        self.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        if len(cycles) <= 20:
            legend_fontsize = 11
        elif len(cycles) <= 40:
            legend_fontsize = 10
        else:
            legend_fontsize = 9

        if len(cycles) < 31:
            self.ax.legend(loc="upper right", fontsize=legend_fontsize, frameon=True, ncol=1)
        else:
            self.ax.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=legend_fontsize,
                frameon=True,
                ncol=1,
                borderaxespad=0.
            )

        self.canvas.figure.subplots_adjust(right=0.8)
        self.canvas.draw()
        self.last_plotted_cycles = cycles
        self.update_status(f"プロット完了: {len(cycles)} サイクル表示中")

    def plot_dis_cap_efficiency(self):
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

        self.ax.set_title("Max DIS Capacity & Coulombic Efficiency", fontname="Arial")
        self.ax.set_xlabel("Cycle Number", fontname="Arial")
        self.ax.set_ylabel("Max DIS Capacity (mAh/g)", fontname="Arial", color="dodgerblue")
        ln1 = self.ax.plot(cycles, max_dis_caps, color="dodgerblue", label="Max DIS Capacity", marker='o')[0]
        self.ax.tick_params(axis='both', direction='in', colors="dodgerblue")
        self.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel("Coulombic Efficiency (%)", fontname="Arial", color="darkorange")
        ln2 = self.ax2.plot(
            cycles, [e * 100 for e in efficiencies],
            color="darkorange", label="Efficiency", marker='x')[0]
        self.ax2.tick_params(axis='both', direction='in', colors="darkorange")
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
            fontsize=legend_fontsize,
            frameon=True, ncol=2
        )

        self.canvas.figure.subplots_adjust(bottom=0.25)
        self.canvas.draw()
        self.update_status("DIS容量とクーロン効率を表示しました")

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

class CyclePlotterMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cycle Plotter")
        self.resize(1000, 700)
        self.central_widget = CyclePlotterWidget()
        self.setCentralWidget(self.central_widget)
        self.statusBar().showMessage("Ready")
        self.progressBar = QProgressBar()
        self.progressBar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progressBar)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = CyclePlotterMainWindow()
    main_window.show()
    sys.exit(app.exec_())
