import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem, QHBoxLayout, QMessageBox, QSplitter, QFrame,
    QSizePolicy, QProgressBar
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.ticker import PercentFormatter

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
        main_layout.addWidget(splitter)

        # ==== Left Panel ====
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.StyledPanel)
        left_frame.setStyleSheet("background-color: #f9f9f9; border: 1px solid #ccc;")
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(8, 8, 8, 8)

        self.list_widget = QListWidget()
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        btn_box = QHBoxLayout()
        btn_box.addWidget(self.select_all_btn)
        btn_box.addWidget(self.deselect_all_btn)
        left_layout.addWidget(self.list_widget)
        left_layout.addLayout(btn_box)
        left_frame.setMaximumWidth(180)
        splitter.addWidget(left_frame)

        # ==== Right Panel ====
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(8, 8, 8, 8)
        self.canvas = FigureCanvas(plt.Figure())
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.subplots()
        splitter.addWidget(right_frame)
        splitter.setStretchFactor(1, 5)

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

    def update_status(self, message):
        # 親ウィンドウが QMainWindow であれば組み込みの statusBar() を更新
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
            # 進捗表示用に QProgressBar を取得（MainWindow の statusBar に追加済みのもの）
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

    def plot_data(self, cycles):
        self.ax.clear()
        self.clear_right_axis()

        cmap = plt.cm.get_cmap("tab10")
        total_cycles = len(cycles)
        for idx, cycle in enumerate(cycles):
            df = self.data_by_cycle[cycle]
            cycle_num = int(cycle)

            group_index = cycle_num // 10
            within_group = cycle_num % 10

            color = cmap(group_index % cmap.N)
            alpha = 1.0 - 0.1 * within_group

            label = f"Cycle {cycle}"
            first = True
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

        self.ax.set_xlabel("Capacity (mAh/g)")
        self.ax.set_ylabel("Voltage (V)")
        self.ax.set_title("Capacity-Voltage by Cycle")
        self.ax.grid(True)
        self.ax.legend(
            loc="lower left",
            bbox_to_anchor=(1.0, 0),
            fontsize="small",
            frameon=True,
            ncol=2 if len(cycles) > 30 else 1
        )
        self.canvas.figure.tight_layout(rect=[0, 0, 0.85, 1])
        self.canvas.draw()
        self.last_plotted_cycles = cycles
        self.update_status(f"プロット完了: {total_cycles} サイクル表示中")

    def select_all_items(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Checked)

    def deselect_all_items(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Unchecked)

    def plot_dis_cap_efficiency(self):
        self.ax.clear()
        self.clear_right_axis()
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

        self.ax.set_title("Max DIS Capacity & Coulombic Efficiency")
        self.ax.set_xlabel("Cycle Number")
        self.ax.set_ylabel("Max DIS Capacity (mAh/g)", color="blue")
        ln1 = self.ax.plot(cycles, max_dis_caps, color="blue", label="Max DIS Capacity", marker='o')
        self.ax.tick_params(axis='y', labelcolor="blue")
        self.ax.grid(True)

        self.ax2 = self.ax.twinx()
        self.ax2.set_ylabel("Coulombic Efficiency (%)", color="orange")
        ln2 = self.ax2.plot(
            cycles,
            [e * 100 for e in efficiencies],
            color="orange",
            label="Efficiency",
            marker='x'
        )
        self.ax2.tick_params(axis='y', labelcolor="orange")
        self.ax2.set_ylim(0, 110)
        self.ax2.yaxis.set_major_formatter(PercentFormatter())

        lines = ln1 + ln2
        labels = [line.get_label() for line in lines]
        self.ax.legend(lines, labels, loc="best", fontsize="small", frameon=True)
        self.canvas.figure.tight_layout(rect=[0, 0, 0.85, 1])
        self.canvas.draw()
        self.update_status("DIS容量とクーロン効率を表示しました")

    def export_to_excel(self):
        if not self.last_plotted_cycles:
            QMessageBox.warning(self, "No Data", "プロットされたデータがありません")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Excel File", filter="Excel Files (*.xlsx)")
        if not path:
            return

        import xlsxwriter
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            workbook = writer.book
            ws1 = workbook.add_worksheet("GraphData")
            writer.sheets["GraphData"] = ws1

            chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
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
                chart.add_series({
                    'name': f"=GraphData!${chr(65 + col)}$1",
                    'categories': ["GraphData", 2, col, len(cap) + 1, col],
                    'values': ["GraphData", 2, col + 1, len(vol) + 1, col + 1],
                    'marker': {'type': 'none'},
                    'line': {'width': 1.5},
                })
                col += 2

            chart.set_title({'name': 'Capacity-Voltage Scatter Plot'})
            chart.set_x_axis({'name': 'Capacity (mAh/g)', 'min': 0})
            chart.set_y_axis({'name': 'Voltage (V)', 'min': 0.5, 'max': 3.5})
            chart.set_style(11)
            chart.set_size({'width': 500, 'height': 600})
            ws1.insert_chart("K2", chart)

            if self.efficiency_data:
                ws2 = workbook.add_worksheet("EfficiencyData")
                ws2.write(0, 0, "Cycle")
                ws2.write(0, 1, "Max DIS Capacity")
                ws2.write(0, 2, "Coulombic Efficiency (%)")
                for i, (cyc, dcap, eff) in enumerate(self.efficiency_data, start=1):
                    ws2.write(i, 0, cyc)
                    ws2.write(i, 1, dcap)
                    ws2.write(i, 2, eff * 100 if eff is not None else None)

        self.update_status(f"Excelに保存しました: {os.path.basename(path)}")

class CyclePlotterMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cycle Plotter")
        self.resize(1200, 700)
        self.central_widget = CyclePlotterWidget()
        self.setCentralWidget(self.central_widget)
        self.statusBar().showMessage("Ready")
        # ステータスバーに進捗表示用の QProgressBar を追加
        self.progressBar = QProgressBar()
        self.progressBar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progressBar)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = CyclePlotterMainWindow()
    main_window.show()
    sys.exit(app.exec_())
