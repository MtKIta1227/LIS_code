import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QListWidget, QListWidgetItem, QHBoxLayout, QLabel, QMessageBox, QSplitter, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class CyclePlotter(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cycle Plotter")
        self.resize(1100, 600)
        self.current_folder = ""
        self.last_plotted_cycles = []

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Top control bar
        top_bar = QHBoxLayout()
        self.load_btn = QPushButton("LOAD")
        self.all_plot_btn = QPushButton("All Plot")
        self.select_plot_btn = QPushButton("Select Plot")
        self.load_btn.setFixedWidth(80)
        self.all_plot_btn.setFixedWidth(100)
        self.select_plot_btn.setFixedWidth(100)

        top_bar.addWidget(self.load_btn)
        top_bar.addWidget(self.all_plot_btn)
        top_bar.addWidget(self.select_plot_btn)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # Main area with list and plot
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        main_layout.addWidget(splitter)

        # Left list section
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(5)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        left_layout.addWidget(self.list_widget)

        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        self.select_all_btn.clicked.connect(self.select_all_items)
        self.deselect_all_btn.clicked.connect(self.deselect_all_items)

        btn_box = QHBoxLayout()
        btn_box.addWidget(self.select_all_btn)
        btn_box.addWidget(self.deselect_all_btn)
        left_layout.addLayout(btn_box)

        left_frame.setMaximumWidth(160)
        splitter.addWidget(left_frame)

        # Right plot section
        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(5)

        self.canvas = FigureCanvas(plt.Figure())
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.subplots()

        splitter.addWidget(right_frame)
        splitter.setStretchFactor(1, 5)

        # Bottom button row (only toEXCEL now)
        bottom_buttons = QHBoxLayout()
        self.to_excel_btn = QPushButton("toEXCEL")
        self.to_excel_btn.setFixedWidth(100)
        bottom_buttons.addStretch()
        bottom_buttons.addWidget(self.to_excel_btn)
        main_layout.addLayout(bottom_buttons)

        # Connect actions
        self.load_btn.clicked.connect(self.load_files)
        self.all_plot_btn.clicked.connect(self.plot_all)
        self.select_plot_btn.clicked.connect(self.plot_selected)
        self.to_excel_btn.clicked.connect(self.export_to_excel)

        self.data_by_cycle = {}

    def load_files(self):
        # Update window title with folder path
        if self.current_folder:
            self.setWindowTitle(self.current_folder)
        self.data_by_cycle.clear()
        self.list_widget.clear()

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.current_folder = QFileDialog.getExistingDirectory(self, "Select Folder", options=options)

        if self.current_folder:
            self.setWindowTitle(self.current_folder)
            files = [os.path.join(self.current_folder, f) for f in os.listdir(self.current_folder) if f.endswith(".CSV")]
            for file in sorted(files):
                cycle = os.path.splitext(os.path.basename(file))[0]
                try:
                    df = pd.read_csv(file, encoding="shift_jis", skiprows=3)
                    df.columns = ["Mode", "Voltage(V)", "Capacity(mAh/g)", "dV(V)", "dQ(mAh/g)", "dQ/dV"]
                    df = df[df["Mode"].isin(["DIS", "CHG"])]
                    df["Cycle"] = cycle
                    self.data_by_cycle[cycle] = df
                    item = QListWidgetItem(f"Cycle {cycle}")
                    item.setCheckState(False)
                    self.list_widget.addItem(item)
                except Exception as e:
                    print(f"Error loading {file}: {e}")

    def select_all_items(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Checked)

    def deselect_all_items(self):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(Qt.Unchecked)

    def plot_data(self, cycles):
        self.ax.clear()
        colormap = plt.cm.get_cmap('tab10')

        for idx, cycle in enumerate(cycles):
            df = self.data_by_cycle[cycle]
            cycle_num = int(cycle)
            group_index = (cycle_num // 10) % 10
            alpha_val = 1.0 - 0.1 * (cycle_num % 10)
            color = colormap(group_index)
            label = f"Cycle {cycle}"
            first = True
            for mode in ["DIS", "CHG"]:
                df_mode = df[df["Mode"] == mode]
                self.ax.plot(
                    df_mode["Capacity(mAh/g)"],
                    df_mode["Voltage(V)"],
                    label=label if first else None,
                    color=color,
                    alpha=alpha_val
                )
                first = False

        self.ax.set_xlabel("Capacity (mAh/g)")
        self.ax.set_ylabel("Voltage (V)")
        self.ax.set_title("Capacity-Voltage by Cycle")
        self.ax.grid(True)

        # サイクル数に応じて凡例表示方法を切り替え
        if len(cycles) <= 30:
            # グラフ内に通常表示
            self.ax.legend(loc='best', fontsize="small")
            self.canvas.figure.tight_layout()
        else:
            # 30超えたら右側に2列で表示
            self.ax.legend(
                loc='center left',
                bbox_to_anchor=(1.0, 0.5),
                ncol=2,
                fontsize='small'
            )
            self.canvas.figure.tight_layout(rect=[0, 0, 0.85, 1])  # 凡例スペースを確保

        self.canvas.draw()
        self.last_plotted_cycles = cycles

    def plot_all(self):
        self.plot_data(list(self.data_by_cycle.keys()))

    def plot_selected(self):
        selected_cycles = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState():
                selected_cycles.append(item.text().split()[-1])

        if not selected_cycles:
            QMessageBox.warning(self, "No selection", "Please check at least one cycle to plot.")
            return
        self.plot_data(selected_cycles)

    def export_to_excel(self):
        if not self.last_plotted_cycles:
            QMessageBox.warning(self, "No Data", "No data plotted to export.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Excel File", filter="Excel Files (*.xlsx)")
        if not save_path:
            return

        import xlsxwriter
        import numpy as np

        with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            worksheet = workbook.add_worksheet("GraphData")
            writer.sheets["GraphData"] = worksheet

            chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})

            col = 0
            for cycle in self.last_plotted_cycles:
                df = self.data_by_cycle[cycle]
                df = df[df["Mode"].isin(["DIS", "CHG"])]

                cap_list = []
                vol_list = []

                for mode in ["DIS", "CHG"]:
                    df_mode = df[df["Mode"] == mode].sort_values("Capacity(mAh/g)")
                    cap = df_mode["Capacity(mAh/g)"].tolist()
                    vol = df_mode["Voltage(V)"].tolist()

                    cap_list.extend(cap)
                    vol_list.extend(vol)

                    # 継ぎ目に None を挿入（線を切る）
                    cap_list.append(None)
                    vol_list.append(None)

                # データ書き込み
                worksheet.write(0, col, f"Cycle {cycle}")
                worksheet.write(1, col, "Capacity")
                worksheet.write(1, col + 1, "Voltage")
                for row in range(len(cap_list)):
                    worksheet.write(row + 2, col, cap_list[row])
                    worksheet.write(row + 2, col + 1, vol_list[row])

                # グラフに1系列だけ追加
                chart.add_series({
                    'name':       f"=GraphData!${chr(65 + col)}$1",
                    'categories': ["GraphData", 2, col, len(cap_list) + 1, col],
                    'values':     ["GraphData", 2, col + 1, len(vol_list) + 1, col + 1],
                    'marker':     {'type': 'none'},
                    'line':       {'width': 1.5},
                })

                col += 2

            chart.set_title({'name': 'Capacity-Voltage Scatter Plot'})
            chart.set_x_axis({'name': 'Capacity (mAh/g)', 'min': 0})
            chart.set_y_axis({'name': 'Voltage (V)', 'min': 0.5, 'max': 3.5})
            chart.set_legend({'position': 'right', 'overlay': True})
            chart.set_style(11)
            chart.set_size({'width': 500, 'height': 600})

            worksheet.insert_chart("K2", chart)

        QMessageBox.information(self, "Exported", f"Graph data and chart saved to {save_path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CyclePlotter()
    window.show()
    sys.exit(app.exec_())
