import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QLineEdit, QMessageBox, QTableWidget, QTableWidgetItem
)


class CycleAnalysisGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Cycle Analysis GUI')
        self.setGeometry(200, 200, 800, 550)

        layout = QVBoxLayout()

        self.load_btn = QPushButton('LOAD')
        self.load_btn.clicked.connect(self.load_folder)

        self.folder_label = QLabel('選択されたフォルダ: なし')

        cycle_layout = QHBoxLayout()
        self.cycle_input = QLineEdit()
        self.cycle_input.setPlaceholderText('サイクル数を入力')

        self.ok_btn = QPushButton('OK')
        self.ok_btn.clicked.connect(self.analyze_cycles)

        cycle_layout.addWidget(self.cycle_input)
        cycle_layout.addWidget(self.ok_btn)

        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText('Sample No.を検索')
        self.search_btn = QPushButton('検索')
        self.search_btn.clicked.connect(self.search_sample_no)

        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_btn)

        # 表形式で結果を表示
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(['Sample No.', 'Sample Name', '放電容量 (mAh/g)', 'クーロン効率 (%)'])

        layout.addWidget(self.load_btn)
        layout.addWidget(self.folder_label)
        layout.addLayout(cycle_layout)
        layout.addLayout(search_layout)
        layout.addWidget(self.result_table)

        self.setLayout(layout)

        self.base_folder = ''

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.base_folder = folder
            self.folder_label.setText(f'選択されたフォルダ: {folder}')

    def analyze_cycles(self):
        cycle_number = self.cycle_input.text().strip()
        if not cycle_number.isdigit():
            QMessageBox.warning(self, 'Invalid Input', '有効なサイクル番号を入力してください（数字のみ）。')
            return

        if not self.base_folder:
            QMessageBox.warning(self, 'No Folder', 'フォルダを選択してください。')
            return

        self.results = []

        for subfolder in sorted(os.listdir(self.base_folder)):
            subfolder_path = os.path.join(self.base_folder, subfolder)
            if os.path.isdir(subfolder_path):
                parts = subfolder.split('_')
                sample_no = parts[0].replace('YM', '')
                sample_name = '_'.join(parts[1:])
                cycle_file = os.path.join(subfolder_path, f'{int(cycle_number):05d}.CSV')
                if os.path.exists(cycle_file):
                    try:
                        df = pd.read_csv(cycle_file, encoding="shift_jis", skiprows=3)
                        df.columns = ["Mode", "Voltage(V)", "Capacity(mAh/g)", "dV(V)", "dQ(mAh/g)", "dQ/dV"]
                        dis_df = df[df["Mode"] == "DIS"]
                        chg_df = df[df["Mode"] == "CHG"]
                        if dis_df.empty or chg_df.empty:
                            dis_max, efficiency = 'データなし', 'データなし'
                        else:
                            dis_max = f"{dis_df['Capacity(mAh/g)'].max():.1f}"
                            chg_max = chg_df["Capacity(mAh/g)"].max()
                            efficiency = f"{(chg_max / float(dis_max)) * 100:.1f}" if float(dis_max) else '0.00'
                    except Exception as e:
                        dis_max, efficiency = f"エラー({e})", f"エラー({e})"
                else:
                    dis_max, efficiency = 'ファイルなし', 'ファイルなし'

                self.results.append((sample_no, sample_name, dis_max, efficiency))

        self.display_results(self.results)

    def display_results(self, results):
        self.result_table.setRowCount(len(results))
        for row, (sample_no, sample_name, dis_cap, coul_eff) in enumerate(results):
            self.result_table.setItem(row, 0, QTableWidgetItem(sample_no))
            self.result_table.setItem(row, 1, QTableWidgetItem(sample_name))
            self.result_table.setItem(row, 2, QTableWidgetItem(dis_cap))
            self.result_table.setItem(row, 3, QTableWidgetItem(coul_eff))

    def search_sample_no(self):
        query = self.search_input.text().strip()
        if query:
            filtered_results = [r for r in self.results if query in r[0]]
            self.display_results(filtered_results)
        else:
            self.display_results(self.results)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = CycleAnalysisGUI()
    gui.show()
    sys.exit(app.exec_())