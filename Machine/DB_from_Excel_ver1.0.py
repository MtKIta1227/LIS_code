import os
import re
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, QProgressBar
)

class ExcelGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Excel Data Loader")
        self.loaded_files = set()
        cols = [
            "filename", "cycleA", "capacityA", "cycleB", "capacityB",
            "活物質重量", "E/S", "Cレート", "電圧範囲",
            "試験温度", "合材", "電解液", "セパレータ"
        ]
        self.summary_df = pd.DataFrame(columns=cols)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ボタン群
        self.load_btn = QPushButton("Load Excel Files")
        self.load_btn.clicked.connect(self.load_data_files)
        layout.addWidget(self.load_btn)

        self.export_csv_btn = QPushButton("Output CSV")
        self.export_csv_btn.clicked.connect(self.export_csv)
        layout.addWidget(self.export_csv_btn)

        self.export_excel_btn = QPushButton("Output Excel")
        self.export_excel_btn.clicked.connect(self.export_excel)
        layout.addWidget(self.export_excel_btn)

        # プログレスバー
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # 結果表示テーブル
        self.table = QTableWidget()
        layout.addWidget(self.table)

    def load_data_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Excel Files", "", "Excel Files (*.xlsx *.xls *.xlsm *csv)"
        )
        to_load = [f for f in files if f not in self.loaded_files]
        if not to_load:
            return

        # プログレスバー設定
        self.progress.setVisible(True)
        self.progress.setMinimum(0)
        self.progress.setMaximum(len(to_load))
        self.progress.setValue(0)

        numeric_labels = ["活物質重量", "E/S", "Cレート", "電圧範囲", "試験温度"]
        num_pattern = re.compile(r'^\d+(\.\d+)?$')

        for count, file in enumerate(to_load, start=1):
            QApplication.processEvents()  # UI 更新

            self.loaded_files.add(file)
            df = pd.read_excel(file, header=None)

            # サイクル数と放電容量の取得
            cycleA = df.iloc[2, 3]
            disD = df.iloc[:, 3]
            maskD = disD.iloc[2:].isna()
            capacityA = disD.iloc[maskD[maskD].index[0] - 1] if maskD.any() else disD.iloc[-1]

            cycleB = df.iloc[2, 11]
            disL = df.iloc[:, 11]
            maskL = disL.iloc[2:].isna()
            capacityB = disL.iloc[maskL[maskL].index[0] - 1] if maskL.any() else disL.iloc[-1]

            # capacityA, capacityB の数値判定（数値以外は None）
            if not num_pattern.fullmatch(str(capacityA).strip()):
                capacityA = None
            if not num_pattern.fullmatch(str(capacityB).strip()):
                capacityB = None

            # electrode_Info の読み込み＆パース
            elec_df = df.iloc[3:18, 46:53]
            elec_flat = elec_df.values.flatten().tolist()
            labels = [
                "活物質重量", "E/S", "Cレート", "電圧範囲",
                "試験温度", "合材", "電解液", "セパレータ"
            ]
            parsed = {lbl: None for lbl in labels}
            for lbl in labels:
                for i, v in enumerate(elec_flat):
                    if str(v).strip() == lbl:
                        for nxt in elec_flat[i+1:]:
                            if pd.notna(nxt) and str(nxt).strip() != "":
                                parsed[lbl] = nxt
                                break
                        break

            # 数値チェック：指定ラベル以外はそのまま、指定ラベルは数値のみ許可
            for lbl in numeric_labels:
                val = parsed.get(lbl)
                if val is None:
                    continue
                if not num_pattern.fullmatch(str(val).strip()):
                    parsed[lbl] = None

            new_row = {
                "filename": os.path.basename(file),
                "cycleA": cycleA,
                "capacityA": capacityA,
                "cycleB": cycleB,
                "capacityB": capacityB,
                **parsed
            }
            self.summary_df = pd.concat(
                [self.summary_df, pd.DataFrame([new_row])],
                ignore_index=True
            )

            # プログレスバー更新
            self.progress.setValue(count)

        self.progress.setVisible(False)
        self.update_table()

    def update_table(self):
        df = self.summary_df
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns.tolist())
        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                val = row[col]
                self.table.setItem(i, j, QTableWidgetItem("" if pd.isna(val) else str(val)))

    def export_csv(self):
        if self.summary_df.empty:
            QMessageBox.warning(self, "Warning", "出力するデータがありません。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "summary.csv", "CSV Files (*.csv)"
        )
        if path:
            self.summary_df.to_csv(path, index=False)
            QMessageBox.information(self, "Saved", f"{path} に出力しました。")

    def export_excel(self):
        if self.summary_df.empty:
            QMessageBox.warning(self, "Warning", "出力するデータがありません。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Excel", "summary.xlsx", "Excel Files (*.xlsx)"
        )
        if path:
            try:
                self.summary_df.to_excel(path, index=False, sheet_name="Summary")
                QMessageBox.information(self, "Saved", f"{path} に出力しました。")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Excel 出力に失敗しました:\n{e}")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    gui = ExcelGUI()
    gui.resize(800, 600)
    gui.show()
    sys.exit(app.exec_())