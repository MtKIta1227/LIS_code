import sys
import csv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QMessageBox, QTextEdit
)
from PyQt5.QtGui import QClipboard

class CellParameterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("セル仕様エディタ")
        self.resize(500, 600)
        self.layout = QVBoxLayout(self)

        self.fields = [
            ("Cell Name", "ラミセル", ""),
            ("活物質重量", "106.63", "mg"),
            ("面積", "23×25", "cm"),
            ("硫黄含有量", "9.3", "mg/cm2"),
            ("E/S", "3", ""),
            ("電解液量", "0.320", "mL"),
            ("Cレート", "0.1", "C"),
            ("電圧範囲", "1.0-3.0", "V"),
            ("試験温度", "25", "℃"),
            ("合材", "MSC(69%):AB:x:CMC:SBR=90:5:1:1.5:2.5", ""),
            ("電解液", "1MLiTFSI/FEC:D2+5wt%LiFSI", ""),
            ("セパレータ", "P1F16", "")
        ]

        self.input_fields = []

        for label, value, unit in self.fields:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            input_box = QLineEdit(value)
            row.addWidget(input_box)
            self.input_fields.append((label, input_box))
            if unit:
                row.addWidget(QLabel(unit))
            self.layout.addLayout(row)

        button_layout = QHBoxLayout()
        save_button = QPushButton("CSV保存")
        save_button.clicked.connect(self.save_csv)
        copy_button = QPushButton("テキストコピー")
        copy_button.clicked.connect(self.copy_text)
        button_layout.addWidget(save_button)
        button_layout.addWidget(copy_button)
        self.layout.addLayout(button_layout)

    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存先を選択", "", "CSV Files (*.csv)")
        if path:
            try:
                with open(path, "w", newline="", encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["項目", "値"])
                    for label, input_box in self.input_fields:
                        writer.writerow([label, input_box.text()])
                QMessageBox.information(self, "保存完了", "CSVに保存しました。")
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"保存できませんでした:\n{e}")

    def copy_text(self):
        text = ""
        for label, input_box in self.input_fields:
            text += f"{label}: {input_box.text()}\n"
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "コピー完了", "クリップボードにコピーしました。")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CellParameterGUI()
    gui.show()
    sys.exit(app.exec_())
