import sys
import csv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QMessageBox, QPlainTextEdit
)
from PyQt5.QtGui import QClipboard

class CellSpecEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("セル仕様エディタ (テキスト貼り付け対応)")
        self.resize(600, 800)
        self.layout = QVBoxLayout(self)

        # 1. 貼り付け用テキストエリア
        self.paste_label = QLabel("セル仕様情報を下記に貼り付けてください（タブ区切りを想定）:")
        self.layout.addWidget(self.paste_label)

        self.paste_text = QPlainTextEdit()
        self.paste_text.setPlaceholderText("例:\nCell Name\tラミセル\n活物質重量\t99.92\tmg\n面積\t23×25\tcm\n...")
        self.layout.addWidget(self.paste_text)

        self.parse_button = QPushButton("テキスト解析")
        self.parse_button.clicked.connect(self.parse_text)
        self.layout.addWidget(self.parse_button)

        # 2. 編集用フィールド
        self.field_layout = QVBoxLayout()
        self.fields = [
            "Cell Name", "活物質重量", "面積", "硫黄含有量", "E/S", "電解液量",
            "Cレート", "電圧範囲", "試験温度", "合材", "電解液", "セパレータ"
        ]
        self.field_widgets = {}
        self.units = {
            "活物質重量": "mg", "面積": "mm", "硫黄含有量": "㎎/cm2",
            "電解液量": "mL", "Cレート": "C", "電圧範囲": "V", "試験温度": "℃"
        }

        for field in self.fields:
            row = QHBoxLayout()
            label = QLabel(field)
            row.addWidget(label)

            line_edit = QLineEdit()
            line_edit.textChanged.connect(self.update_dependent_fields)
            row.addWidget(line_edit)

            unit = self.units.get(field, "")
            if unit:
                unit_label = QLabel(unit)
                row.addWidget(unit_label)

            self.field_widgets[field] = line_edit
            self.field_layout.addLayout(row)

        self.layout.addLayout(self.field_layout)

        # 3. 保存/コピーボタン
        button_layout = QHBoxLayout()
        save_button = QPushButton("CSV保存")
        save_button.clicked.connect(self.save_csv)
        button_layout.addWidget(save_button)

        copy_button = QPushButton("テキストコピー")
        copy_button.clicked.connect(self.copy_text)
        button_layout.addWidget(copy_button)

        self.layout.addLayout(button_layout)

    
    def update_dependent_fields(self):
        try:
            weight = float(self.field_widgets["活物質重量"].text())
        except ValueError:
            weight = None

        try:
            area_text = self.field_widgets["面積"].text()
            if "×" in area_text:
                w, h = map(float, area_text.split("×"))
                area = w * h
            elif "*" in area_text:
                w, h = map(float, area_text.split("*"))
                area = w * h
            else:
                area = float(area_text)
        except:
            area = None

        try:
            es = float(self.field_widgets["E/S"].text())
        except ValueError:
            es = None

        # 硫黄含有量 = 活物質重量 / 面積 / 2
        if weight is not None and area is not None:
            sulfur_content = weight / area / 2 * 100  # ㎎/cm2
            self.field_widgets["硫黄含有量"].setText(f"{sulfur_content:.1f}")
        else:
            self.field_widgets["硫黄含有量"].setText("")

        # 電解液量 = 活物質重量 × E/S / 1000
        if weight is not None and es is not None:
            electrolyte_volume = weight * es / 1000
            self.field_widgets["電解液量"].setText(f"{electrolyte_volume:.3f}")
        else:
            self.field_widgets["電解液量"].setText("")


    
    
    
    def parse_text(self):
        import re
        text = self.paste_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "警告", "貼り付けたテキストがありません。")
            return

        lines = text.splitlines()
        data = {}
        for line in lines:
            if ":" in line:
                key, value = [p.strip() for p in line.split(":", 1)]
            elif "\t" in line:
                key, value = [p.strip() for p in line.split("\t", 1)]
            elif " " in line:
                parts = [p.strip() for p in line.split()]
                key = parts[0]
                value = parts[1]
            else:
                continue
            # 単位除去対象に "Cレート" を追加
            if key in ["活物質重量", "面積", "硫黄含有量", "電解液量", "試験温度", "電圧範囲", "Cレート"]:
                match = re.search(r"[\d\.\-×\*〜~toTO]+", value)
                if match:
                    value = match.group()
            data[key] = value

        for field in self.fields:
            if field in data:
                self.field_widgets[field].setText(data[field])
            else:
                self.field_widgets[field].setText("")

        QMessageBox.information(self, "完了", "テキストの解析が完了しました。")




    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存先を選択", "", "CSV Files (*.csv)")
        if path:
            try:
                with open(path, "w", newline="", encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["項目", "値", "単位"])
                    for field in self.fields:
                        value = self.field_widgets[field].text()
                        unit = self.units.get(field, "")
                        writer.writerow([field, value, unit])
                QMessageBox.information(self, "保存完了", "CSVに保存しました。")
            except Exception as e:
                QMessageBox.critical(self, "エラー", f"保存できませんでした:\n{e}")

    def copy_text(self):
        text = ""
        for field in self.fields:
            value = self.field_widgets[field].text()
            unit = self.units.get(field, "")
            text += f"{field}: {value} {unit}\n" if unit else f"{field}: {value}\n"
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "コピー完了", "クリップボードにコピーしました。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CellSpecEditor()
    gui.show()
    sys.exit(app.exec_())
