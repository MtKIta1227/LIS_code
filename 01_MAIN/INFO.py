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

        # 解析ボタン
        self.parse_button = QPushButton("テキスト解析")
        self.parse_button.clicked.connect(self.parse_text)
        self.layout.addWidget(self.parse_button)

        # 2. 編集用フィールドエリア（解析後に表示）
        self.field_layout = QVBoxLayout()
        self.fields = [
            "Cell Name", "活物質重量", "面積", "硫黄含有量", "E/S", "電解液量",
            "Cレート", "電圧範囲", "試験温度", "合材", "電解液", "セパレータ"
        ]
        # 各項目の (ラベル, QLineEdit, 単位（必要に応じて）) のディクショナリ
        self.field_widgets = {}

        # 単位情報を辞書で管理（あれば）
        self.units = {
            "活物質重量": "mg",
            "面積": "cm",
            "硫黄含有量": "㎎/cm2",
            "電解液量": "mL",
            "Cレート": "C",
            "電圧範囲": "V",
            "試験温度": "℃",
        }

        for field in self.fields:
            row = QHBoxLayout()
            label = QLabel(field)
            row.addWidget(label)

            line_edit = QLineEdit()
            row.addWidget(line_edit)
            unit = self.units.get(field, "")
            if unit:
                unit_label = QLabel(unit)
                row.addWidget(unit_label)

            self.field_widgets[field] = line_edit
            self.field_layout.addLayout(row)

        self.layout.addLayout(self.field_layout)

        # 3. ボタンエリア（CSV保存、クリップボードコピー）
        button_layout = QHBoxLayout()
        save_button = QPushButton("CSV保存")
        save_button.clicked.connect(self.save_csv)
        button_layout.addWidget(save_button)

        copy_button = QPushButton("テキストコピー")
        copy_button.clicked.connect(self.copy_text)
        button_layout.addWidget(copy_button)

        self.layout.addLayout(button_layout)

    def parse_text(self):
        """
        貼り付けテキストから各項目を抽出してフィールドに展開する。
        入力想定はタブ区切りまたは空白区切りのデータ。
        """
        text = self.paste_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "警告", "貼り付けたテキストがありません。")
            return

        # 行ごとに分割
        lines = text.splitlines()
        # 辞書に抽出結果を格納
        data = {}
        for line in lines:
            # タブ区切り、または複数の空白で区切る
            parts = [p.strip() for p in line.split('\t') if p.strip()]
            if len(parts) < 2:
                # 半角スペースでの分割も試みる
                parts = [p.strip() for p in line.split() if p.strip()]
            if len(parts) >= 2:
                key = parts[0]
                value = parts[1]
                # 単位が明示されている場合は値に追加せずGUI上の固定単位を利用
                data[key] = value

        # GUIの各フィールドに反映
        for field in self.fields:
            if field in data:
                self.field_widgets[field].setText(data[field])
            else:
                # もし項目が見つからなければ空欄
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
            if unit:
                text += f"{field}: {value} {unit}\n"
            else:
                text += f"{field}: {value}\n"
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "コピー完了", "クリップボードにコピーしました。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = CellSpecEditor()
    gui.show()
    sys.exit(app.exec_())
