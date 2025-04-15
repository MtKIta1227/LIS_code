import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QGridLayout,
    QPushButton, QMessageBox, QGroupBox
)


class CellSpecApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("セル仕様入力")
        self.resize(400, 600)

        self.labels = [
            "Cell Name", "活物質重量 (mg)", "面積 (mm)", "硫黄含有量 (mg/cm2)",
            "E/S", "電解液量 (mL)", "Cレート", "電圧範囲 (V)", "試験温度 (℃)",
            "合材", "電解液", "セパレータ"
        ]

        self.inputs = {}
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        grid = QGridLayout()

        # 通常入力欄
        for i, label_text in enumerate(self.labels):
            label = QLabel(label_text)
            input_field = QLineEdit()
            self.inputs[label_text] = input_field
            grid.addWidget(label, i, 0)
            grid.addWidget(input_field, i, 1)

        main_layout.addLayout(grid)

        # 計算値グループ（囲み）
        calc_group = QGroupBox("計算値")
        calc_layout = QGridLayout()

        self.sin_input = QLineEdit()
        self.electrolyte_calc_input = QLineEdit()

        self.sin_input.setReadOnly(False)
        self.electrolyte_calc_input.setReadOnly(False)

        calc_layout.addWidget(QLabel("硫黄含有率:"), 0, 0)
        calc_layout.addWidget(self.sin_input, 0, 1)
        calc_layout.addWidget(QLabel("電解液量 (mL):"), 1, 0)
        calc_layout.addWidget(self.electrolyte_calc_input, 1, 1)

        calc_group.setLayout(calc_layout)
        main_layout.addWidget(calc_group)

        # ボタン
        button = QPushButton("to クリップボード")
        button.clicked.connect(self.copy_to_clipboard)
        main_layout.addWidget(button)

        # 自動計算トリガー
        self.inputs["活物質重量 (mg)"].textChanged.connect(self.update_calculations)
        self.inputs["面積 (mm)"].textChanged.connect(self.update_calculations)
        self.inputs["E/S"].textChanged.connect(self.update_calculations)

        self.setLayout(main_layout)

    def update_calculations(self):
        try:
            weight = float(self.inputs["活物質重量 (mg)"].text())
            area = self.inputs["面積 (mm)"].text()
            es = float(self.inputs["E/S"].text())

            if "×" in area:
                a, b = area.split("×")
                area_val = float(a) * float(b)
            elif "x" in area:
                a, b = area.split("x")
                area_val = float(a) * float(b)
            else:
                area_val = float(area)

            sin = weight / area_val / 2 /1000
            electrolyte_vol = weight * es / 1000

            self.sin_input.setText(f"{sin:.1f}")
            self.electrolyte_calc_input.setText(f"{electrolyte_vol:.3f}")
        except:
            self.sin_input.setText("")
            self.electrolyte_calc_input.setText("")

    def copy_to_clipboard(self):
        text_lines = []

        for key in self.labels:
            val = self.inputs[key].text()
            # 「: 」形式で整形
            label_clean = key.replace("(mg)", "").replace("(mm)", "").replace("(mg/cm2)", "").replace("(mL)", "").replace("(℃)", "").replace("(V)", "").strip()
            text_lines.append(f"{label_clean}: {val}")

        # 計算値も追加
        text_lines.append(f"硫黄含有率: {self.sin_input.text()}")
        text_lines.append(f"電解液量 (mL): {self.electrolyte_calc_input.text()}")

        final_text = "\n".join(text_lines)
        QApplication.clipboard().setText(final_text)
        QMessageBox.information(self, "コピー完了", "クリップボードにコピーしました！")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CellSpecApp()
    win.show()
    sys.exit(app.exec_())
