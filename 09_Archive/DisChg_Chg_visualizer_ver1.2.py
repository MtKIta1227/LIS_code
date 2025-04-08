
(コードが長いため、実際のファイルには元のコードに以下の変更が含まれています)

# インポートの追加
from PyQt5.QtWidgets import QTextEdit

# CyclePlotterWidgetの__init__に追加するGUI部品の設定:
# テキストボックスとボタン追加
info_layout = QVBoxLayout()
self.info_textbox = QTextEdit()
self.info_textbox.setPlaceholderText("ここにINFOを入力してください")
self.info_save_btn = QPushButton("INFO Save")
self.info_save_btn.setMinimumWidth(120)

info_layout.addWidget(self.info_textbox)
info_layout.addWidget(self.info_save_btn)

# レイアウトを横に並べる
right_side_layout = QHBoxLayout()
right_side_layout.addWidget(self.canvas, 4)  # 元々のグラフエリアを広めに
right_side_layout.addLayout(info_layout, 1)  # INFOエリアはやや狭めに

right_layout.addWidget(self.toolbar)
right_layout.addLayout(right_side_layout)

# save_info_textメソッドを追加:
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

# ボタンに機能を接続:
self.info_save_btn.clicked.connect(self.save_info_text)

# load_filesメソッド末尾にINFOテキストファイル読み込み追加:
info_dir = os.path.join(os.getcwd(), "INFO")
folder_name = os.path.basename(self.current_folder)
info_path = os.path.join(info_dir, f"{folder_name}.txt")

if os.path.exists(info_path):
    with open(info_path, "r", encoding="utf-8") as file:
        self.info_textbox.setPlainText(file.read())
else:
    self.info_textbox.clear()
