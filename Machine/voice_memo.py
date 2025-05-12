# 必要なライブラリをインストール
# pip install openai-whisper tkinter ffmpeg-python

import whisper
import tkinter as tk
from tkinter import filedialog, messagebox

def transcribe_audio():
    # ファイル選択ダイアログ
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.m4a *.mp3 *.wav *.flac")])
    if not file_path:
        return

    # Whisperモデル読み込み
    messagebox.showinfo("Whisper", "文字起こし中です。少々お待ちください…")
    model = whisper.load_model("base")

    # 文字起こし
    result = model.transcribe(file_path)

    # 結果を表示
    transcription = result['text']
    text_output.delete(1.0, tk.END)
    text_output.insert(tk.END, transcription)

# GUI ウィンドウの作成
root = tk.Tk()
root.title("Whisper 音声文字起こしツール")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

button = tk.Button(frame, text="音声ファイルを選択して文字起こし", command=transcribe_audio)
button.pack(pady=10)

text_output = tk.Text(frame, width=80, height=20)
text_output.pack()

root.mainloop()
