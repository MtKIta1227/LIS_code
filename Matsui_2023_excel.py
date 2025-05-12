#GUIでエクセルファイルを選択する。PyQt5を使用。
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
import pandas as pd
import openpyxl
import os
# PyQt5を使用してGUIでエクセルファイルを選択する。その後、選択したエクセルファイルを読み込み、最初のシートのD列三行目以降のデータを取得し、リストに格納する。

def select_excel_file():
    app = QApplication(sys.argv)
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFiles)
    file_dialog.setNameFilter("Excel Files (*.xlsx *.xls *.xlsm)")
    file_dialog.setViewMode(QFileDialog.List)
    
    if file_dialog.exec_():
        selected_files = file_dialog.selectedFiles()
        return selected_files
    else:
        return []
def read_excel_data(file_path):
    # xlsmファイルを読み込む
    workbook = openpyxl.load_workbook(file_path, data_only=True)
    # 最初のシートを取得
    sheet = workbook.active
    # D列の3行目以降のデータを取得する
    data = []
    for row in sheet.iter_rows(min_row=3, min_col=4, max_col=4):
        for cell in row:
            data.append(cell.value)
    

    return data
def main():
    selected_files = select_excel_file()
    if not selected_files:
        print("No file selected.")
        return

    for file_path in selected_files:
        print(f"Selected file: {file_path}")
        data = read_excel_data(file_path)
        print("Data from D column (3rd row onwards):")
        print(data)
if __name__ == "__main__":
    main()