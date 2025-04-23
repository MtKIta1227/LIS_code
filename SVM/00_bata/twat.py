import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QFileDialog, QComboBox, QTextEdit
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

class MLApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("放電容量予測ツール")
        self.resize(600, 400)

        self.df = None
        self.model = None

        self.load_button = QPushButton("Excelファイル読み込み")
        self.sheet_select = QComboBox()
        self.train_button = QPushButton("モデル学習")
        self.result_view = QTextEdit()
        self.result_view.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(QLabel("シート選択"))
        layout.addWidget(self.sheet_select)
        layout.addWidget(self.train_button)
        layout.addWidget(QLabel("結果:"))
        layout.addWidget(self.result_view)
        self.setLayout(layout)

        self.load_button.clicked.connect(self.load_excel)
        self.train_button.clicked.connect(self.train_model)

    def load_excel(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Excelファイルを選択", "", "Excel Files (*.xlsx)")
        if file_path:
            xls = pd.ExcelFile(file_path)
            self.sheet_select.clear()
            self.sheet_select.addItems(xls.sheet_names)
            self.excel_data = xls

    def train_model(self):
        sheet_name = self.sheet_select.currentText()
        df = self.excel_data.parse(sheet_name)

        df = df.rename(columns=lambda x: x.strip())
        df = df.dropna(subset=['2nd.放電容量'])
        df = df[df['2nd.放電容量'].apply(lambda x: str(x).replace('.', '', 1).isdigit())]  # 数値以外除外
        df['2nd.放電容量'] = df['2nd.放電容量'].astype(float)

        feature_cols = [col for col in df.columns if col in ['活物質', '導電助剤', 'binder', '電解液', 'セパレータ', 'ロード量'] and col in df.columns]
        X = df[feature_cols]
        y = df['2nd.放電容量']

        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ], remainder='passthrough')

        model = Pipeline([
            ("pre", preprocessor),
            ("reg", RandomForestRegressor(random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        df.loc[:, '2nd.放電容量'] = df['2nd.放電容量'].astype(float)  # ←警告解消
        
        # RMSE計算の修正
        from sklearn.metrics import mean_squared_error
        import numpy as np
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        self.model = model
        self.df = df

        self.result_view.setText(f"モデル学習完了！\nR²: {r2:.3f}\nRMSE: {rmse:.2f}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MLApp()
    win.show()
    sys.exit(app.exec_())
