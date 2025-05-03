# -*- coding: utf-8 -*-
# SHAP Beeswarm GUI with XGBoost + NavigationToolbar

import sys, os, re, traceback, unicodedata
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QScrollArea
import shap
import xgboost as xgb
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib import font_manager

# --- 日本語フォント自動設定 ---
jp_font_candidates = [
    "IPAexGothic", "IPA Gothic", "Noto Sans CJK JP",
    "Yu Gothic", "Meiryo", "Hiragino Sans", "TakaoPGothic", "VL Gothic"
]
installed = {f.name for f in font_manager.fontManager.ttflist}
for jp in jp_font_candidates:
    if jp in installed:
        plt.rcParams["font.family"] = jp
        break
plt.rcParams["axes.unicode_minus"] = False

COMPOSITE_LABELS = {
    "msc/benzene": "msc_benzene",
    "paa+ppt": "paa_ppt"
}

def normalize_token(tok):
    t = unicodedata.normalize("NFKC", tok.strip()).lower()
    return re.sub(r"[.\s]+$", "", t)

def unify_composites(txt):
    out = str(txt)
    for key, repl in COMPOSITE_LABELS.items():
        out = re.sub(re.escape(key), repl, out, flags=re.IGNORECASE)
    return out

def encode_multilabel(df, col, prefix, sep=r"[+&/;,]+", min_count=2):
    df = df.copy()
    counter = {}
    for raw in df[col].dropna():
        text = unify_composites(raw)
        for part in re.split(sep, text):
            tok = normalize_token(part)
            if tok:
                counter[tok] = counter.get(tok, 0) + 1
    valid_tokens = {t for t, c in counter.items() if c >= min_count}
    def safe_col(t): return f"{prefix}_{re.sub(r'[^0-9A-Za-z\u3040-\u30ff\u4e00-\u9faf]', '_', t)}"
    for tok in sorted(valid_tokens):
        new_col = safe_col(tok)
        pattern = fr"(?i)(^|[+&/;,])\s*{re.escape(tok)}\s*($|[+&/;,])"
        df[new_col] = (df[col]
                       .fillna("")
                       .apply(unify_composites)
                       .str.contains(pattern)
                       .astype(int))
    df.drop(columns=[col], inplace=True)
    return df

def wide_transform(df):
    df = df.copy()
    salts = pd.unique(df[["塩1", "塩2"]].values.ravel("K"))
    solvents = pd.unique(df[["溶媒1", "溶媒2", "溶媒3"]].values.ravel("K"))
    salts = [s for s in salts if pd.notna(s)]
    solvents = [s for s in solvents if pd.notna(s)]
    for s in salts:
        df[f"salt_{s}"] = 0.0
    for s in solvents:
        df[f"sol_{s}"] = 0.0
    for i in (1, 2):
        nm, val = f"塩{i}", f"塩{i}濃度(M)"
        if nm in df and val in df:
            mask_all = df[nm].notna() & df[val].notna()
            for s in salts:
                m = mask_all & (df[nm] == s)
                df.loc[m, f"salt_{s}"] = pd.to_numeric(df.loc[m, val], errors="coerce").fillna(0.0)
    for i in (1, 2, 3):
        nm, val = f"溶媒{i}", f"溶媒{i}割合"
        if nm in df and val in df:
            mask_all = df[nm].notna() & df[val].notna()
            for s in solvents:
                m = mask_all & (df[nm] == s)
                df.loc[m, f"sol_{s}"] = pd.to_numeric(df.loc[m, val], errors="coerce").fillna(0.0)
    if "添加剤" in df.columns and "添加剤量(%)" in df.columns:
        df["add_norm"] = (df["添加剤"].fillna("").apply(unify_composites).apply(normalize_token))
        additives = [a for a in df["add_norm"].unique() if a]
        for a in additives:
            col = f"additive_{re.sub(r'[^0-9A-Za-z\u3040-\u30ff\u4e00-\u9faf]', '_', a)}"
            df[col] = 0.0
            mask = df["add_norm"] == a
            df.loc[mask, col] = pd.to_numeric(df.loc[mask, "添加剤量(%)"], errors="coerce").fillna(0.0)
        df.drop(columns=["添加剤", "add_norm", "添加剤量(%)"], inplace=True)
    for col, pfx in [("活物質", "act"), ("導電助剤", "add"), ("バインダー", "bind")]:
        if col in df.columns:
            df = encode_multilabel(df, col, pfx)
    df.drop(columns=[c for c in df.columns if c.startswith("塩") or c.startswith("溶媒")], inplace=True, errors=True)
    return df.apply(pd.to_numeric, errors="ignore").fillna(0.0)

class ShapApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SHAP Beeswarm GUI (XGBoost + Toolbar)")
        self.resize(1400, 700)
        self.df = self.model = self.shap_values = self.shap_features = None
        self.main = QtWidgets.QHBoxLayout(self)
        self._build_left_panel()
        self._build_scroll_area()

    def _build_left_panel(self):
        panel = QtWidgets.QVBoxLayout()
        self.main.addLayout(panel, 0)
        self.btn_load = QtWidgets.QPushButton("Load Excel")
        self.btn_load.clicked.connect(self.load_excel)
        panel.addWidget(self.btn_load)
        panel.addWidget(QtWidgets.QLabel("Target (目的変数)"))
        self.cmb_target = QtWidgets.QComboBox()
        panel.addWidget(self.cmb_target)
        self.btn_run = QtWidgets.QPushButton("Run SHAP")
        self.btn_run.clicked.connect(self.run_shap)
        self.btn_run.setEnabled(False)
        panel.addWidget(self.btn_run)
        self.btn_export = QtWidgets.QPushButton("Export SHAP CSV")
        self.btn_export.clicked.connect(self.export_csv)
        self.btn_export.setEnabled(False)
        panel.addWidget(self.btn_export)
        panel.addStretch(1)
        self.txt_log = QtWidgets.QPlainTextEdit(readOnly=True)
        panel.addWidget(self.txt_log, 1)

    def _build_scroll_area(self):
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        fig = plt.Figure(figsize=(7, 5))
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.plot_layout = QtWidgets.QVBoxLayout()
        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas)
        container = QtWidgets.QWidget()
        container.setLayout(self.plot_layout)
        self.scroll_area.setWidget(container)
        self.main.addWidget(self.scroll_area, 1)

    def load_excel(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Excel", "", "Excel (*.xlsx *.xls);;All Files (*.*)")
        if not path:
            return
        try:
            df_raw = pd.read_excel(path, na_values=["", " "])
            self.df = wide_transform(df_raw)
            self.cmb_target.clear()
            self.cmb_target.addItems(self.df.columns.astype(str).tolist())
            self.btn_run.setEnabled(True)
            self.btn_export.setEnabled(False)
            self._reset_canvas()
            self.log(f"Loaded {os.path.basename(path)}")
        except Exception as e:
            self.err(e)


    
    def run_shap(self):
        try:
            import numpy as np
            import matplotlib.cm as cm
            from sklearn.preprocessing import StandardScaler

            target = self.cmb_target.currentText()
            y_raw = pd.to_numeric(self.df[target], errors="coerce")
            X = self.df.drop(columns=[target])
            mask = y_raw.notna()
            y_raw, X = y_raw[mask], X[mask]
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

            # --- 標準化（ここが今回の追加）---
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y_raw.values.reshape(-1, 1)).ravel()

            # --- モデル学習・SHAP解析 ---
            self.model = xgb.XGBRegressor(n_estimators=5000, max_depth=6, learning_rate=0.05, verbosity=0)
            self.model.fit(X, y_scaled)
            explainer = shap.Explainer(self.model)
            self.shap_values = explainer(X)
            self.shap_features = X

            # --- SHAP値処理・TopN抽出 ---
            shap_vals = self.shap_values.values
            mean_shap = np.abs(shap_vals).mean(axis=0)
            top_n = 20
            feature_order = np.argsort(mean_shap)[::-1][:top_n]
            ordered_features = [X.columns[i] for i in feature_order]
            ordered_means = mean_shap[feature_order]
            shap_selected = shap_vals[:, feature_order]
            X_selected = X[ordered_features].values

            # --- 描画準備 ---
            plt.close("all")
            fig, (ax_bar, ax_bee) = plt.subplots(1, 2, figsize=(14, 0.5 * top_n + 2), width_ratios=[1, 2])
            fig.subplots_adjust(wspace=0.05)
            y_pos = np.arange(top_n)

            # --- bar plot (左) ---
            ax_bar.barh(y_pos, ordered_means, color="#1f77b4", height=0.6)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(ordered_features, fontsize=11)
            ax_bar.set_ylim(top_n - 0.5, -0.5)
            ax_bar.set_xlabel("Mean(|SHAP value|)", fontsize=12)
            ax_bar.set_title("Global feature importance", fontsize=13)
            ax_bar.tick_params(axis='x', labelsize=10)
            ax_bar.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax_bar.set_xlim(left=0)

            # --- beeswarm plot (右) ---
            cmap = cm.get_cmap("coolwarm")
            for i in range(top_n):
                vals = shap_selected[:, i]
                feat_vals = X_selected[:, i]
                normed = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-8)
                ax_bee.scatter(vals, np.full_like(vals, i), c=cmap(normed), alpha=0.6, s=18, edgecolor='none')

            ax_bee.set_yticks(y_pos)
            ax_bee.set_yticklabels([''] * top_n)
            ax_bee.set_ylim(top_n - 0.5, -0.5)
            ax_bee.set_xlabel("SHAP value (impact on standardized output)", fontsize=12)
            ax_bee.set_title("Local explanation summary", fontsize=13)
            ax_bee.tick_params(axis='x', labelsize=10)
            ax_bee.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax_bee.axvline(0, color='gray', linewidth=1.0, linestyle='-')

            # --- カラーバー ---
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax_bee, pad=0.01)
            cbar.set_label("Feature value", rotation=270, labelpad=15, fontsize=11)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["Low", "High"])

            # --- GUI描画 ---
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, self)
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            container = QtWidgets.QWidget()
            container.setLayout(layout)
            self.scroll_area.takeWidget()
            self.scroll_area.setWidget(container)

            self.canvas = canvas
            self.toolbar = toolbar
            self.btn_export.setEnabled(True)
            self.log("SHAP summary (standardized y) plotted successfully.")

        except Exception as e:
            self.err(e)

    def export_csv(self):
        if self.shap_values is None:
            return
        try:
            df_out = pd.DataFrame(self.shap_values.values, columns=self.shap_features.columns)
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "shap_values.csv", "CSV (*.csv)")
            if path:
                df_out.to_csv(path, index=False)
                self.log(f"Saved to {path}")
        except Exception as e:
            self.err(e)

    def _reset_canvas(self):
        self.scroll_area.takeWidget()
        fig = plt.Figure(figsize=(7, 5))
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.scroll_area.setWidget(container)

    def log(self, msg):
        self.txt_log.appendPlainText(str(msg))

    def err(self, exc):
        self.txt_log.appendPlainText(f"Error: {exc}\n{traceback.format_exc()}")
        QtWidgets.QMessageBox.critical(self, "Error", str(exc))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = ShapApp()
    w.show()
    sys.exit(app.exec_())
