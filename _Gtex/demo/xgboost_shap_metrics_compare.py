# -*- coding: utf-8 -*-
# SHAP Beeswarm GUI with XGBoost + Metrics Tab + Algorithm Comparison
# Features:
#  - SHAP summary (bar + beeswarm)
#  - Actual vs Predicted scatter plot
#  - SHAP dependence plot
#  - Metrics tab showing RMSE, MAE, R2 on test split
#  - One-click comparison of multiple regression algorithms

import sys, os, re, traceback, unicodedata, math
import pandas as pd
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView
)
import shap
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
import warnings

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib import font_manager

# --- Japanese font ---
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

# --- helper functions from original code (wide_transform etc.) ---
COMPOSITE_LABELS = {"msc/benzene": "msc_benzene", "paa+ppt": "paa_ppt"}

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
    valid_tokens = {t for t,c in counter.items() if c>=min_count}
    def safe_col(t): return f"{prefix}_{re.sub(r'[^0-9A-Za-z\u3040-\u30ff\u4e00-\u9faf]', '_', t)}"
    for tok in sorted(valid_tokens):
        new_col = safe_col(tok)
        pattern = fr"(?i)(^|[+&/;,])\s*{re.escape(tok)}\s*($|[+&/;,])"
        df[new_col] = (
            df[col].fillna("")
            .apply(unify_composites)
            .str.contains(pattern)
            .astype(int)
        )
    df.drop(columns=[col], inplace=True)
    return df

def wide_transform(df):
    df = df.copy()
    salts = pd.unique(df[["塩1","塩2"]].values.ravel("K"))
    solvents = pd.unique(df[["溶媒1","溶媒2","溶媒3"]].values.ravel("K"))
    salts = [s for s in salts if pd.notna(s)]
    solvents = [s for s in solvents if pd.notna(s)]
    for s in salts: df[f"salt_{s}"] = 0.0
    for s in solvents: df[f"sol_{s}"] = 0.0
    for i in (1,2):
        nm,val = f"塩{i}",f"塩{i}濃度(M)"
        if nm in df and val in df:
            mask = df[nm].notna() & df[val].notna()
            for s in salts:
                m = mask & (df[nm]==s)
                df.loc[m, f"salt_{s}"] = pd.to_numeric(df.loc[m,val], errors="coerce").fillna(0.0)
    for i in (1,2,3):
        nm,val = f"溶媒{i}",f"溶媒{i}割合"
        if nm in df and val in df:
            mask = df[nm].notna() & df[val].notna()
            for s in solvents:
                m = mask & (df[nm]==s)
                df.loc[m, f"sol_{s}"] = pd.to_numeric(df.loc[m,val], errors="coerce").fillna(0.0)
    if "添加剤" in df.columns and "添加剤量(%)" in df.columns:
        df["add_norm"] = df["添加剤"].fillna("").apply(unify_composites).apply(normalize_token)
        additives = [a for a in df["add_norm"].unique() if a]
        for a in additives:
            col = f"additive_{re.sub(r'[^0-9A-Za-z\u3040-\u30ff\u4e00-\u9faf]', '_', a)}"
            df[col] = 0.0
            mask = df["add_norm"] == a
            df.loc[mask, col] = pd.to_numeric(df.loc[mask,"添加剤量(%)"], errors="coerce").fillna(0.0)
        df.drop(columns=["添加剤","add_norm","添加剤量(%)"], inplace=True)
    for col,pfx in [("活物質","act"),("導電助剤","add"),("バインダー","bind")]:
        if col in df.columns:
            df = encode_multilabel(df, col, pfx)
    df.drop(columns=[c for c in df.columns if c.startswith("塩") or c.startswith("溶媒")], inplace=True, errors=True)
    return df.apply(pd.to_numeric, errors="ignore").fillna(0.0)

# --- main GUI class ---
class ShapApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SHAP GUI + Metrics & Comparison")
        self.resize(1500, 800)
        self.df = self.model = self.shap_values = self.shap_features = None
        self.scaler_y = None

        self.main = QtWidgets.QHBoxLayout(self)
        self._build_left_panel()
        self._build_tabs()

    # ---------- UI ----------
    def _build_left_panel(self):
        panel = QtWidgets.QVBoxLayout()
        self.main.addLayout(panel, 0)

        self.btn_load = QtWidgets.QPushButton("Load Excel")
        self.btn_load.clicked.connect(self.load_excel)
        panel.addWidget(self.btn_load)

        panel.addWidget(QtWidgets.QLabel("Target (目的変数)"))
        self.cmb_target = QtWidgets.QComboBox()
        panel.addWidget(self.cmb_target)

        self.btn_run = QtWidgets.QPushButton("Run SHAP (XGBoost)")
        self.btn_run.clicked.connect(self.run_shap)
        self.btn_run.setEnabled(False)
        panel.addWidget(self.btn_run)

        self.btn_compare = QtWidgets.QPushButton("Run Algorithm Comparison")
        self.btn_compare.clicked.connect(self.run_comparison)
        self.btn_compare.setEnabled(False)
        panel.addWidget(self.btn_compare)

        self.btn_plot_pred = QtWidgets.QPushButton("Actual vs Predicted (XGB)")
        self.btn_plot_pred.clicked.connect(self.plot_actual_vs_predicted)
        self.btn_plot_pred.setEnabled(False)
        panel.addWidget(self.btn_plot_pred)

        panel.addWidget(QtWidgets.QLabel("Feature for Dependence Plot"))
        self.cmb_feature = QtWidgets.QComboBox()
        panel.addWidget(self.cmb_feature)

        self.btn_depend = QtWidgets.QPushButton("SHAP Dependence Plot")
        self.btn_depend.clicked.connect(self.plot_dependence)
        self.btn_depend.setEnabled(False)
        panel.addWidget(self.btn_depend)

        panel.addStretch(1)
        self.txt_log = QtWidgets.QPlainTextEdit(readOnly=True)
        panel.addWidget(self.txt_log, 1)

    def _build_tabs(self):
        self.tabs = QTabWidget()
        self.main.addWidget(self.tabs, 1)

        # Tab 0: Plot area (scroll)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.tabs.addTab(self.scroll_area, "Plots")

        # Tab 1: Metrics table
        self.tbl_metrics = QTableWidget(0, 4)
        self.tbl_metrics.setHorizontalHeaderLabels(["Algorithm", "RMSE", "MAE", "R²"])
        self.tbl_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.tbl_metrics, "Metrics")

        # initialize empty widget inside scroll_area
        fig = plt.Figure(figsize=(6,4))
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.scroll_area.setWidget(container)

    # ---------- Data loading ----------
    def load_excel(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Excel", "", "Excel (*.xlsx *.xls)"
        )
        if not path:
            return
        try:
            df_raw = pd.read_excel(path, na_values=["", " "])
            self.df = wide_transform(df_raw)
            self.cmb_target.clear()
            self.cmb_target.addItems(self.df.columns.astype(str).tolist())
            self.btn_run.setEnabled(True)
            self.btn_compare.setEnabled(True)
            self.btn_plot_pred.setEnabled(False)
            self.btn_depend.setEnabled(False)
            self._reset_canvas()
            self.log(f"Loaded {os.path.basename(path)}")
        except Exception as e:
            self.err(e)

    # ---------- SHAP workflow for XGB ----------
    def run_shap(self):
        try:
            target = self.cmb_target.currentText()
            y_raw = pd.to_numeric(self.df[target], errors="coerce")
            X = self.df.drop(columns=[target])
            mask = y_raw.notna()
            y_raw, X = y_raw[mask], X[mask]
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

            self.scaler_y = StandardScaler()
            y_scaled = self.scaler_y.fit_transform(y_raw.values.reshape(-1,1)).ravel()

            self.model = xgb.XGBRegressor(
                n_estimators=500, max_depth=3, learning_rate=0.05, verbosity=0,
                random_state=42
            )
            self.model.fit(X, y_scaled)

            explainer = shap.Explainer(self.model)
            self.shap_values = explainer(X)
            self.shap_features = X

            self._draw_shap_summary()
            self.btn_plot_pred.setEnabled(True)
            self.cmb_feature.clear()
            self.cmb_feature.addItems(self.shap_features.columns.tolist())
            self.btn_depend.setEnabled(True)
            self.log("SHAP summary plotted.")
        except Exception as e:
            self.err(e)

    def _draw_shap_summary(self):
        shap_vals = self.shap_values.values
        mean_shap = np.abs(shap_vals).mean(axis=0)
        top_n = 20
        idx = np.argsort(mean_shap)[::-1][:top_n]
        feats = [self.shap_features.columns[i] for i in idx]
        means = mean_shap[idx]
        vals = shap_vals[:, idx]
        Xv = self.shap_features[feats].values

        plt.close("all")
        fig, (ax_bar, ax_bee) = plt.subplots(
            1, 2, figsize=(14, 0.5*top_n+2), width_ratios=[1,2]
        )
        fig.subplots_adjust(wspace=0.05)

        y_pos = np.arange(top_n)
        ax_bar.barh(y_pos, means, color="#1f77b4", height=0.6)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(feats, fontsize=11)
        ax_bar.set_ylim(top_n-0.5, -0.5)
        ax_bar.set_xlabel("Mean(|SHAP|)", fontsize=12)
        ax_bar.set_title("Global feature importance", fontsize=13)
        ax_bar.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax_bar.set_xlim(left=0)

        cmap = cm.get_cmap("coolwarm")
        for i in range(top_n):
            nv = vals[:, i]
            fv = Xv[:, i]
            norm = (fv - fv.min()) / (np.ptp(fv) + 1e-8)
            ax_bee.scatter(nv, np.full_like(nv, i), c=cmap(norm), alpha=0.6, s=18, edgecolor='none')

        ax_bee.set_yticks(y_pos)
        ax_bee.set_yticklabels(['']*top_n)
        ax_bee.set_ylim(top_n-0.5, -0.5)
        ax_bee.set_xlabel("SHAP value")
        ax_bee.set_title("Local explanation summary", fontsize=13)
        ax_bee.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax_bee.axvline(0, color='gray', linewidth=1)

        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax_bee, pad=0.01).set_label("Feature value", rotation=270, labelpad=15)

        # embed in scroll_area
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.scroll_area.setWidget(container)
        self.tabs.setCurrentIndex(0)

    # ---------- Additional plots ----------
    def plot_actual_vs_predicted(self):
        try:
            if self.model is None:
                self.log("Run SHAP first.")
                return
            target = self.cmb_target.currentText()
            y_act = pd.to_numeric(self.df[target], errors="coerce").dropna()
            X = self.df.drop(columns=[target]).loc[y_act.index]
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            y_pred_s = self.model.predict(X)
            y_pred = self.scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).ravel()

            fig, ax = plt.subplots(figsize=(6,6))
            ax.scatter(y_act, y_pred, alpha=0.7, label='Data')
            mn, mx = y_act.min(), y_act.max()
            ax.plot([mn,mx],[mn,mx],'r--', linewidth=2, label='Ideal')
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted (XGB)"); ax.legend(); ax.grid(True)
            plt.tight_layout(); plt.show()
        except Exception as e:
            self.err(e)

    def plot_dependence(self):
        try:
            feat = self.cmb_feature.currentText()
            if not feat:
                return
            shap.dependence_plot(
                feat, self.shap_values.values, self.shap_features,
                show=False
            )
            plt.tight_layout(); plt.show()
        except Exception as e:
            self.err(e)

    # ---------- Algorithm comparison ----------
    def run_comparison(self):
        try:
            target = self.cmb_target.currentText()
            data = self.df.copy()
            y = pd.to_numeric(data[target], errors="coerce")
            X = data.drop(columns=[target])
            mask = y.notna()
            y, X = y[mask], X[mask]
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

            # standardize y for models that are sensitive (we will inverse later)
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y.values.reshape(-1,1)).ravel()

            X_train, X_test, y_train, y_test_scaled = train_test_split(
                X, y_scaled, test_size=0.2, random_state=42
            )
            y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1,1)).ravel()

            algorithms = {
                "Lasso": Lasso(alpha=0.001, max_iter=10000, random_state=42),
                "Ridge": Ridge(alpha=1.0, random_state=42),
                "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=42),
                "PLS": PLSRegression(n_components=min(10, X_train.shape[1])),
                "SVR(RBF)": SVR(kernel='rbf', C=10, gamma='scale'),
                "RandomForest": RandomForestRegressor(n_estimators=500, random_state=42),
                "XGBoost": xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
            }
            if HAS_LGBM:
                algorithms["LightGBM"] = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)

            # metrics storage
            rows = []
            for name, model in algorithms.items():
                try:
                    model.fit(X_train, y_train:=y_train if 'y_train' in locals() else y_train)
                except:
                    model.fit(X_train, y_train)
                pred_scaled = model.predict(X_test)
                if pred_scaled.ndim>1:
                    pred_scaled = pred_scaled.ravel()
                y_pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).ravel()
                rmse = math.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rows.append((name, rmse, mae, r2))
                self.log(f"{name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
            # update table
            self.tbl_metrics.setRowCount(len(rows))
            for r,(alg,rmse,mae,r2) in enumerate(sorted(rows, key=lambda x:x[1])):
                self.tbl_metrics.setItem(r,0,QTableWidgetItem(alg))
                self.tbl_metrics.setItem(r,1,QTableWidgetItem(f"{rmse:.3f}"))
                self.tbl_metrics.setItem(r,2,QTableWidgetItem(f"{mae:.3f}"))
                self.tbl_metrics.setItem(r,3,QTableWidgetItem(f"{r2:.3f}"))
            self.tabs.setCurrentIndex(1)
        except Exception as e:
            self.err(e)

    # ---------- utils ----------
    def _reset_canvas(self):
        self.scroll_area.takeWidget()
        fig = plt.Figure(figsize=(6,4))
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

if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    warnings.filterwarnings("ignore", category=UserWarning)
    w = ShapApp()
    w.show()
    sys.exit(app.exec_())
