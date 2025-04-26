# catboost_shap_gui.py
# --------------------------------------------------------------
# CatBoost + SHAP + HeatMap + Optuna ベイズ最適化 GUI
#   * 全16パラメータ対応
#   * SHAP ≤0.42 互換 (ax 指定を回避)
#   * SHAP summary で max_display=16 を指定
#   * 単独・全ペア HeatMap
#   * Optuna ベイズ最適化 (2ステップ溶媒比サンプリング)
#   * Optuna 全試行結果散布図
# --------------------------------------------------------------

import sys
import platform
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QComboBox, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QProgressBar,
    QTabWidget, QTableWidget, QTableWidgetItem
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import shap
import optuna

# ---------- 日本語フォント設定 ----------
def set_jp_font():
    plat = platform.system()
    if plat == "Darwin":
        fam = "Hiragino Maru Gothic Pro"
    elif plat == "Windows":
        for cand in ["Yu Gothic", "Meiryo", "MS Gothic"]:
            if cand in [f.name for f in fm.fontManager.ttflist]:
                fam = cand; break
        else:
            fam = "MS Gothic"
    else:
        fam = "DejaVu Sans"
    plt.rcParams["font.family"] = fam
    plt.rcParams["axes.unicode_minus"] = False

set_jp_font()

class CatBoostShapApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CatBoost + SHAP GUI (HeatMap & Optuna)")
        self.resize(950, 750)

        # UI
        self.label        = QLabel("Excel ファイルを選択してください")
        self.btn_load     = QPushButton("ファイルを選択…")
        self.btn_train    = QPushButton("学習＆SHAP")
        self.heatmap_x_cb = QComboBox()
        self.heatmap_y_cb = QComboBox()
        self.btn_heatmap  = QPushButton("HeatMap 再描画")
        self.btn_allhm    = QPushButton("全ペアHeatMap")
        self.btn_optuna   = QPushButton("ベイズ最適化")
        self.btn_scatter  = QPushButton("Optuna 散布図")
        for b in (self.btn_train, self.btn_heatmap, self.btn_allhm, self.btn_optuna, self.btn_scatter):
            b.setEnabled(False)
        self.progress = QProgressBar()

        # Tabs
        self.tabs = QTabWidget()
        # SHAP summary
        self.fig_shap, _ = plt.subplots(figsize=(6,4))
        self.canvas_shap = FigureCanvas(self.fig_shap)
        self.tabs.addTab(self.canvas_shap, "SHAP Summary")
        # single heatmap
        self.fig_hm, _   = plt.subplots(figsize=(6,4))
        self.canvas_hm   = FigureCanvas(self.fig_hm)
        self.tabs.addTab(self.canvas_hm, "HeatMap")
        # optuna table
        self.table_optuna = QTableWidget()
        self.tabs.addTab(self.table_optuna, "Optuna 結果")

        # Layout
        top = QHBoxLayout()
        top.addWidget(self.btn_load)
        top.addWidget(self.btn_train)
        top.addWidget(QLabel("X軸:"))
        top.addWidget(self.heatmap_x_cb)
        top.addWidget(QLabel("Y軸:"))
        top.addWidget(self.heatmap_y_cb)
        top.addWidget(self.btn_heatmap)
        top.addWidget(self.btn_allhm)
        top.addWidget(self.btn_optuna)
        top.addWidget(self.btn_scatter)

        main = QVBoxLayout(self)
        main.addWidget(self.label)
        main.addLayout(top)
        main.addWidget(self.progress)
        main.addWidget(self.tabs)

        # Signals
        self.btn_load.clicked.connect(self.load_file)
        self.btn_train.clicked.connect(self.train_and_shap)
        self.btn_heatmap.clicked.connect(self.draw_heatmap)
        self.btn_allhm.clicked.connect(self.draw_all_heatmaps)
        self.btn_optuna.clicked.connect(self.run_optuna)
        self.btn_scatter.clicked.connect(self.draw_optuna_scatter)

        # Members
        self.file_path = None
        self.df        = None
        self.model     = None
        self.cat_cols  = None
        self.cat_idx   = None
        self.X_train   = None
        self.study     = None

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Excel ファイルを開く", "", "Excel Files (*.xlsx *.xls)"
        )
        if path:
            self.file_path = path
            self.label.setText(f"選択中: {path.split('/')[-1]}")
            self.btn_train.setEnabled(True)

    @staticmethod
    def preprocess(df):
        df['2nd.放電容量'] = pd.to_numeric(df['2nd.放電容量'], errors='coerce')
        df = df.dropna(subset=['ロード量']).copy()
        for c in df.columns:
            df[c] = df[c].fillna(0)
        return df

    def train_and_shap(self):
        try:
            self.progress.setValue(10)
            df = pd.read_excel(self.file_path, sheet_name="Raw_Data")
            df = self.preprocess(df)
            self.df = df

            y = df['2nd.放電容量'].values
            X = df.drop(columns=['2nd.放電容量'])
            # detect categorical
            self.cat_cols = X.select_dtypes('object').columns.tolist()
            # ensure導電助剤も categorical
            for c in ['導電助剤','塩2','溶媒3']:
                if c in X.columns and c not in self.cat_cols:
                    self.cat_cols.append(c)
            self.cat_idx = [X.columns.get_loc(c) for c in self.cat_cols]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            self.X_train = X_train

            pool_tr = Pool(X_train, y_train, cat_features=self.cat_idx)
            pool_te = Pool(X_test,  y_test,  cat_features=self.cat_idx)

            self.model = CatBoostRegressor(
                iterations=400, depth=6, learning_rate=0.05,
                loss_function='RMSE', random_state=42, verbose=0
            )
            self.model.fit(pool_tr)

            preds = self.model.predict(pool_te)
            mae, r2 = mean_absolute_error(y_test,preds), r2_score(y_test,preds)
            QMessageBox.information(self, "モデル評価", f"MAE={mae:.2f}\nR²={r2:.2f}")

            # SHAP summary with all 16 features
            explainer = shap.TreeExplainer(self.model)
            shap_vals = explainer.shap_values(self.X_train)
            self.fig_shap.clf()
            plt.figure(self.fig_shap.number)
            shap.summary_plot(
                shap_vals,
                self.X_train,
                show=False,
                max_display=16
            )
            self.canvas_shap.draw()

            # populate heatmap combos
            self.heatmap_x_cb.clear(); self.heatmap_y_cb.clear()
            for c in self.cat_cols:
                self.heatmap_x_cb.addItem(c)
                self.heatmap_y_cb.addItem(c)
            self.heatmap_x_cb.setCurrentText(self.cat_cols[0])
            self.heatmap_y_cb.setCurrentText(self.cat_cols[1])

            for b in (self.btn_heatmap, self.btn_allhm, self.btn_optuna, self.btn_scatter):
                b.setEnabled(True)
            self.progress.setValue(0)
        except Exception as e:
            QMessageBox.critical(self, "エラー", str(e))
            self.progress.setValue(0)

    def draw_heatmap(self):
        try:
            self.fig_hm.clf()
            preds = self.model.predict(
                Pool(self.df.drop(columns=['2nd.放電容量']), cat_features=self.cat_idx)
            )
            tmp = self.df.copy(); tmp['PredCap'] = preds
            xcol = self.heatmap_x_cb.currentText()
            ycol = self.heatmap_y_cb.currentText()
            pivot = tmp.groupby([ycol,xcol])['PredCap'].mean().unstack()

            ax = self.fig_hm.add_subplot(111)
            im = ax.imshow(pivot, cmap='viridis')
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=7)
            ax.set_xlabel(xcol); ax.set_ylabel(ycol)
            ax.set_title(f'{ycol}×{xcol} 平均予測容量')
            self.fig_hm.colorbar(im, ax=ax, label='mAh g$^{-1}$')
            self.canvas_hm.draw()
            self.tabs.setCurrentWidget(self.canvas_hm)
        except Exception as e:
            QMessageBox.critical(self, "エラー", str(e))

    def draw_all_heatmaps(self):
        try:
            preds = self.model.predict(
                Pool(self.df.drop(columns=['2nd.放電容量']), cat_features=self.cat_idx)
            )
            tmp = self.df.copy(); tmp['PredCap'] = preds
            cats = self.cat_cols
            pairs = [(cats[i],cats[j]) for i in range(len(cats)) for j in range(i+1,len(cats))]
            k = len(pairs)
            cols = math.ceil(math.sqrt(k)); rows = math.ceil(k/cols)
            fig = plt.figure(figsize=(4*cols,4*rows))
            for idx,(ycol,xcol) in enumerate(pairs, start=1):
                ax = fig.add_subplot(rows,cols,idx)
                pv = tmp.groupby([ycol,xcol])['PredCap'].mean().unstack()
                im = ax.imshow(pv, cmap='viridis')
                ax.set_xticks(range(len(pv.columns)))
                ax.set_xticklabels(pv.columns, rotation=90, fontsize=5)
                ax.set_yticks(range(len(pv.index)))
                ax.set_yticklabels(pv.index, fontsize=5)
                ax.set_xlabel(xcol,fontsize=6); ax.set_ylabel(ycol,fontsize=6)
            fig.suptitle("全カテゴリペア平均予測容量 HeatMap",fontsize=16)
            fig.tight_layout(rect=[0,0.03,1,0.95])
            fig.show()
        except Exception as e:
            QMessageBox.critical(self, "エラー", str(e))

    @staticmethod
    def populate_table(qt: QTableWidget, df: pd.DataFrame):
        qt.clear()
        qt.setRowCount(len(df)); qt.setColumnCount(df.shape[1])
        qt.setHorizontalHeaderLabels(df.columns.tolist())
        for i,row in df.iterrows():
            for j,val in enumerate(row):
                txt = "—" if pd.isna(val) else str(round(val,2)) if isinstance(val,(int,float)) else str(val)
                qt.setItem(i,j,QTableWidgetItem(txt))
        qt.resizeColumnsToContents()

    def run_optuna(self):
        if self.model is None:
            return
        self.progress.setValue(5)

        def nz(col):
            return [v for v in self.df[col].unique()
                    if not ((isinstance(v,str) and v=="0") or (isinstance(v,(int,float)) and v==0))]

        choices = {
            '活物質': nz('活物質'),
            '導電助剤': nz('導電助剤'),
            'バインダー': nz('バインダー'),
            '塩1': nz('塩1'), '塩2': nz('塩2'),
            '溶媒1': nz('溶媒1'), '溶媒2': nz('溶媒2'), '溶媒3': nz('溶媒3'),
            '添加剤1': nz('添加剤1')
        }
        ranges = {
            'ロード量': (int(self.df['ロード量'].min()), int(self.df['ロード量'].max())),
            '塩1濃度(M)': (0.1, 2.0), '塩2濃度(M)': (0.0, 2.0),
            '添加剤1量(%)': (0,10)
        }
        all_cols = list(self.X_train.columns)
        for c in ['塩2','塩2濃度(M)','溶媒3','溶媒3割合','導電助剤']:
            if c not in all_cols: all_cols.append(c)
        cat_set = set(self.cat_cols)

        sampler = optuna.samplers.TPESampler(seed=42,multivariate=True)
        study   = optuna.create_study(direction="minimize", sampler=sampler)
        self.study = study

        N_TRIAL = 500
        for n in range(N_TRIAL):
            def obj(trial):
                rec = {k: trial.suggest_categorical(k,v) for k,v in choices.items()}
                rec['ロード量']      = trial.suggest_int('ロード量', *ranges['ロード量'])
                rec['塩1濃度(M)']   = trial.suggest_float('塩1濃度(M)', *ranges['塩1濃度(M)'], step=0.05)
                rec['塩2濃度(M)']   = trial.suggest_float('塩2濃度(M)', *ranges['塩2濃度(M)'], step=0.05)
                rec['添加剤1量(%)'] = trial.suggest_int('添加剤1量(%)', *ranges['添加剤1量(%)'])
                # 2-step solvent ratios
                s1 = trial.suggest_int('溶媒1割合',0,100)
                s2 = trial.suggest_int('溶媒2割合',0,100-s1)
                rec['溶媒1割合'],rec['溶媒2割合'],rec['溶媒3割合'] = s1,s2,100-s1-s2
                row = {c:(rec[c] if c in rec else ("0" if c in cat_set else 0)) for c in all_cols}
                cap = self.model.predict(Pool(pd.DataFrame([row]),cat_features=self.cat_idx))[0]
                return -cap

            study.optimize(obj,n_trials=1,catch=(Exception,))
            if n%10==9:
                self.progress.setValue(5+int(90*(n+1)/N_TRIAL))
                QApplication.processEvents()

        self.progress.setValue(95)
        completed = [t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
        completed.sort(key=lambda t: t.value)
        rows=[] 
        for t in completed[:10]:
            d={**t.params,'PredCap':-t.value}
            rows.append(d)
        df_top=pd.DataFrame(rows).reindex(columns=[
            '活物質','導電助剤','バインダー','ロード量',
            '塩1','塩1濃度(M)','塩2','塩2濃度(M)',
            '溶媒1','溶媒1割合','溶媒2','溶媒2割合','溶媒3','溶媒3割合',
            '添加剤1','添加剤1量(%)','PredCap'
        ])
        self.populate_table(self.table_optuna,df_top)
        self.tabs.setCurrentWidget(self.table_optuna)
        QMessageBox.information(
            self,"Optuna 最良レシピ",
            "\n".join(f"{k}: {df_top.iloc[0][k]}" for k in df_top.columns)
        )
        self.progress.setValue(0)

    def draw_optuna_scatter(self):
        if self.study is None:
            return
        df_trials=pd.DataFrame([
            {**t.params,'PredCap':-t.value}
            for t in self.study.trials
            if t.state==optuna.trial.TrialState.COMPLETE
        ])
        # filter out penalty runs
        df_trials=df_trials[df_trials['PredCap']>-1e5]
        num_cols=[
            'ロード量','塩1濃度(M)','塩2濃度(M)',
            '溶媒1割合','溶媒2割合','溶媒3割合',
            '添加剤1量(%)'
        ]
        k=len(num_cols); cols=3; rows=math.ceil(k/cols)
        fig,axes=plt.subplots(rows,cols,figsize=(5*cols,4*rows))
        axes=axes.flatten()
        for ax,col in zip(axes,num_cols):
            if col in df_trials:
                ax.scatter(df_trials[col],df_trials['PredCap'],alpha=0.6)
                ax.set_xlabel(col); ax.set_ylabel('PredCap')
                ax.set_title(f'{col} vs PredCap')
            else:
                ax.set_visible(False)
        fig.tight_layout(); fig.show()

if __name__=="__main__":
    app=QApplication(sys.argv)
    w=CatBoostShapApp(); w.show()
    sys.exit(app.exec_())
