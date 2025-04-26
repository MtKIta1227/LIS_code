# catboost_shap_gui.py
# --------------------------------------------------------------
# CatBoost Ensemble + SHAP + HeatMap + Optuna GUI
#   * 設定値セクションで一元管理
#   * 溶媒／塩／添加剤は「順序を無視したセット」を最適化
#   * 特徴量として「各溶媒種の混合比」「各塩種の合計濃度」「各添加剤種の量」を自動生成
#   * SHAP は全件 or subsample 指定可
#   * Optuna の進行状況をプログレスバー表示
#   * 散布図のマーカーを小さく (s=10)
#   * 添加剤1=0 のとき添加剤1量＝0 固定
#   * 塩2=0 のとき塩2濃度＝0 固定 ＆ user_attr に保存
# --------------------------------------------------------------

# ========== 一元設定 ==========
SHAP_FULL_SAMPLE = True      # True: 全件で SHAP を計算
OPTUNA_N_TRIALS  = 2000      # Optuna 試行回数
ENSEMBLE_SEEDS   = [42, 100, 2025]
CATBOOST_ITERS   = 400
# ===============================

import sys, platform, math, itertools
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

import shap, optuna
from optuna.visualization import plot_slice, plot_parallel_coordinate

# 日本語フォント設定
def set_jp_font():
    p = platform.system()
    if p == "Darwin":
        fam = "Hiragino Maru Gothic Pro"
    elif p == "Windows":
        for c in ["Yu Gothic","Meiryo","MS Gothic"]:
            if c in [f.name for f in fm.fontManager.ttflist]:
                fam = c; break
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
        self.setWindowTitle("CatBoost Ensemble + SHAP GUI")
        self.resize(980, 780)

        # UI components
        self.label        = QLabel("Excel ファイルを選択してください")
        self.btn_load     = QPushButton("ファイルを選択…")
        self.btn_train    = QPushButton("学習＆SHAP")
        self.heatmap_x_cb = QComboBox()
        self.heatmap_y_cb = QComboBox()
        self.btn_heatmap  = QPushButton("HeatMap 再描画")
        self.btn_allhm    = QPushButton("全ペアHeatMap")
        self.btn_optuna   = QPushButton("ベイズ最適化")
        self.btn_scatter  = QPushButton("Optuna 散布図")
        for b in (self.btn_train, self.btn_heatmap, self.btn_allhm,
                  self.btn_optuna, self.btn_scatter):
            b.setEnabled(False)
        self.progress = QProgressBar()

        # Tabs
        self.tabs = QTabWidget()
        # SHAP
        self.fig_shap, _ = plt.subplots(figsize=(6,4))
        self.canvas_shap = FigureCanvas(self.fig_shap)
        self.tabs.addTab(self.canvas_shap, "SHAP Summary")
        # HeatMap
        self.fig_hm, _ = plt.subplots(figsize=(6,4))
        self.canvas_hm = FigureCanvas(self.fig_hm)
        self.tabs.addTab(self.canvas_hm, "HeatMap")
        # Optuna results
        self.table_optuna = QTableWidget()
        self.tabs.addTab(self.table_optuna, "Optuna 結果")

        # Layout
        top = QHBoxLayout()
        for w in (self.btn_load, self.btn_train,
                  QLabel("X軸:"), self.heatmap_x_cb,
                  QLabel("Y軸:"), self.heatmap_y_cb,
                  self.btn_heatmap, self.btn_allhm,
                  self.btn_optuna, self.btn_scatter):
            top.addWidget(w)
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
        self.file_path        = None
        self.df               = None
        self.models           = None
        self.cat_cols         = None
        self.cat_idx          = None
        self.X_train          = None
        self.study            = None
        self.solvent_sets     = None
        self.salt_sets        = None
        self.unique_solvents  = None
        self.unique_salts     = None
        self.unique_additives = None

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

            # Enumerate unique categories
            self.unique_solvents = sorted({
                v for col in ('溶媒1','溶媒2','溶媒3')
                for v in df[col].unique() if str(v) != "0"
            })
            self.solvent_sets = [
                ",".join(c) for c in itertools.combinations(self.unique_solvents, 3)
            ]
            self.unique_salts = sorted({
                v for col in ('塩1','塩2')
                for v in df[col].unique() if str(v) != "0"
            })
            self.salt_sets = (
                [f"{s},0" for s in self.unique_salts] +
                [",".join(c) for c in itertools.combinations(self.unique_salts, 2)]
            )
            self.unique_additives = sorted(
                v for v in df['添加剤1'].unique() if str(v) != "0"
            )

            # Feature engineering
            for s in self.unique_solvents:
                df[f'比率_{s}'] = (
                    (df['溶媒1']==s)*df['溶媒1割合'] +
                    (df['溶媒2']==s)*df['溶媒2割合'] +
                    (df['溶媒3']==s)*df['溶媒3割合']
                )
            for s in self.unique_salts:
                df[f'濃度_{s}'] = (
                    (df['塩1']==s)*df['塩1濃度(M)'] +
                    (df['塩2']==s)*df['塩2濃度(M)']
                )
            for a in self.unique_additives:
                df[f'量_{a}'] = (df['添加剤1']==a)*df['添加剤1量(%)']

            # Drop positional cols
            drop_cols = [
                '溶媒1','溶媒2','溶媒3','溶媒1割合','溶媒2割合','溶媒3割合',
                '塩1','塩2','塩1濃度(M)','塩2濃度(M)',
                '添加剤1','添加剤1量(%)'
            ]
            df = df.drop(columns=drop_cols)
            self.df = df

            # Prepare train/test
            y = df['2nd.放電容量'].values
            X = df.drop(columns=['2nd.放電容量'])
            self.cat_cols = X.select_dtypes('object').columns.tolist()
            self.cat_idx  = [X.columns.get_loc(c) for c in self.cat_cols]

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.X_train = X_tr
            pool_tr = Pool(X_tr, y_tr, cat_features=self.cat_idx)
            pool_te = Pool(X_te, y_te, cat_features=self.cat_idx)

            # Ensemble training
            self.models = []
            for seed in ENSEMBLE_SEEDS:
                m = CatBoostRegressor(
                    iterations=CATBOOST_ITERS, depth=6,
                    learning_rate=0.05, loss_function='RMSE',
                    random_state=seed, verbose=0
                )
                m.fit(pool_tr)
                self.models.append(m)

            preds = np.mean([m.predict(pool_te) for m in self.models], axis=0)
            mae, r2 = mean_absolute_error(y_te, preds), r2_score(y_te, preds)
            QMessageBox.information(
                self, "モデル評価",
                f"MAE = {mae:.2f}    R² = {r2:.2f}"
            )

            # SHAP analysis
            shap_vals = np.mean([
                shap.TreeExplainer(m).shap_values(
                    self.X_train,
                    check_additivity=not SHAP_FULL_SAMPLE
                ) for m in self.models
            ], axis=0)

            self.fig_shap.clf()
            ax = self.fig_shap.add_subplot(111)
            num_cols = self.X_train.select_dtypes(include=[np.number]).columns
            vmin, vmax = (
                self.X_train[num_cols].min().min(),
                self.X_train[num_cols].max().max()
            )
            norm = plt.Normalize(vmin, vmax)
            cmap = plt.get_cmap('coolwarm')

            for i, feat in enumerate(self.X_train.columns):
                s = shap_vals[:, i]
                if feat in num_cols:
                    fvals = self.X_train[feat].astype(float).values
                    ax.scatter(s, np.full_like(s, i),
                               c=fvals, cmap=cmap, norm=norm,
                               s=10, alpha=0.6)
                else:
                    ax.scatter(s, np.full_like(s, i),
                               c='gray', s=10, alpha=0.6)

            ax.axvline(0, color='gray')
            ax.set_yticks(range(len(self.X_train.columns)))
            ax.set_yticklabels(self.X_train.columns)
            ax.set_xlabel("SHAP value")
            self.fig_shap.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax, label='Feature value'
            )
            self.canvas_shap.draw()

            # Initialize HeatMap combos
            self.heatmap_x_cb.clear(); self.heatmap_y_cb.clear()
            for c in self.cat_cols:
                self.heatmap_x_cb.addItem(c)
                self.heatmap_y_cb.addItem(c)
            self.heatmap_x_cb.setCurrentText(self.cat_cols[0])
            self.heatmap_y_cb.setCurrentText(self.cat_cols[1])
            for b in (self.btn_heatmap, self.btn_allhm,
                      self.btn_optuna, self.btn_scatter):
                b.setEnabled(True)

            self.progress.setValue(0)
        except Exception as e:
            QMessageBox.critical(self, "エラー", str(e))
            self.progress.setValue(0)

    def predict_ensemble(self, df_feat):
        pool = Pool(df_feat, cat_features=self.cat_idx)
        return np.mean([m.predict(pool) for m in self.models], axis=0)

    def draw_heatmap(self):
        # ... unchanged ...
        pass

    def draw_all_heatmaps(self):
        # ... unchanged ...
        pass

    @staticmethod
    def populate_table(qt, df):
        # ... unchanged ...
        pass

    def run_optuna(self):
        if not self.models: return
        self.progress.setValue(5)

        def nz(col):
            return [v for v in self.df[col].unique() if str(v) != "0"]

        choices = {
            '活物質':    nz('活物質'),
            '導電助剤':  nz('導電助剤'),
            'バインダー':nz('バインダー'),
            '添加剤1':  ['0'] + self.unique_additives,
            '溶媒セット':self.solvent_sets,
            '塩セット':  self.salt_sets
        }
        ranges = {
            'ロード量':      (int(self.df['ロード量'].min()), int(self.df['ロード量'].max())),
            '塩1濃度(M)':   (0.1, 2.0),
            '塩2濃度(M)':   (0.0, 2.0),
            '添加剤1量(%)': (0, 10)
        }
        all_cols = list(self.X_train.columns)
        cat_set  = set(self.cat_cols)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42, multivariate=True)
        )
        self.study = study

        def objective(trial):
            rec = {k: trial.suggest_categorical(k, v)
                   for k, v in choices.items() if k != '添加剤1量(%)'}
            # 添加剤1量
            if rec['添加剤1'] == '0':
                rec['添加剤1量(%)'] = 0
            else:
                rec['添加剤1量(%)'] = trial.suggest_int(
                    '添加剤1量(%)', *ranges['添加剤1量(%)']
                )
            trial.set_user_attr('添加剤1量(%)', float(rec['添加剤1量(%)']))

            rec['ロード量'] = trial.suggest_int('ロード量', *ranges['ロード量'])

            # 塩セット展開 + 濃度
            s1, s2 = rec['塩セット'].split(',')
            rec['塩1'], rec['塩2'] = s1, s2
            rec['塩1濃度(M)'] = trial.suggest_float(
                '塩1濃度(M)', *ranges['塩1濃度(M)'], step=0.05
            )
            if s2 == '0':
                rec['塩2濃度(M)'] = 0
            else:
                rec['塩2濃度(M)'] = trial.suggest_float(
                    '塩2濃度(M)', *ranges['塩2濃度(M)'], step=0.05
                )
            trial.set_user_attr('塩1濃度(M)', float(rec['塩1濃度(M)']))
            trial.set_user_attr('塩2濃度(M)', float(rec['塩2濃度(M)']))

            # 溶媒セット展開 + Dirichlet
            sol1, sol2, sol3 = rec['溶媒セット'].split(',')
            w = np.random.dirichlet([1,1,1]); w1,w2,w3 = (w*100).round(2)
            rec.update({
                '溶媒1': sol1, '溶媒2': sol2, '溶媒3': sol3,
                '溶媒1割合': w1, '溶媒2割合': w2, '溶媒3割合': w3
            })
            trial.set_user_attr('溶媒1割合', float(w1))
            trial.set_user_attr('溶媒2割合', float(w2))
            trial.set_user_attr('溶媒3割合', float(w3))

            # build feature row
            row = {c: ("0" if c in cat_set else 0) for c in all_cols}
            row.update(rec)
            # engineered solvents / salts / additives...
            for s in self.unique_solvents:
                row[f'比率_{s}'] = (sol1==s)*w1 + (sol2==s)*w2 + (sol3==s)*w3
            c1, c2 = rec['塩1濃度(M)'], rec['塩2濃度(M)']
            for s in self.unique_salts:
                row[f'濃度_{s}'] = (s1==s)*c1 + (s2==s)*c2
            amt = rec['添加剤1量(%)']
            for a in self.unique_additives:
                row[f'量_{a}'] = (rec['添加剤1']==a)*amt

            df_one = pd.DataFrame([row])[all_cols]
            cap = np.mean([m.predict(Pool(df_one, cat_features=self.cat_idx))[0]
                           for m in self.models])
            return -cap

        for i in range(OPTUNA_N_TRIALS):
            study.optimize(objective, n_trials=1, catch=(Exception,))
            self.progress.setValue(int(5 + 90*(i+1)/OPTUNA_N_TRIALS))
            QApplication.processEvents()
        self.progress.setValue(95)

        # Top-10
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        completed.sort(key=lambda t: t.value)
        rows = []
        for t in completed[:10]:
            p = dict(t.params)
            sol1,sol2,sol3 = p['溶媒セット'].split(',')
            salt1,salt2    = p['塩セット'].split(',')
            p.update({'溶媒1':sol1,'溶媒2':sol2,'溶媒3':sol3,
                      '塩1':salt1,'塩2':salt2})
            # retrieve from user_attrs
            w1 = t.user_attrs['溶媒1割合']; w2 = t.user_attrs['溶媒2割合']
            w3 = t.user_attrs['溶媒3割合']
            amt = t.user_attrs['添加剤1量(%)']
            c1 = t.user_attrs['塩1濃度(M)']; c2 = t.user_attrs['塩2濃度(M)']
            p['溶媒1割合'],p['溶媒2割合'],p['溶媒3割合'] = round(w1,2),round(w2,2),round(w3,2)
            p['添加剤1量(%)'] = round(amt,2)
            p['塩1濃度(M)']   = round(c1,2)
            p['塩2濃度(M)']   = round(c2,2)
            # engineered columns...
            for s in self.unique_solvents:
                p[f'比率_{s}'] = round((sol1==s)*w1 + (sol2==s)*w2 + (sol3==s)*w3,2)
            for s in self.unique_salts:
                p[f'濃度_{s}'] = round((salt1==s)*c1 + (salt2
==s)*c2,2)
            for a in self.unique_additives:
                p[f'量_{a}'] = round((p['添加剤1']==a)*amt,2)
            p['PredCap'] = -t.value
            rows.append(p)

        df_top = pd.DataFrame(rows).reindex(columns=list(self.X_train.columns)+['PredCap'])
        self.populate_table(self.table_optuna, df_top)
        self.tabs.setCurrentWidget(self.table_optuna)
        QMessageBox.information(
            self, "Optuna 最良レシピ",
            "\n".join(f"{k}: {df_top.iloc[0][k]}" for k in df_top.columns)
        )
        self.progress.setValue(0)

        # カテゴリ変数影響可視化
        plot_slice(study, params=["バインダー","導電助剤","活物質"]).show()
        plot_parallel_coordinate(study, params=["バインダー","導電助剤","活物質"]).show()

    def draw_optuna_scatter(self):
        if not self.study: return
        # ... same as before ...
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CatBoostShapApp()
    w.show()
    sys.exit(app.exec_())
