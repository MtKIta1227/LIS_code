# -*- coding: utf-8 -*-
# beiz_ver1.0_binder_shap_full.py
# --------------------------------------------------------------
# CatBoost Ensemble + SHAP + HeatMap + Optuna GUI  (2025-04-27 修正版)
# - SHAP全体表示 + バインダー別SHAP表示機能を追加
# --------------------------------------------------------------

import sys, platform, itertools, logging
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
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

logging.basicConfig(level=logging.INFO)

# 日本語フォント設定
def set_jp_font():
    p = platform.system()
    if p == "Darwin": fam = "Hiragino Maru Gothic Pro"
    elif p == "Windows":
        for c in ["Yu Gothic","Meiryo","MS Gothic"]:
            if c in [f.name for f in fm.fontManager.ttflist]: fam = c; break
        else: fam = "MS Gothic"
    else: fam = "DejaVu Sans"
    plt.rcParams["font.family"] = fam
    plt.rcParams["axes.unicode_minus"] = False
set_jp_font()

# 定数
SHAP_FULL_SAMPLE = True
OPTUNA_N_TRIALS   = 2000
ENSEMBLE_SEEDS    = [42, 100, 2025]
CATBOOST_ITERS     = 400

class CatBoostShapApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CatBoost Ensemble + SHAP GUI (fixed)")
        self.resize(1100, 800)

        # --- UI components ---
        self.label    = QLabel("Excel ファイルを選択してください")
        self.btn_load = QPushButton("ファイルを選択…")
        self.btn_train= QPushButton("学習＆SHAP")

        # SHAP全体 & バインダー別
        self.binder_cb            = QComboBox()
        self.btn_shap_by_binder   = QPushButton("Binder別SHAP")
        self.btn_shap_by_binder.setEnabled(False)

        # HeatMap controls
        self.heatmap_x_cb = QComboBox()
        self.heatmap_y_cb = QComboBox()
        self.btn_heatmap  = QPushButton("HeatMap 再描画")
        self.btn_allhm    = QPushButton("全ペアHeatMap")
        self.btn_heatmap.setEnabled(False)
        self.btn_allhm.setEnabled(False)

        # Optuna controls
        self.btn_optuna  = QPushButton("ベイズ最適化")
        self.btn_scatter = QPushButton("Optuna 散布図")
        self.btn_optuna.setEnabled(False)
        self.btn_scatter.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setFormat("%p%")

        # Tabs
        self.tabs = QTabWidget()

        # --- SHAP まとめタブ (ツールバー付き) ---
        self.fig_shap    = plt.figure(figsize=(6,4))
        self.canvas_shap = FigureCanvas(self.fig_shap)
        self.toolbar_shap = NavigationToolbar(self.canvas_shap, self)
        
        # ツールバー＋キャンバスをまとめる
        shap_tab    = QWidget()
        shap_layout = QVBoxLayout()
        shap_layout.addWidget(self.toolbar_shap)
        shap_layout.addWidget(self.canvas_shap)
        shap_tab.setLayout(shap_layout)
        self.tabs.addTab(shap_tab, "SHAP Summary")
        self.fig_hm, _ = plt.subplots(figsize=(6,4))
        self.canvas_hm = FigureCanvas(self.fig_hm)
        self.tabs.addTab(self.canvas_hm, "HeatMap")
        self.table_optuna = QTableWidget()
        self.tabs.addTab(self.table_optuna, "Optuna 結果")

        # Layout
        top = QHBoxLayout()
        for w in [self.btn_load, self.btn_train, QLabel("バインダー:"), self.binder_cb,
                  self.btn_shap_by_binder, QLabel("X軸:"), self.heatmap_x_cb, QLabel("Y軸:"),
                  self.heatmap_y_cb, self.btn_heatmap, self.btn_allhm,
                  self.btn_optuna, self.btn_scatter]:
            top.addWidget(w)
        main = QVBoxLayout(self)
        main.addWidget(self.label)
        main.addLayout(top)
        main.addWidget(self.progress)
        main.addWidget(self.tabs)

        # Signals
        self.btn_load.clicked.connect(self.load_file)
        self.btn_train.clicked.connect(self.train_and_shap)
        self.btn_shap_by_binder.clicked.connect(self.draw_shap_by_binder)
        self.btn_heatmap.clicked.connect(self.draw_heatmap)
        self.btn_allhm.clicked.connect(self.draw_all_heatmaps)
        self.btn_optuna.clicked.connect(self.run_optuna)
        self.btn_scatter.clicked.connect(self.draw_optuna_scatter)

        # Data holders
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
        self.sample_X         = None
        self.shap_vals        = None

    def _safe_process_events(self):
        QApplication.processEvents()

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
        return df.fillna(0)

    def train_and_shap(self):
        try:
            self.progress.setValue(5)
            df = pd.read_excel(self.file_path, sheet_name="Raw_Data")
            df = self.preprocess(df)
            # カテゴリ候補生成
            self.unique_solvents = sorted({v for c in ('溶媒1','溶媒2','溶媒3') for v in df[c].unique() if str(v)!='0'})
            if len(self.unique_solvents)>=3:
                self.solvent_sets = [','.join(c) for c in itertools.combinations(self.unique_solvents,3)]
            else:
                fill = self.unique_solvents + ['0']*(3-len(self.unique_solvents))
                self.solvent_sets = sorted({','.join(sorted(c)) for c in itertools.combinations_with_replacement(fill,3)})
            self.unique_salts = sorted({v for c in ('塩1','塩2') for v in df[c].unique() if str(v)!='0'})
            if len(self.unique_salts)>=2:
                self.salt_sets = [f"{s},0" for s in self.unique_salts] + [','.join(c) for c in itertools.combinations(self.unique_salts,2)]
            else:
                s = self.unique_salts[0] if self.unique_salts else '0'
                self.salt_sets = [f"{s},0"]
            self.unique_additives = sorted(v for v in df['添加剤1'].unique() if v not in (0,0.0,'0'))
            # 特徴量エンジニアリング
            for s in self.unique_solvents:
                df[f'比率_{s}'] = ((df['溶媒1']==s)*df['溶媒1割合'] + (df['溶媒2']==s)*df['溶媒2割合'] + (df['溶媒3']==s)*df['溶媒3割合'])
            for s in self.unique_salts:
                df[f'濃度_{s}'] = ((df['塩1']==s)*df['塩1濃度(M)'] + (df['塩2']==s)*df['塩2濃度(M)'])
            for a in self.unique_additives:
                df[f'量_{a}'] = (df['添加剤1']==a)*df['添加剤1量(%)']
            drop_cols = ['溶媒1','溶媒2','溶媒3','溶媒1割合','溶媒2割合','溶媒3割合',
                         '塩1','塩2','塩1濃度(M)','塩2濃度(M)','添加剤1','添加剤1量(%)']
            df = df.drop(columns=drop_cols)
            self.df = df
            # 学習データ準備
            y = df['2nd.放電容量'].values
            X = df.drop(columns=['2nd.放電容量'])
            self.cat_cols = X.select_dtypes('object').columns.tolist()
            self.cat_idx  = [X.columns.get_loc(c) for c in self.cat_cols]
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            self.X_train = X_tr
            pool_tr = Pool(X_tr, y_tr, cat_features=self.cat_idx)
            pool_te = Pool(X_te, y_te, cat_features=self.cat_idx)
            # アンサンブル学習
            self.progress.setValue(15)
            self.models = []
            for seed in ENSEMBLE_SEEDS:
                m = CatBoostRegressor(iterations=CATBOOST_ITERS, depth=6, learning_rate=0.05, loss_function='RMSE', random_state=seed, verbose=0)
                m.fit(pool_tr)
                self.models.append(m)
                self._safe_process_events()
            preds = np.mean([m.predict(pool_te) for m in self.models], axis=0)
            mae, r2 = mean_absolute_error(y_te, preds), r2_score(y_te, preds)
            QMessageBox.information(self, "モデル評価", f"MAE = {mae:.2f}    R² = {r2:.2f}")
            # SHAP値算出
            self.progress.setValue(30)
            sample_X = self.X_train if SHAP_FULL_SAMPLE or len(self.X_train)<=1000 else self.X_train.sample(1000, random_state=42)
            self.sample_X = sample_X
            self.shap_vals = np.mean([shap.TreeExplainer(m).shap_values(sample_X, check_additivity=False) for m in self.models], axis=0)
            self.progress.setValue(60)
            # SHAP全体表示
            self._render_shap(self.sample_X, self.shap_vals)
            # バインダー候補
            self.binder_cb.clear()
            for b in sorted(self.sample_X['バインダー'].unique(), key=lambda x: str(x)):
                self.binder_cb.addItem(str(b))

            self.btn_shap_by_binder.setEnabled(True)
            # HeatMap準備
            self.heatmap_x_cb.clear(); self.heatmap_y_cb.clear()
            for c in self.cat_cols: self.heatmap_x_cb.addItem(c); self.heatmap_y_cb.addItem(c)
            self.btn_heatmap.setEnabled(True); self.btn_allhm.setEnabled(True)
            self.btn_optuna.setEnabled(True); self.btn_scatter.setEnabled(True)
            self.progress.setValue(0)
        except Exception as e:
            QMessageBox.critical(self, "エラー", str(e)); logging.exception(e); self.progress.setValue(0)

    def _render_shap(self, X, shap_vals, title="SHAP Summary"):
        self.fig_shap.clf()
        ax = self.fig_shap.add_subplot(111)
        num_cols = X.select_dtypes(include=[np.number]).columns
        norm = plt.Normalize(X[num_cols].min().min(), X[num_cols].max().max())
        cmap = plt.get_cmap('viridis')
        y_positions = []
        y_labels = []
        base_idx = 0
        for i, feat in enumerate(X.columns):
            if feat == 'バインダー':
                raw_cats = X['バインダー'].unique()
                cats = sorted(raw_cats, key=lambda x: str(x))
                for cat in cats:
                    mask = (X['バインダー'] == cat)
                    vals = shap_vals[mask, i]
                    ax.scatter(vals, np.full_like(vals, base_idx), c='gray', s=10, alpha=0.6)
                    y_positions.append(base_idx); y_labels.append(f"バインダー={cat}")
                    base_idx += 1
            else:
                vals = shap_vals[:, i]
                if feat in num_cols:
                    fvals = X[feat].astype(float).values
                    ax.scatter(vals, np.full_like(vals, base_idx), c=fvals, cmap=cmap, norm=norm, s=10, alpha=0.6)
                else:
                    ax.scatter(vals, np.full_like(vals, base_idx), c='gray', s=10, alpha=0.6)
                y_positions.append(base_idx); y_labels.append(feat)
                base_idx += 1
        ax.axvline(0, color='gray')
        ax.set_yticks(y_positions); ax.set_yticklabels(y_labels)
        ax.set_xlabel("SHAP value"); ax.set_title(title)
        self.fig_shap.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Feature value')
        self.canvas_shap.draw()

    def draw_shap_by_binder(self):
        binder = self.binder_cb.currentText()
        mask = (self.sample_X['バインダー'] == binder)
        Xb = self.sample_X[mask]; sb = self.shap_vals[mask, :]
        self._render_shap(Xb, sb, title=f"SHAP for Binder: {binder}")

    def predict_ensemble(self, df_feat):
        pool = Pool(df_feat, cat_features=self.cat_idx)
        return np.mean([m.predict(pool) for m in self.models], axis=0)

    def draw_heatmap(self):
        xcol = self.heatmap_x_cb.currentText(); ycol = self.heatmap_y_cb.currentText()
        df = self.df.copy(); df['Pred'] = self.predict_ensemble(df.drop(columns=['2nd.放電容量']))
        pivot = df.pivot_table(values='Pred', index=ycol, columns=xcol, aggfunc=np.mean)
        self.fig_hm.clf(); ax = self.fig_hm.add_subplot(111)
        im = ax.imshow(pivot.values, aspect='auto', origin='lower')
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=90)
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
        ax.set_xlabel(xcol); ax.set_ylabel(ycol)
        self.fig_hm.colorbar(im, ax=ax, label='Predicted Capacity'); self.canvas_hm.draw()

    def draw_all_heatmaps(self):
        for i, xcol in enumerate(self.cat_cols):
            for ycol in self.cat_cols[i+1:]:
                df = self.df.copy(); df['Pred'] = self.predict_ensemble(df.drop(columns=['2nd.放電容量']))
                pivot = df.pivot_table(values='Pred', index=ycol, columns=xcol, aggfunc=np.mean)
                fig, ax = plt.subplots()
                im = ax.imshow(pivot.values, aspect='auto', origin='lower')
                ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, rotation=90)
                ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
                ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(f'{ycol} vs {xcol}')
                fig.colorbar(im, ax=ax, label='Predicted Capacity'); fig.show()

    def populate_table(self, widget, df):
        widget.clear()
        widget.setRowCount(len(df))
        widget.setColumnCount(len(df.columns))
        widget.setHorizontalHeaderLabels(df.columns)
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                widget.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))
        widget.resizeColumnsToContents()

    def run_optuna(self):
        if not self.models: return
        self.progress.setValue(5)
        
        def nz(col): return [v for v in self.df[col].unique() if str(v)!="0"]
        choices={
            '活物質':    nz('活物質'),
            '導電助剤':  nz('導電助剤'),
            'バインダー':nz('バインダー'),
            '添加剤1':  ['0']+self.unique_additives,
            '溶媒セット':self.solvent_sets,
            '塩セット':  self.salt_sets
        }
        ranges={
            'ロード量':      (int(self.df['ロード量'].min()),int(self.df['ロード量'].max())),
            '塩1濃度(M)':   (0.1,2.0),
            '塩2濃度(M)':   (0.0,2.0),
            '添加剤1量(%)': (0,10)
        }
        all_cols=list(self.X_train.columns)
        cat_set=set(self.cat_cols)

        study=optuna.create_study(direction="minimize",
                                  sampler=optuna.samplers.TPESampler(seed=42,multivariate=True))
        self.study=study

        def objective(trial):
            rec={k:trial.suggest_categorical(k,v)
                 for k,v in choices.items() if k!='添加剤1量(%)'}
            if rec['添加剤1']=='0': rec['添加剤1量(%)']=0
            else: rec['添加剤1量(%)']=trial.suggest_int('添加剤1量(%)',*ranges['添加剤1量(%)'])
            trial.set_user_attr('添加剤1量(%)',float(rec['添加剤1量(%)']))
            rec['ロード量']=trial.suggest_int('ロード量',*ranges['ロード量'])
            s1,s2=rec['塩セット'].split(','); rec['塩1'],rec['塩2']=s1,s2
            rec['塩1濃度(M)']=trial.suggest_float('塩1濃度(M)',*ranges['塩1濃度(M)'],step=0.05)
            rec['塩2濃度(M)']=0 if s2=='0' else trial.suggest_float('塩2濃度(M)',*ranges['塩2濃度(M)'],step=0.05)
            trial.set_user_attr('塩1濃度(M)',float(rec['塩1濃度(M)'])); trial.set_user_attr('塩2濃度(M)',float(rec['塩2濃度(M)']))
            sol1,sol2,sol3=rec['溶媒セット'].split(',')
            w1=trial.suggest_float('溶媒1割合',0,100); w2=trial.suggest_float('溶媒2割合',0,100-w1); w3=100-w1-w2
            rec.update({'溶媒1':sol1,'溶媒2':sol2,'溶媒3':sol3,'溶媒1割合':w1,'溶媒2割合':w2,'溶媒3割合':w3})
            trial.set_user_attr('溶媒1割合',float(w1)); trial.set_user_attr('溶媒2割合',float(w2)); trial.set_user_attr('溶媒3割合',float(w3))
            row={c:("0" if c in cat_set else 0) for c in all_cols}
            row.update(rec)
            for s in self.unique_solvents: row[f'比率_{s}']=(sol1==s)*w1+(sol2==s)*w2+(sol3==s)*w3
            for s in self.unique_salts: row[f'濃度_{s}']=(s1==s)*rec['塩1濃度(M)']+(s2==s)*rec['塩2濃度(M)']
            for a in self.unique_additives: row[f'量_{a}']=(rec['添加剤1']==a)*rec['添加剤1量(%)']
            df_one=pd.DataFrame([row])[all_cols]
            cap=np.mean([m.predict(Pool(df_one,cat_features=self.cat_idx))[0] for m in self.models])
            return -cap

        # Optimize
        for i in range(OPTUNA_N_TRIALS):
            study.optimize(objective,n_trials=1,catch=(Exception,))
            self.progress.setValue(int(5+90*(i+1)/OPTUNA_N_TRIALS))
            self._safe_process_events()
        self.progress.setValue(95)

        # ---- Top-20 PredCap 降順 ----
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        # t.value は -PredCap なので、小さい順にソート → PredCap大きい順になる
        completed.sort(key=lambda t: t.value)

        rows = []
        for t in completed[:20]:
            p = dict(t.params)
            # 元のコードと同様にソルベント／ソルト欄を展開
            sol1, sol2, sol3 = p['溶媒セット'].split(',')
            salt1, salt2    = p['塩セット'].split(',')
            p.update({'溶媒1':sol1, '溶媒2':sol2, '溶媒3':sol3,
                      '塩1':salt1, '塩2':salt2})
            # user_attrs から取り出した割合・濃度を丸め
            w1 = t.user_attrs['溶媒1割合']; w2 = t.user_attrs['溶媒2割合']; w3 = t.user_attrs['溶媒3割合']
            amt = t.user_attrs['添加剤1量(%)']
            c1 = t.user_attrs['塩1濃度(M)']; c2 = t.user_attrs.get('塩2濃度(M)', 0.0)
            p['溶媒1割合'],p['溶媒2割合'],p['溶媒3割合'] = round(w1,2),round(w2,2),round(w3,2)
            p['添加剤1量(%)'] = round(amt,2)
            p['塩1濃度(M)'], p['塩2濃度(M)'] = round(c1,2), round(c2,2)
            # engineered 列も丸めて挿入
            for s in self.unique_solvents:
                p[f'比率_{s}'] = round((sol1==s)*w1 + (sol2==s)*w2 + (sol3==s)*w3, 2)
            for s in self.unique_salts:
                p[f'濃度_{s}'] = round((salt1==s)*c1 + (salt2==s)*c2, 2)
            for a in self.unique_additives:
                p[f'量_{a}']  = round((p['添加剤1']==a)*amt, 2)

            # PredCap を追加
            p['PredCap'] = -t.value
            rows.append(p)

        # DataFrame 化してテーブルにセット
        df_top = pd.DataFrame(rows).reindex(columns=list(self.X_train.columns)+['PredCap'])
        self.populate_table(self.table_optuna, df_top)
        self.tabs.setCurrentWidget(self.table_optuna)
        QMessageBox.information(self,"Optuna 最良レシピ",
                                "\n".join(f"{k}: {df_top.iloc[0][k]}" for k in df_top.columns))
        self.progress.setValue(0)

        try:
            plot_slice(study,params=["バインダー","導電助剤","活物質"]).show()
            plot_parallel_coordinate(study,params=["バインダー","導電助剤","活物質"]).show()
        except Exception as e:
            logging.warning("Optuna plot skipped: %s",e)

    def draw_optuna_scatter(self):
        if not self.study: return
        trials=[t for t in self.study.trials if t.state==optuna.trial.TrialState.COMPLETE]
        xs=list(range(1,len(trials)+1)); ys=[-t.value for t in trials]
        fig,ax=plt.subplots()
        scatter=ax.scatter(xs,ys,c=xs,cmap='viridis',s=20,edgecolor='k')
        ax.set_xlabel("Trial"); ax.set_ylabel("Predicted Capacity")
        ax.set_title("Optuna Trials vs Predicted Capacity")
        cbar=fig.colorbar(scatter,ax=ax); cbar.set_label("Trial number")
        fig.show()

if __name__=="__main__":
    app=QApplication(sys.argv)
    w=CatBoostShapApp()
    w.show()
    sys.exit(app.exec_())
