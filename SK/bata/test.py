import pandas as pd

# 元ファイルパス
IN_PATH  = '/mnt/data/20250416FECDMED2電解液_特許2_北山編集.xlsx'
OUT_PATH = '/mnt/data/transformed.xlsx'

# 1) ヘッダ２行を MultiIndex で読み込み
df = pd.read_excel(IN_PATH, sheet_name="LR23特許", header=[1, 2])

# 2) MultiIndex を結合してフラットな列名に
df.columns = [
    f"{str(a).strip()}_{str(b).strip()}" if str(a).strip() and str(b).strip() else str(a).strip() or str(b).strip()
    for a, b in df.columns
]

# 3) コードの FEATURE_ORDER に合わせてリネーム
rename_map = {
    # 例: Excel 側の "2サイクル目の放電容量_ｍAh/g" を "2nd.放電容量" に
    "2サイクル目の放電容量_ｍAh/g": "2nd.放電容量",
    # 以下は実際の列名に合わせて必要なだけ追加してください
    "炭素種_Unnamed: 1_level_1": "導電助剤",
    "硫黄重量_mg":                "活物質",
    "硫黄重量_mg/cm2":            "ロード量",
    "塩　_塩　":                 "塩1",
    # （塩1濃度がない場合は後でゼロ埋め）
    "溶媒種_溶媒１":             "溶媒1",
    "割合 vol.%_溶媒１":        "溶媒1割合",
    "溶媒種_溶媒２":             "溶媒2",
    "割合 vol.%_溶媒２":        "溶媒2割合",
    "溶媒種_溶媒３":             "溶媒3",
    "割合 vol.%_溶媒３":        "溶媒3割合",
    # もし添加剤やバインダーの列があればここで対応
    # "添加剤列名": "添加剤1",
    # "添加剤割合列名": "添加剤1量(%)",
}

df = df.rename(columns=rename_map)

# 4) CODE 側で期待しているFEATURE_ORDER
FEATURE_ORDER = [
    '活物質','導電助剤','バインダー','ロード量',
    '塩1','塩1濃度(M)','塩2','塩2濃度(M)',
    '溶媒1','溶媒1割合','溶媒2','溶媒2割合',
    '溶媒3','溶媒3割合','添加剤1','添加剤1量(%)'
]

# 5) 無い列は 0 で補完
for col in FEATURE_ORDER + ['2nd.放電容量']:
    if col not in df.columns:
        df[col] = 0

# 6) 最終的に順序を揃えて出力
out_df = df[FEATURE_ORDER + ['2nd.放電容量']]
out_df.to_excel(OUT_PATH, sheet_name="Raw_Data", index=False)

print(f"変換完了: {OUT_PATH}")
