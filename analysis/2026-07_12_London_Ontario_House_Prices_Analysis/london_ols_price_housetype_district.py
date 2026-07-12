import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 若沒有這個字體請自行更換成系統中支援中文的字體

# 資料檔案路徑
FILE_PATH = r'C:\Users\flyin\OneDrive\桌面\新代碼\Linear_OLS_Regression\data\london_zolo_house_prices.csv'

# 應變數欄位名稱
TARGET_COL = 'price'

# 類別型影響因素設定：欄位名稱對應到基準類別
# 基準類別不會產生虛擬變數，效果會反映在截距 const 上
# 未來要新增類別型因素（例如學區、屋齡分級等），只需要在這裡加一組 欄位名稱: 基準類別
CATEGORICAL_FEATURES = {
    'property_type': 'House',
    'district': 'London North',
}

# 數值型影響因素清單，目前沒有數值型變數
# 未來爬蟲取得的連續型資料（例如坪數、屋齡年數、距離市中心距離等）直接加進這個清單即可
NUMERIC_FEATURES = []


# 讀取並清理資料
def load_data(path, categorical_features, numeric_features, target_col):
    raw = pd.read_csv(path)
    columns_needed = [target_col] + list(categorical_features.keys()) + numeric_features
    data = raw[columns_needed].copy()
    data.dropna(inplace=True)
    return data


# 依照 CATEGORICAL_FEATURES 與 NUMERIC_FEATURES 自動組出設計矩陣
def build_design_matrix(data, categorical_features, numeric_features):
    dummy_frames = []
    for col, base_category in categorical_features.items():
        dummies = pd.get_dummies(data[col], prefix=col).astype(int)
        base_col = f'{col}_{base_category}'
        if base_col in dummies.columns:
            dummies = dummies.drop(columns=[base_col])
        dummy_frames.append(dummies)
    X = pd.concat(dummy_frames, axis=1)
    if numeric_features:
        X = pd.concat([X, data[numeric_features]], axis=1)
    return X


# 建立並訓練 OLS 模型
def fit_ols(data, target_col, categorical_features, numeric_features):
    X = build_design_matrix(data, categorical_features, numeric_features)
    X = sm.add_constant(X)
    y = data[target_col]
    model = sm.OLS(y, X).fit()
    return model, X, y


# 印出精簡統計表：coef、std err、p value、R squared
def print_summary_table(model):
    table = pd.DataFrame({
        'coef': model.params,
        'std err': model.bse,
        'p value': model.pvalues,
    })
    print(table.round(3))
    print(f'R squared = {model.rsquared:.4f}')
    print(f'Adjusted R squared = {model.rsquared_adj:.4f}')


# 視覺化分析：係數與標準誤、實際vs預測、殘差圖
def plot_results(model, y):
    fitted = model.fittedvalues
    resid = model.resid

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 係數與標準誤：長條代表 coef，誤差線代表 std err
    coef = model.params.drop('const')
    err = model.bse.drop('const')
    axes[0].barh(coef.index, coef.values, xerr=err.values, color='steelblue')
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_title('係數與標準誤')
    axes[0].set_xlabel('對 price 的影響（元）')

    # 實際值 vs 預測值：越貼近對角線代表模型解釋力越好
    axes[1].scatter(y, fitted, alpha=0.5, color='darkorange')
    lims = [min(y.min(), fitted.min()), max(y.max(), fitted.max())]
    axes[1].plot(lims, lims, color='black', linewidth=0.8)
    axes[1].set_title(f'實際值 vs 預測值 (R平方 = {model.rsquared:.3f})')
    axes[1].set_xlabel('實際 price')
    axes[1].set_ylabel('預測 price')

    # 殘差圖：檢查殘差是否隨機散布，若有明顯型態代表模型可能遺漏重要變數
    axes[2].scatter(fitted, resid, alpha=0.5, color='seagreen')
    axes[2].axhline(0, color='black', linewidth=0.8)
    axes[2].set_title('殘差圖')
    axes[2].set_xlabel('預測 price')
    axes[2].set_ylabel('殘差')

    plt.tight_layout()
    plt.savefig('House_Prices_Analysis.png', dpi=150)
    plt.show()


# 主流程
data = load_data(FILE_PATH, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL)
model, X, y = fit_ols(data, TARGET_COL, CATEGORICAL_FEATURES, NUMERIC_FEATURES)

print_summary_table(model)
plot_results(model, y)