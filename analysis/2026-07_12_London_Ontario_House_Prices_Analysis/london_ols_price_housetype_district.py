# ── 套件匯入 ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

# ── 參數設定 ──────────────────────────────────────────────────────────────────
DATA_PATH = r'C:\Users\flyin\OneDrive\桌面\新代碼\Linear_OLS_Regression\data\london_zolo_house_prices.csv'

# ── 讀取資料 ──────────────────────────────────────────────────────────────────
data = pd.read_csv(DATA_PATH)

# log_price：房價分佈右偏（少數豪宅價格極高），取log後較接近常態分佈
# 這點和股價一樣，log轉換是為了讓變數更適合線性模型
data['log_price'] = np.log(data['price'])

# ── 剔除極值：以log_price的z-score判斷 ────────────────────────────────────────
# z-score = (數值 - 平均數) / 標準差，代表這筆資料距離平均數幾個標準差。
# 這裡採用業界常見的門檻：|z-score| > 3 視為極端值並剔除，
# 因為在常態分佈假設下，落在3個標準差之外的資料理論上只佔不到0.3%，
# 屬於相對罕見的極端案例。
log_price_mean = data['log_price'].mean()
log_price_std = data['log_price'].std()
z_score = (data['log_price'] - log_price_mean) / log_price_std

before_count = len(data)
data = data.loc[z_score.abs() <= 3].reset_index(drop=True)
after_count = len(data)
print(f'剔除極值：原本{before_count}筆，剔除{before_count - after_count}筆，剩餘{after_count}筆')

# ------ Define the List of House Price Factors -------------------------------
# 原本命名為char：char在程式語言慣例中通常代表「單一字元」，用來裝一整組
# 候選特徵名稱的list容易讓人誤解，這裡改名為candidate_factors。
# 另外原本的list裡'property_type'重複出現兩次，這裡也一併移除重複項。
candidate_factors = [
    'district',
    'property_type',
    'bedrooms',
    'house_age_years',
    'distance_to_western_km']

# ── 缺失值檢查：candidate_factors裡house_age_years有缺失值 ───────────────────
# 這是把所有candidate_factors都放進迴歸模型時「必要」的前置處理：
# sklearn的LinearRegression和statsmodels的OLS都不接受NaN，只要
# candidate_factors裡任何一個欄位有缺失，這筆資料就無法用來配適模型，
# 這裡直接刪除這些列，並印出刪除筆數方便追蹤資料量的變化。
missing_mask = data[candidate_factors].isnull().any(axis=1)
before_dropna_count = len(data)
data = data.loc[~missing_mask].reset_index(drop=True)
after_dropna_count = len(data)
print(f'因candidate_factors存在缺失值，剔除{before_dropna_count - after_dropna_count}筆，'
      f'剩餘{after_dropna_count}筆')

# ── 基本統計：分組敘述統計 ────────────────────────────────────────────────────
def basic_statistical_analysis(data: pd.DataFrame, factor: str) -> pd.DataFrame:

    print(f'=== Statistical Price Analysis on {factor} ===')
    result = data.groupby(factor)['price'].agg(['count', 'mean', 'median', 'std']).round(0)
    print(result)
    
    return result

# ── 畫圖：單一因素 vs 房價散點圖 ──────────────────────────────────────────────
# 這裡把參數命名為factor而不是char：跟basic_statistical_analysis的參數命名
# 保持一致，也避免用char這個容易讓人聯想成「單一字元」的名稱。
# 用散點圖（而非箱型圖）是因為candidate_factors裡同時有類別變數（district、
# property_type）跟連續變數（bedrooms、house_age_years、
# distance_to_western_km），散點圖對這兩種型態的x軸都適用，
# 不需要為了因素型態的不同另外寫兩種畫圖邏輯。
def plot_factor_vs_price(data: pd.DataFrame, factor: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(data[factor], data['price'], alpha=0.4, s=15, color='steelblue')
    ax.set_title(f'{factor} vs 房價')
    ax.set_xlabel(factor)
    ax.set_ylabel('價格')
    plt.tight_layout()
    plt.savefig(f'london_price_vs_{factor}.png', dpi=150)
    plt.show()

# ── One Hot Encoding：將類別變數轉為數值虛擬變數 ──────────────────────────────
# drop_first=True：避免虛擬變數陷阱（dummy variable trap），
# 被丟掉的類別會變成基準組，其餘係數代表相對基準組的差異
# 原本命名為Coded_Char：大小寫混用（PascalCase）跟其餘變數（data、model、
# pred這類全小寫snake_case）風格不一致，這裡改名為encoded_features，
# 名稱上也更直接對應「編碼後的特徵矩陣」這個實際內容。
#
# candidate_factors裡district、property_type是類別變數，需要one hot
# encoding；bedrooms、house_age_years、distance_to_western_km本身已經是
# 數值，直接併入即可，不需要額外編碼。
categorical_factors = ['district', 'property_type']
numeric_factors = [f for f in candidate_factors if f not in categorical_factors]

encoded_categorical = pd.get_dummies(data[categorical_factors], drop_first=True, dtype=int)
encoded_features = pd.concat([encoded_categorical, data[numeric_factors]], axis=1)
y = data['log_price']

print('\n=== 編碼後特徵欄位（基準組已省略）===')
print(encoded_features.columns.tolist())

# ── OLS回歸（sklearn版本，用於預測與R平方）────────────────────────────────────
model = LinearRegression()
model.fit(encoded_features, y)
pred = model.predict(encoded_features)

print('\n=== 模型績效 ===')
print(f'R平方：{r2_score(y, pred):.4f}')
print(f'截距（基準組log價格）：{model.intercept_:.4f}')
print('各特徵係數（log價格差異）：')
for name, coef in zip(encoded_features.columns, model.coef_):
    print(f'  {name}：{coef:+.4f}（換算倍數：{np.exp(coef):.2f}倍）')

# ── OLS回歸（statsmodels版本，用於統計顯著性檢定）─────────────────────────────
# t檢定與p值可判斷該類別對房價的影響是否統計顯著（一般以p<0.05為門檻）
encoded_features_sm = sm.add_constant(encoded_features.astype(float))
sm_model = sm.OLS(y, encoded_features_sm).fit()
print('\n=== 統計顯著性檢定（statsmodels OLS摘要）===')
print(sm_model.summary())

# ── 圖3：實際值 vs 預測值散點圖 ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y, pred, alpha=0.4, s=15, color='steelblue')
lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
ax.plot(lims, lims, color='red', linestyle='--', linewidth=1)
ax.set_title('實際log價格 vs 預測log價格')
ax.set_xlabel('實際log價格')
ax.set_ylabel('預測log價格')
plt.tight_layout()
plt.savefig('london_ols_actual_vs_pred.png', dpi=150)
plt.show()

# ── 圖4：標準化殘差分析（每個殘差除以殘差標準差，換算成標準化殘差）────────────
# 標準化殘差 = 殘差 / 殘差的標準差，換算後可以直接用「幾個標準差」來判斷
# 離群程度，方便和常態分佈的68%、95%經驗法則對照，而不需要另外查殘差的
# 原始標準差是多少。
residuals = y - pred
standardized_residuals = residuals / residuals.std()

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(pred, standardized_residuals, alpha=0.4, s=15, color='darkorange')
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.set_title('標準化殘差圖（預測值 vs 標準化殘差）')
ax.set_xlabel('預測log價格')
ax.set_ylabel('標準化殘差')
plt.tight_layout()
plt.savefig('london_ols_residuals.png', dpi=150)
plt.show()

basic_statistical_analysis(data, 'distance_to_western_km')
plot_factor_vs_price(data, 'distance_to_western_km')
