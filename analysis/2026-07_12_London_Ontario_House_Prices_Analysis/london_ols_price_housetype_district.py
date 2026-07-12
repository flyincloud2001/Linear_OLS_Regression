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

# ── 基本統計：分組敘述統計 ────────────────────────────────────────────────────
print('=== 各區房價敘述統計 ===')
print(data.groupby('district')['price'].agg(['count', 'mean', 'median', 'std']).round(0))

print('\n=== 各房型房價敘述統計 ===')
print(data.groupby('property_type')['price'].agg(['count', 'mean', 'median', 'std']).round(0))

# ── 圖1：各區房價箱型圖 ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
data.boxplot(column='price', by='district', ax=ax)
ax.set_title('各區房價分佈')
ax.set_xlabel('區域')
ax.set_ylabel('價格')
plt.suptitle('')
plt.tight_layout()
plt.savefig('london_price_boxplot_district.png', dpi=150)
plt.show()

# ── 圖2：各房型房價箱型圖 ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
data.boxplot(column='price', by='property_type', ax=ax)
ax.set_title('各房型房價分佈')
ax.set_xlabel('房型')
ax.set_ylabel('價格')
plt.suptitle('')
plt.tight_layout()
plt.savefig('london_price_boxplot_type.png', dpi=150)
plt.show()

# ── One Hot Encoding：將類別變數轉為數值虛擬變數 ──────────────────────────────
# drop_first=True：避免虛擬變數陷阱（dummy variable trap），
# 被丟掉的類別會變成基準組，其餘係數代表相對基準組的差異
X = pd.get_dummies(data[['district', 'property_type']], drop_first=True, dtype=int)
y = data['log_price']

print('\n=== 編碼後特徵欄位（基準組已省略）===')
print(X.columns.tolist())

# ── OLS回歸（sklearn版本，用於預測與R平方）────────────────────────────────────
model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)

print('\n=== 模型績效 ===')
print(f'R平方：{r2_score(y, pred):.4f}')
print(f'截距（基準組log價格）：{model.intercept_:.4f}')
print('各特徵係數（log價格差異）：')
for name, coef in zip(X.columns, model.coef_):
    print(f'  {name}：{coef:+.4f}（換算倍數：{np.exp(coef):.2f}倍）')

# ── OLS回歸（statsmodels版本，用於統計顯著性檢定）─────────────────────────────
# t檢定與p值可判斷該類別對房價的影響是否統計顯著（一般以p<0.05為門檻）
X_sm = sm.add_constant(X.astype(float))
sm_model = sm.OLS(y, X_sm).fit()
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

# ── 圖4：殘差分析（檢查是否有明顯規律，判斷模型假設是否成立）────────────────────
residuals = y - pred
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(pred, residuals, alpha=0.4, s=15, color='darkorange')
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.set_title('殘差圖（預測值 vs 殘差）')
ax.set_xlabel('預測log價格')
ax.set_ylabel('殘差')
plt.tight_layout()
plt.savefig('london_ols_residuals.png', dpi=150)
plt.show()