# ── Import Modules ──────────────────────────────────────────────────────────────────
import pandas as pd
import statsmodels.api as sm

# ── Parameters Setup ──────────────────────────────────────────────────────────────────
FILE_PATH = r'C:\Users\flyin\OneDrive\桌面\新代碼\Linear_OLS_Regression\data\london_zolo_house_prices.csv'


# ── Read Data and Preprocess Data ──────────────────────────────────────────────────────────
def load_data(path):
    # 讀取原始 CSV 檔案
    raw = pd.read_csv(path)
    # 只保留分析所需欄位：price、property_type、district
    data = raw[['price', 'property_type', 'district']].copy()
    # 刪除缺失值，避免影響迴歸結果
    data.dropna(inplace=True)
    return data


# ── Dummy Variable Encoding：房型 ────────────────────────────────────────────────────────
def encode_property_type(data):
    # property_type 只產生 is_townhouse 與 is_condo 兩欄，House 為基準類別（不建立 is_house）
    #   House     -> (is_townhouse=0, is_condo=0)　← 基準類別，效果反映在截距上
    #   Townhouse -> (is_townhouse=1, is_condo=0)
    #   Condo     -> (is_townhouse=0, is_condo=1)
    data = data.copy()
    data['is_townhouse'] = (data['property_type'] == 'Townhouse').astype(int)
    data['is_condo'] = (data['property_type'] == 'Condo').astype(int)
    return data


# ── Dummy Variable Encoding：地區 ────────────────────────────────────────────────────────
def encode_district(data):
    # district 只產生 is_south、is_east、is_west 三欄，London North 為基準類別（不建立 is_north）。
    # 這是標準的 dummy variable 編碼：類別數為 n 時，只需要 n-1 欄虛擬變數，
    # 因為第 n 個類別的資訊可以由「其餘欄位皆為 0」完整表示；
    # 若連基準類別也建立虛擬變數，會與截距項（常數項）產生完美線性重合（dummy variable trap），
    # 導致設計矩陣不滿秩、OLS 無法唯一求解係數。
    #
    # 資料備註：原始 district 欄位實際只有「London North」「London South」「London East」
    # 與未細分方位的「London」四種值，並沒有明確標示為「London West」的資料。
    # 依需求仍要產生四個地區類別（North/South/East/West）對應的 3 個 dummy 欄位，
    # 因此將這 10 筆僅標示為「London」（無方位細分）的資料歸入第四類，以 is_west 代表。
    data = data.copy()
    data['is_south'] = (data['district'] == 'London South').astype(int)
    data['is_east'] = (data['district'] == 'London East').astype(int)
    data['is_west'] = (data['district'] == 'London').astype(int)
    return data


# ── 建立並訓練 OLS 模型 ──────────────────────────────────────────────────────────────────
def fit_ols_model(data):
    feature_columns = ['is_townhouse', 'is_condo', 'is_south', 'is_east', 'is_west']

    # 自變數：房型與地區的 dummy 欄位
    X = data[feature_columns]
    # 加上常數項（截距），代表基準類別（House、London North）的平均 price
    X = sm.add_constant(X)
    # 應變數：price
    y = data['price']

    model = sm.OLS(y, X).fit()
    return model


# ── 印出係數白話解讀 ────────────────────────────────────────────────────────────────────
def print_coefficient_interpretation(model):
    # 每個係數對應的白話意義說明：相對於基準類別（House、London North），
    # 該虛擬變數為 1 時，price 平均會增加或減少多少元
    interpretation_map = {
        'const': '基準類別（House、London North）的平均 price（截距）',
        'is_townhouse': '相對於 House 基準類別，Townhouse 的 price 平均增加/減少多少元',
        'is_condo': '相對於 House 基準類別，Condo 的 price 平均增加/減少多少元',
        'is_south': '相對於 London North 基準類別，London South 的 price 平均增加/減少多少元',
        'is_east': '相對於 London North 基準類別，London East 的 price 平均增加/減少多少元',
        'is_west': '相對於 London North 基準類別，London West（原始資料中未細分方位的「London」）的 price 平均增加/減少多少元',
    }

    print('===== 係數白話解讀 =====')
    for variable_name, meaning in interpretation_map.items():
        coefficient = model.params[variable_name]
        p_value = model.pvalues[variable_name]
        significance = '（統計顯著，p < 0.05）' if p_value < 0.05 else '（不顯著，p >= 0.05）'
        print(f'{variable_name:<14} 係數 = {coefficient:>14,.2f}　{significance}')
        print(f'{"":<14} 白話意義：{meaning}')


# ── Main ────────────────────────────────────────────────────────────────────────────────
data = load_data(FILE_PATH)
data = encode_property_type(data)
data = encode_district(data)

model = fit_ols_model(data)

print('===== OLS 模型完整 Summary =====')
print(model.summary())

print_coefficient_interpretation(model)

print('===== 整體模型解釋力 =====')
print(f'R-squared = {model.rsquared:.4f}')
print(f'Adjusted R-squared = {model.rsquared_adj:.4f}')
