# ── Package imports ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

# ── Parameter settings ────────────────────────────────────────────────────────
DATA_PATH = r'C:\Users\flyin\OneDrive\桌面\新代碼\Linear_OLS_Regression\data\london_zolo_house_prices.csv'

# ── Load data ──────────────────────────────────────────────────────────────────
data = pd.read_csv(DATA_PATH)

# log_price: house price distribution is right-skewed (a few luxury homes have
# extremely high prices), taking the log makes it closer to a normal
# distribution. Same reasoning as with stock prices: the log transform makes
# the variable more suitable for a linear model.
data['log_price'] = np.log(data['price'])

# ── Remove outliers: judged by the z-score of log_price ──────────────────────
# z-score = (value - mean) / std, representing how many standard deviations
# this observation is away from the mean.
# Here we use a common industry threshold: |z-score| > 3 is treated as an
# outlier and removed, because under a normal distribution assumption, values
# beyond 3 standard deviations theoretically account for less than 0.3% of
# observations and are relatively rare extreme cases.
log_price_mean = data['log_price'].mean()
log_price_std = data['log_price'].std()
z_score = (data['log_price'] - log_price_mean) / log_price_std

before_count = len(data)
data = data.loc[z_score.abs() <= 3].reset_index(drop=True)
after_count = len(data)
print(f'Outlier removal: originally {before_count} rows, removed {before_count - after_count} rows, '
      f'{after_count} rows remaining')

# ------ Define the List of House Price Factors -------------------------------
# Originally named char: char conventionally refers to a "single character" in
# programming, so using it for a list holding a whole set of candidate feature
# names is misleading. Renamed here to candidate_factors.
# Also, 'property_type' originally appeared twice in the list; the duplicate
# has been removed here.
candidate_factors = [
    'district',
    'property_type',
    'bedrooms',
    'house_age_years',
    'distance_to_western_km']

# ── Missing value check: house_age_years in candidate_factors has missing values ─
# This is a "necessary" preprocessing step when putting all candidate_factors
# into the regression model: neither sklearn's LinearRegression nor
# statsmodels' OLS accepts NaN, so if any column in candidate_factors is
# missing for a given row, that row cannot be used to fit the model. Here we
# directly drop these rows and print the number of dropped rows to keep track
# of how the sample size changes.
missing_mask = data[candidate_factors].isnull().any(axis=1)
before_dropna_count = len(data)
data = data.loc[~missing_mask].reset_index(drop=True)
after_dropna_count = len(data)
print(f'Rows dropped due to missing values in candidate_factors: '
      f'{before_dropna_count - after_dropna_count}, '
      f'{after_dropna_count} rows remaining')

# ── Basic statistics: descriptive statistics by group ─────────────────────────
def basic_statistical_analysis(data: pd.DataFrame, factor: str) -> pd.DataFrame:

    print(f'=== Statistical Price Analysis on {factor} ===')
    result = data.groupby(factor)['price'].agg(['count', 'mean', 'median', 'std']).round(0)
    print(result)
    
    return result

# ── Plot: single factor vs house price scatter plot ───────────────────────────
# The parameter here is named factor instead of char: this keeps it consistent
# with the parameter naming in basic_statistical_analysis, and also avoids
# using char, a name that's easily mistaken for a "single character".
# A scatter plot (rather than a boxplot) is used because candidate_factors
# contains both categorical variables (district, property_type) and
# continuous variables (bedrooms, house_age_years, distance_to_western_km);
# a scatter plot works for both types of x-axis, so there's no need to write
# two separate plotting routines for the different factor types.
def plot_factor_vs_price(data: pd.DataFrame, factor: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(data[factor], data['price'], alpha=0.4, s=15, color='steelblue')
    ax.set_title(f'{factor} vs House Price')
    ax.set_xlabel(factor)
    ax.set_ylabel('Price')
    plt.tight_layout()
    plt.savefig(f'london_price_vs_{factor}.png', dpi=150)
    plt.show()

# ── One Hot Encoding: convert categorical variables into numeric dummy variables ─
# drop_first=True: avoids the dummy variable trap. The dropped category
# becomes the baseline group, and the remaining coefficients represent the
# difference relative to that baseline group.
# Originally named Coded_Char: the mixed-case naming (PascalCase) was
# inconsistent with the style of the other variables (data, model, pred,
# which are all lowercase snake_case), so it has been renamed to
# encoded_features, which also more directly reflects what it actually
# contains: the encoded feature matrix.
#
# Among candidate_factors, district and property_type are categorical
# variables and need one hot encoding; bedrooms, house_age_years, and
# distance_to_western_km are already numeric and can be merged in directly
# without any additional encoding.
categorical_factors = ['district', 'property_type']
numeric_factors = [f for f in candidate_factors if f not in categorical_factors]

encoded_categorical = pd.get_dummies(data[categorical_factors], drop_first=True, dtype=int)
encoded_features = pd.concat([encoded_categorical, data[numeric_factors]], axis=1)
y = data['log_price']

print('\n=== Encoded feature columns (baseline categories omitted) ===')
print(encoded_features.columns.tolist())

# ── OLS regression (sklearn version, for prediction and R-squared) ───────────
model = LinearRegression()
model.fit(encoded_features, y)
pred = model.predict(encoded_features)

print('\n=== Model performance ===')
print(f'R-squared: {r2_score(y, pred):.4f}')
print(f'Intercept (baseline group log price): {model.intercept_:.4f}')
print('Coefficients for each feature (log price difference):')
for name, coef in zip(encoded_features.columns, model.coef_):
    print(f'  {name}: {coef:+.4f} (multiplier: {np.exp(coef):.2f}x)')

# ── OLS regression (statsmodels version, for statistical significance testing) ─
# The t-test and p-values can be used to determine whether the effect of a
# given category on house price is statistically significant (commonly using
# p < 0.05 as the threshold).
encoded_features_sm = sm.add_constant(encoded_features.astype(float))
sm_model = sm.OLS(y, encoded_features_sm).fit()
print('\n=== Statistical significance test (statsmodels OLS summary) ===')
print(sm_model.summary())

# ── Figure 3: actual vs predicted values scatter plot ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y, pred, alpha=0.4, s=15, color='steelblue')
lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
ax.plot(lims, lims, color='red', linestyle='--', linewidth=1)
ax.set_title('Actual log Price vs Predicted log Price')
ax.set_xlabel('Actual log Price')
ax.set_ylabel('Predicted log Price')
plt.tight_layout()
plt.savefig('london_ols_actual_vs_pred.png', dpi=150)
plt.show()

# ── Figure 4: standardized residual analysis (each residual divided by the ───
# residual standard deviation, converted into a standardized residual) ───────
# Standardized residual = residual / std of residuals. After this conversion,
# you can directly judge the degree of outlying behavior in terms of "how
# many standard deviations away", which makes it convenient to compare against
# the normal distribution's 68% and 95% empirical rules, without separately
# looking up what the raw residual standard deviation is.
residuals = y - pred
standardized_residuals = residuals / residuals.std()

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(pred, standardized_residuals, alpha=0.4, s=15, color='darkorange')
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.set_title('Standardized Residual Plot (Predicted Value vs Standardized Residual)')
ax.set_xlabel('Predicted log Price')
ax.set_ylabel('Standardized Residual')
plt.tight_layout()
plt.savefig('london_ols_residuals.png', dpi=150)
plt.show()

basic_statistical_analysis(data, 'distance_to_western_km')
plot_factor_vs_price(data, 'distance_to_western_km')
