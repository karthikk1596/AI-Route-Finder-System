import os
import urllib.request
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title='WattWise | Energy Intelligence', page_icon='⚡', layout='wide')
DATA_FILE = 'energydata_complete.csv'
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv'

@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_FILE):
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    df = pd.read_csv(DATA_FILE, parse_dates=['date'])
    df = df.drop_duplicates().sort_values('date').reset_index(drop=True)
    return df

st.title('⚡ WattWise')
st.subheader('Uncertainty-Aware Energy Intelligence & Anomaly Detection')
st.caption('A Statistical Machine Learning project using real appliance-energy data from the UCI Machine Learning Repository.')

with st.spinner('Loading real energy dataset (first run may take a minute)...'):
    try:
        df = load_data()
    except Exception as e:
        st.error('Could not download the dataset automatically. Check your internet connection and try again. Details: ' + str(e))
        st.markdown('Dataset link: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction')
        st.stop()

# Feature engineering: only predictors available in the dataset, plus time features.
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
TARGET = 'Appliances'
features = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T_out', 'RH_out', 'Windspeed', 'Visibility', 'hour', 'day_of_week', 'is_weekend']
features = [c for c in features if c in df.columns]

# Sidebar controls
st.sidebar.header('Project controls')
model_choice = st.sidebar.selectbox('Prediction model', ['Random Forest', 'Linear Regression'])
train_fraction = st.sidebar.slider('Training data proportion (chronological split)', 0.60, 0.90, 0.80, 0.05)
cluster_k = st.sidebar.slider('Usage clusters (K-means)', 2, 5, 3)

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric('Dataset records', f'{len(df):,}')
c2.metric('Features used', str(len(features)))
c3.metric('Average appliance use', f'{df[TARGET].mean():.1f} Wh')
c4.metric('Missing values', f'{int(df.isna().sum().sum()):,}')

st.markdown('### 1. Consumption overview')
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    daily = df.set_index('date')[TARGET].resample('D').mean()
    ax.plot(daily.index, daily.values)
    ax.set_title('Daily average appliance energy use')
    ax.set_ylabel('Energy (Wh per 10-minute reading)')
    ax.set_xlabel('Date')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(df[TARGET], bins=35, edgecolor='white')
    ax.set_title('Distribution of appliance energy use')
    ax.set_xlabel('Energy (Wh)')
    ax.set_ylabel('Frequency')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown('### 2. Predictive modelling')
X = df[features].replace([np.inf, -np.inf], np.nan).fillna(df[features].median(numeric_only=True))
y = df[TARGET].astype(float)
split = int(len(df) * train_fraction)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
if model_choice == 'Random Forest':
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
else:
    model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
a, b, c = st.columns(3)
a.metric('MAE', f'{mae:.2f} Wh', help='Mean absolute prediction error; lower is better.')
b.metric('RMSE', f'{rmse:.2f} Wh', help='Penalizes larger errors; lower is better.')
c.metric('R²', f'{r2:.3f}', help='Relative fit score; may be negative on a difficult test set.')

fig, ax = plt.subplots(figsize=(11, 3.5))
show_n = min(300, len(y_test))
ax.plot(df['date'].iloc[split:split+show_n], y_test.iloc[:show_n].values, label='Actual', linewidth=1.5)
ax.plot(df['date'].iloc[split:split+show_n], pred[:show_n], label='Predicted', linewidth=1.2)
ax.set_title('Actual vs predicted consumption (first test observations)')
ax.set_ylabel('Energy (Wh)')
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Bootstrap interval for the mean absolute error, descriptive uncertainty estimate.
rng = np.random.default_rng(42)
abs_errors = np.abs(y_test.to_numpy() - pred)
if len(abs_errors) > 1:
    boot_means = [rng.choice(abs_errors, size=len(abs_errors), replace=True).mean() for _ in range(500)]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    st.info(f'Bootstrap 95% interval for mean absolute error: **{lo:.2f} to {hi:.2f} Wh**. This describes uncertainty in the test-set MAE, not a guaranteed interval for an individual forecast.')

st.markdown('### 3. Statistical insight: weekday vs weekend')
weekdays = df.loc[df['is_weekend'] == 0, TARGET]
weekends = df.loc[df['is_weekend'] == 1, TARGET]
if len(weekdays) and len(weekends):
    t_stat, p_value = stats.ttest_ind(weekdays, weekends, equal_var=False, nan_policy='omit')
    u, v = st.columns(2)
    u.metric('Weekday mean', f'{weekdays.mean():.2f} Wh')
    v.metric('Weekend mean', f'{weekends.mean():.2f} Wh')
    st.write(f'Welch two-sample t-test: **t = {t_stat:.3f}**, **p-value = {p_value:.4g}**.')
    if p_value < 0.05:
        st.success('At the 5% significance level, this dataset shows a statistically significant difference between weekday and weekend average consumption. This does not prove that the day type causes the difference.')
    else:
        st.warning('At the 5% significance level, this test does not provide sufficient evidence of a weekday/weekend mean difference.')

st.markdown('### 4. Unusual consumption detector')
contamination = st.slider('Expected fraction of unusual readings (setting)', 0.01, 0.10, 0.03, 0.01)
iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
iso.fit(X)
df_display = df[['date', TARGET]].copy()
df_display['anomaly'] = iso.predict(X) == -1
anoms = df_display[df_display['anomaly']].sort_values(TARGET, ascending=False)
st.write(f'Flagged **{len(anoms):,}** readings as unusual out of **{len(df):,}** records. These are statistical flags, not proof of a fault or energy waste.')
st.dataframe(anoms.head(12).rename(columns={'date':'Timestamp','Appliances':'Energy (Wh)','anomaly':'Flagged unusual'}), use_container_width=True, hide_index=True)

st.markdown('### 5. Consumption pattern discovery (K-means + PCA)')
cluster_features = ['Appliances', 'lights', 'T1', 'RH_1', 'T_out', 'RH_out', 'Windspeed']
cluster_features = [c for c in cluster_features if c in df.columns]
# Sample for responsive clustering; deterministic for repeatable demo.
sample = df[cluster_features].sample(n=min(5000, len(df)), random_state=42).copy()
scaler = StandardScaler()
scaled = scaler.fit_transform(sample)
km = KMeans(n_clusters=cluster_k, random_state=42, n_init=10)
labels = km.fit_predict(scaled)
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(scaled)
plot_df = pd.DataFrame({'PC1': coords[:,0], 'PC2': coords[:,1], 'Cluster': labels.astype(str)})
fig, ax = plt.subplots(figsize=(8, 4))
for label in sorted(plot_df['Cluster'].unique()):
    part = plot_df[plot_df['Cluster'] == label]
    ax.scatter(part['PC1'], part['PC2'], s=10, alpha=0.55, label=f'Cluster {label}')
ax.set_title('Usage clusters visualized in 2D PCA space')
ax.set_xlabel('Principal component 1')
ax.set_ylabel('Principal component 2')
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption(f'The first two principal components explain {pca.explained_variance_ratio_.sum()*100:.1f}% of variance in this standardized sample.')
cluster_summary = sample.assign(Cluster=labels).groupby('Cluster')[cluster_features].mean().round(2)
st.write('Average feature values by cluster (use these values to interpret the groups):')
st.dataframe(cluster_summary, use_container_width=True)

st.markdown('### 6. Try a sample prediction')
st.write('Change a few input values to see how the trained model responds. Values are illustrative inputs based on the dataset ranges.')
latest = X.median(numeric_only=True)
with st.form('prediction_form'):
    inp = {}
    cols = st.columns(3)
    for i, feat in enumerate(features):
        if feat in ['hour']:
            mn, mx, step = 0, 23, 1
        elif feat == 'day_of_week':
            mn, mx, step = 0, 6, 1
        elif feat == 'is_weekend':
            mn, mx, step = 0, 1, 1
        else:
            mn = float(X[feat].quantile(0.01))
            mx = float(X[feat].quantile(0.99))
            step = max((mx - mn) / 100, 0.1)
        inp[feat] = cols[i % 3].number_input(feat, min_value=float(mn), max_value=float(mx), value=float(np.clip(latest[feat], mn, mx)), step=float(step), key='input_'+feat)
    submitted = st.form_submit_button('Predict energy use')
if submitted:
    row = pd.DataFrame([inp], columns=features)
    value = float(model.predict(row)[0])
    st.success(f'Predicted appliance energy use: **{value:.2f} Wh per 10-minute interval**')

st.markdown('---')
st.caption('Dataset: Candanedo et al. (2017), Appliances Energy Prediction, UCI Machine Learning Repository, CC BY 4.0. This is an educational prototype; it is not a utility billing tool or a safety monitor.')
