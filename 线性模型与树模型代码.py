import warnings, pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")
main_color  = "#FFA500"
shade_color = "#FFD27F"
grid_color  = "#E8E8E8"

TRAIN_PATH = r"C:\Users\banana\Desktop\privacy\school\data mining\数据挖掘期末\pubg_train.csv\pubg_train.csv"
TEST_PATH  = r"C:\Users\banana\Desktop\privacy\school\data mining\数据挖掘期末\pubg_test.csv\pubg_test.csv"
LATENT_DIM  = 16
VAE_EPOCHS  = 50
OUTLIER_STRATEGY = "iqr"

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"训练集: {train_df.shape}, 测试集: {test_df.shape}")
print("训练集列名:", train_df.columns)
print("测试集列名:", test_df.columns)
print("\n查看前几行数据：")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(train_df.head())
print("\n描述性统计（均值、方差、偏度、峰度）:")
desc_stats = train_df.agg(['mean','std','min','max','skew','kurt']).T
print(desc_stats.round(3))

y = train_df["team_placement"]
train_df.drop(columns=["team_placement"], inplace=True)

plt.figure(figsize=(10, 4))
skews = train_df.skew().sort_values()
ax = skews.plot(kind='bar', color='orange')
plt.title("Feature Skewness")
plt.ylabel("Skewness")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
plt.show()

features = [
    "kill_distance_x_max",
    "kill_distance_y_max",
    "kill_distance_x_min",
    "kill_distance_y_min",
    "player_dist_ride",
    "player_dist_walk"
]

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
axs = axs.flatten()

for i, col in enumerate(features):
    data = train_df[col].dropna()
    density = data.plot(kind="density", color=main_color, linewidth=2, ax=axs[i])

    xs, ys = density.get_lines()[0].get_data()
    axs[i].fill_between(xs, ys, color=shade_color, alpha=0.5)

    axs[i].set_title(f"{col} Distribution", fontsize=11)
    axs[i].grid(True, color=grid_color, linestyle="--", linewidth=0.5)
    axs[i].set_yticks([])

plt.suptitle("Highly Skewed Feature Distributions", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

corr = train_df.copy()
corr['team_placement'] = y
plt.figure(figsize=(14, 10))
sns.heatmap(corr.corr(), cmap='YlOrBr', annot=True, fmt=".2f", linewidths=0.5,
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap (with target)")
plt.tight_layout()
plt.show()

num_cols = train_df.select_dtypes("number").columns
cat_cols = train_df.select_dtypes("object").columns
train_df[num_cols] = train_df[num_cols].apply(lambda c: c.fillna(c.mean()))
test_df[num_cols]  = test_df[num_cols].apply(lambda c: c.fillna(c.mean()))
train_df[cat_cols] = train_df[cat_cols].fillna("Missing")
test_df[cat_cols]  = test_df[cat_cols].fillna("Missing")

if OUTLIER_STRATEGY == "zscore":
    mask = (np.abs(zscore(train_df[num_cols])) < 3).all(axis=1)
    train_df, y = train_df[mask], y[mask]
else:
    def iqr_clean(df, cols):
        for c in cols:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[(df[c] >= q1 - 4.5*iqr) & (df[c] <= q3 + 4.5*iqr)]
        return df
    mask_iqr = iqr_clean(train_df[num_cols], num_cols).index
    train_df, y = train_df.loc[mask_iqr], y.loc[mask_iqr]

num_cols = train_df.select_dtypes("number").columns
cat_cols = train_df.select_dtypes("object").columns
scaler = StandardScaler()
pre = ColumnTransformer([
    ("num", scaler, num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])
X_train_scaled = pre.fit_transform(train_df)
X_test_scaled  = pre.transform(test_df)


def add_features(df):
    df["total_dist"] = df["player_dist_ride"] + df["player_dist_walk"]
    df["kd_ratio"]   = df["player_kills"] / (1 + df["player_dbno"])
    df["kd_ratio"]   = df["kd_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df
train_df = add_features(train_df)
test_df  = add_features(test_df)

def log_transform(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df

log_cols = [
    "player_kills",
    "player_dmg",
    "player_dist_walk",
    "player_dist_ride",
    "kd_ratio"
]
train_df = log_transform(train_df, log_cols)
test_df  = log_transform(test_df, log_cols)

class VAE(nn.Layer):
    def __init__(self, inp, latent):
        super().__init__()
        self.fc1  = nn.Linear(inp, 128)
        self.fc21 = nn.Linear(128, latent)
        self.fc22 = nn.Linear(128, latent)
        self.fc3  = nn.Linear(latent, 128)
        self.fc4  = nn.Linear(128, inp)
    def encode(self, x):
        h = F.relu(self.fc1(x));
        return self.fc21(h), self.fc22(h)
    def reparam(self, mu, logv):
        std = paddle.exp(0.5*logv);
        eps = paddle.randn(std.shape)
        return mu + eps*std
    def decode(self, z):
        return self.fc4(F.relu(self.fc3(z)))
    def forward(self, x):
        mu, logv = self.encode(x)
        z = self.reparam(mu, logv)
        return self.decode(z), mu, logv
vae = VAE(X_train_scaled.shape[1], LATENT_DIM)
opt = paddle.optimizer.Adam(parameters=vae.parameters())
x_t = paddle.to_tensor(X_train_scaled.astype("float32"))
for ep in range(VAE_EPOCHS):
    vae.train()
    recon, mu, logv = vae(x_t)
    loss = F.mse_loss(recon, x_t) + -0.5*paddle.mean(1 + logv - mu**2 - paddle.exp(logv))
    loss.backward(); opt.step(); opt.clear_grad()
    if ep % 10 == 0: print(f"Epoch {ep} | loss {loss.numpy():.4f}")
vae.eval()
with paddle.no_grad():
    z_train = vae.reparam(*vae.encode(x_t)).numpy()
    z_test  = vae.reparam(*vae.encode(
        paddle.to_tensor(X_test_scaled.astype("float32"))
    )).numpy()

X_train_all = np.hstack([X_train_scaled, z_train])
X_test_all  = np.hstack([X_test_scaled,  z_test])

samp = np.random.choice(len(y), 15000, replace=False)
pear = np.array([abs(np.corrcoef(X_train_all[samp, i], y.iloc[samp])[0,1])
                     for i in range(X_train_all.shape[1])])
mask_pear = pear > np.percentile(pear, 60)
mi = mutual_info_regression(X_train_all, y, random_state=42)
mask_mi = mi > np.percentile(mi, 60)
mask = mask_pear | mask_mi
X_final = X_train_all[:, mask]
X_test_final = X_test_all[:, mask]
print("After FS:", X_final.shape)

X_tr, X_val, y_tr, y_val = train_test_split(X_final, y, test_size=0.2, random_state=42)
X_glm = sm.add_constant(X_final)
X_tr_glm, X_val_glm, y_tr, y_val = train_test_split(X_glm, y, test_size=0.2, random_state=42)
glm_gamma = sm.GLM(y_tr, X_tr_glm, family=sm.families.Gamma(sm.families.links.log()))
result = glm_gamma.fit()
y_pred_glm = result.predict(X_val_glm)
mae_glm = mean_absolute_error(y_val, y_pred_glm)
mse_glm = mean_squared_error(y_val, y_pred_glm)
r2_glm  = r2_score(y_val, y_pred_glm)
print(f"GLM(Gamma) | MAE={mae_glm:.4f} | MSE={mse_glm:.4f} | R²={r2_glm:.4f} | Score={100 - mae_glm:.2f} ")

models = {
    "GLM_Gamma": result,
    "XGB": XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.01, objective="reg:absoluteerror"),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=250, random_state=42)
}
scores = {}
preds_all = {}
for name, model in models.items():
    if name == "GLM_Gamma":
        pred = model.predict(X_val_glm)
    else:
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
    preds_all[name] = pred
    mae = mean_absolute_error(y_val, pred)
    mse = mean_squared_error(y_val, pred)
    r2 = r2_score(y_val, pred)
    scores[name] = 100 - mae
    print(f"{name:10} | MAE={mae:.4f} | MSE={mse:.4f} | R²={r2:.4f}| Score={scores[name]:.2f}")

best = max(scores, key=scores.get)
print("\n最优模型:", best)

final_pred = np.clip(np.round(models[best].predict(X_test_final)), 1, None)
pd.DataFrame({"team_placement": final_pred.astype(int)}).to_csv("submission.csv", index=False)
print("submission.csv 已保存")

plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
for name, pred in preds_all.items():
    sns.scatterplot(x=y_val, y=pred, label=name, alpha=0.4)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=1.5, label="Ideal")
plt.title("Prediction vs True (Validation)")
plt.xlabel("True team_placement")
plt.ylabel("Predicted team_placement")
plt.legend()
plt.subplot(1, 2, 2)
for name, pred in preds_all.items():
    error = pred - y_val
    sns.kdeplot(error, label=name, fill=True, alpha=0.4)
plt.axvline(0, color='k', linestyle='--')
plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.legend()
plt.tight_layout()
plt.show()

print("\n各模型预测误差标准差:")
for name, pred in preds_all.items():
    error = pred - y_val
    std_err = np.std(error)
    print(f"{name:10} | 误差标准差 = {std_err:.4f}")
