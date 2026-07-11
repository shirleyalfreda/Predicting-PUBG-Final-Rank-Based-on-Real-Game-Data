import warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from scipy.stats import zscore
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


#PUBG 排名预测 · 数据分析 & 特征工程 (VAE + ResNet版本)

TRAIN_PATH = r"C:\Users\banana\Desktop\privacy\school\data mining\数据挖掘期末\pubg_train.csv\pubg_train.csv"
TEST_PATH  = r"C:\Users\banana\Desktop\privacy\school\data mining\数据挖掘期末\pubg_test.csv\pubg_test.csv"

LATENT_DIM  = 16
VAE_EPOCHS  = 50
SCALER_TYPE = "zscore"
OUTLIER_STRATEGY = "zscore"
USE_PEARSON_MI_VOTE = True


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")

# 2. 数据读取
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

y = train_df["team_placement"]
train_df.drop(columns=["team_placement"], inplace=True)

#  3. 缺失值处理
num_cols = train_df.select_dtypes("number").columns
cat_cols = train_df.select_dtypes("object").columns

train_df[num_cols] = train_df[num_cols].apply(lambda c: c.fillna(c.mean()))
test_df[num_cols]  = test_df[num_cols].apply(lambda c: c.fillna(c.mean()))
train_df[cat_cols] = train_df[cat_cols].fillna("Missing")
test_df[cat_cols]  = test_df[cat_cols].fillna("Missing")

#  4. 异常值处理
if OUTLIER_STRATEGY == "zscore":
    mask = (np.abs(zscore(train_df[num_cols])) < 3).all(axis=1)
    train_df, y = train_df[mask], y[mask]
else:
    def iqr_clean(df, cols):
        for c in cols:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[(df[c] >= q1 - 1.5*iqr) & (df[c] <= q3 + 1.5*iqr)]
        return df
    mask_iqr = iqr_clean(train_df[num_cols], num_cols).index
    train_df, y = train_df.loc[mask_iqr], y.loc[mask_iqr]

#  5. 特征构造
def add_features(df):
    df["total_dist"] = df["player_dist_ride"] + df["player_dist_walk"]
    df["kd_ratio"]   = df["player_kills"] / (1 + df["player_dbno"])
    df["kd_ratio"]   = df["kd_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df
train_df = add_features(train_df)
test_df  = add_features(test_df)

#  6. 编码 + 缩放
num_cols = train_df.select_dtypes("number").columns
cat_cols = train_df.select_dtypes("object").columns

scaler = StandardScaler() if SCALER_TYPE == "zscore" else MinMaxScaler()
pre = ColumnTransformer([
    ("num", scaler, num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

X_train_scaled = pre.fit_transform(train_df)
X_test_scaled  = pre.transform(test_df)

print("\n 查看前几行数据：")
print(train_df.head())
print("\n 所有特征列（按类型分类）：")
print(f"数值特征 ({len(num_cols)}):", list(num_cols))
print(f"类别特征 ({len(cat_cols)}):", list(cat_cols))
print("\n 描述性统计：")
desc_stats = train_df[num_cols].agg(['mean','std','min','max','skew','kurt']).T
print(desc_stats.round(3))

# ---------- 7. VAE ----------
class VAE(nn.Layer):
    def __init__(self, inp, latent):
        super().__init__()
        self.fc1  = nn.Linear(inp, 128)
        self.fc21 = nn.Linear(128, latent)
        self.fc22 = nn.Linear(128, latent)
        self.fc3  = nn.Linear(latent, 128)
        self.fc4  = nn.Linear(128, inp)
    def encode(self, x):
        h = F.relu(self.fc1(x)); return self.fc21(h), self.fc22(h)
    def reparam(self, mu, logv):
        std = paddle.exp(0.5*logv); eps = paddle.randn(std.shape)
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
    z_test  = vae.reparam(*vae.encode(paddle.to_tensor(X_test_scaled.astype("float32")))).numpy()

#  8. 合并 VAE 潜变量
X_train_all = np.hstack([X_train_scaled, z_train])
X_test_all  = np.hstack([X_test_scaled,  z_test])

# 9. 特征选择
if USE_PEARSON_MI_VOTE:
    samp = np.random.choice(len(y), 15000, replace=False)
    pear = np.array([abs(np.corrcoef(X_train_all[samp, i], y.iloc[samp])[0,1])
                     for i in range(X_train_all.shape[1])])
    mask_pear = pear > np.percentile(pear, 60)
    mi = mutual_info_regression(X_train_all, y, random_state=42)
    mask_mi = mi > np.percentile(mi, 60)
    mask = mask_pear | mask_mi
else:
    mask = np.ones(X_train_all.shape[1], dtype=bool)

X_final = X_train_all[:, mask]
X_test_final = X_test_all[:, mask]
print("After FS:", X_final.shape)

# 10. 使用残差网络进行回归预测
class ResBlock(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    def forward(self, x):
        identity = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.relu(out + identity)

class ResNetRegressor(nn.Layer):
    def __init__(self, inp):
        super().__init__()
        self.input = nn.Linear(inp, 128)
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)
        self.out = nn.Linear(128, 1)
    def forward(self, x):
        x = F.relu(self.input(x))
        x = self.res1(x)
        x = self.res2(x)
        return self.out(x)

# 划分训练集和验证集
X_tr, X_val, y_tr, y_val = train_test_split(X_final, y, test_size=0.2, random_state=42)

X_tr_t = paddle.to_tensor(X_tr.astype("float32"))
y_tr_t = paddle.to_tensor(y_tr.values.reshape(-1,1).astype("float32"))
X_val_t = paddle.to_tensor(X_val.astype("float32"))
y_val_t = paddle.to_tensor(y_val.values.reshape(-1,1).astype("float32"))

model = ResNetRegressor(X_tr.shape[1])
opt = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=1e-3)

# 模型训练
for epoch in range(100):
    model.train()
    pred = model(X_tr_t)
    loss = F.mse_loss(pred, y_tr_t)
    loss.backward()
    opt.step()
    opt.clear_grad()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} - Train MSE Loss: {loss.numpy()[0]:.4f}")

#  11. 模型评估
model.eval()
with paddle.no_grad():
    y_pred = model(X_val_t).numpy().flatten()
    y_true = y_val.values

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
r2  = r2_score(y_true, y_pred)
score = 100 - mae

print("\n 模型评估指标：")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"R²  : {r2:.4f}")
print(f"自定义评分 (100 - MAE) = {score:.4f}")
