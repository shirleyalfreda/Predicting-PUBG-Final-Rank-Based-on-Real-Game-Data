import zipfile
import warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import zscore
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

TRAIN_PATH = 'D:\\360Downloads\\pubg_train.csv'
TEST_PATH = 'D:\\360Downloads\\pubg_test.csv'

LATENT_DIM  = 16   # VAE 隐变量
VAE_EPOCHS  = 50
SCALER_TYPE = "zscore"  # "zscore" or "minmax"
OUTLIER_STRATEGY = "zscore"  # "zscore" or "iqr"
USE_PEARSON_MI_VOTE = True   # 是否执行皮尔逊+互信息投票

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
paddle.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

y = train_df["team_placement"]
train_df.drop(columns=["team_placement"], inplace=True)

#  缺失值处理
num_cols = train_df.select_dtypes("number").columns
cat_cols = train_df.select_dtypes("object").columns

train_df[num_cols] = train_df[num_cols].apply(lambda c: c.fillna(c.mean()))
test_df[num_cols]  = test_df[num_cols].apply(lambda c: c.fillna(c.mean()))
train_df[cat_cols] = train_df[cat_cols].fillna("Missing")
test_df[cat_cols]  = test_df[cat_cols].fillna("Missing")

#  异常值处理
if OUTLIER_STRATEGY == "zscore":
    mask = (np.abs(zscore(train_df[num_cols])) < 3).all(axis=1)
    train_df, y = train_df[mask], y[mask]
else:  # IQR
    def iqr_clean(df, cols):
        for c in cols:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[(df[c] >= q1 - 1.5*iqr) & (df[c] <= q3 + 1.5*iqr)]
        return df
    mask_iqr = iqr_clean(train_df[num_cols], num_cols).index
    train_df, y = train_df.loc[mask_iqr], y.loc[mask_iqr]

#  特征构造
def add_features(df):
    df["total_dist"] = df["player_dist_ride"] + df["player_dist_walk"]
    df["kd_ratio"]   = df["player_kills"] / (1 + df["player_dbno"])
    df["kd_ratio"]   = df["kd_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df
train_df = add_features(train_df)
test_df  = add_features(test_df)

#  编码 + 缩放
num_cols = train_df.select_dtypes("number").columns
cat_cols = train_df.select_dtypes("object").columns

scaler = StandardScaler() if SCALER_TYPE == "zscore" else MinMaxScaler()

pre = ColumnTransformer([
    ("num", scaler, num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

X_train_scaled = pre.fit_transform(train_df)
X_test_scaled  = pre.transform(test_df)


#  VAE
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
    z_test  = vae.reparam(*vae.encode(
        paddle.to_tensor(X_test_scaled.astype("float32"))
    )).numpy()

#  合并 VAE 潜变量
X_train_all = np.hstack([X_train_scaled, z_train])
X_test_all  = np.hstack([X_test_scaled,  z_test])

#  特征选择 (皮尔逊 + 互信息)
if USE_PEARSON_MI_VOTE:
    # 皮尔逊
    samp = np.random.choice(len(y), 15000, replace=False)
    pear = np.array([abs(np.corrcoef(X_train_all[samp, i], y.iloc[samp])[0,1])
                     for i in range(X_train_all.shape[1])])
    mask_pear = pear > np.percentile(pear, 60)
    # 互信息
    mi = mutual_info_regression(X_train_all, y, random_state=42)
    mask_mi = mi > np.percentile(mi, 60)
    mask = mask_pear | mask_mi
else:
    mask = np.ones(X_train_all.shape[1], dtype=bool)

X_final = X_train_all[:, mask]
X_test_final = X_test_all[:, mask]
print("After FS:", X_final.shape)

#  模型 & 评估
y_norm = y / 100.0  # 若你的目标是 team_placement

X_train, X_val, y_train, y_val = train_test_split(X_final, y_norm, test_size=0.2, random_state=42)

X_train_paddle = paddle.to_tensor(X_train, dtype='float32')
X_val_paddle   = paddle.to_tensor(X_val, dtype='float32')
y_train_paddle = paddle.to_tensor(y_train.to_numpy().reshape(-1, 1), dtype='float32')
y_val_paddle   = paddle.to_tensor(y_val.to_numpy().reshape(-1, 1), dtype='float32')

# 构建模型
class VAEEnhancedMLP(nn.Layer):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

model = VAEEnhancedMLP(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)

# 训练模型
# ---------------------------
for epoch in range(300):
    model.train()
    preds = model(X_train_paddle)
    loss = criterion(preds, y_train_paddle)
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()

    model.eval()
    val_preds = model(X_val_paddle)
    val_loss = criterion(val_preds, y_val_paddle)
    print(f"Epoch {epoch+1}, Train Loss: {float(loss.numpy()):.4f}, Val Loss: {float(val_loss.numpy()):.4f}")

# 测试集预测 + 反归一化
model.eval()
X_test_paddle = paddle.to_tensor(X_test_final, dtype='float32')
with paddle.no_grad():
    y_pred_norm = model(X_test_paddle).numpy().flatten()

y_pred = (y_pred_norm * test_df['game_size']).round().astype(int)
y_pred = np.clip(y_pred, 1, 100)

# 文件
submission = pd.DataFrame({'team_placement': y_pred})
submission.to_csv('submission.csv', index=False)

import zipfile
with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('submission.csv')

print(" 成功生成 submission.zip，可提交！")


model.eval()
with paddle.no_grad():
    y_val_pred = model(X_val_paddle).numpy().flatten()

# 真实排名预测
y_val_true = (y_val * 100).values
y_val_pred_real = y_val_pred * 100
residuals = y_val_true - y_val_pred_real

# 计算指标
mae = mean_absolute_error(y_val_true, y_val_pred_real)
r2 = r2_score(y_val_true, y_val_pred_real)

print("\n 模型评估指标：")
print(f"MAE（绝对误差）: {mae:.4f}")
print(f"R²（决定系数）: {r2:.4f}")
print("残差均值：", np.mean(residuals))
print("残差标准差：", np.std(residuals))




