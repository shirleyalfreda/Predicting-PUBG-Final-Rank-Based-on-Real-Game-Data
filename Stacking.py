import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
train_df = pd.read_csv('D:\\360Downloads\\pubg_train.csv')
test_df = pd.read_csv('D:\\360Downloads\\pubg_test.csv')

y = train_df["team_placement"]
x = train_df.drop(columns=["team_placement"])
x = x.fillna(x.mean())
test_df = test_df.fillna(test_df.mean())
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# 基模型
base_models = [
    ('xgb', XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)),
    ('lgb', LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42))
]

# Stacking模型（元模型：岭回归）
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
    cv=5, n_jobs=-1
)

# 训练模型
stacking_model.fit(X_train, y_train)

# 验证评估
y_pred = stacking_model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Stacking 模型评估：")
print(f"MAE: {mae:.4f}")
print(f"R² : {r2:.4f}")

# 测试集预测（team_placement 记得乘以 game_size）
test_features = test_df.copy()
test_features = test_features.fillna(0)  # 填补缺失值
test_features = test_features[x.columns]

y_test_pred = stacking_model.predict(test_features)
y_test_pred = np.clip(y_test_pred.round().astype(int), 1, 100)

# 导出结果
import pandas as pd
submission = pd.DataFrame({'team_placement': y_test_pred})
submission.to_csv('submission_stacking.csv', index=False)

import zipfile
with zipfile.ZipFile('submission_stacking.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('submission_stacking.csv')

print(" 集成模型提交文件已生成：submission_stacking.zip")

from sklearn.metrics import mean_squared_error


mse = mean_squared_error(y_val, y_pred)
score = 100 - mae

print("\n 验证集评估指标：")
print(f"MAE  (平均绝对误差): {mae:.4f}")
print(f"MSE  (均方误差):     {mse:.4f}")
print(f"R²   (决定系数):     {r2:.4f}")
print(f"竞赛评分 (100 - MAE): {score:.4f}")


#  MAE 可视化

mae_per_sample = np.abs(y_val - y_pred)

plt.figure(figsize=(12, 4))
plt.plot(mae_per_sample, color='orange', linewidth=1)
plt.xlabel("sample")
plt.ylabel("MAE (|y_true - y_pred|)")
plt.title("MAE error")
plt.grid(True)
plt.tight_layout()
plt.show()
