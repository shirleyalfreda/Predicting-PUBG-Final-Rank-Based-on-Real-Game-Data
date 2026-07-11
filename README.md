# 基于真实应用场景的数据分析及数学建模-吃鸡排名预测
## 摘要
PUBG是一款战术竞技类游戏，玩家需在缩小的安全区内收集资源并击败对手，最终生存者将获得"大吉大利，晚上吃鸡"的胜利。数据集包含150万条玩家行为记录，目标变量为团队排名（team_placement）。
在实现过程中，我们首先进行了数据探索与预处理，包括处理极端偏态分布（如对数变换）、缺失值填充（均值填充数值型特征，"Missing"填充类别型特征）以及异常值剔除（Z-score和IQR方法）。随后，通过特征工程构造新特征（如总移动距离total_dist和击杀/倒地比率kd_ratio），并采用变分自编码器（VAE）进行非线性特征提取，以捕捉潜在的游戏行为模式。在建模阶段，我们对比了广义线性模型（GLM）、XGBoost、极端随机森林（ExtraTrees）、多层感知机（MLP）、残差网络（ResNet）以及集成学习（Stacking）共6种模型，最终集成学习方法表现最优（R²=0.8543，MAE=4.7325）。
已实现功能：
1.	数据预处理：缺失值填充、异常值检测、偏态分布校正（对数变换）
2.	特征工程：构造新特征（total_dist、kd_ratio）、VAE特征降维
3.	模型构建与评估：6种模型的对比实验，集成学习优化预测性能
4.	可视化分析：绘图采用PUBG游戏风格的橙黄色系（参考游戏内UI配色），如热力图用暖色调突出特征相关性，直方图用深橙色表示玩家击杀分布，与游戏中的伤害数值显示风格一致。

通过本项目，我们不仅验证了数据挖掘方法在游戏行为分析中的有效性，也为未来优化玩家匹配、反作弊系统等提供了数据支持。

## 项目亮点：
1、完整的数据预处理流程（缺失值填充、异常值检测、偏态分布校正）
2、手工特征工程 + VAE 非线性特征提取
3、6 种模型对比实验（GLM、XGBoost、ExtraTrees、MLP、ResNet、Stacking）
4、可视化分析采用 PUBG 游戏风格的橙黄色系

## 数据集描述
### 数据来源
飞桨学习赛：吃鸡排名预测挑战赛
### 数据规模
数据集	行数	列数
训练集	1,500,000	16
测试集	500,000	15（不含目标变量）
字段含义
字段名	含义
match_id	本局游戏 ID
team_id	本局游戏中队伍 ID
game_size	本局队伍数量
party_size	本局游戏中队伍人数
player_assists	玩家助攻数
player_dbno	玩家击倒数
player_dist_ride	玩家车辆行驶距离
player_dist_walk	玩家步行距离
player_dmg	输出伤害值
player_kills	玩家击杀数
player_name	玩家名称（全局唯一）
kill_distance_x_min	击杀时最小的 x 坐标间隔
kill_distance_x_max	击杀时最大的 x 坐标间隔
kill_distance_y_min	击杀时最小的 y 坐标间隔
kill_distance_y_max	击杀时最大的 y 坐标间隔
team_placement	目标变量：队伍排名

## 数据预处理
### 1. 缺失值填充
数值型特征：使用该列的均值（mean）填充
类别型特征：使用字符串 "Missing" 填充
num_cols = train_df.select_dtypes("number").columns
cat_cols = train_df.select_dtypes("object").columns
train_df[num_cols] = train_df[num_cols].apply(lambda c: c.fillna(c.mean()))
train_df[cat_cols] = train_df[cat_cols].fillna("Missing")

### 2. 异常值检测
采用 Z-score 方法，剔除 |z| > 3 的样本：
from scipy.stats import zscore
mask = (np.abs(zscore(train_df[num_cols])) < 3).all(axis=1)
train_df, y = train_df[mask], y[mask]

### 3. 偏态分布校正
对严重右偏的特征进行对数变换（log1p），压缩长尾分布：
log_cols = [
    "kill_distance_x_max", "kill_distance_y_max",
    "kill_distance_x_min", "kill_distance_y_min",
    "player_dist_ride", "player_dist_walk"
]
def log_transform(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df

### 4. 标准化与编码
数值特征：Z-score 标准化（StandardScaler）
类别特征：独热编码（OneHotEncoder）
使用 ColumnTransformer 统一处理训练集和测试集，防止数据泄露
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])
X_train_scaled = pre.fit_transform(train_df)
X_test_scaled = pre.transform(test_df)

## 特征工程
基于游戏经验构造了两个新特征：
新特征	计算公式	业务含义
1、total_dist：计算公式为player_dist_ride + player_dist_walk	总移动距离，反映玩家活跃度
2、kd_ratio：计算公式为player_kills / (1 + player_dbno)	击杀/倒地比率，衡量玩家战斗效率
def add_features(df):
    df["total_dist"] = df["player_dist_ride"] + df["player_dist_walk"]
    df["kd_ratio"] = df["player_kills"] / (1 + df["player_dbno"])
    df["kd_ratio"] = df["kd_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df
### 1. VAE 非线性特征提取
我们引入了变分自编码器（VAE），对标准化后的特征进行非线性压缩，提取 16 维潜在特征，捕捉游戏行为中的深层模式。
VAE 网络结构：输入层 (原始特征) → 128 → [μ, logσ²] → 采样 z (16维) → 128 → 重构层
import paddle.nn as nn
import paddle.nn.functional as F
class VAE(nn.Layer):
    def __init__(self, inp, latent):
        super().__init__()
        self.fc1 = nn.Linear(inp, 128)
        self.fc21 = nn.Linear(128, latent)  # 均值 μ
        self.fc22 = nn.Linear(128, latent)  # 方差对数 logσ²
        self.fc3 = nn.Linear(latent, 128)
        self.fc4 = nn.Linear(128, inp)
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)
    def reparam(self, mu, logv):
        std = paddle.exp(0.5 * logv)
        eps = paddle.randn(std.shape)
        return mu + eps * std
    def decode(self, z):
        return self.fc4(F.relu(self.fc3(z)))
    def forward(self, x):
        mu, logv = self.encode(x)
        z = self.reparam(mu, logv)
        return self.decode(z), mu, logv
训练完成后，将 VAE 提取的潜变量与原始特征拼接，形成增强特征集：
X_train_all = np.hstack([X_train_scaled, z_train])
X_test_all = np.hstack([X_test_scaled, z_test])

### 2. 特征选择
采用 “皮尔逊相关系数 + 互信息”双指标投票机制，保留被任一方法认定为重要的特征：
from sklearn.feature_selection import mutual_info_regression
#### 皮尔逊相关系数（捕捉线性关系）
pear = np.array([abs(np.corrcoef(X_train_all[samp, i], y.iloc[samp])[0,1]) 
                  for i in range(X_train_all.shape[1])])
mask_pear = pear > np.percentile(pear, 60)
#### 互信息（捕捉非线性关系）
mi = mutual_info_regression(X_train_all, y, random_state=42)
mask_mi = mi > np.percentile(mi, 60)
#### 并集保留
mask = mask_pear | mask_mi
X_final = X_train_all[:, mask]
筛选后特征维度降至 16 维。

## 模型与实验结果
1、GLM（广义线性模型）	线性模型	可解释性强，计算效率高
2、XGBoost	树模型	非线性拟合强，内置正则化
3、ExtraTrees	树模型	极端随机化，训练速度快
4、MLP	神经网络	多层感知机，4层（128-64-32-1）
5、ResNet	神经网络	引入残差块，缓解退化问题
6、Stacking	集成学习	XGBoost + LightGBM + MLP → RidgeCV
### 1. 评估指标
MAE（平均绝对误差）：衡量预测排名与实际排名的平均偏差
R²（决定系数）：衡量模型对目标变量的解释程度
Score = 100 - MAE（竞赛评分标准）
### 2. 实验结果对比
模型	MAE	R²	Score (100 - MAE)
GLM（Gamma 分布 + Log 链接）	5.0165	0.7593	94.98
XGBoost	4.5380	0.7669	95.46
ExtraTrees	4.8252	0.7725	95.17
MLP	15.36	0.5681	95.13
ResNet	13.34	0.6507	86.66
Stacking（集成学习）	4.7325	0.8543	95.27
最佳 R² 由 Stacking 集成学习取得（0.8543），单模型最佳 MAE 由 XGBoost 取得（4.5380）。
### 3. 可视化结果
线性模型 vs 树模型
https://media/linear_tree_compare.png
神经网络模型对比
https://media/nn_compare.png

## 观察结论：
树模型（XGBoost、ExtraTrees） 在结构化数据上表现优异，误差分布集中稳定
GLM 虽简单，但经过充分特征工程后表现不俗
神经网络 在此类表格数据上需要更长的训练时间，本项目中循环次数不足限制了其潜力
Stacking 通过融合不同模型的优势，取得了最高的 R²

## 项目结构
PUBG-Rank-Prediction/
│
├── data/                          # 数据集（不包含在仓库中）
│   ├── train.csv                  # 训练集（150万行）
│   └── test.csv                   # 测试集（50万行）
│
├── src/                           # 源代码
│   ├── preprocessing.py           # 数据预处理（缺失值、异常值、对数变换）
│   ├── features.py                # 特征工程（手工特征 + VAE）
│   ├── models.py                  # 各模型定义（GLM、XGBoost、ExtraTrees）
│   ├── neural_nets.py             # 神经网络（MLP、ResNet）
│   ├── stacking.py                # Stacking 集成学习
│   └── utils.py                   # 工具函数（可视化、评估指标）
│
├── notebooks/                     # Jupyter Notebook（EDA 与实验记录）
│   └── eda.ipynb
│
├── results/                       # 实验结果
│   ├── models/                    # 训练好的模型权重
│   └── logs/                      # 训练日志
│
├── media/                         # README 图片资源
│   ├── linear_tree_compare.png
│   └── nn_compare.png
│
├── requirements.txt               # 依赖包列表
├── README.md                      # 项目说明（本文件）
└── LICENSE                        # MIT 许可证

## 快速开始
### 环境要求
Python 3.12
PaddlePaddle 2.6+
其他依赖见 requirements.txt

## 模型优缺点分析
### 1. GLM（广义线性模型）
✅ 优点：结构简单、计算高效、可解释性强、不易过拟合
❌ 缺点：无法自动捕捉非线性关系，依赖特征工程质量
### 2. XGBoost
✅ 优点：非线性拟合能力强、内置正则化防过拟合、自动选择最优分裂特征
❌ 缺点：参数众多调优成本高、训练时间长、需对类别特征预处理
### 3. ExtraTrees
✅ 优点：随机性大减少计算量、对数据波动不敏感、无需标准化
❌ 缺点：缺少残差迭代机制、局部拟合能力弱于 XGBoost、精度相对较低
### 4. MLP
✅ 优点：适用于非结构化数据、能拟合复杂非线性关系、高度并行化训练
❌ 缺点：在表格数据上需要大量训练轮次、黑盒难解释、超参数敏感
### 5. ResNet
✅ 优点：通过残差块解决深层网络退化、非线性表达能力强
❌ 缺点：参数多训练慢、网络稳定性依赖精细调参、内存占用高
### 6. Stacking（集成学习）
✅ 优点：融合多模型优势、无需精细特征选择、内置交叉验证防过拟合
❌ 缺点：组合与参数调试困难、逻辑路径不透明、训练成本高

## 未来展望
基于本次项目的经验，未来可以从以下方向继续优化：
自动特征工程：尝试 OpenFE 工具自动生成有意义的特征组合（如比值、差值、多项式），已编写相关代码但受限于环境配置未能实际运行
注意力机制：引入 Attention 机制，让模型像职业玩家一样聚焦关键行为特征
正则化优化：通过 L1/L2 正则化保持模型“轻量化”，防止过拟合
深度模型优化：增加 MLP/ResNet 的训练轮次和网络深度，挖掘其在此类数据上的潜力
模型部署：将训练好的模型封装为轻量级 API，提供实时比赛排名预测服务


⭐ 如果这个项目对你有帮助，欢迎给一个 Star！;)

以上是完整的中文版 README。你可以直接复制粘贴到你的 GitHub 仓库中，需要额外做的准备工作：
创建 media/ 文件夹，把报告中的两张对比图提取为独立图片文件放入其中
创建 requirements.txt 列出所有依赖
将代码文件整理到 src/ 目录下
如果需要我帮你生成 requirements.txt 或其他辅助文件，随时告诉我！
