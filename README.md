# 分数阶方程的 Fourier 复合神经网络方法

本仓库提供四个数值算例的可运行代码、论文 PDF、解与误差图、最终误差表，以及各算例的网络架构和复现说明。每个算例均包含 α=0.4、0.8、1.2、1.6、2.0 五组实际运行结果。

配套论文：[求解分数阶 Laplace 问题的 Fourier 复合神经网络方法](paper/%E6%B1%82%E8%A7%A3%E5%88%86%E6%95%B0%E9%98%B6%20Laplace%20%E9%97%AE%E9%A2%98%E7%9A%84%20Fourier%20%E5%A4%8D%E5%90%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%96%B9%E6%B3%95.pdf)。

## 算例入口

| 算例 | 架构、参数与使用说明 | 运行代码 | 结果图 | 最终误差表 |
|---|---|---|---|---|
| 1 · 一维平滑扩散 | [README](example1/readme.md) | [main.py](example1/main.py) | [images](example1/images/) | [CSV](example1/results_summary.csv) |
| 2 · 一维对流扩散 | [README](example2/readme.md) | [main.py](example2/main.py) | [images](example2/images/) | [CSV](example2/results_summary.csv) |
| 3 · 一维非正则初值 | [README](example3/readme.md) | [main.py](example3/main.py) | [images](example3/images/) | [CSV](example3/results_summary.csv) |
| 4 · 二维 Gaussian 扩散 | [README](example4/readme.md) | [main.py](example4/main.py) | [images](example4/images/) | [CSV](example4/results_summary.csv) |

## 目录

```text
example1/                 # example2、example3、example4 的结构相同
  main.py                 # 单个 α 的运行入口
  readme.md               # 架构、参数、复现方法与结果分析
  results_summary.csv     # 五种 α 的最终误差
  images/
    alpha_0_4/ ...         # 解对比、误差分布、初值及局部放大图
  other/
    support/              # 主程序依赖的辅助模块
    alpha_0_4/ ...         # 权重、数据、预测数组、配置与指标
      training_image/     # 对应实际训练历史的曲线
paper/                    # 配套论文 PDF
README.md
requirements.txt
```

## 运行

本次运行环境为 Python 3.11、PyTorch 2.5.1+cu121、NumPy 2.4.6、SciPy 1.17.1、Matplotlib 3.11.1，GPU 为 NVIDIA RTX 3060 Laptop（6 GB）。依赖版本见 [requirements.txt](requirements.txt)。CUDA 不可用时程序使用 CPU，运行时间会变长。例 1–3 的网络使用 float32；例 4 的 PINN 使用 float64，FNO 使用 float32，训练和评估均关闭 TF32。

在仓库根目录安装依赖，然后选择一个算例：

```bash
python -m pip install -r requirements.txt
cd example1

# 加载已有权重，重新计算 α=1.6 的指标和图片
python main.py --stage evaluate --alpha 1.6

# 例 1–3：从头训练并评估，另存到新目录
python main.py --stage train --alpha 1.6 --out other/rerun_alpha_1_6
```

例 4 在 `example4` 目录中执行：

```bash
# 加载已有权重并评估
python main.py --stage evaluate --alpha 1.6

# 生成数值训练数据、训练两个分支并评估
python main.py --stage all --alpha 1.6 --out other/rerun_alpha_1_6
```

每次命令只处理一个 α，可将 `--alpha` 改为 `0.4`、`0.8`、`1.2`、`1.6` 或 `2.0`。不同 α 使用各自的权重和数据。默认输出对应 `other/alpha_*/`，主要结果图位于 `images/alpha_*/`；`--out` 用于另存一次复现实验。请保留 `other/support/` 以及所选 α 的权重、配置和数据文件。具体阶段、参数及输出说明见各算例 README。

## 方法与评价范围

代码将解分为齐次分支 u1 与源项分支 u2。PINN 根据频域方程和已知初值学习 u1；FNO 根据源项及坐标学习响应，再按时间累积得到 u2；最终预测为两支之和。例 1–3 的 FNO 使用独立参考标签，例 4 使用独立数值求解生成的标签；标签均不通过减去 PINN 预测来补偿 PINN 误差。

例 4 直接使用独立的 kx、ky、t 输入及完整二维 Fourier 逆积分，不施加径向网络约化，也不对预测做旋转或反射平均。参考 u、参考 u1 及 u2=u−u1 用于最终误差比较。

这些结果评估每个给定源项轨迹上的重构能力；例 4 另使用与训练时间不重合的数值验证切片。仓库目前不包含针对未见源项的算子泛化实验。更换方程、初值或源项时，需要相应检查数据生成与参考计算，并重新训练模型。

## 结果概览

以下为总解的全时空 MAE。例 1–3 使用 81×200 个评价点，例 4 使用 81×64×64 个评价点；算例的尺度和观察区域不同，数值不用于直接评判不同问题的难度。

| α | Example 1 | Example 2 | Example 3 | Example 4 |
|---:|---:|---:|---:|---:|
| 0.4 | 4.318243e-04 | 4.068056e-04 | 1.801671e-02 | 8.327292e-05 |
| 0.8 | 2.065068e-04 | 2.823200e-04 | 9.468455e-03 | 7.483836e-05 |
| 1.2 | 1.117233e-04 | 9.183636e-05 | 3.053683e-03 | 9.591628e-05 |
| 1.6 | 9.786462e-05 | 1.093217e-04 | 3.458013e-03 | 1.073409e-04 |
| 2.0 | 3.334498e-05 | 1.119016e-04 | 1.725840e-03 | 1.204819e-04 |

各例 `results_summary.csv` 同时给出 u1、u2、总解的 MAE、RMSE、最大绝对误差、相对 L2，以及 t=1 的相同指标。相对 L2 以比值记录，不是百分数。初始时刻、局部与观察窗口边缘的误差见各例图片和 README；全局平均误差较小并不保证每个局部的相对误差都小。

以例 4、α=1.6 为例，下面同时展示两支和总解的参考值、预测值与绝对误差：

![Example 4 的分支和总解比较](example4/images/alpha_1_6/branch_comparison.png)
