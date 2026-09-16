# 复现与佐证材料

- `support/`：主程序依赖的辅助模块。
- `alpha_*/`：各阶数的权重（`.pt`）、数据与预测数组（`.npz`）、训练历史（`.csv`）、配置、运行时间和核对指标（`.json`）。
- `alpha_*/training_image/`：训练 loss 及训练过程曲线。
- `verification/`：低 α 参考积分的独立核对记录。
- [低 α 参考积分说明](smallalpha_compatibility.md)：适用参数与积分精度检查。

主要结果图在上一级 [`images/`](../images/)，最终误差在上一级 [`results_summary.csv`](../results_summary.csv)。加载现有权重复现时，需要保留 `support/` 与所选 α 的数据目录。
