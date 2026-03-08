# DDNet 项目运行手册

本手册基于当前仓库代码整理，覆盖环境准备、训练、测试、自动多模型对比与常见问题。

## 1. 环境准备

### 1.1 Python 环境

## 2. 路径与数据

### 2.1 主路径设置

在 `path_config.py` 修改：

- `main_dir`：项目根目录（示例已是 `D:/gyq/coding/ddnet/`）

### 2.2 数据目录格式

以 `CurveFaultA` 为例：

- `data/CurveFaultA/train_data/seismic/seismic*.npy`
- `data/CurveFaultA/train_data/vmodel/vmodel*.npy`
- `data/CurveFaultA/test_data/seismic/seismic*.npy`
- `data/CurveFaultA/test_data/vmodel/vmodel*.npy`

## 3. 全局参数配置

在 `param_config.py` 修改：

- `dataset_name`
- `model_type`
- `device_mode`（`auto|cpu|gpu`）
- `train_size` / `test_size`
- `train_batch_size` / `test_batch_size`
- 学习率与各阶段 epoch

## 4. 训练

### 4.1 DDNet 主训练（课程学习）

先在 `model_train.py` 底部修改 `TRAIN_MANUAL_CONFIG`（不再命令行传参）：

- `model_type`
- `load_pretrained`
- `finetune_lr_scale`

```bash
python model_train.py
```

### 4.2 InversionNet

```bash
python inversionnet_train.py
```

### 4.3 FCNVMB

```bash
python fcnvmb_train.py
```

模型输出在：`models/<dataset_name>Model/*.pkl`

## 5. 测试（单模型）

先在 `model_test.py` 底部修改 `TEST_MANUAL_CONFIG`（不再命令行传参）：

- `compare_config`

```bash
python model_test.py
```

默认受 `model_test.py` 末尾 `__main__` 区域控制：

- `batch_of_single = 1`：跑 `batch_test`（全测试集）
- `batch_of_single != 1`：跑 `single_test`（单样本）

### 5.1 预览图开关（single_test）

`single_test(...)` 支持：

- `save_preview=1/0`：保存预览图
- `show_preview=1/0`：是否弹窗

输出目录：`results/<dataset_name>Results/previews/`

## 6. 自动多模型对比（已支持）

`model_test.py` 已支持读取 JSON 配置后自动批量对比。

### 6.1 配置文件

- `compare_models.json`：默认 Single + Batch
- `compare_models.single.json`：只跑 Single
- `compare_models.batch.json`：只跑 Batch

### 6.2 运行

### 6.3 对比输出

自动生成：

- `[CompareSingle]<dataset>_<timestamp>.csv`
- `[CompareBatch]<dataset>_<timestamp>.csv`

位置：`results/<dataset_name>Results/`

表头包含：

- `rank, status, alias, model_type, model_path`
- `mse_mean, mae_mean, uqi_mean, lpips_mean`
- `elapsed_seconds, per_sample_seconds, error`

默认按 `sort_by`（默认 `mae_mean`）排序。

## 7. 推荐运行流程

1. 在 `param_config.py` 选定数据集与规模（先小规模冒烟，再全量）。
2. 训练得到多个 `.pkl` 模型。
3. 在 `compare_models*.json` 填好模型清单。
4. 先跑 `compare_models.single.json` 快速看可视化与粗指标。
5. 再跑 `compare_models.batch.json` 获取最终公平对比结果。

## 8. 常见问题

### 8.1 OpenMP 冲突（Windows 常见）

报错关键词：`libiomp5md.dll already initialized`。

可临时用：

```bash
set KMP_DUPLICATE_LIB_OK=TRUE && python model_test.py
```

或在 Python 入口前设置同名环境变量。

### 8.2 模型文件不存在

自动对比不会整体中断，会在结果里标记：

- `status=missing`
- `error=model_file_not_found`

### 8.3 模型类型写错

自动对比结果会标记：

- `status=invalid_model_type`

合法类型：`DDNet, DDNet70, InversionNet, FCNVMB, SDNet, SDNet70`

## 9. 常用命令速查

## 10. 预训练 + 微调 + 对照评测（新增）

### 10.1 预训练入口（`pretrain_entry.py`）

当前预训练入口支持多源域顺序训练、异常跳过与总结文件输出。

先在 `pretrain_entry.py` 顶部修改 `PRETRAIN_MANUAL_CONFIG`（不再命令行传参）。

运行：

```bash
python pretrain_entry.py
```

关键配置字段（在字典中修改）：

- `source_datasets`：源域列表（逗号分隔）
- `target_dataset`：目标域标签（用于输出目录）
- `pretrain_epochs`：每个源域训练轮数
- `pretrained_path`：可选初始化权重
- `fallback_init_scratch`：初始化失败时回退从头训练
- `strict_source_check`：任一源域异常即失败（默认关闭）
- `nan_retry_scale`：出现 NaN 后单次重试学习率缩放（默认 `0.1`）

输出：

- 预训练权重：`models_pretrain/<target_dataset>Model/*.pkl`
- 过程总结：`models_pretrain/<target_dataset>Model/*_summary.json`

### 10.2 目标域微调入口（`model_train.py`）

`model_train.py` 已支持加载预训练权重与微调学习率缩放（通过文件内配置）。

先修改 `TRAIN_MANUAL_CONFIG`，再运行：

```bash
python model_train.py
```

关键配置字段：

- `load_pretrained`：预训练 `.pkl` 路径
- `finetune_lr_scale`：对 `param_config.py` 中学习率的缩放系数

### 10.3 预训练方案对照评测配置

新增配置文件：

- `compare_models.pretrain.json`（Single + Batch）
- `compare_models.pretrain.single.json`（仅 Single）
- `compare_models.pretrain.batch.json`（仅 Batch）

这些配置默认按 `mse_mean` 排序。请先把其中占位路径替换成真实模型路径：

- `REPLACE_WITH_BASELINE_MODEL.pkl`
- `REPLACE_WITH_FINETUNED_MODEL.pkl`

评测方式：

1) 在 `model_test.py` 底部 `TEST_MANUAL_CONFIG["compare_config"]` 中设置配置文件名
2) 直接运行 `model_test.py`

```bash
python model_test.py
```
