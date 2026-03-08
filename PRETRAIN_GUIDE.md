# 预训练说明

本项目当前已具备可执行的独立预训练入口，用于在不影响既有训练/测试脚本的前提下，完成“源域预训练 -> 目标域微调 -> 对照评测”的流程。

## 1. 入口与目录

- 预训练入口：`pretrain_entry.py`
- 预训练权重目录：`models_pretrain/<target_dataset>Model/`
- 正式实验权重目录：`models/<dataset_name>Model/`

目录分离原则：

- 预训练只写 `models_pretrain/...`
- 正式训练只写 `models/...`

## 2. 当前能力（已实现）

`pretrain_entry.py` 已支持：

1. 多源域顺序监督预训练（同一模型权重连续继承）
2. 源域数据有效性检查
3. 源域异常跳过（非严格模式）
4. 预训练初始化失败时可回退从头训练
5. 训练出现 NaN 时单次降学习率重试
6. 输出 checkpoint 与 JSON 汇总

`model_train.py` 已支持：

1. 通过 `--load-pretrained` 显式加载预训练权重微调
2. 通过 `--finetune-lr-scale` 控制微调学习率缩放

## 3. 预训练运行方式（手动改文件参数）

不再通过命令行传参。请先修改 `pretrain_entry.py` 顶部的 `PRETRAIN_MANUAL_CONFIG`，再直接运行：

```bash
python pretrain_entry.py
```

常改字段：

- `source_datasets`：源域列表（逗号分隔）
- `target_dataset`：目标域标签
- `pretrain_epochs`：每个源域训练轮数
- `batch_size`、`lr`
- `pretrained_path`：可选初始权重
- `fallback_init_scratch`、`strict_source_check`、`nan_retry_scale`
- `dry_run`：只做检查不训练

## 4. 参数位置（`pretrain_entry.py`）

参数集中在 `PRETRAIN_MANUAL_CONFIG` 字典中，直接在文件里改值即可。

## 5. 输出说明

预训练结束后会生成：

- checkpoint：`models_pretrain/<target_dataset>Model/<name>.pkl`
- summary：`models_pretrain/<target_dataset>Model/<name>_summary.json`

summary 包含：

- 请求的源域列表
- 跳过的源域及原因
- 成功训练的源域及状态（`ok`/`ok_after_retry`）
- 初始化方式（pretrained/scratch/scratch_fallback）

## 6. 微调运行方式（`model_train.py`）

不再通过命令行传参。请先修改 `model_train.py` 底部 `TRAIN_MANUAL_CONFIG`，再运行：

```bash
python model_train.py
```

常改字段：

- `model_type`
- `load_pretrained`（留空表示从头训练）
- `finetune_lr_scale`（会缩放 `param_config.py` 的 `learning_rate`）

## 7. 对照评测（M4）

已提供预训练对照配置模板：

- `compare_models.pretrain.json`
- `compare_models.pretrain.single.json`
- `compare_models.pretrain.batch.json`

先替换其中占位模型路径；
然后在 `model_test.py` 底部 `TEST_MANUAL_CONFIG["compare_config"]`
中设置所用配置文件（如 `compare_models.pretrain.json`），再执行：

```bash
python model_test.py
```

默认按 `mse_mean` 排序。
