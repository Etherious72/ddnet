# 预训练说明

本项目已新增独立预训练入口（框架版），用于在不影响现有训练/测试脚本的前提下，单独管理预训练流程。

## 1. 入口与目录

- 入口脚本：`pretrain_entry.py`
- 预训练模型目录：`models_pretrain/<dataset_name>Model/`
- 当前默认数据集（来自 `param_config.py`）：`CurveFaultA`
- 当前默认目录示例：`models_pretrain/CurveFaultAModel/`

说明：预训练目录与实验模型目录分离。

- 预训练目录：`models_pretrain/...`
- 正式实验目录：`models/...`

## 2. 当前状态（框架版）

`pretrain_entry.py` 当前只提供流程骨架，不执行真实模型训练。它会：

1. 解析参数
2. 创建/确认预训练目录
3. 打印预训练流程信息
4. 可选写入占位文件（用于验证流程）

## 3. 使用方式

### 3.1 仅检查流程（不写文件）

```bash
python pretrain_entry.py --dry-run
```

### 3.2 生成占位结果文件

```bash
python pretrain_entry.py --epochs 5 --save-name demo_pretrain_stub
```

执行后会在预训练目录下生成：

- `models_pretrain/CurveFaultAModel/demo_pretrain_stub.txt`

## 4. 参数说明

- `--dataset`：数据集名称，默认取 `param_config.py` 中的 `dataset_name`
- `--model-type`：模型类型，默认取 `param_config.py` 中的 `model_type`
- `--epochs`：预训练轮数（当前为占位参数）
- `--save-name`：占位文件名（不含后缀）
- `--dry-run`：仅打印流程，不写入占位文件

## 5. 与现有脚本关系

本次新增不修改以下脚本逻辑：

- `model_train.py`
- `model_test.py`
- `inversionnet_train.py`
- `fcnvmb_train.py`

也就是说，现有训练和测试流程保持原样可用。

## 6. 后续接入真实预训练时建议

当你需要把框架升级为真实预训练流程时，建议保持以下原则：

- 只向 `models_pretrain/...` 写预训练权重
- 正式实验继续写到 `models/...`
- 在正式训练入口通过参数显式指定是否加载预训练权重
