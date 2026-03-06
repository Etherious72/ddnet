# DDNet 项目运行手册

本手册基于当前仓库代码整理，覆盖环境准备、训练、测试、自动多模型对比与常见问题。

## 1. 环境准备

### 1.1 Python 环境

建议使用你当前已验证可运行的 Conda 环境：`dataset-profiler`。

示例：

```bash
"D:\ProgramData\miniconda3\Scripts\conda.exe" run -n dataset-profiler python --version
```

### 1.2 PyTorch GPU 检查

```bash
"D:\ProgramData\miniconda3\Scripts\conda.exe" run -n dataset-profiler python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

输出中 `torch.cuda.is_available()` 为 `True` 表示可用。

## 2. 路径与数据

### 2.1 主路径设置

在 `path_config.py` 修改：

- `main_dir`：项目根目录（示例已是 `D:/gyq/coding/ddnet/`）

程序会自动拼出：

- `data/<dataset_name>/`
- `models/<dataset_name>Model/`
- `results/<dataset_name>Results/`

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

```bash
python model_test.py
```

默认行为受 `model_test.py` 末尾 `__main__` 区域控制：

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

### 6.2 运行命令

```bash
python model_test.py --compare-config compare_models.json
python model_test.py --compare-config compare_models.single.json
python model_test.py --compare-config compare_models.batch.json
```

### 6.3 对比输出

自动生成：

- `[CompareSingle]<dataset>_<timestamp>.csv`
- `[CompareBatch]<dataset>_<timestamp>.csv`

位置：`results/<dataset_name>Results/`

表头包含：

- `rank, status, alias, model_type, model_path`
- `mse_mean, mae_mean, uqi_mean, lpips_mean`
- `elapsed_seconds, per_sample_seconds, error`

默认按 `sort_by`（通常 `mae_mean`）排序。

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
set KMP_DUPLICATE_LIB_OK=TRUE && python model_test.py --compare-config compare_models.batch.json
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

```bash
# 单模型测试（按 model_test.py 默认入口）
python model_test.py

# 自动对比：Single+Batch
python model_test.py --compare-config compare_models.json

# 自动对比：只跑 Single
python model_test.py --compare-config compare_models.single.json

# 自动对比：只跑 Batch
python model_test.py --compare-config compare_models.batch.json
```
