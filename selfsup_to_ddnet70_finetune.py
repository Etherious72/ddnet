"""将自监督预训练权重用于 DDNet 系列微调（独立脚本）。

用途：
1) 独立于既有入口，通过文件内参数控制微调。
2) 在启动前检查预训练权重与目标网络是否完全兼容。
3) 不兼容时可按配置回退为从头训练（避免流程中断）。

说明：
- 本脚本复用 model_train.py 的课程学习训练流程。
- 该脚本只负责“选择是否加载预训练权重 + 回退策略”。
"""

import os

import torch

from model_train import curriculum_learning_training
from net.DDNet import DDNetModel, SDNetModel
from net.DDNet70 import DDNet70Model, SDNet70Model
from net.FCNVMB import FCNVMB
from net.InversionNet import InversionNet
from param_config import (
    classes,
    data_dim,
    dataset_name,
    device_mode,
    display_step,
    inchannels,
    learning_rate,
    loss_weight,
    model_dim,
    model_type,
)

# 手动配置区（不使用命令行参数）
MANUAL_CONFIG = {
    "model_type": model_type,
    "pretrained_path": "",  # 例如：models_pretrain/CurveFaultAModel/xxx.pth 或 xxx.pkl
    "finetune_lr_scale": 0.1,
    "strict_pretrained_match": False,   # True: 不兼容直接报错
    "allow_scratch_fallback": True,     # True: 不兼容时从头训练
}


# 函数作用：按 model_type 创建未加载权重的网络骨架。
def build_model_skeleton(model_type):
    if model_type == "DDNet":
        return DDNetModel(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    if model_type == "DDNet70":
        return DDNet70Model(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    if model_type == "SDNet":
        return SDNetModel(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    if model_type == "SDNet70":
        return SDNet70Model(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    if model_type == "InversionNet":
        return InversionNet()
    if model_type == "FCNVMB":
        return FCNVMB(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    raise ValueError("未支持的 model_type: {}".format(model_type))


# 函数作用：标准化 checkpoint 的键名（去掉 DataParallel 前缀 module.）。
def normalize_state_dict_keys(state):
    out = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


# 函数作用：检查 checkpoint 与目标模型是否“完全兼容可直接 load_state_dict”。
def check_full_compatibility(model_type, pretrained_path):
    if not os.path.exists(pretrained_path):
        return False, "pretrained_file_not_found"

    state = torch.load(pretrained_path, map_location="cpu")
    if not isinstance(state, dict):
        return False, "checkpoint_not_state_dict"

    state = normalize_state_dict_keys(state)
    model = build_model_skeleton(model_type)
    model_state = model.state_dict()

    if len(state) != len(model_state):
        return False, "key_count_mismatch"

    for k, v in model_state.items():
        if k not in state:
            return False, "missing_key:{}".format(k)
        if tuple(v.shape) != tuple(state[k].shape):
            return False, "shape_mismatch:{}".format(k)

    return True, "ok"


# 函数作用：根据兼容性与策略决定最终使用的初始化权重路径。
def resolve_init_checkpoint(cfg):
    path = cfg["pretrained_path"].strip()
    if not path:
        print("[Finetune] 未指定 pretrained_path，将从头训练。")
        return ""

    ok, reason = check_full_compatibility(cfg["model_type"], path)
    if ok:
        print("[Finetune] 预训练权重兼容，加载: {}".format(path))
        return path

    msg = "[Finetune] 预训练权重不兼容: {} ({})".format(path, reason)
    if cfg["strict_pretrained_match"]:
        raise RuntimeError(msg)

    print(msg)
    if cfg["allow_scratch_fallback"]:
        print("[Finetune] 回退策略启用：改为从头训练。")
        return ""

    raise RuntimeError("预训练权重不兼容且未允许回退。")


# 函数作用：脚本入口，执行微调训练。
def main():
    cfg = MANUAL_CONFIG
    if cfg["finetune_lr_scale"] <= 0:
        raise ValueError("finetune_lr_scale 必须 > 0")

    init_ckpt = resolve_init_checkpoint(cfg)
    curriculum_learning_training(
        model_type=cfg["model_type"],
        init_model_src=init_ckpt,
        finetune_lr_scale=cfg["finetune_lr_scale"],
    )


if __name__ == "__main__":
    main()
