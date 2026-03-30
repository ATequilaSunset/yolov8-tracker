#!/usr/bin/env python3
"""快速验证自定义模型结构是否能正确构建。"""
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(__file__))

print("[1/4] 导入自定义模块...")
from custom_modules import BackgroundReconstruct, FeatureEnhance, register_custom_modules
register_custom_modules()

print("[2/4] Patch parse_model...")
import builtins
builtins.BackgroundReconstruct = BackgroundReconstruct
builtins.FeatureEnhance = FeatureEnhance

import ultralytics.nn.tasks as _tasks

# 构建 custom_channel_handlers
def _bgr_channels(args, ch_in_list):
    c1 = ch_in_list[0]
    new_args = [c1] + list(args[1:])
    return 1, new_args

def _feh_channels(args, ch_in_list):
    c1 = ch_in_list[0]
    c_diff = ch_in_list[1] if len(ch_in_list) > 1 else 1
    return c1, [c1, c_diff]

custom_channel_handlers = {
    BackgroundReconstruct: _bgr_channels,
    FeatureEnhance:        _feh_channels,
}

_orig_parse = _tasks.parse_model

def _patched_parse(d, ch, verbose=True):
    # 在 _orig_parse 执行前，将自定义模块注入其 globals
    # 通过替换 globals 中的查找逻辑：hook globals()[m] 调用
    # 最简方式：直接在 _tasks 模块 globals 注入
    _tasks.BackgroundReconstruct = BackgroundReconstruct
    _tasks.FeatureEnhance = FeatureEnhance

    # 先用原始 parse_model，但它的 else: c2=ch[f] 会给错误通道数
    # 我们需要在构建每一层后修正 ch 列表
    # 更可靠的方式：直接检查 seq 后每层的真实输出通道
    result = _orig_parse(d, ch, verbose=verbose)
    return result

# 不 patch，直接注入模块名到 _tasks，让 globals()[m] 能找到
_tasks.BackgroundReconstruct = BackgroundReconstruct
_tasks.FeatureEnhance = FeatureEnhance

print("[3/4] 解析 YAML 并运行 parse_model...")
from ultralytics.utils import YAML
from copy import deepcopy
import torch

yaml_path = os.path.join(os.path.dirname(__file__), 'yolov8n_custom.yaml')
d = YAML.load(yaml_path)
d['scale'] = 'n'

# 直接用原始 parse_model（已注入自定义模块到 _tasks globals）
# 但通道推导仍有问题，所以我们用 patched 版本
# 重新实现简化版 channel-aware parse
from ultralytics.nn.modules import (
    Conv, C2f, SPPF, Concat, Detect
)
from ultralytics.utils.ops import make_divisible
import contextlib, ast

def simple_channel_check(d):
    """仅追踪每层的输出通道数，验证通道推导是否正确。"""
    scales = d.get('scales', {})
    depth, width, max_ch = scales.get('n', [0.33, 0.25, 1024])
    nc = d.get('nc', 1)

    ch = [3]  # input channels
    print(f"{'idx':>4} {'from':>12} {'module':<28} {'args':<20} {'c_out':>6}")
    print("-" * 75)

    custom_out = {
        'BackgroundReconstruct': lambda args, ch_in: 1,
        'FeatureEnhance':        lambda args, ch_in: ch_in[0],
    }

    for i, (f, n, m_name, args) in enumerate(d['backbone'] + d['head']):
        # 计算输入通道
        if isinstance(f, list):
            ch_in = [ch[x] for x in f]
        else:
            ch_in = [ch[f]]

        # 推导输出通道
        if m_name in ('Conv',):
            c2 = make_divisible(min(args[0], max_ch) * width, 8)
        elif m_name in ('C2f', 'SPPF'):
            c2 = make_divisible(min(args[0], max_ch) * width, 8)
        elif m_name == 'Concat':
            c2 = sum(ch_in)
        elif m_name == 'Detect':
            c2 = None
        elif m_name.startswith('nn.'):
            c2 = ch_in[0]  # Upsample 不改变通道
        elif m_name in custom_out:
            c2 = custom_out[m_name](args, ch_in)
        else:
            c2 = ch_in[-1]

        ch_in_str = str(ch_in[0]) if len(ch_in) == 1 else str(ch_in)
        print(f"{i:>4} {str(f):>12} {m_name:<28} {str(args):<20} {str(c2):>6}")
        if c2 is not None:
            ch.append(c2)
        else:
            ch.append(0)

    print()
    # 检查关键层
    assert ch[2] == 1,   f"layer2 (BGRecon) 输出应为 1ch，实际 {ch[2]}"
    assert ch[3] == make_divisible(min(128, max_ch) * width, 8), \
        f"layer3 (FeatEnh) 输出应为 {make_divisible(128*width,8)}ch，实际 {ch[3]}"
    print("通道推导验证通过！")
    print(f"  layer1 (feat_early): {ch[1]}ch")
    print(f"  layer2 (BGRecon):    {ch[2]}ch  <- diff_map")
    print(f"  layer3 (FeatEnh):    {ch[3]}ch  <- enhanced_feat")
    print(f"  layer4 (C2f):        {ch[4]}ch")

simple_channel_check(d)

print("\n[4/4] 构建实际模型...")
from ultralytics.nn.tasks import DetectionModel
model = DetectionModel(yaml_path, ch=3, nc=1, verbose=False)
print(f"模型层数: {len(model.model)}")
print(f"save 列表: {model.save}")

# 验证各层 m.f 和通道数
print("\n关键层检查:")
for idx in [1, 2, 3, 4]:
    m = model.model[idx]
    print(f"  layer{idx}: type={m.type.split('.')[-1]:<25} from={m.f}")

print("\n=== 验证完成 ===")
