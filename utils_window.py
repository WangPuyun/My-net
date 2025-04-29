# utils_window.py
import torch
from math import pi

PATCH   = 256          # patch 尺寸
OVERLAP = 64           # 建议 32~64，自己微调
STRIDE  = PATCH - OVERLAP

def hann2d(size: int, device):
    """生成 2-D Hann window，范围 0~1"""
    w = torch.hann_window(size, periodic=False, device=device)   # (size,)
    window2d = torch.outer(w, w)                                 # (size,size)
    return window2d / window2d.max()                             # 归一化
