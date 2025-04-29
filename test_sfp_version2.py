# test.py
import argparse
import math
import os
from torch.nn.functional import normalize
from torchvision.transforms.functional import affine
import DCC
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
from torch.backends import cudnn
import UPIE
import Unet
import config_U2 as config
from math import pi
from Datasets_U2 import RandomMove, unfold_image, concat_image
from torchvision.utils import save_image
from AttentionU2Net import CAOutside
from AttentionU2Net import U2Net_with_enhance_img
from utils_window import PATCH, OVERLAP, STRIDE, hann2d
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3,7'

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Network Testing')
    # 设置加载的模型名称 (checkpoint 文件)
    parser.add_argument("--model_name", type=str, default=None,
                        help="加载已训练的模型文件路径，例如 'xxx.pth'")
    # 测试批次大小
    parser.add_argument("--test_batch_size", type=int, default=3, help="测试批次大小")
    # # 测试数据所在的目录或其他参数 (视情况添加)
    # parser.add_argument("--test_data_dir", type=str, default="./test_data",
    #                     help="测试数据的路径")
    # 其他必要的参数按需添加
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # 获取可用GPU数量
    args.nprocs = torch.cuda.device_count()
    # 多卡测试可以使用与训练相同的多进程方式
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    # 初始化分布式环境
    args.local_rank = local_rank
    config.init_distributed(local_rank=args.local_rank, nprocs=args.nprocs)

    # 构建模型与优化器（测试时通常无需使用optimizer，但此处为了演示可共用）
    model = U2Net_with_enhance_img.net
    model = model.cuda(args.local_rank)

    # 加载指定的checkpoint
    checkpoint = torch.load('./pt/1000.pth')
    model.load_state_dict(checkpoint['model'])

    # 同步BN、防止多卡测试时因BN计算导致结果不一致
    model = config.wrap_model_distributed(model, local_rank=local_rank)
    model.eval()  # 进入测试模式

    # 创建损失函数（若需要在测试环节计算 loss，可保留）
    criterion = nn.CosineSimilarity().cuda(args.local_rank)

    # 构造测试集的数据加载器
    # 可以复用与训练集/验证集类似的创建函数，也可写单独的测试数据加载逻辑
    # 假设 config.create_dataloaders() 里可以根据标志返回测试集加载器
    # 或者自行实现一个 config.create_test_loader(args)。这里只作示例：
    test_loader, _ = config.test_dataloaders(args)

    # 若需要加速
    # cudnn.benchmark = True

    # 在每张卡上循环测试 (或者仅主进程执行)
    # 此处写一个简单的测试流程，可根据需要做更详细的结果保存或指标计算
    device = torch.device(f'cuda:{local_rank}')
    window = hann2d(PATCH, device).unsqueeze(0).unsqueeze(0)

    total_loss, total_samples = 0., 0
    
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            inputs   = sample['input'].cuda(device)
            gt       = sample['ground_truth'].float().cuda(device) / 255.
            mask     = sample['mask'].unsqueeze(1).cuda(device)
            gt *= mask

            H, W     = inputs.shape[2:]
            # -----------准备空容器-----------
            out_sum = torch.zeros(1, 3, H, W, device=device)
            w_sum   = torch.zeros(1, 1, H, W, device=device)

            #  -----------滑窗推理-----------
            for y in range(0, H - PATCH + 1, STRIDE):
                for x in range(0, W - PATCH + 1, STRIDE):
                    patch = inputs[..., y:y+PATCH, x:x+PATCH]
                    pred, *_ = model(patch)
                    pred = pred * window
                    out_sum[..., y:y+PATCH, x:x+PATCH] += pred
                    w_sum[...,  y:y+PATCH, x:x+PATCH] += window

            # -----------归一化得到整幅结果-----------
            full_pred = out_sum / w_sum.clamp_min(1e-6)
            full_pred = torch.nn.functional.normalize(full_pred, dim=1)
            full_pred *= mask

            # -----------计算角度 MAE-----------
            M  = torch.sum(mask)
            m  = torch.sum(criterion(full_pred , gt )) / M
            mae = torch.acos(m.clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / pi

            total_loss   += mae.item()
            total_samples += 1

            filename = sample['filename']
            save_image(full_pred, './results_sfp/{}_{}.bmp'.format(filename[0], mae))
            
            

if __name__ == "__main__":
    main()
