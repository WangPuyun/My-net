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
    model = CAOutside.net
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
    total_loss = 0.0
    total_samples = 0
    random_move = RandomMove()
    images = torch.zeros([1, 3, 1024, 1224]).cuda(local_rank)
    with torch.no_grad():
        for i, sample_raw in enumerate(test_loader):
            images.zero_()
            mae = float('nan')
            while math.isnan(mae):
                for j in range(32):
                    sample = random_move(sample_raw)
                    distant = sample['distant']
                    original = [-distant[0], -distant[1]]
                    sample = unfold_image(sample)
                    mask = sample['mask'].cuda(local_rank)
                    inputs = sample['input']
                    # inputs = inputs[:,0:4,:,:]
                    inputs = inputs.cuda(local_rank, non_blocking=True)
                    outputs, *_ = model(inputs)
                    outputs *= mask
                    outputs = concat_image(outputs)
                    pad = nn.ZeroPad2d(padding=(0, 200, 0, 0))
                    outputs = pad(outputs)
                    outputs = affine(outputs, 0, original, 1, [0.0])
                    images += outputs
                images = torch.div(images, 32)
                images = normalize(images, dim=1)
                ground_truths = sample_raw['ground_truth']
                # ground_truths = ground_truths[:, 0:4, :, :]
                ground_truths = ground_truths.cuda(local_rank, non_blocking=True)
                mask = sample_raw['mask'].cuda(local_rank, non_blocking=True)
                mask = mask.unsqueeze(1)
                ground_truths = ground_truths * mask
                ground_truths = ground_truths.float() / 255.0
                M = torch.sum(torch.sum(mask, dim=1))
                
                m = torch.sum(torch.sum(criterion(images, ground_truths))) / M
                # torch.distributed.barrier()# 同步进程
                angle = torch.acos(m)
                mae = angle * 180 / pi
                print(mae)
            filename = sample_raw['filename']
            save_image(images, './results_sfp/{}_{}.bmp'.format(filename[0], mae))
            
            

if __name__ == "__main__":
    main()
