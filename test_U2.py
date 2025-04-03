# test.py
import argparse
import os
from locale import normalize
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
from Datasets_U2 import RandomMove, unfold_image, concat_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Network Testing')
    # 设置加载的模型名称 (checkpoint 文件)
    parser.add_argument("--model_name", type=str, default=None,
                        help="加载已训练的模型文件路径，例如 'xxx.pth'")
    # 测试批次大小
    parser.add_argument("--test_batch_size", type=int, default=1, help="测试批次大小")
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
    model = UPIE.MLFE()
    model = model.cuda(args.local_rank)

    # 加载指定的checkpoint
    checkpoint = torch.load('./pt/200.pth')
    model.load_state_dict(checkpoint['model'])

    # 同步BN、防止多卡测试时因BN计算导致结果不一致
    model = config.wrap_model_distributed(model, local_rank=local_rank)
    model.eval()  # 进入测试模式

    # 创建损失函数（若需要在测试环节计算 loss，可保留）
    criterion = nn.L1Loss().cuda(local_rank)

    # 构造测试集的数据加载器
    # 可以复用与训练集/验证集类似的创建函数，也可写单独的测试数据加载逻辑
    # 假设 config.create_dataloaders() 里可以根据标志返回测试集加载器
    # 或者自行实现一个 config.create_test_loader(args)。这里只作示例：
    test_loader, _ = config.test_dataloaders(args)

    # 若需要加速
    cudnn.benchmark = False

    # 在每张卡上循环测试 (或者仅主进程执行)
    # 此处写一个简单的测试流程，可根据需要做更详细的结果保存或指标计算
    total_loss = 0.0
    total_samples = 0
    random_move = RandomMove()
    images = torch.zeros([1, 4, 1024, 1224]).cuda(local_rank)
    with torch.no_grad():
        for i, sample_raw in enumerate(test_loader):
            images.zero_()
            for j in range(32):
                sample = random_move(sample_raw)
                distant = sample['distant']
                original = [-distant[0], -distant[1]]
                sample = unfold_image(sample)
                mask = sample['mask'].cuda(local_rank)
                inputs = sample['input']
                inputs = inputs[:,0:4,:,:]
                inputs = inputs.cuda(local_rank, non_blocking=True)
                outputs = model(inputs)
                _, _, h, w = outputs.size()
                outputs *= mask
                outputs = concat_image(outputs)
                pad = nn.ZeroPad2d(padding=(0, 200, 0, 0))
                outputs = pad(outputs)
                outputs = affine(outputs, 0, original, 1, [0.0])
                images += outputs
            images = torch.div(images, 32)

            ground_truths = sample_raw['CleanWater']
            ground_truths = ground_truths[:, 0:4, :, :]
            ground_truths = ground_truths.cuda(local_rank, non_blocking=True)
            mask = sample_raw['mask'].cuda(local_rank, non_blocking=True)
            mask = mask.unsqueeze(1)
            ground_truths = ground_truths * mask

            loss = criterion(images, ground_truths)
            total_loss += loss.item() * ground_truths.size(0)
            total_samples += ground_truths.size(0)

            filename = os.path.splitext(os.path.basename(sample['mat_path'][0]))[0]
            for channel in range(4):
                image = images[:, channel:channel + 1, :, :]
                ground_truth = ground_truths[:, channel:channel + 1, :, :]
                # 将 output 与 groundtruth 水平拼接
                # combined = torch.cat([image, ground_truth], dim=3)  # dim=3 表示沿宽度拼接

                torchvision.utils.save_image(image, f'./results/{filename}_偏振通道{channel}.png')
                # torchvision.utils.save_image(ground_truth, f'./GT/{filename}_{channel}.png')

    # 汇总结果 (若使用分布式，需要手动做reduce或gather)
    avg_loss = total_loss / (total_samples + 1e-5)
    print(f"[Rank {local_rank}] 测试集平均Loss: {avg_loss}")

if __name__ == "__main__":
    main()
