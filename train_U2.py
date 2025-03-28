import argparse
import os
import torch.multiprocessing as mp
import config_U2 as config
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
import torch.nn as nn
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'

parser = argparse.ArgumentParser(description='PyTorch Network Training')
parser.add_argument("--model_name", type=str, default=None, help="是否加载模型继续训练，重头开始训练 defaule=None, 继续训练defaule设置为'/**.pth'")
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument("--train_batch_size", type=int, default=12, help="分布训练批次大小")
parser.add_argument("--val_batch_size", type=int, default=4, help="分布验证批次大小")
parser.add_argument('--event_dir', default="./runs", help='tensorboard事件文件的地址')
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument('--warmup_epochs', type=int, default=50, help='学习率预热epoch数')
parser.add_argument('--checkpoints_dir', default="./pt", help='模型检查点文件的路径(以继续培训)')
args = parser.parse_args()

train_loss_list = []  # 只在主进程维护一个 loss_list
val_loss_list = []
lr_list = []

def main():
    args.nprocs = torch.cuda.device_count()

    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))  # 单机多卡

def main_worker(local_rank, nprocs,args):
    global train_loss_list, val_loss_list, lr_list
    # 1.初始化分布式训练环境
    args.local_rank = local_rank
    config.init_distributed(local_rank=args.local_rank, nprocs=args.nprocs)

    # 2.创建模型与优化器
    model, optimizer,scheduler = config.create_model_and_optimizer(args)

    # 3.可选：加载已有checkpoint
    start_epoch = 0
    if args.model_name:
        model, optimizer, start_epoch = config.load_checkpoint(model, optimizer,checkpoints_dir=args.checkpoints_dir,
            model_name=args.model_name, local_rank=args.local_rank )

    # 4.同步BN并封装DDP
    model = config.wrap_model_distributed(model, local_rank=args.local_rank)

    # 5.创建损失函数
    criterion = nn.CosineSimilarity().cuda(args.local_rank)

    # 6.创建数据加载器
    train_loader, val_loader, train_sampler, val_sampler = config.create_dataloaders(args)

    # 7.初始化cudnn与TensorBoard
    cudnn.benchmark = False # 设置为True追求速度，会消耗大量显存；False训练速度慢，占用显存少
    writer = SummaryWriter(args.event_dir)  # 创建事件文件

    # 8.训练与验证循环
    epoch_iter = tqdm(range(start_epoch, args.epochs), desc="Epoch Progress")

    for epoch in epoch_iter:
        # 为了分布式训练的数据随机性一致，需要在每个epoch设置sampler的epoch
        train_sampler.set_epoch(epoch)  # 将在for循环训练时候的epoch传入,达到乱序效果
        val_sampler.set_epoch(epoch)

        # # 学习率预热逻辑
        # if epoch < args.warmup_epochs:
        #     warmup_lr = args.lr * (epoch + 1) / args.warmup_epochs
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = warmup_lr
        # else:
        #     scheduler.step(val_loss_list[-1])

        model,train_loss_list = config.train_sfp(train_loader, model, criterion, optimizer, epoch, writer, args.local_rank, args, train_loss_list)
        val_loss_list = config.val_sfp(val_loader, model, writer, epoch, args.local_rank, args, criterion, val_loss_list)
        torch.distributed.barrier()  # 等待所有进程计算完毕
        # scheduler.step(val_loss_list[-1])  # 更新学习率
        # 记录学习率（仅在主进程）
        if args.local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'current_lr:{current_lr}')
            lr_list.append(current_lr)

        # 在进度条中显示最新的loss
        epoch_iter.set_postfix(train_loss=train_loss_list[-1], val_loss=val_loss_list[-1])
    # 9.周期性保存模型
        if (epoch + 1) % 100 == 0 and args.local_rank == 0:
            config.save_checkpoint(model, optimizer,epoch+1, checkpoints_dir=args.checkpoints_dir)

    if args.local_rank == 0:
        writer.close()
        config.draw_curve(train_loss_list, 'train loss')
        config.draw_curve(val_loss_list, 'val loss')
        config.draw_curve(lr_list, 'learning rate')
        config.draw_two_curve(train_loss_list, val_loss_list, 'train loss', 'val loss')

if __name__ == "__main__":
    main()