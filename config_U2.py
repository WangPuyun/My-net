import torch.distributed as dist
import torch
import UPIE
import Unet
import ResNet
import DCC
from torchvision.transforms.functional import affine
from torchvision.utils import save_image
from torch.nn.functional import normalize
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Datasets_U2 import MyDataset, RandomCrop, FixedCrop, RandomMove, unfold_image, concat_image, unfold_enhanced_image, RandomMovePad, concat_enhanced_image
from torch.utils.data import DataLoader
from AttentionU2Net import U2Net_with_enhance_img
from math import pi
import math
from utils_window import PATCH, OVERLAP, STRIDE, hann2d
def init_distributed(local_rank, nprocs, url='tcp://localhost:25484'):
    """
    初始化分布式训练环境。
    """
    dist.init_process_group(
        backend='nccl',  # 使用 nccl 后端
        init_method=url,  # 指定初始化通讯方式（ip+端口）
        world_size=nprocs,  # 总进程数
        rank=local_rank  # 当前进程 rank
    )
    # 设置当前 GPU
    torch.cuda.set_device(local_rank)


def create_model_and_optimizer(args):
    """
    训练时调用
    创建模型和优化器，返回(model, optimizer)。
    """
    model = Unet.U_Net(4,4)

    # 将模型移动到指定设备（本地 GPU）
    # print(args.local_rank)
    model = model.cuda(args.local_rank)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    return model, optimizer, scheduler


def load_checkpoint(model, optimizer, checkpoints_dir, model_name, local_rank):
    """
    从给定的 checkpoints_dir + model_name 路径中加载模型和优化器参数。
    返回 (model, optimizer, start_epoch)。
    """
    checkpoint_path = checkpoints_dir + model_name
    print("加载模型：", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{local_rank}')
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]

    return model, optimizer, start_epoch


def wrap_model_distributed(model, local_rank):
    """
    将模型的 BatchNorm 转换为 SyncBatchNorm 并包装为 DistributedDataParallel。
    """
    # 将模型中的 BatchNorm 层替换为 SyncBatchNorm
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 使用分布式并行包装
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    return model


def create_dataloaders(args):
    """
    创建训练/验证集的 Dataset & DataLoader，并返回 (train_loader, val_loader, train_sampler, val_sampler)
    """
    # 训练集
    train_set = MyDataset(
        csv_file='Underwater Dataset/train_list_withoutcleanwater.csv',
        root_dir='Underwater Dataset/Underwater Dataset',
        transform=RandomCrop()  # RandomCrop 是数据增强
    )

    # 验证集
    val_set = MyDataset(
        csv_file='Underwater Dataset/val_list_withoutcleanwater.csv',
        root_dir='Underwater Dataset/Underwater Dataset',
        transform=False  
    )

    # 分布式采样器
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False, drop_last=True)

    # 调整后的 batch_size（总的 batch_size / nprocs）
    train_batch_size = int(args.train_batch_size / args.nprocs)
    val_batch_size = int(args.val_batch_size / args.nprocs)

    # DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True
    )

    return train_loader, val_loader, train_sampler, val_sampler


def test_dataloaders(args):
    """
    创建测试集的 Dataset & DataLoader，并返回 (test_loader, test_sampler)
    """

    # 验证集
    test_set = MyDataset(
        csv_file='Underwater Dataset/test_list_withoutcleanwater.csv',
        root_dir='Underwater Dataset/Underwater Dataset',
        transform=False
    )

    # 分布式采样器
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False, drop_last=False)

    # 调整后的 batch_size（总的 batch_size / nprocs）
    test_batch_size = int(args.test_batch_size / args.nprocs)

    # DataLoader
    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=test_sampler,
        drop_last=True
    )

    return test_loader, test_sampler


def save_checkpoint(model, optimizer, epoch, checkpoints_dir):
    """
    保存 checkpoint，包括模型参数、优化器参数和当前 epoch。
    """
    checkpoint = {
        "model": model.module.state_dict(),  # DDP情形下，需要访问 model.module
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    save_path = f"{checkpoints_dir}/{epoch}.pth"
    torch.save(checkpoint, save_path)
    print(f"模型已保存到：{save_path}")


def train(train_loader, model, criterion, optimizer, epoch, writer, local_rank, args, train_loss_list):
    model.train()
    running_loss = 0
    total_samples = 0
    for i, sample in enumerate(train_loader):
        # 4通道分别是4个偏振态的图像，同时输入模型，旨在让模型学习到各个通道之间的相关性
        ground_truths = sample['CleanWater']
        ground_truths = ground_truths[:,0:4,:,:]
        ground_truths = ground_truths.cuda(local_rank, non_blocking=True)
        inputs = sample['input']
        inputs = inputs[:,0:4,:,:]
        inputs = inputs.cuda(local_rank, non_blocking=True)
        mask = sample['mask']
        mask = mask.unsqueeze(1)
        mask = mask.cuda(local_rank, non_blocking=True)
        mask = mask.expand_as(ground_truths)
        

        optimizer.zero_grad()  # 清除之前的梯度
        # 前向传播
        outputs = model(inputs)
        _, _, h, w = outputs.size()
        outputs *= mask
        ground_truths *= mask

        loss = criterion(outputs, ground_truths)

        # 反向传播
        loss.backward()  # 计算本批次梯度
        optimizer.step()  # 更新参数

        batch_size = ground_truths.size(0)  # 获取 batch_size
        total_samples += batch_size
        running_loss += loss.item() * batch_size

    # 计算所有 GPU 的 loss 总和和样本总数
    running_loss_tensor = torch.tensor([running_loss], dtype=torch.float32, device='cuda')
    total_samples_tensor = torch.tensor([total_samples], dtype=torch.float32, device='cuda')

    # 让所有 GPU 计算的 running_loss 和 total_samples 求和
    running_loss_tensor = sync_tensor(running_loss_tensor)
    total_samples_tensor = sync_tensor(total_samples_tensor)

    epoch_loss = running_loss_tensor.item() / total_samples_tensor.item()
    if local_rank == 0:
        writer.add_scalar('training_loss', epoch_loss, epoch + 1)
    train_loss_list.append(epoch_loss)

    return model, train_loss_list
def train_sfp(train_loader, model, criterion, optimizer, epoch, writer, local_rank, args, train_loss_list):
    model.train()
    running_loss = 0
    total_samples = 0
    for i, sample in enumerate(train_loader):
        ground_truths = sample['ground_truth']
        ground_truths = ground_truths.cuda(local_rank, non_blocking=True)
        mask = sample['mask']
        mask = mask.cuda(local_rank, non_blocking=True)  # False or True
        mask1 = torch.unsqueeze(mask, 1)
        inputs = sample['input']
        inputs.requires_grad_(True)
        inputs = inputs.cuda(local_rank, non_blocking=True)

        outputs, _, _, _, _, _, _, = model(inputs)
        outputs = outputs * mask1
        outputs = normalize(outputs, dim=1)
        ground_truths = ground_truths * mask1

        cosine = 1 - criterion(outputs, ground_truths)
        num_cosine = torch.sum(torch.sum(torch.sum(cosine, dim=1), dim=1))
        M = torch.sum(torch.sum(torch.sum(mask, dim=1), dim=1))  # 物体像素
        back_ground = (train_loader.batch_size * 256 * 256) - M  # 背景像素
        loss_cosine = num_cosine - back_ground
        loss = loss_cosine / M

        torch.distributed.barrier()  # 在所有进程运行到这一步之前，先完成此前代码的进程会等待其他进程。这使得我们能够得到准确、有序的输出。

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = ground_truths.size(0)  # 获取 batch_size
        total_samples += batch_size
        running_loss += loss.item() * batch_size

    # 计算所有 GPU 的 loss 总和和样本总数
    running_loss_tensor = torch.tensor([running_loss], dtype=torch.float64, device='cuda')
    total_samples_tensor = torch.tensor([total_samples], dtype=torch.float64, device='cuda')

    # 让所有 GPU 计算的 running_loss 和 total_samples 求和
    running_loss_tensor = sync_tensor(running_loss_tensor)
    total_samples_tensor = sync_tensor(total_samples_tensor)

    epoch_loss = running_loss_tensor.item() / total_samples_tensor.item()
    if local_rank == 0:
        writer.add_scalar('training_loss', epoch_loss, epoch + 1)
    train_loss_list.append(epoch_loss)
        
    return model, train_loss_list



def val(val_loader, model, writer, epoch, local_rank, args, criterion, val_loss_list):
    # assert val_loader.batch_size == 1# 与image的batch_size一致
    model.eval()
    total_loss = 0
    total_samples = 0
    epoch_nonzero = 0
    epoch_total   = 0
    random_move = RandomMovePad()
    image = torch.zeros([ 1, 4, 1024, 1224]).cuda(local_rank)
    with torch.no_grad():
        for i, sample_raw in enumerate(val_loader):
            image.zero_()
            for j in range(32):
                sample = random_move(sample_raw)
                distant = sample['distant']
                original = [-distant[0], -distant[1]]
                sample = unfold_enhanced_image(sample)
                mask = sample['mask'].cuda(local_rank)

                inputs = sample['input']
                inputs = inputs[:,0:4,:,:]
                inputs = inputs.cuda(local_rank, non_blocking=True)
                outputs = model(inputs)
                # outputs *= mask
                outputs = concat_enhanced_image(outputs)
                outputs = affine(outputs, 0, original, 1, [0.0])
                outputs = outputs[..., 128:128+1024,28:28+1224]

                image += outputs
            image = torch.div(image, 32)
            # draw_tensor_image(image, denormalize=False)
            ground_truths = sample_raw['CleanWater']
            ground_truths = ground_truths[:, 0:4, :, :]
            ground_truths = ground_truths.cuda(local_rank, non_blocking=True)
            mask = sample_raw['mask'].cuda(local_rank, non_blocking=True)
            mask = mask.unsqueeze(1)
            ground_truths = ground_truths * mask
            image = image * mask

            batch_loss = criterion(image, ground_truths).item()
            # print('batch_loss:', batch_loss)
            batch_size = ground_truths.size(0)  # 获取 batch_size
            total_loss += batch_loss * batch_size
            total_samples += batch_size
    # 计算所有 GPU 的 loss 总和和样本总数
    running_loss_tensor = torch.tensor([total_loss], dtype=torch.float32, device='cuda')
    total_samples_tensor = torch.tensor([total_samples], dtype=torch.float32, device='cuda')

    # 让所有 GPU 计算的 running_loss 和 total_samples 求和
    running_loss_tensor = sync_tensor(running_loss_tensor)
    total_samples_tensor = sync_tensor(total_samples_tensor)

    val_loss = running_loss_tensor.item() / total_samples_tensor.item()

    if local_rank == 0:
        writer.add_scalar('validation_loss', val_loss, epoch + 1)
    val_loss_list.append(val_loss)
    return val_loss_list

def val_PlanB(val_loader, model, writer, epoch, local_rank, args, criterion, val_loss_list):
    # assert val_loader.batch_size == 1# 与image的batch_size一致
    model.eval()
    device = torch.device(f'cuda:{local_rank}')
    window = hann2d(PATCH,device).unsqueeze(0).unsqueeze(0)

    total_loss, total_samples = 0., 0

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            input = sample['input'][:,0:4,:,:].cuda(device)
            gt = sample['CleanWater'].cuda(device)
            mask = sample['mask'].unsqueeze(1).cuda(device)
            mask = mask.expand_as(gt)
            gt *= mask

            H, W = input.shape[2:]
            # 准备空容器
            out_sum = torch.zeros(1, 4, H, W, device=device)
            w_sum = torch.zeros(1, 1, H, W, device=device)

            # 滑窗推理
            for y in range(0, H - PATCH + 1, STRIDE):
                for x in range(0, W - PATCH + 1, STRIDE):
                    patch = input[..., y:y+PATCH, x:x+PATCH]
                    pred = model(patch)
                    pred = pred * window
                    out_sum[..., y:y+PATCH, x:x+PATCH] += pred
                    w_sum[..., y:y+PATCH, x:x+PATCH] += window
            # 归一化得到整幅结果
            full_pred = out_sum / w_sum.clamp_min_(1e-3)
            # full_pred = torch.nn.functional.normalize(full_pred, dim=1)
            full_pred *= mask
            
            # 计算损失
            loss = criterion(full_pred, gt).item()
            # print('loss:',loss)
            batch_size = gt.size(0)
            total_loss += loss * batch_size
            total_samples += batch_size

    # 计算所有GPU的loss总和和样本总数
    running_loss_tensor = torch.tensor([total_loss], dtype=torch.float32, device='cuda')
    total_samples_tensor = torch.tensor([total_samples], dtype=torch.float32, device='cuda')

    # 让所有 GPU 计算的 running_loss 和 total_samples 求和
    running_loss_tensor = sync_tensor(running_loss_tensor)
    total_samples_tensor = sync_tensor(total_samples_tensor)

    val_loss = running_loss_tensor.item() / total_samples_tensor.item()

    if local_rank == 0:
        writer.add_scalar('validation_loss', val_loss, epoch + 1)
    val_loss_list.append(val_loss)

    return val_loss_list

def val_sfp(val_loader, model, writer, epoch, local_rank, args, criterion, val_loss_list):
    model.eval()
    total_loss = 0
    total_samples = 0
    random_move = RandomMove()
    image = torch.zeros([ 1, 3, 1024, 1224]).cuda(local_rank)
    with torch.no_grad():
        for i, sample_raw in enumerate(val_loader):
            image.zero_()
            mae = float('nan')
            while math.isnan(mae):
                for j in range(32):
                    sample = random_move(sample_raw)
                    filename = sample['filename'][0]
                    distant = sample['distant']
                    original = [-distant[0],-distant[1]]
                    sample1 = unfold_image(sample)
                    inputs = sample1['input'].cuda(local_rank)
                    mask = sample1['mask'].cuda(local_rank)
                    outputs, _, _, _, _, _, _, = model(inputs)
                    outputs = outputs * mask
                    out_put = concat_image(outputs)
                    pad = nn.ZeroPad2d(padding=(0, 200, 0, 0))
                    output_original = pad(out_put)
                    angle = 0
                    scale = 1
                    shear = [0.0]
                    output_original = affine(output_original, angle, original, scale, shear)
                    image = image + output_original
                ground_truth = sample_raw['ground_truth'].cuda(local_rank)
                # ground_truth 原始是 uint8，转为 float 并归一化
                ground_truth = ground_truth.float() / 255.0

                mask1 = sample_raw['mask'].squeeze(0).cuda(local_rank)

                M = torch.sum(torch.sum(mask1, dim=1))

                image = torch.div(image, 32)
                image = normalize(image, dim=1)
                if i%10 == 0 and local_rank == 0:
                    save_image(image, f'./results_sfp/{filename}.bmp')
                m = torch.sum(torch.sum(criterion(image, ground_truth))) / M
                mae = torch.acos(m) * 180 / pi

            batch_size = ground_truth.size(0)
            total_loss += mae * batch_size
            total_samples += batch_size
    # 计算所有 GPU 的 loss 总和和样本总数
    running_loss_tensor = torch.tensor([total_loss], dtype=torch.float64, device='cuda')
    total_samples_tensor = torch.tensor([total_samples], dtype=torch.float64, device='cuda')

    # 让所有 GPU 计算的 running_loss 和 total_samples 求和
    running_loss_tensor = sync_tensor(running_loss_tensor)
    total_samples_tensor = sync_tensor(total_samples_tensor)

    val_loss = running_loss_tensor.item() / total_samples_tensor.item()

    if local_rank == 0:
        writer.add_scalar('validation_loss', val_loss, epoch + 1)
    val_loss_list.append(val_loss)
    return val_loss_list

def val_sfp_PlanB(val_loader, model, writer, epoch, local_rank, args, criterion, val_loss_list):

    model.eval()
    device = torch.device(f'cuda:{local_rank}')
    # H, W = 1024, 1224                       # 原图大小，若变化自行获取
    window = hann2d(PATCH, device).unsqueeze(0).unsqueeze(0)     # (1,1,256,256)

    total_loss, total_samples = 0., 0

    with torch.no_grad():
        for i, sample in enumerate(val_loader):

            inputs   = sample['input'].cuda(device)              # B=1, C=3/4, H, W
            gt       = sample['ground_truth'].float().cuda(device) / 255.
            mask     = sample['mask'].unsqueeze(1).cuda(device)  # (1,1,H,W)
            gt *= mask

            H, W     = inputs.shape[2:]
            # -----------准备空容器-----------
            out_sum = torch.zeros(1, 3, H, W, device=device)
            w_sum   = torch.zeros(1, 1, H, W, device=device)

            # -----------滑窗推理-----------
            for y in range(0, H - PATCH + 1, STRIDE):
                for x in range(0, W - PATCH + 1, STRIDE):

                    patch = inputs[..., y:y+PATCH, x:x+PATCH]     # (1,C,256,256)
                    pred, *_ = model(patch)                      # (1,3,256,256)  ← 改成你的输出
                    pred = pred * window                         # 加权
                    out_sum[..., y:y+PATCH, x:x+PATCH] += pred
                    w_sum[...,  y:y+PATCH, x:x+PATCH] += window

            # -----------归一化得到整幅结果-----------
            full_pred = out_sum / w_sum.clamp_min(1e-6)           # (1,3,H,W)
            full_pred = torch.nn.functional.normalize(full_pred, dim=1)
            full_pred *= mask

            # -----------计算角度 MAE-----------
            M  = torch.sum(mask)                                  # 有效像素
            m  = torch.sum(criterion(full_pred , gt )) / M
            mae = torch.acos(m.clamp(-1 + 1e-6, 1 - 1e-6)) * 180 / pi

            total_loss   += mae.item()
            total_samples += 1       # batch_size=1

            if i % 10 == 0 and local_rank == 0:
                save_image(full_pred, f'./results_sfp/{sample["filename"][0]}_{epoch}.bmp')

    # -----------DDP 同步 + 记录-----------
    val_mae_tensor = torch.tensor([total_loss], device=device)
    val_samples_tensor = torch.tensor([total_samples], device=device)
    val_mae_tensor = sync_tensor(val_mae_tensor)
    val_samples_tensor = sync_tensor(val_samples_tensor)
    val_mae_tensor = val_mae_tensor / val_samples_tensor
    if local_rank == 0:
        writer.add_scalar('validation_mae', val_mae_tensor.item(), epoch+1)
    val_loss_list.append(val_mae_tensor.item())
    return val_loss_list

def ssim(img1, img2, window_size=11, size_average=True, val_range=1.0):
    # 确保输入在 [0, val_range] 范围内
    img1 = torch.clamp(img1, 0, val_range)
    img2 = torch.clamp(img2, 0, val_range)

    # 计算参数
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2

    # 创建高斯窗口
    window = create_window(window_size, 1).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=1)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=1) - mu1_mu2

    # SSIM 公式
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


def create_window(window_size, channel):
    # 生成高斯窗口
    sigma = 1.5  # 经验值
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window = g.ger(g).unsqueeze(0).unsqueeze(0)
    return window.repeat(channel, 1, 1, 1)


def draw_curve(train_loss_list, title):
    # 用 matplotlib 画图
    plt.figure()
    plt.plot(train_loss_list, label=f'{title}')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'{title} Curve')
    plt.legend()
    plt.grid(True)  # 添加网格线以增强可读性
    plt.savefig(f'{title}.png')

def draw_two_curve(list_1, list_2, title_1, title_2):
    plt.figure()
    plt.plot(list_1, label=f'{title_1}', linestyle='-', marker='o')
    plt.plot(list_2, label=f'{title_2}', linestyle='--', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(f'{title_1} and {title_2} Curve')
    plt.legend()
    plt.grid(True)  # 添加网格线以增强可读性
    plt.savefig('two_curve.png')

def total_variation_loss(img):
    # batch_size, channels, height, width = img.size()
    h_variation = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    v_variation = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    loss_tv = (h_variation + v_variation)
    return loss_tv

class loss_function(nn.Module):
    def __init__(self, c=0.5, val_range=1.0):
        """
        c:   L1 占比权重
        val_range: 若图像是 [0,1] 区间可传 1.0, 若 [0,255] 就传 255
        """
        super(loss_function, self).__init__()
        self.c = c
        self.val_range = val_range
        self.l1_loss = nn.L1Loss()

    def forward(self, output, target):
        # L1
        L1 = self.l1_loss(output, target)
        # SSIM
        # Lssim_val = 1.0 - ssim(output, target, val_range=self.val_range)
        Lssim_val = 0
        for channel in range(output.shape[1]):
            Lssim_val += 1.0 - ssim(output[:,channel:channel+1,:,:], target[:,channel:channel+1,:,:], val_range=self.val_range)

        # 计算TV损失
        Ltv_val = total_variation_loss(output)

        total_loss = 10 * L1 + 1 * Lssim_val + 10 * Ltv_val
        # print(f"L1: {L1.item()}, Lssim: {Lssim_val.item()}, Ltv: {Ltv_val.item()}")
        return total_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'  # 当前值+平均值
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]  # 花括号及其里面的字符 (称作格式化字段) 将会被 format() 中的参数替换
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
def draw_tensor_image(tensor_img, title="Channel", denormalize=True):
    """
    可视化PyTorch tensor图像
    参数：
        tensor_img: 输入的4D tensor [batch, channels, H, W] 或 3D tensor [channels, H, W]
        title: 图像标题（默认"Image"）
        denormalize: 是否执行反归一化（默认True）
    """
    # 转换到CPU并转为numpy
    img = tensor_img.detach().cpu().numpy()

    # 反归一化（假设归一化到[0,1]）
    if denormalize:
        img = img * 255.0
        img = img.clip(0, 255).astype('uint8')

    # 调整维度顺序
    if img.ndim == 4:  # 如果有batch维度
        img = img[0]    # 取第一个样本
    img = img.transpose(1, 2, 0)  # CHW -> HWC

    # 显示图像
    plt.figure(figsize=(20, 15))
    for channel in range(img.shape[-1]):
        plt.subplot(2,2,channel+1)
        plt.imshow(img[:,:,channel].squeeze(), cmap='gray')
        plt.title(f"{title} : {channel}(shape: {tensor_img.shape})")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    # plt.pause(0.001)  # 用于在训练过程中实时更新显示
    plt.close()

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    args.warmup_epochs = 0
    if epoch < args.warmup_epochs:
        lr *= float(epoch) / float(max(1.0, args.warmup_epochs))
        if epoch == 0:
            lr = 1e-6
    else:
        # progress after warmup
        if args.cos:  # cosine lr schedule
            # lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
            progress = float(epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
            lr *= 0.5 * (1. + math.cos(math.pi * progress))
            # print("adjust learning rate now epoch %d, all epoch %d, progress"%(epoch, args.epochs))
        else:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:  # 第一种定义方法
        param_group['lr'] = lr
    print("Epoch-{}, base lr {}, optimizer.param_groups[0]['lr']".format(epoch+1, args.lr),
          optimizer.param_groups[0]['lr'])

def sync_tensor(tensor):
    """在所有 GPU 之间同步张量，确保 loss 计算正确"""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  # 所有 GPU 求和
    return tensor