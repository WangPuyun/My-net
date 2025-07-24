from __future__ import print_function, division
import os
import torch
import pandas as pd
import torchvision
import scipy.io as scio
from torchvision.transforms.functional import affine
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import warnings
import torchvision.transforms.functional as F
warnings.filterwarnings("ignore")

plt.ion()  # interactive mode

class MyDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform):
        """
        Args:
            csv_file (string): 带有注释的 csv 文件的路径。
            root_dir (string): 包含所有图像的目录。
            transform (callable, optional): 应用于样本的可选变换。
        """
        self.image_gt = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_gt)

    def __getitem__(self, idx):
        img_gt_file_path = os.path.join(self.root_dir,
                                        self.image_gt.iloc[idx, 0])  # os.path.join连接两个或更多的路径名组件
        img_gt = scio.loadmat(img_gt_file_path)
        image = torch.as_tensor(img_gt['images'], dtype=torch.float32).permute(2, 0, 1)
        # enhanced_images = torch.as_tensor(img_gt['enhanced_images'], dtype=torch.float32).permute(2, 0, 1)
        CleanWater = torch.as_tensor(img_gt['CleanWater'], dtype=torch.float32).permute(2, 0, 1)
        ground_truth = torch.as_tensor(img_gt['I_Normal_gt'], dtype=torch.float32).permute(2, 0, 1)
        mask = torch.as_tensor(img_gt['mask'], dtype=torch.float32)

        N1 = torch.as_tensor(img_gt['Diffuse'], dtype=torch.float32).permute(2, 0, 1)
        N2 = torch.as_tensor(img_gt['Specular1'], dtype=torch.float32).permute(2, 0, 1)
        N3 = torch.as_tensor(img_gt['Specular2'], dtype=torch.float32).permute(2, 0, 1)
        N = torch.cat([N1, N2, N3], dim=0)
        input = torch.cat([image, N], dim=0)

        # P = img_gt['P']
        # P = P[:, :, 1:5]
        # P1 = torch.as_tensor(P, dtype=torch.float32).permute(2, 0, 1)
        # input = torch.cat([image, P1], dim=0)

        # input = torch.cat([enhanced_images, input], dim=0)
        
        filename = self.image_gt.iloc[idx, 0].rstrip(".mat")
        sample = { 'input': input, 'ground_truth': ground_truth, 'mask': mask, 'CleanWater': CleanWater, 'mat_path': img_gt_file_path, 'P': img_gt['P'], 'filename':filename, 'image': image}
        if self.transform:
            sample = self.transform(sample)

        return sample

class FixedCrop(object):
    """先裁剪掉多余的像素，使图像大小为 16 的倍数"""

    def __init__(self, target_size=(1024, 1216)):  # 确保 1024 × 1216
        self.target_size = target_size

    def __call__(self, sample):
        input, ground_truth, mask, CleanWater, mat_path = sample['input'], sample['ground_truth'], sample['mask'], sample['CleanWater'], sample['mat_path']

        # 获取原始尺寸
        _, h, w = input.shape

        # 计算需要裁剪的区域（居中裁剪）
        crop_h = min(h, self.target_size[0])
        crop_w = min(w, self.target_size[1])
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2

        input = input[:, top:top + crop_h, left:left + crop_w]
        CleanWater = CleanWater[:, top:top + crop_h, left:left + crop_w]
        ground_truth = ground_truth[:, top:top + crop_h, left:left + crop_w]
        mask = mask[top:top + crop_h, left:left + crop_w]

        return { 'input': input, 'ground_truth': ground_truth, 'mask': mask, 'CleanWater': CleanWater, 'mat_path': mat_path, 'filename':sample['filename']}

class RandomCrop(object):
    """随机裁剪样本中的图像。

    Args:
        output_size (tuple or int): 所需的输出大小。 如果是 int，则进行方形裁剪。
    """

    def __call__(self, sample):
        input, ground_truth, mask, CleanWater, mat_path, image = sample['input'], sample['ground_truth'], sample['mask'], sample['CleanWater'], sample['mat_path'], sample['image']

        crop = torchvision.transforms.RandomCrop(256)

        while True:
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            mask_temp = crop(mask)
            a = torch.sum(torch.sum(mask_temp))
            # print(a)
            if a / 65536 > 0.5:  # Iraw为0.25 原来为0.5
                torch.random.manual_seed(seed)
                image1 = crop(image)
                torch.random.manual_seed(seed)
                input = crop(input)
                torch.random.manual_seed(seed)
                ground_truth = crop(ground_truth)
                torch.random.manual_seed(seed)
                CleanWater = crop(CleanWater)
                break

        return {'input': input, 'ground_truth': ground_truth, 'mask': mask_temp, 'CleanWater': CleanWater, 'mat_path': mat_path, 'filename':sample['filename'], 'image': image1}

class RandomMovePad(object):
    """随机平移（含对称 Padding 以防像素丢失）"""
    def __init__(self, max_translate=128, pad_mode='reflect'):
        self.max_t = max_translate
        self.pad_mode = pad_mode

    def _pad(self, tensor):
        # (左, 右, 上, 下) 统一补 self.max_t
        return F.pad(tensor, (self.max_t-100, self.max_t, self.max_t-100, self.max_t),
                     padding_mode=self.pad_mode)

    def __call__(self, sample):
        # -------- 克隆源数据，保证不污染 sample_raw ----------
        img_src   = sample['input']
        gt_src    = sample['ground_truth']
        mask_src  = sample['mask']
        clean_src = sample['CleanWater']

        # *clone* 保证后续 in-place 操作不影响原张量
        img   = self._pad(img_src.clone())
        gt    = self._pad(gt_src.clone())
        mask  = self._pad(mask_src.clone())
        clean = self._pad(clean_src.clone())

        # 随机平移
        tx = random.randint(-self.max_t, self.max_t)
        ty = random.randint(-self.max_t+100, self.max_t-100)
        translate = [tx, ty]

        seed = torch.random.seed()       # 保证四路一致
        for t in (img, gt, mask, clean):
            torch.random.manual_seed(seed)
            t[:] = F.affine(t, angle=0, translate=translate,
                            scale=1.0, shear=[0.0])

        sample = {'input':img, 'ground_truth':gt, 'mask':mask, 'CleanWater':clean,
                       'mat_path': sample['mat_path'], 'distant':translate, 'filename':sample['filename']}
        return sample


class RandomMove(object):# 我发现师兄用的RandomMove和TranLate是一样的
    def __call__(self, sample, target_size=(1024, 1216)):
        input, ground_truth, mask, CleanWater, mat_path, image = sample['input'], sample['ground_truth'], sample['mask'], sample['CleanWater'], sample['mat_path'], sample['image']

        mask = torch.unsqueeze(mask, 0)
        angle = 0
        scale = 1
        shear = [0.0]
        ls = list(range(-200, 0))
        translate = [random.choice(ls), random.choice(ls)]
        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        image_move = affine(image, angle, translate, scale, shear)
        torch.random.manual_seed(seed)
        input = affine(input, angle, translate, scale, shear)
        torch.random.manual_seed(seed)
        ground_truth = affine(ground_truth, angle, translate, scale, shear)
        torch.random.manual_seed(seed)
        mask = affine(mask, angle, translate, scale, shear)
        mask = torch.squeeze(mask)
        torch.random.manual_seed(seed)
        CleanWater = affine(CleanWater, angle, translate, scale, shear)
        return {'input': input, 'ground_truth': ground_truth, 'mask': mask, 'CleanWater': CleanWater, 'mat_path': mat_path, 'distant': translate, 'filename':sample['filename'], 'image': image_move}

def unfold_image(sample):
    input, ground_truth, mask, CleanWater, mat_path, image = sample['input'], sample['ground_truth'], sample['mask'], sample['CleanWater'], sample['mat_path'], sample['image']
    input, mask, image = input.squeeze(0), mask.squeeze(0), image.squeeze(0)
    patches1 = input.unfold(2, 256, 256).unfold(3, 256, 256)
    patches1 = patches1.reshape(13, -1, 256, 256)
    patches1.transpose_(0, 1)
    print("unfold_image input.shape:", input.shape)

    patches2 = mask.unfold(1, 256, 256).unfold(2, 256, 256)
    patches2 = patches2.reshape(1, -1, 256, 256)
    patches2.transpose_(0, 1)

    patches3 = image.unfold(2, 256, 256).unfold(3, 256, 256)
    patches3 = patches3.reshape(4, -1, 256, 256)
    patches3.transpose_(0, 1)
    # else:
    #     patches1 = input.unfold(2, 256, 256).unfold(3, 256, 256)
    #     patches1 = patches1.reshape(8,4,-1,256,256)
    #     patches1 = patches1.permute(0, 2, 1, 3, 4)  # 交换 batch 和 patch 维度
    #
    #     patches2 = mask.unfold(1, 256, 256).unfold(2, 256, 256)
    #     patches2 = patches2.reshape(8, 1, -1, 256, 256)
    #     patches2 = patches2.permute(0, 2, 1, 3, 4)  # 交换 batch 和 patch 维度

    return {'input': patches1, 'ground_truth': ground_truth, 'mask': patches2, 'CleanWater': CleanWater, 'mat_path': mat_path, 'filename':sample['filename'], 'image': patches3}

def unfold_enhanced_image(sample):
    input, ground_truth, mask, CleanWater, mat_path = sample['input'], sample['ground_truth'], sample['mask'], sample['CleanWater'], sample['mat_path']
    input, mask = input.squeeze(0), mask.squeeze(0)
    patches1 = input.unfold(1, 256, 256).unfold(2, 256, 256)
    patches1 = patches1.reshape(8, -1, 256, 256)
    patches1.transpose_(0, 1)

    patches2 = mask.unfold(0, 256, 256).unfold(1, 256, 256)
    patches2 = patches2.reshape(1, -1, 256, 256)
    patches2.transpose_(0, 1)
    # else:
    #     patches1 = input.unfold(2, 256, 256).unfold(3, 256, 256)
    #     patches1 = patches1.reshape(8,4,-1,256,256)
    #     patches1 = patches1.permute(0, 2, 1, 3, 4)  # 交换 batch 和 patch 维度
    #
    #     patches2 = mask.unfold(1, 256, 256).unfold(2, 256, 256)
    #     patches2 = patches2.reshape(8, 1, -1, 256, 256)
    #     patches2 = patches2.permute(0, 2, 1, 3, 4)  # 交换 batch 和 patch 维度

    return {'input': patches1, 'ground_truth': ground_truth, 'mask': patches2, 'CleanWater': CleanWater, 'mat_path': mat_path, 'filename':sample['filename']}

def concat_image(outputs):
    image1 = torch.cat((outputs[0], outputs[1]), dim=2)
    image1 = torch.cat((image1, outputs[2]), dim=2)
    image1 = torch.cat((image1, outputs[3]), dim=2)

    image2 = torch.cat((outputs[4], outputs[5]), dim=2)
    image2 = torch.cat((image2, outputs[6]), dim=2)
    image2 = torch.cat((image2, outputs[7]), dim=2)

    image3 = torch.cat((outputs[8], outputs[9]), dim=2)
    image3 = torch.cat((image3, outputs[10]), dim=2)
    image3 = torch.cat((image3, outputs[11]), dim=2)

    image4 = torch.cat((outputs[12], outputs[13]), dim=2)
    image4 = torch.cat((image4, outputs[14]), dim=2)
    image4 = torch.cat((image4, outputs[15]), dim=2)

    image1 = torch.cat((image1, image2), dim=1)
    image1 = torch.cat((image1, image3), dim=1)
    image1 = torch.cat((image1, image4), dim=1)
    return image1.unsqueeze(0)

def concat_enhanced_image(outputs):
    image1 = torch.cat((outputs[0], outputs[1]), dim=2)
    image1 = torch.cat((image1, outputs[2]), dim=2)
    image1 = torch.cat((image1, outputs[3]), dim=2)
    image1 = torch.cat((image1, outputs[4]), dim=2)

    image2 = torch.cat((outputs[5], outputs[6]), dim=2)
    image2 = torch.cat((image2, outputs[7]), dim=2)
    image2 = torch.cat((image2, outputs[8]), dim=2)
    image2 = torch.cat((image2, outputs[9]), dim=2)

    image3 = torch.cat((outputs[10], outputs[11]), dim=2)
    image3 = torch.cat((image3, outputs[12]), dim=2)
    image3 = torch.cat((image3, outputs[13]), dim=2)
    image3 = torch.cat((image3, outputs[14]), dim=2)

    image4 = torch.cat((outputs[15], outputs[16]), dim=2)
    image4 = torch.cat((image4, outputs[17]), dim=2)
    image4 = torch.cat((image4, outputs[18]), dim=2)
    image4 = torch.cat((image4, outputs[19]), dim=2)

    image5 = torch.cat((outputs[20], outputs[21]), dim=2)
    image5 = torch.cat((image5, outputs[22]), dim=2)
    image5 = torch.cat((image5, outputs[23]), dim=2)
    image5 = torch.cat((image5, outputs[24]), dim=2)

    image1 = torch.cat((image1, image2), dim=1)
    image1 = torch.cat((image1, image3), dim=1)
    image1 = torch.cat((image1, image4), dim=1)
    image1 = torch.cat((image1, image5), dim=1)

    image = image1.unsqueeze(0)
    return image