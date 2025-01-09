import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomAffine, RandomHorizontalFlip, RandomCrop
from PIL import Image
# ToTensor 将 PIL 图像或 ndarray 转换为张量（Tensor），并归一化到[0, 1]范围内，如果输入数据是一个多通道图像，该函数还会在转换过程中调整维度顺序，以便将通道维度放在第一维
# RandomAffine 是一个随机仿射变换函数，用于对图像进行随机旋转、缩放、平移、剪切等操作，从而生成具有一定变换程度的新图像
# RandomHorizontalFlip 是一个随机水平翻转函数，用于随机地将图像水平翻转（即左右翻转）
# RandomCrop 是一个随机裁剪函数，用于从图像中随机裁剪出指定大小的区域，随机裁剪可以有效地提高模型对输入图像的位置不变性，从而增强模型的泛化能力，并且可以有效地处理输入图像尺寸不一致的问题


def ImageTransform(loadSize):
    return {"train": Compose([
        RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        # pad_if_needed: bool值, 避免数组越界; padding_mode：填充模式, “constant”: 利用常值进行填充; fill: 用于填充的常值
        RandomAffine(10, fill=255),
        # degrees：可从中选择的度数范围, 如果为非零数字, 旋转角度从(-degrees,+degress), 或者可设置为(min,max)
        RandomHorizontalFlip(p=0.2),
        # p: 随即水平翻转概率
        ToTensor(),
    ]), "test": Compose([
        ToTensor(),
    ]), "train_gt": Compose([
        RandomCrop(loadSize, pad_if_needed=True, padding_mode='constant', fill=255),
        RandomAffine(10, fill=255),
        RandomHorizontalFlip(p=0.2),
        ToTensor(),
    ])}


class DocData(Dataset):
    def __init__(self, path_img, path_gt, loadSize, mode=1):
        super().__init__()
        self.path_gt = path_gt
        self.path_img = path_img
        self.data_gt = os.listdir(path_gt)
        # os.listdir() 用于返回对应路径下所有文件和文件夹
        self.data_img = os.listdir(path_img)
        self.mode = mode
        if mode == 1:
            self.ImgTrans = (ImageTransform(loadSize)["train"], ImageTransform(loadSize)["train_gt"])
        else:
            self.ImgTrans = ImageTransform(loadSize)["test"]
        # 用 mode 这个 int 常量来设定模式，如果 mode = 1，则为训练模式，应用训练集的图像处理；如果为测试模式，则采用测试集图像处理

    def __len__(self):
        return len(self.data_gt)
        # 用于返回ground truth文件夹下总共有多少张图片

    def __getitem__(self, idx):

        gt = Image.open(os.path.join(self.path_gt, self.data_img[idx]))
        img = Image.open(os.path.join(self.path_img, self.data_img[idx]))
        # os.path.join() 用于组合文件路径，即返回一个'path_gt/data_img[idx]'的文件路径
        # 但这里有一点要注意，无论是 gt 还是 img 引用的都是对应路径下的，但文件名都选用的是 img 的文件，因此我们在应用时要保证 gt 和 img 这两个路径下所有文件的名称和数量都应该相同
        img = img.convert('RGB')
        gt = gt.convert('RGB')
        # 如果不使用.convert(‘RGB’)进行转换的话，读出来的图像是RGBA四通道的，A通道为透明通道，因此使用convert(‘RGB’)进行通道转换
        if self.mode == 1:
            seed = torch.random.seed()
            # 这句代码设置了一个随机种子，并将其赋值给变量 seed，随机种子是生成随机数的起始点，
            # 通过设置相同的随机种子可以确保在随机数生成器中生成的随机数序列是可复现的，即每次运行代码时都会得到相同的随机数序列
            torch.random.manual_seed(seed)
            # 这句代码将上面设置的随机种子应用到 PyTorch 的随机数生成器上，确保 PyTorch 在接下来的随机操作中使用相同的随机种子
            img = self.ImgTrans[0](img)
            torch.random.manual_seed(seed)
            gt = self.ImgTrans[1](gt)
        else:
            img = self.ImgTrans(img)
            gt = self.ImgTrans(gt)
        name = self.data_img[idx]
        return img, gt, name
