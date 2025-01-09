from typing import Tuple, Union, List
import torch
from torch import nn
from models.TrebleAFF_MIMOUNet import MIMOUNet
from models.Denoiser import DenoiserUNet


class DocDiff(nn.Module):
    def __init__(self, input_channels: int = 2, output_channels: int = 1, n_channels: int = 32,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 n_blocks: int = 1):
        super(DocDiff, self).__init__()
        self.denoiser = DenoiserUNet(input_channels, output_channels, n_channels, ch_mults, n_blocks, is_noise=True)
        self.init_predictor = MIMOUNet(num_res=16)
        # self.init_predictor = UNet(input_channels, output_channels, 2 * n_channels, ch_mults, n_blocks)
        # 这里的结构略有疑惑，首先就是对于输入的文档图像究竟是几通道图像，以我现在的认知，我暂时认为输入的文档图像都是灰度图，即1通道
        # 这样的话，正好和输出通道数相对应，接下来我们要看，在DocDiff中初始化了denoiser和init_predictor两个网络，两个网络均是前面的Unet
        # 唯一的区别就在输入通道数一个是2，另一个是1，再结合文章中的网络结构图，我发现文章中提到的粗预测网络和denoiser网络实际上的核心Unet结构
        # 是相同的，这点也在代码上得到了作证，二者的区别在于粗预测网络的输入是最初的模糊文档图像，而denoiser网络的输入是初步去噪之后的结果和
        # 扩散之后的噪点在通道维度连接得到的结果，假设扩散噪点通道数为1，这样连接之后正好通道数可以变为2
        # 那么接下来的的问题是如何将该网络应用到彩色图像上，即三通道图像上；有一点需要我进行研究，就是由3通道图像扩散出来的噪点是几通道的
        # 最终我们应该得到这样两句初始化代码
        # self.init_predictor = UNet(input_channels=3, output_channels=3, n_channels, ch_mults, n_blocks, is_noise=False)
        # self.denoiser = UNet(input_channels=3+扩散噪点通道数, output_channels=3, n_channels, ch_mults, n_blocks, is_noise=True)

    def forward(self, x, condition, t, diffusion):
        # 这里的输入分别表示x：ground truth；condition：blur image；t：时间步；diffusion：高斯扩散
        # 这里实际上是在对残差结果进行了扩散“去模糊”过程，这么做的目的是为了降低计算量
        x_list = self.init_predictor(condition, t)
        x_ = x_list[-1]
        residual = x - x_
        noisy_image, noise_ref = diffusion.noisy_image(t, residual)
        x__ = self.denoiser(torch.cat((noisy_image, x_.clone().detach()), dim=1), t)
        return x_, x__, noisy_image, noise_ref, x_list
    # 这里是很重要的代码，这里完整的介绍了在训练过程中图像的输入输出过程
    # 首先我们假定网络的输入是需要去模糊的图片 x，将 x 输入 init_predictor 之后得到了粗预测结果 x_
    # 在这里网络计算了一个奇怪的变量：residual = x - x_；这是由模糊图像减去粗预测去模糊结果计算得到的
    # 在前面并没有提到这个变量，但结合文章中的网络结构图发现，扩散模型前向扩散过程得到的结果名称为 Noisy Residual
    # 正好和变量 residual 相对应，由此可以推断，Noisy Residual 就是由 residual 扩散加噪得到的结果
    # 然后我们将扩散加噪结果 noisy_image 和初步预测去噪结果 x_ 连接在一起输入 Denoiser 网络中
    # Denoiser 网络的输出结果是 x__，根据推测就是图中的 Residual Prediction
    # 目前的问题就是不清楚变量 noise_ref 究竟是什么，且将其返回的意义是什么


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
# 上面的 class 的作用应该是用于更新模型的参数
