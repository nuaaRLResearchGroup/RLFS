import torch
from torch import nn
import torchvision.utils as vutils
import numpy


class BasicEncoder(nn.Module):   #nn.Module定义了神经网络模型
    """
    The BasicEncoder module takes an cover image and a data tensor and combines
    them into a steganographic image.

    """

    def _conv2d(self, in_channels, out_channels):    #创建二维卷积层
        return nn.Conv2d(
            in_channels=in_channels,   #输入通道数，即输入特征图的通道数。这个参数指定了输入数据的深度。
            out_channels=out_channels, #输出通道数，即卷积层的滤波器数量。这个参数决定了卷积层的输出特征图的深度。
            kernel_size=3,             #卷积核大小
            padding=1                  #填充大小
        )

    def _build_models(self):  #结合结构图
        self.conv1 = nn.Sequential(   #封面图片处理
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv2 = nn.Sequential(   #封面图片+隐藏信息
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size, 3),
        )
        return self.conv1, self.conv2, self.conv3, self.conv4

    def __init__(self, data_depth, hidden_size):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self._models = self._build_models()

    def forward(self, image, data):  #模型的前向传播函数
        x = self._models[0](image)   #将输入的 image 通过第一个卷积层 _models[0] 进行前向传播
        x_1 = self._models[1](torch.cat([x] + [data], dim=1))   #将 x 和 data 沿着通道维度（dim=1）进行拼接，输入到第二个卷积层传播
        x_2 = self._models[2](x_1)
        x_3 = self._models[3](x_2)

        # 保存卷积层x_3对应的图片
        #vutils.save_image(x_3, 'x3_image.png')
        return x_3


class ResidualEncoder(BasicEncoder):

    def forward(self, image, data):
        return image + super().forward(image, data)


class DenseEncoder(ResidualEncoder):

    def _build_models(self):
        self.conv1 = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv2 = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv3 = nn.Sequential(
            self._conv2d(self.hidden_size * 2 +
                         self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.conv4 = nn.Sequential(
            self._conv2d(self.hidden_size * 3 + self.data_depth, 3)
        )

        return self.conv1, self.conv2, self.conv3, self.conv4

    def forward(self, image, data):
        x = self._models[0](image)
        x_list = [x]
        x_1 = self._models[1](torch.cat(x_list+[data], dim=1))
        x_list.append(x_1)
        x_2 = self._models[2](torch.cat(x_list+[data], dim=1))
        x_list.append(x_2)
        x_3 = self._models[3](torch.cat(x_list+[data], dim=1))
        x_list.append(x_3)
        return image + x_3
