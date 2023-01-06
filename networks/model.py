import torch
import torch.nn as nn
from pytorch3d.transforms import so3_exponential_map as rodrigues

class Lifter(nn.Module):
    def __init__(self):
        super(Lifter, self).__init__()

        self.upscale = nn.Linear(32, 1024)

        self.res_pose1 = res_block()
        self.res_pose2 = res_block()
        self.pose3d = nn.Linear(1024, 48)

        self.drop = nn.Dropout(p=0.5)   # or 1
        

    def forward(self, p2d):

        x = p2d

        # First layer for dimensionality up to linear_size
        x = nn.LeakyReLU()(self.upscale(x))     # 32 -> 1024
        # x = self.drop(x)

        # Multiple bi-linear layers : res-block number is 2
        xp = self.res_pose1(x)
        xp = self.res_pose2(xp)

        # Last linear lyaer for Human_3D_Size in output
        xp = self.pose3d(xp)


        return xp


class res_block(nn.Module):
    def __init__(self):
        super(res_block, self).__init__()
        self.l1 = nn.Linear(1024, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x):
        inp = x
        x = self.l1(x)
        # x = self.bn1(x)
        x = nn.LeakyReLU()(x)
        # x = self.drop1(x)

        x = self.l2(x)
        # x = self.bn1(x)
        x = nn.LeakyReLU()(x)
        # x = self.drop2(x)

        x += inp

        return x

