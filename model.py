import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ConvolutionalLayer(nn.Module):
    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(ConvolutionalLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class ResidualLayer(nn.Module):
    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(ResidualLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ValueHead(nn.Module):
    def __init__(self, inplanes, hidden_layer_size, width, height, norm_layer=None):
        super(ValueHead, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, 1)
        self.bn1 = norm_layer(1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc1 = nn.Linear(width * height, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = torch.tanh(out)
        return out


class PolicyHead(nn.Module):
    def __init__(self, inplanes, width, height, norm_layer=None):
        super(PolicyHead, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, 2)
        self.bn1 = norm_layer(2)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc1 = nn.Linear(width * height * 2, width * height * 8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = F.softmax(out)
        return out


class ResidualCNN(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        residual_layers,
        vh_hidden_layer_size,
        width,
        height,
        norm_layer=None,
    ):
        super(ResidualCNN, self).__init__()
        self.width = width
        self.height = height
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv_layer = ConvolutionalLayer(inplanes, planes, norm_layer=norm_layer)
        self.res_layers = []
        for _ in range(residual_layers):
            self.res_layers.append(
                ResidualLayer(planes, planes, norm_layer=norm_layer)
            )
        self.value_head = ValueHead(
            planes, vh_hidden_layer_size, width, height, norm_layer=norm_layer
        )
        self.policy_head = PolicyHead(planes, width, height, norm_layer=norm_layer)

    def forward(self, x):
        out = self.conv_layer(x)
        for layer in self.res_layers:
            out = layer(out)

        value_in = out
        policy_in = out
        value_out = self.value_head(value_in)
        policy_out = self.policy_head(policy_in)

        return policy_out, value_out

    def predict(self, state):
        # TODO what size for ints here
        t = torch.zeros([3, self.width, self.height], dtype=torch.int32)
        for coord, cell in state.grid.items():
            t[max(0, cell.race), coord.x, coord.y] = cell.count
        t[2,:] = state.current_player * torch.ones([self.width, self.height])
        t = t.view(1, 3, self.width, self.height).float()
        p, v = self(t)
        return p[0], v[0]

def vamperouge_net(config):
    return ResidualCNN(
        config.nn_inplanes,
        config.nn_planes,
        config.nn_residual_layers,
        config.nn_vh_hidden_layer_size,
        config.board_width,
        config.board_height,
    )
