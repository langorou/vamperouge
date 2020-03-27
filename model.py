import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import config
import game
from progress.bar import Bar
from progress.misc import AverageMeter


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
        out = F.log_softmax(out, dim=1)
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
        self.cuda_available = torch.cuda.is_available()
        self.width = width
        self.height = height
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv_layer = ConvolutionalLayer(inplanes, planes, norm_layer=norm_layer)
        res_layers = []
        for _ in range(residual_layers):
            res_layers.append(ResidualLayer(planes, planes, norm_layer=norm_layer))
        self.seq_res_layer = nn.Sequential(*res_layers)
        self.value_head = ValueHead(
            planes, vh_hidden_layer_size, width, height, norm_layer=norm_layer
        )
        self.policy_head = PolicyHead(planes, width, height, norm_layer=norm_layer)
        if self.cuda_available:
            self.cuda()

    def forward(self, x):
        out = self.conv_layer(x)
        out = self.seq_res_layer(out)
        value_out = self.value_head(out)
        policy_out = self.policy_head(out)
        return policy_out, value_out

    def train_from_samples(self, train_samples):
        """
        samples: (state, policy, value)
        """
        optimizer = optim.Adam(self.parameters())

        for epoch in range(config.train_epochs):
            print(f"epoch {str(epoch+1)}")
            self.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            policy_losses = AverageMeter()
            value_losses = AverageMeter()

            size = int(len(train_samples) / config.train_bs)
            bar = Bar("Training NN", max=size)
            batch_idx = 0

            while batch_idx < size:
                start = time.time()
                sample_ids = np.random.randint(len(train_samples), size=config.train_bs)
                states, policies, values = list(
                    zip(*[train_samples[i] for i in sample_ids])
                )
                states = self._states_to_tensor(states)
                target_policies = torch.FloatTensor(np.array(policies))
                target_values = torch.FloatTensor(np.array(values).astype(np.float64))

                if self.cuda_available:
                    states = states.contiguous().cuda()
                    target_policies = target_policies.contiguous().cuda()
                    target_values = target_values.contiguous().cuda()

                data_time.update(time.time() - start)

                # get output
                out_policies, out_values = self(states)
                loss_policy = self.loss_policy(target_policies, out_policies)
                loss_value = self.loss_value(target_values, out_values)
                total_loss = loss_policy + loss_value

                policy_losses.update(loss_policy.item(), states.size(0))
                value_losses.update(loss_value.item(), states.size(0))

                # gradient and SGD
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_time.update(time.time() - start)
                batch_idx += 1

                # plot progress
                bar.suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_policy: {lp:.4f} | Loss_value: {lv:.3f}".format(
                    batch=batch_idx,
                    size=size,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lp=policy_losses.avg,
                    lv=value_losses.avg,
                )
                bar.next()
            bar.finish()

    def loss_policy(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_value(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def _states_to_tensor(self, states):
        t = torch.zeros([len(states), 3, self.width, self.height], dtype=torch.int32)
        for i, state in enumerate(states):
            for coord, cell in state.grid.items():
                if cell.race == game.VAMPIRE:
                    t[i, 0, coord.x, coord.y] = cell.count
                elif cell.race == game.WEREWOLF:
                    t[i, 1, coord.x, coord.y] = cell.count
                elif cell.race == game.HUMAN:
                    t[i, 2, coord.x, coord.y] = cell.count
        # t doesn't have information about the current player because the states are always
        # canonical here, so vampires are the current players
        return t.float()

    def predict(self, state):
        t = self._states_to_tensor([state])
        if self.cuda_available:
            t = t.contiguous().cuda()
        self.eval()
        with torch.no_grad():
            p, v = self(t)
        return torch.exp(p).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(f"Checkpoint directory doesn't exist. Creating directory {folder}")
            os.mkdir(folder)
        torch.save({"state_dict": self.state_dict(),}, filepath)

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"No model in path {filepath}")
        map_location = None if self.cuda_available else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint["state_dict"])


def vamperouge_net(config):
    return ResidualCNN(
        config.nn_inplanes,
        config.nn_planes,
        config.nn_residual_layers,
        config.nn_vh_hidden_layer_size,
        config.board_width,
        config.board_height,
    )
