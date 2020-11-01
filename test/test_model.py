from simtool.model import *
import pytest
import torch
import numpy as np

class TestConvBuilder:
    def test_groupnorm(self):
        builder = ConvBuilder(norm='groupnorm',group_size=8)
        seq = builder(32,64)
        assert len(seq) == 3
        assert type(seq[1]) == torch.nn.modules.normalization.GroupNorm

    def test_batchnorm(self):
        builder = ConvBuilder(norm='batchnorm',eps=1e-06, momentum=0.2, affine=False, track_running_stats=False)
        seq = builder(32,64)
        assert len(seq) == 3
        assert type(seq[1]) == torch.nn.modules.batchnorm.BatchNorm2d
        assert seq[1].eps == 1e-06
        assert seq[1].momentum == 0.2

    def test_conv(self):
        builder = ConvBuilder(norm=False)
        seq = builder(32,64, stride=2, kernel=1, activate=False)
        assert len(seq) == 1
        assert seq[0].stride[0] == 2
        assert seq[0].kernel_size[0] == 1

    def test_dropout(self):
        builder = ConvBuilder(norm=False)
        seq = builder(32,64, dropout=True)
        assert len(seq) == 3
        assert type(seq[2]) == torch.nn.modules.dropout.Dropout2d

class TestBasicBlock:
    def test_basicblock(self):
        builder = ConvBuilder()
        block = BasicBlock(3,3,builder)
        x = torch.rand(1,3,28,28)
        out = block(x)
        assert out.size(-1) == 28

    def test_downsample(self):
        builder = ConvBuilder()
        downsample = builder(3,6,stride=2,kernel=1,padding=0,activate=False)
        block = BasicBlock(3,6,builder,stride=2,downsample=downsample)
        x = torch.rand(1,3,28,28)
        out = block(x)
        assert list(out.size()) == [1,6,14,14]


class TestRockClassifier:
    def test_build_layer(self):
        model = RockClassifier(3)
        conv = model._build_layer(3,6,2,3,2)
        assert len(conv) ==2
        x = torch.rand(1,3,28,28)
        out = conv(x)
        assert list(out.size()) == [1,6,14,14]


    def test_build_blocks(self):
        model = RockClassifier(3)
        layer = model._build_blocks()
        assert len(layer) == len(model.blocks)

    def test_simple_model(self):
        model = RockClassifier(3, blocks=[1,1], embedding_dim=64, strides = [2,2,2],
                               padding=[1,1,1], kernels= [3,3,3])

        x = torch.rand(1,3,28,28)
        out,_ = model(x)
        assert list(out.size()) == [1,6]
        assert np.allclose(out.exp().sum(1).item(), 1.)
