import torch
import torch.nn as nn
import torch.nn.functional as F

nonlinearity_map = {"relu":nn.ReLU,
                  "prelu": nn.PReLU,
                  "selu": nn.SELU,
                  "elu": nn.ELU}





class ConvBuilder(nn.Module):
    """ A Method stack repetitive convolution related operations
        Conv -> Normalization -> Nonlinearity -> Dropout
            Args:
                conv (nn operation): Which convolution to use from torch.nn
                norm (str): Which normalization, batchnorm or groupnorm
                nonlinearity (str): Which activation function to use
                dropout (nn operation): Which drouput function to use from torch.nn
                dropout_prob (float): Percent of values to keep
                norm_args (kwargs): Any parameters needed for the chosen normalization
    """
    def __init__(self,conv=nn.Conv2d,
                 norm='batchnorm',nonlinearity='selu',
                 dropout = nn.Dropout2d, dropout_prob=0.8, **norm_args
                ):
        super().__init__()
        self.something = 10
        self.norm = self._build_normalization(norm)
        self.norm_args = norm_args
        self.nonlinearity = nonlinearity_map[nonlinearity]
        self.conv = conv
        self.dropout = dropout
        self.dropout_prob = dropout_prob


    def _build_normalization(self,norm):
        normalization_map = {'batchnorm':self._build_batchnorm,
                             'groupnorm':self._build_groupnorm}

        return normalization_map.get(norm,False)
    @staticmethod
    def _build_batchnorm(m,**kwargs):
        mapping={3:nn.BatchNorm1d,
                 4:nn.BatchNorm2d,
                 5:nn.BatchNorm3d}
        return mapping[m.weight.dim()](m.out_channels, **kwargs)

    @staticmethod
    def _build_groupnorm(m,group_size=4):
        out_channels = m.out_channels
        if out_channels % group_size != 0:
            num_groups = 1
        else:
            num_groups = out_channels // group_size
        return nn.GroupNorm(num_groups,m.out_channels)

    def __call__(self,in_channels,out_channels,
                 stride=1,kernel=3,padding=1,
                 activate=True, dropout=False):
        modules = []
        modules.append(self.conv(in_channels, out_channels, kernel, stride, padding))
        if self.norm is not False:
            modules.append(self.norm(modules[-1],**self.norm_args))
        if activate:
            modules.append(self.nonlinearity())
        if dropout:
            modules.append(self.dropout(self.dropout_prob))
        return nn.Sequential(*modules)



class BasicBlock(nn.Module):
    """ Basic Residual block from https://arxiv.org/pdf/1512.03385.pdf
            Args:
                in_channels (int): input channels to the layer
                out_channels (int): output channels to the layer
                builder (class): an intialized ConvBuilder()
                stride (int): stride value to use for the first conv
                kernel (int): Kernel size for conv, 3 -> 3x3
                padding (int): Padding to add to input
                downsample (layer): A conv operation to downsample the input
                                    before the residual connection
    """
    def __init__(self,in_channels, out_channels, builder,
                 stride=1, kernel=3, padding=1, downsample=None):
        super().__init__()

        self.builder = builder
        self.conv1 = builder(in_channels, out_channels,stride=stride, kernel=kernel, padding=padding)
        self.conv2 = builder(out_channels, out_channels, kernel=kernel, padding=padding)
        self.downsample = downsample
        self.nonlinearity = builder.nonlinearity()

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        return self.nonlinearity(residual + out)


block_map = {
    "basic": BasicBlock
}

class RockClassifier(nn.Module):
    """ A Residual network to perform rock classification
            Args:
                in_features (int): Intitial channels in the input tensor
                initial_channels (int): Number of features for the first conv
                blocks (list): Number of repeats for each residual layer
                embedding_dim (int): Size of embedding before output
                strides (list): Stride for each residual block
                padding (list): Padding for each residual block
                kernels (list): Kernel size for each residual block
                block (str): Which type of residual block to use
                conv (nn operation): Which convolution to use from torch.nn
                norm (str): Which normalization, batchnorm or groupnorm
                nonlinearity (str): Which activation function to use
                dropout (nn operation): Which drouput function to use from torch.nn
                dropout_prob (float): Percent of values to keep
                norm_args (kwargs): Any parameters needed for the chosen normalization
                num_classes (int): Output channels
                img_h (int): Size of input image height 

    """
    def __init__(self,in_features,
                 initial_channels=32, blocks=[1,1,1,1],
                 embedding_dim = 128, strides=[2,1,2,1],
                 padding=[1,1,1,1], kernels=[3,3,3,3],
                 block='basic', conv=nn.Conv2d,
                 norm='batchnorm', nonlinearity='selu',
                 dropout = nn.Dropout2d, dropout_prob=0.2,
                 num_classes = 6,img_h = 28, **norm_args):
        super().__init__()

        self.initial_channels = initial_channels
        self.blocks = blocks
        self.strides = strides
        self.padding = padding
        self.kernels = kernels
        self.block = block_map[block]
        self.builder = ConvBuilder(conv=nn.Conv2d, norm=norm,
                                   nonlinearity='selu', dropout = nn.Dropout2d,
                                    dropout_prob=0.2,**norm_args)


        self.initial_conv = self.builder(in_features,initial_channels)
        self.layers = self._build_blocks()
        reshaped_dim = round(float(img_h) / 2**len(blocks))**2 * (initial_channels * 2**len(blocks))
        self.embedding = nn.Linear(reshaped_dim,embedding_dim)
        self.fc = nn.Linear(embedding_dim,num_classes)




    def _build_blocks(self):
        layers = []
        channels = self.initial_channels
        #assert len(blocks) + len(strides) + len(padding) + len(kernels) == 4*len(blocks)
        for b,s,p,k in zip(self.blocks, self.strides, self. padding, self.kernels):
            layers.append(
                self._build_layer(channels,channels*2,s,k,b,p)
            )
            channels *= 2
        return nn.ModuleList(layers)


    def _build_layer(self,in_channels, out_channels,
                     stride=1, kernel=3, layers=1, pad=1):

        if (stride !=1) or (in_channels != out_channels):
            downsample = self.builder(in_channels, out_channels,
                                      stride, kernel=1,
                                      padding=0, activate=False)

        layer = [self.block(in_channels, out_channels,self.builder,
                            stride=stride, padding=pad, downsample=downsample)]

        for _ in range(1,layers):
            layer.append(
                self.block(out_channels, out_channels,self.builder)
            )

        return nn.Sequential(*layer)


    def forward(self,imgs):
        out = self.initial_conv(imgs)
        for l in self.layers:
            out = l(out)
        embedding = self.embedding(out.reshape(out.size(0),-1))
        output = self.fc(embedding)
        return F.log_softmax(output,1), embedding.detach()
