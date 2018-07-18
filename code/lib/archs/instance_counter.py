import torch.nn as nn
from modules.coord_conv import CoordConvNet


class InstanceCounter(nn.Module):

    r"""Instance Counter Module. Basically, it is a convolutional network
    to count instances for a given feature map.

    Args:
        input_n_filters (int): Number of channels in the input image
        use_coordinates (bool, optional): If `True`, adds coordinate
            information to input image and hidden state. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, 1)`

    Examples:
        >>> ins_cnt = InstanceCounter(3, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = ins_cnt(input)

        >>> ins_cnt = InstanceCounter(3, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = ins_cnt(input)
    """

    def __init__(self, input_n_filters, use_coordinates=False,
                 usegpu=True):
        super(InstanceCounter, self).__init__()

        self.input_n_filters = input_n_filters
        self.n_filters = 32
        self.use_coordinates = use_coordinates
        self.usegpu = usegpu

        self.__generate_cnn()

        self.output = nn.Sequential()
        self.output.add_module('linear', nn.Linear(self.n_filters,
                                                   1))
        self.output.add_module('sigmoid', nn.Sigmoid())

    def __generate_cnn(self):

        self.cnn = nn.Sequential()
        self.cnn.add_module('pool1', nn.MaxPool2d(2, stride=2))
        self.cnn.add_module('conv1', nn.Conv2d(self.input_n_filters,
                                               self.n_filters,
                                               kernel_size=(3, 3),
                                               stride=(1, 1),
                                               padding=(1, 1)))
        self.cnn.add_module('relu1', nn.ReLU())
        self.cnn.add_module('conv2', nn.Conv2d(self.n_filters,
                                               self.n_filters,
                                               kernel_size=(3, 3),
                                               stride=(1, 1),
                                               padding=(1, 1)))
        self.cnn.add_module('relu2', nn.ReLU())
        self.cnn.add_module('pool2', nn.MaxPool2d(2, stride=2))
        self.cnn.add_module('conv3', nn.Conv2d(self.n_filters,
                                               self.n_filters,
                                               kernel_size=(3, 3),
                                               stride=(1, 1),
                                               padding=(1, 1)))
        self.cnn.add_module('relu3', nn.ReLU())
        self.cnn.add_module('conv4', nn.Conv2d(self.n_filters,
                                               self.n_filters,
                                               kernel_size=(3, 3),
                                               stride=(1, 1),
                                               padding=(1, 1)))
        self.cnn.add_module('relu4', nn.ReLU())
        self.cnn.add_module('pool3', nn.AdaptiveAvgPool2d((1, 1)))
        # b, nf, 1, 1

        if self.use_coordinates:
            self.cnn = CoordConvNet(self.cnn, with_r=True,
                                    usegpu=self.usegpu)

    def forward(self, x):

        x = self.cnn(x)
        if self.use_coordinates:
            x = x[-1]
        x = x.squeeze(3).squeeze(2)
        x = self.output(x)

        return x
