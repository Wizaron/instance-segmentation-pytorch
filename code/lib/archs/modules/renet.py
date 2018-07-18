import torch.nn as nn
from torch.nn import functional as F
from coord_conv import AddCoordinates


class ReNet(nn.Module):

    r"""ReNet Module as defined in 'ReNet: A Recurrent Neural
    Network Based Alternative to Convolutional Networks'
    (https://arxiv.org/pdf/1505.00393.pdf).

    Args:
        n_input (int): Number of channels in the input image
        n_units (int): Number of channels produced by ReNet
        patch_size (tuple): Patch size in the input of ReNet
        use_coordinates (bool, optional): If `True`, adds coordinate
            information to input image and hidden state. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{out}, W_{out})`

    Examples:
        >>> renet = ReNet(3, 16, (2, 2), True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = renet(input)

        >>> renet = ReNet(3, 16, (2, 2), True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = renet(input)
    """

    def __init__(self, n_input, n_units, patch_size=(1, 1),
                 use_coordinates=False, usegpu=True):
        super(ReNet, self).__init__()

        self.use_coordinates = use_coordinates
        self.usegpu = usegpu

        # Determine whether to do tiling and patch sizes
        self.patch_size_height = int(patch_size[0])
        self.patch_size_width = int(patch_size[1])

        assert self.patch_size_height >= 1
        assert self.patch_size_width >= 1

        self.tiling = False if ((self.patch_size_height == 1) and (
            self.patch_size_width == 1)) else True

        if self.use_coordinates:
            self.coord_adder = AddCoordinates(with_r=True,
                                              usegpu=self.usegpu)

        # Determine RNNs
        # Horizontal RNN
        rnn_hor_n_inputs = n_input * self.patch_size_height * \
            self.patch_size_width
        if self.use_coordinates:
            rnn_hor_n_inputs += 3

        self.rnn_hor = nn.GRU(rnn_hor_n_inputs, n_units,
                              num_layers=1, batch_first=True,
                              bidirectional=True)

        # Vertical RNN
        self.rnn_ver = nn.GRU(n_units * 2, n_units,
                              num_layers=1, batch_first=True,
                              bidirectional=True)

    def __tile(self, x):

        if (x.size(2) % self.patch_size_height) == 0:
            n_height_padding = 0
        else:
            n_height_padding = self.patch_size_height - \
                x.size(2) % self.patch_size_height
        if (x.size(3) % self.patch_size_width) == 0:
            n_width_padding = 0
        else:
            n_width_padding = self.patch_size_width - \
                x.size(3) % self.patch_size_width

        n_top_padding = n_height_padding / 2
        n_bottom_padding = n_height_padding - n_top_padding

        n_left_padding = n_width_padding / 2
        n_right_padding = n_width_padding - n_left_padding

        x = F.pad(x, (n_left_padding, n_right_padding,
                      n_top_padding, n_bottom_padding))

        b, n_filters, n_height, n_width = x.size()

        assert n_height % self.patch_size_height == 0
        assert n_width % self.patch_size_width == 0

        new_height = n_height / self.patch_size_height
        new_width = n_width / self.patch_size_width

        x = x.view(b, n_filters, new_height, self.patch_size_height,
                   new_width, self.patch_size_width)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.contiguous()
        x = x.view(b, new_height, new_width, self.patch_size_height *
                   self.patch_size_width * n_filters)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()

        return x

    def __swap_hw(self, x):

        # x : b, nf, h, w
        x = x.permute(0, 1, 3, 2)
        x = x.contiguous()
        #  x : b, nf, w, h

        return x

    def rnn_forward(self, x, hor_or_ver):

        # x : b, nf, h, w
        assert hor_or_ver in ['hor', 'ver']

        if hor_or_ver == 'ver':
            x = self.__swap_hw(x)

        x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
        b, n_height, n_width, n_filters = x.size()
        # x : b, h, w, nf

        x = x.view(b * n_height, n_width, n_filters)
        # x : b * h, w, nf
        if hor_or_ver == 'hor':
            x, _ = self.rnn_hor(x)
        elif hor_or_ver == 'ver':
            x, _ = self.rnn_ver(x)

        x = x.contiguous()
        x = x.view(b, n_height, n_width, -1)
        # x : b, h, w, nf

        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()
        # x : b, nf, h, w

        if hor_or_ver == 'ver':
            x = self.__swap_hw(x)

        return x

    def forward(self, x):

        # x : b, nf, h, w
        if self.tiling:
            x = self.__tile(x)

        if self.use_coordinates:
            x = self.coord_adder(x)

        x = self.rnn_forward(x, 'hor')
        x = self.rnn_forward(x, 'ver')

        return x
