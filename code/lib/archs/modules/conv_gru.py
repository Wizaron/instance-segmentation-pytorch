import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from coord_conv import CoordConv

# Adapted from: https://github.com/bionick87/ConvGRUCell-pytorch


class ConvGRUCell(nn.Module):

    r"""Convolutional GRU Module as defined in 'Delving Deeper into
    Convolutional Networks for Learning Video Representations'
    (https://arxiv.org/pdf/1511.06432.pdf).

    Args:
        input_size (int): Number of channels in the input image
        hidden_size (int): Number of channels produced by the ConvGRU
        kernel_size (int or tuple): Size of the convolving kernel
        use_coordinates (bool, optional): If `True`, adds coordinate
            information to input image and hidden state. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input:
            - `x` : `(N, C_{in}, H_{in}, W_{in})`
            - `hidden` : `(N, C_{out}, H_{in}, W_{in})` or `None`
        - Output: `next_hidden` : `(N, C_{out}, H_{in}, W_{in})`

    Examples:
        >>> n_hidden = 16
        >>> conv_gru = ConvGRUCell(3, n_hidden, 3, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> hidden = torch.rand(8, n_hidden, 64, 64)
        >>> output = conv_gru(input, None)
        >>> output = conv_gru(input, hidden)

        >>> n_hidden = 16
        >>> conv_gru = ConvGRUCell(3, n_hidden, 3, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> hidden = torch.rand(8, n_hidden, 64, 64).cuda()
        >>> output = conv_gru(input, None)
        >>> output = conv_gru(input, hidden)
    """

    def __init__(self, input_size, hidden_size, kernel_size,
                 use_coordinates=False, usegpu=True):
        super(ConvGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.use_coordinates = use_coordinates
        self.usegpu = usegpu

        _n_inputs = self.input_size + self.hidden_size
        if self.use_coordinates:
            self.conv_gates = CoordConv(_n_inputs,
                                        2 * self.hidden_size,
                                        self.kernel_size,
                                        padding=self.kernel_size // 2,
                                        with_r=True,
                                        usegpu=self.usegpu)

            self.conv_ct = CoordConv(_n_inputs, self.hidden_size,
                                     self.kernel_size,
                                     padding=self.kernel_size // 2,
                                     with_r=True,
                                     usegpu=self.usegpu)
        else:
            self.conv_gates = nn.Conv2d(_n_inputs,
                                        2 * self.hidden_size,
                                        self.kernel_size,
                                        padding=self.kernel_size // 2)

            self.conv_ct = nn.Conv2d(_n_inputs, self.hidden_size,
                                     self.kernel_size,
                                     padding=self.kernel_size // 2)

    def forward(self, x, hidden):

        batch_size, _, height, width = x.size()

        if hidden is None:
            size_h = [batch_size, self.hidden_size, height, width]
            hidden = Variable(torch.zeros(size_h))

            if self.usegpu:
                hidden = hidden.cuda()

        c1 = self.conv_gates(torch.cat((x, hidden), dim=1))
        rt, ut = c1.chunk(2, 1)

        reset_gate = F.sigmoid(rt)
        update_gate = F.sigmoid(ut)

        gated_hidden = torch.mul(reset_gate, hidden)

        ct = F.tanh(self.conv_ct(torch.cat((x, gated_hidden), dim=1)))

        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct

        return next_h


if __name__ == '__main__':
    def test(use_coordinates, usegpu):
        n_timesteps, batch_size, n_channels = 3, 8, 3
        hidden_size, kernel_size, image_size = 64, 3, 32
        max_epoch = 10

        model = ConvGRUCell(n_channels, hidden_size, kernel_size,
                            use_coordinates, usegpu)

        input = Variable(torch.rand(batch_size, n_channels,
                                    image_size, image_size))
        hidden = Variable(torch.rand(batch_size, hidden_size,
                                     image_size, image_size))

        if usegpu:
            model = model.cuda()
            input = input.cuda()
            hidden = hidden.cuda()

        print '\n* Model :\n\n', model

        out1 = model(input, None)
        out2 = model(input, hidden)

        print '\n* Success!'

    print '\n### CPU without coordinates ###'
    test(False, False)
    print '\n### CPU with coordinates ###'
    test(True, False)
    print '\n### GPU without coordinates ###'
    test(False, True)
    print '\n### GPU with coordinates ###'
    test(True, True)
