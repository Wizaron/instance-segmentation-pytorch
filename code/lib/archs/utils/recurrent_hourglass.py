from torch import nn
from torch.nn import functional as F
from conv_gru import ConvGRUCell
from coord_conv import CoordConv
from list_module import ListModule


class RecurrentHourglass(nn.Module):

    r"""RecurrentHourglass Module as defined in
    Instance Segmentation and Tracking with Cosine Embeddings and Recurrent
    Hourglass Networks (https://arxiv.org/pdf/1806.02070.pdf).

    Args:
        input_n_filters (int): Number of channels in the input image
        hidden_n_filters (int): Number of channels produced by Convolutional
            GRU module
        kernel_size (int or tuple): Size of the convolving kernels
        n_levels (int): Number of timesteps to unroll Convolutional GRU
            module
        embedding_size (int): Number of channels produced by Recurrent
            Hourglass module
        use_coordinates (bool, optional): If `True`, adds coordinate
            information to input image and hidden state. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, C_{out}, H_{in}, W_{in})`

    Examples:
        >>> hg = RecurrentHourglass(3, 16, 3, 5, 32, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = hg(input)

        >>> hg = RecurrentHourglass(3, 16, 3, 5, 32, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = hg(input)
    """

    def __init__(self, input_n_filters, hidden_n_filters, kernel_size,
                 n_levels, embedding_size, use_coordinates=False,
                 usegpu=True):
        super(RecurrentHourglass, self).__init__()

        assert n_levels >= 1, 'n_levels should be greater than or equal to 1.'

        self.input_n_filters = input_n_filters
        self.hidden_n_filters = hidden_n_filters
        self.kernel_size = kernel_size
        self.n_levels = n_levels
        self.embedding_size = embedding_size
        self.use_coordinates = use_coordinates
        self.usegpu = usegpu

        self.convgru_cell = ConvGRUCell(self.hidden_n_filters,
                                        self.hidden_n_filters,
                                        self.kernel_size,
                                        self.use_coordinates,
                                        self.usegpu)

        self.__generate_pre_post_convs()

    def __generate_pre_post_convs(self):

        if self.use_coordinates:
            def __get_conv(input_n_filters, output_n_filters):
                return CoordConv(input_n_filters, output_n_filters,
                                 self.kernel_size,
                                 padding=self.kernel_size // 2,
                                 with_r=True,
                                 usegpu=self.usegpu)
        else:
            def __get_conv(input_n_filters, output_n_filters):
                return nn.Conv2d(input_n_filters, output_n_filters,
                                 self.kernel_size,
                                 padding=self.kernel_size // 2)

        # Pre Conv Layers
        self.pre_conv_layers = [__get_conv(self.input_n_filters,
                                           self.hidden_n_filters), ]
        for _ in range(self.n_levels - 1):
            self.pre_conv_layers.append(__get_conv(self.hidden_n_filters,
                                                   self.hidden_n_filters))
        self.pre_conv_layers = ListModule(*self.pre_conv_layers)

        # Post Conv Layers
        self.post_conv_layers = [__get_conv(self.hidden_n_filters,
                                            self.embedding_size), ]
        for _ in range(self.n_levels - 1):
            self.post_conv_layers.append(__get_conv(self.hidden_n_filters,
                                                    self.hidden_n_filters))
        self.post_conv_layers = ListModule(*self.post_conv_layers)

    def forward_encoding(self, x):

        convgru_outputs = []
        hidden = None
        for i in range(self.n_levels):
            x = F.relu(self.pre_conv_layers[i](x))
            hidden = self.convgru_cell(x, hidden)
            convgru_outputs.append(hidden)

        return convgru_outputs

    def forward_decoding(self, convgru_outputs):

        _last_conv_layer = self.post_conv_layers[self.n_levels - 1]
        _last_output = convgru_outputs[self.n_levels - 1]

        post_feature_map = F.relu(_last_conv_layer(_last_output))
        for i in range(self.n_levels - 1)[::-1]:
            post_feature_map += convgru_outputs[i]
            post_feature_map = self.post_conv_layers[i](post_feature_map)
            post_feature_map = F.relu(post_feature_map)

        return post_feature_map

    def forward(self, x):

        x = self.forward_encoding(x)
        x = self.forward_decoding(x)

        return x

if __name__ == '__main__':
    from torch.autograd import Variable
    import torch
    import time

    def test(use_coordinates, usegpu):

        n_epochs, batch_size, image_size = 10, 4, 36

        input_n_filters, hidden_n_filters = 3, 64
        kernel_size, n_levels, embedding_size = 3, 4, 8

        hg = RecurrentHourglass(input_n_filters, hidden_n_filters,
                                kernel_size, n_levels,
                                embedding_size, use_coordinates,
                                usegpu)

        input = Variable(torch.rand(batch_size, input_n_filters,
                                    image_size, image_size))

        if usegpu:
            hg = hg.cuda()
            input = input.cuda()

        print hg

        output = hg(input)

        print input.size(), output.size()

    print '\n### CPU without Coords ###'
    test(False, False)
    print '\n### CPU with Coords ###'
    test(True, False)
    print '\n### GPU without Coords ###'
    test(False, True)
    print '\n### GPU with Coords ###'
    test(True, True)
