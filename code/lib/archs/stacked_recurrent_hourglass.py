import torch
import torch.nn as nn

from modules.vgg16 import VGG16
from modules.recurrent_hourglass import RecurrentHourglass
from modules.renet import ReNet
from instance_counter import InstanceCounter


class StackedRecurrentHourglass(nn.Module):

    r"""Stacked Recurrent Hourglass Module for instance segmentation
    as defined in 'Instance Segmentation and Tracking with Cosine
    Embeddings and Recurrent Hourglass Networks'
    (https://arxiv.org/pdf/1806.02070.pdf).

    * First four layers of VGG16
    * Two RecurrentHourglass layers
    * Two ReNet layers
    * Two transposed convolutional layers for upsampling
    * Three heads for semantic segmentation, instance segmentation and
        instance counting.

    Args:
        n_classes (int): Number of semantic classes
        use_instance_seg (bool, optional): If `False`, does not perform
            instance segmentation. Default: `True`
        pretrained (bool, optional): If `True`, initializes weights of the
            VGG16 using weights trained on ImageNet. Default: `True`
        use_coordinates (bool, optional): If `True`, adds coordinate
            information to input image and hidden state. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output:
            - Semantic Seg: `(N, N_{class}, H_{in}, W_{in})`
            - Instance Seg: `(N, 32, H_{in}, W_{in})`
            - Instance Cnt: `(N, 1)`

    Examples:
        >>> srhg = StackedRecurrentHourglass(4, True, True, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> outputs = srhg(input)

        >>> srhg = StackedRecurrentHourglass(4, True, True, True, True)
        >>> srhg = srhg.cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> outputs = srhg(input)
    """

    def __init__(self, n_classes, use_instance_seg=True, pretrained=True,
                 use_coordinates=False, usegpu=True):
        super(StackedRecurrentHourglass, self).__init__()

        self.n_classes = n_classes
        self.use_instance_seg = use_instance_seg
        self.use_coords = use_coordinates
        self.pretrained = pretrained
        self.usegpu = usegpu

        # Encoder
        # BaseCNN
        self.base_cnn = self.__generate_base_cnn()

        # Encoder Stacked Hourglass
        self.enc_stacked_hourglass = self.__generate_enc_stacked_hg(64, 3)

        # ReNets
        self.stacked_renet = self.__generate_stacked_renet(64, 2)

        # Decoder
        self.decoder = self.__generate_decoder(64)

        # Heads
        self.semantic_seg, self.instance_seg, self.instance_count = \
            self.__generate_heads(64, 32)

    def __generate_base_cnn(self):

        base_cnn = VGG16(n_layers=4, pretrained=self.pretrained,
                         use_coordinates=self.use_coords,
                         return_intermediate_outputs=False,
                         usegpu=self.usegpu)

        return base_cnn

    def __generate_enc_stacked_hg(self, input_n_filters, n_levels):

        stacked_hourglass = nn.Sequential()
        stacked_hourglass.add_module('Hourglass_1',
                                     RecurrentHourglass(
                                         input_n_filters=input_n_filters,
                                         hidden_n_filters=64,
                                         kernel_size=3,
                                         n_levels=n_levels,
                                         embedding_size=64,
                                         use_coordinates=self.use_coords,
                                         usegpu=self.usegpu))
        stacked_hourglass.add_module('pool_1',
                                     nn.MaxPool2d(2, stride=2))
        stacked_hourglass.add_module('Hourglass_2',
                                     RecurrentHourglass(
                                         input_n_filters=64,
                                         hidden_n_filters=64,
                                         kernel_size=3,
                                         n_levels=n_levels,
                                         embedding_size=64,
                                         use_coordinates=self.use_coords,
                                         usegpu=self.usegpu))
        stacked_hourglass.add_module('pool_2',
                                     nn.MaxPool2d(2, stride=2))

        return stacked_hourglass

    def __generate_stacked_renet(self, input_n_filters, n_renets):

        assert n_renets >= 1, 'n_renets should be 1 at least.'

        renet = nn.Sequential()
        renet.add_module('ReNet_1', ReNet(input_n_filters, 32,
                                          patch_size=(1, 1),
                                          use_coordinates=self.use_coords,
                                          usegpu=self.usegpu))
        for i in range(1, n_renets):
            renet.add_module('ReNet_{}'.format(i + 1),
                             ReNet(32 * 2, 32, patch_size=(1, 1),
                                   use_coordinates=self.use_coords,
                                   usegpu=self.usegpu))

        return renet

    def __generate_decoder(self, input_n_filters):

        decoder = nn.Sequential()
        decoder.add_module('ConvTranspose_1',
                           nn.ConvTranspose2d(input_n_filters,
                                              64,
                                              kernel_size=(2, 2),
                                              stride=(2, 2)))
        decoder.add_module('ReLU_1', nn.ReLU())
        decoder.add_module('ConvTranspose_2',
                           nn.ConvTranspose2d(64, 64,
                                              kernel_size=(2, 2),
                                              stride=(2, 2)))
        decoder.add_module('ReLU_2', nn.ReLU())

        return decoder

    def __generate_heads(self, input_n_filters, embedding_size):

        semantic_segmentation = nn.Sequential()
        semantic_segmentation.add_module('Conv_1',
                                         nn.Conv2d(input_n_filters,
                                                   self.n_classes,
                                                   kernel_size=(1, 1),
                                                   stride=(1, 1)))

        if self.use_instance_seg:
            instance_segmentation = nn.Sequential()
            instance_segmentation.add_module('Conv_1',
                                             nn.Conv2d(input_n_filters,
                                                       embedding_size,
                                                       kernel_size=(1, 1),
                                                       stride=(1, 1)))
        else:
            instance_segmentation = None

        instance_counting = InstanceCounter(input_n_filters,
                                            use_coordinates=self.use_coords,
                                            usegpu=self.usegpu)

        return semantic_segmentation, instance_segmentation, instance_counting

    def forward(self, x):

        x = self.base_cnn(x)
        x = self.enc_stacked_hourglass(x)
        x = self.stacked_renet(x)
        x = self.decoder(x)

        sem_seg_out = self.semantic_seg(x)
        if self.use_instance_seg:
            ins_seg_out = self.instance_seg(x)
        else:
            ins_seg_out = None

        ins_count_out = self.instance_count(x)

        return sem_seg_out, ins_seg_out, ins_count_out
