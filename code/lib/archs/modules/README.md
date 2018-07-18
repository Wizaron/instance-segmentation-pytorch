# Modules

## Convolutional GRU

* Proposed in [Delving Deeper into Convolutional Networks for Learning Video Representations](https://arxiv.org/pdf/1511.06432.pdf)
* Can be found at `conv_gru.py`

```
    Convolutional GRU Module as defined in 'Delving Deeper into
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
```

## Coordinate Convolution

* Proposed in [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/pdf/1807.03247.pdf)
* Can be found at `coord_conv.py`

```
    Coordinate Adder Module as defined in 'An Intriguing Failing of
    Convolutional Neural Networks and the CoordConv Solution'
    (https://arxiv.org/pdf/1807.03247.pdf).

    This module concatenates coordinate information (`x`, `y`, and `r`) with
    given input tensor.

    `x` and `y` coordinates are scaled to `[-1, 1]` range where origin is the
    center. `r` is the Euclidean distance from the center and is scaled to
    `[0, 1]`.

    Args:
        with_r (bool, optional): If `True`, adds radius (`r`) coordinate
            information to input image. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: `(N, (C_{in} + 2) or (C_{in} + 3), H_{in}, W_{in})`

    Examples:
        >>> coord_adder = AddCoordinates(True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = coord_adder(input)

        >>> coord_adder = AddCoordinates(True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = coord_adder(input)
```

## Recurrent Hourglass

* Proposed in [Instance Segmentation and Tracking with Cosine Embeddings and Recurrent Hourglass Networks](https://arxiv.org/pdf/1806.02070.pdf)
* Can be found at `recurrent_hourglass.py`

```
    RecurrentHourglass Module as defined in
    'Instance Segmentation and Tracking with Cosine Embeddings and Recurrent
    Hourglass Networks' (https://arxiv.org/pdf/1806.02070.pdf).

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
```

## ReNet

* Proposed in [ReNet: A Recurrent Neural Network Based Alternative to Convolutional Networks](https://arxiv.org/pdf/1505.00393.pdf)
* Can be found at `renet.py`

```
    ReNet Module as defined in 'ReNet: A Recurrent Neural
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
```

## VGG16

* Proposed in [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
* Can be found at `vgg16.py`

```
    A module that augments VGG16 as defined in 'Very Deep Convolutional
    Networks for Large-Scale Image Recognition'
    (https://arxiv.org/pdf/1409.1556.pdf).

    1. It can return first `n_layers` of the VGG16.
    2. It can add coordinates to feature maps prior to each convolution.
    3. It can return all outputs (including intermediate outputs) of the
        VGG16.

    Args:
        n_layers (int): Use first `n_layers` layers of the VGG16
        pretrained (bool, optional): If `True`, initializes weights of the
            VGG16 using weights trained on ImageNet. Default: `True`
        use_coordinates (bool, optional): If `True`, adds `x`, `y` and radius
            (`r`) coordinates to feature maps prior to each convolution.
            Weights to process these coordinates are initialized as zero.
            Default: `False`
        return_intermediate_outputs (bool, optional): If `True`, return
            outputs of the each layer in the VGG16 as a list otherwise
            return output of the last layer of first `n_layers` layers
            of the VGG16. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: Output of the last layer of the selected subpart of VGG16
            or the list that contains outputs of the each layer depending on
            `return_intermediate_outputs`

    Examples:
        >>> vgg16 = VGG16(16, True, True, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = vgg16(input)

        >>> vgg16 = VGG16(16, True, True, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = vgg16(input)
```
