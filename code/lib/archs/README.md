# Architectures

## ReSeg

* Proposed in [ReSeg: A Recurrent Neural Network-based Model for Semantic Segmentation](https://arxiv.org/pdf/1511.07053.pdf)
* Can be found at `reseg.py`

```
    ReSeg Module (with modifications) as defined in 'ReSeg: A Recurrent
    Neural Network-based Model for Semantic Segmentation'
    (https://arxiv.org/pdf/1511.07053.pdf).

    * VGG16 with skip Connections as base network
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
        >>> reseg = ReSeg(3, True, True, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> outputs = reseg(input)

        >>> reseg = ReSeg(3, True, True, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> outputs = reseg(input)
```

## Stacked Recurrent Hourglass

* Proposed in []()
* Can be found at `stacked_recurrent_hourglass.py`

```
    Stacked Recurrent Hourglass Module for instance segmentation
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
```
