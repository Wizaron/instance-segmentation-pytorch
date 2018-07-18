from torch import nn

# https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219


class ListModule(nn.Module):

    def __init__(self, *args):
        super(ListModule, self).__init__()

        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):

        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)

        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)
