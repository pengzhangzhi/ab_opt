import copy
import torch
# from torchvision.transforms import Compose


_TRANSFORM_DICT = {}

class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
    
def register_transform(name):
    def decorator(cls):
        _TRANSFORM_DICT[name] = cls
        return cls
    return decorator


def get_transform(cfg):
    if cfg is None or len(cfg) == 0:
        return None
    tfms = []
    for t_dict in cfg:
        t_dict = copy.deepcopy(t_dict)
        cls = _TRANSFORM_DICT[t_dict.pop('type')]
        tfms.append(cls(**t_dict))
    return Compose(tfms)


def _index_select(v, index, n):
    if isinstance(v, torch.Tensor) and v.size(0) == n:
        return v[index]
    elif isinstance(v, list) and len(v) == n:
        return [v[i] for i in index]
    else:
        return v


def _index_select_data(data, index):
    return {
        k: _index_select(v, index, data['aa'].size(0))
        for k, v in data.items()
    }


def _mask_select(v, mask):
    if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0):
        return v[mask]
    elif isinstance(v, list) and len(v) == mask.size(0):
        return [v[i] for i, b in enumerate(mask) if b]
    else:
        return v


def _mask_select_data(data, mask):
    return {
        k: _mask_select(v, mask)
        for k, v in data.items()
    }
