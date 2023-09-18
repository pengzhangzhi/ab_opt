
_MODEL_DICT = {}


def register_model(name):
    def decorator(cls):
        _MODEL_DICT[name] = cls
        return cls
    return decorator


def get_model(cfg):
    if cfg.type == 'diff_gnn':
        return _MODEL_DICT[cfg.type](**cfg)
    else:
         return _MODEL_DICT[cfg.type](cfg)
