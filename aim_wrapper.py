import copy
import os
import types

import numpy as np

_g_session = None
_g_context = {}

def init(exp_parent_dir, run_group=None):
    return
    global _g_session
    assert _g_session is None, 'aim_wrapper.init() should be called only once.'
    _g_session = Session(repo=os.path.realpath(os.path.abspath(exp_parent_dir)),
                         experiment=run_group,
                         flush_frequency=64)

def close():
    return
    global _g_session
    if _g_session is not None:
        _g_session.close()
        _g_session = None

def track(*args, **kwargs):
    return
    global _g_session
    global _g_context
    assert _g_session is not None, 'aim_wrapper.init() should be called before calling aim_wrapper.track().'
    return _g_session.track(*args, **kwargs, **_g_context)

def set_params(params, name, *, excluded_keys=[]):
    return
    global _g_session
    assert _g_session is not None, 'aim_wrapper.init() should be called before calling aim_wrapper.set_params().'
    params = make_aim_compatible(params, excluded_keys=excluded_keys)
    return _g_session.set_params(params=params, name=name)


# Makes given object compatible with aim for logging.
def make_aim_compatible(item, excluded_keys=[]):
    # https://stackoverflow.com/a/12569453/2182622
    if isinstance(item, (np.ndarray, np.generic)):
        return make_aim_compatible(item.tolist())

    # E.g. torch.relu, functions, ...
    if isinstance(item, (types.BuiltinFunctionType, types.FunctionType)):
        return item.__name__

    if not isinstance(item, (dict, list, tuple, set,)):
        if isinstance(item, (str, int, float, bool,)):
            return item
        return str(item)

    if isinstance(item, (list, tuple, set)):
        return type(item)([make_aim_compatible(i) for i in item])

    if isinstance(item, dict):
        return dict([
            (k if isinstance(k, (str, int, tuple)) else str(k), make_aim_compatible(v))
            for k, v in item.items()
            if k not in excluded_keys
        ])

    assert False


class AimContext:
    def __init__(self, context):
        self.context = context

    def __enter__(self):
        global _g_context
        self.prev_g_context = _g_context
        _g_context = self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _g_context
        _g_context = self.prev_g_context

def get_metric_prefixes():
    global _g_context
    prefix = ''
    if 'phase' in _g_context:
        prefix += _g_context['phase'].capitalize()
    if 'policy' in _g_context:
        prefix += {'sampling': 'Sp', 'option': 'Op'}.get(
                _g_context['policy'].lower(), _g_context['policy'].lower()).capitalize()

    if len(prefix) == 0:
        return '', ''

    return prefix + '/', prefix + '__'

def get_context():
    global _g_context
    return copy.copy(_g_context)

