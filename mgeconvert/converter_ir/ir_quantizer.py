import json

import megengine
import numpy as np
from megengine.core._imperative_rt.core2 import apply
from megengine.core.ops.builtin import FakeQuant
from tmconvert.converter_ir.ir_tensor import IRTensor


class IRQuantizer(object):
    def __init__(self, require_quantize=False, param_fake_quant=False):
        super().__init__()
        self.require_quantize = require_quantize
        if require_quantize:
            assert (
                param_fake_quant
            ), "Doesn't support param fake quant when require_quantize is True"
        self.param_fake_quant = param_fake_quant

    # def get_dtype_bound(self, dt):

    def quantize(self, tensor: IRTensor):
        assert self.require_quantize, "This net do not require true quantize."
        value = tensor.np_data
        if isinstance(value, megengine.tensor):
            value = value.numpy()
        if tensor.scale:
            value = value / tensor.scale
            value = np.round(value)
        if tensor.zero_point:
            value += tensor.zero_point
        dt = (
            np.dtype(tensor.q_dtype)
            if isinstance(tensor.q_dtype, str)
            else tensor.q_dtype
        )
        if np.issubdtype(dt, np.integer):
            v_min = np.iinfo(dt).min
            v_max = np.iinfo(dt).max
            value = np.clip(value, v_min, v_max)
        value = value.astype(tensor.q_dtype)
        return value

    def save_quantize_params(self, tensors, path="quant_params.json"):
        quant_params = {}
        for t in tensors:
            dt = np.dtype(t.q_dtype)
            v_max, v_min = None, None
            is_weight = True if t.np_data is not None else False
            if np.issubdtype(dt, np.integer):
                v_min = np.iinfo(dt).min
                v_max = np.iinfo(dt).max
            if self.param_fake_quant and is_weight:
                if t.scale is not None:
                    inp = megengine.tensor(t.np_data)
                    scale = megengine.tensor(t.scale)
                    zp = float(t.zero_point) if t.zero_point else 0.0
                    zero_point = megengine.tensor(zp)
                    t.np_data = apply(
                        FakeQuant(qmin=v_min, qmax=v_max), inp, scale, zero_point
                    )[0].numpy()
            else:
                param = {
                    "dtype": str(dt),
                    "qmin": str(v_min),
                    "qmax": str(v_max),
                    "scale": str(t.scale),
                    "zero_point": str(t.zero_point),
                    "is_weight": is_weight,
                }
                quant_params[t.name] = param

        params = json.dumps(quant_params, indent=4)
        with open(path, "w") as f:
            f.write(params)
