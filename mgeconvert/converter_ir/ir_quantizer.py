import json

import megengine
import numpy as np
from mgeconvert.converter_ir.ir_tensor import IRTensor


class IRQuantizer(object):
    def __init__(self, require_quantize=False):
        super().__init__()
        self.require_quantize = require_quantize
    
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

    def save_quantize_params(self, irgraph, path="quant_params.json"):
        all_tensors = set()
        for opr in irgraph.all_oprs:
            for t in opr.inp_tensors + opr.out_tensors:
                all_tensors.add(t)
        quant_params = {}
        for t in all_tensors:
            dt = np.dtype(t.q_dtype)
            v_max, v_min = None, None
            is_weight = True if t.np_data is not None else False
            if np.issubdtype(dt, np.integer):
                v_min = np.iinfo(dt).min
                v_max = np.iinfo(dt).max
            param = {
                "dtype": str(dt),
                "qmin": str(v_min),
                "qmax": str(v_max),
                "scale": str(t.scale),
                "zero_point": str(t.zero_point),
                "is_weight": is_weight
            }
            quant_params[t.name] = param

        params = json.dumps(quant_params, indent=4)
        with open(path, "w") as f:
            f.write(params)
