from abc import ABC

from mgeconvert.converter_ir.ir_tensor import AxisOrder

from ....converter_ir.ir_op import (
    AbsOpr,
    AddOpr,
    CeilOpr,
    ExpOpr,
    FloorOpr,
    FuseMulAdd3Opr,
    LeakyReluOpr,
    LogOpr,
    MaxOpr,
    MinOpr,
    MulOpr,
    PowOpr,
    ReluOpr,
    SigmoidOpr,
    SubOpr,
    TanHOpr,
    TrueDivOpr,
)
from ..mge_utils import get_shape
from .base import OpGenBase, _register_op


class GenFuseMulAdd3Oprs(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        self.op = FuseMulAdd3Opr()
        self.add_tensors(mge_opr)


mode_opr_map = {
    "add": AddOpr,
    "fuse_add_relu": AddOpr,
    "sigmoid": SigmoidOpr,
    "mul": MulOpr,
    "abs": AbsOpr,
    "ceil": CeilOpr,
    "exp": ExpOpr,
    "floor": FloorOpr,
    "log": LogOpr,
    "max": MaxOpr,
    "min": MinOpr,
    "pow": PowOpr,
    "relu": ReluOpr,
    "sub": SubOpr,
    "tanh": TanHOpr,
    "true_div": TrueDivOpr,
    "fuse_mul_add3": GenFuseMulAdd3Oprs,
}


@_register_op("Elemwise")
class GenElemwiseOpr(OpGenBase):
    def __init__(self, mge_opr, irgraph):
        super().__init__(mge_opr, irgraph)
        try:
            self.mode = self.params["mode"]
        except RuntimeError:
            self.mode = "NONE"
        if self.mode.lower() in ["fuse_mul_add3"]:
            self.op = mode_opr_map[self.mode.lower()](mge_opr, irgraph).get_opr()
        else:
            self.op = mode_opr_map[self.mode.lower()]()
            if "RELU" in self.mode:
                self.op.activation = "RELU"
            self.add_tensors(mge_opr)
