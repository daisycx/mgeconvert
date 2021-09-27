from abc import ABC

import megengine.functional as F
import megengine.module as M
import megengine.module.qat as QAT
from megengine.traced_module.expr import CallFunction, CallMethod

from ....converter_ir.ir_op import Conv2dOpr
from ....converter_ir.ir_tensor import AxisOrder
from ..tm_utils import get_logger
from .base import EXPR2OP, OpGenBase, _register_op

logger = get_logger(__name__)


class GenConvBase(OpGenBase, ABC):
    def __init__(self, expr, irgraph, op_cls):
        super().__init__(expr, irgraph)
        if isinstance(expr, CallMethod):
            conv_module = expr.inputs[0].expr.value
            self.weight = conv_module.weight
            self.bias = conv_module.bias
            self.stride = conv_module.stride
            self.padding = conv_module.padding
            self.dilation = conv_module.dilation
            self.groups = conv_module.groups
        elif isinstance(expr, CallFunction):
            self.weight = None
            self.stride = self.expr.args[3]
            self.padding = self.expr.args[4]
            self.dilation = self.expr.args[5]
            self.groups = self.expr.args[6]
            if len(expr.args) > 7:
                assert self.expr.args[7] == "cross_correlation"
            if len(expr.args) > 8:
                assert self.expr.args[8] == "default"
        self.op = op_cls(self.stride, self.padding, self.dilation, self.groups,)

    def add_opr_vars(self, weight_format):
        self.add_weight_bias_tensors(weight_format)
        self.add_opr_out_tensors()

    def add_weight_bias_tensors(self, weight_format):
        if isinstance(self.expr, CallMethod):
            for i in self.expr.args[1:]:
                t = self.resolver.get_ir_tensor(i, user_opr=self.op)
                self.op.add_inp_tensors(t)
            self.add_const_inputs(weight_format)
        elif isinstance(self.expr, CallFunction):
            inp_tensor = self.resolver.get_ir_tensor(
                self.expr.args[0], user_opr=self.op
            )
            self.op.add_inp_tensors(inp_tensor)
            weight_tensor = self.resolver.get_ir_tensor(
                self.expr.args[1], user_opr=self.op, axis_order=weight_format,
            )
            weight_tensor.axis_order = weight_format
            self.op.add_inp_tensors(weight_tensor)
            if self.expr.args[2]:
                bias = self.expr.args[2]
                bias.shape = bias.shape[1]
                bias_tensor = self.resolver.get_ir_tensor(
                    bias, name=self.expr.args[0]._name + "_bias", user_opr=self.op
                )
                self.op.add_inp_tensors(bias_tensor)

    def add_const_inputs(self, weight_format):
        if self.weight is not None:
            weight_tensor = self.resolver.get_ir_tensor(
                self.weight,
                name=self.expr.inputs[0]._name + "_weight",
                user_opr=self.op,
                axis_order=weight_format,
            )
            weight_tensor.axis_order = weight_format
            self.op.add_inp_tensors(weight_tensor)
        if self.bias is not None:
            bias_tensor = self.resolver.get_ir_tensor(
                self.bias, name=self.expr.inputs[0]._name + "_bias", user_opr=self.op,
            )
            self.op.add_inp_tensors(bias_tensor)


@_register_op(M.Conv2d, F.conv2d)
class GenConv2dOpr(GenConvBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph, Conv2dOpr)
        self.add_opr_vars(AxisOrder.OIHW)


class GenQConvBase(GenConvBase):
    def __init__(self, expr, irgraph, op_cls):
        conv_module = expr.inputs[0].expr.value
        if hasattr(conv_module.weight_fake_quant, "get_qparams"):
            self.weight_qparams = conv_module.weight_fake_quant.get_qparams()
            self.weight_dtype = self.weight_qparams.dtype_meta.np_dtype_str
            self.act_qparams = conv_module.act_fake_quant.get_qparams()
            self.act_dtype = self.act_qparams.dtype_meta.np_dtype_str
        elif hasattr(conv_module.weight_observer, "get_qparams"):
            self.weight_qparams = conv_module.weight_observer.get_qparams()
            self.weight_dtype = self.weight_qparams.dtype_meta.np_dtype_str
            self.act_qparams = conv_module.act_observer.get_qparams()
            self.act_dtype = self.act_qparams.dtype_meta.np_dtype_str
        else:
            logger.error("Observer and FakeQuantize do not have get_qparams().")
        super().__init__(expr, irgraph, op_cls)

    def add_opr_out_tensors(self):
        for o in self.expr.outputs:
            out_tensor = self.resolver.get_ir_tensor(o, self.op)
            out_tensor.set_qparams(
                *self.resolver.resolve_qparams(
                    self.act_qparams.scale, self.act_qparams.zero_point
                )
            )
            out_tensor.q_dtype = self.act_dtype
            self.op.add_out_tensors(out_tensor)

    def add_const_inputs(self, weight_format):
        if self.weight is not None:
            weight_tensor = self.resolver.get_ir_tensor(
                self.weight,
                user_opr=self.op,
                name=self.expr.inputs[0]._name + "_weight",
                axis_order=weight_format,
            )
            weight_tensor.set_qparams(
                *self.resolver.resolve_qparams(
                    self.weight_qparams.scale, self.weight_qparams.zero_point
                )
            )
            weight_tensor.q_dtype = self.weight_dtype
            self.op.add_inp_tensors(weight_tensor)

        if self.bias is not None:
            bias_tensor = self.resolver.get_ir_tensor(
                self.bias, user_opr=self.op, name=self.expr.inputs[0]._name + "_bias"
            )
            bias_tensor.set_qparams(
                self.op.inp_tensors[0].scale * weight_tensor.scale, 0
            )
            bias_tensor.q_dtype = "int32"
            self.op.add_inp_tensors(bias_tensor)


@_register_op(QAT.Conv2d)
class GenQConv2dOpr(GenQConvBase):
    def __init__(self, expr, irgraph):
        super().__init__(expr, irgraph, Conv2dOpr)
        self.add_opr_vars(AxisOrder.OIHW)
