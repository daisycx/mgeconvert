import megengine as mge
import numpy as np
from megengine.traced_module.expr import Constant, GetAttr
from megengine.traced_module.node import TensorNode

from ...converter_ir.ir_graph import IRGraph
from ...converter_ir.ir_tensor import AxisOrder, IRTensor


class TensorNodeResolver:
    __const_id = 0

    def __init__(self, irgraph: IRGraph) -> None:
        self.irgraph = irgraph

    def resolve(self, inp, owner_opr=None, param_name=None, axis_order=AxisOrder.NCHW):
        self.axis_order = axis_order
        self.scale = None
        self.zero_point = None
        self.owner_opr = owner_opr

        if isinstance(inp, TensorNode):
            self.shape = inp.shape
            self.dtype = inp.dtype
            if isinstance(inp.expr, Constant):
                self.np_data = inp.expr.value.numpy()
            elif isinstance(inp.expr, GetAttr):
                self.np_data = getattr(inp.expr.owner, inp.expr.name).numpy()
            else:
                self.np_data = None
            self.name = inp._name
            self.ori_id = inp._id
        elif isinstance(inp, (int, float, list, np.ndarray)):
            self.np_data = np.array(inp)
            self.dtype = self.np_data.dtype.type
            self.shape = self.np_data.shape
            self.name = "const_val_" + str(TensorNodeResolver.__const_id)
            TensorNodeResolver.__const_id += 1
            self.ori_id = None
        elif isinstance(inp, mge.Tensor):
            self.name = param_name
            self.shape = inp.shape
            self.dtype = inp.dtype
            self.np_data = inp.numpy()
            self.ori_id = None

        return IRTensor(
            self.name,
            self.shape,
            self.dtype,
            scale=self.scale,
            zero_point=self.zero_point,
            np_data=self.np_data,
            owner_opr=self.owner_opr,
            axis=self.axis_order,
        )

    def get_ir_tensor(
        self, inp, owner_opr=None, user_opr=None, name=None, axis_order=AxisOrder.NCHW
    ):
        ir_tensor = self.resolve(
            inp, owner_opr=owner_opr, param_name=name, axis_order=axis_order
        )
        ori_tensor = self.irgraph.get_tensor(self.ori_id, ir_tensor)
        if user_opr is not None and user_opr not in ori_tensor.user_opr:
            ori_tensor.add_user_opr(user_opr)
        return ori_tensor

    def resolve_qparams(self, scale, zero_point):
        if isinstance(scale, mge.Tensor):
            scale = scale.numpy()
        if zero_point:
            if isinstance(scale, mge.Tensor):
                zero_point = zero_point.numpy()
        return scale, zero_point
