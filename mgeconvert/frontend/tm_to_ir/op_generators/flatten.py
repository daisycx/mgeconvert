import megengine.functional as F
from megengine.traced_module.expr import CallFunction

from ....converter_ir.ir_op import FlattenOpr
from ..tm_utils import get_logger
from .base import OpGenBase, _register_op

logger = get_logger(__name__)


@_register_op(F.flatten)
class GenFlattenOpr(OpGenBase):
    def __init__(self, expr, irgraph) -> None:
        super().__init__(expr, irgraph)
        assert isinstance(self.expr, CallFunction)
        self.start_axis = 0
        self.end_axis = -1
        if len(expr.args) >= 2:
            self.start_axis = int(expr.args[1])
        if len(expr.args) >= 3:
            self.end_axis = int(expr.args[2])
        self.op = FlattenOpr(self.start_axis, self.end_axis)
        self.add_opr_vars()

    def add_opr_vars(self):
        inp = self.expr.args[0]
        inp_tensor = self.resolver.get_ir_tensor(inp, user_opr=self.op)
        self.op.add_inp_tensors(inp_tensor)
        self.add_opr_out_tensors()
