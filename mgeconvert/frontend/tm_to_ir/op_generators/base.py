from abc import ABC

from ..tm_tensor_resolver import TensorNodeResolver

EXPR2OP = {}


def _register_op(*ops):
    def callback(impl):
        for op in ops:
            EXPR2OP[op] = impl
        return impl

    return callback


class OpGenBase(ABC):
    def __init__(self, expr, irgraph) -> None:
        self.expr = expr
        self.irgraph = irgraph
        self.resolver = TensorNodeResolver(self.irgraph)

    def get_opr(self):
        return self.op

    def add_opr_out_tensors(self):
        if len(self.op.inp_tensors) > 0:
            is_qat = (
                hasattr(self.op.inp_tensors[0], "scale")
                and self.op.inp_tensors[0].scale is not None
            )
        else:
            is_qat = False
        for o in self.expr.outputs:
            t = self.resolver.get_ir_tensor(o, owner_opr=self.op)
            if is_qat:
                t.scale = self.op.inp_tensors[0].scale
                t.zero_point = self.op.inp_tensors[0].zero_point
                t.q_dtype = self.op.inp_tensors[0].q_dtype
            self.op.add_out_tensors(t)
