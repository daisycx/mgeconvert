from typing import List, Sequence

import megengine
from megengine.core._imperative_rt.core2 import Tensor as RawTensor
from megengine.module.qat import QuantStub
from megengine.traced_module import TracedModule
from megengine.traced_module.expr import (
    Apply,
    CallFunction,
    CallMethod,
    Constant,
    GetAttr,
    Input,
)
from megengine.traced_module.module_tracer import BUILTIN_ARRAY_METHOD
from megengine.traced_module.node import ModuleNode, TensorNode

from ...converter_ir.ir_graph import IRGraph
from .op_generators import EXPR2OP
from .pattern_utils import DEFAULT_FUSION_PATTERNS
from .qat_pattern import find_match_pattern
from .tm_tensor_resolver import TensorNodeResolver
from .tm_utils import get_logger

logger = get_logger(__name__)


class TM_FrontEnd:
    def __init__(self, traced_module):
        if isinstance(traced_module, TracedModule):
            self.module = traced_module.flatten()
        elif isinstance(traced_module, str):
            self.module = megengine.load(traced_module)
        self.inputs: List[TensorNode] = self.module.graph.inputs[1:]
        self.outputs: List[TensorNode] = self.module.graph.outputs

        self.irgraph = IRGraph()
        self.tensor_resolver = TensorNodeResolver(self.irgraph)

    def resolve(self):
        self.add_net_inputs()
        self.get_all_oprs()
        self.add_net_outputs()
        return self.irgraph

    def add_net_inputs(self):
        for node in self.inputs:
            inp_tensor = self.tensor_resolver.get_ir_tensor(node, owner_opr=self)
            if node.qparams is not None:
                inp_tensor.set_qparams(
                    *self.tensor_resolver.resolve_qparams(
                        node.qparams.scale, node.qparams.zero_point
                    )
                )
                inp_tensor.q_dtype = node.qparams.dtype_meta.np_dtype_str
            self.irgraph.add_net_inputs(inp_tensor)

    def get_all_oprs(self):
        expr_iter = iter(self.module.graph._exprs)
        while True:
            try:
                expr = next(expr_iter)
                if isinstance(expr, Constant):
                    if isinstance(expr.value, RawTensor):
                        op_gen_cls = EXPR2OP.get("Constant")
                        op = op_gen_cls(expr, self.irgraph).get_opr()
                        self.irgraph.add_op(op)
                elif isinstance(expr, GetAttr):
                    if isinstance(expr.outputs[0], TensorNode):
                        op_gen_cls = EXPR2OP.get("Constant")
                        op = op_gen_cls(expr, self.irgraph).get_opr()
                        self.irgraph.add_op(op)
                elif isinstance(expr, CallMethod):
                    if expr.method in BUILTIN_ARRAY_METHOD:
                        # generate array_method op
                        op_gen_cls = EXPR2OP.get(expr.method, None)
                        assert op_gen_cls, "METHOD {} is not supported.".format(
                            expr.method
                        )
                        op = op_gen_cls(expr, self.irgraph).get_opr()
                        self.irgraph.add_op(op)
                    elif expr.method == "__new__":
                        # TODO
                        pass
                    elif expr.method == "__call__":
                        m = expr.inputs[0]
                        assert isinstance(m, ModuleNode)
                        assert isinstance(m.expr, Constant)
                        if isinstance(m.expr.value, TracedModule):
                            module = m.expr.value
                            assert module.is_qat
                            pats = find_match_pattern(module.graph)
                            pat, end_expr = pats[0]
                            fusion_op = DEFAULT_FUSION_PATTERNS.get(pat)
                            ops = fusion_op(
                                module,
                                end_expr,
                                expr,
                                self.irgraph,
                                self.tensor_resolver,
                            )
                            ops = (ops,) if not isinstance(ops, Sequence) else ops
                            # if len(ops) >1:
                            #     import pdb;pdb.set_trace()
                            for op in ops:
                                self.irgraph.all_oprs.append(op)
                                self.irgraph._opr_ids.append(id(op))
                        elif isinstance(m.expr.value, QuantStub):
                            module = m.expr.value
                            inp_tensor = self.tensor_resolver.get_ir_tensor(
                                expr.inputs[1]
                            )
                            out_tensor = self.irgraph.get_tensor(
                                expr.outputs[0]._id, None, origin_tensor=inp_tensor
                            )
                            qdtype = module.get_activation_dtype()
                            qparams = (
                                module.act_fake_quant.get_qparams()
                                if hasattr(module.act_fake_quant, "get_qparams")
                                else module.act_observer.get_qparams()
                            )
                            scale = qparams.scale
                            zero_point = qparams.zero_point
                            out_tensor.q_dtype = qdtype
                            out_tensor.scale = float(scale)
                            out_tensor.zero_point = (
                                int(zero_point) if zero_point else None
                            )

                        else:
                            op_gen_cls = EXPR2OP.get(type(m.expr.value), None)
                            assert op_gen_cls, "Module {} is not supported.".format(
                                type(m.expr.value)
                            )
                            op = op_gen_cls(expr, self.irgraph).get_opr()
                            self.irgraph.add_op(op)
                elif isinstance(expr, CallFunction):
                    f = expr.func  # expr.func.__module__ + "." + expr.func.__name__
                    op_gen_cls = EXPR2OP.get(f, None)
                    assert op_gen_cls, "FUNCTION {} is not supported.".format(f)
                    op = op_gen_cls(expr, self.irgraph).get_opr()
                    self.irgraph.add_op(op)
                elif isinstance(expr, Apply):
                    opdef = expr.opdef
                    op_gen_cls = EXPR2OP.get(str(opdef), None)
                    assert op_gen_cls, "OPDEF {} is not supported.".format(str(opdef))
                    op = op_gen_cls(expr, self.irgraph).get_opr()
                    self.irgraph.add_op(op)
                elif isinstance(expr, Input):
                    logger.warning("Do not suppot Input Expr.")
            except StopIteration:
                break

    def add_net_outputs(self):
        for node in self.outputs:
            assert (
                node._id in self.irgraph._tensor_ids
            ), "output node is not generated by any opr"
            out_tensor = self.tensor_resolver.get_ir_tensor(node, self)
            self.irgraph.add_net_outputs(out_tensor)