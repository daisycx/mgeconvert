from collections import OrderedDict
from enum import Enum
from functools import cmp_to_key
from typing import Set  # pylint: disable=unused-import
from typing import Callable, Dict, Sequence

import numpy as np
from megengine.core.tensor.dtype import qint32

from ..converter_ir.ir_graph import IRGraph
from .ir_op import (
    AddOpr,
    Conv2dOpr,
    Deconv2dOpr,
    ElemwiseOpr,
    ExpOpr,
    FuseMulAdd3Opr,
    GetSubTensorOpr,
    LeakyReluOpr,
    LinearOpr,
    MaxOpr,
    MinOpr,
    MulOpr,
    OpBase,
    PadOpr,
    ReduceOpr,
    ReluOpr,
    ReshapeOpr,
    ResizeOpr,
    SoftmaxOpr,
    SqueezeOpr,
    SubOpr,
    TanHOpr,
    TransposeOpr,
    TrueDivOpr,
    _PoolOpr,
)
from .ir_tensor import AxisOrder, IRTensor


class TransformerRule(Enum):
    # general rules
    NOPE = 1

    # for TFLite
    REDUCE_AXIS_AS_INPUT = 100
    REMOVE_RESHAPE_INPUT = 101
    # FUSE_FOR_RELU6 pass should happen before FUSE_ACTIVATION
    FUSE_FOR_RELU6 = 102  ##
    FUSE_ACTIVATION = 103
    CONV_ADD_ZERO_BIAS = 104
    DECONV_ADD_ZERO_BIAS = 105
    # DEPTHWISE_CONV_RESHAPE_WEIGHT requirs RESHAPE_BIAS_TO_1DIM
    DEPTHWISE_CONV_RESHAPE_WEIGHT = 106
    FUSE_SOFTMAX = 107
    # RESHAPE_BIAS_TO_1DIM should happen before DECONV_SHAPE_AS_INPUT
    RESHAPE_BIAS_TO_1DIM = 108
    DECONV_SHAPE_AS_INPUT = 109
    FUSE_ASTYPE = 110  ##
    PADDING_FOR_CONV_AND_POOLING = 111
    TRANSPOSE_PATTERN_AS_INPUT = 112
    # FUSE_FOR_LEAKY_RELU should happen before EXPAND_MUL_ADD3
    FUSE_FOR_LEAKY_RELU = 113
    EXPAND_MUL_ADD3 = 114
    EXPAND_ADD_SIGMOID = 115  ##
    FUSE_FOR_CONV_BIAS = 116
    FUSE_FOR_DECONV_BIAS = 117
    FUSE_FOR_FULLY_CONNECTED = 118  ##
    # for TFLite Converter
    SLICE_PARAMS_AS_INPUTS_AND_MAKE_SQUEEZE = 119
    RESIZE_PARAMS_AS_INPUT = 120

    # remove reshape
    REMOVE_RESHAPE_REALTED_OP = 121
    REMOVE_UNRELATED_IROP = 122


def cmp_rules(a, b):
    if a.value < b.value:
        return -1
    if a.value > b.value:
        return 1
    return 0


class IRTransform:
    def __init__(self, transformer_options):
        if not isinstance(transformer_options, Sequence):
            transformer_options = [
                transformer_options,
            ]

        # bias of depthwise_conv must be 1 dim
        if TransformerRule.DEPTHWISE_CONV_RESHAPE_WEIGHT in transformer_options:
            if TransformerRule.RESHAPE_BIAS_TO_1DIM not in transformer_options:
                transformer_options.append(TransformerRule.RESHAPE_BIAS_TO_1DIM)

        self.trans_options = sorted(transformer_options, key=cmp_to_key(cmp_rules))

    def transform(self, ir_graph):
        for option in self.trans_options:
            TRANSFORMMAP[option](ir_graph)
        return ir_graph


TRANSFORMMAP: Dict[Enum, Callable] = {}


def _register_tranformation_rule(transformer_option):
    def callback(impl):
        TRANSFORMMAP[transformer_option] = impl

    return callback


def cal_pad_mode(tm_opr):
    out_shape = tm_opr.out_tensors[0].shape
    inp_shape = tm_opr.inp_tensors[0].shape
    if out_shape[2:] == inp_shape[2:]:
        return "SAME"
    else:
        return "VALID"


@_register_tranformation_rule(TransformerRule.REMOVE_RESHAPE_INPUT)
def _remove_reshape_input(net):
    for op in net.all_oprs:
        if not isinstance(op, ReshapeOpr):
            continue

        if len(op.inp_tensors) == 2:
            del op.inp_tensors[1]
            # TODO： delete _tensor_ids, all_tensors


@_register_tranformation_rule(TransformerRule.TRANSPOSE_PATTERN_AS_INPUT)
def _transpose_pattern_as_input(net):
    for op in net.all_oprs:
        if not isinstance(op, TransposeOpr):
            continue

        perm_tensor = IRTensor(
            name=op.inp_tensors[0].name + "_perm",
            shape=np.array(op.pattern).shape,
            dtype=np.int32,
            np_data=np.array(op.pattern, dtype=np.int32),
            owner_opr=op,
            q_type=np.int32,
            axis=None,
        )
        op.add_inp_tensors(perm_tensor)


@_register_tranformation_rule(TransformerRule.REDUCE_AXIS_AS_INPUT)
def _reduce_axis_as_input(net):
    for op in net.all_oprs:
        if not isinstance(op, ReduceOpr):
            continue

        axis_tensor = IRTensor(
            name=op.inp_tensors[0].name + "_axis",
            shape=[1],
            dtype=np.int32,
            np_data=np.array(op.axis, dtype=np.int32),
            owner_opr=op,
            q_type=np.int32,
            axis=None,
        )
        op.add_inp_tensors(axis_tensor)


@_register_tranformation_rule(TransformerRule.PADDING_FOR_CONV_AND_POOLING)
def _make_padding(net: IRGraph):
    def have_padding(opr):
        if isinstance(opr, Conv2dOpr):
            if cal_pad_mode(opr) == "SAME":
                return False
        if hasattr(opr, "padding") and (opr.padding[0] > 0 or opr.padding[1] > 0):
            return True
        return False

    insert_intended = OrderedDict()  # type: OrderedDict
    for op in net.all_oprs:
        if not isinstance(op, (Conv2dOpr, _PoolOpr)):
            continue

        if have_padding(op):
            assert op.inp_tensors[0].ndim == 4, "ERROR: unsupported padding mode"
            np_data = np.array(
                [
                    0,
                    0,
                    op.padding[0],
                    op.padding[0],
                    op.padding[1],
                    op.padding[1],
                    0,
                    0,
                ],
                dtype=np.int32,
            )

            new_tensor_id = max(net._tensor_ids) + 1
            pad_in_tensor = IRTensor(
                name=op.inp_tensors[0].name + "_paddings",
                shape=[4, 2],
                dtype=np.int32,
                owner_opr=None,
                np_data=np_data,
                q_type=np.int32,
                axis=None,
            )
            net.add_tensor(new_tensor_id, pad_in_tensor)

            shape = list(op.inp_tensors[0].shape)
            new_tensor_id = max(net._tensor_ids) + 1
            pad_out_tensor = IRTensor(
                name=op.inp_tensors[0].name + "_pad_out",
                shape=[
                    shape[0],
                    shape[1],
                    shape[2] + op.padding[0] * 2,
                    shape[3] + op.padding[1] * 2,
                ],
                dtype=op.inp_tensors[0].dtype,
            )
            if (
                hasattr(op.inp_tensors[0], "scale")
                and op.inp_tensors[0].scale is not None
            ):
                pad_out_tensor.scale = op.inp_tensors[0].scale
                pad_out_tensor.q_dtype = op.inp_tensors[0].q_dtype
            if hasattr(op.inp_tensors[0], "zero_point"):
                pad_out_tensor.zero_point = op.inp_tensors[0].zero_point
            net.add_tensor(new_tensor_id, pad_out_tensor)

            pad_opr = PadOpr()
            pad_opr.inp_tensors = [op.inp_tensors[0], pad_in_tensor]
            index = op.inp_tensors[0].user_opr.index(op)
            op.inp_tensors[0].user_opr[index] = pad_opr
            pad_opr.out_tensors = [pad_out_tensor]
            pad_out_tensor.owner_opr = pad_opr
            op.inp_tensors = [pad_out_tensor] + op.inp_tensors[1:]
            pad_out_tensor.user_opr.append(op)

            index = net._opr_ids.index(id(op))
            insert_intended[index] = (id(pad_opr), pad_opr)

    for index, generated_pair in list(insert_intended.items())[::-1]:
        net._opr_ids.insert(index, generated_pair[0])
        net.all_oprs.insert(index, generated_pair[1])


@_register_tranformation_rule(TransformerRule.DECONV_SHAPE_AS_INPUT)
def _deconv_shape_as_input(net: IRGraph):
    for op in net.all_oprs:
        if not isinstance(op, Deconv2dOpr):
            continue

        result_shape = op.out_tensors[0].shape
        np_data = np.array(
            [result_shape[0], result_shape[2], result_shape[3], result_shape[1],],
            dtype=np.int32,
        )
        new_tensor_id = max(net._tensor_ids) + 1
        shape_symvar = IRTensor(
            name=op.inp_tensors[0].name + "_deconv_out_shape",
            shape=[4],
            dtype=np.int32,
            owner_opr=op,
            np_data=np_data,
            q_type=np.int32,
            axis=None,
        )
        shape_tensor = net.get_tensor(new_tensor_id, shape_symvar)
        if len(op.inp_tensors) == 2:
            op.inp_tensors = [
                shape_tensor,
                op.inp_tensors[1],
                op.inp_tensors[0],
            ]
        else:
            op.inp_tensors = [
                shape_tensor,
                op.inp_tensors[1],
                op.inp_tensors[0],
                op.inp_tensors[2],
            ]


@_register_tranformation_rule(TransformerRule.RESIZE_PARAMS_AS_INPUT)
def _resize_params_as_input(net):
    for op in net.all_oprs:
        if not isinstance(op, ResizeOpr):
            continue

        if len(op.inp_tensors) == 2:
            continue

        out_size_tensor = IRTensor(
            name=op.inp_tensors[0].name + "_out_size",
            shape=(2,),
            dtype=np.int32,
            np_data=np.array(op.out_size, dtype=np.int32),
            q_type=np.int32,
            axis=None,
        )
        op.add_inp_tensors(out_size_tensor)


@_register_tranformation_rule(TransformerRule.CONV_ADD_ZERO_BIAS)
def _add_bias_for_conv(net: IRGraph):
    for op in net.all_oprs:
        if not isinstance(op, Conv2dOpr):
            continue
        if len(op.inp_tensors) == 3:
            continue

        weight_shape = op.inp_tensors[1].shape
        bias_shape = (
            weight_shape[0]
            if len(weight_shape) == 4
            else weight_shape[0] * weight_shape[1]
        )
        bias_shape = (1, bias_shape, 1, 1)
        bias = np.zeros(bias_shape, dtype=np.float32)
        bias_tensor = IRTensor(
            name=op.inp_tensors[0].name + "_bias",
            shape=bias_shape,
            dtype=np.float32,
            np_data=bias,
            axis=AxisOrder.NCHW,
        )
        if op.inp_tensors[0].scale and op.inp_tensors[1].scale:
            bias_tensor.set_qparams(
                op.inp_tensors[0].scale * op.inp_tensors[1].scale, 0
            )
            bias_tensor.q_dtype = "int32"
        op.inp_tensors.append(bias_tensor)


@_register_tranformation_rule(TransformerRule.DECONV_ADD_ZERO_BIAS)
def _add_bias_for_deconv(net: IRGraph):
    for op in net.all_oprs:
        if not isinstance(op, Deconv2dOpr):
            continue
        if len(op.inp_tensors) == 3:
            continue

        weight_shape = op.inp_tensors[1].shape
        bias_shape = (
            weight_shape[1]
            if len(weight_shape) == 4
            else weight_shape[0] * weight_shape[2]
        )
        bias_shape = (1, bias_shape, 1, 1)
        bias = np.zeros(bias_shape, dtype=np.float32)
        bias_tensor = IRTensor(
            name=op.inp_tensors[0].name + "_bias",
            shape=bias_shape,
            dtype=np.float32,
            np_data=bias,
            axis=AxisOrder.NCHW,
        )
        if op.inp_tensors[0].scale and op.inp_tensors[1].scale:
            bias_tensor.set_qparams(
                op.inp_tensors[0].scale * op.inp_tensors[1].scale, 0
            )
            bias_tensor.q_dtype = "int32"
        op.inp_tensors.append(bias_tensor)


@_register_tranformation_rule(TransformerRule.RESHAPE_BIAS_TO_1DIM)
def _reshape_bias_to_1dim(net: IRGraph):
    for op in net.all_oprs:
        if not isinstance(op, (Deconv2dOpr, Conv2dOpr)):
            continue
        if len(op.inp_tensors) == 2:
            continue

        bias = op.inp_tensors[2]
        if bias.ndim == 4:
            bias.shape = (bias.shape[1],)
            bias.np_data = bias.np_data.reshape(-1)


@_register_tranformation_rule(TransformerRule.DEPTHWISE_CONV_RESHAPE_WEIGHT)
def _depthwise_conv_reshape_weight(net: IRGraph):
    # general group conv is not supported for TFLite
    for op in net.all_oprs:
        if not isinstance(op, Conv2dOpr):
            continue
        if op.groups == 1:
            continue

        weight = op.inp_tensors[1]  # G, oc/G, ic/G, kh, kw
        ic, cm = weight.shape[1] * op.groups, weight.shape[2]
        h, w = weight.shape[3:5]
        weight.shape = (ic, cm, h, w)  # oc, ic/G, kh, kw
        weight.np_data = weight.np_data.reshape(ic, cm, h, w)


@_register_tranformation_rule(TransformerRule.FUSE_ACTIVATION)
def _fuse_activation(net):
    delete_intended = []

    for op_id, op in zip(net._opr_ids, net.all_oprs):
        if isinstance(op, (ReluOpr, TanHOpr)):
            prev_ops = net.find_inp_oprs(op)
            if len(prev_ops) == 0:
                continue
            prev_op = prev_ops[0]
            if not isinstance(prev_op, OpBase):
                continue
            if prev_op.activation != "IDENTITY" or prev_op.name == "Deconv2d":
                continue

            activation = op.name.upper()
            prev_op.activation = activation
            prev_op.out_tensors = op.out_tensors

            delete_intended.append(net._opr_ids.index(op_id))

    for delete_idx in delete_intended[::-1]:
        net.delete_ops(delete_idx)


@_register_tranformation_rule(TransformerRule.SLICE_PARAMS_AS_INPUTS_AND_MAKE_SQUEEZE)
def _make_slice_as_inputs(net: IRGraph):
    for op in net.all_oprs:
        if not isinstance(op, GetSubTensorOpr):
            continue

        ndim = op.inp_tensors[0].ndim

        def make_input(axis, param, init_value):
            # make inputs: begin, end and step.
            ret = [init_value] * ndim
            for k, v in zip(axis, param):
                ret[k] = v
            ret = IRTensor(
                name=op.name + "_fake_input",
                shape=[len(ret)],
                dtype=np.int32,
                np_data=np.array(ret, dtype=np.int32),
                owner_opr=op,
                q_type=np.int32,
            )
            return ret

        begins_tensor = make_input(op.axis, op.begin_params, 0)
        ends_tensor = make_input(op.axis, op.end_params, np.iinfo(np.int32).max)
        steps_tensor = make_input(op.axis, op.step_params, 1)

        op.inp_tensors = [op.inp_tensors[0], begins_tensor, ends_tensor, steps_tensor]

        # TFLite slice do not support squeeze axis, so insert a squeeze opr here.
        # infer actual output shape of tflite slice
        desired_out_shape = op.out_tensors[0].shape
        actual_out_shape = [1] * ndim
        idx = 0
        for i in range(ndim):
            if i in op.squeeze_axis:
                continue
            actual_out_shape[i] = desired_out_shape[idx]
            idx += 1
        slice_out_tensor = IRTensor(
            name=op.name + "fake_output",
            shape=actual_out_shape,
            dtype=op.out_tensors[0].dtype,
            q_type=op.out_tensors[0].q_dtype,
            owner_opr=op,
        )
        old_out = op.out_tensors
        op.out_tensors = [slice_out_tensor]

        squeeze = SqueezeOpr(op.squeeze_axis)
        squeeze.inp_tensors = [slice_out_tensor]
        squeeze.out_tensors = old_out

        idx = net._opr_ids.index(id(op)) + 1
        net.add_op(squeeze, idx)


# caffe transormer rules
class PatternNode:
    def __init__(self, type, is_output=False, const_value=None):
        self.op = None
        self.type = type
        self.inp_oprs = []
        self.inp_const = []
        self.inp_tensors = []
        self.is_output = is_output
        self.const_value = const_value

    def check_const_value(self, op):
        inp_tensors = [v.np_data for v in op.inp_tensors]
        for const in self.const_value:
            idx = const[0]
            if idx == -1:
                find = False
                for index, v in enumerate(inp_tensors):
                    if np.array_equal(const[1], v):
                        find = True
                        del inp_tensors[index]
                        break
                if not find:
                    return False
            elif not np.array_equal(const[1], inp_tensors[idx]):
                return False
        return True


get_type = lambda op: type(op).__name__


def match(node, opr):
    node_queue = [node]
    opr_queue = [opr]
    matched_opr = set()
    matched_node = set()
    while len(node_queue) != 0:
        cur_node = node_queue.pop(0)
        cur_opr = opr_queue.pop(0)
        if cur_node.type != get_type(cur_opr) and cur_node.type != "*" or cur_opr.skip:
            return False
        if cur_node.op == None:
            cur_node.op = cur_opr
            if cur_node.const_value != None:
                if not cur_node.check_const_value(cur_opr):
                    return False
        elif cur_node.op != cur_opr:
            return False

        matched_opr.add(cur_opr)
        matched_node.add(cur_node)
        for i, var in enumerate(cur_opr.inp_tensors):
            if var.np_data is not None:
                cur_node.inp_const.append([i, var.np_data])
            else:
                cur_node.inp_tensors.append([i, var])
        if len(cur_node.inp_oprs) == 0:
            continue
        if len(cur_node.inp_oprs) != len(cur_opr.inp_oprs):
            return False

        for i, j in zip(cur_node.inp_oprs, cur_opr.inp_oprs):
            node_queue.append(i)
            opr_queue.append(j)

    for n in matched_node:
        if n.is_output:
            continue
        for op in n.op.out_oprs:
            if op not in matched_opr:
                return False

    return True


def get_softmax_axis(ndim: int) -> int:
    if ndim in (0, 1, 3):
        return 0
    return 1


@_register_tranformation_rule(TransformerRule.FUSE_SOFTMAX)
def _fuse_softmax(net: IRGraph):
    matches = OrderedDict()  # type: OrderedDict

    for op in net.all_oprs:
        if not isinstance(op, TrueDivOpr):
            continue
        try:
            prev_op = net.find_inp_oprs(op)[1]
            cur_index = net._opr_ids.index(id(op))
            if (
                not isinstance(prev_op, ReduceOpr)
                or prev_op.mode != "SUM"
                or prev_op.axis != get_softmax_axis(prev_op.inp_tensors[0].ndim)
                or net._opr_ids.index(id(prev_op)) != cur_index - 1
            ):
                continue
            prev_op = net.find_inp_oprs(op)[0]
            if (
                not isinstance(prev_op, ExpOpr)
                or net._opr_ids.index(id(prev_op)) != cur_index - 2
            ):
                continue
            prev_op = net.find_inp_oprs(prev_op)[0]
            if (
                not isinstance(prev_op, SubOpr)
                or net._opr_ids.index(id(prev_op)) != cur_index - 3
            ):
                continue

            prev_op = net.find_inp_oprs(prev_op)[1]
            if (
                not isinstance(prev_op, ReduceOpr)
                or prev_op.mode != "MAX"
                or prev_op.axis != get_softmax_axis(prev_op.inp_tensors[0].ndim)
                or net._opr_ids.index(id(prev_op)) != cur_index - 4
            ):
                continue
        except IndexError:  # doesn't match
            continue

        softmax_opr = SoftmaxOpr(axis=get_softmax_axis(prev_op.inp_tensors[0].ndim))
        softmax_opr.beta = 1
        softmax_opr.inp_tensors = prev_op.inp_tensors[:1]
        for i in softmax_opr.inp_tensors:
            i.user_opr.append(softmax_opr)
        softmax_opr.out_tensors = op.out_tensors
        softmax_out_oprs = net.find_out_oprs(op)
        matches[id(prev_op)] = (id(prev_op), softmax_opr, softmax_out_oprs)

    for original_id, generated_pair in list(matches.items())[::-1]:
        index = net._opr_ids.index(original_id)
        for out_op in generated_pair[2]:
            generated_pair[1].out_tensors[0].user_opr.append(out_op)

        del net._opr_ids[index : index + 5]
        del net.all_oprs[index : index + 5]

        net._opr_ids.insert(index, generated_pair[0])
        net.all_oprs.insert(index, generated_pair[1])


@_register_tranformation_rule(TransformerRule.FUSE_FOR_LEAKY_RELU)
def _fuse_leaky_relu(net: IRGraph):
    """
    Elemwise(ADD) + Elemwise(MUL) + Elemwise(MAX) + Elemwise(MIN) -> LeakyRelu
    """
    for opr in net.all_oprs:
        if (
            opr.name == "Add"
            and len(net.find_inp_oprs(opr)) == 2
            and net.find_inp_oprs(opr)[0].name == "Max"
            and net.find_inp_oprs(opr)[1].name == "Mul"
        ):
            max_op = net.find_inp_oprs(opr)[0]
            mul_op = net.find_inp_oprs(opr)[1]
            if not mul_op.inp_tensors[1].shape == (1,):
                continue
            if not max_op.inp_tensors[1].shape == (1,):
                continue
            if (
                len(net.find_inp_oprs(mul_op)) != 1
                or net.find_inp_oprs(mul_op)[0].name != "Min"
                or net.find_inp_oprs(mul_op)[0].inp_tensors[1].shape != (1,)
            ):
                continue
            min_op = net.find_inp_oprs(mul_op)[0]
            if not min_op.inp_tensors[1].shape == (1,):
                continue
            if max_op.inp_tensors[0] != min_op.inp_tensors[0]:
                continue
            leaky_relu = LeakyReluOpr(
                negative_slope=float(mul_op.inp_tensors[1].np_data)
            )
            leaky_relu.inp_tensors = [max_op.inp_tensors[0]]
            max_op.inp_tensors[0].user_opr.remove(max_op)
            max_op.inp_tensors[0].user_opr.remove(min_op)
            max_op.inp_tensors[0].user_opr.append(leaky_relu)
            leaky_relu.out_tensors = opr.out_tensors
            opr.out_tensors[0].owner_opr = leaky_relu

            index = net.all_oprs.index(max_op)
            del net.all_oprs[index : index + 4]
            del net._opr_ids[index : index + 4]
            net.add_op(leaky_relu, index)


@_register_tranformation_rule(TransformerRule.FUSE_FOR_CONV_BIAS)
def _fuse_for_conv_bias(net: IRGraph):
    """
    ConvolutionForward + Elemwise(ADD) -> ConvForwardBias
    """
    for opr in net.all_oprs:
        if (
            opr.name == "Conv2d"
            and len(net.find_out_oprs(opr)) == 1
            and net.find_out_oprs(opr)[0].name == "Add"
        ):
            bias_op = net.find_out_oprs(opr)[0]
            if not (
                (
                    bias_op.inp_tensors[1].np_data is not None
                    and len(bias_op.inp_tensors[1].np_data.reshape(-1))
                    == opr.inp_tensors[1].shape[0]
                )
                or (
                    (
                        bias_op.inp_tensors[0].np_data is not None
                        and len(bias_op.inp_tensors[0].np_data.reshape(-1))
                        == opr.inp_tensors[1].shape[0]
                    )
                )
            ):
                continue
            bias_idx = 0 if bias_op.inp_tensors[0].np_data is not None else 1
            if len(opr.inp_tensors) == 2:
                opr.inp_tensors.append(bias_op.inp_tensors[bias_idx])
            else:
                opr.inp_tensors[2].np_data += bias_op.inp_tensors[
                    bias_idx
                ].np_data.reshape(-1)
            if bias_op in opr.out_tensors[0].user_opr:
                opr.out_tensors[0].user_opr.remove(bias_op)
            bias_out_op = net.find_out_oprs(bias_op)
            if len(bias_out_op) > 0:
                for op in bias_out_op:
                    op.inp_tensors[0] = opr.out_tensors[0]
                    opr.out_tensors[0].user_opr.append(op)
            else:
                # last op of the graph
                assert bias_op.out_tensors[0] in net.graph_outputs
                index = net.graph_outputs.index(bias_op.out_tensors[0])
                net.graph_outputs[index] = opr.out_tensors[0]
            opr.activation = bias_op.activation
            index = net.all_oprs.index(bias_op)
            del net.all_oprs[index]
            del net._opr_ids[index]


@_register_tranformation_rule(TransformerRule.FUSE_FOR_DECONV_BIAS)
def _fuse_for_deconv_bias(net: IRGraph):
    for opr in net.all_oprs:
        if (
            opr.name == "Deconv2d"
            and len(net.find_out_oprs(opr)) == 1
            and net.find_out_oprs(opr)[0].name == "Add"
        ):
            bias_op = net.find_out_oprs(opr)[0]
            if not (
                (
                    bias_op.inp_tensors[1].np_data is not None
                    and len(bias_op.inp_tensors[1].np_data.reshape(-1))
                    == opr.inp_tensors[1].shape[1]
                )
                or (
                    (
                        bias_op.inp_tensors[0].np_data is not None
                        and len(bias_op.inp_tensors[0].np_data.reshape(-1))
                        == opr.inp_tensors[1].shape[1]
                    )
                )
            ):
                continue
            bias_idx = 0 if bias_op.inp_tensors[0].np_data is not None else 1
            if len(opr.inp_tensors) == 3:
                opr.inp_tensors.append(bias_op.inp_tensors[bias_idx])
            else:
                opr.inp_tensors[2].np_data += bias_op.inp_tensors[
                    bias_idx
                ].np_data.reshape(-1)
            if bias_op in opr.out_tensors[0].user_opr:
                opr.out_tensors[0].user_opr.remove(bias_op)
            bias_out_op = net.find_out_oprs(bias_op)
            if len(bias_out_op) > 0:
                for op in bias_out_op:
                    op.inp_tensors[0] = opr.out_tensors[0]
                    opr.out_tensors[0].user_opr.append(op)
            else:
                # last op of the graph
                assert bias_op.out_tensors[0] in net.graph_outputs
                index = net.graph_outputs.index(bias_op.out_tensors[0])
                net.graph_outputs[index] = opr.out_tensors[0]
            opr.activation = bias_op.activation
            index = net.all_oprs.index(bias_op)
            del net.all_oprs[index]
            del net._opr_ids[index]


@_register_tranformation_rule(TransformerRule.EXPAND_MUL_ADD3)
def _expand_mul_add3(net: IRGraph):

    for op in net.all_oprs:
        if not isinstance(op, FuseMulAdd3Opr):
            continue

        last_op = net.find_inp_oprs(op)
        assert len(last_op) == 1
        mul_out_tensor = IRTensor(
            name=op.inp_tensors[0].name + "_mul_out",
            shape=op.inp_tensors[0].shape,
            dtype=op.inp_tensors[0].dtype,
        )
        new_tensor_id = max(net._tensor_ids) + 1
        net.add_tensor(new_tensor_id, mul_out_tensor)

        mul_op = MulOpr()
        mul_out_tensor.owner_opr = mul_op
        mul_op.inp_tensors = op.inp_tensors[:2]
        for o in mul_op.inp_tensors:
            index = o.user_opr.index(op)
            o.user_opr[index] = mul_op
        mul_op.out_tensors = [mul_out_tensor]

        add_op = AddOpr()
        add_op.inp_tensors = [mul_out_tensor, op.inp_tensors[2]]
        mul_out_tensor.user_opr.append(add_op)
        add_op.out_tensors = op.out_tensors

        index = net._opr_ids.index(id(op))
        net.delete_ops(index)
        net.add_op(mul_op, index)
        net.add_op(add_op, index + 1)


@_register_tranformation_rule(TransformerRule.REMOVE_RESHAPE_REALTED_OP)
def _remove_reshape_tensors(net: IRGraph):
    for opr in net.all_oprs:
        if isinstance(opr, ReshapeOpr) and len(opr.inp_tensors) > 1:
            opr.inp_tensors = opr.inp_tensors[:1]


visited_tensor = set()  # type: set


def _dfs_recursive(op_set, tensor):
    owner_opr = tensor.owner_opr
    op_set.add(owner_opr)
    if tensor in visited_tensor:
        return
    visited_tensor.add(tensor)
    if isinstance(owner_opr, IRGraph) or owner_opr is None:
        return
    for tt in owner_opr.inp_tensors:
        _dfs_recursive(op_set, tt)


@_register_tranformation_rule(TransformerRule.REMOVE_UNRELATED_IROP)
def _remove_unrelated_op(net: IRGraph):
    match_sets = set()  # type: Set[OpBase]
    for out_tensor in net.graph_outputs:
        _dfs_recursive(match_sets, out_tensor)
    remove_idx = []
    for opr in net.all_oprs:
        if opr not in match_sets:
            index = net._opr_ids.index(id(opr))
            remove_idx.append(index)
    for i in remove_idx[::-1]:
        net.delete_ops(i)