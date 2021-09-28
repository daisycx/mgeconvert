from typing import List  # pylint: disable=unused-import

from .ir_tensor import IRTensor  # pylint: disable=unused-import


class OpBase:
    skip = False
    name = ""

    def __init__(self) -> None:
        self.inp_tensors = []  # type: List[IRTensor]
        self.out_tensors = []  # type: List[IRTensor]
        self.activation = "IDENTITY"

    def add_inp_tensors(self, ir_tensor):
        self.inp_tensors.append(ir_tensor)

    def add_out_tensors(self, ir_tensor):
        self.out_tensors.append(ir_tensor)


####################  conv related ############################
class _ConvOpr(OpBase):
    def __init__(self, stride, padding, dilation, groups):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups


class Conv2dOpr(_ConvOpr):
    name = "Conv2d"


class Deconv2dOpr(_ConvOpr):
    name = "Deconv2d"


class ConvolutionBackwardFilterOpr(OpBase):
    name = "ConvolutionBackwardFilter"

    def __init__(
        self, stride, padding, dilation, group, kernel_shape, src_shape, grad_out_shape
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.group = group
        self.kernel_shape = kernel_shape
        self.src_shape = src_shape
        self.grad_out_shape = grad_out_shape


class _PoolOpr(OpBase):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


class MaxPool2dOpr(_PoolOpr):
    name = "MaxPool2d"

    def __init__(self, kernel_size, stride, padding):
        super().__init__(kernel_size, stride, padding)
        self.mode = "MAX"


class AvgPool2dOpr(_PoolOpr):
    name = "AvgPool2d"

    def __init__(
        self, kernel_size, stride, padding, mode="AVERAGE_COUNT_EXCLUDE_PADDING"
    ):
        super().__init__(kernel_size, stride, padding)
        self.mode = mode


class PadOpr(OpBase):
    name = "Pad"


class BatchNormalizationOpr(OpBase):
    name = "BatchNormalization"

    def __init__(self, output_idx=0):
        super().__init__()
        self.output_idx = output_idx


class AdaptiveAvgPool2dOpr(OpBase):
    name = "AdaptiveAvgPool2d"

    def __init__(self, out_shape):
        super().__init__()
        self.out_shape = out_shape


####################  math related ############################


class MatMulOpr(OpBase):
    name = "MatMul"

    def __init__(
        self,
        transpose_a=False,
        transpose_b=False,
        compute_mode="default",
        format="default",
    ):
        super().__init__()
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.compute_mode = compute_mode
        self.format = format


class LinearOpr(MatMulOpr):
    name = "Linear"

    def __init__(self, has_bias=False):
        super().__init__(transpose_b=True)
        self.has_bias = has_bias


class ReduceOpr(OpBase):
    name = "Reduce"

    def __init__(self, axis, mode, keep_dims):
        super().__init__()
        self.axis = axis
        self.mode = mode
        self.keep_dims = keep_dims


class SoftmaxOpr(OpBase):
    name = "Softmax"

    def __init__(self, axis=None, beta=1):
        super().__init__()
        self.axis = axis
        self.beta = beta


####################  tensor related ############################
class FlattenOpr(OpBase):
    name = "Flatten"

    def __init__(self, start_axis=0, end_axis=-1):
        super().__init__()
        self.start_axis = start_axis
        self.end_axis = end_axis


class DropoutOpr(OpBase):
    name = "Dropout"

    def __init__(self, drop_prob=0, training=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.training = training


class ConstantOpr(OpBase):
    name = "Constant"


class MultipleDeviceTensorHolderOpr(OpBase):
    name = "MultipleDeviceTensorHolder"


class SharedDeviceTensorOpr(OpBase):
    name = "SharedDeviceTensorOpr"


class VolatileSharedDeviceTensorOpr(OpBase):
    name = "VolatileSharedDeviceTensor"


class GetVarShapeOpr(OpBase):
    name = "GetVarShape"


class IndexingOneHotOpr(OpBase):
    name = "IndexingOneHotOpr"


class LinspaceOpr(OpBase):
    name = "Linspace"


class WarpPerspectiveForwardOpr(OpBase):
    name = "WarpPerspectiveForward"


class IdentityOpr(OpBase):
    name = "Identity"

    def __init__(self):
        super().__init__()
        self.mode = "Identity"


class ConcatOpr(OpBase):
    name = "Concat"

    def __init__(self, axis):
        super().__init__()
        self.axis = axis


class ReshapeOpr(OpBase):
    name = "Reshape"

    def __init__(self, out_shape):
        super().__init__()
        self.out_shape = out_shape


class TransposeOpr(OpBase):
    name = "Transpose"

    def __init__(self, pattern: list):
        super().__init__()
        self.pattern = pattern


class SqueezeOpr(OpBase):
    name = "Squeeze"

    def __init__(self, squeeze_dims):
        super().__init__()
        self.squeeze_dims = squeeze_dims


class GetSubTensorOpr(OpBase):
    name = "GetSubTensor"

    def __init__(self, axis, begin_params, end_params, step_params, squeeze_axis=None):
        super().__init__()
        self.axis = axis
        self.begin_params = begin_params
        self.end_params = end_params
        self.step_params = step_params
        self.squeeze_axis = squeeze_axis


class ResizeOpr(OpBase):
    name = "Resize"

    def __init__(
        self, out_size, scale_factor=None, mode="bilinear", align_corners=None
    ):
        super().__init__()
        self.out_size = out_size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners


class AxisAddRemoveOpr(OpBase):
    name = "AxisAddRemove"

    def __init__(self, out_shape, desc):
        super().__init__()
        self.out_shape = out_shape
        self.desc = desc


class BroadcastOpr(OpBase):
    name = "Broadcast"


############################ elemwise ########################


class ElemwiseOpr(OpBase):
    pass


class AddOpr(ElemwiseOpr):
    name = "Add"


class SubOpr(ElemwiseOpr):
    name = "Sub"


class MulOpr(ElemwiseOpr):
    name = "Mul"


class TrueDivOpr(ElemwiseOpr):
    name = "TrueDiv"


class PowOpr(ElemwiseOpr):
    name = "Pow"


class ExpOpr(ElemwiseOpr):
    name = "Exp"


class FloorOpr(ElemwiseOpr):
    name = "Floor"


class FloorDivOpr(ElemwiseOpr):
    name = "FloorDiv"


class CeilOpr(ElemwiseOpr):
    name = "Ceil"


class MaxOpr(ElemwiseOpr):
    name = "Max"


class MinOpr(ElemwiseOpr):
    name = "Min"


class AbsOpr(ElemwiseOpr):
    name = "Abs"


class LogOpr(ElemwiseOpr):
    name = "Log"


class FuseMulAdd3Opr(OpBase):
    name = "FuseMulAdd3"


############################# activation ###########################


class Relu6Opr(OpBase):
    name = "Relu6"


class ReluOpr(OpBase):
    name = "Relu"


class SigmoidOpr(OpBase):
    name = "Sigmoid"


class HardSigmoidOpr(OpBase):
    name = "HardSigmoid"


class SiLUOpr(OpBase):
    name = "SiLU"


class TanHOpr(OpBase):
    name = "TanH"


class LeakyReluOpr(OpBase):
    name = "LeakyRelu"

    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope


class TypeCvtOpr(OpBase):
    name = "TypeCvt"

    def __init__(self, out_dtype):
        super().__init__()
        self.out_dtype = out_dtype


class HardSwishOpr(OpBase):
    name = "HardSwish"


class RepeatOpr(OpBase):
    name = "Repeat"

    def __init__(self, repeats, axis):
        super().__init__()
        self.repeats = repeats
        self.axis = 0 if axis is None else axis