from typing import Sequence, Union

import megengine as mge
from megengine.core.tensor import dtype
from megengine.quantization.utils import create_qparams
from megengine.traced_module import TracedModule
from mgeconvert.backend.ir_to_tflite import TFLiteConverter, set_platform
from mgeconvert.converter_ir.ir_quantizer import IRQuantizer
from mgeconvert.converter_ir.ir_transform import IRTransform, TransformerRule
from mgeconvert.frontend import TM_FrontEnd


def tracedmodule_to_tflite(
    traced_module,
    output="out.tflite",
    *,
    data_type: str = None,
    scales: Union[float, Sequence] = None,
    zero_points: Union[int, Sequence] = None,
    require_quantize=False,
    param_fake_quant=False,
    quantize_file_path="quant_params.json",
    graph_name="graph",
    mtk=False,
):
    """
    Convert traced model to TFLite,
    and save the TFLite model to file `output`.

    :param traced_module: a traced module or the file path of a traced module.
    :param output: the filename used for the saved model.
    :param data_type: data type of input

    :param graph_name: the name of the TFLite graph.
    :param mtk: if this TFLite will be run on mtk.
    :type mtk: bool
    """
    if isinstance(traced_module, str):
        traced_module = mge.load(traced_module)
    assert isinstance(
        traced_module, TracedModule
    ), "Input should be a traced module or a path of traced module."

    if data_type is not None:
        for i in range(len(traced_module.graph.inputs[1:])):
            if traced_module.graph.inputs[i + 1].qparams is None:
                traced_module.graph.inputs[i + 1].qparams = create_qparams()
            traced_module.graph.inputs[
                i + 1
            ].qparams.dtype_meta = dtype._builtin_quant_dtypes[data_type]
    if scales is not None:
        if not isinstance(scales, Sequence):
            scales = (scales,)
        for i in range(len(traced_module.graph.inputs[1:])):
            scale = scales[i] if i < len(scales) else scales[-1]
            traced_module.graph.inputs[i + 1].qparams.scale = mge.tensor(float(scale))
    if zero_points is not None:
        if not isinstance(zero_points, Sequence):
            zero_points = (zero_points,)
        for i in range(len(traced_module.graph.inputs[1:])):
            zero_point = zero_points[i] if i < len(zero_points) else zero_points[-1]
            traced_module.graph.inputs[i + 1].qparams.zero_point = mge.tensor(
                int(zero_point)
            )

    irgraph = TM_FrontEnd(traced_module).resolve()

    transformer_options = [
        TransformerRule.REDUCE_AXIS_AS_INPUT,
        TransformerRule.PADDING_FOR_CONV_AND_POOLING,
        TransformerRule.CONV_ADD_ZERO_BIAS,
        TransformerRule.DECONV_SHAPE_AS_INPUT,
        TransformerRule.DEPTHWISE_CONV_RESHAPE_WEIGHT,
        TransformerRule.RESHAPE_BIAS_TO_1DIM,
        TransformerRule.FUSE_ACTIVATION,
        TransformerRule.SLICE_PARAMS_AS_INPUTS_AND_MAKE_SQUEEZE,
        TransformerRule.RESIZE_PARAMS_AS_INPUT,
        TransformerRule.TRANSPOSE_PATTERN_AS_INPUT,
    ]
    if mtk:
        # MTK devices only support batch_size 1
        set_platform("mtk")
        transformer_options.append(TransformerRule.DECONV_ADD_ZERO_BIAS,)

    transformer = IRTransform(transformer_options)
    transformed_irgraph = transformer.transform(irgraph)

    quantizer = IRQuantizer(
        require_quantize=require_quantize, param_fake_quant=param_fake_quant
    )

    if not require_quantize:
        quantizer.save_quantize_params(transformed_irgraph, path=quantize_file_path)

    converter = TFLiteConverter(transformed_irgraph, graph_name, quantizer=quantizer)
    model = converter.convert()

    assert isinstance(output, str), "tflite_fpath must be string"
    with open(output, "wb") as fout:
        fout.write(model)
