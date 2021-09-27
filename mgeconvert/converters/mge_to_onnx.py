from mgeconvert.backend.ir_to_onnx import OnnxConverter
from mgeconvert.converter_ir.ir_transform import IRTransform, TransformerRule

from ..frontend import MGE_FrontEnd


def remove_initializer_from_input(model):
    if model.ir_version < 4:
        print(
            "Model with ir_version below 4 requires to include initilizer in graph input"
        )
        return model

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    return model


def mge_to_onnx(
    mge_fpath, output="out.onnx", *, graph_name="graph", opset=8, outspec=None
):
    assert isinstance(mge_fpath, str), "mge_fpath must be string"
    irgraph = MGE_FrontEnd(mge_fpath).resolve()
    transformer_options = [
        TransformerRule.FUSE_SOFTMAX,
        TransformerRule.EXPAND_MUL_ADD3,
    ]
    transformer = IRTransform(transformer_options)
    transformed_irgraph = transformer.transform(irgraph)
    converter = OnnxConverter(transformed_irgraph, opset, graph_name)
    model = converter.convert()
    model = remove_initializer_from_input(model)

    assert isinstance(output, str), "onnx_fpath must be string"
    with open(output, "wb") as fout:
        fout.write(model.SerializeToString())
