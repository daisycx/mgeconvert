from typing import Sequence, Union

import megengine as mge
from megengine.traced_module import TracedModule
from mgeconvert.backend.ir_to_onnx.onnx_converter import OnnxConverter
from mgeconvert.converter_ir.ir_transform import IRTransform, TransformerRule
from mgeconvert.frontend import TM_FrontEnd


def tracedmodule_to_onnx(
    traced_module, output="out.onnx", *, graph_name="graph", opset=8, outspec=None
):
    """
    Convert megengine model to ONNX,
    and save the ONNX model to file `output`.

    :param mge_fpath: the file path of megengine model.
    :type fpath: str
    :param output: the filename used for the saved model.
    :type output: str
    :param graph_name: the name of the ONNX graph.
    :type graph_name: str
    :param opset: opset version of ONNX model.
    :type opset: int
    """
    if isinstance(traced_module, str):
        traced_module = mge.load(traced_module)
    assert isinstance(
        traced_module, TracedModule
    ), "Input should be a traced module or a path of traced module."

    irgraph = TM_FrontEnd(traced_module).resolve()
    transformer_options = [
        TransformerRule.REMOVE_RESHAPE_REALTED_OP,
        TransformerRule.REMOVE_UNRELATED_IROP,
    ]
    transformer = IRTransform(transformer_options)
    transformed_irgraph = transformer.transform(irgraph)

    converter = OnnxConverter(irgraph, opset, graph_name)
    model = converter.convert()

    assert isinstance(output, str), "onnx_fpath must be string"
    with open(output, "wb") as fout:
        fout.write(model.SerializeToString())
