from typing import Sequence, Union

import megengine as mge
from megengine.traced_module import TracedModule
from mgeconvert.backend.ir_to_caffe.caffe_converter import CaffeConverter
from mgeconvert.converter_ir.ir_transform import IRTransform, TransformerRule
from mgeconvert.frontend import TM_FrontEnd


def tracedmodule_to_caffe(
    traced_module,
    prototxt="out.prototxt",
    caffemodel="out.caffemodel",
    use_empty_blobs=False,
):
    """
    Convert megengine model to Caffe,
    and save caffe model to `prototxt` and `caffemodel`.

    :param mge_fpath: the file path of megengine model.
    :type mge_fpath: str
    :param prototxt: the filename used for saved model definition.
    :type prototxt: str
    :param caffemodel: the filename used for saved model weights.
    :type caffemodel: str
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
    converter = CaffeConverter(transformed_irgraph, use_empty_blobs)
    converter.convert()

    assert isinstance(prototxt, str) and isinstance(
        caffemodel, str
    ), "'prototxt' and 'caffemodel' must be string"
    converter.dump(prototxt, caffemodel)
