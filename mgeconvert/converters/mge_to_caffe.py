from mgeconvert.backend.ir_to_caffe import CaffeConverter
from mgeconvert.converter_ir.ir_quantizer import IRQuantizer
from mgeconvert.converter_ir.ir_transform import IRTransform, TransformerRule

from ..frontend import MGE_FrontEnd


def mge_to_caffe(
    mge_fpath,
    prototxt="out.prototxt",
    graph_name="graph",
    caffemodel="out.caffemodel",
    outspec=None,
    use_empty_blobs=False,
):
    assert isinstance(mge_fpath, str), "mge_fpath must be string"
    irgraph = MGE_FrontEnd(mge_fpath).resolve()

    transformer_options = [
        TransformerRule.EXPAND_MUL_ADD3,
        TransformerRule.FUSE_FOR_LEAKY_RELU,
    ]
    transformer = IRTransform(transformer_options)
    transformed_irgraph = transformer.transform(irgraph)

    converter = CaffeConverter(transformed_irgraph, use_empty_blobs)
    converter.convert()

    assert isinstance(prototxt, str) and isinstance(
        caffemodel, str
    ), "'prototxt' and 'caffemodel' must be string"
    converter.dump(prototxt, caffemodel)
