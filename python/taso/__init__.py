from .core import *
import onnx
from onnx import helper, TensorProto, numpy_helper

# the following packages are for graph partitioning
import graph_tool as gt
from graph_tool import flow#, draw
# from graph_tool imprt totopology
# import numpy as np

class InputNotFoundError(Exception):
    """Raised when cannot find input tensors """
    pass

# correspond to https://github.com/onnx/onnx/blob/master/onnx/onnx.proto
def onnx_datatype_tostring(dtype):
    if dtype == 0:
        return 'UNDEFINED'
    elif dtype == 1:
        return 'FLOAT'
    elif dtype == 2:
        return 'UINT8'
    elif dtype == 3:
        return 'INT8'
    elif dtype == 4:
        return 'UINT16'
    elif dtype == 5:
        return 'INT16'
    elif dtype == 6:
        return 'INT32'
    elif dtype == 7:
        return 'INT64'
    elif dtype == 8:
        return 'STRING'
    elif dtype == 9:
        return 'BOOL'
    elif dtype == 10:
        return 'FLOAT16'
    elif dtype == 11:
        return 'DOUBLE'
    elif dtype == 12:
        return 'UINT32'
    elif dtype == 13:
        return 'UINT64'
    elif dtype == 14:
        return 'COMPLEX64'
    elif dtype == 15:
        return 'COMPLEX128'
    elif dtype == 16:
        return 'BFLOAT16'
    else:
        raise Exception('Unknown onnx datatype')

def _check_output(taso_output, onnx_output):
    # TODO: check output match
    return True

def _parse_attribute(attributes):
    atts = dict()
    for att in attributes:
        if att.type == onnx.AttributeProto.INT:
            atts[att.name] = att.i
        elif att.type == onnx.AttributeProto.INTS:
            atts[att.name] = att.ints
        elif att.type == onnx.AttributeProto.FLOAT:
            atts[att.name] = att.f
        elif att.type == onnx.AttributeProto.STRING:
            atts[att.name] = att.s
        elif att.type == onnx.AttributeProto.TENSOR:
            atts[att.name] = att.t
        else:
            assert False, "Unsupported Attribute Type: {}".format(att.type)
    return atts

def _get_conv_pool_pads_attr(attrs):
    if ("auto_pad" in attrs):
        padding = attrs["auto_pad"]
        if isinstance(padding, bytes):
            padding = padding.decode()
        if (padding=='SAME_LOWER') or (padding=='SAME_UPPER'):
            pads = "SAME"
        elif (padding=='VALID'):
            pads = "VALID"
        else:
            assert padding=='NOTSET', "Unrecogonized auto_pad value: {}".format(padding)
        # Note that we always think conv1x1 has SAME padding
        # This will allow fusing enlarged convs
        if sum(attrs['kernel_shape']) <= 2:
            pads = "SAME"
        if padding != 'NOTSET':
            return pads
    # Note that we always think conv1x1 has SAME padding
    # This will allow fusing enlarged convs
    if sum(attrs["pads"]) == 0 and sum(attrs['kernel_shape']) > 2:
        pads = "VALID"
    else:
        pads = "SAME"
    return pads

def _get_list_from_initializer(initializer, name):
    for data in initializer:
        if data.name == name:
            ret = list()
            if data.int64_data != []:
                for dim in data.int64_data:
                    ret.append(dim)
            elif data.raw_data and data.raw_data != []:
                ret_in_array = numpy_helper.to_array(data)
                for dim in ret_in_array:
                        ret.append(dim)
            return ret
    raise InputNotFoundError
    return []

def _get_inputs(op, graph, tensors, initializer):
    inputs = list()
    for i in op.input:
        input_tensor = None
        if i in tensors:
            input_tensor = tensors[i]
        else:
            for init in initializer:
                if init.name == i:
                    input_tensor = graph.new_weight(
                        dims=tuple(list(init.dims)), data=numpy_helper.to_array(init))
                    break
        if input_tensor is None:
            raise InputNotFoundError
            return []
        inputs.append(input_tensor)
    return inputs

def _add(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    outputs = graph.add(inputs[0], inputs[1])
    return outputs

def _argmax(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "ArgMax requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    keepdims = attrs["keepdims"]
    axis = attrs["axis"]
    axes_list = [axis]
    outputs = graph.reduce_argmax(input=inputs[0], axes=tuple(axes_list), keepdims=keepdims)
    return outputs

def _argmin(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "ArgMin requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    keepdims = attrs["keepdims"]
    axis = attrs["axis"]
    axes_list = [axis]
    outputs = graph.reduce_argmin(input=inputs[0], axes=tuple(axes_list), keepdims=keepdims)
    return outputs

def _batchnorm(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    attrs = _parse_attribute(op.attribute)
    outputs = graph.batchnorm(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
    return outputs

def _cast(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    #assert len(op.input) == 1, "Cast requires exactly one input"
    #input_tensor = None
    #if op.input[0] in tensors:
    #    input_tensor = tensors[op.input[0]]
    #else:
    #    for init in initializer:
    #        if init.name == op.input[0]:
    #            input_tensor = graph.new_weight(
    #                dims=tuple(list(init.dims)), data=numpy_helper.to_array(init))
    #            break
    #assert input_tensor is not None, "Input Tensor Not Found"
    attrs = _parse_attribute(op.attribute)
    to_type = onnx_datatype_tostring(attrs["to"])
    outputs = graph.cast(input=inputs[0], datatype=to_type)
    return outputs

def _ceil(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Ceil requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.ceil(inputs[0])
    return outputs

def _concat(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    attrs = _parse_attribute(op.attribute)
    axis = attrs["axis"]
    outputs = graph.concat(axis, inputs)
    return outputs

def _constant(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    attrs = _parse_attribute(op.attribute)
    # TODO: Currently do not support sparse value
    assert "value" in attrs, "Do not support sparse value for Constant"
    tensor = attrs["value"]
    dims = list()
    for dim in tensor.dims:
        dims.append(dim)
    weight_data = numpy_helper.to_array(tensor)
    outputs = graph.new_weight(dims=tuple(dims), data=weight_data)
    return outputs

def _conv2d(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    attrs = _parse_attribute(op.attribute)
    if "group" not in attrs:
        group = 1 # default 1
    else:
        group = attrs["group"]
    pads = _get_conv_pool_pads_attr(attrs)
    strides = attrs["strides"]
    outputs = graph.conv2d(input=inputs[0], weight=inputs[1], strides=strides, padding=pads)
    return outputs

def _div(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Div takes exactly two inputs"
    outputs = graph.div(x=inputs[0], y=inputs[1])
    return outputs

def _dropout(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Dropout takes exactly one input"
    attrs = _parse_attribute(op.attribute)
    rate = attrs["ratio"]
    outputs = graph.dropout(input=inputs[0], rate=rate)
    return outputs

def _equal(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Equal takes exactly two inputs"
    outputs = graph.equal(x=inputs[0], y=inputs[1])
    return outputs

def _exp(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Exp requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.exp(input=inputs[0])
    return outputs

def _gemm(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    attrs = _parse_attribute(op.attribute)
    if "transA" in attrs:
        inputs[0] = graph.transpose(inputs[0], (1,0), shuffle=True)
    if "transB" in attrs:
        inputs[1] = graph.transpose(inputs[1], (1,0), shuffle=True)
    outputs = graph.matmul(inputs[0], inputs[1])
    return outputs

def _greater(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Greater takes exactly two inputs"
    outputs = graph.greater(inputs[0], inputs[1])
    return outputs

def _identity(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Identity takes exactly one input"
    outputs = graph.dropout(inputs[0], 0.0)
    return outputs

def _leakyrelu(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "LeakyRelu requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    alpha = attrs["alpha"]
    outputs = graph.leakyrelu(input=inputs[0], alpha=alpha)
    return outputs

def _less(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Less takes exactly two inputs"
    outputs = graph.less(inputs[0], inputs[1])
    return outputs

def _log(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Log requires exactly one input"
    #input_tensor = None
    #if op.input[0] in tensors:
    #    input_tensor = tensors[op.input[0]]
    #else:
    #    for init in initializer:
    #        if init.name == op.input[0]:
    #            input_tensor = graph.new_weight(
    #                dims=tuple(list(init.dims)), data=numpy_helper.to_array(init))
    #            break
    #assert input_tensor is not None, "Input Tensor Not Found"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.log(input=inputs[0])
    return outputs

def _logical_not(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Not requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.logical_not(input=inputs[0])
    return outputs

def _matmul(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Matmul takes exactly two inputs"
    outputs = graph.matmul(inputs[0], inputs[1])
    return outputs

def _min(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Min takes exactly two inputs"
    outputs = graph.min(inputs[0], inputs[1])
    return outputs

def _mul(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Mul takes exactly two inputs"
    outputs = graph.mul(inputs[0], inputs[1])
    return outputs

def _pad(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    attrs = _parse_attribute(op.attribute)
    # Currently treat pad as a no op
    assert sum(attrs['pads']) == 0
    return inputs

def _max(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Max takes exactly two inputs"
    outputs = graph.max(inputs[0], inputs[1])
    return outputs

def _maxpool2d(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "MaxPool2D requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    kernels = attrs["kernel_shape"]
    strides = attrs["strides"]
    pads = _get_conv_pool_pads_attr(attrs)
    outputs = graph.maxpool2d(input=inputs[0], kernels=kernels, strides=strides, padding=pads)
    return outputs

def _avgpool2d(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "AvgPool2D requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    kernels = attrs["kernel_shape"]
    strides = attrs["strides"]
    pads = _get_conv_pool_pads_attr(attrs)
    outputs = graph.avgpool2d(input=inputs[0], kernels=kernels, strides=strides, padding=pads)
    return outputs

def _reducemax(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "ReduceMax requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    keepdims = attrs["keepdims"]
    axes_ints = attrs["axes"]
    axes_list = list()
    for i in axes_ints:
        axes_list.append(i)
    outputs = graph.reduce_max(input=inputs[0], axes=tuple(axes_list), keepdims=keepdims)
    return outputs

def _reducemean(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "ReduceMean requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    keepdims = attrs["keepdims"]
    axes_ints = attrs["axes"]
    axes_list = list()
    for i in axes_ints:
        axes_list.append(i)
    outputs = graph.reduce_mean(input=inputs[0], axes=tuple(axes_list), keepdims=keepdims)
    return outputs

def _reducemin(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "ReduceMin requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    keepdims = attrs["keepdims"]
    axes_ints = attrs["axes"]
    axes_list = list()
    for i in axes_ints:
        axes_list.append(i)
    outputs = graph.reduce_min(input=inputs[0], axes=tuple(axes_list), keepdims=keepdims)
    return outputs

def _reduceprod(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "ReduceProd requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    keepdims = attrs["keepdims"]
    axes_ints = attrs["axes"]
    axes_list = list()
    for i in axes_ints:
        axes_list.append(i)
    outputs = graph.reduce_prod(input=inputs[0], axes=tuple(axes_list), keepdims=keepdims)
    return outputs

def _reducesum(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "ReduceSum requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    keepdims = attrs["keepdims"]
    axes_ints = attrs["axes"]
    axes_list = list()
    for i in axes_ints:
        axes_list.append(i)
    outputs = graph.reduce_sum(input=inputs[0], axes=tuple(axes_list), keepdims=keepdims)
    return outputs

def _reshape(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2
    shape = list()
    for data in initializer:
        if data.name == op.input[1]:
            shape = list()
            if data.int64_data != []:
                for dim in data.int64_data:
                    shape.append(dim)
            elif data.raw_data and data.raw_data != []:
                shape_in_array = numpy_helper.to_array(data)
                for dim in shape_in_array:
                    shape.append(dim)
    outputs = graph.reshape(inputs[0], tuple(shape))
    return outputs

def _resize(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) >= 2, "Resize takes at least two inputs"
    outputs = graph.resize(inputs[0], inputs[1])
    return outputs

# TensorFlow resize_nearest_neighbor
# https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/resize-nearest-neighbor
def _resize_nearest_neighbor(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "ResizeNearestNeighbor takes exactly two inputs"
    shape = list()
    for data in initializer:
        if data.name == op.input[1]:
            for dim in data.int64_data:
                shape.append(dim)
    assert len(shape) == 2, "ResizeNeareestNeighbor: new size cannot be statically inferred"
    outputs = graph.resize_nearest_neighbor(input=inputs[0], new_height=shape[0], new_width=shape[1])
    return outputs

# TensorFlow crop_and_resize
# https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/crop-and-resize
def _crop_and_resize(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 4, "CropAndResize takes exactly four inputs"
    outputs = graph.crop_and_resize(inputs[0], inputs[1], inputs[2], inputs[3])
    return outputs

def _relu(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Relu requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.relu(input=inputs[0])
    return outputs

def _round(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Round requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.round(inputs[0])
    return outputs

def _shape(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs)== 1, "Shape requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.shape(inputs[0])
    return outputs

def _size(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Size requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.size(inputs[0])
    return outputs

def _slice(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) >= 3, "Slice requires at least 3 inputs"
    assert len(inputs) <= 5, "Slice takes at most 5 inputs"
    start = _get_list_from_initializer(initializer, op.input[1])
    # replace INT_MAX with 999999
    for i in range(len(start)):
        start[i] = min(999999, start[i])
    end = _get_list_from_initializer(initializer, op.input[2])
    # replace INT_MAX with 999999
    for i in range(len(end)):
        end[i] = min(999999, end[i])
    if len(op.input) > 3:
        axes = _get_list_from_initializer(initializer, op.input[3])
    else:
        axes = None
    if len(op.input) > 4:
        steps = _get_list_from_initializer(initializer, op.input[4])
    else:
        steps = None
    outputs = graph.slice(inputs[0], start, end, axes, steps)
    return outputs

def _split(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Split requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    axis = attrs["axis"]
    split_ints = attrs["split"]
    if type(split_ints) is not list:
        origin_dim = inputs[0].dim(axis)
        split_list = [int(origin_dim / split_ints)] * split_ints
        outputs = graph.split(inputs[0], axis, split_list)
    else:
        split_list = list()
        for i in split_ints:
            split_list.append(i)
        outputs = graph.split(inputs[0], axis, split_list)
    return outputs

def _sqrt(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Sqrt requires exactly one input"
    #input_tensor = None
    #if op.input[0] in tensors:
    #    input_tensor = tensors[op.input[0]]
    #else:
    #    for init in initializer:
    #        if init.name == op.input[0]:
    #            input_tensor = graph.new_weight(
    #                dims=tuple(list(init.dims)), data=numpy_helper.to_array(init))
    #            break
    #assert input_tensor is not None, "Input Tensor Not Found"
    attrs = _parse_attribute(op.attribute)
    outputs = graph.sqrt(input=inputs[0])
    return outputs

def _squeeze(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Squeeze takes exactly one input"
    attrs = _parse_attribute(op.attribute)
    axes_ints = attrs["axes"]
    axes = list()
    for i in axes_ints:
        axes.append(i)
    outputs = graph.squeeze(input=inputs[0], axes=tuple(axes))
    return outputs

def _strided_slice(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 4, "StrideSlice takes exactly four inputs"
    start = _get_list_from_initializer(initializer, op.input[1])
    end = _get_list_from_initializer(initializer, op.input[2])
    steps = _get_list_from_initializer(initializer, op.input[3])
    attrs = _parse_attribute(op.attribute)
    begin_mask = attrs["begin_mask"]
    end_mask = attrs["end_mask"]
    ellipsis_mask = attrs["ellipsis_mask"]
    new_axis_mask = attrs["new_axis_mask"]
    shrink_axis_mask = attrs["shrink_axis_mask"]
    # TODO: support new_axis and shrink axis
    assert new_axis_mask == 0, "Non zero new_axis_mask is not supported yet"
    assert shrink_axis_mask == 0, "Non zero shrink_axis_mask is not supported yet"
    # TODO: current assume that strided slice returns the original tensor
    outputs = graph.slice(inputs[0], None, None, None, None)
    return outputs

def _sub(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 2, "Sub takes exactly two inputs"
    outputs = graph.sub(x=inputs[0], y=inputs[1])
    return outputs

def _transpose(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Transpose requires exactly one input"
    attrs = _parse_attribute(op.attribute)
    perm_ints = attrs["perm"]
    perm = list()
    for i in perm_ints:
        perm.append(i)
    outputs = graph.transpose(inputs[0], tuple(perm), shuffle=True)
    return outputs

def _unsqueeze(op, graph, tensors, initializer):
    inputs = _get_inputs(op, graph, tensors, initializer)
    assert len(inputs) == 1, "Unsqueeze takes exactly one input"
    #input_tensor = None
    #if op.input[0] in tensors:
    #    input_tensor = tensors[op.input[0]]
    #else:
    #    for init in initializer:
    #        if init.name == op.input[0]:
    #            input_tensor = graph.new_weight(
    #                dims=tuple(list(init.dims)), data=numpy_helper.to_array(init))
    #            break
    #assert input_tensor is not None, "Input Tensor Not Found"
    attrs = _parse_attribute(op.attribute)
    axes_ints = attrs["axes"]
    axes = list()
    for i in axes_ints:
        axes.append(i)
    outputs = graph.unsqueeze(input=inputs[0], axes=tuple(axes))
    return outputs

# Add all supported operators
xf_operators = dict()
xf_operators['Add'] = _add
xf_operators['ArgMax'] = _argmax
xf_operators['ArgMin'] = _argmin
xf_operators['BatchNormalization'] = _batchnorm
xf_operators['Cast'] = _cast
xf_operators['Ceil'] = _ceil
xf_operators['Concat'] = _concat
xf_operators["Constant"] = _constant
xf_operators['Conv'] = _conv2d
xf_operators['Div'] = _div
xf_operators['Dropout'] = _dropout
xf_operators['Equal'] = _equal
xf_operators['Exp'] = _exp
xf_operators['Gemm'] = _gemm
xf_operators['Greater'] = _greater
xf_operators['Identity'] = _identity
xf_operators['LeakyRelu'] = _leakyrelu
xf_operators['Less'] = _less
xf_operators['Log'] = _log
xf_operators['Pad'] = _pad
xf_operators['ReduceMax'] = _reducemax
xf_operators['ReduceMean'] = _reducemean
xf_operators['ReduceMin'] = _reducemin
xf_operators['ReduceProd'] = _reduceprod
xf_operators['ReduceSum'] = _reducesum
xf_operators['Reshape'] = _reshape
xf_operators['Relu'] = _relu
xf_operators['Round'] = _round
xf_operators['Matmul'] = _matmul
xf_operators['Max'] = _max
xf_operators['MaxPool'] = _maxpool2d
xf_operators['Min'] = _min
xf_operators['Mul'] = _mul
xf_operators['Not'] = _logical_not
xf_operators['AveragePool'] = _avgpool2d
xf_operators['Shape'] = _shape
xf_operators['Size'] = _size
xf_operators['Slice'] = _slice
xf_operators['Split'] = _split
xf_operators['Sqrt'] = _sqrt
xf_operators['Squeeze'] = _squeeze
xf_operators['StridedSlice'] = _strided_slice
xf_operators['Sub'] = _sub
xf_operators['Transpose'] = _transpose
xf_operators['Unsqueeze'] = _unsqueeze

def new_graph(print_measurements = False):
    graph = core.PyGraph()
    if print_measurements:
        graph.print_measurements()
    return graph

def load_onnx(filename):
    '''
    Load a onnx file and return a Graph

    @params
    filename is a string containing a file name
    
    @return
    Loaded in-memory Graph
    '''
    graph = core.PyGraph()
    model = onnx.load(filename)
    tensors = dict()
    for t in model.graph.input:
        dims = list()
        for d in t.type.tensor_type.shape.dim:
            dims.append(d.dim_value)
        weight_data = None
        for weight in model.graph.initializer:
            if (weight.name == t.name):
                weight_data = numpy_helper.to_array(weight)
        # We classify an input to be a pure input if we cannot find its weights
        if weight_data is None:
            tensors[t.name] = graph.new_input(dims=tuple(dims))
        else:
            tensors[t.name] = graph.new_weight(dims=tuple(dims), data=weight_data)

    # Add initializers not in the inputs
    for weight in model.graph.initializer:
        if weight.name not in tensors:
            if weight.dims:
                dims = list(weight.dims)
                weight_data = numpy_helper.to_array(weight)
                tensors[weight.name] = graph.new_weight(dims=tuple(dims), data=weight_data)

    # Reorder nodes to satisfy data dependencies
    tensor_owner = dict()
    name_to_op = dict()
    for op in model.graph.node:
        name_to_op[op.name] = op
        for output in op.output:
            tensor_owner[output] = op.name
    out_edges = dict()
    dependents = dict()
    node_list = list()
    for op in model.graph.node:
        dependents[op.name] = 0
        for input in op.input:
            if input in tensor_owner:
                dependents[op.name] += 1
                input_node = tensor_owner[input]
                if input_node not in out_edges:
                    out_edges[input_node] = list()
                out_edges[input_node].append(op.name)
        if dependents[op.name] == 0:
            node_list.append(op.name)
    idx = 0
    while idx < len(node_list):
        opname = node_list[idx]
        if opname in out_edges:
            for e in out_edges[opname]:
                dependents[e] -= 1
                if dependents[e] == 0:
                    node_list.append(e)
        idx += 1
    assert len(node_list) == len(model.graph.node), "Internal error when reording ONNX operators"

    # Add nodse into TASO graph
    cnt = 0
    for opname in node_list:
        op = name_to_op[opname]
        print(cnt, op.op_type)
        cnt += 1
        if op.op_type in xf_operators:
            try:
                outputs = xf_operators[op.op_type](op, graph, tensors, model.graph.initializer)
                if not isinstance(outputs, list):
                    outputs = [outputs]
                assert len(outputs) == len(op.output), "Number of output tensors mismatch"
                for i in range(len(outputs)):
                    assert _check_output(outputs[i], op.output[i])
                    tensors[op.output[i]] = outputs[i]
            except InputNotFoundError:
                print("Cannot find input tensor for operator: name({}) type({}) (Skipped)".format(opname, op.op_type))
                continue
        else:
            print("Found unsupported ONNX operator: {} (Skipped)".format(op.op_type))
            continue
    return graph

input_weight_names = dict()
input_weight_names['Add'] = ['input1', 'input2']
input_weight_names['AveragePool'] = ['input']
input_weight_names['BatchNormalization'] = ['input', 'scale', 'bias', 'mean', 'var']
input_weight_names['Concat'] = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6']
input_weight_names['Conv'] = ['input', 'weight', 'bias']
input_weight_names['Matmul'] = ['input', 'weight']
input_weight_names['Mul'] = ['input1', 'input2']
input_weight_names['Reshpe'] = ['input', 'shape']

operator_attrs = dict()
operator_attrs['Add'] = []
operator_attrs['ArgMax'] = []
operator_attrs['ArgMin'] = []
operator_attrs['AveragePool'] = ['kernel_shape', 'pads', 'strides']
operator_attrs['BatchNormalization'] = [] # TODO: Add epsilon and momentum
operator_attrs['Cast'] = []
operator_attrs['Ceil'] = []
operator_attrs['Concat'] = ['axis']
operator_attrs['Conv'] = ['group', 'kernel_shape', 'pads', 'strides']
operator_attrs['Div'] = []
operator_attrs['Dropout'] = []
operator_attrs['Gemm'] = []
operator_attrs['Greater'] = []
operator_attrs['Identity'] = []
operator_attrs['Less'] = []
operator_attrs['Log'] = []
operator_attrs['Pad'] = []
operator_attrs['Matmul'] = []
operator_attrs['MaxPool'] = ['kernel_shape', 'pads', 'strides']
operator_attrs['Mul'] = []
operator_attrs['Shape'] = []
operator_attrs['Sigmoid'] = []
operator_attrs['Slice'] = []
operator_attrs['Split'] = ['axis', 'split']
operator_attrs["Squeeze"] = ['axes']
operator_attrs['StridedSlice'] = []
operator_attrs['Relu'] = []
operator_attrs['Reshape'] = []
operator_attrs['Tanh'] = []
operator_attrs['Transpose'] = ['perm']
operator_attrs['Unsqueeze'] = ['axes']

def _input_tensor_name(graph, inedge, op):
    intype = graph.get_operator_type(inedge['srcOp'])
    if intype == "Input":
        return "data"
    elif intype == "Weight":
        mytype = graph.get_operator_type(op)
        return "{}{}_{}".format(mytype, op['guid'], input_weight_names[mytype][inedge['dstIdx']])
    else:
        return _output_tensor_name(graph, inedge['srcOp'], inedge['srcIdx'])

def _output_tensor_name(graph, op, idx):
    type = graph.get_operator_type(op)
    return "{}{}_fwd{}".format(type, op['guid'], idx)

def _add_node_attribute(graph, node, op, optype):
    for key in operator_attrs[optype]:
        val = graph.get_operator_attr(op, key)
        attr = helper.make_attribute(key, val)
        node.attribute.append(attr)

def export_onnx(graph):
    '''
    Export a XFlow graph to an ONNX graph
    
    @params
    graph is a XFlow graph

    @return
    A in-memory ONNX graph
    '''
    opList = graph.get_operator_list()
    graph_nodes = list()
    graph_inputs = list()
    graph_initializers = list()
    graph_outputs = list()
    output_guids = dict()
    for op in opList:
        mytype = graph.get_operator_type(op)
        inedges = graph.get_input_edges(op)
        #print("op.guid={} mytype={} inedges={}".format(op['guid'], mytype, len(inedges)))
        inputs = list()
        for e in inedges:
            intype = graph.get_operator_type(e['srcOp'])
            inputs.append(_input_tensor_name(graph, e, op))
            output_guids.pop((e['srcOp']['guid'], e['srcIdx']), None)
            if intype == 'Input' or intype == 'Weight':
                graph_inputs.append(helper.make_tensor_value_info(_input_tensor_name(graph, e, op),
                                    TensorProto.FLOAT, graph.get_input_dims(op, e['dstIdx'])))
            if intype == 'Weight':
                graph_initializers.append(helper.make_tensor(_input_tensor_name(graph, e, op),
                                          TensorProto.FLOAT, graph.get_input_dims(op, e['dstIdx']),
                                          graph.get_weight_value(e['srcOp'])))

        # add a second input for Reshape
        if mytype == 'Reshape':
            inputs.append('Reshape_attr{}'.format(op['guid']))
            shape = graph.get_output_dims(op, 0)
            graph_inputs.append(helper.make_tensor_value_info('Reshape_attr{}'.format(op['guid']), TensorProto.INT64, [len(shape)]))
            graph_initializers.append(helper.make_tensor('Reshape_attr{}'.format(op['guid']), TensorProto.INT64, [len(shape)], shape))
        outputs = list()
        for i in range(graph.get_num_outputs(op)):
            outputs.append(_output_tensor_name(graph, op, i))
            output_guids[(op['guid'], i)] = op
        node = helper.make_node(mytype, inputs, outputs, '{}{}'.format(mytype, op['guid']))
        _add_node_attribute(graph, node, op, mytype)
        graph_nodes.append(node)
    for guid, idx in output_guids:
        op = output_guids[(guid, idx)]
        graph_outputs.append(helper.make_tensor_value_info(_output_tensor_name(graph, op, idx),
                             TensorProto.FLOAT, graph.get_output_dims(op, idx)))
    onnx_graph = helper.make_graph(graph_nodes, 'main', graph_inputs, graph_outputs, graph_initializers)
    onnx_model = helper.make_model(onnx_graph, producer_name='TASO Optimized Model')
    return onnx_model

def get_vertex_weight(graph):
    return graph.get_vertex_weight()


def optimize_prune(graph, alpha = 1.0, budget = 1000, print_subst = False):
    return graph.optimize_prune(alpha, budget, print_subst)

def optimize_enumeration(graph, alpha = 1.0, budget = 1000, print_subst = False):
    return graph.optimize_enumeration(alpha, budget, print_subst)


def optimize_reuse(graph, alpha = 1.0, budget = 1000, print_subst = False):
    return graph.optimize_reuse(alpha, budget, print_subst)

def optimize_sampletrick(graph, alpha = 1.0, budget = 1000, print_subst = False, sample_size = 20):
    return graph.optimize_sampletrick(alpha, budget, print_subst, sample_size)

def optimize_sampletrick_local(graph, alpha = 1.0, budget = 1000, print_subst = False, sample_size = 20):
    return graph.optimize_sampletrick_local(alpha, budget, print_subst, sample_size)

def optimize_sampletrick_newreuse(graph, alpha = 1.0, budget = 1000, print_subst = False, sample_size = 20):
    return graph.optimize_sampletrick_newreuse(alpha, budget, print_subst, sample_size)

def optimize_sampletrick_newreuse_2step(graph, alpha = 1.0, budget = 1000, print_subst = False, sample_size = 20):
    return graph.optimize_sampletrick_newreuse_2step(alpha, budget, print_subst, sample_size)

def optimize_sampletrick_newreuse_2samplestep(graph, alpha = 1.0, budget = 1000, print_subst = False, sample_size = 20, do_weight_process=True):
    return graph.optimize_sampletrick_newreuse_2samplestep(alpha, budget, print_subst, sample_size, do_weight_process)

def optimize_sampletrick_truenewreuse(graph, which_sample = 0, alpha = 1.0, budget = 1000, print_subst = False, sample_size = 20):
    return graph.optimize_sampletrick_truenewreuse(which_sample, alpha, budget, print_subst, sample_size)


def optimize_partition(graph, alpha = 1.0, budget = 1000, print_subst = False, eraly_stop_num = 10000, partitions = None, do_weight_process=True):
    return graph.optimize_partition(alpha, budget, print_subst, eraly_stop_num, partitions, do_weight_process)

def optimize_sysmltrick(graph, alpha = 1.0, budget = 1000, print_subst = False, eraly_stop_num = 10000, do_weight_process=True):
    return graph.optimize_sysmltrick(alpha, budget, print_subst, eraly_stop_num, do_weight_process)

def minstcut(outedges, s, t, capacity):
    '''transform the input directed graph to another directed graph
        (Since in this transaction, we regard the vertex cut problem as
        solving the vertex cut in the underlying undirected graph.)
        solve the min s-t cut in the transformed directed graph
    input:
        outedges: dictionary of outedges of vertices, every vertex has an entry
        s: the index of the source node for the s-t cut problem
        t: the index of the sink node for the s-t cut problem
        RMK: s and t should not be adjacent
        capacity: the capacity of each vertex, capacity[i] is for vertex index i
    output:
        the vertex cut and the cost
    '''
    new_edges = list()
    ver_num = len(outedges.keys())
    # add an arc to split a vertex which is neither s nor t
    for i in range(ver_num):
        if (i != s) and (i != t):
            new_edges.append(list([i,i+ver_num]))

    # add other edges
    for i in outedges.keys():
        if (i == s) or (i == t):
            for j in outedges[i]:
                # j is not s and t
                new_edges.append(list([i, j]))
                new_edges.append(list([j+ver_num, i]))
        else:
            # i is not s and t
            for j in outedges[i]:
                if (j == s) or (j == t):
                    new_edges.append(list([i+ver_num, j]))
                    new_edges.append(list([j, i]))
                else:
                    new_edges.append(list([i+ver_num, j]))
                    new_edges.append(list([j+ver_num, i]))

    # find min s-t cut
    init_g = gt.Graph()
    init_g.add_vertex(ver_num*2)
    init_g.add_edge_list(iter(new_edges))
    
    infi_cap = max(capacity)*(ver_num + 1) + 1
    cap = init_g.new_edge_property("int")
    for e in init_g.edges():
        if (int(e.source()) + ver_num == int(e.target())):
            cap[e] = capacity[int(e.source())]
        else:
            cap[e] = infi_cap
    init_g.edge_properties["cap"] = cap

    src, tgt = init_g.vertex(s), init_g.vertex(t)
    res = flow.boykov_kolmogorov_max_flow(init_g, src, tgt, cap)
    part = flow.min_st_cut(init_g, src, cap, res)
    mc = sum([cap[e] - res[e] for e in init_g.edges() if part[e.source()] != part[e.target()]])
    # print("minimum cost: ", mc)

    # get the minimum cut
    # mc, part = flow.min_cut(init_g, cap)

    # get vertex cut
    vertex_cut = list()
    # source_part = list()
    # sink_part = list()
    # source_part.append(s)
    # sink_part.append(t)
    
    # visited = [False] * len(graph_infor[0])
    for e in init_g.edges():
        # if (part[e.source()] != part[e.target()]) and (cap[e] - res[e] > 0):
        if (part[e.source()] == True) and (part[e.target()] == False):
            assert int(e.source()) + ver_num == int(e.target()), 'wrong cut'
            assert cap[e] - res[e] == capacity[int(e.source())], "wrong flow"
            vertex_cut.append(int(e.source()))
            # visited[int(e.source())] = True
        
        # elif (part[e.source()] == part[e.target()]) and (int(e.source()) + ver_num == int(e.target())):
        #     if (part[e.source()] == True):
        #         source_part.append(int(e.source()))
        #     else:
        #         sink_part.append(int(e.source()))

    # mc = sum([cap[e] - res[e] for e in init_g.edges() if part[e.source()] != part[e.target()]])
    mc = sum([capacity[i] for i in vertex_cut])
    # print("minimum cost: ", mc)

    return vertex_cut, mc#, source_part, sink_part

def find_min_degree(out_edges):
    '''find the vertex with the minimum degree
        return the corresponding vertex and its neighbours
    '''
    neighbours = dict()
    for s in out_edges.keys():
        neighbours[s] = list()
    for s in out_edges.keys():
        for t in out_edges[s]:
            neighbours[s].append(t)
            neighbours[t].append(s)

    # do experiments, just select vertex 0 as the "minimum degree" vertex
    return 0, neighbours[0]

    min_degree = min([len(i) for i in neighbours.values()])
    for s in range(len(neighbours.keys())): 
        # iterate from small index to large
        if len(neighbours[s]) == min_degree:
            return s, neighbours[s]


def partition_edges_old(vertex_cut, outedges):
    '''This function partitions the edges in the graph into two parts
        return the nodes in two parts (each part has a copy of vertex cut nodes)
    '''
    new_edges = list()
    for s in outedges.keys():
        for t in outedges[s]:
            if not((s in vertex_cut) or (t in vertex_cut)):
                new_edges.append(list([s,t]))

    # find all connected components
    ver_num = len(outedges.keys())
    g = gt.Graph()
    g.add_vertex(ver_num)
    g.add_edge_list(iter(new_edges))

    comp, hist = gt.topology.label_components(g, directed=False) # treat the graph as undirected
    comp_num = hist.size
    out_comps = {comp[g.vertex(t)] for s in vertex_cut for t in outedges[s] if (t not in vertex_cut)}
    assert len(out_comps) + len(vertex_cut) != comp_num, "all components are out components"

    # partiotion vertices according to the components
    out_part = {i for i in range(ver_num) if (comp[g.vertex(i)] in out_comps)}
    in_part = set(range(ver_num)) - out_part
    out_part.update(vertex_cut)

    return in_part, out_part


def partition_edges_paperversion(vertex_cut, outedges, total_cost, capacity):
    '''This function partitions the edges in the graph into two parts
        return the nodes in two parts (each part has a copy of vertex cut nodes), and the NEW VERTEX CUT
        This partition is the same as mentioned in the paper, all reachable nodes from vertex_cut is in the out_part, others in the in_part
        THIS FUNCTION WOULD UPDATE VERTEX CUT
        total_cost: the first element records the total partition cost
    '''
    ver_num = len(outedges.keys())
    to_search = vertex_cut
    out_part = set(vertex_cut)
    while len(to_search) > 0:
        new_discovered = list()
        for s in to_search:
            for t in outedges[s]:
                if t not in out_part:
                    new_discovered.append(t)
        out_part.update(new_discovered)
        to_search = new_discovered
    in_part = set(range(ver_num)) - out_part

    # to make use of the do_partition function, we need to make sure every edge incident on in_part can be partitioned into in part edges
    new_in_nodes = set()
    for s in in_part:
        for t in outedges[s]:
            if t not in in_part:
                new_in_nodes.add(t)
    in_part.update(new_in_nodes)
    # in_part.update(vertex_cut)

    # update the total partition cost
    for v in new_in_nodes:
        total_cost[0] = total_cost[0] + capacity[v]

    return in_part, out_part, new_in_nodes

    # partiotion vertices according to the components

    # out_part = {i for i in range(ver_num) if (comp[g.vertex(i)] in out_comps)}
    # in_part = set(range(ver_num)) - out_part
    # out_part.update(vertex_cut)

    # return in_part, out_part
        

def partition_edges(vertex_cut, outedges):
    '''This function partitions the edges in the graph into two parts
        return the nodes in two parts (each part has a copy of vertex cut nodes)
        IF ALL COMPONENTS ARE OUT COMPONENTS, this function would divide the largest component and all the other components into two parts
    '''
    new_edges = list()
    for s in outedges.keys():
        for t in outedges[s]:
            if not((s in vertex_cut) or (t in vertex_cut)):
                new_edges.append(list([s,t]))

    # find all connected components
    ver_num = len(outedges.keys())
    g = gt.Graph()
    g.add_vertex(ver_num)
    g.add_edge_list(iter(new_edges))

    comp, hist = gt.topology.label_components(g, directed=False) # treat the graph as undirected
    comp_num = hist.size
    out_comps = {comp[g.vertex(t)] for s in vertex_cut for t in outedges[s] if (t not in vertex_cut)}
    
    # assert len(out_comps) + len(vertex_cut) != comp_num, "all components are out components"
    if (len(out_comps) + len(vertex_cut) == comp_num):
        # if all components are out components, we need to select the largest one as the out component
        # list_hist = list(hist)
        out_comp_sizes = list()
        out_comp_labels = list()
        for ii in range(comp_num):
            if ii in out_comps:
                out_comp_labels.append(ii)
                out_comp_sizes.append(hist[ii])
        # get the label of the component with the max size
        max_comp_label = out_comp_labels[out_comp_sizes.index(max(out_comp_sizes))]
        out_comps = {max_comp_label}

    # partiotion vertices according to the components
    out_part = {i for i in range(ver_num) if (comp[g.vertex(i)] in out_comps)}
    in_part = set(range(ver_num)) - out_part
    out_part.update(vertex_cut)

    return in_part, out_part

def turn_outedge_to_list(outedges):
    edges = list()
    for s in outedges.keys():
        for t in outedges[s]:
            edges.append(list([s,t]))

    return edges


def get_components(edges, ver_num):
    '''get the largest component'''
    g = gt.Graph()
    g.add_vertex(ver_num)
    g.add_edge_list(iter(edges))
    largest = list()
    comp = gt.topology.label_largest_component(g, directed=False) # treat the graph as undirected
    for i in range(ver_num):
        if comp[g.vertex(i)]:
            largest.append(i)

    return largest

def do_partition(G1_cap, G2_cap, G1_ver_index, G2_ver_index, G1_index_ver, G2_index_ver, G1_outedges, G2_outedges, capacity, out_edges, ver_index, index_ver, in_nodes, out_nodes, vertex_cut):
    '''patition nodes and edges according to the vertex cut'''
    ver_num = len(capacity)
    G1_cnt = 0
    G2_cnt = 0
    for i in range(ver_num):
        # if visited[i]:
        if i in out_nodes:
            G2_index_ver[G2_cnt] = index_ver[i]
            G2_ver_index[index_ver[i]] = G2_cnt
            G2_cap.append(capacity[i]) # store the corresponding cap
            G2_cnt += 1
        # else:
        if i in in_nodes:
            G1_index_ver[G1_cnt] = index_ver[i]
            G1_ver_index[index_ver[i]] = G1_cnt
            G1_cap.append(capacity[i]) # store the corresponding cap
            G1_cnt += 1

   # store the adjacent matrix
    # G1_outedges = dict()
    # G2_outedges = dict()
    for s_v in G1_ver_index.keys():
        s_in = G1_ver_index[s_v]
        s_io = ver_index[s_v]
        G1_outedges[s_in] = list()
        for t_io in out_edges[s_io]:
            t_v = index_ver[t_io]
            if t_v in G1_ver_index.keys(): # if two end points are in the same part
                if not((s_io in vertex_cut) and (t_io in vertex_cut)):
                    t_in = G1_ver_index[t_v]
                    G1_outedges[s_in].append(t_in)
                # G1.append(list([s_in, t_in]))
    for s_v in G2_ver_index.keys():
        s_in = G2_ver_index[s_v]
        s_io = ver_index[s_v]
        G2_outedges[s_in] = list()
        for t_io in out_edges[s_io]:
            t_v = index_ver[t_io]
            if t_v in G2_ver_index.keys(): # if two end points are in the same part
                t_in = G2_ver_index[t_v]
                G2_outedges[s_in].append(t_in)
                # G2.append(list([s_in, t_in]))


def recursive_graph_partition(threshold, capacity, out_edges, ver_index, index_ver, partitions, wei_ops, total_cost):
    '''
    threshold: int. the size threshold of the graph which does not need partition
    capacity: list of list. capacities
    out_edges: store adjacency matrix in dictionary
    ver_index: dictionary. from vertex to index
    index_ver: dictionary. from index to vertex
    partitions: list of list. each element list is the list of all vertices (instead of indices)
    wei_ops: dict from int to int. the weight ops which belongs to this partition and their mapper ops, store vertex not index of ops
    total_cost: list of one int, store the total cost
    '''
    # print("num of nodes in this partition:", len(capacity))
    # print("capacity:")
    # for i in range(len(capacity)):
    #     print(i, capacity[i])

    ver_num = len(capacity)

    edges = turn_outedge_to_list(out_edges) # store edge pairs
    vertex_cut = None
    in_nodes = None
    out_nodes = None

    if (ver_num <= threshold):
        # print("find a partition small enough")
        # do not need partition
        # return edges in this partition

        # need to add edges of wei_op->op
        final_edges = list()
        for s_i in out_edges:
            for t_i in out_edges[s_i]:
                final_edges.append(list([index_ver[s_i], index_ver[t_i]]))
        for s in wei_ops:
            final_edges.append(list([s, wei_ops[s]]))

        partitions.append(final_edges)
        # partitions.append(list(ver_index.keys())) # store the vertices in this partition
    else:
        # print("continue partition---------------")

        # need first check how many connected components there are and if more than one how large is the biggest one
        l_comp = get_components(edges, ver_num)
        if len(l_comp) < ver_num:
            # the graph is not connected
            # print("the graph is not connected")
            in_nodes = set(range(ver_num)) - set(l_comp)
            out_nodes = set(l_comp)
            vertex_cut = []
        else:
            min_v, v_neighbors = find_min_degree(out_edges)
            # print("min_v, v_neighbors", min_v, v_neighbors)

            global_mc = sum(capacity) + 1
            global_vercut = None
            # global_sourcepart = None
            # global_sinkpart = None

            # print("first kind partition")
            for t in range(ver_num-1, -1, -1):
                # iterate from large index to small
                if (t != min_v) and (t not in v_neighbors):
                    vercut, mc = minstcut(out_edges, min_v, t, capacity)
                    # print("s, t: ", min_v, t, "minimum cost: ", mc)
                    if mc < global_mc:
                        global_mc = mc
                        global_vercut = vercut
                        # global_sourcepart = sp
                        # global_sinkpart = tp
                        if global_mc == 0:
                            # we already find some good enough cut
                            break

            if global_mc != 0:
                # print("second kind partition")
                for s in v_neighbors:
                    for t in v_neighbors:
                        if (s != t):
                            # check s and t are not adjacent
                            if (t not in out_edges[s]) and (s not in out_edges[t]):
                                # s and t are not adjacent
                                vercut, mc = minstcut(out_edges, s, t, capacity)
                                # print("s, t: ", s, t, "minimum cost: ", mc)
                                if mc < global_mc:
                                    global_mc = mc
                                    global_vercut = vercut
                                    # global_sourcepart = sp
                                    # global_sinkpart = tp
                                    if global_mc == 0:
                                        # we already find some good enough cut
                                        break
                    if global_mc == 0:
                        break
            
            assert global_vercut != None, "no vertex cut can be found"

            # print("global_mc, global_vercut: ", global_mc, global_vercut)

            # update total_cost, comment this code because this cost is fake now (after doing partition according to sysml 19)
            # do not comment this line now
            total_cost[0] = total_cost[0] + global_mc

            # get the two vertex sets of two parts in the partition, identified by visited
            vertex_cut = global_vercut
            in_nodes, out_nodes = partition_edges(vertex_cut, out_edges)
            # partition edges, this function would update the vertex cut because of the edge partition, and update the total cost
            # in_nodes, out_nodes, vertex_cut = partition_edges(vertex_cut, out_edges, total_cost, capacity)

        # visited = [False] * ver_num
        # for i in vertex_cut:
        #     visited[i] = True
        # for i in vertex_cut:
        #     for j in out_edges[i]:
        #         if j in global_sinkpart:
        #             for k in global_sinkpart:
        #                 visited[j] = True
        #         else:
        #             for k in global_sourcepart:
        #                 visited[j] = True

        # all vertices which are reachable from vertex cut
        # out_from = vertex_cut
        # while (len(out_from) > 0):
        #     new_found = list()
        #     for s in out_from:
        #         # s must has an entry in out_edges
        #         for t in out_edges[s]:
        #             if visited[t] == False:
        #                 new_found.append(t)
        #                 visited[t] = True
        #     out_from = new_found

        # store the capacity of vertices in G1 and G2
        G1_cap = list() 
        G2_cap = list()
        G1_ver_index = dict()
        G2_ver_index = dict()
        G1_index_ver = dict()
        G2_index_ver = dict()
        G1_outedges = dict()
        G2_outedges = dict()

        do_partition(G1_cap, G2_cap, G1_ver_index, G2_ver_index, G1_index_ver, G2_index_ver, G1_outedges, G2_outedges, capacity, out_edges, ver_index, index_ver, in_nodes, out_nodes, vertex_cut)

        G1_wei_ops = dict()
        G2_wei_ops = dict()

        # partition weight ops
        for s in wei_ops:
            t = wei_ops[s]
            t_i = ver_index[t]            
            if t_i in out_nodes:
                G2_wei_ops[s] = t
            else:
                G1_wei_ops[s] = t


        # now one iteration of graph partition is done, and the new graph infor has been prepared
        # recursively partition the graph
        recursive_graph_partition(threshold, G1_cap, G1_outedges, G1_ver_index, G1_index_ver, partitions, G1_wei_ops, total_cost)
        recursive_graph_partition(threshold, G2_cap, G2_outedges, G2_ver_index, G2_index_ver, partitions, G2_wei_ops, total_cost)

def del_weight_op(outedges):
    '''This function delete weight op from the initial computation graph
    Return the set of weight ops'''
    w2op = dict()
    newedges = dict()
    for s in outedges.keys():
        if len(outedges[s]) == 1:
            # s has only one output
            find = False
            for p in outedges.keys():
                if s in outedges[p]:
                    # if p->s exists, i.e., s has input, it cannot be a weight op
                    find = True
                    break
            if not find:
                # s is a weight op, or s can be regarded as a weight op
                w2op[s] = outedges[s][0]

    # for i in outedges.keys():
    #     if i not in w2op:
    #         newedges[i] = outedges[i]
    #     else:
    #         newedges[i] = list()
    
    # return w2op, newedges
    return w2op

def draw_partitions(ver_num, partitions):
    g = gt.Graph()
    g.add_vertex(ver_num)
    par = g.new_edge_property("int")
    for i in range(len(partitions)):
        g.add_edge_list(iter(partitions[i]))
        for e in partitions[i]:
            par[g.edge(e[0], e[1])] = i+1
    
    g.edge_properties["par"] = par
    gt.draw.graph_draw(g, edge_color=par, output="partition_results.pdf")


def modify_capacity(capacity):
    '''
    This function change the 0 capacity to an epsilon value so that the algorithm would find a min s-t vertex cut with the minimum size
    '''
    ver_num = len(capacity)
    mul_const = ver_num + 1
    for i in range(ver_num):
        if (capacity[i] == 0):
            capacity[i] = 1
        else:
            capacity[i] = mul_const * capacity[i]




# the algorithm performs recursive graph partitioning
def graph_partition(graph, threshold, partitions = list()):
    graph_infor = get_vertex_weight(graph)

    capacity = graph_infor[0]

    # modify the vertex capacity
    modify_capacity(capacity)

    # store the vertex-index mapping infor, key: verID value:index in the graph
    ver_index = dict()
    # store the index-vertex mapping infor
    index_ver = dict()
    for i in range(len(graph_infor[0])):
        ver_index[i] = i
        index_ver[i] = i

    # build the edge dictionary
    # in_edges = dict()
    out_edges = dict()
    for i in range(len(graph_infor[0])):
        out_edges[i] = list()
    for i in graph_infor[1:]:
        # if (i[0] in out_edges.keys()):
        out_edges[i[0]].append(i[1])
        # else:
        #     out_edges[i[0]] = list([i[1]])


    ver_num = len(capacity)
    wei_ops = del_weight_op(out_edges)
    in_nodes = set(wei_ops.keys())
    out_nodes = set(range(ver_num)) - in_nodes
    vertex_cut = []

    G1_cap = list() 
    G2_cap = list()
    G1_ver_index = dict()
    G2_ver_index = dict()
    G1_index_ver = dict()
    G2_index_ver = dict()
    G1_outedges = dict()
    G2_outedges = dict()

    do_partition(G1_cap, G2_cap, G1_ver_index, G2_ver_index, G1_index_ver, G2_index_ver, G1_outedges, G2_outedges, capacity, out_edges, ver_index, index_ver, in_nodes, out_nodes, vertex_cut)

    total_cost = list([0])

    # recursive_graph_partition(threshold, capacity, new_outedges, ver_index, index_ver, partitions)
    recursive_graph_partition(threshold, G2_cap, G2_outedges, G2_ver_index, G2_index_ver, partitions, wei_ops, total_cost)

    # print("total_cost: ", total_cost[0])

    assert len(graph_infor[1:]) == sum([len(i) for i in partitions]), "not every edge belongs to exactly one partition"

    # print(partitions)

    # draw the partitions
    # draw_partitions(ver_num, partitions)

    # new_par = [j for i in partitions for j in i]




# Current TASO Version
__version__ = "0.1.0"
