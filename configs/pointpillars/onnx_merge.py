import onnx

from onnx import TensorProto, ModelProto
from onnx.helper import (make_model, make_graph, make_node, make_tensor_value_info, get_attribute_value)
from onnx.checker import check_model

if __name__ == "__main__":
  pfe_path = "onnx_merge/sim_pfe_10_cls.onnx"
  rpn_path = "onnx_merge/sim_rpn_10_cls.onnx"

  list_path = [pfe_path, rpn_path]
  list_model = []
  model_output = "onnx_merge/pointpillars.onnx"

  for path in list_path:
    model = onnx.load_model(path)
    list_model.append(model)


  list_model[0] = onnx.shape_inference.infer_shapes(list_model[0])    


  model:ModelProto = ModelProto(ir_version=list_model[0].ir_version,
    producer_name=list_model[0].producer_name,
    producer_version=list_model[0].producer_version,
    opset_import=list_model[0].opset_import)


  # add input
  model.graph.name = list_model[0].graph.name
  # model.graph.input.extend(list_model[0].graph.input)
  model.graph.output.extend(list_model[1].graph.output)
  model.graph.value_info.extend(list_model[0].graph.value_info)
  model.graph.value_info.extend(list_model[1].graph.value_info)
  pillars_input = make_tensor_value_info(
    "pillars_input",
    TensorProto.FLOAT,
    # [1, 1, "valid_pillars", 4]
    [-1, 32, 10]

  )
  coords = make_tensor_value_info(
    "coords",
    TensorProto.FLOAT,
    [1, 1, -1, 4]

  )
  params = onnx.helper.make_tensor_value_info(
    "valid_pillars",
    TensorProto.FLOAT,
    [1,]
  )

  model.graph.input.extend([pillars_input, coords, params])
  # make node
  ScatterBev = make_node(op_type="SRR", name="srr_bev" , \
    inputs=["pillars_feats" , "coords", "valid_pillars"],\
    outputs=["rpn_input"], domain="custom")

  list_model[1].graph.node[0].input[0] = "rpn_input"

  model.graph.node.extend(list_model[0].graph.node)
  model.graph.initializer.extend(list_model[0].graph.initializer)
  model.graph.node.append(ScatterBev)
  model.graph.node.extend(list_model[1].graph.node)
  model.graph.initializer.extend(list_model[1].graph.initializer)

  onnx.save(model, model_output)