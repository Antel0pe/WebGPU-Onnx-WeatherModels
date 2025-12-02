import onnx

model = onnx.load("./public/pangu_weather_1.onnx")
graph = model.graph

print("Inputs:")
for inp in graph.input:
    print(inp.name, inp.type.tensor_type.shape)

print("Outputs:")
for out in graph.output:
    print(out.name, out.type.tensor_type.shape)
