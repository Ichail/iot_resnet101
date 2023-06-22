import torch.onnx
import onnx

path_to_torch_model = input("Enter path to .pt model: ")
model = torch.jit.load(path_to_torch_model)
model.eval().cuda()
dummy_input = torch.autograd.Variable(torch.randn(1, 3, 224, 224))

input_names = ["data"]
output_names = ["output"]
torch.onnx.export(model,
                  dummy_input.cuda(),
                  'resnet-erasing.onnx',
                  input_names=input_names,
                  output_names=output_names,
                  opset_version=11)
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
