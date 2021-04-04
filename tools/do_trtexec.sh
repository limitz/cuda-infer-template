trtexec --onnx=resnet101-fcn-480x270.onnx --saveEngine=/home/limitz/resnet101-fcn-480x270.engine --explicitBatch --optShapes=input:1x3x270x480 --workspace=2048
