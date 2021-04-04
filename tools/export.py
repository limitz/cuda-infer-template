#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from PIL import Image
from io import BytesIO
import requests

output_image="input.ppm"

# Read sample image input and save it in ppm format

import torch
import torch.nn as nn

square = 0
scale = 8
output_onnx="resnet101-fcn-480x270.onnx"

wf = 3840 / scale
hf = 2160 / scale
if square:
    wf = hf

w = int(wf)
h = int(hf)
print(wf,hf,w,h)

# FC-ResNet101 pretrained model from torch-hub extended with argmax layer
class FCN_ResNet101(nn.Module):
    def __init__(self):
        super(FCN_ResNet101, self).__init__()
        #self.model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
        #self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS', pretrained=True)
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True)
        #self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', 
        #        model_math="fp32", pretrained=True)

    def forward(self, inputs):
        x = self.model(inputs)
        #print(x.shape)
        #x = (x[0], torch.nn.functional.softmax(x[1], dim=2))
        x = x['out'].argmax(1, keepdims=True)
        return x;

model = FCN_ResNet101()
model.eval()


# Generate input tensor with random values
input_tensor = torch.rand(1, 3, h, w)

# Export torch model to ONNX
print("Exporting ONNX model {}".format(output_onnx))
torch.onnx.export(model, input_tensor, output_onnx,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    #dynamic_axes={"input": {0: "batch"},
    #              "output":  {0: "batch"}},
    verbose=False)

