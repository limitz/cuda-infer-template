#!/bin/python3
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
# MODIFIED BY LIMITZ

import os
import torch
import torch.nn as nn

width = 1920
height = 1080
scale = 2
model_load = ('pytorch/vision:v0.6.0', 'fcn_resnet101' )
model_width = int(width / scale)
model_height = int(height / scale)
model_format = "{name}.{width}x{height}.{ext}"
onnx_name = model_format.format(name = model_load[1], width = model_width, height = model_height, ext= "onnx");
engine_name = model_format.format(name = model_load[1], width = model_width, height = model_height, ext= "engine");


# FC-ResNet101 pretrained model from torch-hub extended with argmax layer
class fcn_resnet101(nn.Module):
    def __init__(self):
        super(fcn_resnet101, self).__init__()
        self.model = torch.hub.load(model_load[0], model_load[1], pretrained=True)

    def forward(self, inputs):
        x = self.model(inputs) 
        #x = (x[0], torch.nn.functional.softmax(x[1], dim=2))
        x = x['out'].argmax(1, keepdims=True)
        return x;

model = fcn_resnet101()
model.eval()
input_tensor = torch.rand(1, 3, model_height, model_width)
torch.onnx.export(model, input_tensor, onnx_name,
    opset_version=12,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    verbose=True)


os.system(
        "trtexec --onnx={onnx} --saveEngine={engine} --optShapes=1x3x{height}x{width} --workspace=2048 --best"
        .format(onnx = onnx_name, engine = engine_name, width = model_width, height = model_height)
);
