import cv2
from retinaface_detection import RetinaFaceDetection
import torch

model = RetinaFaceDetection()
onnx_file_path = "resnet_50_2.onnx"
x = torch.randn(2, 3, 512, 512, requires_grad=True).to('cuda:0')
torch.onnx.export(model.net,           
                  x,                   
                  onnx_file_path,     
                  export_params=True,  
                  verbose = False,
                  opset_version=11,   
                  do_constant_folding=True,  
                  input_names=['input'],   
                  output_names=['conf', 'loc','landmarks'], 
                  dynamic_axes={'input': {0: 'batch_size'},    
                                'conf': {0: 'batch_size'}, 
                                'loc': {0: 'batch_size'},
                                'landmarks': {0: 'batch_size'}}) 