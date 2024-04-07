from face_detect.retinaface_detection import RetinaFaceDetection
import torch
import argparse

parser = argparse.ArgumentParser(description="GPEN torch to ONNX")
parser.add_argument('--origin_model_path', type=str)
parser.add_argument('--export_model_path', type=str)
parser.add_argument('--gpu_id', type=str)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

FaceDetect_model = RetinaFaceDetection(pretrained_path=args.origin_model_path, device=device)

dummy_input = torch.randn(1, 3, 512, 512, requires_grad=True).to(device)
torch.onnx.export(FaceDetect_model.net,           
                  dummy_input,                   
                  args.export_model_path,     
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

print('ONNX model has been exported successfully!')
