import torch
import argparse
from face_model.gpen_model import FullGenerator


parser = argparse.ArgumentParser(description="GPEN torch to ONNX")
parser.add_argument('--origin_model_path', type=str)
parser.add_argument('--export_model_path', type=str)
parser.add_argument('--gpu_id', type=str)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
GPEN_model = FullGenerator(256, 512, 8, 1, narrow=0.5).to(device)
GPEN_model.eval()
state_dict = torch.load(args.origin_model_path)

GPEN_model.load_state_dict(state_dict)

dummy_input = torch.randn(1, 3, 256, 256).to(device)
torch.onnx.export(GPEN_model,
                  dummy_input, 
                  args.export_model_path, 
                  export_params=True, 
                  opset_version=10, 
                  do_constant_folding=True, 
                  input_names=['input'], 
                  output_names=['output'])

print('ONNX model has been exported successfully!')
