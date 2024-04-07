import numpy as np
import torch
from collections import OrderedDict
import onnxruntime
import argparse
from GFPGAN import GFPGAN

parser = argparse.ArgumentParser("GFPGAN torch to ONNX")
parser.add_argument('--origin_model_path', type=str, default=None)
parser.add_argument('--export_model_path', type=str, default=None)
parser.add_argument('--gpu_id', type=str)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

enhance_model = GFPGAN().to(device)
enhance_model.eval()


state_dict = torch.load(args.origin_model_path)['params_ema']
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if "stylegan_decoder" in k:
        k = k.replace('.', 'dot')
        new_state_dict[k] = v
        k = k.replace('dotweight', '.weight')
        k = k.replace('dotbias', '.bias')
        new_state_dict[k] = v
    else:
        new_state_dict[k] = v
     
model.load_state_dict(new_state_dict, strict=False)


onnx_input_shape = torch.rand(1, 3, 512, 512).to(device)
torch.onnx.export(enhance_model, 
                  onnx_input_shape, 
                  args.export_model_path,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output'])


print('ONNX model has been exported successfully!')
