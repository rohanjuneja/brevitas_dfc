import numpy as np
import math
import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM

criticalPath = np.loadtxt("data/time.txt")
max_elem = criticalPath.max()
print("max " + str(max_elem))

# https://discuss.pytorch.org/t/how-to-load-pytorch-model/66432
quantized_model = OPTForCausalLM.from_pretrained('facebook/opt-1.3b', torch_dtype=torch.float16, device_map='auto')
quantized_model.load_state_dict('data/weights_sq_w8a8_sd.pth')

for name, param in quantized_model.named_parameters():
    print(name, param.size())

# for layer in quantized_model.children():
#     weights =   list(layer.parameters())
