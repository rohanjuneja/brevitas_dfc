import torch
from torch.nn.utils import parameters_to_vector
from brevitas.nn import QuantConv2d, QuantLinear, QuantMultiheadAttention
from brevitas.graph.base import ModuleToModuleByInstance
from brevitas.graph.utils import get_module
from torch import nn
from brevitas import config

QUANT_LAYER_MAP = {
    nn.Conv2d: QuantConv2d, 
    nn.Linear: QuantLinear,
    nn.MultiheadAttention: QuantMultiheadAttention}
config.IGNORE_MISSING_KEYS = True  # required to avoid errors on the weight scale parameter not being part of the resnet18 state_dict

# def threshold():
def threshold(model):
    transforms = []
    graph_model = torch.fx.symbolic_trace(model)
    for node in graph_model.graph.nodes:
        if node.op == 'call_module':
            module = get_module(model, node.target)
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                print(module)
                # transform = ModuleToModuleByInstance(
                #     module, QUANT_LAYER_MAP[module.__class__], 
                #     input_quant=act_quant,
                #     weight_quant=weight_quant,
                #     **act_kwargs_prefix('input_'),
                #     **weight_kwargs)
            elif isinstance(module, nn.MultiheadAttention):
                print(module)
                # transform = ModuleToModuleByInstance(
                #     module, QUANT_LAYER_MAP[module.__class__], 
                #     in_proj_input_quant=act_quant,
                #     in_proj_weight_quant=weight_quant,
                #     in_proj_bias_quant=None,
                #     attn_output_weights_quant=act_quant,
                #     q_scaled_quant=act_quant,
                #     k_transposed_quant=act_quant,
                #     v_quant=act_quant,
                #     out_proj_input_quant=act_quant,
                #     out_proj_weight_quant=weight_quant,
                #     out_proj_bias_quant=None,
                #     **act_kwargs_prefix('in_proj_input_'),
                #     **act_kwargs_prefix('attn_output_weights_'),       
                #     **act_kwargs_prefix('q_scaled_'),
                #     **act_kwargs_prefix('k_transposed_'),
                #     **act_kwargs_prefix('v_'),
                #     **act_kwargs_prefix('out_proj_input_'),                 
                #     **weight_kwargs_prefix('in_proj_'),
                #     **weight_kwargs_prefix('out_proj_'))
            # transforms.append(transform)   

    # for t in transforms:
    #     model = t.apply(model)
    # for name, module in model.named_modules():
    #     if isinstance(module, (nn.Linear, nn.Conv2d)) and not isinstance(module, (QuantLinear, QuantConv2d, QuantMultiheadAttention)):
    #         raise RuntimeError(f"Unquantized {name}")
    return model


    # for layer in model.children():
    # # while 1:
    #     # print(list(layer.parameters()))
    #     weight_og   =   parameters_to_vector(layer.parameters())

    #     #assuming 3d float32 weight tensors
    #     # weight_og = torch.randn(1, 4, 8)
    #     print("weight_og")
    #     print(weight_og)

    #     # weight-freq distribution
    #     weight_dist = torch.arange(-256, 256)
    #     print("weight_dist")
    #     print(weight_dist)
    #     freq_mapping  = torch.randn(512)
    #     print("freq_mapping")
    #     print(freq_mapping)
    #     target_freq = 2.0
    #     #list of optimal weight values. It should be a 1D list?
    #     weight_optimal = weight_dist[freq_mapping < target_freq]
    #     print("weight_optimal")
    #     print(weight_optimal)

    #     flattened_weights = weight_og.flatten()
    #     # Convert the quantize_range to a PyTorch tensor
    #     quantize_range_tensor = torch.tensor(weight_optimal)
    #     # Quantize each element to its closest value in quantize_range
    #     quantized_weights = quantize_range_tensor[torch.argmin(torch.abs(flattened_weights.unsqueeze(0) - quantize_range_tensor.unsqueeze(1)), dim=0)]
    #     # Reshape the quantized array back to the original tensor shape
    #     quantized_weights = quantized_weights.reshape(weight_og.shape)

    #     print("quantized_weights")
    #     print(quantized_weights)
    #     break


if __name__ == "__main__":
    threshold()
