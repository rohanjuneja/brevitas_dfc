import torch
from torch.nn.utils import parameters_to_vector

# def threshold():
def threshold(model):
    for layer in model.parameters():
    # while 1:
        # print(list(layer.parameters()))
        weight_og   =   layer.quant_weight().int()

        #assuming 3d float32 weight tensors
        # weight_og = torch.randn(1, 4, 8)
        print("weight_og")
        print(weight_og)

        # weight-freq distribution
        weight_dist = torch.arange(-256, 256)
        print("weight_dist")
        print(weight_dist)
        freq_mapping  = torch.randn(512)
        print("freq_mapping")
        print(freq_mapping)
        target_freq = 2.0
        #list of optimal weight values. It should be a 1D list?
        weight_optimal = weight_dist[freq_mapping < target_freq]
        print("weight_optimal")
        print(weight_optimal)

        flattened_weights = weight_og.flatten()
        # Convert the quantize_range to a PyTorch tensor
        quantize_range_tensor = torch.tensor(weight_optimal)
        # Quantize each element to its closest value in quantize_range
        quantized_weights = quantize_range_tensor[torch.argmin(torch.abs(flattened_weights.unsqueeze(0) - quantize_range_tensor.unsqueeze(1)), dim=0)]
        # Reshape the quantized array back to the original tensor shape
        quantized_weights = quantized_weights.reshape(weight_og.shape)

        print("quantized_weights")
        print(quantized_weights)
        break


if __name__ == "__main__":
    threshold()
