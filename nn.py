import torch
import math


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = torch.randn(out_features, in_features) * math.sqrt(2.0 / in_features)

        if bias:
            self.bias = torch.zeros(out_features)
        else:
            self.bias = None

        self.weight_grad = None
        self.bias_grad = None
        self.input_cache = None

    def forward(self, x):
        self.input_cache = x.clone()

        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)

        output = torch.matmul(x, self.weight.t())

        if self.use_bias:
            output = output + self.bias

        return output
    
    def backward(self, grad_output):
        x = self.input_cache
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)

        self.weight_grad = torch.matmul(grad_output.t(), x)

        if self.use_bias:
            self.bias_grad = torch.sum(grad_output, dim=0)

        grad_input = torch.matmul(grad_output, self.weight)

        return grad_input

    def parameters(self):
        params = [self.weight]
        if self.use_bias:
            params.append(self.bias)
        return params