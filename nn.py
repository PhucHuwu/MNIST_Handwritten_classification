import torch
import numpy as np
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


class ReLU:
    def __init__(self):
        self.input_cache = None

    def forward(self, x):
        self.input_cache = x.clone()
        return torch.clamp(x, min=0)

    def backward(self, grad_output):
        relu_mask = (self.input_cache > 0).float()
        return grad_output * relu_mask

    def parameters(self):
        return []


class Softmax:
    def __init__(self, dim=-1):
        self.dim = dim
        self.output_cache = None

    def forward(self, x):
        x_max = torch.max(x, dim=self.dim, keepdim=True)[0]
        x_shifted = x - x_max

        exp_x = torch.exp(x_shifted)
        softmax_output = exp_x / torch.sum(exp_x, dim=self.dim, keepdim=True)

        self.output_cache = softmax_output.clone()
        return softmax_output

    def backward(self, grad_output):
        y = self.output_cache
        grad_input = y * grad_output
        sum_grad = torch.sum(grad_input, dim=self.dim, keepdim=True)
        grad_input = grad_input - y * sum_grad
        return grad_input

    def parameters(self):
        return []


class Flatten:
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape

        if self.end_dim == -1:
            return x.view(x.size(0), -1)
        else:
            return x.flatten(self.start_dim, self.end_dim)

    def backward(self, grad_output):
        return grad_output.view(self.input_shape)

    def parameters(self):
        return []


class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class CrossEntropyLoss:
    def __init__(self):
        self.input_cache = None
        self.target_cache = None
        self.softmax = Softmax(dim=1)

    def forward(self, input_logits, target):
        self.input_cache = input_logits.clone()
        self.target_cache = target.clone()

        probabilities = self.softmax.forward(input_logits)

        batch_size = target.size(0)

        if len(target.shape) == 1:
            target_one_hot = torch.zeros_like(probabilities)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        else:
            target_one_hot = target

        epsilon = 1e-15
        probabilities = torch.clamp(probabilities, epsilon, 1 - epsilon)

        loss = -torch.sum(target_one_hot * torch.log(probabilities)) / batch_size
        return loss

    def backward(self):
        probabilities = self.softmax.output_cache
        batch_size = self.target_cache.size(0)

        if len(self.target_cache.shape) == 1:
            target_one_hot = torch.zeros_like(probabilities)
            target_one_hot.scatter_(1, self.target_cache.unsqueeze(1), 1.0)
        else:
            target_one_hot = self.target_cache

        grad_input = (probabilities - target_one_hot) / batch_size
        return grad_input


class Adam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        self.m = []
        self.v = []

        for param in parameters:
            self.m.append(torch.zeros_like(param))
            self.v.append(torch.zeros_like(param))

    def step(self, gradients):
        self.t += 1

        for i, (param, grad) in enumerate(zip(self.parameters, gradients)):
            if grad is None:
                continue

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)

            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        pass


class Net:
    def __init__(self):
        self.flatten = Flatten()

        self.first_layer = Linear(784, 256)
        self.first_activation = ReLU()

        self.second_layer = Linear(256, 128)
        self.second_activation = ReLU()

        self.output_layer = Linear(128, 10)
        self.output_activation = Softmax(dim=1)

        self._parameters = []
        self._parameters.extend(self.first_layer.parameters())
        self._parameters.extend(self.second_layer.parameters())
        self._parameters.extend(self.output_layer.parameters())

    def forward(self, x):
        x = self.flatten.forward(x)
        x = self.first_layer.forward(x)
        x = self.first_activation.forward(x)
        x = self.second_layer.forward(x)
        x = self.second_activation.forward(x)
        x = self.output_layer.forward(x)
        x = self.output_activation.forward(x)
        return x

    def backward(self, grad_output):
        grad = self.output_activation.backward(grad_output)
        grad = self.output_layer.backward(grad)
        grad = self.second_activation.backward(grad)
        grad = self.second_layer.backward(grad)
        grad = self.first_activation.backward(grad)
        grad = self.first_layer.backward(grad)
        grad = self.flatten.backward(grad)
        return grad

    def get_gradients(self):
        gradients = []
        gradients.extend([self.first_layer.weight_grad, self.first_layer.bias_grad])
        gradients.extend([self.second_layer.weight_grad, self.second_layer.bias_grad])
        gradients.extend([self.output_layer.weight_grad, self.output_layer.bias_grad])
        return gradients

    def parameters(self):
        return self._parameters

    def train(self):
        pass

    def eval(self):
        pass


if __name__ == "__main__":
    print("Testing  Linear Layer...")
    linear = Linear(784, 256)
    x = torch.randn(32, 784)
    output = linear.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    print("\nTesting  ReLU...")
    relu = ReLU()
    relu_output = relu.forward(output)
    print(f"ReLU output shape: {relu_output.shape}")
    print(f"ReLU applied (first 5 values): {relu_output[0][:5]}")

    print("\nTesting  Softmax...")
    softmax = Softmax()
    final_layer = Linear(256, 10)
    logits = final_layer.forward(relu_output)
    softmax_output = softmax.forward(logits)
    print(f"Softmax output shape: {softmax_output.shape}")
    print(f"Softmax probabilities (first sample): {softmax_output[0]}")
    print(f"Sum of probabilities: {torch.sum(softmax_output[0])}")
