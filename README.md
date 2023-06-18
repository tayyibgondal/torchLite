# torchLite

This is a basic neural network library that provides functionality for creating and training neural networks. The library includes a scalar-valued autograd engine, which allows for automatic computation of gradients using the chain rule.

## Base Engine

The `Value` class serves as the core building block of the autograd engine. It stores a single scalar value and its gradient. The class supports various arithmetic operations such as addition, multiplication, and exponentiation. Additionally, it provides methods like `relu()` for applying the rectified linear unit (ReLU) activation function and `backward()` for computing gradients using the chain rule.

## Neuron

The `Neuron` class represents a single neuron in a neural network. During initialization, it takes the number of inputs (`n_in`) as a parameter and randomly initializes its weights (`w`) and bias (`b`) using the `Value` class. The neuron can be called with an input `x`, which performs the forward pass through the neuron by calculating the weighted sum of the inputs, adding the bias, and applying the hyperbolic tangent (tanh) activation function.

## Layer

The `Layer` class represents a layer of neurons in a neural network. It takes the number of inputs (`n_in`) and the number of outputs (`n_out`) as parameters during initialization. The layer consists of multiple `Neuron` instances, and calling the layer with an input `x` performs the forward pass by passing the input through each neuron in the layer.

## MLP (Multi-Layer Perceptron)

The `MLP` class represents a multi-layer perceptron, which is a type of neural network architecture. It takes the number of inputs (`n_in`) and a list of output dimensions (`n_outs`) for each layer as parameters during initialization. The `MLP` class contains multiple `Layer` instances, and calling the `MLP` with an input `x` performs the forward pass by passing the input through each layer successively.

## Usage

Here's an example demonstrating how to use this neural network library:

```python
from engine import Value
from neuralnet import MLP

# Create an MLP with 2 inputs, 2 hidden layers (3 neurons each), and 1 output
model = MLP(2, [3, 3, 1])

# Generate sample input
x = [Value(0.5), Value(0.3)]

# Perform forward pass
output = model(x)

# Print the output
print(output)

# Compute gradients using backpropagation
output.backward()

# Access the gradients of the model's parameters
parameters = model.parameters()
for param in parameters:
    print(param.grad)
