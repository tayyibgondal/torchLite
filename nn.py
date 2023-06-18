import random
from engine import Value

class Neuron:
  # A neuron needs to know its dimension during initialization
  
  def __init__(self, n_in):
    self.w = [Value(random.uniform(-1,1)) for _ in range(n_in)]
    self.b = Value(random.uniform(-1,1))
  
  def __call__(self, x):
    # w * x + b
    z = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = z.tanh()
    return out
  
  def parameters(self):
    return self.w + [self.b]
  
class Layer:
  
  def __init__(self, n_in, n_out):
    self.neurons = [Neuron(n_in) for _ in range(n_out)]
  
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs
  
  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]
  
class MLP:
  
  def __init__(self, n_in, n_outs):
    units = [n_in] + n_outs
    self.layers = [Layer(units[i], units[i+1]) for i in range(len(n_outs))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]