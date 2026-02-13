import collections
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def sum(x, axis=None, keepdims=False):
    return torch.sum(x, dim=axis, keepdim=keepdims)

def mean(x, axis=None, keepdims=False):
    return torch.mean(x, dim=axis, keepdim=keepdims)

def var(x, axis=None, keepdims=False):
    meanx = mean(x, axis=axis, keepdim=keepdims)
    return mean(torch.square(x - meanx), axis=axis, keepdim=keepdims)

def std(x, axis=None, keepdims=False):
    return torch.sqrt(var(x, axis=axis, keepdim=keepdims))

def max(x, axis=None, keepdims=False):
    return torch.max(x, dim=axis, keepdim=keepdims)[0]

def min(x, axis=None, keepdims=False):
    return torch.min(x, dim=axis, keepdim=keepdims)[0]

def concatenate(arrs, axis=0):
    return torch.cat(arrs, dim=axis)

def argmax(x, axis=None):
    return torch.argmax(x, dim=axis)

def softmax(x, axis=None):
    return F.softmax(x, dim=axis)

# ================================================================
# Mathematical utils
# ================================================================

def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return torch.where(
        torch.abs(x) < delta,
        torch.square(x) * 0.5,
        delta * (torch.abs(x) - 0.5 * delta)
    )

# ================================================================
# Optimizer utils
# ================================================================

def minimize_and_clip(optimizer, objective, parameters, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables while
    ensuring the norm of the gradients for each parameter is clipped to `clip_val`
    """
    optimizer.zero_grad()
    objective.backward()

    if clip_val is not None:
        torch.nn.utils.clip_grad_norm_(parameters, clip_val)

    optimizer.step()

# ================================================================
# Global session and device management
# ================================================================

def get_device():
    """Returns the device to use (CPU or GPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_device(device):
    """Sets the device to use"""
    if isinstance(device, str):
        device = torch.device(device)
    torch.set_default_device(device)

def get_session():
    """Compatibility function - returns device"""
    return get_device()

# ================================================================
# Model utilities
# ================================================================

def mlp(input_size, hidden_sizes, activation=nn.ReLU, output_activation=nn.Identity):
    """Create an MLP network"""
    layers = []
    sizes = [input_size] + list(hidden_sizes)

    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:  # Don't add activation after the last layer
            layers.append(activation())
        elif output_activation != nn.Identity:
            layers.append(output_activation())

    return nn.Sequential(*layers)

# ================================================================
# Variable scope utilities (compatibility layer)
# ================================================================

class VariableScope:
    def __init__(self, name, reuse=None):
        self.name = name
        self.reuse = reuse
        self.params = {}
        self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_variable(self, name, shape, initializer=None, trainable=True):
        key = f"{self.name}/{name}"
        if key in self.params:
            if self.reuse:
                return self.params[key]
            else:
                raise ValueError(f"Variable {key} already exists and reuse is False")

        if initializer is not None:
            if callable(initializer):
                param = torch.empty(shape, device=get_device())
                param = initializer(param)
            else:
                param = torch.tensor(initializer, device=get_device(), dtype=torch.float32)
        else:
            param = torch.empty(shape, device=get_device())
            nn.init.xavier_uniform_(param)

        if trainable:
            param.requires_grad_(True)

        self.params[key] = param
        self._initialized = True
        return param

    def trainable_variables(self):
        return [param for param in self.params.values() if param.requires_grad]

# Global scope stack for compatibility
_scope_stack = []

def variable_scope(name, reuse=None):
    scope = VariableScope(name, reuse)
    return scope

def get_variable_scope():
    return _scope_stack[-1] if _scope_stack else VariableScope("global", reuse=False)

def absolute_scope_name(relative_scope_name):
    """Appends parent scope name to `relative_scope_name`"""
    current_scope = get_variable_scope()
    if current_scope.name:
        return current_scope.name + "/" + relative_scope_name
    return relative_scope_name

# ================================================================
# Saving and loading utilities
# ================================================================

def save_state(model, path):
    """Save model state"""
    torch.save(model.state_dict(), path)

def load_state(model, path):
    """Load model state"""
    model.load_state_dict(torch.load(path, map_location=get_device()))
    return model

# ================================================================
# Function utilities (compatibility layer)
# ================================================================

def function(inputs, outputs, updates=None, givens=None):
    """Simplified version of TensorFlow's function for PyTorch"""
    class TorchFunction:
        def __init__(self, inputs, outputs):
            self.inputs = inputs
            self.outputs = outputs

        def __call__(self, *args, **kwargs):
            # Handle positional arguments
            feed_dict = {}
            for i, (input_tensor, value) in enumerate(zip(self.inputs, args)):
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value).float().to(get_device())
                feed_dict[input_tensor] = value

            # Handle keyword arguments
            for name, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value).float().to(get_device())
                feed_dict[name] = value

            # Run forward pass
            if isinstance(self.outputs, (list, tuple)):
                results = [output(**feed_dict) for output in self.outputs]
            else:
                results = [self.outputs(**feed_dict)]

            # Convert to numpy if needed
            results = [result.detach().cpu().numpy() if isinstance(result, torch.Tensor) else result
                      for result in results]

            return results[0] if len(results) == 1 else results

    return TorchFunction(inputs, outputs)

# ================================================================
# Placeholder utilities (compatibility layer)
# ================================================================

class Placeholder:
    def __init__(self, shape=None, dtype=None, name=None):
        self.shape = shape
        self.dtype = dtype or torch.float32
        self.name = name

    def __repr__(self):
        return f"Placeholder(shape={self.shape}, dtype={self.dtype}, name={self.name})"

def placeholder(shape=None, dtype=None, name=None):
    """Create a placeholder for compatibility with TensorFlow code"""
    return Placeholder(shape, dtype, name)

# ================================================================
# Initializers
# ================================================================

def xavier_uniform_initializer():
    return lambda x: nn.init.xavier_uniform_(x)

def zeros_initializer():
    return lambda x: nn.init.zeros_(x)

# ================================================================
# Global variable management
# ================================================================

def global_variables():
    """Get all global variables (for compatibility)"""
    all_params = []
    # This would need to be implemented based on your model tracking
    return all_params

def variables_initializer(variables):
    """Initialize variables (for compatibility)"""
    # In PyTorch, variables are initialized when created
    def init_fn():
        pass
    return init_fn