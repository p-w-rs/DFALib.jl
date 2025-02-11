# DFALib.jl

A Julia implementation of Direct Feedback Alignment (DFA), an alternative to backpropagation for training neural networks. DFA uses random feedback weights to propagate error signals directly to each layer, allowing for more parallelizable and biologically plausible learning.

## Installation

```julia
] add DFALib
```

## Quick Start

```julia
using DFALib

# Create a DFA network with 784 inputs and 10 outputs
nn = DFANet(
    Dense(784, 256, Sigmoid),  # Input layer
    Dense(256, 128, Sigmoid), # Second hidden layer
    Dense(128, 10) # Output layer
)

# Forward pass
x = rand(Float32, 784)    # Input data
y = nn(x)               # Predictions

# Learning step
error = target - y       # Calculate error
η = 0.01f0              # Learning rate
feedback!(nn, error, η) # Update weights using DFA
```

## Features

- **Direct Feedback Alignment** implementation with parallel weight updates
- **Multiple layer types** supported:
  - Dense (Fully connected) layers
  - Convolutional layers (TODO)
- **Common activation functions**: ReLU, Sigmoid, CELU, LReLU, Softmax, etc.
- **Various weight initializers**: Glorot, He, LeCun, Orthogonal, etc.
- **Type-stable implementation** supporting any AbstractFloat type
- **Multi-threaded feedback** for faster training

## Why DFA?

Direct Feedback Alignment is an alternative to backpropagation that:

- Allows parallel weight updates across layers
- Provides more biological plausibility
- Reduces the memory requirements during training
- Can potentially improve training speed on specialized hardware

## License

MIT License

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
