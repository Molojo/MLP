import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple, List

def batch_generator(train_x: np.ndarray, train_y: np.ndarray, batch_size: int):
    """
    Generator that yields mini-batches of data from the training set.
    
    :param train_x: Input features array of shape (n_samples, n_features).
    :param train_y: Target values array of shape (n_samples, n_outputs).
    :param batch_size: Number of samples per batch.
    :yield: A tuple (batch_x, batch_y) for each batch.
    """
    n = train_x.shape[0]
    # Loop through the dataset in steps of batch_size
    for start_idx in range(0, n, batch_size):
        end_idx = start_idx + batch_size  # Determine the end index of the batch
        batch_x = train_x[start_idx:end_idx]
        batch_y = train_y[start_idx:end_idx]
        yield batch_x, batch_y  # Yield the mini-batch


##############################################################################################################################
#                                               Activation Functions (Abstract Base Class)                                 #
##############################################################################################################################

class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the activation function for the given input.
        
        :param x: Input array.
        :return: Activated output.
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the activation function with respect to the input.
        
        :param x: Input array.
        :return: Derivative of the activation function evaluated at x.
        """
        pass


class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the sigmoid activation: 1 / (1 + exp(-x))."""
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the sigmoid function.
        The derivative is given by: sigmoid(x) * (1 - sigmoid(x)).
        """
        sig = self.forward(x)
        return sig * (1.0 - sig)


class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the hyperbolic tangent activation."""
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of tanh.
        Since tanh'(x) = 1 - tanh(x)^2, we compute that directly.
        """
        return 1.0 - np.tanh(x)**2


class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the ReLU activation: max(0, x)."""
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of ReLU.
        It returns 1 when x > 0, and 0 when x <= 0.
        """
        return np.where(x > 0, 1, 0)


class Softmax(ActivationFunction):
    """
    Softmax activation is generally used in the final layer for multi-class classification.
    Note: The derivative for Softmax is not computed element-wise here because, when used
    in conjunction with cross-entropy loss, the gradient simplifies to (y_pred - y_true).
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the softmax activation in a numerically stable manner.
        
        This function subtracts the maximum value per sample to avoid numerical overflow.
        
        :param x: Input array of shape (n_samples, n_classes).
        :return: Softmax probabilities of the same shape.
        """
        shifted_x = x - np.max(x, axis=1, keepdims=True)  # Shift for numerical stability
        exp_scores = np.exp(shifted_x)  # Exponentiate the shifted scores
        # Normalize so that probabilities sum to 1 for each sample
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Element-wise derivative is not implemented for softmax.
        When used with cross-entropy loss, the gradient is directly computed as (y_pred - y_true).
        """
        raise NotImplementedError("Derivative for Softmax activation is not implemented; "
                                  "use the cross-entropy loss derivative directly.")


class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Linear activation simply returns the input as output."""
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """The derivative of a linear function is 1 for all elements."""
        return np.ones_like(x)


class Softplus(ActivationFunction):
    """
    Softplus activation function: f(x) = ln(1 + exp(x)).
    Acts as a smooth approximation to the ReLU function.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the softplus activation using the log1p function for numerical stability."""
        return np.log1p(np.exp(x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the softplus function,
        which is equivalent to the sigmoid function: 1 / (1 + exp(-x)).
        """
        return 1 / (1 + np.exp(-x))


class Mish(ActivationFunction):
    """
    Mish activation function: f(x) = x * tanh(softplus(x)).
    
    Mish is a self-regularized non-monotonic activation function.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the Mish activation."""
        return x * np.tanh(np.log1p(np.exp(x)))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the Mish activation.
        A common formulation is:
          d/dx Mish(x) = tanh(softplus(x)) + x * sigmoid(x) * (1 - tanh(softplus(x))^2)
        where sigmoid(x) = 1 / (1 + exp(-x)).
        """
        sp = np.log1p(np.exp(x))  # softplus(x)
        tanh_sp = np.tanh(sp)     # tanh(softplus(x))
        sigmoid = 1 / (1 + np.exp(-x))  # sigmoid(x)
        return tanh_sp + x * sigmoid * (1 - tanh_sp**2)


##############################################################################################################################
#                                                  Loss Functions (Abstract Base Class)                                #
##############################################################################################################################

class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the loss value for a batch of predictions.
        
        :param y_true: True target values.
        :param y_pred: Predicted values from the network.
        :return: A scalar loss value representing the error.
        """
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the loss function with respect to the predictions.
        
        :param y_true: True target values.
        :param y_pred: Predicted values.
        :return: The gradient of the loss with respect to y_pred.
        """
        pass


class SquaredError(LossFunction):
    """
    Mean Squared Error (MSE) loss function, scaled by 1/2 for convenience.
    
    L = 0.5 * mean((y_pred - y_true)^2)
    """
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute the squared error loss by averaging over all samples."""
        return 0.5 * np.mean((y_true - y_pred)**2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the squared error loss.
        
        The gradient is (y_pred - y_true) averaged over the batch.
        This scaling ensures that the update magnitude is independent of the batch size.
        """
        return (y_pred - y_true) / y_true.shape[0]


class CrossEntropy(LossFunction):
    """
    Multi-class cross-entropy loss function.
    
    L = -1/n * sum_over_samples( sum_over_classes( y_true * log(y_pred) ) )
    """
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the cross-entropy loss over a batch.
        
        Clipping y_pred prevents numerical issues with log(0).
        """
        eps = 1e-9  # Small epsilon to prevent taking log(0)
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the cross-entropy loss.
        
        For softmax outputs, the derivative simplifies to (y_pred - y_true) averaged over the batch.
        """
        return (y_pred - y_true) / y_true.shape[0]


##############################################################################################################################
#                                                        Layer                                                           #
##############################################################################################################################

class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, dropout_rate: float = 0.0):
        """
        Initialize a neural network layer with the given parameters.

        :param fan_in: Number of input neurons from the previous layer.
        :param fan_out: Number of neurons in the current layer.
        :param activation_function: Instance of an ActivationFunction to apply.
        :param dropout_rate: Fraction of neurons to drop during training (0.0 means no dropout).
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        self.activations = None  # Store activations after forward pass
        self.dropout_mask = None  # Store dropout mask for backpropagation
        self.delta = None  # Store the delta (error term) during backpropagation

        # Glorot (Xavier) uniform initialization for weights.
        # This helps maintain the variance of activations across layers.
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))
        self.b = np.zeros((1, fan_out))  # Bias initialized to zeros

    def forward(self, h: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform the forward pass for the layer.

        :param h: Input data (shape: [batch_size, fan_in]).
        :param training: Flag indicating if dropout should be applied.
        :return: Output after applying the activation function (and dropout if training).
        """
        # Compute the pre-activation (linear combination)
        z = np.dot(h, self.W) + self.b
        # Apply the activation function to get activations
        a = self.activation_function.forward(z)
        
        # If in training mode and dropout is enabled, create a dropout mask
        if training and self.dropout_rate > 0.0:
            # Create a binary mask where each neuron's output is kept with probability (1 - dropout_rate)
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape)
            a = a * self.dropout_mask  # Apply the mask to the activations
        else:
            self.dropout_mask = None
        
        self.activations = a  # Save activations for use in the backward pass
        return a

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform the backward pass for the layer.

        :param h: Input to the layer during the forward pass.
        :param delta: Gradient of the loss with respect to this layer's output.
        :return: Tuple (dL_dW, dL_db, delta to propagate to previous layer).
        """
        # If dropout was applied, adjust the delta to only include the active neurons.
        if self.dropout_mask is not None:
            delta = delta * self.dropout_mask

        # Recompute the linear combination (pre-activation) for derivative computation.
        z = np.dot(h, self.W) + self.b

        # For Softmax, we assume the loss function handles the derivative,
        # so we use delta directly. Otherwise, compute the derivative of the activation.
        if isinstance(self.activation_function, Softmax):
            d_out = delta
        else:
            d_act = self.activation_function.derivative(z)
            d_out = delta * d_act  # Element-wise multiplication of delta and activation derivative

        # Compute gradients with respect to weights and biases.
        dL_dW = np.dot(h.T, d_out)  # Gradient of loss with respect to weights
        dL_db = np.sum(d_out, axis=0, keepdims=True)  # Gradient of loss with respect to biases

        # Compute delta to pass to the previous layer using the chain rule.
        self.delta = np.dot(d_out, self.W.T)
        return dL_dW, dL_db, self.delta


##############################################################################################################################
#                                        Multilayer Perceptron (MLP)                                                   #
##############################################################################################################################

class MultilayerPerceptron:
    def __init__(self, layers: List[Layer]):
        """
        Initialize the Multilayer Perceptron (MLP).

        :param layers: List of Layer objects in order from the first to the last layer.
        """
        self.layers = layers
        self.h_list = []  # To store activations from each layer during forward pass

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform forward propagation through the network.

        :param x: Input data.
        :param training: If True, applies dropout as configured in each layer.
        :return: Network output.
        """
        self.h_list = [x]  # Initialize list with input data
        out = x
        # Pass input through each layer sequentially
        for layer in self.layers:
            out = layer.forward(out, training=training)
            self.h_list.append(out)  # Save the output (activation) of each layer
        return out

    def backward(self, loss_grad: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backpropagation through the network.

        :param loss_grad: Gradient of the loss with respect to the network's output.
        :return: Two lists containing gradients for weights and biases for each layer.
        """
        dl_dw_all = []  # List to collect gradients for weights
        dl_db_all = []  # List to collect gradients for biases
        delta = loss_grad  # Start with the loss gradient at the output layer

        # Propagate gradients backward through each layer in reverse order
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            h = self.h_list[i]  # Get the corresponding input for this layer
            dW, dB, delta = layer.backward(h, delta)  # Compute gradients and delta for previous layer
            dl_dw_all.insert(0, dW)  # Insert at beginning to maintain layer order
            dl_db_all.insert(0, dB)
        return dl_dw_all, dl_db_all

    def train(
        self, 
        train_x: np.ndarray, 
        train_y: np.ndarray, 
        val_x: np.ndarray, 
        val_y: np.ndarray, 
        loss_func: LossFunction, 
        learning_rate: float = 1E-3, 
        batch_size: int = 16, 
        epochs: int = 100,
        momentum: float = 0.0,
        rmsprop: bool = True,
        rmsprop_decay: float = 0.9,
        epsilon: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the MLP using mini-batch stochastic gradient descent (SGD).

        Supports optional momentum and RMSProp optimizers.

        :param train_x: Training input data.
        :param train_y: Training target values.
        :param val_x: Validation input data.
        :param val_y: Validation target values.
        :param loss_func: Loss function used for training.
        :param learning_rate: Step size for weight updates.
        :param batch_size: Number of samples per mini-batch.
        :param epochs: Number of passes over the training data.
        :param momentum: Momentum factor for gradient descent (if used).
        :param rmsprop: Flag to indicate if RMSProp should be used.
        :param rmsprop_decay: Decay rate for RMSProp caches.
        :param epsilon: Small constant to avoid division by zero in RMSProp.
        :return: Arrays containing the training and validation loss per epoch.
        """
        n_samples = train_x.shape[0]
        training_losses = []  # Record training loss for each epoch
        validation_losses = []  # Record validation loss for each epoch

        # Initialize velocity terms for momentum (if used)
        velocities_W = [np.zeros_like(layer.W) for layer in self.layers]
        velocities_b = [np.zeros_like(layer.b) for layer in self.layers]
        
        # Initialize RMSProp caches if RMSProp is enabled
        if rmsprop:
            caches_W = [np.zeros_like(layer.W) for layer in self.layers]
            caches_b = [np.zeros_like(layer.b) for layer in self.layers]

        # Loop over the number of epochs
        for epoch in range(epochs):
            # Shuffle training data at the start of each epoch for stochasticity
            indices = np.random.permutation(n_samples)
            train_x_shuffled = train_x[indices]
            train_y_shuffled = train_y[indices]

            total_train_loss = 0.0  # Accumulate loss over batches
            num_batches = 0

            # Iterate over mini-batches generated from the shuffled data
            for batch_x, batch_y in batch_generator(train_x_shuffled, train_y_shuffled, batch_size):
                # Forward pass: compute the network's predictions
                y_pred = self.forward(batch_x, training=True)
                # Compute the loss for the current mini-batch
                loss_val = loss_func.loss(batch_y, y_pred)
                total_train_loss += loss_val
                num_batches += 1

                # Compute the gradient of the loss with respect to the predictions
                loss_gradient = loss_func.derivative(batch_y, y_pred)
                # Backpropagate the loss gradient to compute gradients for all layers
                dW_all, dB_all = self.backward(loss_gradient)

                # Update the weights and biases for each layer
                for i, layer in enumerate(self.layers):
                    if rmsprop:
                        # Update RMSProp caches with the square of current gradients
                        caches_W[i] = rmsprop_decay * caches_W[i] + (1 - rmsprop_decay) * (dW_all[i] ** 2)
                        caches_b[i] = rmsprop_decay * caches_b[i] + (1 - rmsprop_decay) * (dB_all[i] ** 2)
                        # Update parameters using RMSProp adjustment
                        layer.W -= learning_rate * dW_all[i] / (np.sqrt(caches_W[i]) + epsilon)
                        layer.b -= learning_rate * dB_all[i] / (np.sqrt(caches_b[i]) + epsilon)
                    elif momentum > 0.0:
                        # Update velocities for momentum-based optimization
                        velocities_W[i] = momentum * velocities_W[i] - learning_rate * dW_all[i]
                        velocities_b[i] = momentum * velocities_b[i] - learning_rate * dB_all[i]
                        # Apply the momentum updates to parameters
                        layer.W += velocities_W[i]
                        layer.b += velocities_b[i]
                    else:
                        # Simple gradient descent update
                        layer.W -= learning_rate * dW_all[i]
                        layer.b -= learning_rate * dB_all[i]

            # Compute average training loss for the epoch
            avg_train_loss = total_train_loss / num_batches
            training_losses.append(avg_train_loss)

            # Compute validation loss (forward pass without dropout)
            val_pred = self.forward(val_x, training=False)
            val_loss = loss_func.loss(val_y, val_pred)
            validation_losses.append(val_loss)

            # Print the loss values for this epoch
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

        return np.array(training_losses), np.array(validation_losses)


def plot_losses(training_losses: np.ndarray, validation_losses: np.ndarray):
    """
    Plot the training and validation loss curves over epochs.

    :param training_losses: Array of training loss values per epoch.
    :param validation_losses: Array of validation loss values per epoch.
    """
    epochs = np.arange(1, len(training_losses) + 1)
    plt.plot(epochs, training_losses, label="Training Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.show()

    # Save the plot as a PNG file
    plt.savefig('losses.png')

