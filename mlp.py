import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List

def batch_generator(train_x: np.ndarray, train_y: np.ndarray, batch_size: int):
    """
    Generator that yields mini-batches from the training data.

    :param train_x: Input features of shape (n_samples, n_features).
    :param train_y: Target values of shape (n_samples, n_outputs).
    :param batch_size: Number of samples per mini-batch.
    :yield: Tuple (batch_x, batch_y) for each mini-batch.
    """
    n = train_x.shape[0]
    for start_idx in range(0, n, batch_size):
        end_idx = start_idx + batch_size
        yield train_x[start_idx:end_idx], train_y[start_idx:end_idx]


##############################################################################################################################
#                                               Activation Functions (Abstract Base Class)                                 #
##############################################################################################################################

class ActivationFunction(ABC):
    """
    Abstract base class for activation functions.
    """
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the activation function.

        :param x: Input array.
        :return: Activated output.
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the activation function.

        :param x: Input array (typically the pre-activation value).
        :return: Derivative evaluated at x.
        """
        pass

class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    def derivative(self, x: np.ndarray) -> np.ndarray:
        sig = self.forward(x)
        return sig * (1.0 - sig)

class Tanh(ActivationFunction):
    """
    Hyperbolic tangent activation function.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(x)**2

class Relu(ActivationFunction):
    """
    Rectified Linear Unit (ReLU) activation function.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

class Softmax(ActivationFunction):
    """
    Softmax activation function, used in multi-class classification.
    Note: Its derivative is usually combined with cross-entropy loss.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted_x = x - np.max(x, axis=1, keepdims=True)  # For numerical stability
        exp_scores = np.exp(shifted_x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use cross-entropy derivative directly with Softmax.")

class Linear(ActivationFunction):
    """
    Linear activation function (identity function).
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

class Softplus(ActivationFunction):
    """
    Softplus activation function: f(x) = ln(1 + exp(x)).
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(x))
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

class Mish(ActivationFunction):
    """
    Mish activation function: f(x) = x * tanh(softplus(x)).
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * np.tanh(np.log1p(np.exp(x)))
    def derivative(self, x: np.ndarray) -> np.ndarray:
        sp = np.log1p(np.exp(x))  # softplus(x)
        tanh_sp = np.tanh(sp)     # tanh(softplus(x))
        sigmoid = 1 / (1 + np.exp(-x))  # sigmoid(x)
        return tanh_sp + x * sigmoid * (1 - tanh_sp**2)


##############################################################################################################################
#                                                  Loss Functions (Abstract Base Class)                                #
##############################################################################################################################

class LossFunction(ABC):
    """
    Abstract base class for loss functions.
    """
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the loss value.

        :param y_true: True target values.
        :param y_pred: Predicted values.
        :return: Scalar loss value.
        """
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the loss with respect to predictions.

        :param y_true: True target values.
        :param y_pred: Predicted values.
        :return: Gradient of the loss.
        """
        pass

class SquaredError(LossFunction):
    """
    Mean Squared Error (MSE) loss function, scaled by 1/2 for convenience.
    
    Loss = 0.5 * mean((y_pred - y_true)^2)
    """
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 0.5 * np.mean((y_true - y_pred)**2)
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (y_pred - y_true) / y_true.shape[0]

class CrossEntropy(LossFunction):
    """
    Multi-class cross-entropy loss function.
    """
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        eps = 1e-9  # Prevent log(0)
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (y_pred - y_true) / y_true.shape[0]


##############################################################################################################################
#                                                        Layer                                                           #
##############################################################################################################################

class Layer:
    """
    Represents a single layer in the neural network.
    """
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, dropout_rate: float = 0.0):
        """
        Initialize the layer with weights, biases, and an activation function.

        :param fan_in: Number of input neurons.
        :param fan_out: Number of output neurons.
        :param activation_function: Activation function instance.
        :param dropout_rate: Fraction of neurons to drop during training (0.0 for no dropout).
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        self.activations = None      # Store post-activation outputs
        self.dropout_mask = None     # For dropout regularization
        self.delta = None            # For backpropagation error signal
        self.z = None                # Cache pre-activation (linear combination)

        # Xavier/Glorot uniform initialization for weights.
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))
        self.b = np.zeros((1, fan_out))

    def forward(self, h: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Compute the forward pass of the layer.

        :param h: Input data (batch_size x fan_in).
        :param training: If True, apply dropout.
        :return: Output activations.
        """
        # Compute the linear combination and cache it.
        self.z = np.dot(h, self.W) + self.b
        # Apply the activation function.
        a = self.activation_function.forward(self.z)
        # Apply dropout if training is True.
        if training and self.dropout_rate > 0.0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape)
            a *= self.dropout_mask
        else:
            self.dropout_mask = None
        self.activations = a
        return a

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform the backward pass for the layer.

        :param h: Input to this layer during the forward pass.
        :param delta: Gradient of loss with respect to the layer's output.
        :return: Gradients with respect to weights, biases, and the propagated delta.
        """
        # Adjust delta for dropout.
        if self.dropout_mask is not None:
            delta *= self.dropout_mask

        # For Softmax, use delta directly; otherwise compute using activation derivative.
        if isinstance(self.activation_function, Softmax):
            d_out = delta
        else:
            d_act = self.activation_function.derivative(self.z)  # Use cached pre-activation
            d_out = delta * d_act

        # Compute gradients for weights and biases.
        dL_dW = np.dot(h.T, d_out)
        dL_db = np.sum(d_out, axis=0, keepdims=True)

        # Compute delta to propagate to the previous layer.
        self.delta = np.dot(d_out, self.W.T)
        return dL_dW, dL_db, self.delta


##############################################################################################################################
#                                        Multilayer Perceptron (MLP)                                                   #
##############################################################################################################################

class MultilayerPerceptron:
    """
    Represents a multi-layer perceptron neural network.
    """
    def __init__(self, layers: List[Layer]):
        """
        Initialize the MLP with a list of layers.

        :param layers: List of Layer objects (ordered from input to output).
        """
        self.layers = layers
        self.h_list = []  # To store activations from each layer during forward pass

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform forward propagation through the network.

        :param x: Input data.
        :param training: Whether the network is in training mode (affects dropout).
        :return: Network output.
        """
        self.h_list = [x]
        out = x
        for layer in self.layers:
            out = layer.forward(out, training=training)
            self.h_list.append(out)
        return out

    def backward(self, loss_grad: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform backpropagation through the network.

        :param loss_grad: Gradient of the loss with respect to the network's output.
        :return: Lists of gradients for weights and biases for each layer.
        """
        dl_dw_all = []
        dl_db_all = []
        delta = loss_grad
        # Propagate gradients backwards through layers.
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            h = self.h_list[i]
            dW, dB, delta = layer.backward(h, delta)
            dl_dw_all.insert(0, dW)
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
        batch_size: int = 32, 
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
        :param train_y: Training target data.
        :param val_x: Validation input data.
        :param val_y: Validation target data.
        :param loss_func: Loss function instance.
        :param learning_rate: Learning rate for updates.
        :param batch_size: Number of samples per mini-batch.
        :param epochs: Number of training epochs.
        :param momentum: Momentum factor for gradient updates.
        :param rmsprop: Flag to enable RMSProp updates.
        :param rmsprop_decay: Decay rate for RMSProp.
        :param epsilon: Small constant for numerical stability in RMSProp.
        :return: Arrays of training and validation loss per epoch.
        """
        n_samples = train_x.shape[0]
        training_losses = []
        validation_losses = []

        # Initialize momentum velocities.
        velocities_W = [np.zeros_like(layer.W) for layer in self.layers]
        velocities_b = [np.zeros_like(layer.b) for layer in self.layers]
        # Initialize RMSProp caches if enabled.
        if rmsprop:
            caches_W = [np.zeros_like(layer.W) for layer in self.layers]
            caches_b = [np.zeros_like(layer.b) for layer in self.layers]

        for epoch in range(epochs):
            # Shuffle training data each epoch.
            indices = np.random.permutation(n_samples)
            train_x_shuffled = train_x[indices]
            train_y_shuffled = train_y[indices]
            total_train_loss = 0.0
            num_batches = 0

            for batch_x, batch_y in batch_generator(train_x_shuffled, train_y_shuffled, batch_size):
                # Forward pass on current batch.
                y_pred = self.forward(batch_x, training=True)
                loss_val = loss_func.loss(batch_y, y_pred)
                total_train_loss += loss_val
                num_batches += 1

                # Compute gradient of loss with respect to predictions.
                loss_gradient = loss_func.derivative(batch_y, y_pred)
                # Backward pass: compute gradients for all layers.
                dW_all, dB_all = self.backward(loss_gradient)

                # Update weights and biases.
                for i, layer in enumerate(self.layers):
                    if rmsprop:
                        # Update RMSProp caches.
                        caches_W[i] = rmsprop_decay * caches_W[i] + (1 - rmsprop_decay) * (dW_all[i] ** 2)
                        caches_b[i] = rmsprop_decay * caches_b[i] + (1 - rmsprop_decay) * (dB_all[i] ** 2)
                        # Update parameters using RMSProp.
                        layer.W -= learning_rate * dW_all[i] / (np.sqrt(caches_W[i]) + epsilon)
                        layer.b -= learning_rate * dB_all[i] / (np.sqrt(caches_b[i]) + epsilon)
                    elif momentum > 0.0:
                        # Update momentum velocities.
                        velocities_W[i] = momentum * velocities_W[i] - learning_rate * dW_all[i]
                        velocities_b[i] = momentum * velocities_b[i] - learning_rate * dB_all[i]
                        # Update parameters using momentum.
                        layer.W += velocities_W[i]
                        layer.b += velocities_b[i]
                    else:
                        # Simple gradient descent update.
                        layer.W -= learning_rate * dW_all[i]
                        layer.b -= learning_rate * dB_all[i]

            avg_train_loss = total_train_loss / num_batches
            training_losses.append(avg_train_loss)

            # Compute validation loss (no dropout during validation).
            val_pred = self.forward(val_x, training=False)
            val_loss = loss_func.loss(val_y, val_pred)
            validation_losses.append(val_loss)

            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

        return np.array(training_losses), np.array(validation_losses)
