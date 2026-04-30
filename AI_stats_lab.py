'''
AI_stats_lab.py
 
Neural Networks Lab: 3-Layer Forward Pass and Backpropagation
 
Implement all functions.
Do NOT change function names.
Do NOT print inside functions.
'''
 
import numpy as np
 
 
def sigmoid(z):
    """
    sigmoid(z) = 1 / (1 + exp(-z))
    
    Converts any number into a value between 0 and 1.
    Think of it like a "squeeze function" - it compresses numbers into 0-1 range.
    
    Args:
        z: input value or array
    
    Returns:
        sigmoid output (value between 0 and 1)
    """
    return 1 / (1 + np.exp(-z))
 
 
def forward_pass(X, W1, W2, W3):
    """
    3-layer neural network forward pass.
    
    Think of it like a factory assembly line:
    Input -> Layer 1 -> Layer 2 -> Output
    
    Each layer processes information and passes it to the next layer.
 
    Layer 1 (Hidden Layer 1):
        First, we mix inputs using weight matrix W1
        Then apply sigmoid to squash values between 0-1
        h1 = sigmoid(X @ W1)
 
    Layer 2 (Hidden Layer 2):
        h1 is now input to second layer
        h2 = sigmoid(h1 @ W2)
 
    Output layer:
        Final predictions
        y = sigmoid(h2 @ W3)
 
    Args:
        X: Input data (shape: n_samples × input_features)
        W1: Weights from input to hidden layer 1 (shape: input_features × hidden1_size)
        W2: Weights from hidden layer 1 to hidden layer 2 (shape: hidden1_size × hidden2_size)
        W3: Weights from hidden layer 2 to output (shape: hidden2_size × 1)
 
    Returns:
        h1: Output of layer 1
        h2: Output of layer 2
        y: Final predictions
    """
    # Layer 1: input -> hidden layer 1
    z1 = X @ W1  # Matrix multiplication: input × weights
    h1 = sigmoid(z1)  # Apply sigmoid activation
    
    # Layer 2: hidden layer 1 -> hidden layer 2
    # ⚠️ IMPORTANT: Use h1 here, NOT X
    z2 = h1 @ W2
    h2 = sigmoid(z2)
    
    # Output layer: hidden layer 2 -> output
    z3 = h2 @ W3
    y = sigmoid(z3)
    
    return h1, h2, y
 
 
def backward_pass(X, h1, h2, y, label, W1, W2, W3):
    """
    Backpropagation for a 3-layer sigmoid neural network.
    
    This is how the network learns! We go BACKWARDS from output to input,
    calculating how much each weight contributed to the error.
    
    Think of it like:
    1. Calculate error at output
    2. Figure out which weights caused the error
    3. Adjust weights to reduce error
    
    The chain rule from calculus helps us do this efficiently.
 
    Steps:
    1. Calculate output error (how far we were from correct answer)
    2. Backpropagate to layer 2 weights
    3. Backpropagate to layer 1 weights
    4. Calculate loss (overall error)
 
    Args:
        X: Input data
        h1: Hidden layer 1 activations
        h2: Hidden layer 2 activations
        y: Predicted output
        label: True label (correct answer)
        W1, W2, W3: Weight matrices
 
    Returns:
        dW1: Gradient for W1 (how much to change W1)
        dW2: Gradient for W2 (how much to change W2)
        dW3: Gradient for W3 (how much to change W3)
        loss: Binary cross-entropy loss (how wrong we were)
    """
    
    n_samples = X.shape[0]
    
    # Reshape label if it's a scalar
    if np.isscalar(label):
        label = np.array([[label]])
    elif label.ndim == 1:
        label = label.reshape(-1, 1)
    
    # Step 1: Calculate loss (Binary Cross-Entropy)
    # This measures how far our prediction is from the true answer
    # Loss = -[label*log(y) + (1-label)*log(1-y)]
    loss = -np.mean(label * np.log(y + 1e-8) + (1 - label) * np.log(1 - y + 1e-8))
    
    # Step 2: Output layer error
    # How much did we miss? (prediction - true answer)
    dz3 = y - label  # This is the error at output
    
    # Step 3: Calculate gradient for W3
    # dW3 = h2.T @ dz3 / n_samples
    # We use h2 because W3 connects h2 to output
    dW3 = (h2.T @ dz3) / n_samples
    
    # Step 4: Backpropagate error to h2
    # Multiply error by W3 and apply sigmoid derivative
    dh2 = dz3 @ W3.T  # Pass error backwards through W3
    dz2 = dh2 * h2 * (1 - h2)  # Apply sigmoid derivative
    
    # Step 5: Calculate gradient for W2
    # dW2 = h1.T @ dz2 / n_samples
    dW2 = (h1.T @ dz2) / n_samples
    
    # Step 6: Backpropagate error to h1
    # Multiply error by W2 and apply sigmoid derivative
    dh1 = dz2 @ W2.T
    dz1 = dh1 * h1 * (1 - h1)
    
    # Step 7: Calculate gradient for W1
    # dW1 = X.T @ dz1 / n_samples
    dW1 = (X.T @ dz1) / n_samples
    
    return dW1, dW2, dW3, loss
