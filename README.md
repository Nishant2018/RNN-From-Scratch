## Recurrent Neural Network (RNN)

### Introduction

Recurrent Neural Networks (RNNs) are a class of artificial neural networks designed to recognize patterns in sequences of data, such as time series, speech, or text. Unlike traditional neural networks, RNNs have connections that form directed cycles, allowing information to persist and making them ideal for sequential data.

### Why Use RNN?

- **Sequential Data Handling**: RNNs are designed to work with sequences and time-series data.
- **Memory**: They can maintain memory of previous inputs in the sequence, which is crucial for tasks like language modeling and speech recognition.
- **Versatility**: Useful in various applications like machine translation, sentiment analysis, and video processing.

### How RNN Works

1. **Recurrent Connection**: The key feature of RNNs is their ability to maintain a hidden state that captures information about the sequence. At each time step \( t \), the hidden state \( h_t \) is updated based on the previous hidden state \( h_{t-1} \) and the current input \( x_t \).

2. **Hidden State Update**:
   \[
   h_t = \tanh(W_h h_{t-1} + W_x x_t + b_h)
   \]
   Where:
   - \( h_t \) is the hidden state at time step \( t \).
   - \( W_h \) and \( W_x \) are weight matrices for the hidden state and input, respectively.
   - \( b_h \) is the bias term.
   - \( \tanh \) is the hyperbolic tangent activation function.

3. **Output Calculation**:
   \[
   y_t = W_y h_t + b_y
   \]
   Where:
   - \( y_t \) is the output at time step \( t \).
   - \( W_y \) is the weight matrix for the output.
   - \( b_y \) is the bias term for the output.

### Example Code

Here is a simple example of how to use RNN for a sequence prediction task using Python's `Keras` library:

```python
import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# Generate dummy data
x_train = np.random.random((1000, 10, 1))
y_train = np.random.randint(2, size=(1000, 1))

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(32, input_shape=(10, 1)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
accuracy = model.evaluate(x_train, y_train)
print("Accuracy:", accuracy)
