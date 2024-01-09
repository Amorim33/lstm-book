# LSTM Book
A repository to store studies and POCs related to LSTM neural networks.

The main goal is to implement a LSTM neural network from scratch.

## LSTM
### Problem
`RNNs` are great at processing sequential data, but they suffer from the vanishing or exploding gradient problem. 

Recurrent neural networks have a hard time learning long-term dependencies in the data, that is, when there is a long gap between the relevant information in the sequence.
> "The child ran away from the dining table so he could play." Here, the word "child" is relevant to the word "play", but it is separated by a long distance. we need to remember the word "child" for a long time to make the connection and predict the word "play" correctly.

The vanishing gradient problem is when the gradient shrinks as it back-propagates through time. This makes it hard for the model to learn dependencies that are many time steps away. The exploding gradient problem is the opposite, the gradient grows exponentially.

### Solution

`LSTMs` solve the vanishing gradient problem that occurs in traditional RNNs. LSTMs have a forget gate in addition to the input and output gates of traditional RNNs. The forget gate allows the LSTM to drop information that is no longer relevant to the prediction at a certain time step. This allows the LSTM to retain long-term dependencies in the data.
Also the cell state allows the network to retain information unchanged over many time steps.

LSTMs have three main gates. Each gate is like a filter that either lets information through or blocks it. They are:
- Forget Gate: decides what information to throw away from the cell state.
- Input Gate: decides which values from the input to update the cell state.
- Output Gate: decides what to output based on input and the memory of the cell.

Additionally it has a cell state:
- Cell State: the actual memory of the cell based on Forget and input gates.

<!-- 
TODO: review this section, maybe the concept of the gates is not correct.

## Forward pass

### 1. Propose new candidate cell state

The vector $\tilde{C_t}$ is the new candidate values, that could be added to the state. It is computed based on the current input $x_t$ and the previous hidden state $h_{t-1}$.

$$\tilde{C_t}  = tanh(U_{c}x_t + V_{c}h_{t-1} + b_{c})$$

For the cell state a tanh activation function is used to squish the values between -1 and 1. Basically, the values will float around zero.

### 2. Decide what to keep/forget from the last cell state

The `forget gate` decides what information to throw away from the cell state. It looks at $h_{t-1}$ and $x_t$ and outputs a number between 0 and 1 for each number in the cell state $C_{t-1}$. A 1 represents "completely keep this" while a 0 represents "completely get rid of this".

$$f_t = \sigma(U_{f}x_t + V_{f}h_{t-1} + b_{f})$$

The forget gate is a sigmoid function, because we want to keep values between 0 and 1. This output will be useful when we element-wise multiply it with the cell state.

### 3. Decide what to keep/forget from the new candidate cell state

The `input gate` decides what new information to store in the cell state. It is basically the same of the `forget gate`, but applied to the new candidate cell state $\tilde{C}_t$.

$$i_t = \sigma(U_{i}x_t + V_{i}h_{t-1} + b_{i})$$

### 4. Create new cell state

The new cell state is a combination of the previous cell state and the new candidate cell state. The previous cell state is multiplied by $f_t$ to forget the things we decided to forget earlier. Then we add $i_t*\tilde{C}_t$. This is the new candidate values, scaled by how much we decided to update each state value.

$$C_t = f_t * C_{t-1} + i_t * \tilde{C_t}$$

> It is important to note that the operations are element-wise.

### 5. Decide what to output

The `output gate` decides what to output based on the cell state. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid function on the input $x_t$ and the previous hidden state $h_{t-1}$. Then, we multiply that with the cell state. This way, we only output the parts we decided to.

$$o_t = \sigma(U_{o}x_t + V_{o}h_{t-1} + b_{o})$$

### 6. Output (hidden state)

Finally, we have to compute the hidden state $h_t$. This will be our output for the current time step. First, we run a tanh function on the cell state. Then, we multiply it by the output $o_t$ of the output gate.

$$h_t = o_t*tanh(C_t)$$

### 7. Prediction (final gate)

The prediction is made based on the hidden state $h_t$.

$$\hat{y_t} = softmax(V_{y}h_t + b_{y})$$

The softmax function is used to normalize the output of the network to a probability distribution.

### Additional context

The variables $U$, $V$ and $b$ are the parameters of the LSTM. They are adjusted during the training process. The matrices $U$ and $V$ are the weights and they are multiplied by the input $x_t$ and the previous hidden state $h_{t-1}$ respectively. The bias vector $b$ is added to the result.

 -->
