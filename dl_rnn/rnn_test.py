import copy,numpy as np


def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output


def sigmoid_output_to_derivative(output):
    return output * (1-output)


int2binary = {}
binary_dim = 8

largest_num = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_num)], dtype=np.uint8).T, axis=1)
for i in range(largest_num):
    int2binary[i] = binary[i]

alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

# print(synapse_0)

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

for i in range(1):
    a_int = np.random.randint(largest_num / 2)
    a = int2binary[a_int]

    b_int = np.random.randint(largest_num / 2)
    b = int2binary[b_int]

    c_int = a_int + b_int
    c = int2binary[c_int]

    d = np.zeros_like(c)
    # print(d)

    overallError = 0
    layer_2_deltas = list()
    # hidden output
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

    for position in range(binary_dim):
        # input
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        # output
        y = np.array([[c[binary_dim - position - 1]]]).T

        # 隐层的输出
        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

        # 输出层
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))
        layer_2_error = y - layer_2
        # delta wo
        layer_2_deltas.append(layer_2_error * sigmoid_output_to_derivative(layer_2))

        overallError += np.abs(layer_2_error[0])
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        # 保存隐藏层的值
        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])
        layer_1 = layer_1_values[-position - 1]
        prev_layer_1 = layer_1_values[-position - 2]
        # error at output layer
        layer_2_delta = layer_2_deltas[-position - 1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(
            synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again

        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        future_layer_1_delta = layer_1_delta

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0








