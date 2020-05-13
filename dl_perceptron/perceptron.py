import numpy as np


# one perceptron has single neuron, so it's unnecessary to set neuron number, the dimension of hidden layer
class Perceptron:
    def __init__(self, input_dim, activator):
        self.activator = activator
        self.weights = np.zeros(input_dim, float)
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        wx = np.sum(input_vec * self.weights + self.bias)
        return self.activator(wx)

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    # one iteration, over the whole data
    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for input_vec, label in samples:
            output = self.predict(input_vec)
            self._update_weights(np.array(input_vec), output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        self.weights += rate * delta * input_vec
        self.bias += rate * delta


# step function
def f(x):
    return 1 if x > 0 else 0


def get_training_dataset():
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels


def train_and_perceptron():
    p = Perceptron(2, f)
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p


if __name__ == '__main__':
    and_perceptron = train_and_perceptron()
    print(and_perceptron)
    print(and_perceptron.predict([1, 1]))
    print(and_perceptron.predict([1, 0]))
    print(and_perceptron.predict([0, 0]))
    print(and_perceptron.predict([0, 1]))

