import numpy as np
from tabulate import tabulate


# 2 entradas , 1 saida , rede bipolar (-1 e 1)
# 2 inputs, 1 output, bipolar net (-1,1)


#train
# initialize bias
def train(w, inputs, target):
    b = 1
    for epoch in range(1):
        for j in range(4):
            for i in range(2):
                w[i] = w[i] + (inputs[j, i] * target[j])

            # update bias
            b += target[j]
    print('novos pesos')
    print(w)
    print(f"bias={b}")
    return w, b


# Activation
def activation(a):
    if a < 0:
        return -1
    else:
        return 1
    pass


def test(w, inputs, b):
    y = np.zeros((4, 1))
    y1 = np.zeros((4, 1))
    for j in range(4):
        for i in range(2):
            y[j] += w[i] * inputs[j, i]
        y[j] += b
        y1[j] = activation(y[j])

    # OUTPUT before activation
    print(f"y={y.T}")
    return y1


def printresults(inputs, y1, target):
    # Print output
    y1 = np.append(inputs, y1, axis=1)
    y1 = np.append(y1, target.reshape(-1, 1), axis=1)

    headers = [" in1", "in2", "y1", "target"]
    table = tabulate(y1, headers, tablefmt="fancy_grid")
    print(table)


if __name__ == '__main__':
    print('weights')
    w = np.zeros([2, 1])
    print(w)

    inputs = np.array([
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1]])
    print('inputs')
    print(inputs)

    ##targets
    # AND PORT
    target_1 = np.array([-1, -1, -1, 1])

    # OR PORT
    target_2 = np.array([-1, 1, 1, 1])

    # select target!
    target = target_1

    # train
    w, b = train(w, inputs, target)

    # test
    y1 = test(w, inputs, b)

    # print
    printresults(inputs, y1, target)
