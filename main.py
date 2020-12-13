from matplotlib import pylab as plt
import pandas as pd
from scipy.misc import derivative

def graph(x):
    return ((x+22)+10)**2
def dergraph(x):
    eps = 8e-9
    return (graph(x + eps)-graph(x))/eps
def sDefGraph(x):
    return derivative(graph, x)

der_custom = dergraph(0)
der_lib = sDefGraph(0)
print('custom  lib = ', der_custom)
print('standart lib = ', der_lib)

#gradient descent
def grad_desc(alpha, eps, epoch, deritarive):
    x_prev = 0
    y_prev = graph(x_prev)
    Y = {x_prev: y_prev}
    for i in range (epoch):
        x_new = x_prev - alpha *deritarive(x_prev)
        y_new = graph(x_new)
        Y[x_new] = y_new

        if abs(x_new - x_prev) <= eps:
            return Y
        x_prev = x_new
    return Y

#global minimum
gradient = grad_desc(0.3, 10e-3, 1000, dergraph)

some_res = pd.DataFrame(gradient.items(), columns=['x', 'y'])
print(some_res)
res = some_res.tail(1)
print('\nGlobal minimum = \n', res, '\n')
x = list(gradient.keys())
y = list(gradient.values())
f_x = range(int(min(x[0], x[-1])) - 3, int(max(x[0], x[-1])) +3)
f_y = [graph(xi) for xi in f_x]
plt.plot(f_x, f_y, x, y, 'ro-', res['x'], res['y'], 'go')
plt.legend(['y = ((x+22)+10)**2', 'gradient descent', 'result'])
plt.show()

# testing result
alpha = 0.1
eps = 0.0001
epoch = 10
gradient = grad_desc(alpha, eps, epoch, dergraph)
x = list(gradient.keys())[-1]
y = gradient[x]
print('Global minimum for alpha = ',alpha,  ', eps = ',eps, ', iteration = ', len(gradient))
print('custom deriative: x = ', x, ', y = ', y)
std_gradient = grad_desc(alpha, eps, epoch, sDefGraph)
std_x = list(std_gradient.keys())[-1]
std_y = std_gradient[std_x]
print('standart library deriative: x = ', std_x, ', y = ', std_y)
print('precision x:', std_x/x*100, 'y: ', std_y/y*100 )

