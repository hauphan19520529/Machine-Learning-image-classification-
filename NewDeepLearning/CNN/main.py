# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from optimizer import *
from simple_convnet import SimpleConvNet


# 0:Load MNIST data
#(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label=False)
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

# 1:Settings
network = SimpleConvNet(input_dim=(1,28,28),
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
optimizer = Adam()   #SGD(lr=0.01)
max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
iter_per_epoch = max(train_size / batch_size, 1)
max_iter = int(max_epochs * iter_per_epoch)

train_loss_list = []
train_acc_list = []
test_acc_list = []

epoch_cnt = 0

for i in range(max_iter):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


# Plot
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
"""
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

n_data = 20
row = 4
col = 5
fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(8, 6))

fig.suptitle("MNIST data-set", fontsize=24, color='white')
for i, img in enumerate(x_train[:n_data]):
    _r = i // col
    _c = i % col
    ax[_r,_c].set_title(t_train[i], fontsize=16, color='black')
    ax[_r,_c].axes.xaxis.set_visible(False)
    ax[_r,_c].axes.yaxis.set_visible(False)
    ax[_r,_c].imshow(img.reshape(28, 28), cmap='Greys')

plt.show()
plt.close()
"""
