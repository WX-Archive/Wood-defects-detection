import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import os
import numpy as np
import time

def inception(x, W, name_in):
    ## Branch: 1x1 ##
    y1 = tf.nn.conv2d(x, W[0], strides = [1, 1, 1, 1], padding = "SAME")
    ## Branch: 3x3 ##
    y = tf.nn.conv2d(x, W[1], strides = [1, 1, 1, 1], padding = "SAME")
    y3 = tf.nn.conv2d(y, W[2], strides = [1, 1, 1, 1], padding = "SAME")
    ## Branch: 5x5 ##
    y = tf.nn.conv2d(x, W[3], strides = [1, 1, 1, 1], padding = "SAME")
    y5 = tf.nn.conv2d(y, W[4], strides = [1, 1, 1, 1], padding = "SAME")
    ## Branch: maxpool ##
    y = tf.nn.max_pool(x, ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = "SAME") # maxpool
    ymax = tf.nn.conv2d(y, W[5], strides = [1, 1, 1, 1], padding = "SAME")
    #print(y1.shape, y3.shape, y5.shape, ymax.shape)
    y = tf.concat((y1, y3, y5, ymax), 3, name_in)
    ## Activation ##
    #m, v = tf.nn.moments(y, [0])
    #sd = W[6] * (y - m) / tf.pow((v + eps), 0.5) + W[7]
    #y_0 = tf.nn.relu(y, name_in)
    #print(d)
    return y

def convolution2D(x, W, srd, pad, par, name_in):
    y = tf.nn.conv2d(x, W[0], strides = [1, 1, 1, 1], padding = pad)
    if par == 1:
        m, v = tf.nn.moments(y, [0])
        d = W[1] * (y - m) / tf.pow((v + eps), 0.5) + W[2]
    if par == 0:
        d = y
    #d = tf.nn.relu(sd, name_in)
    return d

def convolution2D_res(x, W, srd, pad, par, name_in):
    s = 0
    y = tf.nn.conv2d(x, W[0], strides = [1, 1, 1, 1], padding = pad)
    #d = tf.nn.leaky_relu(y, s, name_in)
    y = tf.nn.conv2d(y, W[1], strides = [1, srd, srd, 1], padding = pad)
    if par == 1:
        m, v = tf.nn.moments(y, [0])
        d = W[2] * (y - m) / tf.pow((v + eps), 0.5) + W[3]
    if par == 0:
        d = y
    #result = tf.nn.leaky_relu(inp + y2, s, name_in)
    result = d
    return result

def fractorial(x):
    y = 1
    for i in range(x):
        y = (i + 1) * y
    return y

def mish(x):
    y = x * tf.nn.tanh(tf.log(1 + tf.exp(x)))
    return y

def swish(x):
    y = x / (1 + tf.exp(-x))
    return y

def activation(x, num, W):
    #print(W)
    cons = 1e-2
    if num == 0:
        result = tf.nn.relu(x, 'x')
        #result = tf.nn.relu(W[0] * x / cons, 'x')
    if num == 1: # sin
        result = tf.sin(x, 'x')
        #result = tf.sin(x * W[0] / cons, 'x')
        #result = (W[0] * tf.sin(x) + (W[1] - cons) * tf.sin(2 * x)) / cons # order 2
        #result = (W[0] * tf.sin(x) + (W[1] - cons) * tf.sin(2 * x) + (W[2] - cons) * tf.sin(3 * x) + (W[3] - cons) * tf.sin(4 * x)) / cons # order 4
    if num == 2: # Quadratic Unit
        #result = (W[0] * x + W[1] * x**2) / cons
        result = x + x**2
    if num == 3: # Mish
        result = x * tf.nn.tanh(tf.log(1 + tf.exp(x)))
        #result = mish(mish(mish(mish(W[0] * x / cons + (W[1] - cons) / cons) * W[2] / cons + (W[3] - cons) / cons) * W[4] / cons + (W[5] - cons) / cons) * W[6] / cons + (W[7] - cons) / cons)
        #result = mish(mish(mish(W[0] * x / cons + (W[1] - cons) / cons) * W[2] / cons + (W[3] - cons) / cons) * W[4] / cons + (W[5] - cons) / cons)
        #result = mish(mish(W[0] * x / cons + (W[1] - cons) / cons) * W[2] / cons + (W[3] - cons) / cons)
        #result = mish(W[0] * x / cons + (W[1] - cons) / cons)
    if num == 4: # leaky_relu
        result = tf.nn.leaky_relu(x, 0.2, 'x')
    if num == 5: # Cubic Unit
        #result = W[0] * x * 100 - W[1] * 100 * x**3
        result = x - x**3
    if num == 6: # Gaussian
        result = tf.exp(-x**2)
    if num == 7: # Taylor
        #result = (W[0] * x + (W[1] - cons) * x**2) / cons
        #result = (W[0] * x + (W[1] - cons) * x**2 + (W[2] - cons) * x**3) / cons
        #result = (W[0] * x + (W[1] - cons) * x**2 + (W[2] - cons) * x**3 + (W[3] - cons) * x**4) / cons
        #result = (W[0] * x + (W[1] - cons) * x**2 + (W[2] - cons) * x**3 + (W[3] - cons) * x**4 + (W[4] - cons) * x**5) / cons
        result = (W[0] * x + (W[1] - cons) * x**2 + (W[2] - cons) * x**3 + (W[3] - cons) * x**4 + (W[4] - cons) * x**5 + (W[5] - cons) * x**6) / cons
        #result = (W[0] * x + W[1] * x**2 + (W[2] - cons) * x**3 + (W[3] - cons) * x**4 + (W[4] - cons) * x**5 + (W[5] - cons) * x**6 + (W[6] - cons) * x**7) / cons
        #result = (W[0] * x + (W[1] - cons) * x**2 + (W[2] - cons) * x**3 + (W[3] - cons) * x**4 + (W[4] - cons) * x**5 + (W[5] - cons) * x**6 + (W[6] - cons) * x**7 + (W[7] - cons) * x**8) / cons
        #result = (W[0] * x + W[1] * x**2 + (W[2] - cons) * x**3 + (W[3] - cons) * x**4 + (W[4] - cons) * x**5 + (W[5] - cons) * x**6 + (W[6] - cons) * x**7 + (W[7] - cons) * x**8 + (W[8] - cons) * x**9) / cons
        #result = (W[0] * x + W[1] * x**2 + (W[2] - cons) * x**3 + (W[3] - cons) * x**4 + (W[4] - cons) * x**5 + (W[5] - cons) * x**6 + (W[6] - cons) * x**7 + (W[7] - cons) * x**8 + (W[8] - cons) * x**9 + (W[9] - cons) * x**10) / cons
    if num == 8: # SiLU or swish
        result = swish(x)
        #result = swish(swish(W[0] * x / cons + (W[1] - cons) / cons) * W[2] / cons + (W[3] - cons) / cons)
    if num == 9: # Taylor
        #result = (W[0] * x + (W[1] - cons) * x**2) / cons
        #result = (W[0] * x + W[1] * x**2 + (W[2] - cons) * x**3) / cons
        #result = (W[0] * x + W[1] / fractorial(2) * x**2 + (W[2] - cons) / fractorial(3) * x**3) / cons
        result = (W[0] * x + W[1] / fractorial(2) * x**2 + (W[2] - cons) / fractorial(3) * x**3 + (W[3] - cons) / fractorial(4) * x**4) / cons
        #result = (W[0] * x + W[1] * x**2 + (W[2] - cons) * x**3 + (W[3] - cons) * x**4 + (W[4] - cons) * x**5) / cons
        #result = (W[0] * x + W[1] / fractorial(2) * x**2 + (W[2] - cons) / fractorial(3) * x**3 + (W[3] - cons) / fractorial(4) * x**4 + (W[4] - cons) / fractorial(5) * x**5) / cons
        #result = (W[0] * x + W[1] / fractorial(2) * x**2 + (W[2] - cons) / fractorial(3) * x**3 + (W[3] - cons) / fractorial(4) * x**4 + (W[4] - cons) / fractorial(5) * x**5 + (W[5] - cons) / fractorial(6) * x**6) / cons
        #result = (W[0] * x + W[1] / fractorial(2) * x**2 + (W[2] - cons) / fractorial(3) * x**3 + (W[3] - cons) / fractorial(4) * x**4 + (W[4] - cons) / fractorial(5) * x**5) / cons
        #result = (W[0] * x + W[1] * x**2 + (W[2] - cons) * x**3 + (W[3] - cons) * x**4 + (W[4] - cons) * x**5 + (W[5] - cons) * x**6 + (W[6] - cons) * x**7) / cons
        #result = (W[0] * x + W[1] * x**2 + (W[2] - cons) * x**3 + (W[3] - cons) * x**4 + (W[4] - cons) * x**5 + (W[5] - cons) * x**6 + (W[6] - cons) * x**7 + (W[7] - cons) * x**8) / cons
        #result = (W[0] * x + W[1] * x**2 + (W[2] - cons) * x**3 + (W[3] - cons) * x**4 + (W[4] - cons) * x**5 + (W[5] - cons) * x**6 + (W[6] - cons) * x**7 + (W[7] - cons) * x**8 + (W[8] - cons) * x**9) / cons
        #result = (W[0] * x + W[1] * x**2 + (W[2] - cons) * x**3 + (W[3] - cons) * x**4 + (W[4] - cons) * x**5 + (W[5] - cons) * x**6 + (W[6] - cons) * x**7 + (W[7] - cons) * x**8 + (W[8] - cons) * x**9 + (W[9] - cons) * x**10) / cons
    if num == 10: # GELU
        result = .5 * x * (1 + tf.nn.tanh((2 / np.pi)**0.5 * (x + .044745 * x**3)))
    return result

def activation2(x, num, W):
    if num == 0:
        result = tf.nn.relu(x, 'x')
    if num == 1: # sin
        result = tf.sin(x, 'x')
        #result = (W[0] * x - W[1] * x**3 / fractorial(3) + W[2] * x**5 / fractorial(5) - W[3] * x**7 / fractorial(7) + W[4] * x**9 / fractorial(9))
        #result = W[0] * 1 * tf.sin(x, 'x') + W[1] * tf.sin(2 * x, 'x')
        #result = tf.sin(W[0] * 10 * x, 'x') + tf.sin(W[1] * 20 * x, 'x') + tf.sin(W[2] * 30 * x, 'x')
    if num == 2: # Quadratic Unit
        #result = W[0] * x * 100 + W[1] * x**2 * 100 / fractorial(2)
        #result = W[0] * 100 * x + W[1] / fractorial(2) * 100 * x**2 + W[2] / fractorial(3) * x**3 * 100 + W[3] / fractorial(4) * x**4 * 100
        result = -x
    if num == 3: # Mish
        result = x * tf.nn.tanh(tf.log(1 + tf.exp(x)))
    if num == 4: # leaky_relu
        result = tf.nn.leaky_relu(x, 0.1, 'x')
    if num == 5: # Cubic Unit
        result = W[0] * x * 100 + W[1] * 100 * x**3 / fractorial(3)
        #result = x + x**3
    if num == 6: # Gaussian
        result = tf.exp(-x**2)
    if num == 7: # new
        #result = W[0] * 100 * x + W[1] / fractorial(2) * 100 * x**2 + W[2] / fractorial(3) * x**3 * 1.0 + W[3] / fractorial(4) * x**4 * 1.0
        result = W[0] * x * 100 + W[1] * x**2 * 100 / fractorial(2)
    print(result)
    return result

def discriminator_ResNet(x, W, r, n1, n2):
    eps = 1e-7
    #n1 = 7
    #n2 = 0
    bn = 3
    print('This is ResNet, AF1 number:', n1)
    print('AF2 number:', n2)
    print('learning rate:', r)
    t = W
    print('W:', len(t))
    d = x
    fig = [0 for i in range(5)] # initiate a list of 5 elements
    ## discriminator: 1 ##
    p = t[0]
    print(d.shape)
    print(p)
    y = tf.nn.conv2d(d, p, strides = [1, 2, 2, 1], padding = 'SAME', name = 'y') # 112
    fig[0] = y
    d = activation(y, n1, t[21])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # 56
    print('d:', d)
    da = d
    ## discriminator: 2 ##
    p = t[1]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    fig[1] = y
    d = activation(y, n1, t[22])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 3 ##
    p = t[2]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    fig[2] = y
    d = activation(y + da, n1, t[23])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    da = d
    ## discriminator: 4 ##
    p = t[3]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    fig[3] = y
    d = activation(y, n2, t[24])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 5 ##
    p = t[4]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    fig[4] = y
    d = activation(y + da, n2, t[25])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    da = tf.nn.conv2d(d, t[17], strides = [1, 2, 2, 1], padding = 'SAME', name = 'y') # 28 dimension modification
    ## discriminator: 6 ##
    p = t[5]
    y = tf.nn.conv2d(d, p, strides = [1, 2, 2, 1], padding = 'SAME', name = 'y') # 28
    d = activation(y, n2, t[26])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 7 ##
    p = t[6]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 28
    d = activation(y + da, n2, t[27])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    da = d
    ## discriminator: 8 ##
    p = t[7]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 28
    d = activation(y, n2, t[28])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 9 ##
    p = t[8]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 28
    d = activation(y + da, n2, t[29])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    da = tf.nn.conv2d(d, t[18], strides = [1, 2, 2, 1], padding = 'SAME', name = 'y') # 14 dimension modification
    ## discriminator: 10 ##
    p = t[9]
    y = tf.nn.conv2d(d, p, strides = [1, 2, 2, 1], padding = 'SAME', name = 'y') # 14
    d = activation(y, n2, t[30])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 11 ##
    p = t[10]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    d = activation(y + da, n2, t[31])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    da = d
    ## discriminator: 12 ##
    p = t[11]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    d = activation(y, n2, t[32])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 13 ##
    p = t[12]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    d = activation(y + da, n2, t[33])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    da = tf.nn.conv2d(d, t[19], strides = [1, 2, 2, 1], padding = 'SAME', name = 'y') # 7 dimension modification
    ## discriminator: 14 ##
    p = t[13]
    y = tf.nn.conv2d(d, p, strides = [1, 2, 2, 1], padding = 'SAME', name = 'y') # 7
    d = activation(y, n2, t[34])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 15 ##
    p = t[14]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 7
    d = activation(y + da, n2, t[35])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    da = d
    ## discriminator: 16 ##
    p = t[15]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 7
    d = activation(y, n2, t[36])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 17 ##
    p = t[16]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 7
    d = activation(y + da, n2, t[37])
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    d = tf.nn.avg_pool(d, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'VALID', name = 'd') # 1
    #d = tf.nn.dropout(d, keep_prob)
    ## discriminator: FC1 ##
    p = t[20]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    d0 = tf.nn.softmax(y, name = 'cls0') # classification
    print('d0:', d0)
    return d0, fig

def discriminator_VGG(x, W):
    n1 = 2
    t = W
    print('W:', len(t))
    d = x
    fig = [0 for i in range(5)] # initiate a list of 5 elements
    ## discriminator: 1 ##
    p = t[0]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 224
    fig[0] = y
    d = activation(y, n1, p)
    print('d:', d)
    ## discriminator: 2 ##
    p = t[1]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 224
    fig[1] = y
    d = activation(y, n1, p)
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 112
    ## discriminator: 3 ##
    p = t[2]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 112
    fig[2] = y
    d = activation(y, n1, p)
    print('d:', d)
    ## discriminator: 4 ##
    p = t[3]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 112
    fig[3] = y
    d = activation(y, n1, p)
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 56
    ## discriminator: 5 ##
    p = t[4]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    fig[4] = y
    d = activation(y, n1, p)
    print('d:', d)
    ## discriminator: 6 ##
    p = t[5]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    d = activation(y, n1, p)
    ## discriminator: 7 ##
    p = t[6]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    d = activation(y, n1, p)
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 28
    print('d:', d)
    ## discriminator: 8 ##
    p = t[7]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 28
    d = activation(y, n1, p)
    ## discriminator: 9 ##
    p = t[8]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 28
    d = activation(y, n1, p)
    print('d:', d)
    ## discriminator: 10 ##
    p = t[9]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 28
    d = activation(y, n1, p)
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 14
    ## discriminator: 11 ##
    p = t[10]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    d = activation(y, n1, p)
    print('d:', d)
    ## discriminator: 12 ##
    p = t[11]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    d = activation(y, n1, p)
    ## discriminator: 13 ##
    p = t[12]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    d = activation(y, n1, p)
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 7
    print('d:', d)
    ## discriminator: 14 ##
    p = t[13]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    test_x1 = y
    d = activation(y, n1, p)
    test_y1 = d
    #t1 = tf.nn.dropout(d_a1, keep_prob)
    #t2 = tf.nn.dropout(d_a2, keep_prob)
    d = tf.nn.dropout(d, keep_prob)
    #print('t1:', t1)
    #d = tf.concat((t1, t2), 3, 'concat') # 2个激活函数合并
    ## discriminator: 15 ##
    p = t[14]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    test_x2 = y
    d = activation(y, n1, p)
    test_y2 = d
    print('d:', d)
    ## discriminator: 16 ##
    p = t[15]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    d0 = tf.nn.softmax(y, name = 'cls0') # classification
    print('d0:', d0)
    return d0, fig[0], fig[1], fig[2], fig[3], fig[4], test_x1, test_y1, test_x2, test_y2

def discriminator_VGG_short(x, W):
    #n = 1
    n1 = 3
    n2 = 3
    t = W
    print('W:', len(t))
    d = x
    fig = [0 for i in range(4)] # initiate a list of 5 elements
    ## discriminator: 1 ##
    p = t[0]
    y = tf.nn.conv2d(d, p, strides = [1, 2, 2, 1], padding = 'SAME', name = 'y') # 112
    #print(t[7])
    d = activation(y, n1, t[7])
    m, v = tf.nn.moments(d, axes = [0])
    #d = (d - m) / tf.pow((v + eps), 0.5)
    #d1 = tf.nn.max_pool(d, ksize = [1, 112, 112, 1], strides = [1, 1, 1, 1], padding = 'VALID') # test
    #d1 = tf.reduce_max(d, reduction_indices = [1, 2, 3])
    #d = d / d1
    #print('d1:', d1)
    fig[0] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 56
    print('d:', d)
    ## discriminator: 2 ##
    p = t[1]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    d = activation(y, n1, t[8])
    m, v = tf.nn.moments(d, axes = [0])
    #d = (d - m) / tf.pow((v + eps), 0.5)
    #d1 = tf.nn.max_pool(d, ksize = [1, 56, 56, 1], strides = [1, 1, 1, 1], padding = 'VALID') # test
    #d = d / d1
    fig[1] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 28
    print('d:', d)
    ## discriminator: 3 ##
    p = t[2]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 28
    d = activation(y, n1, t[9])
    m, v = tf.nn.moments(d, axes = [0])
    #d = (d - m) / tf.pow((v + eps), 0.5)
    #d1 = tf.nn.max_pool(d, ksize = [1, 28, 28, 1], strides = [1, 1, 1, 1], padding = 'VALID') # test
    #d = d / d1
    fig[2] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 14
    print('d:', d)
    ## discriminator: 4 ##
    p = t[3]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    d = activation(y, n1, t[10])
    m, v = tf.nn.moments(d, axes = [0])
    #d = (d - m) / tf.pow((v + eps), 0.5)
    #d1 = tf.nn.max_pool(d, ksize = [1, 14, 14, 1], strides = [1, 1, 1, 1], padding = 'VALID') # test
    #d = d / d1
    fig[3] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 7
    print('d:', d)
    ## discriminator: 5 ##
    p = t[4]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    test_x1 = y
    d = activation(y, n2, t[11])
    m, v = tf.nn.moments(d, axes = [0])
    #d = (d - m) / tf.pow((v + eps), 0.5)
    test_y1 = d
    d = tf.nn.dropout(d, keep_prob)
    ## discriminator: 6 ##
    p = t[5]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    test_x2 = y
    d = activation(y, n2, t[12])
    m, v = tf.nn.moments(d, axes = [0])
    #d = (d - m) / tf.pow((v + eps), 0.5)
    test_y2 = d
    #d = tf.nn.dropout(d, keep_prob)
    print('d:', d)
    ## discriminator: 7 ##
    p = t[6]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    d0 = tf.nn.softmax(y, name = 'cls0') # classification
    print('d0:', d0)
    return d0, fig[0], fig[1], fig[2], fig[3],test_x1, test_y1, test_x2, test_y2

def discriminator_VGG_short2(x, W):
    #n = 1
    n1 = 1
    n2 = 1
    t = W
    print('W:', len(t))
    d = x
    fig = [0 for i in range(4)] # initiate a list of 5 elements
    ## discriminator: 1 ##
    p = t[0]
    y = tf.nn.conv2d(d, p, strides = [1, 2, 2, 1], padding = 'SAME', name = 'y') # 112
    #print(t[7])
    d = activation2(y, n1, t[7:12])
    fig[0] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 56
    print('d:', d)
    ## discriminator: 2 ##
    p = t[1]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    d = activation2(y, n1, t[12:17])
    fig[1] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 28
    print('d:', d)
    ## discriminator: 3 ##
    p = t[2]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 28
    d = activation2(y, n1, t[17:22])
    fig[2] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 14
    print('d:', d)
    ## discriminator: 4 ##
    p = t[3]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    d = activation2(y, n1, t[22:27])
    fig[3] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 7
    print('d:', d)
    ## discriminator: 5 ##
    p = t[4]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    test_x1 = y
    d = activation2(y, n2, t[27:32])
    m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    test_y1 = d
    d = tf.nn.dropout(d, keep_prob)
    print('d:', d)
    ## discriminator: 6 ##
    p = t[5]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    test_x2 = y
    d = activation2(y, n2, t[32:37])
    m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    test_y2 = d
    print('d:', d)
    ## discriminator: 7 ##
    p = t[6]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    d0 = tf.nn.softmax(y, name = 'cls0') # classification
    print('d0:', d0)
    return d0, fig[0], fig[1], fig[2], fig[3],test_x1, test_y1, test_x2, test_y2

def discriminator_scale(x, W, opt, r, n1, n2):
    eps = 1e-7
    #n1 = 7
    #n2 = 0
    #opt: enable BN, 0: no, 1 yes
    bn = 3
    print('This is GoogleNet, AF1 number:', n1)
    print('AF2 number:', n2)
    print('learning rate:', r)
    t = W
    print('W:', len(t))
    d = x
    fig = [0 for i in range(8)] # initiate a list of 4 elements
    ## discriminator: 1 ##
    p = t[0] * 1
    y = tf.nn.conv2d(d, p, strides = [1, 2, 2, 1], padding = 'SAME', name = 'y') # 112
    fig[0] = y
    #d = activation(y, n1, t[11][:, 0:1, 0:1, 0:1]) # feature map mode
    #d = activation(y, n1, t[11][:, 0:1, 0:1, :]) # channel mode
    d = activation(y, n1, t[11]) # node mode
    print('t:', t[11][:, 0:1, 0:1, 0:1])
    #p = t[1]
    #y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 112
    #d = activation(y, n1, t[12])
    fig[1] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 56
    m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5) * opt + d * (1 - opt)
    print('d:', d)
    d1 = d
    ## discriminator: 2 ##
    p = t[2] * 1
    #print('d', d)
    #print('p', p)
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    fig[2] = y
    #d = activation(y, n1, t[13][:, 0:1, 0:1, 0:1]) # feature map mode
    #d = activation(y, n1, t[13][:, 0:1, 0:1, :]) # channel mode
    d = activation(y, n1, t[13]) # node mode
    #print('d:', d)
    #p = t[3]
    #y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    #d = activation(y, n1, t[14])
    fig[3] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 28
    m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5) * opt + d * (1 - opt)
    print('d:', d)
    ## discriminator: 3 ##
    p = t[4] * 1
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 28
    fig[4] = y
    #d = activation(y, n1, t[15][:, 0:1, 0:1, 0:1]) # feature map mode
    #d = activation(y, n1, t[15][:, 0:1, 0:1, :]) # channel mode
    d = activation(y, n1, t[15]) # node mode
    #print('d:', d)
    #p = t[5]
    #y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 28
    #d = activation(y, n1, t[16])
    fig[5] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 14
    m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5) * opt + d * (1 - opt)
    print('d:', d)
    ## discriminator: 4 ##
    p = t[6] * 1
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    fig[6] = y
    #d = activation(y, n1, t[17][:, 0:1, 0:1, 0:1]) # feature map mode
    #d = activation(y, n1, t[17][:, 0:1, 0:1, :]) # channel mode
    d = activation(y, n1, t[17]) # node mode
    #print('d:', d)
    #p = t[7]
    #y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    #d = activation(y, n1, t[18])
    fig[7] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 7
    m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5) * opt + d * (1 - opt)
    print('d:', d)
    ## discriminator: 5 ##
    p = t[8] * 1
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    #y = tf.nn.avg_pool(d, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'VALID', name = 'd') # pooling 1
    test_x1 = y
    #d = activation(y, n2, t[19][:, :, :, 0:1]) # feature map mode
    d = activation(y, n2, t[19]) # node mode
    test_y1 = d
    m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5) * opt + d * (1 - opt)
    #d = tf.nn.dropout(d, keep_prob)
    print('d:', d)
    ## discriminator: 6 ##
    p = t[9] * 1
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    test_x2 = y
    #d = activation(y, n2, t[20][:, :, :, 0:1]) # feature map mode
    d = activation(y, n2, t[20]) # node mode
    test_y2 = d
    m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5) * opt + d * (1 - opt)
    #d = tf.nn.dropout(d, keep_prob)
    print('d:', d)
    ## discriminator: 7 ##
    p = t[10] * 1
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    #y = tf.nn.conv2d(d_1, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1 # 之前的操作d_1，导致成功率不高
    d0 = tf.nn.softmax(y, name = 'cls0') # classification
    #d0 = tf.nn.sigmoid(y, name = 'cls0')
    print('d0:', d0)
    return d0, fig, test_x1, test_y1, test_x2, test_y2, y

def discriminator_scale2(x, W): # for Scalability testing
    n1 = 0
    n2 = 0
    t = W
    print('W:', len(t))
    d = x
    fig = [0 for i in range(8)] # initiate a list of 4 elements
    ## discriminator: 1 ##
    p = t[0] * 1
    y = tf.nn.conv2d(d, p, strides = [1, 2, 2, 1], padding = 'SAME', name = 'y') # 112
    fig[0] = y
    d = activation(y, n1, t[11]) # node mode
    #m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    #d = (d - m) / tf.pow((v + eps), 0.5)
    #p = t[1]
    #y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 112
    #d = activation(y, n1, t[12])
    fig[1] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 56
    m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    d1 = d
    ## discriminator: 2 ##
    p = t[2] * 1
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    fig[2] = y
    d = activation(y, n1, t[13]) # node mode
    #m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    #d = (d - m) / tf.pow((v + eps), 0.5)
    #p = t[3]
    #y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    #d = activation(y, n1, t[14])
    fig[3] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 28
    m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 3 ##
    p = t[4] * 1
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 28
    fig[4] = y
    d = activation(y, n1, t[15]) # node mode
    #m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    #d = (d - m) / tf.pow((v + eps), 0.5)
    p = t[5]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 28
    d = activation(y, n1, t[16])
    fig[5] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 14
    m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 4 ##
    p = t[6] * 1
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    fig[6] = y
    d = activation(y, n1, t[17]) # node mode
    #m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    #d = (d - m) / tf.pow((v + eps), 0.5)
    p = t[7]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    d = activation(y, n1, t[18])
    fig[7] = d
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # pooling 7
    m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 5 ##
    p = t[8] * 1
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    #y = tf.nn.avg_pool(d, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'VALID', name = 'd') # pooling 1
    test_x1 = y
    d = activation(y, n2, t[19]) # node mode
    test_y1 = d
    m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    #d = tf.nn.dropout(d, keep_prob)
    print('d:', d)
    ## discriminator: 6 ##
    p = t[9] * 1
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    test_x2 = y
    d = activation(y, n2, t[20]) # node mode
    test_y2 = d
    m, v = tf.nn.moments(d, axes = [3], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    #d = tf.nn.dropout(d, keep_prob)
    print('d:', d)
    ## discriminator: 7 ##
    p = t[10] * 1
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    d0 = tf.nn.softmax(y, name = 'cls0') # classification
    #d0 = tf.nn.sigmoid(y, name = 'cls0')
    print('d0:', d0)
    return d0, fig, test_x1, test_y1, test_x2, test_y2

def discriminator_GoogleNet(x, W, keep_prob, r, n1, n2):
    eps = 1e-7
    #n1 = 7
    #n2 = 0
    bn = 3
    print('This is GoogleNet, AF1 number:', n1)
    print('AF2 number:', n2)
    print('learning rate:', r)
    t = W
    print('W:', len(t))
    d = x
    fig = [0 for i in range(5)] # initiate a list of 5 elements
    ## discriminator: 1 ##
    p = t[0]
    y = tf.nn.conv2d(d, p, strides = [1, 2, 2, 1], padding = 'SAME', name = 'y') # 112
    fig[0] = y
    d = activation(y, n1, t[57]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # 56
    print('d:', d)
    ## discriminator: 2 ##
    p = t[1]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    fig[1] = y
    d = activation(y, n1, t[58]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # 28
    print('d:', d)
    ## discriminator: 3a ##
    p = t[2:8]
    y = inception(d, p, 'y') # 28
    fig[2] = y
    d = activation(y, n1, t[59]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 3b ##
    p = t[8:14]
    y = inception(d, p, 'y') # 28
    fig[3] = y
    d = activation(y, n1, t[60]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # 14
    d_con = d
    print('d:', d)
    ## discriminator: 4a ##
    p = t[14:20]
    y = inception(d, p, 'y') # 14
    fig[4] = y
    d = activation(y, n1, t[61]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d_4a = d
    print('d:', d)
    ## discriminator: 4b ##
    p = t[20:26]
    y = inception(d, p, 'y') # 14
    d = activation(y, n1, t[62]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d_4b = d
    print('d:', d)
    ## discriminator: 4c ##
    p = t[26:32]
    y = inception(d, p, 'y') # 14
    d = activation(y, n1, t[63]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d_4c = d
    print('d:', d)
    ## discriminator: 4d ##
    p = t[32:38]
    y = inception(d, p, 'y') # 14
    d = activation(y, n1, t[64]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d_4d = d
    print('d:', d)
    ## discriminator: 4e ##
    p = t[38:44]
    y = inception(d, p, 'y') # 14
    d = activation(y, n1, t[65]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d_4e = d
    #d = tf.concat((d_con, d_4e), 3, 'd_Con')
    print('d:', d)
    d = tf.nn.max_pool(d, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = 'd') # 7
    print('d:', d)
    ## discriminator: 5a ##
    p = t[44:50]
    y = inception(d, p, 'y') # 7
    d = activation(y, n2, t[66]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 5b ##
    p = t[50:56]
    y = inception(d, p, 'y') # 7
    d = activation(y, n2, t[67]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = tf.nn.avg_pool(d, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'VALID', name = 'd') # 1
    d = tf.nn.dropout(d, keep_prob)
    print('d:', d)
    ## discriminator: FC ##
    p = t[56]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 1
    d0 = tf.nn.softmax(y, name = 'cls0') # classification
    print('d0:', d0)
    return d0, fig

def discriminator_MobileNet(x, W, r, n1, n2):
    t = W
    eps = 1e-7
    #n1 = 7
    #n2 = 0
    bn = 3
    print('This is MobileNet, AF1 number:', n1)
    print('AF2 number:', n2)
    print('learning rate:', r)
    print('W:', len(t))
    d = x
    fig = [0 for i in range(5)] # initiate a list of 5 elements
    ## discriminator: 1 ##
    p = t[0]
    y = tf.nn.conv2d(d, p, strides = [1, 2, 2, 1], padding = 'SAME', name = 'y') # 112
    fig[0] = y
    d = activation(y, n1, t[28]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 2 ##
    p = t[1]
    y = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 112
    fig[1] = y
    d = activation(y, n1, t[29]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 3 ##
    p = t[2]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 112
    fig[2] = y
    d = activation(y, n1, t[30]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 4 ##
    p = t[3]
    y = tf.nn.depthwise_conv2d(d, p, strides = [1, 2, 2, 1], rate = [1, 1], padding = 'SAME') # 56
    fig[3] = y
    d = activation(y, n1, t[31]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 5 ##
    p = t[4]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    fig[4] = y
    d = activation(y, n1, t[32]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 6 ##
    p = t[5]
    y = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 56
    d = activation(y, n1, t[33]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 7 ##
    p = t[6]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    d = activation(y, n1, t[34]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 8 ##
    p = t[7]
    y = tf.nn.depthwise_conv2d(d, p, strides = [1, 2, 2, 1], rate = [1, 1], padding = 'SAME') # 28
    d = activation(y, n1, t[35]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 9 ##
    p = t[8]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 28
    d = activation(y, n2, t[36]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 10 ##
    p = t[9]
    y = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 28
    d = activation(y, n2, t[37]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 11 ##
    p = t[10]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 28
    d = activation(y, n2, t[38]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 12 ##
    p = t[11]
    y = tf.nn.depthwise_conv2d(d, p, strides = [1, 2, 2, 1], rate = [1, 1], padding = 'SAME') # 14
    d = activation(y, n2, t[39]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 13 ##
    p = t[12]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    d = activation(y, n2, t[40]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 14 ##
    p = t[13]
    y = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 14
    d = activation(y, n2, t[41]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 15 ##
    p = t[14]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    d = activation(y, n2, t[42]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 14_2 ##
    p = t[15]
    y = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 14
    d = activation(y, n2, t[43]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 15_2 ##
    p = t[16]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    d = activation(y, n2, t[44]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 14_3 ##
    p = t[17]
    y = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 14
    d = activation(y, n2, t[45]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 15_3 ##
    p = t[18]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    d = activation(y, n2, t[46]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 14_4 ##
    p = t[19]
    y = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 14
    d = activation(y, n2, t[47]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 15_4 ##
    p = t[20]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    d = activation(y, n2, t[48]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 14_5 ##
    p = t[21]
    y = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 14
    d = activation(y, n2, t[49]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 15_5 ##
    p = t[22]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    d = activation(y, n2, t[50]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 16 ##
    p = t[23]
    y = tf.nn.depthwise_conv2d(d, p, strides = [1, 2, 2, 1], rate = [1, 1], padding = 'SAME') # 7
    d = activation(y, n2, t[51]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 17 ##
    p = t[24]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 7
    d = activation(y, n2, t[52]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 18 ##
    p = t[25]
    y = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 7
    d = activation(y, n2, t[53]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    ## discriminator: 19 ##
    p = t[26]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 7
    d = activation(y, n2, t[54]) # node mode
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    print('d:', d)
    d = tf.nn.avg_pool(d, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'VALID', name = 'd') # 1
    ## discriminator: 20 ##
    p = t[27]
    y = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 7
    d = tf.nn.softmax(y, name = 'cls') # classification
    print('d:', d)
    return d, fig

def discriminator_MobileNet2(x, W):
    t = W
    eps = 1e-7
    bn = 3
    n1 = 7
    print('W:', len(t))
    d = x
    fig = [0 for i in range(5)] # initiate a list of 5 elements
    ## discriminator: 1 ##
    p = t[0]
    d = tf.nn.conv2d(d, p, strides = [1, 2, 2, 1], padding = 'SAME', name = 'y') # 112
    fig[0] = d
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[28]) # node mode
    print('d:', d)
    ## discriminator: 2 ##
    p = t[1]
    d = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 112
    fig[1] = d
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[29]) # node mode
    print('d:', d)
    ## discriminator: 3 ##
    p = t[2]
    d = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 112
    fig[2] = d
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[30]) # node mode
    print('d:', d)
    ## discriminator: 4 ##
    p = t[3]
    d = tf.nn.depthwise_conv2d(d, p, strides = [1, 2, 2, 1], rate = [1, 1], padding = 'SAME') # 56
    fig[3] = d
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[31]) # node mode
    print('d:', d)
    ## discriminator: 5 ##
    p = t[4]
    d = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    fig[4] = d
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[32]) # node mode
    print('d:', d)
    ## discriminator: 6 ##
    p = t[5]
    d = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 56
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[33]) # node mode
    print('d:', d)
    ## discriminator: 7 ##
    p = t[6]
    d = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[34]) # node mode
    print('d:', d)
    ## discriminator: 8 ##
    p = t[7]
    d = tf.nn.depthwise_conv2d(d, p, strides = [1, 2, 2, 1], rate = [1, 1], padding = 'SAME') # 28
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[35]) # node mode
    print('d:', d)
    ## discriminator: 9 ##
    p = t[8]
    d = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[36]) # node mode
    print('d:', d)
    ## discriminator: 10 ##
    p = t[9]
    d = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 56
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[37]) # node mode
    print('d:', d)
    ## discriminator: 11 ##
    p = t[10]
    d = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 56
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[38]) # node mode
    print('d:', d)
    ## discriminator: 12 ##
    p = t[11]
    d = tf.nn.depthwise_conv2d(d, p, strides = [1, 2, 2, 1], rate = [1, 1], padding = 'SAME') # 14
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[39]) # node mode
    print('d:', d)
    ## discriminator: 13 ##
    p = t[12]
    d = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[40]) # node mode
    print('d:', d)
    ## discriminator: 14 ##
    p = t[13]
    d = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 14
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[41]) # node mode
    print('d:', d)
    ## discriminator: 15 ##
    p = t[14]
    d = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[42]) # node mode
    print('d:', d)
    ## discriminator: 14_2 ##
    p = t[15]
    d = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 14
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[43]) # node mode
    print('d:', d)
    ## discriminator: 15_2 ##
    p = t[16]
    d = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[44]) # node mode
    print('d:', d)
    ## discriminator: 14_3 ##
    p = t[17]
    d = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 14
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[45]) # node mode
    print('d:', d)
    ## discriminator: 15_3 ##
    p = t[18]
    d = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[46]) # node mode
    print('d:', d)
    ## discriminator: 14_4 ##
    p = t[19]
    d = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 14
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[47]) # node mode
    print('d:', d)
    ## discriminator: 15_4 ##
    p = t[20]
    d = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[48]) # node mode
    print('d:', d)
    ## discriminator: 14_5 ##
    p = t[21]
    d = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 14
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[49]) # node mode
    print('d:', d)
    ## discriminator: 15_5 ##
    p = t[22]
    d = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 14
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[50]) # node mode
    print('d:', d)
    ## discriminator: 16 ##
    p = t[23]
    d = tf.nn.depthwise_conv2d(d, p, strides = [1, 2, 2, 1], rate = [1, 1], padding = 'SAME') # 7
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[51]) # node mode
    print('d:', d)
    ## discriminator: 17 ##
    p = t[24]
    d = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 7
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[52]) # node mode
    print('d:', d)
    ## discriminator: 18 ##
    p = t[25]
    d = tf.nn.depthwise_conv2d(d, p, strides = [1, 1, 1, 1], rate = [1, 1], padding = 'SAME') # 7
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[53]) # node mode
    print('d:', d)
    ## discriminator: 19 ##
    p = t[26]
    d = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'SAME', name = 'y') # 7
    m, v = tf.nn.moments(d, axes = [bn], keep_dims = True)
    d = (d - m) / tf.pow((v + eps), 0.5)
    d = activation(d, n1, t[54]) # node mode
    print('d:', d)
    d = tf.nn.avg_pool(d, ksize = [1, 7, 7, 1], strides = [1, 1, 1, 1], padding = 'VALID', name = 'd') # 1
    ## discriminator: 20 ##
    p = t[27]
    d = tf.nn.conv2d(d, p, strides = [1, 1, 1, 1], padding = 'VALID', name = 'y') # 7
    d = tf.nn.softmax(d, name = 'cls') # classification
    print('d:', d)
    return d, fig


#噪声产生的函数
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size = [m, 1, 1, n])

def rand_array(Lv, Hv, num):
    sample = np.arange(Lv, Hv + 1, 1) #生成Lv到Hv的等差数列(Hv - Lv个数)
    output = np.random.rand(1, num)   #初始化输出数列
    for i in range(num):
        #print(i)
        val = np.random.rand(1) * (Hv - Lv + 1 - i)
        ind = int(np.floor(val))
        #print(val, ind, sample[ind])
        output[0, i] = sample[ind] #赋值
        sample = np.delete(sample, ind) #删除当前序号中的值
    return output
# 函数：rearrange
def array_assign(x, index): # 以index为依据重新分配x的元素
    y = np.zeros(x.shape, dtype = np.float32)
    for i in range(x.shape[0]):
        n = int(index[0, i])
        y[i, :] = x[n, :]
    return y
