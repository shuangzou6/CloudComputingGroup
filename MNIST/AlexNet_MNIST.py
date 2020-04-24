import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data
#定义网络超参数
learning_rate = 0.0001
train_epochs = 130
batch_size = 256
display = 10
t = 50      # 每轮随机抽的次数
# 网络参数
n_input = 784
n_classes = 10
dropout = 0.75     # dropout正则,保留神经元节点的概率
# 定义占位符
images = tf.placeholder(tf.float32,[None,n_input])
labels = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)

# 随机抽取训练数据
# 随机选取mini_batch
def get_random_batchdata(n_samples, batchsize):
    start_index = np.random.randint(0, n_samples - batchsize)
    return (start_index, start_index + batchsize)

# 权重,偏置初始化
# 在卷积层和全连接层我使用不同的初始化操作
# 卷积层使用截断正太分布初始化,shape为列表对象

def conv_weight_init(shape,stddev):
    weight = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=shape, stddev=stddev))
    return weight

# 全连接层使用xavier初始化
def xavier_init(layer1, layer2, constant = 1):
    Min = -constant * np.sqrt(6.0 / (layer1 + layer2))
    Max = constant * np.sqrt(6.0 / (layer1 + layer2))
    weight = tf.random_uniform((layer1, layer2), minval = Min, maxval = Max, dtype = tf.float32)
    return tf.Variable(weight)

# 偏置
def biases_init(shape):
    biases = tf.Variable(tf.random_normal(shape, dtype=tf.float32))
    return biases

def conv2d(image, weight, stride=1):
    return tf.nn.conv2d(image, weight, strides=[1,stride,stride,1],padding='SAME')

def max_pool(tensor, k = 2, stride=2):
    return tf.nn.max_pool(tensor, ksize=[1,k,k,1], strides=[1,stride,stride,1],padding='SAME')

def lrnorm(tensor):
    return tf.nn.lrn(tensor,depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)

# 初始化权重
# wc = weight convolution
wc1 = conv_weight_init([5, 5, 1, 12], 0.05)
wc2 = conv_weight_init([5, 5, 12, 32], 0.05)
wc3 = conv_weight_init([3, 3, 32, 48], 0.05)
wc4 = conv_weight_init([3, 3, 48, 48], 0.05)
wc5 = conv_weight_init([3, 3, 48, 32], 0.05)
# wf : weight fullconnection
wf1 = xavier_init(4*4*32, 512)
wf2 = xavier_init(512, 512)
wf3 = xavier_init(512, 10)

# 初始化偏置
bc1 = biases_init([12])
bc2 = biases_init([32])
bc3 = biases_init([48])
bc4 = biases_init([48])
bc5 = biases_init([32])
# full connection
bf1 = biases_init([512])
bf2 = biases_init([512])
bf3 = biases_init([10])
# 转换image的shape
imgs = tf.reshape(images,[-1, 28, 28, 1])
# 卷积１
c1     = conv2d(imgs, wc1)    # 未激活
conv1 = tf.nn.relu(c1 + bc1)
lrn1      = lrnorm(conv1)
pool1     = max_pool(lrn1)
# 卷积２
conv2 = tf.nn.relu(conv2d(pool1, wc2) + bc2)
lrn2  = lrnorm(conv2)
pool2 = max_pool(lrn2)
# 卷积３－－５，不在进行ＬＲＮ，会严重影响网络前馈，反馈的速度（效果还不明显）
conv3 = tf.nn.relu(conv2d(pool2, wc3) + bc3)
#pool3 = max_pool(conv3)
# 卷积４
conv4 = tf.nn.relu(conv2d(conv3, wc4) + bc4)
#pool4 = max_pool(conv4)
# 卷积５
conv5 = tf.nn.relu(conv2d(conv4, wc5) + bc5)
pool5 = max_pool(conv5)

# 转换pool5的shape
reshape_p5 = tf.reshape(pool5, [-1, 4*4*32])
fc1 = tf.nn.relu(tf.matmul(reshape_p5, wf1) + bf1)
# 正则化
drop_fc1 = tf.nn.dropout(fc1, keep_prob)
# full connect 2
fc2 = tf.nn.relu(tf.matmul(drop_fc1, wf2) + bf2)
drop_fc2 = tf.nn.dropout(fc2, keep_prob)
# full connect 3 (输出层)未激活
output = tf.matmul(drop_fc2, wf3) + bf3


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(labels, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


mnist = input_data.read_data_sets('MNIST/mnist',one_hot=True)
n_examples = int(mnist.train.num_examples)

# 变量初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # 变量初始化

Cost = []
Accu = []
for i in range(train_epochs):
    for j in range(t):
        start_idx, end_idx = get_random_batchdata(n_examples, batch_size)
        batch_images = mnist.train.images[start_idx: end_idx]
        batch_labels = mnist.train.labels[start_idx: end_idx]
        # 更新weights,biases
        sess.run(optimizer, feed_dict={images:batch_images, labels:batch_labels,keep_prob:0.65})
        c , accu = sess.run([cost,accuracy],feed_dict={images:batch_images,labels:batch_labels,keep_prob:1.0})
        Cost.append(c)
        Accu.append(accu)
    if i % display ==0:
        print( 'epoch : %d,cost:%.5f,accu:%.5f'%(i+10,c,accu))
        #result = sess.run(merged,feed_dict={imgaes:xxx,labels:yyy,keep_prob:1.0})
        # merged也需要ｒｕｎ一下，ｉ为ｘ轴（对应所以可视化对象的ｘ轴）
print( 'Training Finish !')

fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12,7))
ax1.plot(Cost,label='Train loss',c='red')
ax1.set_title('Train loss')
ax1.grid(True)
ax1.legend(loc=0)
ax2.set_title('Accuacy')
ax2.plot(Accu,label='Accuracy',c='green')
ax2.grid(True)
plt.legend(loc=5)
plt.show()