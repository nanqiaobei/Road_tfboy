import tensorflow as tf
from tensorflow.keras import layers
##这个主要实现了定义层的方法，以及如何定义可训练与不可训练参数，并介绍了add_weight 这个快速初始化方式
##实现这层的功能，实际就是实现当前层的weight和输入数据的计算，实现transformation 从输入到输出
class Linear(layers.Layer):
    def __init__(self,units=32,input_dim=32):
        super(Linear,self).__init__()
        ###默认w的初始的权重方式，这个和pytorch源码很像
        w_init=tf.random_normal_initializer()
        ###用tf.Variable 包裹起来，加入到变量图中，使之识别是变量，亦可以指定其能否训练
        self.w=tf.Variable(initial_value=w_init(shape=(input_dim,units),dtype='float32'),
                           trainable=True)
        b_init=tf.zeros_initializer()
        self.b=tf.Variable(initial_value=b_init(shape=(units,),dtype='float32'),trainable=True)
    def call(self,inputs):
        return tf.matmul(inputs,self.w)+self.b
x=tf.ones((2,2))
linear1=Linear(4,2)
y=linear1(x)
print(y)

###如果需要 让w 和 b能够自动被追踪，可通过在层上被设置的层属性
assert linear1.weights==[linear1.w,linear1.b]  ##断言这个是否包含有这2个weight,
print(linear1.weights[0].numpy())  ##可通过这种方式取出这个里面w的参数值,

###方法2：
#3如果你想要快速的为这层增加
class Linear_two(layers.Layer):
    def __init__(self,units=32,input_dim=32):
        super(Linear_two,self).__init__()
        ##遇到trainable 时用True 和 False 表示其是训练
        self.w=self.add_weight(shape=(input_dim,units),initializer='random_normal',trainable=True)
        self.b=self.add_weight(shape=(units,),initializer='zeros',trainable=True)
    def call(self,inputs):
        return tf.matmul(inputs,self.w)+self.b

x2=tf.ones((2,2))
linear2=Linear_two(4,2)
y2=linear2(x2)
print(y2)

###除了在层中定义可训练的权重变量，也可以定义不可训练的权重变量，不可训练的权重变量是指：不会考虑进反向传播中，当训练这一层

class ComputeSum(layers.Layer):
    def __init__(self,input_dim):
        super(ComputeSum,self).__init__()
        self.total=tf.Variable(initial_value=tf.zeros((input_dim,)),trainable=False)

    def call(self,inputs):
        ##assign_add 函数是相加的意思  total+=value 这个式子，最后相加的值赋值给total
        self.total.assign_add(tf.reduce_sum(inputs,axis=0))
        return self.total

x3=tf.ones((2,2))
my_sum=ComputeSum(2)
y3=my_sum(x3)
print(y3.numpy()) ##这步的输出和下面的输出是不一样的
y3=my_sum(x3)
print(y3.numpy())
###2步输出是不一样的，第一次是[2,2] 第2次是[4,4] 是因为经过第一次运算之后，self.total 的值为第一次total的值
