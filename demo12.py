###再许多情况下，你不知道你输入的维度大小，但是你想要创建权重当变量知道时，可以过些时间再实例化这个层
###再keras.API.我们推荐再build(input_shape )中创建层的权重,
import tensorflow as tf
from tensorflow.keras  import layers

class Linear(layers.Layer):
    def __init__(self,units=32):
        super(Linear,self).__init__()
        self.units=units

    def build(self,input_shape):
        self.w=self.add_weight(shape=(input_shape[-1],self.units),
                               initializer='random_normal',
                               trainable=True)
        self.b=self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

    def call(self,inputs):
        return tf.matmul(inputs,self.w)+self.b

##the _call_method of you layer will automatically run build the first time is called
linear1=Linear(32) ###这个实现才进行实例化

x=tf.ones((2,3))
y=linear1(x)
print(y)

###如果你分配一个实列性的层作为另一个层的属性，则最外面的层将会开始追踪内部层的权重
class MLPBlock(layers.Layer):
    def __init__(self):
        super(MLPBlock,self).__init__()
        self.linear_1=Linear(32)
        self.linear_2=Linear(32)
        self.linear_3=Linear(1)

    def call(self,inputs):
        x=self.linear_1(inputs)
        x=tf.nn.relu(x)
        x=self.linear_2(x)
        x=tf.nn.relu(x)
        x=self.linear_3(x)
        return x

mlp=MLPBlock()
y=mlp(tf.ones(shape=(3,64)))
###只会标识出参与训练的参数个数，不训练的参数个数不知道
print("weights:",len(mlp.weights))
print("trainable weight:",len(mlp.trainable_weights))