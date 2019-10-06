import tensorflow as tf
from tensorflow.keras import layers
###有2中方式去实例化一个model
##第一种（我喜欢叫他函数式模型编程，像一步步写数学公式） 是使用 "functional API" 使用input的地方，将会链式的连接每一层，并且通过每一层的call(定义前向传播)
##区分出模型的前向传播，最后，创建了一个从输入到输出的模型

inputs=tf.keras.Input(shape=(3,),name='inputs')

###填写激活函数的参数名字有2种，一种是采用参数的string 名字，一种是指定函数,激活函数，激活函数默认无
##x=tf.keras.layers.Dense(4,activation="relu")(inputs)
###通过 name=""可以为每一层指定名字
x=tf.keras.layers.Dense(4,activation=tf.nn.relu,name='dense_4')(inputs)
outpus=tf.keras.layers.Dense(5,activation=tf.nn.softmax)(x)
model=tf.keras.Model(inputs,outpus,name='model')  ##这个name 为整个模型指定名字
model.summary()
####第2种（这个跟pytorch 很像只不过一个call,另一个forward()）(我喜欢叫他面向对象式)
###定义自己的layer时 通过继承 Model 这个类，必须在定义的layer 中继承 __init__这个方法，和前向传播 call
class Mymodel(tf.keras.Model):
    def __init__(self):
        super(Mymodel,self).__init__()
        self.dense1=tf.keras.layers.Dense(4,activation=tf.nn.relu)
        self.dense2=tf.keras.layers.Dense(5,activation=tf.nn.softmax)

    ##在call 中传入你需要通过这层的参数，可以是一个，也可以是多个
    def call(self,inputs):
        x=self.dense1(inputs)
        return self.dense2(x)
###如果是通过继承Model 定义layer 的方法时，可以通过在 call 方法中增加 training=False 指定其行为
class ONEmodel(tf.keras.Model):
    def __init__(self):
        super(ONEmodel,self).__init__()
        self.dense1=tf.keras.layers.Dense(4,activation=tf.nn.relu)
        self.dense2=tf.keras.layers.Dense(5,activation=tf.nn.softmax)
        self.dropout=tf.keras.layers.Dropout(0.6)
    def call(self,inputs,training=False):
        x=self.dense1(inputs)
        ###这步定义可以的，使得在预测的使用让其不可以使用
        if training:
            x=self.dropout(x,training=training)
        return self.dense2(x)

model=ONEmodel()




