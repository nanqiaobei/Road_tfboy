import tensorflow as tf
from tensorflow.keras import layers

##你可以序列化你自己定义的层为功能模型的一部分，你可以选择地实现 get_config 方法
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

    def get_config(self):
        return {'units':self.units}

##现在可以重新创建这个层从他的config

layer=Linear(64)
config=layer.get_config()
print(config)
new_layer=Linear.from_config(config)


class Linear2(layers.Layer):

    def __init__(self,units=32,**kwargs):
        super(Linear2,self).__init__(**kwargs)
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

    def get_config(self):
        config=super(Linear2,self).get_config()
        config.update({'units':self.units})
        return config
layer2=Linear2(64)
config=layer2.get_config()
##config 会打印出继承的名字
print(config)
new_layer2=Linear2.from_config(config)