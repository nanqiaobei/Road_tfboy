###解决如何自定义层
##自定义层必须继承tf.keras.layers.Layer，
##必须实现以下方法
##1.__init__
##2.build 这个是为了创建层的权重，添加参数方式为 add_weight 方法
##3.call  定义前向传播
##4.可供选择实现的2方法:get_config 和from_config
import tensorflow as tf
from tensorflow.keras import layers
class MyLayer(tf.keras.layers.Layer):
    def __init__(self,output_dim,**kwargs):
        super(MyLayer,self).__init__(**kwargs)

    def build(self,input_shape):
        ##创建可训练的权重变量为这一层
        self.kernel=self.add_weight(name='keral',shape=(input_shape[1],self.output_dim),
                                    initializer='uniform',
                                    trainable=True)

    def call(self,inputs):
        return tf.matmul(inputs,self.kernel)

    def get_config(self):
        base_config=super(MyLayer,self).get_config()
        base_config['output_dim']=self.ouput_dim
        return base_config

    @classmethod
    def from_config(cls,config):
        return cls(**config)

model = tf.keras.Sequential([
    MyLayer(10),
    layers.Activation('softmax')])

