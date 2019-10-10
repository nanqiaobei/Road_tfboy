###对待是训练还是测试，batchNormalization和Dropout 是不同对待的

import tensorflow as tf
from tensorflow.keras import layers
class CustomDropout(layers.Layer):
    def __init__(self,rate,**kwargs):
        super(CustomDropout,self).__init__(**kwargs)
        self.rate=rate

    def call(self,inputs,training=None):
        if training:
            return tf.nn.dropout(inputs,rate=self.rate)
        return inputs



###总体来说层的定义就讲完了,,主要用来定义内部计算模块的，我们使用Model 类来定义最外层的模型-（我们需要训练的目标）