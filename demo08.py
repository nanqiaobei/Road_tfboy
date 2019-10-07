###关于一个模型 实现配置多个不同的评价标准和Loss 方式，简单的配置方法
##也可以说，一个模型的重复使用
import tensorflow as tf
from tensorflow.keras import layers
def get_uncompile_model():
    inputs=tf.keras.Input(shape=(784,),name='digits')
    x=layers.Dense(64,activation=tf.nn.relu,name='dense_1')(inputs)
    x=layers.Dense(64,activation=tf.nn.relu,name='dense_2')(x)
    output=layers.Dense(10,activation=tf.nn.softmax,name='predictions')(x)
    model=tf.keras.Model(inputs=inputs,outputs=output)
    return model

def get_compiled_model():
    model=get_uncompile_model()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=tf.keras.metrics.SparseCategoricalAccuracy())

    return model
###有时候提供给我们的损失函数，和评价指标，优化方式，不是自己想要，需要重新定义，请看demo09