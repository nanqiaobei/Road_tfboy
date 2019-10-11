import tensorflow as tf
from tensorflow.keras import layers
###关于模型的多输入和毒品输出

image_input=tf.keras.Input(shape=(32,32,3),name='img_input')
timeseries_input=tf.keras.Input(shape=(None,10),name='ts_input')

x1=layers.Conv2D(3,3)(image_input)
x1=layers.GlobalMaxPooling2D()(x1)

x2=layers.Conv1D(3,3)(timeseries_input)
x2=layers.GlobalMaxPooling1D()(x2)

x=layers.concatenate([x1,x2])

score_output=layers.Dense(1,name='score_out')(x)
class_out=layers.Dense(5,activation='softmax',name='class_out')(x)

model=tf.keras.Model(inputs=[image_input,timeseries_input],outputs=[score_output,class_out])

model.summary()


###推荐下面这种形式去配置多输入和多输出模型（在多任务学习中，经常看到）这种字典形式的设置更好
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(1e-3),
    loss={'score_out': tf.keras.losses.MeanSquaredError(),
          'class_out': tf.keras.losses.CategoricalCrossentropy()},
    metrics={'score_out': [tf.keras.metrics.MeanAbsolutePercentageError(),
                              tf.keras.metrics.MeanAbsoluteError()],
             'class_out': [tf.keras.metrics.CategoricalAccuracy()]},
    loss_weights={'score_out': 2., 'class_out': 1.})