import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
print(tf.version)
print(tf.keras.__version__)
###简单构建模型是层的堆叠：tf.keras.Sequential
model=tf.keras.Sequential()
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metric=[tf.keras.metrics.categorical_accuracy])

train_x=np.random.random((1000,72))
train_y=np.random.random((1000,10))
val_x=np.random.random((200,72))

val_y=np.random.random((200,10))
####竟然都不用打印出来，舒服啊  方式一：
#model.fit(train_x,train_y,epochs=10,batch_size=100,validation_data=(val_x,val_y))

###方式二：
dataset=tf.data.Dataset.from_tensor_slices((train_x,train_y))
dataset=dataset.batch(32)
dataset=dataset.repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_dataset = val_dataset.batch(32)
val_dataset = val_dataset.repeat()
##方式2
model.fit(dataset, epochs=10, steps_per_epoch=30,
#           validation_data=val_dataset, validation_steps=3)
