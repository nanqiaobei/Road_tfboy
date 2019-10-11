###自定义训练过程循环
import tensorflow as tf
from tensorflow.keras import layers
image_input=tf.keras.Input(shape=(None,784),name='inputs1')
x=layers.Dense(64,activation='relu')(image_input)
x=layers.Dense(64,activation='relu')(x)
output=layers.Dense(10,activation='softmax',name='predict')
model=tf.keras.Model(inputs=image_input,outputs=x)
model.summary()
optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-2)
loss_fn=tf.keras.losses.SparseCategoricalCrossentropy()