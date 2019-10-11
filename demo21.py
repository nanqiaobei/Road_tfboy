##使用学习率时间表
##训练深度学习模型常见的模式就是随着训练的进行逐渐减少学习，通常称为”学习率的衰减“
###学习率可以是静态的，也可以是动态的（根据当前模型的当前行为）
##把时间表给优化器
import tensorflow as tf
from tensorflow.keras import layers

initial_learning_rate=0.1
##时间表
lr_schedule=tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True
)
optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

###内置多种时间表可供选择：ExponentialDecay,PiecewiseConstantDecay,PolynomiaLDecay 和InverseTimeDecay
###使用回调实现动态学习率计划
#由于优化程序无法访问验证指标，因此无法使用这些计划对象来实现动态学习率机会（例如：当损失不在改善时。降低学习率）
##但是回调可以访问所有指标，包括验证指标，因此，可以通过回调来修改优化程序上的当前学习率
##从而实现此模型，它甚至可以作为ReduceLROnPlateau 回调内置的

