import tensorflow as tf
from tensorflow.keras import layers

#####注意tf.keras.input shape 中的batch_size 是省略的
###layers中包含着层的模块
inputs1=tf.keras.Input(shape=(784),name='img')

inputs2=tf.keras.Input(shape=(32,32,3))
###包含数据的维度信息和数据类型信息
print(inputs2.shape)  ###(none,32,32,3)
print(inputs2.shape[1]) ###而且这个shape 也是元组类型，支持元组的操作取值
print(inputs2.dtype)


##output=activation(dot(inout,kernel)+bias)
##跟pytorch 不过pyotch 中使用的 forward 而tf2.0中采用的call()
dense=layers.Dense(10,activation='relu')
x=dense(inputs1)
x=layers.Dense(64,activation='relu')(x)
outputs=layers.Dense(10,activation='softmax')(x)
###在这个点上，我们建立模型，通过指定他的input和output就行
##model=tf.keras.Model(inputs=inputs1,outputs=outputs,name='mnist_model')  报错
model=tf.keras.Model(inputs1,outputs,name='mnist_model')
####通过model.summary()可查看模型的架构
model.summary()
####可通过keras.utils.plot_model(model,path) 画出模型图
tf.keras.utils.plot_model(model,"first_model.png")
####如果选择需要展现每一层的input shape 和output shape 则需要
###另外需要注意的是画图是需要安装pydot 和graphviz 的，直接使用conda install pydot 即可全部安装
##keras.utils.plot_model(model,"first_model.png,show_shape="True") ##设定了一个参数 show_shape="true"
