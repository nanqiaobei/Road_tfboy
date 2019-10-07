####模型有多个输入和多个输出
###采用function API
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
num_tag=12
vocab_size=6000
num_class=4
num_words=10000
###注意这个shape=（None ,）这个逗号一定要加，
inputs1=tf.keras.Input(shape=(None,),name='title') ##这里的命名，方便通过指定命名的方式输入数据
inputs2=tf.keras.Input(shape=(None,),name='body')
inputs3=tf.keras.Input(shape=(num_tag,),name='tag')

title_feature=layers.Embedding(num_words,64)(inputs1)
body_feature=layers.Embedding(num_words,64)(inputs2)
title_feature=layers.LSTM(128)(title_feature)
body_feature=layers.LSTM(32)(body_feature)
x=layers.concatenate([title_feature,body_feature,inputs3])

predicate1=layers.Dense(1,activation=tf.nn.sigmoid,name="predicate1")(x)
predicate2=layers.Dense(num_class,activation=tf.nn.softmax,name="predicate2")(x)
model=tf.keras.Model(inputs=[inputs1,inputs2,inputs3],outputs=[predicate1,predicate2])
#model.summary()

###通过model.comile（）这个函数配置模型的损失函数，优化器
#model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss=['binary_crossentropy','categorical_crossentropy'],loss_weights=[1.,0.2])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss={'predicate1':'binary_crossentropy','predicate2':'categorical_crossentropy'},loss_weights=[1.,0.2])

title_data=np.random.randint(num_words,size=(1280,10))
body_data=np.random.randint(num_words,size=(1280,10))
tags_data=np.random.randint(2,size=(1280,num_tag)).astype('float32')
predicate1_tagets=np.random.random(size=(1280,1))
predicate2_tagets=np.random.randint(2,size=(1280,num_class))

###采用fit()函数进行配置模型的输入，epoch 和batch_size##通过字典的键值形式进行配置
model.fit({'title':title_data,'body':body_data,'tag':tags_data},{'predicate1':predicate1_tagets,'predicate2':predicate2_tagets},epochs=2,batch_size=32)
