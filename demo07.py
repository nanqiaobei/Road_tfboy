###这部分主要针对模型的train and evaluate
##有2种方式，一种是用写好的 API 例如:model.fit(),model.evaluate(),model.predict()
###第2种方式，自己写通过eager ececution and GradientTape ，写自己的训练和评价
import tensorflow as tf
from tensorflow.keras import layers

inputs1=tf.keras.Input(shape=(784,),name='digits')
x=layers.Dense(64,activation='relu',name='dense_1')(inputs1)
x=layers.Dense(64,activation=tf.nn.relu,name='dense_2')(x)
output=layers.Dense(10,activation=tf.nn.softmax,name='pred')(x)

model=tf.keras.Model(inputs=inputs1,outputs=output)
model.summary()
###返回的都是Numpy arrays
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data(path='mnist.npz')##
x_train=x_train.reshape(60000,784).astype('float32')/255
x_test=x_test.reshape(10000,784).astype('float32')/255

y_train=y_train.astype('float32')
y_test=y_test.astype('float32')
###Reserve 1000 sample for validation
x_val=x_train[-10000:]
y_val=y_train[-10000:]
x_train=x_train[:-10000]
y_train=y_train[:-10000]

####制定训练模型的配置，优化器，loss,评价指标
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              ###当有多个loss值，是需要用字典形式，根据最后一层loss值的名称进行使用多个loss functin ,见demo06
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              ####list of metrics to monitor
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()] )

###通过指定batch_size 大小，把数据切分成一个个batch,将切分的batch 喂给网络训练，直到重复迭代完所有的batch
##一个epochs 代码一次完整的训练数据
print("# Fit model on training data")
history=model.fit(x_train,y_train,batch_size=64,epochs=3,validation_data=(x_val,y_val))
###返回的"history" object 拥有 loss 值和评价指标的值在训练中 ##这个存放的是每一轮的评价loss值和accuracy 值
print('\nhitory dict:',history.history)

###Evaluate the model on the test data using 'evaluate
print("\n# Evaluate on test data")

results=model.evaluate(x_test,y_test,batch_size=128)
print('test loss : %3.3f,test acc: %3.3f:'%(results[0],results[1]))

###generate predictions(probabilities--the output of the last layer)
##on new data using predict
print("\n# Genrate predictions for 3 samples")
predictions=model.predict(x_test[:3])
print("predictions shape:",predictions.shape)