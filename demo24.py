###Define the keras model to add callback
##在keras 中，Callback 是python 一个类，可以提供特定的功能，并站在训练的各个
##阶段（包括批处理和epochs的开始和结束）测试和预测中调用一组方法，回调在对于训练期间
##了解模型的内部状态和统计信息很有用，
import tensorflow as tf
from tensorflow.keras import layers
import datetime
import numpy as np

def get_model():
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1,activation='linear',input_dim=784))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.1),loss='mean_squared_error',metrics=['mae'])
    return model

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()


x_train=x_train.reshape(60000,784).astype('float32')/255
x_test=x_test.reshape(10000,784).astype('float32')/255
###定义一个简单的自定义回归以跟踪每批数据的开始和结束，在这些调用期间，它将打印当前
##批次的索引

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self,batch,logs=None):
        ###这里的batch 是关于batch 的索引
        print('Training:batch{}begins at{}'.format(batch,datetime.datetime.now().time()))

    def on_train_batch_end(self,batch,logs=None):
        print("Training:batch{}ends at {}".format(batch,datetime.datetime.now().time()))

    def on_test_batch_begin(self, batch, logs=None):
        print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    def on_test_batch_end(self, batch, logs=None):
        print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

model = get_model()
_ = model.fit(x_train, y_train,
          batch_size=64,
          epochs=1,
          steps_per_epoch=5,
          verbose=0,
          callbacks=[MyCustomCallback()])

###对于training，testing and predicting 提供以下方法进行重写
##on_(train|test|predict)_begin(self,logs=None)
##Called at the beginning of fit /evaluate/predict
#on_(train|test|predict)_end(self,logs=None)
##Called at the end of fit/evaluate/predict
##on_(train|test|predict)_batch_begin(self,batch,logs=None)
###这些方法中.logs 是一个字典，包含batch的索引和batch 的大小
###呈现出当前batch的索引和batch的大小的
##在epoch开始的时候:
#on_epoch_begin(self,epoch,logs=None)
##on_epoch_end(self,epoch,logs=None)
##其中logs 也是包含loss的额值，
###logs 字典中包含很多 ，可以自己打印出来看看

class LossAndErrorPrintCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self,batch,logs=None):
        print("for batch {} loss is {:7.2f}.".format(batch,logs['loss']))

    def on_epoch_end(self,epoch,logs=None):
        print("the average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}".format(epoch,logs['loss'],logs['mae']))
model2=get_model()
_=model.fit(x_train,y_train,batch_size=64,steps_per_epoch=5,epochs=3,verbose=0,callbacks=[LossAndErrorPrintCallback()])
print("model 2 执行结束")
###
class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

      Arguments:
          patience: Number of epochs to wait after min has been hit. After this
          number of no improvement, training stops.
      """
    def __init__(self,patience=0):
        super(EarlyStoppingAtMinLoss,self).__init__()
        self.patience=patience
        ##当出现最小的loss值时候，保存这时候的模型里面的权重值
        self.best_weights=None

    def on_train_begin(self,logs=None):
        ##当最小loss值不在最小是，需要等待的epoch
        self.wait=0
        #训练停止时候的epoch
        self.stopped_epoch=0
        ##初始化best 为infinity
        self.best=np.Inf

    def on_epoch_end(self,epoch,logs=None):
        current=logs.get('loss')
        if np.less(current,self.best):
            self.best=current
            self.wait=0
            ##记录最下loss时的模型权重
            self.best_weights=self.model.get_weights()
        else:
            self.wait+=1
            if self.wait>=self.patience:
                self.stopped_epoch=epoch
                self.model.stop_training=True
                print("Restoring model weights from the end of the best epoch")
                ##模型采用最好的权重
                self.model.set_weights(self.best_weights)
    def on_train_end(self,logs=None):
        if self.stopped_epoch>0:
            print("Epoch %05d : early stoping"%(self.stopped_epoch+1))

model3=get_model()
_=model3.fit(x_train,y_train,batch_size=64,steps_per_epoch=5,epochs=30,verbose=0,callbacks=[LossAndErrorPrintCallback(),EarlyStoppingAtMinLoss()]

 ###学习率的安排，设置2个学习率，使用
class LearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

     Arguments:
         schedule: a function that takes an epoch index
             (integer, indexed from 0) and current learning rate
             as inputs and returns a new learning rate as output (float).
     """
    def __init__(self,schedule):
        super(LearningRateScheduler,self).__init__()
        self.schedule=schedule
    def on_epoch_begin(self,epoch,logs=None):
        if not hasattr(self.model.optimizer,'lr'):
            raise ValueError('optimizer must have a "lr" attributer')
        ##从模型的优化器中获得当前的学习率
        lr=float(tf.keras.backend.get_value(self.model.optimizer.lr)))
        #
        scheduled_lr=self.schedule(epoch,lr)
        ##在新的epoch 开始前，设置好，新的学习率的值返回优化器中
        tf.keras.backend.set_value(self.model.optimizer.lr,scheduled_lr)


    ###(epoch to start,learning rate)tuples
LR_SCHEDULE = [(3, 0.05), (6, 0.01), (9, 0.005), (12, 0.001)]


def lr_shcedule(epoch,lr):
        if epoch<LR_SCHEDULE[0][0] or epoch>LR_SCHEDULE[-1][0]:
            return lr
        for i in range(len(LR_SCHEDULE)):
            if epoch==LR_SCHEDULE[i][0]:
                if epoch==LR_SCHEDULE[i][0]:
                    return LR_SCHEDULE[i][1]
        return lr

model4=get_model()
model = get_model()
_ = model.fit(x_train, y_train,
          batch_size=64,
          steps_per_epoch=5,
          epochs=15,
          verbose=0,
          callbacks=[LossAndErrorPrintingCallback(), LearningRateScheduler(lr_schedule)])