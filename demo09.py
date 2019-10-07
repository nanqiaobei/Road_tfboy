import tensorflow as tf
from tensorflow.keras import layers
###传统的loss 不满足自己想要的，就需要自己去定义，有2种放法去定义,
##第一种是创建一个函数 接受输入 y_true 和 y_pred
def get_uncompile_model():
    inputs=tf.keras.Input(shape=(784,),name='digits')
    x=layers.Dense(64,activation=tf.nn.relu,name='dense_1')(inputs)
    x=layers.Dense(64,activation=tf.nn.relu,name='dense_2')(x)
    output=layers.Dense(10,activation=tf.nn.softmax,name='predictions')(x)
    model=tf.keras.Model(inputs=inputs,outputs=output)
    return model

def basic_loss_function(y_true,y_pred):
    return tf.math.reduce_mean(y_true-y_pred)


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

model=get_uncompile_model()
# model.compile(optimizer=tf.keras.optimizers.RMSprop(),
#               ###直接当数据的形式来使用，都不用写出函数式，传输入输出的
#               loss=basic_loss_function)
#
# model.fit(x_train,y_train,batch_size=64,epochs=3)

###第2种方式面向对象式，继承的方法，需要y_true和y_pred之间的参数，可以继承tf.keras,lossees.Loss 这个类
##必须包含下面2个方法：1.__init__ 和 call(self,y_true,y_pred) ,__init_能够在call 方法种调用
class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    '''
    Args:
        pos_weight:Scalar to affect the positive labels of the loss function
        weight:Scalar to affect the entirety of the loss function
        from_logits:Whethe to computer loss form logist or the probability
        reduction:Type of the tf.keras.losses.Reducation to apply to loss
        name:Name of the loss function## 这个就是
    '''
    def __init__(self,pos_weight,weight,from_logits=False,reduction=tf.keras.losses.Reduction.AUTO,name='weight_binary_crossentropy'):
        super(WeightedBinaryCrossEntropy,self).__init__(reduction=reduction,name=name)

        self.pos_weight=pos_weight
        self.weight=weight
        self.from_logits=from_logits

    def call(self,y_true,y_pred):
        if not self.from_logits:
            x_1=y_true*self.pos_weight*tf.math.log(y_pred+1e-6)
            x_2=(1-y_true)*-tf.math.log(1-y_pred+1e-6)

            return tf.add(x_1,x_2)*self.weight
        return tf.nn.weighted_cross_entropy_with_logits(y_true,y_pred,self.pos_weight)*self.weight

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=WeightedBinaryCrossEntropy(0.5,2))

model.fit(x_train,y_train,batch_size=64,epochs=3)
####Custom metrics
###如果你需要的评价指标没有API，可以自己创建一个 通过继承Metric class ，你需要实现下面4个
class CatogricalTruePositives(tf.keras.metrics.Metric):
    '''
    :arg
     __init__:在这里创建评价指标的目前变量，目任初始状态应当为0
     update_state(self,y_true,y_pred,sample_weight=None)使用真是标签与预测标签进行计算，来更新目前的变量
     result(self),使用当前的状态变量，计算最终的结果
     reset_states(self),在每一轮结束后重置为0
    '''
    def __init__(self,name='categorical_true_positives',**kwargs):
        super(CatogricalTruePositives,self).__init__(name=name,**kwargs)
        ###通过add_weight 来创建对象
        self.true_positives=self.add_weight(name='tp',initializer='zeros')

    def update_state(self,y_true,y_pred,sample_weight=None):
        y_pred=tf.reshape(tf.argmax(y_pred,axis=1),shape=(-1,1))
        ###数据类型强制转化函数tf.cast()
        values=tf.cast(y_true,'int32')==tf.cast(y_pred,'int32')
        values=tf.cast(values,'float32')
        if sample_weight is not None:
            sample_weight=tf.cast(sample_weight,'float32')
            values=tf.multiply(values,sample_weight)

        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        ###the state of the metric will be reset at the start of each epoch
        self.true_positives.assign(0.)
