##RNN 代码
import tensorflow as tf
from tensorflow.keras import layers

class CustomRNN(layers.Layer):
    def __init__(self,units):
        super(CustomRNN,self).__init__()
        self.units=units
        self.dense1=layers.Dense(units=units,acitivation='tanh')
        self.dense2=layers.Dense(units=units,activation='tanh')

    def call(self,inputs):
        outputs=[]
        state=tf.zeros((inputs.shape[0],self.units))
        for t in range(inputs.shape[1]):
            x=inputs[:,t,:]
            h=self.dense1(x)
            y=h+self.dense2(state)
            state=y
            outputs.append(y)
        features=tf.stack(outputs,axis=1)
