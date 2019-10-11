import tensorflow as tf
from tensorflow.keras import layers

##using callbacks
##keras 中的回调是在训练期间的（在某个时期开始时， 在某个batch结束时，在某个epoch结束时等）在不同时间点，调用对象
##这些对象可以实现下列行为
#1.在训练过程中的不同时间点进行验证（除了内置的按时间段验证）
#2.定期或超过特定精度阈值时对模型进行检查
#3.当训练视乎停滞不前时，更改模型的学习率
#4.当训练停滞不前，对模型顶层进行微调
#5在某个epoch等结束时，发生信息

###回调可以作为列表传递给fit
callbacks=[
    tf.keras.callbacks.EarlyStopping(
        ##stop training when 'val_loss is no longer improving
        monitor='val_loss',
        ##
        min_delta=1e-2,
        patience=2,
        verbose=1
    )
]

###里面存在许多内置的回调:
# tf.keras.callbacks.ModelCheckpoint:定期保存模型
# tf.keras.callbacks.EarlyStopping :当训练不在改善验证指标时，停止训练
# tf.keras.callbacks.TensorBoard:定期编写可在 TensorBoard 中可视化的模型日志
# tf.keras.callbacks.CSVLogger :将损失和指标数据流失传到CSV文件中
##等等
##编写自己的回调
##可以通过扩展基类 keras.callbacks.Callback 来创建自定义回调，回调可以
##通过class 属性访问器关联的模型 self.model
##简单的实列，咋爱训练过程中，保存每批次的损失值列表：
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self,logs):
        self.losses=[]

    def on_batch_end(self,batch,logs):
        self.losses.append(logs.get('loss'))

##在相对较大的训练数据集上，特别重要的是定期保存模型的检查点
##最简单的方式是使用ModelCheckpoint 回调：
callbacks2=[
    tf.keras.callbacks.ModelCheckpoint(
        filepaht='mymodel_{epochs}.h5', #你想要保存模型的位置
        save_best_only=True, #关于保存当前的节点
        monitor='val_loss', ##hte 'val_loss has improved
        verbose=1
    )
]



