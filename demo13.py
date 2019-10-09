###当我们再写call方法时候，你可以创建一个loss tensor  如果你想要使用再最后，
import tensorflow as tf
from tensorflow.keras import layers

##a layer that crates an activity regularization loss
###注意这里实现的是计算输入值tensor的loss
class ActivityRegularizationLayer(layers.Layer):
    def __init__(self,rate=1e-2):
        super(ActivityRegularizationLayer,self).__init__()
        self.rate=rate

    def call(self,inputs):
        self.add_loss(self.rate*tf.reduce_sum(inputs))
        return inputs
###这些losses（包括那些任何创建的内层）都可以重新被获取，通过layer.losses。这一特性被重新设置再每一次调用
#__call__ 到顶端的层，所以layer.losses 永远包含创建这些损失值再最后的前向传播中
class OuterLayer(layers.Layer):
    def __init__(self):
        super(OuterLayer,self).__init__()
        self.activity_reg=ActivityRegularizationLayer(1e-2)

    def call(self,inputs):
        return self.activity_reg(inputs)

layer=OuterLayer()
assert len(layer.losses)==0  ##没有损失值，因为这个层还没有被调用
_=layer(tf.ones(2,1))  ##这个可以打印出来
print(layer.losses[0])
assert len(layer.losses)==1 ##我们创建了一个loss值，因为调用了

###这些loss 特性，应该也包含正则的loss 为每一个内部层的权重创建
class OuterLayer_two(layers.Layer):
    def __init__(self):
        super(OuterLayer_two,self).__init__()
        self.dense=layers.Dense(32,kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self,inputs):
        return self.dense(inputs)
layer2=OuterLayer_two()
_=layer2(tf.zeros((1,1)))
###注意layer2.losses 是一个列表，有id标识
print(layer2.losses)

### the loss 可以被考虑进来，当写训练循环的时候
##实例化一个优化器
optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
###迭代batch
for x_batch_train,y_batch_train in train_dataset:
    with tf.GradientTape() as tape:
        logits=layer(x_batch_train)
        ##loss value for this minibatch
        loss_value=loss_fn(y_batch_train,logits)
        ##add extra losses created during this forward pass
        loss_value+=sum(model.losses)
    ##通过loss_value 反向传播计算模型中可训练参数的梯度
    grads=tape.gradient(loss_value,model.trainable_weights)
    ##优化器利用梯度下降算法，来优化这些模型中可训练的权重（更新权重）
    optimizer.apply_gradients(zip(grads,model.trainable_weights))