import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
###对于暑假的填充和截断 采用 tf.keras.preprocessing.sequence.pad_sequences
raw_inputs = [
  [83, 91, 1, 645, 1253, 927],
  [73, 8, 3215, 55, 927],
  [711, 632, 71]
]

###采用0进行填充，填充到数据一样的长度大小，样本必须拥有一样的大小，其中填充的部分应该被忽略，采用masking
##默认采用0进行填充，这个可以通过 value 参数进行设置
##你可以是使用‘pre’ padding (在文本的开始)，或者采用‘post’在文本结尾
##推荐在文本结尾

padding_input=tf.keras.preprocessing.sequence.pad_sequences(raw_inputs,padding='post')
print(padding_input)
####在keras 模型中有2中输入可以masking 1.keras.layers.Masking ，
##2.使用配置keras.layers.Embedding 层中mask_zero=True
##3在调用这层时，支持这个mask 这个参数
##方式二的使用
embedding=layers.Embedding(input_dim=5000,output_dim=16,mask_zero=True)
print(embedding)
mask_output=embedding(padding_input)
print(mask_output._keras_mask)

###方式一的使用
mask_layer=layers.Masking()
##Simulate the embedding lookup by enpand the 2d to 3d with embedding dimension of10
unmasked_embedding=tf.cast(tf.tile(tf.expand_dims(padding_input,axis=-1),[1,1,10]),
                           tf.float32)

masked_embedding=mask_layer(unmasked_embedding)
print(masked_embedding._keras_mask)
###其中每个单独的False条目指示在处理过程中忽略相应的时间步长。
###当使用Function API 或者Sequential API 时候，又Embedding or Masking layer 生成
##mask 可以通过任何层和任何网络,keras 将会自动获取输入相对应的mask，并将
##其传递给知道如何使用该遮罩的任何层
##但是注意，如果是通过 call 子类模型或图层的方法中，mask 不会自动传播
##因此需要手动将mask 参数传递给需要一个图层的任何层
###将遮罩张量直接传递到图层
##可以可以处理mask 的层（例如LSTM）在其__call__方法里面有mask 这个参数
##于此同时，人可以产生mask的层（Embedding）有compute_mask(input,previous_mask)
##方法被调用
###注意embedding 是embedding      mask 是mask
class Mylayer(layers.Layer):
    def __init__(self,**kwargs):
        super(Mylayer,self).__init__(**kwargs)
        self.embedding=layers.Embedding(input_dim=5000,output_dim=16,mask_zero=True)
        self.lstm=layers.LSTM(32)

    def call(self,inputs):
        x=self.embedding(inputs)
        ##应该需要准备mask tensor
        ##仅仅是需要boolean tensor
        ## 需要正确的 shape [batch_size,timesteps]
        mask=self.embedding.compute_mask(inputs)
        output=self.lstm(x,mask=mask) ##这一层将要忽略mask values
        return output

layer=Mylayer()
x=np.random.random((32,10))*100
x=x.astype('float32')
y=layer(x)
print(y)
##自定义图层的mask

##有时候，你可以编写生成的mask 图层,
##连接时间的维度层，大多数都要修改当前掩码，以便下游层能够适当考虑掩码的时间步长
##因此你的图层应当实现 layer.compute_mask()方法，该方法根据当前输入和遮赵生成一个新的遮罩
class TemporalSplit(tf.keras.layers.Layer):
    def call(self,inputs):
        return tf.split(inputs,2,axis=1)

    def compute_mask(self,inputs,mask=None):
        if mask is None:
            return None
        return  tf.split(mask,2,axis=1)
first_half,second_half=TemporalSplit()(masked_embedding)
print(first_half._keras_mask)
print(second_half._keras_mask)


class CustomEmbedding(tf.keras.layers.Layer):

    def __init__(self, input_dim, output_dim, mask_zero=False, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer='random_normal',
            dtype='float32')

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings, inputs)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs, 0)


layer2 = CustomEmbedding(10, 32, mask_zero=True)
x1 = np.random.random((3, 10)) * 9
x1= x1.astype('int32')

y = layer2(x1)
mask = layer2.compute_mask(x)

print(mask)