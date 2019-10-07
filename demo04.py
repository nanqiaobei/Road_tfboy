import tensorflow as tf
from tensorflow.keras import layers
#####这是一段autoencoder 自编码器的代码实现，介绍了常用的卷积的一些参数设定

encoder_input=tf.keras.Input(shape=(28,28,1),name='inputs')
####input_shape=(128,128,3)  #(height,width ,channel) _data_format="channels_last"
###重要参数第一个为filters(过滤器的个数，即生成的特征数)，keras_size(过滤器的尺寸大小，分为宽和高，)（3）=（3，3），（3，128）
###当2个不一样的时候，就需要全写除了，默认padding='valid',默认步长strides=(1,1)
x=layers.Conv2D(16,3,activation=tf.nn.relu)(encoder_input)
x=layers.Conv2D(32,3,activation=tf.nn.relu)(x)
###最大池化第一参数是池化层的尺寸大小(3,3)=3,
x=layers.MaxPooling2D(3)(x)
x=layers.Conv2D(32,3,activation='relu')(x)
x=layers.Conv2D(16,3,activation='relu')(x)
encoder_output=layers.GlobalMaxPooling2D()(x)
encoder=tf.keras.Model(encoder_input,encoder_output,name='encoder')
encoder.summary()
x=layers.Reshape((4,4,1))(encoder_output)
x=layers.Conv2DTranspose(16,3,activation=tf.nn.relu)(x)
x=layers.Conv2DTranspose(32,3,activation=tf.nn.relu)(x)
x=layers.UpSampling2D(3)(x)
x=layers.Conv2DTranspose(16,3,activation=tf.nn.relu)(x)
decoder_out=layers.Conv2DTranspose(1,3,activation='relu')(x)
autoencoder=tf.keras.Model(encoder_input,decoder_out,name='autoencoder')
autoencoder.summary()