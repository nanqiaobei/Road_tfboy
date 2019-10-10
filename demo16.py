import tensorflow as tf
from tensorflow.keras import layers

###VAE的代码实现 变分推断  inputs 经过nn网络（任意）输出 m_i,和c_i，d_i=exp(c_i)xe_i+m_i （e_i 是随机产生的正态分布） 为decoer所作事情,编码出d_i，
##d_i 经过NN网络（任意）可得到输出:
#1.loss1= 输入与输出的重构之差，即输出与输出的2分布的距离（min）,loss2 min-batch(1+c_i-(m_i)^2-exp(c_i))(min)
####m_i 代表了搞事混合模型的均值，c_i 代表了方差
###VAE 先将分布映射到隐空间，研究数据的分布，从隐空间中进行生成

class Sampling(layers.Layer):
    """"
    """
    def call(self,inputs):
        z_mean,z_log_var=inputs
        batch=tf.shape(z_mean)[0]
        dim=tf.shape(z_mean)[1]
        epsilon=tf.keras.backend.random_normal(shape=(batch,dim))
        return z_mean+tf.exp(0.5*z_log_var)*epsilon

class Encoder(layers.Layer):
    def __init__(self,
                 latent_dim=32,
                 intermediate_dim=64,
                 name='encoder',
                 **kwargs):
        super(Encoder,self).__init__(name=name,**kwargs)
        self.dense_proj=layers.Dense(intermediate_dim,activation='relu')
        self.dense_mean=layers.Dense(latent_dim)
        self.dense_log_var=layers.Dense(latent_dim)
        self.sampling=Sampling()

    def call(self,inputs):
        x=self.dense_proj(inputs)
        z_mean=self.dense_log_var(x)
        z_log_var=self.dense_log_var(x)
        z=self.sampling((z_mean,z_log_var))
        return z_mean,z_log_var,z


class Dencoder(layers.Layer):
    """converts,z the encoded digit vector ,back into a readable digit"""

    def __init__(self,original_dim,intermediate_dim=64,name='decoder',**kwargs):
        super(Dencoder,self).__init__(name=name,**kwargs)
        self.dense_proj=layers.Dense(intermediate_dim,activation='relu')
        self.dense_ouput=layers.Dense(original_dim,activation='sigmoid')

    def call(self,inputs):
        x=self.dense_proj(inputs)
        return self.dense_ouput(x)


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end_to_end model for training"""
    def __init__(self,original_dim,intermediate_dim=64,latent_dim=32,name='autoencoder',**kwargs):
        super(VariationalAutoEncoder,self).__init__(name=name)
        self.original_dim=original_dim
        self.encoder=Encoder(latent_dim=latent_dim,intermediate_dim=intermediate_dim)
        self.decoder=Dencoder(original_dim,intermediate_dim=intermediate_dim)

    def call(self,inputs):
        z_mean,z_log_var,z=self.encoder(inputs)
        reconstructed=self.decoder(z)
        ##add KL 散度  正则化loss
        kl_loss=-0.5*tf.reduce_sum(z_log_var-tf.square(z_mean)-tf.exp(z_log_var)+1)
        self.add_loss(kl_loss)
        return reconstructed

# original_dim=784
# vae=VariationalAutoEncoder(original_dim,64,32)
optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
# mse_loss_fn=tf.keras.losses.MeanSquaredError()
#
# loss_metric=tf.keras.metrics.Mean()

(x_train,_),_=tf.keras.datasets.mnist.load_data()
x_train=x_train.reshape(60000,784).astype('float32')/255
# train_dataset=tf.data.Dataset.from_tensor_slices(x_train)
# train_dataset=train_dataset.shuffle(buffer_size=1024).batch(64)
#

vae=VariationalAutoEncoder(784,64,32)
vae.compile(optimizer,loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train,x_train,epochs=3,batch_size=64)