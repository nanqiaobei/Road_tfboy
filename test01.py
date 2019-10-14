import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def get_angles(pos,i,d_model):
    angle_rates=1/np.power(10000,(2*(i//2))/np.float32(d_model))
    return pos*angle_rates
def positional_encding(position,d_model):
    angle_rads=get_angles(np.arange(position)[:,np.newaxis],
                          np.arange(d_model)[np.newaxis,:],
                          d_model)
    ##apply sin to even indices in the array :2i
    angle_rads[:,0::2]=np.sin(angle_rads[:,0::2])
    angle_rads[:,1::2]=np.cos(angle_rads[:,1::2])
    pos_encoding=angle_rads[np.newaxis,...]
    return tf.cast(pos_encoding,dtype=tf.float32)
def point_wise_feed_forward_network(d_model,dff):
    ##1 (batch_size,seq_len,dff) ->(batch_size,seq_len,d_model)
    model=tf.keras.Sequential([tf.keras.layers.Dense(dff,activation='relu'),tf.keras.layers.Dense(d_model)])
def scaled_dit_product_attention(q,k,v,mask):
    ###Calculate the attention weight
    ##seq_lenk=seq_len_v
    ##it must be broadcastable for addition
    ##q:query shape (...,seqlen_q,deoth)
    ##k key shape=(..,seqlen_k,depth)
    ### v vlaue shape===(...seq_len_v,depth_v)
    ##mask Float tensor with shape broadcastable
    ### return output ,attention_weights
    matmul_qk=tf.matmul(q,k,transpose_b=True)  ###(..,seq_len_q,seq_leng_k)
    ##scale matmul_qk
    dk=tf.cast(tf.shape(k)[-1],tf.float32)
    scaled_attention_logits=matmul_qk/tf.math.sqrt(k)
    if mask is not None:
        scaled_attention_logits+=(mask*1e-9)

    ##sofrmax
    attention_weight=tf.nn.softmax(scaled_attention_logits,axis=-1)
    output=tf.matmul(attention_weight,v) ###(..,seq_len_q,depth_v)

    return output,attention_weight
class MultiHeadAttention(layers.Layer):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()
        self.num_heads=num_heads
        self.d_model=d_model

        assert d_model%self.num_heads==0

        self.depth=d_model//self.num_heads

        self.wq=tf.keras.layers.Dense(d_model)
        self.wk=tf.keras.layers.Dense(d_model)
        self.wv=tf.keras.layers.Dense(d_model)

        self.dense=tf.keras.layers.Dense(d_model)

    def split_heads(self,x,batch_size):
        ###split the last dimension into (num_heads,depth)
        ##transpose the result such that the shape is (batch_size,num_heads,seq_len,depth)
        x=tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))
        return tf.transpose(x,per=[0,2,1,3])

    def call(self,v,k,q,mask):
        batch_size=tf.shape(q)[0]

        q=self.wq(q)  ##(batch_size,seq_len,d_model)
        k=self.wk(k)  ##(batch_size,seq_len,d_model)
        v=self.wv(v)  ###(batch_sizemseq_len,d_model)

        ###scaled_attention.shape=(batch_size,num_heads,seq_len_q,depth)
        scaled_attention=scaled_dit_product_attention(q,k,v,mask)
        concat_attention=tf.reshape(scaled_attention,(batch_size,-1,self.d_model))

class Encoder(layers.Layer):
    def __init__(self,d_model,num_heads,diff,rate=0.1):
        super(Encoder).__init__()
        self.mha=MultiHeadAttention(d_model,num_heads)

        self.layernorm1=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm1=tf.keras.layers.LayerNormalization(epsilon=1e-6)


        self.droput1=tf.keras.layers.Dropout(rate)
        self.droput2=tf.keras.layers.Dropout(rate)
    def call(self,x,training,mask):
        attn_output,_=self.mha(x,x,x,mask) ##(batch_siz,input_seq_len,d_model)
        attn_output=self.droput1(attn_output,training=training)
        out1=self.layernorm1(x+attn_output)#(batch_size,input_seq_len,d_model)
        ###下面使用了位置编码

class Net(tf.keras.Model):
    def __init__(self,vocab_size,embedding_size,):
        super(Net,self).__init__()
        self.embedding=layers.Embedding()
