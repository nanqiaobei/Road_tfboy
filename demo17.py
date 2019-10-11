import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
#抽取和重用计算图中的层节点

from  tensorflow.keras.applications import VGG19
vgg19=VGG19()
###
##通过查询图的数据结构获得
features_list=[layer.output for layer in vgg19.layers]

##我们可以使用这些features 来创建一个新的特征抽取模型
feat_extraction_model=tf.keras.Model(inputs=vgg19.input,output=features_list)

img=np.random.random((1,224,24,3)).astype('float32')
extracted_features=feat_extraction_model(img)
