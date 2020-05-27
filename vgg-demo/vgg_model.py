'''
import os 
from keras.applications.vgg19 import VGG19 
from keras.preprocessing import image 
from keras.applications.vgg19 import preprocess_input 
from keras.models import Model 
import numpy as np 

base_model = VGG19(weights= 'imagenet' ) 
model = Model(inputs=base_model.input, outputs=base_model.get_layer( 'flatten' ).output) 
model.summary()
'''
import keras
import tensorflow as tf

def model_vgg16():
    model = tf.keras.applications.VGG16(
            weights = 'imagenet', #指定模型初始化的权重检查点
            include_top = False, #指定模型最后是否包含密集连接分类器（Dense）层，VGG16是1000个类别
            #input_shape = (150,150,3) #输入到网络中的张量形状，这个可以不用输入，能够自动检测
            )
    model.summary()
    tf.keras.utils.plot_model(model, 'VGG16_with_shape_info.png', show_shapes=True)

def model_vgg19():
    model = tf.keras.applications.VGG19(
            include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
            pooling=None, classes=1000
            )
    model.summary()
    tf.keras.utils.plot_model(model, 'VGG19_with_shape_info.png', show_shapes=True)
    
if __name__ == '__main__':
    model_vgg19() 
