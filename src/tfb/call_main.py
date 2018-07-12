from style_transfer import StyleTransfer
from time import strftime
import numpy as np
import vgg19
import os
import collections
import tensorflow as tf
import utils
from time import strftime
import ntpath
from os import listdir
from os.path import isfile, join

"""add one dim for batch"""
# VGG19 requires input dimension to be (batch, height, width, channel)
def add_one_dim(image):
    shape = (1,) + image.shape
    return np.reshape(image, shape)

def transfer_runner(content_image, style_image):
    model_path='tfb/pre_trained_model'    #The directory where the pre-trained model was saved
    output='results/'+strftime("%Y-%m-%d-%H:%M")+'.jpg'  #File path of output image
    loss_ratio=1e-3                   #Weight of content-loss relative to style-loss
    content_layers=['conv4_2']        #VGG19 layers used for content loss
    style_layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'] #VGG19 layers used for style loss
    content_layer_weights=[1.0]       #Content loss for each content is multiplied by corresponding weight
    style_layer_weights=[.2,.2,.2,.2,.2] #Style loss for each content is multiplied by corresponding weight
    initial_type='content'            #choices = ['random','content','style'], The initial image for optimization (notation in the paper : x)
    max_size=101#512                      #The maximum width or height of input images
    content_loss_norm_type=3          #choices=[1,2,3],  Different types of normalization for content loss
    num_iter=400                     #The number of iterations to run

    try:
        assert len(content_layers) == len(content_layer_weights)
    except:
        raise ('content layer info and weight info must be matched')
    try:
        assert len(style_layers) == len(style_layer_weights)
    except:
        raise('style layer info and weight info must be matched')

    try:
        assert max_size > 100
    except:
        raise ('Too small size')


    model_file_path = model_path + '/' + vgg19.MODEL_FILE_NAME
    assert os.path.exists(model_file_path)
    try:
        assert os.path.exists(model_file_path)
    except:
        raise Exception('There is no %s'%model_file_path)

    try:
        size_in_KB = os.path.getsize(model_file_path)
        assert abs(size_in_KB - 534904783) < 10
    except:
        print('check file size of \'imagenet-vgg-verydeep-19.mat\'')
        print('there are some files with the same name')
        print('pre_trained_model used here can be downloaded from bellow')
        print('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat')
        raise()

    # initiate VGG19 model
    model_file_path = model_path + '/' + vgg19.MODEL_FILE_NAME
    vgg_net = vgg19.VGG19(model_file_path)

    # initial guess for output
    if initial_type == 'content':
        init_image = content_image
    elif initial_type == 'style':
        init_image = style_image
    elif initial_type == 'random':
        init_image = np.random.normal(size=content_image.shape, scale=np.std(content_image))

    # check input images for style-transfer
    # utils.plot_images(content_image,style_image, init_image)

    # create a map for content layers info
    CONTENT_LAYERS = {}
    for layer, weight in zip(content_layers,content_layer_weights):
        CONTENT_LAYERS[layer] = weight

    # create a map for style layers info
    STYLE_LAYERS = {}
    for layer, weight in zip(style_layers, style_layer_weights):
        STYLE_LAYERS[layer] = weight
    with tf.Graph().as_default():
        # open session
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # build the graph
        st = StyleTransfer(session = sess,
                  content_layer_ids = CONTENT_LAYERS,
                  style_layer_ids = STYLE_LAYERS,
                  init_image = add_one_dim(init_image),
                  content_image = add_one_dim(content_image),
                  style_image = add_one_dim(style_image),
                  net = vgg_net,
                  num_iter = num_iter,
                  loss_ratio = loss_ratio,
                  content_loss_norm_type = content_loss_norm_type,
                  )
        # launch the graph in a session
        result_image = st.update()
        # close session
        sess.close()
    # remove batch dimension
    shape = result_image.shape
    result_image = np.reshape(result_image,shape[1:])
    # save result
    #utils.save_image(result_image,get_output_filepath(content, style))
    return result_image
