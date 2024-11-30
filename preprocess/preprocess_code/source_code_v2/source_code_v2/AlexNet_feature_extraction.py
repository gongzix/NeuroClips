# -*- coding: utf-8 -*-
# set up Python2.7 environment: numpy for numerical routines, and matplotlib for plotting
#
# Reference: 
# Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). 
# Imagenet classification with deep convolutional neural networks.
# In Advances in neural information processing systems (pp. 1097-1105).
 

import numpy as np
import matplotlib.pyplot as plt
import h5py
import time

# The caffe module needs to be on the Python path;
caffe_root = '/path/to/caffe/'  # this is the path to caffe, {caffe_root}/examples
data_root = '/path/to/videoframes/'
import caffe
caffe.set_device(0) # if use gpu
caffe.set_mode_gpu()

######################################## Load AlexNet model ###################################################
model_def = caffe_root + 'models/model.prototxt' # load model definition
model_weights = caffe_root + 'models/model.caffemodel' # load model weights
imsize = 227

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode 
                
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu[:,15:242,15:242]  # the mean (BGR) pixel values
    
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel


####################################### Classify example images ##############################################  
image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
plt.imshow(image)

# set the size of the input
net.blobs['data'].reshape(1,        # batch size
                          3,        # 3-channel (BGR) images
                          imsize, imsize)  # image size is 227x227
net.blobs['data'].data[...] = transformed_image

# perform classification
output = net.forward()
output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
print 'predicted class is:', output_prob.argmax()

# load ImageNet labels
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'    
labels = np.loadtxt(labels_file, str, delimiter='\t')
print 'output label:', labels[output_prob.argmax()]

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print 'probabilities and labels:'
zip(output_prob[top_inds], labels[top_inds])


########################################### Define layer labels ###################################################
layer_name_list = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8']

########################################## process training movies ################################################
# read middle layer feature maps
start = time.time()
srate = 30
Ns = 18 # number of training movie segments
numOfimages = srate*8*60
numlist = np.arange(0,numOfimages,int(30/srate)) # subsamle if needed
for seg in range(1,Ns+1):
    foldpath = data_root+'/AlexNet_feature_maps_seg' + str(seg)+'.h5'
    store = h5py.File(foldpath,'w')
    act={}
    for lay_idx in range(0,len(layer_name_list)): 
        layer_name = layer_name_list[lay_idx]
        grp1 = store.create_group(layer_name)
        temp = net.blobs[layer_name].data.shape 
        temp = list(temp)
        temp[0]=len(numlist)
        temp = tuple(temp)
        act[lay_idx]=grp1.create_dataset('data', temp, dtype='float16')
    k = 0
    for im in numlist:
        print "segment %d, frame %d" %(seg,im+1)
        image = caffe.io.load_image(data_root+'/frames/seg'+str(seg)+'/im-'+str(im+1)+'.jpg')
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        for lay_idx in range(0,len(layer_name_list)): 
            layer_name = layer_name_list[lay_idx]
            act[lay_idx][k,:] = net.blobs[layer_name].data
        k = k + 1    
    store.close()        
endtime = time.time()
    
print("--- %s seconds ---" % (endtime - start))

########################################## process testing movies ################################################
# read middle layer feature maps
start = time.time()
srate = 30
Ns = 5 # number of testing movie segments
numOfimages = srate*8*60
numlist = np.arange(0,numOfimages,int(30/srate)) # subsamle if needed
for seg in range(1,Ns+1):
    foldpath = data_root+'/AlexNet_feature_maps_test' + str(seg)+'.h5'
    store = h5py.File(foldpath,'w')
    act={}
    for lay_idx in range(0,len(layer_name_list)): 
        layer_name = layer_name_list[lay_idx]
        grp1 = store.create_group(layer_name)
        temp = net.blobs[layer_name].data.shape 
        temp = list(temp)
        temp[0]=len(numlist)
        temp = tuple(temp)
        act[lay_idx]=grp1.create_dataset('data', temp, dtype='float16')
    k = 0
    for im in numlist:
        print "test %d, frame %d" %(seg,im+1)
        image = caffe.io.load_image(data_root+'/frames/test'+str(seg)+'/im-'+str(im+1)+'.jpg')
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image
        output = net.forward()
        for lay_idx in range(0,len(layer_name_list)): 
            layer_name = layer_name_list[lay_idx]
            act[lay_idx][k,:] = net.blobs[layer_name].data
        k = k + 1    
    store.close()        
endtime = time.time()
    
print("--- %s seconds ---" % (endtime - start))
