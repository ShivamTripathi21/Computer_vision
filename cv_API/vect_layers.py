from __future__ import print_function
from __future__ import division
import numpy as np
# try:
#     from cs231n.im2col_cython import col2im_cython, im2col_cython
#     from cs231n.im2col_cython import col2im_6d_cython
# except ImportError:
#     print('run the following from the cs231n directory and try again:')
#     print('python setup.py build_ext --inplace')
#     print('You may also need to restart your iPython kernel')

def create_im2col(x, field_height, field_width, pad=1, stride=1):
	""" An implementation of im2col based on some fancy indexing """
	
	# 0 pad the input
	
	p = pad
	x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
	
	# First figure out what the size of the output should be
	N, C, H, W = x.shape
    
    
    #output size should be (c * filt_h * filt_w) X (out_height * out_width)
	assert (H + 2 * pad - field_height) % stride == 0
	assert (W + 2 * pad - field_height) % stride == 0
	out_height = (H + 2 * pad - field_height) // stride + 1
	out_width = (W + 2 * pad - field_width) // stride + 1
    
	i0 = np.repeat(np.arange(field_height), field_width)
	i0 = np.tile(i0, C)
	i1 = stride * np.repeat(np.arange(out_height), out_width)
	j0 = np.tile(np.arange(field_width), field_height * C)
	j1 = stride * np.tile(np.arange(out_width), out_height)
	i = i0.reshape(-1, 1) + i1.reshape(1, -1)
	j = j0.reshape(-1, 1) + j1.reshape(1, -1)

	k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    
    # now use k, i, j
    
	cols = x_padded[:, k, i, j]
	cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    
	return cols
	
	

def conv_forward_im2col(x, w, b, conv_param):
	"""
	A ast implementation of the forward pass for a conv layer
	based on im2col and col2im.
	"""
	N, C, H, W = x.shape
	num_filters, _, filter_height, filter_width =w.shape
	stride, pad = conv_param['stride'], conv_param['pad']
	
	#check dimensions
	
	assert (W + 2*pad - filter_width) % stride == 0, 'width does not work'
	assert (H + 2*pad - filter_height) % stride == 0, 'height does not work'
	
	#create output
	# we will use // because we want to get integer in our parameter
	our_height = (H + 2 * pad - filter_height) // stride + 1
	our_width = (W + 2 * pad - filter_width) // stride + 1
	
	out = np.zeros((N, num_filters, our_height, our_width), dtype=x.dtype)
	
	# now we have to find x_col of size 11X11X3 X 55X55
	# required parameter is x, filter_height, filter_width, pad, stride
	
	x_col = create_im2col(x, filter_height, filter_width, pad=pad, stride=stride)
	
	res = w.reshape((w.shape[0], -1)).dot(x_col) + b.reshape(-1, 1)
	
	out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
	out = out.transpose(3, 0, 1, 2)
	
	cache = (x, w, b, conv_param, x_col)
	return out,cache
	


	
	
	
