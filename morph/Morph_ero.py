from keras.engine.topology import Layer,InputSpec
import keras.backend as K
import numpy as np
import tensorflow as tf

class Erossion(Layer):
    def __init__(self,filters=1,
                 kernel_size=(3,3),
                 strides=(1,1),
                 data_format='channels_last',
                 operation = 'e',
                 **kwargs):
        super(Erossion, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.data_format = data_format
        self.operation = operation
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],self.filters)

    def build(self,input_shape):
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) + '; Received input shape:',
                             str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]
        # kernel_shape = (self.filters,) + self.kernel_size + (input_dim,)
        kernel_shape = (self.filters,) + (np.prod(self.kernel_size)*input_dim,)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer='uniform',
                                      name='kernel',
                                      dtype='float32',
                                      trainable=True)
        super(Erossion, self).build(input_shape)
        # self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        # self.built = True

    def call(self, x):
        if self.operation == 'e':
            return self.erosion_appox(x)
        # elif self.operation == 'e':
        #     return x
        else:
            raise ValueError('Operation not supported.')

    ### helpers
    def erosion_appox(self,img):
        shape = K.int_shape(img)
        height, width, channel = shape[1],shape[2],shape[3],

        kernal = [1] + [self.kernel_size[0],self.kernel_size[1]] + [1]
        stride = [1] + [self.strides[0],self.strides[1]] + [1]
        rates = [1, 1, 1, 1]
        padding = "SAME"

        patches = tf.extract_image_patches(images=img, ksizes=kernal, strides=stride, \
                                           rates=rates, padding=padding)
        kernal_weight = self.kernel

        for kn in range(0,self.filters):
            weighted_patch = patches * kernal_weight[kn,:]  # w .* each patch

            ### weighted_patch norm
            weighted_patch_min = K.min(weighted_patch)
            weighted_patch_max = K.max(weighted_patch)
            if weighted_patch_min != weighted_patch_max:
                range_constant = 80  # tf.exp can take 88 as the max input
                weighted_patch_norm = (weighted_patch - weighted_patch_min) / \
                                      (weighted_patch_max - weighted_patch_min) * range_constant
                flag = 0
            else:
                if weighted_patch_max == 0:
                    weighted_patch_max == 1e-4
                weighted_patch_norm = weighted_patch / weighted_patch_max  * range_constant
                flag = 1

            max_of_patch = -K.log(K.sum(K.exp(-weighted_patch_norm), axis=-1))

            ### inverse norm of weighted patch
            if flag == 0:
                max_of_patch_norm = max_of_patch * (weighted_patch_max - weighted_patch_min) \
                                    / range_constant + weighted_patch_min
            else:
                max_of_patch_norm = max_of_patch / range_constant * weighted_patch_max
            max_of_patch_norm = K.expand_dims(max_of_patch_norm,axis=-1)

            if kn == 0:
                output = max_of_patch_norm
            else:
                output = K.concatenate([output,max_of_patch_norm],axis=-1)
        return output

    def extract_image_patches(self,X, ksizes, ssizes, border_mode="same", dim_ordering="tf"):
        kernel = [1, ksizes[0], ksizes[1], 1]
        strides = [1, ssizes[0], ssizes[1], 1]
        padding = border_mode.upper()
        if dim_ordering == "th":
            X = K.permute_dimensions(X, [0, 2, 3, 1])
        bs_i, w_i, h_i, ch_i = K.int_shape(X)
        patches = tf.extract_image_patches(X, kernel, strides, [1, 1, 1, 1], padding)
        # Reshaping to fit Theano
        bs, w, h, ch = K.int_shape(patches)
        patches = tf.reshape(tf.transpose(tf.reshape(patches, [bs, w, h, -1, ch_i]), [0, 1, 2, 4, 3]),
                                [bs, w, h, ch_i, ksizes[0], ksizes[1]])
        if dim_ordering == "tf":
            patches = K.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
        return patches

    def oper(self,kerneln,patchn):
        return -K.log(K.sum(K.exp(-patchn*kerneln)))
