from keras import *
from keras.layers import *
from keras import backend as K

class CustomPooling(Layer):
    def __init__(self, **kwargs):
        super(CustomPooling, self).__init__(**kwargs)

    def call(self, x):
        first_layer = GlobalMaxPooling1D(data_format='channels_last')(x)
        second_layer = GlobalAveragePooling1D(data_format='channels_last')(x)
        pooling = Add()([first_layer, second_layer])
        return pooling

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


class CustomResidual(Layer):
    def __init__(self, input_shape=(128, 8), **kwargs):
        super(CustomResidual, self).__init__(**kwargs)
        self.input_shapes = input_shape
        self.kernel_initializer = initializers.get('glorot_uniform')
        self.bias_initializer = initializers.get('zeros')
        self.kernel_regularizer = regularizers.get(None)
        self.bias_regularizer = regularizers.get(None)
        self.activation = activations.get('relu')
        self.kernel_constraint = constraints.get(None)
        self.bias_constraint = constraints.get(None)
        self.dilation_rate = conv_utils.normalize_tuple(1, 1,
                                                        'dilation_rate')
        self.filters = 256
        self.rank = 1
        self.padding = conv_utils.normalize_padding('valid')
        self.kernel_size = conv_utils.normalize_tuple(4, self.rank,
                                                      'kernel_size')
        self.strides = conv_utils.normalize_tuple(1, self.rank, 'strides')
        self.data_format = "channels_last"

    def call(self, x):
        #         print(np.shape(x), self.kernels[0])
        first_layer = self.activation(K.bias_add(
            K.conv1d(
                x,
                self.kernels[0],
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0]),
            self.biases[0],
            data_format=self.data_format)
        )

        #         print(np.shape(first_layer), self.kernels[1])
        x = self.activation(K.bias_add(
            K.conv1d(
                first_layer,
                self.kernels[1],
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0]),
            self.biases[1],
            data_format=self.data_format)
        )
        x = self.activation(K.bias_add(
            K.conv1d(
                x,
                self.kernels[2],
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0]),
            self.biases[2],
            data_format=self.data_format)
        )
        x = ZeroPadding1D(padding=3)(x)
        residual = Add()([x, first_layer])
        #         residual = Activation("relu")(residual)
        return residual

    def build(self, input_shape):
        input_dim = input_shape[-1]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernels = []
        self.biases = []
        self.kernels.append(self.add_weight(shape=kernel_shape,
                                            initializer=self.kernel_initializer,
                                            name='kernel1',
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint))
        input_dim = 256
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernels.append(self.add_weight(shape=kernel_shape,
                                            initializer=self.kernel_initializer,
                                            name='kernel2',
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint))
        input_dim = 256
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernels.append(self.add_weight(shape=kernel_shape,
                                            initializer=self.kernel_initializer,
                                            name='kernel3',
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint))
        self.biases.append(self.add_weight(shape=(self.filters,),
                                           initializer=self.bias_initializer,
                                           name='bias1',
                                           regularizer=self.bias_regularizer,
                                           constraint=self.bias_constraint))
        self.biases.append(self.add_weight(shape=(self.filters,),
                                           initializer=self.bias_initializer,
                                           name='bias2',
                                           regularizer=self.bias_regularizer,
                                           constraint=self.bias_constraint))
        self.biases.append(self.add_weight(shape=(self.filters,),
                                           initializer=self.bias_initializer,
                                           name='bias3',
                                           regularizer=self.bias_regularizer,
                                           constraint=self.bias_constraint))
        super(CustomResidual, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 125, 256)


def load_custom_model():
    model = Sequential()
    model.add(CustomResidual(input_shape=(128, 8)))
    model.add(CustomPooling())
    model.add(Dense(300, activation=None))
    model.add(LeakyReLU())
    model.add(Dense(150, activation=None))
    model.add(LeakyReLU())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adadelta(), metrics=['accuracy'])
    model.build(input_shape=(128, 8))
    return model
