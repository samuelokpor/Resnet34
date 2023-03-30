import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, GlobalAvgPool2D, Dense, Flatten, add
from keras import Input, Model, Sequential

num_classes = 3

class ResBlock(tf.keras.layers.Layer):
    #conv
    #batch Norm
    #relu
    #conv
    #batch Norm
    # add input to last layer
    # - identify if input and output shapes are same
    # - conv 1x1 if input and output shapes different
    # Resnet 34 has two conv blocks in resblock

    def __init__(self, num_channels, use_conv=False, strides=1):
        super(ResBlock, self).__init__()
        self.conv1 = Conv2D(num_channels, kernel_size=3, strides=strides, padding='same')
        self.conv2 = Conv2D(num_channels, kernel_size=3, padding='same')
        self.conv3 = None
        if use_conv:
            self.conv3 = Conv2D(num_channels, kernel_size=1, strides=strides)
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()

    def call(self, in_):
        x = ReLU()(self.bn1(self.conv1(in_)))
        x = self.bn2(self.conv2(x))
        if self.conv3 is not None:
            in_ = self.conv3(in_)
        x = add([x, in_])
        return ReLU()(x)

class ResLayer(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False):
        super(ResLayer, self).__init__()
        self.num_channels = num_channels
        self.num_residuals = num_residuals
        self.first_block = first_block
        self.residual_layers = []
        for i in range(num_residuals):
            if i==0 and not first_block:
                self.residual_layers.append(
                    ResBlock(num_channels, use_conv=True, strides=2)
                )
            else:
                self.residual_layers.append(ResBlock(num_channels))


    def call(self, x):
        for layer in self.residual_layers:
            x = layer(x)
        return x
    
    def get_config(self):

        config = super().get_config()
        config.update({
            'num_channels':self.num_channels,
            'num_residuals':self.num_residuals,
            'first_block':self.first_block
        })
        return config
    
def ResNet34():
    return Sequential([
    Conv2D(64, kernel_size=7, strides=2, padding='same', input_shape=(224, 224, 3)),
    BatchNormalization(),
    ReLU(),
    MaxPool2D(pool_size=3, strides=2, padding='same'),
    ResLayer(64, 3, first_block=True),
    ResLayer(128, 4),
    ResLayer(256, 6),
    ResLayer(512, 3),
    GlobalAvgPool2D(),
    Dense(num_classes, activation='softmax')
    ])



resnet34instance = ResNet34()
print(resnet34instance.summary())