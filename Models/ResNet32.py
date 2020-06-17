class ResNet32:
    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes
        return

    def compositeConv(self, filters, kernel_size, num_convs, inputs, residual_connection=False):
        outputs = inputs
        for i in range(num_convs):
            outputs = Conv2D(filters=filters, kernel_size=kernel_size, padding="same")(outputs)
            outputs = Activation("relu")(outputs)
            outputs = BatchNormalization(axis=-1)(outputs)
        if residual_connection: return inputs+outputs
        else: return outputs

    def create_model(self):
        inputs = Input(shape=self.input_shape)

        outputs = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same")(inputs)
        outputs = Activation("relu")(outputs)
        outputs = BatchNormalization(axis=-1)(outputs)
        outputs = Dropout(0.25)(outputs)

        for i in range(3):
            outputs = self.compositeConv(filters=64, kernel_size=(3, 3), num_convs=2, inputs=outputs)
        outputs = Dropout(0.25)(outputs)

        outputs = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same")(outputs)
        outputs = Activation("relu")(outputs)
        outputs = BatchNormalization(axis=-1)(outputs)
        outputs = Conv2D(filters=128, kernel_size=(3, 3), padding="same")(outputs)
        outputs = Activation("relu")(outputs)
        outputs = BatchNormalization(axis=-1)(outputs)

        for i in range(3):
            outputs = self.compositeConv(filters=128, kernel_size=(3, 3), num_convs=2, inputs=outputs)
        outputs = Dropout(0.25)(outputs)

        outputs = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same")(outputs)
        outputs = Activation("relu")(outputs)
        outputs = BatchNormalization(axis=-1)(outputs)
        outputs = Conv2D(filters=256, kernel_size=(3, 3), padding="same")(outputs)
        outputs = Activation("relu")(outputs)
        outputs = BatchNormalization(axis=-1)(outputs)

        for i in range(5):
            outputs = self.compositeConv(filters=256, kernel_size=(3, 3), num_convs=2, inputs=outputs)
        outputs = Dropout(0.25)(outputs)

        outputs = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same")(outputs)
        outputs = Activation("relu")(outputs)
        outputs = BatchNormalization(axis=-1)(outputs)
        outputs = Conv2D(filters=512, kernel_size=(3, 3), padding="same")(outputs)
        outputs = Activation("relu")(outputs)
        outputs = BatchNormalization(axis=-1)(outputs)

        for i in range(2):
            outputs = self.compositeConv(filters=512, kernel_size=(3, 3), num_convs=2, inputs=outputs)
        outputs = Dropout(0.25)(outputs)

        outputs = MaxPooling2D()(outputs)

        outputs = Flatten()(outputs)

        outputs = Dense(self.classes)(outputs) # self.classes contains the number of classes
        outputs = Activation("softmax")(outputs) # softmax classifier

        model = Model(inputs, outputs)
        return model
