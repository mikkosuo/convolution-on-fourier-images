# Define the models and a function for initializing them

from tensorflow import keras
from keras import layers

def InitializeModels(ModelInfo):
    for modelName, ModelInfo in ModelInfo.items():

        fourierInput = ModelInfo["Parameters"]["FourierInput"]
        augmentedLayer = ModelInfo["Parameters"]["AugmentedLayer"]
        size = ModelInfo["Parameters"]["Size"]
        numOfClasses = ModelInfo["Parameters"]["NumOfClasses"]

        blackAndWhite = False
        if ModelInfo["Dataset"] == "MNIST":
            blackAndWhite = True

        print(f"Initializing a model for {modelName} ... ", end='')
        print("WITH " if augmentedLayer else "WITHOUT ", end='')
        print("flip and rotate layer... ", end='')

        if ModelInfo["Architecture"] == "simpleConv":
            ann = simple_convolution(FourierInput=fourierInput, AugmentmentLayer=augmentedLayer,Size = size,
                                    NumOfClasses=numOfClasses, BlackAndWhite=blackAndWhite)
        elif ModelInfo["Architecture"] == "xception":
            ann = Xception(FourierInput=fourierInput, AugmentmentLayer=augmentedLayer,Size = size,
                                    NumOfClasses=numOfClasses, BlackAndWhite=blackAndWhite)
        else:
            print("Error, wrong architecture name")
            exit()
        
        ann.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])    
        ann.save(ModelInfo["ModelPath"] + ".keras")
        print("Model saved.")

def FlipAndRotateLayer():
    data_augmentation = keras.Sequential(
    [   
        layers.RandomFlip(mode="horizontal"),
        layers.RandomRotation(factor=0.03, fill_mode="reflect", interpolation="bilinear"),
    ]
    )
    return data_augmentation

def simple_convolution(FourierInput = False, AugmentmentLayer = True, Size = [32, 32], NumOfClasses = 10, BlackAndWhite = False):

    channels = 1 if BlackAndWhite else 3
    if FourierInput:
        channels *= 2

    inputs = keras.Input(shape=(Size[0], Size[1], channels))
    augmented = FlipAndRotateLayer()(inputs)

    x = layers.BatchNormalization()(augmented if AugmentmentLayer else inputs)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    outputs = layers.Dense(NumOfClasses, activation='softmax')(x)

    return keras.Model(inputs, outputs)


def Xception(FourierInput = False, AugmentmentLayer = True, Size = [32, 32], NumOfClasses = 10, BlackAndWhite = False):

    channels = 1 if BlackAndWhite else 3
    if FourierInput:
        channels *= 2

    inputs = keras.Input(shape=(Size[0], Size[1], channels))
    augmented = FlipAndRotateLayer()(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(augmented if AugmentmentLayer else inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    activation = "softmax"
    units = NumOfClasses

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)