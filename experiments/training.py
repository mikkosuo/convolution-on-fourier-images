# Initialize, train, and save the models (and training hisotry)

import tensorflow as tf
from tensorflow.signal import fft2d
from tensorflow.math import real, imag
from tensorflow import keras
import pickle
import keras
from keras import datasets
import models
from header import ModelInfo, TRAINING_HISTORY_FOLDER_PATH
print(f"Tensorflow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


MODELS_INITIALIZED = False  # True = train the existing models further
                            # False = reset/initialize all models
# the history files will get reseted in all cases

if input("Retrain all models? y/n: " if not MODELS_INITIALIZED else 
         "Reset the history data? (models will be trained further) y/n: ") != "y":
    exit()

NUM_OF_TRAINING_EPOCHS = 120
VALIDATION_SPLIT = 0.2

if not MODELS_INITIALIZED: models.InitializeModels(ModelInfo)

# train the models
for modelName, modelInfo in ModelInfo.items():
    if modelInfo["Dataset"] == "Cifar10":
        (X_train, y_train), (_,_) = datasets.cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train)
    elif modelInfo["Dataset"] == "MNIST":
        (X_train, y_train), (_,_) = datasets.mnist.load_data()
        y_train = keras.utils.to_categorical(y_train)
    else:
        print(f"Error wrong dataset name")
        break

    print(f"Now training {modelName}")
    loaded_model = keras.models.load_model(modelInfo["ModelPath"] + ".keras")

    # Fourier transformation
    if modelInfo["Parameters"]["FourierInput"]:
        # keras.Input doesnt support complex64 tensors, so stack the imaginary part along new dimension
        if modelInfo["Dataset"] == "Cifar10":
            X_train = tf.concat([real(fft2d(X_train)), imag(fft2d(X_train))], axis=3)
        elif modelInfo["Dataset"] == "MNIST":
            X_train = tf.concat([real(fft2d(X_train))[..., tf.newaxis], imag(fft2d(X_train))[..., tf.newaxis]], axis=-1)    
        else:
            print(f"Error wrong dataset name")
            break
        print(f"Shape after fourier transform: {X_train.shape}")
        
    _E = NUM_OF_TRAINING_EPOCHS # shorten the variable name
    history = loaded_model.fit(X_train, y_train, epochs=NUM_OF_TRAINING_EPOCHS, validation_split = VALIDATION_SPLIT, validation_freq = 1,
                               callbacks=[keras.callbacks.LearningRateScheduler(
                                   lambda epoch, lr: lr if epoch < _E * 3 / 5 else (1e-5 if epoch < _E*4/5 else 1e-6))])
    
    # Save training history and model for evaluation
    with open((TRAINING_HISTORY_FOLDER_PATH + "/" + modelInfo["ShortModelName"] + "_training_history" + ".pkl"), 'wb') as file:
        pickle.dump(history.history, file)
    loaded_model.save(modelInfo["ModelPath"] + ".keras")

print("Training done, see test results in testing.py")
