#
# The only purpose of this file is to print some metrics
# and generate the figures. 
#
# Messy code, most likely unreadable.

import numpy as np
import tensorflow as tf
from tensorflow.signal import fft2d
from tensorflow.math import real, imag
import keras
from keras import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay , classification_report, accuracy_score
from header import ModelInfo, TRAINING_HISTORY_FOLDER_PATH, CIFAR10_CLASS_LABELS, LINE_STYLE, LINE_COLORS
import pickle
#import visualkeras
from PIL import ImageFont

# If the models aren't trained yet, train them with training.py 
# before running this file
print(keras.__version__)
PLOT_MODELS = False

''' Iterate the models and fill these dictionaries'''
testingAccuracies = {}
validationAccuracies = {}
trainingAccuracies = {}
trainingLosses = {}
confusionMatrices = {}
for modelName, modelInfo in ModelInfo.items():
    print(modelInfo["ModelPath"])
    loaded_model = tf.keras.models.load_model(modelInfo["ModelPath"]+".keras")
    if PLOT_MODELS:
        path = "figures/model_" + modelInfo["ShortModelName"] + ".png"
        font = ImageFont.truetype("arial.ttf", 12)
        visualkeras.layered_view(loaded_model, to_file="figures/" +modelInfo["ShortModelName"] + ".png", font=font, spacing = 23,
                                type_ignore=[tf.keras.layers.BatchNormalization])
    if modelInfo["Dataset"] == "Cifar10":
        (_, _), (X_test,Y_test) = datasets.cifar10.load_data()
    elif modelInfo["Dataset"] == "MNIST":
        (_, _), (X_test,Y_test) = datasets.mnist.load_data()
    else:
        print(f"Error wrong dataset name")
        break
    print(f"Now testing {modelName}")

    # Fourier transformation
    if modelInfo["Parameters"]["FourierInput"]:
        # keras.Input doesnt support complex64 tensors, so stack the imaginary part along new dimension
        if modelInfo["Dataset"] == "Cifar10":
            X_test = tf.concat([real(fft2d(X_test)), imag(fft2d(X_test))], axis=3)
        elif modelInfo["Dataset"] == "MNIST":
            X_test = tf.concat([real(fft2d(X_test))[..., tf.newaxis], imag(fft2d(X_test))[..., tf.newaxis]], axis=-1)    
        else:
            print(f"Error wrong dataset name")
            break
    
    y_pred = loaded_model.predict(X_test)
    y_pred_classes = [np.argmax(element) for element in y_pred]
    
    print("Classification Report: \n", classification_report(Y_test, y_pred_classes))
    print(f"Number of parameters: {loaded_model.count_params()}")
    print(f"Accuracy score: {accuracy_score(Y_test, y_pred_classes)}")
    testingAccuracies[modelName] = accuracy_score(Y_test, y_pred_classes)

    # Load the training history from the file
    with open((TRAINING_HISTORY_FOLDER_PATH + "/" + modelInfo["ShortModelName"] + "_training_history" + ".pkl"), 'rb') as file:
        loaded_history = pickle.load(file)

    validationAccuracies[modelName] = loaded_history['val_accuracy']
    trainingAccuracies[modelName] = loaded_history['accuracy']
    trainingLosses[modelName] = loaded_history['loss']
    confusionMatrices[modelName] = confusion_matrix(Y_test, y_pred_classes, normalize='true')

    
# ↓↓↓ All the following code is for plotting ↓↓↓

''' CONFUSION MATRICES'''
for modelName, cm in confusionMatrices.items():
    fig, ax = plt.subplots(figsize=(14, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CIFAR10_CLASS_LABELS if ModelInfo[modelName]["Dataset"] == "Cifar10" else None)
    disp.plot(ax=ax, values_format= '.2f', cmap='cividis')
    plt.title(f"Normalized confusion matrix - {modelName}")
    plt.savefig("figures/" + modelName + "_confusion_matrix")

''' DIFF MATRICES SIMPLE CONV'''
noFlipAndRotationDiff = confusionMatrices["Cifar10"] - confusionMatrices["Fourier cifar10"]
fig, ax = plt.subplots(figsize=(14, 14))
disp = ConfusionMatrixDisplay(confusion_matrix=noFlipAndRotationDiff, display_labels=CIFAR10_CLASS_LABELS)
disp.plot(ax=ax, values_format='.2f', cmap='coolwarm')
plt.title(f"Difference of the normalized confusion matrices - comparing CIFAR10 and Fourier CIFAR10")
plt.savefig("figures/no_flip_diff_confusion_matrix")
plt.close()

flipAndRotationDiff = confusionMatrices["Cifar10 with flip and rotate layer"] - confusionMatrices["Fourier cifar10 with flip and rotate layer"]
fig, ax = plt.subplots(figsize=(14, 14))
disp = ConfusionMatrixDisplay(confusion_matrix=flipAndRotationDiff, display_labels=CIFAR10_CLASS_LABELS)
disp.plot(ax=ax, values_format='.2f', cmap='coolwarm')
plt.title(f"Difference of the normalized confusion matrices - comparing augmented CIFAR-10 and augmented Fourier CIFAR-10")
plt.savefig("figures/flip_diff_confusion_matrix")
plt.close()

noFourierDiff = confusionMatrices["Cifar10 with flip and rotate layer"] - confusionMatrices["Cifar10"]
fig, ax = plt.subplots(figsize=(14, 14))
disp = ConfusionMatrixDisplay(confusion_matrix=noFourierDiff, display_labels=CIFAR10_CLASS_LABELS)
disp.plot(ax=ax, values_format='.2f', cmap='coolwarm')
plt.title(f"Difference of the normalized confusion matrices - comparing augmented CIFAR-10 with CIFAR10")
plt.savefig("figures/cifar_diff_confusion_matrix")
plt.close()

fourierDiff = confusionMatrices["Fourier cifar10 with flip and rotate layer"] - confusionMatrices["Fourier cifar10"]
fig, ax = plt.subplots(figsize=(14, 14))
disp = ConfusionMatrixDisplay(confusion_matrix=fourierDiff, display_labels=CIFAR10_CLASS_LABELS)
disp.plot(ax=ax, values_format='.2f', cmap='coolwarm')
plt.title(f"Difference of the normalized confusion matrices - comparing augmented Fourier CIFAR-10 with Fourier CIFAR10")
plt.savefig("figures/fourier_diff_confusion_matrix")
plt.close()

''' DIFF MATRICES XCEPTION'''
noFlipAndRotationDiff_xception = confusionMatrices["Cifar10 (xception)"] - confusionMatrices["Fourier cifar10 (xception)"]
fig, ax = plt.subplots(figsize=(14, 14))
disp = ConfusionMatrixDisplay(confusion_matrix=noFlipAndRotationDiff_xception, display_labels=CIFAR10_CLASS_LABELS)
disp.plot(ax=ax, values_format='.2f', cmap='coolwarm')
plt.title(f"Difference of the normalized confusion matrices - comparing CIFAR10 and Fourier CIFAR10 (Xceptions)")
plt.savefig("figures/no_flip_diff_confusion_matrix_xception")
plt.close()

flipAndRotationDiff_xception = confusionMatrices["Cifar10 with flip and rotate layer (xception)"] - confusionMatrices["Fourier cifar10 with flip and rotate layer (xception)"]
fig, ax = plt.subplots(figsize=(14, 14))
disp = ConfusionMatrixDisplay(confusion_matrix=flipAndRotationDiff_xception, display_labels=CIFAR10_CLASS_LABELS)
disp.plot(ax=ax, values_format='.2f', cmap='coolwarm')
plt.title(f"Difference of the normalized confusion matrices - comparing augmented CIFAR-10 and augmented Fourier CIFAR-10 (Xception)")
plt.savefig("figures/flip_diff_confusion_matrix_xception")
plt.close()

noFourierDiff_xception = confusionMatrices["Cifar10 with flip and rotate layer (xception)"] - confusionMatrices["Cifar10 (xception)"]
fig, ax = plt.subplots(figsize=(14, 14))
disp = ConfusionMatrixDisplay(confusion_matrix=noFourierDiff_xception, display_labels=CIFAR10_CLASS_LABELS)
disp.plot(ax=ax, values_format='.2f', cmap='coolwarm')
plt.title(f"Difference of the normalized confusion matrices - comparing augmented CIFAR-10 with CIFAR10 (xception)")
plt.savefig("figures/cifar_diff_confusion_matrix_xception")
plt.close()

fourierDiff_xception = confusionMatrices["Fourier cifar10 with flip and rotate layer (xception)"] - confusionMatrices["Fourier cifar10 (xception)"]
fig, ax = plt.subplots(figsize=(14, 14))
disp = ConfusionMatrixDisplay(confusion_matrix=fourierDiff_xception, display_labels=CIFAR10_CLASS_LABELS)
disp.plot(ax=ax, values_format='.2f', cmap='coolwarm')
plt.title(f"Difference of the normalized confusion matrices - comparing augmented Fourier CIFAR-10 with Fourier CIFAR10 (xception)")
plt.savefig("figures/fourier_diff_confusion_matrix_xception")
plt.close()


# ↓↓↓ Plot the cifar with simpleconv ↓↓↓

''' Plot training losses '''
plt.figure()
for modelName, loss in trainingLosses.items():
    if ModelInfo[modelName]["Dataset"] != "Cifar10" or ModelInfo[modelName]["Architecture"] != "simpleConv":
        continue
    plt.plot(loss, label=modelName, 
             linestyle = LINE_STYLE["spectral" if ModelInfo[modelName]["Parameters"]["FourierInput"] else "spatial"],
             color = LINE_COLORS["Training"]["Loss"]["flipAndRotate" if ModelInfo[modelName]["Parameters"]["AugmentedLayer"] else "noTransform"])
plt.title('Training Loss Development Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(framealpha=1.0)
plt.savefig("figures/training_losses")
plt.close()

''' Plot validation accuracies '''
plt.figure()
for modelName, loss in validationAccuracies.items():
    if ModelInfo[modelName]["Dataset"] != "Cifar10" or ModelInfo[modelName]["Architecture"] != "simpleConv":
        continue
    plt.plot(loss, label=modelName,
             linestyle = LINE_STYLE["spectral" if ModelInfo[modelName]["Parameters"]["FourierInput"] else "spatial"],
             color = LINE_COLORS["Training"]["Accuracy"]["flipAndRotate" if ModelInfo[modelName]["Parameters"]["AugmentedLayer"] else "noTransform"])
plt.title('Validation Accuracy Development Over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(framealpha=1.0, loc='lower right')
plt.grid(True)
plt.yticks(np.arange(0, 1.05, 0.05))
plt.ylim(0,1)
plt.savefig("figures/validation_accuracies")
plt.close()

''' Plot training accuracies '''
plt.figure()
for modelName, accuracies in trainingAccuracies.items():
    if ModelInfo[modelName]["Dataset"] != "Cifar10" or ModelInfo[modelName]["Architecture"] != "simpleConv":
        continue
    plt.plot(accuracies, label=modelName,
             linestyle = LINE_STYLE["spectral" if ModelInfo[modelName]["Parameters"]["FourierInput"] else "spatial"],
             color = LINE_COLORS["Training"]["Accuracy"]["flipAndRotate" if ModelInfo[modelName]["Parameters"]["AugmentedLayer"] else "noTransform"])
plt.title('Training Accuracy Development Over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(framealpha=1.0, loc='lower right')
plt.grid(True)
plt.yticks(np.arange(0, 1.05, 0.05))
plt.ylim(0,1)
plt.savefig("figures/training_accuracies")
plt.close()

''' Plot training and validation accuracies '''
plt.figure() # Spatial
for modelName, accuracies in trainingAccuracies.items():
    if ModelInfo[modelName]["Dataset"] != "Cifar10" or ModelInfo[modelName]["Architecture"] != "simpleConv":
        continue
    if ModelInfo[modelName]["Parameters"]["FourierInput"] or ModelInfo[modelName]["Architecture"] != "simpleConv":
        continue
    plt.plot(accuracies, label=modelName + ( "(training)"),
             linestyle = "dashed" if ModelInfo[modelName]["Parameters"]["AugmentedLayer"] else "solid",
             color = "royalblue")
    plt.plot(validationAccuracies[modelName], label = modelName + ( "(validation)"),
            linestyle = "dashed" if ModelInfo[modelName]["Parameters"]["AugmentedLayer"] else "solid",
             color = "forestgreen")
plt.title('Training And Validation Accuracies Over Epochs (Spatial)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(framealpha=1.0, loc='lower right')
plt.grid(True)
plt.yticks(np.arange(0, 1.05, 0.05))
plt.ylim(0,1)
plt.savefig("figures/training_and_validation_accuracies_spatial")
plt.close()

plt.figure() # Fourier
for modelName, accuracies in trainingAccuracies.items():
    if ModelInfo[modelName]["Dataset"] != "Cifar10" or ModelInfo[modelName]["Architecture"] != "simpleConv":
        continue
    if not ModelInfo[modelName]["Parameters"]["FourierInput"] or ModelInfo[modelName]["Architecture"] != "simpleConv":
        continue
    plt.plot(accuracies, label=modelName + ( "(training)"),
             linestyle = "dashed" if ModelInfo[modelName]["Parameters"]["AugmentedLayer"] else "solid",
             color = "royalblue")
    plt.plot(validationAccuracies[modelName], label = modelName + ( "(validation)"),
            linestyle = "dashed" if ModelInfo[modelName]["Parameters"]["AugmentedLayer"] else "solid",
             color = "forestgreen")
plt.title('Training And Validation Accuracies Over Epochs (Fourier)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(framealpha=1.0, loc='lower right')
plt.grid(True)
plt.yticks(np.arange(0, 1.05, 0.05))
plt.ylim(0,1)
plt.savefig("figures/training_and_validation_accuracies_fourier")
plt.close()

# ↓↓↓ Plot the mnist ↓↓↓
''' Plot validation accuracies '''
plt.figure()
for modelName, loss in validationAccuracies.items():
    if ModelInfo[modelName]["Dataset"] != "MNIST" or ModelInfo[modelName]["Architecture"] != "simpleConv":
        continue
    plt.plot(loss, label=modelName,
             linestyle = LINE_STYLE["spectral" if ModelInfo[modelName]["Parameters"]["FourierInput"] else "spatial"],
             color = LINE_COLORS["Training"]["Accuracy"]["flipAndRotate" if ModelInfo[modelName]["Parameters"]["AugmentedLayer"] else "noTransform"])
plt.title('Validation Accuracy Development Over Epochs (MNIST)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(framealpha=1.0, loc='lower right')
plt.grid(True)
plt.yticks(np.arange(0, 1.05, 0.05))
plt.ylim(0,1)
plt.savefig("figures/MNIST_validation_accuracies")
plt.close()

# ↓↓↓ Plot the cifar with xception ↓↓↓

''' Plot training losses '''
plt.figure()
for modelName, loss in trainingLosses.items():
    if ModelInfo[modelName]["Dataset"] != "Cifar10" or ModelInfo[modelName]["Architecture"] != "xception":
        continue
    plt.plot(loss, label=modelName, 
             linestyle = LINE_STYLE["spectral" if ModelInfo[modelName]["Parameters"]["FourierInput"] else "spatial"],
             color = LINE_COLORS["Training"]["Loss"]["flipAndRotate" if ModelInfo[modelName]["Parameters"]["AugmentedLayer"] else "noTransform"])
plt.title('Training Loss Development Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(framealpha=1.0)
plt.savefig("figures/training_losses_xception")
plt.close()

''' Plot validation accuracies '''
plt.figure()
for modelName, loss in validationAccuracies.items():
    if ModelInfo[modelName]["Dataset"] != "Cifar10" or ModelInfo[modelName]["Architecture"] != "xception":
        continue
    plt.plot(loss, label=modelName,
             linestyle = LINE_STYLE["spectral" if ModelInfo[modelName]["Parameters"]["FourierInput"] else "spatial"],
             color = LINE_COLORS["Training"]["Accuracy"]["flipAndRotate" if ModelInfo[modelName]["Parameters"]["AugmentedLayer"] else "noTransform"])
plt.title('Validation Accuracy Development Over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(framealpha=1.0, loc='lower right')
plt.grid(True)
plt.yticks(np.arange(0, 1.05, 0.05))
plt.ylim(0,1)
plt.savefig("figures/validation_accuracies_xception")
plt.close()

''' Plot training accuracies '''
plt.figure()
for modelName, accuracies in trainingAccuracies.items():
    if ModelInfo[modelName]["Dataset"] != "Cifar10" or ModelInfo[modelName]["Architecture"] != "xception":
        continue
    plt.plot(accuracies, label=modelName,
             linestyle = LINE_STYLE["spectral" if ModelInfo[modelName]["Parameters"]["FourierInput"] else "spatial"],
             color = LINE_COLORS["Training"]["Accuracy"]["flipAndRotate" if ModelInfo[modelName]["Parameters"]["AugmentedLayer"] else "noTransform"])
plt.title('Training Accuracy Development Over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(framealpha=1.0, loc='lower right')
plt.grid(True)
plt.yticks(np.arange(0, 1.05, 0.05))
plt.ylim(0,1)
plt.savefig("figures/training_accuracies_xception")
plt.close()

''' Plot training and validation accuracies '''
plt.figure() # Spatial
for modelName, accuracies in trainingAccuracies.items():
    if ModelInfo[modelName]["Dataset"] != "Cifar10" or ModelInfo[modelName]["Architecture"] != "xception":
        continue
    if ModelInfo[modelName]["Parameters"]["FourierInput"] or ModelInfo[modelName]["Architecture"] != "xception":
        continue
    plt.plot(accuracies, label=modelName + ( "(training)"),
             linestyle = "dashed" if ModelInfo[modelName]["Parameters"]["AugmentedLayer"] else "solid",
             color = "royalblue")
    plt.plot(validationAccuracies[modelName], label = modelName + ( "(validation)"),
            linestyle = "dashed" if ModelInfo[modelName]["Parameters"]["AugmentedLayer"] else "solid",
             color = "forestgreen")
plt.title('Training And Validation Accuracies Over Epochs (Spatial)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(framealpha=1.0, loc='lower right')
plt.grid(True)
plt.yticks(np.arange(0, 1.05, 0.05))
plt.ylim(0,1)
plt.savefig("figures/training_and_validation_accuracies_spatial_xception")
plt.close()

plt.figure() # Fourier
for modelName, accuracies in trainingAccuracies.items():
    if ModelInfo[modelName]["Dataset"] != "Cifar10" or ModelInfo[modelName]["Architecture"] != "xception":
        continue
    if not ModelInfo[modelName]["Parameters"]["FourierInput"] or ModelInfo[modelName]["Architecture"] != "xception":
        continue
    plt.plot(accuracies, label=modelName + ( "(training)"),
             linestyle = "dashed" if ModelInfo[modelName]["Parameters"]["AugmentedLayer"] else "solid",
             color = "royalblue")
    plt.plot(validationAccuracies[modelName], label = modelName + ( "(validation)"),
            linestyle = "dashed" if ModelInfo[modelName]["Parameters"]["AugmentedLayer"] else "solid",
             color = "forestgreen")
plt.title('Training And Validation Accuracies Over Epochs (Fourier)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(framealpha=1.0, loc='lower right')
plt.grid(True)
plt.yticks(np.arange(0, 1.05, 0.05))
plt.ylim(0,1)
plt.savefig("figures/training_and_validation_accuracies_fourier_xception")
plt.close()

print("\nAll plots saved to figures folder.")
#plt.show()

print("Testing accuracies:")
for model, accuracy in testingAccuracies.items():
    print(f"{model}: {accuracy:.2%}")