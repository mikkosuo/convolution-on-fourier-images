# This file contains some information of the paths
# and set parameters. Include this in testing.py and training.py.

MODELS_FOLDER_PATH = "models"
TRAINING_HISTORY_FOLDER_PATH = "training history"
CIFAR10_CLASS_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

ModelInfo = {
            "Fourier cifar10 with flip and rotate layer" : {"ModelPath" : MODELS_FOLDER_PATH + "/fourier_cifar10_with_augment_layer",
                            "Parameters" : {"FourierInput" : True,
                                            "AugmentedLayer" : True,
                                            "Size" : [32, 32],
                                            "NumOfClasses" : 10},
                           "Dataset" : "Cifar10",
                           "ShortModelName" : "FourierCifar10WithFlipAndRotate",
                           "Architecture" : "simpleConv"},
        
        
            "Cifar10 with flip and rotate layer" : {"ModelPath" : MODELS_FOLDER_PATH + "/cifar10_with_augment_layer",
                                            "Parameters" : {"FourierInput" : False,
                                            "AugmentedLayer" : True,
                                            "Size" : [32, 32],
                                            "NumOfClasses" : 10},
                           "Dataset" : "Cifar10",
                           "ShortModelName" : "cifar10withFlipAndRotate",
                           "Architecture" : "simpleConv"},
    
    
            "Cifar10" : {"ModelPath" : MODELS_FOLDER_PATH + "/cifar10_no_augment_layer",
                          "Parameters" : {"FourierInput" : False,
                                          "AugmentedLayer" : False,
                                          "Size" : [32, 32],
                                          "NumOfClasses" : 10},
                           "Dataset" : "Cifar10",
                           "ShortModelName" : "cifar10",
                           "Architecture" : "simpleConv"},


            "Fourier cifar10" : {"ModelPath" : MODELS_FOLDER_PATH + "/fourier_cifar10_no_augment_layer",
                            "Parameters" : {"FourierInput" : True,
                                            "AugmentedLayer" : False,
                                            "Size" : [32, 32],
                                            "NumOfClasses" : 10},
                           "Dataset" : "Cifar10",
                           "ShortModelName" : "FourierCifar10",
                           "Architecture" : "simpleConv"},
            
            "MNIST" : {"ModelPath" : MODELS_FOLDER_PATH + "/MNIST_no_augment_layer",
                          "Parameters" : {"FourierInput" : False,
                                          "AugmentedLayer" : False,
                                          "Size" : [28, 28],
                                          "NumOfClasses" : 10},
                           "Dataset" : "MNIST",
                           "ShortModelName" : "MNIST",
                           "Architecture" : "simpleConv"},
            
            "MNIST with flip and rotate layer" : {"ModelPath" : MODELS_FOLDER_PATH + "/MNIST_with_augment_layer",
                "Parameters" : {"FourierInput" : False,
                                "AugmentedLayer" : True,
                                "Size" : [28, 28],
                                "NumOfClasses" : 10},
                "Dataset" : "MNIST",
                "ShortModelName" : "MNISTwithFlipAndRotate",
                "Architecture" : "simpleConv"},

            "Fourier MNIST" : {"ModelPath" : MODELS_FOLDER_PATH + "/fourier_MNIST_no_augment_layer",
                            "Parameters" : {"FourierInput" : True,
                                            "AugmentedLayer" : False,
                                            "Size" : [28, 28],
                                            "NumOfClasses" : 10},
                           "Dataset" : "MNIST",
                           "ShortModelName" : "FourierMNIST",
                           "Architecture" : "simpleConv"},

            "Fourier MNIST with flip and rotate layer" : {"ModelPath" : MODELS_FOLDER_PATH + "/fourier_MNIST_with_augment_layer",
                            "Parameters" : {"FourierInput" : True,
                                            "AugmentedLayer" : True,
                                            "Size" : [28, 28],
                                            "NumOfClasses" : 10},
                            "Dataset" : "MNIST",
                            "ShortModelName" : "FourierMNISTwithFlipAndRotate",
                            "Architecture" : "simpleConv"},
            

            #xception
            "Cifar10 (xception)" : {"ModelPath" : MODELS_FOLDER_PATH + "/cifar10_no_augment_layer_xception",
                          "Parameters" : {"FourierInput" : False,
                                          "AugmentedLayer" : False,
                                          "Size" : [32, 32],
                                          "NumOfClasses" : 10},
                           "Dataset" : "Cifar10",
                           "ShortModelName" : "cifar10_xception",
                           "Architecture" : "xception"},

            "Cifar10 with flip and rotate layer (xception)" : {"ModelPath" : MODELS_FOLDER_PATH + "/cifar10_with_augment_layer_xception",
                                            "Parameters" : {"FourierInput" : False,
                                            "AugmentedLayer" : True,
                                            "Size" : [32, 32],
                                            "NumOfClasses" : 10},
                           "Dataset" : "Cifar10",
                           "ShortModelName" : "cifar10withFlipAndRotate_xception",
                           "Architecture" : "xception"},
            "Fourier cifar10 (xception)" : {"ModelPath" : MODELS_FOLDER_PATH + "/fourier_cifar10_no_augment_layer_xception",
                            "Parameters" : {"FourierInput" : True,
                                            "AugmentedLayer" : False,
                                            "Size" : [32, 32],
                                            "NumOfClasses" : 10},
                           "Dataset" : "Cifar10",
                           "ShortModelName" : "FourierCifar10_xception",
                           "Architecture" : "xception"},

            "Fourier cifar10 with flip and rotate layer (xception)" : {"ModelPath" : MODELS_FOLDER_PATH + "/fourier_cifar10_with_augment_layer_xception",
                            "Parameters" : {"FourierInput" : True,
                                            "AugmentedLayer" : True,
                                            "Size" : [32, 32],
                                            "NumOfClasses" : 10},
                           "Dataset" : "Cifar10",
                           "ShortModelName" : "FourierCifar10WithFlipAndRotate_xception",
                           "Architecture" : "xception"},

            "MNIST (xception)" : {"ModelPath" : MODELS_FOLDER_PATH + "/MNIST_without_augment_layer_xception",
                "Parameters" : {"FourierInput" : False,
                                "AugmentedLayer" : False,
                                "Size" : [28, 28],
                                "NumOfClasses" : 10},
                "Dataset" : "MNIST",
                "ShortModelName" : "MNISTwithoutFlipAndRotate_xception",
                "Architecture" : "xception"},

            "MNIST fourier (xception)" : {"ModelPath" : MODELS_FOLDER_PATH + "/MNIST_fourier_without_augment_layer_xception",
                "Parameters" : {"FourierInput" : True,
                                "AugmentedLayer" : False,
                                "Size" : [28, 28],
                                "NumOfClasses" : 10},
                "Dataset" : "MNIST",
                "ShortModelName" : "fourierMNISTwithoutFlipAndRotate_xception",
                "Architecture" : "xception"},                

            "MNIST with flip and rotate layer (xception)" : {"ModelPath" : MODELS_FOLDER_PATH + "/MNIST_with_augment_layer_xception",
                "Parameters" : {"FourierInput" : False,
                                "AugmentedLayer" : True,
                                "Size" : [28, 28],
                                "NumOfClasses" : 10},
                "Dataset" : "MNIST",
                "ShortModelName" : "MNISTwithFlipAndRotate_xception",
                "Architecture" : "xception"},

            "Fourier MNIST with flip and rotate layer (xception)" : {"ModelPath" : MODELS_FOLDER_PATH + "/fourier_MNIST_with_augment_layer_xception",
                            "Parameters" : {"FourierInput" : True,
                                            "AugmentedLayer" : True,
                                            "Size" : [28, 28],
                                            "NumOfClasses" : 10},
                            "Dataset" : "MNIST",
                            "ShortModelName" : "FourierMNISTwithFlipAndRotate_xception",
                            "Architecture" : "xception"},                                           
              }

LINE_COLORS = {"Training" : {"Accuracy" : {
                                        "noTransform" : "royalblue",
                                        "flipAndRotate" : "forestgreen"},
                             "Loss" : {
                                        "noTransform" : "royalblue",
                                        "flipAndRotate" : "forestgreen"}},
               "Validation" : {"Accuracy" : {
                                        "noTransform" : "orange",
                                        "flipAndRotate" : "red"},
                             "Loss" : {
                                        "noTransform" : "orange",
                                        "flipAndRotate" : "red"}}}

LINE_STYLE = {
    "spectral" : "dashed",
    "spatial" : "solid"

}