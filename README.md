# convolution-on-fourier-images
Code used comparing a simple CNN's and Xceptions performance on Fourier transformed images, and code for creating the figures in the experiments sections of my bachelor thesis.

The experiments contain the files in a state that they were at the end of the study. Including the saved models, training history, and figures. The saved models can be tested by
running the testing.py. By running the training.py the models can be retrained or trained further. The history file will get overwritten in the case of further training. To train further
the MODELS_INITIALIZED flag has to be changed to True. Otherwise, the models are initialized.

These libraries have to be installed to run the code:
- Tensorflow (keras has to be version 3) 
- Numpy
- Matplotlib
- Scikit-learn

Use the included requirements.txt to install dependencies (for Linux users since Windows doesn't seem to support this version of Tensorflow):
- pip install -r /path_to_folder/requirements.txt
