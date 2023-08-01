# Root_Image_Segmentation 

Data:
Roots were grown in a mesh frame that supported the root system after removing the soil. Images were then captured using a camera at the University of Georgia.

The data can be found in the folder "data/root_data2."

Data Augmentation:
The training data consists of 30 3456x5184 images, which were augmented using ImageDataGenerator to feed a deep learning neural network.

For more details, refer to preProcess.py.

Model:
This deep neural network is implemented using the Keras functional API, allowing easy experimentation with various architectures.

The network's output is a 512x512 mask that needs to be learned, and the sigmoid activation function ensures that mask pixels are in the [0, 1] range.

Training:
The model was trained for 5 epochs, and after this training, the calculated accuracy is approximately 0.97.

The loss function used during training is binary crossentropy.
