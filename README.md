# DeepLearning18
ETH students in Deep Learning attempting to solve kaggle competition airbus-ship-detection.

The problem consists of detecting all ships in satellite colour images. 
A bounding box must be generated around every instance of a ship in an image.

Credits to https://github.com/MrGemy95/Tensorflow-Project-Template for the amazing template

# SLIC model
Our SLIC model uses the superpixel segmentation SLIC algorithm ("SLIC Superpixels, Achanta et al., 2010") and apply on top of it a convolutional neural network to classify whether each patch contains a boat. 

To test the SLIC model, download the test images from https://polybox.ethz.ch/index.php/s/ltwnzltPug10ggc and save the file in data folder. Select the number of images you want to test in `configs/baseline.jsn` (default is 20). 
Then run `python mains/baseline_tester.py -c configs/baseline.jsn`.
It will output each image with the predicted mask.
