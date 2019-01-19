# DeepLearning18
ETH students in Deep Learning attempting to solve kaggle competition airbus-ship-detection.

The problem consists of detecting all ships in satellite colour images. 
A bounding box must be generated around every instance of a ship in an image.

Credits to https://github.com/MrGemy95/Tensorflow-Project-Template for the amazing template

# SLIC model
Our SLIC model uses the superpixel segmentation SLIC algorithm ("SLIC Superpixels, Achanta et al., 2010") and apply on top of it a convolutional neural network to classify whether each patch contains a boat. 

To test the SLIC model, download the test images from https://polybox.ethz.ch/index.php/s/0gBtvmVmJyrX8a2 and save the file in data folder. Then download the checkpoints of the CNN training https://polybox.ethz.ch/index.php/s/fUwhqZnAhiy6cgF and save the baseline file in experiments folder. Select the number of images you want to test in `configs/baseline.jsn` (default is 20). 
Then run `python mains/baseline_tester.py -c configs/baseline.jsn`.
It will output each image with the predicted mask.

# Adapted Faster R-CNN
The model is essentially a modified version of the Region Proposal Network of Faster R-CNN. The multiclass classifier of Faster R-CNN is not needed since our problem is a binary task (ship or background), hence we chop that part off our model. The RPN consists mainly on a FCN that generates a feature map of an image, plus another CNN with two branches, one proposing bounding boxes and the other giving probabilities of those boxes containing ships.
