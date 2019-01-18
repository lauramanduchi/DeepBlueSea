# DeepLearning18
ETH students in Deep Learning attempting to solve kaggle competition airbus-ship-detection

Credits to https://github.com/MrGemy95/Tensorflow-Project-Template for the amazing template

# SLIC model
To test the SLIC model, download the test images from https://polybox.ethz.ch/index.php/s/ltwnzltPug10ggc and save the file in data folder. Select the number of images you want to test in `configs/baseline.jsn` (default is 20). 
Then run `python mains/baseline_tester.py -c configs/baseline.jsn`.
It will output each image with the predicted mask.
