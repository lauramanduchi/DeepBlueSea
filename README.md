# DeepLearning18
ETH students in Deep Learning attempting to solve kaggle competition airbus-ship-detection.

The problem consists of detecting all ships in satellite colour images. 
A bounding box must be generated around every instance of a ship in an image.

## Data
The data can be found on the official kaggle web page https://www.kaggle.com/c/airbus-ship-detection/data.
We sampled 10'000 pictures out of the full dataset which can be found on https://polybox.ethz.ch/index.php/s/Td5L3qgKayPIm9s.

## SLIC model
Our SLIC model uses the superpixel segmentation SLIC algorithm ("SLIC Superpixels, Achanta et al., 2010") and apply on top of it a convolutional neural network to classify whether each patch contains a boat. 

To test the SLIC model, download the test images from https://polybox.ethz.ch/index.php/s/0gBtvmVmJyrX8a2 and save the file in data folder. Then download the checkpoints of the CNN training https://polybox.ethz.ch/index.php/s/fUwhqZnAhiy6cgF and save the baseline file in experiments folder. Select the number of images you want to test in `configs/baseline.jsn` (default is 20). 
Then run `python mains/baseline_tester.py -c configs/baseline.jsn`.
It will output each image with the predicted mask.

## Adapted Faster R-CNN
The model is essentially a modified version of the Region Proposal Network of Faster R-CNN. The multiclass classifier of Faster R-CNN is not needed since our problem is a binary task (ship or background), hence we chop that part off our model. The RPN consists mainly on a FCN that generates a feature map of an image, plus another CNN with two branches, one proposing bounding boxes and the other giving probabilities of those boxes containing ships.
#### Training
To train the model from scratch run `python mains/faster_rcnnNoC.py -c configs/faster_rcnn_5layersNorm.json`. We ourselves trained the network on 8 cores GPUs with 20'000MB of memory each. Although we had spare memory, we advise to be prepared for high memory requirements.
To train the model from the latest checkpoint, download the checkpoint from the link below under `experiments/faster_rcnn_5layersNorm/checkpoint` and run the same command as per the model from scratch.
Checkpoints: https://polybox.ethz.ch/index.php/s/kE1YRyRUPmBCGmB.
#### Testing
To test the model, download the test images from the link below and save them under `data\test_sample\`, then run `python mains/faster_rcnnNoC_tester.py -c "configs/faster_rcnn_5layersNorm.json.json`. This will create a file in the predictions directory containing the untreated bounding boxes proposals under a `faster_rcnn_5layersNorm.json` file. For more details, the tester program chooses the boxes with the top 2000 probabilities and then runs NMS on these 2000 boxes with a NMS threshold of 0.2 (both configurable in the faster_rcnnNoC_tester.py file). Running NMS before only taking 2000 boxes would potentially be too slow. 
Finally run `python testing/evaluate.py`. This file takes two json files (a groundtruth and prediction) and calculates mAP and plots a curve.
Test images: https://polybox.ethz.ch/index.php/s/0gBtvmVmJyrX8a2

## Credits
Credits to https://github.com/MrGemy95/Tensorflow-Project-Template for the amazing template!
mAP scoring code adapted from https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734
