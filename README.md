# Yolo3

2017 witnessed some real fight for the best Object Detection model with RetinaNet (another one-stage detector), Faster RCNN with FPN with ResNext as the backbone and Mask RCNN with ResNext backbone and then RetinaNet with the ResNext backbone topping the charts with an MAP of 61 on COCO dataset for 0.5 IOU. RetinaNet being a one-stage detector was faster than the rest. With no new version of YOLO in 2017, 2018 came with best RetinaNet(the one I mentioned above) and then now YOLO V3!. The paper is written by again, Joseph Redmon and Ali Farhad and named YOLOv3: An Incremental Improvement. This brought the fast YOLOv2 at par with best accuracies. YOLOv3 gives a MAP of 57.9 on COCO dataset for IOU 0.5. For comparisons just refer the table below:


Mean Average Precision Comparisons 2018! (Source : YOLOv3 paper)
Now you can observe, 57.9 is at par with all the two stage detectors. YOLO608(best YOLO with high dimensional input images) is still almost 4x times faster than best RetinaNet and 2x faster than second best RetinaNet. The YOLO320 has same accuracy as the RetinaNet with ResNet50 backbone being 4x times faster. This makes YOLOv3 clearly very efficient for any general object detection use-case.

What changed ? What are the so called Incremental Improvements?

Bounding Box Predictions : YOLOv3 just like YOLOv2 uses dimension clusters to generate Anchor Boxes. Now as YOLOv3 is a single network the loss for objectiveness and classification needs to be calculated separately but from the same network. YOLOv3 predicts the objectiveness score using logistic regression where 1 means complete overlap of bounding box prior over the ground truth object. It will predict only 1 bonding box prior for one ground truth object( unlike Faster RCNN) and any error in this would incur for both classification as well as detection (objectiveness) loss. There would also be other bounding box priors which would have objectiveness score more than the threshold but less than the best one, for these error will only incur for the detection loss and not for the classification loss.
Class Predictions : YOLOv3 uses independent logistic classifiers for each class instead of a regular softmax layer. This is done to make the classification multi-label classification. What it means and how it adds value? Take an example, where a woman is shown in the picture and the model is trained on both person and woman, having a softmax here will lead to the class probabilities been divided between these 2 classes with say 0.4 and 0.45 probabilities. But independent classifiers solves this issue and gives a yes vs no probability for each class, like what’s the probability that there is a woman in the picture would give 0.8 and what’s the probability that there is a person in the picture would give 0.9 and we can label the object as both person and woman.

# source: 

- https://github.com/arunponnusamy/object-detection-opencv

- https://www.arunponnusamy.com/yolo-object-detection-opencv-python.html
        
# (for Videos )we will nead the below library to read from youtube link

- pip install --upgrade youtube-dl

- pip install pafy

tutorial: https://www.youtube.com/watch?v=w0tDDFip7KM
