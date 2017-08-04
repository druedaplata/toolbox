# Toolbox

## 1. ImageMarker
This tool helps labeling files for object detection in Digits.
Each mark created in an image is transformed to KITTI format labeling.
It supports several labels.

### How to use?
```
./ImageMarkerTool.py input/ output/ labels
```
* 'a' - move left on images list
* 'd' - move right on images list
* '1' - cycle trough labels
* 'p' - disable mark size 
* 's' - find next image without labels
* 'r' - remove last label created
* 'g' - generate kitti labels
* 'q' - close 

### Requirements
* numpy==1.11.0
* cv2==3.2.0
