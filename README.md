# Toolbox

## 1. ImageMarker
This tool assists in labeling images for object detection and segmentation.

Formats supported for Detection:
* KITTI - Used in Nvidia Digits
* VOC   - *Not implemented yet*

Formats supported for Segmentation:
* Grayscale Image Labels
* RGB Image Labels - *Not implemented yet*

### How to use?
```bash
python ImageMarkerTool.py

# Arguments
--input, -in                # Path to images folder.
--output, -out              # Path where labels will be saved.
--labels, -l                # List of objects to label, MUST BE IN ALPHABETICAL ORDER.
--mode, -m                  # Labeling mode used.
                            #   - detection (default)
                            #   - segmentation.
--label_format, -lf         # Optional argument for detection labeling.
                            #   - kitti (default)
                            #   - voc
```


### Requirements

* Python 3.6+
* Opencv 3.0
* Numpy 1.11+
