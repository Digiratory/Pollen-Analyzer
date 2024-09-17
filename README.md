# Pollen-Analyzer


# Instance Segmentation

The following segment represents the instance-segmentaion part of the project. You can reproduce our results by following the steps presented below.

## Prerequisites

The code is based on the following modules and frameworks:

- Python 3.9
- Core: PyTorch, OpenCV, TorchVison
- Functional: NumPy, random, PIL, datumaro, etc.

You can install all the dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Dataset

Also you will need to acquire our dataset to reproduce model training. You can make it with  the following command or by follow the disk link.

```bash
wget "link"
```

## Modules directory structure and brief description

### train_model.py

The module's aid is to configure and train the model, as well as saving the metrics, sample images during the way.

- noise_dir - source path to Noise(background) images
- pollen_dir - path to a directory where pollen samples are held
- PATH_TO_MODEL - path, where trained model weights will be saved
- image_save_dir - path, where processed generated images will be saving during the training phase
- log_dir - path to a directory where training metrics will be saved

### test_model.py

This module is creating the test dataset and processes it's evaluation. The given results is the IoU metrics: bbox (bounding box), segm (masks)

- test_root - path to CVAT dataset, wich helds the test images and mask polygons for model evaluation
- model_root - path to model weights

### image_from_mask.py

This module is inference mode, performing the model's task: instance segmentation (extracting the objects for the passed images).

- path_to_model - path to model weights
- input_dir - path to directory, where target images are held (microscopic images, where the object segmetation is needed)
- output_dir - path to a directory, where finded objects are held, the fromat looks like: sample (1).jpg -> sample(1)_0.jpg, sample(1)_1.jpg...

The following values defines the confidence level, when the model should refer the ecnountered object as a target object (which should be returned)

- score_threshold - (float) overall model's score threshold
- mask_threshold - (float) mask confidence threshold






