# Thyroid Nodule Detection with AFP
This is the implementation code of the work titled _"Self-supervised Enhanced Thyroid Nodule Detection in Ultrasound Examination Video Sequences with Multi-Perspective Evaluation"_

## Three parts of this work
- Self-supervised pre-training model based on [SOLO]
- Ultrasound viodeo object detection model with adjacent frame perception (AFP) implemented by [MMTracking] toolbox
- The multi-perspective evaluation of thyrpid nodule detection

## Features
- Extend the thyroid nodule detection from 2D image to video sequence
- Enhancement of the self-supervied model trained on the unlabeled ultrasound images
- Evluations from nodule localization, frame image, video sequence, nodule instance perspectives    

## Contents
This repository does not contain code from the base libraries of [SOLO] and [MMTracking], and the following contents are included:
- Patch shuffle (PS) augmentation for ultrsound video
- The RoI head and BBox heads in for the Implementation of AFP
- Configs for training APF and PASS
- Data  pre- and post-processing and pipeline for multi-perspective evaluation

## Prerequisites
- Python 3.6+
- PyTorch 1.10.0
- MMCV Full 1.4.2
- MMDetection 2.23.0
- MMTracking 0.12.0
- Numpy 1.24.4
- CUDA 11.1
- GCC 5+

## Getting Started
You need to copy the model implementation and configuration files to the correct folders (described by the comments in the code file) in [SOLO] and [MMTracking] respectively, and [MMTracking] needs to be recompiled.

The code for data processing and model evaluation is independent, and each function corresponds to a complete pipeline.

## License

MIT


[//]:
   [SOLO]: <https://github.com/vturrisi/solo-learn>
   [MMTracking]: <https://github.com/open-mmlab/mmtracking>
