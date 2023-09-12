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


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
   [SOLO]: <https://github.com/vturrisi/solo-learn>
   [MMTracking]: <https://github.com/open-mmlab/mmtracking>
   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
