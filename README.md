# Video Imagination from a Single Image with Transformation Generation
This repository contains an implementation of Video Imagination from a Single Image with Transformation Generation. The framework can synthesize multiple imaginary video from a single image.

## Imaginary Video Example
We randomly pick some imaginary videos synthesized by our framework. The input is a single image from UCF101 dataset, and the output imaginary video contains five frames.
The following gif picture is a demo of synthesized imaginary video. The network may bring some delay, please wait a while fro the demonstration.
> Imaginary Video 
> 
> ![gif](https://github.com/gitpub327/VideoImagination/blob/master/Imaginary_Video.gif)

> Input image
> 
>![im](https://github.com/gitpub327/VideoImagination/blob/master/Example_of_Input_Image.png)


## Data
The framework can be trained on three datasets : moving MNIST, 2D shape, UCF101. No pre-process is needed except normalizing images to be in the range [0, 1].
The videos (or image tuples) needs to be convert to tfrecords at first. 
## Training
The code requires a TensorFlowr r1.0 installation

To train the framework, after you prepare the tfrecords, run main.py. This file will build model and graph, and train the networks.

## Notes
The code is modified based on [A Tensorflow Implementation of DCGAN](https://github.com/bamos/dcgan-completion.tensorflow).  The on-the-fly 2D shape dataset generating codes are modified from [the author of the dataset](https://github.com/tensorflow/models/tree/master/next_frame_prediction).

