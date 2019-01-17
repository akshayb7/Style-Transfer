# Style-Transfer

## Introduction

This repository contains a PyTorch-based implementation of "A Neural Algorithm of Artistic Style" by L. Gatys, A. Ecker, and M. Bethge, which presents a method for transferring the artistic style of one input image onto another. You can read the paper here: http://arxiv.org/abs/1508.06576. 

## Requirements

 - Python 3.0 or above
 - PyTorch 1.0.0 (may work with older versions)
 - CUDA (recommended)

CUDA will enable GPU-based computations.

## How to use?

All of the necessary codes are available in the different helper functions for ease of understanding and then combined in the `style transfer.ipynb` notebook. The different helper functions are in image_helpers.py and layer_helpers.py.

A high-level implementation is also done in the `Style Transfer (Automatized + Visual Examples).ipynb` notebook, with most of the code abstracted away in the automation.py file.

The user may use either of the two notebooks.

## Sample

Some sample implementations:


![](Images/results/2.JPG?raw=true)
![](Images/results/3.JPG?raw=true)
![](Images/results/4.JPG?raw=true)