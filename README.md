# Vision Transformer
Custom implementation of the original [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) vision transformer.

## Overview
In this repository, I have built and annotated a simple, lightweight version of the vision transformer (ViT) from basic PyTorch components. I hope that it will be a useful resource for learning about the basics of this architecture, and will provide a helpful jumping-off point for more complex applications of transformers for computer vision.

### Architecture
I explain the architecture in greater detail my accompanying [blog post](link), but essentially the vision transformer consists of the following components:
1. A simple method for cutting images up into patches and flattening them, and turning them into sequences
2. A concatenated learnable class embedding
3. An added learnable positional embedding
4. A transformer, consisting of a number of layers of encoders (and no decoders)
5. A two-layer MLP for classification

### Data
For this project, I trained versions of the transformer on CIFAR-10 and CIFAR-100. Pytorch Lightning data modules for preparing these datasets are included.

## Environment & Setup
This model was trained with the following packages:
- `pytorch 1.8.2`
- `torchvision 0.9.2`
- `pytorch-lightning 1.6.1`
- `torchmetrics 0.8.0`
- `pl_bolts 0.5.0`

## Repo Structure
### data
Data modules for CIFAR-10 and CIFAR-100. These can be used to download, transform, split and load data into dataloaders.

### vision_transformer/models
- vit_encoder.py - Includes my implementation of the norm-first ViT encoder.
- vit_classifier.py - Includes my implementation of the overall architecture.
- vit_pl_train_module.py - Contains training loop, evaluation methods and other Pytorch Lightning code.

## Usage
### Training
To train this model with CIFAR-100, simply run `python train.py` (edit for different options) or run through the `vit_demo.ipynb` notebook.

## Results
This model was able to get 78.8% accuracy on CIFAR-10 and 49.8% on CIFAR-100. This is far from SOTA, but does demonstrate that the method works for computer vision. Generally, vision transformers on begin to exceed CNN performance when trained on enormous datasets (like JFT-300M), which is difficult for individual practitioners or smaller companies, but it is still quite useful to see how the mechanism works!