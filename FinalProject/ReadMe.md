# Coding Three Final Project -Experimenting with Interactive Possibilities of DCGAN

Yuzhu Xiong

## Basic introduction

In this project, I explored the potential for interaction using a [DCGAN.](https://arxiv.org/abs/1511.06434) The model was trained on the [Art by Ai - Neural Style Transfer](https://www.kaggle.com/datasets/vbookshelf/art-by-ai-neural-style-transfer) dataset, and I achieved real-time interaction with the training outcomes.

To accomplish this, I utilized various references and resources, including [Using Interact](https://colab.research.google.com/drive/1CXrsbypB-BZY6J6fvsrUgogBGof5gedN#scrollTo=3noK7P5_9gpv), the [dcgan_faces_tutorial](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/5f81194dd43910d586578638f83205a3/dcgan_faces_tutorial.ipynb#scrollTo=qBfeHTsY_NuQ), [Forms](https://colab.research.google.com/notebooks/forms.ipynb#scrollTo=ig8PIYeLtM8g), and [BabyGAN](https://colab.research.google.com/github/tg-bomze/BabyGAN/blob/master/BabyGAN_(ENG).ipynb). Throughout the process, I received assistance from ChatGPT for code rephrasing and debugging.

The inspiration for this project came from the intriguing work called [Latent Flowers GANden](https://observablehq.com/@stwind/latent-flowers-ganden).

**Video of visualizing G’s output on the fixed_noise batch for every epoch**

**Video of interact with results**

**Video of interact with epoch**

**Video of the complete running process**

## Development process

### Motivation
Taking inspiration from [Latent Flowers GANden](https://observablehq.com/@stwind/latent-flowers-ganden), my aim is to push the boundaries of accessibility and usability for the algorithm. I want to transform it into a tool rather than just a collection of code snippets. Following the approach used in [Latent Flowers GANden](https://observablehq.com/@stwind/latent-flowers-ganden), which utilized [DCGAN](https://arxiv.org/abs/1511.06434)  for real-time interaction with the latent space, I have also chosen DCGAN for my project.

During my research, I compared different resources and tutorials, and I found that the [dcgan_faces_tutorial](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/5f81194dd43910d586578638f83205a3/dcgan_faces_tutorial.ipynb#scrollTo=qBfeHTsY_NuQ) exhibited better performance when training image datasets compared to other [dcgan](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb) tutorials. Hence, I decided to use the [dcgan_faces_tutorial](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/5f81194dd43910d586578638f83205a3/dcgan_faces_tutorial.ipynb#scrollTo=qBfeHTsY_NuQ) as a reference for my project.

**P.S. - Choosing the Machine Learning Model**
Throughout my exploration, I have experimented with several machine learning models and attempted to modify their datasets. However, I encountered challenges with some of them. For instance, the [BabyGAN](https://github.com/tg-bomze/BabyGAN) and [Fix the Noise: Disentangling Source Feature for Controllable Domain Translation](https://github.com/LeeDongYeun/FixNoise) models were unsuccessful in running properly. Additionally, I faced limitations in changing the dataset for models such as [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [FixNoise](https://github.com/LeeDongYeun/FixNoise).

**Record of Additional Failed Experiments**



### Changing dataset


### Input experiments

### Interaction possibilities tests



## Dataset

## Third-party resources

## Clarification


