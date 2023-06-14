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



### Train1-**Changing dataset**

The original dataset of  [dcgan_faces_tutorial](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/5f81194dd43910d586578638f83205a3/dcgan_faces_tutorial.ipynb#scrollTo=qBfeHTsY_NuQ) is the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). I changed the original dataset by  downloading the [Art by Ai - Neural Style Transfer](https://www.kaggle.com/datasets/vbookshelf/art-by-ai-neural-style-transfer) dataset from Kaggle，the reference of importing dataset from kaggle to colab is [here](https://www.analyticsvidhya.com/blog/2021/06/how-to-load-kaggle-datasets-directly-into-google-colab/) .

- code modification
<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/Screenshot%202023-06-14%20at%2011.17.13.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

- training results(num_epoch = 5,batch_size = 128, wokers = 4, nz(latent vector) = )

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/epoch%20%3D%205.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
     
<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/5epoch.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

### Train2-**Changing epoch**

After changing the dataset, I conducted two tests to evaluate the performance of the model with different epoch values, in addition to the original results obtained with 5 epochs. The tests were performed with 40 and 100 epochs respectively.

- test1(num_epoch = 40)

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/epoch%20%3D%2040.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/40epoch.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
     
- test2(num_epoch = 100)

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/epoch%20%3D%20100.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/100epoch.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
     
**Insights:** 

1. Time and Training: Increasing the epoch value results in a longer training time as the model requires more iterations to learn. However, it also provides more opportunities for the model to converge and reach a stable equilibrium.
 
2. Result Quality: I observed that with higher epoch values, the model tends to produce better results. This can be attributed to the increased learning time, allowing the model to capture more nuanced patterns and generate higher-quality outputs.

3. Discriminator Accuracy: More epochs also provide the discriminator with more training iterations, enabling it to improve its accuracy in distinguishing between real and fake images. This improved accuracy puts more pressure on the generator to produce more realistic images.

### Train3-Changing batch_size
The default value of batch_size in this notebook  is 128, I made 2 tests to change the batch_size and comparing the differences of outcomes.

- test1(batch_size = 64)
<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/batch_size%20%3D%2064.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/batch64.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

- test2(batch_size  = 256)
<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/batch_size%20%3D%20256.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/batch256.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

**Insights:** 

The choice of batch size can indeed have a noticeable impact on the outcome performance, particularly in terms of color accuracy.

1. Color Accuracy: When using a batch size of 128, the discriminator showed a better performance in color accuracy. The generated outcomes exhibited more accurate and visually appealing colors compared to lower or higher batch size values. This suggests that a batch size of 128 allows the model to learn and capture the intricate color patterns more effectively.

2. Overfitting and Underfitting: Using a batch size that is too low or too high can lead to a decrease in color accuracy. This might because with a very low batch size, the model may not receive enough diverse samples in each iteration to learn the complex color representations properly. This can result in underfitting, where the model fails to capture the full color range and variations in the dataset. Conversely, an excessively high batch size can lead to overfitting, where the model becomes too specialized to the training data, ignoring subtle color details and generalizing poorly to unseen examples.

### Train4-**Changing number of workers**

The workers parameter in the DataLoader class determines the number of worker threads used for loading the data. It specifies how many subprocesses to use for data loading. I conducted two tests where I varied the workers value, and it appears that changing the workers value did not have a direct impact on the training outcome.

- test1(workers = 8)
<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/workers%20%3D%208.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/8workers.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

- test2(workers = 16)
<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/workers%20%3D%2016.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/16workers.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

**Insights:** 

The primary purpose of using multiple worker threads is to enhance data loading efficiency, particularly when dealing with large datasets. However, the impact on the training outcome may not be immediately apparent in all scenarios. The influence of the workers parameter depends on various factors, including the complexity of the data loading process, hardware configuration, and dataset size.

## Dataset

## Third-party resources

## Clarification


