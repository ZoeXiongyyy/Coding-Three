# Coding Three Final Project

Yuzhu Xiong 21003975

**FinalProject Colab link:** https://colab.research.google.com/drive/1lIiJ-Zj3uaQI-s3h0VYwr-8qQfj30omG?usp=sharing

## Basic introduction

In this project, I explored the potential for interaction using a [DCGAN.](https://arxiv.org/abs/1511.06434) The model was trained on the [Art by Ai - Neural Style Transfer](https://www.kaggle.com/datasets/vbookshelf/art-by-ai-neural-style-transfer) dataset, and I achieved real-time interaction with the training outcomes.

To accomplish this, I utilized various references and resources, including [Using Interact](https://colab.research.google.com/drive/1CXrsbypB-BZY6J6fvsrUgogBGof5gedN#scrollTo=3noK7P5_9gpv), the [dcgan_faces_tutorial](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/5f81194dd43910d586578638f83205a3/dcgan_faces_tutorial.ipynb#scrollTo=qBfeHTsY_NuQ), [Forms](https://colab.research.google.com/notebooks/forms.ipynb#scrollTo=ig8PIYeLtM8g), and [BabyGAN](https://colab.research.google.com/github/tg-bomze/BabyGAN/blob/master/BabyGAN_(ENG).ipynb). Throughout the process, I received assistance from **ChatGPT**, especiallly in **code rephrasing and debugging**.

The inspiration for this project came from the intriguing work called [Latent Flowers GANden](https://observablehq.com/@stwind/latent-flowers-ganden).

### Video recording

**Note:** All videos have been accelerated due to the slow loading of the code.

**Video of visualizing G’s output on the fixed_noise batch for every epoch**

https://youtube.com/shorts/LMCqDXUKD2I?feature=share

[![Alt Text](https://img.youtube.com/vi/LMCqDXUKD2I/0.jpg)](https://www.youtube.com/watch?v=LMCqDXUKD2I)

**Video of interact with results**

https://youtu.be/mlKNuTo6hj8

[![Alt Text](https://img.youtube.com/vi/mlKNuTo6hj8/0.jpg)](https://www.youtube.com/watch?v=mlKNuTo6hj8)

**Video of interact with epoch**

https://youtu.be/_zgJmDx9LkM

[![Alt Text](https://img.youtube.com/vi/_zgJmDx9LkM/0.jpg)](https://www.youtube.com/watch?v=_zgJmDx9LkM)

**Video of the complete running process**

https://youtu.be/Tikv14jzoU4

[![Alt Text](https://img.youtube.com/vi/Tikv14jzoU4/0.jpg)](https://www.youtube.com/watch?v=Tikv14jzoU4)

-------

## Development process

### Motivation
Taking inspiration from [Latent Flowers GANden](https://observablehq.com/@stwind/latent-flowers-ganden), my aim is to push the boundaries of accessibility and usability for the algorithm. I want to transform it into a tool rather than just a collection of code snippets. Following the approach used in [Latent Flowers GANden](https://observablehq.com/@stwind/latent-flowers-ganden), which utilized [DCGAN](https://arxiv.org/abs/1511.06434)  for real-time interaction with the latent space, I have also chosen DCGAN for my project.

During my research, I compared different resources and tutorials, and I found that the [dcgan_faces_tutorial](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/5f81194dd43910d586578638f83205a3/dcgan_faces_tutorial.ipynb#scrollTo=qBfeHTsY_NuQ) exhibited better performance when training image datasets compared to other [dcgan](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb) tutorials. Hence, I decided to use the [dcgan_faces_tutorial](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/5f81194dd43910d586578638f83205a3/dcgan_faces_tutorial.ipynb#scrollTo=qBfeHTsY_NuQ) as a reference for my project.

**P.S. - Choosing the Machine Learning Model**

Throughout my exploration, I have experimented with several machine learning models and attempted to modify their datasets. However, I encountered challenges with some of them. For instance, the [BabyGAN](https://github.com/tg-bomze/BabyGAN) and [StyleGAN2: projecting images](https://github.com/woctezuma/stylegan2-projecting-images) models were unsuccessful in running properly. Additionally, I faced limitations in changing the dataset for models such as [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [FixNoise](https://github.com/LeeDongYeun/FixNoise).

______

### Train1-**Changing dataset**

The original dataset of  [dcgan_faces_tutorial](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/5f81194dd43910d586578638f83205a3/dcgan_faces_tutorial.ipynb#scrollTo=qBfeHTsY_NuQ) is the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). I changed the original dataset by  downloading the [Art by Ai - Neural Style Transfer](https://www.kaggle.com/datasets/vbookshelf/art-by-ai-neural-style-transfer) dataset from Kaggle，the reference of importing dataset from kaggle to colab is [here](https://www.analyticsvidhya.com/blog/2021/06/how-to-load-kaggle-datasets-directly-into-google-colab/) .

- **code modification**
<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/Screenshot%202023-06-14%20at%2011.17.13.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

- **training results(num_epoch = 5,batch_size = 128, wokers = 4, nz(latent vector) = 100)**

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/epoch%20%3D%205.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
     
<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/5epoch.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

--------     

### Train2-Changing epoch

After changing the dataset, I conducted two tests to evaluate the performance of the model with different epoch values, in addition to the original results obtained with 5 epochs. The tests were performed with 40 and 100 epochs respectively.

- **test1(num_epoch = 40)**

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/epoch%20%3D%2040.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/40epoch.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
     
- **test2(num_epoch = 100)**

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

------------

### Train3-Changing batch_size
The default value of batch_size in this notebook  is 128, I made 2 tests to change the batch_size and comparing the differences of outcomes.

- **test1(batch_size = 64)**
<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/batch_size%20%3D%2064.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/batch64.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

- **test2(batch_size  = 256)**
<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/batch_size%20%3D%20256.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/batch256.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

**Insights:** 

The choice of batch size can indeed have a noticeable impact on the outcome performance, particularly in terms of color accuracy.

1. When using a batch size of 128, the discriminator showed a better performance in color accuracy. The generated outcomes exhibited more accurate and visually appealing colors compared to lower or higher batch size values. This suggests that a batch size of 128 allows the model to learn and capture the intricate color patterns more effectively.

2. Using a batch size that is too low or too high can lead to a decrease in color accuracy. This might because with a very low batch size, the model may not receive enough diverse samples in each iteration to learn the complex color representations properly. Conversely, an excessively high batch size can lead to the model become too specialized to the training data, ignoring subtle color details and generalizing poorly to unseen examples.

---------------

### Train4-Changing number of workers

The workers parameter in the DataLoader class determines the number of worker threads used for loading the data. It specifies how many subprocesses to use for data loading. I conducted two tests where I varied the workers value, and it appears that changing the workers value did not have a direct impact on the training outcome.

- **test1(workers = 8)**
<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/workers%20%3D%208.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/8workers.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

- **test2(workers = 16)**
<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/workers%20%3D%2016.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/16workers.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

**Insights:** 

The primary purpose of using multiple worker threads is to enhance data loading efficiency, particularly when dealing with large datasets. However, the impact on the training outcome may not be immediately apparent in all scenarios. The influence of the workers parameter depends on various factors, including the complexity of the data loading process, hardware configuration, and dataset size.

-----------

### Train5-Changing latent vector

the latent vector in the original notebook is called nz, and the default value of nz is 100， and I made 2 changes(nz = 200, nz = 50) to observe the outcome.

- **test1(nz = 200)**
<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/nz%20%3D%20200.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/200nz.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

- **test2(nz= 50)**

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/nz%3D50.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

<img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/50nz.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

**Insights:** 

I observed that using a latent vector size of 100 leads to better training outcomes compared to smaller (e.g., 50) or larger (e.g., 200) sizes. I think this might because of :

1. The generator and discriminator architectures in the tutorial are carefully designed and tuned based on empirical observations and best practices. These architectures are optimized for a latent vector size of 100. When using a smaller or larger latent vector size, the model may not perform optimally because the architecture and hyperparameters are not specifically tailored to those sizes.

2. The latent vector size should ideally match the complexity and dimensionality of the underlying data distribution. If the latent vector size is too small (e.g., 50), it may not provide enough capacity for the generator to learn the intricate details of the data distribution.  On the other hand, if the latent vector size is too large (e.g., 200), it introduces unnecessary complexity and can make the training process more challenging without providing significant improvements in the generated samples.

------------

### Train6 & Final-Interactive possibilities experiements

In the Train6 and Final experiments, I aimed to make the model more interactive by referencing various resources such as [Using Interact](https://colab.research.google.com/drive/1CXrsbypB-BZY6J6fvsrUgogBGof5gedN#scrollTo=3noK7P5_9gpv), [Forms](https://colab.research.google.com/notebooks/forms.ipynb#scrollTo=ig8PIYeLtM8g), and [BabyGAN](https://colab.research.google.com/github/tg-bomze/BabyGAN/blob/master/BabyGAN_(ENG).ipynb).

I explored two different approaches to enable interaction with the code. The first approach involved using markdown language to create interactive user interfaces. However, this method only allowed for changing the input values in the original code and after the code was run, those elements could not directly interact with the running results.

- **interactive UI written by markdown**

    <img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/ui02.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

    <img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/ui01.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

    <img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/ui03.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

The second method involved importing ipywidgets, which allowed me to create UI elements such as buttons and sliders that could directly interact with the running code and manipulate the results. This approach provided more dynamic and real-time interaction with the model, allowing users to input information, open or hide images, and even filter the result process.

- **interaction created with the import of ipywidgets**

    <img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/widget01.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

    <img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/widget02.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

    <img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/widget03.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

**Insights:** 

Enabling code and result interaction through UI can greatly enhance the usability and accessibility of the model for a wider audience. However, there are still several limitations.

When the epoch value is set too high or the number of frames in the animation of *visualizing G’s output on the fixed_noise batch for every epoch* is too large, the process can become problematic. It may lead to the running process becoming collapsed or unstable.
- **collaspe**
  
  <img src="https://github.com/ZoeXiongyyy/Coding-Three/blob/main/FinalProject/Video%26Pic/collapse.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

One possible reason for this issue is setting the animation.embed_limit too small (e.g., plt.rcParams['animation.embed_limit'] = 30). But when adjusting the value of animation.embed_limit, the program may still experience disconnections or errors , I think this may related the collab platform itself. 

Moreover, if the size of the animation exceeds the imposed limitation, it can result in dropped frames and incomplete animations. In some cases, it may even cause the program to disconnect with the host on colab entirely.

Additionally, the running time of experiments can be prolonged, especially when loading animations and images.

----------

## Dataset
Art by Ai - Neural Style Transfer

https://www.kaggle.com/datasets/vbookshelf/art-by-ai-neural-style-transfer

## Original code resource
dcgan_faces_tutoria

https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/5f81194dd43910d586578638f83205a3/dcgan_faces_tutorial.ipynb#scrollTo=qBfeHTsY_NuQ

------------

## Third-party resources

### Platform
Google Colab

https://colab.research.google.com/?utm_source=scs-index

### Coding assistance
ChatGpt

https://openai.com/blog/chatgpt

### Reference
https://colab.research.google.com/github/tg-bomze/BabyGAN/blob/master/BabyGAN_(ENG).ipynb#scrollTo=iBZJPkI5Yz0v

https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb#scrollTo=NFC2ghIdiZYE

https://observablehq.com/@stwind/latent-flowers-ganden

https://colab.research.google.com/github/tg-bomze/BabyGAN/blob/master/BabyGAN_(ENG).ipynb#scrollTo=iBZJPkI5Yz0v

http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/5f81194dd43910d586578638f83205a3/dcgan_faces_tutorial.ipynb#scrollTo=qBfeHTsY_NuQ

https://www.analyticsvidhya.com/blog/2021/06/how-to-load-kaggle-datasets-directly-into-google-colab/
https://github.com/woctezuma/stylegan2-projecting-images

