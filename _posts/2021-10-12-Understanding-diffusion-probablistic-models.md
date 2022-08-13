---
layout: post
title: "Understanding diffusion probablistic models"
categories: diffusion-models
author:
- Nigama Vykari 
meta: "Springfield"
---

A generative model is a mathematical model that can generate data from random variables. Essentially, they are models that can generate data based on some set of rules.

They are capable of generating data points from the training set, and these points are not part of the training set and have not been seen by the model before.The main goal of these models is to create new, meaningful and relevant content for a given topic or niche.

This blog will discuss about an interesting [research paper](https://arxiv.org/abs/2102.09672) that describe the performance of **Denoising Diffusion Probablistic Models** and how they are used in image synthesis.

#### **Forward Noising Process**

The idea behind forward noising process involves adding extra bit of noise ($$ \epsilon $$) in each step sampled from a standard distribution ($$ N $$) and we eventually end up with only random noise after thousands of steps. 

If we repeat this for infinitely many steps, it leads to more random noise and we don't have any information about the original image anymore.

![](/assets/images/fnp.png)

We have now defined a process where we take the image from a data space to a known distribution, which is the normal distribution ($$ N $$).

The logic behind this technique is to invert the noisy process and identify the image it came from. We do not give any information about the noise except for it's distribution. The question now is - Can we create a function that learns this reverse process?

In mathematical terms, lets say we need to create a function $$ \phi $$ where it takes $$ x $$ image as an input at time $$ t=50 $$. What is the image at $$ t = 49 $$ ? $$ \phi (x, t=50), (t=49, x=?) $$

**Note :** Generating training data for this neural network will be easy because we just take the data and run them through the noise process. This creates plenty of training data for each step. We also don't define a new $$ \phi $$ function at every step since it can take time as an input referring to specific image.

In a given data distribution of $$ q(x_{0}) $$, we define a sample data of $$ x_{0} $$ through a forward noising process $$ q $$. This produces noisy images $$ x_{1} $$ through $$ x_{T} $$ by adding Gaussian Noise at time $$ t $$ with some variance ( $$ \beta_{t} \in (0, 1) $$ ). 

The distribution of the things produced via this noise, given that we start with the data sample $$ x_{0} $$, we simply define it as product of distributions. 

$$ q(x_{1},....,x_{T} | x_{0}) := \prod_{t=1}^{T} q(x_{t} | x_{t-1}) $$

The distribution of the next sample in these steps is going to be a normal distribution with a mean and variance.

$$ q(x_{t} | x_{t-1}) := N(x_{t} ; \sqrt{1-\beta_{t}} x_{t-1}, \beta_{t}I) $$

The assumption here is that we use noise that has diagonal co-variance matrix ($$ \beta_{t}I $$) and the gaussian noise is centered at the last sample ($$ x_{t-1} $$) but down-sampled by $$ \sqrt{1-\beta_{t}} $$.

According to the original paper:

> Given sufficiently large $$T$$ and a well behaved schedule of $$ \beta_{t} $$, the latent $$x_{T}$$ (the very last step) is nearly an isotropic Gaussian distribution.

It is important to remember that to implement the inverse process, the model needs an understanding of the entire data distribution. Since its not possible take our entire data into account, we approximate one of these steps with a neural network as follows:

$$ p_{\theta}(x_{t-1}|x_{t}) := N(x_{t-1};\mu_{\theta}(x_{t},t), \sum_{\theta}(x_{t},t)) $$

The function takes the noised version as an input ($$ \mu_{\theta}(x_{t},t) $$) and outputs distribution over images where this could have originated from. The neural network produces the mean ($$ \mu_{\theta} $$) and the co-variance variance matrix ($$\sum_{\theta}$$), given the image ($$x_{t}$$).

**Note :** It is a strong assumption that the data would be from a Gaussian distribution. This only applies because we are taking small steps and may not hold true if we take large enough steps. 

A variational lower bound can be written to check if the image taken from the data matches our output or if they are close together. 

$$ L_{t-1} := D_{KL}(q(x_{t-1}|x_{t},x_{0}) || p_{\theta}(x_{t-1}|x_{t})) $$

#### **Training In Practice**

Besides predicting the original image, the neural network could also tell us the noise that has been added. The authors of the paper Improved Denoising Diffusion Probabilistic Models state that modelling the noise is the best approach for trining so far. So they define a new loss function -

$$ L_{simple} = E_{t, x_{0}, \epsilon}[|| \epsilon - \epsilon_{\theta} (x_{t},t) ||^{2}] $$

The new loss function simply estimates the noise output from the neural network to approximately match the added noise. 

#### **Learing The Co-Variance Matrix**

In order to predict the preceeding distribution of a noisy image, the authors say that there are two things that need to be modelled. 

1. Learning the Co-Variance Matrix
2. Modelling the noise instead of the mean. 

If we want to fix the co-variance matrix, it is necessary to know at what scale you need to fix it at. This is entirely dependent on the noise applied in the forward process. 

After adding the noise, we can calculate the avaerage co-variance of the reverse step should be at that particular time. Infact, we can derive the upper and lower bounds.

$$ \therefore $$ If $$ \beta $$ is the schedule for the noise, then : $$ \sigma_{t}^{2} = \beta_{t} $$ or $$ \sigma_{t}^{2} = \beta_{t}^{-}$$ where  $$ \beta_{t} $$ is the noise scale and $$ \beta_{t}^{-} $$ is the accumulated noise until that step. 

If we plot a graph for the ratio of upper and lower bounds as a function of the diffusion step, we can observe an immediate clamp at $$ 1 $$ (especially for large amt. of steps) as shown in the figure. 

![](/assets/images/dsr.png) 

This means that there is little difference between the upper and lower bounds and one of them is enough. However, if a neural network were to learn any number/numbers (say from 1 to 1000), and if the real answer lies anywhere between two number with a small difference, it would have a hard time predicting these values. 

To solve this, the authors re-paramatrize how they predict the co-variance matrix. They do this by learning an interpolation parameter ($$ v $$) to interpolate between the upper and lower bound. This technique turned out to be great since the neural network can now predict a number $$ v $$, for each dimension which is only between $$ 0 $$ and $$ 1 $$.

#### **Final Thoughts**

An issue that variational auto encoders had for a long time was that the images were blurry, and recent developments in DDP models can fix these kind of issues.

Recently, diffusion models have also been reported to acheive competetive results compared to GANs with high quality in image data. 

The final results show that DDPMs beat some of the GAN models on a lot of tasks. It is quite possible that these models will go beyond GANs as we continue to improve their architecture.

