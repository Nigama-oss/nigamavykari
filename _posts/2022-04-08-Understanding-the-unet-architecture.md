---
layout: post
title: "U-Net Architecture For Semantic Image Segmentation"
categories: ml
author:
- Nigama Vykari 
---
The U-Net architecture was one of the most influential papers released in 2015. Although there are advanced algorithms that may perform better at segmentation, U-Net is still very popular since it can achieve good results on limited training samples. 

This network was trained end-to-end, which was said to have outperformed the previous best method at the time (a sliding window CNN). With U-Net, the input image first goes through a contraction path, where the input size is downsampled, and then it has an expansive path, where the image is upsampled. In between the contraction and expansive paths, it has *skip connections*. 

Before trying to understand further, I higly recommend checking out how [Convolutional Neural Networks work](http://127.0.0.1:4000/ml/2020/03/07/Convolutional-neural-networks.html).

### Understanding U-Net Architecture

U-Net architecture was developed to not only understand what the image was, but also to get the location of the object and indentify it's area. Since it was developed keeping medical images in mind, we can say that it can perform well even with some unlabelled/unstructured data. As quoted in the paper, there are three important parts that we need to discuss to understand the network completely.

> The architecture consists of a contracting path to capture context and a symmetric expanding that enables precise localization.

![](/assets/images/u_net.png)

You can think of semantic segmentation as a variation of image classification. But the network only has convolutional layers and doesn't have any fully connected or dense layers. The image shown below is from the original paper [U-Net Architecture for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf). Lets understand how the architecture works and also implement it from scratch. 

#### The Encoder

The *contracting path* in the above quote refers to the encoder block. As you can see from the animation below, the architecture is built in levels. Each level consists of two $$3x3$$ convolutional layers, each followed by a ReLU activation unit.  

The transition between these layers is handled by $$2x2$$ max pooling units (represented by a red downward arrow) with a stride of 2 for downsampling. Maxpooling is done to reduce the size of the input to the next level. **As we go down the levels of u-shape, the size of our input halves, but the number of channels doubles.** This way, the network is able to learn more complex relationships in the image data. By the time we reach the very bottom, the network knows the 'what' of the image fairly well. 

> Encoding in a nutshell is - condensing the information with each layer and widening the receptive field. 

![](/assets/gifs/encoder.gif) 

Observe that the steps of each level are repeating the same process by only changing the number of filters and input size. This means that we can define a function to perform this repetetive task. Since each of these levels is a convolution unit, let's define a `conv_block` for the convolutions at each stage and an `encoder_block` which includes max-pooling along with the `conv_block`.

```python
# Convolution block
def conv_block(input, num_filters):
	x = Conv2D(num_filters, 3, padding="same")(input)
	x = Activation("relu")
	x = BatchNormalization()(x)

	x = Conv2D(num_filters, 3, padding="same")(input)
	x = Activation("relu")
	x = BatchNormalization()(x)

	return x

```

```python
# Encoder block
def encoder_block(input, num_filters):
	x = conv_block(input, num_filters)
	p = MaxPool2D((2,2))(x)
	return x, p
```

At this point for a normal classification problem, we usually add some dense layers and predict the output class. But for semantic segmentation, this is not good enough. We also need to know the 'where' of the image and its area. This is done by a decoder block or an expansive process (the path moving up the U-shape).

#### The Decoder and Skip Connections

The decoder part of the network is the same architecture with slight variation. This variation is from localization of *skip connections*. This means that the feature maps of the encoder are concatenated (with the help of skip connections) to the output of the transposed convolutions of the same level. Since we already learnt the feature mapping in the encoding process, we might as well use them to provide more informatiom while decoding.

![](/assets/gifs/decoder.gif)

Since our goal is to get an output that is similar in size to the output, we use 2x2 transposed convolutions (also known as deconvolutions) to upsample the condensed representation of the image that we now have at the bottom of our U-Net shape back to its input size.

Therefore instead of max pooling between the levels, there are two inputs at each level - One from the transposed convolutions from the encoder, and the second output from the previous level. Let's create a function called `decoder_block` that defines this process.

```python
def decoder_block(input, num_filters):
	x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
	x = Concatenate() ([x, skip_connections])
	x = conv_block(x, num_filters)
	return x

```

Note that in the decoder part, the procedure of the encoder is basically reversed. Here, **as we upsample the data, the image gets bigger and the number of channels halves.**

Now, we can build the U-Net by creating a separate function called `build_unet` with all it's building blocks. 

```python
def build_unet(input_shape):
	inputs = Input(input_shape)

	s1, p1 = encoder_block(inputs, 64)
	s2, p2 = encoder_block(p1, 128)
	s3, p3 = encoder_block(p2, 256)
	s4, p4 = encoder_block(p3, 512)

	b1 = conv_block(p4, 1024)

	d1 = decoder_block(b1, s4, 512)
	d2 = decoder_block(d1, s3, 256)
	d3 = decoder_block(d2, s2, 128)
	d4 = decoder_block(d3, s1, 64)

	outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
	model = Model(inputs, outpust, name="U-Net")
	return model
```

In the paper, the authors used the unpadded convolutions, and because of this, the output came out to be much smaller than the input. But, it might be a good idea to use padding to keep the original image size. 

Similar to many visual tasks, and especially in biomedical image processing, object localization and mapping is required in the output. Meaning, class labels should be assigned to each pixel as shown in the picture.

![](/assets/images/semantic.jpg)

This method was developed by *Ciresan et al* who trained network using sliding window setup to predict class label of each pixel.

Although the Sliding Window technique has some benefits like localization, and much larger training data for patches, it did have some flaws. One of them was the method being very slow.

We could also just input and image, then create the same 3x3 convolutions and continue doing it until we get some output.

But this cannot work for two reasons - 
1. we're not building a very large receptive field, and 
2. It's very expensive.

#### Getting Results

After constructing our model, we need to provide it with proper training data, inputs, and the target variabes, so it can learn where the objects are located in the inputs that we present. The original paper used cross entropy loss function to train the U-Net and soft marks as an activation function for the final convolution layer.

In single class segmentation, we usually need to provide an image of black and white segmentation mask to mark the area of the onjects in the original image. As a target in multi class segmentation, we usually need to use different colours to mask the different objects. But since ML models also need numbers for calculations, we also need to load and convert the image to a 3D volume (a multi dimensional array). Thankfully, [tensorflow](https://www.tensorflow.org/tutorials/images/segmentation) provides all these methods we need to get our results. 


