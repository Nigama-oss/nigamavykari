---
layout: post
title: "Machine Learning in fluid dynamics"
categories: machine-learning
author:
- Nigama Vykari
---

I recently came across a [video](https://www.youtube.com/watch?v=8e3OT2K99Kw&list=PLMrJAkhIeNNQWO3ESiccZmPssvUDFHL4M&index=6) that explained about a research paper on [Machine Learning In Fluid Dynamics](https://www.youtube.com/redirect?q=https%3A%2F%2Fwww.annualreviews.org%2Fdoi%2Fpdf%2F10.1146%2Fannurev-fluid-010719-060214&redir_token=QUFFLUhqa1VTSVVidXE3Qlk4cEJZdmFXc2o2N2NpRjRmZ3xBQ3Jtc0tsa1YtcFhlQW00Z3FWOHVETXhMeWxOMWRkQi1MM3huSVdxZ0EyMC1FX2NPVDlsbFZOcmp6ZkFibURsOFhOLUExZ19jOXdtakszbWFYRzJkSFBjN1UyMTM3dTBVZWVGTElkVzg2YVZ6dEFXUmdBM3FHVQ%3D%3D&v=3fOXIbycAmc&event=video_description). I found the concept to be quite intriguing and I wanted to share this knowledge.

There are incredible algorithms in ML on image recognition, and object detection, and apparently we could apply those techniques in fluid mechanics if we imagine the *'fluid'* or the *'flow field'* as a picture. And the reason these two fields go together is because, a lot of tasks performed in fluid mechanics are the fundamental concepts of machine learning.

**Note:** Fluids are substances that has no fixed shape and yields easily to external pressure. It might be gases or liquids. Flow field is the distribution of density & velocity of a fluid over space and time. Almost every trillion dollar industry like medical & transportation, involves working with fluids.

> The question is how much of these advancements in ML can be actually converted into physical sciences & engineering?

#### Patterns Exist

Patterns exist everywhere. Patterns exist both in machine learning & fluid mechanics. When we assume our flow fields as images, every little detail does not matter. What matters is the dominant patterns which defines that particular flow field or the image. Interestingly enough, a lot of machine learning ideas of image recognition were from fluid mechanics.

For example, below is an image of cloud flow over Rishiri Islands in Japan, which resemble a simple structure that we can control and simulate. To learn more about extracting patterns, [checkout this paper](https://arc.aiaa.org/doi/10.2514/1.J056060). 

![](/assets/images/rishiri.png)

**This very existence of patterns in fluids is the entry point of machine learning into fluid mechanics.**

The concept of extracting patterns from images was first introduced by *Sirovich* in 1987 in his paper [Eigenfaces](https://www.face-rec.org/interesting-papers/General/ld.pdf). In this paper, he experimented by taking a huge dataset of human face images and took the defining or dominant components of that library to get the so called *'eigenfaces'*. Basically, he was able take out the most common features of a human face from that dataset and extract those patterns. **He applied the same concept to the flow field images in the same year.** One of the go to algorithms developed later in fluid mechanics to extract patterns is POD or [Proper Orthogonal Decomposition](https://www.youtube.com/playlist?list=PLnMuXMcchFUQnpTWYx6byi-vYNgIY9vZp).

> Therefore we have to keep in mind - patterns exist, we can extract those patterns and get a much simpler representations of our flow.

This idea that there exists lower dimensional patterns in fluids, allows us to perform very powerful things such as taking algorithms directly from image processing and applying them to flow field. It can also be used to perform tasks that need high computation power like closure models, which have very complex flow data.

The goal of this post was not to explain how all of these concepts work, but to introduce you to the idea. I hope you find the resources in this post useful & explore more in this field.