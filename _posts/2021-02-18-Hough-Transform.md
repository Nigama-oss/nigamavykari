---
layout: post
title: "Understanding Hough Transform Technique"
categories: machine-learning
author:
- Nigama Vykari 
meta: "Springfield"
---

*Originally posted on [Paperspace Blog.](https://blog.paperspace.com/understanding-hough-transform-lane-detection/)*

The hough transform technique is an amazing tool that can be used for locating shapes in images. It is often used to detect circles, ellipses, and lines to get the exact location or geometrical understanding of the image. This ability of the Hough transform to identify shapes makes it an ideal tool for detecting lane lines for a self-driving car.

Lane detection is one of the basic concepts when trying to understand how self-driving cars work. In this article, we will build a program that can identify lane lines in a picture or a video and learn how the Hough transform plays a huge role in achieving this task. Hough transform comes almost as the last step in detecting straight lines in the region of interest. 

#### **Hough Transform**

As mentioned above, hough transform is applied as a last step on images which has already gone through canny edge detection. The picture you now have is just a series of pixels and we cannot find a geometrical representation directly to know the slope and intercepts.

Since images are never perfect, we cannot loop through the pixels to find the slope and intercept since it would be a very difficult task. This is where hough transform can be used. It helps us to figure out the prominent lines and connect the disjoint edge points in an image.

Let's understand this concept by comparing normal X-Y co-ordinate space and hough space (M-C space).

![](/assets/images/mcxy.png)

A point in an XY plane can have any number of lines passing through it. The same is true if we take a line in the same XY plane, many points are passing through that line.

To identify the lines in our picture, we have to imagine each edge pixel as a point in a co-ordinate space and we are going to convert that point into a line in the hough space.

![](/assets/images/xytomc.png)

We need to find two or more lines representing their corresponding points(on XY-plane) that intersect in the hough space to detect the lines. This way we know that the two points belong to the same line.

The idea of finding possible lines from a series of points is how we can find the lines in our gradient image. But, the model also needs parameters of the lines to identify them.

To get these parameters, we first divide the hough space into a grid containing small squares as shown in the figure. The values of c and m corresponding to the square with the most number of intersections will be used to draw the best fit line.

![](/assets/images/grid.png)

This approach works well with lines that have slopes. But, if we are dealing with a straight line, the slope will always be infinity and we cannot exactly work with that value. So, we make a small change in our approach.

Instead of representing our line equation as $$y = mx + c$$, we express it in the polar co-ordinate system i.e; $$ρ = Xcosθ + Ysinθ$$.

- $$ρ$$ = perpendicular distance from the origin.
- $$θ$$ = angle of inclination of the normal line from the x-axis.

![](/assets/images/xytomc--2-.png)

By representing the line in polar coordinates, we get a sinusoidal curve in the hough space instead of straight lines. This curve is identified for all of the different values for $$ρ$$ and $$θ$$ of lines passed through our points. 

![](/assets/images/curve.png)

If we have more points, they create more curves in our hough space. Similar to the previous case, the values corresponding to the most intersecting curves, will be taken to create the best fit line.

