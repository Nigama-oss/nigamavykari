---
layout: post
title: "Module Extensions for Python"
categories: machine-learning
author:
- Nigama Vykari 
meta: "python"
---

Any code that can be integrated or imported to another python script is an *extension*. This new code can be written in python, C or C++. A great feature of python is that the underlying details of the module imports are hidden unless the client searches for it. Therefore you wouldn't be able to tell whether the module was written in python or some compiled language. 

> Note that the extensions are generally available in a development environment and so it is required that you perform your tasks in a similar environment if you intend to create new extensions. 

Here we discuss how to take externally written code and integrate it into our own python environment.

#### **Writing Extensions**

There are three main steps involved while creating extensions for python:

- Creating the application code.
- Wrapping code wit boilerplates.
- Compilation and testing.

Lets uncover each of these steps with examples.

**1. Creating Application Code**

Firstly, create a *library* before your code becomes an *extension*. You need to keep in mind that this code will later communicate with and share data with your C code. 


Second, create a test code for your software. 