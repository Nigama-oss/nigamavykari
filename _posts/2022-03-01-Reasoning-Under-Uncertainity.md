---
layout: post
title: "Reasoning Under Uncertainity"
categories: ml
author:
- Nigama Vykari 
---

Probability theory and the associated notion of uncertainity is not just a technical or mathematical concept used in machine learning, but a much broader term that applies to most of our daily life. It is a part of an important process that we might describe as *human intelligence*.

> **"Life's most important problems are for the most part, problems of probablility."**    
**- Laplace.**

Uncertainity is discussed in many crucial concepts including scientific reasoning, political issues, and is obviously an important part of computer science. 

#### **Computational difficulties of probability theory**

Lets consider a situation where there are $$26$$ variables from A to Z $$ (n = 26) $$ that we would like to make statements about. In a classical propositional framework, it would work the following way:

- Firstly we assign a binary value to the variables. (Ex: A = True, B = True, C = False and so on...)
- We assign true values to a certain subset of these $$26$$ variables and also use propositional statements like: $$A \rightarrow {B}$$, $$C = A + B, F$$ is true $$\forall {D}$$ & $$E$$ etc.

- Then we use the rules of propositional logic to semi-automatically derive truth values to other variables in these alphabets.

Remember that we are doing this computationally, and by the end of this process we would've assigned truth values to all variables, and doing so requires a storage of $$26$$ bits.

If we extend this framework to a probablistic setting, we do not just assign true or false, but instead we have to assign a probability to every possible variation of T or F for all of the 26 variables. Storing these requires us to store real numbers between $$0$$ and $$1$$ for $$ 2^{26} (2^{26} = 67108864) $$ different staements. 

This is excluding the fact that we store real numbers rather than binary, which would further complicate things. It approximately measures to $$67$$ megabytes of storage for this task, which is disastrous. 

***Why do we need to make this sacrifice?***

If we are uncertain, we need to keep track of every single hypothesis. Every one of them could be the right one and there isn't a unique solution anymore. This is a fundamental aspect of trying to be uncertain.

The challenge of probablistic reasoning in practice lies in the cost of being uncertain, which is much more expensive in terms of computation and memory than simply committing to a single hypothesis.

#### **Defining Independence and Conditional Independence**

Two variables A & B are independent only if their joint distributions factorizes into marginal distributions i.e; $$ P(A, B) = P(A) . P(B) $$. There is a generalization of this definition which is straight forward to conditional distribution. 

If two variables A & B are conditionally independent given a variable C, only if their conditonal distribution factorizes the following -

$$ P(A, B | C) = P(A | C) P(B | C) $$

In such a case, we have:

$$ P(A|B, C) = {P(A|C)} $$ 

i.e; in light of information about C, B provides no further information about A. This concept of independence and conditional independence can be used to simplify computation.

To see this in action, lets consider an example directly from the book *[Information theory inference and learning algorithms by David MacKay](https://www.inference.org.uk/itprnn/book.pdf)* where he considered four binary variables and assigned probablity to all possible states. The variables are as follows:

- A = The event where the alarm has triggered
- E = The event where there was an earthquake
- B = The event where there was a break-in
- R = The event where the announcement is made on the radio

This joint probability distribution has $$ 2^{4} - 1 = 15 $$ parameters.

The probability of all these events occuring would be - 

$$ P(A, E, B, R) = P(A|R, E, B).P(R|E, B).P(E|B).P(B) $$

However, we might get lucky and there might be some additional information that we have available from the designing process or domain knowledge of the problem that can simplify the computation.

Removing the irrelevant conditions (domain knowledge) reduces the parameters to $$ 8 \rightarrow{[4 + 2 + 1 + 1]}. $$

$$ \therefore{P(A, E, B, R) = P(A|E, B).P(R|E).P(E).P(B)} $$

This is one way where independence can help us drastically improve computation where the count of parameters can be reduced to almost half or sometimes even lower. 

#### **Representation**

Representing our variables in a graphical model helps us to do inference. 

![](/assets/images/rep.png)

In probablistic reasoning, the process of inference is completely mechanical. We just have to write down numbers and use bayes theorem to compute posteriors. So, lets start with some numbers in the following example. 

For probability -

$$ P(A, E, B, R) = P(A|E, B).P(R|E).P(E).P(B) $$

$$ P(B = 1) = 10^{-3} $$              |               $$ P(B = 0) = 1 - 10^{-3} $$ 
$$ P(E = 1) = 10^{-3} $$              |               $$ P(E = 0) = 1 - 10^{-3} $$ 

and 

$$ P(R = 1 | E = 1) = 1.0 $$ 

$$ P(R = 1 | B = 0) = 0.0 $$ 


Using $$ f = 10^{-3}, \alpha_{b} = 0.999, \alpha_{e} = 0.001 $$

$$ f $$ = Probablibity of a false alarm.

$$ \alpha_b $$ = Probability the alarm goes off if there is a burglary.

$$ \alpha_b $$ = Probability the alarm goes off if there is an earthquake.

$$ P(A=0|B=0, E=0) = 0.999 & P(A=1|B=0, E=0) = 0.001 $$

$$ P(A=0|B=1, E=0) = 0.00999 & P(A=1|B=1, E=0) = 0.99001 $$

$$ P(A=0|B=0, E=1) = 0.98901 & P(A=1|B=0, E=1) = 0.01099 $$

$$ P(A=0|B=1, E=1) = 0.0098901 & P(A=1|B=1, E=1) = 0.9901099 $$

**Q:** What is the probability that there was a break-in and/or an earthquake given that the alarm went off?

**A:** 

$$ P(B, E|A = 1) = \frac{P(A=1 | B, E) P(B)P(E)} {P(A=1)} $$

$$ P(A=1) = P(A=1|B=0, E=0) P(B=0) P(E=0) + $$

$$P(A=1|B=0, E=1) P(B=0) P(E=1) + $$

$$P(A=1|B=1, E=0) P(B=1) P(E=0) + $$

$$P(A=1|B=1, E=1) P(B=1) P(E=1) $$

$$= 0.000998 + 0.000989 + 0.000010979 + 0.00000099 = 0.002 $$

$$ \therefore $$ Note the conditional independence. 

If we take a look at the conditional probabilities for earthquakes and burglaries, we notice that the values are not independent anymore.

$$ P(B=0, E=0|A=1) = 0.4993 $$

$$ P(B=0, E=1|A=1) = 0.0055 $$

$$ P(B=1, E=0|A=1) = 0.4947 $$

$$ P(B=1, E=1|A=1) = 0.0005 $$

Although the model initially assumed that B & E happen independent of each other, after we observe the alarm we have "conditional independence" between these variables. This is an intersting effect you observe that- once a variable that can be caused by two different causes or generative processes they become dependent on eachother. 

**Once we get the additional information,** we can ask the question: What is the probability for a break-in given the alarm goes off and an announcement is made on the radio?

**A:** 

$$ P(B=0|E=1, A=0) = \frac{P(B=0, E=1|A=1)}{P(E=1|A=1)} $$

$$ = \frac{P(B=0, E=1|A=1)}{P(B=0, E=1|A=1) + P(B=1, E=1|A=1)} = 0.92 = 92. $$

$$ P(B=1|E=1, A=1) = \frac{P(B=1, E=1|A=1)}{P(E=1|A=1)} $$

$$ = \frac{P(B=1, E=1|A=1)}{P(B=0, E=1|A=1) + P(B=1, E=1|A=1)} = 0.08 =8. $$ 

The above example didn't just show the structure in our reasoning process, but also served to show a simple case of cooking recipie for solving inference problems. Therefore to do inference, **we always need to write down the probability of everything.**

To summarize the process :

1. Identify all the relevant variables. Ex: A, R, E, B.
2. Define a joint probability that assigns a probability measure to all the variables (Also known as writing a generative model).
3. Once we have our probability space, we can now see where our observations come from. Then we fix the values. Ex: A=1, B=0 etc.
4. Perform inference by applying Bayes theorem.

The key conceptual strength of probablistic reasoning is that once you agree on a generative model, there is no question anymore about what the correct answer to inference should be.

***A philosophical thought to ponder:***  *It is the very nature of independence that is the biggest philosophical issue maybe for probability theory and not the proor or likelihood and what it means to be uncertain, but actually what it means to be independent.* 

#### **Conclusion**

We have seen that probability theory, even though is a beautiful mechanical process, has a computational issue: If we keep track of uncertainity about several possible hypothesis, then we might face a combinatorily hard problem. 

This is a fundamental part of reasoning under uncertainity. It is not solved by any other framework and is just **caused** by the fact that we want to keep track of several possibilities. Conditional independence is going to be a computational tool for us that simplifies computational tasks in machine learning.

One of the many different ways to deal with this computationally is **the notion of conditional independence**. It reduces complexity and helps us make things tractable by separating sums from each other so that they can be solved with way less degrees of freedom.






