---
layout: post
title: "Calculating Eigen values and eigen vectors"
categories: math
author:
- Nigama Vykari 
meta: "Springfield"
---

Eigen Values and eigen vectors are used to reduce matrices into much simpler forms which allows in easier calculations. They play a crucial role in many real-life applications of linear-algebra, differential equations and physical sciences.

Here, we review how we can calculate eigen vectors and eigen values with examples. 

Suppose we are given a square matrix $$ M $$, and were told that $$ \overline{M} \overline{V} = \lambda \overline{V}$$ for some scalar $$\lambda$$ (+ve or -ve) and some column matrix $$\overline{V}$$ (anything except 0), then such $$\lambda$$ is an *eigen value* and the $$\overline{V}$$ is an *eigen vector*.

If we are given a condidate $$\overline{V}$$ to try, then we find the values the following way:

$$ \begin{equation*} \overline{M} = \begin{pmatrix}
2 & 4 \\
1 & -1 \\
\end{pmatrix}
\end{equation*} $$ and $$ \begin{equation*} \overline{V} = \begin{pmatrix}
1 \\
-1 \\
\end{pmatrix}
\end{equation*} 
$$

Try 

$$ \begin{equation*} \overline{M} \overline{V} = \begin{pmatrix}
2 & 4 \\
1 & -1 \\
\end{pmatrix}
\end{equation*}  \begin{pmatrix}
1 \\
-1 \\
\end{pmatrix} = \begin{pmatrix}
-2 \\
2 \\
\end{pmatrix}$$ 

we get

$$ \overline{V} = -2 \begin{pmatrix}
1 \\
-1 \\
\end{pmatrix}$$ 

and

$$\lambda = -2$$

What if $$\overline{V}$$ /eigen vector is not given? In such case, 

1. We find the $$ \lambda $$ values.
2. For each value, we can find the corresponding $$\overline{V}$$ values.

$$ \overline{M} \overline{V} = \lambda \overline{V} \rightarrow \overline{M} \overline{V} - \lambda \overline{V} = 0 $$

$$ \overline{M} \overline{V} - \lambda \overline{I} \overline{V} = 0 $$ 

where $$ \overline{I}$$ is an indentity matrix. 

$$ (\overline{M} - \lambda \overline{I}) \overline{V} = 0 $$

This can only be solved for $$ \overline{V} \neq 0 $$ if the following equation is true.

$$ \overline{M} - \lambda \overline{I} = 0 $$

This equation allows us to find all the eigen values. 

**Example-1:**

$$ \begin{equation*} \overline{M} = \begin{pmatrix}
2 & 4 \\
1 & -1 \\
\end{pmatrix}
\end{equation*} $$ and $$ \begin{equation*} \overline{M} \overline{I} = \begin{pmatrix}
\lambda & 0 \\
0 & \lambda \\
\end{pmatrix}
\end{equation*} $$

$$ \overline{M} - \lambda \overline{I} = \left | {\begin{array}{cc}
   2-\lambda & 4 \\
   1 & -1-\lambda \\
  \end{array} } \right | = 0 $$ 

  $$ (2-\lambda)(-1-\lambda)-4 = 0 $$

  $$ \lambda^{2} - \lambda - 6 = 0 $$

  $$ (\lambda - 3) (\lambda + 2) = 0 $$

  $$ \therefore \lambda = 3, -2 $$


$$ \overline{M} = \left | {\begin{array}{cc}
   -2 & 1 & 3 \\
   1 & -1 & 0 \\
  -1 & 1 & 2 \\
  \end{array} } \right | $$ 

we know that $$ \overline{M} - \lambda \overline{I} = 0 $$

$$ \rightarrow \left | {\begin{array}{cc}
   -2 & 1 & 3 \\
   1 & -1 & 0 \\
  -1 & 1 & 2 \\
  \end{array} } \right | - \left | {\begin{array}{cc}
   \lambda & 0 \\
   0 & \lambda \\
  \end{array} } \right | = \left | {\begin{array}{cc}
   -2-\lambda & 1 & 3 \\
   1 & -1-\lambda & 0 \\
  -1 & 1 & 2-\lambda \\
  \end{array} } \right | $$

$$ -1 \left | {\begin{array}{cc}
   1 & 3 \\
   1 & 2-\lambda \\
  \end{array} } \right | + (-1-\lambda) \left | {\begin{array}{cc}
   -2-\lambda & 3 \\
   -1 & 2-\lambda \\
  \end{array} } \right | = 0 $$

$$ -(2-\lambda-3) - (1+\lambda) [(-2-\lambda)(2-\lambda)+3] = 0 $$

$$ (\lambda+1) - (\lambda+1)(-4+2\lambda - 2\lambda + \lambda^{2} + 3) = 0$$

$$ (\lambda + 1) [1-(\lambda^{2} - 1)] = 0 $$

$$ \lambda = -1, 2 $$

$$ 1-(\lambda^{2} - 1) = 2 - \lambda^{2} \rightarrow \lambda^{2} = 2 $$

$$ \lambda \pm \sqrt{2} $$

$$ \therefore \lambda = -1, \sqrt{2}, -\sqrt{2} $$

#### **Rules for Solving Eigen Vectors**

1. Find the possible $$\lambda$$ values using $$ M - \lambda I = 0 $$
2. Generally there are $$'n'$$ solutions for an $$ n \times n $$ matrix.
3. For each lambda value, figure out an acceptable $$ \overline{V} $$ value.
4. You only have to choose $$n-1$$ of the rows.
5. You might be asked to normalize i.e; scale the values to $$ \hat{V} = 1 $$.

**Example-2:**

$$ \overline{M} = \left | {\begin{array}{cc}
   -2 & 1 & 3 \\
   1 & -1 & 0 \\
   -1 & 1 & 2 \\
  \end{array} } \right | = 0 $$  and $$ \lambda = -1, \lambda_{2} = \sqrt{2}, \lambda_{3} = -\sqrt{2} $$

From the equation $$ (\bar{M} -\lambda \bar{I}) \bar{V} = 0 $$

$$ \begin{equation*} \lambda_{2} = \sqrt{2} \begin{pmatrix}
-2-\sqrt{2} & 1 & 3 \\
1 & -1-\sqrt{2} & 0 \\
-1 & 1 & 2-\sqrt{2} \\
\end{pmatrix}
\end{equation*}  \begin{pmatrix}
x \\
y \\
z \\
\end{pmatrix} = \begin{pmatrix}
0 \\
0 \\
0 \\
\end{pmatrix}$$ 

$$ x + (-1-\sqrt{2})y = 0 $$

$$ y = 1 \rightarrow x = 1+\sqrt{2} $$

$$ (-2-\sqrt{2})x + y + 3z = 0 \rightarrow (1+\sqrt{2})(-2-\sqrt{2})+1 $$

$$ \begin{equation*} \lambda_{2} = \begin{pmatrix}
1 + \sqrt{2} \\
     1       \\     
1 + \sqrt{2} \\
\end{pmatrix}
\end{equation*} $$

$$ \begin{equation*} \lambda_{3} = \sqrt{2} \begin{pmatrix}
-2+\sqrt{2} & 1 & 3 \\
1 & -1+\sqrt{2} & 0 \\
-1 & 1 & 2+\sqrt{2} \\
\end{pmatrix}
\end{equation*}  \begin{pmatrix}
x \\
y \\
z \\
\end{pmatrix} = \begin{pmatrix}
0 \\
0 \\
0 \\
\end{pmatrix}$$

$$ x+(\sqrt{2}-1)y = 0 $$

$$ y = 1 \rightarrow x = 1-\sqrt{2} $$

$$ -x +y+(2+\sqrt{2})z = 0 $$

$$ (2+\sqrt{2})z = 0 $$

$$ z = \frac{-1}{\sqrt{2}+1} $$







