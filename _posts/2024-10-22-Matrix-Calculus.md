---
layout: post
title:  "Matrix Calculus for Deep Learning"
date:   2024-10-22
---

## Introduction

When beginning to dive into machine learning, matrix calculus is one of the more intimidating tools that users must familiarise themselves with. Rigorous mathematical approaches usually contain tensor products, Kroeneker deltas, Jacobian tensors, seemingly arbitrarily transposed matrices and vector outputs, and derivatives of matrices written in Einstein notation using dummy indices. This post will hopefully make the basics, especially the essential components applicable to deep learning, a little more approachable. We'll start by revisiting scalar calculus and gradually build up to the richer world of vector and matrix operations. We'll cover element-wise operations, including addition, multiplication, and non-linear functions, before moving on to the more advanced topic of matrix-vector operations. For each operation, we'll derive the Jacobian matrix, highlighting its relevance in deep learning. By the end, you'll have a solid foundation in the basic matrix calculus for deep learning. Future posts will build on these ideas and show how they can be concretely applied to training real world problems.

For readers interested in a more comprehensive exploration of matrix calculus, Parr and Howard's essential guide ([explained.ai/matrix-calculus](https://explained.ai/matrix-calculus/)) provides an in-depth treatment of the subject. While that resource offers a wider overview, this post focuses on explicitly solving the simplest cases.

## Beyond Scalars

Practitioners should be familiar with standard scalar calculus, where we (typically) have some function $f : \mathbb{R} \mapsto \mathbb{R}$, which simply means we are taking a single real number, and mapping that to another real number. The derivative of this function is another function, and describes how sensitive the output of the function is to small perturbations around valid inputs. In this local region, our function can be viewed as linear, and the derivative defines the slope of the tangent line at this input. Below is an example:

$$
\begin{split}
f(x) & = x^3 + 2x + 1, \\
\frac{df}{dx} & = 3x^2 + 2
\end{split}
$$

A simple multivariate extension would be to change our scalar function, such that we now take in two inputs. Formally, we say $f : \mathbb{R}^2 \mapsto \mathbb{R}$. The input is now a vector: 

$$
\begin{split}
f(\mathbf{x}) = f\begin{pmatrix}x_1 \\ x_2 \end{pmatrix} & = x_1^2 + 2x_1x_2 + 1 \\
\end{split}
$$

We have two options for calculating the derivative of this function, by taking the partial derivative of the function with respect to either input. Grouping these two functions into a single row vector, we get the Jacobian, $J$, our workhorse for calculus beyond scalar valued functions:

$$
\begin{split}
J = 
\nabla_{\mathbf{x}} f =
\nabla f
\begin{pmatrix}
x_1 \\
x_2 
\end{pmatrix} & = \begin{pmatrix} \frac{\partial{f}}{\partial{x_1}} & \frac{\partial{f}}{\partial{x_2}} \end{pmatrix} \\
& = \begin{pmatrix} 2x_1 + 2x_2 & & 2x_1 \end{pmatrix}
\end{split}
$$

This Jacobian now parameterises how our output changes with respect to changes in either (or both) of our inputs. Now our vector valued Jacobian describes the slope of the tangent line in two-dimensions, which is still approximately linear in the neighbourhood of any valid $x_1$ and $x_2$.

We could similarly consider a multivariate extension where we have a single scalar input, but our function is vector valued with two outputs. This can be thought of as having two separate mapping functions, $f_1 : \mathbb{R} \mapsto \mathbb{R}$ and $f_2 : \mathbb{R} \mapsto \mathbb{R}$, combined into a vector $\mathbf{f} : \mathbb{R} \mapsto \mathbb{R}^2$, for example:

$$
\begin{split}
\mathbf{f}(x) = 
\begin{pmatrix}
f_1(x) \\ 
f_2(x) 
\end{pmatrix} = 
\begin{pmatrix} 3x^2 \\
6x + 1 
\end{pmatrix}
\end{split}
$$

Here our Jacobian, now a column vector, $J$, takes the form:

$$
\begin{split}
J = \nabla_{x} \mathbf{f} = 
\begin{pmatrix} 
\frac{\partial{f_1}}{\partial{x}} \\
\frac{\partial{f_2}}{\partial{x}} 
\end{pmatrix}
= \begin{pmatrix} 6x \\ 6 \end{pmatrix}
\end{split}
$$

Note that we chose to arrange our Jacobian such that we take partial derivatives of multiple inputs (with a scalar function) as a row vector, and vector-valued function (with a single scalar input) as a column vector. This is an arbitrary choice, and is known as the numerator layout, we could quite easily have chosen to do the opposite (the denominator layout), the most important point is to pick one, and be consistent. The numerator layout tends to be the most common form chosen in the ML community, so we will go with that.

The next step is a little bit more abstract, but hopefully you can see what's coming. A more general multivariate case is when we have an arbitrary sized vector input, mapping to an arbitrarily sized vector output. Formally, we can say $\mathbf{f} : \mathbb{R}^n \mapsto \mathbb{R}^m$:

$$
\begin{split}
\mathbf{f}(\mathbf{x}) = 
\begin{pmatrix}
f_1(\mathbf{x}) \\
f_2(\mathbf{x}) \\
\vdots \\
f_m(\mathbf{x})
\end{pmatrix} = 
\begin{pmatrix}
f_1\mathbf{\begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}} \\
\vdots \\
f_m\mathbf{\begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}} \\
\end{pmatrix}
\end{split}
$$

Our generalised Jacobian, $J$, takes the form of an $m$ by $n$ matrix, as we have $m$ outputs, and $n$ inputs, and describes the partial derivative of each input with respect to each output:

$$
\begin{split}
J = \nabla{\mathbf{f}}_\mathbf{x} = 
\begin{pmatrix}
\nabla{f_1}(\mathbf{x}) \\
\nabla{f_2}(\mathbf{x}) \\
\vdots \\
\nabla{f_m}(\mathbf{x})
\end{pmatrix} = 
\begin{pmatrix}
\frac{\partial{f_1}}{\partial{x_1}} & \frac{\partial{f_1}}{\partial{x_2}} & \dots & \frac{\partial{f_1}}{\partial{x_n}} \\ 
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial{f_m}}{\partial{x_1}} & \frac{\partial{f_m}}{\partial{x_2}} & \dots & \frac{\partial{f_m}}{\partial{x_n}} \\
\end{pmatrix}
\end{split}
$$

For now, we will avoid making this more abstract by considering functions of higher rank objects (i.e., tensors), as the Jacobian will be a higher order tensor. In the following sections, we will look at the most common matrix calculus operations applicable to deep learning, defining the Jacobian for element-wise operations on vectors (e.g., scalar and vector addition, scalar and vector multiplication, and the application of an arbitrary non-linear function). We will also detail matrix-vector multiplication, here the Jacobian will be a third order tensor, but due to the simple linear nature of the operation, this high dimensional tensor will have a particularly friendly form.

## Element-wise Operations

### Vector-scalar Addition

Given a vector $\mathbf{x} \in \mathbb{R}^n$, and a scalar $a \in \mathbb{R}$, we define our output $\mathbf{y} \in \mathbb{R}^n$ as adding $a$ to every component of $\mathbf{x}$:

$$
\begin{split}
\mathbf{y} = f(\mathbf{x} + a) = 
\begin{pmatrix}x_1 + a \\ x_2 + a \\ \vdots \\ x_n + a 
\end{pmatrix}
=
\begin{pmatrix}y_1\\ y_2 \\ \vdots \\ y_n 
\end{pmatrix}
\end{split}
$$

This results in two Jacobians, depending on which of our inputs we are differentiating with respect to, $\frac{\partial\mathbf{y}}{\partial\mathbf{x}}$ and $\frac{\partial\mathbf{y}}{\partial a}$. The simpler case is $\frac{\partial\mathbf{y}}{\partial a}$, as we are differentiating with respect to a single scalar value:

$$
\begin{split} 
\frac{\partial\mathbf{y}}{\partial a} =
\frac{\partial f(\mathbf{x} + a)}{\partial a} = 
\begin{pmatrix}
\frac{\partial (x_1 + a)}{\partial a} \\
\frac{\partial (x_2 + a)}{\partial a} \\
\vdots \\
\frac{\partial (x_n + a)}{\partial a}
\end{pmatrix}
=
\begin{pmatrix}
1 \\
1 \\
\vdots \\
1
\end{pmatrix}
= \mathbf{1} \in \mathbb{R}^n
\end{split}
$$

This should look familiar to single-variable scalar case. We can also easily extend this to matrix-scalar addition, where our Jacobian will instead be $\mathbf{1} \in \mathbb{R}^{m\times n}$, when we have an $m\times n$ matrix. $\frac{\partial\mathbf{y}}{\partial\mathbf{x}}$ is slightly more complicated, as we need to take partial derivatives with respect to every component of our vector valued input:

$$
\begin{split}
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} =
\frac{\partial f(\mathbf{x} + a)}{\partial \mathbf{x}} & = 
\begin{pmatrix}
\frac{\partial (x_1 + a)}{\partial x_1} & \frac{\partial (x_1 + a)}{\partial x_2} & \dots & \frac{\partial (x_1 + a)}{\partial x_n}  \\
\frac{\partial (x_2 + a)}{\partial x_1} & \frac{\partial (x_2 + a)}{\partial x_2} & \dots &  \frac{\partial (x_2 + a)}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial (x_n + a)}{\partial x_1} & \frac{\partial (x_n + a)}{\partial x_2} & \dots & \frac{\partial (x_n + a)}{\partial x_n}
\end{pmatrix} \\
& =
\begin{pmatrix}
1 & 0 & \dots & 0 \\
0 & 1 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & 1
\end{pmatrix}
= \mathbf{I} \in \mathbb{R}^{n\times n}
\end{split}
$$

Our output is the identity matrix, where we have ones along the diagonals, and zeros everywhere else. We call it the identity matrix because applying it to any matrix (of equivalent shape) returns the matrix unchanged. In the context of DL, vector-scalar addition is commonly utilised whenever we add a learnable bias terms to intermediate layers, and also in sophisticated gradient based optimisers like Adam.

### Vector Addition

Given two vectors $\mathbf{x}, \mathbf{z} \in \mathbb{R}^n$, we define our output $\mathbf{y} \in \mathbb{R}^n$ as the elementwise addition of each component or $x_i \in \mathbf{x}$ with $z_i \in \mathbf{z}$:

$$
\begin{split}
\mathbf{y} = f(\mathbf{x} + \mathbf{z}) = 
\begin{pmatrix}x_1 + z_1 \\ x_2 + z_2 \\ \vdots \\ x_n + z_n 
\end{pmatrix}
=
\begin{pmatrix}y_1\\ y_2 \\ \vdots \\ y_n \end{pmatrix}
\end{split}
$$

Our two Jacobians, $\frac{\partial\mathbf{y}}{\partial\mathbf{x}}$ and $\frac{\partial\mathbf{y}}{\partial\mathbf{z}}$ can be derived as:

$$
\begin{split}
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} =
\frac{\partial f(\mathbf{x} + \mathbf{z})}{\partial \mathbf{x}} & = 
\begin{pmatrix}
\frac{\partial (x_1 + z_1)}{\partial x_1} & \frac{\partial (x_1 + z_1)}{\partial x_2} & \dots & \frac{\partial (x_1 + z_1)}{\partial x_n}  \\
\frac{\partial (x_2 + z_2)}{\partial x_1} & \frac{\partial (x_2 + z_2)}{\partial x_2} & \dots &  \frac{\partial (x_2 + z_2)}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial (x_n + z_n)}{\partial x_1} & \frac{\partial (x_n + z_n)}{\partial x_2} & \dots & \frac{\partial (x_n + z_n)}{\partial x_n}
\end{pmatrix} \\
& =
\begin{pmatrix}
1 & 0 & \dots & 0 \\
0 & 1 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & 1
\end{pmatrix}
= \mathbf{I} \in \mathbb{R}^{n\times n}
\end{split}
$$


and,

$$
\begin{split}
\frac{\partial \mathbf{y}}{\partial \mathbf{z}} =
\frac{\partial f(\mathbf{x} + \mathbf{z})}{\partial \mathbf{z}} & = 
\begin{pmatrix}
\frac{\partial (x_1 + z_1)}{\partial z_1} & \frac{\partial (x_1 + z_1)}{\partial z_2} & \dots & \frac{\partial (x_1 + z_1)}{\partial z_n}  \\
\frac{\partial (x_2 + z_2)}{\partial z_1} & \frac{\partial (x_2 + z_2)}{\partial z_2} & \dots &  \frac{\partial (x_2 + z_2)}{\partial z_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial (x_n + z_n)}{\partial z_1} & \frac{\partial (x_n + z_n)}{\partial z_2} & \dots & \frac{\partial (x_n + z_n)}{\partial z_n}
\end{pmatrix} \\
& =
\begin{pmatrix}
1 & 0 & \dots & 0 \\
0 & 1 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & 1
\end{pmatrix}
= \mathbf{I} \in \mathbb{R}^{n\times n}
\end{split}
$$

These operations are commonly utilised inside most standard neural network layers, where we do a matrix multiplication with our input vector (more on this later) and sum the resulting output vector to a bias vector. We also regularly encounter vector addition in residual layers, when combining the outputs from multi-head attention back to the residual stream, or when summing the outputs of multiple different task specific heads into a single feature vector.

### Vector-scalar Multiplication

Hopefully by now the process defining vector operations in terms of their simpler scalar components is starting to become less daunting. Let's quickly run through the process for multiplication. Given a vector $\mathbf{x} \in \mathbb{R}^n$, and a scalar $a \in \mathbb{R}$, we define our output $\mathbf{y} \in \mathbb{R}^n$ as multiplying $a$ with every component of $\mathbf{x}$:

$$
\begin{split}
\mathbf{y} = f(a \mathbf{x}) = 
\begin{pmatrix}ax_1 \\ ax_2 \\ \vdots \\ ax_n 
\end{pmatrix}
=
\begin{pmatrix}y_1\\ y_2 \\ \vdots \\ y_n 
\end{pmatrix}
\end{split}
$$

Our first Jacobian, $\frac{\partial\mathbf{y}}{\partial a}$ is:

$$
\begin{split}
\frac{\partial\mathbf{y}}{\partial a} =
\frac{\partial f(a\mathbf{x})}{\partial a} = 
\begin{pmatrix}
\frac{\partial (ax_1)}{\partial a} \\
\frac{\partial (ax_2)}{\partial a} \\
\vdots \\
\frac{\partial (ax_n)}{\partial a}
\end{pmatrix}
=
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{pmatrix}
= \mathbf{x}
\end{split}
$$

This can be easily extended for matrix-scalar multiplication, where the output of our Jacobian would instead be original matrix, $X \in \mathbb{R}^{m\times n}$, where $X$ has m-rows and m-columns. Our other Jacobian, $\frac{\partial\mathbf{y}}{\partial \mathbf{x}}$ takes the form:

$$
\begin{split}
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} =
\frac{\partial f(a\mathbf{x})}{\partial \mathbf{x}} & = 
\begin{pmatrix}
\frac{\partial (ax_1)}{\partial x_1} & \frac{\partial (ax_1)}{\partial x_2} & \dots & \frac{\partial (ax_1)}{\partial x_n}  \\
\frac{\partial (ax_2)}{\partial x_1} & \frac{\partial (ax_2)}{\partial x_2} & \dots &  \frac{\partial (ax_2)}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial (ax_n)}{\partial x_1} & \frac{\partial (ax_n)}{\partial x_2} & \dots & \frac{\partial (ax_n)}{\partial x_n}
\end{pmatrix} \\
& =
\begin{pmatrix}
a & 0 & \dots & 0 \\
0 & a & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & a
\end{pmatrix}
= a\mathbf{I} \in \mathbb{R}^{n\times n}
\end{split}
$$

In practical applications, this Jacobian for vector-scalar multiplication plays a fundamental role in normalisation layers (such as batch normalisation and layer normalisation), where we learn a scalar parameter, $\gamma$, that linearly transforms the normalised outputs (typically a vector). Thus this Jacobian enables us to learn the optimal scaling factor for each output layer through gradient-based optimisation.

### Vector Multiplication

Given vectors $\mathbf{x}, \mathbf{z} \in \mathbb{R}^n$, we define our output $\mathbf{y} \in \mathbb{R}^n$ as the elementwise multiplication (also known as the Hadamard product) of each component or $x_i \in \mathbf{x}$ with $z_i \in \mathbf{z}$:

$$
\begin{split}
\mathbf{y} = f(\mathbf{x} * \mathbf{z}) = 
\begin{pmatrix}x_1 z_1 \\ x_2 z_2 \\ \vdots \\ x_n z_n 
\end{pmatrix}
=
\begin{pmatrix}y_1\\ y_2 \\ \vdots \\ y_n \end{pmatrix}
\end{split}
$$

Our two Jacobians, $\frac{\partial\mathbf{y}}{\partial\mathbf{x}}$ and $\frac{\partial\mathbf{y}}{\partial\mathbf{z}}$ can be derived as:

$$
\begin{split}
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} =
\frac{\partial f(\mathbf{x} * \mathbf{z})}{\partial \mathbf{x}} & = 
\begin{pmatrix}
\frac{\partial (x_1 z_1)}{\partial x_1} & \frac{\partial (x_1 z_1)}{\partial x_2} & \dots & \frac{\partial (x_1 z_1)}{\partial x_n}  \\
\frac{\partial (x_2 z_2)}{\partial x_1} & \frac{\partial (x_2 z_2)}{\partial x_2} & \dots &  \frac{\partial (x_2 z_2)}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial (x_n z_n)}{\partial x_1} & \frac{\partial (x_n z_n}{\partial x_2} & \dots & \frac{\partial (x_n z_n)}{\partial x_n}
\end{pmatrix} \\
& =
\begin{pmatrix}
z_1 & 0 & \dots & 0 \\
0 & z_2 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & z_n
\end{pmatrix}
= diag({\mathbf{z}}) \in \mathbb{R}^{n\times n}
\end{split}
$$

and,

$$
\begin{split}
\frac{\partial \mathbf{y}}{\partial \mathbf{z}} =
\frac{\partial f(\mathbf{x} * \mathbf{z})}{\partial \mathbf{z}} & = 
\begin{pmatrix}
\frac{\partial (x_1 z_1)}{\partial z_1} & \frac{\partial (x_1 z_1)}{\partial z_2} & \dots & \frac{\partial (x_1 z_1)}{\partial z_n}  \\
\frac{\partial (x_2 z_2)}{\partial z_1} & \frac{\partial (x_2 z_2)}{\partial z_2} & \dots &  \frac{\partial (x_2 z_2)}{\partial z_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial (x_n z_n)}{\partial z_1} & \frac{\partial (x_n z_n}{\partial z_2} & \dots & \frac{\partial (x_n z_n)}{\partial z_n}
\end{pmatrix} \\
& =
\begin{pmatrix}
x_1 & 0 & \dots & 0 \\
0 & x_2 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & x_n
\end{pmatrix}
= diag({\mathbf{x}}) \in \mathbb{R}^{n\times n}
\end{split}
$$

Here $diag(\mathbf{a})$ refers to a diagonal matrix, where the entries in the diagonal correspond to each element in the original vector $\mathbf{a}$. Notice the continuing theme of sparsity in our higher dimensional Jacobians, off diagonal elements all go to zero. Libraries for performing DL (such as tensorflow and PyTorch) will not formulate the entire Jacobian for these operations, as this sparsity can be leveraged for more optimised solutions (we will illustrate an example of this later).

### Vector Sums

Another commonly used operation in DL involves taking the sum of a vector, and using this sum in a subsequent operation applied to each component of the original vector (a good example is the denominator in the softmax function). This may look intimidating, due to the mathematical summation notation, but it just as straight forward as all the operations we have described so far. Given a vector $\mathbf{x} \in \mathbb{R}^n$, we define each element in our output vector $\mathbf{y} \in \mathbb{R}^n$ as the sum of all the components in $\mathbf{x}$:

$$
\begin{split}
\mathbf{x}  = 
\begin{pmatrix}x_1 \\ x_2 \\ \vdots \\ x_n 
\end{pmatrix}
\mathbf{y}  =
\begin{pmatrix}
\sum_{i=1}^{n} x_i \\ \sum_{i=1}^{n} x_i \\ \vdots \\ \sum_{i=1}^{n} x_i
\end{pmatrix}
=
\begin{pmatrix}
x_1 + x_2 + ... + x_n \\ x_1 + x_2 + ... + x_n \\ \vdots \\ x_1 + x_2 + ... + x_n
\end{pmatrix}
=
\begin{pmatrix}
y_1 \\ y_2 \\ \vdots \\ y_n
\end{pmatrix}
\end{split}
$$

Our Jacobian:

$$
\begin{split}
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} &=
\begin{pmatrix}
\frac{\partial (\sum_{i=1}^{n} x_i)}{\partial x_1} & \frac{\partial (\sum_{i=1}^{n} x_i)}{\partial x_2} & \dots & \frac{\partial (\sum_{i=1}^{n} x_i)}{\partial x_n}  \\
\frac{\partial (\sum_{i=1}^{n} x_i)}{\partial x_1} & \frac{\partial (\sum_{i=1}^{n} x_i)}{\partial x_2} & \dots &  \frac{\partial (\sum_{i=1}^{n} x_i)}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial (\sum_{i=1}^{n} x_i)}{\partial x_1} & \frac{\partial (\sum_{i=1}^{n} x_i)}{\partial x_2} & \dots & \frac{\partial (\sum_{i=1}^{n} x_i)}{\partial x_n}
\end{pmatrix} \\
&=
\begin{pmatrix}
1 & 1 & \dots & 1 \\
1 & 1 & \dots & 1 \\
\vdots & \vdots & \ddots & \vdots \\
1 & 1 & \dots & 1
\end{pmatrix}
= \mathbf{1} \in \mathbb{R}^{n\times n}
\end{split}
$$

### Non-linear Operations

So far we have looked at simple addition and multiplication operations. Expanding upon this, we also typically apply non-linearities at the end each layer in a neural network, increasing the expressivity of our network, allowing for more complicated behaviors to be captured. Let's take the simple example of applying the natural exponent to a vector (corresponding to the numerator of the softmax function):

$$
\begin{split}
\mathbf{x}  = 
\begin{pmatrix}x_1 \\ x_2 \\ \vdots \\ x_n 
\end{pmatrix}
\mathbf{y}  =
\begin{pmatrix}
e^{x_1} \\ e^{x_2} \\ \vdots \\ e^{x_3}
\end{pmatrix}
=
\begin{pmatrix}
y_1 \\ y_2 \\ \vdots \\ y_n
\end{pmatrix}
\end{split}
$$

Our Jacobian:

$$
\begin{split}
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} &=
\begin{pmatrix}
\frac{\partial (e^{x_1})}{\partial x_1} & \frac{\partial (e^{x_1})}{\partial x_2} & \dots & \frac{\partial (e^{x_1})}{\partial x_n}  \\
\frac{\partial (e^{x_2})}{\partial x_1} & \frac{\partial (e^{x_2})}{\partial x_2} & \dots &  \frac{\partial (e^{x_2})}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial (e^{x_n})}{\partial x_1} & \frac{\partial (e^{x_n})}{\partial x_2} & \dots & \frac{\partial (e^{x_n})}{\partial x_n}
\end{pmatrix} \\
&=
\begin{pmatrix}
e^{x_1} & 0 & \dots & 0 \\
0 & e^{x_2} & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & e^{x_n}
\end{pmatrix}
= diag(\mathbf{y}) \in \mathbb{R}^{n\times n}
\end{split}
$$

This summarises the majority of simple Jacobian formulations for element-wise operations relevant to deep learning. A useful exercise may be to see how to perform similar operations for subtraction and division, just follow the same steps and see what answers you come up with. A more advanced and useful exercise would be to try to derive the Jacobian for an operation combining a number of these steps (for example the entire softmax activation function on a vector). Just remember to simplify to a scalar case, and apply the associated single variable product, quotient, or chain rules accordingly.

## Matrix Vector Multiplication

Perhaps the most common linear operation in deep learning is transforming a vector via matrix multiplication. It is the backbone of multi-layer perceptrons, convolutional neural networks, transformers and more. Moreover, taking the gradient of a matrix-vector product with respect to a vector is a key operation in back propagation, as we need to know the gradient of the output layer to calculate the gradient in the input layer. The gradient of a matrix-vector product with respect to the matrix is essential to understand how we should change the weights in any given layer to decrease our loss, which ultimately trains our model to perform a particular task.

Given a matrix $A \in \mathbb{R}^{m \times n}$ and a vector $\mathbf{x} \in \mathbb{R}^n$, we define their product $\mathbf{y} = A\mathbf{x}$, as a vector, $\mathbf{y} \in \mathbb{R}^m$, where each element of $\mathbf{y}$ is the dot product between each row of $A$ with $\mathbf{x}$:

$$
\begin{split}
A =
\begin{pmatrix}
a_{11} & a_{12} & \dots & a_{1n}  \\
a_{21} & a_{22} & \dots & a_{2n}  \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \dots & a_{mn}  \\
\end{pmatrix}
\mathbf{x}  = 
\begin{pmatrix}x_1 \\ x_2 \\ \vdots \\ x_n 
\end{pmatrix}
\end{split}
$$

$$
\begin{split}
\mathbf{y}  =  f(A * \mathbf{x})  =
\begin{pmatrix}
a_{11} x_1 & a_{12} x_2 & \dots & a_{1n} x_n \\
a_{21} x_1 & a_{22} x_2 & \dots & a_{2n} x_n  \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} x_1 & a_{m2} x_2 & \dots & a_{mn} x_n  \\
\end{pmatrix} =
\begin{pmatrix}y_1 \\ y_2 \\ \vdots \\ y_m 
\end{pmatrix}
\end{split}
$$


This product is commonly expressed more succinctly using summation notation:

$$
y_{i} = \sum\limits_{j=1}^{n} a_{ij} x_{j}  = a_{i1}x_1 + a_{i2}x_2 + ... + a_{in}x_n
$$

Where the indices $i$ refer to the row (of the input matrix, and output vector), and $j$ is the index we sum over (the columns of $A$, and rows of $\mathbf{x}$). Let's start with the simpler case and take the Jacobian with respect to the vector $\mathbf{x}$:

$$
\begin{split}
\frac{\partial \mathbf{y}}{\partial \mathbf{x}} &= \frac{\partial f(A * \mathbf{x})}{\partial \mathbf{x}} =
\begin{pmatrix}
\frac{\partial y_1}{\partial \mathbf{x}} \\ \frac{\partial y_2}{\partial \mathbf{x}}  \\ \vdots \\ \frac{\partial y_m}{\partial \mathbf{x}} 
\end{pmatrix} =
\begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \dots & \frac{\partial y_1}{\partial x_n}
\\ 
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \dots & \frac{\partial y_2}{\partial x_n}
\\
\vdots & \vdots & \ddots & \vdots
\\
\frac{\partial y_m}{\partial x_1} & \frac{\partial y_m}{\partial x_2} & \dots & \frac{\partial y_m}{\partial x_n}
\end{pmatrix} =
\begin{pmatrix}
a_{11} & a_{12} & \dots & a_{1n}
\\ 
a_{21} & a_{22} & \dots & a_{2n}
\\
\vdots & \vdots & \ddots & \vdots
\\
a_{m1} & a_{m2} & \dots & a_{mn}
\end{pmatrix}
\end{split}
$$

Here $diag(\mathbf{a})$ refers to a diagonal matrix, where the entries in the diagonal correspond to each element in the original vector $\mathbf{a}$. Notice the continuing theme of sparsity in our higher dimensional Jacobians, off diagonal elements all go to zero. Libraries for performing DL (such as tensorflow and pytorch) will not formulate the entire Jacobian for these operations, as this sparsity can be leveraged for more optimised solutions (we will illustrate an example of this later).

Here our Jacobian reduces similarly to the scalar case, $\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial f(A * \mathbf{x})}{\partial \mathbf{x}} = A \in \mathbb{R}^{m \times n}$. Finally, let's look towards the most complicated Jacobian we are going to treat, the Jacobian with respect to the matrix $A$. We can formulate this in exactly the same fashion:

$$
\begin{split}
\frac{\partial \mathbf{y}}{\partial A} &=
\begin{pmatrix}
\frac{\partial y_1}{\partial A} \\ \frac{\partial y_2}{\partial A}  \\ \vdots \\ \frac{\partial y_m}{\partial A} 
\end{pmatrix}
\end{split}
$$

However, this Jacobian is a little more complicated, each entry in our Jacobian is a partial derivative with respect to each element in the matrix $A$:

$$
\begin{split}
\frac{\partial \mathbf{y}}{\partial A} &=
\begin{pmatrix}
\begin{bmatrix}
\frac{\partial y_1}{\partial a_{11}} & \frac{\partial y_1}{\partial a_{12}}  & \dots & \frac{\partial y_1}{\partial a_{1n}} \\
\frac{\partial y_1}{\partial a_{21}} & \frac{\partial y_1}{\partial a_{22}}  & \dots & \frac{\partial y_1}{\partial a_{2n}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial y_1}{\partial a_{m1}} & \frac{\partial y_1}{\partial a_{m2}}  & \dots & \frac{\partial y_1}{\partial a_{mn}}
\end{bmatrix} \\
\\
\begin{bmatrix}
\frac{\partial y_2}{\partial a_{11}} & \frac{\partial y_2}{\partial a_{12}}  & \dots & \frac{\partial y_2}{\partial a_{1n}} \\
\frac{\partial y_2}{\partial a_{21}} & \frac{\partial y_2}{\partial a_{22}}  & \dots & \frac{\partial y_2}{\partial a_{2n}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial y_2}{\partial a_{m1}} & \frac{\partial y_2}{\partial a_{m2}}  & \dots & \frac{\partial y_2}{\partial a_{mn}}
\end{bmatrix} \\
\vdots \\
\begin{bmatrix}
\frac{\partial y_m}{\partial a_{11}} & \frac{\partial y_m}{\partial a_{12}}  & \dots & \frac{\partial y_m}{\partial a_{1n}} \\
\frac{\partial y_m}{\partial a_{21}} & \frac{\partial y_m}{\partial a_{22}}  & \dots & \frac{\partial y_m}{\partial a_{2n}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial y_m}{\partial a_{m1}} & \frac{\partial y_m}{\partial a_{m2}}  & \dots & \frac{\partial y_m}{\partial a_{mn}}
\end{bmatrix}
\end{pmatrix} = 
\begin{pmatrix}
\begin{bmatrix}
x_1 & x_2 & \dots & x_n \\
0 & 0 & \dots &  0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots &  0 \\
\end{bmatrix} \\
\\
\begin{bmatrix}
0 & 0 & \dots &  0 \\
x_1 & x_2 & \dots & x_n \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots &  0 \\
\end{bmatrix} \\
\vdots \\
\begin{bmatrix}
0 & 0 & \dots &  0 \\
0 & 0 & \dots &  0 \\
\vdots & \vdots & \ddots & \vdots \\
x_1 & x_2 & \dots & x_n \\
\end{bmatrix}
\end{pmatrix} \in \mathbb{R}^{m\times m\times n}
\end{split}
$$

This results in the Jacobian taking the form of an $m\times m\times n$ tensor (where each of the $m$ matrices is of size $m\times n$). As this tensor is sparse (the vast majority of the terms are zero), we can succinctly summarise the same tensor in more compact form:

$$
\frac{\partial y_i}{\partial a_{jk}} = \begin{cases}
x_k & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}
$$

Furthermore, formulating the entire sparse Jacobian would result in lots of wasted computation, so is never done in practice. In deep learning, we always have a scalar loss $L$, and want to calculate gradients at any layer with respect to this loss. This is where the chain rule for back propagation saves the day, for any input, we calculate the gradient of the loss on this input, by multiplying the gradient of the loss w.r.t the output together with gradient of the output w.r.t the input (i.e. the Jacobian). In this case, our gradients would take the form:

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial A}
$$

We also want the shapes of these gradient terms to match, so $\frac{\partial L}{\partial \mathbf{y}}$ matches with the output $\mathbf{y} \in \mathbb{R}^{m}$, and $\frac{\partial L}{\partial A}$ is the same shape as $A \in \mathbb{R}^{m\times n}$. Thus our third order tensor, $\frac{\partial \mathbf{y}}{\partial A}$ behaves like row vector $\in \mathbb{R}^{1\times n}$ in an outer product. From our full Jacobian above, we can see that for each output $y_i$, its derivative with respect to the $i$th row of $A$ is $\mathbf{x}^T$. From this, the outer product conveniently simplifies to:

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial \mathbf{y}} \mathbf{x}^T
$$

This is known as the Vector-Jacobian product (VJP). Even though $\frac{\partial \mathbf{y}}{\partial A}$ is technically a third-order tensor, when computing gradients for backpropagation, we avoid forming the full tensor and can directly use $\frac{\partial \mathbf{y}}{\partial A} = \mathbf{x}^T \in \mathbb{R}^{1\times n}$. This simplified computation is what makes training deep neural networks computationally feasible. All modern DL frameworks utilise the VJP to simplify sparse higher order Jacobians. We have focused on the most complicated case, a useful exercise would be to see if the Jacobians described in previous sections can be similarly reduced by leveraging the VJP.

## Conclusion

Throughout this post, we've systematically broken down the essential matrix calculus operations that form the backbone of deep learning. Starting from familiar scalar operations, we've seen how vector and matrix operations can be understood by considering their component-wise behavior, leading to elegant and often sparse Jacobian matrices. These seemingly abstract mathematical constructs have direct practical implications.

Understanding these fundamentals helps explain why modern deep learning frameworks can efficiently handle networks with billions of parameters. The sparsity patterns we've identified in the Jacobians directly influence how these frameworks implement automatic differentiation. While we've focused on the basics here, these same principles extend to more complex operations like convolutions, attention mechanisms, and sophisticated optimssation techniques.

For practitioners looking to deepen their understanding, we recommend experimenting with these operations in isolation before tackling more complex architectures. While DL frameworks have essentially abstracted away the gradients for key neural network operations, practitioners should have a deep understanding of what is going on under the hood. The ability to reason about gradients and understand how they flow through a network remains an invaluable skill for debugging models, designing new architectures, and optimising training procedures.
