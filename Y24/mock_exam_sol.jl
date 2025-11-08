### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 315c3954-e5a2-48d7-b12a-048f284d89c8
using PlutoUI

# ╔═╡ 336a1db9-2050-4528-8005-796f58c43c97
using PlutoTeachingTools

# ╔═╡ ca5301bd-9e3b-4e7f-8b5b-ca8ebb76e1d2
begin
	# using TikzGraphs
	# using Graphs
	using LaTeXStrings
end

# ╔═╡ 54e98c53-3c3e-454b-94d8-1f60cec1c67c
using HypertextLiteral

# ╔═╡ 30e0e1ef-c4a1-4a14-8a55-781974543a69
md"""

# CS5014 Mock Exam

\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 7ee3e2b0-69e9-11ee-3665-89484daabae1
md"""



## Question 1


###### Consider ``\mathbb{R}^2 \rightarrow \mathbb{R}`` function ``h(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + w_0``, where ``\mathbf{w} = [1, 2]^\top`` is known. Assume we are at $\mathbf{x}_0 = [1,1]^\top$;


* find the gradient expression $\nabla_\mathbf{x}h(\mathbf{x})$


* which direction to follow to decrease the function the most?


* one has repeated the following gradient descent step 100 times (with a learning rate of ``0.1``),
  
  $$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - 0.1 \cdot\nabla_\mathbf{x}h(\mathbf{x}_{t-1})$$
  * what is $\mathbf{x}_{100}$?
  * assume we continue the gradient descent step indefinitely, will the algorithm converge?

"""

# ╔═╡ c24431f1-ea80-4385-8746-b22a3c1f1fc6
Foldable("Solution", md"""
### Solution

The gradient is
$$\nabla_\mathbf{x}h(\mathbf{x})= \mathbf{w}  = \begin{bmatrix}1 \\ 2 \end{bmatrix}$$


The greatest descent direction is opposite the gradient direction: $-\nabla_\mathbf{x}h(\mathbf{x}_0) = -\begin{bmatrix}1 \\ 2 \end{bmatrix}$


Since the gradient is constant, we have 

$\mathbf{x}_{100} = \mathbf{x}_0 - 0.1 \times 100\cdot  \mathbf{w} = \begin{bmatrix}1\\ 1\end{bmatrix} - 10\begin{bmatrix}1 \\ 2 \end{bmatrix} = \begin{bmatrix}-9 \\ -19 \end{bmatrix}$

It will not converge. The gradient of a hyperplane will never vanish. In other words, a hyperplane has no local minimum. If implemented like this, the value will eventually become negative infinite ``-\infty``.
""")

# ╔═╡ 55e36373-cdc0-4922-852b-73646681c5db
md"""
## Question 2


Given bi-variate quadratic function 

$$h(\mathbf{x}) = \mathbf{x}^\top \begin{bmatrix}1 & 0 \\ 0 & 2 \end{bmatrix}\mathbf{x} + \mathbf{b}^\top\mathbf{x} + \ln 2^{10000},$$

where $\mathbf{b} = [1, 2]^\top$


* Find the gradient expression $\nabla_{\mathbf{x}}h(\mathbf{x})$


* Optimise $h$ directly and give an expression for the optimum $\mathbf{x}$ 


* Is the optimum a minimum or maximum?


* Given an initial guess $\mathbf{x}_0= [1,1]^\top$, if one uses Newton's method to minimise $h(\mathbf{x})$, will the algorithm converge? how many iterations are needed?


"""

# ╔═╡ ad422f41-2869-49b0-bbf2-6ff1a5c3268c
Foldable("Solution", md"""
The gradient is 

$$\nabla_{\mathbf{x}}h(\mathbf{x})=2 \begin{bmatrix}1 & 0 \\ 0 & 2 \end{bmatrix}\mathbf{x} +  \begin{bmatrix}1 \\ 2\end{bmatrix}=\begin{bmatrix}2 & 0 \\ 0 & 4 \end{bmatrix}\mathbf{x} +  \begin{bmatrix}1 \\ 2\end{bmatrix}$$


To optimise $h$, set the gradient to zero and solve it

$$\begin{align}&\;\;\;\;\;\;\nabla_{\mathbf{x}}h(\mathbf{x})=\begin{bmatrix}2 & 0 \\ 0 & 4 \end{bmatrix}\mathbf{x} +  \begin{bmatrix}1 \\ 2\end{bmatrix} =\mathbf{0}\\
&\Rightarrow \begin{bmatrix}2 & 0 \\ 0 & 4 \end{bmatrix}\mathbf{x}=-   \begin{bmatrix}1 \\ 2\end{bmatrix}\\
&\Rightarrow \mathbf{x}=- \begin{bmatrix}\frac{1}{2} & 0 \\ 0 & \frac{1}{4} \end{bmatrix} \begin{bmatrix}1 \\ 2\end{bmatrix} = - \begin{bmatrix}\frac{1}{2} \\ \frac{1}{2}\end{bmatrix}
\end{align}$$

It is a minimum, since $\begin{bmatrix}1 & 0 \\ 0 & 2 \end{bmatrix}$ is positive definite: 

$\mathbf{v}^\top\begin{bmatrix}1 & 0 \\ 0 & 2 \end{bmatrix}\mathbf{v} = v_1^2 + 2v_2^2 >0\;\; \text{for all }\mathbf{v}\neq \mathbf{0}$


It will converge in one step. Newton's method approximate a function locally with a quadratic and then optimise the approximation. And this function is a quadratic function; therefore, the approximation is exact. Newton's step essentially optimises the quadratic which leads to the solution in one step.

If you do not believe me, check further. Newton's step is

$$\mathbf{x}_1 \leftarrow \mathbf{x}_0 - \mathbf{H}^{-1}\nabla_{\mathbf{x}}h(\mathbf{x}_0)$$

where $\mathbf{H} =\begin{bmatrix}2 & 0 \\ 0 & 4 \end{bmatrix}$, therefore, after one iteration, the algorithm converges to the minimum straightaway:

$$\begin{align}\mathbf{x}_1 &= \mathbf{x}_0 - \begin{bmatrix}2 & 0 \\ 0 & 4 \end{bmatrix}^{-1}\left (\begin{bmatrix}2 & 0 \\ 0 & 4 \end{bmatrix}\mathbf{x}_0 +  \begin{bmatrix}1 \\ 2\end{bmatrix}\right)\\
&=\cancel{\mathbf{x}_0 - \mathbf{x}_0} -  \begin{bmatrix}2 & 0 \\ 0 & 4 \end{bmatrix}^{-1}\begin{bmatrix}1 \\ 2\end{bmatrix} \\
&=- \begin{bmatrix}\frac{1}{2} \\ \frac{1}{2}\end{bmatrix}\end{align}$$

And further updates have no effect:  since the gradient will be zero and $\mathbf{H}^{-1}\nabla_{\mathbf{x}}h(\mathbf{x}_t) =\mathbf{H}^{-1}\mathbf{0}=\mathbf{0}$.

Also note Newton's method would converge in one step, regardless of the choice of the initial guess $\mathbf{x}_0$.
""")

# ╔═╡ 22e48319-535d-49aa-9f04-5fe9a775e54d
md"""

## Question 3


Consider a regularised supervised learning problem. The training data is $$\{\mathbf{x}^{(i)}, y^{(i)}\}$$ and the loss is defined as 


$$\mathcal{L}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n \ell^{(i)}(\hat{y}^{(i)}, {y}^{(i)}) + \lambda\cdot  f(\mathbf{w}),$$

where $\ell^{(i)}$ is the $i$-th observation's loss, $\hat{y}^{(i)} = h(\mathbf{w}, \mathbf{x}^{(i)})$ is the supervised learning's model output, and $f(\mathbf{w})$ is some appropriate regularisation function.

* Write down $f(\mathbf{w})$'s expression for ridge regression
  * outline a learning algorithm to learn $\mathbf{w}$

* Write down $f(\mathbf{w})$'s expression for lasso regression
  * outline a learning algorithm to learn $\mathbf{w}$


* Assume the training process has finished and the learnt parameter is $\hat{\mathbf{w}}$. And you are asked to "attack" the trained model. That is to make a small change on $\mathbf{x}$ by adding a small pertubation $\Delta\mathbf{x}$:

  $$\mathbf{x}^\ast = \mathbf{x} + \Delta\mathbf{x}$$
  such that the model's prediction $\hat{y}^\ast = h(\hat{\mathbf{w}}, \mathbf{x}^\ast)$ deteriorates the most.  What $\Delta\mathbf{x}$ would you choose? 
"""

# ╔═╡ 58910ad4-8889-4217-b370-5f8068314497
aside(tip(md"""
This is known as *adversarial attack* in deep learning.
"""))

# ╔═╡ ec8a00f6-315f-4966-9aac-a98a7f217937
Foldable("Solution", md"""

For convenience, denote the un-penalised loss as 

$$\ell(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n\ell^{(i)}(\mathbf{w})$$


Ridge regression: $$f(\mathbf{w}) = \sum_{j=1}^m w_j^2$$
* learning with gradient descent: *i.e.* old gradient plus weight decay term: $2\lambda \cdot \mathbf{w}_t$,
* random guess $\mathbf{w}_0$, repeat until converge
$$\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma \cdot (\nabla \ell(\mathbf{w}_{t-1}) + 2 \lambda \mathbf{w}_{t-1})$$

Lasso regression: $$f(\mathbf{w}) = \sum_{j=1}^m |w_j|$$
* learning with gradient descent: old gradient plus sign of the gradient: $\lambda\cdot \texttt{sign}(\mathbf{w}_t)$ where the $\texttt{sign}$ is applied element wisedly
* random guess $\mathbf{w}_0$, repeat until converge
$$\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma \cdot (\nabla \ell(\mathbf{w}_{t-1}) + \lambda \texttt{sign}(\mathbf{w}_{t-1}))$$


The most sensible choice is the (scaled) gradient of the loss $\ell$ w.r.t to $\mathbf{x}$, *i.e.*

$\Delta\mathbf{x} = \epsilon \frac{\nabla \ell_{\mathbf{x}}(\mathbf{x})}{\|\nabla \ell_{\mathbf{x}}(\mathbf{x})\|_2}$

Why?
* first note that $\ell$ is also a function of $\mathbf{x}$

* we want to increase the loss (deteriorates the performance); if $\ell$ is differentiable, then the gradient direction will increases the loss the most locally at $\mathbf{x}$; 

* and the objective is to attack, we follow the gradient direction (rather than the opposite)

* lastly, we scale the gradient to a unit vector and scale it back with a budget $\epsilon$: which basically implies $\|\Delta \mathbf{x}\|_2 = \epsilon$, which serves the same purpose as a learning rate
""")

# ╔═╡ 29f3b258-82e8-411f-9758-1337d8d6f3c5
md"""

## Question 4*


ResNets are very popular neural network architecture for deep neural networks. They usually achieve better results than ordinary neural networks with dense layers. ResNets consists of a sequence of blocks named `residual layer` and a `residual layer` forward computation is

```math
\begin{align}
\mathbf{z}_1 &= \mathbf{W}_1\mathbf{z}_0 + \mathbf{b}_1 \\
\mathbf{a}_1 &= \text{relu}\odot(\mathbf{z}_1)\\
\mathbf{z}_2 &= \mathbf{W}_2\mathbf{a}_1 + \mathbf{b}_2 \\
\mathbf{z}_3 &=  \mathbf{z}_0 + \mathbf{z}_2
\end{align}
```

where the input signal is $\mathbf{z}_0$ and the final output of the block is $\mathbf{z}_3$, the learnable parameters are $$\{\mathbf{W}_1, \mathbf{W}_2, \mathbf{b}_1, \mathbf{b}_2\}$$

* Draw the associated computational acyclic graph for the forward computation


* Given adjoint $$\bar{\mathbf{z}}_3 = \nabla_{\mathbf{z}_3} \ell$$ (*i.e.* the gradient of final loss $\ell$ w.r.t. ${\mathbf{z}}_3$), outline how reverse-mode auto-differentiation (aka backpropagation) can be used to compute the gradients of the learnable parameters (*i.e.* provide the backward pass steps of the backpropagation algorithm)


* Based on your backward pass steps, explain why `residual block` should be preferred than the regular `dense layer` for deep networks
"""

# ╔═╡ 5a48b927-06f2-4623-889c-96844643dfe1
Foldable("Solution (Backprop)", md"""



Backward pass (note that $\mathbf{z}_0$ folks out; its gradient should sum both path):

```math
\begin{align}
\bar{\mathbf{z}}_2 &= \bar{\mathbf{z}}_3 \\
\bar{\mathbf{W}}_2 &= \bar{\mathbf{z}}_2 \mathbf{a}_1^\top \\
\bar{\mathbf{a}}_1 &= \mathbf{W}_2^\top\bar{\mathbf{z}}_2 \\
\bar{\mathbf{b}}_2 &= \bar{\mathbf{z}}_2 \\

\bar{\mathbf{z}}_1 &= \mathbb{1}(\mathbf{z}_1 > \mathbf{0}) \odot  \bar{\mathbf{a}}_1 \tag{element wise *} \\
\bar{\mathbf{W}}_1 &= \bar{\mathbf{z}}_1 \mathbf{z}_0^\top \\
\bar{\mathbf{b}}_1 &= \bar{\mathbf{z}}_1 \\
\bar{\mathbf{z}}_0 &= \bar{\mathbf{z}}_3 + \mathbf{W}_1^\top \bar{\mathbf{z}}_1
\end{align}
```


A simple dense linear layer (if we get ride of the extra operations) is 

```math
\begin{align}
\mathbf{z}_1 &= \mathbf{W}_1\mathbf{z}_0 + \mathbf{b}_1 \\
\mathbf{z}_3 &= \text{relu}\odot(\mathbf{z}_1)\\
\end{align}
```

And the backward pass is 

```math
\begin{align}
\bar{\mathbf{z}}_1 &= \mathbb{1}(\mathbf{z}_1 > \mathbf{0}) \odot \bar{\mathbf{z}}_3  \\
\bar{\mathbf{z}}_0 &= \mathbf{W}_1^\top \bar{\mathbf{z}}_1
\end{align}
```
Compared with `Residual layer`'s gradient, the addition operation of the residual layer add the incoming gradient $\bar{\mathbf{z}}_3$ on top, which allows deeper layers' gradient pass through **directly** without multiplying any weight as a shortcut, which solves the gradient vanish problem (if you multiply layers of weights, if local gradients are $-1< w<1$, then the gradient will vanish eventually if we have a lot of layers.)

""")

# ╔═╡ 56a7f6c7-08ea-4234-a1be-e5f37b742c2c
md"""

## Question 5


* Outline the EM algorithm for finite mixture of Gaussians


* Assume you are told that the covariance matrices are all identity matrices, that is $\mathbf{\Sigma}_k =\mathbf{I}$, simplify and outline the updated EM algorithm


* Furthermore, assume a hard E-step is used in the E-step, that is 
  
  $$r_{ik} \leftarrow \begin{cases}1 & k = \arg\max_{k'} p(z^{(i)}=k'|\mathbf{x}^{(i)}) \\0 & \text{otherwise}\end{cases},$$
  update your EM algorithm further.
"""

# ╔═╡ f10dc34b-3fa8-4575-9558-a1d7b70e37db
Foldable("Solution", md"""



Check lecture slides.



Since the covariances are $\mathbf{I}$, the Gaussian's density reduces to Euclidean distance: the Gaussian's kernel is $-\frac{1}{2}(\mathbf{x} -\boldsymbol{\mu}_k)^\top \mathbf{I}^{-1}(\mathbf{x} -\boldsymbol{\mu}_k) = -\frac{1}{2}\|\mathbf{x} -\boldsymbol{\mu}_k\|_2^2$ 

And the E step becomes 

$$r_{ik} \propto  \pi_k\cdot \exp \left(- \frac{1}{2} \|\mathbf{x}^{(i)} -\boldsymbol{\mu}_k\|_2^2\right)$$

M step is simplied accordingly as well (just remove the reestimation for $\Sigma$);



The hard E step becomes 

$$r_{ik} \leftarrow \begin{cases}1 & k = \arg\max_{k'}\left\{ \ln\pi_k - \frac{1}{2} \|\mathbf{x}^{(i)} -\boldsymbol{\mu}_k\|_2^2\right \} \\0 & \text{otherwise}\end{cases},$$

if we further assume $\pi_k = \frac{1}{K}$, i.e. uniform distributed prior, then 

$$r_{ik} \leftarrow \begin{cases}1 & k = \arg\min_{k'} \|\mathbf{x}^{(i)} -\boldsymbol{\mu}_k\|_2^2 \\0 & \text{otherwise}\end{cases},$$

the algorithm reduces to K-means!
""")

# ╔═╡ 82e954b6-559d-47e0-a6f3-4a3d0631a312
md"""

## Question 6


You are given a binary classification dataset collected by an unreliable politician. The politician is careless and often drunk. He made a lot of mistakes while documenting the data. In particular, some random fraction of the targets $y^{(i)} \in \{0,1\}$ has been replaced with random guesses instead of the true targets. Unfortunately, you do not know which of the targets have been corrupted. The input features vectors $\{\mathbf{x}^{(i)}\}$, however, are recorded correctly without error. You may assume the randomly guessed label is 50/50 split between label 0 and label 1.  You may also assume roughly $\epsilon$ of the data is corrupted but we do not know the percentage.


For each observation $y^{(i)}$, introduce a latent binary variable $z^{(i)} \in \{0, 1\}$ to mark whether the observation is a random guess. That is when $z^{(i)} = 1$, the politician has randomly guessed $y^{(i)}$; otherwise, he has copied $y^{(i)}$ correctly.

Assume we want to learn the model parameter $\mathbf{w}, \epsilon$ based on $\{\mathbf{x}^{(i)}, y^{(i)}\}$, where $\mathbf{w}$ is the un-contaminated logistic regression's parameter.

1. Find an expression for the complete likelihood: $p(y^{(i)}, z^{(i)}|\mathbf{w}, \epsilon)$;


2. Show the marginal distribution $p(y^{(i)}|\mathbf{w}, \epsilon)$ is a finite mixture model (where the mixture size $K=2$)


3. Find the complete joint log likelihood $\ln p(\{y^{(i)}, z^{(i)}\}|\mathbf{w}, \epsilon)$


4. Outline an EM algorithm to learn the parameter
"""

# ╔═╡ 0b6cc0dd-ea27-49f5-898a-52312bae0797
Foldable("Solution", md"""




The complete likelihood is (just apply product rule):

$p(y^{(i)}, z^{(i)}|\mathbf{w}, \epsilon) = p(z^{(i)}|\cancel{\mathbf{w}}, \epsilon)p(y^{(i)}|z^{(i)}, \mathbf{w}, \epsilon)$

where the prior for $z^{(i)}$ is a Bernoulli

$\begin{align}p(z^{(i)}|\epsilon) &= \begin{cases}1-\epsilon & z^{(i)} = 0 \\\epsilon &z^{(i)}=1\end{cases} \\
&= \epsilon^{z^{(i)}} (1-\epsilon)^{1-z^{(i)}}\end{align}$

and the likelihood is 

$\begin{align}p(y^{(i)}|z^{(i)}, \mathbf{w}, \epsilon) &=\begin{cases}\mathcal{Ber}(y^{(i)}; 0.5) & z^{(i)} = 1\\\mathcal{Ber}(y^{(i)}; \sigma^{(i)}) & z^{(i)} = 0\end{cases} \\
&= \mathcal{Ber}(y^{(i)}; 0.5)^{z^{(i)}}\mathcal{Ber}(y^{(i)}; \sigma^{(i)})^{1-z^{(i)}}\end{align}$

* where $\sigma^{(i)} = \frac{1}{1+e^{-\mathbf{w}^\top\mathbf{x}^{(i)}- w_0}}$ is the ordinary logistic regression
* it basically implies, if $z=1$, $y$ is a random guess (a Bernoulli with bias $.5$); otherwise, it is a Bernoulli with bias specified as logistic regression's output $\sigma^{(i)}$


Due to sum rule, the marginal is mixture of two Bernoulli distributions 


$\begin{align}p(y^{(i)}|\mathbf{w}, \epsilon) &= \sum_{z^{(i)}=0,1}p(y^{(i)}, z^{(i)}|\mathbf{w}, \epsilon) \\
&=\sum_{z^{(i)}=0,1}p(z^{(i)}| \epsilon)p(y^{(i)}|z^{(i)}, \mathbf{w}, \epsilon)\\
&=  (1-\epsilon)\cdot\mathcal{Ber}(y^{(i)}; \sigma^{(i)}) + \epsilon\cdot \mathcal{Ber}(y^{(i)}; 0.5)
\end{align}$


The complete log likelihood of $i$-th observation is 

$\begin{align}\ln &p(y^{(i)}, z^{(i)}|\mathbf{w}, \epsilon) =  \ln p(z^{(i)}| \epsilon)+\ln p(y^{(i)}|z^{(i)}, \mathbf{w}, \epsilon)\\
&=\ln \left(\epsilon^{z^{(i)}} (1-\epsilon)^{1-z^{(i)}}\right ) + \ln \left(\mathcal{Ber}(y^{(i)}; 0.5)^{z^{(i)}}\mathcal{Ber}(y^{(i)}; \sigma^{(i)})^{1-z^{(i)}}\right) \\
&= z^{(i)} \ln \epsilon + (1-z^{(i)}) \ln (1-\epsilon) + z^{(i)}\ln \mathcal{Ber}(y^{(i)}; 0.5) + (1-z^{(i)})\ln \mathcal{Ber}(y^{(i)}; \sigma^{(i)})\\
&= z^{(i)} \left\{\ln\epsilon +\ln \mathcal{Ber}(y^{(i)}; 0.5)   \right\} + (1-z^{(i)}) \left \{\ln (1-\epsilon) + \ln \mathcal{Ber}(y^{(i)}; \sigma^{(i)})\right\}
\end{align}$

The joint log complete likelihood, due to independence, is the sum 

$\begin{align}\ln &p(\{y^{(i)}, z^{(i)}\}|\mathbf{w}, \epsilon)=\ln \left\{\prod_{i=1}^n p(y^{(i)}, z^{(i)}|\mathbf{w}, \epsilon)\right\}\\
&=\sum_{i=1}^n\ln p(y^{(i)}, z^{(i)}|\mathbf{w}, \epsilon)\\
&=\sum_{i=1}^n\left\{z^{(i)} \left(\ln\epsilon +\ln \mathcal{Ber}(y^{(i)}; 0.5)   \right) + (1-z^{(i)}) \left (\ln (1-\epsilon) + \ln \mathcal{Ber}(y^{(i)}; \sigma^{(i)})\right) \right\}
\end{align}$

""")

# ╔═╡ b995fb4a-11a8-4db4-9d08-3daac84b0e00
Foldable("Solution (EM)", md"""
#### EM algorithm

You can either derive the EM algorithm based on the fact that the model is a finite mixture model or follow the EM algorithm's steps from scratch. 


##### Based on the mixture model directly

For a mixture model, E step computes the responsiblity vector, for $i=1,\ldots, n$; $k=0,1$


$$r_{i1} \leftarrow p(z^{(i)}=1|y^{(i)}, \mathbf{x}^{(i)}, \epsilon, \mathbf{w})\propto \epsilon \cdot \mathcal{Ber}(y^{(i)}; 0.5)$$

$$r_{i0} \leftarrow p(z^{(i)}=0|y^{(i)}, \mathbf{x}^{(i)}, \epsilon, \mathbf{w})\propto (1-\epsilon) \cdot \mathcal{Ber}(y^{(i)}; \sigma^{(i)})$$

To be more specific,

$r_{i1} =\frac{\epsilon \cdot 0.5 }{\epsilon \cdot 0.5 + (1-\epsilon)\cdot \mathcal{Ber}(y^{(i)}; \sigma^{(i)})};\;\; r_{i0} = 1-r_{i1}$
* basically, for each observation, we calculate how likely, $r_{i1}$, $y^{(i)}$ is a random guess based on the current parameter 
M-step, reestimate the parameters (weighted MLE)

$\epsilon \leftarrow \frac{\sum_{i=1}^n r_{i1}}{n}$

$\mathbf{w} \leftarrow \arg\max_{\mathbf{w}}\sum_{i=1}^n r_{i0} \cdot \ln \mathcal{Bern}(y^{(i)}; \sigma^{(i)})$


* it makes sense, for each observation, if we believe it is a good observation, then $r_{i0}$ will be close to 1, it is given a higher weight; otherwise, if $r_{i0} \rightarrow 0$, it is a random guess, we ignore it in $\mathbf{w}$'s estimation.

* we can use gradient descent to re-estimate $\mathbf{w}$, the gradient should be (just weighted sum of the logistic regression's gradient)

$$\sum_{i=1}^n r_{i0}\cdot  (\sigma^{(i)} - y^{(i)})\mathbf{x}^{(i)}$$



EM algorithm will repeat the E & M step until convergence. 
""")

# ╔═╡ 3e206c68-1ca1-48c5-8f5e-9a9908b454ac
Foldable("EM from first principle*", md"""



##### Alternatively, EM from scratch


EM algorithm aims at optimising the expectation of the complete log-likelihood function. Denoting the model parameter $\theta\triangleq\{\epsilon, \mathbf{w}\}$, the function that EM aims to optimise is 


$$\small\begin{align}Q(\theta, \theta_t) &= \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z}|\mathbf{y}, \theta_t)}\left[\ln p(\{y^{(i)}, z^{(i)}\}|\mathbf{w}, \epsilon)\right] \\
&=\mathbb{E}_{\mathbf{z} \sim p(\mathbf{z}|\mathbf{y}, \theta_t)}\left[\sum_{i=1}^n\left\{z^{(i)} \left(\ln\epsilon +\ln \mathcal{Ber}(y^{(i)}; 0.5)   \right) + (1-z^{(i)}) \left (\ln (1-\epsilon) + \ln \mathcal{Ber}(y^{(i)}; \sigma^{(i)})\right) \right\}\right]\\
&=\sum_{i=1}^n\left\{\mathbb{E}_{z^{(i)} \sim p(z^{(i)}|y^{(i)}, \mathbf{w}_t, \epsilon_t)}[z^{(i)}] \left(\ln\epsilon +\ln \mathcal{Ber}(y^{(i)}; 0.5)   \right) \right.  \\
&\;\;\;\;\;\;\;\;\;\;\;\;\; \left. + (1-\mathbb{E}_{z^{(i)} \sim p(z^{(i)}|y^{(i)}, \mathbf{w}_t, \epsilon_t)}[z^{(i)}]) \left (\ln (1-\epsilon) + \ln \mathcal{Ber}(y^{(i)}; \sigma^{(i)})\right) \right\}
\end{align}$$

where we have used expectation's linearity property.
Note that the expectation is just the "responsiblities"

$\begin{align}\mathbb{E}_{z^{(i)} \sim p(z^{(i)}|y^{(i)}, \mathbf{w}_t, \epsilon_t)}[z^{(i)}] &=1 \cdot p(z^{(i)}=1|y^{(i)}, \mathbf{w}_t, \epsilon_t) + 0 \cdot p(z^{(i)}=0|y^{(i)}, \mathbf{w}_t, \epsilon_t) \\&=p(z^{(i)}=1|y^{(i)}, \mathbf{w}_t, \epsilon_t)\\
&\triangleq r_{i1}\end{align}$

and we denote $r_{i0} = p(z^{(i)}=0|y^{(i)}, \mathbf{w}_t, \epsilon_t)= 1-r_{i1}$

Therefore, the surrogate function becomes (which completes the E-step)

$\begin{align}
Q(\theta, \theta_t) &=\sum_{i=1}^n\left\{r_{i1} \left(\ln\epsilon +\ln \mathcal{Ber}(y^{(i)}; 0.5)   \right) + (1-r_{i1}) \left (\ln (1-\epsilon) + \ln \mathcal{Ber}(y^{(i)}; \sigma^{(i)})\right) \right\}\\
&=\sum_{i=1}^n\left\{r_{i1} \left(\ln\epsilon +\ln \mathcal{Ber}(y^{(i)}; 0.5)   \right) + r_{i0} \left (\ln (1-\epsilon) + \ln \mathcal{Ber}(y^{(i)}; \sigma^{(i)})\right) \right\}
\end{align}$


The M-step aims at optimising $Q$:

$\theta_{t+1} \leftarrow \arg\max_{\theta}Q(\theta, \theta_t);$


Take derivative w.r.t $\epsilon$ and solve it for zero, we have

$$\begin{align}\frac{dQ}{d\epsilon} &= \sum_{i=1}^n \left\{r_{i1} \frac{1}{\epsilon} - r_{i0}\frac{1}{1-\epsilon}\right\} =0

\Rightarrow (1-\epsilon)\sum_{i=1}^n r_{i1} - \epsilon\sum_{i=1}^n r_{i0} =0\end{align}$$

$$\Rightarrow \epsilon = \frac{\sum_{i=1}^n r_{i1}}{\sum_{i=1}^n r_{i1} + \sum_{i=1}^n r_{i0}}=\frac{\sum_{i=1}^n r_{i1}}{n}$$
* where we have used the fact that $r_{i0}=1-r_{i1}$



To optimise $\mathbf{w}$, we first isolate the term that is related to $\mathbf{w}$ first from $Q$, since all the un-related terms are constant from $\mathbf{w}$'s perspective. Based on the model, only $\sigma^{(i)} = \frac{1}{1+e^{-\mathbf{w}^\top\mathbf{x}^{(i)}}}$ depends on $\mathbf{w}$ (we assume the bias $w_0$ is rolled into $\mathbf{w}$ by introducing dummy one). Therefore,
$\begin{align}
Q(\mathbf{w}) 
&=\sum_{i=1}^n\left\{\cancel{r_{i1} \left(\ln\epsilon +\ln \mathcal{Ber}(y^{(i)}; 0.5)   \right) }+ r_{i0} \left (\cancel{\ln (1-\epsilon)} + \ln \mathcal{Ber}(y^{(i)}; \sigma^{(i)})\right) \right\}\\
&=\sum_{i=1}^nr_{i0} \ln \mathcal{Ber}(y^{(i)}; \sigma^{(i)})
\end{align}$

Optimising the above is the same as solving a weighted MLE problem, where the weights are $r_{i0}$:

$$\begin{align}\mathbf{w} &\leftarrow \arg\max_{\mathbf{w}}\sum_{i=1}^nr_{i0} \ln \mathcal{Ber}(y^{(i)}; \sigma^{(i)})\\
&= \arg\min_{\mathbf{w}}-\sum_{i=1}^nr_{i0} \ln \mathcal{Ber}(y^{(i)}; \sigma^{(i)})\\
&=\arg\min_{\mathbf{w}} - \sum_{i=1}^n r_{i0} \left\{ y^{(i)} \ln \sigma^{(i)} + (1-y^{(i)}) \ln(1-\sigma^{(i)}) \right\}
\end{align}$$

* we recognise the above is a weighted cross entropy loss, we resort to gradient descent to optimise the parameter

* the gradient should be the weighted version of the odinary logistic regression's gradient

$$\sum_{i=1}^n r_{i0}\cdot  (\sigma^{(i)} - y^{(i)})\mathbf{x}^{(i)}$$


""")

# ╔═╡ b5e975c7-f576-4cfc-a4a9-932315375b7d
begin
	# using HypertextLiteral
	
	function show_img(path_to_file; center=true, h = 400, w = nothing)
		if center
			if isnothing(w)
				@htl """<center><img src= $(figure_url * path_to_file) height = '$(h)' /></center>"""
			else
				@htl """<center><img src= $(figure_url * path_to_file) width = '$(w)' /></center>"""
			end
	
		else
			if isnothing(w)
				@htl """<img src= $(figure_url * path_to_file) height = '$(h)' />"""
			else
				@htl """<img src= $(figure_url * path_to_file) width = '$(w)' />"""
			end
		end
	end;
	figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";
end;

# ╔═╡ 6e6ee623-9063-41db-8ff2-f0cb8de56fb6
Foldable("Solution (DAG)", md"""

$(show_img("CS5914/backprop/res_block.svg", w=750))
""")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.0"
PlutoTeachingTools = "~0.2.14"
PlutoUI = "~0.7.58"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "c5364c6df1753eb766a46694d2907eda18845041"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "2984284a8abcfcc4784d95a9e2ea4e352dd8ede7"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.36"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "260dc274c1bc2cb839e758588c63d9c8b5e639d1"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.0.5"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "5d9ab1a4faf25a62bb9d07ef0003396ac258ef1c"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.15"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "7f4228017b83c66bd6aa4fddeb170ce487e53bc7"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.6.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─30e0e1ef-c4a1-4a14-8a55-781974543a69
# ╟─315c3954-e5a2-48d7-b12a-048f284d89c8
# ╟─336a1db9-2050-4528-8005-796f58c43c97
# ╟─7ee3e2b0-69e9-11ee-3665-89484daabae1
# ╟─c24431f1-ea80-4385-8746-b22a3c1f1fc6
# ╟─ca5301bd-9e3b-4e7f-8b5b-ca8ebb76e1d2
# ╟─55e36373-cdc0-4922-852b-73646681c5db
# ╟─ad422f41-2869-49b0-bbf2-6ff1a5c3268c
# ╟─22e48319-535d-49aa-9f04-5fe9a775e54d
# ╟─58910ad4-8889-4217-b370-5f8068314497
# ╟─ec8a00f6-315f-4966-9aac-a98a7f217937
# ╟─29f3b258-82e8-411f-9758-1337d8d6f3c5
# ╟─6e6ee623-9063-41db-8ff2-f0cb8de56fb6
# ╟─5a48b927-06f2-4623-889c-96844643dfe1
# ╟─56a7f6c7-08ea-4234-a1be-e5f37b742c2c
# ╟─f10dc34b-3fa8-4575-9558-a1d7b70e37db
# ╟─82e954b6-559d-47e0-a6f3-4a3d0631a312
# ╟─0b6cc0dd-ea27-49f5-898a-52312bae0797
# ╟─b995fb4a-11a8-4db4-9d08-3daac84b0e00
# ╟─3e206c68-1ca1-48c5-8f5e-9a9908b454ac
# ╟─54e98c53-3c3e-454b-94d8-1f60cec1c67c
# ╟─b5e975c7-f576-4cfc-a4a9-932315375b7d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
