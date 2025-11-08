### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 30549f16-8156-4bcd-a7c2-51cf58a45afb
md"""

# MLE of multivariate Gaussian
#### CS5014 Machine Learning
##### Lei Fang

\



In this note, we derive the maximum likelihood estimator for multivariate Gaussian. Given $\{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots, \mathbf{x}^{(n)}\}$, assume $\mathbf{x}^{(i)} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$. Find the ML estimate of

$$\boldsymbol{\mu},\; \mathbf{\Sigma}$$

The log likelihood is:

$$\begin{align*}
 \mathcal{L}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) &=  \ln p(\{\mathbf{x}^{(i)}\}_1^n|\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \sum_{i=1}^n \ln \mathcal{N}(\mathbf{x}^{(i)}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) \\
 &= \sum_{i=1}^n -\frac{1}{2} \ln |\boldsymbol{\Sigma}| -\frac{1}{2}(\mathbf{x}^{(i)} -\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}^{(i)}-\boldsymbol{\mu}) -\frac{d}{2} \ln 2\pi \\
&= -\frac{n}{2} \ln |\boldsymbol{\Sigma}| - \frac{1}{2}\sum_{i=1}^n(\mathbf{x}^{(i)} -\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}^{(i)}-\boldsymbol{\mu}) +\text{const.}
\end{align*}$$

Now ready to compute the ML estimator


"""

# ╔═╡ d0a490c5-9b57-4364-baba-bed33a6fe7d0
md"""

### MLE for $\hat{\boldsymbol{\mu}}$


The MLE are defined as usual:

$$\hat{\boldsymbol{\mu}}, \hat{\boldsymbol{\Sigma}} \leftarrow \arg\max_{\boldsymbol{\mu}, \boldsymbol{\Sigma}} \mathcal{L}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$


Remember I have mentioned that quadratic form is multivariate generalisation of quadratic function: 

$\mathbf{x}^\top \mathbf{A x}$ is simular to ${x}\cdot a \cdot x$; their gradients are similar as well

$\frac{\partial xax}{\partial x} = \frac{\partial ax^2}{\partial x} = (a+a) x = 2ax$

The gradient of quadratic form is 

$\frac{\partial \mathbf{x}^\top \mathbf{A x}}{\partial \mathbf{x}} = (\mathbf{A}+\mathbf{A}^\top) \mathbf{x}   = 2 \mathbf{Ax}$

if we assume $\mathbf{A}$ is symmetric, then $\mathbf{A}^\top +\mathbf{A} = 2\mathbf{A}.$

Take derivative w.r.t $\boldsymbol{\mu}$ (notice it is a quadratic form w.r.t $\boldsymbol{\mu}$) and set it to zero:

$$\nabla_{\boldsymbol{\mu}} \mathcal{L} = -\frac{1}{2} \sum_{i=1}^n 2\cdot (-1)\cdot \boldsymbol{\Sigma}^{-1} (\mathbf{x}^{(i)}-\boldsymbol{\mu})= \sum_{i=1}^n \boldsymbol{\Sigma}^{-1}  (\mathbf{x}^{(i)} - \boldsymbol{\mu}) = \mathbf{0}$$
which leads to 


$$\begin{align}&\Rightarrow {\boldsymbol{\Sigma}}^{-1}  \sum_{i=1}^n  (\mathbf{x}^{(i)} - \boldsymbol{\mu}) = \mathbf{0} \\

&\Rightarrow \sum_{i}^n  (\mathbf{x}^{(i)} - \boldsymbol{\mu}) = \mathbf{0} \\
&\Rightarrow n\cdot \boldsymbol{\mu} =\sum_{i=1}^n  \mathbf{x}^{(i)}  \\
&\Rightarrow \hat{\boldsymbol{\mu}} = \frac{1}{n} \sum_{i=1}^n \mathbf{x}^{(i)}
\end{align}$$

where 
* the first step left multiplies $\boldsymbol{\Sigma}$ on both sides, $\boldsymbol{\Sigma} \boldsymbol{\Sigma}^{-1} =\mathbf{I}$

"""

# ╔═╡ e9f50c74-d29a-4fee-b2e8-d1cea1bd7656
md"""

### MLE for $\hat{\boldsymbol{\Sigma}}$

The same deal, we need to take derivative and then set the derivative to zero. 


We will use the trace trick here first to rewrite the log likelihood function;

Note that 

$\text{Tr}(\mathbf{X}) = \sum_{i} \mathbf{X}_{ii},$ the sum of the diagonal entries of a matrix; and also ``\text{Tr}(c) = c``, i.e. trace of a scalar is itself; and trace is a linear operator,

$$\text{Tr}\left (\sum_{i=1}^n  \mathbf{A}_i\right ) = \sum_{i=1}^n  \text{Tr}(\mathbf{A}_i),$$ and trace has a cyclic property:

$$\text{Tr}\left( \mathbf{ABC}\right )= \text{Tr}\left( \mathbf{CAB}\right )= \text{Tr}\left( \mathbf{BCA}\right)$$

We are ready to rewrite the log likelihood now:

$$\begin{align*}
 \mathcal{L}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) &= -\frac{n}{2} \log |\boldsymbol{\Sigma}| - \frac{1}{2}\sum_{i=1}^n (\mathbf{x}^{(i)} -\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}^{(i)}- \boldsymbol{\mu}) +\text{const.} \\
 &= -\frac{n}{2} \log |\boldsymbol{\Sigma}| - \frac{1}{2}\sum_{i=1}^n \text{Tr}\left ((\mathbf{x}^{(i)} -\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}^{(i)}- \boldsymbol{\mu})\right ) +\text{const.} \\
 &= -\frac{n}{2} \log |\boldsymbol{\Sigma}| - \frac{1}{2}\sum_{i=1}^n \text{Tr}\left (\boldsymbol{\Sigma}^{-1}(\mathbf{x}^{(i)}- \boldsymbol{\mu})(\mathbf{x}^{(i)} -\boldsymbol{\mu})^\top \right ) +\text{const.} \\
 &= -\frac{n}{2} \log |\boldsymbol{\Sigma}| - \frac{1}{2}\text{Tr}\left ( \boldsymbol{\Sigma}^{-1}\sum_{i=1}^n(\mathbf{x}^{(i)}- \boldsymbol{\mu})(\mathbf{x}^{(i)} -\boldsymbol{\mu})^\top \right ) +\text{const.}
\end{align*}$$

where 
* line two has used the property that trace of a scalar is itself; a quadratic form is a scalar; 
* line three: cyclic property
* line four: linear operator 

To make the notation less clutter, define the sum of squared error matrix (aka scatter matrix)

```math
\mathbf{S} = \sum_{i=1}^n(\mathbf{x}^{(i)}- \boldsymbol{\mu})(\mathbf{x}^{(i)} -\boldsymbol{\mu})^\top ,
```
The log likelihood becomes

$$\begin{align*}
 \mathcal{L}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) 
 &= -\frac{n}{2} \log |\boldsymbol{\Sigma}| - \frac{1}{2}\text{Tr}\left ( \boldsymbol{\Sigma}^{-1}\sum_{i=1}^n(\mathbf{x}^{(i)}- \boldsymbol{\mu})(\mathbf{x}^{(i)} -\boldsymbol{\mu})^\top \right ) +\text{const.}\\
&= -\frac{n}{2} \log |\boldsymbol{\Sigma}| - \frac{1}{2}\text{Tr}\left ( \boldsymbol{\Sigma}^{-1}\mathbf{S} \right ) +\text{const.}
\end{align*}$$

Check [matrix cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) for some useful matrix gradient identities:

$$\frac{\partial \ln |\mathbf{X}|}{\partial \mathbf{X}} = (\mathbf{X}^{-1})^\top,\;\;\frac{\partial \text{Tr}(\mathbf{AX}^{-1}\mathbf{B})}{\partial \mathbf{X}} = -(\mathbf{X}^{-1}\mathbf{BAX}^{-1})^\top$$

Then the gradient becomes:

$$\begin{align}
\nabla_{\boldsymbol{\Sigma}} \mathcal L = - \frac{n}{2} \left(\boldsymbol{\Sigma}^{-1}\right)^\top - \frac{1}{2}\left[- \left(\boldsymbol{\Sigma}^{-1} \mathbf{S}\boldsymbol{\Sigma}^{-1}\right)^\top\right ]
\end{align}$$

* where ``\mathbf{A} =\mathbf{I}``, and ``\mathbf{B}= \mathbf{S}``
"""



# ╔═╡ 5bb35fcc-181c-4bf1-acf8-92bf068004d7
md"""

Set the derivative to zero, we have 

$$\begin{align}
&- \frac{n}{2} \left(\boldsymbol{\Sigma}^{-1}\right)^\top - \frac{1}{2}\left[- \left(\boldsymbol{\Sigma}^{-1} \mathbf{S} \boldsymbol{\Sigma}^{-1}\right)^\top\right ] =\mathbf{0}\\

&\Rightarrow n \cdot \boldsymbol{\Sigma}^{-1} = \left(\boldsymbol{\Sigma}^{-1} \mathbf{S} \boldsymbol{\Sigma}^{-1}\right)\\
&\Rightarrow \boldsymbol{\Sigma} \cdot n =  \mathbf{S} \\
&\Rightarrow\boldsymbol{\Sigma} = \frac{1}{n} \mathbf{S}\\
&\Rightarrow \boldsymbol{\Sigma} = \frac{1}{n}  \sum_{i=1}^n (\mathbf{x}^{(i)}- \boldsymbol{\mu})(\mathbf{x}^{(i)} -\boldsymbol{\mu})^\top 
\end{align}$$

Some explanation due here: 
* the first step multiplies ``-2`` on both side and move the second term to the right hand side of the equation; and then take transpose on both side: i.e. $(\mathbf{A}^\top)^\top =\mathbf{A}$
* the second step multplies both left and right hand side with $\boldsymbol{\Sigma}$, note $\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma} = \boldsymbol{\Sigma}\boldsymbol{\Sigma}^{-1}= \mathbf{I}$


In summary, the MLE for multivariate Gaussians are 


$$\begin{equation*}
\hat{\boldsymbol{\mu}} = \frac{1}{n} \sum_{i=1}^n \mathbf{x}^{(i)}; \;\;\; 
\hat{\boldsymbol{\Sigma}}= \frac{1}{n} \sum_{i=1}^n (\mathbf{x}^{(i)}- \hat{\boldsymbol{\mu}})(\mathbf{x}^{(i)}-\hat{\boldsymbol{\mu}})^\top
\end{equation*}$$

- intuitive results: empirical sample mean and covariance are the estimators!
"""

# ╔═╡ 4a18116d-889c-4e7b-9d01-423c97e0db6c
# md"""


# **Weighted MLE** should be very similar. Typing latex is very painful. I will stop here and leave it as an exercise...

# """

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.1"
manifest_format = "2.0"
project_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

[deps]
"""

# ╔═╡ Cell order:
# ╟─30549f16-8156-4bcd-a7c2-51cf58a45afb
# ╟─d0a490c5-9b57-4364-baba-bed33a6fe7d0
# ╟─e9f50c74-d29a-4fee-b2e8-d1cea1bd7656
# ╟─5bb35fcc-181c-4bf1-acf8-92bf068004d7
# ╟─4a18116d-889c-4e7b-9d01-423c97e0db6c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
