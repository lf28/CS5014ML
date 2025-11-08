### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 9f90a18b-114f-4039-9aaf-f52c77205a49
begin
	using LinearAlgebra
	using PlutoUI
	using PlutoTeachingTools
	using LaTeXStrings
	using Latexify
	using Random
	using Statistics
	using HypertextLiteral
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	
end

# ╔═╡ 01585427-e485-4421-8590-1c1964f8bb28
using Symbolics

# ╔═╡ c34e56cb-5a92-4282-aa62-4c11deaff0fd
using Zygote

# ╔═╡ fd39b942-5066-4ae9-803d-112001a10e0b
using StatsBase

# ╔═╡ 3e2e1ea8-3a7d-462f-ac38-43a087907a14
TableOfContents()

# ╔═╡ 275e626f-026b-4167-acaa-d9faa590bed7
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ 7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
ChooseDisplayMode()

# ╔═╡ bc96a33d-9011-41ec-a19e-d472cbaafb70
md"""

# CS5014 Machine Learning


#### Auto-differentiation


\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 8e98d5c2-3744-49b8-95f8-d832c1b28632
md"""

## Reading & references

##### Suggested reading 


* [_Automatic Differentiation in Machine Learning: a Survey_ by _Baydin A. et al_:](https://arxiv.org/abs/1502.05767)

"""

# ╔═╡ 42952923-cd9b-4a05-a439-ee382d4466c3
md"""

## Recap: gradient descent



#### Gradient descent solves an optimisation problem

$$\Large \mathbf{w}\leftarrow \arg\min_{\mathbf{w}} \mathcal{L}(\mathbf{w}; \{\mathbf{x}^{(i)}, y^{(i)}\})$$


* ##### it works by moving opposite the gradient direction iteratively

```math
\Large
\boxed{\mathbf{w}_{new} \leftarrow \mathbf{w}_{old} - \gamma\cdot \nabla \mathcal{L}(\mathbf{w}_{old}; \{\mathbf{x}^{(i)}, y^{(i)}\})}
```


* ##### and the algorithm requires the gradient 

$\large \nabla_{\mathbf{w}} \mathcal{L} = \begin{bmatrix}
\frac{\partial \mathcal{L} }{\partial w_1}\\
\frac{\partial \mathcal{L} }{\partial w_2}\\
\vdots\\

\frac{\partial \mathcal{L} }{\partial w_m}
\end{bmatrix}$



"""

# ╔═╡ 3747dda8-b8ae-49f5-9b08-0508abedc4a6
md"""

## Univariate chain rule

##### Composite function, denoted as ``f_2 \circ f_1`` 

```math
\Large 
(f_2 \circ f_1) (x) \triangleq f_2(f_1(x))
```



"""

# ╔═╡ 1a4d3572-438d-4295-a4a3-ca603cc8d238
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/chainrulefwd.svg' width = '400' /></center>"

# ╔═╡ 2076ccbd-01a2-48a3-898d-2a464f1a20b6
md"""


* ##### the derivative (by chain rule)
  * multiplication of the local gradients 

```math
\large
\frac{d (f_2 \circ f_1)}{dx} = \frac{d f_2}{d f_1} \frac{d f_1}{d x}
```



"""

# ╔═╡ 6e3892f0-0ddf-4367-b6b3-88e573f62b09
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/chainrulebwd2.svg' width = '400' /></center>"

# ╔═╡ 3c51d1e9-dfc1-44fc-a877-3e454de06736
md"""

## (Multivariate) chain rule 

#### A variable forks out (used multiple places)
* ##### terminology in directed acyclic graph: fan-out ``>1``


```math
\Large 
f(x(t), y(t))
```


"""

# ╔═╡ 963dd96b-94f3-4bc0-ab20-4ce8ee9f1e37
md"""


* ##### multivariate chain rule
  * ###### add all paths' gradients

```math
\large
\frac{d f}{dt} = \underbrace{\frac{dx}{dt}\frac{\partial f}{\partial x}}_{\text{path 1}} + \underbrace{\frac{dy}{dt}\frac{\partial f}{\partial y}}_{\text{path 2}}
```



"""

# ╔═╡ 01d0df22-a170-41af-99fa-137f477d1a58
md"""

## (Multivariate) chain rule : general case


#### More generally, consider a $$\mathbb{R}^n \rightarrow \mathbb{R}^m$$ function

$$\Large\mathbf{y}_{m\times 1} = f(\mathbf{x}_{n\times 1})$$

* if $\mathbf{x}, \mathbf{y}$ is (are) matrix(cies), we stack the columns to form a big vector
"""

# ╔═╡ 6a17976f-e242-446f-a631-849663486ffe
md"""

#### The partials $$\partial y_j/\partial x_i$$ for $i =1\ldots n; j = 1,\ldots, m$

* ###### the collection of the $n\times m$ partials is called `jacobian matrix`
$$\large\frac{\partial \mathbf{y}}{\partial \mathbf{x}} =\begin{bmatrix}\frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \ldots & \frac{\partial y_1}{\partial x_n} \\
\frac{\partial y_2}{\partial x_1} & \frac{\partial y_2}{\partial x_2} & \ldots & \frac{\partial y_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots\\
\frac{\partial y_m}{\partial x_1} &\frac{\partial y_m}{\partial x_2}& \ldots & \frac{\partial y_m}{\partial x_n}\\
\end{bmatrix}_{m\times n}$$


* ##### the partial is the sum of all hightlighted gradient paths
"""

# ╔═╡ d09cbf01-a61c-4a49-990e-ef4a6eaca12c
md"""
## What is auto-differentiation?


#### Auto-differentiation: software computes $$\nabla_{\mathbf{w}} \mathcal{L}$$


* ##### input: a function that computes $\mathcal{L}$, *e.g.*

$$\begin{align}
\mathcal{L} = \frac{1}{2}(\mathbf{w}^\top\mathbf{x} - y)^2 +\frac{\lambda}{2}\mathbf{w}^\top
\mathbf{w}\tag{regularised sse error}
\end{align}$$


* ##### output: the gradient (a function) $\nabla \mathcal{L}(\mathbf{w})$

* ##### a modularised and efficient implementation
"""

# ╔═╡ 12fe101b-f602-4570-98a8-6362098662b2
md"""

## Auto-diff is NOT: Finite Differences

* #### One-sided finite difference method

```math
\large
\frac{\partial f(\mathbf{x})}{\partial x_i} \approx \frac{f(\mathbf{x}+ \epsilon \cdot \mathbf{e}_i) - f(\mathbf{x})}{\epsilon}
```

* #### Two-sided central difference method (more accurate)

```math
\large
\frac{\partial f(\mathbf{x})}{\partial x_i} \approx \frac{f(\mathbf{x}+ \epsilon \cdot \mathbf{e}_i) - f(\mathbf{x} - \epsilon \mathbf{e}_i)}{2\epsilon}
```

"""

# ╔═╡ 70aae3d9-de8d-411e-8eac-759092e474e3
md"""

## Auto-diff is NOT: Finite Differences (cont.)


* #### Autodiff is not finite differences
  * ###### finite differences are expensive (forward pass for _each_ $x_i$ twice)
  * ###### truncation and rounding errors
  * ###### only used for testing


* #### Autodiff: more efficient and numerically stable
"""

# ╔═╡ d1076724-3e36-4801-aef3-ec19f3bcce72
md"""

## Auto-diff is NOT: Symbolic differentiation


* #### Autodiff is not `Symbolic differenetiation` (e.g. `Mathematica`)

  * ##### `Symbolic diff` problem: "expression swelling"

  * ##### _e.g._ consider  logistic regression: ``\sigma(x) = \frac{1}{1+e^{-wx-b}}``


"""

# ╔═╡ 96e85812-0f24-476b-8b1c-5c443e80852f
md"""
* ##### If we apply two more linear layer + `sigmoid` activation, the `symbolic diff` returns (note that this is only one partial!)
"""

# ╔═╡ 69f6e60e-6c75-4be1-b718-259ca0cfe97b
md"""


* ##### The goal of autodiff is not an expression of the gradient, but a procedure for computing the gradient
"""

# ╔═╡ 9444fffc-57ce-4a38-af09-ceaf25100da8
md"""

## Auto-diff: two modes

#### Forward mode auto-diff 
  * ##### easier to implement than reverse mode
  * ##### slow for functions: $\mathbb{R}^n \rightarrow \mathbb{R}^m$ ($n > m$)


#### Reverse mode auto-diff 
  * ##### efficient for many to one function: $\mathbb{R}^n \rightarrow \mathbb{R}^m$ ($n > m$), (e.g. for most ML models, the loss is a scalar, *i.e.* ``m=1``)
    
  * ##### more widely used in ML application, also known as backpropagation
"""

# ╔═╡ dac1bf8b-e538-4282-ba0c-cca030ecc5aa
md"""

## Confusing terminologies


* #### `Autodiff`: a general way of computing gradients given a function
  * ###### quite likely with the help of graph

* #### `Backpropagation`: a special case of `reverse-mode` auto-diff applied to neural nets


* #### `Autgrad` (`Python`), `Zygote.jl` (`Julia`) are particular `autodiff` software packages
  * more packages exisit: `ForwardDiff.jl`, `PyTorch`'s grad

"""

# ╔═╡ 094f8059-38c5-4ddf-b5c1-a0d502d8fca7
md"""
## Computational graph

#### Computations can be represented as a `directed acyclic graph` (DAG)




#### For example,  to compute

$$\Large y = e^{-\sin(x^2)+5}$$

* it is broken down into 

```julia
t1 = x^2
t2 = sin(t1)
t3 = - t2
t4 = t3 + 5
y = e^t4
```

* a graph can be explicitly constructed by tracing the intermediate variables
"""

# ╔═╡ 2dfcc44b-40e9-4aef-8b09-44ec39cf6f35
md"""

## More example 
"""

# ╔═╡ b3dcb542-e1d2-45e0-b9c1-99a544c930e8
md"""

## Why graph?


#### We are computer scientists, and we know data structures well!
* a DAG is a common data structure 


#### Local gradients of `primitive` operations can be mechanically computed
* *e.g.* ``t_1=x^2``, ``(\cdot)^2`` is a primitive operation, and ``dt_1/dx = 2x``
* *e.g.* ``z=e^x``, ``e^{(\cdot)}`` is a primitive operation, and ``dz/dx = e^x``

"""

# ╔═╡ 62c244b3-fa36-472b-98a3-9993172ae9a4
md"""

#### It is modularised and mechanical 

* therefore, can be implemented as sequential steps 
"""

# ╔═╡ 86c5226a-97ce-4afe-a673-c799f78765ea
md"""
## Forward mode auto-diff


#### It accumulates (multiply) gradients forwardly

* ##### in one pass 


"""

# ╔═╡ df80e910-b219-40d6-9d66-6a8264fed915
md"""
## Forward mode auto-diff (cont.)


#### It accumulates (multiply) gradients forwardly


* ##### The change rate between node $t_1$ and the input $x$:

$$\Large\color{red}\frac{dt_1}{dx}$$
"""

# ╔═╡ 6c8aabfd-3a24-45ff-801a-caf8f07b1c4a
md"""
## Forward mode auto-diff (cont.)


#### It accumulates (multiply) gradients forwardly


* ##### At each node, we compute the derivatives:

$$\Large\color{red}\frac{dt_i}{dx}$$
"""

# ╔═╡ 02d9c597-c4be-471c-acc5-6dc3c0c56b1f
md"""
## Forward mode auto-diff (cont.)


#### A short hand notation, "dot"


$$\Large\color{red}\frac{dt_i}{dx} =\dot{t}_i$$

* #####  change ratio between the input $x$ (the root node) and all the intermediate nodes $t_i$
"""

# ╔═╡ 244030b7-b0fe-4b51-951e-3142346d03f2
md"""

## Forward mode (how?)


#### Dynamic programming !

* ##### instead of multiplying all gradients from the `root` over and over again

```math
\Large \dot{t}_4 = \frac{dt_4}{dt_3}\underbrace{\frac{dt_3}{dt_2}\frac{dt_2}{dt_1}\frac{dt_1}{dx}}_{\dot{t}_3}
```


* ##### we accumulate the gradients (by multiplying and cache)

  * ##### $\dot{t}_3 =\frac{dt_3}{dt_2}\frac{dt_2}{dt_1}\frac{dt_1}{dx}$ is computed and cached
  
```math
\Large \dot{t}_4 = \frac{dt_4}{dt_3}\dot{t}_3
```

"""

# ╔═╡ 49c5f93f-8b24-4703-95ed-3177ff3cdc1a
md"""

## Forward mode: initilisation

#### Initialise the root node's gradient as $1$, since 
```math
\Large
\dot{x} =\frac{dx}{dx} = 1
```

* basically `identity` function's gradient
"""

# ╔═╡ 7e8c4ebc-67f4-446d-932f-817382979854
md"""

## Forward mode: conti.

#### Forwardly accumulates gradients
```math
\Large
\dot{t}_1 =\dot{x} \frac{dt_1}{dx}
```

"""

# ╔═╡ 6405e556-5a7f-4e1f-8ccc-f5ccce191c5b
md"""

## Forward mode: conti.

#### Forwardly accumulates gradients


"""

# ╔═╡ caf32470-457e-4f0a-9f1a-9a90704a103c
md"""
## Forward mode -- the problem


#### Forward mode can be inefficient for a $\mathbb{R}^n \rightarrow \mathbb{R}^m (n \gg m)$ function

* ##### many inputs $n$, and *e.g.* scalar output $m=1$ 


"""

# ╔═╡ b002eb10-1c33-47a7-84bf-239b67c26988
md"""

## Forward mode -- the problem (cont.)


#### But why inefficient?

* ##### recall for each $$\large\colorbox{pink}{$x_i$}$$, we compute the partials (for each nodes) 

$$\Large \dot{\text{v}} = \frac{\partial\text{v}}{\partial\colorbox{pink}{$x_i$}}$$


* ##### the infinitesimal change rate (aka derivative) between all the intermediate nodes $\large \text{v}\small s$ and **the root node** $$\colorbox{pink}{$x_i$}$$ 
  * ##### (while holding the rest $x_{j\neq i}$ constant)
"""

# ╔═╡ c96d7374-99de-4475-9f63-021208959af6
md"""

## Forward mode -- the problem (cont.)


#### But why inefficient?

* ##### recall for each $$\large\colorbox{pink}{$x_i$}$$, we compute the partials (for each nodes) 

$$\Large \dot{\text{v}} = \frac{\partial\text{v}}{\partial\colorbox{pink}{$x_i$}}$$


* ##### the gradient we aim to compute is the collection of the partials

$$\Large\begin{align}\nabla_{\mathbf{x}} \text{y} &= \begin{bmatrix}\frac{\partial\text{y}}{\partial\colorbox{lightblue}{$x_1$}} & \frac{\partial\text{y}}{\partial\colorbox{orange}{$x_2$}} & \ldots & \frac{\partial\text{y}}{\partial\colorbox{lightgreen}{$x_n$}}\end{bmatrix}^\top\end{align}$$
#### Therefore, we need to _repeat_ the process $n$ times
* ##### one for each input dimension $x_i$


"""

# ╔═╡ 4253d341-e1b9-4764-bcf7-7e467b21b225
md"""
## Forward mode -- the problem (cont.)

### To be more specific

----
* #### _For input_ dimension $x_i$
  * ##### initialise $\dot{x}_i=1$ and $\dot{x}_{j\neq i} =0$ (why?)
  * ##### do the forward accumulation to compute $\frac{\partial y}{\partial x_i}$

* #### _Collect the partials to form the gradient_ $\nabla_{\mathbf{x}} y$

$$\Large\begin{align}\nabla_{\mathbf{x}} \text{y} &= \begin{bmatrix}\frac{\partial\text{y}}{\partial x_1} & \frac{\partial\text{y}}{\partial x_2} & \ldots & \frac{\partial\text{y}}{\partial x_n}\end{bmatrix}^\top\end{align}$$
----
"""

# ╔═╡ 24d74ebe-eb37-4556-bab5-b668f5e7af9c
Foldable("Why?", md"""

Note that when computing the partials w.r.t $x_i$, all the partials are defined as

$$\dot{v} = \frac{\partial \text{v}}{\partial x_i}$$


* therefore, $\large\dot{x}_i = \frac{\partial \text{x}_i}{\partial x_i}=1$, (the derivative of a variable w.r.t itself is always 1)


* and for all $j\neq i$: $\large\dot{x}_{j} = \frac{\partial \text{x}_j}{\partial x_i}=0$ (in other words, the $j$-th input is independent from $i$-th input as there is no directed path from $x_i$ to $x_j$)


* it can also be motivated more formally from the **definition of the partials** (perturbing one dimension each time while holding the rest constant)
""")

# ╔═╡ b0ec7a68-d684-4eac-91d5-a7fe71c40c69
md"#### _For example, for_ $x_1$"

# ╔═╡ 754d270f-6d24-4240-b638-3003eb5d4420
md"#### _For example, for_ $x_i$"

# ╔═╡ 433d7769-a9fb-4b57-b50c-6e5ccef506f4
md"#### _For example, for the last input_ $x_n$"

# ╔═╡ 7e58f5a9-5378-4f09-8993-b9dc6766f6ac
md"""

## Forward mode -- implementation*


#### Forward auto-diff is surprisingly simple to implement
* ##### with ``\approx`` 20 lines of code (if we use, _e.g._ `Julia`, or any functional programming language)
* ##### in essence, for `primitive` operations, we define how gradient propagate forwardly
* ##### easiest to implement with `Dual number`


#### `Dual number`, basically a node with 
  * the value ``\text{v}`` and 
  * the partials ``\dot{\text{v}}``

"""

# ╔═╡ 39850af7-6d6e-48d4-8d19-f96e3966571f
begin
	struct Dual{T} <: Number
	    v::T
	    v̇::T
	end
end;

# ╔═╡ 930bd2d3-0028-46aa-a0be-cb0e4973943e
md"""
#### Sum rule 

* ##### How to compute the function value
$$\large h(x) = f(x)+g(x)$$


* ##### How to forwardly accumulate the gradient

$$\large h'(x) =f'(x) + g'(x)$$


"""

# ╔═╡ 186988ed-6373-416c-84c4-a9cbd0a6a169
begin
	## forward rules for addition
	Base.:+(f::Dual, g::Dual) = Dual(f.v + g.v, f.v̇ +g.v̇)
	## add a dual number with another constant
	Base.:+(f::Dual, α::Number) = Dual(f.v  + α, f.v̇)
	## no need to redefine! addition is commutative
	Base.:+(α::Number, f::Dual) = f + α
end

# ╔═╡ 3d4c3e71-2b33-4e00-a710-792c604466c6
md"""
#### Substract rule ? no need! reuse sum rule

$$\Large f(x) - g(x) = f(x) + (-1)*g(x)$$

"""

# ╔═╡ ba807661-3c95-4d18-bca4-be2e5606f697
md"""

#### Product rule 
$$\large h(x) = f(x) * g(x)$$

$$\large h'(x)=f'(x)g(x) + f(x)g'(x)$$



#### Quatient rule 

$$\large h(x) = f(x) / g(x)$$

$$\large h'(x)=\frac{g(x)f'(x) - f(x) g'(x)}{g(x)^2}$$

"""

# ╔═╡ 0fb3579f-9cde-44d7-93f0-725939e25ffe
md"""

#### Some extra book-keeping*
"""

# ╔═╡ 93f43a4d-b5f1-456b-8565-dea84eb67790
begin
	## how to convert a plain value x to Dual, the partial is assumed zero for all constants
	Base.convert(::Type{Dual}, x::Real) = Dual(x, zero(x))
	## tell Julia how to deal with the promotion: from a plain number to Dual number
	Base.promote_rule(::Type{Dual{T}}, ::Type{<:Number}) where T = Dual
end

# ╔═╡ 0f52879e-a31f-44cc-9ee3-665fff477f54
promote(Dual(1.0, 0.0), 2.5) ## promotion rule (together with convert) is handy to deal with all other unspecified cases; it promotes a number to dual number

# ╔═╡ ffc4c9af-32b0-4720-ad36-a9fd5fd1df94
md"""
## A few more `primitive` functions


$$\large f(x) = x^n;\;\; f'(x) = n x^{n-1}$$

$$\large f(x) = e^x;\;\; f'(x) = e^x$$

$$\large f(x) = \ln(x);\;\; f'(x) = 1/x$$

$$\large f(x) = \sin(x);\;\; f'(x) = \cos(x)$$


##### Exercise, write a forward function for $f(x) =\cos(x)$

"""

# ╔═╡ 6019dc3b-c0ec-4a28-9539-51e2fa19559e
Base.:^(d::Dual, n::Integer) = Dual((d.v)^n,  n * (d.v)^(n-1) * d.v̇)

# ╔═╡ 3b7546dd-1025-494a-893d-1297f488093a
begin
	# Product Rule
	Base.:*(f::Dual, g::Dual) = Dual(f.v * g.v, f.v̇ * g.v + f.v * g.v̇)
	Base.:*(α::Number, f::Dual) = Dual(f.v * α, f.v̇ * α)
	Base.:*(f::Dual, α::Number) = α * f

	# Quotient Rule
	Base.:/(f::Dual, g::Dual) = Dual(f.v/g.v, (f.v̇*g.v - f.v*g.v̇)/(g.v^2))
	Base.:/(α::Number, f::Dual) = Dual(α/f.v , -α*f.v̇ / f.v^2)
	Base.:/(f::Dual, α::Number) = f * inv(α) # alternatively Dual(f.v /α, f.v̇ * (1/α))
end

# ╔═╡ f6bbddfe-d23d-44d9-ba13-237c55b9567d
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

# ╔═╡ 591c5ddd-7a73-48e9-8f2a-d634fad62ca6
show_img("multichainfwd.svg", h=200)

# ╔═╡ a40a6b23-1853-401c-b470-1ec5779c5df1
show_img("multichainbwd.svg", h=300)

# ╔═╡ bc234d58-096a-4f40-a2e5-fe170d5c0282
show_img("manytomanyf.svg", h=300)

# ╔═╡ c0ca19ab-7223-4125-a656-81fc9eb3db4c
show_img("partial1.svg", h=300)

# ╔═╡ 72873e0f-9cb6-474a-974a-b21289b93ca8
show_img("partial2.svg", h=300)

# ╔═╡ 5dcffcea-864f-4657-9904-9da542b69ed5
show_img("compgraph.svg", h=250)

# ╔═╡ 700fdce1-16d4-4179-8ed7-dce1a19cf8a9
TwoColumn(md"""
\

##### Loss for simple linear regression:

```math
\Large
l = \frac{1}{2}\left((x*w +b) -y\right )^2 
```
\


```python
t₁ = x * w # intermediate value

t₂ = t₁ + b # ŷ

t₃ = t₂ - y # ŷ - y

t₄ = t₃ ** 2 # (ŷ - y)²

l = 1/2 * t₄ # 1/2 * (ŷ - y)²
```

""", 
show_img("CS5914/backprop/linreg_grad.svg", h=400))

# ╔═╡ e5dd83b3-3744-4d1d-9aa2-5f24c036e16f
show_img("localgrads2.svg", w=700)

# ╔═╡ 68f733fa-d6eb-417c-a33e-9b9c2bf803ef
show_img("fw1.svg", w=700)

# ╔═╡ 6e7266ad-dbbf-4058-b650-3d7d013b120b
show_img("fw2_.svg", w=700)

# ╔═╡ 2ca35388-7302-4f04-a67c-fa29c3d51e4d
show_img("fw3.svg", w=700)

# ╔═╡ d6abb07e-595e-42c2-9a98-ca09caf156a2
show_img("fw4.svg", w=700)

# ╔═╡ b41eb406-f681-4c27-a041-6d932e1d19b2
show_img("fwt4_1.svg", w=700)

# ╔═╡ 0c3601f1-406a-4ce8-a3ad-dd6ac5ef478e
show_img("fwt4_2.svg", w=700)

# ╔═╡ 5fcfe9b6-d0e4-4bc9-86d2-4bdac90e1ec7
show_img("fwx1.svg", w=700)

# ╔═╡ 5e7cc34b-73b3-4342-94bb-4be6969fb040
show_img("fwx2.svg", w=700)

# ╔═╡ 33e47f17-b4ef-4206-b719-a2ac81355e56
show_img("fwx3.svg", w=700)

# ╔═╡ c0a731b8-4732-4561-9394-1d9d6b2adfb7
show_img("fwx4.svg", w=700)

# ╔═╡ 2a682b20-c48a-478f-af99-18dcb441c919
show_img("fwx5.svg", w=700)

# ╔═╡ 93925a5f-c287-478a-81da-86eee8aaea08
show_img("fwmulti.svg", h=300)

# ╔═╡ 37bded16-7d2f-4317-b5c1-db6185cc3800
show_img("fwmulti3.svg", h=390)

# ╔═╡ 851a7cea-4fed-4049-933c-92da77d68c0d
show_img("fwmulti2.svg", h=400)

# ╔═╡ b7f16835-f05d-4967-814e-57e11af3dbbc
show_img("fwmulti3.svg", h=400)

# ╔═╡ 75e3a5d9-5631-4170-8ca7-e824c1bef993
show_img("fwmulti4.svg", h=400)

# ╔═╡ a1366fa3-f39d-48f3-80b7-d6174f5019af
begin
	# Program like a mathematician :-): - is just a specific case of +
	Base.:-(f::Dual, g::Dual) = f + (-1) * g
	# alternatively
	# Base.:-(f::Dual, g::Dual) = Dual(f.v  - g.v, f.v̇ - g.v̇)
end

# ╔═╡ b35d4772-d810-4b95-92ac-d5dd87353f86
begin
	Base.:-(f::Dual) = (-1) * f
end

# ╔═╡ 4dc93a97-6c75-45d9-8ed7-aeaaf74389e9
Base.exp(d::Dual) = Dual(exp(d.v), exp(d.v) * d.v̇)

# ╔═╡ f698f63c-ac0f-4891-9c0c-ca42db4b4555
let
	@variables x, w, b
	σ = 1 / (1+exp(-1(x * w + b)))
	D = Differential(w)
	σ, expand_derivatives(D(σ))
	# expand_derivatives(D(σ))
end

# ╔═╡ a8b5b0c0-166d-4422-b5db-41900178eace
let
	@variables x, w₁, b₁, w₂, b₂ 
	z = 1/(1+exp(-(1/(1+exp(-(w₂* (1 / exp(-1(x * w₁ + b₁))) +b₂))))))
	D = Differential(w₁)
	# z
	expand_derivatives(D(z))
end

# ╔═╡ 978da62a-e03c-4539-a859-acd0a78ce425
Base.log(d::Dual) = Dual(log(d.v), 1/(d.v) * d.v̇)

# ╔═╡ 6fcfb337-8e25-4142-9a67-31c9887fbd2f
Base.sin(d::Dual) = Dual(sin(d.v), cos(d.v) * d.v̇)

# ╔═╡ 937072a0-9487-424d-b427-a280f85bc448
let
	gr()
	x₀ = 0.0
	ϵ2_ = 0.3
	Δx = ϵ2_
	xs = -1.2π : 0.1: 1.2π
	f, ∇f = sin, cos
	# anim = @animate for Δx in π:-0.1:0.0
	# Δx = 1.3
	plot(xs, sin, label=L"\sin(x)", ylim = [-1.5, 1.5], xlabel=L"x", lw=2, legend=:outerbottom, framestyle=:semi, title="Finite difference method", legendfontsize=10)
	df = f(x₀ + Δx) - f(x₀ - Δx)
	k = Δx == 0 ? ∇f(x₀) : df/(2*Δx)
	b = f(x₀+ Δx) - k * (x₀+ Δx)
	# the approximating linear function with Δx 
	plot!(xs, (x) -> k*x+b, label="", lw=2)
	scatter!([x₀], [f(x₀)], mc=:white, msc=:gray, msw=2, alpha=0.9, label=L"x_0,\; \sin(x_0)")
	scatter!([x₀+Δx], [f(x₀+Δx)], ms=4, label=L"x_0+Δx,\; \sin(x_0+Δx)")
	scatter!([x₀-Δx], [f(x₀-Δx)], ms=4, label=L"x_0-Δx,\; \sin(x_0-Δx)")

	plot!([x₀-Δx, x₀+Δx], [f(x₀-Δx), f(x₀-Δx)], lc=:gray, label="")
	plot!([x₀+Δx, x₀+Δx], [f(x₀-Δx), f(x₀+Δx)], lc=:gray, label="")
		font_size = Δx < 0.8 ? 12 : 14
		annotate!(x₀+Δx, 0.5 *(f(x₀-Δx) + f(x₀+Δx)), text(L"Δf", font_size, :top, rotation = 90))
		annotate!(0.5*(x₀+x₀), f(x₀-Δx), text(L"Δx", font_size,:top))
		annotate!(-.6, 1, text(L"\frac{Δf}{Δx}=%$(round(k, digits=6))", 15,:top))
end

# ╔═╡ 90784f49-fbed-489c-8f7f-6e182ee68f8b
md"""

## Demonstration

$$\Large y = e^{-\sin(x^2)+5}$$
"""

# ╔═╡ 0b1a867e-fe88-4bd3-b3fb-311506647140
let
	x0 = 2.0
	x = Dual(x0, 1.0)
	y = exp(-sin(x^2) +5)
end

# ╔═╡ 16eefeee-3a1e-491d-8f33-bba535f778ef
let
	function y(x)
		exp(-sin(x^2) +5)
	end
	## finite difference gradient check
	x0 = 2.0
	eps = 1e-5
	y(x0), (y(x0 + eps) - y(x0 - eps))/ (2*eps)
end

# ╔═╡ 0cb656d6-c792-4662-958e-7456f0300a18
md"""

## Demonstration

##### `sigmoid` function
$$\large\sigma(x) = \frac{1}{1+e^{-x}}$$

* note that its derivative is $\sigma'(x) = \sigma(x) (1-\sigma(x))$, therefore ``\sigma'(0.0) = 0.5^2 = 0.25``
"""

# ╔═╡ c9429b12-4afd-4997-b0be-1d1d22f3cc13
function sigmoid(x)
	1 / (1+exp(-x))
end

# ╔═╡ 371d4e51-0c19-4430-96b9-192d72ad5824
sigmoid(Dual(0.0, 1.0)) ## auto-diff with our own implementation

# ╔═╡ 0118fc3f-e0e1-480d-9b23-8bcfa55feebc
let 
	σ_ = sigmoid(0.0)
	σ_*(1-σ_)
end;

# ╔═╡ 3faea791-befb-40ac-85d6-8e6194ccdb6a
md"#### We can modularise `sigmoid` as one primitive operation

* ##### treat `sigmoid` function as one unit
  * ###### it will be  faster 
  * ###### as it bypasses a lot of primitive operations
* ##### most deep learning packages does this for commonly used functions


"

# ╔═╡ 2777a33d-6959-4f6f-a8f7-b666b56dc0a6
function sigmoid_fwd(d::Dual)
	σ = 1/(1+exp(- d.v))
	dσdx = σ * (1-σ)
	return Dual(σ, dσdx * d.v̇)
end

# ╔═╡ 5aba6c98-faf1-45a7-a8c3-325a96e3e444
md"""

We demonstrate below how the `sigmoid`  is used in computing a cross entropy loss for scalar weight $w$

* if the weight is a vector, we need to repeat the process multiple times

"""

# ╔═╡ e1cf2c66-d6aa-4fed-ba4e-d798d36205ca
let
	## cross entropy loss and compute the derivative w.r.t w
	y = true
	x = 1
	w = 2.
	# logit
	z = Dual(w, 1.0) * x
	# sigmoid result
	σ = sigmoid_fwd(z)
	# foward computing the cross entropy loss, forward diff also returns dl/dw
	loss = y ? log(σ) : log(1-σ)
end

# ╔═╡ 8acc5a4b-cc2b-458d-8818-59c5ff52b524
let
	## finite difference method to compute gradient of w 
	y = true
	x = 1
	w = 2.
	loss(w) = y ? log(sigmoid(w * x)) : log(1-sigmoid(w*x))
	ϵ = 1e-5
	dldw = (loss(w + ϵ) - loss(w - ϵ)) / (2ϵ)
end

# ╔═╡ 20c9c798-52b2-4f56-8ca6-7124a778fd43
md"""

## Demonstration

##### We can also auto-diff a program

##### A naive implementation of ``x^n``


```julia
result = 1.0
for i in 1:n
	result *= x
end
```

##### Sanity check: ``(x^n)' = n x^{n-1}``
  * ``x=2, n =10``, then ``x^n = 2^{10}= 1024``
  * ``(x^n)'= 10 \times 2^9=5120``
"""

# ╔═╡ 0dfa109a-4612-479f-9379-61d63ae5295a
Base.one(d::Dual) = one(d.v) 

# ╔═╡ 8c86613a-9a95-4687-bbff-8fe12457e2c9
function x_power_n_naive(x, n)
	xⁿ = one(x)
	for i in 1:n
		xⁿ *= x
	end
	return xⁿ
end

# ╔═╡ ade06efb-109f-4874-8e60-a9faaf4e14ff
x_power_n_naive(Dual(2.0, 1.0), 10)

# ╔═╡ 353346df-3cd9-48bc-8d50-bd0bc1fba07e
md"""

#### Implementation with ``\ln``

```math
\Large
x^n = e^{\ln x^n} = e^{n \ln x}
```
"""

# ╔═╡ cc6ffeb0-8dcb-40f0-a190-475af17ce2f0
function x_power_n_log(x, n)
	nlogx = n * log(x)
	exp(nlogx)
end

# ╔═╡ e1c6edc6-e5c0-4e18-818f-67043977c808
x_power_n_log(Dual(2.0, 1.0), 10)

# ╔═╡ 315b7bfa-c215-4b32-b243-91c0a0a9d577
md"""

#### It works even with recursive functions

##### _e.g._ a fast divide-conquer implementation

```math
\large
x^n = \begin{cases}(x^{n//2})^2  & \text{if n is even} \\

x \times (x^{n//2})^2  & \text{if n is odd}
\end{cases}
```
"""

# ╔═╡ 88f23dfa-615d-49a7-90bc-97b56f73d1af
function x_power_n_fst(x, n)
	if n == 0
		return one(x)
	elseif n > 0
		nhalf = floor(Int, n/2)
		xⁿʰ = x_power_n_fst(x, nhalf)
		if iseven(n)
			return xⁿʰ * xⁿʰ
		else
			return x * xⁿʰ * xⁿʰ
		end
	end
end

# ╔═╡ bedb1024-3458-4c5c-83f3-09cae1034db2
x_power_n_fst(Dual(2., 1.), 10)

# ╔═╡ 18cc560a-eb59-40e2-869f-e7169b0fa9f3
md"""

# Reverse mode auto-diff
"""

# ╔═╡ 0f9f63a8-1232-4fb7-b3b2-4b52cba7728f
md"""

## Reverse mode -- big picture


#### Accumulate gradients backwardly, 


* ##### for each nodes $\large\text{v}\small s$, we compute (y is the final output node)

$$\Large\bar{v}=\frac{\partial \text{y}}{ \partial v}$$


* ##### Initialisation, set the output's adjoint to 1

```math
\Large 
\bar{\text{y}} = \frac{\partial y}{\partial y} =1
```
"""

# ╔═╡ b642f207-5a8e-495d-89e4-b98f13f5a165
show_img("bw1.svg", w=700)

# ╔═╡ 5f84553f-2acf-409d-af06-77462962d1f0
show_img("bw2.svg", w=700)

# ╔═╡ 724e8726-6bd2-4dfc-8394-fe2aee6a9963
show_img("bw3.svg", w=700)

# ╔═╡ c53f3f55-ccc3-493e-b657-fa141134a378
show_img("bw4.svg", w=700)

# ╔═╡ a4e4638a-1d2c-49d1-aa53-2614e6c78c82
md"""

## Forward _vs_ Reverse mode
"""

# ╔═╡ 589b1b4f-ef8b-4f0f-b2b9-099314ec0a1c
md"""
#### Reverse mode

* ##### the sensitity of `the output` when we change an `intermediate node`
"""

# ╔═╡ e6f15720-71c9-4e4a-924a-2671b994aba9
show_img("bw3.svg", w=700)

# ╔═╡ cd916288-15f0-4f9e-be05-b756b55122ca
md"""

#### Forward-mode 


* ##### the sensitity of `intermediate` node when we change an `input node`
"""

# ╔═╡ 0af94490-d524-478c-9d61-c1b00055e3df
show_img("fwx4.svg", w=700)

# ╔═╡ 1c9c8efa-fc0e-47a6-9e94-a8c9aa89bf79
md"""

## Adjoint notation 
#### -- (bar notation ``\bar{v}``)

* ##### for each nodes $\large\text{v}\small s$, we compute

$$\Large\bar{\text{v}} = \frac{\partial \text{output}}{ \partial \text{v}}$$

"""

# ╔═╡ ce47f14e-ab0e-4e9c-ac0f-493bc7e592a5
show_img("bw5.svg", w=700)

# ╔═╡ 65e8cfd2-c430-4479-8311-1cef1dfe03d2
md"""

## Reverse mode

#### A `two pass` algorithm
* ##### to construct the `DAG`, we need do the `forward pass` first
* ##### then `back-propagate` the gradients 

"""

# ╔═╡ 62729b84-6c72-45e5-afa5-b2a5c886d674
show_img("fwdpass.svg", w=700)

# ╔═╡ 0530448d-a50c-4970-8c8e-d69bc1b8dd25
show_img("bwdpass.svg", w=700)

# ╔═╡ a882ceef-e161-4dad-8aab-c9493d3bb99c
md"""

## Why reverse mode ?


#### It is very efficient for $$\mathbb{R}^n \rightarrow \mathbb{R}$$

* ##### just one backward pass is enough to compute all the gradients
"""

# ╔═╡ 7f6532b2-0ab8-4f50-942b-d7a6a12560b1
show_img("fwdpass1.svg", h=400)

# ╔═╡ f5b62aca-380c-4631-97c2-e88fd492c95c
show_img("bwdpass1.svg", h=400)

# ╔═╡ a17a212e-bdd9-423b-8182-571e3571ca01
md"""

## How to back-propagate efficiently?


#### Dynamic programming (again) !

* ##### by definition, the adjoint for $t_1$ is 

```math
\Large \bar{t}_1 = \frac{\partial y}{\partial t_1} = \underbrace{\frac{\partial y}{\partial t_4}\frac{\partial t_4}{\partial t_3}\frac{\partial t_3}{\partial t_2}}_{\partial y/\partial t_2=\bar{t}_2}\frac{\partial t_2}{\partial t_1}
```


"""

# ╔═╡ a0685a47-88f4-4ebb-b6f6-9722f77fbd18
show_img("bwd5_.svg", w=700)

# ╔═╡ e6573211-06a2-4606-9caf-d97fb88f1e00
md"""


* #### we can efficiently compute it 
  * ##### (if $\bar{t}_2 =\frac{\partial y}{\partial t_2}$ is computed and cached!)

  
```math
\Large \bar{t}_1 = \frac{\partial t_2}{\partial t_1} \bar{t}_2
```

* #### more generally, for a `chain` DAG, 

  * ##### $\{t_1, t_2, t_3, \ldots, t_n\}$ form a `chain`, that is $t_{i+1}$ is the child of $t_i$ 

  
```math
\Large \bar{t}_i = \begin{cases}1 & \text{if } t_i \text{ is the end node} \\
\frac{\partial t_{i+1}}{\partial t_{i}}\bar{t}_{i+1} & \text{others}\end{cases}
```
"""

# ╔═╡ d13d806e-ecc8-48d3-a7c5-38a1e4c3cd1b
md"""

## Reverse accumulation in general


#### For a general DAG 

* ##### (due to multivariate chain rule) sum all paths's gradients

"""

# ╔═╡ 59effbcd-4334-4b19-9d00-38805577d4e3
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/fork_rule.svg
# ' height = '350' /></center>"

show_img("CS5914/backprop/fork_rule.svg", h=350)

# ╔═╡ dc012b20-9fb9-4757-98ae-50e81525715b
md"""
## Backpropagation -- _the essence_


#### The essence of backpropagation is 
* ##### mechanical backward *message passing* 
* ##### and caching the adjoints along the way


"""

# ╔═╡ b1dcd923-5170-4aca-ac43-936d36c98a47
show_img("CS5914/backprop/backprop_withfork.png", w=800)

# ╔═╡ b6445302-6a19-4677-8ba6-4e5850a17f91
# show_img("CS5914/backprop/backprop_onefig.png", h=380)

# ╔═╡ 05c921ee-ef17-4e3f-9380-1929a1a98761
md"""

## Backprop operation as `gates`


#### Computers clearly *do not* understand *calculus*

* ##### it makes computers' life (and also our life) easier, let's forget calculus for a moment
* ##### but simply treat those operators mechanically

## `add (+)` gate

#### Consider addition operator: ``\large +``

* ##### forward pass is fairly straightforward

* ##### how about the backward pass? _Hint: what are the local gradients?_
"""

# ╔═╡ d2d57c21-4c16-47d5-b9dc-9e9100d361bc
# html"<img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/plusgate1_.svg
# ' height = '280' />"

show_img("CS5914/backprop/plusgate1_.svg", h=280)

# ╔═╡ 5443b0f4-1b2c-4d23-a2c2-72f6f6dbb8a1
md"""
## 

"""

# ╔═╡ 26bcb663-2eb1-4542-a50f-7f8e26d17c13
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/plusgate.svg
# ' height = '300' /></center>"

show_img("CS5914/backprop/plusgate.svg", h=300)

# ╔═╡ f65deaf9-42c6-4a6a-bde1-45e9f3191294
md"""
## `mult (*)` gate

##### Consider operator: ``\large *``

* ##### forward pass is fairly straightforward

* ##### how about the backward pass? 
"""

# ╔═╡ 8f7d61a6-b580-4277-a70d-00ba79d9547a
# html"<img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/mult_gate1.svg
# ' height = '350' />"

show_img("CS5914/backprop/mult_gate1.svg", h=350)

# ╔═╡ 872f54bc-fc6a-4e52-a99f-357d45c3e68a
md"""

##
"""

# ╔═╡ b798636d-6c13-4aa7-bb9b-fc1e4e01a11b
# html"<img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/mult_gate.svg
# ' height = '350' />"

show_img("CS5914/backprop/mult_gate.svg", h=350)

# ╔═╡ 2aedeac6-d43c-4a0e-8e0b-f46e904600a2
md"""
## ``\texttt{max}``  gate

##### Consider operator: ``\large \texttt{max}(\cdot, \cdot)``, *e.g.* ``\texttt{max}(3, -1) =3``

* ##### forward pass is fairly straightforward

* ##### how about the backward pass? 
"""

# ╔═╡ d689bd7b-ce37-4373-8228-d29ac00adfa0
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/router_gate.svg
# ' width = '500' /></center>"

show_img("CS5914/backprop/router_gate.svg", w=500)

# ╔═╡ af15d1ec-b1ba-4c69-8e46-9b6d6e31523c
md"""

## ``\texttt{max}_c(x)`` gate (or `ReLu`)

"""

# ╔═╡ a855cfae-a5cb-4c94-9d7a-44387ad27060
TwoColumn(md"""
\


##### How about `max` with a constant ``c``? 

```math
\large
\text{max}_c(x) = \text{max}(x, c)
```

* ##### *i.e.* a threshold function

* ##### `ReLu` is a specific case, where ``c=0``



""", plot((x) -> max(x, 0), size=(280,280), ratio=1, framestyle=:origin, lw=2, label=L"\texttt{max}_c(x)"))

# ╔═╡ 166b5b82-3f8c-43e4-a469-1b9cce08bc4e
# html"<img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/relu_gate.svg
# ' height = '350' />"

show_img("CS5914/backprop/relu_gate.svg", h =350)

# ╔═╡ 82b57d6c-befa-4246-9b03-212e08c1d706
md"""

## Example: logistic regression's gradient

#### The logistic regression with cross entropy loss

```math
\Large
l = - \left (y \ln (\sigma(z)) + (1-y) \ln(1-\sigma(z))\right ),
```

* ##### where ``z= w*x +b`` is the logit


* ##### the **gradients** are 

```math
\boxed{
\begin{align}
\frac{\mathrm{d}\, l(w, b)}{\mathrm{d} w} &= \underbrace{(\sigma(wx +b) -y)}_{\hat{y} -y}x\\
\frac{\mathrm{d}\, l(w,b)}{\mathrm{d} b} &= \underbrace{(\sigma(wx +b) -y)}_{\hat{y} -y}\\
\end{align}
}
```

"""

# ╔═╡ 72f36dfe-89b5-4477-a83d-b6fd0046e96d
md"""

## Example: logistic regression's gradient

"""

# ╔═╡ f19c8378-7148-4baf-b664-a00d8805fbe8
TwoColumn(md"""
\

#### Forward pass



```python
t₁ = x * w # intermediate value
t₂ = t₁ + b # z = x*w +b

t₃ = σ(t₂) # ŷ; the output

t₄ = ln(t₃) # ln(σ)
t₅ = ln(1-t₃) # ln(1-σ)

t₆ = t₄ * y # y⋅ln(σ)
t₇ = 1 - y 
t₈ = t₇ * t₅ # (1-y)⋅ln(1-σ)

t₉ = t₆ + t₈ # y⋅ln(σ) + (1-y)⋅ln(1-σ)

l = -1 * t₉ # -1(y⋅ln(σ)+(1-y)⋅ln(1-σ))
```

""", 
	
# 	html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/logis_forward.svg
# ' height = '600' /></center>"

show_img("CS5914/backprop/logis_forward.svg", w=330))

# ╔═╡ cf54b8dd-2020-44f3-a93b-1a449769c528
md"""

## Example: logistic regression's gradient

#### Initialisation: $\bar{l}=1$


"""

# ╔═╡ b3df5d95-f975-47a1-a338-8766c15f027c
#  html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/logis_backward1.svg
# ' height = '600' /></center>" 

 show_img("CS5914/backprop/logis_backward1.svg", h=600)

# ╔═╡ 140e3645-4c1b-4737-a458-45b014299349
md"""

## Example: logistic regression's gradient


#### Let's derive the backprop with gates in mind!
"""

# ╔═╡ 7b0773f7-acad-4773-9207-1d09f632b0a0
TwoColumnWideRight(md""" 

\
\

#### `mult` gate 
* ##### works as exchanger
* ##### exchange ``-1`` here""", 
	
# 	html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/logis_backward2_.svg
# ' height = '600' /></center>"

show_img("CS5914/backprop/logis_backward2_.svg", h=600))

# ╔═╡ e584e6b5-3f2a-4e12-8706-db27a507eeaa
md"""

## Example: logistic regression's gradient


###### Let's derive the backprop with gates function in mind!
"""

# ╔═╡ f9a54b6b-632a-410c-8501-e40109ee7605
TwoColumnWideRight(md""" 

\
\
\

##### `+` gate: as a distributor
* distributes gradient to both branches""", 
	
# 	html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/logis_backward3.svg
# ' height = '600' /></center>" 

show_img("CS5914/backprop/logis_backward3_.svg", h=600))

# ╔═╡ 999f8bef-39c5-4ac3-97b9-60bd09189f26
md"""

## Example: logistic regression's gradient


###### Let's derive the backprop with gates function in mind!
"""

# ╔═╡ 2ba2c7ea-e7a1-474b-a696-1a15c850637b
TwoColumnWideRight(md""" 

\
\
\
\
\
\

##### `*` gate: as an exchanger
* exchange ``y`` with ``{t_4}``'s branch""", 
	
	
# 	html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/logis_backward4.svg
# ' height = '600' /></center>" 

show_img("CS5914/backprop/logis_backward4_.svg", h=600))

# ╔═╡ f2729651-6193-4234-8d04-20edc76bf013
md"""

## Example: logistic regression's gradient


###### Let's derive the backprop with gates function in mind!
"""

# ╔═╡ 82a2d879-170b-469f-a6e9-f2c6269937cc
TwoColumnWideLeft(
	
# 	html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/logis_backward5.svg
# ' height = '600' /></center>"
	
	show_img("CS5914/backprop/logis_backward5_.svg",h=600)
	, md""" 

\
\
\
\
\
\
\

##### `*` gate: as an exchanger
* exchange ``t_7`` with ``{t_5}``""" )

# ╔═╡ 7d50e6d3-013e-4f13-aeb9-5190f8c37e70
md"""

## Example: logistic regression's gradient


###### Let's derive the backprop with gates function in mind!
"""

# ╔═╡ 9298c327-bea7-48ac-bded-315dc9d76208
TwoColumnWideLeft(
	
# 	html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/logis_backward6.svg
# ' height = '600' /></center>"
	show_img("CS5914/backprop/logis_backward6_.svg", h=600)
	
	, md""" 

\
\
\
\
\
\
\
\
\
\
\
\



###### How to backprop ``\overline{t_3}`` ? 
hint: _it is a fork!_ 

""" )

# ╔═╡ 9d3cb890-47ec-42a8-b67e-a6a675a81300
md"""

## Example: logistic regression's gradient


###### Let's derive the backprop with gates function in mind!
"""

# ╔═╡ 5e3687c2-8e18-4fac-81ec-602edb79a091
TwoColumnWideLeft(
# 	html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/logis_backward7.png
# ' height = '600' /></center>"
	
	show_img("CS5914/backprop/logis_backward7_1.png", h=600)
	
	, md""" 

\
\
\
\
\
\
\
\
\
\
\
\
\



###### How to backprop ``\overline{t_3}`` ? 
hint: _it is a fork!_

""" )

# ╔═╡ d474e7df-9dd4-4a52-9644-5e86309a5063
aside(tip(md"""

Recall ``\ln(x)``'s derivative

```math
\small
\frac{d \ln(x)}{dx} = \frac{1}{x},\quad \frac{d \ln(1-x)}{dx} = -\frac{1}{1-x}
```
"""))

# ╔═╡ 8f11acf4-861d-4c43-b24e-6a0f6e095ed4
md"""

## Example: logistic regression's gradient


###### Let's derive the backprop with gates function in mind!
"""

# ╔═╡ f5786bb9-1665-415e-92a1-1d7fc337f9f3
TwoColumnWideLeft(
	
# 	html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/logis_backward9.png
# ' height = '600' /></center>"
	show_img("CS5914/backprop/logis_backward9_1.png", h=600)
	
	, md""" 

\
\
\
\
\
\
\
\
\
\
\
\
\
\






_hint_: logistic function's gradient: ``\small\sigma'(x) = \sigma(x) (1-\sigma(x))``

and recall  ``\small t_3 =\sigma(t_2)``, therefore
	
$\small\frac{d t_3}{d t_2} = t_3 (1-t_3)$


""" )

# ╔═╡ 42ddbd1b-0711-4b5e-b327-81864aac0c62
md"""

## Example: logistic regression's gradient


###### Let's derive the backprop with gates function in mind!
"""

# ╔═╡ 190fac3d-25cc-4d7d-91f4-199d6f2932c0
TwoColumnWideLeft(
	
# 	html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/logis_backward10.png
# ' height = '600' /></center>"
	
	show_img("CS5914/backprop/logis_backward10_.png", h=600), md""" 

\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\





The rest are the same as linear regression's case!
""" )

# ╔═╡ 87620132-1261-4b7c-8ad9-933cc64de277
md"""

## Simplify by merging

#### Merge and define the loss `logit_BCE_loss`:


```math
\texttt{logit\_BCE\_loss}(z, y) = -y \ln(\sigma(z)) - (1-y)\ln(1-\sigma(z))
```

* ##### where ``z`` are called `logits` 
  * more numerically stable 
  * much simpler to implement! *especially*, the backward pass
"""

# ╔═╡ 7c29d7f8-2479-4e5b-94e7-7040453ac4db
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/merge_logis.png
# ' width = '800' /></center>"

show_img("CS5914/backprop/merge_logis.png"; w=800)

# ╔═╡ 03eede9e-d7f4-4a5d-ab0c-9daa9e72594b
md"""


## Show me the code


"""

# ╔═╡ d4ce1390-f96e-45d8-a034-f869b5bd1ee4
TwoColumn(
md"""

```python

## forward pass
t1 = w * x 
t2 = t1 + b
loss = logit_BCE_loss(t2, y) #logit_BCE_loss

## backward pass
dloss = 1 # initialise
## back propogate logit_BCE_loss
dt2 = (sigmoid(t2) - y) * dloss
dt1 = dt2
db = dt2
dw = dt1 * x 


```

""",

	show_img("CS5914/backprop/merge_logis_.png"; w=300)

	
)

# ╔═╡ ccef6f05-4f1b-4377-86e3-cc477176de01
md"""

## Cross entropy loss with `logits`


##### This is actually *the preferred approach* 

* ##### for `PyTorch` and all other Deep Learning packages

"""

# ╔═╡ dd3f1337-c5f9-41b8-8d1b-3b657404d4d3
show_img("CS5914/backprop/pytorch_logits.png"; w=700)

# ╔═╡ 5174b7c2-f1d3-437f-b6b5-5a9f89f555d6
md"#### Below is `Julia`'s Deep Learning `Flux.jl`'s documentation"

# ╔═╡ fb958c4b-5c8a-4bf2-bccc-3a7cafa6406d
show_img("CS5914/backprop/flux_logits.png"; w=700)

# ╔═╡ 35943fab-8e85-40f5-9397-f3577c935802
md"""

## The same idea applies to multi-class 
"""

# ╔═╡ 173b9082-1536-44b0-865e-dfe7e5ee8576

md"""

#### The cross entropy loss for multi-class is

$$
\Large
\begin{align}
L(\mathbf{y}; {\mathbf{z}}) 
&=- \mathbf{y}^\top \ln(\hat{\mathbf{y}})\\
&= - \mathbf{y}^\top  \ln \text{softmax}(\mathbf{z})
\end{align}$$

- ``\mathbf{y}`` is the one-hot encoded label
- ``\hat{\mathbf{y}} = \text{softmax}(\mathbf{z})`` is the softmax output of the logits ``\mathbf{z}``
* the local gradient w.r.t the logits $\mathbf{z}$ is of the same form 

```math
\Large
\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{y}} -\mathbf{y}
```
"""

# ╔═╡ 537fdffa-0343-4b01-8ea2-092049597cdc
show_img("CS5914/backprop/ce_flux.png"; w=700)

# ╔═╡ 87a8fa4c-5da2-48d6-a007-a6952dc4ebfb
md"""


## Backprop -- matrix calculus


#### It seems **daunting** to apply backprop with _matricies_ 
* ##### _matrix calculus_ indeed can be a bit tricky!

\


#### But it is straightforward if we stick to a few rules

> * #### The first rule -- dimension match rule



"""

# ╔═╡ 12399ab0-1c54-42bf-bedc-5e135ee9c035
md"""


## The first rule for matrix backprop

!!! important "The fundamental ground rule"
	#### For any scalar valued function 
	* (*e.g* all neural networks with scalar loss) 
	> #### the gradient of ``\bar{\mathbf{W}}`` has the same dimension as ``\mathbf{W}``
	>
	> * ##### *i.e.* `W̄.shape == W.shape`

##

*e.g.* if ``\mathbf{W}`` is a 2 by 3 matrix, 

```math
\large
	{\mathbf{W}} = \begin{bmatrix} {W}_{1,1} & {W}_{1,2}  & {W}_{1,3} \\
	{W}_{2,1} &  {W}_{2,2}  &  {W}_{2,3}
	\end{bmatrix}_{2\times 3}
```

Then its the gradient w.r.t the final scalar loss $l$ (or adjoint) is also ``2\times 3``

```math
\large
	\bar{\mathbf{W}} = \nabla_{\mathbf{w}}l = \begin{bmatrix}\frac{\partial l}{\partial {W}_{1,1}} & \frac{\partial l}{\partial {W}_{1,2}}  & \frac{\partial l}{\partial {W}_{1,3}} \\
	\frac{\partial l}{\partial {W}_{2,1}} & \frac{\partial l}{\partial {W}_{2,2}}  & \frac{\partial l}{\partial {W}_{2,3}}
	\end{bmatrix}_{2\times 3}
```

* ##### this turns out to be the most important and useful result

"""

# ╔═╡ 1de3e833-cf73-45b5-83cd-9b1cfaec6fd4
md"""
## Recap: scalar product `(*)` gate


##### Backward operate as an exchanger 
"""

# ╔═╡ cb4d586d-1b8e-41a2-bc98-e9f70d0a45d0
show_img("CS5914/backprop/scalar_prod_gate.svg", h=350)

# ╔═╡ 18db9aae-ea09-4a2b-848c-05c10416a733
md"""

## How about other multiplications ?


### For example, `inner product` `(*⊤)`

$$\large\begin{align}
\mathbf{w}^\top\mathbf{x} &= \begin{bmatrix}w_1&\ldots & w_n\end{bmatrix} \begin{bmatrix}x_1\\ \vdots\\ x_n\end{bmatrix}\\
&= \sum_{i=1}^n x_i w_i
\end{align}$$

* ###### the result is a scalar
* ###### still backprop as an exchanger!


"""

# ╔═╡ 29ac1207-584c-4c12-86c2-1d8ee2168db1
md"""

## Inner product gate `(*⊤)`

#### Recall the definition of inner product

```math
\large z= \mathbf{w}^\top\mathbf{x} = \sum_{i=1}^n w_ix_i
```

* #### and the local gradients/derivatives are 

```math
\large \frac{\partial z}{\partial\mathbf{x}} =\mathbf{w};\quad\frac{\partial z}{\partial\mathbf{w}} =\mathbf{x}
```

* ##### note that the dimension match!
"""

# ╔═╡ 11624c75-0199-499a-8393-e06295763fb8
# html"<img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/inner_gate.svg
# ' height = '350' />"

show_img("CS5914/backprop/inner_gate.svg", h=350)

# ╔═╡ 8cddc677-5796-489e-bb42-2ae13471090d
md"""

## Matrix vector `multi` `(@)`

#### How about matrix *times* vector

```math
\Large
\mathbf{z}= \mathbf{W}\mathbf{x} 
```

"""

# ╔═╡ 1110d206-09c3-4bbb-8932-aec832548a1b
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/matmul_gate_.svg
# ' height = '300' /><center>"
show_img("CS5914/backprop/matmul_gate_.svg", w=800)

# ╔═╡ a33e7154-6440-44f2-b73a-94a60aff6bd1
md"""
## How to remember it?



#### Note that 

```math
\Large
\mathbf{z}= \mathbf{W}\mathbf{x} 
```
#### The backward rule:
```math
\Large
\bar{\mathbf{W}} = \bar{\mathbf{z}} \mathbf{x}^\top;\;\;\bar{\mathbf{x}} = \mathbf{W}^\top\bar{\mathbf{z}}
```

* ##### *left* or *right* multiply *does not* change
  * ###### _e.g._ to backprop $\bar{\mathbf{W}}$, we need to multiply $\mathbf{x}$ (exchanger)
    * ###### the forward pass `right multiply` $\mathbf{x}$ (as ``\mathbf{z}= \mathbf{W}\mathbf{x} ``)
    * ###### the backward pass still `right multiply` $\mathbf{x}$ (with the incoming message ``\bar{\mathbf{z}}``)
* ##### but we need to apply ``^\top`` on top 
"""

# ╔═╡ 7131db97-f850-43d6-9841-fa21c5e44527
md"""


#### if you can't remember it, just do dims-check
* ##### if matches, you are correct! 

```math
\large
\begin{align}
\bar{\mathbf{W}}_{n\times m} = \bar{\mathbf{z}}_{n\times 1} \mathbf{x}^\top_{1\times m}\quad \# \text{match!} \\
\\
\bar{\mathbf{x}}_{m\times 1} = \mathbf{W}^\top_{m\times n}\bar{\mathbf{z}}_{n \times 1}\quad \# \text{match!}
\end{align}
```
"""

# ╔═╡ d54e2b98-be92-4e54-9a87-f262ee44bd06
md"""

## But why? *
"""

# ╔═╡ 877b97f3-7984-4ba6-bb79-33e8036271eb
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/matmul1.svg
# ' height = '350' /><center>"

show_img("CS5914/backprop/matmul1.svg", h=350)

# ╔═╡ 3aa8a6ed-009e-48a5-beea-84cf2e9a8b7f
md"""



## But why? *


##### Inner product ``*^\top``: acts like an _exchanger_!
"""

# ╔═╡ fb5bc004-9de4-4907-9ed9-36dd0c248508
TwoColumnWideRight(md"""


""", 
	
# 	html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/matmul2_.png
# ' height = '350' /><center>"


show_img("CS5914/backprop/matmul2_.png", h=350))

# ╔═╡ e80b5308-d37c-49aa-8779-f0602dbcd8ae
md"""

## But why? *

##### Inner product ``*^\top``: acts like an _exchanger_!
"""

# ╔═╡ d28b50c4-6785-4db1-808f-c694a91b56ff
TwoColumnWideRight(md"""
\

```math
\small
\color{salmon}
\begin{align}
\overline{\mathbf{W}} &=\begin{bmatrix}- & \overline{\mathbf{w}}_1^\top&-\\
- & \vdots & -\\
- & \overline{\mathbf{w}}_n^\top&-\end{bmatrix}
\\
&= \begin{bmatrix}- & \bar{z}_1\cdot \mathbf{x}^\top&-\\
- & \vdots & -\\
- & \bar{z}_n \cdot\mathbf{x}^\top&-\end{bmatrix}\\
&= \begin{bmatrix}\bar{z}_1\\
\vdots\\
\bar{z}_n\end{bmatrix}[-\;\; \mathbf{x}^\top -]\\
&= \bar{\mathbf{z}}\mathbf{x}^\top
\end{align}
```

""", 
	
# 	html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/backprop/matmul2_.png
# ' height = '350' /><center>"

show_img("CS5914/backprop/matmul2_.png", h=350))

# ╔═╡ e49bbb63-bf1d-4203-b322-6600324c79e8
md"""

## But why? *


##### -- _fork rule_ for $\mathbf{x}$
"""

# ╔═╡ 42b1e2cb-6476-4079-b75f-da8b65381aa2
md"""

```math
\color{salmon}
\Large\begin{align}
\bar{\mathbf{x}}&=\sum_{i=1}^n \bar{z}_i\mathbf{w}_i
= \begin{bmatrix}\mid &  & \mid \\
\mathbf{w}_1& \ldots &\mathbf{w}_n\\
\mid &  & \mid\end{bmatrix} \begin{bmatrix}\bar{z}_1\\
\vdots\\
\bar{z}_n\end{bmatrix}\\
&=\mathbf{W}^\top \bar{\mathbf{z}}
\end{align}
```

"""

# ╔═╡ db49ebaa-4df7-406a-81ce-ff1c16e52825

show_img("CS5914/backprop/matmul3.svg", w=600)

# ╔═╡ e8691786-8048-4df5-87ba-ccf064be5a44
md"""

## General matrix `multi`


#### More generally, 
```math
\Large
\mathbf{Z}_{n\times k}= \mathbf{W}_{n\times m}\mathbf{X}_{m \times k}
```

#### The backward rule (still exchanger!)
```math
\Large
\bar{\mathbf{W}}_{n\times m} = \bar{\mathbf{Z}}_{n\times k} \mathbf{X}^\top_{k \times m};\;\;\bar{\mathbf{X}}_{m\times k} = \mathbf{W}^\top_{m\times n}\bar{\mathbf{Z}}_{n\times k}
```

* you can check the dimension again 
"""

# ╔═╡ a53023a4-0173-4812-b817-cefce3cfd15b
show_img("CS5914/backprop/mat_mult_gen.svg", w=750)

# ╔═╡ 977a058d-3d1b-4b47-b2b2-1e125f1e8c52
md"""

## Matrtix `add` gate


#### Matrix addition `+` still works the same way

$$\Large\mathbf{C} =\mathbf{A}+\mathbf{B}$$

* ##### distributer
"""

# ╔═╡ cb02a9d1-8b11-406e-a50a-1f75ad159f17
show_img("CS5914/backprop/mat_add_gate.svg", w=750)

# ╔═╡ 50878c28-e279-4d69-acc0-6f8f00513a68
md"""

## More rules


#### Element-wise function 

```math
\Large
\mathbf{C} = g(\mathbf{A}) \Rightarrow  \bar{\mathbf{A}} = g'(\mathbf{A}) \odot \bar{\mathbf{C}}
```

* where ``g'`` is the gradient of $g$
* and $\odot$ denotes element-wise product ("`.*`" in `Julia`, `*` in `numpy`)



#### Reshaping
```math
\Large
\mathbf{C} = r(\mathbf{A}) \Rightarrow  \bar{\mathbf{A}} = r^{-1}( \bar{\mathbf{C}})
```

* where ``r`` denotes any reshaping operation; and ``r^{-1}`` is the reverse operation
* For example, transpose: ``\cdot ^\top``, matrix to vec `vec(A)` 
"""

# ╔═╡ 4b504a4f-7ba9-41f9-9cb1-2c3f7662f9c9
md"""

## An example*


#### Consider $z= \mathbf{x}^\top\mathbf{x}$

* it consists of reshaping $\cdot ^\top$ and matrix multiplication

* note that its local gradient is simply $\nabla_\mathbf{x} z = 2\mathbf{x}$


##### The forward computation
"""

# ╔═╡ 9746712d-49cf-44f1-9b27-b8b9c417a855
show_img("xx_example.svg", h=180)

# ╔═╡ 772a9707-9772-40ef-a4af-8f944c50ffd4
md"""

##### There is a fork, we need to add two paths' gradients

* we first backprop the `mat_mul` as an exchanger (and do not forget the transpose)

"""

# ╔═╡ e477537d-725d-42ea-8110-2cbbacd8605c
show_img("xxb1.svg", h=190)

# ╔═╡ 2aea2856-7bb3-4d3d-8530-9c9558c92f1b
md"""

##### The upper path has applied $(\cdot)^\top$ to ``\mathbf{x}``
* ``\mathbf{x} \rightarrow \mathbf{x}^\top`` (column to row vector), 
* the backward pass reverse the transpose: ``\bar{\mathbf{x}}^\top \rightarrow \bar{\mathbf{x}}`` (row to column vector)

* ###### When we add the two paths together, we get $\bar{\mathbf{x}} =2\mathbf{x}$ as expected
"""

# ╔═╡ 707be73c-509b-42b7-8418-9e9cdec5b18a
show_img("xxb3.svg", h=320)

# ╔═╡ 6d819f43-3b8c-4027-9197-6e3bed3a7921
md"""

## A non-trivial example*

#### Multiple output linear regression (``m`` outputs)
* ##### with the sum of squared errors

$$\Large\begin{align} \mathbf{z}_{m \times 1} &= \mathbf{W}_{m \times d}\mathbf{x}_{d\times 1}+\mathbf{b}_{m\times 1}\\
\mathbf{e}_{m\times 1} &= \mathbf{z}_{m\times 1} - \mathbf{y}_{m\times 1} \\
loss &= \mathbf{e}^\top\mathbf{e} = \sum_{i=1}^m e_i^2
\end{align}$$
"""

# ╔═╡ 7272aa09-278c-4220-9422-a67c616da786
md"""
#### Compare against `Zygote.jl`'s result
"""

# ╔═╡ 88e47f50-77f1-4df6-bd35-9fe6611ffbe9
md"""

## Batch mode*


##### For batch computation, we have to do broadcasting when adding the bias
* luckily, it can be represented as matrix multiplication with a vector of ones

$$\begin{align} \mathbf{Z}_{m \times n} &= \mathbf{W}_{m \times d}\mathbf{X}^\top_{d\times n}+\mathbf{b}_{m\times 1}\mathbf{1}^\top_{1\times n}\\
\mathbf{E}_{m\times n} &= \mathbf{Z}_{m\times n} - \mathbf{Y}^\top_{m\times n} \\
\mathbf{E}^2_{m\times n} &= \mathbf{E}.^2\tag{element-wise square}\\
loss &= \sum_{i=1}^m\sum_{j=1}^n \mathbf{E}^2_{ij}
\end{align}$$
"""

# ╔═╡ 821ae99d-e44f-4baa-af3f-4ae79481f7ae
md"#### Check against `Zygote.jl`"

# ╔═╡ 9eaa6d78-85bc-4d2d-a63f-374d7690ba29
begin
	Random.seed!(123)
	nobs = 100 ## number of obser
	mm = 3 ## number of targets
	dim = 5 ## number of features 
	Xtrain = randn(nobs, dim) ## design matrix
	Ytrain = randn(nobs, mm) ## multi output target
	W = randn(mm, dim)
	b = randn(mm)
end;

# ╔═╡ f1d7a7c5-1b93-4e95-a20d-f7c10cfdbc30
gW, gb = let
	xx = Xtrain[1, :]
	yy = Ytrain[1, :]
	## forward computation
	z = W * xx + b
	e = z - yy ## error
	loss = e' * e ## compute sum of squared error
	## backward computation
	dloss = 1.0 ## initialisation
	de = (2 * e) * dloss
	dz = de ## the forward pass is e = z - yy; addition gate: gradient copier
	dW = dz * xx' ## the forward pass is z = W *xx + b; mat_mul for W: gradient exchanger 
	db = dz ## the forward pass is z = W *xx + b; addition gate for b: gradient copier 
	dW, db
end

# ╔═╡ af3a82c9-d3c6-41a6-beea-3800b7bd705f
gw_zygote, gb_zygote=let
	xx = Xtrain[1, :]
	yy = Ytrain[1, :]
	_, (gw, gb) = Zygote.withgradient(W, b) do ww, b
		z = ww * xx + b
		loss = sum((yy- z).^2)
	end
	gw, gb
end

# ╔═╡ 6900ee45-1d4b-4217-8265-3b2d1de6affc
gw_zygote ≈ gW

# ╔═╡ aca13135-912e-4a1a-a2aa-c2d8f9fc1338
gb_zygote ≈ gb

# ╔═╡ 92cf9398-04c3-4b3e-a9d4-e72e908945a1
let
	## forward computation batch mode
	Z = W * Xtrain' .+ b ## broadcasting which is equivalent to WX' + b1ᵀ
	error = Z - Ytrain' ## error
	error2 = error.^2
	loss = sum(error2) ## compute sum of squared error
	# ## backward computation
	dloss = 1.0 ## initialisation
	derror2 = ones(size(error2)) * dloss ## backprop  sum: as distributor; dimension check!
	derror = derror2 .* (2 * error) ## backprop element wise ^2
	dZ = derror ## backprop error = Z - Ytrain': distributor
	dW = dZ * Xtrain ## backprop matrix multiplication, exchanger
	db = dZ * ones(100)  ## backprop matrix multiplication
	dW, db
end

# ╔═╡ 43b36ecf-63b6-4598-8425-d778f4d9548f
let
	_, (gw, gb) = Zygote.withgradient(W, b) do ww, b
		Z = ww * Xtrain' .+ b
		loss = sum((Ytrain'- Z).^2)
	end
	gw, gb
end

# ╔═╡ 58a5531e-caa1-4a0b-bf84-9ae174fd3aa2
md"""
## Backprop for Neueral Network 


"""

# ╔═╡ b0cd34a5-8b36-4ecd-96d2-b95596af3010
TwoColumn(md"""
#### Consider a neural network 

* #### one hidden layer
  * two hidden units
* #### one output unit 

* ##### biases are skipped here
""", show_img("CS5914/backprop/mlp_backprop2.svg", w=500)
)

# ╔═╡ 1c982ca6-be1d-45c0-8210-11d5cd868cca
md"""

## Computational graph representation 


"""

# ╔═╡ fd357cc4-b18c-480a-a5a7-1683ce93a5ae
show_img("CS5914/backprop/mlp_backprop1.svg", w=450)

# ╔═╡ 0fd35bbf-5f7f-46bf-ba9e-49c1ba2251eb
md"""

#### _Computational flow graph_ representation
* draw all parameters ``\mathbf{w}_1^{(1)}, \mathbf{w}_2^{(1)}, \mathbf{w}^{(2)}`` specifically as nodes
* handy for gradient derivation
"""

# ╔═╡ 5f495ff4-0c07-44f4-a119-b3ac09de1ecf
show_img("CS5914/backprop/mlp_forward.svg", w=600)

# ╔═╡ cba5b614-d554-4c2f-a8cc-6419111fc8e8
md"""

## Flow-graph with matrices
"""

# ╔═╡ a124a584-105b-49a8-9bf2-40ddeef8c895
show_img("CS5914/backprop/nnet_flow_mat.svg", w=640)

# ╔═╡ 6bc6706c-dcf3-4d9c-9a10-a38842207ac2
show_img("CS5914/backprop/nnet_flow_mat1.svg", w=680)

# ╔═╡ 425b3c00-0a97-4074-b065-95c55f85ed7e
md"""

## Backpropagation 


##### _Initialisation_
"""

# ╔═╡ cbf5c53a-c99d-46b5-8b0d-bc6bf76a5320
show_img("CS5914/backprop/nnet_back0.png", w=630)

# ╔═╡ 7296f399-dbfa-47b2-9966-c4ea414ec436
md"""

## Backpropagation (cont.)
 


"""

# ╔═╡ 55491804-e113-4444-89af-d79e3fb05d7c
show_img("CS5914/backprop/nnet_back1.png", w=630)

# ╔═╡ 916eedd2-ae35-4cce-940a-39323953e43f
md"""

## Backpropagation (cont.)
 

##### _matrix mult gate -- exchanger!_
"""

# ╔═╡ 2e0e996a-a023-4923-a81f-764e6dddd115
show_img("CS5914/backprop/nnet_back2.png", w=630)

# ╔═╡ 8e79fc83-0939-436d-8c67-eca94c6cfcba
md"""

## Backpropagation (cont.)
 

##### _backprop element-wise activations_
"""

# ╔═╡ 1108c22f-3b6b-4c2c-abde-d294b97eca01
show_img("CS5914/backprop/nnet_back3.png", w=630)

# ╔═╡ 740f97fc-7604-495c-a058-9c19c3c5fc80
md"""

## Backpropagation (cont.)*
 

##### _backprop element-wise activations_

##### Here is why:
"""

# ╔═╡ 6a3269d2-ca76-4218-a02a-9366c1bfac30
show_img("CS5914/backprop/ele_act.png", w=630, center=false)

# ╔═╡ 2bed91e8-4fac-4629-98a0-31df81a8ca3b
show_img("CS5914/backprop/ele_act1.png", w=650, center=false)

# ╔═╡ b88c7683-8e59-4209-9f2a-0f55ddc70e8d
md"""

## Backpropagation (cont.)
 

##### _backprop element-wise activations_

##### `ReLu` activation example, recall `Relu` backprop as a switch
  * if the input is greater than 0, the gradient is allowed to flow back (otherwise, 0)

###### if we group $\mathbf{a}^{(1)} = \begin{bmatrix}a_1^{(1)}\\ a_2^{(1)}\end{bmatrix}$, and $\mathbf{z}^{(1)} = \begin{bmatrix}z_1^{(1)}\\ z_2^{(1)}\end{bmatrix}$, 

* ###### the vectorised backprop is just


$$\large\bar{\mathbf{z}}^{(1)} = I.(\mathbf{z}^{(1)} .> 0) \odot \bar{\mathbf{a}}^{(1)}$$
"""

# ╔═╡ 07c8157f-795a-4b4e-b895-04fa5598573b
show_img("CS5914/backprop/ele_act_relu.png", w=630)

# ╔═╡ e19e6b81-a0ba-4a51-a2e4-a5139d60c08b
md"""

## Backpropagation (cont.)
 

##### _matrix mult gate -- exchanger!_
"""

# ╔═╡ 053a909c-44c3-42ef-8cde-3b042a51daeb
show_img("CS5914/backprop/nnet_back5.png", w=630)

# ╔═╡ c80d3ea8-c3bd-44b6-80b0-3af9fe71e127
md"""

## Show me the code -- forward pass

"""

# ╔═╡ c628e757-0924-4665-b06c-5eb1c29a5a7b
show_img("CS5914/backprop/nnet_flow_mat1.svg", w=680)

# ╔═╡ 85a614e4-97ce-43b1-990a-67b719cf8499
md"""

```julia
## forward pass
z₁ = W₁ * x # first hidden layer
a₁ = relu.(z₁) # ReLu activations (note the "." for element-wise operation)
z₂ = W₂ * a₁ # output layer
l = 0.5 * (y - z₂)^2 # loss
```

"""

# ╔═╡ 5b885a9d-92f1-46db-836f-7b6bacbbbcd0
md"""

## Show me the code -- backward pass

"""

# ╔═╡ 35e4b98b-3570-4d28-9bb5-268a9d31d9c5
show_img("CS5914/backprop/nnet_back5.png", w=680)

# ╔═╡ e3ddd42d-2170-48e0-9ea8-4edc2f64ae9e
md"""

```julia
## forward pass
z₁ = W₁ * x # first hidden layer
a₁ = relu.(z₁) # ReLu activations (note the "." for element-wise operation)
z₂ = W₂ * a₁ # output layer
l = 0.5 * (y - z₂)^2 # loss
```

"""

# ╔═╡ 43675330-3f08-4d7d-afef-838678433e03
md"""
```julia
## backward pass
dl = 1.0 # initialisation
dz₂ = (z₂ - y) * dl 
dW₂ = dz₂ * a₁'  # mat_mul: exchanger and a₁': a₁ᵀ 
da₁ = W₂' * dz₂  # mat_mul: exchanger
	
dz₁ = (z₁ .> 0) .* da₁ # ReLu: as a switch
dW₁ = dz₁ * x'   # mat_mul: exchanger 
```
"""

# ╔═╡ 44702524-6f20-460f-aab6-c3ecdaf4aa23
md"""

## With biases `b`

#### Remember for `+`, it backprop as gradient copier
"""

# ╔═╡ 9058e842-50e6-416b-baa2-12de099a93ed
md"""

```julia
## forward pass
z₁ = W₁ * x + b₁ # first hidden layer with bias
a₁ = relu.(z₁) # ReLu activations (note the "." for element-wise operation)
z₂ = W₂ * a₁ + b₂ # output layer
l = 0.5 * (y - z₂)^2 # loss
```

"""

# ╔═╡ 4009aa7f-0444-4244-8f1e-4265e80b7a51
md"""
```julia
## backward pass
dl = 1.0 # initialisation
dz₂ = (z₂ - y) * dl 
dW₂ = dz₂ * a₁'  # mat_mul: exchanger and a₁': a₁ᵀ 
db₂ = dz₂ # add gate: gradient copier 
da₁ = W₂' * dz₂  # mat_mul: exchanger
dz₁ = (z₁ .> 0) .* da₁ # ReLu: as a switch
dW₁ = dz₁ * x'   # mat_mul: exchanger 
db₁ = dz₁
```
"""

# ╔═╡ 8c0bfd17-6fae-40b6-a8cc-6f1cf53ce022
md"""

# Case study (Neural Net)

## Why I need to know auto-diff?


* #### Well, being curious about things you don't yet know should be in our DNA (as a homo-sapiens)



* #### You may at some point need to differentiate some functions 
  * e.g. existing `Auto-diff` packages are limited (in-place mutation is now allowed in `Zygote.jl`)



* #### You know why/why not/how Neural Net work at a new level
"""

# ╔═╡ bc373277-9d24-4f85-83b4-96e7f34cfd69
md"""


## Remember the learn to sort problem?



#### _Can we train a computer to learn to sort numbers ?_
* ###### without explicitly programming it
* ###### but treat it as a ML problem 


"""

# ╔═╡ afdfddc7-cb15-4d65-b608-b557981a1506
ThreeColumn(md"""

###### *Input:*
> **unsorted** array 
>
> $$\large[4,3,2]$$

""", 
	
md""" 

\

```math
\Large
\xRightarrow[]{\text{learn to sort}} 
```
""", md""" 
###### *Ouput:*
> **sorted** array 
>
> $$\large[2,3,4]$$

""")

# ╔═╡ 155f2f18-6e36-4619-ab7d-26595c2c7cb1
md"""



#### We are going to implement a neural net from scratch

* ##### "hand-shred" the `backpropagation` algorithm by `LinearAlgebra.jl` only
"""

# ╔═╡ 2d06a027-8e3c-4f0f-b186-166c1a196806
md"""
## The problem
##### Consider a simplified problem (multi target regression)
* ##### _fixed sized_ arrays with *three elements* btw 1 and 100

```math
[40, 2, 78], [2,4, 1], [90,5,10], \ldots
```
"""

# ╔═╡ 2945d608-9df2-4db3-8848-ce658dc86a3b
md"""

* ##### training data

```math
\begin{align}
\mathbf{x:} \text{unsorted} & \;\; & \mathbf{y:} \text{sorted}\\
[40, 2, 78] & &[ 2, 40, 78]\\
[2, 4, 1] & &[ 1, 2, 4]\\
[90, 5, 10] & &[ 5, 10, 90]\\
\vdots & & \vdots

\end{align}
```


* ##### we are going to solve it as a regression problem
  * ##### but multi-output regression $$\large h(\mathbf{x}; \theta) = \hat{\mathbf{y}} \in \mathbb{R}^3$$
  * ##### and the sum of squared error loss
$$\large\mathcal{L} = \frac{1}{n}\sum_{i=1}^n\sum_{d=1}^3 (\mathbf{y}_d^{(i)} - \hat{\mathbf{y}}_d^{(i)})^2$$

"""

# ╔═╡ 9477671c-b275-4a8c-b77d-dc5f986c7148
md"""

## Demo -- _sorting_ as regression 
"""

# ╔═╡ ce314a72-0af2-4b57-97c9-4386bcfdc89b
begin
	Random.seed!(123)
	seq_length = 3
	numbers = 1:100
	total = 10_000
	Xs = [] 
	Ys = []
	for _ in 1:total
		xs_ = sample(numbers, seq_length; replace = false)
		# xs_ = rand(seq_length) * 100
		ys_ = sort(xs_)
		push!(Xs, xs_)
		push!(Ys, ys_)
	end
	Xs = hcat(Xs...)
	Ys = hcat(Ys...)
	test_X = Xs[:, Int(total/2+1):end]
	test_Y = Ys[:, Int(total/2+1):end]
	train_X = Xs[:, 1:Int(total/2)]
	train_Y = Ys[:, 1:Int(total/2)]
end;

# ╔═╡ 5b4c7b7d-7312-4a85-8c12-d5e3cffa0400
md"Training data ``\mathbf{x}``: each row is one training instance"

# ╔═╡ 0e56879c-b494-4ab7-b65f-92ca1fb98b2f
train_X'

# ╔═╡ cd12123a-2519-4d2f-89cd-7d93517724f0
md"Training targets: ``\mathbf{y}``: each row is one training target"

# ╔═╡ 5386acfe-6406-42a9-bcae-03981005449a
train_Y'

# ╔═╡ 79149c07-a8f0-4bce-b58f-8e8569d6f212
md"""

## The neural net


#### A neural network (with two hidden layers)

* ##### the output is multiple output (``3`` outputs)

* ##### first hidden layer: ``50`` neurons: ``\mathbf{W}^{(1)}_{50 \times 3}``, and ``\mathbf{b}^{(1)} \in \mathbb{R}^{50}``
  * `ReLu` activation

* ##### second hidden layer: ``50`` neurons: ``\mathbf{W}^{(2)}_{50 \times 50}``, and ``\mathbf{b}^{(2)} \in \mathbb{R}^{50}``
  * `ReLu` activation


* ##### output layer ``\mathbf{W}^{(3)}_{3 \times 50}, \mathbf{b}^{(3)} \in \mathbb{R}^3``
"""

# ╔═╡ 8ff3c42c-2aae-4a1a-ab99-e09b64f59054
md"""
#### Initialisation
"""

# ╔═╡ 928fe376-19f4-460f-8d90-f89604741c34
begin
	hidden_size = 50
	ntrain = size(train_X)[2]
end;

# ╔═╡ fa314d87-4b29-40cc-9082-a1845fa013fc
begin
	Random.seed!(2345)
	## initialisation
	W1 = randn(hidden_size, seq_length) .* sqrt(2 / (hidden_size * seq_length))
	b1 = zeros(hidden_size)
	W2 = randn(hidden_size, hidden_size) .* sqrt(2 / (hidden_size * hidden_size))
	b2 = zeros(hidden_size)
	W3 = randn(seq_length, hidden_size) .* sqrt(2 / (seq_length * hidden_size))
	b3 = zeros(seq_length)
end;

# ╔═╡ 1980e2dd-da14-4b3b-ab2b-8135c83b8e35
function relu_(z)
	max(z, 0)
end;

# ╔═╡ e4462cca-ba52-41ea-b60f-e1671aa407d5
md"""

#### Forward and backward pass
"""

# ╔═╡ 620bbc2a-849c-4684-8a34-7279b018274a
# let
# 	idx = 1
# 	xi = train_X[:, idx]
# 	yi = train_Y[:, idx]
# 	xi, yi
# 	z1 = W1 * xi + b1
# 	a1 = relu_.(z1)
# 	z2 = W2 * a1 + b2
# 	a2 = relu_.(z2)
# 	y_preds = W3 * a2 + b3
# 	error = y_preds - yi
# 	li = 0.5 * sum(error.^2)
# end

# ╔═╡ 278a11a5-83de-46d8-8552-ab611424b2eb
md"""

```math
l = \frac{1}{2} \|\text{error}\|_2^2 = \frac{1}{2} \text{error}^\top\text{error}
```

* where $\text{error} = \hat{\mathbf{y}} - \mathbf{y}$
```math
\nabla_{\text{error}} l = \frac{1}{\cancel{2}} \cancel{2} \times \text{error} = \text{error}
```
"""

# ╔═╡ 249fc727-a379-452c-af06-9980904db1ec
show_img("learn2sortnnet.svg", w=750)

# ╔═╡ 8ce12db7-6f1d-4223-ae61-98170a477a35
function update!(Θ, gradΘ, γ)
	for (θ, dθ) in zip(Θ, gradΘ)
		θ .-= γ * dθ
	end
end;

# ╔═╡ de2c730a-4069-4219-b2b0-86674bc5ca17
begin
	γ = 0.00001
	losses = []
	nsize = size(train_X)[2] # training size
	Θ = (W1, b1, W2, b2, W3, b3)
	for e in 1:20
		ids = shuffle(1:ntrain)
		loss = 0.0
		for idx in ids
			xi = train_X[:, idx]
			yi = train_Y[:, idx]
			## forward computation
			z1 = W1 * xi + b1
			a1 = relu_.(z1)
			z2 = W2 * a1 + b2
			a2 = relu_.(z2)
			y_preds = W3 * a2 + b3
			error = y_preds - yi ## error = y_preds + (-1)*yi
			li = 0.5 * sum(error.^2)
			loss += li
		
			## backward 
			dli = 1.0 ## ∂li/∂li = 1.0
			derror = error * dli
			dy_preds = derror
			db3 = dy_preds
			da2 = W3' * dy_preds
			dW3 = dy_preds * a2'
			dz2 = (z2 .> 0) .* da2
			db2 = dz2
			dW2 = dz2 * a1'
			da1 = W2' * dz2
			dz1 = (z1 .> 0) .* da1
			dW1 = dz1 * xi'
			db1 = dz1
			## stochastic gradient descent
			gradΘ = (dW1, db1, dW2, db2, dW3, db3)
			update!(Θ, gradΘ, γ)
		end

		push!(losses, loss/nsize)
	end

	# Θ = (W1, b1, W2, b2, W3, b3)

end;

# ╔═╡ d1c0f2b1-4ef0-40a9-b1ae-284f5f4b0af4
plot(losses, xlabel="Epoch", ylabel="Loss", label="")

# ╔═╡ ff05845e-6a87-4601-9b22-2a02ff7430d3
# begin
# 	Random.seed!(123)
# 	## initialisation
# 	W1 = randn(hidden_size, seq_length) .* sqrt(2 / (hidden_size * seq_length))
# 	b1 = zeros(hidden_size)
# 	W2 = randn(hidden_size, hidden_size) .* sqrt(2 / (hidden_size * hidden_size))
# 	b2 = zeros(hidden_size)
# 	W3 = randn(seq_length, hidden_size) .* sqrt(2 / (seq_length * hidden_size))
# 	b3 = zeros(seq_length)
# 	# Θ = (W1, b1, W2, b2, W3, b3)
# 	γ = 0.00001
# 	losses = []
# 	nsize = size(train_X)[2]
# 	for e in 1:20
# 		ids = shuffle(1:ntrain)
# 		loss = 0.0
# 		for idx in ids
# 			xi = train_X[:, idx]
# 			yi = train_Y[:, idx]
# 			## forward computation
# 			z1 = W1 * xi + b1
# 			h1 = relu_.(z1)
# 			z2 = W2 * h1 + b2
# 			h2 = relu_.(z2)
# 			z3 = W3 * h2 + b3
# 			error = z3- yi
# 			li = sum(error.^2)
# 			loss += li
	
# 			## backward 
# 			dloss = 1.0
# 			derror = (2 * error) * dloss
# 			dz3 = derror
# 			dW3 = dz3 * h2' 
# 			db3 = dz3
# 			dh2 = W3' * dz3
# 			dz2 = (z2 .> 0) .* dh2
# 			dW2 = dz2 * h1'
# 			db2 = dz2
# 			dh1 = W2' * dz2
# 			dz1 = (z1 .> 0) .* dh1
# 			dW1 = dz1 * xi'
# 			db1 = dz1
# 			# gradΘ = (dW1, db1, dW2, db2, dW3, db3)
# 			update!(Θ, gradΘ, γ)
# 		end

# 		push!(losses, loss/nsize)
# 	end

# 	Θ = (W1, b1, W2, b2, W3, b3)
# end

# ╔═╡ 0a1fd159-fa08-44ab-81f7-0abf38cba097
md"#### Training accuracy: "

# ╔═╡ 883f90fe-6dc5-4312-938d-8076b7204ed3
md"#### Testing accuracy: "

# ╔═╡ d4cb02d3-fe02-40b7-8e79-857c2c71f7d9
md"""
#### Check against `Zygote.jl`
"""

# ╔═╡ 93dc79cb-3fe2-4603-a57e-782ac6e66abd
# all(grads_zygote .≈ gradΘ)

# ╔═╡ ac889070-c342-4258-ae72-e298d060c90e
grads_zygote = let
	xi = train_X[:, 1]
	yi = train_Y[:, 1]
	grads = Zygote.gradient(W1, b1, W2, b2, W3, b3) do W1_, b1_, W2_, b2_, W3_, b3_
		z1 = W1_ * xi + b1_
		h1 = relu_.(z1)
		z2 = W2_ * h1 + b2_
		h2 = relu_.(z2)
		z3 = W3_ * h2 + b3_
		error = z3 - yi
		loss = .5 * sum(error.^2)
	end
end

# ╔═╡ bbdcd4fb-d5bc-4599-9e01-8739eea70a28
function predict(Θ, X)
	W1, b1, W2, b2, W3, b3 = Θ
	n = size(X)[2]
	z1 = W1 * X .+ b1
	h1 = relu_.(z1)
	z2 = W2 * h1 .+ b2
	h2 = relu_.(z2)
	z3 = W3 * h2 .+ b3
end;

# ╔═╡ c30f667e-e949-4530-ad14-3d620047e027
Int.(round.(predict(Θ, train_X)));

# ╔═╡ a7ea1f45-e511-4feb-abd6-bd0b47e89331
mean(Int.(round.(predict(Θ, train_X))) .== train_Y) ## training accuracy

# ╔═╡ 02bbf5a2-2048-4aaf-8fd6-245943234fe7
mean(Int.(round.(predict(Θ, test_X))) .== test_Y) ## testing accuracy

# ╔═╡ f094c18f-fac3-4aa3-b986-fed742ece212
# let
# 	using Flux
# 	loader_sorting = Flux.DataLoader((train_X, train_Y); batchsize = 50);
# 	Random.seed!(1236789)
# 	nnet1 = Chain(Dense(seq_length => hidden_size, relu), Dense(hidden_size, hidden_size, relu), Dense(hidden_size, seq_length))
# 	optim = Flux.setup(Flux.Adam(0.01), nnet1) 
# 	losses = []
# 	for e in 1:10
# 		for (xs, ys) in loader_sorting
# 			li, grads = Flux.withgradient(nnet1) do model
# 				y_pred = model(xs)
# 				b_size = size(xs)[2]
# 				sum((ys - y_pred).^2)/b_size 
# 			end
# 			Flux.update!(optim, nnet1, grads[1])
# 			push!(losses, li)
# 		end
# 	end
# end;

# ╔═╡ c873b617-731b-4674-be7d-966958e79676
begin
	struct Linear
		W
		b
		act::Function
	end

	# constructor with input and output size and identity act
	Linear(in::Integer, out::Integer) = Linear(randn(out, in), randn(out), identity)

	Linear(in::Integer, out::Integer, a::Function) = Linear(randn(out, in), randn(out), a)
	
	# Overload function, so the object can be used as a function
	(m::Linear)(x) = m.act.(m.W * x .+ m.b)
end;

# ╔═╡ b30971a7-eef9-4405-b0d4-91aa3299186b
begin
	function forward(m::Linear, x)
		out = m(x)
		pullback = function backward(Δout)
			Δz = (out .> 0) .* Δout
			ΔW = Δz * x'
			Δx = m.W' * Δz
			return ΔW, Δz, Δx
		end
		return out, pullback
	end
end

# ╔═╡ 18dc5ddd-f185-4f81-975a-3ca3a095a1d4
let
	ff = function f(x) 
		
			x^2
		end

	ff(2)
end

# ╔═╡ fa02ead1-6b3c-437b-b8c2-b79318c0ad5c
identity(5)

# ╔═╡ b05411c4-1dc5-482e-ac20-110341662f7b
Linear(3, 50, relu_)(randn(3))

# ╔═╡ 0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
md"""

## Appendix
"""

# ╔═╡ 8687dbd1-4857-40e4-b9cb-af469b8563e2
function perp_square(origin, vx, vy; δ=0.1) 
	x = δ * vx/sqrt(norm(vx))
	y = δ * vy/sqrt(norm(vy))
	xyunit = origin+ x + y
	xunit = origin + x
	yunit = origin +y
	Shape([origin[1], xunit[1], xyunit[1], yunit[1]], [origin[2], xunit[2], xyunit[2], yunit[2]])
end

# ╔═╡ fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
# as: arrow head size 0-1 (fraction of arrow length)
# la: arrow alpha transparency 0-1
function arrow3d!(x, y, z,  u, v, w; as=0.1, lc=:black, la=1, lw=0.4, scale=:identity)
    (as < 0) && (nv0 = -maximum(norm.(eachrow([u v w]))))
    for (x,y,z, u,v,w) in zip(x,y,z, u,v,w)
        nv = sqrt(u^2 + v^2 + w^2)
        v1, v2 = -[u,v,w]/nv, nullspace(adjoint([u,v,w]))[:,1]
        v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
        v5 = v4 - 2*(v4'*v2)*v2
        (as < 0) && (nv = nv0) 
        v4, v5 = -as*nv*v4, -as*nv*v5
        plot!([x,x+u], [y,y+v], [z,z+w], lc=lc, la=la, lw=lw, scale=scale, label=false)
        plot!([x+u,x+u-v5[1]], [y+v,y+v-v5[2]], [z+w,z+w-v5[3]], lc=lc, la=la, lw=lw, label=false)
        plot!([x+u,x+u-v4[1]], [y+v,y+v-v4[2]], [z+w,z+w-v4[3]], lc=lc, la=la, lw=lw, label=false)
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
Latexify = "~0.16.1"
Plots = "~1.40.9"
PlutoTeachingTools = "~0.2.14"
PlutoUI = "~0.7.54"
Statistics = "~1.11.1"
StatsBase = "~0.34.2"
Symbolics = "~6.30.0"
Zygote = "~0.6.69"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "b5f047bab68c01612f832b1cc98be8a17d23f463"

[[deps.ADTypes]]
git-tree-sha1 = "e2478490447631aedba0823d4d7a80b2cc8cdb32"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.14.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cd8b948862abee8f3d3e9b73a102a9ca924debb0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.2.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "b5bb4dc6248fde467be2a863eb8452993e74d402"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.1"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bijections]]
git-tree-sha1 = "d8b0439d2be438a5f2cd68ec158fe08a7b2595b7"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.9"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "a975ae558af61a2a48720a6271661bf2621e0f4e"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.72.3"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "bce26c3dab336582805503bed209faab1c279768"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.4"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "03aa5d44647eaec98e1920635cdfed5d5560a8b9"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.117"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "a7e9f13f33652c533d49868a534bfb2050d1365f"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.7.15"

    [deps.DomainSets.extensions]
    DomainSetsMakieExt = "Makie"

    [deps.DomainSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DynamicPolynomials]]
deps = ["Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Reexport", "Test"]
git-tree-sha1 = "9a3ae38b460449cc9e7dd0cfb059c76028724627"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.6.1"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExproniconLite]]
git-tree-sha1 = "c13f0b150373771b0fdc1713c97860f8df12e6c2"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.14"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "846f7026a9decf3679419122b49f8a1fdb48d2d5"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.16+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "Statistics"]
git-tree-sha1 = "eea7b3a1964b4de269bb380462a9da604be7fcdb"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.2.2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "0ff136326605f8e06e9bcf085a356ab312eef18a"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.13"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "9cb62849057df859575fc1dda1e91b82f8609709"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.13+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "c67b33b085f6e2faf8bf79a61962e7339a81129c"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.15"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "2bd56245074fab4015b9174f24ceba8293209053"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.27"

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

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "950c3717af761bc3ff906c2e8e52bd83390b6ec2"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.14"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "71b48d857e86bf7a1838c4736545699974ce79a2"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.9"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.Jieko]]
deps = ["ExproniconLite"]
git-tree-sha1 = "2f05ed29618da60c06a87e9c033982d4f71d0b6c"
uuid = "ae98c720-c025-4a4a-838c-29b094483192"
version = "0.2.1"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "a434e811d10e7cbf4f0674285542e697dca605d0"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.42"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "80d268b2f4e396edc5ea004d1e0f569231c71e9e"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.34"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "5fcfea6df2ff3e4da708a40c969c3812162346df"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.2.0"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "4b5ad6a4ffa91a00050a964492bc4f86bb48cea0"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.35+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cd714447457c660382fe634710fb56eb255ee42e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.6"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

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

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "df37206100d39f79b3376afb6b9cee4970041c61"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.1+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89211ea35d9df5831fca5d33552c02bd33878419"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e888ad02ce716b319e6bdb985d2ef300e7089889"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.3+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "688d6d9e098109051ae33d126fcfc88c4ce4a021"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "1833212fd6f580c20d4291da9c1b4e8a655b128e"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.0.0"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.Moshi]]
deps = ["ExproniconLite", "Jieko"]
git-tree-sha1 = "453de0fc2be3d11b9b93ca4d0fddd91196dcf1ed"
uuid = "2e0e35c7-a2e4-4343-998d-7ef72827ed2d"
version = "0.3.5"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "8d39779e29f80aa6c071e7ac17101c6e31f075d7"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.5.7"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "491bdcdc943fcbc4c005900d7463c9f216aabf4c"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.6.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "cc0a5deefdb12ab3a096f00a6d42133af4560d71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "5e1897147d1ff8d98883cda2be2187dcf57d8f0c"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.15.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a9697f1d06cc3eb3fb3ad49cc67f2cfabaac31ea"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.16+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "966b85253e959ea89c53a9abebbf2e964fbf593b"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.32"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3b31172c032a1def20c98dae3f2cdc9d10e3b561"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "dae01f8c2e069a683d3a6e17bbae5070ab94786f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.9"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

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
git-tree-sha1 = "7e71a55b87222942f0f9337be62e26b1f103d3e4"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.61"

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

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "cb420f77dc474d23ee47ca8d14c90810cafe69e7"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.6"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "e96b644f7bfbf1015f8e42a7c7abfae2a48fafbf"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.31.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "9bb80533cb9769933954ea4ffbecb3025a783198"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.7.2"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "04c968137612c4a5629fa531334bb81ad5680f00"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.13"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Moshi", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "ee305515b0946db5f56af699e8b5804fee04146c"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.75.1"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseMLStyleExt = "MLStyle"
    SciMLBaseMakieExt = "Makie"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    MLStyle = "d8e11817-5142-5d16-987a-aa16d5891078"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["Accessors", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "6149620767866d4b0f0f7028639b6e661b6a1e44"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.12"
weakdeps = ["SparseArrays", "StaticArraysCore"]

    [deps.SciMLOperators.extensions]
    SciMLOperatorsSparseArraysExt = "SparseArrays"
    SciMLOperatorsStaticArraysCoreExt = "StaticArraysCore"

[[deps.SciMLStructures]]
deps = ["ArrayInterface"]
git-tree-sha1 = "566c4ed301ccb2a44cbd5a27da5f885e0ed1d5df"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "1147f140b4c8ddab224c94efa9569fc23d63ab44"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.3.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0feb6b9031bd5c51f9072393eb5ab3efd31bf9e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.13"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "5a3a31c41e15a1e042d60f2f4942adccba05d3c9"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.0"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "d6c04e26aa1c8f7d144e1a8c47f1c73d3013e289"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.38"

[[deps.SymbolicLimits]]
deps = ["SymbolicUtils"]
git-tree-sha1 = "fabf4650afe966a2ba646cabd924c3fd43577fc3"
uuid = "19f23fe9-fdab-4a78-91af-e7b7767979c3"
version = "0.2.2"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "ArrayInterface", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "ExproniconLite", "LinearAlgebra", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicIndexingInterface", "TaskLocalValues", "TermInterface", "TimerOutputs", "Unityper", "WeakValueDicts"]
git-tree-sha1 = "e3a430bead8cfc69d951e721cb10330bb7ac1fce"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "3.17.1"

    [deps.SymbolicUtils.extensions]
    SymbolicUtilsLabelledArraysExt = "LabelledArrays"
    SymbolicUtilsReverseDiffExt = "ReverseDiff"

    [deps.SymbolicUtils.weakdeps]
    LabelledArrays = "2ee39098-c373-598a-b85f-a56591580800"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

[[deps.Symbolics]]
deps = ["ADTypes", "ArrayInterface", "Bijections", "CommonWorldInvalidations", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "DynamicPolynomials", "LaTeXStrings", "Latexify", "Libdl", "LinearAlgebra", "LogExpFunctions", "MacroTools", "Markdown", "NaNMath", "OffsetArrays", "PrecompileTools", "Primes", "RecipesBase", "Reexport", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArraysCore", "SymbolicIndexingInterface", "SymbolicLimits", "SymbolicUtils", "TermInterface"]
git-tree-sha1 = "b587aa93817acfa302f12a03d6d8d4ff5d1eac2b"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "6.30.0"

    [deps.Symbolics.extensions]
    SymbolicsForwardDiffExt = "ForwardDiff"
    SymbolicsGroebnerExt = "Groebner"
    SymbolicsLuxExt = "Lux"
    SymbolicsNemoExt = "Nemo"
    SymbolicsPreallocationToolsExt = ["PreallocationTools", "ForwardDiff"]
    SymbolicsSymPyExt = "SymPy"

    [deps.Symbolics.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Groebner = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
    Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
    Nemo = "2edaba10-b0f1-5616-af89-8c11ac63239a"
    PreallocationTools = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TaskLocalValues]]
git-tree-sha1 = "d155450e6dff2a8bc2fcb81dcb194bd98b0aeb46"
uuid = "ed4db957-447d-4319-bfb6-7fa9ae7ecf34"
version = "0.1.2"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TermInterface]]
git-tree-sha1 = "d673e0aca9e46a2f63720201f55cc7b3e7169b16"
uuid = "8ea1fca8-c5ef-4a55-8b96-4e9afe9c9a3c"
version = "2.0.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f57facfd1be61c42321765d3551b3df50f7e09f6"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.28"

    [deps.TimerOutputs.extensions]
    FlameGraphsExt = "FlameGraphs"

    [deps.TimerOutputs.weakdeps]
    FlameGraphs = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

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

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unityper]]
deps = ["ConstructionBase"]
git-tree-sha1 = "25008b734a03736c41e2a7dc314ecb95bd6bbdb0"
uuid = "a7c27f48-0311-42f6-a7f8-2c11e75eb415"
version = "0.1.6"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.WeakValueDicts]]
git-tree-sha1 = "98528c2610a5479f091d470967a25becfd83edd0"
uuid = "897b6980-f191-5a31-bcb0-bf3c4585e0c1"
version = "0.1.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56c6604ec8b2d82cc4cfe01aa03b00426aac7e1f"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.4+1"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+3"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e9216fdcd8514b7072b43653874fd688e4c6c003"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.12+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89799ae67c17caa5b3b5a19b8469eeee474377db"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.5+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c57201109a9e4c0585b208bb408bc41d205ac4e9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.2+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6dba04dbfb72ae3ebe5418ba33d087ba8aa8cb00"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.1+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "0b3c944f5d2d8b466c5d20a84c229c17c528f49e"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.75"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "434b3de333c75fc446aa0d19fc394edafd07ab08"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.7"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0ba42241cb6809f1a278d0bcb976e0483c3f1f2d"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+1"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "068dfe202b0a05b8332f1e8e6b4080684b9c7700"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.47+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╟─9f90a18b-114f-4039-9aaf-f52c77205a49
# ╟─3e2e1ea8-3a7d-462f-ac38-43a087907a14
# ╟─275e626f-026b-4167-acaa-d9faa590bed7
# ╟─f6bbddfe-d23d-44d9-ba13-237c55b9567d
# ╟─7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
# ╟─bc96a33d-9011-41ec-a19e-d472cbaafb70
# ╟─8e98d5c2-3744-49b8-95f8-d832c1b28632
# ╟─42952923-cd9b-4a05-a439-ee382d4466c3
# ╟─3747dda8-b8ae-49f5-9b08-0508abedc4a6
# ╟─1a4d3572-438d-4295-a4a3-ca603cc8d238
# ╟─2076ccbd-01a2-48a3-898d-2a464f1a20b6
# ╟─6e3892f0-0ddf-4367-b6b3-88e573f62b09
# ╟─3c51d1e9-dfc1-44fc-a877-3e454de06736
# ╟─591c5ddd-7a73-48e9-8f2a-d634fad62ca6
# ╟─963dd96b-94f3-4bc0-ab20-4ce8ee9f1e37
# ╟─a40a6b23-1853-401c-b470-1ec5779c5df1
# ╟─01d0df22-a170-41af-99fa-137f477d1a58
# ╟─bc234d58-096a-4f40-a2e5-fe170d5c0282
# ╟─6a17976f-e242-446f-a631-849663486ffe
# ╟─c0ca19ab-7223-4125-a656-81fc9eb3db4c
# ╟─72873e0f-9cb6-474a-974a-b21289b93ca8
# ╟─d09cbf01-a61c-4a49-990e-ef4a6eaca12c
# ╟─12fe101b-f602-4570-98a8-6362098662b2
# ╟─937072a0-9487-424d-b427-a280f85bc448
# ╟─70aae3d9-de8d-411e-8eac-759092e474e3
# ╟─d1076724-3e36-4801-aef3-ec19f3bcce72
# ╠═01585427-e485-4421-8590-1c1964f8bb28
# ╠═f698f63c-ac0f-4891-9c0c-ca42db4b4555
# ╟─96e85812-0f24-476b-8b1c-5c443e80852f
# ╠═a8b5b0c0-166d-4422-b5db-41900178eace
# ╟─69f6e60e-6c75-4be1-b718-259ca0cfe97b
# ╟─9444fffc-57ce-4a38-af09-ceaf25100da8
# ╟─dac1bf8b-e538-4282-ba0c-cca030ecc5aa
# ╟─094f8059-38c5-4ddf-b5c1-a0d502d8fca7
# ╟─5dcffcea-864f-4657-9904-9da542b69ed5
# ╟─2dfcc44b-40e9-4aef-8b09-44ec39cf6f35
# ╟─700fdce1-16d4-4179-8ed7-dce1a19cf8a9
# ╟─b3dcb542-e1d2-45e0-b9c1-99a544c930e8
# ╟─e5dd83b3-3744-4d1d-9aa2-5f24c036e16f
# ╟─62c244b3-fa36-472b-98a3-9993172ae9a4
# ╟─86c5226a-97ce-4afe-a673-c799f78765ea
# ╟─68f733fa-d6eb-417c-a33e-9b9c2bf803ef
# ╟─df80e910-b219-40d6-9d66-6a8264fed915
# ╟─6e7266ad-dbbf-4058-b650-3d7d013b120b
# ╟─6c8aabfd-3a24-45ff-801a-caf8f07b1c4a
# ╟─2ca35388-7302-4f04-a67c-fa29c3d51e4d
# ╟─02d9c597-c4be-471c-acc5-6dc3c0c56b1f
# ╟─d6abb07e-595e-42c2-9a98-ca09caf156a2
# ╟─244030b7-b0fe-4b51-951e-3142346d03f2
# ╟─b41eb406-f681-4c27-a041-6d932e1d19b2
# ╟─0c3601f1-406a-4ce8-a3ad-dd6ac5ef478e
# ╟─49c5f93f-8b24-4703-95ed-3177ff3cdc1a
# ╟─5fcfe9b6-d0e4-4bc9-86d2-4bdac90e1ec7
# ╟─7e8c4ebc-67f4-446d-932f-817382979854
# ╟─5e7cc34b-73b3-4342-94bb-4be6969fb040
# ╟─6405e556-5a7f-4e1f-8ccc-f5ccce191c5b
# ╟─33e47f17-b4ef-4206-b719-a2ac81355e56
# ╟─c0a731b8-4732-4561-9394-1d9d6b2adfb7
# ╟─2a682b20-c48a-478f-af99-18dcb441c919
# ╟─caf32470-457e-4f0a-9f1a-9a90704a103c
# ╟─93925a5f-c287-478a-81da-86eee8aaea08
# ╟─b002eb10-1c33-47a7-84bf-239b67c26988
# ╟─37bded16-7d2f-4317-b5c1-db6185cc3800
# ╟─c96d7374-99de-4475-9f63-021208959af6
# ╟─4253d341-e1b9-4764-bcf7-7e467b21b225
# ╟─24d74ebe-eb37-4556-bab5-b668f5e7af9c
# ╟─b0ec7a68-d684-4eac-91d5-a7fe71c40c69
# ╟─851a7cea-4fed-4049-933c-92da77d68c0d
# ╟─754d270f-6d24-4240-b638-3003eb5d4420
# ╟─b7f16835-f05d-4967-814e-57e11af3dbbc
# ╟─433d7769-a9fb-4b57-b50c-6e5ccef506f4
# ╟─75e3a5d9-5631-4170-8ca7-e824c1bef993
# ╟─7e58f5a9-5378-4f09-8993-b9dc6766f6ac
# ╠═39850af7-6d6e-48d4-8d19-f96e3966571f
# ╟─930bd2d3-0028-46aa-a0be-cb0e4973943e
# ╠═186988ed-6373-416c-84c4-a9cbd0a6a169
# ╟─3d4c3e71-2b33-4e00-a710-792c604466c6
# ╠═a1366fa3-f39d-48f3-80b7-d6174f5019af
# ╠═b35d4772-d810-4b95-92ac-d5dd87353f86
# ╟─ba807661-3c95-4d18-bca4-be2e5606f697
# ╠═3b7546dd-1025-494a-893d-1297f488093a
# ╟─0fb3579f-9cde-44d7-93f0-725939e25ffe
# ╠═93f43a4d-b5f1-456b-8565-dea84eb67790
# ╠═0f52879e-a31f-44cc-9ee3-665fff477f54
# ╟─ffc4c9af-32b0-4720-ad36-a9fd5fd1df94
# ╠═6019dc3b-c0ec-4a28-9539-51e2fa19559e
# ╠═4dc93a97-6c75-45d9-8ed7-aeaaf74389e9
# ╠═978da62a-e03c-4539-a859-acd0a78ce425
# ╠═6fcfb337-8e25-4142-9a67-31c9887fbd2f
# ╟─90784f49-fbed-489c-8f7f-6e182ee68f8b
# ╠═0b1a867e-fe88-4bd3-b3fb-311506647140
# ╠═16eefeee-3a1e-491d-8f33-bba535f778ef
# ╟─0cb656d6-c792-4662-958e-7456f0300a18
# ╠═c9429b12-4afd-4997-b0be-1d1d22f3cc13
# ╠═371d4e51-0c19-4430-96b9-192d72ad5824
# ╟─0118fc3f-e0e1-480d-9b23-8bcfa55feebc
# ╟─3faea791-befb-40ac-85d6-8e6194ccdb6a
# ╠═2777a33d-6959-4f6f-a8f7-b666b56dc0a6
# ╟─5aba6c98-faf1-45a7-a8c3-325a96e3e444
# ╠═e1cf2c66-d6aa-4fed-ba4e-d798d36205ca
# ╠═8acc5a4b-cc2b-458d-8818-59c5ff52b524
# ╟─20c9c798-52b2-4f56-8ca6-7124a778fd43
# ╠═8c86613a-9a95-4687-bbff-8fe12457e2c9
# ╠═ade06efb-109f-4874-8e60-a9faaf4e14ff
# ╠═0dfa109a-4612-479f-9379-61d63ae5295a
# ╟─353346df-3cd9-48bc-8d50-bd0bc1fba07e
# ╠═cc6ffeb0-8dcb-40f0-a190-475af17ce2f0
# ╠═e1c6edc6-e5c0-4e18-818f-67043977c808
# ╟─315b7bfa-c215-4b32-b243-91c0a0a9d577
# ╠═88f23dfa-615d-49a7-90bc-97b56f73d1af
# ╠═bedb1024-3458-4c5c-83f3-09cae1034db2
# ╟─18cc560a-eb59-40e2-869f-e7169b0fa9f3
# ╟─0f9f63a8-1232-4fb7-b3b2-4b52cba7728f
# ╟─b642f207-5a8e-495d-89e4-b98f13f5a165
# ╟─5f84553f-2acf-409d-af06-77462962d1f0
# ╟─724e8726-6bd2-4dfc-8394-fe2aee6a9963
# ╟─c53f3f55-ccc3-493e-b657-fa141134a378
# ╟─a4e4638a-1d2c-49d1-aa53-2614e6c78c82
# ╟─589b1b4f-ef8b-4f0f-b2b9-099314ec0a1c
# ╟─e6f15720-71c9-4e4a-924a-2671b994aba9
# ╟─cd916288-15f0-4f9e-be05-b756b55122ca
# ╟─0af94490-d524-478c-9d61-c1b00055e3df
# ╟─1c9c8efa-fc0e-47a6-9e94-a8c9aa89bf79
# ╟─ce47f14e-ab0e-4e9c-ac0f-493bc7e592a5
# ╟─65e8cfd2-c430-4479-8311-1cef1dfe03d2
# ╟─62729b84-6c72-45e5-afa5-b2a5c886d674
# ╟─0530448d-a50c-4970-8c8e-d69bc1b8dd25
# ╟─a882ceef-e161-4dad-8aab-c9493d3bb99c
# ╟─7f6532b2-0ab8-4f50-942b-d7a6a12560b1
# ╟─f5b62aca-380c-4631-97c2-e88fd492c95c
# ╟─a17a212e-bdd9-423b-8182-571e3571ca01
# ╟─a0685a47-88f4-4ebb-b6f6-9722f77fbd18
# ╟─e6573211-06a2-4606-9caf-d97fb88f1e00
# ╟─d13d806e-ecc8-48d3-a7c5-38a1e4c3cd1b
# ╟─59effbcd-4334-4b19-9d00-38805577d4e3
# ╟─dc012b20-9fb9-4757-98ae-50e81525715b
# ╟─b1dcd923-5170-4aca-ac43-936d36c98a47
# ╟─b6445302-6a19-4677-8ba6-4e5850a17f91
# ╟─05c921ee-ef17-4e3f-9380-1929a1a98761
# ╟─d2d57c21-4c16-47d5-b9dc-9e9100d361bc
# ╟─5443b0f4-1b2c-4d23-a2c2-72f6f6dbb8a1
# ╟─26bcb663-2eb1-4542-a50f-7f8e26d17c13
# ╟─f65deaf9-42c6-4a6a-bde1-45e9f3191294
# ╟─8f7d61a6-b580-4277-a70d-00ba79d9547a
# ╟─872f54bc-fc6a-4e52-a99f-357d45c3e68a
# ╟─b798636d-6c13-4aa7-bb9b-fc1e4e01a11b
# ╟─2aedeac6-d43c-4a0e-8e0b-f46e904600a2
# ╟─d689bd7b-ce37-4373-8228-d29ac00adfa0
# ╟─af15d1ec-b1ba-4c69-8e46-9b6d6e31523c
# ╟─a855cfae-a5cb-4c94-9d7a-44387ad27060
# ╟─166b5b82-3f8c-43e4-a469-1b9cce08bc4e
# ╟─82b57d6c-befa-4246-9b03-212e08c1d706
# ╟─72f36dfe-89b5-4477-a83d-b6fd0046e96d
# ╟─f19c8378-7148-4baf-b664-a00d8805fbe8
# ╟─cf54b8dd-2020-44f3-a93b-1a449769c528
# ╟─b3df5d95-f975-47a1-a338-8766c15f027c
# ╟─140e3645-4c1b-4737-a458-45b014299349
# ╟─7b0773f7-acad-4773-9207-1d09f632b0a0
# ╟─e584e6b5-3f2a-4e12-8706-db27a507eeaa
# ╟─f9a54b6b-632a-410c-8501-e40109ee7605
# ╟─999f8bef-39c5-4ac3-97b9-60bd09189f26
# ╟─2ba2c7ea-e7a1-474b-a696-1a15c850637b
# ╟─f2729651-6193-4234-8d04-20edc76bf013
# ╟─82a2d879-170b-469f-a6e9-f2c6269937cc
# ╟─7d50e6d3-013e-4f13-aeb9-5190f8c37e70
# ╟─9298c327-bea7-48ac-bded-315dc9d76208
# ╟─9d3cb890-47ec-42a8-b67e-a6a675a81300
# ╟─5e3687c2-8e18-4fac-81ec-602edb79a091
# ╟─d474e7df-9dd4-4a52-9644-5e86309a5063
# ╟─8f11acf4-861d-4c43-b24e-6a0f6e095ed4
# ╟─f5786bb9-1665-415e-92a1-1d7fc337f9f3
# ╟─42ddbd1b-0711-4b5e-b327-81864aac0c62
# ╟─190fac3d-25cc-4d7d-91f4-199d6f2932c0
# ╟─87620132-1261-4b7c-8ad9-933cc64de277
# ╟─7c29d7f8-2479-4e5b-94e7-7040453ac4db
# ╟─03eede9e-d7f4-4a5d-ab0c-9daa9e72594b
# ╟─d4ce1390-f96e-45d8-a034-f869b5bd1ee4
# ╟─ccef6f05-4f1b-4377-86e3-cc477176de01
# ╟─dd3f1337-c5f9-41b8-8d1b-3b657404d4d3
# ╟─5174b7c2-f1d3-437f-b6b5-5a9f89f555d6
# ╟─fb958c4b-5c8a-4bf2-bccc-3a7cafa6406d
# ╟─35943fab-8e85-40f5-9397-f3577c935802
# ╟─173b9082-1536-44b0-865e-dfe7e5ee8576
# ╟─537fdffa-0343-4b01-8ea2-092049597cdc
# ╟─87a8fa4c-5da2-48d6-a007-a6952dc4ebfb
# ╟─12399ab0-1c54-42bf-bedc-5e135ee9c035
# ╟─1de3e833-cf73-45b5-83cd-9b1cfaec6fd4
# ╟─cb4d586d-1b8e-41a2-bc98-e9f70d0a45d0
# ╟─18db9aae-ea09-4a2b-848c-05c10416a733
# ╟─29ac1207-584c-4c12-86c2-1d8ee2168db1
# ╟─11624c75-0199-499a-8393-e06295763fb8
# ╟─8cddc677-5796-489e-bb42-2ae13471090d
# ╟─1110d206-09c3-4bbb-8932-aec832548a1b
# ╟─a33e7154-6440-44f2-b73a-94a60aff6bd1
# ╟─7131db97-f850-43d6-9841-fa21c5e44527
# ╟─d54e2b98-be92-4e54-9a87-f262ee44bd06
# ╟─877b97f3-7984-4ba6-bb79-33e8036271eb
# ╟─3aa8a6ed-009e-48a5-beea-84cf2e9a8b7f
# ╟─fb5bc004-9de4-4907-9ed9-36dd0c248508
# ╟─e80b5308-d37c-49aa-8779-f0602dbcd8ae
# ╟─d28b50c4-6785-4db1-808f-c694a91b56ff
# ╟─e49bbb63-bf1d-4203-b322-6600324c79e8
# ╟─42b1e2cb-6476-4079-b75f-da8b65381aa2
# ╟─db49ebaa-4df7-406a-81ce-ff1c16e52825
# ╟─e8691786-8048-4df5-87ba-ccf064be5a44
# ╟─a53023a4-0173-4812-b817-cefce3cfd15b
# ╟─977a058d-3d1b-4b47-b2b2-1e125f1e8c52
# ╟─cb02a9d1-8b11-406e-a50a-1f75ad159f17
# ╟─50878c28-e279-4d69-acc0-6f8f00513a68
# ╟─4b504a4f-7ba9-41f9-9cb1-2c3f7662f9c9
# ╟─9746712d-49cf-44f1-9b27-b8b9c417a855
# ╟─772a9707-9772-40ef-a4af-8f944c50ffd4
# ╟─e477537d-725d-42ea-8110-2cbbacd8605c
# ╟─2aea2856-7bb3-4d3d-8530-9c9558c92f1b
# ╟─707be73c-509b-42b7-8418-9e9cdec5b18a
# ╟─6d819f43-3b8c-4027-9197-6e3bed3a7921
# ╠═f1d7a7c5-1b93-4e95-a20d-f7c10cfdbc30
# ╟─7272aa09-278c-4220-9422-a67c616da786
# ╠═6900ee45-1d4b-4217-8265-3b2d1de6affc
# ╠═aca13135-912e-4a1a-a2aa-c2d8f9fc1338
# ╠═c34e56cb-5a92-4282-aa62-4c11deaff0fd
# ╠═af3a82c9-d3c6-41a6-beea-3800b7bd705f
# ╟─88e47f50-77f1-4df6-bd35-9fe6611ffbe9
# ╠═92cf9398-04c3-4b3e-a9d4-e72e908945a1
# ╟─821ae99d-e44f-4baa-af3f-4ae79481f7ae
# ╠═43b36ecf-63b6-4598-8425-d778f4d9548f
# ╟─9eaa6d78-85bc-4d2d-a63f-374d7690ba29
# ╟─58a5531e-caa1-4a0b-bf84-9ae174fd3aa2
# ╟─b0cd34a5-8b36-4ecd-96d2-b95596af3010
# ╟─1c982ca6-be1d-45c0-8210-11d5cd868cca
# ╟─fd357cc4-b18c-480a-a5a7-1683ce93a5ae
# ╟─0fd35bbf-5f7f-46bf-ba9e-49c1ba2251eb
# ╟─5f495ff4-0c07-44f4-a119-b3ac09de1ecf
# ╟─cba5b614-d554-4c2f-a8cc-6419111fc8e8
# ╟─a124a584-105b-49a8-9bf2-40ddeef8c895
# ╟─6bc6706c-dcf3-4d9c-9a10-a38842207ac2
# ╟─425b3c00-0a97-4074-b065-95c55f85ed7e
# ╟─cbf5c53a-c99d-46b5-8b0d-bc6bf76a5320
# ╟─7296f399-dbfa-47b2-9966-c4ea414ec436
# ╟─55491804-e113-4444-89af-d79e3fb05d7c
# ╟─916eedd2-ae35-4cce-940a-39323953e43f
# ╟─2e0e996a-a023-4923-a81f-764e6dddd115
# ╟─8e79fc83-0939-436d-8c67-eca94c6cfcba
# ╟─1108c22f-3b6b-4c2c-abde-d294b97eca01
# ╟─740f97fc-7604-495c-a058-9c19c3c5fc80
# ╟─6a3269d2-ca76-4218-a02a-9366c1bfac30
# ╟─2bed91e8-4fac-4629-98a0-31df81a8ca3b
# ╟─b88c7683-8e59-4209-9f2a-0f55ddc70e8d
# ╟─07c8157f-795a-4b4e-b895-04fa5598573b
# ╟─e19e6b81-a0ba-4a51-a2e4-a5139d60c08b
# ╟─053a909c-44c3-42ef-8cde-3b042a51daeb
# ╟─c80d3ea8-c3bd-44b6-80b0-3af9fe71e127
# ╟─c628e757-0924-4665-b06c-5eb1c29a5a7b
# ╟─85a614e4-97ce-43b1-990a-67b719cf8499
# ╟─5b885a9d-92f1-46db-836f-7b6bacbbbcd0
# ╟─35e4b98b-3570-4d28-9bb5-268a9d31d9c5
# ╟─e3ddd42d-2170-48e0-9ea8-4edc2f64ae9e
# ╟─43675330-3f08-4d7d-afef-838678433e03
# ╟─44702524-6f20-460f-aab6-c3ecdaf4aa23
# ╟─9058e842-50e6-416b-baa2-12de099a93ed
# ╟─4009aa7f-0444-4244-8f1e-4265e80b7a51
# ╟─8c0bfd17-6fae-40b6-a8cc-6f1cf53ce022
# ╟─bc373277-9d24-4f85-83b4-96e7f34cfd69
# ╟─afdfddc7-cb15-4d65-b608-b557981a1506
# ╟─155f2f18-6e36-4619-ab7d-26595c2c7cb1
# ╟─2d06a027-8e3c-4f0f-b186-166c1a196806
# ╟─2945d608-9df2-4db3-8848-ce658dc86a3b
# ╟─fd39b942-5066-4ae9-803d-112001a10e0b
# ╟─9477671c-b275-4a8c-b77d-dc5f986c7148
# ╟─ce314a72-0af2-4b57-97c9-4386bcfdc89b
# ╟─5b4c7b7d-7312-4a85-8c12-d5e3cffa0400
# ╟─0e56879c-b494-4ab7-b65f-92ca1fb98b2f
# ╟─cd12123a-2519-4d2f-89cd-7d93517724f0
# ╟─5386acfe-6406-42a9-bcae-03981005449a
# ╟─79149c07-a8f0-4bce-b58f-8e8569d6f212
# ╟─8ff3c42c-2aae-4a1a-ab99-e09b64f59054
# ╟─fa314d87-4b29-40cc-9082-a1845fa013fc
# ╟─928fe376-19f4-460f-8d90-f89604741c34
# ╟─1980e2dd-da14-4b3b-ab2b-8135c83b8e35
# ╟─d1c0f2b1-4ef0-40a9-b1ae-284f5f4b0af4
# ╟─e4462cca-ba52-41ea-b60f-e1671aa407d5
# ╟─620bbc2a-849c-4684-8a34-7279b018274a
# ╟─278a11a5-83de-46d8-8552-ab611424b2eb
# ╟─249fc727-a379-452c-af06-9980904db1ec
# ╟─de2c730a-4069-4219-b2b0-86674bc5ca17
# ╟─8ce12db7-6f1d-4223-ae61-98170a477a35
# ╟─ff05845e-6a87-4601-9b22-2a02ff7430d3
# ╟─c30f667e-e949-4530-ad14-3d620047e027
# ╟─0a1fd159-fa08-44ab-81f7-0abf38cba097
# ╠═a7ea1f45-e511-4feb-abd6-bd0b47e89331
# ╟─883f90fe-6dc5-4312-938d-8076b7204ed3
# ╠═02bbf5a2-2048-4aaf-8fd6-245943234fe7
# ╟─d4cb02d3-fe02-40b7-8e79-857c2c71f7d9
# ╠═93dc79cb-3fe2-4603-a57e-782ac6e66abd
# ╠═ac889070-c342-4258-ae72-e298d060c90e
# ╟─bbdcd4fb-d5bc-4599-9e01-8739eea70a28
# ╟─f094c18f-fac3-4aa3-b986-fed742ece212
# ╠═c873b617-731b-4674-be7d-966958e79676
# ╠═b30971a7-eef9-4405-b0d4-91aa3299186b
# ╠═18dc5ddd-f185-4f81-975a-3ca3a095a1d4
# ╠═fa02ead1-6b3c-437b-b8c2-b79318c0ad5c
# ╠═b05411c4-1dc5-482e-ac20-110341662f7b
# ╟─0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
# ╟─8687dbd1-4857-40e4-b9cb-af469b8563e2
# ╟─fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
