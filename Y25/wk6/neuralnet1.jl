### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ ab7b3ac3-953d-41fb-af70-7e92ab06b8c7
begin
	using Zygote
	# using Flux
	# import Flux: logitcrossentropy, normalise, onecold, onehotbatch
	using DataFrames
	using Random
	using StatsPlots
	using Latexify, LaTeXStrings
	using Statistics: mean
	using PlutoTeachingTools
	using PlutoUI
	using LinearAlgebra
	using LogExpFunctions
	# using StatsBase
	using Distributions
	using HypertextLiteral
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style	
end

# ╔═╡ 08f341b4-1e45-4c1d-990d-e2671ebc70b2
begin 
	using Flux
	import Flux: logitcrossentropy, normalise, onecold, onehotbatch
end

# ╔═╡ 28754109-9d20-4c20-a2b1-491dd56dcc2f
using Images

# ╔═╡ f72cacd8-f5c2-43a7-b737-71c6b5185880
begin
	using Logging
	Logging.disable_logging(Logging.Info); # or e.g. Logging.Info
end;

# ╔═╡ 58d7047d-0173-4297-891b-74dd92e13afb
using MLJBase

# ╔═╡ 50103dcc-034e-4c22-953c-1b5fd781c070
TableOfContents()

# ╔═╡ fdab762e-4d32-43f9-8a0d-267fec8e491e
ChooseDisplayMode()

# ╔═╡ 78aad856-dc62-430d-ad3a-a32b0d268b0d
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ caa7b56c-6a21-413d-816e-00bb0d16fdcd
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

# ╔═╡ a9c35a78-c179-45f4-ba25-18e744b30129
md"""

# CS5014 Machine Learning


#### Neural Network

\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 1e180936-7d39-4895-b526-9a09d7024946
# md"""

# ## This time

# An introduction to Neural Networks (NN)
# * intuitively: stack multiple single neurons together 


# * some commonly used NN constructs and their implementation
#   * Linear layer 
#   * activation functions


# * and their implementations from scratch


# * regularisation techniques of neural networks
# \

# ## Next time

# Other topics to discuss (Next topic)


# * learning: backpropagation (next time)
# * some other tricks and practices: vanishing gradient, batch normalisation
# * advanced gradient descent algorithms
# """

# ╔═╡ 16a460dd-a1bb-4304-bd80-cdeaa82c1cac
md"""

## Recap: logistic regression
#### also known as a single-neuron 

```math
\large
\textcolor{darkorange}{\sigma(\mathbf{x})} = \textcolor{darkorange}{\texttt{logistic}}\left (\underbrace{\color{darkblue}{\begin{bmatrix} w_1&  w_2& \ldots&   w_m \end{bmatrix}}  \color{darkgreen}{\begin{bmatrix} x_1\\  x_2\\ \vdots\\   x_m \end{bmatrix}} + \textcolor{darkblue}{b} \color{black}}_{z} \right )
```
* ``\color{Periwinkle}z = {\color{darkblue}\mathbf{w}}^\top{\color{darkgreen}\mathbf{x}} +{\color{darkblue}b}`` : pre-activation (a hyperplane)
* ``{\texttt{logistic}}(z) =\frac{1}{1+e^{-z}}``
"""

# ╔═╡ 70d6cadd-04e4-4bb5-b38e-8d510a3f2bcf
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/single_logreg.png
' width = '500' /></center>"

# ╔═╡ f413ae69-9df1-4447-84b0-8f35b39f55cd
md"""
## Recap: logistic regression


"""

# ╔═╡ 5262a879-a236-44ee-8d00-ef8547de579b
wv_ = [1, 1] * 1

# ╔═╡ 78d6618f-cd68-434b-93e4-afd36c9e3247
# md"""

# ## Recap: single neuron (linear regression)



# A **linear regression** can be viewed as a _special case_, where

# ```math
# \large
# \sigma(\mathbf{x}) = \texttt{identity}(\mathbf{w}^\top\mathbf{x})
# ```
# * activation function is a boring _identity_ function: $\texttt{identity}(z) =z$


# """

# ╔═╡ 0d0b78f8-8b53-4f11-94d9-d436c84ae976
# md"""

# ```math
# \textcolor{darkorange}{\sigma(\mathbf{x})} = \texttt{I}\left (\color{darkblue}{\begin{bmatrix} w_1&  w_2& \ldots&   w_m \end{bmatrix}}  \color{darkgreen}{\begin{bmatrix} x_1\\  x_2\\ \vdots\\   x_m \end{bmatrix}} + {\color{darkblue} b} \color{black} \right );
# ```

# """

# ╔═╡ a7b797a8-d791-4d10-9ac2-7bd9c0815688
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/single_linreg.png
# ' width = '400' /></center>"

# ╔═╡ bbb08fad-6d96-4735-9ac5-a03f27773af1
# md"""

# ## Recap: binary classification cross entropy loss


# Binary classification usually use binary cross entropy (BCE) loss:


# ```math
# \large
# \begin{align}
#  \text{BCE\_loss}(\hat{\mathbf{y}}, \mathbf{y}) = - \sum_{i=1}^n {y^{(i)}} \ln \sigma^{(i)}+ (1- y^{(i)}) \ln (1-\sigma^{(i)})
# \end{align}
# ```
# """

# ╔═╡ 7159fa02-e801-4a86-8653-aae1dda2e7ac
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/logistic_reg_loss.png
# ' width = '600' /></center>"

# ╔═╡ f697e1bb-4e8b-458c-8ea1-d85c403fd913
# md"""

# ## Recap: binary classification learning


# """

# ╔═╡ 0b945b04-2808-4e81-9c2f-18ef5c891366
# md"""

# Based on the BCE loss, we have derived the gradients for ``\{\mathbf{w}, b\}`` 


# ```math
# \large
# \begin{align}
# \nabla_{{\mathbf{w}}}L(y, \hat{y}) &= - (y- \hat{y})\cdot \mathbf{x} \\
# \nabla_{b}L(y, \hat{y}) &= - (y- \hat{y})\cdot 1

# \end{align}
# ```

# * where $\hat{y} = \sigma(\mathbf{w}^\top\mathbf{x}+b)$ is the prediction output


# ##### Show me the code

# _Gradient descent_

# ```python
# while True:
# 	gradw, gradb = evaluate_gradient(train_x, train_y, w, b)
# 	w -= lr * gradw
# 	b -= lr * gradb
# ```
# """

# ╔═╡ f510ba7f-8ca5-42ed-807d-67a8710d2e00
# aside(tip(md"""
# If we use the dummy predictor notation, the gradient simplified to


# ```math
# \nabla_{\tilde{\mathbf{w}}}L(y, \hat{y}) = - (y- \hat{y})\cdot \mathbf{x},
# ```

# where ``\mathbf{x} = [1, \mathbf{x}]^\top`` and ``\tilde{\mathbf{w}} \triangleq [b, \mathbf{w}]^\top``

# """))

# ╔═╡ 45788db8-4d37-4b7a-aadb-6c6993bf37e8
md"""

## Recap: logistic regression learning demonstration


"""

# ╔═╡ f000ec11-e922-4262-b985-3882b2e6c2ee
md"""

## However 



#### Real world data sets are rarely _linear_

* ###### a single neuron model is rarely enough
"""

# ╔═╡ 2945e298-1983-4e6b-a903-bead1ac833a8
begin
	#Auxiliary functions for generating our data
	function generate_real_data(n)
	    x1 = rand(1,n) .- 0.5
	    x2 = (x1 .* x1)*3 .+ randn(1,n)*0.1
	    return vcat(x1,x2)
	end
	function generate_fake_data(n)
	    θ  = 2*π*rand(1,n)
	    r  = rand(1,n)/3
	    x1 = @. r*cos(θ)
	    x2 = @. r*sin(θ)+0.5
	    return vcat(x1,x2)
	end
	gr()
	# Creating our data
	Random.seed!(23456)
	train_size = 1000
	real = generate_real_data(train_size)
	fake = generate_fake_data(train_size)
	test_size = 500
	real_test = generate_real_data(test_size)
	fake_test = generate_fake_data(test_size)
	# Visualizing
	plt_smile = scatter(real[1,1:500],real[2,1:500], label="class 1", title="A non-linear classification dataset")
	scatter!(fake[1,1:500],fake[2,1:500], ratio=1, label="class 2", framestyle=:origin)
end

# ╔═╡ c95275ed-b682-49b8-ae54-9e9e84d401e6
# md"""
# ##

# ##### Non-linear regression 
# \

# A non-linear regression example
# """

# ╔═╡ 1d968144-0bf7-4e08-9600-6fc33cdbdb52
# begin
# 	gr()
# 	scatter(x_input, y_output, label="Observations", title="A non-linear regression dataset")	
# 	plot!(true_f, lw=2, xlim=[-10, 10], framestyle=:default,  lc=:gray, ls=:dash,label="true function", xlabel=L"x", ylabel=L"y")
# end

# ╔═╡ 7dcff9b2-763b-4424-97d7-48da1e3d2aff
begin
	function true_f(x)
		-5*tanh(0.5*x) * (1- tanh(0.5*x)^2)
	end
	Random.seed!(100)
	x_input = collect(range(-8, 8, 50))
	
	y_output = true_f.(x_input) .+ sqrt(0.05) * randn(length(x_input))
	# xμ, xσ = mean(x_input), sqrt(var(x_input))
	# x_input = (x_input .- xμ) / xσ
end;

# ╔═╡ 6d84e048-d215-4597-b1f9-dbc24e031791
# md"""

# ##

# ##### Non-linear classification

# \

# A non-linear decision boundary example

# """

# ╔═╡ 95ed0e5c-1d01-49d1-be25-a751f125eb76
# TwoColumn(md"""
# \



# * #### Not separable by _one linear_  boundary
# \


# * #### In other words, a single neuron is clearly not enough


# """, let

# 	# θs = range(π/4, π/4 + 2π, 20)
# 	# r = 5.0
# 	# ws = r * [cos.(θs) sin.(θs)]

# 	# anim = @animate for w in eachrow(ws)
# 	@df df_class scatter(:x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", ylabel=L"x_2", title="Non-linear separable data with single neuron", framestyle=:origin, ratio=1, size=(350,350), titlefontsize=8)
# 	# plot!(-7:0.1:7, -7:0.1:7, (x,y)-> logistic(dot([x, y], w)), c=:jet, st=:contour, alpha=0.5, colorbar=false)
# 	# end
# 	# gif(anim, fps = 1)
# end)

# ╔═╡ 1fb81ca3-2fa9-4862-bad1-00446515a23a
md"""

## Another non-linear example


"""

# ╔═╡ fe45432f-9224-4e1e-a981-0136db240085
md"""

## How about two neurons?


#### Each pair of datasets can be handled by one neuron


#### How about stack two neurons together?
"""

# ╔═╡ 6886fe8d-9645-4094-85f8-6b3fb5d409e4
md"""

## How about two neurons? (conti.)

#### ``z_1, z_2`` are two neuron's pre-activated outputs
\

```math 
\Large
\begin{align}
z_1 = \mathbf{w}_1^\top\mathbf{x} +b_1\\ 
z_2 = \mathbf{w}_2^\top \mathbf{x} + b_2
\end{align}
``` 

"""

# ╔═╡ f52e374a-2031-4bd9-b58c-16598ffac15a
TwoColumn(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-01.png", :height=>300), Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-02.png", :height=>300))

# ╔═╡ 4b2fd42a-e88d-4117-af79-84cfaf902122
md"""

## Matrix notation
"""

# ╔═╡ 5bdd174d-9250-4864-80af-1360240ccc93

md"""

#### ``z_1, z_2`` are two neuron's outputs

```math 
\large
\begin{align}
z_1 = \mathbf{w}_1^\top\mathbf{x} +b_1\\ 
z_2 = \mathbf{w}_2^\top \mathbf{x} + b_2
\end{align}
``` 

#### Written in matrix notation


```math
\Large
\underbrace{\begin{bmatrix}
z_1 \\
z_2
\end{bmatrix}}_{\mathbf{z}} = \underbrace{\begin{bmatrix} \rule[.5ex]{2.5ex}{0.5pt} &\mathbf{w}_1^\top & \rule[.5ex]{2.5ex}{0.5pt} \\

\rule[.5ex]{2.5ex}{0.5pt} &\mathbf{w}_2^\top & \rule[.5ex]{2.5ex}{0.5pt}
\end{bmatrix}}_{\mathbf{W}} \underbrace{\begin{bmatrix} x_1 \\

x_2
\end{bmatrix}}_{\mathbf{x}} + \underbrace{\begin{bmatrix} b_1 \\
b_2
\end{bmatrix} }_{\mathbf{b}}
```

"""

# ╔═╡ 36a00007-0da1-48a4-a225-d400d9c61a37
md"""

## Matrix notation
"""

# ╔═╡ ca75fcaf-9b6d-4d1d-a97e-1b0167b1e2d8

md"""
#### In matrix notation


```math
\large
\underbrace{\begin{bmatrix}
z_1 \\
z_2
\end{bmatrix}}_{\mathbf{z}} = \underbrace{\begin{bmatrix} \rule[.5ex]{2.5ex}{0.5pt} &\mathbf{w}_1^\top & \rule[.5ex]{2.5ex}{0.5pt} \\

\rule[.5ex]{2.5ex}{0.5pt} &\mathbf{w}_2^\top & \rule[.5ex]{2.5ex}{0.5pt}
\end{bmatrix}}_{\mathbf{W}} \underbrace{\begin{bmatrix} x_1 \\

x_2
\end{bmatrix}}_{\mathbf{x}} + \underbrace{\begin{bmatrix} b_1 \\
b_2
\end{bmatrix} }_{\mathbf{b}}
```

#### or simply

```math
\LARGE
\texttt{Linear}(\mathbf{x}, \mathbf{W}, \mathbf{b}) =\boxed{\mathbf{z} = \mathbf{W} \mathbf{x} +\mathbf{b}}
```

* ##### the function _Linear Layer_
* where note that

```math
\large
\mathbf{W} = \begin{bmatrix} w_{11} & w_{12} \\
w_{21} & w_{22}
\end{bmatrix} =\begin{bmatrix} \rule[.5ex]{2.5ex}{0.5pt} &\mathbf{w}_1^\top & \rule[.5ex]{2.5ex}{0.5pt} \\

\rule[.5ex]{2.5ex}{0.5pt} &\mathbf{w}_2^\top & \rule[.5ex]{2.5ex}{0.5pt}
\end{bmatrix}
```
* ##### Two neurons: therefore two rows in ``\mathbf{W}``

"""

# ╔═╡ f6e77a11-8e1a-44e2-a2d7-cd29939dcc30
md"""

## Linear layer (generalisation)


##### ``n`` neurons Linear Layer 

* ##### input: ``\mathbf{x} \in \mathbb{R}^m``
* ##### output: ``\mathbf{z} =[z_1, z_2, \ldots, z_n] \in \mathbb{R}^n``
"""

# ╔═╡ 001a7866-f0c4-4cea-bda3-d5fea2ce56b4
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/linear1.svg
' height = '400' /></center>"

# ╔═╡ 2271ac6d-4bb7-4683-8672-c280c3e57fd3
md"""

## Linear layer (generalisation)


##### ``n`` neurons Linear Layer 

* ##### input: ``\mathbf{x} \in \mathbb{R}^m``
* ##### output: ``\mathbf{z} =[z_1, z_2, \ldots, z_n] \in \mathbb{R}^n``
"""

# ╔═╡ 350146ba-6e11-41f8-b451-b3c842722de9
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/linear2.svg
' height = '400' /></center>"

# ╔═╡ fcb991b7-e3e1-4f82-8a8b-c1f2a99d6cce
md"""

## Linear layer (generalisation)


##### ``n`` neurons Linear Layer 

* ##### input: ``\mathbf{x} \in \mathbb{R}^m``
* ##### output: ``\mathbf{z} =[z_1, z_2, \ldots, z_n] \in \mathbb{R}^n``
"""

# ╔═╡ 8971420b-7687-4a89-8cfa-da2483be8730
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/linear3.svg
' height = '400' /></center>"

# ╔═╡ a7051da0-f28a-4261-b71c-f7450341e1bf
# md"""

# ## Linear Layer implementation


# Just a many to many function


# ```math
# \LARGE
# \texttt{Linear}(\mathbf{x}; \mathbf{W}, \mathbf{b}) = \mathbf{W} \mathbf{x} +\mathbf{b}
# ```


# * ``\mathbf{W}``: size ``n \times m``
#   * ``n``: # of hidden neurons
#   * ``m``: # of input (or input dimension)


# * ``\mathbf{b}``: size ``n\times 1``
#   * bias vector: one bias for each of the ``n`` neurons

# """

# ╔═╡ 500d6f5f-30ee-45a0-9e3c-5f8ba5274e75
begin
	# That's it!
	function denselayer(X, W, b; σ = identity)
		h = W * X .+ b
		σ.(h)
	end
end;

# ╔═╡ 2ec97a4c-0ddc-4918-91a1-0bf177b4a202
md"""

## Linear Layer implementation 


#### In maths, Linear layer is 


```math
\LARGE
\texttt{Linear}(\mathbf{x}; \mathbf{W}, \mathbf{b}) = \mathbf{W} \mathbf{x} +\mathbf{b}
```


* ``\mathbf{W}``: size ``n \times m``
  * ``n``: # of output hidden neurons
  * ``m``: # of input (or input dimension)


* ``\mathbf{b}``: size ``n\times 1``
  * bias vector: one bias for each of the ``n`` neurons


## Linear Layer implementation in Python*


#### *Python* (PyTorch, numpy) is a *row-major* programming language

* ##### a vector ``\texttt{x}`` is assumed a row vector by default


#### Therefore, when it comes to implementation in Python

```math
\LARGE
\texttt{z}_{1\times n} = \texttt{x}_{1\times m} @ \texttt{W}_{m\times n}  +\texttt{b}_{1\times n}
```


* ##### that is ``\texttt{z} = \mathbf{z}^\top``, ``\texttt{W} = \mathbf{W}^\top``, ``\texttt{b} =\mathbf{b}^\top``
"""

# ╔═╡ b7d84cb8-59ee-41e6-a04e-a4be9f905f95
md"""

#### In particular, $\texttt{W}_{\text{in} \times \text{out}} \in \mathbb{R}^{\text{in} \times \text{out}}$

$$\texttt{W}= \begin{bmatrix} \mid &\mid &  \mid &\mid \\
\mathbf{w}_1 &\mathbf{w}_2 & \ldots & \mathbf{w}_{\text{out}} \\
\mid &\mid &  \mid &\mid 
\end{bmatrix}$$
"""

# ╔═╡ d6023ad2-2467-44a4-8c57-d12024e76536
aside(tip(md"""
```math
\begin{align}
\mathbf{z} &=  \mathbf{Wx} + \mathbf{b}\\
\mathbf{z}^\top &= \mathbf{x}^\top \mathbf{W}^\top + \mathbf{b}^\top
\end{align}
```
"""))

# ╔═╡ 82a2a13c-cdbe-4901-8b2e-86a5d3cd99e5
# md"""

# !!! note "Linear Layer" 
# 	```python
# 	class Linear:  
# 	    def __init__(self, in_size, out_size):
# 	        self.weight = torch.randn((in_size, out_size), generator=g) 
# 	        self.bias = torch.zeros(out_size) 
	  
# 	    def __call__(self, x):
# 	        self.out = x @ self.weight + self.bias
# 	        return self.out

# 	    ...
# 	```

# """

# ╔═╡ afce64cb-cdca-42a1-8e92-f1e9634af976
# md"""


# ```python
# l1 = Linear(2, 2) # create a 2 to 2 linear layer
# z = l1(torch.rand(2))
# ```
# """

# ╔═╡ 7841a726-072a-41a3-a10e-4a4a5256aef1
md"""
## Linear Layer implementation in Julia*

#### `Julia` is a modern language designed for numerical computing

* the implementation is just direct translation from maths
"""

# ╔═╡ ddb1bd96-e316-450a-a7b5-f5561a890266
begin
	struct Linear
		W
		b
	end

	# constructor with input and output size and identity act
	Linear(in::Integer, out::Integer) = Linear(randn(out, in), randn(out))
	
	# Overload function, so the object can be used as a function
	(m::Linear)(x) = m.W * x .+ m.b
end;

# ╔═╡ 0fe62e04-559d-4140-970b-6baca845be63
Linear(2, 2)(rand(2));

# ╔═╡ 8d009757-ab79-47e5-8be0-080ef091b302
begin
	# That's it!
	function linearlayer(X, W, b)
		z = W * X .+ b
	end
end;

# ╔═╡ f079377b-d71f-4f70-aece-f04d12cd3127
md"""

## Activation functions

#### We apply non-linear _activation_ functions *element-wisedly* to ``\mathbf{z}``


"""

# ╔═╡ fe63494b-716c-4718-afcd-026291dc7b94
TwoColumn(md"""
\
\
\

```math 
\large
\begin{align}
a(z_1) = \sigma(\underbrace{\mathbf{w}_1^\top\mathbf{x} +b_1}_{z_1})\\ 
a(z_2) = \sigma(\underbrace{\mathbf{w}_2^\top \mathbf{x} + b_2}_{z_2})
\end{align}
``` 

""", 

Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/linear5_act.svg", :height=>300)
)

# ╔═╡ e6d3f1d1-21cb-4662-8806-77b88038be7c
md"""

#### Which can be written as 


```math
\large
\mathbf{a} = \sigma.(\mathbf{Wx}+\mathbf{b})
```

* ``\sigma\,`` +``\,`` dot : element-wise operation, apply the function to each element
* some use ``\odot`` to denote element-wise operation, *e.g.* ``\mathbf{a} = \sigma \odot (\mathbf{Wx}+\mathbf{b})``
"""

# ╔═╡ cb1bc71b-8c8b-4873-8f28-2935c0972b8c
md"""

## Activations

##### It is also common to draw the activation and linear layer together 
"""

# ╔═╡ 4a0195bf-2d5c-478b-b571-edd4b3e532eb
TwoColumn(html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/linear5_act.svg
' height = '300' /></center>" , html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-03.png
' height = '255' /></center>")

# ╔═╡ 13786181-9280-4260-9705-f2aa419e9c26
# md"""
# ## Linear layer + activations


# Now consider ``m``-dimensional input and ``n`` neurons

# * ``m`` input/predictors
# * ``n`` output hidden neurons


# ##### We use _super-index_ ``^{(l)}`` to index the layer of a NN
# """

# ╔═╡ fec26acd-d545-428d-a90d-17a89b2e11c9
# ThreeColumn(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-08.png", :height=>350), Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-09.png", :height=>350), Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-10.png", :height=>350))

# ╔═╡ 6d443b0c-becc-459a-b4ee-9a863c7d69b3
md"""

## Linear Layer + activation


#### **_Linear_** Layer + element-wise **activation** forms a ``\mathbb{R}^m \rightarrow \mathbb{R}^n`` function

```math
\Large
 \mathbf{x} \mapsto \sigma.(\texttt{Linear}(\mathbf{x}; \mathbf{W}, \mathbf{b}))
```

* ##### which is a layer of neural net


"""

# ╔═╡ be7087be-f948-4701-b5c2-00580ca27d9b
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-11.png
' height = '400' /></center>"

# ╔═╡ c98638f4-5bb1-4550-aa51-945c4632b801
md"""

## Activation functions


#### Some common activation functions are



"""

# ╔═╡ eef80655-c86f-4947-be7e-ec5b7ff13ad2
ThreeColumn(
md"""


**Identity**: 

```math
\small
\sigma(z) = z
```

**Logistic**: 

```math
\small

\sigma(z) = \frac{1}{1+e^{-z}}
```

**Tanh**: 

```math
\small

\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z}+ e^{-z}}
```



""",


md"""


**ReLu** (Rectified linear unit): 

```math
\small
\begin{align}
\text{ReLu}(z) &= \begin{cases}0 & z < 0\\ z & z\geq 0 \end{cases}\\
&=\max(z, 0)

\end{align}
```


**Leaky ReLu**: 

```math
\scriptsize
\text{LeakyReLu}(z) = \begin{cases}0.01 z & z < 0\\ z & z\geq 0 \end{cases}
```


**softplus**: 

```math
\small

\text{softplus}(z) = \ln(1+e^z)
```

"""
	
	
	
	,
	
md"""


**Gelu** (Gaussian Error Linear Unit): 

```math
\scriptsize
\begin{align}
\text{Gelu}(z) &= 0.5x *\\
&\;\;(1 + \tanh(\sqrt{\tfrac{2}{\pi}} (x + 0.044715x^3)))
\end{align}
```


**Swish**: 

```math
\small

\text{swish}(z) =  \frac{z}{1+e^{-z}}
```


**CeLu**: 

```math
\small

\text{Celu}(z, \alpha) =  \begin{cases}z & z \geq 0\\ \alpha\left(e^{z/\alpha} -1\right) & \text{otherwise}\end{cases}
```

"""

	
)

# ╔═╡ 8aa1f78a-9cf8-4186-b9a0-31b8a00eabfc
let
	gr()
	p_relu = plot(relu, lw=1.5, lc=1, label=L"\texttt{ReLu}(z)",legend=:topleft)
	p_logis = plot(logistic, lw=1.5, lc=3,label=L"\texttt{logistic}(z)", legend=:topleft)
	plot!(tanh, lw=1.5, lc=2, label=L"\tanh(z)",legend=:topleft)
	# plot!(p_relu, leakyrelu, lw=1.5, lc=4, label=L"\texttt{LReLu}(z)", legend=:topleft)
	plot!(p_relu, Flux.softplus, lw=1.5, lc=4, label=L"\texttt{softplus}(z)", legend=:topleft)
	p_gelu = plot(gelu, lw=1.5, lc=6, label=L"\texttt{gelu}(z)", legend=:topleft)
	plot!(swish, lc=7, label=L"\texttt{swish}(z)", lw=2)

	p_lrelu = plot(leakyrelu, lw=1.5, lc=6,label=L"\texttt{LeakyReLu}(z)", legend=:topleft)

	plot!(celu, lc=7, lw=1.5,label=L"\texttt{celu}(z)")
	plot(p_logis, p_relu, p_gelu, p_lrelu)
end

# ╔═╡ 81c2fa7b-d3b3-4fe3-b2e6-831a10616ed0
# md"
# A more user-friendly implementation

# * use `struct`
# * a layer with ``n`` neurons can be constructed specifically
# * the function map is overloaded
# "

# ╔═╡ 361d76ed-7952-4c64-830c-27c6273272d7
# md"""

# ## Implementations -- Python
# """

# ╔═╡ 7001dfa7-fb39-40fc-acac-750c7f0d6605
# md"""

# As an example, ReLu can be implemented as

# ```python
# class ReLu:
#     def __call__(self, x):
#         self.out = torch.clip(x, 0) ## clip will return the max between x and 0
#         return self.out
# ```

# or simply use the built-in method

# ```python
# class ReLu:
#     def __call__(self, x):
#         self.out = torch.relu(x) ## clip will return the max between x and 0
#         return self.out
# ```
# """

# ╔═╡ 743030fd-4462-4563-8767-2d4bfbb17f5b
# md"""

# ```python
# l1 = Linear(2, 5) # a linear layer with 2 input and 5 neurons output
# relu = ReLu() # instantiate a ReLu() activation function
# l1relux = relu((l1(x))) # element wise activations
# ```
# """

# ╔═╡ f3ca219a-e2af-45f7-85bf-3c050227a33b
md"""

## Implementation -- Julia*


"""

# ╔═╡ 19142836-f9c0-45c1-a05b-24d810a06f8f
begin
	struct DenseLayer
		W
		b
		act::Function
	end

	# constructor with input and output size and identity act
	DenseLayer(in::Integer, out::Integer) = DenseLayer(randn(out, in), randn(out), identity)

	# constructor with non-default act function
	DenseLayer(in::Integer, out::Integer, a::Function) = DenseLayer(randn(out, in), randn(out), a)
	
	# Overload function, so the object can be used as a function
	(m::DenseLayer)(x) = m.act.(m.W * x .+ m.b)
end;

# ╔═╡ 378c520d-881c-4056-a3d0-03fcddf2126c
# md"""

# ## Why (non-linear) activations ?



# """

# ╔═╡ dbddfc50-5d66-45c9-b61a-a554282e38c7
# TwoColumn(md"""
# #### _Non-linear_ activations are essential

# * ##### it makes the final function **non-linear**


# #### **Without** _non-linear_ *activations*, a neural network becomes another linear model

# \

# * To see this, we set ``\sigma = \texttt{idensity}``, *i.e.* no non-linear activations, then

# ```math
# \mathbf{a} = \mathbf{Wx}+\mathbf{b} 
# ```



# """, html"<br><br><br><br><br><center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp_reg.svg
# ' width = '345' /></center>")

# ╔═╡ 0f6a50ea-6165-4e9b-b332-641fed50eea9
# md"""

# ## Why (non-linear) activations ?


# """

# ╔═╡ bad0d1a7-f93a-4af8-a453-4ea8e1385d7c
# TwoColumn(md"""
# ###### _Non-linear_ activations are essential for neural networks

# * it makes the final function **non-linear**


# ###### **Without** _non-linear_ *activations*, a neural network becomes another linear model

# \

# To see this, we set ``\sigma = \texttt{idensity}``, then

# ```math
# \mathbf{a} = \mathbf{Wx}+\mathbf{b} 
# ```

# and

# ```math
# \begin{align}
# z &= \mathbf{v}^\top (\underbrace{ \mathbf{Wx}+\mathbf{b}}_{\mathbf{a}})+ b_0 \\
# \end{align}
# ```

# """, html"<br><br><br><br><br><center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp_reg.svg
# ' width = '345' /></center>")

# ╔═╡ c6de70e6-14f4-4b1a-8097-4e086a7fedde
# md"""

# ## Why (non-linear) activations ?


# """

# ╔═╡ bb317788-8d04-45f8-84f7-7a0226a7df6f
# TwoColumn(md"""
# ###### _Non-linear_ activations are essential for neural networks

# * it makes the final function **non-linear**


# ###### **Without** _non-linear_ *activations*, a neural network becomes another linear model

# \

# To see this, we set ``\sigma = \texttt{idensity}``, then

# ```math
# \mathbf{a} = \mathbf{Wx}+\mathbf{b} 
# ```

# and

# ```math
# \begin{align}
# z &= \mathbf{v}^\top (\underbrace{ \mathbf{Wx}+\mathbf{b}}_{\mathbf{a}})+ b_0 \\
# &= \underbrace{\mathbf{v}^\top \mathbf{W}}_{\boldsymbol{\omega^\top}}\,\mathbf{x} + \underbrace{\mathbf{v}^\top \mathbf{b} +b_0}_{\omega_0} \\
# &= \boldsymbol{\omega}^\top\mathbf{x}+ \omega_0 \;\; \text{\# a linear function}
# \end{align}
# ```

# """, html"<br><br><br><br><br><center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp_reg.svg
# ' width = '345' /></center>")

# ╔═╡ a8f32a69-3150-4639-9cd4-546ab8442d90
md"""

## Why we need non-linear activation?

##### A showllow neural network with three hidden units
"""

# ╔═╡ 6daa681e-3974-497c-baec-347740b2a29e
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ShallowNet.svg
' height = '350' /></center>"

# ╔═╡ 3db05083-dc9e-49d9-9d11-68562eac5827
md"""

##### The pre-activation functions
"""

# ╔═╡ 8674f942-dfae-4891-b3e1-a38b79a2b621
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ShallowBuildUp0.svg
' width = '750' /></center>"

# ╔═╡ 1be45d72-0661-4a3a-875f-888d0c12b5e4
md"""
##
##### The after-activation hidden layers (ReLu)
"""

# ╔═╡ d6353c42-b07b-483a-a669-857c480385b9
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ShallowBuildUp1.svg
' width = '750' /></center>"

# ╔═╡ 0e929a8d-2c43-4e51-b990-8855963b4b4d
md"""
##
### The final output

* ##### Without the non-linear activations, the final function will be a linear combinations of linear functions, which is still linear
"""

# ╔═╡ 0fe8412f-4dc8-4698-b434-81763038e768
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ShallowBuildUp1.svg
' width = '750' /></center>"

# ╔═╡ 6a4faffd-39e9-4576-a820-3f3abb724055
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ShallowBuildUp2.svg
' width = '750' /></center>"

# ╔═╡ a9427f86-20d2-43ca-b1ca-9724252315da
md"""

[Source: Understanding Deep Learning](https://udlbook.github.io/udlbook/)
"""

# ╔═╡ afe83e7a-5e5f-4fe4-aa82-082c7fcdfed3
md"""

## Universal approximator



#### Neural network is *universal function approximator*
- ##### with *infinite* number of neurons, it can approximate any function well




"""

# ╔═╡ 13e0cd61-2f3f-40be-ad4a-7a2fb13661bf
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ShallowApproximate.svg
' width = '750' /></center>"

# ╔═╡ a5efe47c-71fb-4b1e-8417-47036b96238d
md"""
## Finalising: output layer (regression)

#### For **regression**, we just output the final output as it is


#### And squared error loss

$$\Large\text{loss}(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$$


"""

# ╔═╡ bd6ccedb-00a7-4f9d-8ec9-861d8a8ada11
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp_reg.svg
' height = '345' /></center>"

# ╔═╡ 939c8687-3620-4e98-b9f5-0c304312ef2f
md"""

## Finalising: output layer (classification)

#### For binary *classification*, the output is squeezed by `logistic`

* ##### the same idea as logistic regression: apply logistic transformation


#### loss function: binary cross entropy

"""

# ╔═╡ a617405e-fc19-4032-8642-7f01fb486c74
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-05.png
' height = '350' /></center>"

# ╔═╡ 5af83948-c7dd-46b5-a920-4bfb514e4f9c
# md"""

# ## Multi-class classification: softmax function


# To idea is similar to softmax regression, we apply softmax function to the final output

# """

# ╔═╡ 8243b153-0d03-4efa-9fcc-98ab42008826
# aside(tip(md"""

# It is a *soft* version of a *hardmax* function:

# ```math
# \begin{bmatrix}
# 1.3\\
# 5.1\\
# 2.2\\
# 0.7\\
# 1.1
# \end{bmatrix} \Rightarrow \texttt{hardmax} \Rightarrow \begin{bmatrix}
# 0\\
# 1\\
# 0\\
# 0\\
# 0
# \end{bmatrix},
# ```

# also known as winner-take-all. 

# Mathematically, each element is: ``I(\mathbf{a}_i = \texttt{max}(\mathbf{a}))``

# """))

# ╔═╡ 4514df4f-0fee-4e30-8f43-d68a73a56105
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/softmax.jpeg
# ' width = '500' /></center>"

# ╔═╡ 271874ee-3db9-4abd-8fb4-96f4266cec25
# md"figure source: [^1]"

# ╔═╡ 65efc486-8797-442f-a66e-32da5b860635
# function soft_max_naive(x)
# 	ex = exp.(x)
# 	ex ./ sum(ex)
# end

# ╔═╡ b517aaba-e653-491c-8dc2-af86a300b62e
md"""

## Finalsing: add output layer:  multiclass classification



#### For multi-class **classification**, we apply `softmax` to the logits $\mathbf{z}$

$$\Large\text{softmax}_k(\mathbf{z}) = \frac{e^{z_k}}{\sum_j e^{z_j}}$$



#### loss: cross entropy

"""

# ╔═╡ 9d0e9137-f53b-4329-814e-20e842033f41
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnmulticlass.png
' width = '600' /></center>"

# ╔═╡ 1287ed6d-8f1e-40f3-8d46-00ab4b267681
  
md"""figure source [^2]"""

# ╔═╡ 9a97cc7a-3b3f-41bf-a41a-80dde51a01cc
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/fixedbasisreg_.svg
# ' height = '450' /></center>"

# ╔═╡ a594cfb0-bd4d-427e-ba00-c665a2c41f54
md"""

## Fixed basis regression vs Neural network



"""

# ╔═╡ 9907c198-8f88-46cf-804d-ba6bde31770b
md"""

#### Fixed basis regression is a specific case  of Neural network

* ##### one hidden layer

* ##### and the hidden layers' parameters are fixed

$\{\mathbf{W}, \mathbf{b}\} \text{: are fixed rather than adaptively learnt}$



#### As an example, for polynomial regression with regression function

$\large h(x) = v_1 x^2 + v_2 x_2^2 +v_0$

* ##### the corresponding fixed hidden layer are 
$$\begin{align}a_1 = x_1^2 \\ a_2 = x_2^2 \end{align}$$
"""

# ╔═╡ 61873a36-2262-4a3b-bb2d-83417c76914a
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/fixed_basis_mlp_.svg
' width = '450' /></center>"

# ╔═╡ cf69619e-9313-4d14-826b-9a249e1e6b06
# md"""

# ## Make it deep



# """

# ╔═╡ 2c42c70f-05d9-4d6e-8ddd-d225d65a39a6
# TwoColumn(md"""

# \
# \
# \
# \


# We can add more than one hidden layer


# * hidden layer ``(l)`` to layer ``(l+1)``

#   * inputs: ``\mathbf{a}^{(i)} \in \mathbb{R}^m``
#   * output: ``\mathbf{a}^{(i+1)} \in \mathbb{R}^n``

# * super-index ``(l)`` index the i-th layer of a NN

# """, 
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-13.png
# ' height = '450' /></center>"
# )

# ╔═╡ 3fde5353-863b-4fd0-8ce2-c6c26d934a14
md"""

# Deep neural network
"""

# ╔═╡ e1a62d74-aaae-491e-b5ef-89e3c1516ca9
md"""

## Make it deeper


#### We can introduce more than one **hidden layer**

* ##### this is a neural network with two hidden layers
* ##### also known as *Multi-layer perceptron (MLP)*
"""

# ╔═╡ 1fe38da7-09ca-4006-8c18-97f4a5c5ce78
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp21.svg
' width = '500' /></center>"

# ╔═╡ 8b66daf7-4bf7-4323-be48-a6defefcf022
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp20.svg
# ' width = '450' /></center>"

# ╔═╡ 91deb3a5-5a59-4a71-a5f0-8037024e1e97
# md"""

# ## Make it deeper


# Here, we introduce two **additional hidden layer**

# * ##### this is a neural network with three hidden layers
# """

# ╔═╡ 9dea643a-9f72-404b-b57f-496933cc8013

md"""
## Why deeper is better ? An example


#### A _deep-ish_ network
* ##### with a bottlenet middle layer with one neuron (with identity activation)
* ##### basically **two** neural networks connected together
"""

# ╔═╡ 5e7d2395-0095-4c3f-bf21-b2c99fa34e4f
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/DeepConcat1.svg
' width = '700' /></center>"

# ╔═╡ 73b965c6-4f67-4130-ba81-e72b809b00b6
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/DeepConcat0.svg
# ' width = '700' /></center>"

# ╔═╡ b5a49d08-1786-405b-8b6c-17a9ead69cc2
# md"""
# ## Why deeper is better ?



# #### A _deep-ish_ network
# * ##### with a bottlenet middle layer with one neuron (with identity activation)
# * ##### basically **two** neural networks connected together
# """

# ╔═╡ f3f3036f-2744-47bc-95a6-6512460023f7
md"""
## Why deeper is better ?


#### A _deep-ish_ network
* ##### with a bottlenet middle layer with one neuron (with identity activation)
* ##### basically **two** neural networks connected together
"""

# ╔═╡ f815155f-7d28-473d-8fa9-e0e7f7d81df3
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/DeepConcat2.svg
' width = '700' /></center>"

# ╔═╡ ca91ae11-7b2f-4b83-a06c-e66a51ec53a7
md"""
## Why deeper is better ?


#### A _deep-ish_ network
* ##### the final function when **two** neural networks **composed together**
  * #####   ``3^2=9`` regions!
  * ##### comparing with a shallow net with ``6`` neurons

"""

# ╔═╡ 5159ff24-8746-497b-be07-727f9b7a4d82
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/DeepConcat.svg
' width = '700' /></center>"

# ╔═╡ 6a20011e-c89b-46b4-a1f9-243bd401c8fa
md"""

[Source: Understanding Deep Learning](https://udlbook.github.io/udlbook/)
"""

# ╔═╡ be0d4a9f-de82-4e40-94a9-5eda37aaeb35
md"""

## Recall: universal approximator



#### Neural network is *universal function approximator*



"""

# ╔═╡ 85bd6d6d-4d78-45cd-a4b6-3e819dd069c8
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ShallowApproximate.svg
' width = '750' /></center>"

# ╔═╡ 9c2bb240-0bf9-4add-9244-e1efa678c71f
md"""
## Demonstration

#### Learn to remember an image

"""

# ╔═╡ 185ab1d0-bdd4-4ba4-a6a6-0829237040aa
md"""
## Train a net to remember an image

#### We can train a neural net to "remember" the image 

* ##### simply approximate a $\mathbb{R}^2 \rightarrow \mathbb{R}$ function 
$$\Large f\left (\begin{bmatrix}i\\ j \end{bmatrix}\right ) = \mathbf{X}_{i,j}$$

* ##### ``i, j`` are the row/col number; the output is the pixel value
"""

# ╔═╡ c598e82d-7a32-47d9-8f4d-fa0145f0b46e
imgs = let
	filesnames = ["winnie", "alanturing", "smiley"]
	imgs = []
	for filename in filesnames
		image_url = "https://sites.cs.st-andrews.ac.uk/people/lf28//data/" *filename*".png"
		# smiley = CSV.File(download(image_url)) 
		push!(imgs, download(image_url) |> load)
	end
	[imresize(i, (256, 256)) for i in imgs]
end;

# ╔═╡ c170521b-db8f-4729-83cc-6da5d4f187e1
TwoColumn(md"
\
\


##### An image is just a matrix of pixels

$$\Large\mathbf{X} \in \mathbb{R}^{256\times 256}$$
* ##### ``256 \times 256`` pixels
", Gray.(imgs[1]))

# ╔═╡ a4edf393-5bb5-401e-808c-7cd8c00692ad
Float32.(Gray.(imgs[2]))

# ╔═╡ a430325b-f800-4f0e-93ad-f9193a9d1b01
md"""

## Demonstration: deep vs shallow

#### We train two networks (same size of parameter set)

* ##### `shallow_net`: 1 hidden layer with 10200 neurons


* ##### `deep_net`: 5 hidden layers each with 100 neurons
"""

# ╔═╡ 0d55eb49-d4a5-4c19-8520-dbf8fb062106
shallow_net = Chain(Dense(2 => 10200, relu), Dense(10200 => 1))

# ╔═╡ 7d14cc47-d326-46cc-99bc-2b6e45122fbd
deep_net = Chain(Dense(2 => 100, relu), Dense(100 => 100, relu), Dense(100 => 100, relu), Dense(100 => 100, relu), Dense(100 => 100, relu),  Dense(100 => 1))

# ╔═╡ 64baea35-079c-4ed1-b486-0442a555e062
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/lossshallowvsdeep.png
' width = '600' /></center>"

# ╔═╡ 59a750de-3594-4222-91b7-9e86feae3e1c
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/shallowvsdeep.gif
' width = '600' /></center>"

# ╔═╡ 0fbac3bc-3397-49f6-9e9d-1e31908f530e
# md"""

# ## MLP 


# ##### Neural network is a ``\mathbb{R}^m \rightarrow \mathbb{R}^n`` function 




# ```math
# \large
# \begin{align}
# \texttt{nnet}(\mathbf{x}) &= \texttt{Layer}^{(L)}(\\
# &\;\;\;\;\;\texttt{Layer}^{(L-1)}(\\
# &\;\;\;\;\;\;\;\;\; \vdots \\
# &\;\;\;\;\;\texttt{Layer}^{(2)}(\\
# &\;\;\;\;\;\texttt{Layer}^{(1)}(\mathbf{x}))\ldots)

# \end{align}
# ```

# * where $\texttt{Layer}(\mathbf{x}) = \sigma.(\texttt{Linear}(\mathbf{x};  \mathbf{W}, \mathbf{b}))$
# * ``L`` function compositions

# """

# ╔═╡ 96e3e41c-0c50-4e9b-b308-2b307fe54fc8
# TwoColumn(md"""

# ### Show me the maths

# Neural network is a ``\mathbb{R}^m \rightarrow \mathbb{R}^n`` function 




# ```math
# \begin{align}
# \texttt{nnet}(\mathbf{x}) &= \texttt{Layer}^{(L)}(\\
# &\;\;\;\;\;\texttt{Layer}^{(L-1)}(\\
# &\;\;\;\;\;\;\;\;\; \vdots \\
# &\;\;\;\;\;\texttt{Layer}^{(2)}(\\
# &\;\;\;\;\;\texttt{Layer}^{(1)}(\mathbf{x}))\ldots)

# \end{align}
# ```

# * where $\texttt{Layer}(\mathbf{x}) = \sigma.(\texttt{Linear}(\mathbf{x};  \mathbf{W}, \mathbf{b}))$
# * ``L`` function compositions
# """, md"""

# ### Show me the code


# ```python

# # create a list of layers of Linear + activations

# Layers = [Linear(in_dim, h1), 
# 		  ReLu(), 
# 		  Linear(h1, h2), 
# 		  ReLu(), ..., 
# 		  Linear(hL, ho)]

# # forward computation: function composition
# output = x
# for layer in layers:
# 	output = layer(output)
# ```
# """)

# ╔═╡ 1e2db1d6-e487-4783-ae6c-b230f8566732
# md"""

# ## Question


# !!! question "Question"
# 	What if only linear activation functions are used? 
#     * for example, identity activation functions are used only
# 	```math
# 		\sigma(z) = z
# 	```

# """

# ╔═╡ 08ed93ed-304a-46a5-bfc3-c8f45df246ed
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp21.svg
# ' height = '220' /></center>"

# ╔═╡ 24d1beda-717c-4f5b-8220-d8671dfc8187
# Foldable("Answer", md"""

# The neural network reduces to a linear function at the end!

# With identity activation, a dense layer is just a few hyperplanes (to be more mathematically correct, affine transforms)

# ```math

# \texttt{Dense}(\mathbf{x}) = \mathbf{W} \mathbf{x} +\mathbf{b}
# ```

# Consider a NN with 2 layers: composing them together is just another set of hyperplanes

# ```math
# \begin{align}
# \texttt{nnet}(\mathbf{x}) &= \texttt{Dense}^{(2)}(\texttt{Dense}^{(1)}(\mathbf{x})) \\
# &= \texttt{Dense}^{(2)}(\mathbf{W}^{(1)} \mathbf{x} +\mathbf{b}^{(1)})\\
# &= \mathbf{W}^{(2)} (\mathbf{W}^{(1)} \mathbf{x} +\mathbf{b}^{(1)}) + \mathbf{b}^{(2)}\\
# &= \underbrace{\mathbf{W}^{(2)}\mathbf{W}^{(1)}}_{\tilde{\mathbf{W}}} \mathbf{x} +\underbrace{\mathbf{W}^{(2)}\mathbf{b}^{(1)} + \mathbf{b}^{(2)}}_{\tilde{\mathbf{b}}}\\
# &= \tilde{\mathbf{W}} \mathbf{x} + \tilde{\mathbf{b}}
# \end{align}
# ```

# Adding more layers do not change a thing, at the end still a linear function!

# """)

# ╔═╡ 495ff258-ba4a-410e-b5e5-30aba7aaa95e
# md"""

# ## Implementation -- MLP
# ##### _be classy_

# """

# ╔═╡ 1d5accd1-3582-4c76-8680-30e34d27142b
# md"""
# ```python
# class MLP:
#     def __init__(self, layers):
#         self.layers = layers
  
#     def __call__(self, x):
#         yhat = x
#         for layer in self.layers:
#             yhat = layer(yhat)
#         return yhat
#     def parameters(self):
#         return [p for layer in self.layers for p in layer.parameters()]
# ```

# """

# ╔═╡ 0be45c76-6127-48e8-95fa-28827de2f38e
# TwoColumn(md"""

# For example, a MLP like this: 

# $(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp6.svg", :width=>300))

# """, 
# md"""
# Implementation:
# \
# \

# ```python
# n_i, n_h, n_o = 2, 2, 1 ## number of input and hidden units

# ## instantiate a MLP with ReLu activation
# nnet = MLP([Linear(n_i, n_h), ReLu(), Linear(n_h, n_o)])
# output = nnet(x)
# ```

# """

# )

# ╔═╡ 5c7370bc-7ba6-4449-b963-448283c80315
nn1 = let
	Random.seed!(111)
	input_dim, hidden_dim, out_dim = 2, 2, 1
	l1 = DenseLayer(input_dim, hidden_dim, tanh)
	l2 = DenseLayer(hidden_dim, out_dim, σ)
	neural_net(x) = (l2 ∘ l1)(x)
end;

# ╔═╡ 63937001-c4d9-423f-9491-b4f35342f5a4
# let
# 	i = 1
# 	nn1(D[i, :])
# end

# ╔═╡ e259603f-2baa-4242-bc04-791d1c8b168e
# nn1(D')

# ╔═╡ bedac901-af79-4798-b7b3-c9a730220351
# md"""

# ## *Why transpose

# Note that

# ```math
# \mathbf{X}=\begin{bmatrix}(\mathbf{x}^{(1)})^\top \\ (\mathbf{x}^{(2)})^\top \\ \vdots \\  (\mathbf{x}^{(n)})^\top \end{bmatrix}

# ```

# * apply transpose ``^\top`` to ``\mathbf{X}``: the columns become the observations

# ```math
# \mathbf{X}^\top=\begin{bmatrix}\mathbf{x}^{(1)}& \mathbf{x}^{(2)} &\ldots& \mathbf{x}^{(n)}\end{bmatrix}

# ```
# * consider one layer 

# ```math

# \mathbf{W} \underbrace{\begin{bmatrix}\mathbf{x}^{(1)}& \mathbf{x}^{(2)} &\ldots& \mathbf{x}^{(n)}\end{bmatrix}}_{\mathbf{X}^\top} +\mathbf{b} = \begin{bmatrix}\mathbf{W}\mathbf{x}^{(1)}+\mathbf{b}&  \mathbf{W}\mathbf{x}^{(2)}+\mathbf{b} &\ldots& \mathbf{W}\mathbf{x}^{(n)}+\mathbf{b}\end{bmatrix} 
# ```


# * the i-th column is the output of the i-th observation
# """

# ╔═╡ eeb9e21e-d891-4b7b-a92a-d0d0d70c1517
# md"""

# ## Implementation -- NN

# Neural net is just composition of layers of different constructs

# * *e.g.* `DenseLayer`

# Composing functions together is simple in Julia 

# * just use ``∘`` (just type "\circ" + `tab`): the same way as you write maths!


# ```julia
# (f ∘ g)(x) # composing functions together; the same as f(g(x))
# ```
# """

# ╔═╡ 6a378bb3-a9b9-4d94-9407-b5020fefcf39
# md"""

# ## Cross entropy loss (multiclass classification)

# Multiclass cross entropy loss

# $$
# \begin{align}
# L(\mathbf{y}; \hat{\mathbf{y}}) &= - \sum_{j=1}^C  \mathbf{y}_j \ln \underbrace{P(y =j| \mathbf{x})}_{\text{NNs softmax: } \hat{\mathbf{y}}_j}\\
# &=- \sum_{j=1}^C  \mathbf{y}_j \ln \hat{\mathbf{y}}_j
# \end{align}$$

# - ``\mathbf{y}`` is the one-hot encoded label
# - ``\hat{\mathbf{y}}`` is the softmax output
# - the binary case is just a specific case




# """

# ╔═╡ 188654a1-5211-4f87-ae1c-7a64233f8d28
# (Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp7.svg", :width=>700))


# ╔═╡ cd5c4feb-8da9-41f3-8a71-430eb254fb95
# md"""

# ##### MSE loss for regression

# """

# ╔═╡ 14482e14-e64c-4b78-abf4-469f1c404776
# (Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/mlp_reg_loss.svg", :width=>700))


# ╔═╡ dcaa1b2d-6f6f-4635-ae39-06a9e9555bce
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-06.png
# ' width = '640' /></center>"

# ╔═╡ 44833991-5520-4046-acc9-c63a9700acf3
# md"""


# ## Gradient calculation - Backpropagation





# **Backpropagation** (BP) is an efficient algorithm to compute the gradient for NN
# * a specific case of reverse-mode auto-differentiation algorithm
# * check the appendix for details of BP algorithm as well as a BP implementation


# ##### We will talk about the gradient calculation in detail next time

# """

# ╔═╡ fd1b3955-85eb-45a4-9798-7d453f1cdd28
# md"""


# ## Gradient calculation - Backpropagation





# #### **Backpropagation** (BP) is an efficient algorithm to compute the gradient for NN
# * ##### a specific case of reverse-mode auto-differentiation algorithm


# ##### We will talk about the gradient calculation in a minute

# """

# ╔═╡ 231e0908-a543-4fd9-80cd-249961a8ddaa
md"""

## Learning 


#### Minimise a loss (or equivalently maximise the log likelihood)

```math
\Large 
\hat{\boldsymbol{\Theta}} \leftarrow \arg\min_{\boldsymbol{\Theta}}\underbrace{\frac{1}{n} \sum_{i=1}^n \ell(y^{(i)}, \hat{y}^{(i)}) }_{\ell(\Theta)}
```

* ##### denote the parameter together ``\boldsymbol{\Theta} = \{\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \mathbf{W}^{(2)}, \mathbf{b}^{(2)}, \ldots\}``
  * where ``\hat{y}^{(i)}`` is the output of the neural network
  * e.g. for binary classification, we use cross entropy

* ##### gradient descent


```math
\Large
\boxed{\mathbf{\Theta}^{(t)} \leftarrow \mathbf{\Theta}^{(t-1)} - γ\, \nabla \ell(\mathbf{\Theta}^{(t-1)})}
```

* ##### we will talk about the gradient computation in a minute

"""

# ╔═╡ 231fd696-89f3-4bf8-9466-ddcea3789d21
md"""

## Demonstrations

"""

# ╔═╡ 41b78734-0fae-4323-9af1-e5e0deda584c
md"""

### Create the NN

"""

# ╔═╡ 8a8c1903-45c9-455f-a02a-fa858ae7bada
TwoColumn(md"""


* ##### input size: 2
* ##### hidden size: 2 (activation function e.g. `tanh`)
* ##### output size: 1 (binary classification)
""", 
	
	html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-05.png
' width = '400' /></center>")

# ╔═╡ f892235e-50cc-4d74-b8cc-61768450a9e3
begin
	Random.seed!(1234)
	input_dim, hidden_dim, out_dim = 2, 2, 1
	l1 = DenseLayer(input_dim, hidden_dim, tanh)
	l2 = DenseLayer(hidden_dim, out_dim)
	neural_net(x) = (l2 ∘ l1)(x)
end;

# ╔═╡ 2fbcf298-13e7-4c1e-be10-de8ca82c9385
md"""
##
### Learning

"""

# ╔═╡ ec5c193a-33de-48c9-a941-ceffc811596f
function cross_entropy_loss(y, ŷ)
	# deal with boundary cases such as σ = 0 or 1, log(0) gracefully
	# sum(y .* log.(σ) + (1 .- y).* log.(1 .- σ))
	# rather you should use xlogy and xlog1py
	-sum(xlogy.(y, ŷ) + xlog1py.(1 .-y, -ŷ))
end;

# ╔═╡ 5c5538ce-9e0e-4fba-9dfb-efa37cd43b9b
# md"""

# Use Zygote.jl to compute the gradient ``\nabla_\boldsymbol{\Theta}L``
# """

# ╔═╡ 8a654c85-7095-4f91-82d0-2393f90b3aa8
# let
# 	Θ = [l1, l2]
# 	∇g_ = Zygote.gradient(() -> cross_entropy_loss(targets', neural_net(D')), Params(Θ)) # Params(θ) tells Zygote what parameters' gradient we are computing
# 	∇g_[l1], ∇g_[l2]
# end

# ╔═╡ e746c069-f18e-4367-8cdb-7ffaac0f9ace
# md"""

# Helper method for gradient **update**
# * it becomes handy if we have a very deep NN (many layers)
# """

# ╔═╡ b1edd6a5-c00d-4ef5-949f-ce12e9507c58
function update!(layer::DenseLayer, ∇layer, γ)
	layer.W .= layer.W + γ * ∇layer.W
	layer.b .= layer.b + γ * ∇layer.b
end;

# ╔═╡ 596a1ddd-1aaa-4b02-ab87-be0a6c3fbdfd
accuracy(ŷ, y) = mean(ŷ .== y); # helper method for accuracy;

# ╔═╡ 69f26e0f-20c3-4e1d-95db-0c365f272f6d
md"""

## What neural network is doing?

#### Let's inspect the hidden neurons after training

* ##### note that hidden layer + output layer is a ordinary logistic regression
  * ###### from ``\mathbf{a} \in \mathbb{R}^2`` perspective
"""

# ╔═╡ bff67477-f769-44b2-bd07-21b439eced35
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/logis_mlp.svg
' width = '500' /></center>"

# ╔═╡ 3bfccf63-8658-4c52-b2cf-047fcafec8e7
md"""

## What neural network is doing?

"""

# ╔═╡ 47cc554e-e2f8-4e90-84a8-64b628ff4703
TwoColumn(md"""

#### NN *cleverly* and *automatically* *engineered* **good** features 


```math
\mathbf{a} = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix}
```

* the learnt features become **linearly separable** for the output layer

""", html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/logis_mlp.svg
' width = '500' /></center>")

# ╔═╡ 93ce5d39-5489-415c-9709-8740a016db06
md"""

## Another example -- the smile data

"""

# ╔═╡ b40fd686-c82b-465c-ad8f-bcea54d62aac
begin
	gr()
	# Visualizing
	scatter(real[1,1:500],real[2,1:500], label="class 1",title="A non-linear classification")
	scatter!(fake[1,1:500],fake[2,1:500], ratio=1, label ="class 2", framestyle=:origin)
end

# ╔═╡ 8067e844-14ed-4fb9-b609-ff598d61cf9e
begin
	D_smile = [real'; fake']
	# idx_shuffle = shuffle(1:size(D_smile))
	targets_smile = [ones(size(real)[2]); zeros(size(fake)[2])]
	idx_shuffle = shuffle(1:length(targets_smile))

	D_smile = D_smile[idx_shuffle, :]
	targets_smile = targets_smile[idx_shuffle]
end;

# ╔═╡ 18f85862-5b89-4358-8431-db7fdd900b9b
begin
	D_smile_test = [real_test'; fake_test'];
	targets_smile_test = [ones(size(real_test)[2]); zeros(size(fake_test)[2])]
end;

# ╔═╡ 743e1b17-23b9-4a2a-9466-4c0ff03e8888
md"""
##

### Create the network
"""

# ╔═╡ baca7037-f1eb-4b61-b157-9f5e12523894
# md"""

# #### Let's add one more hidden layer

# ```math
# \texttt{nnet}(\mathbf{x}) = (\underbrace{\texttt{Layer}^{(3)}}_{\text{hidden 2 to output}} \circ \underbrace{\texttt{Layer}^{(2)}}_{\text{hidden 1 to hidden 2}} \circ \underbrace{\texttt{Layer}^{(1)}}_{\text{input to hidden 1}}) (\mathbf{x})
# ```

# * where $\texttt{Layer}(\mathbf{x}) = \sigma.(\texttt{Linear}(\mathbf{x};  \mathbf{W}, \mathbf{b}))$
# """

# ╔═╡ 3202e689-edb5-4dd5-b8ac-74a4fd0251e6
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-16.png
' width = '500' /></center>"

# ╔═╡ fbc066bb-90ab-4d9d-8f77-010662290f60
begin
	Random.seed!(123)
	# create a neural network
	hidden_1 = 20
	hidden_2 = 3
	l1_2 = DenseLayer(2, hidden_1, tanh)
	l2_2 = DenseLayer(hidden_1, hidden_2, tanh)
	l3_2 = DenseLayer(hidden_2, 1)
	nnet2(x) = (l3_2 ∘ l2_2 ∘ l1_2)(x)
end;

# ╔═╡ 9b7a4c44-76c0-4627-a226-e43c78141031
loss_smile, anim_smile = let
	gr()
	# learning rate
	γ = 0.0001
	iters = 1000
	losses = zeros(iters)
	layers = [l1_2, l2_2, l3_2]
	anim = @animate for i in 1: iters
		# ∇gt = Zygote.gradient(() -> cross_entropy_loss(targets_smile[:], nnet2(D_smile')[:] ), Params(layers))

		loss, ∇gt = Zygote.withgradient(() -> Flux.logitbinarycrossentropy(nnet2(D_smile'), targets_smile'; agg=sum), Params(layers))
		for l in layers
			update!(l, ∇gt[l], -γ)
		end
		losses[i] = loss
		if i % 20 == 1
			ŷ = nnet2(D_smile')[:] .> 0.5
			@info "Iteration, accuracy: ", i , accuracy(targets_smile[:], ŷ)*100
		end

		scatter(real[1,1:100],real[2,1:100], zcolor=nnet2(real)', ratio=1, colorbar=false, framestyle=:origin)
		scatter!(fake[1,1:100],fake[2,1:100],zcolor=nnet2(fake)',legend=false, title="Iteration: "*string(i))
	end every 50
	losses, anim
end;

# ╔═╡ 79eab834-9a65-4c94-9466-6e9de387dbca
# gif(anim_smile, fps=5)

# ╔═╡ 93a6a56d-ebbd-4f1b-8d65-fbcbecadeef7
md"""
## 

### The decision boundary after training
"""

# ╔═╡ c14d5645-4ccb-413c-ad54-ee9d45706405
begin
	gr()
	plot(-0.7:0.1:.7, -0.25:0.1:1, (x, y) -> nnet2([x, y])[1] < 0.5, alpha=0.4,  st=:contourf, c=:jet, framestyle=:origin)
	scatter!(real[1,1:100],real[2,1:100], c=1, ratio=1, colorbar=false)
	scatter!(fake[1,1:100],fake[2,1:100],c=2,legend=false)
end

# ╔═╡ 298fbc0b-5e94-447b-8a6c-a0533490e8e1
md"""

## What neural network is doing?

"""

# ╔═╡ 4590f3b7-031d-4039-b125-945e3dcdcea7
TwoColumn(md"""

#### NN *cleverly* and *automatically* *engineered* **good** features again


```math
\mathbf{a}^{(2)} = \begin{bmatrix} a_1^{(2)} \\ a_2^{(2)} \\ a_3^{(2)} \end{bmatrix}
```

* the learnt features become **linearly separable** for the output layer

""", html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/smile_logits.svg
' width = '500' /></center>")

# ╔═╡ d12ab2fb-a91d-498a-844f-0148e56110d7
let
	import PlotlyBase
	plotly()
	hidden_output = l2_2(l1_2((D_smile')))
	scatter(hidden_output[1, targets_smile .== 0], hidden_output[2, targets_smile .==0], hidden_output[3, targets_smile .==0], legend=:outerright, label="class 1", markersize=2)
	scatter!(hidden_output[1, targets_smile .== 1], hidden_output[2,targets_smile .==1], hidden_output[3, targets_smile .== 1], label="class 2", xlabel="a1", ylabel="a2", zlabel="a3", size=(500,400), title="The hidden layer's output",markersize=2)
end

# ╔═╡ 575455dc-81bb-4529-b971-b554b5601fce
# zs = (xys[:, 1])[:] ;

# ╔═╡ eff06e68-b8b6-4009-b3f2-0c117f03bfe0
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.

# ╔═╡ e4f0b81f-12bb-4014-bc54-69cc8701a28a
begin
	nobs_moon = 200
	X_moon, y_moon = MLJBase.make_moons(nobs_moon)
	# Matrix(X_moon)
	X_moon = matrix(X_moon)
	y_moon = Array(y_moon)
end;

# ╔═╡ fad7cf61-f270-4a25-b69c-ff37e3da5468
project_A(a) = a * a' / dot(a,a); # return the projection matrix

# ╔═╡ d9f87602-1d22-4967-ac5b-d7bc39cf4931
A_proj_1 = project_A(ones(2));

# ╔═╡ d18c622a-5c17-4a32-9bf7-f58d34e7a10a
A_strech_vert = let
	δ₂ = 0.5
	A_strech_vert = [1 0; 0 1+δ₂]
end;

# ╔═╡ a991e15e-54cb-4698-9481-58cfe9b12f55
A_strech = let
	δ = 0.5
	A_strech = [1 + δ 0; 0 1]
end;

# ╔═╡ 5bc25e85-fb53-47ad-b19b-2697b80927b3
Rmat(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)];

# ╔═╡ 19958842-85f0-4475-a29b-0b282232d9a4
R = Rmat(5);

# ╔═╡ 30f16cf7-0cd7-42bc-857e-9390b404a1dd
A_sheering = let
	δ = -0.5
	A = [1 δ; 0 1]
end;

# ╔═╡ 569d9b78-1684-4db5-b938-59ff6422345b
A_proj_x = let
	A = project_A([0.5, 1.5])
end;

# ╔═╡ 68376541-4b03-4eb2-b4e2-d1469d8dd07e
A_id = I

# ╔═╡ 1d67a460-c117-49f0-b161-7ae28dd43107
md"""


## Recall: linear transformation


$$\Large \mathbf{x} \rightarrow \mathbf{Wx}$$


* ##### input $\mathbf{x} \in \mathbb{R}^2$

* ##### output $\mathbf{Wx} \in \mathbb{R}^2$

* ##### $\mathbf{W}$: _e.g._ a 2 by 2 matrix

#### Composition of linear transformations: matrix multiplications



$$\Large \mathbf{x} \rightarrow \mathbf{W}_k \ldots\mathbf{W}_2\mathbf{W}_1\mathbf{x}$$


* ##### it is still a linear transformation: ``\mathbf{W} = \mathbf{W}_k \ldots \mathbf{W}_2\mathbf{W}_1``

"""

# ╔═╡ a42b6041-3d89-409c-94fe-357e41565dbb
A_compose = R * A_strech;

# ╔═╡ 7cebfb96-1650-4450-842f-dab1289bd36d
begin
	Random.seed!(123)
	A_rand = randn(2,2)

	# A_rand = A_rand * A_rand'
end;

# ╔═╡ e4c92595-d3df-4235-a728-aa1bace3dbc3
@bind A_mat Select([ A_compose => "composition", R => "rotate", A_strech => "stretch horizontal" , A_strech_vert => "stretch vert", A_sheering => "sheering", A_proj_1 => "project to 1", A_id => "identity", A_rand => "random"])

# ╔═╡ 946bceab-4256-42e8-a492-ee4e6153d116
md"Add non-linear activation: $(@bind add_non_linear CheckBox(default=false))"

# ╔═╡ 8e71ddf1-b888-4b16-b796-5dad56da733c
@bind A_mat_ Select([ A_compose => "composition", R => "rotate", A_strech => "stretch horizontal" , A_strech_vert => "stretch vert", A_sheering => "sheering", A_proj_1 => "project to 1", A_id => "identity", A_rand => "random"])

# ╔═╡ 8505d6c3-155d-4644-82ba-dda4c523ad2a
act_fun = relu

# ╔═╡ d80d54c3-a936-4044-acb0-95181b03ee70
let
	plotly()
	A_mat = A_mat_
	act = act_fun
	ms = 1.5
	xs = range(-1, 1, 10)
	ys = range(-1, 1, 10)
	xxs, yys = meshgrid(xs, ys)
	xys = [xxs yys]
	xys_ = xys * A_mat';
	depths = [1:3;]
	
# zs

	plt =  plot(xys[:, 1], xys[:, 2], depths[1] * ones(length(xys[:, 1])), c =:blue, st=:scatter, markersize=ms,   alpha=0.3, label="", ratio = 1.0,  framestyle=:zerolines, grid=false, zaxis =false,  colorbar=:false)


	


	if act == Flux.softmax
		newxyz = act(xys_, dims =2)
	else
		newxyz = act.(xys_)
	end

	if add_non_linear
		plot!(xys_[:, 1], xys_[:, 2], depths[2] * ones(length(xys_[:, 1])), st=:scatter,  c =:blue, markersize=ms, alpha=0.4, label="", title = "Non-linear activation: $(act)", titlefont = font("Courier", 12))
	
		for i in 1:(size(xys)[1])
			plot!([xys[i, 1], xys_[i, 1]], [xys[i, 2], xys_[i, 2]], [depths[1], depths[2]], lw=1.5,  ls=:solid, lc=:gray, alpha=0.75,label="")
		end
		
		plot!(newxyz[:, 1], newxyz[:, 2], depths[3] * ones(length(newxyz[:, 1])), c =:blue, st=:scatter, markersize=ms,   alpha=0.7, label="")
	
		for i in 1:(size(xys)[1])
			plot!([xys_[i, 1], newxyz[i, 1]], [xys_[i, 2], newxyz[i, 2]], [depths[2], depths[3]], lw=1.5,  ls=:solid, lc=:gray, alpha=0.75,label="")
		end

	else
		plot!(xys_[:, 1], xys_[:, 2], depths[2] * ones(length(xys_[:, 1])), st=:scatter,  c =:blue, markersize=ms, alpha=0.8, label="")
	
		for i in 1:(size(xys)[1])
			plot!([xys[i, 1], xys_[i, 1]], [xys[i, 2], xys_[i, 2]], [depths[1], depths[2]], lw=1.5,  ls=:solid, lc=:gray, alpha=0.75,label="")
		end

		
	end
	plt
end

# ╔═╡ d1224150-d1d0-4871-a81a-f6eb1897b420
# begin
# 	max_lim = 5
# 	xs = range(-1, 1, 20)
# 	ys = range(-1, 1, 20)
# 	xxs, yys = meshgrid(xs, ys)
# 	xys = [xxs yys]
# end;

# ╔═╡ 3613fb7c-61aa-4dc6-bc11-05717b78f75c
begin
	max_lim = 5
	xs = range(-5, 5, 15)
	ys = range(-5, 5, 15)
	xxs, yys = meshgrid(xs, ys)
	xys = [xxs yys]
end;

# ╔═╡ 6fe3655b-5b39-4058-90f8-b1f16639c360
rows_l, rows_r, cols_b, cols_t=let
	off_set = 0.8
	rows_left = []
	rows_right = []
	columns_bottom = []
	columns_top = []
	# for each rows
	for y in ys
		# the left and right ends
		push!(rows_left, [extrema(xs)[1] - off_set, y])
		push!(rows_right, [extrema(xs)[2] + off_set, y])
	end
	# for each columns
	for x in xs
		# the bottom and top ends
		# push!(columns_bottom_top, [[x, ys.start], [x, ys.stop]])

		push!(columns_bottom, [x, extrema(ys)[1] - off_set])
		push!(columns_top, [x, extrema(ys)[2] + off_set])
	end

	vcat(rows_left'...), vcat(rows_right'...), vcat(columns_bottom'...),vcat(columns_top'...)
end;

# ╔═╡ 35c07542-381c-4cd2-b605-c68fbde146e3
colorch = cgrad(:hot, rev=true, scale =:linear);

# ╔═╡ e5c3e1da-4e2c-46e0-8be2-2a6416958b00
zs = (xys[:, 1])[:] ;

# ╔═╡ b3599b16-f003-4d36-9092-5140d4eccb27
let
	gr()

	xys_ = xys * A_mat';
	# zs =

	# plt = plot(xys_[:, 1], xys_[:, 2], st=:scatter, marker_z = zs, c=colorch, colorbar=:false,  markersize=4, alpha=0.8, label="", ratio = 1.0,xlim = extrema(xys_[:, 1]) .+ (-.5, .5), ylim =extrema(xys_[:, 2]) .+ (-.5, .5), framestyle=:origin, grid=false )

	plt = plot(xys_[:, 1], xys_[:, 2], st=:scatter, marker_z = zs, c=colorch, colorbar=:false,  markersize=4, alpha=0.8, label="", ratio = 1.0, xlim = [-9, 9], ylim =extrema(xys_[:, 2]) .+ (-.5, .5), framestyle=:origin, grid=false )


	plot!(xys[:, 1], xys[:, 2], st=:scatter, markersize=4,  c=:gray, alpha=0.2, label="")

	rows_l_ = rows_l * A_mat'
	rows_r_ = rows_r * A_mat'
	for (l, r) in zip(eachrow(rows_l_), eachrow(rows_r_))
		# l_ = A_mat * l_
		plot!([l[1], r[1]], [l[2], r[2]], lc=:gray, alpha=0.75,label="")
	end

	
	cols_b_ = cols_b * A_mat'
	cols_t_ = cols_t * A_mat'

	for (b, t) in zip(eachrow(cols_b_), eachrow(cols_t_))
		plot!([b[1], t[1]], [b[2], t[2]], lc=:gray, alpha=0.75,label="")
	end

	plt
end

# ╔═╡ 3ceef48c-09b0-496e-a65b-34cc01f3f3eb
md"""

## Recall: _non-linear_ transform?

```math
\large
f\left (\begin{bmatrix}x_1\\ x_2 \end{bmatrix}\right ) = \begin{bmatrix}x_1+x_2 \\ \sin(x_1)\cos(x_2) +x_2\end{bmatrix}
```
"""

# ╔═╡ d9bf168e-96cc-4161-aabd-b2c1ce19abcb
f(x) = [x[1]+x[2], sin(x[1]) * cos(x[2]) + x[2]]

# ╔═╡ d03452d1-fc84-411f-86e2-c8fce6cff01e
# f(v) = [v[1]^2 * v[2] , v[2]^2 * v[1]] ./10

# ╔═╡ 387128c4-1829-4bc8-afdb-ee4d66e3650e
# f(v) = [v[1]^2 * v[2]^2 , v[2]^2 * v[1]]

# ╔═╡ 8eb4ec4b-521a-4957-8a70-9da05bf0f5af
let
	# xys_ = xys * A_mat';
	gr()

	xys_ = vcat(f.(eachrow(xys))'...)
	
	plt = plot(xys_[:, 1], xys_[:, 2], st=:scatter, ratio=1, marker_z=zs, c=colorch, colorbar=:false, framestyle=:origin, markersize=4, alpha=0.8, label="", xlim = extrema(xys_[:, 1]) .+ (-1, 1), ylim =extrema(xys_[:, 2]) .+ (-1, 1))


	# for eachcol(xys)

	xs_ = range(extrema(xs)..., 100)
	for y in ys
		ps = [xs_ ones(length(xs_)) * y]
		ps_ = vcat(f.(eachrow(ps))'...)
		plot!(ps_[:, 1], ps_[:,2], st=:path, alpha=0.5,  lc=:gray,label="")
	end

	ys_ = range(extrema(ys)..., 100)
	for x in xs
		
		ps = [ones(length(ys_)) * x ys_]
		ps_ = vcat(f.(eachrow(ps))'...)
		plot!(ps_[:, 1], ps_[:,2], st=:path, alpha=0.5, lc=:gray,label="")
	end

	
	# cols_b_ = cols_b * A_mat'
	# cols_t_ = cols_t * A_mat'

	# for (b, t) in zip(eachrow(cols_b_), eachrow(cols_t_))
	# 	plot!([b[1], t[1]], [b[2], t[2]], lc=:gray, label="")
	# end

	plt
end

# ╔═╡ 7d5d8ec2-2980-4af0-94f5-31e934416261
# let
# 	gr()
# 	A_mat = A_sheer_lr
# 	xys_ = xys * A_mat';
# 	# zs =

# 	# plt = plot(xys_[:, 1], xys_[:, 2], st=:scatter, marker_z = zs, c=colorch, colorbar=:false,  markersize=4, alpha=0.8, label="", ratio = 1.0,xlim = extrema(xys_[:, 1]) .+ (-.5, .5), ylim =extrema(xys_[:, 2]) .+ (-.5, .5), framestyle=:origin, grid=false )

# 	plt = plot(xys_[:, 1], xys_[:, 2], st=:scatter, marker_z = zs, c=colorch, colorbar=:false,  markersize=4, alpha=0.8, label="", ratio = 1.0, xlim = [-9, 9], ylim =extrema(xys_[:, 2]) .+ (-.5, .5), framestyle=:origin, grid=false )


# 	plot!(xys[:, 1], xys[:, 2], st=:scatter, markersize=4,  c=:gray, alpha=0.2, label="")

# 	rows_l_ = rows_l * A_mat'
# 	rows_r_ = rows_r * A_mat'
# 	for (l, r) in zip(eachrow(rows_l_), eachrow(rows_r_))
# 		# l_ = A_mat * l_
# 		plot!([l[1], r[1]], [l[2], r[2]], lc=:gray, alpha=0.75,label="")
# 	end

	
# 	cols_b_ = cols_b * A_mat'
# 	cols_t_ = cols_t * A_mat'

# 	for (b, t) in zip(eachrow(cols_b_), eachrow(cols_t_))
# 		plot!([b[1], t[1]], [b[2], t[2]], lc=:gray, alpha=0.75,label="")
# 	end

# 	plt
# end

# ╔═╡ 54fc77fd-f26a-44cd-90d1-85a4a917223f
md"""

## What neural network is doing? (cont.)

#### Neural network (``K`` layer MLP) _without_ non-linear activations
$$\Large \mathbf{x} \rightarrow (\mathbf{W}_K \ldots(\mathbf{W}_2(\mathbf{W}_1\mathbf{x}))\ldots)$$
  * #####  the biases are omit here for simplicity but the effect is the same



"""

# ╔═╡ e9b6e233-1e1f-436e-8701-5e1092f27c07
md"""

## What neural network is doing? (cont.)

#### Neural network (MLP with ``K`` layers) _without_ activations

$$\Large \mathbf{x} \rightarrow (\mathbf{W}_K \ldots(\mathbf{W}_2(\mathbf{W}_1\mathbf{x}))\ldots)$$

$\Large\Downarrow$

$$\Large \mathbf{x} \rightarrow \underbrace{\mathbf{W}_K \ldots\mathbf{W}_2\mathbf{W}_1}_{\mathbf{W}}\mathbf{x}$$

  * #####  the biases are omit here for simplicity but effect is the same



* ##### and what a linear transformation can do is very limited 
"""

# ╔═╡ 1235b084-12bb-4537-a0da-1c3df7e0db3f
@bind δ_lt Slider(-2:0.1:2)

# ╔═╡ 6fcdf294-95b7-40ef-a2b5-b37b46ff3ba0
A_sheer_lr = [1 δ_lt; 0 1];

# ╔═╡ 884c04d6-553d-42a2-b831-5163d078c8b7
A_I = Matrix(I,2,2);

# ╔═╡ 04ec6ff8-7a99-4acc-a192-3ee93a3f9d60
@bind A_mat2 Select([A_sheer_lr => "sheering",  A_I => "identity", A_compose => "composition", R => "rotate", A_strech => "stretch horizontal" , A_strech_vert => "stretch vert", A_proj_1 => "project to 1"])

# ╔═╡ 33b4bade-3ca4-41c5-aa9a-44571339160a
let
	A_mat = A_mat2
	plt = plot(ratio=1, xlabel=L"x_1", ylabel=L"x_2", framestyle =:zerolines)

	X_moon_new = X_moon * A_mat'
	for c in [0, 1]
		scatter!(X_moon_new[y_moon .== c, 1], X_moon_new[y_moon .== c, 2], c= c+1, label="class $(c)")
	end

	plt
end

# ╔═╡ a38d9dab-575f-4299-87a1-0913f347ee3a
md"""

## What neural network is doing? (cont.)

#### Neural network _with_ non-linear activations

$$\Large \mathbf{x} \rightarrow σ.(\mathbf{W}_K \ldots \, σ.(\mathbf{W}_2 \,\sigma.(\mathbf{W}_1\mathbf{x})))$$

* ##### it becomes a nonlinear transformation

"""

# ╔═╡ ebac79e3-2a84-4513-a9d3-3fecf5c24413
md"""

## What neural network is doing? (cont.)

#### Neural network _with_ non-linear activations


$$\Large \mathbf{x} \rightarrow σ.(\mathbf{W}_K \ldots \, σ.(\mathbf{W}_2 \,\sigma.(\mathbf{W}_1\mathbf{x})))$$

* ##### it becomes a nonlinear transformation


#### Equivalently: a sequence of $K$ `non-linear` transformations

$$\Large \mathbf{x} \rightarrow \mathbf{x}_1 \rightarrow \mathbf{x}_2 \rightarrow \ldots \rightarrow \mathbf{x}_K$$


  * ##### where $\mathbf{x}_k = \sigma.(\mathbf{W}_k \mathbf{x}_{k-1})$

* ##### we can visualise these hidden outputs $\{\mathbf{x}_k\}_{k=1}^K$
"""

# ╔═╡ 4ae14b4a-9f87-49cf-9f70-6f191a00e78c
md"""
## What neural network is doing? (cont.)



$$\Large \mathbf{x} \rightarrow \mathbf{x}_1 \rightarrow \mathbf{x}_2 \rightarrow \ldots \rightarrow \mathbf{x}_K$$

  * ##### ``K=5`` here
  * ##### where $\mathbf{x}_k = \sigma.(\mathbf{W}_k \mathbf{x}_{k-1})$

* ##### the hidden outputs $\{\mathbf{x}_k\}_{k=1}^K$ are plotted below
"""

# ╔═╡ f6554d6f-3856-436d-a33a-3ca8a132877a
begin
	Random.seed!(12345)
	nnet_moon = Chain(
	    Dense(2 => 2, tanh),   # activation function inside layer
	    Dense(2 => 2, tanh),
		Dense(2 => 2, tanh),
		Dense(2 => 2, tanh),
		Dense(2 => 2, tanh),
		Dense(2=>1,)
	) |> f64
end

# ╔═╡ b913038f-f883-4fec-9bb6-1f427382cf26
begin

	y_moon_one_hot = Flux.onehotbatch(y_moon, [true, false]) 
	# moon_loader = Flux.DataLoader((X_moon', y_moon_one_hot), batchsize=100, shuffle=true);
	moon_loader = Flux.DataLoader((X_moon', y_moon'), batchsize=50, shuffle=true);

	optim = Flux.setup(Flux.Adam(0.01), nnet_moon)  # will store optimiser momentum, etc.

# # Train
	losses_moon = []
	for i in 1:5_00
		for (xtrain, ytrain) in moon_loader
	 		loss, grad = Flux.withgradient(nnet_moon) do m
				y_preds = m(xtrain) 
				# Flux.Losses.logitcrossentropy(y_preds, ytrain)
				Flux.Losses.logitbinarycrossentropy(y_preds, ytrain)
			end
			Flux.update!(optim, nnet_moon, grad[1])
			push!(losses_moon, loss)
		end
		
	end
end

# ╔═╡ 81458084-db0a-4140-a42b-c536ad8391a9
begin
	Random.seed!(2345)
	nnet_moon2 = Chain(
	    Dense(2 => 2, relu; init = 	Flux.kaiming_normal),   # activation function inside layer
	    Dense(2 => 2, relu; init = 	Flux.kaiming_normal),
		# Dense(2 => 2, tanh; init = 	Flux.kaiming_normal),
		# Dense(2 => 2, tanh; init = 	Flux.kaiming_normal),
		# Dense(2 => 2, tanh; init = 	Flux.kaiming_normal),
		Dense(2=>2),
		Flux.softmax
	) |> f64
	moon_loader2 = Flux.DataLoader((X_moon', y_moon_one_hot), batchsize=50, shuffle=true);


	optim2 = Flux.setup(Flux.Adam(0.01), nnet_moon2)  # will store optimiser momentum, etc.
	losses_moon2 = []
	for i in 1:5_00
		for (xtrain, ytrain) in moon_loader2
	 		loss, grad = Flux.withgradient(nnet_moon2) do m
				y_preds = m(xtrain) 
			# 	# Flux.Losses.logitcrossentropy(y_preds, ytrain)
				Flux.Losses.crossentropy(y_preds, ytrain)
			end
			Flux.update!(optim2, nnet_moon2, grad[1])
			push!(losses_moon2, loss)
		end
	end

end

# ╔═╡ a549be88-f812-40e6-8100-d4702303e956
# plot(losses_moon2)

# ╔═╡ bbffe411-a6d8-40e0-9005-c9a4170b3cec
plts = let
	gr()
	plts = []
	xnew = X_moon'
	plt = plot(ratio=1, title="Layer 0: "*L"\mathbf{x}", framestyle=:zerolines)

	xs = range(extrema(X_moon[:, 1]) .+ (-.5, .5)..., 10)
	ys = range(extrema(X_moon[:, 2]) .+ (-.5, .5)..., 10)
	# xs_ = range(extrema(xnew)..., 100)
	row_lines = []
	for y in ys
		xs_ = range(extrema(X_moon[:, 1]) .+ (-.5, .5)..., 50)
		ps = [xs_ ones(length(xs_)) * y]
		ps_ = ps
		plot!(ps_[:, 1], ps_[:,2], st=:path, alpha=0.5,  lc=:gray,label="")
		push!(row_lines, ps_)
	end

	col_lines = []
	for x in xs
		ys_ =  range(extrema(X_moon[:, 2]) .+ (-.5, .5)..., 50)
		ps = [ones(length(ys_))*x ys_]
		ps_ = ps
		plot!(ps_[:, 1], ps_[:,2], st=:path, alpha=0.5,  lc=:gray,label="")
		push!(col_lines, ps_)
	end

	for c in [0, 1]
		scatter!(xnew[1, y_moon .== c], xnew[2, y_moon .== c], c= c+1, label="class $(c)", alpha=0.5)
	end

	push!(plts, plt)
	for (i, f) in enumerate(nnet_moon[1:(end-1)])
		xnew = f(xnew)
		plt = plot(ratio=1, title="Layer $(i): "*L"\mathbf{x}_{%$(i)}", framestyle=:zerolines)
		for (ri, row) in enumerate(row_lines)
			# ps = [xs ones(length(xs)) * y]
			# xs_ = range(extrema(X_moon[:, 1]) .+ (-.5, .5)..., 50)
			# ps = [xs_ ones(length(xs_)) * y]
			# ps = [xs ones(length(xs)) * y]
			ps_ = f(row')'
			plot!(ps_[:, 1], ps_[:,2], st=:path, alpha=0.5,  lc=:gray,label="")
			row_lines[ri] = ps_
		end
		for (ci, col) in enumerate(col_lines)
			# ys_ =  range(extrema(X_moon[:, 2]) .+ (-.5, .5)..., 50)
			# ps = [ones(length(ys_))*x ys_]
			ps_ = f(col')'
			plot!(ps_[:, 1], ps_[:,2], st=:path, alpha=0.5,  lc=:gray,label="")
			col_lines[ci] = ps_
		end
		
		for c in [0, 1]
			scatter!(xnew[1, y_moon .== c], xnew[2, y_moon .== c], c= c+1, label="class $(c)",  alpha=0.5)
		end
		push!(plts, plt)
	end
	
	plts
end;

# ╔═╡ 734aaf86-6146-4fc8-921c-9689aac9e639
plot(plts..., titlefontsize=12, size=(800,600))

# ╔═╡ 1393ed38-0945-43ee-8a35-3f4ebda1fec0
# let
# 	plotly()
# 	plts = []
# 	xnew = X_moon'
# 	# plt = plot(title="Layer 0: "*L"\mathbf{x}", framestyle=:one)
# 	plt = plot(title="", framestyle=:none, camera = (40, 30))

# 	xs = range(extrema(X_moon[:, 1]) .+ (-.5, .5)..., 10)
# 	ys = range(extrema(X_moon[:, 2]) .+ (-.5, .5)..., 10)
# 	row_lines = []
# 	for y in ys
# 		xs_ = range(extrema(X_moon[:, 1]) .+ (-.5, .5)..., 50)
# 		ps = [xs_ ones(length(xs_)) * y]
# 		ps_ = ps
# 		plot!(ps_[:, 1], ps_[:,2], zeros(length(ps_[:, 1])), st=:path, alpha=1,  lc=:gray,label="")
# 		push!(row_lines, ps_)
# 	end

# 	col_lines = []
# 	for x in xs
# 		ys_ =  range(extrema(X_moon[:, 2]) .+ (-.5, .5)..., 50)
# 		ps = [ones(length(ys_))*x ys_]
# 		ps_ = ps
# 		plot!(ps_[:, 1], ps_[:,2], zeros(length(ps_[:, 1])), st=:path, alpha=1,  lc=:gray,label="")
# 		push!(col_lines, ps_)
# 	end

# 	for c in [0, 1]
# 		scatter!(xnew[1, y_moon .== c], xnew[2, y_moon .== c], zeros(length(xnew[1, y_moon .== c])), c= c+1, label="", ms=1, alpha=1, zaxis=false, xaxis=false, yaxis=false)
# 	end

# 	plt

# 	# nnet

# 	# f1 = nnet_moon2[1]
# 	# layer = nnet_moon2[1]
# 	# f(x) = layer.weight * x .+ layer.bias
# 	# f(x) = layer(x)
# 	# xnew = f(xnew)

# 	# push!(plts, plt)
# 	for (i, f) in enumerate(nnet_moon2[1:(end)])
# 		xnew = f(xnew)
# 		depth = i * 5
# 	# 	plt = plot(ratio=1, title="Layer $(i): "*L"\mathbf{x}_{%$(i)}", framestyle=:zerolines)
# 		for (ri, row) in enumerate(row_lines)
# 			# ps = [xs ones(length(xs)) * y]
# 			# xs_ = range(extrema(X_moon[:, 1]) .+ (-.5, .5)..., 50)
# 			# ps = [xs_ ones(length(xs_)) * y]
# 			# ps = [xs ones(length(xs)) * y]
# 			ps_ = f(row')'
# 			plot!(ps_[:, 1], ps_[:,2], depth * ones(length(ps_[:, 1])), st=:path, alpha=0.5,  lc=:gray,label="")
# 			row_lines[ri] = ps_
# 		end

# 	# plt
# 		for (ci, col) in enumerate(col_lines)
# 			# ys_ =  range(extrema(X_moon[:, 2]) .+ (-.5, .5)..., 50)
# 			# ps = [ones(length(ys_))*x ys_]
# 			ps_ = f(col')'
# 			plot!(ps_[:, 1], ps_[:,2], depth * ones(length(ps_[:, 1])), st=:path, alpha=0.5,  lc=:gray,label="")
# 			col_lines[ci] = ps_
# 		end

# 	# plt
		
# 		for c in [0, 1]
# 			scatter!(xnew[1, y_moon .== c], xnew[2, y_moon .== c], depth * ones(length(xnew[1, y_moon .== c])), c= c+1,ms=1, label="",  alpha=0.5)
# 		end
# 		# push!(plts, plt)
# 	end
# 	plt
	
# 	# plts
# end

# ╔═╡ b00bf65f-3f6d-40b2-a95a-22a7ad17af96
md"""

## The smile dataset


"""

# ╔═╡ 30810c50-83f7-4fac-8492-0e0018911dfa
plt_smile

# ╔═╡ 776ec1b6-02f8-4dfd-b3c4-563552de1d6a
md"""


#### Alternative activations: any $\sigma$ (as long as non-linear) works to certain degree

* ##### _e.g._ here what is shown below is $\sigma \triangleq \sin(z)$
"""

# ╔═╡ d3f5bf51-44ae-4231-afcd-c118f6e599a3
begin
	Random.seed!(123456)
	act_smile = sin ## use sin as activation function
	nnet_smile = Chain(
	    Dense(2 => 2, act_smile),   # activation function inside layer
	    Dense(2 => 2, act_smile),
		Dense(2 => 2, act_smile),
		Dense(2 => 2, act_smile),
		Dense(2 => 2, act_smile),
		Dense(2=>1,)
	) |> f64

		
	smile_loader = Flux.DataLoader((D_smile', targets_smile'), batchsize=100, shuffle=true);

end;

# ╔═╡ 25b06875-81c4-4d5e-86c8-f744f4e45c08
md"""

### The hidden transformations
"""

# ╔═╡ abf54f94-93b2-4275-b59b-b501b40feb84
begin

	optim_smile = Flux.setup(Flux.Adam(0.001), nnet_smile)  # will store optimiser 
	losses_smile = []
	for i in 1:10_00
		for (xtrain, ytrain) in smile_loader
	 		loss, grad = Flux.withgradient(nnet_smile) do m
				y_preds = m(xtrain) 
				Flux.Losses.logitbinarycrossentropy(y_preds, ytrain)
			end
			Flux.update!(optim_smile, nnet_smile, grad[1])
			push!(losses_smile, loss)
		end
		
	end

	smile_train_acc = mean((nnet_smile(D_smile')[:] .> 0) .== targets_smile[:]);

	smile_test_acc = mean((nnet_smile(D_smile_test')[:] .> 0) .== targets_smile_test[:]);
end;

# ╔═╡ a2512481-deac-4529-ae1c-6f8d8eae9613
let
	gr()
	plot(losses_smile, dpi=250, xlabel="Iteration", xscale=:log10, label="loss")
	n = length(smile_loader)
	plot!(n:n:length(losses_smile), mean.(Iterators.partition(losses_smile, n)),
    label="epoch mean", dpi=250, title="Loss curve for the smile dataset")
end

# ╔═╡ 9bcba339-0d81-4126-b1cd-0a779b801e38
md"#### Training accuracy: $(smile_train_acc * 100) %"

# ╔═╡ 5b461b02-a6b7-43a0-af64-ceca6e25cc3e
md"#### Testing accuracy: $(smile_test_acc * 100) %"

# ╔═╡ 42f0ef1f-69de-4934-bea8-572c45767c13
plts_smile = let
	gr()
	plts = []
	XX = D_smile[1:300, :]
	# xnew = D_smile'
	yy = targets_smile[1:300]
	plt = plot(ratio=1, title="Layer 0: "*L"\mathbf{x}", framestyle=:zerolines)

	xs = range(extrema(XX[:, 1]) .+ (-.5, .5)..., 10)
	ys = range(extrema(XX[:, 2]) .+ (-.5, .5)..., 10)
	# xs_ = range(extrema(xnew)..., 100)
	row_lines = []
	for y in ys
		xs_ = range(extrema(XX[:, 1]) .+ (-.5, .5)..., 50)
		ps = [xs_ ones(length(xs_)) * y]
		ps_ = ps
		plot!(ps_[:, 1], ps_[:,2], st=:path, alpha=0.5,  lc=:gray,label="")
		push!(row_lines, ps_)
	end

	col_lines = []
	for x in xs
		ys_ =  range(extrema(XX[:, 2]) .+ (-.5, .5)..., 50)
		ps = [ones(length(ys_))*x ys_]
		ps_ = ps
		plot!(ps_[:, 1], ps_[:,2], st=:path, alpha=0.5,  lc=:gray,label="")
		push!(col_lines, ps_)
	end

	for c in [0, 1]
		scatter!(XX[yy .== c, 1], XX[yy .== c, 2], c= c+1, label="class $(c)", alpha=0.5, m=3)
	end

	push!(plts, plt)
	xnew = XX'
	for (i, f) in enumerate(nnet_smile[1:(end-1)])
		xnew = f(xnew)
		plt = plot(ratio=1, title="Layer $(i): "*L"\mathbf{x}_{%$(i)}", framestyle=:zerolines)
		for (ri, row) in enumerate(row_lines)
			ps_ = f(row')'
			plot!(ps_[:, 1], ps_[:,2], st=:path, alpha=0.5,  lc=:gray,label="")
			row_lines[ri] = ps_
		end
		for (ci, col) in enumerate(col_lines)
			# ys_ =  range(extrema(X_moon[:, 2]) .+ (-.5, .5)..., 50)
			# ps = [ones(length(ys_))*x ys_]
			ps_ = f(col')'
			plot!(ps_[:, 1], ps_[:,2], st=:path, alpha=0.5,  lc=:gray,label="")
			col_lines[ci] = ps_
		end
		
		for c in [0, 1]
			scatter!(xnew[1, yy .== c], xnew[2, yy .== c], c= c+1, label="class $(c)",  alpha=0.5, m=3)
		end
		push!(plts, plt)
	end
	
	plts
end;

# ╔═╡ 8401435b-2b61-4187-9369-d6256bfe67f4
plot(plts_smile..., layout =(2,3), titlefontsize=10, size =(800,500))

# ╔═╡ 39eb062f-f5e9-46f0-80df-df6eab168ebd
sse_loss(y, ŷ) = 0.5 * sum((y .- ŷ).^2) / length(y);

# ╔═╡ 9b4ae619-2951-4f89-befb-b85411826233
md"""

## Demonstration: regression

"""

# ╔═╡ 5621b8d4-3649-4a3e-ab78-40ed9fd0d865
begin
	gr()
	scatter(x_input, y_output, label="Observations", title="A non-linear regression dataset")	
	plot!(true_f, lw=2, xlim=[-10, 10], framestyle=:default,  lc=:gray, ls=:dash,label="true function", xlabel=L"x", ylabel=L"y")
end

# ╔═╡ a6c97f1e-bde0-427e-b621-799f4857347d
md"""
##
### Create the network
"""

# ╔═╡ 3dd87e40-38b8-4330-be87-c901ffb7121a
TwoColumn(md"""
\
\



#### A NN with the following setup
```math
\Large
1 \Rightarrow 12 \Rightarrow 1
``` 

#### loss: mean squared error

""", html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/nnet-17.png
' width = '350' /></center>")

# ╔═╡ 2805fb07-2931-4a29-acc2-d6c17a5dbd97
let
	gr()
	Random.seed!(111)
	# create a NN with one hidden layer and 12 neurons
	hidden_size = 12
	l1_3 = DenseLayer(1, hidden_size, (x) -> relu(x))
	l2_3 = DenseLayer(hidden_size, 1)
	nnet3(x) = (l2_3 ∘ l1_3)(x)
	layers = [l1_3, l2_3]
	γ = 0.004
	anim = @animate for i in 1 : 2000
		∇gt = Zygote.gradient(() -> sse_loss(nnet3(x_input')[:], y_output[:]), Params(layers))
		for l in layers
			update!(l, ∇gt[l], -γ)
		end
		l = sse_loss(nnet3(x_input')[:], y_output[:])
		scatter(x_input, y_output, label="Observations", framestyle=:origin)	
		plot!(x_input, nnet3(x_input')[:], label="prediction", lw=2, title="With ReLu activation: Iteration: "*L"%$(i)"*"; loss: "*L"%$(round(l;digits=2))")
	end every 100
	gif(anim, fps=5)
end

# ╔═╡ 101e52d3-906e-46cd-a55c-a4a13f4d19d1
begin
	x_input_f = Float32.(x_input)
	y_output_f = Float32.(y_output)
	# x_test_f = 
end;

# ╔═╡ 5d679888-578e-49b1-b4a9-d2616d21badf
md"Choose activation function: $(@bind actfun Select([tanh, relu, sin, logistic, Flux.softplus]))"

# ╔═╡ c94e1c57-1bab-428b-936f-a6c1e2ba8237
begin
	Random.seed!(111)
	# create a NN with one hidden layer and 12 neurons
	hidden_size = 12
	l1_3 = DenseLayer(1, hidden_size, (x) -> actfun(x))
	l2_3 = DenseLayer(hidden_size, 1)
	nnet3(x) = (l2_3 ∘ l1_3)(x)
end

# ╔═╡ 9dd4de90-3417-4f6e-b459-a415f4450d7f
begin
	Random.seed!(123)
	nnet_reg = Chain(Dense(1, hidden_size, actfun), Dense(hidden_size, 1))
end;

# ╔═╡ c94dd5af-85d7-4435-ad85-8c408364df69
let
	gr()
	# layers = [l1_3, l2_3]
	γ = 0.005
	optim = Flux.setup(Descent(γ), nnet_reg)

	anim = @animate for i in 1 : 3000
		# ∇gt = Zygote.gradient(() -> sse_loss(nnet3(x_input')[:], y_output[:]), Params(layers))
		# for l in layers
		# 	update!(l, ∇gt[l], -γ)
		# end
		# l = sse_loss(nnet3(x_input')[:], y_output[:])

		l, grads = Flux.withgradient(nnet_reg) do m
			Flux.mse(m(x_input_f'), y_output_f')
		end
		Flux.update!(optim, nnet_reg, grads[1])
		
		scatter(x_input, y_output, label="Observations", framestyle=:origin)	
		plot!(x_input_f, nnet_reg(x_input_f')[:], label="prediction", lw=2, title="Activation: " * string(actfun)*"; Iteration: "*L"%$(i)"*"; loss: "*L"%$(round(l;digits=2))")
	end every 100
	gif(anim, fps=8)
end

# ╔═╡ e91e3b9e-d130-49d6-b334-f5c99fe39d49
# let
# 	gr()
# 	layers = [l1_3, l2_3]
# 	γ = 0.002
# 	anim = @animate for i in 1 : 3000
# 		∇gt = Zygote.gradient(() -> sse_loss(nnet3(x_input')[:], y_output[:]), Params(layers))
# 		for l in layers
# 			update!(l, ∇gt[l], -γ)
# 		end
# 		l = sse_loss(nnet3(x_input')[:], y_output[:])
# 		scatter(x_input, y_output, label="Observations", framestyle=:origin)	
# 		plot!(x_input, nnet3(x_input')[:], label="prediction", lw=2, title="Activation: " * string(actfun)*"; Iteration: "*L"%$(i)"*"; loss: "*L"%$(round(l;digits=2))")
# 	end every 100
# 	gif(anim, fps=8)
# end

# ╔═╡ bce39bea-830f-4b42-a3c9-46af364dd845
# md"""

# # Demonstrations (Python)

# #### Check `Jupyter notebook`


# """

# ╔═╡ 0b03941d-b3fd-45b1-b0b4-7576004b2676
function soft_max(x)  # the input can be a matrix; apply softmax to each column
	ex = exp.(x .- maximum(x, dims=1))
	ex ./ sum(ex, dims = 1)
end;

# ╔═╡ cdba14b1-28dc-4c43-b51f-895f8fc80143
function cross_en_loss(y, ŷ) # cross entropy for multiple class; add ϵ for numerical issue, log cannot take 0 as input!
	-sum(y .* log.(ŷ .+ eps(eltype(y)))) /size(y)[2]
end;

# ╔═╡ fdd378e4-b341-4e2b-a269-1b8dd1751501
# TwoColumn(md"""
# #### Consider a neural network 

# * #### one hidden layer
#   * two hidden units
# * #### one output unit 

# * ##### biases are skipped here
# """, show_img("CS5914/backprop/mlp_backprop2.svg", w=500)
# )

# ╔═╡ 44c1078d-2c1f-40d3-af18-bdedd8e7427d
md"""
# Backprop for Neueral Network 


"""

# ╔═╡ f9488864-7ebf-40e7-9a1e-4dd0a56acde9
md"""

## Flow graph


"""

# ╔═╡ 03a4b301-fb4e-482a-83b6-6475cc128c41
show_img("CS5914/backprop/mlp_backprop1.svg", w=450)

# ╔═╡ 635ab7b5-f2b4-43ec-8737-c48a13bc03ec
md"""

#### Flow graph representation
* ##### draw all parameters ``\mathbf{w}_1^{(1)}, \mathbf{w}_2^{(1)}, \mathbf{w}^{(2)}`` specifically as nodes
* ##### handy for gradient derivation
"""

# ╔═╡ 3093d6b2-71a8-4445-a999-667949d27462
show_img("CS5914/backprop/mlp_forward.svg", w=600)

# ╔═╡ 8bb64e37-3618-4b11-87a0-ad7f9f2acb3c
md"""

## Flow-graph with matrices
"""

# ╔═╡ d913e595-109b-4f42-9e0d-e6b22300b2cc
show_img("CS5914/backprop/nnet_flow_mat.svg", w=640)

# ╔═╡ f3c7df0d-c314-4ad6-be95-dbdd6a35a2b1
show_img("CS5914/backprop/nnet_flow_mat1.svg", w=680)

# ╔═╡ 703bce35-14ec-4ed3-b370-0d6ec1583a0d
md"""

## Backpropagation 


##### _Initialisation_
"""

# ╔═╡ c3cc6324-cd17-4b78-ab3b-d7c8a8452668
show_img("CS5914/backprop/nnet_back0.png", w=630)

# ╔═╡ a4716c8d-df90-458d-8e36-292ae1e90ca4
md"""

## Backpropagation (cont.)
 


"""

# ╔═╡ 12a39fde-7de5-4274-a840-bda2e5030806
show_img("CS5914/backprop/nnet_back1.png", w=630)

# ╔═╡ e100e657-a193-4f7a-9ba4-d15653878dc4
md"""

## Backpropagation (cont.)
 

##### _matrix mult gate -- exchanger!_
"""

# ╔═╡ b90da4b7-9f63-4d63-8b1e-78e7ce91546f
show_img("CS5914/backprop/nnet_back2.png", w=630)

# ╔═╡ 895092e4-f03a-41ce-8a64-d618f83f4e81
md"""

## Backpropagation (cont.)
 

##### _backprop element-wise activations_
"""

# ╔═╡ f56f982d-79b4-4dd0-9479-3b3d48c76a7f
show_img("CS5914/backprop/nnet_back3.png", w=630)

# ╔═╡ 6a05f3fd-6435-4c9f-b07e-4df25b560881
md"""

## Backpropagation (cont.)
 

#### _backprop element-wise activations_

#### `ReLu` activation

$$\Large \texttt{relu}(z) = \text{max}(0, z)$$

#### The backprop works as a switch
  * if the input ``z>0``, the gradient is allowed to flow back (otherwise, 0)

#### if we group $\mathbf{a}^{(1)} = \begin{bmatrix}a_1^{(1)}\\ a_2^{(1)}\end{bmatrix}$, and $\mathbf{z}^{(1)} = \begin{bmatrix}z_1^{(1)}\\ z_2^{(1)}\end{bmatrix}$, 

* ##### the backprop is


$$\large\bar{\mathbf{z}}^{(1)} = {I}.(\mathbf{z}^{(1)} .> 0) \odot \bar{\mathbf{a}}^{(1)}$$
"""

# ╔═╡ 74591882-db43-40c2-8e10-3e154559b882
show_img("CS5914/backprop/ele_act_relu.png", w=630)

# ╔═╡ 1bc9c2bb-d3ae-4072-919d-0e8475f8fa6a
md"""

## Backpropagation (cont.)
 

##### _matrix mult gate -- exchanger!_
"""

# ╔═╡ 025ae1b6-a9ee-4f93-bfb4-53791db8b420
show_img("CS5914/backprop/nnet_back5.png", w=630)

# ╔═╡ 0a4ef24a-2da9-4594-9238-64aa52df9c20
md"""

## Show me the code -- forward pass

"""

# ╔═╡ 90ae5b87-0eef-40a2-8d24-27d6ac9288b0
show_img("CS5914/backprop/nnet_flow_mat1.svg", w=680)

# ╔═╡ 2201093c-da89-4ec5-bc72-015aa31dce46
md"""

```julia
## forward pass
z₁ = W₁ * x # first hidden layer
a₁ = relu.(z₁) # ReLu activations (note the "." for element-wise operation)
z₂ = W₂ * a₁ # output layer
l = 0.5 * (y - z₂)^2 # loss
```

"""

# ╔═╡ d468416b-b0b6-44af-99af-b48c63df1e18
md"""

## Show me the code -- backward pass

"""

# ╔═╡ 4de5da8b-90fd-4551-b840-be25311f9f56
show_img("CS5914/backprop/nnet_back5.png", w=680)

# ╔═╡ d93488ea-0e11-4eba-bda6-3a6097720c9f
md"""

```julia
## forward pass
z₁ = W₁ * x # first hidden layer
a₁ = relu.(z₁) # ReLu activations (note the "." for element-wise operation)
z₂ = W₂ * a₁ # output layer
l = 0.5 * (y - z₂)^2 # loss
```

"""

# ╔═╡ e32a58aa-a6c4-4feb-a259-190c6ff1e3b9
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

# ╔═╡ 2dd671aa-b417-48be-a33e-5977eb68f304
md"""

## With biases `b`

#### Remember for `+`, it backprop as gradient copier
"""

# ╔═╡ 7c96a599-58e9-4097-a3be-794a4d5e469e
md"""

```julia
## forward pass
z₁ = W₁ * x + b₁ # first hidden layer with bias
a₁ = relu.(z₁) # ReLu activations (note the "." for element-wise operation)
z₂ = W₂ * a₁ + b₂ # output layer
l = 0.5 * (y - z₂)^2 # loss
```

"""

# ╔═╡ fa5b2fc4-0607-4c63-9c92-7be1e41ecb63
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

# ╔═╡ bb72ca19-6b92-4894-be41-541f0c03f153
md"""

# Non-stable gradients

## An example

#### An neural network with 8 hidden layer
* ##### `Sigmoid` activation (or logistic)
* ##### 8 neurons each layer


"""

# ╔═╡ 24180616-05e6-4d7a-8a2f-85a4f224e9d2
rand_uniform(b = 1.0) = (x...) -> Flux.rand(x...) * (2*b) .- b;

# ╔═╡ 6d4fe85f-9749-42ef-a5fd-a690238ff57e
md"""
## Vanishing gradient problem -- example

#### `Sigmoid` activation

* ##### Note the gradients getting smaller; eventually vanishes for earlier layers
"""

# ╔═╡ 25a4afb0-025e-4ce7-a720-74f6e59271ad
rand_u(u= 1.0f0) = (x...) -> Flux.rand32(x...) * u;

# ╔═╡ 6fb4815e-8e98-4fc5-b386-758fd5f77ed4
md"""
#### `Tanh` does not work well either!
"""

# ╔═╡ 80f45a4f-6a87-43d4-b46a-8f38b1af0db3
# md"""

# ## Why gradients _vanish_?
# """

# ╔═╡ d719f785-3189-4bad-bc4b-15993bd04770
# md"""

# ##### Forward pass with a ``K``--layer--network
# """

# ╔═╡ f017fb27-c8d7-4fe7-97ed-468ab3e6dad1
# show_img("CS5914/backprop/multi_feedforward_net1.svg", w=750)

# ╔═╡ fd1e3794-1e74-44cb-aa0a-b9a6e2e4e824
md"""

## Why gradients _vanish_?
"""

# ╔═╡ c20394c6-f09d-4c1e-9f09-49b2b7018c47
md"""

##### Backward pass with a ``K``--layer--network


"""

# ╔═╡ f93c1e9a-9f52-4850-aed5-013de4f28811
show_img("CS5914/backprop/multi_backward_net2.svg", w=750)

# ╔═╡ 5922b21b-9814-4c8e-94fa-4717b4910ea4
md"""

#### *Backpropagation*: multiplies *all the local gradients*

```math
\begin{align}
\frac{\partial \ell}{\partial \mathbf{W}^{(k)}} &= \frac{\partial \ell}{\partial \mathbf{z}^{(K)}} \frac{\partial \mathbf{z}^{(K)}}{\partial \mathbf{a}^{(K-1)}} \frac{\partial \mathbf{a}^{(K-1)}}{\partial \mathbf{z}^{(K-1)}}  \frac{\partial \mathbf{z}^{(K-1)}}{\partial \mathbf{a}^{(K-2)}} \frac{\partial \mathbf{a}^{(K-2)}}{\partial \mathbf{z}^{(K-2)}}\ldots  \frac{\partial \mathbf{z}^{(k+1)}}{\partial \mathbf{a}^{(k)}} \frac{\partial \mathbf{a}^{(k)}}{\partial \mathbf{z}^{(k)}}\frac{\partial \mathbf{z}^{(k)}}{\partial \mathbf{W}^{(k)}}\\
\end{align}
```


* ##### if any local gradient is zero (or even between (0, 1))
  * the gradients will **vanish** eventually
  * if ``0 \leq a  \leq 1``, ``a^{K} \rightarrow 0`` 

* ##### if all local gradients are too large (or $>1$)
  * the gradients will **explode** eventually (if infinite deep)
  * if ``|a| > 1``, ``a^{K} \rightarrow \infty`` 
"""

# ╔═╡ 72c4261a-0a7a-42ac-b797-100e249e5485
md"""

## Why gradients _vanish_ ?



"""

# ╔═╡ deec453b-c4f0-40b4-ab15-9046cbe0bd59
show_img("CS5914/backprop/multi_backward_net2.svg", w=750)

# ╔═╡ afe2f096-0f49-4492-8341-364ac707bdf8
md"""

#### *Backpropagation*: multiplies *all the local gradients*

```math
\begin{align}
\frac{\partial \ell}{\partial \mathbf{W}^{(k)}} &= \frac{\partial \ell}{\partial \mathbf{z}^{(K)}} \boxed{\frac{\partial \mathbf{z}^{(K)}}{\partial \mathbf{a}^{(K-1)}} \frac{\partial \mathbf{a}^{(K-1)}}{\partial \mathbf{z}^{(K-1)}}}  \,\boxed{\frac{\partial \mathbf{z}^{(K-1)}}{\partial \mathbf{a}^{(K-2)}} \frac{\partial \mathbf{a}^{(K-2)}}{\partial \mathbf{z}^{(K-2)}}}\ldots  \boxed{\frac{\partial \mathbf{z}^{(k+1)}}{\partial \mathbf{a}^{(k)}} \color{red}{\frac{\partial \mathbf{a}^{(k)}}{\partial \mathbf{z}^{(k)}}}}\frac{\partial \mathbf{z}^{(k)}}{\partial \mathbf{W}^{(k)}}\\
\end{align}
```

#### Let's inspect the activation's backward pass

```math
{\color{red}{\frac{\partial \mathbf{a}^{(k)}}{\partial \mathbf{z}^{(k)}}} = \text{act}'(\mathbf{z}^{(k)}) }\quad \# \text{abuse of notation}
```


* ##### ``\text{act}'(\mathbf{z}^{(k)})``: the derivative of the activation function
"""

# ╔═╡ a1326f81-37b3-4c86-b012-4b0a80b51562
aside(tip(md"""

To be more formal, the local gradient should be a diagonal matrix (known as Jacobian matrix):

```math
\small
\begin{align}
&\frac{\partial \mathbf{a}^{(k)}}{\partial \mathbf{z}^{(k)}} = \texttt{diag}(\text{act}'(\mathbf{z}^{(k)}))\\
&=\begin{bmatrix}\text{act}'(z^{(k)}_1) & & &&\\
& \text{act}'(z^{(k)}_2) &&&\\
&&&& \\
&&&\text{act}'(z^{(k)}_m)&
\end{bmatrix}
\end{align}
```


"""))

# ╔═╡ bacb7cc0-506d-4bb2-a87b-94d0fddc03eb
md"""

## Why gradients _vanish_ ?


#### `Sigmoid` activation as an example


```math
\color{red}
\frac{\partial \mathbf{a}^{(k)}}{\partial \mathbf{z}^{(k)}} = \texttt{sigmoid}'(\mathbf{z}^{(k)}) 
```


* ``\texttt{sigmoid}'(z)``: the gradient of the activation function
"""

# ╔═╡ 6447f211-1313-439c-a33b-3efdbd54ade0
md"Add saturate area: $(@bind add_sat CheckBox(default=false))"

# ╔═╡ ae07651e-6bad-4005-80d9-f48d2b3ae036
TwoColumn(
	let
		x0 =6
	plt = plot(-10:0.1:10, sigmoid, lw=2, label=L"\texttt{Sigmoid}", title=L"\texttt{act}(z):\,\texttt{Sigmoid}",xlabel=L"z", size=(350,250), xlim =[-10,10])
		if add_sat
	vspan!([-11, -x0],  alpha=0.5, c=:gray, label="outputs saturate!")
	vspan!([x0, 11], alpha=0.5, c=:gray, label="", xlim =[-10,10])		
		end
	plt
	end, 
let
	
plt = plot(-10:0.1:10, (x) -> Flux.gradient(sigmoid, x)[1], lw=2,lc=2, label=L"\texttt{Sigmoid}'",title=L"\texttt{act}'(z):\, \texttt{Sigmoid}"*"'s grad",xlabel=L"z", size=(350,250), xlim =[-10,10])
x0 = 6 
	if add_sat
	vspan!([-11, -x0],  alpha=0.5, c=:gray, label="grad vanish!")
	vspan!([x0, 11], alpha=0.5, c=:gray, label="", xlim =[-10,10])		
		end
	plt
end)

# ╔═╡ c5929f4b-4f03-40f1-a249-2d19f70909ab
md"""

## Solution -- do not use `Sigmoid`


#### Use those *non-saturating activation functions*
* ##### *e.g.* `ReLu`, `GeLu`, etc
"""

# ╔═╡ aad4c4a1-6eef-4668-ad6e-468c28a6b97b
TwoColumn(plot(-10:0.1:10, relu, lw=2, label=L"\texttt{ReLu}", title=L"\texttt{act}(z):\, \texttt{ReLu}",xlabel=L"z", size=(350,250)), 
	
	
	begin
	plot(-10:0.1:10, (x) -> Flux.gradient(relu, x)[1], lw=2,lc=1, label=L"\texttt{ReLu}'",title=L"\texttt{act}'(z):\, \texttt{ReLu}"*"'s grad",xlabel=L"z", size=(350,250))
plot!((x) -> Flux.gradient(sigmoid, x)[1], lw=2,lc=2, label=L"\texttt{Sigmoid}'",xlabel=L"z", size=(350,250))
	end
)

# ╔═╡ 65db5ff1-5505-4b94-bdc8-d60a12e3e35c
md"""
## Other solutions


#### There are a wide range of remedies and hacks

* ##### Better initialisation:  `kaiming_uniform_`, `xavier_uniform_` etc.



* ##### Batch normalisation




* ##### Residual layer or skip connection 
"""

# ╔═╡ 4bc50945-6002-444b-bf1a-afad4d529a30
md"""

# Appendix

"""

# ╔═╡ c21bb9d9-dcc0-4879-913b-f9471d606e7b
md"""

## Image sources
[^1]: https://miro.medium.com/max/1400/1*ReYpdIZ3ZSAPb2W8cJpkBg.jpeg

[^2]: Machine Learning: A probability perspective; Kevin Murphy 2012

[^3]: https://d33wubrfki0l68.cloudfront.net/8850c924730b56bbbe7955fd6593fd628249ecff/275c5/images/multiclass-classification.png
"""

# ╔═╡ c5beb4c7-9135-4706-b4af-688d8f088e21
md"""

## Data

"""

# ╔═╡ 7fb8a3a4-607e-497c-a7a7-b115b5e161c0
begin
	Random.seed!(123)
	n_= 30
	D1 = [0 0] .+ randn(n_,2)
	# D1 = [D1; [-5 -5] .+ randn(n_,2)]
	D2_1 = [5 5] .+ randn(n_,2)
	D2_2 = [-5 -5] .+ randn(n_,2)
	D2 = [D2_1; D2_2]
	D = [D1; D2]
	targets_ = [repeat(["class 1"], n_); repeat(["class 2"], n_*2)]
	targets = [zeros(n_); ones(n_*2)]
	df_class = DataFrame(x₁ = D[:, 1], x₂ = D[:, 2], y=targets_)
end;

# ╔═╡ fd7e3596-f567-4fca-a28b-883ada2122f9
TwoColumn(md"""
\


* #### Not separable by _one linear_  boundary
\


* #### In other words, a single neuron is clearly not enough


""", let

	θs = range(π/4, π/4 + 2π, 20)
	r = 5.0
	ws = r * [cos.(θs) sin.(θs)]

	anim = @animate for w in eachrow(ws)
		@df df_class scatter(:x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", ylabel=L"x_2", title="Non-linear separable data with single neuron", framestyle=:origin, ratio=1, size=(350,350), titlefontsize=8)
		plot!(-7:0.1:7, -7:0.1:7, (x,y)-> logistic(dot([x, y], w)), c=:jet, st=:contour, alpha=0.5, colorbar=false)
	end
	gif(anim, fps = 1)
end)

# ╔═╡ cf29359e-1a43-46f6-a272-61d19d926f86
p_nl_cls = let
	@df df_class scatter(:x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", ylabel=L"x_2", title="Non-linear separable data")

	w1 = 0.5 * [0, 5, 5]
	# w2 = [-23, -5, -5]

	plot!(-7:0.1:7, -7:0.1:7, (x,y)-> logistic(dot([1, x, y], w1)), c=:jet, st=:contour, alpha=0.5)
end;

# ╔═╡ e92f6642-bc77-423c-8071-f2f82b738da1
let
	pl1 = @df df_class scatter(:x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", ylabel=L"x_2", title="One neuron", alpha=0.1, label="", framestyle=:origin)

	scatter!(D2_1[:, 1], D2_1[:, 2], c=2, label="")
	scatter!(D1[:, 1], D1[:, 2], c=1, label="")


	pl2 = @df df_class scatter(:x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", ylabel=L"x_2", title="Another neuron", alpha=0.1, label="",framestyle=:origin)

	scatter!(D2_2[:, 1], D2_2[:, 2], c=2, label="")
	scatter!(D1[:, 1], D1[:, 2], c=1, label="")


	w1 = [-23, 5, 5]
	w2 = [-23, -5, -5]

	plot!(pl1, -7:0.1:7, -7:0.1:7, (x,y)-> logistic(dot([1, x, y], w1)), c=:jet, st=:contour)


	plot!(pl2, -7:0.1:7, -7:0.1:7, (x,y)-> logistic(dot([1, x, y], w2)), c=:jet, st=:contour)

	plot(pl1, pl2, size=(900,400))
end

# ╔═╡ ae6953c9-c202-43bd-b563-7a982179e053
let
	gr()
	@df df_class scatter(:x₁, :x₂, group=:y, legend=:right, xlabel=L"x_1", ylabel=L"x_2", title="Non-linearly separable data", ratio=1, size=(400,400), framestyle=:origin)
end

# ╔═╡ 87e4e6b7-881e-4430-b731-5a1990b5d583
losses1 = let
	γ = 0.5 # learning rate
	iters = 2000
	losses = zeros(iters)
	for i in 1: iters
		# ∇gt = Zygote.gradient(() -> cross_entropy_loss(targets[:], neural_net(D')[:]), Params([l1, l2]))
		∇gt = Zygote.gradient(() -> Flux.logitbinarycrossentropy(neural_net(D'), targets'), Params([l1, l2]))
		update!(l1, ∇gt[l1], -γ)
		update!(l2, ∇gt[l2], -γ)
		losses[i] = Flux.logitbinarycrossentropy(neural_net(D'), targets')
		if i % 1 == 0
			ŷ = neural_net(D')[:] .> 0.5
			@info "Iteration, accuracy: ", i , accuracy(targets[:], ŷ)*100
		end
	end
	losses
end;

# ╔═╡ ff665805-7e65-4fcb-bc0a-0ab323b595f9
let
	gr()
	plot(losses1, label="Loss", xlabel="Epochs", ylabel="Loss", title="Learning: loss vs epochs")
end

# ╔═╡ c1ac246e-a8b3-4eae-adb6-c8fe12e039f2
let
	plotly()
	scatter(D1[:, 1], D1[:, 2], - 0.01 .+ zeros(size(D1)[1]), ms=3, label="class 1")
	scatter!(D2[:, 1], D2[:, 2], -0.01 .+  zeros(size(D2)[1]), ms=3, label="class 2")
	plot!(-6:0.2:6, -6:0.2:6, (x,y) -> l1([x, y])[1], st=:surface, c=:jet, alpha=0.8, colorbar=false, xlabel="x1", ylabel="x2", zlabel="a(x)")
	plot!(-6:0.2:6, -6:0.2:6, (x,y) -> l1([x, y])[2], st=:surface, alpha=0.8,c=:jet, title="Hidden layer's functions")
end

# ╔═╡ 21abc2a8-4270-4e47-9d1b-059d31382c00
let
	gr()
	hidden_output = l1(D')
	scatter(hidden_output[1, targets .== 0], hidden_output[2,targets .==0], legend=:outerright, label="class 1")
	scatter!(hidden_output[1, targets .== 1], hidden_output[2,targets .==1], label="class 2", xlabel=L"a_1", ylabel=L"a_2", size=(500,400), title="The hidden layer's output")
end

# ╔═╡ 150f3914-d493-4ece-b661-03abc8c28de9
begin
	function logistic_loss(w, X, y)
		σ = logistic.(X * w)
		# deal with boundary cases such as σ = 0 or 1, log(0) gracefully
		# sum(y .* log.(σ) + (1 .- y).* log.(1 .- σ))
		# rather you should use xlogy and xlog1py
		-sum(xlogy.(y, σ) + xlog1py.(1 .-y, -σ))
	end
end

# ╔═╡ 2f4a1f66-2293-44c9-b75f-f212c1d522fb
function ∇logistic_loss(w, X, y)
	σ = logistic.(X * w)
	X' * (σ - y)
end

# ╔═╡ 6e52567f-4d86-47f4-a228-13c0ff6910ce
D₂, targets_D₂, targets_D₂_=let
	Random.seed!(123)
	D_class_1 = rand(MvNormal(zeros(2), Matrix([1 -0.8; -0.8 1.0])), 30)' .+2
	D_class_2 = rand(MvNormal(zeros(2), Matrix([1 -0.8; -0.8 1.0])), 30)' .-2
	data₂ = [D_class_1; D_class_2]
	D₂ = [ones(size(data₂)[1]) data₂]
	targets_D₂ = [ones(size(D_class_1)[1]); zeros(size(D_class_2)[1])]
	targets_D₂_ = [ones(size(D_class_1)[1]); -1 *ones(size(D_class_2)[1])]
	D₂, targets_D₂,targets_D₂_
end

# ╔═╡ 5c102abf-dca7-48e5-a233-9558559e3fb4
losses_logitstic, wws=let
	# a very bad starting point: completely wrong prediction function
	ww = [0, -5, -5]
	γ = 0.05
	iters = 30
	losses = zeros(iters+1)
	wws = Matrix(undef, 3, iters+1)
	losses[1] = logistic_loss(ww, D₂, targets_D₂)
	wws[:, 1] = ww 
	for i in 1:iters
		# gw = ∇logistic_loss(ww, D₂, targets_D₂)
		loss, gw = Flux.withgradient(ww) do w
			zs = D₂ * w 
			Flux.logitbinarycrossentropy(zs', targets_D₂'; agg = sum)
		end
		if i > 10
			HH = Zygote.hessian(ww) do w
				zs = D₂ * w
				Flux.logitbinarycrossentropy(zs', targets_D₂'; agg = sum)
			end
			d = HH \ gw[1]
			ww = ww - d
		else
			ww = ww - γ * gw[1]
		end
		
		wws[:, i+1] = ww 
		losses[i+1] = logistic_loss(ww, D₂, targets_D₂)
	end
	losses, wws
end;

# ╔═╡ ed15aaea-712b-4dcf-89d4-224e81549135
let
	gr()
	anim = @animate for t in [1:10; 11:2:size(wws)[2];]
		plt1 = plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], st=:scatter, label=L"y^{(i)} = 1", xlabel=L"x_1", ylabel=L"x_2", c=2, size=(350,350))
		plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], st=:scatter, c=1, framestyle=:origin, label=L"y^{(i)} = 0", xlim=[-8, 8], ylim=[-6, 6])
		w₀, w₁, w₂ = (wws[:, t])
		# if w₂ != 0
		# 	plot!(-5:1:5, (x) -> -w₀ / w₂ - w₁/w₂ * x, lc=:gray, lw=2, title="Iteration: "*L"%$(t);"*" loss: " *L"%$(round(losses_logitstic[t]; digits=1))", label="decision boundary")
		# end

		plot!(-6:0.5:6, -6:0.5:6, (x, y) -> logistic(w₀ + w₁ * x + w₂ * y), st=:contour, fill=false, c=:jet, title="Iteration: "*L"%$(t);"*" loss: " *L"%$(round(losses_logitstic[t]; digits=1))")

		plt2 = plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)), st=:scatter, label="class 1", c=2, size=(350,350))
		plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], zeros(sum(targets_D₂ .== 0)), st=:scatter, framestyle=:origin, label="class 2", xlim=[-8, 8], legend=:topleft, c=1)
		w₀, w₁, w₂ = (wws[:, t])
		plot!(-5:0.1:5, -5:0.1:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.5, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="Iteration: "*L"%$(t);"*" loss: " *L"%$(round(losses_logitstic[t]; digits=1))", ratio=1)
		plot(plt1, plt2, layout=(1,2), size =(800, 400))
	end
	gif(anim, fps=3)
end

# ╔═╡ 48847533-ca98-43ce-be04-1636728becc4
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

# ╔═╡ f119f829-51c9-4c15-b0a2-6bbd1db5a428
let
	plotly()
	w₀ = 0
	w₁, w₂ = wv_[1], wv_[2]
	p1 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> w₀+ w₁* x + w₂ * y, st=:surface, c=:jet, colorbar=false, alpha=0.8, framestyle=:zerolines, xlabel="x₁", ylabel="x₂", title="h(x) = wᵀ x")
	arrow3d!([0], [0], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
	p2 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.8, zlim=[-0.1, 1.1],  xlabel="x₁", ylabel="x₂", title="σ(wᵀx)")

	arrow3d!([0], [0], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
	plot(p1, p2)
end

# ╔═╡ 3b4ecf3e-f652-4337-b45d-17e9b6f0ac58
begin
	# plotly()
	function gen_ball_data(n_obs=1000, n_dims=10; seed= 123)
		Random.seed!(seed)
		radius = sqrt(n_dims)
		points = randn(n_dims, n_obs)
		sphere = radius * points ./ sqrt.(sum(points.^2, dims=1))
	
		# Xs = sphere * 
		Xs = sphere .* rand(1, n_obs) .^ (1/ n_dims)
		adjustment = 1/ std(Xs)
		radius *= adjustment
		Xs *= adjustment
		ys = abs.(sum(Xs, dims=1)) .> radius * 0.5
		return Xs .|> Float32, ys .|> Float32
	end
end

# ╔═╡ 83b7f44e-85c0-4d7f-b57f-db0ef24dbda4
X, y= gen_ball_data(100)

# ╔═╡ 8e7b34c2-c4f1-4175-bbca-02db06165a94
function build_nnet(n_input=2, n_output=1, n_hidden_units = 100, n_layers =5; act_fun= relu, use_bn=false, init_method = Flux.glorot_uniform)
	if n_hidden_units isa Vector
		n_units = n_hidden_units
		@assert length(n_units) == n_layers
	else
		n_units = repeat([n_hidden_units], n_layers)
	end
	layers = []
	if use_bn
		push!(layers, Dense(n_input, n_units[1], identity; init=init_method))
		push!(layers, BatchNorm(n_units[1], act_fun))
	else
		push!(layers, Dense(n_input, n_units[1], act_fun; init=init_method))
	end
	for l in 2:n_layers
		# push!(layers, Dense(n_units[l-1], n_units[l], act_fun; init=init_method))
		# if use_bn
		# 	push!(layers, BatchNorm(n_units[l]))
		# end

		if use_bn
			push!(layers, Dense(n_units[l-1], n_units[l], identity; init=init_method))
			push!(layers, BatchNorm(n_units[l], act_fun))
		else
			push!(layers, Dense(n_units[l-1], n_units[l], act_fun; init=init_method))
		end
	end

	push!(layers, Dense(n_units[end], n_output; init=init_method))

	return Chain(layers...)
end

# ╔═╡ 6cd72266-ac9b-4d5d-a02b-336596c6a9f7
function produce_plots(hlist, glist; act_fun = sigmoid, init_method= "")
	plt1 = produce_plots(hlist; grad =false, act_fun = act_fun)
	plt2 = produce_plots(glist; grad =true, act_fun = act_fun)
	plot!(plt1, size=(350,350))
	plot!(plt2, size=(350,350))
	plot(plt1, plt2, size=(700,400), labelfontsize=12, titlefontsize=10)

end

# ╔═╡ 6a8e975c-6e2b-4631-9c54-5048f9c3feb9
function produce_plots(acts; grad =false, act_fun = :sigmoid)
	gr()
	title = "Hidden layer activations with " * L"\texttt{%$(act_fun)}"
	ylabel = "Hidden output"
	if grad
		title = "Gradients with " * L"\texttt{%$(act_fun)}"
		ylabel = "Gradients"
	end
	plt = violin(acts[1], label="Layer $(1)", legend=false, xlabel="Layers", ylabel=ylabel, title=title, titlefontsize =12)
	for (i, a) in enumerate(acts[2:end])
		violin!(a, label="Layer $(i+1)")
	end
	plt
end

# ╔═╡ da2bf7f6-9da5-4693-a806-add291fc4a85
function fb_pass(nnet; X=X, y = y, use_bn = false)
	acts_list =[]
	grad_list = []
	output = X


	for (li, layer) in enumerate(nnet[1:end-1])
		output = layer(output)
		push!(acts_list, vec((output)))
		# acts["a"*string(li)] = vec(output)
	end

	if use_bn
		acts_list = acts_list[2:2:end]
	end
	
	l, grad = Flux.withgradient(nnet) do m
		y_pred = m(X)
		Flux.Losses.logitbinarycrossentropy(y_pred, y)
	end
		
	for (li, gl) in enumerate(grad[1].layers[1:end-1])
		# output = layer(output)
		# # push!(acts, vec(copy(hl)))
		if :weight ∈ keys(gl) 
			push!(grad_list, vec((gl.weight)))
		end
		# acts["a"*string(li)] = vec(output)
	end
	acts_list, grad_list

	# acts_ = vcat(acts[1:end-1]...)
end

# ╔═╡ f2f43d52-f045-49c8-91ac-6e5f35191801
begin
	Random.seed!(4321)
	act_fun_ = sigmoid
	nnet_sigmoid = build_nnet(10, 1, 8, 8; act_fun = act_fun_, use_bn=false, init_method = rand_uniform())
	hlist, glist = fb_pass(nnet_sigmoid)
	# plt = produce_plots(hlist, glist; act_fun = act_fun_)

	# plot!(plt, size=(700,420))
end;

# ╔═╡ cce66409-2d25-4f04-b257-367b31b4c78a
nnet_sigmoid

# ╔═╡ ad325885-449e-43d2-8b15-63f442510326
TwoColumn(
plot!(produce_plots(hlist; act_fun=:Sigmoid), size=(350,350), titlefontsize=11)
,
plot!(produce_plots(glist; act_fun=:Sigmoid, grad=true), size=(350,350), ylim =(-0.015, 0.015),titlefontsize=11)
)

# ╔═╡ 8aea059c-6926-47b2-90a7-b76bae2522c1
TwoColumn(
plot!(produce_plots(hlist; act_fun=:Sigmoid), size=(350,350), titlefontsize=11, title="Hidden layer activations with "*L"\texttt{Sigmoid}")
,
plot!(produce_plots(glist; act_fun=:Sigmoid, grad=true), title ="Vanishing gradient with "*L"\texttt{Sigmoid}", titlefontsize=11, size=(350,350), ylim =(-0.015, 0.015))
)

# ╔═╡ cd8da191-3a04-4cce-954e-b3fb6fe69d9a
let
	Random.seed!(4321)
	act_fun_ = tanh
	nnet_ = build_nnet(10, 1, 10, 8; act_fun = act_fun_, use_bn=false, init_method = rand_u(0.6))
	hlist_, glist = fb_pass(nnet_)
	# plt = produce_plots(hlist, glist; act_fun = act_fun_)

	TwoColumn(
	plot!(produce_plots(hlist_; act_fun=act_fun_), size=(350,350), title="Hidden layer activations with "*L"\texttt{Tanh}")
	,
	plot!(produce_plots(glist; act_fun=act_fun_, grad=true), title="Gradients with "*L"\texttt{Tanh}", ylim = (-0.015, 0.015),size=(350,350))
	)
	# plot!(plt, size=(700,420))
end

# ╔═╡ 39fb6b13-408f-45d0-9314-9f4f47ac642e
let
	Random.seed!(123)
	act_fun_ = relu
	nnet_ = build_nnet(10, 1, 8, 8; act_fun = act_fun_, use_bn=false, init_method = rand_uniform())
	hlist, glist = fb_pass(nnet_)
	# plt = produce_plots(hlist, glist; act_fun = act_fun_)

	TwoColumn(
	plot!(produce_plots(hlist; act_fun=act_fun_), size=(350,350), title="Hidden layer activations with "*L"\texttt{ReLu}")
	,
	plot!(produce_plots(glist; act_fun=act_fun_, grad=true), size=(350,350),title="Gradients with "*L"\texttt{ReLu}")
	)
	# plot!(plt, size=(700,420))
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
MLJBase = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
PlotlyBase = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "a37bed79701e7ecdf72af8bd611a8dd57855c602"

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
git-tree-sha1 = "7e35fca2bdfba44d797c53dfe63a51fabf39bfc0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.4.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "d81ae5489e13bc03567d4fbbb06c546a5e53c857"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.22.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceMetalExt = "Metal"
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
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "29bb0eb6f578a587a49da16564705968667f5fa8"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.2"

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

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "a49f9342fc60c2a2aaa4e0934f06755464fcf438"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.6"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Preferences", "Static"]
git-tree-sha1 = "f3a21d7fc84ba618a779d1ed2fcca2e682865bab"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.7"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.CategoricalArrays]]
deps = ["Compat", "DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "73acb4ed51b1855e1b5ce5c610334363a98d13f1"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "1.0.2"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysArrowExt = "Arrow"
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStatsBaseExt = "StatsBase"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.CategoricalDistributions]]
deps = ["CategoricalArrays", "Distributions", "Missings", "OrderedCollections", "Random", "Reexport", "ScientificTypes"]
git-tree-sha1 = "dc7a65b2a9e16e4f39a94eb706815a5547f1c31e"
uuid = "af321ab8-2d2e-40a6-b165-3d674595d28e"
version = "0.2.0"

    [deps.CategoricalDistributions.extensions]
    UnivariateFiniteDisplayExt = "UnicodePlots"

    [deps.CategoricalDistributions.weakdeps]
    UnicodePlots = "b8865327-cd53-5732-bb35-84acbb429228"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3b704353e517a957323bd3ac70fa7b669b5f48d4"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.72.6"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkCodecCore]]
git-tree-sha1 = "51f4c10ee01bda57371e977931de39ee0f0cdb3e"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.0"

[[deps.ChunkCodecLibZlib]]
deps = ["ChunkCodecCore", "Zlib_jll"]
git-tree-sha1 = "cee8104904c53d39eb94fd06cbe60cb5acde7177"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.0.0"

[[deps.ChunkCodecLibZstd]]
deps = ["ChunkCodecCore", "Zstd_jll"]
git-tree-sha1 = "34d9873079e4cb3d0c62926a225136824677073f"
uuid = "55437552-ac27-4d47-9aa3-63184e8fd398"
version = "1.0.0"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

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
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "a692f5e257d332de1e554e4566a4e5a8a72de2b2"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.4"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d8928e9169ff76c6281f39a659f9bca3a573f24c"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.1"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "f0d05ae68d39d73a96883fc89c61bfe127290472"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.2"

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
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

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

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3bc002af51045ca3b47d2e1787d6ce02e68b943a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.122"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EnzymeCore]]
git-tree-sha1 = "f91e7cb4c17dae77c490b75328f22a226708557c"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.15"
weakdeps = ["Adapt"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"

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
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccc81ba5e42497f4e76553a5545665eed577a663"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.0.0+0"

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "d60eb76f37d7e5a40cc2e7c36974d864b82dc802"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.1"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "173e4d8f14230a7523ae11b9a3fa9edb3e0efd78"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.14.0"
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

[[deps.Flux]]
deps = ["Adapt", "ChainRulesCore", "Compat", "EnzymeCore", "Functors", "LinearAlgebra", "MLCore", "MLDataDevices", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "Setfield", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "d0751ca4c9762d9033534057274235dfef86aaf9"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.16.5"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxCUDAcuDNNExt = ["CUDA", "cuDNN"]
    FluxEnzymeExt = "Enzyme"
    FluxMPIExt = "MPI"
    FluxMPINCCLExt = ["CUDA", "MPI", "NCCL"]

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "ba6ce081425d0afb2bedd00d9884464f764a9225"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.2.2"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "60a0339f28a233601cb74468032b5c302d5067de"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "f52c27dd921390146624f3aab95f4e8614ad6531"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.18"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b0406b866ea9fdbaf1148bc9c0b887e59f9af68"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.18+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "50c11ffab2a3d50192a228c313f05b5b5dc5acb2"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.0+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "7a98c6502f4632dbe9fb1973a4244eaa3324e84d"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e6fe50ae7f23d171f44e311c2960294aaa0beb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.19"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HistogramThresholding]]
deps = ["ImageBase", "LinearAlgebra", "MappedArrays"]
git-tree-sha1 = "7194dfbb2f8d945abdaf68fa9480a965d6661e69"
uuid = "2c695a8d-9458-5d45-9878-1b8a99cf7853"
version = "0.3.1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "8e070b599339d622e9a081d17230d74a5c473293"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.17"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

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
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "57e9ce6cf68d0abf5cb6b3b4abf9bedf05c939c0"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.15"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageBinarization]]
deps = ["HistogramThresholding", "ImageCore", "LinearAlgebra", "Polynomials", "Reexport", "Statistics"]
git-tree-sha1 = "33485b4e40d1df46c806498c73ea32dc17475c59"
uuid = "cbc4b850-ae4b-5111-9e64-df94c024a13d"
version = "0.3.1"

[[deps.ImageContrastAdjustment]]
deps = ["ImageBase", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "eb3d4365a10e3f3ecb3b115e9d12db131d28a386"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.12"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageCorners]]
deps = ["ImageCore", "ImageFiltering", "PrecompileTools", "StaticArrays", "StatsBase"]
git-tree-sha1 = "24c52de051293745a9bad7d73497708954562b79"
uuid = "89d5987c-236e-4e32-acd0-25bd6bd87b70"
version = "0.1.3"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "08b0e6354b21ef5dd5e49026028e41831401aca8"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.17"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "52116260a234af5f69969c5286e6a5f8dc3feab8"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.12"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "8e64ab2f0da7b928c8ae889c514a52741debc1c2"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.4.2"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Bzip2_jll", "FFTW_jll", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Zlib_jll", "Zstd_jll", "libpng_jll", "libwebp_jll", "libzip_jll"]
git-tree-sha1 = "d670e8e3adf0332f57054955422e85a4aec6d0b0"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "7.1.2005+0"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "895205d762ae24a01689f8cc7ad584b55f1fd005"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.7"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "783b70725ed326340adf225be4889906c96b8fd1"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.7"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "8071ca812183ee9acb8e93e8d59c66a7d8742d5c"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.10.0"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "dfde81fafbe5d6516fb864dc79362c5c6b973c82"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.10.2"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageBinarization", "ImageContrastAdjustment", "ImageCore", "ImageCorners", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "a49b96fd4a8d1a9a718dfd9cde34c154fc84fcd5"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.26.2"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "b842cbff3f44804a84fda409745cc8f04c029a20"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.6"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "65d505fa4c0d7072990d659ef3fc086eb6da8208"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.2"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalSets]]
git-tree-sha1 = "5fbb102dcb8b1a858111ae81d56682376130517d"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.11"
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

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["ChunkCodecLibZlib", "ChunkCodecLibZstd", "FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues"]
git-tree-sha1 = "da2e9b4d1abbebdcca0aa68afa0aa272102baad7"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.2"
weakdeps = ["UnPack"]

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "b5a371fcd1d989d844a4354127365611ae1e305f"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.39"
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "ba51324b894edaf1df3ab16e2cc6bc3280a2f1a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.10"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

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
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LearnAPI]]
deps = ["Preferences"]
git-tree-sha1 = "a205f0181e25a22089a62a56b9e537b889540dfb"
uuid = "92ad9a40-7767-427a-9ee6-6e577f1266cb"
version = "2.0.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.11.1+1"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3acf07f130a76f87c041cfb2ff7d7284ca67b072"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.2+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2a7a12fc0a4e7fb773450d17975322aa77142106"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.2+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll"]
git-tree-sha1 = "8e6a74641caf3b84800f2ccd55dc7ab83893c10b"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.17.0+0"

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
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "a9fc7883eb9b5f04f46efb9a540833d1fad974b3"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.173"
weakdeps = ["ChainRulesCore", "ForwardDiff", "NNlib", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    ForwardDiffNNlibExt = ["ForwardDiff", "NNlib"]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLDataDevices]]
deps = ["Adapt", "Functors", "Preferences", "Random", "SciMLPublic"]
git-tree-sha1 = "fb607ece419d500b7fb9a3cdd045906467809695"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.15.0"

    [deps.MLDataDevices.extensions]
    MLDataDevicesAMDGPUExt = "AMDGPU"
    MLDataDevicesCUDAExt = "CUDA"
    MLDataDevicesChainRulesCoreExt = "ChainRulesCore"
    MLDataDevicesChainRulesExt = "ChainRules"
    MLDataDevicesComponentArraysExt = "ComponentArrays"
    MLDataDevicesFillArraysExt = "FillArrays"
    MLDataDevicesGPUArraysExt = "GPUArrays"
    MLDataDevicesMLUtilsExt = "MLUtils"
    MLDataDevicesMetalExt = ["GPUArrays", "Metal"]
    MLDataDevicesOneHotArraysExt = "OneHotArrays"
    MLDataDevicesReactantExt = "Reactant"
    MLDataDevicesRecursiveArrayToolsExt = "RecursiveArrayTools"
    MLDataDevicesReverseDiffExt = "ReverseDiff"
    MLDataDevicesSparseArraysExt = "SparseArrays"
    MLDataDevicesTrackerExt = "Tracker"
    MLDataDevicesZygoteExt = "Zygote"
    MLDataDevicescuDNNExt = ["CUDA", "cuDNN"]
    MLDataDevicesoneAPIExt = ["GPUArrays", "oneAPI"]

    [deps.MLDataDevices.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.MLJBase]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Dates", "DelimitedFiles", "Distributed", "Distributions", "InteractiveUtils", "InvertedIndices", "LearnAPI", "LinearAlgebra", "MLJModelInterface", "Missings", "OrderedCollections", "Parameters", "PrettyTables", "ProgressMeter", "Random", "RecipesBase", "Reexport", "ScientificTypes", "Serialization", "StatisticalMeasuresBase", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "b4dcf3affab14227aab7ec6cdf0fc89eabb9f9d9"
uuid = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
version = "1.10.0"

    [deps.MLJBase.extensions]
    DefaultMeasuresExt = "StatisticalMeasures"

    [deps.MLJBase.weakdeps]
    StatisticalMeasures = "a19d573c-0a75-4610-95b3-7071388c7541"

[[deps.MLJModelInterface]]
deps = ["InteractiveUtils", "REPL", "Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "ccaa3f7938890ee8042cc970ba275115428bd592"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.12.0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "a772d8d1987433538a5c226f79393324b55f7846"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.8"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3cce3511ca2c6f87b19c34ffc623417ed2798cbd"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.10+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "3a8f462a180a9d735e340f4e8d5f364d411da3a4"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.8.1"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.5.20"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "eb6eb10b675236cee09a81da369f94f16d77dc2f"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.31"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ca7e18198a166a1f3eb92a3650d53d94ed8ca8a1"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.22"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "bfe8e84c71972f77e775f75e6d8048ad3fdbe8bc"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.10"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "libpng_jll"]
git-tree-sha1 = "215a6666fee6d6b3a6e75f2cc22cb767e2dd393a"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.5.5+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "386b47442468acfb1add94bf2d85365dea10cbab"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "131dc319e7c58317e8c6d5170440f6bdaee0a959"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.6"

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"

    [deps.Optimisers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "d922b4d80d1e12c658da7785e754f4796cc1d60d"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.36"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "cf181f0b1e6a18dfeb0ee8acc4a9d1672499626c"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.4"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1f7f9bbd5f7a2e5a9f7d96e51c9754454ea7f60b"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.4+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

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

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "28278bb0053da0fd73537be94afd1682cc5a0a83"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.21"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "12ce661880f8e309569074a61d3767e5756a199f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.1"

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

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "dacc8be63916b078b592806acd13bb5e5137d7e9"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.6"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3faff84e6f97a7f18e0dd24373daa229fd358db5"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.73"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "972089912ba299fba87671b025cd0da74f5f54f7"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.0"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "d95ed0324b0799843ac6f7a6a85e65fe4e5173f0"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.5"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "34f7e5d2861083ec7596af8b8c092531facf2192"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "8f528b0851b5b7025032818eb5abbeb8a736f853"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "994cc27cdacca10e68feb291673ec3a76aa2fae9"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

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

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

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

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "5680a9276685d392c87407df00d57c9924d9f11e"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.7.1"
weakdeps = ["RecipesBase"]

    [deps.Rotations.extensions]
    RotationsRecipesBaseExt = "RecipesBase"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "456f610ca2fbd1c14f5fcf31c6bfadc55e7d66e0"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.43"

[[deps.SciMLPublic]]
git-tree-sha1 = "ed647f161e8b3f2973f24979ec074e8d084f1bee"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.0"

[[deps.ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "3ad7f09ae97806e86b3ef28cd50c52ca18bd2d80"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.1.1"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "3e5f165e58b18204aed03158664c4982d691f454"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.5.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "49440414711eddc7227724ae6e570c7d5559a086"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.3.1"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.StatisticalMeasuresBase]]
deps = ["CategoricalArrays", "InteractiveUtils", "MLUtils", "MacroTools", "OrderedCollections", "PrecompileTools", "ScientificTypesBase", "Statistics"]
git-tree-sha1 = "201079ca2c48e5edfaabfc34dec1a5ad2e59476e"
uuid = "c062fc1d-0d66-479b-b6ac-8b44719de4cc"
version = "0.1.3"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "89f86d9376acd18a1a4fbef66a56335a3a7633b8"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.5.0"

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
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "a136f98cefaf3e2924a66bd75173d1c891ab7453"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.7"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "88cf3587711d9ad0a55722d339a013c4c56c5bbc"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.8"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "a2c37d815bf00575332b7bd0389f771cb7987214"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.2"
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
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "d969183d3d244b6c33796b5ed01ab97328f2db85"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.5"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "98b9352a24cb6a2066f9ababcc6802de9aed8ad8"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.6"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "SplittablesBase", "Tables"]
git-tree-sha1 = "4aa1fdf6c1da74661f6f5d3edfd96648321dade9"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.85"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

    [deps.UnsafeAtomics.weakdeps]
    LLVM = "929cbde3-209d-540e-8aea-75f648917ca0"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "d1d9a935a26c475ebffd54e9c7ad11627c43ea85"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.72"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "e9aeb174f95385de31e70bd15fa066a505ea82b9"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.7"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a29cbf3968d36022198bcc6f23fdfd70f7caf737"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.7.10"

    [deps.Zygote.extensions]
    ZygoteAtomExt = "Atom"
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Atom = "c52e3926-4ff0-5f6e-af25-54175e0327b1"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "434b3de333c75fc446aa0d19fc394edafd07ab08"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.7"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.libzip_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "OpenSSL_jll", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "86addc139bca85fdf9e7741e10977c45785727b7"
uuid = "337d8026-41b4-5cde-a456-74a10e5b31d1"
version = "1.11.3+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.5.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "fbf139bce07a534df0e699dbb5f5cc9346f95cc1"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.9.2+0"
"""

# ╔═╡ Cell order:
# ╟─ab7b3ac3-953d-41fb-af70-7e92ab06b8c7
# ╟─08f341b4-1e45-4c1d-990d-e2671ebc70b2
# ╟─28754109-9d20-4c20-a2b1-491dd56dcc2f
# ╟─50103dcc-034e-4c22-953c-1b5fd781c070
# ╟─caa7b56c-6a21-413d-816e-00bb0d16fdcd
# ╟─f72cacd8-f5c2-43a7-b737-71c6b5185880
# ╟─fdab762e-4d32-43f9-8a0d-267fec8e491e
# ╟─78aad856-dc62-430d-ad3a-a32b0d268b0d
# ╟─a9c35a78-c179-45f4-ba25-18e744b30129
# ╟─1e180936-7d39-4895-b526-9a09d7024946
# ╟─16a460dd-a1bb-4304-bd80-cdeaa82c1cac
# ╟─70d6cadd-04e4-4bb5-b38e-8d510a3f2bcf
# ╟─f413ae69-9df1-4447-84b0-8f35b39f55cd
# ╠═5262a879-a236-44ee-8d00-ef8547de579b
# ╟─f119f829-51c9-4c15-b0a2-6bbd1db5a428
# ╟─78d6618f-cd68-434b-93e4-afd36c9e3247
# ╟─0d0b78f8-8b53-4f11-94d9-d436c84ae976
# ╟─a7b797a8-d791-4d10-9ac2-7bd9c0815688
# ╟─bbb08fad-6d96-4735-9ac5-a03f27773af1
# ╟─7159fa02-e801-4a86-8653-aae1dda2e7ac
# ╟─f697e1bb-4e8b-458c-8ea1-d85c403fd913
# ╟─0b945b04-2808-4e81-9c2f-18ef5c891366
# ╟─f510ba7f-8ca5-42ed-807d-67a8710d2e00
# ╟─45788db8-4d37-4b7a-aadb-6c6993bf37e8
# ╟─ed15aaea-712b-4dcf-89d4-224e81549135
# ╟─5c102abf-dca7-48e5-a233-9558559e3fb4
# ╟─f000ec11-e922-4262-b985-3882b2e6c2ee
# ╟─2945e298-1983-4e6b-a903-bead1ac833a8
# ╟─c95275ed-b682-49b8-ae54-9e9e84d401e6
# ╟─1d968144-0bf7-4e08-9600-6fc33cdbdb52
# ╟─7dcff9b2-763b-4424-97d7-48da1e3d2aff
# ╟─6d84e048-d215-4597-b1f9-dbc24e031791
# ╟─95ed0e5c-1d01-49d1-be25-a751f125eb76
# ╟─1fb81ca3-2fa9-4862-bad1-00446515a23a
# ╟─fd7e3596-f567-4fca-a28b-883ada2122f9
# ╟─cf29359e-1a43-46f6-a272-61d19d926f86
# ╟─fe45432f-9224-4e1e-a981-0136db240085
# ╟─e92f6642-bc77-423c-8071-f2f82b738da1
# ╟─6886fe8d-9645-4094-85f8-6b3fb5d409e4
# ╟─f52e374a-2031-4bd9-b58c-16598ffac15a
# ╟─4b2fd42a-e88d-4117-af79-84cfaf902122
# ╟─5bdd174d-9250-4864-80af-1360240ccc93
# ╟─36a00007-0da1-48a4-a225-d400d9c61a37
# ╟─ca75fcaf-9b6d-4d1d-a97e-1b0167b1e2d8
# ╟─f6e77a11-8e1a-44e2-a2d7-cd29939dcc30
# ╟─001a7866-f0c4-4cea-bda3-d5fea2ce56b4
# ╟─2271ac6d-4bb7-4683-8672-c280c3e57fd3
# ╟─350146ba-6e11-41f8-b451-b3c842722de9
# ╟─fcb991b7-e3e1-4f82-8a8b-c1f2a99d6cce
# ╟─8971420b-7687-4a89-8cfa-da2483be8730
# ╟─a7051da0-f28a-4261-b71c-f7450341e1bf
# ╟─500d6f5f-30ee-45a0-9e3c-5f8ba5274e75
# ╟─2ec97a4c-0ddc-4918-91a1-0bf177b4a202
# ╟─b7d84cb8-59ee-41e6-a04e-a4be9f905f95
# ╟─d6023ad2-2467-44a4-8c57-d12024e76536
# ╟─82a2a13c-cdbe-4901-8b2e-86a5d3cd99e5
# ╟─afce64cb-cdca-42a1-8e92-f1e9634af976
# ╟─7841a726-072a-41a3-a10e-4a4a5256aef1
# ╠═ddb1bd96-e316-450a-a7b5-f5561a890266
# ╠═0fe62e04-559d-4140-970b-6baca845be63
# ╟─8d009757-ab79-47e5-8be0-080ef091b302
# ╟─f079377b-d71f-4f70-aece-f04d12cd3127
# ╟─fe63494b-716c-4718-afcd-026291dc7b94
# ╟─e6d3f1d1-21cb-4662-8806-77b88038be7c
# ╟─cb1bc71b-8c8b-4873-8f28-2935c0972b8c
# ╟─4a0195bf-2d5c-478b-b571-edd4b3e532eb
# ╟─13786181-9280-4260-9705-f2aa419e9c26
# ╟─fec26acd-d545-428d-a90d-17a89b2e11c9
# ╟─6d443b0c-becc-459a-b4ee-9a863c7d69b3
# ╟─be7087be-f948-4701-b5c2-00580ca27d9b
# ╟─c98638f4-5bb1-4550-aa51-945c4632b801
# ╟─eef80655-c86f-4947-be7e-ec5b7ff13ad2
# ╟─8aa1f78a-9cf8-4186-b9a0-31b8a00eabfc
# ╟─81c2fa7b-d3b3-4fe3-b2e6-831a10616ed0
# ╟─361d76ed-7952-4c64-830c-27c6273272d7
# ╟─7001dfa7-fb39-40fc-acac-750c7f0d6605
# ╟─743030fd-4462-4563-8767-2d4bfbb17f5b
# ╟─f3ca219a-e2af-45f7-85bf-3c050227a33b
# ╠═19142836-f9c0-45c1-a05b-24d810a06f8f
# ╟─378c520d-881c-4056-a3d0-03fcddf2126c
# ╟─dbddfc50-5d66-45c9-b61a-a554282e38c7
# ╟─0f6a50ea-6165-4e9b-b332-641fed50eea9
# ╟─bad0d1a7-f93a-4af8-a453-4ea8e1385d7c
# ╟─c6de70e6-14f4-4b1a-8097-4e086a7fedde
# ╟─bb317788-8d04-45f8-84f7-7a0226a7df6f
# ╟─a8f32a69-3150-4639-9cd4-546ab8442d90
# ╟─6daa681e-3974-497c-baec-347740b2a29e
# ╟─3db05083-dc9e-49d9-9d11-68562eac5827
# ╟─8674f942-dfae-4891-b3e1-a38b79a2b621
# ╟─1be45d72-0661-4a3a-875f-888d0c12b5e4
# ╟─d6353c42-b07b-483a-a669-857c480385b9
# ╟─0e929a8d-2c43-4e51-b990-8855963b4b4d
# ╟─0fe8412f-4dc8-4698-b434-81763038e768
# ╟─6a4faffd-39e9-4576-a820-3f3abb724055
# ╟─a9427f86-20d2-43ca-b1ca-9724252315da
# ╟─afe83e7a-5e5f-4fe4-aa82-082c7fcdfed3
# ╟─13e0cd61-2f3f-40be-ad4a-7a2fb13661bf
# ╟─a5efe47c-71fb-4b1e-8417-47036b96238d
# ╟─bd6ccedb-00a7-4f9d-8ec9-861d8a8ada11
# ╟─939c8687-3620-4e98-b9f5-0c304312ef2f
# ╟─a617405e-fc19-4032-8642-7f01fb486c74
# ╟─5af83948-c7dd-46b5-a920-4bfb514e4f9c
# ╟─8243b153-0d03-4efa-9fcc-98ab42008826
# ╟─4514df4f-0fee-4e30-8f43-d68a73a56105
# ╟─271874ee-3db9-4abd-8fb4-96f4266cec25
# ╟─65efc486-8797-442f-a66e-32da5b860635
# ╟─b517aaba-e653-491c-8dc2-af86a300b62e
# ╟─9d0e9137-f53b-4329-814e-20e842033f41
# ╟─1287ed6d-8f1e-40f3-8d46-00ab4b267681
# ╟─9a97cc7a-3b3f-41bf-a41a-80dde51a01cc
# ╟─a594cfb0-bd4d-427e-ba00-c665a2c41f54
# ╟─9907c198-8f88-46cf-804d-ba6bde31770b
# ╟─61873a36-2262-4a3b-bb2d-83417c76914a
# ╟─cf69619e-9313-4d14-826b-9a249e1e6b06
# ╟─2c42c70f-05d9-4d6e-8ddd-d225d65a39a6
# ╟─3fde5353-863b-4fd0-8ce2-c6c26d934a14
# ╟─e1a62d74-aaae-491e-b5ef-89e3c1516ca9
# ╟─1fe38da7-09ca-4006-8c18-97f4a5c5ce78
# ╟─8b66daf7-4bf7-4323-be48-a6defefcf022
# ╟─91deb3a5-5a59-4a71-a5f0-8037024e1e97
# ╟─9dea643a-9f72-404b-b57f-496933cc8013
# ╟─5e7d2395-0095-4c3f-bf21-b2c99fa34e4f
# ╟─73b965c6-4f67-4130-ba81-e72b809b00b6
# ╟─b5a49d08-1786-405b-8b6c-17a9ead69cc2
# ╟─f3f3036f-2744-47bc-95a6-6512460023f7
# ╟─f815155f-7d28-473d-8fa9-e0e7f7d81df3
# ╟─ca91ae11-7b2f-4b83-a06c-e66a51ec53a7
# ╟─5159ff24-8746-497b-be07-727f9b7a4d82
# ╟─6a20011e-c89b-46b4-a1f9-243bd401c8fa
# ╟─be0d4a9f-de82-4e40-94a9-5eda37aaeb35
# ╟─85bd6d6d-4d78-45cd-a4b6-3e819dd069c8
# ╟─9c2bb240-0bf9-4add-9244-e1efa678c71f
# ╟─c170521b-db8f-4729-83cc-6da5d4f187e1
# ╟─a4edf393-5bb5-401e-808c-7cd8c00692ad
# ╟─185ab1d0-bdd4-4ba4-a6a6-0829237040aa
# ╟─c598e82d-7a32-47d9-8f4d-fa0145f0b46e
# ╟─a430325b-f800-4f0e-93ad-f9193a9d1b01
# ╟─0d55eb49-d4a5-4c19-8520-dbf8fb062106
# ╟─7d14cc47-d326-46cc-99bc-2b6e45122fbd
# ╟─64baea35-079c-4ed1-b486-0442a555e062
# ╟─59a750de-3594-4222-91b7-9e86feae3e1c
# ╟─0fbac3bc-3397-49f6-9e9d-1e31908f530e
# ╟─96e3e41c-0c50-4e9b-b308-2b307fe54fc8
# ╟─1e2db1d6-e487-4783-ae6c-b230f8566732
# ╟─08ed93ed-304a-46a5-bfc3-c8f45df246ed
# ╟─24d1beda-717c-4f5b-8220-d8671dfc8187
# ╟─495ff258-ba4a-410e-b5e5-30aba7aaa95e
# ╟─1d5accd1-3582-4c76-8680-30e34d27142b
# ╟─0be45c76-6127-48e8-95fa-28827de2f38e
# ╟─5c7370bc-7ba6-4449-b963-448283c80315
# ╟─63937001-c4d9-423f-9491-b4f35342f5a4
# ╟─e259603f-2baa-4242-bc04-791d1c8b168e
# ╟─bedac901-af79-4798-b7b3-c9a730220351
# ╟─eeb9e21e-d891-4b7b-a92a-d0d0d70c1517
# ╟─6a378bb3-a9b9-4d94-9407-b5020fefcf39
# ╟─188654a1-5211-4f87-ae1c-7a64233f8d28
# ╟─cd5c4feb-8da9-41f3-8a71-430eb254fb95
# ╟─14482e14-e64c-4b78-abf4-469f1c404776
# ╟─dcaa1b2d-6f6f-4635-ae39-06a9e9555bce
# ╟─44833991-5520-4046-acc9-c63a9700acf3
# ╟─fd1b3955-85eb-45a4-9798-7d453f1cdd28
# ╟─231e0908-a543-4fd9-80cd-249961a8ddaa
# ╟─231fd696-89f3-4bf8-9466-ddcea3789d21
# ╟─ae6953c9-c202-43bd-b563-7a982179e053
# ╟─41b78734-0fae-4323-9af1-e5e0deda584c
# ╟─8a8c1903-45c9-455f-a02a-fa858ae7bada
# ╟─f892235e-50cc-4d74-b8cc-61768450a9e3
# ╟─87e4e6b7-881e-4430-b731-5a1990b5d583
# ╟─2fbcf298-13e7-4c1e-be10-de8ca82c9385
# ╟─ff665805-7e65-4fcb-bc0a-0ab323b595f9
# ╟─ec5c193a-33de-48c9-a941-ceffc811596f
# ╟─5c5538ce-9e0e-4fba-9dfb-efa37cd43b9b
# ╟─8a654c85-7095-4f91-82d0-2393f90b3aa8
# ╟─e746c069-f18e-4367-8cdb-7ffaac0f9ace
# ╟─b1edd6a5-c00d-4ef5-949f-ce12e9507c58
# ╟─596a1ddd-1aaa-4b02-ab87-be0a6c3fbdfd
# ╟─69f26e0f-20c3-4e1d-95db-0c365f272f6d
# ╟─bff67477-f769-44b2-bd07-21b439eced35
# ╟─3bfccf63-8658-4c52-b2cf-047fcafec8e7
# ╟─47cc554e-e2f8-4e90-84a8-64b628ff4703
# ╟─c1ac246e-a8b3-4eae-adb6-c8fe12e039f2
# ╟─21abc2a8-4270-4e47-9d1b-059d31382c00
# ╟─93ce5d39-5489-415c-9709-8740a016db06
# ╟─b40fd686-c82b-465c-ad8f-bcea54d62aac
# ╟─8067e844-14ed-4fb9-b609-ff598d61cf9e
# ╟─18f85862-5b89-4358-8431-db7fdd900b9b
# ╟─743e1b17-23b9-4a2a-9466-4c0ff03e8888
# ╟─baca7037-f1eb-4b61-b157-9f5e12523894
# ╟─3202e689-edb5-4dd5-b8ac-74a4fd0251e6
# ╟─fbc066bb-90ab-4d9d-8f77-010662290f60
# ╟─9b7a4c44-76c0-4627-a226-e43c78141031
# ╟─79eab834-9a65-4c94-9466-6e9de387dbca
# ╟─93a6a56d-ebbd-4f1b-8d65-fbcbecadeef7
# ╟─c14d5645-4ccb-413c-ad54-ee9d45706405
# ╟─298fbc0b-5e94-447b-8a6c-a0533490e8e1
# ╟─4590f3b7-031d-4039-b125-945e3dcdcea7
# ╟─d12ab2fb-a91d-498a-844f-0148e56110d7
# ╟─575455dc-81bb-4529-b971-b554b5601fce
# ╟─eff06e68-b8b6-4009-b3f2-0c117f03bfe0
# ╟─58d7047d-0173-4297-891b-74dd92e13afb
# ╟─e4f0b81f-12bb-4014-bc54-69cc8701a28a
# ╟─d9f87602-1d22-4967-ac5b-d7bc39cf4931
# ╟─fad7cf61-f270-4a25-b69c-ff37e3da5468
# ╟─d18c622a-5c17-4a32-9bf7-f58d34e7a10a
# ╟─a991e15e-54cb-4698-9481-58cfe9b12f55
# ╟─19958842-85f0-4475-a29b-0b282232d9a4
# ╟─5bc25e85-fb53-47ad-b19b-2697b80927b3
# ╟─30f16cf7-0cd7-42bc-857e-9390b404a1dd
# ╠═569d9b78-1684-4db5-b938-59ff6422345b
# ╠═68376541-4b03-4eb2-b4e2-d1469d8dd07e
# ╟─1d67a460-c117-49f0-b161-7ae28dd43107
# ╠═a42b6041-3d89-409c-94fe-357e41565dbb
# ╟─e4c92595-d3df-4235-a728-aa1bace3dbc3
# ╟─b3599b16-f003-4d36-9092-5140d4eccb27
# ╟─7cebfb96-1650-4450-842f-dab1289bd36d
# ╟─946bceab-4256-42e8-a492-ee4e6153d116
# ╟─8e71ddf1-b888-4b16-b796-5dad56da733c
# ╟─8505d6c3-155d-4644-82ba-dda4c523ad2a
# ╟─d80d54c3-a936-4044-acb0-95181b03ee70
# ╟─d1224150-d1d0-4871-a81a-f6eb1897b420
# ╟─3613fb7c-61aa-4dc6-bc11-05717b78f75c
# ╟─6fe3655b-5b39-4058-90f8-b1f16639c360
# ╟─35c07542-381c-4cd2-b605-c68fbde146e3
# ╟─e5c3e1da-4e2c-46e0-8be2-2a6416958b00
# ╟─3ceef48c-09b0-496e-a65b-34cc01f3f3eb
# ╠═d9bf168e-96cc-4161-aabd-b2c1ce19abcb
# ╠═d03452d1-fc84-411f-86e2-c8fce6cff01e
# ╠═387128c4-1829-4bc8-afdb-ee4d66e3650e
# ╟─8eb4ec4b-521a-4957-8a70-9da05bf0f5af
# ╟─6fcdf294-95b7-40ef-a2b5-b37b46ff3ba0
# ╟─7d5d8ec2-2980-4af0-94f5-31e934416261
# ╟─54fc77fd-f26a-44cd-90d1-85a4a917223f
# ╟─e9b6e233-1e1f-436e-8701-5e1092f27c07
# ╟─1235b084-12bb-4537-a0da-1c3df7e0db3f
# ╟─04ec6ff8-7a99-4acc-a192-3ee93a3f9d60
# ╟─884c04d6-553d-42a2-b831-5163d078c8b7
# ╟─33b4bade-3ca4-41c5-aa9a-44571339160a
# ╟─a38d9dab-575f-4299-87a1-0913f347ee3a
# ╟─ebac79e3-2a84-4513-a9d3-3fecf5c24413
# ╟─4ae14b4a-9f87-49cf-9f70-6f191a00e78c
# ╠═f6554d6f-3856-436d-a33a-3ca8a132877a
# ╟─734aaf86-6146-4fc8-921c-9689aac9e639
# ╟─b913038f-f883-4fec-9bb6-1f427382cf26
# ╟─81458084-db0a-4140-a42b-c536ad8391a9
# ╟─a549be88-f812-40e6-8100-d4702303e956
# ╟─bbffe411-a6d8-40e0-9005-c9a4170b3cec
# ╟─1393ed38-0945-43ee-8a35-3f4ebda1fec0
# ╟─b00bf65f-3f6d-40b2-a95a-22a7ad17af96
# ╟─30810c50-83f7-4fac-8492-0e0018911dfa
# ╟─776ec1b6-02f8-4dfd-b3c4-563552de1d6a
# ╠═d3f5bf51-44ae-4231-afcd-c118f6e599a3
# ╟─a2512481-deac-4529-ae1c-6f8d8eae9613
# ╟─9bcba339-0d81-4126-b1cd-0a779b801e38
# ╟─5b461b02-a6b7-43a0-af64-ceca6e25cc3e
# ╟─25b06875-81c4-4d5e-86c8-f744f4e45c08
# ╟─8401435b-2b61-4187-9369-d6256bfe67f4
# ╟─abf54f94-93b2-4275-b59b-b501b40feb84
# ╟─42f0ef1f-69de-4934-bea8-572c45767c13
# ╟─39eb062f-f5e9-46f0-80df-df6eab168ebd
# ╟─9b4ae619-2951-4f89-befb-b85411826233
# ╟─5621b8d4-3649-4a3e-ab78-40ed9fd0d865
# ╟─a6c97f1e-bde0-427e-b621-799f4857347d
# ╟─3dd87e40-38b8-4330-be87-c901ffb7121a
# ╟─2805fb07-2931-4a29-acc2-d6c17a5dbd97
# ╟─101e52d3-906e-46cd-a55c-a4a13f4d19d1
# ╟─c94e1c57-1bab-428b-936f-a6c1e2ba8237
# ╟─9dd4de90-3417-4f6e-b459-a415f4450d7f
# ╟─5d679888-578e-49b1-b4a9-d2616d21badf
# ╟─c94dd5af-85d7-4435-ad85-8c408364df69
# ╟─e91e3b9e-d130-49d6-b334-f5c99fe39d49
# ╟─bce39bea-830f-4b42-a3c9-46af364dd845
# ╟─0b03941d-b3fd-45b1-b0b4-7576004b2676
# ╟─cdba14b1-28dc-4c43-b51f-895f8fc80143
# ╟─fdd378e4-b341-4e2b-a269-1b8dd1751501
# ╟─44c1078d-2c1f-40d3-af18-bdedd8e7427d
# ╟─f9488864-7ebf-40e7-9a1e-4dd0a56acde9
# ╟─03a4b301-fb4e-482a-83b6-6475cc128c41
# ╟─635ab7b5-f2b4-43ec-8737-c48a13bc03ec
# ╟─3093d6b2-71a8-4445-a999-667949d27462
# ╟─8bb64e37-3618-4b11-87a0-ad7f9f2acb3c
# ╟─d913e595-109b-4f42-9e0d-e6b22300b2cc
# ╟─f3c7df0d-c314-4ad6-be95-dbdd6a35a2b1
# ╟─703bce35-14ec-4ed3-b370-0d6ec1583a0d
# ╟─c3cc6324-cd17-4b78-ab3b-d7c8a8452668
# ╟─a4716c8d-df90-458d-8e36-292ae1e90ca4
# ╟─12a39fde-7de5-4274-a840-bda2e5030806
# ╟─e100e657-a193-4f7a-9ba4-d15653878dc4
# ╟─b90da4b7-9f63-4d63-8b1e-78e7ce91546f
# ╟─895092e4-f03a-41ce-8a64-d618f83f4e81
# ╟─f56f982d-79b4-4dd0-9479-3b3d48c76a7f
# ╟─6a05f3fd-6435-4c9f-b07e-4df25b560881
# ╟─74591882-db43-40c2-8e10-3e154559b882
# ╟─1bc9c2bb-d3ae-4072-919d-0e8475f8fa6a
# ╟─025ae1b6-a9ee-4f93-bfb4-53791db8b420
# ╟─0a4ef24a-2da9-4594-9238-64aa52df9c20
# ╟─90ae5b87-0eef-40a2-8d24-27d6ac9288b0
# ╟─2201093c-da89-4ec5-bc72-015aa31dce46
# ╟─d468416b-b0b6-44af-99af-b48c63df1e18
# ╟─4de5da8b-90fd-4551-b840-be25311f9f56
# ╟─d93488ea-0e11-4eba-bda6-3a6097720c9f
# ╟─e32a58aa-a6c4-4feb-a259-190c6ff1e3b9
# ╟─2dd671aa-b417-48be-a33e-5977eb68f304
# ╟─7c96a599-58e9-4097-a3be-794a4d5e469e
# ╟─fa5b2fc4-0607-4c63-9c92-7be1e41ecb63
# ╟─bb72ca19-6b92-4894-be41-541f0c03f153
# ╟─cce66409-2d25-4f04-b257-367b31b4c78a
# ╟─24180616-05e6-4d7a-8a2f-85a4f224e9d2
# ╟─f2f43d52-f045-49c8-91ac-6e5f35191801
# ╟─6d4fe85f-9749-42ef-a5fd-a690238ff57e
# ╟─ad325885-449e-43d2-8b15-63f442510326
# ╟─25a4afb0-025e-4ce7-a720-74f6e59271ad
# ╟─6fb4815e-8e98-4fc5-b386-758fd5f77ed4
# ╟─cd8da191-3a04-4cce-954e-b3fb6fe69d9a
# ╟─80f45a4f-6a87-43d4-b46a-8f38b1af0db3
# ╟─d719f785-3189-4bad-bc4b-15993bd04770
# ╟─f017fb27-c8d7-4fe7-97ed-468ab3e6dad1
# ╟─fd1e3794-1e74-44cb-aa0a-b9a6e2e4e824
# ╟─c20394c6-f09d-4c1e-9f09-49b2b7018c47
# ╟─f93c1e9a-9f52-4850-aed5-013de4f28811
# ╟─5922b21b-9814-4c8e-94fa-4717b4910ea4
# ╟─72c4261a-0a7a-42ac-b797-100e249e5485
# ╟─deec453b-c4f0-40b4-ab15-9046cbe0bd59
# ╟─afe2f096-0f49-4492-8341-364ac707bdf8
# ╟─a1326f81-37b3-4c86-b012-4b0a80b51562
# ╟─bacb7cc0-506d-4bb2-a87b-94d0fddc03eb
# ╟─6447f211-1313-439c-a33b-3efdbd54ade0
# ╟─ae07651e-6bad-4005-80d9-f48d2b3ae036
# ╟─8aea059c-6926-47b2-90a7-b76bae2522c1
# ╟─c5929f4b-4f03-40f1-a249-2d19f70909ab
# ╟─aad4c4a1-6eef-4668-ad6e-468c28a6b97b
# ╟─39fb6b13-408f-45d0-9314-9f4f47ac642e
# ╟─65db5ff1-5505-4b94-bdc8-d60a12e3e35c
# ╟─4bc50945-6002-444b-bf1a-afad4d529a30
# ╟─c21bb9d9-dcc0-4879-913b-f9471d606e7b
# ╟─c5beb4c7-9135-4706-b4af-688d8f088e21
# ╠═7fb8a3a4-607e-497c-a7a7-b115b5e161c0
# ╠═150f3914-d493-4ece-b661-03abc8c28de9
# ╠═2f4a1f66-2293-44c9-b75f-f212c1d522fb
# ╠═6e52567f-4d86-47f4-a228-13c0ff6910ce
# ╠═48847533-ca98-43ce-be04-1636728becc4
# ╠═3b4ecf3e-f652-4337-b45d-17e9b6f0ac58
# ╠═83b7f44e-85c0-4d7f-b57f-db0ef24dbda4
# ╠═8e7b34c2-c4f1-4175-bbca-02db06165a94
# ╠═6cd72266-ac9b-4d5d-a02b-336596c6a9f7
# ╠═6a8e975c-6e2b-4631-9c54-5048f9c3feb9
# ╠═da2bf7f6-9da5-4693-a806-add291fc4a85
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
