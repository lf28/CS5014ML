### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 17a3ac47-56dd-4901-bb77-90171eebc8c4
begin
	using PlutoTeachingTools
	using PlutoUI
	# using Plots
	using LinearAlgebra
	using StatsPlots
	using LogExpFunctions
	using StatsBase
	using LaTeXStrings
	using Latexify
	using Random
	using HypertextLiteral
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	using ForwardDiff
end

# ╔═╡ a26e482a-f925-48da-99ba-c23ad0a9bed6
using Zygote

# ╔═╡ 29998665-0c8d-4ba4-8232-19bd0de71477
begin
	using DataFrames, CSV
	using MLDatasets
	# using Images
end

# ╔═╡ 253958f1-ae84-4230-bfd1-6023bdffee26
using BenchmarkTools

# ╔═╡ f79bd8ab-894e-4e7b-84eb-cf840baa08e4
using Logging

# ╔═╡ cb72ebe2-cea8-4467-a211-5c3ac7af74a4
TableOfContents()

# ╔═╡ f9023c9e-c529-48a0-b94b-31d822dd4a11
ChooseDisplayMode()

# ╔═╡ d11b231c-3d4d-4fa2-8b1c-f3dd742f8977
md"""

# CS5014 Machine Learning


#### Linear regression 1
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 580b2af0-5a8f-45b3-bc34-ff7bf6c0c221
md"""

## Reading & references

##### Essential reading 


* [_Understanding deep learning_ by _Simon Prince._: Chapter 2](https://github.com/udlbook/udlbook/releases/download/v.1.20/UnderstandingDeepLearning_16_1_24_C.pdf)


##### Suggested reading 
* [_Machine Learning refined_ by Jeremy Watt, Reza Borhani and Aggeos Katsaggelos](https://github.com/jermwatt/machine_learning_refined/blob/gh-pages/sample_chapters/2nd_ed/chapter_5.pdf) Chapter 5.1-5.2



* [_Pattern recognition and Machine Learning_ by _Chris Bishop._: Chapter 3.1-3.2](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)


"""

# ╔═╡ e48c618f-10ec-4ae0-8274-24780417f456
md"""

## Notations


#### Scalars: normal case letters
* ``x,y,\beta,\gamma``


#### Vectors: **Bold-face** smaller case
* ``\mathbf{x},\mathbf{y}, \boldsymbol{\beta}``
* ``\mathbf{x}^\top``: row vector

#### Matrices: **Bold-face** capital case
* ``\mathbf{X},\mathbf{A}, \boldsymbol{\Gamma}``  

#### Tensors: *sans serif* font
* ``\mathsf{X, A, \Gamma}``

"""

# ╔═╡ cf9c3937-3d23-4d47-b329-9ecbe0006a1e
md"""

## Notations


"""

# ╔═╡ ec6242db-e69e-429c-b82c-2c22a9b232f1
TwoColumn(md"""
\

##### Super-index with bracket: ``\large \mathbf{x}^{(i)}``

* ###### index observations/data/rows
* *e.g.* ``y^{(i)}`` the ``i``-th target
* *e.g.* ``\mathbf{x}^{(i)}`` the ``i``-th observation's features
* usually use ``n``: number of observations

""", html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/iindex.svg" width = "400"/></center>""")

# ╔═╡ 2a381c96-57ed-42f0-9cca-fc2b84e017a5
TwoColumn(md"""

##### Sub-index: ``\large \mathbf{x}_j``

* ###### index feature/columns
* *e.g.* ``\mathbf{x}^{(3)}_2``: the second feature of ``i``-th observation
* usually use letter ``m``: number of features 

""", html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/jindex.svg" width = "400"/></center>""")

# ╔═╡ 98951c85-ecc7-45af-b429-dbe11b4aac38
# begin
# 	gr()
# 	gf2 = x -> ForwardDiff.gradient(f2, x)
# 	xylim = π * 1.4
# 	x1_ = range(-xylim, stop =xylim, length=100)
# 	x2_ = range(-xylim, stop =xylim, length=100)
# 	p1_ = plot(x1_, x2_, (a, b) -> f2([a, b]), st=:surface, xlabel ="x", ylabel="y", zlabel="f", colorbar=false, color=:jet)
# 	# p2_= plot(contour(x1_, x2_, (a, b) -> f2([a, b])), colorbar=false)
# 	# plot(p1_, p2_)
# 	# plot(p1_,
# 	plot!(x1_, x2_, (a, b) -> tf2([a, b]), st=:surface, c=:gray, alpha=0.9, zlim = [-2, 9])
# 	scatter!([location[1]], [location[2]], [f2(location)], ms=2)
# 	p1_
# end

# ╔═╡ dfcfd2c0-9f51-48fb-b91e-629b6934dc0f
md"""

# Linear regression


"""

# ╔═╡ 8257d59b-b8f3-44aa-879a-ecaa40c511d6
md"""

## What is regression ?

"""

# ╔═╡ 6073463a-ca24-4ddc-b83f-4c6ff5033b3b
md"""
##
#### Prediction objective

* ###### for some ``\mathbf{x}_{test}``: predict ``h(\mathbf{x}_{test})``
"""

# ╔═╡ ed20d0b0-4e1e-4ec5-92b0-2d4938c249b9
@bind x_test_0 Slider(3.5:0.2:9, default=7, show_value=false)

# ╔═╡ c802c807-971b-426a-be13-a382688de911
begin
	ENV["DATADEPS_ALWAYS_ACCEPT"] = true;
	X_housing = (BostonHousing().features |> Matrix)'
	df_house = MLDatasets.BostonHousing().dataframe
	df_house[!, :target] = (MLDatasets.BostonHousing().targets |> Matrix)[:]
end;

# ╔═╡ 86f09ee8-087e-47ac-a81e-6f8c38566774
# md"""

# ## Regression's objective


# !!! note "Objective"
# 	###### Predict house price ``y_{test}`` based on ``\mathbf{x}_{test}``:
# 	```math
# 	\Large
# 		h(\mathbf{x}_{test}): \text{ regression function}
# 	```

# """

# ╔═╡ e3797d79-8007-4c8b-8f4f-ff2cb4461020
md"""

## _Linear_ regression 
"""

# ╔═╡ 564ff88e-b310-4ac1-85a0-28640ee015bb
md"""

## _Terminologies_
"""

# ╔═╡ fe3e7f8f-e8a0-4a1f-ad9b-a880a00f49aa
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/regression_ano.svg" width = "900"/></center>"""

# ╔═╡ 65f28dfb-981d-4361-b37c-12af3c7995cd
linear_reg_normal_eq(X, y) = (X' * X)^(-1) * X' * y;

# ╔═╡ 6af0c378-fbac-4d22-a046-de37b5c90f20
TwoColumn(md"""
!!! information "Regression"
	###### _Supervised learning_ with _continuous_ targets ``y^{(i)} \in \mathbb{R}``
    * input feature ``\mathbf{x}^{(i)}``
    * target ``y^{(i)}``

##### *Example*: *house price* prediction ``\{\mathbf{x}^{(i)}, y^{(i)}\}`` for ``i=1,2,\ldots, n``

* ``y^{(i)} \in \mathbb{R}``:  house _price_ is continuous
* ``\mathbf{x}^{(i)}``: the average number of rooms""", 
	
	let
	gr()
	@df df_house scatter(:RM, :target, xlabel="room", ylabel="price", label="", title="House price prediction", size=(350,300), alpha=0.8, ms=3)
	x_room = df_house[:, :RM]
	# x_room_sq = x_room.^2
	X_train_room = [ones(length(x_room)) x_room]
	c, b = linear_reg_normal_eq(X_train_room, df_house.target)

	plot!(3.5:0.5:9, (x) -> b* x+ c, lw=3, label="")
end)

# ╔═╡ 774c46c0-8a62-4635-ab56-662267e67511
let
	gr()
	@df df_house scatter(:RM, :target, xlabel="room", ylabel="price", label="", title="House price prediction: regression")
	x_room = df_house[:, :RM]
	# x_room_sq = x_room.^2
	X_train_room = [ones(length(x_room)) x_room]
	c, b= linear_reg_normal_eq(X_train_room, df_house.target)

	plot!(3.5:0.5:9, (x) -> b* x+ c, lw=3, label=L"h(\mathbf{x})", legend=:outerbottom)

	scatter!([x_test_0], [0], c= :gray, markershape=:x,  label=L"\mathbf{x}_{\textit{test}}")
	plot!([x_test_0], [ b* x_test_0+ c], st=:sticks, line=:dash, c=:gray, lw=2, label="", framestyle=:zerolines, xlim =[1.5, 10])
end

# ╔═╡ 75c38735-ba5b-4040-9c1b-77f963c6cccc
TwoColumn(md"""
\
\

!!! note "Linear regression"
	**Linear regression**: the prediction function ``h(\cdot)`` is assumed **linear**

	```math
	\Large
	h(x_{\text{room}}) = w_0 + w_1 x_{\text{room}} 
	```


""", let
	@df df_house scatter(:RM, :target, xlabel="room", ylabel="price", label="", title="Linear regression", size=(350,300))
	x_room = df_house[:, :RM]
	# x_room_sq = x_room.^2
	X_train_room = [ones(length(x_room)) x_room]
	c, b = linear_reg_normal_eq(X_train_room, df_house.target)

	plot!(3.5:0.5:9, (x) -> b* x+ c, lw=3, label=L"h(x) = w_0 + w_1x", legendfontsize=14)
end)

# ╔═╡ e196e010-6e45-486b-957d-b3d5910dc887
md"""
## Multiple linear regression

#### _House_ dataset has 
* ##### *14 predictors*: `room`, `crime rate`, `age`, `dis`, and so on
"""

# ╔═╡ e5c1570d-1620-4e90-afbd-97e9c8151c51
# first(df_house, 5)

# ╔═╡ 6970801f-4d88-4baf-9c06-5feb9bb3ed2f
md"""

## Multiple linear regression



!!! note "Simple linear regression"
	##### *Linear regression* with a single predictor

	```math
	\large
	h(x_{\text{room}}) = w_0 + w_1 x_{\text{room}} 
	```

```math
\Huge
\Downarrow \text{\small 14 predictors: \texttt{room, crime rate, age, ...}
}
```


!!! note "Multiple linear regression"
	##### *Linear regression* with _multiple_ predictors

	```math
	\large
	h(\mathbf{x}) = w_0 + \boxed{w_1 x_{\text{room}} + w_2 x_{\text{crime}}+ w_3 x_{\text{age}} + \ldots+ w_{14} x_{\text{rad}}}
	```
"""

# ╔═╡ c3d82ace-cfdc-44f6-93e2-d9e20a10cdee
md"""

## Multiple linear regression



!!! note "Simple linear regression"
	##### *Linear regression* with a single predictor

	```math
	\large
	h(x_{\text{room}}) = w_0 + w_1 x_{\text{room}} 
	```

```math
\Huge
\Downarrow \text{\small 14 predictors: \texttt{room, crime rate, age, ...}
}
```


!!! note "Multiple linear regression"
	##### *Linear regression* with _multiple_ predictors

	```math
	\large
	\begin{align}
	h(\mathbf{x}) &= w_0 + \boxed{w_1 x_{\text{room}} + w_2 x_{\text{crime}}+ w_3 x_{\text{age}} + \ldots+ w_{14} x_{\text{rad}}} \\
	&=  w_0 + \mathbf{w}^\top\mathbf{x}
	\end{align}
	```

	* where $\large \mathbf{x} = \begin{bmatrix}x_{\text{room}} \\ x_{\text{crime}} \\\vdots \\ x_{\text{rad}}\end{bmatrix}_{14\times 1}$
"""

# ╔═╡ 24eb939b-9568-4cfd-bfe5-0191eada253a
# md"""

# ## Multiple linear regression



# !!! note "Multiple Linear regression"
# 	##### Prediction function ``h(\mathbf{x})``

# 	```math
# 	\large
# 	h(\mathbf{x}) = w_0 + w_1 x_{1} + w_2 x_2 + \ldots + w_m x_m  = \mathbf{w}^\top \mathbf{x}
# 	```

# 	* for convenience, we add 1 dummy predictor one to ``\mathbf{x}``:

# 	```math
# 		\mathbf{x} =\begin{bmatrix}1\\ x_1 \\ x_2 \\ \vdots\\ x_m \end{bmatrix}
# 	```

# 	* then 

# 	```math
# 		\mathbf{w}^\top\mathbf{x} =\begin{bmatrix}w_0 & w_1 & w_2 & \ldots & w_m \end{bmatrix}  \begin{bmatrix}1\\ x_1 \\ x_2 \\ \vdots\\ x_m \end{bmatrix} = w_0 + w_1 x_{1} + w_2 x_2 + \ldots + w_m x_m 
# 	```

#     * we sometimes write ``h(\mathbf{x}; \mathbf{w}) = \mathbf{w}^\top \mathbf{x}`` or ``h_{\mathbf{w}}(\mathbf{x})``

# """

# ╔═╡ 5d96f623-9b30-49a4-913c-6dee65ae0d23
md"""

## Hyperplane ``h(\mathbf{x}) = w_0 + \mathbf{w}^\top \mathbf{x} ``


#### Geometrically, we predict with a _hyperplane_

"""

# ╔═╡ fe533cf8-5ff2-4094-b5bc-08bd43f3f9de
md"""



## A handy notation: _dummy one_ 

!!! note "Linear regression - generalisation"

	##### *Linear regression* with _multiple_ predictors


	```math
	\Large
	\begin{align}
	h(\mathbf{x}) &= w_0 + w_1 x_{1} + w_2 x_2 + \ldots + w_m x_m \\
	&=\begin{bmatrix}w_0 & w_1 & w_2 & \ldots & w_m \end{bmatrix}  \begin{bmatrix}\colorbox{orange}{$1$}\\ x_1 \\ x_2 \\ \vdots\\ x_m \end{bmatrix}\\
	&= \boxed{\mathbf{w}^\top\mathbf{x} }
	\end{align}
	```


* for convenience, we add a ``\textcolor{orange}{\rm dummy \, predictor\, 1}`` to ``\mathbf{x}``:

```math
\large
	\mathbf{x} =\begin{bmatrix}\colorbox{orange}{$1$}\\ x_1 \\ x_2 \\ \vdots\\ x_m \end{bmatrix}
```

* sometimes we write ``h(\mathbf{x}; \mathbf{w}) = \mathbf{w}^\top \mathbf{x}`` or ``h_{\mathbf{w}}(\mathbf{x})``

"""

# ╔═╡ 9d9c7233-9535-48c4-8add-68415217e1dd
md"""

## "Learning" 

#### -- Empirical risk minimization 


#### In many ways, machine "*learning*" is 

* ###### _looking for_ some *good* ``\hat{h}(\mathbf{x})`` from hypothesis set ``\{h_1, h_2, \ldots\}``

* ###### minimise the *empirical loss/risk* 
  * ###### *empirical*: loss of the given _training data_ only **not** the whole population 
"""

# ╔═╡ 492f4ed9-5434-4c5a-8baf-3ecd79345ddc
TwoColumn(html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/mlgoodness.png' width = '250' /></center>", let
	gr()
	Random.seed!(123)
	n = 20
	w0, w1 = 1, 2
	Xs = rand(n) * 2 .+ 1
	ys = (w0 .+ Xs .* w1) .+ randn(n)/2
	Xs = [Xs; 0.25]
	ys = [ys; 3.5]
	plt = plot(Xs, ys, st=:scatter, markersize=4, label="", xlabel="height", ylabel="weight", size=(400,300), framestyle=:semi)
	plot!(0:0.5:3.0, (x) -> w0 + w1 * x , lw=3, label=L"\hat{h}(x)", legend=:outerright)


	for i in 1:15
		w0_, w1_ = randn(2) ./ [3, 1]  + [w0, w1] 
		if i == 15
			plot!(0:0.5:3.0, (x) -> w0_ + w1_ * x , lw=1., label=L"\ldots")
		else
			plot!(0:0.5:3.0, (x) -> w0_ + w1_ * x , lw=1., label=L"h_{%$(i)}(x)")
		end
	end
	plt
end)

# ╔═╡ fc826717-2a28-4b86-a52b-5c133a50c2f9
md"""

## Defining a *good* ``h(\cdot)``



#### *Prediction errors* are

```math
\Large
\begin{align}
\text{error}^{(i)} &= y^{(i)} - h(\mathbf{x}^{(i)}; \mathbf{w})\\
&= y^{(i)} - \hat{y}^{(i)}
\end{align}
```

* ##### ``\hat{y}^{(i)} \triangleq h(\mathbf{x}^{(i)}; \mathbf{w})`` for ``i =1 \ldots n``  
  * notation ``\hat{{\theta}}``: estimate/guess/prediction of ``\theta``
"""

# ╔═╡ 93600d0c-fa7e-4d38-bbab-5adcf54d0c90
let
	gr()
	Random.seed!(123)
	n = 10
	w0, w1 = 1, 1
	Xs = range(-2, 2, n)
	ys = (w0 .+ Xs .* w1) .+ randn(n)/1
	Xs = [Xs; 0.5]
	ys = [ys; 3.5]
	plt = plot(Xs, ys, st=:scatter, markersize=4, alpha=0.5,  label="", xlabel=L"x", ylabel=L"y", ratio=1, title="Prediction error: "*L"y^{(i)} - \hat{y}^{(i)}")
	plot!(-2.9:0.1:2.9, (x) -> w0 + w1 * x , xlim=[-3, 3], lw=2, label=L"h_w(x)", legend=:topleft, framestyle=:axes, legendfontsize=13)
	ŷs = Xs .* w1 .+ w0
	for i in 1:length(Xs)
		plot!([Xs[i], Xs[i]], [ys[i], ŷs[i] ], arrow =:both, lc=:gray, lw=1, label="")
	end

	
	annotate!(Xs[end], 0.5*(ys[end] + ŷs[end]), text(L"y^i - \hat{y}^{(i)}", 15, :black, :top, rotation = -90 ))
	plt
end

# ╔═╡ cec496e3-a249-4516-83d0-1b8c34efa4c4
md"""
## Defining a *good* ``h(\cdot)``


##### One possible empirical loss or (negative **goodness**) is 

> ##### **S**um of **S**quared **E**rrors (MSE)

```math
\Large
\begin{align}
L(\mathbf{w}) &= \frac{1}{2}\sum_{i=1}^n \left (\text{error}^{(i)}\right ) ^2
\\

&=\frac{1}{2}\sum_{i=1}^n (y^{(i)} - h(\mathbf{x}^{(i)}; \mathbf{w}))^2 \\
&=\frac{1}{2}\sum_{i=1}^n (y^{(i)} - \hat{y}^{(i)})^2\;\;\; \Leftarrow \text{SSE}
\end{align}
```

* ``\dfrac{1}{2}`` introduced for mathematical convenience (it will be clear soon)
"""

# ╔═╡ c1a8b895-55ea-4a63-9327-1f961dbc25e1
md"``w_1``: $(@bind w₁ Slider(-1:0.1:2; default = 1.0)), ``w_0``: $(@bind w₀ Slider(-1:0.1:3; default=1))"

# ╔═╡ dead4d31-8ed4-4599-a3f7-ff8b7f02548c
md"""
## Least square estimation

##### Learning: aim to *minimise* the loss (or achieve the best **goodness**)


```math
\Large
\begin{align}
\hat{\mathbf{w}} &\leftarrow \arg\min_{\mathbf{w}}L(\mathbf{w}) \\
&= \arg\min_{\mathbf{w}} \frac{1}{2}\sum_{i=1}^n (y^{(i)} - \hat{y}^{(i)})^2
\end{align}
```


* #### optimisation: _good old calculus!_
"""

# ╔═╡ d70102f1-06c0-4c5b-8dfd-e41c4a455181
md"""

## Some examples


"""

# ╔═╡ b1c58114-6999-4789-aa12-7454aa1e0927
TwoColumn(md"""
* top left -- the **zero** function
  
* bottom left -- **under** estimate
""", md"""
  * top right -- **over** estimate

  * bottom right -- seems **perfect**

""")

# ╔═╡ a02a28d4-d015-4a99-906f-0aed0ada51af
md"""

## Towards matrix notation

##### For the training instances ``\mathbf{x}^{(i)} \in \mathcal{X}_{train}``, the _predictions_ are


* ###### for house 1, with ``\mathbf{x}^{(1)}``, its prediction is
```math
\large
\hat{y}^{(1)} = w_0 + w_1 x_{\text{room}}^{(1)} + w_2 x_{\text{crime}}^{(1)}+ w_3 x_{\text{age}}^{(1)} + \ldots+ w_{14} x_{\text{rad}}^{(1)} 
```

* ###### for house 2, with ``\mathbf{x}^{(2)}``, its prediction is
```math
\large
\hat{y}^{(2)} = w_0 + w_1 x_{\text{room}}^{(2)} + w_2 x_{\text{crime}}^{(2)}+ w_3 x_{\text{age}}^{(2)} + \ldots+ w_{14} x_{\text{rad}}^{(2)}
```

```math
\Large
\vdots
```

* ###### for house ``n``, with ``\mathbf{x}^{(n)}``, its prediction is
```math
\large
\hat{y}^{(n)} = w_0 + w_1 x_{\text{room}}^{(n)} + w_2 x_{\text{crime}}^{(n)}+ w_3 x_{\text{age}}^{(n)} + \ldots+ w_{14} x_{\text{rad}}^{(n)}
```

"""

# ╔═╡ c7aced31-0d39-4a8f-927b-4977156e486e
md"""

## Towards matrix notation

##### For the training instances ``\mathbf{x}^{(i)} \in \mathcal{X}_{train}``, the _predictions_ are

* ##### as a _systems of equations_ (overly _tedious_ to write)



```math
\large
\begin{align}
\hat{y}^{(1)} &= w_0 + w_1 x_{\text{room}}^{(1)} + w_2 x_{\text{crime}}^{(1)}+ w_3 x_{\text{age}}^{(1)} + \ldots+ w_{14} x_{\text{rad}}^{(1)} \\
\hat{y}^{(2)}  &= w_0 + w_1 x_{\text{room}}^{(2)} + w_2 x_{\text{crime}}^{(2)}+ w_3 x_{\text{age}}^{(2)} + \ldots+ w_{14} x_{\text{rad}}^{(2)}\\
\hat{y}^{(3)}  &= w_0 + w_1 x_{\text{room}}^{(3)} + w_2 x_{\text{crime}}^{(3)}+ w_3 x_{\text{age}}^{(3)} + \ldots+ w_{14} x_{\text{rad}}^{(3)}\\
&\vdots\\


\hat{y}^{(n)} &= w_0 + w_1 x_{\text{room}}^{(n)} + w_2 x_{\text{crime}}^{(n)}+ w_3 x_{\text{age}}^{(n)} + \ldots+ w_{14} x_{\text{rad}}^{(n)}

\end{align}
```



"""

# ╔═╡ c4107744-f07e-4b4f-bc04-64a5dc059801
md"""

## Towards matrix notation

##### For the training instances ``\mathbf{x}^{(i)} \in \mathcal{X}_{train}``, the _predictions_ are

* ##### as a _systems of equations_ (overly _tedious_ to write)




```math
\large
\begin{align}
\hat{y}^{(1)} &= w_0 + w_1 x_{\text{room}}^{(1)} + w_2 x_{\text{crime}}^{(1)}+ w_3 x_{\text{age}}^{(1)} + \ldots+ w_{14} x_{\text{rad}}^{(1)} \\
\hat{y}^{(2)}  &= w_0 + w_1 x_{\text{room}}^{(2)} + w_2 x_{\text{crime}}^{(2)}+ w_3 x_{\text{age}}^{(2)} + \ldots+ w_{14} x_{\text{rad}}^{(2)}\\
\hat{y}^{(3)}  &= w_0 + w_1 x_{\text{room}}^{(3)} + w_2 x_{\text{crime}}^{(3)}+ w_3 x_{\text{age}}^{(3)} + \ldots+ w_{14} x_{\text{rad}}^{(3)}\\
&\vdots\\


\hat{y}^{(n)} &= w_0 + w_1 x_{\text{room}}^{(n)} + w_2 x_{\text{crime}}^{(n)}+ w_3 x_{\text{age}}^{(n)} + \ldots+ w_{14} x_{\text{rad}}^{(n)}

\end{align}
```



* ##### by vector notation (slightly better)


```math
\large
\begin{equation}
\begin{cases}
\hat{y}^{(1)} &= \mathbf{w}^\top \mathbf{x}^{(1)}\\
\hat{y}^{(2)} &= \mathbf{w}^\top \mathbf{x}^{(2)}\\
\hat{y}^{(3)} &= \mathbf{w}^\top \mathbf{x}^{(3)}\\
&\vdots\\
\hat{y}^{(n)} &= \mathbf{w}^\top \mathbf{x}^{(n)}
\end{cases} 
\end{equation} 
```



"""

# ╔═╡ ce1a022c-66e3-4f96-b2a3-9caee9ebbcaf
md"""

## Towards matrix notation

##### For the training instances ``\mathbf{x}^{(i)} \in \mathcal{X}_{train}``, the _predictions_ are

* ##### as a _systems of equations_ (overly _tedious_ to write)



```math
\large
\begin{align}
\hat{y}^{(1)} &= w_0 + w_1 x_{\text{room}}^{(1)} + w_2 x_{\text{crime}}^{(1)}+ w_3 x_{\text{age}}^{(1)} + \ldots+ w_{14} x_{\text{rad}}^{(1)} \\
\hat{y}^{(2)}  &= w_0 + w_1 x_{\text{room}}^{(2)} + w_2 x_{\text{crime}}^{(2)}+ w_3 x_{\text{age}}^{(2)} + \ldots+ w_{14} x_{\text{rad}}^{(2)}\\
\hat{y}^{(3)}  &= w_0 + w_1 x_{\text{room}}^{(3)} + w_2 x_{\text{crime}}^{(3)}+ w_3 x_{\text{age}}^{(3)} + \ldots+ w_{14} x_{\text{rad}}^{(3)}\\
&\vdots\\


\hat{y}^{(n)} &= w_0 + w_1 x_{\text{room}}^{(n)} + w_2 x_{\text{crime}}^{(n)}+ w_3 x_{\text{age}}^{(n)} + \ldots+ w_{14} x_{\text{rad}}^{(n)}

\end{align}
```


* ##### by vector notation (slightly better)


```math
\large
\begin{equation}
\begin{cases}
\hat{y}^{(1)} &= \mathbf{w}^\top \mathbf{x}^{(1)}\\
\hat{y}^{(2)} &= \mathbf{w}^\top \mathbf{x}^{(2)}\\
\hat{y}^{(3)} &= \mathbf{w}^\top \mathbf{x}^{(3)}\\
&\vdots\\
\hat{y}^{(n)} &= \mathbf{w}^\top \mathbf{x}^{(n)}
\end{cases} 
\end{equation}  \xRightarrow[]{\mathbf{a}^\top\mathbf{b} = \mathbf{b}^\top\mathbf{a}} \begin{cases}
\hat{y}^{(1)} &= (\mathbf{x}^{(1)})^\top \mathbf{w} \\
\hat{y}^{(2)} &= (\mathbf{x}^{(2)})^\top \mathbf{w}\\
\hat{y}^{(3)} &= (\mathbf{x}^{(3)})^\top \mathbf{w}\\
&\vdots\\
\hat{y}^{(n)} &= (\mathbf{x}^{(n)})^\top \mathbf{w}
\end{cases} 
```



"""

# ╔═╡ 7258e0a7-d805-4d14-93a5-4ce15df30346
md"""

## Towards matrix notation

##### For the training instances ``\mathbf{x}^{(i)} \in \mathcal{X}_{train}``, the _predictions_ are

* ##### as a _systems of equations_ (overly _tedious_ to write)


```math
\large
\begin{align}
\hat{y}^{(1)} &= w_0 + w_1 x_{\text{room}}^{(1)} + w_2 x_{\text{crime}}^{(1)}+ w_3 x_{\text{age}}^{(1)} + \ldots+ w_{14} x_{\text{rad}}^{(1)} \\
\hat{y}^{(2)}  &= w_0 + w_1 x_{\text{room}}^{(2)} + w_2 x_{\text{crime}}^{(2)}+ w_3 x_{\text{age}}^{(2)} + \ldots+ w_{14} x_{\text{rad}}^{(2)}\\
\hat{y}^{(3)}  &= w_0 + w_1 x_{\text{room}}^{(3)} + w_2 x_{\text{crime}}^{(3)}+ w_3 x_{\text{age}}^{(3)} + \ldots+ w_{14} x_{\text{rad}}^{(3)}\\
&\vdots\\


\hat{y}^{(n)} &= w_0 + w_1 x_{\text{room}}^{(n)} + w_2 x_{\text{crime}}^{(n)}+ w_3 x_{\text{age}}^{(n)} + \ldots+ w_{14} x_{\text{rad}}^{(n)}

\end{align}
```


* ##### by vector notation (slightly better)



```math
\large
\begin{equation}
\begin{cases}
\hat{y}^{(1)} &= \mathbf{w}^\top \mathbf{x}^{(1)}\\
\hat{y}^{(2)} &= \mathbf{w}^\top \mathbf{x}^{(2)}\\
\hat{y}^{(3)} &= \mathbf{w}^\top \mathbf{x}^{(3)}\\
&\vdots\\
\hat{y}^{(n)} &= \mathbf{w}^\top \mathbf{x}^{(n)}
\end{cases} 
\end{equation}  \xRightarrow[]{\mathbf{a}^\top\mathbf{b} = \mathbf{b}^\top\mathbf{a}} \begin{cases}
\hat{y}^{(1)} &= (\mathbf{x}^{(1)})^\top \mathbf{w} \\
\hat{y}^{(2)} &= (\mathbf{x}^{(2)})^\top \mathbf{w}\\
\hat{y}^{(3)} &= (\mathbf{x}^{(3)})^\top \mathbf{w}\\
&\vdots\\
\hat{y}^{(n)} &= (\mathbf{x}^{(n)})^\top \mathbf{w}
\end{cases} 
```




* ##### by matrix notation

```math
\Large
\begin{bmatrix}\hat{y}^{(1)}\\  \hat{y}^{(2)} \\ \vdots \\ \hat{y}^{(n)}\end{bmatrix} = \begin{bmatrix}  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(1)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(2)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\ & \vdots & \\  \rule[.5ex]{2.5ex}{0.5pt} &(\mathbf{x}^{(n)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\end{bmatrix} \begin{bmatrix} \vert \\ \mathbf{w} \\\vert\end{bmatrix} =\begin{bmatrix}  (\mathbf{x}^{(1)})^\top \mathbf{w}\\  (\mathbf{x}^{(2)})^\top \mathbf{w}\\  \vdots  \\ (\mathbf{x}^{(n)})^\top \mathbf{w}\end{bmatrix}
```

"""

# ╔═╡ b2f1592d-5fbf-4c7c-9d6d-562e87fb703e
md"""

* ###### where note that 
"""

# ╔═╡ 1862e10d-f9f8-4b3f-a112-2387e3204d85
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/houseX.svg" width = "500"/></center>"""

# ╔═╡ 0cae3899-b3a9-4f28-9fa4-748779afed41
md"""

## Summary, the predictions

##### For the training instances ``\mathbf{x}^{(i)} \in \mathcal{X}_{train}``, the _predictions_ are


* ##### by matrix 

```math
\large
\begin{bmatrix}\hat{y}^{(1)}\\  \hat{y}^{(2)} \\ \vdots \\ \hat{y}^{(n)}\end{bmatrix}_{\Large\hat{\mathbf{y}}} = \begin{bmatrix}  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(1)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(2)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\ & \vdots & \\  \rule[.5ex]{2.5ex}{0.5pt} &(\mathbf{x}^{(n)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\end{bmatrix}_{\Large\mathbf{X}} \begin{bmatrix} \vert \\ \mathbf{w} \\\vert\end{bmatrix}
```
\

* ##### or more concisely

```math
\huge
\boxed{
\hat{\mathbf{y}} = \mathbf{Xw} }                                      
```
"""

# ╔═╡ 0675e184-fbef-410c-bdd2-a5b0e77bbabf
# md"""


# * #####  *i.e.*

# ```math
# \Large
# \begin{bmatrix}\hat{y}^{(1)}\\  \hat{y}^{(2)} \\ \vdots \\ \hat{y}^{(n)}\end{bmatrix}  = \begin{bmatrix}  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(1)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(2)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\ & \vdots & \\  \rule[.5ex]{2.5ex}{0.5pt} &(\mathbf{x}^{(n)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\end{bmatrix} \begin{bmatrix} \vert \\ \mathbf{w} \\\vert\end{bmatrix} =\mathbf{Xw}
# ```

# """

# ╔═╡ 485d6d3d-8e80-4165-ba7e-4a0c0b661ef5
# md"""

# ## Towards matrix notation

# ##### For the training instances ``\mathbf{x}^{(i)} \in \mathcal{X}_{train}``, the _predictions_ are


# * ##### by matrix notation (very concise)

# ```math
# \large
# \begin{bmatrix}h(\mathbf{x}^{(1)})\\  h(\mathbf{x}^{(2)}) \\ \vdots \\ h(\mathbf{x}^{(n)})\end{bmatrix} = \begin{bmatrix}  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(1)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(2)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\ & \vdots & \\  \rule[.5ex]{2.5ex}{0.5pt} &(\mathbf{x}^{(n)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\end{bmatrix} \begin{bmatrix} \vert \\ \mathbf{w} \\\vert\end{bmatrix} =\begin{bmatrix}  (\mathbf{x}^{(1)})^\top \mathbf{w}\\  (\mathbf{x}^{(2)})^\top \mathbf{w}\\  \vdots  \\ (\mathbf{x}^{(n)})^\top \mathbf{w}\end{bmatrix}
# ```
# \

# * ##### or _simply_

# ```math
# \huge
# \boxed{
# \mathbf{h} = \mathbf{Xw} }                                      
# ```

# * ##### "hat" symbol $\Large \hat{\cdot}$ : remind us its _precdiction_ or _estimation_ of ``\mathbf{y}`` (from on the model)


# ```math
# \boxed{\huge
# \hat{\mathbf{y}} = \mathbf{Xw} }                                      
# ```

# """

# ╔═╡ a5323b5c-9904-4aaf-a569-4057f3c7dac6
# md"""

# ## Towards matrix notation

# ##### For the training instances ``\mathbf{x}^{(i)} \in \mathcal{X}_{train}``, the _predictions_ are


# * ##### by matrix notation (very concise)

# ```math
# \large
# \begin{bmatrix}h(\mathbf{x}^{(1)})\\  h(\mathbf{x}^{(2)}) \\ \vdots \\ h(\mathbf{x}^{(n)})\end{bmatrix} = \begin{bmatrix}  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(1)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(2)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\ & \vdots & \\  \rule[.5ex]{2.5ex}{0.5pt} &(\mathbf{x}^{(n)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\end{bmatrix} \begin{bmatrix} \vert \\ \mathbf{w} \\\vert\end{bmatrix} =\begin{bmatrix}  (\mathbf{x}^{(1)})^\top \mathbf{w}\\  (\mathbf{x}^{(2)})^\top \mathbf{w}\\  \vdots  \\ (\mathbf{x}^{(n)})^\top \mathbf{w}\end{bmatrix}
# ```
# \

# * ##### or _simply_

# ```math
# \huge
# \boxed{
# \mathbf{h} = \mathbf{Xw} }                                      
# ```

# * ##### "hat" symbol $\Large \hat{\cdot}$ : remind us they are precdiction or estimation from the model


# ```math
# \boxed{\huge
# \hat{\mathbf{y}} = \mathbf{Xw} }                                      
# ```

# """

# ╔═╡ 1431515c-3e1e-4629-b79b-21c62b222eee
md"""

## Loss in matrix notation


```math
\Large
\begin{align}
L(\mathbf{w}) &= \frac{1}{2}\sum_{i=1}^n (y^{(i)} - \hat{y}^{(i)})^2 \\
&= \boxed{\frac{1}{2} (\mathbf{y} - \hat{\mathbf{y}})^\top (\mathbf{y} -\hat{\mathbf{y}})} 
\end{align}
```

* ##### where ``\boxed{\hat{\mathbf{y}} = \mathbf{Xw}}`` are the model predictions

"""

# ╔═╡ 9d784098-5150-4a66-b2ea-7e86820a165a
md"""
##

##### *Note that*, ``\mathbf{y} - \hat{\mathbf{y}}`` is



```math
\Large
\mathbf{y} - \hat{\mathbf{y}} = \underbrace{\begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(n)}\end{bmatrix}}_{\mathbf{y}} - \underbrace{\begin{bmatrix}\hat{y}^{(1)}\\  \hat{y}^{(2)} \\ \vdots \\ \hat{y}^{(n)}\end{bmatrix}}_{\mathbf{Xw}} = \underbrace{\begin{bmatrix}y^{(1)}-\hat{y}^{(1)} \\ y^{(2)}-\hat{y}^{(2)} \\\vdots \\ y^{(n)}- \hat{y}{(n)}\end{bmatrix}}_{\text{pred. error vector}}

```

"""

# ╔═╡ 956e0525-3627-4626-bb7b-fee6299b544d
md"""


##

##### _Therefore_, the inner product ``(\mathbf{y} - \hat{\mathbf{y}})^\top (\mathbf{y} - \hat{\mathbf{y}})`` 

```math
\Large
\begin{align}
(\mathbf{y} &- \hat{\mathbf{y}})^\top (\mathbf{y} - \hat{\mathbf{y}})= \\

& {\begin{bmatrix}y^{(1)}-\hat{y}^{(1)} & y^{(2)}-\hat{y}^{(2)}& \ldots & y^{(n)}- \hat{y}^{(n)}\end{bmatrix}} {\begin{bmatrix} y^{(1)}-\hat{y}^{(1)} \\ y^{(2)}-\hat{y}^{(2)}\\ \vdots \\ y^{(n)}- \hat{y}^{(n)}\end{bmatrix}}\\
&= \boxed{\sum_{i=1}^n (y^{(i)}-\hat{y}^{(i)})^2}
\end{align}
```
* ##### the just _sum of squared errors_ (SSE)

"""

# ╔═╡ 76d88d8b-09f9-462f-a2fe-4aa14e11226c
# md"""

# ##
# ##### _Recall_ the ``i``-th prediction error is

# ```math
# \large
# \text{error}^{(i)} = y^{(i)} - h(\mathbf{x}^{(i)})
# ``` 


# """

# ╔═╡ a0ad1190-da30-45ae-a00d-189eae6ed516
md"""
##  Loss in matrix notation: *summary*

!!! note "SSE Loss in matrix notation"

	```math
	\Large
	\begin{align}
	L(\mathbf{w}) &= \frac{1}{2}\sum_{i=1}^n (y^{(i)} - \hat{y}^{(i)})^2 \\
	&= \frac{1}{2} (\mathbf{y} - \hat{\mathbf{y}})^\top (\mathbf{y} -\hat{\mathbf{y}})\\
	&= {\frac{1}{2} \|\mathbf{y} - \hat{\mathbf{y}}\|^2_2}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; {\normalsize \text{recall }\mathbf{a}^\top \mathbf{a}: \text{returns norm of vector}}
	\end{align}
	```


"""

# ╔═╡ b2309f9b-516c-4852-ad21-a1f43e3b5c7a
# md"""
# ##

# **Next** stack ``n`` training inputs ``\{\mathbf{x}^{(i)}\}`` to form a ``n\times m`` matrix ``\mathbf{X}``

# * ``\mathbf{X}``: ``n\times m`` matrix;  called **design matrix**

#   * ``n``: # of observations
#   * ``m``: # of features

# ```math
# \Large
# \mathbf{X} = \begin{bmatrix}  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(1)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(2)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\ & \vdots & \\  \rule[.5ex]{2.5ex}{0.5pt} &(\mathbf{x}^{(n)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\end{bmatrix}
# ```

# """

# ╔═╡ 01e4c730-0bf9-418b-a9d8-16ad76e2e56b
# md"""
# ##  Loss in matrix notation: *summary*

# !!! note "SSE Loss in matrix notation"

# 	```math
# 	\Large
# 	\begin{align}
# 	L(\mathbf{w}) &= \frac{1}{2}\sum_{i=1}^n (y^{(i)} - h(\mathbf{x}^{(i)}; \mathbf{w}))^2 \\
# 	&= \frac{1}{2} (\mathbf{y} - \mathbf{Xw})^\top (\mathbf{y} -\mathbf{Xw})\\
# 	&= {\frac{1}{2} \|\mathbf{y} - \mathbf{Xw}\|^2_2}\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; {\normalsize \mathbf{a}^\top \mathbf{a}: \text{returns norm of vector}}\\
# 	&= \boxed{\frac{1}{2} \|\mathbf{error}\|^2_2}
# 	\end{align}
# 	```

# 	* ##### the norm (squared length) of the error vector 

# 	```math
# 	\Large
# 		\mathbf{error} = \mathbf{y} - \mathbf{Xw} = \begin{bmatrix}y^{(1)} - h(\mathbf{x}^{(1)}) \\ y^{(2)} - h(\mathbf{x}^{(2)})\\ \vdots \\ y^{(n)} - h(\mathbf{x}^{(n)})\end{bmatrix}
# 	```
# """

# ╔═╡ 67d37ec2-5087-42c6-9400-5255db269f74
# md"For our housing dataset:"

# ╔═╡ 3967f515-3be3-46f8-8bf7-1128a8f0ed54
# aside(tip(md"Recall ``\mathbf{x}^\top`` means a row vector"))

# ╔═╡ 202f922d-0f10-4a11-a426-c39887bf8ea1
# md"""
# ##

# ##### *In summary,*

# !!! note "SSE Loss in matrix notation"

# 	```math
# 	\Large
# 	\begin{align}
# 	L(\mathbf{w}) &= \frac{1}{2}\sum_{i=1}^n (y^{(i)} - h(\mathbf{x}^{(i)}; \mathbf{w}))^2 \\
# 	&= \boxed{\frac{1}{2} (\mathbf{y} - \mathbf{Xw})^\top (\mathbf{y} -\mathbf{Xw})}
# 	\end{align}
# 	```

# """

# ╔═╡ c22b7b09-35da-4d87-b30f-9a3b525f90bf
md"""

## Why bother using matrix?


#### *Elegant* and *Concise*: ``L(\mathbf{w}) =\frac{1}{2}\|\textbf{error}\|_2^2``


#### *Efficiency*

* ###### use matrix is known as **vectorisation** 
  * Linear Algebra packages (e.g. `numpy` or `PyTorch`)are highly optimised
* ###### way more efficient than loop!



"""

# ╔═╡ 687fd7cd-8b5e-47d6-b186-4d143d61f366
TwoColumn(md"
```julia
# loss: vectorised
function loss(w, X, y) 
	error = y - X * w
	return 0.5 * dot(error, error) 
end
```
", md"

```julia
# loss: with loop
function loss_loop(w, X, y) 
	# number of observations
	n = length(y) 
	loss = 0
	for i in 1:n
		li = (y[i] - dot(X[i,:], w))^2
		loss += li
	end
	return .5 * loss 
end
```
")

# ╔═╡ 1398bda3-5b94-4d54-a553-ca1ac1bc6ce9
function loss(w, X, y) # in matrix notation
	error = y - X * w
	0.5 * (error' * error) 
end;

# ╔═╡ 4e430aa8-9c74-45c1-8771-b33e558208cf
function loss_loop(w, X, y) # no matrix notation
	n = length(y) # number of observations
	.5 * sum([(y[i] - X[i,:]' * w )^2 for i in 1:n])
end;

# ╔═╡ 8e6a0d85-e421-44e5-9c23-661b076382b0
begin
	nobs, ndims = 100_000, 1_000
	# XX_test, yy_test, ww_test = rand(nobs,ndims), rand(nobs), rand(ndims)
end;

# ╔═╡ 0a162252-359a-4be9-96d9-d66d4dca926c
@benchmark loss(rand(ndims), rand(nobs,ndims), rand(nobs))

# ╔═╡ 648ef3b0-2198-44eb-9085-52e2362b7f88
@benchmark loss_loop(rand(ndims), rand(nobs, ndims), rand(nobs))

# ╔═╡ 1b08a455-85b2-450b-b5bd-5c52694a740b
# md"""
# ## What the loss function looks like?



# !!! note "The loss: a quadratic function of w"

# 	```math
# 	\Large
# 	L(\mathbf{w}) = \frac{1}{2} (\mathbf{y} - \mathbf{Xw})^\top (\mathbf{y} -\mathbf{Xw})
# 	```

# 	* the loss is a quadratic function *w.r.t* ``\mathbf{w}`` (if it's not obvious to you now, you will see why soon)
# """

# ╔═╡ c6f8f57a-f487-4105-8088-acd91f3c43fa
md"""

## Optimise ``L(\mathbf{w})``: _least square estimation_


```math
\Large
\begin{align}
\hat{\mathbf{w}} &\leftarrow \arg\min_{\mathbf{w}}L(\mathbf{w})
\end{align}
```

* ##### _where_ 
```math
\large
	L(\mathbf{w}) = \frac{1}{2}\sum_{i=1}^n (y^{(i)} - \hat{y}^{(i)})^2 =\frac{1}{2} (\mathbf{y} - \hat{\mathbf{y}})^\top (\mathbf{y} -\hat{\mathbf{y}})
```

> ##### *Optimisation*: **_Good old calculus_**

"""

# ╔═╡ 788efd7e-7a95-4134-a514-6a03654a4af5
md"""

## ``\arg\min``, ``\arg\max`` ``\Leftrightarrow`` ``\nabla L(\mathbf{w}) =0``



#### `argmin` or `argmax` (an optimisation problem)
``\;\;\;\;``_if_ ``L(\mathbf{w})`` is continuous and differentiable 


```math
\huge
\Downarrow 
```

* #### _First_, find gradient ``\nabla L(\mathbf{w})`` 

* #### _Second_, _solve_ it for ``\mathbf{0}``

```math
\Large
\nabla L(\mathbf{w}) =\mathbf{0}
```
"""

# ╔═╡ 158d1c6e-fbf5-454d-bd50-e60aafdc291d
# md"""

# ## _First_, find the gradient ``\nabla L(\mathbf{w})``
# """

# ╔═╡ b89c8eee-e34d-4e25-aca3-68fada91ed89
# md"""

# ##
# ### Recap: some matrix calculus results


# ```math
# \large
# h(\mathbf{w}) = \mathbf{b}^\top \mathbf{w} + c
# ```

# * its gradients *w.r.t* ``\mathbf{w}`` is


# ```math
# \boxed{
# \large
# \nabla_\mathbf{w} h(\mathbf{w}) =  \mathbf{b}}
# ```

# * which is just generalisation of ``h(w) =bw +c``  derivative

# ```math
# \large
# h'(w) = b
# ```


# """

# ╔═╡ eb97eef7-e07e-429f-ae40-809761a75abc
# md"""
# ##
# ### Recap: some matrix calculus results


# ```math
# \large
# f(\mathbf{w}) = \mathbf{w}^\top \mathbf{Aw} + \mathbf{b}^\top \mathbf{w} + c
# ```

# * its gradients *w.r.t* ``\mathbf{w}`` (symmetric ``\mathbf{A}``)

# ```math
# \boxed{
# \large
# \nabla_\mathbf{w} f(\mathbf{w}) = 2 \mathbf{A}\mathbf{w} + \mathbf{b}}
# ```

# * which is just generalisation of ``f(w) =\underbrace{aw^2}_{w\cdot a\cdot w}+ bw +c``'s derivative:

# ```math
# f'(w) = 2aw +b
# ```


# """

# ╔═╡ 073c6ed9-a4cc-489d-9609-2d710aa7740f
md"""

## Method 1: _with the first definition_



```math
\Large
L(\mathbf{w}) = \frac{1}{2}\sum_{i=1}^n (y^{(i)} - \hat{y}^{(i)})^2
```


##### Consider the ``i``th loss only

* ##### the ``i``-th loss is

```math
\Large
L^{(i)}(\mathbf{w}) = (y^{(i)} - \hat{y}^{(i)})^2
```

* ##### we are going to use _chain rule_ to find the gradient
"""

# ╔═╡ e57e21a0-01f7-470a-912e-c2ebb10560b6
md"""

## Method 1: _with the first definition_




##### Consider the ``i``th loss only


```math
\Large
L^{(i)}(\mathbf{w}) = (y^{(i)} - \hat{y}^{(i)})^2
```
* ##### remember ``\hat{y}^{(i)} = \mathbf{w}^\top\mathbf{x}^{(i)}``
* ##### loss computation as a dependence graph
"""

# ╔═╡ 701f988a-139a-4b81-97d1-46b24ac6335e
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/ssegrad1.svg' width = '600' /></center>"

# ╔═╡ be664fd1-7cc3-478d-8e75-86ff0643933c
md"""

##

##### The local derivatives/gradients
"""

# ╔═╡ 7fc17194-f462-4326-b22a-b64dbce6b944
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/ssegrad2.svg' width = '600' /></center>"

# ╔═╡ 697700d6-a6d9-4a05-97e0-967ebf22454f
md"""

##

##### Chain rule: collect and mutiply all local derivatives/differentials
"""

# ╔═╡ df1491b1-d27a-4c25-a721-34928cef33cd
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/ssegrad3.svg' width = '600' /></center>"

# ╔═╡ 7d68dba0-3093-4495-89d9-2dd81b86ef29
md"""

##
"""

# ╔═╡ 5a995196-6c0a-4c6c-b6a6-a7c55d506aae
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/ssegrad4.svg' width = '600' /></center>"

# ╔═╡ 87372788-5f01-4bf4-be10-099ba07d5078
md"""

#### Therefore, the _total gradient_ is:


```math
\Large
\begin{align}
\nabla L(\mathbf{w}) &= \frac{1}{2} \sum_{i=1}^n\; \color{red}{\nabla L^{(i)}(\mathbf{w})} \\
&=  \frac{1}{2} \sum_{i=1}^n \color{red}{-2 (y^{(i)} - \hat{y}^{(i)}) \cdot \mathbf{x}^{(i)}} \\
&=-\sum_{i=1}^n { (y^{(i)} - \hat{y}^{(i)}) \cdot \mathbf{x}^{(i)}} 
\end{align}
```
"""

# ╔═╡ d57d1d6e-db6f-4ea6-b0fc-0f1d7cc4aa3b
# aside(tip(md"""Remember 

# ```math
# h(\mathbf{x}; \mathbf{w}) = \mathbf{w}^\top \mathbf{x}
# ```

# And its gradient w.r.t ``\mathbf{w}`` is

# ```math
# \nabla_{\mathbf{w}} \mathbf{w}^\top \mathbf{x} = \mathbf{x}
# ```
# """))

# ╔═╡ 86a64b00-28e8-4ee0-b285-44d80e74ccfd
md"""

## Method 2: vector norm definition


```math
\boxed{
\Large
L(\mathbf{w}) = \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})}
```


#### *Expand the quadratic form* first
```math
\begin{align}
L(\mathbf{w}) &= \boxed{\large \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})}\\


\end{align}
```

"""

# ╔═╡ d4e9d38f-40e8-40b7-a454-25f6d7adb64c
md"""

## Method 2: vector norm definition


```math
\boxed{
\Large
L(\mathbf{w}) = \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})}
```


#### *Expand the quadratic form* first
```math
\begin{align}
L(\mathbf{w}) &=  \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})\\
&= \boxed{\large\frac{1}{2} (\mathbf{y}^\top-\hat{\mathbf{y}}^\top) (\mathbf{y}-\hat{\mathbf{y}})} \tag{apply $\top$} \\

\end{align}
```

"""

# ╔═╡ b7d9dd45-1a9d-4ff9-a6b6-19c1c7d8ca2e
md"""

## Method 2: vector norm definition


```math
\boxed{
\Large
L(\mathbf{w}) = \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})}
```


#### *Expand the quadratic form* first
```math
\begin{align}
L(\mathbf{w}) &=  \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})\\
&= \frac{1}{2} (\mathbf{y}^\top-\hat{\mathbf{y}}^\top) (\mathbf{y}-\hat{\mathbf{y}}) \tag{apply $\top$} \\
&= \boxed{\large\frac{1}{2} \left( \mathbf{y}^\top(\mathbf{y}-\hat{\mathbf{y}}) - \hat{\mathbf{y}}^\top(\mathbf{y}-\hat{\mathbf{y}})\right )}\tag{distributive law}\\

\end{align}
```

"""

# ╔═╡ 9db7c362-916e-4e0c-8a81-2450a7156760
md"""

## Method 2: vector norm definition


```math
\boxed{
\Large
L(\mathbf{w}) = \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})}
```


#### *Expand the quadratic form* first
```math
\begin{align}
L(\mathbf{w}) &=  \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})\\
&= \frac{1}{2} (\mathbf{y}^\top-\hat{\mathbf{y}}^\top) (\mathbf{y}-\hat{\mathbf{y}}) \tag{apply $\top$} \\
&= \frac{1}{2} \left( \mathbf{y}^\top(\mathbf{y}-\hat{\mathbf{y}}) - \hat{\mathbf{y}}^\top(\mathbf{y}-\hat{\mathbf{y}})\right )\tag{distributive law}\\
&= \boxed{\large{\frac{1}{2}\left( \mathbf{y}^\top\mathbf{y}-\mathbf{y}^\top\hat{\mathbf{y}} - \hat{\mathbf{y}}^\top\mathbf{y}+\hat{\mathbf{y}}^\top\hat{\mathbf{y}}\right ) }}\tag{distributive law} \\

\end{align}
```

"""

# ╔═╡ ac28562a-4501-4906-8160-828be49b500f
md"""

## Method 2: vector norm definition


```math
\boxed{
\Large
L(\mathbf{w}) = \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})}
```


#### *Expand the quadratic form* first
```math
\begin{align}
L(\mathbf{w}) &=  \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})\\
&= \frac{1}{2} (\mathbf{y}^\top-\hat{\mathbf{y}}^\top) (\mathbf{y}-\hat{\mathbf{y}}) \tag{apply $\top$} \\
&= \frac{1}{2} \left( \mathbf{y}^\top(\mathbf{y}-\hat{\mathbf{y}}) - \hat{\mathbf{y}}^\top(\mathbf{y}-\hat{\mathbf{y}})\right )\tag{distributive law}\\
&= \frac{1}{2}\left( \mathbf{y}^\top\mathbf{y}-\mathbf{y}^\top\hat{\mathbf{y}} - \hat{\mathbf{y}}^\top\mathbf{y}+\hat{\mathbf{y}}^\top\hat{\mathbf{y}}\right ) \tag{distributive law} \\
&= \boxed{\large\frac{1}{2}\left( \mathbf{y}^\top\mathbf{y} -2 \mathbf{y}^\top\hat{\mathbf{y}}+\hat{\mathbf{y}}^\top\hat{\mathbf{y}}\right ) }\tag{$\mathbf{y}^\top\hat{\mathbf{y}} =\hat{\mathbf{y}}^\top\mathbf{y}$}
\end{align}
```

"""

# ╔═╡ 6f7a822b-8fc4-4fe1-bd5c-fb7da07e49b0
md"""

## Method 2: vector norm definition

```math
\boxed{
\Large
L(\mathbf{w}) = \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})}
```


#### *Expand the quadratic form* first
```math
\begin{align}
L&(\mathbf{w}) =  \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})\\
&= \frac{1}{2} (\mathbf{y}^\top-\hat{\mathbf{y}}^\top) (\mathbf{y}-\hat{\mathbf{y}}) \tag{apply $\top$} \\
&= \frac{1}{2} \left( \mathbf{y}^\top(\mathbf{y}-\hat{\mathbf{y}}) - \hat{\mathbf{y}}^\top(\mathbf{y}-\hat{\mathbf{y}})\right )\tag{distributive law}\\
&= \frac{1}{2}\left( \mathbf{y}^\top\mathbf{y}-\mathbf{y}^\top\hat{\mathbf{y}} - \hat{\mathbf{y}}^\top\mathbf{y}+\hat{\mathbf{y}}^\top\hat{\mathbf{y}}\right ) \tag{distributive law} \\
&= \frac{1}{2}\left( \mathbf{y}^\top\mathbf{y} -2 \mathbf{y}^\top\hat{\mathbf{y}}+\hat{\mathbf{y}}^\top\hat{\mathbf{y}}\right ) \tag{$\mathbf{a}^\top\mathbf{b} =\mathbf{b}^\top\mathbf{a}$}\\
&= \boxed{\large\frac{1}{2}\left( \mathbf{y}^\top\mathbf{y} -2\mathbf{y}^\top (\mathbf{Xw})+(\mathbf{Xw})^\top(\mathbf{Xw})\right )} \tag{$\hat{\mathbf {y} }=\mathbf{Xw}$}
\end{align}
```

"""

# ╔═╡ 8131a33f-b3ac-4d6f-8cf8-26cc71786177
md"""

## Method 2: vector norm definition

```math
\boxed{
\Large
L(\mathbf{w}) = \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})}
```


#### *Expand the quadratic form* first
```math
\begin{align}
L&(\mathbf{w}) =  \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})\\
&= \frac{1}{2} (\mathbf{y}^\top-\hat{\mathbf{y}}^\top) (\mathbf{y}-\hat{\mathbf{y}}) \tag{apply $\top$} \\
&= \frac{1}{2} \left( \mathbf{y}^\top(\mathbf{y}-\hat{\mathbf{y}}) - \hat{\mathbf{y}}^\top(\mathbf{y}-\hat{\mathbf{y}})\right )\tag{distributive law}\\
&= \frac{1}{2}\left( \mathbf{y}^\top\mathbf{y}-\mathbf{y}^\top\hat{\mathbf{y}} - \hat{\mathbf{y}}^\top\mathbf{y}+\hat{\mathbf{y}}^\top\hat{\mathbf{y}}\right ) \tag{distributive law} \\
&= \frac{1}{2}\left( \mathbf{y}^\top\mathbf{y} -2 \mathbf{y}^\top\hat{\mathbf{y}}+\hat{\mathbf{y}}^\top\hat{\mathbf{y}}\right ) \tag{$\mathbf{a}^\top\mathbf{b} =\mathbf{b}^\top\mathbf{a}$}\\
&= \frac{1}{2}\left( \mathbf{y}^\top\mathbf{y} -2\mathbf{y}^\top (\mathbf{Xw})+(\mathbf{Xw})^\top(\mathbf{Xw})\right ) \tag{$\hat{\mathbf {y} }=\mathbf{Xw}$}\\
&= \boxed{\large\frac{1}{2}\left( \mathbf{y}^\top\mathbf{y} -2 \mathbf{y}^\top \mathbf{Xw}+\mathbf{w}^\top\mathbf{X}^\top\mathbf{Xw}\right ) }\tag{$(\mathbf{Xw})^\top=\mathbf{w}^\top\mathbf{X}^\top$}\\
\end{align}
```

"""

# ╔═╡ 79a7094e-a6e4-44bf-baa7-eb3974360d4c
md"""

## Method 2: vector norm definition

```math
\boxed{
\Large
L(\mathbf{w}) = \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})}
```


#### *Expand the quadratic form* first
```math
\begin{align}
&L(\mathbf{w}) =  \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})\\
&= \frac{1}{2} (\mathbf{y}^\top-\hat{\mathbf{y}}^\top) (\mathbf{y}-\hat{\mathbf{y}}) \tag{apply $\top$} \\
&= \frac{1}{2} \left( \mathbf{y}^\top(\mathbf{y}-\hat{\mathbf{y}}) - \hat{\mathbf{y}}^\top(\mathbf{y}-\hat{\mathbf{y}})\right )\tag{distributive law}\\
&= \frac{1}{2}\left( \mathbf{y}^\top\mathbf{y}-\mathbf{y}^\top\hat{\mathbf{y}} - \hat{\mathbf{y}}^\top\mathbf{y}+\hat{\mathbf{y}}^\top\hat{\mathbf{y}}\right ) \tag{distributive law} \\
&= \frac{1}{2}\left( \mathbf{y}^\top\mathbf{y} -2 \mathbf{y}^\top\hat{\mathbf{y}}+\hat{\mathbf{y}}^\top\hat{\mathbf{y}}\right ) \tag{$\mathbf{a}^\top\mathbf{b} =\mathbf{b}^\top\mathbf{a}$}\\
&= \frac{1}{2}\left( \mathbf{y}^\top\mathbf{y} -2\mathbf{y}^\top (\mathbf{Xw})+(\mathbf{Xw})^\top(\mathbf{Xw})\right ) \tag{$\hat{\mathbf {y} }=\mathbf{Xw}$}\\
&= \frac{1}{2}\left( \mathbf{y}^\top\mathbf{y} -2 \mathbf{y}^\top \mathbf{Xw}+\mathbf{w}^\top\mathbf{X}^\top\mathbf{Xw}\right ) \tag{$(\mathbf{Xw})^\top=\mathbf{w}^\top\mathbf{X}^\top$}\\
&=\boxed{\frac{1}{2}\left( \mathbf{y}^\top\mathbf{y} -2 (\mathbf{X}^\top\mathbf{y})^\top\mathbf{w}+\mathbf{w}^\top\mathbf{X}^\top\mathbf{Xw}\right )} \tag{$\mathbf{y}^\top\mathbf{X} = (\mathbf{X}^\top\mathbf{y})^\top$}
\end{align}
```

"""

# ╔═╡ 11babbc2-bd40-4636-9aa8-a23a8568ea05
md"""

## Method 2: vector norm definition


```math
\boxed{
\Large
L(\mathbf{w}) = \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})}
```


#### *Expand the quadratic form*, we have
```math
\Large
\begin{align}
L(\mathbf{w}) 
&= \frac{1}{2}\left (\mathbf{w}^\top\mathbf{X}^\top\mathbf{Xw}   -2 (\mathbf{X}^\top\mathbf{y})^\top\mathbf{w}+\mathbf{y}^\top\mathbf{y}\right )
\end{align}
```

"""

# ╔═╡ 4c3b3d0f-3800-443f-80f8-244e57304bdc
md"""

## Method 2: vector norm definition


```math
\boxed{
\Large
L(\mathbf{w}) = \frac{1}{2} (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}})}
```


#### *Expand the quadratic form*, we have
```math
\Large
\begin{align}
L(\mathbf{w}) 

&= \frac{1}{2}\left (\mathbf{w}^\top\underbrace{\mathbf{X}^\top\mathbf{X}}_{\mathbf{A}}\mathbf{w}  \;\;\underbrace{ -2 (\mathbf{X}^\top\mathbf{y}}_{\mathbf{b}})^\top\mathbf{w}\;+\;\underbrace{\mathbf{y}^\top\mathbf{y}}_c\right )
\end{align}
```

* #### this is a quadratic function: 
  * ##### quadratic term: ``\mathbf{A} = \mathbf{X}^\top\mathbf{X}``, 
  * ##### linear term ``\mathbf{b}=-2\mathbf{X}^\top\mathbf{y}``, 
  * ##### and constant ``c=\mathbf{y}^\top\mathbf{y}``
"""

# ╔═╡ 898b97f0-519c-4b8a-ad1f-5e9b2164ae51
md"""

## Question


```math
\Large
\begin{align}
L(\mathbf{w}) 
&= \frac{1}{2}\left (\mathbf{w}^\top\underbrace{\mathbf{X}^\top\mathbf{X}}_{\mathbf{A}}\mathbf{w}\;  \underbrace{ -2 (\mathbf{X}^\top\mathbf{y}}_{\mathbf{b}})^\top\mathbf{w}+\underbrace{\mathbf{y}^\top\mathbf{y}}_c\right )
\end{align}
```

* #### this is a quadratic function with ``\mathbf{A} = \mathbf{X}^\top\mathbf{X}``

!!! question "Question"
	> ##### Question: what does it imply?
	###### _hint:_ is ``\mathbf{X}^\top\mathbf{X}`` positive/negative definite?
"""

# ╔═╡ 8467d20a-65dc-403f-933f-3932c5b14d13
Foldable("Implication", md"""
#### ``L(\mathbf{w})`` is a quadratic function facing up; there is a minimum
* since ``\mathbf{X}^\top\mathbf{X}`` is positive (semi-)definite:

```math
\mathbf{w}^\top(\mathbf{X}^\top\mathbf{X})\mathbf{w} \geq 0; \text{ for all }\mathbf{w}\neq \mathbf{0} \in \mathbb{R}^{m}
```

* to see this, denote ``\mathbf{u} =\mathbf{Xw}``, then ``\mathbf{u}^\top\mathbf{u} = \mathbf{w}^\top(\mathbf{X}^\top\mathbf{X})\mathbf{w} = u_1^2 + u_2^2 + \ldots u_m^2 \geq 0``

$(
let
	gr()
	Random.seed!(111)
	num_features = 1
	num_data = 100
	true_w = [1,2] 
	# simulate the design matrix or input features
	X_train_ = [ones(num_data) range(-.5, 1; length=num_data)]
	# generate the noisy observations
	y_train_ = X_train_ * true_w
	xlength, ylength = 100,100
	plt1=plot(range(true_w[1] .+ [-xlength, xlength]..., 50),  range(true_w[2] .+ [-ylength, ylength]..., 50), (x,y) -> loss([x, y], X_train_, y_train_), st=:surface, colorbar=false, xlabel=L"w_0", ylabel=L"w_1",c=:coolwarm, zlabel="loss",alpha=0.8, title=L"L(\mathbf{w})")
	
	scatter!([true_w[1]], [true_w[2]], [0], m=:x, ms=5, mc=:black, label="minimum")
end

)
""")

# ╔═╡ 92d7be2f-bdb5-457c-8bfd-2fe98cc32e31
md"""


## Method 2: continue


#### So the loss's a quadratic function, how to find its minimum?

```math
\Large
\begin{align}
L(\mathbf{w}) 
&= \frac{1}{2} \left  (\underbrace{\mathbf{w}^\top(\mathbf{X}^\top\mathbf{X})\mathbf{w}}_{\mathbf{w}^\top \mathbf{A} \mathbf{w}}\;\; \underbrace{-2(\mathbf{X}^\top \mathbf{y})^\top\mathbf{w}}_{\mathbf{b}^\top \mathbf{w}} + \underbrace{\mathbf{y}^\top \mathbf{y}}_{c} \right ) \\
&\\
&\text{We still need to find the gradient!}
\end{align}
```


"""

# ╔═╡ 629e38b9-353c-4b55-9ffe-4b1b249843eb
md"""
## Tip


```math
\boxed{
\Large
\nabla_\mathbf{w} \mathbf{w}^\top \mathbf{Aw} = 2 \mathbf{A}\mathbf{w}}
```

```math
\boxed{
\Large
\nabla_\mathbf{w}  \mathbf{b}^\top \mathbf{w} = \mathbf{b}}
```

```math
\boxed{
\Large
\nabla_\mathbf{w}  c = \mathbf{0}}
```

"""

# ╔═╡ 64ba3967-ba5f-454e-9809-867b4544dd5a
md"""


## Method 2: continue


#### We realise the loss's a quadratic function

```math
\Large
\begin{align}
L(\mathbf{w}) 
&= \frac{1}{2} \left  (\underbrace{\mathbf{w}^\top(\mathbf{X}^\top\mathbf{X})\mathbf{w}}_{\mathbf{w}^\top \mathbf{A} \mathbf{w}}\;\; \underbrace{-2(\mathbf{X}^\top \mathbf{y})^\top\mathbf{w}}_{\mathbf{b}^\top \mathbf{w}} + \underbrace{\mathbf{y}^\top \mathbf{y}}_{c} \right ) \\
&\\
&\text{We still need to find the gradient!}\\
&\\
\nabla L(\mathbf{w}) &= \frac{1}{2} \left( 2\cdot \underbrace{(\mathbf{X}^\top\mathbf{X})}_{\mathbf{A}}\;\, \mathbf{w}\;\; \underbrace{-\; 2 \mathbf{X}^\top\mathbf{y}}_{\mathbf{b}} \;\;\;\;\;+\;\; \mathbf{0}\right) \\
\end{align}
```
##


##### The *gradient* is


```math
\Large

\begin{align}
\nabla L(\mathbf{w}) &= \frac{1}{\cancel {2}} \left( \cancel {2}\cdot {\mathbf{X}^\top\mathbf{X}}\,\mathbf{w} - \cancel {2} {\mathbf{X}^\top\mathbf{y}} \right) \\
&=  \mathbf{X}^\top\mathbf{Xw} -  \mathbf{X}^\top\mathbf{y}\\
&= \boxed{\mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} )}
\end{align}
```

"""

# ╔═╡ 350f2a70-405c-45dc-bfcd-913bc9a7de75
md"""

## Exercise

!!! question "Exercise"
	##### Verify that  the two gradient expressions are the same
	```math
	\Large
		\mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) = -\sum_{i=1}^n   (y^{(i)} -  \mathbf{w}^\top \mathbf{x}^{(i)})) \cdot  \mathbf{x}^{(i)}
	```

"""

# ╔═╡ a044c05e-2513-48c6-a1cb-002acc3eacfd
md"""

## ``\arg\min``, ``\arg\max`` ``\Leftrightarrow`` ``\nabla L(\mathbf{w}) =\mathbf{0}``


* #### _First_, find gradient ``\nabla L(\mathbf{w})``  (_just done_)


```math
\Large
\nabla L(\mathbf{w}) = \mathbf{X}^\top (\mathbf{Xw} - \mathbf{y})
```

* #### _Second_, _solve_ it for ``\mathbf{0}`` (_**still needs to do this**_)

```math
\Huge
\boxed{\nabla L(\mathbf{w}) =\mathbf{0}}
```
"""

# ╔═╡ 0adbec1a-4962-4e4a-8cec-e57ff9abb6d6
md"""


## Least square estimation -- normal equation

#### To optimise

```math
\Large
\mathbf{w}_{\text{LSE}} \leftarrow \arg\min_{\mathbf{w}} L(\mathbf{w})
```


* #### set the gradient to **zero** and solve it!


```math
\Huge
\boxed{\nabla L(\mathbf{w}) = \mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) = \mathbf{0}}
```


* #### this is called the _normal equation_ 

"""

# ╔═╡ 970ccf20-d9a4-4740-9c1f-d31aa72f1856
md"""


## Least square estimation -- normal equation



* ##### set the gradient to **zero** and solve it!


```math
\begin{align}
\nabla L(&\mathbf{w}) 
= \mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) = \mathbf{0} \\
&\Rightarrow \boxed{\Large\mathbf{X}^\top\mathbf{Xw} - \mathbf{X}^\top\mathbf{y} =\mathbf{0}}\tag{distributive law}\\

\end{align}

```

"""

# ╔═╡ 8d788ccb-4f2a-4e15-ac2f-16c87aeb0a38
md"""


## Least square estimation -- normal equation





* ##### set the gradient to **zero** and solve it!


```math
\begin{align}
\nabla L(&\mathbf{w}) 
= \mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) = \mathbf{0} \\
&\Rightarrow \mathbf{X}^\top\mathbf{Xw} - \mathbf{X}^\top\mathbf{y} =\mathbf{0}\tag{distributive law}\\
&\Rightarrow \boxed{\Large\mathbf{X}^\top\mathbf{Xw} = \mathbf{X}^\top\mathbf{y}} \tag{add $ \mathbf{X}^\top\mathbf{y}$ on both sides}\\

\end{align}

```

"""

# ╔═╡ 661ef8d4-f93f-4980-b7ea-fd7f9d77b76a
md"""


## Least square estimation -- normal equation





* ##### set the gradient to **zero** and solve it!


```math
\begin{align}
&\nabla L(\mathbf{w}) 
= \mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) = \mathbf{0} \\
&\Rightarrow \mathbf{X}^\top\mathbf{Xw} - \mathbf{X}^\top\mathbf{y} =\mathbf{0}\tag{distributive law}\\
&\Rightarrow \mathbf{X}^\top\mathbf{Xw} = \mathbf{X}^\top\mathbf{y} \tag{add $ \mathbf{X}^\top\mathbf{y}$}\\
&\Rightarrow \boxed{\large(\mathbf{X}^\top\mathbf{X})^{-1}\cdot\mathbf{X}^\top\mathbf{Xw} = (\mathbf{X}^\top\mathbf{X})^{-1} \cdot\mathbf{X}^\top\mathbf{y}} \tag{left * $(\mathbf{X}^\top\mathbf{X})^{-1}$ }\\
\end{align}

```

"""

# ╔═╡ efcc2e8f-6401-4256-bc51-556be014823e
md"""


## Least square estimation -- normal equation



* ##### set the gradient to **zero** and solve it!


```math
\begin{align}
\nabla L(&\mathbf{w}) 
= \mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) = \mathbf{0} \\
&\Rightarrow \mathbf{X}^\top\mathbf{Xw} - \mathbf{X}^\top\mathbf{y} =\mathbf{0}\tag{distributive law}\\
&\Rightarrow \mathbf{X}^\top\mathbf{Xw} = \mathbf{X}^\top\mathbf{y} \tag{add $ \mathbf{X}^\top\mathbf{y}$}\\
&\Rightarrow (\mathbf{X}^\top\mathbf{X})^{-1}\cdot\mathbf{X}^\top\mathbf{Xw} = (\mathbf{X}^\top\mathbf{X})^{-1} \cdot\mathbf{X}^\top\mathbf{y} \tag{left * $(\mathbf{X}^\top\mathbf{X})^{-1}$ }\\
&\Rightarrow \boxed{\Large\mathbf{Iw} = (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{X}^\top\mathbf{y} }\tag{inverse definition} 
\end{align}

```

"""

# ╔═╡ cea0a958-8128-4c76-9e5f-f3614026b01e
md"""


## Least square estimation -- normal equation




* ##### set the gradient to **zero** and solve it!


```math
\begin{align}
\nabla L(&\mathbf{w}) 
= \mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) = \mathbf{0} \\
&\Rightarrow \mathbf{X}^\top\mathbf{Xw} - \mathbf{X}^\top\mathbf{y} =\mathbf{0}\tag{distributive law}\\
&\Rightarrow \mathbf{X}^\top\mathbf{Xw} = \mathbf{X}^\top\mathbf{y} \tag{add $ \mathbf{X}^\top\mathbf{y}$}\\
&\Rightarrow (\mathbf{X}^\top\mathbf{X})^{-1}\cdot\mathbf{X}^\top\mathbf{Xw} = (\mathbf{X}^\top\mathbf{X})^{-1} \cdot\mathbf{X}^\top\mathbf{y} \tag{left * $(\mathbf{X}^\top\mathbf{X})^{-1}$ }\\
&\Rightarrow \mathbf{Iw} = (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{X}^\top\mathbf{y} \tag{inverse definition} \\
&\Rightarrow \boxed{\Large\mathbf{w} = (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{X}^\top\mathbf{y}} \tag{$\mathbf{Iw} =\mathbf{w}$}
\end{align}

```

"""

# ╔═╡ fd6beddc-1a0a-4256-b978-4c0da9f4875b
md"""


## Least square estimation -- normal equation


#### To optimise

```math
\Large
\mathbf{w}_{\text{LSE}} \leftarrow \arg\min_{\mathbf{w}} L(\mathbf{w})
```


* ##### set the gradient to **zero** and solve it!


```math
\Large
\begin{align}
\nabla L(\mathbf{w}) 
= \mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) = \mathbf{0} 
\Rightarrow \boxed{\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}}
\end{align}

```

* #### this is known as the *normal equation* approach 
"""

# ╔═╡ 6da86877-1d96-4a73-af75-e4ae3eea1499
md"""
## Naive implementation

```math
\Large
\begin{align}
\boxed{\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}}
\end{align}

```
"""

# ╔═╡ 8fbcf6c3-320c-47ae-b4d3-d710a120eb1a
function naive_normal_eq_est(X, y) # naive implementation
	# a naive method to do least square
	(X'X)^(-1) * X' * y
end

# ╔═╡ f90bf568-33fc-475a-9079-99f79af891f5
md"Estiamte: $(@bind estimate_ CheckBox(default=false))"

# ╔═╡ 4d0c089b-2ed3-4787-9c36-2941600b079f
md"""
## Implementation 


```math
\Large
\begin{align}
 \boxed{\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}}
\end{align}

```

#### In practice, we **DO NOT** directly invert ``\mathbf{X}^\top\mathbf{X}`` 

* it is **computational expensive** for large models (*i.e.* a lot of features)
  * inverting a ``m\times m`` matrix is expensive: ``O(m^3)``: ``m`` is number of features
* also not **numerical stable**

##

### *Python*: we use `numpy`'s `np.linalg.lstsq()`
* `lstsq`: least square 

```python
# add dummy ones
X_bias = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)
# NumPy shapes: w_fit is (M+1,) if X is (N,M+1) and yy is (N,)
w_fit = np.linalg.lstsq(X_bias, yy, rcond=None)[0]
```

"""

# ╔═╡ 088fc2bb-fc33-499a-8baf-a9a8628bd5c8
md"""

## Implementation in `Julia`/`Matlab`

##### Deadly simple: `w = X \ y`
* ##### `\`: `mldivide` (matrix left divide)

```julia
# for both Julia and Matlab
X_bias = [ones(size(X, 1)) X]
w_fit = X_bias \ yy;
```
"""

# ╔═╡ 14032a47-b996-43f1-bf47-b9ef33d92f76
function least_square_est(X, y) # implement the method here!
	X \ y
end;

# ╔═╡ 08897017-3fd8-43ff-8922-2c496b9aefd8
md"Estiamte: $(@bind estimate CheckBox(default=false))"

# ╔═╡ e4d5e0ec-8eb2-4af5-87a3-d9ecad0f104f
# md"""

# ## What can go wrong in practice ?


# ##### What if the features are _highly_ correlated?
# * one feature: ``{x}_1`` (height in `cm`)
# * another feature ``{x}_2`` (height in `m`): then ``{x}_2 = 0.01 \times {x}_1`` for all observations




# """

# ╔═╡ bb2b313a-b5ec-4246-8e2e-fe4e6aab9d6c
# md"""

# ## When ``\mathbf{X}^\top\mathbf{X}`` not invertible?

# #### *for example*, colinear features
# * (*more correctly: _linearly dependent feature vectors_)
# """

# ╔═╡ da365bdc-da2b-4da2-8e2e-3d2f88a85fec
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/collinearX_.svg' width = '600' /></center>"

# ╔═╡ 38ec0405-16a2-42ee-a4af-2a61c6f87ba1
# md"""

# Add targets: $(@bind add_ys CheckBox(false));Add fitted plane: $(@bind add_fit_plane CheckBox(default=false)); move me: $(@bind w_1 Slider(-1.2:0.1:2.0, default =0.6))
# """

# ╔═╡ 7bfa4bab-e563-4298-acf4-ed955b6bdabd
# let
# 	plotly()
# 	xs = -3:0.6:3
# 	ys = (-3:0.6:3)/100
# 	# w_1 = 0.6
# 	w_2 = 2 - w_1
# 	w = [w_1, w_2]
# 	c = 0
# 	# plt = scatter([x1s], [x2s], zeros(length(x1s)), ms=1, c=:gray)
# 	if add_ys
# 		plt = scatter([x1s], [x2s], [x1s+ 100* x2s], label="",   zlim = [-6, 6], ms=2, c=1, xlabel="x1 " *" in (cm)", ylabel="x2 "*" in (m)", zlabel="weight")
# 	else
# 		plt = scatter([x1s], [x2s], zeros(length(x1s)), label="",   zlim = [-1, 6], ms=2, m=:cross, c=:gray, xlabel="x1 " *" in (cm)", ylabel="x2 "*" in (m)", zlabel="weight", framestyle=:zerolines)
# 	end
# 	if add_fit_plane
# 		plot!(xs, ys, (x,y) -> dot([x,y*100], w) + c, st=:surface, alpha=0.8, c=:coolwarm, display_option=Plots.GR.OPTION_MESH, colorbar=false, zlim = [-6, 6])
# 	end
# 	plt
# end

# ╔═╡ fc3fe35d-8437-46d4-9ea0-57c4c48f3ae0
# md"""

# !!! qustion "Question"
# 	#### How many solutions ? 


# $(Foldable("Answer", md"
# ##### infinite !"))
	
# \

# """

# ╔═╡ cb2e0882-dc92-421e-9352-cfdf3f9da96b
# begin
# 	Random.seed!(122)
# 	nobs_ = 25
# 	x1s = rand(nobs_) * 5 .- 5/2
# 	x2s = x1s/100
# end;

# ╔═╡ 619819c3-b8bd-456b-85fd-f0c6e74d6edc
# md"""

# ## Loss view -- "ill-conditioned" loss

# ##### -- where is the minimum?
# """

# ╔═╡ b0d9538d-584c-44ea-8221-3727180d7e17
# let
# 	gr()
# 	ys = x1s + 100 * x2s
# 	XX = [x1s x2s]
# 	plt1 = plot(-5:0.5:9, -30:0.5:30, (w1,w2) -> dot(ys - XX*[w1, w2], ys - XX*[w1, w2]), st=:surface, c=:jet, title="Loss surface: ill-conditioned", xlabel="w1", ylabel="w2", zlabel="loss", colorbar=false)

# 	plt2= plot(-5:0.5:9, -30:0.5:30, (w1,w2) -> dot(ys - XX*[w1, w2], ys - XX*[w1, w2]), st=:contour, c=:jet, title="Loss contour", xlabel="w1", ylabel="w2", zlabel="loss", colorbar=false,ratio=1)

# 	plot(plt1, plt2)
# end

# ╔═╡ 0fe59b22-f12d-4b4d-9d18-0c5b53cc0336
# md"""

# ## What if ``\mathbf{X}^\top\mathbf{X}`` not invertible?


# #### The *implication* in _linear algebra term_

# ```math
# \large
# \mathbf{X}^\top\mathbf{X}:\; \color{red}\text{  not invertible!}
# ```


# ```math
# \large
# \begin{align}
# {\hat{\mathbf{w}} = \underbrace{(\mathbf{X}^\top\mathbf{X})^{-1}}_{\color{red}\text{NOT defined}}\mathbf{X}^\top \mathbf{y}}: \;\; \color{red}{\text{cannot be used!}}
# \end{align}
# ```

# * ##### ``\hat{\mathbf{w}}``: no unique solution
# """

# ╔═╡ 547cec7f-edff-4283-89ee-788a7bda5f16
# md"""

# #### What to do ? 


# * ##### _option 1_: remove the redundant columns (may not be obvious how to)

# * ##### _option 2_: add small constants ``\epsilon`` to the diagonal entries of ``\mathbf{X}^\top\mathbf{X}`` to make it invertible 

# ```math
# \large
# \begin{align}
# {\hat{\mathbf{w}} = \underbrace{(\mathbf{X}^\top\mathbf{X})^{-1}}_{\color{red}\text{NOT defined}}\mathbf{X}^\top \mathbf{y}}: \;\; \color{red}{\text{cannot be used!}}
# \end{align}
# ```

# ```math
# \large
# \begin{align}
# {\hat{\mathbf{w}} = \underbrace{(\mathbf{X}^\top\mathbf{X} + \epsilon \mathbf{I})^{-1}}_{\color{red}\text{usually  invertible}}\mathbf{X}^\top \mathbf{y}}: \;\; \color{red}{\text{can be used!}}
# \end{align}
# ```
# """

# ╔═╡ 31e9f475-119e-4126-b556-89202f161ccd
md"""

# Why _normal_?*

## The *Normal Equation* 

```math
\LARGE\boxed{\nabla L(\mathbf{w}) 
= \mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) = \mathbf{0} }
```


* #### Normal  ``\Longleftrightarrow`` Orthogonal


## Recap

### Recall that ``\mathbf{a} \perp \mathbf{b} \Leftrightarrow \mathbf{a}^\top\mathbf{b} = 0``, due to 
\


```math
\Large
\mathbf{a}^\top\mathbf{b} = \|\mathbf{a}\|_2 \, \|\mathbf{b}\|_2\, \cos\frac{\pi}{2} = 0
```
"""

# ╔═╡ 485ebf22-2307-476c-a448-21e2377513e2
md"""

## Why called _normal_ equation?

#### The *Normal Equation* 

```math
\LARGE\boxed{\nabla L(\mathbf{w}) 
= \mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) = \mathbf{0} }
```


* #### Normal  ``\Longleftrightarrow`` Orthogonal


### Recall that ``\mathbf{X}``'s columns are features
"""

# ╔═╡ f0256550-ad5f-4cfe-ac93-685ab2c584d3
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/houseX.svg" width = "500"/></center>"""

# ╔═╡ 243b9d86-8592-420c-9d0b-659722feaf32
md"""

##

### ``\mathbf{X}^\top``: _rows_ now are features 
"""

# ╔═╡ f111e5f7-36c7-4df2-8eee-944ec770fd66
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/houseXtrans.svg" width = "550"/></center>"""

# ╔═╡ 608883c2-8cf5-4a6e-8260-0799e58b87f5
md"""

#### _In general_, with ``m`` features, we have
```math
\Large
\mathbf{X}^\top= \begin{pmatrix}
\rule[.5ex]{2.5ex}{0.5pt} & & \mathbf{1}^\top &  & \rule[.5ex]{2.5ex}{0.5pt} \\
\rule[.5ex]{2.5ex}{0.5pt} & & \mathbf{x}_1^\top & & \rule[.5ex]{2.5ex}{0.5pt}  \\
& & \vdots & &\\
\rule[.5ex]{2.5ex}{0.5pt} & & \mathbf{x}_m^\top & & \rule[.5ex]{2.5ex}{0.5pt} 
\end{pmatrix}_{m \times n}
```
"""

# ╔═╡ 6de0c68f-fc63-41f5-8615-967bb480d2fe
md"""

#### Denote ``\hat{\boldsymbol{e}} \triangleq \mathbf{Xw} -\mathbf{y}=\hat{\mathbf{y}}-\mathbf{y}`` for convenience, we have

```math
\Large
\mathbf{X}^\top \underbrace{(\mathbf{Xw}-\mathbf{y})}_{\hat{\boldsymbol{{e}}}} =\mathbf{0} 
```
```math
\Large
\begin{align}
&\Rightarrow \begin{pmatrix}
\rule[.5ex]{2.5ex}{0.5pt} & \mathbf{1}^\top & \rule[.5ex]{2.5ex}{0.5pt} \\
\rule[.5ex]{2.5ex}{0.5pt} & \mathbf{x}_1^\top & \rule[.5ex]{2.5ex}{0.5pt}  \\
& \vdots &\\
\rule[.5ex]{2.5ex}{0.5pt} & \mathbf{x}_m^\top & \rule[.5ex]{2.5ex}{0.5pt} 
\end{pmatrix}\begin{bmatrix} 
\vert \\
\hat{\boldsymbol{{e}}} \\
\vert 
\end{bmatrix}=\begin{bmatrix}0 \\ 0 \\ \vdots \\0 \end{bmatrix}\\
&\Rightarrow\begin{cases}
 \mathbf{1}^\top\hat{\boldsymbol{{e}}} &=0  \\
 \mathbf{x}_1^\top\hat{\boldsymbol{{e}}} &=0\\
&\vdots \\
\mathbf{x}_m^\top\hat{\boldsymbol{{e}}}  &=0
\end{cases}
\end{align}
```


> #### Implication: features and prediction errors are _orthogonal_
> ```math
> \Large
> \begin{cases}
> \mathbf{1} \perp \hat{\boldsymbol{{e}}}   \\
> \mathbf{x}_1 \perp \hat{\boldsymbol{{e}}} \\
> \vdots & \\
> \mathbf{x}_m \perp \hat{\boldsymbol{{e}}}  
> \end{cases}
> ```

> ### Now we know why it is called "normal"  
> #### -- _but why least square estimate has to be "normal"?_

"""

# ╔═╡ 107949f6-3fda-43bb-874b-d12605f20380
md"""

## Correlation view


##### Inner product measures correlation
```math
\large 
\mathbf{x}^\top \hat{\mathbf{e}}=0 \Rightarrow \text{residuals } \hat{\mathbf{e}} \text{ not correlated with } \mathbf{x}
```
"""

# ╔═╡ ff2b28fd-5a7c-46b8-ba32-763640b54b9a
md"Change the slope: $(@bind k_v Slider(-0.1:0.05:2.05, default=0)), Show inner product: $(@bind showinner CheckBox(false)), Set least square: $(@bind setlsq CheckBox(false))"

# ╔═╡ 21458c5d-ffc8-483a-a375-477f5fead1f8
TwoColumn(begin
	gr()
	Random.seed!(123123123123)
	xsv = 1:6
	truek = 1
	ysv = truek * xsv + repeat([1,-1], 3) .* randn(length(xsv))

	k̂ = xsv \ ysv
	if setlsq
		k = k̂
	else
		k = k_v
	end
	
	
	plt_normal = plot(-1:1:7, x -> k*x, framestyle=:zerolines,lc=1, lw=2, label="fitted line",xlim =[-1,6.5],ratio=0.5, ylim =[-1, 12.5], legend=:topleft, size=(350,350))
	scatter!(xsv, ysv, c=1,  label="training data")
	ŷs = k * xsv
	for i in 1:length(xsv)
		plot!([xsv[i], xsv[i]], [ysv[i], ŷs[i] ], arrow =:both, lc=:orange, lw=1.25, label="")
		annotate!([xsv[i]+0.05], 0.5 * (ysv[i]+ ŷs[i]) +0.4, Plots.text(L"\hat{e}^{(%$(i))}",:red,  :left))
	end

	ê = ŷs - ysv 
	if showinner
		if setlsq
			title!("Least square fit: "*L"\mathbf{x}^\top\hat{\mathbf{e}} = %$(round(xsv'*ê; digits=2))")
			plot!(ratio=1, xlim =[-1,6.5], ylim =[-1, 6.9])

		else
			title!(L"\mathbf{x}^\top\hat{\mathbf{e}} = %$(round(xsv'*ê; digits=2))")
		end
			
	end
	plt_normal
end, 
md"""
\
\

``\;\;\;\;\;{\mathbf{x}} =\begin{bmatrix}1\\2\\3\\4\\5\\6 \end{bmatrix}\;\;\;\;\;\;\;\hat{\mathbf{e}}=``$(latexify_md(round.(ê; digits=1)))""")

# ╔═╡ c71e94e9-39d9-4b25-ab14-a0ad65b338ea
md"""


## A better view: projection view
\


> #### Normal equation: _project_ ``\mathbf{y}`` to _columns_ of ``\mathbf{X}``



## Digress: solve ``\mathbf{y}=\mathbf{Xw}``


!!! warning ""
	##### We are often asked to solve

	$$\Large \mathbf{y} =\mathbf{Xw}\;\;  \text{for } \mathbf{w} \in \mathbb{R}^m$$



#### *For example*

```math
\Large
\underbrace{\begin{bmatrix}3 \\ 1\end{bmatrix}_{2\times 1}}_{\mathbf{y}} = \underbrace{\begin{bmatrix}2 & 0 \\ 0 & 3\end{bmatrix}_{2\times 2}}_{\mathbf{X}}\underbrace{\begin{bmatrix}\columncolor{\lightsalmon}w_1 \\ w_2 \end{bmatrix}_{2\times 1}}_{\mathbf{w}} 
```





"""

# ╔═╡ b73ee6cb-f6e0-4a0d-afac-24148447425e
aside(tip(md"If ``\mathbf{X}`` is **invertible**, the solution is simple

```math
\large
\mathbf{w} = \mathbf{X}^{-1}\mathbf{y}
```
"))

# ╔═╡ d5345afe-84e6-4922-a800-e8289a95d048
md"""

## _For example_, linear regression


#### *Linear regression* is a special case with ``\approx``


```math
\Large
{\begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\\vdots \\ y^{(n)}\end{bmatrix}} \approx  \begin{bmatrix}  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(1)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(2)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\ & \vdots & \\   & \vdots &\\\rule[.5ex]{2.5ex}{0.5pt} &(\mathbf{x}^{(n)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\end{bmatrix}_{n>m} \begin{bmatrix} \vert \\ \mathbf{w} \\\vert\end{bmatrix}
```
\

* ##### ``n > m``, over-determined and *no* exact solution 
  * ``n``: number of observations
  * ``m``: number of features
* ##### but we want small difference, *i.e.* ``\approx`` 



### This also explains `Julia`/`Matlab`'s `\` operator
* ##### matrix left divide: ` w = X \ y`
"""

# ╔═╡ b2a10215-7595-4a92-94ac-dad3fe80e8f6
md"""


## Recap: matrix vector product ``\mathbf{Xw}``


#### Note that ``\mathbf{X}`` is  a collection of ``\large m`` *column (feature) vectors*

```math
\Large
\mathbf{X}  =\begin{bmatrix}
	\vert & \vert &  & \vert\\
	\mathbf{x}_1 & \mathbf{x}_2 & \ldots & \mathbf{x}_{m} \\
	\vert & \vert & & \vert 
	\end{bmatrix}\;  \text{and}\; \mathbf{w} =\begin{bmatrix}
	w_1\\
	w_2 \\
	\vdots\\
	
	w_m
	\end{bmatrix}

```



```math
\large

```


!!! important "Matrix vector: linear combo view"	
	```math
	\begin{align}
	\mathbf{X}\mathbf{w} &=  \begin{bmatrix}
	\vert & \vert &  & \vert\\
	\mathbf{x}_1 & \mathbf{x}_2 & \ldots & \mathbf{x}_{m} \\
	\vert & \vert & & \vert 
	\end{bmatrix}  \begin{bmatrix}
	w_1\\
	w_2 \\
	\vdots\\
	w_m
	\end{bmatrix}  =  w_1\begin{bmatrix}
           \vert\\
           \mathbf{x}_{1}\\
           \vert
         \end{bmatrix} + w_2\begin{bmatrix}
           \vert\\
           \mathbf{x}_{2}\\
           \vert
         \end{bmatrix} + \ldots w_m\begin{bmatrix}
           \vert\\
           \mathbf{x}_{m}\\
           \vert
         \end{bmatrix}\\
	& = \sum_{i=1}^m w_i \mathbf{x}_i
	\end{align}
	```
	##### In summary, 
	* ##### ``\mathbf{Xw}`` is a linear combination of the column vectors of ``\mathbf{X}``, 
	* ##### _where_ ``\mathbf{w}`` are the coefficients


"""

# ╔═╡ c6c78c39-f9c3-4bb4-96d7-5c447b96130a
md"""

## Example
"""

# ╔═╡ 113f4be3-0ae8-4acf-a04b-aba8bac5449c
md"
Add combination $(begin @bind add_u CheckBox(default=false) end) ;
Add column space $(begin @bind add_av CheckBox(default=false) end) 
"

# ╔═╡ 149acbd6-6200-4d2d-9aa3-a5b6f6e6fd86
md"""

 
``w_1`` = $(@bind v₁_ Slider(-1.5:0.1:1.5, default=.5, show_value=true)) 
``w_2`` = $(@bind v₂_ Slider(-1.5:.1:1.5, default=-.9, show_value=true))

"""

# ╔═╡ be7bcc2a-3cef-40a9-84d9-6afedfae2c4b
md"rotate: $(@bind ang11 Slider(-90:1:90, default=20)); up/down: $(@bind ang21 Slider(-90:1:90, default=25))"

# ╔═╡ d18a8d63-80e0-4a6a-bf6e-96b857721de8
Foldable("The column space", md"
> The column space ?
> 
>  ``\{w_1 \textcolor{red}{\mathbf{x}_1} + w_2 \textcolor{green}{\mathbf{x}_2}\}:`` the whole shaded **plane**

")

# ╔═╡ 5b2a8f5a-74da-464e-a3bf-172454642d8f
begin
	a1 = [1.5, 3, 0]
	a2 = [3, 0, 0]
end;

# ╔═╡ 12bab87e-f3f0-4124-b48d-677f47bba686
# md"""
# ## Geometric interpretation of ``\mathbf{y}=\mathbf{Xw}``


# !!! warning "Question"
# 	##### What is the *geometric interpretation* of solving

# 	$$\Large \mathbf{y} = \mathbf{Xw}$$ 
# 	##### *for*  ``\mathbf{w} \in \mathbb{R}^m`` ?

	

# """

# ╔═╡ 362400de-d0e3-4fc2-a99e-a1e51eb005a8
# md"""
# !!! answer "Answer"
# 	```math
# 	\Large
# 		\begin{bmatrix} \vert\\ \mathbf{y} \\ \vert\end{bmatrix} = w_1\begin{bmatrix}
#            \vert\\
#            \mathbf{x}_{1}\\
#            \vert
#          \end{bmatrix} + w_2\begin{bmatrix}
#            \vert\\
#            \mathbf{x}_{2}\\
#            \vert
#          \end{bmatrix} + \ldots w_m\begin{bmatrix}
#            \vert\\
#            \mathbf{x}_{m}\\
#            \vert
#          \end{bmatrix}  
# 	```
# 	> ##### Does ``\mathbf{y}`` lives in the column space of ``\mathbf{X}``?
# 	* ###### if so, there is (are) solutions(s)
# 	* ###### _otherwise_, there is no **exact** solution



# """

# ╔═╡ 998c9b63-36f3-496b-9db7-6913e7c67805
md"""

## Geometric view of 

#### _Solving_ ``\mathbf{y} = \mathbf{Xw}``

"""

# ╔═╡ 0c0894be-2042-4797-9c20-7fc1239a2cad
md"
Add ``\mathbf{y}`` $(begin @bind add_b CheckBox(default=false) end) ,
Add solution $(begin @bind add_sol CheckBox(default=false) end) 

rotate vertically: $(@bind ang2 Slider(-90:1:90, default=25)),
rotate horizontally: $(@bind ang1_ Slider(0:1:90, default=50))
"

# ╔═╡ de76e3a6-5e59-4dd1-8704-0fa540e6d34d
bv = [3.,1.5,0];

# ╔═╡ caed0f95-84cb-4d3a-929f-85b6d61280d7
let

 	A = [a1 a2]

	A * (A \ bv) ≈ bv

end;

# ╔═╡ 98615427-e2dd-4268-8534-c95ac192cc9e
md"""

## Geometric view of 

#### _Solving_ ``\mathbf{y} = \mathbf{Xw}``

"""

# ╔═╡ 28487f32-ce91-4ba8-a544-cb1c8bffb73e
Foldable("The approx. solution is actually", md"
* *i.e.* its **projection** 
")

# ╔═╡ 3477d5f8-1e27-45bf-a675-d781e478c302
md"
Add ``\hat{\mathbf{y}}=\mathbf{Xw}`` $(begin @bind add_proj CheckBox(default=false) end) 
"

# ╔═╡ 514c9139-7c19-4615-8ce5-9fd1b6e2c2f6
md"""

 
``w_1`` = $(@bind v₁ Slider(0:0.5:5, default=2)) 
``w_2`` = $(@bind v₂ Slider(0:0.5:5, default=2))
rotate: $(@bind ang1 Slider(-90:1:90, default=20))
"""

# ╔═╡ bf43112d-22fc-4782-9112-d587bce9dd76
bp = [v₁,v₂,0];

# ╔═╡ c7ad6de9-0d7d-4343-b94d-d7a9cb7785da
md"""

## Recap: simple projection


"""

# ╔═╡ 121fdd48-6372-4ca9-bac9-6664a1664098
md"""

## General projection


> #### The error vector should be ``\perp`` to the column space vectors
"""

# ╔═╡ 3d22ee0d-9fdf-4b43-9a05-6a905a4cd603
md"""

## General projection


> ##### To project, the error vector ``\perp`` the columns of ``\mathbf{X}``

##### In maths,

```math
\large
\underbrace{(\hat{\mathbf{y}} -\mathbf{y})}_{\hat{\boldsymbol{e}}} \perp \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_m\}
```

##### Or equivalently, by inner product

```math
\large
\begin{align}
\mathbf{x}_1^\top (\hat{\mathbf{y}} -\mathbf{y}) &= 0\\
\mathbf{x}_2^\top (\hat{\mathbf{y}} -\mathbf{y})  &= 0\\
&\vdots\\
\mathbf{x}_m^\top (\hat{\mathbf{y}} -\mathbf{y})  &= 0\\
\end{align}
```


"""

# ╔═╡ 37692f49-6a9d-42e4-9611-edca98360ea8
md"""



##### *In matrix* notations, 
> ```math
> \large
> \begin{align}
> \mathbf{X}^\top \underbrace{(\mathbf{Xw} -\mathbf{y})}_{\hat{\mathbf{y}} -\mathbf{y}}= \mathbf{0} 
> \end{align}
> ```

"""

# ╔═╡ 8101f514-f5b3-4468-bc66-d9863c0d1111
md"""

## Reading & references

##### Essential reading 


* [_Understanding deep learning_ by _Simon Prince._: Chapter 2](https://github.com/udlbook/udlbook/releases/download/v.1.20/UnderstandingDeepLearning_16_1_24_C.pdf)


##### Suggested reading 
* [_Machine Learning refined_ by Jeremy Watt, Reza Borhani and Aggeos Katsaggelos](https://github.com/jermwatt/machine_learning_refined/blob/gh-pages/sample_chapters/2nd_ed/chapter_5.pdf) Chapter 5.1-5.2



* [_Pattern recognition and Machine Learning_ by _Chris Bishop._: Chapter 3.1-3.2](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)


"""

# ╔═╡ 70836590-75d0-4091-908c-3e164f6142f3
# md"""


# ## A related problem
# """

# ╔═╡ b35340f9-6d86-4cd7-bb5d-75a876378656
# md"""
# Add solutions: $(@bind add_fit_line CheckBox(default=false)); 
# """

# ╔═╡ 444b1a52-3beb-4d08-9cb5-99def7da04ba
# let
# 	gr()
# 	xs = -1:1:6
# 	plt = scatter([3], [3],   ms=8, xlim =[-2, 6], ylim =[0, 6],xlabel=L"x", ylabel=L"y", label ="training data",framestyle=:origin, title="How many solutions?")
# 	if add_fit_line
# 		for k in -2:0.1:2
# 			# 3k + b = 3
# 			b = 3 - 3*k
# 			plot!(xs, (x) -> k *x + b, label="", lw=1.5, alpha=0.4)
# 		end
# 		title!("How many solutions? infinite!")
# 	end
# 	plt
# end

# ╔═╡ da548f5b-9912-4044-89c6-af431d84169f
# md"""
# ## Regularisation

# !!! question "Question"
# 	#### Which $$h(x)=w_0 + wx$$ shall I choose?
# """

# ╔═╡ ff12f43e-a2a7-4fc0-993a-16cb09052c55
# Foldable("Answer", md"""


# ```math
# \Large
# h(x) = 0 \cdot x + b\;\; \text{is a safe choice!}
# ```




# #### *i.e.* assume there is no relationship
# """)

# ╔═╡ 10e1a07d-e031-46ae-ad35-6ee6f674f40d
# md"""
# The slope ``w=``$(@bind kk Slider(-2:0.05:2, default = 1.0; show_value=true))
# """

# ╔═╡ 76d44e52-2ee2-4e34-aa07-c77b529b81e6
# let
# 	gr()
# 	xs = -1:1:6
# 	plt = scatter([3], [3],  label ="training data",ms=8,  xlim =[-2, 6], ylim =[0, 6],xlabel=L"x", ylabel=L"y", framestyle=:origin, xticks = -2:1:6)

# 	for k in -2:0.15:2
# 		# 3k + b = 3
# 		b = 3 - 3*k
# 		plot!(xs, (x) -> k *x + b, label="", lw=0.5, alpha=0.4)
# 	end
# 	title!("Which " * L"h(x) = %$(kk)x+w_0" *" shall I choose?")
	
# 	plot!(xs, (x) -> kk *x + (3 -3* kk), label="", lw=3, c=:blue)
# 	plt
# end

# ╔═╡ e3d3a4f5-5297-4cd8-b975-139162f71692
# md"""

# ## Regularisation


# To reinforce it, add a **penalty** term to make ``w`` close to 0!


# ```math
# \large
# L(\mathbf{w}) = \frac{1}{2}\sum_{i=1}^n (y^{(i)} - wx^{(i)} -w_0)^2 + \boxed{\frac{\lambda}{2} w^2}_{\text{penalty term}}
# ```


# * ``\lambda > 0``:  a hyperparameter

# * we do not usually penalise the bias/intercept ``w_0``
# """



# ╔═╡ cb2e4c43-c1a1-43e4-8f8d-65a5ec8679a6
# md"""

# ## Regularisation

# More generally,


# ```math
# \large
# L(\mathbf{w}) = \underbrace{\frac{1}{2}\sum_{i=1}^n (y^{(i)} - h(\mathbf{x}^{(i)}; \mathbf{w}))^2}_{\text{previous loss}} + \boxed{\frac{\lambda}{2} \sum_{j=1}^{m} w_j^2}_{\text{penalty term}}
# ```
# * where ``\lambda \geq 0`` is a hyperparameter


# ##

# Or in matrix notation


# ```math
# \large
# L(\mathbf{w}) = \frac{1}{2}  (\mathbf{y}-\mathbf{Xw})^\top (\mathbf{y}-\mathbf{Xw}) + \boxed{\frac{\lambda}{2} \mathbf{w}^\top \mathbf{w}}
# ```
# * recall that ``\mathbf{w}^\top \mathbf{w} = \sum_j w_j^2``


# > And this is called **`Ridge` regression**.

# ##

# ###### _How_ and _Why_ it works ?


# ```math
# \large 

# \mathbf{w}^\top \mathbf{w} = (\mathbf{w} -\mathbf{0})^\top (\mathbf{w} - \mathbf{0})

# ```

# * essentially measures the squared distance from ``\mathbf{w}`` to ``\mathbf{0}``

# * large ``w_j \neq 0`` therefore are penalised



# """

# ╔═╡ c5a180b1-0424-48fe-b33c-d1706051c95d
# aside(tip(md"""

# Recall 

# ```math
# \nabla_{\mathbf{w}} \mathbf{w}^\top\mathbf{w} = 2 \mathbf{w}
# ```

# """))

# ╔═╡ e5954020-5d7a-4f7f-8bee-056a77f2a12f
# md"""
# ## Ridge regression -- learning

# The problem now is to optimise the regularised loss

# ```math
# \large 
# \mathbf{w}_{ridge} \leftarrow \arg\min_{\mathbf{w}} \frac{1}{2}  (\mathbf{y}-\mathbf{Xw})^\top (\mathbf{y}-\mathbf{Xw}) + \frac{\lambda}{2} \mathbf{w}^\top \mathbf{w}
# ```

# * its gradient is 


# ```math
# \nabla L(\mathbf{w}) = \mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) + \lambda\mathbf{w}
# ```

# * gradient descent can be used
#   * very easy to implement: just add ``\lambda \mathbf{w}`` 
#   * it is known as weight decay
# """

# ╔═╡ 1fc91017-1b9a-4dd2-986b-ee48eeda0b17
# md"""


# ## Alternatively, closed form solution

# Alternatively, set the gradient to zero, and solve it


# !!! exercise "Ridge regression solution"
# 	Show that the closed form solution for ridge regression is
# 	```math
# 	\large
# 		\mathbf{w}_{ridge} = (\mathbf{X}^\top\mathbf{X} +\lambda \mathbf{I})^{-1} \mathbf{X}^\top\mathbf{y}
# 	```



# !!! hint "Hint"
# 	```math
# 		\lambda \mathbf{w} = \lambda \mathbf{Iw},
# 	```
# 	where ``\mathbf{I}`` is the identity matrix.

# """

# ╔═╡ a633e6af-7d71-4234-81c5-df861ccda5ee
# md"""

# ## Gradient descent 

# ##### We can also use *gradient descent* to **minimise** the loss

# ```math
# \large
# \hat{\mathbf{w}} \leftarrow \arg\min_{\mathbf{w}} L(\mathbf{w})
# ```


# * recall the gradient is 
# ```math
# \Large
# \begin{align}
# \nabla L(&\mathbf{w}) 
# = \mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) 
# \end{align}

# ```

# * gradient descent:

# ```math
# \LARGE
# \mathbf{w}_{new} = \mathbf{w}_{old} - \underbrace{\colorbox{lightgreen}{$\gamma$}}_{{\color{green}\small \rm learning\; rate}} \nabla L(\mathbf{w}_{old})
# ```

# """

# ╔═╡ 882ba5fd-d533-453f-9d34-bcde5623a1ff
# md"""
# ## Demo: gradient descent 


# Gradient descent with
# * ``\mathbf{w}_0 = \mathbf{0}``: a horizontal plane
# * learning rate ``\gamma=0.01``
# * it converges to the same result as the normal equation solution
# """

# ╔═╡ 42037da5-792c-407e-935f-534bf35a739b
# function ∇L(w, X, y)
# 	# gradient of linear regression
# 	X' * (X*w -y)
# end

# ╔═╡ 89a8e5bd-f049-491e-a3ff-9e33bfccb289
# gif(anim, fps=8)

# ╔═╡ 2b4a0dc3-8bad-4589-ae97-746bdd705050
# ws_history, losses=let
# 	∇l(x) = ∇L(x,  X_train, y_train)
# 	max_iters = 2000
# 	losses = []
# 	# random starting point
# 	w₀ = zeros(num_features+1)
# 	push!(losses, loss(w₀, X_train, y_train))
# 	ws_history = zeros(num_features+1, max_iters+1)
# 	ws_history[:, 1] = w₀
# 	γ = 0.01
# 	for i in 1:max_iters
# 		w₀ = w₀ - γ * ∇l(w₀)
# 		push!(losses, loss(w₀, X_train, y_train)) # book keeping; optional
# 		ws_history[:, i+1] = w₀ # book keeping; optional
# 	end
# 	ws_history, losses
# end;

# ╔═╡ 5f70d015-c2c6-4c45-badf-688613f0ea69
# anim = let
# 	gr()
# 	w_lse = least_square_est(X_train, y_train)
# 	anim = @animate for i in [1:15; 16:50:1000; 1001:100:2000]
# 		# plot(1:10)
# 		scatter(X_train[:, 2], X_train[:,3], y_train, markersize=2, label="", zlim =[2.,12])
# 		w0 = ws_history[:, i]
# 		surface!(0:0.5:1, 0:0.5:1.0, (x1, x2) -> dot([1, x1, x2], w0), c=:jet,  colorbar=false, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"y", alpha=0.5, title="Iteration "*string(i), display_option=Plots.GR.OPTION_Z_SHADED_MESH)

# 		surface!(0:0.5:1, 0:0.5:1.0, (x1, x2) -> dot([1, x1, x2], w_lse))
# 	end
# end;

# ╔═╡ e17f634a-64a8-4271-9657-248fb2b5c688
# md"""

# ## What gradient descent is doing?



# #### The _gradient_ is


# ```math
# \large
# \nabla L(\mathbf{w}) =  -\sum_{i=1}^n \underbrace{(y^{(i)} - \mathbf{w}^\top\mathbf{x}^{(i)})}_{\text{prediction error: } e^{(i)}}\, \cdot \,\mathbf{x}^{(i)}
# ```

# #### Let's gain some _insights_ of *gradient descent*


# * consider ``\mathbf{x}^{(i)}, y^{(i)}`` only, the gradient is
# ```math
# \large
# \begin{align}
# \nabla L^{(i)}(\mathbf{w}_{old}) &=  -\underbrace{(y^{(i)} - \mathbf{w}_{old}^\top\mathbf{x}^{(i)})}_{\text{prediction error: } e^{(i)}} \, \cdot \, \mathbf{x}^{(i)} \\
# &= - e^{(i)}\cdot \mathbf{x}^{(i)}

# \end{align}
# ```

# * gradient descent applies the **negative** gradient direction, which becomes 

# ```math
# \mathbf{w}_{new} \leftarrow \mathbf{w}_{old} -\gamma  (-e^{(i)}\cdot \mathbf{x}^{(i)})
# ``` 

# ```math
# \large
# \boxed{\mathbf{w}_{new} \leftarrow \mathbf{w}_{old} + \gamma\cdot e^{(i)} \cdot \mathbf{x}^{(i)}\;\; \# \texttt{gradient step}}
# ```
# * where ``e^{(i)} = y - \mathbf{w}_{old}^\top\mathbf{x}^{(i)}`` is the prediction error
# """

# ╔═╡ 635fc846-2711-41d4-b452-7a96e1a167c1
# md"""

# ## What gradient descent is doing?

# ```math
# \Large
# \boxed{\mathbf{w}_{new} = \mathbf{w}_{old} + \gamma\cdot e \cdot\mathbf{x}\;\; \# \texttt{gradient step}}
# ```


# * where ``e = y - \mathbf{w}_{old}^\top\mathbf{x}``

# After the update, the **new prediction** becomes

# ```math
# \large
# \begin{align}\hat{h}_{new} &=\mathbf{w}_{new}^\top \mathbf{x} = \mathbf{w}_{old}^\top \mathbf{x} + \gamma\cdot e\cdot {\mathbf{x}}^\top \mathbf{x}\\
# &=\colorbox{pink}{$\hat{h}_{old} + \boxed{\gamma}_{\small >0}\cdot e\cdot \boxed{{\mathbf{x}}^\top \mathbf{x}}_{\geq 0}$}
# \end{align}
# ```


# ##### when error ``e \approx 0``, or ``y \approx \mathbf{w}_{old}^\top\mathbf{x}  ``, the current model is perfect 
# * _GD_ makes no change


# ##### when error ``e > 0``, or ``y >\mathbf{w}_{old}^\top\mathbf{x}``, the current model under predicts 
# * GD **increases** the prediction by a little next time

# ##### when error ``e < 0``, or ``y < \mathbf{w}_{old}^\top\mathbf{x}``, the current model  over predicts  
# * GD **decreases** the prediction by a little next time
# """

# ╔═╡ f65644e7-cb25-46ad-b146-87cf7de69f72
# md"""

# ## Polynomial regression

# The following function seems fitting the data better: a **quadratic function**
# ```math
# \large
# h(x) = w_0 + w_1 x + w_2 x^2
# ```


# """

# ╔═╡ 1581b975-6c0a-4bc1-8dc1-a6ea273e14f9
# quadratic_fit

# ╔═╡ f8f34671-ef4c-4300-a983-10d42d43fb9f
# quadratic_fit=let
# 	gr()
# 	@df df_house scatter(:RM, :target, xlabel="room", ylabel="price", label="", title="House price prediction: non-linear regression")
# 	x_room = df_house[:, :RM]
# 	x_room_sq = x_room.^2
# 	X_train_room = [ones(length(x_room)) x_room x_room_sq]
# 	c, b, a = linear_reg_normal_eq(X_train_room, df_house.target)

# 	plot!(3.5:0.5:9, (x) -> a* x^2 + b* x+ c, lw=3, label=L"h(\mathbf{x})", legend=:outerbottom)
# end;

# ╔═╡ 22cbb1aa-c53f-45d6-891a-90c6f2b9e886
# md"""

# ## Free lunch -- fixed basis expansion


# ##### *First*, expand the features as new columns
# * for each ``x^{(i)}``, input room, expand another feature ``(x^{(i)})^2``

# ```math
# \large
# \mathbf{X} = \begin{bmatrix}1 & x^{(1)} & \columncolor{pink}(x^{(1)})^2 \\
# 1 & x^{(2)} & (x^{(2)})^2 \\
# \vdots & \vdots & \vdots \\
# 1 & x^{(n)} & (x^{(n)})^2
# \end{bmatrix}

# ```

# ##### *Then*, linear model's prediction ``\hat{\mathbf{y}}``
# ```math
# \large
# \hat{\mathbf{{y}}} =\mathbf{Xw} = \begin{bmatrix}1 & x^{(1)} & \columncolor{pink}(x^{(1)})^2 \\
# 1 & x^{(2)} & (x^{(2)})^2 \\
# \vdots & \vdots & \vdots \\
# 1 & x^{(n)} & (x^{(n)})^2
# \end{bmatrix}\begin{bmatrix}w_0\\ w_1 \\ w_2\end{bmatrix} = \begin{bmatrix}w_0 + w_1 x^{(1)} + w_2 (x^{(1)})^2 \\
# w_0 + w_1 x^{(2)} + w_2 (x^{(2)})^2 \\
# \vdots  \\
# w_0 + w_1 x^{(n)} + w_2 (x^{(n)})^2
# \end{bmatrix}

# ```


# ##### *Lastly*, regress with the expanded design matrix (now a ``n \times 3`` matrix)

# ```math
# \large
# \hat{\mathbf{w}} \leftarrow \arg\min_{\mathbf{w}} \frac{1}{2} (\mathbf{y} -\hat{\mathbf{y}})^\top (\mathbf{y} -\hat{\mathbf{y}})
# ```

# """

# ╔═╡ ce35ddcb-5018-4cb9-b0c9-01fb4b14be40
# begin
# 	x_room = df_house[:, :RM]
# 	x_room_sq = x_room.^2 # squared x_room^2
# 	X_room_expanded = [ones(length(x_room)) x_room x_room_sq]
# end;

# ╔═╡ 59a3e04f-c842-4b6d-b067-0525c9dda70b
# md"Then fit with normal equation:";

# ╔═╡ 660b612b-fbc6-434a-b8ae-69f1213dfad4
# (X_room_expanded' * X_room_expanded)^(-1) * X_room_expanded' * df_house.target;

# ╔═╡ f3e404b7-3419-43e7-affa-217324a65534
# quadratic_fit;

# ╔═╡ 2cdd8751-7ec8-47b0-a174-fdc23e176921
# md"""

# ## Higher orders?


# ```math
# \large
# h(x) = w_0 + w_1 x + w_2 x^2 +\ldots + w_p w^p
# ```
#  * still **free lunch**: regress with a ``n\times (p+1)`` matrix

# """

# ╔═╡ 720774c4-9aec-4329-bc90-51350fea0191
# md"""

# ```math
# \large
# \mathbf{X} = \begin{bmatrix}1 & x^{(1)} & (x^{(1)})^2  & \ldots & (x^{(1)})^p \\
# 1 & x^{(2)} & (x^{(2)})^2 & \ldots &(x^{(2)})^p\\
# \vdots & \vdots & \vdots & \ddots & \vdots \\
# 1 & x^{(n)} & (x^{(n)})^2 & \ldots & (x^{(n)})^p
# \end{bmatrix}
# ```
# """

# ╔═╡ edc245bc-6571-4e65-a50c-0bd4b8d63b74
# function poly_expand(x; order = 2) # expand the design matrix to the pth order
# 	n = length(x)
# 	return hcat([x.^p for p in 0:order]...)
# end;

# ╔═╡ 4154585d-4eff-4ee9-8f33-dad0dfdd143c
# function poly_reg(x, y; order = 2) # fit a polynomial regression to the input x; x is assumed a vector
# 	X = poly_expand(x; order=order)
# 	# a better method to do least square 
# 	w = X \ y
# 	l = loss(w, X, y)
# 	return w, l
# end;

# ╔═╡ d5cc7ea6-d1a0-46e1-9fda-b7f7092eb73c
# poly_reg(x_room, df_house.target; order=3);

# ╔═╡ f90957ea-4123-461f-9a3e-ae23a2848264
# md"The polynomial order:"

# ╔═╡ 8b596e98-4d0c-471e-bb25-20f492a9199b
# @bind poly_order Slider(0:25, default =2, show_value=true)

# ╔═╡ 9aff2235-7da4-41a9-9926-4fe291d9a638
# let
# 	gr()
# 	step = 10
# 	w, l = poly_reg(x_room[1:step:end], df_house.target[1:step:end]; order=poly_order)
# 	@df df_house scatter(:RM, :target, xlabel="room", ylabel="price", label="", title="Poly-regression order: "* L"%$(poly_order)" *"; training loss = "*L"%$(round(l/length(df_house.target); digits=2))")
# 	plot!(3:0.05:9, (x) -> poly_fun(x, w), lw=2, label=L"h(\mathbf{x})", legend=:outerbottom, ylim=[-5, 65])
# end

# ╔═╡ 72440a56-7a47-4a32-9244-cb0424a6fd79
# md"


# ##### Note that the _loss_ here is the training loss
# * sum of squared error on the training data
# " 

# ╔═╡ 46180264-bddc-47d8-90a7-a16d0ea87cfe
# poly_fun(x, w) = sum([x^p for p in 0:length(w)-1] .* w);

# ╔═╡ 882da1c1-1358-4a40-8f69-b4d7cbc9387e
# md"""

# ## Another dataset


# ##### The true function is

# ```math
# \large h(x) = -2 x + 2 x^2
# ```
# """

# ╔═╡ 0dff10ec-dd13-4cc2-b092-9e72454763cc
# begin
# 	gr()
# 	Random.seed!(123)
# 	x_poly = [range(-1.3, -0.5, length=8)... range(.5, 1.3, length=8)...][:]
# 	x_poly_test = -2 : 0.1: 2
# 	w_poly = [0,-2,2]
# 	y_poly = [poly_fun(x, w_poly) for x in x_poly] + randn(length(x_poly))
# 	y_poly_test = [poly_fun(x, w_poly) for x in x_poly_test] + randn(length(x_poly_test))
# 	plot(x_poly, y_poly, st=:scatter, label="training data")

# 	plot!(x_poly_test, y_poly_test, st=:scatter, lc=2, alpha= .5, label="testing data")
# 	plot!(-2:0.1:2, (x) -> poly_fun(x, w_poly), label="true "*L"h(x)", lw=2, lc=2, size=(600, 400))
# end

# ╔═╡ 726a1008-26e6-417d-a73a-57e32cb224b6
# md"""

# ## Overfitting

# ###### Polynomial regression with orders: ``\large p = 1, 2, 4, 7, 10, 15``
# """

# ╔═╡ 8cc45465-78e7-4777-b0c7-bb842e4e51a8
# let
# 	gr()
# 	poly_order = [1, 2, 4, 7, 10, 15]

# 	plots_ =[]
# 	for p in poly_order
# 		plt = plot(x_poly, y_poly, st=:scatter, label="")
# 		w, loss = poly_reg(x_poly, y_poly; order=p)
# 		plot!(-1.5:0.02:1.5, (x) -> poly_fun(x, w), lw=2, lc=:red,  label="Order: "*L"p=%$(p)", legend=:outerbottom, ylim=[-1, 7], title="training loss: "*L"%$(round(loss; digits=2))")
# 		push!(plots_, plt)
# 	end
# 	plot(plots_..., size=(900, 600))
# end

# ╔═╡ fc2206f4-fd0f-44f8-ae94-9dae77022fab
# md"""


# !!! note "Overfitting"
# 	Higher-order models lead to overly complicated models
#     * checking training errors only does not help
#     * training error always favours complicated models
# """

# ╔═╡ c91fc381-613c-4c9b-98b6-f4dd897a69a5
# md"""
# ## Testing performance


# For ``({x}^{(i)}, y^{(i)}) \in \mathcal{D}_{test}``:
# ```math
# L_{test}(\mathbf{w}) = \frac{1}{2}\sum_{i=1}^n (y^{(i)} - h(\mathbf{x}^{(i)}; \hat{\mathbf{w}}))^2
# ```
# * where ``\hat{\mathbf{w}}`` is estimated by the least squared method
# """

# ╔═╡ 21b6b547-66ed-4231-830b-1a09adaf400c
# let
# 	gr()
# 	poly_order = [1, 2, 4, 7, 10, 15]


# 	plots_ =[]
# 	for p in poly_order
# 		plt = plot(x_poly, y_poly, st=:scatter, ms=3, mc=1,alpha=0.5, label="train data")
# 		plot!(x_poly_test, y_poly_test, st=:scatter, ms=3, mc=2,alpha=0.5,  label="test data")
# 		w, loss = poly_reg(x_poly, y_poly; order=p)
# 		loss_test = norm([poly_fun(x, w) for x in x_poly_test] - y_poly_test)/length(y_poly_test)
# 		plot!(-1.5:0.01:1.5, (x) -> poly_fun(x, w), lw=2, lc=:red,  label="Order: "*L"p=%$(p)", legend=:outerbottom, ylim=[-1, 7], title="test loss: "*L"%$(round(loss_test; digits=2))")
# 		push!(plots_, plt)
# 	end
# 	plot(plots_..., size=(900, 600))
# end

# ╔═╡ 2ee96524-7abc-41af-a38c-f71635c738dc
# md"""

# ## Regularisation



# One technique to avoid **overfitting** is **regularisation**

# \

# Regularisation: add a **penalty** term 

# ```math
# \large
# L(\mathbf{w}) = \underbrace{\frac{1}{2}\sum_{i=1}^n (y^{(i)} - h(\mathbf{x}^{(i)}; \mathbf{w}))^2}_{\text{previous loss}} + \boxed{\frac{\lambda}{2} \sum_{j=1}^{m} w_j^2}_{\text{penalty term}}
# ```
# * where ``\lambda \geq 0`` is a hyperparameter


# ##

# Or in matrix notation


# ```math
# \large
# L(\mathbf{w}) = \frac{1}{2}  (\mathbf{y}-\mathbf{Xw})^\top (\mathbf{y}-\mathbf{Xw}) + \boxed{\frac{\lambda}{2} \mathbf{w}^\top \mathbf{w}}
# ```

# > And this is called **`Ridge` regression**.

# ##

# ###### _How_ and _Why_ it works ?
# * the idea: large ``\|\mathbf{w}\|_2 = \sqrt{ \mathbf{w}^\top \mathbf{w}}`` implies very wiggly prediction function
# * large ``w_j``s therefore are penalised

# Recall that ``\mathbf{w}^\top \mathbf{w} = \sum_j w_j^2``


# """

# ╔═╡ 8436c5ce-d7be-4c02-9e89-7da701303263
# aside(tip(md"""

# Recall 

# ```math
# \nabla_{\mathbf{w}} \mathbf{w}^\top\mathbf{w} = 2 \mathbf{w}
# ```

# """))

# ╔═╡ 6a9e0bd9-8a2a-40f0-aaac-bf32d39ffac8
# md"""
# ## Ridge regression -- learning

# The problem now is to optimise the regularised loss

# ```math
# \large 
# \mathbf{w}_{ridge} \leftarrow \arg\min_{\mathbf{w}} \frac{1}{2}  (\mathbf{y}-\mathbf{Xw})^\top (\mathbf{y}-\mathbf{Xw}) + \frac{\lambda}{2} \mathbf{w}^\top \mathbf{w}
# ```

# * its gradient is 


# ```math
# \nabla L(\mathbf{w}) = \mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) + \lambda\mathbf{w}
# ```

# * gradient descent can be used
#   * very easy to implement: just add ``\lambda \mathbf{w}`` 
#   * it is known as weight decay
# """

# ╔═╡ df62e796-d711-4ea1-a0c2-e3ec6c501a78
# md"""


# ## Alternatively, closed form solution

# Alternatively, set the gradient to zero, and solve it


# !!! exercise "Ridge regression solution"
# 	Show that the closed form solution for ridge regression is
# 	```math
# 	\large
# 		\mathbf{w}_{ridge} = (\mathbf{X}^\top\mathbf{X} +\lambda \mathbf{I})^{-1} \mathbf{X}^\top\mathbf{y}
# 	```



# !!! hint "Hint"
# 	```math
# 		\lambda \mathbf{w} = \lambda \mathbf{Iw},
# 	```
# 	where ``\mathbf{I}`` is the identity matrix.

# """

# ╔═╡ 8493d307-9158-4821-bf3b-c368d9cd5fc5
# ridge_reg(X, y; λ = 1) = (X' * X + λ *I)^(-1) * X' * y;

# ╔═╡ 88d98d87-f3cf-42f4-9282-1e6f383934cd
# md"""

# ## Effect of hyperparameter


# Polynomial (10-order) ridge regression  with different hyperparameters

# ```math
# \large
# \lambda = [ 0, 1, 5, 10, 20, e^{20}];\;\;
# e^{20} \approx 4.85 \times 10^8
# ```

# """

# ╔═╡ 0e60180d-c4eb-4c83-9301-e3151ab828d5
# let
# 	gr()
# 	poly_order = 10
# 	λs = [0, 1, 5, 10, 20, exp(20)]
# 	plots_ =[]
# 	for λ in λs
# 		plt = plot(x_poly, y_poly, st=:scatter, ms=3, mc=1,alpha=0.5, label="train data")
# 		plot!(x_poly_test, y_poly_test, st=:scatter, ms=3, mc=2,alpha=0.5,  label="test data")
# 		x_p = poly_expand(x_poly; order = poly_order)
# 		w = ridge_reg(x_p, y_poly; λ = λ)
# 		loss_test = norm([poly_fun(x, w) for x in x_poly_test] - y_poly_test)/length(y_poly_test)
# 		plot!(-2:0.05:2, (x) -> poly_fun(x, w), lw=2, lc=:red,  label="", legend=:outerbottom, ylim=[-1, 7], title="test loss: "*L"%$(round(loss_test; digits=2));\;" * L"\lambda=%$(round(λ;digits=1))")
# 		push!(plots_, plt)
# 	end
# 	plot(plots_..., size=(900, 600))
# end

# ╔═╡ 58d6a11e-a375-4f0b-84d7-b95e7cfd3033
# md"""

# ## Question 



# !!! question "Question"
# 	What is the optimisation result?
# 	```math
# 	\mathbf{w} \leftarrow \arg\min_{\mathbf{w}} \frac{1}{2}  (\mathbf{y}-\mathbf{Xw})^\top (\mathbf{y}-\mathbf{Xw}) + \frac{e^{20}}{2} (\mathbf{w} - \mathbf{1})^\top (\mathbf{w} - \mathbf{1})
# 	```

# """

# ╔═╡ 974f1b58-3ec6-447a-95f2-6bbeda43f12f
md"""

# Appendix
"""

# ╔═╡ f1261f00-7fc6-41bb-8706-0b1973d72955
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

# ╔═╡ 627ac720-41dc-4df0-94b1-b62abe15296a
plt_av = let
	gr()
	a1 = a1
	a2 = a2
	A= hcat([a1, a2]...)
 	plt = plot( zlim =[-1,1.5], framestyle=:zerolines, camera=(ang11,ang21), size=(400,400))
	arrow3d!([0], [0], [0], [a1[1]], [a1[2]], [a1[3]]; as=0.1, lc=2, la=1, lw=2, scale=:identity)
	arrow3d!([0], [0], [0], [a2[1]], [a2[2]], [a2[3]]; as=0.1, lc=3, la=1, lw=2, scale=:identity)
	scatter!([0], [0], [0], mc=:black, ms =1, label="")
	normv = cross(a1, a2)
	bv = v₁_ * a1 + v₂_ * a2 
	if add_u
		arrow3d!([0], [0], [0], [bv[1]], [bv[2]], [bv[3]]; as=0.1, lc=1, la=1, lw=3, scale=:identity)
		if add_av
			surface!(-4:.2:5, -4:.2:5, (x,y) -> - x * normv[1]/normv[3]- y * normv[2]/normv[3], colorbar=false, alpha=0.25)
		end
	end
	plt
end;

# ╔═╡ 6f97e1be-0a1a-4adc-916b-26df3ad3ed8d
TwoColumn(md"""##### Let's consider a case ``m=2``, 

```math 
\mathbf{X} = \begin{bmatrix} \columncolor{lightsalmon}\textcolor{red}{\vert} &\columncolor{lightgreen} \textcolor{green}{ \vert} \\
\textcolor{red}{\mathbf{x}_1} & \textcolor{green}{\mathbf{x}_2}\\
\textcolor{red}{\vert} & \vert
 \end{bmatrix},\;\; \mathbf{w} = \begin{bmatrix} w_1 \\ w_2\end{bmatrix}
```

##### _for example_, ``\mathbf{Xw}`` becomes

```math
\large
	w_1\textcolor{red}{\begin{bmatrix}
	   1.5\\
	   3\\
	   0
	 \end{bmatrix}} + w_2 \textcolor{green}{\begin{bmatrix}
	  2\\
	  0\\
	  0
	 \end{bmatrix}}
```

* ``\textcolor{red}{\mathbf{x}_1}, \textcolor{green}{\mathbf{x}_2}``: the ``\textcolor{red}{\rm red}`` and ``\textcolor{green}{\rm green}`` vectors

 
>  #### What is span ``\{w_1 \textcolor{red}{\mathbf{x}_1} + w_2 \textcolor{green}{\mathbf{x}_2}\}:``?
>
> * ##### *aka* column space
""", plt_av)

# ╔═╡ 45363ea2-7f01-49c2-833d-9fe5147c5aed
plt_bv = let
	gr()
	a1 = [1.5,3,0]
	a2 = [3,0,0]
	A= hcat([a1, a2]...)
 	plt = plot(xlim=[-1,5], ylim=[-1, 4], zlim =[-2,4], framestyle=:zerolines, camera=(ang1_,ang2), size=(400,400))
	arrow3d!([0], [0], [0], [a1[1]], [a1[2]], [a1[3]]; as=0.1, lc=2, la=1, lw=2, scale=:identity)
	arrow3d!([0], [0], [0], [a2[1]], [a2[2]], [a2[3]]; as=0.1, lc=3, la=1, lw=2, scale=:identity)
	scatter!([0], [0], [0], mc=:black, ms =1, label="")
	normv = cross(a1, a2)
	surface!(-1:0.1:5, -1:0.1:5, (x,y) -> - x * normv[1]/normv[3]- y * normv[2]/normv[3], colorbar=false, alpha=0.25)
	if add_b

		arrow3d!([0], [0], [0], [bv[1]], [bv[2]], [bv[3]]; as=0.1, lc=1, la=1, lw=3, scale=:identity)
		if add_sol
			A = [a1 a2]
			v = A \ bv 
			va1 = v[1] * a1
			arrow3d!([0], [0], [0], [va1[1]], [va1[2]], [va1[3]]; as=0.1, lc=2, la=1, lw=3, scale=:identity)
			plot!([bv[1], va1[1]], [bv[2], va1[2]], [bv[3], va1[3]], lw=2, lc=:gray, ls=:dash, label="")
			va2 = v[2] * a2
			arrow3d!([0], [0], [0], [va2[1]], [va2[2]], [va2[3]]; as=0.1, lc=3, la=1, lw=3, scale=:identity)
			plot!([bv[1], va2[1]], [bv[2], va2[2]], [bv[3], va2[3]], lw=2, lc=:gray, ls=:dash, label="")
		end
	end
	plt
end;

# ╔═╡ 62c51913-5742-41a4-8e58-29a88279bc38
TwoColumn(md"""
\
\
\
\


##### When ``\color{blue}\mathbf{y}``: the ``\textcolor{blue}{\rm blue}`` vector 

$\large\color{blue}\mathbf{y}\in \{\mathbf{Xw}; \mathbf{w}\in \mathbb{R}^m\}$
* ###### *there is (are) solution(s)*

""", plt_bv)

# ╔═╡ 9169557a-054b-421a-927a-66435e4eb0b8
plt_av_nosol = let
	gr()
	a1 = [2,4,0]
	a2 = [3,0,0]
	b = [1,1,4]
	# c = [1,1,4]
	A= hcat([a1, a2]...)
	# bp= A*inv(A'*A)*A'*b
 	plt=plot(xlim=[-1,5], ylim=[-1, 5], zlim =[-2,5], framestyle=:zerolines, camera=(ang1,15), size=(400,400))
	arrow3d!([0], [0], [0], [b[1]], [b[2]], [b[3]]; as=0.1, lc=1, la=1, lw=2, scale=:identity)
	# annotate!(c[1], c[2], c[3], text("c"))
	arrow3d!([0], [0], [0], [a1[1]], [a1[2]], [a1[3]]; as=0.1, lc=2, la=1, lw=2, scale=:identity)
	arrow3d!([0], [0], [0], [a2[1]], [a2[2]], [a2[3]]; as=0.1, lc=3, la=1, lw=2, scale=:identity)
	scatter!([0], [0], [0], mc=:black, label="")

	if add_proj
		arrow3d!([0], [0], [0], [bp[1]], [bp[2]], [bp[3]]; as=0.1, lc=4, la=0.9, lw=2, scale=:identity)
		plot!([b[1], bp[1]], [b[2], bp[2]], [b[3], bp[3]], lw=2, lc=:gray, ls=:dash, label="")
	end

	surface!(-1:0.1:5, -1:0.1:5, (x,y) -> 0, colorbar=false, alpha=0.35)
	plt
end;

# ╔═╡ ae2bfc51-dc67-4ed4-8340-365a4c0b46d7
TwoColumn(md"""
\
\
\

##### When ``\color{blue}\mathbf{y}``: the ``\textcolor{blue}{\rm blue}`` vector _sticks out_, or

```math
\Large\textcolor{blue}{\mathbf{y}} \notin \{\mathbf{Xw}, \mathbf{w}\in \mathbb{R}^m\}
```



* ##### *NO solution*
* ##### _nevertheless_, we can find an *approximated* solution
""", plt_av_nosol)

# ╔═╡ 57ff6470-631a-474c-b94d-80b27c33142b
TwoColumn(md"""
\
\


#### In maths,
\

```math
\Large
\underbrace{(\textcolor{blue}{\textcolor{purple}{\hat{\mathbf{y}}} - \mathbf{y}} )}_{\color{brown}\text{error } \hat{\mathbf{e}}} \perp \{\textcolor{red}{\mathbf{x}_1}, \textcolor{green}{\mathbf{x}_2}, \ldots\}
```

* ##### note that ``\mathbf{Xw} \triangleq \hat{\mathbf{y}}``

""", let
	gr()
	a1 = [2,4,0]
	a2 = [3,0,0]
	b = [1,1,4]
	# c = [1,1,4]
	A= hcat([a1, a2]...)
	bp= A*inv(A'*A)*A'*b
 	plt = plot(xlim=[-1,5], ylim=[-1, 5], zlim =[-2,5], framestyle=:zerolines, camera=(20,15), size=(400,400))
	arrow3d!([0], [0], [0], [b[1]], [b[2]], [b[3]]; as=0.1, lc=1, la=1, lw=2, scale=:identity)
	# annotate!(c[1], c[2], c[3], text("c"))
	arrow3d!([0], [0], [0], [a1[1]], [a1[2]], [a1[3]]; as=0.1, lc=3, la=1, lw=2, scale=:identity)
	arrow3d!([0], [0], [0], [a2[1]], [a2[2]], [a2[3]]; as=0.1, lc=2, la=1, lw=2, scale=:identity)
	scatter!([0], [0], [0], mc=:black, label="")
	arrow3d!([0], [0], [0], [bp[1]], [bp[2]], [bp[3]]; as=0.1, lc=4, la=0.9, lw=2, scale=:identity)
	plot!([b[1], bp[1]], [b[2], bp[2]], [b[3], bp[3]], lw=2, lc=:gray, ls=:solid, arrow=:arrow,  label="")
	error = bp -b 
	arrow3d!(b[1], b[2], b[3], [error[1]], [error[2]], [error[3]]; as=0.1, lc=5, la=0.9, lw=3, scale=:identity)
	# arrow3d!([b[1]], [b[2]], [b[3]], [bp[1]], [bp[2]], [bp[3]]; as=0.1, lc=4, la=0.9, lw=2, scale=:identity)
	surface!(-1:0.1:5, -1:0.1:5, (x,y) -> 0, colorbar=false, alpha=0.35)
	plt
end)

# ╔═╡ 238e7b56-fb3a-4e9b-9c31-09c1f4a1df2a
begin
	Random.seed!(111)
	num_features = 2
	num_data = 25
	true_w = rand(num_features+1) * 10
	# simulate the design matrix or input features
	X_train = [ones(num_data) rand(num_data, num_features)]
	# generate the noisy observations
	y_train = X_train * true_w + randn(num_data)
end;

# ╔═╡ 69005e98-5ef3-4376-9eed-919580e5de53
let
	plotly()
	# plot(X_train[:,2], y_train, st=:scatter, label="Observations")
	scatter(X_train[:, 2], X_train[:,3], y_train, markersize=1.5, label="observations", title="Linear regression assumption", xlabel="x₁", ylabel="x₂", zlabel="y")
	surface!(0:0.5:1, 0:0.5:1.0, (x1, x2) -> dot([1, x1, x2], true_w),  colorbar=false, xlabel="x₁", ylabel="x₂", zlabel="y", alpha=0.8, label="h(x)",c=:coolwarm)
end

# ╔═╡ 57b77a3c-7424-4215-850e-b0c77036b993
let
	gr()
	Random.seed!(111)
	ws = [zeros(3) rand(3) * 15  rand(3)*5   true_w]
	plots_frames = []
	# plot(X_train[:,2], y_train, st=:scatter, label="Observations")
	for i in 1 : 4
		plt = scatter(X_train[:, 2], X_train[:,3], y_train, markersize=1.5, label="", title="Loss is $(round(loss(ws[:, i], X_train, y_train);digits=2))", xlabel=L"x_1", ylabel=L"x_2", zlabel=L"y")
		surface!(0:0.1:1, 0:0.1:1.0, (x1, x2) -> dot([1, x1, x2], ws[:, i]),  colorbar=false,  alpha=0.8, label="", c=:coolwarm)
		push!(plots_frames, plt)
	end
	
	plot(plots_frames..., layout=4)
end

# ╔═╡ 9877b87b-6cbc-4c28-8b8b-a93baa647482
let
	w_lse = zeros(size(X_train)[2])
	if estimate_
		w_lse = least_square_est(X_train, y_train)
	end
	plotly()
	scatter(X_train[:, 2], X_train[:,3], y_train, markersize=1.5, label="observations", title="Linear regression: normal equation", xlabel="x₁", ylabel="x₂", zlabel="y")

	surface!(0:0.5:1, 0:0.5:1.0, (x1, x2) -> dot([1, x1, x2], w_lse),  colorbar=false, xlabel="x₁", α=0.8, ylabel="x₂", zlabel="y", c=:coolwarm)
end

# ╔═╡ 2c7b22f5-1b99-4f9a-b502-c3e069a15cae
let
	w_lse = zeros(size(X_train)[2])
	if estimate
		w_lse = least_square_est(X_train, y_train)
	end
	plotly()
	scatter(X_train[:, 2], X_train[:,3], y_train, markersize=1.5, label="observations", title="Linear regression: normal equation", xlabel="x₁", ylabel="x₂", zlabel="y")

	surface!(0:0.5:1, 0:0.5:1.0, (x1, x2) -> dot([1, x1, x2], w_lse),  colorbar=false, xlabel="x₁", α=0.8, ylabel="x₂", zlabel="y", c=:coolwarm)
end

# ╔═╡ c9b5e47c-e0f1-4496-a342-e37df85d6de9
begin
	# define a function that returns a Plots.Shape
	rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
end;

# ╔═╡ fa7ec40f-6986-4e0a-93cc-6ad71c3b7938
let
	gr()
	Random.seed!(123)
	n = 10
	w0, w1 = w₀, w₁
	truew0, truew1 = 1, 1
	Xs = range(-2, 2, n)
	ys = (truew0 .+ Xs .* truew1) .+ randn(n)/1
	Xs = [Xs; 0.5]
	ys = [ys; 3.5]
	ŷs = (w0 .+ Xs .* w1)
	loss = sum((ys - ŷs).^2)
	plt = plot(Xs, ys, st=:scatter, markersize=5, alpha=0.9, label="", xlabel=L"x", ylabel=L"y", ratio=1, title="SSE loss: "*L"\sum (y^{(i)} - \hat{y}^{(i)})^2=%$(round(loss; digits=3))")
	plot!(-4:0.1:4, (x) -> w0 + w1 * x , ylim =[-2,4], lw=2, label=L"h_w(x)", legend=:topleft, framestyle=:origin)
	ŷs = Xs .* w1 .+ w0
	for i in 1:length(Xs)
		plot!([Xs[i], Xs[i]], [ys[i], ŷs[i] ], lc=:gray, lw=1.5, label="")
		iidx = i
			if (ys[iidx] -  ŷs[iidx]) > 0 
		li = -(ŷs[iidx] - ys[iidx] )
		plot!(rectangle(li, li, Xs[iidx], ŷs[iidx]), lw=1.5, color=:gray, opacity=.3, label="")
	else
		li = -(ŷs[iidx] - ys[iidx] )
		plot!(rectangle(abs(li), li, Xs[iidx], ŷs[iidx]), lw=1.5, color=:gray, opacity=.3, label="")
		# annotate!(.5*(Xs[iidx] + abs(li)), 0.5*(ys[iidx] + ŷs[iidx]), text(L"(y^i - h(x^{(i)}))^2", 10, :black ))

	end
	end


	plt
end

# ╔═╡ ddbd8d06-4ff0-4a1c-8dfe-092072fd88d5
function perp_square(origin, vx, vy; δ=0.1) 
	x = δ * vx/sqrt(norm(vx))
	y = δ * vy/sqrt(norm(vy))
	xyunit = origin+ x + y
	xunit = origin + x
	yunit = origin +y
	Shape([origin[1], xunit[1], xyunit[1], yunit[1]], [origin[2], xunit[2], xyunit[2], yunit[2]])
end

# ╔═╡ e4e1b9b4-a078-42dd-9fad-726a7941a1c7
plt_proj_to_a = let
	gr()
 	plt = plot(xlim=[-1,3], ylim=[-1, 3], ratio=1, framestyle=:origin, size=(300, 300))
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	a = [3,0]
	b= [2,2]
	bp = dot(a,b)/dot(a,a)*a
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lc=2, lw=2)
	quiver!([0], [0],  quiver=([b[1]], [b[2]]), lc=1, lw=2)
	annotate!(0+0.3, 0+0.3, text(L"\theta", :top))
	annotate!(a[1],a[2], text(L"\mathbf{x}", :bottom))
	annotate!(b[1],b[2], text(L"\mathbf{y}", :bottom))
	plot!([b[1], bp[1]], [b[2],bp[2]], ls=:dash, lc=:gray, lw=2, label="")
	annotate!(bp[1],bp[2]-0.1, text(L"\hat{\mathbf{y}}_{\texttt{proj}}", :top))
	quiver!([0], [0],  quiver=([bp[1]], [bp[2]]), lc=4, lw=2)
	plot!(perp_square([bp[1],bp[2]], a, b -bp; δ=0.1), lw=1, fillcolor=false, label="")

	quiver!([b[1]], [b[2]], quiver=([-b[1] + bp[1]], [-b[2] +bp[2]]), lc=:red, lw=3)

	annotate!(0.5 * (bp[1] +b[1]) +0.1,0.5 * (bp[2] +b[2]), text(L"\hat{\mathbf{y}}_{\texttt{proj}} -\mathbf{y}", 10, :red, :left))
	plt
end;

# ╔═╡ 893b6e7d-9381-4726-9d04-2417e6684a32
TwoColumn(md"""
\

##### "_Simple single projection_"

> #### Project ``\mathbf{y}`` to a *vector* ``\mathbf{x}``:
> ```math
> \large
> (\hat{\mathbf{y}}_{\texttt{proj}}-\mathbf{y}) \perp \mathbf{x}
> ```
\

##### Now, the problem is more _general_

> ##### Project to a space formed by **multiple** vectors
> ```math 
> \large \{\mathbf{x}_1, \mathbf{x}_2, \ldots \}
> ```

""", plt_proj_to_a)

# ╔═╡ cb02aee5-d082-40a5-b799-db6b4af557f7
# md"""
# ## More datasets


# It turns out linear correlations are more common than we expect!

# * *e.g.* the flipper size and body mass of Penguins 
# """

# ╔═╡ 8deb1b8c-b67f-4d07-8986-2333dbadcccc
# md"""
# ![](https://allisonhorst.github.io/palmerpenguins/reference/figures/lter_penguins.png)"""

# ╔═╡ af622189-e504-4633-9d9e-ab16c7293f82
df_penguin = let
	Logging.disable_logging(Logging.Warn)
	df_penguin = DataFrame(CSV.File(download("https://gist.githubusercontent.com/slopp/ce3b90b9168f2f921784de84fa445651/raw/4ecf3041f0ed4913e7c230758733948bc561f434/penguins.csv"), types=[Int, String, String, [Float64 for _ in 1:4]..., String, Int]))
	df_penguin[completecases(df_penguin), :]
end;

# ╔═╡ 9267c4a4-74d1-4515-95d6-acc3b12e5ed6
# first(df_penguin, 5)

# ╔═╡ 76cc8ca7-d17e-4cd7-a6de-4d606e0a0985
# @df df_penguin scatter(:flipper_length_mm, :body_mass_g, group = (:species), legend=:topleft, xlabel="Flipper length", ylabel="Body mass");

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
BenchmarkTools = "~1.6.0"
CSV = "~0.10.10"
DataFrames = "~1.7.0"
ForwardDiff = "~0.10.36"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Latexify = "~0.16.5"
LogExpFunctions = "~0.3.23"
MLDatasets = "~0.7.9"
Plots = "~1.38.14"
PlutoTeachingTools = "~0.2.11"
PlutoUI = "~0.7.51"
StatsBase = "~0.34.0"
StatsPlots = "~0.15.5"
Zygote = "~0.6.68"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "a78d9b240dacb858f3d9a16a437433a10d714006"

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
git-tree-sha1 = "0ba8f4c1f06707985ffb4804fdad1bf97b233897"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.41"

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
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "680b3b8759bd4c54052ada14e52355ab69e07876"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

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

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "93da6c8228993b0052e358ad592ee7c1eccaa639"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.0"

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

[[deps.AtomsBase]]
deps = ["LinearAlgebra", "PeriodicTable", "Preferences", "Printf", "Requires", "StaticArrays", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "5f76425eb977584353191c41d739e7783f036b90"
uuid = "a963bdd2-2df7-4f54-a1ee-49d51e6be12a"
version = "0.5.1"

    [deps.AtomsBase.extensions]
    AtomsBaseAtomsViewExt = "AtomsView"

    [deps.AtomsBase.weakdeps]
    AtomsView = "ee286e10-dd2d-4ff2-afcb-0a3cd50c8041"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "2c7cc21e8678eff479978a0a2ef5ce2f51b63dff"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.5.0"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "e2144b631226d9eeab2d746ca8880b7ccff504ae"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.3"

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

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "e38fbc49a620f5d0b660d7f543db1009fe0f8336"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BufferedStreams]]
git-tree-sha1 = "6863c5b7fc997eadcabdbaf6c5f201dc30032643"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.2.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+4"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "4312d7869590fab4a4f789e97bd82f0a04eaaa05"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.72.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Chemfiles]]
deps = ["AtomsBase", "Chemfiles_jll", "DocStringExtensions", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "b7b6140d72877345f03ce09d2cd97c0115f43c52"
uuid = "46823bd8-5fb3-5f92-9aa0-96921f3dd015"
version = "0.10.42"

[[deps.Chemfiles_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f3743181e30d87c23d9c8ebd493b77f43d8f1890"
uuid = "78a364fa-1a3c-552a-b4bb-8fa0f9c1fcca"
version = "0.10.4+0"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "26ec26c98ae1453c692efded2b17e15125a5bea1"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.28.0"

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

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

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

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "f36e5e8fdffcb5646ea5da81495a5a7566005127"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.3"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataDeps]]
deps = ["HTTP", "Libdl", "Reexport", "SHA", "Scratch", "p7zip_jll"]
git-tree-sha1 = "8ae085b71c462c2cb1cfedcb10c3c877ec6cf03f"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.13"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

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

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

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
git-tree-sha1 = "e51db81749b0777b2147fbe7b783ee79045b8e99"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.4+3"

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

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "7de7c78d681078f027389e067864a8d53bd7c3c9"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.1"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+3"

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
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "7878ff7172a8e6beedd1dea14bd27c3c6340d361"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.22"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

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
git-tree-sha1 = "0ef97e93edced3d0e713f4cfd031cc9020e022b0"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.2.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.GZip]]
deps = ["Libdl", "Zlib_jll"]
git-tree-sha1 = "0085ccd5ec327c077ec5b91a5f937b759810ba62"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.6.2"

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

[[deps.Glob]]
git-tree-sha1 = "97285bbd5230dd766e9ef6749b80fc617126d496"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.1"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "e856eef26cf5bf2b0f95f8f4fc37553c72c8641c"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.2"

    [deps.HDF5.extensions]
    MPIExt = "MPI"

    [deps.HDF5.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "38c8874692d48d5440d5752d6c74b0c6b0b60739"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.2+1"

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

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "50aedf345a709ab75872f80a2779568dc0bb461b"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.11.2+3"

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

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "0f14a5456bdc6b9731a5682f439a672750a09e48"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.0.4+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InternedStrings]]
deps = ["Random", "Test"]
git-tree-sha1 = "eb05b5625bc5d821b8075a77e4c421933e20c76b"
uuid = "7d512f48-7fb1-5a58-b986-67e6dc259f01"
version = "0.7.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

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
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "Requires", "TranscodingStreams"]
git-tree-sha1 = "91d501cb908df6f134352ad73cde5efc50138279"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.5.11"

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

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "1d322381ef7b087548321d3f878cb4c9bd8f8f9b"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.1"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "a729439c18f7112cbbd9fcdc1771ecc7f071df6a"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.39"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "d5bc0b079382e89bfa91433639bc74b9f9e17ae7"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.33"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "5fcfea6df2ff3e4da708a40c969c3812162346df"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.2.0"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

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

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

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
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

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

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "1d2dd9b186742b0f317f2530ddcbf00eebb18e96"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.7"

[[deps.MIMEs]]
git-tree-sha1 = "1833212fd6f580c20d4291da9c1b4e8a655b128e"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.0.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

[[deps.MLDatasets]]
deps = ["CSV", "Chemfiles", "DataDeps", "DataFrames", "DelimitedFiles", "FileIO", "FixedPointNumbers", "GZip", "Glob", "HDF5", "ImageShow", "JLD2", "JSON3", "LazyModules", "MAT", "MLUtils", "NPZ", "Pickle", "Printf", "Requires", "SparseArrays", "Statistics", "Tables"]
git-tree-sha1 = "361c2692ee730944764945859f1a6b31072e275d"
uuid = "eb30cadb-4394-5ae3-aed4-317e484a6458"
version = "0.7.18"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "7940c0af802586b97009f254aa6065000a16fa1d"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.5"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "7715e65c47ba3941c502bffb7f266a41a7f54423"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.2.3+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "70e830dab5d0775183c99fc75e4c24c614ed7142"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.1+2"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

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

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

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
version = "2023.12.12"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "bdc9d30f151590aca0af22690f5ab7dc18a551cb"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.27"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NPZ]]
deps = ["FileIO", "ZipFile"]
git-tree-sha1 = "60a8e272fe0c5079363b28b0953831e2dd7b7e6f"
uuid = "15e1cf62-19b3-5cfa-8e77-841668bca605"
version = "0.4.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "fe891aea7ccd23897520db7f16931212454e277e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.1"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "8a3271d8309285f4db73b4f662b1b290c715e85e"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.21"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

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

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "2dace87e14256edb1dd0724ab7ba831c779b96bd"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.6+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+3"

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
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "966b85253e959ea89c53a9abebbf2e964fbf593b"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.32"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ed6834e95bd326c52d5675b4181386dfbe885afb"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.55.5+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.PeriodicTable]]
deps = ["Base64", "Unitful"]
git-tree-sha1 = "238aa6298007565529f911b734e18addd56985e1"
uuid = "7b2266bf-644c-5ea3-82d8-af4bbd25a884"
version = "1.2.1"

[[deps.Pickle]]
deps = ["BFloat16s", "DataStructures", "InternedStrings", "Mmap", "Serialization", "SparseArrays", "StridedViews", "StringEncodings", "ZipFile"]
git-tree-sha1 = "e99da19b86b7e1547b423fc1721b260cfbe83acb"
uuid = "fbb45041-c46e-462f-888f-7c521cafbc2c"
version = "0.3.5"

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
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "9f8675a55b37a70aa23177ec110f6e3f4dd68466"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.17"

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

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

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

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

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

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

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
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

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
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

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

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "02c8bd479d26dbeff8a7eb1d77edfc10dacabc01"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.11"
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

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

[[deps.StridedViews]]
deps = ["LinearAlgebra", "PackageExtensionCompat"]
git-tree-sha1 = "5b765c4e401693ab08981989f74a36a010aa1d8e"
uuid = "4db3bf67-4bd7-4b4e-b153-31dc3fb37143"
version = "0.2.2"

    [deps.StridedViews.extensions]
    StridedViewsCUDAExt = "CUDA"

    [deps.StridedViews.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "b765e46ba27ecf6b44faf70df40c57aa3a547dcb"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a6b1675a536c5ad1a60e5a5153e1fee12eb146e3"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.0"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "9537ef82c42cdd8c5d443cbc359110cbb36bae10"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.21"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

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
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

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

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

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

[[deps.UnitfulAtomic]]
deps = ["Unitful"]
git-tree-sha1 = "903be579194534af1c4b4778d1ace676ca042238"
uuid = "a7773ee8-282e-5fa2-be4e-bd808c38a91a"
version = "1.0.0"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

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

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

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

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

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

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "f492b7fe1698e623024e873244f10d89c95c340a"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "622cf78670d067c738667aaa96c553430b65e269"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+0"

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

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f5733a5a9047722470b95a81e1b172383971105c"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.3+0"

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
git-tree-sha1 = "055a96774f383318750a1a5e10fd4151f04c29c5"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.46+0"

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

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

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
# ╟─17a3ac47-56dd-4901-bb77-90171eebc8c4
# ╟─a26e482a-f925-48da-99ba-c23ad0a9bed6
# ╟─29998665-0c8d-4ba4-8232-19bd0de71477
# ╟─cb72ebe2-cea8-4467-a211-5c3ac7af74a4
# ╟─f9023c9e-c529-48a0-b94b-31d822dd4a11
# ╟─d11b231c-3d4d-4fa2-8b1c-f3dd742f8977
# ╟─580b2af0-5a8f-45b3-bc34-ff7bf6c0c221
# ╟─e48c618f-10ec-4ae0-8274-24780417f456
# ╟─cf9c3937-3d23-4d47-b329-9ecbe0006a1e
# ╟─ec6242db-e69e-429c-b82c-2c22a9b232f1
# ╟─2a381c96-57ed-42f0-9cca-fc2b84e017a5
# ╟─98951c85-ecc7-45af-b429-dbe11b4aac38
# ╟─dfcfd2c0-9f51-48fb-b91e-629b6934dc0f
# ╟─8257d59b-b8f3-44aa-879a-ecaa40c511d6
# ╟─6af0c378-fbac-4d22-a046-de37b5c90f20
# ╟─6073463a-ca24-4ddc-b83f-4c6ff5033b3b
# ╟─ed20d0b0-4e1e-4ec5-92b0-2d4938c249b9
# ╟─774c46c0-8a62-4635-ab56-662267e67511
# ╟─c802c807-971b-426a-be13-a382688de911
# ╟─86f09ee8-087e-47ac-a81e-6f8c38566774
# ╟─e3797d79-8007-4c8b-8f4f-ff2cb4461020
# ╟─75c38735-ba5b-4040-9c1b-77f963c6cccc
# ╟─564ff88e-b310-4ac1-85a0-28640ee015bb
# ╟─fe3e7f8f-e8a0-4a1f-ad9b-a880a00f49aa
# ╟─65f28dfb-981d-4361-b37c-12af3c7995cd
# ╟─e196e010-6e45-486b-957d-b3d5910dc887
# ╟─e5c1570d-1620-4e90-afbd-97e9c8151c51
# ╟─6970801f-4d88-4baf-9c06-5feb9bb3ed2f
# ╟─c3d82ace-cfdc-44f6-93e2-d9e20a10cdee
# ╟─24eb939b-9568-4cfd-bfe5-0191eada253a
# ╟─5d96f623-9b30-49a4-913c-6dee65ae0d23
# ╟─69005e98-5ef3-4376-9eed-919580e5de53
# ╟─fe533cf8-5ff2-4094-b5bc-08bd43f3f9de
# ╟─9d9c7233-9535-48c4-8add-68415217e1dd
# ╟─492f4ed9-5434-4c5a-8baf-3ecd79345ddc
# ╟─fc826717-2a28-4b86-a52b-5c133a50c2f9
# ╟─93600d0c-fa7e-4d38-bbab-5adcf54d0c90
# ╟─cec496e3-a249-4516-83d0-1b8c34efa4c4
# ╟─c1a8b895-55ea-4a63-9327-1f961dbc25e1
# ╟─fa7ec40f-6986-4e0a-93cc-6ad71c3b7938
# ╟─dead4d31-8ed4-4599-a3f7-ff8b7f02548c
# ╟─d70102f1-06c0-4c5b-8dfd-e41c4a455181
# ╟─b1c58114-6999-4789-aa12-7454aa1e0927
# ╟─57b77a3c-7424-4215-850e-b0c77036b993
# ╟─a02a28d4-d015-4a99-906f-0aed0ada51af
# ╟─c7aced31-0d39-4a8f-927b-4977156e486e
# ╟─c4107744-f07e-4b4f-bc04-64a5dc059801
# ╟─ce1a022c-66e3-4f96-b2a3-9caee9ebbcaf
# ╟─7258e0a7-d805-4d14-93a5-4ce15df30346
# ╟─b2f1592d-5fbf-4c7c-9d6d-562e87fb703e
# ╟─1862e10d-f9f8-4b3f-a112-2387e3204d85
# ╟─0cae3899-b3a9-4f28-9fa4-748779afed41
# ╟─0675e184-fbef-410c-bdd2-a5b0e77bbabf
# ╟─485d6d3d-8e80-4165-ba7e-4a0c0b661ef5
# ╟─a5323b5c-9904-4aaf-a569-4057f3c7dac6
# ╟─1431515c-3e1e-4629-b79b-21c62b222eee
# ╟─9d784098-5150-4a66-b2ea-7e86820a165a
# ╟─956e0525-3627-4626-bb7b-fee6299b544d
# ╟─76d88d8b-09f9-462f-a2fe-4aa14e11226c
# ╟─a0ad1190-da30-45ae-a00d-189eae6ed516
# ╟─b2309f9b-516c-4852-ad21-a1f43e3b5c7a
# ╟─01e4c730-0bf9-418b-a9d8-16ad76e2e56b
# ╟─67d37ec2-5087-42c6-9400-5255db269f74
# ╟─3967f515-3be3-46f8-8bf7-1128a8f0ed54
# ╟─202f922d-0f10-4a11-a426-c39887bf8ea1
# ╟─c22b7b09-35da-4d87-b30f-9a3b525f90bf
# ╟─687fd7cd-8b5e-47d6-b186-4d143d61f366
# ╠═0a162252-359a-4be9-96d9-d66d4dca926c
# ╠═648ef3b0-2198-44eb-9085-52e2362b7f88
# ╟─1398bda3-5b94-4d54-a553-ca1ac1bc6ce9
# ╟─4e430aa8-9c74-45c1-8771-b33e558208cf
# ╟─253958f1-ae84-4230-bfd1-6023bdffee26
# ╟─8e6a0d85-e421-44e5-9c23-661b076382b0
# ╟─1b08a455-85b2-450b-b5bd-5c52694a740b
# ╟─c6f8f57a-f487-4105-8088-acd91f3c43fa
# ╟─788efd7e-7a95-4134-a514-6a03654a4af5
# ╟─158d1c6e-fbf5-454d-bd50-e60aafdc291d
# ╟─b89c8eee-e34d-4e25-aca3-68fada91ed89
# ╟─eb97eef7-e07e-429f-ae40-809761a75abc
# ╟─073c6ed9-a4cc-489d-9609-2d710aa7740f
# ╟─e57e21a0-01f7-470a-912e-c2ebb10560b6
# ╟─701f988a-139a-4b81-97d1-46b24ac6335e
# ╟─be664fd1-7cc3-478d-8e75-86ff0643933c
# ╟─7fc17194-f462-4326-b22a-b64dbce6b944
# ╟─697700d6-a6d9-4a05-97e0-967ebf22454f
# ╟─df1491b1-d27a-4c25-a721-34928cef33cd
# ╟─7d68dba0-3093-4495-89d9-2dd81b86ef29
# ╟─5a995196-6c0a-4c6c-b6a6-a7c55d506aae
# ╟─87372788-5f01-4bf4-be10-099ba07d5078
# ╟─d57d1d6e-db6f-4ea6-b0fc-0f1d7cc4aa3b
# ╟─86a64b00-28e8-4ee0-b285-44d80e74ccfd
# ╟─d4e9d38f-40e8-40b7-a454-25f6d7adb64c
# ╟─b7d9dd45-1a9d-4ff9-a6b6-19c1c7d8ca2e
# ╟─9db7c362-916e-4e0c-8a81-2450a7156760
# ╟─ac28562a-4501-4906-8160-828be49b500f
# ╟─6f7a822b-8fc4-4fe1-bd5c-fb7da07e49b0
# ╟─8131a33f-b3ac-4d6f-8cf8-26cc71786177
# ╟─79a7094e-a6e4-44bf-baa7-eb3974360d4c
# ╟─11babbc2-bd40-4636-9aa8-a23a8568ea05
# ╟─4c3b3d0f-3800-443f-80f8-244e57304bdc
# ╟─898b97f0-519c-4b8a-ad1f-5e9b2164ae51
# ╟─8467d20a-65dc-403f-933f-3932c5b14d13
# ╟─92d7be2f-bdb5-457c-8bfd-2fe98cc32e31
# ╟─629e38b9-353c-4b55-9ffe-4b1b249843eb
# ╟─64ba3967-ba5f-454e-9809-867b4544dd5a
# ╟─350f2a70-405c-45dc-bfcd-913bc9a7de75
# ╟─a044c05e-2513-48c6-a1cb-002acc3eacfd
# ╟─0adbec1a-4962-4e4a-8cec-e57ff9abb6d6
# ╟─970ccf20-d9a4-4740-9c1f-d31aa72f1856
# ╟─8d788ccb-4f2a-4e15-ac2f-16c87aeb0a38
# ╟─661ef8d4-f93f-4980-b7ea-fd7f9d77b76a
# ╟─efcc2e8f-6401-4256-bc51-556be014823e
# ╟─cea0a958-8128-4c76-9e5f-f3614026b01e
# ╟─fd6beddc-1a0a-4256-b978-4c0da9f4875b
# ╟─6da86877-1d96-4a73-af75-e4ae3eea1499
# ╟─8fbcf6c3-320c-47ae-b4d3-d710a120eb1a
# ╟─f90bf568-33fc-475a-9079-99f79af891f5
# ╟─9877b87b-6cbc-4c28-8b8b-a93baa647482
# ╟─4d0c089b-2ed3-4787-9c36-2941600b079f
# ╟─088fc2bb-fc33-499a-8baf-a9a8628bd5c8
# ╠═14032a47-b996-43f1-bf47-b9ef33d92f76
# ╟─08897017-3fd8-43ff-8922-2c496b9aefd8
# ╟─2c7b22f5-1b99-4f9a-b502-c3e069a15cae
# ╟─e4d5e0ec-8eb2-4af5-87a3-d9ecad0f104f
# ╟─bb2b313a-b5ec-4246-8e2e-fe4e6aab9d6c
# ╟─da365bdc-da2b-4da2-8e2e-3d2f88a85fec
# ╟─38ec0405-16a2-42ee-a4af-2a61c6f87ba1
# ╟─7bfa4bab-e563-4298-acf4-ed955b6bdabd
# ╟─fc3fe35d-8437-46d4-9ea0-57c4c48f3ae0
# ╟─cb2e0882-dc92-421e-9352-cfdf3f9da96b
# ╟─619819c3-b8bd-456b-85fd-f0c6e74d6edc
# ╟─b0d9538d-584c-44ea-8221-3727180d7e17
# ╟─0fe59b22-f12d-4b4d-9d18-0c5b53cc0336
# ╟─547cec7f-edff-4283-89ee-788a7bda5f16
# ╟─31e9f475-119e-4126-b556-89202f161ccd
# ╟─485ebf22-2307-476c-a448-21e2377513e2
# ╟─f0256550-ad5f-4cfe-ac93-685ab2c584d3
# ╟─243b9d86-8592-420c-9d0b-659722feaf32
# ╟─f111e5f7-36c7-4df2-8eee-944ec770fd66
# ╟─608883c2-8cf5-4a6e-8260-0799e58b87f5
# ╟─6de0c68f-fc63-41f5-8615-967bb480d2fe
# ╟─107949f6-3fda-43bb-874b-d12605f20380
# ╟─ff2b28fd-5a7c-46b8-ba32-763640b54b9a
# ╟─21458c5d-ffc8-483a-a375-477f5fead1f8
# ╟─c71e94e9-39d9-4b25-ab14-a0ad65b338ea
# ╟─b73ee6cb-f6e0-4a0d-afac-24148447425e
# ╟─d5345afe-84e6-4922-a800-e8289a95d048
# ╟─b2a10215-7595-4a92-94ac-dad3fe80e8f6
# ╟─c6c78c39-f9c3-4bb4-96d7-5c447b96130a
# ╟─6f97e1be-0a1a-4adc-916b-26df3ad3ed8d
# ╟─113f4be3-0ae8-4acf-a04b-aba8bac5449c
# ╟─149acbd6-6200-4d2d-9aa3-a5b6f6e6fd86
# ╟─be7bcc2a-3cef-40a9-84d9-6afedfae2c4b
# ╟─d18a8d63-80e0-4a6a-bf6e-96b857721de8
# ╟─627ac720-41dc-4df0-94b1-b62abe15296a
# ╟─5b2a8f5a-74da-464e-a3bf-172454642d8f
# ╟─12bab87e-f3f0-4124-b48d-677f47bba686
# ╟─362400de-d0e3-4fc2-a99e-a1e51eb005a8
# ╟─45363ea2-7f01-49c2-833d-9fe5147c5aed
# ╟─caed0f95-84cb-4d3a-929f-85b6d61280d7
# ╟─998c9b63-36f3-496b-9db7-6913e7c67805
# ╟─62c51913-5742-41a4-8e58-29a88279bc38
# ╟─0c0894be-2042-4797-9c20-7fc1239a2cad
# ╟─de76e3a6-5e59-4dd1-8704-0fa540e6d34d
# ╟─98615427-e2dd-4268-8534-c95ac192cc9e
# ╟─ae2bfc51-dc67-4ed4-8340-365a4c0b46d7
# ╟─28487f32-ce91-4ba8-a544-cb1c8bffb73e
# ╟─3477d5f8-1e27-45bf-a675-d781e478c302
# ╟─514c9139-7c19-4615-8ce5-9fd1b6e2c2f6
# ╟─bf43112d-22fc-4782-9112-d587bce9dd76
# ╟─9169557a-054b-421a-927a-66435e4eb0b8
# ╟─c7ad6de9-0d7d-4343-b94d-d7a9cb7785da
# ╟─893b6e7d-9381-4726-9d04-2417e6684a32
# ╟─e4e1b9b4-a078-42dd-9fad-726a7941a1c7
# ╟─121fdd48-6372-4ca9-bac9-6664a1664098
# ╟─57ff6470-631a-474c-b94d-80b27c33142b
# ╟─3d22ee0d-9fdf-4b43-9a05-6a905a4cd603
# ╟─37692f49-6a9d-42e4-9611-edca98360ea8
# ╟─8101f514-f5b3-4468-bc66-d9863c0d1111
# ╟─70836590-75d0-4091-908c-3e164f6142f3
# ╟─b35340f9-6d86-4cd7-bb5d-75a876378656
# ╟─444b1a52-3beb-4d08-9cb5-99def7da04ba
# ╟─da548f5b-9912-4044-89c6-af431d84169f
# ╟─ff12f43e-a2a7-4fc0-993a-16cb09052c55
# ╟─10e1a07d-e031-46ae-ad35-6ee6f674f40d
# ╟─76d44e52-2ee2-4e34-aa07-c77b529b81e6
# ╟─e3d3a4f5-5297-4cd8-b975-139162f71692
# ╟─cb2e4c43-c1a1-43e4-8f8d-65a5ec8679a6
# ╟─c5a180b1-0424-48fe-b33c-d1706051c95d
# ╟─e5954020-5d7a-4f7f-8bee-056a77f2a12f
# ╟─1fc91017-1b9a-4dd2-986b-ee48eeda0b17
# ╟─a633e6af-7d71-4234-81c5-df861ccda5ee
# ╟─882ba5fd-d533-453f-9d34-bcde5623a1ff
# ╟─42037da5-792c-407e-935f-534bf35a739b
# ╟─89a8e5bd-f049-491e-a3ff-9e33bfccb289
# ╟─2b4a0dc3-8bad-4589-ae97-746bdd705050
# ╟─5f70d015-c2c6-4c45-badf-688613f0ea69
# ╟─e17f634a-64a8-4271-9657-248fb2b5c688
# ╟─635fc846-2711-41d4-b452-7a96e1a167c1
# ╟─f65644e7-cb25-46ad-b146-87cf7de69f72
# ╟─1581b975-6c0a-4bc1-8dc1-a6ea273e14f9
# ╟─f8f34671-ef4c-4300-a983-10d42d43fb9f
# ╟─22cbb1aa-c53f-45d6-891a-90c6f2b9e886
# ╟─ce35ddcb-5018-4cb9-b0c9-01fb4b14be40
# ╟─59a3e04f-c842-4b6d-b067-0525c9dda70b
# ╟─660b612b-fbc6-434a-b8ae-69f1213dfad4
# ╟─f3e404b7-3419-43e7-affa-217324a65534
# ╟─2cdd8751-7ec8-47b0-a174-fdc23e176921
# ╟─720774c4-9aec-4329-bc90-51350fea0191
# ╟─edc245bc-6571-4e65-a50c-0bd4b8d63b74
# ╟─4154585d-4eff-4ee9-8f33-dad0dfdd143c
# ╟─d5cc7ea6-d1a0-46e1-9fda-b7f7092eb73c
# ╟─f90957ea-4123-461f-9a3e-ae23a2848264
# ╟─8b596e98-4d0c-471e-bb25-20f492a9199b
# ╟─9aff2235-7da4-41a9-9926-4fe291d9a638
# ╟─72440a56-7a47-4a32-9244-cb0424a6fd79
# ╟─46180264-bddc-47d8-90a7-a16d0ea87cfe
# ╟─882da1c1-1358-4a40-8f69-b4d7cbc9387e
# ╟─0dff10ec-dd13-4cc2-b092-9e72454763cc
# ╟─726a1008-26e6-417d-a73a-57e32cb224b6
# ╟─8cc45465-78e7-4777-b0c7-bb842e4e51a8
# ╟─fc2206f4-fd0f-44f8-ae94-9dae77022fab
# ╟─c91fc381-613c-4c9b-98b6-f4dd897a69a5
# ╟─21b6b547-66ed-4231-830b-1a09adaf400c
# ╟─2ee96524-7abc-41af-a38c-f71635c738dc
# ╟─8436c5ce-d7be-4c02-9e89-7da701303263
# ╟─6a9e0bd9-8a2a-40f0-aaac-bf32d39ffac8
# ╟─df62e796-d711-4ea1-a0c2-e3ec6c501a78
# ╟─8493d307-9158-4821-bf3b-c368d9cd5fc5
# ╟─88d98d87-f3cf-42f4-9282-1e6f383934cd
# ╟─0e60180d-c4eb-4c83-9301-e3151ab828d5
# ╟─58d6a11e-a375-4f0b-84d7-b95e7cfd3033
# ╟─974f1b58-3ec6-447a-95f2-6bbeda43f12f
# ╠═f1261f00-7fc6-41bb-8706-0b1973d72955
# ╠═238e7b56-fb3a-4e9b-9c31-09c1f4a1df2a
# ╠═c9b5e47c-e0f1-4496-a342-e37df85d6de9
# ╠═ddbd8d06-4ff0-4a1c-8dfe-092072fd88d5
# ╠═cb02aee5-d082-40a5-b799-db6b4af557f7
# ╟─8deb1b8c-b67f-4d07-8986-2333dbadcccc
# ╟─f79bd8ab-894e-4e7b-84eb-cf840baa08e4
# ╟─af622189-e504-4633-9d9e-ab16c7293f82
# ╟─9267c4a4-74d1-4515-95d6-acc3b12e5ed6
# ╟─76cc8ca7-d17e-4cd7-a6de-4d606e0a0985
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
