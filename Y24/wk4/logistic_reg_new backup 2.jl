### A Pluto.jl notebook ###
# v0.19.37

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

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
	using StatsPlots
end

# ╔═╡ 86ac8000-a595-4162-863d-8720ff9dd3bd
begin
	using LogExpFunctions
	using StatsBase
	using Distributions
end

# ╔═╡ ece21354-0718-4afb-a905-c7899f41883b
begin
	using Logging
	Logging.disable_logging(Logging.Info); # or e.g. Logging.Info
end;

# ╔═╡ 2588893b-7978-45af-ac71-20cb2af34b10
using Zygote

# ╔═╡ 1aa0cc79-e9ca-44bc-b9ab-711eed853f00
using MLUtils

# ╔═╡ 99b2a872-7d59-4a7e-ab7e-79e825563f0c
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ 646cafb0-e9c3-4f85-8b38-ae75fafb7c61
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

# ╔═╡ 3e2e1ea8-3a7d-462f-ac38-43a087907a14
TableOfContents()

# ╔═╡ 7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
ChooseDisplayMode()

# ╔═╡ bc96a33d-9011-41ec-a19e-d472cbaafb70
md"""

# CS5014 Machine Learning Algorithms


#### Logistic regression 
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 1d1c759c-9b59-4f6e-a61f-02bd5e279f68
md"""

## Reading & references

##### Essential reading 


* **Logistic regression** [_Understanding Deep Learning_ by _Simon Prince_: Chapter 5.4](https://github.com/udlbook/udlbook/releases/download/v2.00/UnderstandingDeepLearning_28_01_24_C.pdf)



* **Stochastic gradient descent** [_Understanding Deep Learning_ by _Simon Prince_: Chapter 6.2](https://github.com/udlbook/udlbook/releases/download/v2.00/UnderstandingDeepLearning_28_01_24_C.pdf)


##### Suggested reading

* **More on logistic regression** [_Probabilistic Machine Learning_ by _Kevin Murphy_: Chapter 10](https://probml.github.io/pml-book/book1.html)

"""

# ╔═╡ bcc3cea5-0564-481b-883a-a45a1b870ba3
md"""
## Binary classification



"""

# ╔═╡ 2d70884d-c922-45e6-853e-5608affdd860
md"""

## A first attempt -- use regression


"""

# ╔═╡ 8765a8e8-6f73-4ad4-9604-6a9eb5991241
# begin
# 	plotly()
# 	scatter(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)), zlim=[-1.1, 1.1], label="ỹ=1", c=2)
# 	scatter!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], -1 * ones(sum(targets_D₂ .== 0)), label="ỹ=-1", c=1, framestyle=:zerolines)
# 	w_d₂ = linear_reg(D₂, targets_D₂_;λ=0.0)
# 	plot!(-5:1:5, -5:1:5, (x,y) -> w_d₂[1] + w_d₂[2] * x + w_d₂[3] * y, alpha =0.9, st=:surface, colorbar=false, c=:jet, title="First attempt: least square regression")


# 	# plot!(-5:1:5, -5:1:5, (x,y) -> 0, alpha =0.5, st=:surface, c=:gray)
# end

# ╔═╡ 8c69c448-c104-4cae-9352-10d4cec512a9
# md"""

# ## Least square -- classification decision boundary


# """

# ╔═╡ 68a0247b-9346-42fc-b2d2-4373e70a1789
# TwoColumn(md"""


# Note that the fitted hyperplane ``h(\mathbf{x})``

# * ``h(\mathbf{x}^{(i)}) = 0``: the decision boundary
# * ``h(\mathbf{x}^{(i)}) > 0``: one side of the boundary
# * ``h(\mathbf{x}^{(i)}) < 0``: the other side of the boundary


# To classify ``\hat{y}^{(i)}``:

# ```math
# \hat{y}^{(i)} = \begin{cases} \text{class 1} & h(\mathbf{x}^{(i)}) \geq 0 \\

# \text{class 2}& h(\mathbf{x}^{(i)}) < 0 
# \end{cases}
# ```

# """ , let
# 	plotly()
# 	scatter(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)), zlim=[-1.1, 1.1], label="ỹ=1", c=2, titlefontsize=10, legend=:bottom)
# 	scatter!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], -1 * ones(sum(targets_D₂ .== 0)), label="ỹ=-1", framestyle=:zerolines, c=1)
# 	w_d₂ = linear_reg(D₂, targets_D₂_;λ=0.0)
# 	plot!(-5:1:5, -5:1:5, (x,y) -> w_d₂[1] + w_d₂[2] * x + w_d₂[3] * y, alpha =0.9, st=:surface, colorbar=false, c=:jet, title="First attempt: decision boundary h(x) =0")
# 	if true
# 		# the intersection
# 		xs = -5:1:5
# 		ys = @. (- w_d₂[1] - w_d₂[2] * xs) / w_d₂[3]
# 		plot!(xs,  ys, zeros(length(xs)), lw=3, lc=:gray, ls=:solid, label="h(x)=0")
# 	end

# 	plot!(-5:1:5, -5:1:5, (x,y) -> 0, alpha =0.5, st=:surface, c=:gray)
# end)

# ╔═╡ bf83caa9-d9f9-48a2-9563-152fcbb8afdc
# md"""

# ##
# """

# ╔═╡ c78fbd4c-1e5f-4307-9ab3-9303b0744de0
# plot_ls_class

# ╔═╡ ea55660a-090b-453c-a141-b6867670ba48
# md"""

# ## Least square classifier fails


# ##### What if we *add some more data* ?

# * **_ideally_**, the old decision boundary **shouldn't** change much
# """

# ╔═╡ ffb7f8f1-739c-43a5-b13e-4045b50c9e05
# let

# 	gr()
# 	p2 = plot(D₃[targets_D₃ .== 1, 2], D₃[targets_D₃ .== 1, 3], st=:scatter, label="class 1", ratio = 1, c = 2)
# 	plot!(D₃[targets_D₃ .== 0, 2], D₃[targets_D₃ .== 0, 3], st=:scatter, framestyle=:origin, label="class 2", xlim=[-8, 8], legend=:topleft,  title="More data", c=1)
# 	tmin = 0
# 	tmax = 2π
# 	tvec = range(tmin, tmax, length = 100)

# 	plot!(2.2 .+ sin.(tvec) *1.6, -6 .+ cos.(tvec) * 2, lc=1, lw=3, label="", alpha=1.0 , fill=true, fillalpha=0.2, fillcolor=1)
# 	plot(plot(plot_ls_class, legend=:outerbottom), p2)
# end

# ╔═╡ 3315ef07-17df-4fe1-a7a5-3e8f039c0cc1
linear_reg(X, y; λ = 1) = (X' * X + λ *I) \ X' * y;

# ╔═╡ c936fe77-827b-4245-93e5-d239d467bffd
# md"""

# ## Least square classifier fails


# ##### The least-square classifier is overly *sensitive* to the extra data

# """

# ╔═╡ 86420ad6-3995-424c-9bdd-21214bf544df
# let
# 	gr()
# 	plt = plot(D₃[targets_D₃ .== 1, 2], D₃[targets_D₃ .== 1, 3], st=:scatter, c= 2, label="class 1", ratio=1, legend=:bottomright)
# 	plot!(D₃[targets_D₃ .== 0, 2], D₃[targets_D₃ .== 0, 3], st=:scatter, c=1, framestyle=:origin, label="class 2", xlim=[-10, 8])
# 	w_d₃ = linear_reg(D₃, targets_D₃_;λ=0.0)
# 	# w_d₂
# 	plot!(-7.5:1:7, (x) -> - w_d₂[1]/w_d₂[3] - w_d₂[2]/w_d₂[3] * x, lw=2.5, lc=:gray,ls=:solid,  label="", title="")
# 	plot!(-6:1:6, (x) -> - w_d₃[1]/w_d₃[3] - w_d₃[2]/w_d₃[3] * x, lw=4, lc=2, label="New decision boundary: "*L"h(\mathbf{x}) =0", title="Least square classifier with new data")

# 	minx, maxx = minimum(D₃[:,2])-2, maximum(D₃[:,2])+2
# 	miny, maxy = minimum(D₃[:,3])-2, maximum(D₃[:,3])+2
# 	xs, ys = meshgrid(minx:0.25:maxx, miny:0.25:maxy)
# 	colors = [dot(w_d₃ , [1, x, y]) > 0.0   for (x, y) in zip(xs, ys)] .+ 1
# 	scatter!(xs, ys, color = colors,
#             markersize = 1.5, label = nothing, markerstrokewidth = 0,
# 		markeralpha=0.5, xlim= [minx, maxx], ylim =[miny, maxy], framestyle=:origin)


# 	plot(plot_ls_class, plt, legendfontsize=6, titlefontsize=8, legend=:outerbottom)
# end

# ╔═╡ cdccf955-19f3-45bb-8dfe-2e15addcdd32
# md"""

# ## Why?


# **Firstly**, the sum of squared error (SSE) loss doesn't make sense

# ```math
# \large
# \text{SSE Loss}(\mathbf{w}) = \frac{1}{2} \sum_{i=1}^{n} (y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)})^2
# ```

# * least square tries to please every data point (equally) in squared loss sense


# """

# ╔═╡ 4d474d9f-50dc-4c9f-9505-65e024c83f38
# let
# 	plotly()
# 	scatter(D₃[targets_D₃ .== 1, 2], D₃[targets_D₃ .== 1, 3], ones(sum(targets_D₂ .== 1)), zlim=[-2,2], label="y=1", c=2)
# 	scatter!(D₃[targets_D₃ .== 0, 2], D₃[targets_D₃ .== 0, 3], -1 * ones(sum(targets_D₃ .== 0)), label="y=-1", framestyle=:zerolines, c=1)
# 	w_d₃ = linear_reg(D₃, targets_D₃_;λ=0.0)
# 	plot!(-10:1:10, -10:1:10, (x,y) -> w_d₃[1] + w_d₃[2] * x + w_d₃[3] * y, alpha =0.8, st=:surface, color=:jet, colorbar=false, title="Why least square classifier fails")
# end

# ╔═╡ 331571e5-f314-42db-ae56-66e64e485b85
md"""

## Problem with the model

##### The prediction function is unbounded

```math
\Large
\begin{align}
h(\mathbf{x}; \mathbf{w}) &= \mathbf{w}^{\top} \mathbf{x}^{(i)} \\
&\in (-\infty, \infty)
\end{align}
```


##### But, the target ``y`` is *binary*
  * *e.g.* the output ``h(\mathbf{x}) = 100``, or ``-100`` does not make sense


"""

# ╔═╡ c8e55a60-0829-4cc7-bc9b-065809ac791c
md"""

## `Logistic` or `Sigmoid`

"""

# ╔═╡ 5ed6a7a3-7871-467f-9ff0-779e2c403bac
TwoColumn(md"""

\

**`logistic` function** (also known as `sigmoid`)

```math
\large
\texttt{logistic}(x) \triangleq \sigma(x) = \frac{1}{1+e^{-x}}.
``` 

* ``\sigma`` as a shorthand notation for the logistic activation

* it squeezes a line to 0 and 1
  * ``x\rightarrow \infty``, ``\sigma(x) \rightarrow 1``
   * ``x\rightarrow -\infty``, ``\sigma(x) \rightarrow 0``""", begin
gr()
plot(-10:0.2:10, logistic, c=7, xlabel=L"x",  label=L"\texttt{logistic}(x)", legend=:topleft, lw=3, size=(350,300),  framestyle=:origin, ylim =(-0.4, 1.4))

plot!(-10:1:10, x -> 1, c=:gray, lw=2, label="", ls=:dash)
plot!(-10:1:10, x -> 0, c=:gray, lw=2, label="", ls=:dash)

end

)

# ╔═╡ 7ad891af-18d2-48d9-a9a6-f322fdd30bed
md"""
## `logistic` function's derivative ``\sigma'(x)``

```math
\Large
\boxed{\sigma'(x) = \sigma(x) (1-\sigma(x))}
```

* ##### ``x \rightarrow \infty``, ``\sigma(x) \rightarrow 1``, ``\sigma(x)' \rightarrow 0``
* ##### ``x \rightarrow -\infty``, ``\sigma(x) \rightarrow 0``, ``\sigma(x)' \rightarrow 0``
"""

# ╔═╡ 15c50511-3c4a-46b4-8712-fecbdd181bfd
md"Add ``\sigma'(x)``: $(@bind add_dsigma CheckBox(default=false)) Add saturate area: $(@bind add_sat CheckBox(default=false))"

# ╔═╡ ffb51b97-d873-4142-bc99-d48c9a8c390a
let
	x0 =6
	plt = plot(-10:0.1:10, logistic,ylim =[-.25, 1.25],  lw=2, label=L"\sigma(x)", title=L"\sigma(x)" * " and "*L"\sigma'(x)",xlabel=L"x", xlim =[-10,10], yticks =[0.0, 0.25, 0.5, 0.75, 1.0], framestyle=:origin, size=(750,400), legend=:outerright, legendfontsize=14, ytickfontsize=14, xlabelfontsize=14, titlefontsize=16)

	x0 = 6 
	plot!( x -> 1, c=:gray, lw=2, label="", ls=:dash)
	plot!( x -> 0, c=:gray, lw=2, label="", ls=:dash)
	if add_dsigma
		plot!(-10:0.1:10, (x) -> logistic(x) * (1-logistic(x)), lw=4,lc=2, label=L"\sigma'(x)")
	end
	if add_sat
		vspan!([-11, -x0], ylim =(-0.0,1.0),  alpha=0.5, c=:gray, label="")
		vspan!([x0, 11], ylim =(-0.0,1.0), alpha=0.5, c=:gray, label="")	
		annotate!([-8], [0.68], Plots.text(L"\longleftarrow", 30, :blue))
		annotate!([8], [0.68], Plots.text(L"\longrightarrow", 30, :blue))

		annotate!([-8], [0.6], Plots.text(L"0 \leftarrow \sigma(x)", :blue))
		annotate!([8], [0.6], Plots.text(L"\sigma(x)\rightarrow 1", :blue))

		annotate!([-8], [0.48], Plots.text(L"\longleftarrow", 30, :red))
		annotate!([8], [0.48], Plots.text(L"\longrightarrow", 30, :red))
		annotate!([-8], [0.4], Plots.text(L"0 \leftarrow \sigma'(x)",  :red))
		annotate!([8], [0.4], Plots.text(L"\sigma'(x) \rightarrow 0",:red))
	else
		
	end
	plt
end

# ╔═╡ 53e69939-09b2-41b5-a5a1-21b075275f8c
md"""

## `logistic` function


Now feed a linear function ``\large h(x) = w_1 x + w_0`` to ``\large \sigma(\cdot)``


```math
\Large
(\sigma \circ h) (x) = \sigma(h(x)) = \sigma(w_1 x+ w_0) = \frac{1}{1+e^{-w_1x -w_0}}
```
"""

# ╔═╡ 5d15c0a7-1038-4d44-8dc4-986809b5d45e
# md"""

# Add ``\sigma(h(x))``: $(@bind add_sigma_h CheckBox(default=false))
# """

# ╔═╡ e11f7ee9-e50a-4c7c-a5aa-acc6f712a063
# md"Slope: ``w_1`` $(@bind w₁_ Slider([(0.0:0.00001:4.)...; (4.0:0.0001:10)...], default=1.0, show_value=true)),
# Intercept: ``w_0`` $(@bind w₀_ Slider(-10:0.1:10, default=0, show_value=true))"

# ╔═╡ 5997601f-5ee4-4ade-91f6-3bbdaa46f17c
# let
# 	k, b= w₁_, w₀_
# 	gr()
# 	plot(-40:0.1:40, (x) -> logistic( x ), xlabel=L"x", label=L"\sigma(1x + 0.0)", legend=:topleft, lw=3, size=(800,800), c=1, ratio=10)
# 	plot!(-40:0.1:40, (x) -> k * x+ b, ylim=[-2, 2], xlim =[-20, 20],label=L"h(x) =%$(round(w₁_; digits=2)) x + %$(w₀_)", lw=3, lc=7, ls=:dash,  framestyle=:origin, legend=:outerbottom, legendfontsize=14, labelfontsize=14, ytickfontsize=14)
	
# 	x₀_ = -b/k
# 	if add_sigma_h
# 		plot!(-40:0.1:40, (x) -> logistic(k * x + b), xlabel=L"x", label=L"\sigma(h(x))=\sigma(%$(round(w₁_; digits=2)) x + %$(w₀_))", lw=3, lc=7)
# 		plot!([x₀_, x₀_], [0, .5], lw=2, ls=:dash, lc=:gray, st=:line, label="")
# 		plot!([.5, x₀_], [.5, .5], lw=2, ls=:dash, lc=:gray, st=:line, label="")
# 	end


# 	plot!( x -> 1, c=:gray, lw=1, label="", ls=:dash)
# 	plot!( x -> 0, c=:gray, lw=1, label="", ls=:dash)
	
# end

# ╔═╡ 4cac5129-3b3d-42c4-839c-254ac26199f1
md"""

## Logistic function 
##### -- higher-dimensional


##### It *squeezes* a *hyperplane* (or multiple lines) instead of a *line*

```math
\large
h(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}\tag{a hyper-plane}
```


```math
\large
\sigma(h(\mathbf{x})) = σ(\mathbf{w}^\top \mathbf{x}) = \frac{1}{1+e^{-\mathbf{w}^\top \mathbf{x}}}\tag{a S shaped plane}
```
"""

# ╔═╡ 137441e2-2f6c-43bb-b1a2-b240ee3ab6e7
ww = [1, 1]

# ╔═╡ 246b8623-2f47-4ac1-b562-75d50889e6d8
md"Add ``x_2=c`` vertical plane: $(@bind add_x2plane CheckBox(default=false)), add more planes: $(@bind add_more_x2_planes CheckBox(default=false))"

# ╔═╡ 206b39d0-8a4b-43ed-9e56-d09e4da59883


# ╔═╡ 3deafee4-92a0-4879-b618-20d58ec731d1
let
	plotly()
	b = ww
	w₀ = 10
	f(x1, x2) = dot(b, [x1, x2]) + w₀
	plt = plot(-15:2:15, -15:2:15, (x1, x2) -> f(x1, x2), st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.8, framestyle=:zerolines, c=:coolwarm, colorbar=false, camera=(-5, 30))

	min_z, max_z = extrema([f(x1, x2) for x1 in -15:2:15 for x2 in -15:2:15])
	x = range( -15,stop=15,length=10)
	y = range( -15,stop= 15,length=10)
	
	x0 = [0, 0.0]
	eps_x = range(x0[1] -1*eps() ,stop=x0[1]  +eps(),length=10) #create a very small range
	eps_y = range(x0[2]-1*eps(),stop=x0[2] +eps(),length=10)
	X = repeat(range(min_z, max_z, length=10), 1, length(y))
	Y = repeat(range(min_z, max_z, length=10)',length(x),1)
	if add_x2plane
		surface!(x, eps_y, X, surftype=(surface = true, mesh = true), c=2, alpha=0.35)  
		xs = -15:1:15
		ys = x0[2] * ones(length(xs))
		zs = [f(xs[i], ys[i]) for i in 1:length(ys)]
		path3d!(xs, ys, zs, lw=5, label="", c=2)

		if add_more_x2_planes
			# for x2_ in range(-15, 15, 7)
			# 	eps_y_ = range(x2_-1*eps(),stop=x2_ +eps(),length=10)
			# 	surface!(x, eps_y_, X, surftype=(surface = true, mesh = true), c=2, alpha=0.35)  
			# 	ys = x2_ * ones(length(xs))
			# 	zs = [f(xs[i], ys[i]) for i in 1:length(ys)]
			# 	path3d!(xs, ys, zs, lw=5, label="", c=2)
			# end
		end
	end


	# plot!(-15:2:15, -10:2:10, (x1, x2) -> 10.0, st=:surface, alpha=0.8, title="Level set")
	# x

	plt
end

# ╔═╡ 6ba6bed9-d913-4511-9829-787fe8a09fa7
md"""
## Logistic function 
##### -- higher-dimensional
"""

# ╔═╡ 22b15086-cde5-4e07-90e2-0cfd82c34889
wv_ = [1, 1] * 1;

# ╔═╡ a078a211-e390-4158-9076-d154921bf8b4
md"``\mathbf{w}=``$(latexify_md(wv_))"

# ╔═╡ 324ea2b8-c350-438d-8c8f-6404045fc19f
# md"""

# ## Logistic function

# """

# ╔═╡ 206c82ee-389e-4c92-adbf-9578f7125418
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.

# ╔═╡ 1a062043-8cfe-4946-8eb0-688d3896229e
md"""

## Logistic regression -- as a graph


The function is actually a simpliest **neural network**


```math
\large 
\sigma(\mathbf{x}) = \frac{1}{1+ e^{- \mathbf{w}^\top\mathbf{x}}}
``` 


which can be represented as a **computational graph**

* ``z = \mathbf{w}^\top \mathbf{x}`` is called the *logit* value
"""

# ╔═╡ 88093782-d180-4422-bd92-357c716bfc89
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS3105/logistic_reg_neuron.png
' width = '400' /></center>"

# ╔═╡ 91a6f3de-01c6-4215-9928-1ecce113adc1
md"""


## Second attempt: `SSE` loss

##### Can we reuse the sum squared error (SSE) loss?

```math
\large
loss(\hat{\mathbf{y}}, \mathbf{y}) = \frac{1}{2} \sum_{i=1}^n (y^{(i)} - \hat{y}^{(i)})^2 
```


* ##### where ``\hat{y}^{(i)} = \sigma(\mathbf{w}^\top\mathbf{x}^{(i)})`` is the logistic function
"""

# ╔═╡ 2459a2d9-4f48-46ab-82e5-3968e713f15f
html"<center><img src='https://carpentries-incubator.github.io/ml4bio-workshop/assets/logit_nodes.png
' width = '400' /></center>";

# ╔═╡ 32eb5c4f-8fde-44c1-a748-905de6aaf364
md"""

## The loss surface -- visualisation

"""

# ╔═╡ 16d84881-d40a-4694-9812-6896b336accc
function logistic_mse_loss(w, X, y; agg=mean)
	(0.5 * (y .- logistic.(X * w)).^2) |> agg
end;

# ╔═╡ dd0a4446-65ed-4b3d-87a6-ba950457aa72
function zeroone_loss(w, X, y; agg=mean)
	# (0.5 * (y .- logistic.(X * w)).^2) |> agg
	y_ = 2 * y .- 1
	(sign.(X * w) .!= y_) |> agg
end;

# ╔═╡ 72697e34-6f63-4d6b-a9ff-ae23a00d4ce2
md"""

## Non-convex loss: a bad starting point
"""

# ╔═╡ 593eb726-47a5-4783-8872-03a75c6cfb89
md"""

## Non-convex loss: if you are lucky 
"""

# ╔═╡ 27385a11-e8de-4d88-898a-8b9bd0a73ee2
md"""


## `SSE` loss is not convex


##### The loss, *however*, is *non-convex*

> #### It has too many *flat* areas!

##### The *consequence*: gradient descent might struggle!
* ##### quite depends on the *initial starting* guess
"""

# ╔═╡ e7dac1a5-dc63-4364-a7d6-13ab9808c9c6
# let
# 	gr()
# 	w0 = [-4.5, -4.5]
# 	bias = 0.0	
# 	grad = logistic_mse_loss_grad
# 	γ = 0.5
# 	maxiters = 1_000
# 	ws = [w0]
# 	loss = []
# 	for i in 1:maxiters
# 		li, gradw = grad(w0)
# 		w0 -= γ * gradw[1]
# 		push!(ws, w0)
# 		push!(loss, li)
# 	end
# 	ws = hcat(ws...)
# 	traces = ws
# 	plt = plot(-8:0.5:10, -5:0.5:10, (w1, w2) -> logistic_mse_loss([bias, w1, w2], D₂, targets_D₂; agg=sum), st=:contourf, c=:jet, colorbar=false, title="Squared error loss",  ratio=1, alpha=0.8, xlim =(-8, 10), ylim = (-5, 10))
# 	wt = traces[:, 1]
# 	scatter!([wt[1]], [wt[2]], markershape=:xcross, label="start", markerstrokewidth=4, mc=3, markersize=6)
# 	anim = @animate for t in 2:20:1_000
# 		plot!([traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
# 		plot!([wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="", title="Iteration $(t)")
# 		wt = traces[1:2, t]
# 	end 
# 	gif(anim, fps=5)
# end

# ╔═╡ 52ff5315-002c-480b-9a4b-c04124498277
md"""
## Recap: probabilistic linear regression model


> $\large \begin{align}p(y^{(i)}|\mathbf{x}^{(i)}, \mathbf{w}, \sigma^2) &= \mathcal{N}(y^{(i)};  \mathbf{w}^\top \mathbf{x}^{(i)} , \sigma^2)\end{align}$

* ``y^{(i)}`` is a univariate Gaussian with mean $\mathbf{w}^\top \mathbf{x}^{(i)}$ and variance $\sigma^2$ 


"""

# ╔═╡ 8fd7703b-db80-4aa4-977f-c7e3ad1f9fb6
begin
	show_img("plinreggif.gif", w=450)
end

# ╔═╡ 0469b52f-ce98-4cfa-abf3-be53630b30dd
md"""

## Probabilistic model for logistic regression



Since ``y^{(i)} \in \{0,1\}``, a natural choice of **likelihood** function is Bernoulli (the outcome is binary)

> ```math
> \large
> 
> p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) = \texttt{Bernoulli}(\sigma^{(i)}) =\begin{cases}\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} =1
> \\
> 1-\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} = 0   \end{cases}
> ```

* short-hand notation ``\sigma^{(i)} =\sigma(\mathbf{w}^\top\mathbf{x}^{(i)})``


"""

# ╔═╡ 4f884fee-4934-46c3-8352-0105652f8537
md"""


## Probabilistic model for logistic regression (cont.)



Since ``y^{(i)} \in \{0,1\}``, a natural choice of **likelihood** function is Bernoulli (the outcome is binary)

> ```math
> \large
> 
> p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) = \texttt{Bernoulli}(\sigma^{(i)}) =\begin{cases}\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} =1
> \\
> 1-\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} = 0   \end{cases}
> ```

* short-hand notation ``\sigma^{(i)} =\sigma(\mathbf{w}^\top\mathbf{x}^{(i)})``

##### The generative story therefore is

---


Given fixed ``\{\mathbf{x}^{(i)}\}``, which are assumed fixed and non-random


for each ``\mathbf{x}^{(i)}``
  * compute regression function ``\sigma(\mathbf{x}^{(i)}) =\sigma(\mathbf{w}^\top \mathbf{x}^{(i)})``
  * *generate* ``y^{(i)} \sim \texttt{Bernoulli}(\sigma(\mathbf{x}^{(i)}))``

---
"""

# ╔═╡ 9dbf5502-fa44-404f-88ae-be3488e3e41c
md"""

##

"""

# ╔═╡ 5dc4531b-cc10-41de-b055-2ed4270ae2cc
md"Add true function ``\sigma(x; \mathbf{w})``: $(@bind add_h CheckBox(default=false)),Add false function ``1-\sigma(x; \mathbf{w})``: $(@bind add_negh CheckBox(default=false))
Add ``p(y^{(i)}|x^{(i)})``: $(@bind add_pyi CheckBox(default=false)),
Add ``y^{(i)}\sim p(y^{(i)}|x^{(i)})``: $(@bind add_yi CheckBox(default=false))
"

# ╔═╡ 479fb04e-c95f-4522-9195-6e6e9648565b
begin
	Random.seed!(2345)
	n_obs = 20
	# the input x is fixed; non-random
	xs = range(0.2, 19.5; length = n_obs)
	# xs = sort(rand(n_obs) * 20)
	true_w = [-11, 1]/2.25
	# true_σ² = 0.05
	ys = zeros(Bool, n_obs)
	for (i, xⁱ) in enumerate(xs)
		hⁱ = true_w' * [1, xⁱ]
		# ys[i] = hⁱ + rand(Normal(0, sqrt(true_σ²)))
		ys[i] = rand() < logistic(hⁱ)
	end
end

# ╔═╡ 73d71e0a-f957-426c-a356-3238032603d9
md"
Select ``x^{(i)}``: $(@bind add_i Slider(1:length(xs); show_value=true))
"

# ╔═╡ a67f4cb1-6558-466d-9ebc-df21dd83ce96
TwoColumnWideLeft(let
	gr()
	plt = plot(xs, zeros(length(xs)), st=:scatter, framestyle=:origin, labels=L"x", color=:black, ms=5, markershape=:x, xlabel="Weight of the animal (kg)", ylim=[-0.2, 1.2],legend=:outerbottom, size=(450,350))
	xis = xs
	i = add_i
	hxi =  logistic(true_w[1] + true_w[2]*xs[i])
	
	if add_h
		plot!(-0.01:0.01:21, (x) -> logistic(true_w[1] + true_w[2]*x), lw=2, label="the bias: " * L"h(x)")
		plot!([xs[i]], [hxi],st=:scatter, markershape=:circle, ms=4, markerstrokewidth=1, c=2, label="")

		plot!([xs[i], xs[i]], [0,hxi], st=:path, label="", ls=:dash, c=2, lw=1.5, alpha=0.8)
		if add_negh
			plot!(-0.01:0.01:21, (x) -> 1-logistic(true_w[1] + true_w[2]*x), lw=2, ls=:dash, c=1, label="")
			plot!([xs[i]], [1-hxi],st=:scatter, markershape=:circle, ms=4, markerstrokewidth=1, c=1, label="")

			plot!([xs[i], xs[i]], [0,1-hxi], st=:path, label="", ls=:dot, c=1, lw=4, alpha=0.6)
		end

		plot!([xs[i]], [hxi],st=:scatter, markershape=:circle, ms=4, markerstrokewidth=1, c=2, label="")

		plot!([xs[i], xs[i]], [0,hxi], st=:path, label="", ls=:dash, c=2, lw=1.5)
	end

	plot!([xs[i]], [0],st=:scatter, markershape=:x, ms=5, markerstrokewidth=5, c=:black, label="")


	labels = [y == 0 ? text("cat", 10, "Computer Modern" , :blue) : text("dog", 10, "Computer Modern" , :red) for y in ys]
	if add_yi
		shapes = [:diamond, :circle]
		scatter!(xis[1:i],ys[1:i], alpha=0.5, markershape = shapes[ys[i] + 1], label="observation: "*L"y^{(i)}\sim \texttt{Bern}(\sigma^{(i)})", c = ys[1:i] .+1, markersize=8; annotation = (xis[1:i],ys[1:i] .+0.1, labels[1:i]))
	end

	plt
end, 
	let 
	gr()
	xis = xs
	i = add_i
	if add_pyi
		x = xis[i]
		μi = dot(true_w, [1, x])
		σ = logistic(μi)
		# scatter!([x],[μi], markerstrokewidth =1, markershape = :diamond, c=:grey, label="signal: "*L"h(x)", markersize=3)
		# plot!(ys_ .+x, xs_, c=:grey, label="", linewidth=2)
		plot(["y = cat", "y = dog"], [1-σ, σ], c= [1,2], st=:bar, orientation=:v, size=(250,250), title="Bias: "*L"\sigma(x^{(i)})=%$(round(σ, digits=2))", ylim =(0, 1.02), label="", ylabel=L"P(y|x)" )
	else
		plot(size=(250,250))
	end
	
end)

# ╔═╡ 3e980014-daf7-4d8b-b9e6-14a5d078e3b6
md"""
## Demonstration (animation)
"""

# ╔═╡ 6cbddc5d-ae3f-43ac-9b7a-bbc779739353
begin
	bias = 0.0 # 2
	slope = 1 # 10, 0.1
end;

# ╔═╡ 5d2f56e8-21b2-4aa9-b450-40f7881489e0
let
	gr()
	n_obs = 20
	# logistic = σ
	Random.seed!(4321)
	xs = sort(rand(n_obs) * 10 .- 5)
	true_w = [bias, slope]
	# true_σ² = 0.05
	ys = zeros(Bool, n_obs)
	for (i, xⁱ) in enumerate(xs)
		hⁱ = true_w' * [1, xⁱ]
		# ys[i] = hⁱ + rand(Normal(0, sqrt(true_σ²)))
		ys[i] = rand() < logistic(hⁱ)
	end

	x_centre = -true_w[1]/true_w[2]

	plt = plot(xs, zeros(length(xs)), st=:scatter, framestyle=:origin, labels=L"x", color=:black, ms=5, markershape=:x, xlabel=L"x", ylim=[-0.1, 1.2], ylabel=L"y", legend=:outerbottom)
	# true_w =[0, 1]
	plot!(plt, min(-5 + x_centre, -5):0.01:max(x_centre +5, 5), (x) -> logistic(true_w[1] + true_w[2]*x), lw=1.5, label=L"\sigma(x)", title="Probabilistic model for logistic regression")
	plot!(plt, min(-5 + x_centre, -5):0.01:max(x_centre +5, 5), (x) -> 1-logistic(true_w[1] + true_w[2]*x),lc=1, lw=1.5, label=L"1-\sigma(x)")

	xis = xs

	anim = @animate for i in 1:length(xis)
		x = xis[i]
		scatter!(plt, [xis[i]],[ys[i]], markershape = :circle, label="", c=ys[i]+1, markersize=5)
		vline!(plt, [x], ls=:dash, lc=:gray, lw=0.2, label="")
		plt2 = plot(Bernoulli(logistic(true_w[1] + true_w[2]*x)), st=:bar, yticks=(0:1, ["negative", "positive"]), xlim=[0,1.01], orientation=:h, yflip=true, label="", title=L"p(y|{x})", color=1:2)
		plot(plt, plt2, layout=grid(2, 1, heights=[0.85, 0.15]), size=(650,500))
	end
	# ys = xs * true_w[2] .+ true_w[1] + randn(length(xs)) * sqrt(σ²0)
	
	# 	x = xis[i]
	# 	
	# end

	gif(anim; fps=4)
end

# ╔═╡ 64a5e292-14b4-4df0-871d-65d9fec6201d
# let
# 	gr()
# 	Random.seed!(4321)
# 	xs = sort(rand(n_obs) * 10 .- 5)
# 	true_w = [bias, slope]
# 	# true_σ² = 0.05
# 	ys = zeros(Bool, n_obs)
# 	for (i, xⁱ) in enumerate(xs)
# 		hⁱ = true_w' * [1, xⁱ]
# 		# ys[i] = hⁱ + rand(Normal(0, sqrt(true_σ²)))
# 		ys[i] = rand() < logistic(hⁱ)
# 	end

# 	x_centre = -true_w[1]/true_w[2]

# 	plt = plot(xs, zeros(length(xs)), st=:scatter, framestyle=:origin, labels=L"x", color=:black, ms=5, markershape=:x, xlabel=L"x", ylim=[-0.1, 1.2], ylabel=L"y", legend=:outerbottom)
# 	# true_w =[0, 1]
# 	plot!(min(-5 + x_centre, -5):0.01:max(x_centre +5, 5), (x) -> logistic(true_w[1] + true_w[2]*x), lw=2, label="the regression function: " * L"\sigma(x)", title="Probabilistic model for logistic regression")

# 	xis = xs

# 	# ys = xs * true_w[2] .+ true_w[1] + randn(length(xs)) * sqrt(σ²0)
# 	anim = @animate for i in 1:length(xis)
# 		x = xis[i]
# 		scatter!([xis[i]],[ys[i]], markershape = :circle, label="", c=ys[i]+1, markersize=5)
# 	end

# 	gif(anim; fps=5)
# end

# ╔═╡ 96b18c60-87c6-4c09-9882-fbbc5a53f046
md"""

## Probabilistic model for binary classification



> ```math
> \large
> 
> p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) = \texttt{Bernoulli}(\sigma^{(i)}) =\begin{cases}\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} =1
> \\
> 1-\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} = 0   \end{cases}
> ```

* short-hand notation ``\sigma^{(i)} =\sigma(\mathbf{w}^\top\mathbf{x}^{(i)})``


Recall the above likelihood can be written in one-line

```math
\large
p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) =  (\sigma^{(i)})^{y^{(i)}} (1-\sigma^{(i)})^{1- y^{(i)}}
```


And its log transformation (log-likelihood) is 


```math
\large
\ln p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) =  {y^{(i)}} \ln \sigma^{(i)}+ (1- y^{(i)}) \ln (1-\sigma^{(i)})
```
"""

# ╔═╡ 35d5f991-5194-4174-badc-324ad7d15ac3
md"""

## (Log-)likelihood function



Therefore, the log-likelihood function is

```math
\large
\begin{align}
\ln p(\mathcal{D}|\mathbf{w}, \sigma^2, \{\mathbf{x}^{i}\}) &= \sum_{i=1}^n \ln p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) \\

&= \sum_{i=1}^n {y^{(i)}} \ln \sigma^{(i)}+ (1- y^{(i)}) \ln (1-\sigma^{(i)})
\end{align}
```


In practice, we use take the average/mean loss rather than the sum

```math
\large
\frac{1}{n}\ln p(\mathcal{D}|\mathbf{w}, \sigma^2, \{\mathbf{x}^{i}\}) =\colorbox{pink}{$\frac{1}{n}$}\sum_{i=1}^n {y^{(i)}} \ln \sigma^{(i)}+ (1- y^{(i)}) \ln (1-\sigma^{(i)})
```



## MLE

**Learning** is just to maximise the log likelihood function *w.r.t.* ``\mathbf{w}``, which is **MLE**


```math
\large
\hat{\mathbf{w}}_{\text{MLE}} \leftarrow \arg\max_{\mathbf{w}} \ln p(\mathcal{D}|\mathbf{w}, \sigma^2, \{\mathbf{x}^{i}\})
```

or minimise its negative (also known as **cross-entropy** (CE) loss)


```math
\large
\hat{\mathbf{w}}_{\text{MLE}} \leftarrow \arg\min_{\mathbf{w}} \underbrace{-\ln p(\mathcal{D}|\mathbf{w}, \sigma^2, \{\mathbf{x}^{i}\})}_{\text{CE loss}(\mathbf{w})}
```

"""

# ╔═╡ f693814d-9639-4abb-a5b4-e83fe7a28a77
md"""

## Cross-entropy loss -- how ?


##### *Cross-entropy* loss but _WHY_ and _How_?
\

Consider one observation ``\{\mathbf{x}^{(i)}, y^{(i)}\}`` only, the loss is

```math
\large
\begin{align}
 \text{CE}(\mathbf{w}) = - {y^{(i)}} \ln \sigma^{(i)}- (1- y^{(i)}) \ln (1-\sigma^{(i)})
\end{align}
```
When ``y^{(i)} = 1``, *i.e.* the true label is 1, the loss becomes 

```math
\large
\begin{align}
y^{(i)} = 1\; \Rightarrow\; \text{CE error}^{(i)} &= - 1 \ln \sigma^{(i)}- (1- 1) \ln (1-\sigma^{(i)})\\
&= - \ln (\sigma^{(i)})
\end{align}
```
"""

# ╔═╡ 67f7449f-19b8-4607-9e32-0f8a16a806c0
TwoColumn(md"""

\
\


When ``y^{(i)} = 1``, *i.e.* the true label is 1, the loss becomes 

\

```math
\begin{align}
 \text{CE}(\mathbf{w}) &= - 1 \ln \sigma^{(i)}- (1- 1) \ln (1-\sigma^{(i)})\\
&= - \ln (\sigma^{(i)})
\end{align}
```


* when the prediction is correct,  the loss is zero
* when the prediction is wrong, the loss is `Inf` loss

""", let
	gr()
	plot(0:0.005:1, (x) -> -log(x), lw=2, xlabel="Predicted probability: "* L"\sigma^{(i)}", ylabel="loss", label=L"-\ln \sigma^{(i)}", title=L"y^{(i)}=1"* ": i.e. class label is 1", annotate = [(1, 0.9, text("perfect pred", "Computer Modern", :right, rotation = 270 ,:green, 12)), (0.11, 3.5, text("worst pred", :right, "Computer Modern", rotation = 270 ,:red, 12))], size=(350,350))


	quiver!([1], [0.8], quiver=([1-1], [0-0.8]), c=:green, lw=3)
	quiver!([0.07], [5], quiver=([0.0- 0.06], [5-5]), c=:red, lw=3)

	quiver!([0.25], [2], quiver=([0.75- 0.25], [2-2]), c=:green, lw=3)
	quiver!([0.75], [3], quiver=([0.25- 0.75], [3-3]), c=:red, lw=3)
	annotate!(0.5, 2.1, text("better","Computer Modern", :green, :bottom))
	annotate!(0.5, 3.1, text("worse","Computer Modern", :red, :bottom))
end)

# ╔═╡ fb0361a6-c967-4dcd-9002-55ae25e225a5
aside(tip(md"""

Recall ``\sigma^{(i)} = p(y^{(i)}=1|\mathbf{x}^{(i)})``

"""))

# ╔═╡ 5c0eaab3-de6d-457f-a0c3-8ea6b5da2c88
md"""

##
"""

# ╔═╡ de93ac5e-bec6-4764-ac6d-f84076ff20ee
TwoColumn(md"""

\
\


When ``y^{(i)} = 0``, *i.e.* the true label is 0, the loss becomes 

\

```math
\begin{align}
 \text{CE}(\mathbf{w}) &= - 0 \ln \sigma^{(i)}- (1- 0) \ln (1-\sigma^{(i)})\\
&= - \ln (1-\sigma^{(i)})
\end{align}
```
""", let
	gr()
	plot(0:0.005:1, (x) -> -log(1-x), lw=2, xlabel="Predicted probability: "* L"\sigma^{(i)}", ylabel="loss", label=L"-\ln(1-\sigma^{(i)})", title=L"y^{(i)}=0"* ": the true class label is 0", size=(350,350))


	# quiver!([0], [0.8], quiver=([1-1], [0-0.8]), c=:red, lw=3)
	
	# quiver!([0.07], [5], quiver=([0.0- 0.06], [5-5]), c=:green, lw=3)

	quiver!([0.25], [2], quiver=([0.75- 0.25], [2-2]), c=:red, lw=3)
	quiver!([0.75], [3], quiver=([0.25- 0.75], [3-3]), c=:green, lw=3)
	annotate!(0.5, 2.1, text("worse", "Computer Modern", :red, :bottom))
	annotate!(0.5, 3.1, text("better", "Computer Modern", :green, :bottom))
end)

# ╔═╡ a50cc950-45ca-4745-9f47-a3fbb28db782
md"""


## Surrogate loss


"""

# ╔═╡ 1421c748-083b-4977-8a88-6a39c9b9906d
TwoColumn(md"""

\
\

For classification, a more **natural** **loss** is 1/0 classification accuracy:


```math
\large
\begin{align}
\ell_{1/0}(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{n}\sum_{i=1}^n \mathbb{I}(y^{(i)} \neq \hat{y}^{(i)})\\
\text{where }\; \hat{y}^{(i)} = \mathbb{I}(\sigma^{(i)} > 0.5)
\end{align}
```

* however, this loss is **not** differentiable 
* the gradients are zero _everywhere_

""", let
	gr()

	plot(0:0.001:1, (x) -> x < .5, lw=4, xlabel="Predicted probability: "* L"\sigma^{(i)}", ylabel="loss", label=L"1/0"* " loss", title="When the class label is 1", ylim =[-0.2, 4], size=(350,350))

	# plot(0:0.005:1, (x) -> -log(x), lw=2, xlabel="Predicted probability: "* L"\sigma^{(i)}", ylabel="loss", label=L"-\ln \sigma^{(i)}", title=L"y^{(i)}=1"* ": i.e. class label is 1",  size=(550,350))


	# quiver!([1], [0.8], quiver=([1-1], [0-0.8]), c=:green, lw=3)
	# quiver!([0.07], [5], quiver=([0.0- 0.06], [5-5]), c=:red, lw=3)

	quiver!([0.25], [2], quiver=([0.75- 0.25], [2-2]), c=:green, la=0.5, lw=3)
	quiver!([0.75], [3], quiver=([0.25- 0.75], [3-3]), c=:red, lw=3)
	annotate!(0.5, 2.1, text("better", "Computer Modern", :green, :bottom))
	annotate!(0.5, 3.1, text("worse", "Computer Modern", :red, :bottom))
end)

# ╔═╡ 82baa991-6df5-4561-9643-52db96c5e99b
md"""


## Surrogate loss

"""

# ╔═╡ 9a02f4f8-38d6-44a4-9118-1440bfc4d271
TwoColumn(md"""

\
\
\
\

**Cross-entropy**: is a **surrogate loss** for the **1/0** loss
* it **approximates**/**traces** the desired loss

""", let
	gr()

	plot(0:0.001:1, (x) -> x < .5, lw=2, xlabel="Predicted probability: "* L"\sigma^{(i)}", ylabel="loss", label=L"1/0"* " loss", title="When the class label is 1", ylim =[-0.2, 5], size=(350,350))

	plot!(0:0.005:1, (x) -> -log(x), lw=2, xlabel="Predicted probability: "* L"\sigma^{(i)}", ylabel="loss", label="Cross Entropy loss")

	plot!(0:0.005:1, (x) -> (x-1)^2, lw=2, lc=4, label="Squared error loss")

	# plot!(0:0.005:1, (x) -> (1-x)^2+ (x-1)^2, lw=2, lc=4, ls=:dash, label="Bier loss")
	# quiver!([1], [0.8], quiver=([1-1], [0-0.8]), c=:green, lw=3)
	# quiver!([0.07], [5], quiver=([0.0- 0.06], [5-5]), c=:red, lw=3)

	quiver!([0.25], [2], quiver=([0.75- 0.25], [2-2]), c=:green, la=0.5, lw=3)
	quiver!([0.75], [3], quiver=([0.25- 0.75], [3-3]), c=:red, lw=3)
	annotate!(0.5, 2.1, text("better", "Computer Modern", :green, :bottom))
	annotate!(0.5, 3.1, text("worse", "Computer Modern", :red, :bottom))
end)

# ╔═╡ ce822f87-d2c4-434b-9b5d-b629962d2df2
md"""

# Logistic regression -- Learning
"""

# ╔═╡ e67259ac-e6a9-476d-9795-520e4b6a7b9b
md"""

## Learning -- gradient descent


**Learning** is to minimise the **cross-entropy** (CE) loss


```math
\large
\hat{\mathbf{w}}_{\text{MLE}} \leftarrow \arg\min_{\mathbf{w}} \underbrace{\frac{1}{n}\sum_{i=1}^n-{y^{(i)}} \ln \sigma^{(i)}- (1- y^{(i)}) \ln (1-\sigma^{(i)})}_{\text{CE Loss: } L(\mathbf{w})}
```



**Bach gradient descent algorithm:**

-----

* random guess ``\large\mathbf{w}_0``

* while **not converge**
  * ``\large\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma \nabla L(\mathbf{w}_{t-1})``
-----



"""

# ╔═╡ 3abbf43d-a288-46e6-989f-d92a0bef8b18
md"""
## Gradient derivation


Consider one training sample ``(\mathbf{x}^{(i)}, y^{(i)})``'s loss ``L^{(i)} (\mathbf{w})`` and its gradient ``\nabla{L}^{(i)}(\mathbf{w})``

```math
\large
\begin{align}
L^{(i)}(\mathbf{w}) 
&= - {y^{(i)}} \ln \sigma^{(i)}- (1- y^{(i)}) \ln (1-\sigma^{(i)})
\end{align}
```

where

```math
\large
\sigma^{(i)} = \sigma(z^{(i)}),\; z^{(i)} = \mathbf{w}^{\top}\mathbf{x}^{(i)}
```


"""

# ╔═╡ 4927a13b-3c3a-401a-a772-8f1602ea5129
md"""

##### As a flow/dependence graph:
"""

# ╔═╡ ebf67e2b-ac81-4232-b000-209b09e378a8
show_img("/logistic_grad_flow1.svg", w=750)

# ╔═╡ 85fe861d-f9fe-4416-aa3a-f03fb1d6b393
md"""

## Gradient derivation

#### Chain rule by graph

\


Based on chain rule, the gradient is

```math
\Large
\nabla_{\mathbf{w}}L^{(i)} = \frac{\partial L^{(i)}}{\partial \sigma^{(i)}} \frac{\partial \sigma^{(i)}}{\partial z^{(i)}}\frac{\partial z^{(i)}}{\partial \mathbf{w}}
```
"""

# ╔═╡ 4bf3d6c3-1778-400e-8133-6e51a053537e
show_img("/logistic_grad_flow2.svg", w=750)

# ╔═╡ 23e726a1-1620-464b-a6d5-3eca16179534
md"""

## Gradient derivation

#### Chain rule by graph

"""

# ╔═╡ cbe1feda-ae3f-4e4d-a43b-101cc35dfeb1
show_img("/logistic_grad_flow3_.svg", w=750)

# ╔═╡ 5c0bc1cb-d330-45dc-87f2-d33c3a2d41e7
aside(tip(md"""
Recall ``\ln(x)``'s derivative

```math
\ln'(x) = \frac{1}{x}
```
"""))

# ╔═╡ 2ad9f96c-01ee-4410-9940-1feb5df3ba0a
md"""

## Gradient derivation

#### Chain rule by graph

"""

# ╔═╡ 11bc7b24-2c5e-49b6-a153-e1d128d52b4a
show_img("/logistic_grad_flow4__.svg", w=750)

# ╔═╡ a836ddca-e9f0-430f-948b-bc4683c48245
md"""

##### Multiply the three together and tidy up


```math
\large
\begin{align}
\nabla{L}^{(i)}(\mathbf{w}) &= \left(-\frac{y^{(i)}}{\sigma^{(i)}} + \frac{1-y^{(i)}}{1-\sigma^{(i)}} \right)  \left(\sigma^{(i)} (1-\sigma^{(i)} ) \right) \mathbf{x}^{(i)}\\
&\;\;\;\;\;\;\;\;\;\; \vdots \\
&= \LARGE \boxed{-( {y^{(i)}}-\sigma^{(i)} ) \cdot \mathbf{x}^{(i)}}
\end{align}
```
"""

# ╔═╡ 824baded-9755-4ff1-8e26-bf7df96ba0e5
md"""

## Gradient 





##### The *gradient* for *logistic regression* is:


```math
\large
\nabla_{\mathbf{w}}L(\mathbf{w})  = -\sum_{i=1}^n \underbrace{(y^{(i)} - \sigma^{(i)})}_{\text{pred. error for }i} \cdot \mathbf{x}^{(i)} 
```

##### The _gradient_ for *linear regression* is


```math
\large
\nabla_{\mathbf{w}}L(\mathbf{w})  = -\sum_{i=1}^n \underbrace{(y^{(i)} - \mathbf{w}^{\top} \mathbf{x}^{(i)})}_{\text{pred. error for }i} \cdot \mathbf{x}^{(i)} 
```

##### The same idea: the gradient is proportional to the prediction quality
* prediction is perfect: gradient is zero
* otherwise, adjust accordingly
"""

# ╔═╡ 9d25c603-07b3-4758-9b28-0d2041e18569
md"""

## Learning -- gradient descent


**Learning** is to minimise the **cross-entropy** (CE) loss


```math
\large
\hat{\mathbf{w}}_{\text{MLE}} \leftarrow \arg\min_{\mathbf{w}} \underbrace{-\frac{1}{n}\ln p(\mathcal{D}|\mathbf{w}, \{\mathbf{x}^{(i)}\})}_{\text{CE loss}(\mathbf{w})}
```

> Optimisation: **gradient descent**

##

----


**Bach gradient descent algorithm:**



* random guess ``\large\mathbf{w}_0``

* while **not converge**
  * ``\large\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma \nabla L(\mathbf{w}_{t-1})``
-----

and recall the binary **C**ross **E**ntropy (CE) _loss_ is 


```math
\large
L(\mathbf{w}) =\frac{1}{n} \sum_{i=1}^n L^{(i)}(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n \underbrace{-{y^{(i)}} \ln \sigma^{(i)}- (1- y^{(i)}) \ln (1-\sigma^{(i)})}_{L^{(i)}(\mathbf{w})}
```

"""

# ╔═╡ 929916d2-9abf-40b2-9399-85c3ba05989b
md"""

## Learning -- gradient descent


**Learning** is to minimise the **cross-entropy** (CE) loss


```math
\large
\hat{\mathbf{w}}_{\text{MLE}} \leftarrow \arg\min_{\mathbf{w}} \underbrace{-\frac{1}{n}\sum_{i=1}^n {{y^{(i)}} \ln \sigma^{(i)}+ (1- y^{(i)}) \ln (1-\sigma^{(i)})}}_{\text{CE loss}(\mathbf{w})}
```

> Optimisation: **gradient descent**


##

The **gradient** for _logistic regression_ is


```math
\large
\nabla_{\mathbf{w}}L(\mathbf{w})  = -\frac{1}{n}\sum_{i=1}^n \underbrace{(y^{(i)} - \sigma^{(i)})}_{\text{pred. error for }i} \cdot \mathbf{x}^{(i)} 
```

Recall the **gradient** for _linear regression_ is


```math
\large
\nabla_{\mathbf{w}}L(\mathbf{w})  = -\frac{1}{n}\sum_{i=1}^n \underbrace{(y^{(i)} - \mathbf{w}^{\top} \mathbf{x}^{(i)})}_{\text{pred. error for }i} \cdot \mathbf{x}^{(i)} 
```

> The same idea: gradient is proportional to the prediction quality
> * prediction is perfect: gradient is near zero
> * otherwise, increase/decrease accordingly
"""

# ╔═╡ 8743242e-7f7f-4f54-b27c-1f08d1a110e2
aside(tip(md"""


Use the fact

```math
\sigma'(x) = \sigma(x) (1-\sigma(x))
```

Note that 


```math

\sigma^{(i)} = \sigma(\mathbf{w}^\top\mathbf{x}^{(i)}),
```

The gradient *w.r.t.* ``\mathbf{w}`` can be easily derived based on chain rule


Let ``z = \mathbf{w}^\top\mathbf{x}^{(i)}``, and ``\sigma^{(i)} = \sigma(z)``,

Based on the chain rule:

```math
\begin{align}
\nabla_{\mathbf{w}} \sigma^{(i)} &= \frac{\partial \sigma^{(i)}}{\partial z} \frac{\partial z}{\partial \mathbf{w}} \\
&= \sigma^{(i)} (1- \sigma^{(i)}) \cdot \mathbf{x}^{(i)}

\end{align}
```

"""))

# ╔═╡ aaaadaa8-d2e9-4647-82de-5547b2d6ddb4
md"""

## Gradient -- matrix notation

The gradient is


```math
\large
\nabla_{\mathbf{w}}L(\mathbf{w})  = - \frac{1}{n}\sum_{i=1}^n \underbrace{(y^{(i)} - \sigma^{(i)})}_{\text{pred. error for }i} \cdot \mathbf{x}^{(i)} 
```

In matrix notation, 

```math
\large
\nabla L(\mathbf{w}) =\frac{1}{n} \mathbf{X}^\top (\boldsymbol{\sigma} - \mathbf{y})
```

where 

```math
\large
\boldsymbol{\sigma} = \begin{bmatrix}
\sigma({\mathbf{w}^\top\mathbf{x}^{(1)}})\\
\sigma({\mathbf{w}^\top\mathbf{x}^{(2)}})\\
\vdots\\

\sigma({\mathbf{w}^\top\mathbf{x}^{(n)}})
\end{bmatrix} = \begin{bmatrix}
\sigma^{(1)}\\
\sigma^{(2)}\\
\vdots\\

\sigma^{(n)}
\end{bmatrix}
```
"""

# ╔═╡ 6e62b53d-68d8-45de-9a0f-8223b5d2df92
md"""

## Gradient derivation (details)


Consider one training sample ``(\mathbf{x}^{(i)}, y^{(i)})``'s loss ``L^{(i)} (\mathbf{w})`` and its gradient ``\nabla{L}^{(i)}(\mathbf{w})``, where

```math
\begin{align}
L^{(i)}(\mathbf{w}) 
&= - {y^{(i)}} \ln \sigma^{(i)}- (1- y^{(i)}) \ln (1-\sigma^{(i)})
\end{align}
```

Note that the full batch gradient is the mean of the gradients ``\nabla{L}(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n \nabla{L}^{(i)}(\mathbf{w})``.



```math
\begin{align}
\nabla{L}^{(i)}(\mathbf{w}) &= \nabla\left (- {y^{(i)}} \ln \sigma^{(i)}- (1- y^{(i)}) \ln (1-\sigma^{(i)})\right )\\
&= - {y^{(i)}}  \nabla \ln \sigma^{(i)}- (1- y^{(i)}) \nabla \ln (1-\sigma^{(i)})\\
&= - {y^{(i)}} \frac{1}{\sigma^{(i)}} \nabla \sigma^{(i)}- (1- y^{(i)})\frac{1}{1-\sigma^{(i)}} \nabla  (1-\sigma^{(i)})\\
&= - {y^{(i)}} \frac{\sigma^{(i)} (1-\sigma^{(i)})}{\sigma^{(i)}}  \nabla \mathbf{w}^\top \mathbf{x}^{(i)} - (1- y^{(i)})\frac{-\sigma^{(i)} (1-\sigma^{(i)})}{1-\sigma^{(i)}}  \nabla \mathbf{w}^\top \mathbf{x}^{(i)}\\
&= - {y^{(i)}} (1-\sigma^{(i)})\cdot\mathbf{x}^{(i)} - (1- y^{(i)})(-\sigma^{(i)}) \cdot\mathbf{x}^{(i)}\\
&= (- {y^{(i)}} +  {y^{(i)}}\sigma^{(i)} + \sigma^{(i)} -  {y^{(i)}}\sigma^{(i)} ) \cdot\mathbf{x}^{(i)}\\
&= -( {y^{(i)}}-\sigma^{(i)} ) \cdot \mathbf{x}^{(i)}
\end{align}
```
"""

# ╔═╡ 7ed0de94-b513-40c1-9f83-6ed75fcd4cdd
md"""

## Learing algorithm -- Implementation


----


**Bach gradient descent algorithm:**



* random guess ``\large\mathbf{w}_0``

* while **not converge**
  * ``\large\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma \nabla L(\mathbf{w}_{t-1})``
-----

"""

# ╔═╡ 188e9981-9597-483f-b1e3-689e26389b61
md"""


**Implementation in Python**


```python
def logistic(x):    
    return 1/ (1 + np.exp(-x))
```


```python
for i in range(0, max_iters):
	yhats = logistic(Xs@w)
    grad =  Xs.T@(yhats - ys) / np.size(ys)
    w = w - gamma * grad 
```


"""

# ╔═╡ 50d9a624-20d8-4feb-a565-4efcaf22fe27
# md"""

# ## Gradient implementation and check


# Now check the gradient as a routine task
# * checking gradient should become your habit
# """

# ╔═╡ ec78bc4f-884b-4525-8d1b-138b37274ee7
begin
	function logistic_loss(w, X, y; agg=sum)
		σ = logistic.(X * w)
		# deal with boundary cases such as σ = 0 or 1, log(0) gracefully
		# sum(y .* log.(σ) + (1 .- y).* log.(1 .- σ))
		# rather you should use xlogy and xlog1py
		(-(xlogy.(y, σ) + xlog1py.(1 .-y, -σ))) |> agg
	end
end;

# ╔═╡ e88db991-bf54-4548-80cb-7cd307300ec9
function ∇logistic_loss(w, X, y)
	σ = logistic.(X * w)
	X' * (σ - y)
end;

# ╔═╡ 99c3f5ee-63d0-4f6f-90d9-2e524a7e945a
md"""

## Demonstration -- gradient descent


"""

# ╔═╡ ff397d54-42b3-40af-bf36-4716fd9a4419
md"""

## Stochastic gradient descent



##### Batch gradient can be prehibitively *expensive* to compute

```math
\Large
\nabla L(\mathbf{w})  = \boxed{\frac{1}{n}\sum_{i=1}^n \underbrace{\colorbox{pink}{${-(y^{(i)} - \sigma^{(i)})}\cdot \mathbf{x}^{(i)}$}}_{\nabla L^{(i)}(\mathbf{w})}}_{\text{too expensive!}}
```

* ##### it requires to store all ``n`` training data ``\{\mathbf{x}^{(i)}, y^{(i)}\}`` and go through them 





"""

# ╔═╡ fcc49f3c-6219-40b8-a5d0-85fa70738a8d
md"""


## Solution: stochastic gradient descent (SGD)

"""

# ╔═╡ 6dc74312-0e3b-4ead-b369-f9c2b70ab3d3
TwoColumn(md"""

\
\
\
\
\

**Idea**: instead using all ``n``, use just **one random observation**'s gradient
\

```math
\begin{align}
\nabla L(\mathbf{w})  &= {\frac{1}{n}\sum_{i=1}^n \underbrace{\colorbox{pink}{${-(y^{(i)} - \sigma^{(i)})}\cdot \mathbf{x}^{(i)}$}}_{\nabla L^{(i)}(\mathbf{w})}}\\
&\approx \nabla L^{(i)}(\mathbf{w})
\end{align}
```

* noisy gradient instead of the mean
* might even help avoid local optimum as a by-product


""", let
	gr()
	vv = [.5, .5] *2
	plt = plot([0, vv[1]], [0, vv[2]], lc=:blue, arrow=Plots.Arrow(:closed, :head, 10, 10),  st=:path, lw=2, c=:red,  xlim=[-.2, 2], ylim=[-.2, 2], ratio=1, label="",framestyle=:none, legend=:bottomright, size=(350,450), legendfontsize=12)
	annotate!([vv[1] + .1], [vv[2] + .05], (L"\nabla L",14, :blue))
	Random.seed!(123)
	v = vv + randn(2) ./ 2.5
	annotate!([v[1] + .3], [v[2] + .1], (L"\nabla L^{(i)}", 12, :gray))
	plot!([0, v[1]], [0, v[2]], lc=:gray, arrow=Plots.Arrow(:open, :head,1,1),  st=:path, lw=.3, label="")
	for i in 1:15
		v = vv + randn(2) ./ 2.5
		plot!([0, v[1]], [0, v[2]], lc=:gray, arrow = Plots.Arrow(:open, :head,1,1),  st=:path, lw=.3, label="")
	end
	plt
end)

# ╔═╡ e0f3cee1-b6ee-4399-8e4f-c0d70b94723e
md"""

## SGD




**Stochastic gradient descent algorithm:**

-----

* random guess ``\large\mathbf{w}_0``

* for each *epoch*
  * randomly shuffle the data
  * for each ``i \in 1\ldots n``
    * ``\large\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma_t \nabla L^{(i)}(\mathbf{w}_{t-1})``
-----


The learning rate ``\gamma_t`` usually is adaptive or delaying (but small constant works reasonably well)

```math 
\large \gamma_t = \frac{1}{{iter}}\;\; \text{or}\;\; \gamma_t = \frac{1}{1\, +\, \eta\, \cdot\, iter}
```

* as the stochastic gradient can be noisy at the end


"""

# ╔═╡ aa5b5ee6-b5fe-4245-80fd-ab3ab3963d59
md"""

## Random shuffle

##### Three variants:

* ###### SGD: without shuffling


* ###### SS: single shuffle



* ###### RR: repeated random shuffle
"""

# ╔═╡ 819d01e1-4b81-4210-82de-eaf838b6a337
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/sgdcompare_logist.png
' width = '800' /></center>"

# ╔═╡ 4d4a45b5-e221-4713-b3fa-3eb36a813385
md"""

\* Koloskova *et al* (2023) *Shuffle SGD is Always Better than SGD: Improved Analysis of SGD with Arbitrary Data Orders*
"""

# ╔═╡ dc70f9a4-9b52-490a-9a94-95fe903401ce
md"""

## GD vs Mini-batch
"""

# ╔═╡ f7499d8d-e511-4540-ab68-84454b3d9cd9
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ml_stochastic.png
' width = '800' /></center>"

# ╔═╡ 0cf4d7db-5545-4b1c-ba6c-d9a6a3501e0e
md"""
[*Watt, Borhani, and Katsaggelos (2023), Machine Learning Refined
Foundations, Algorithms, and Applications](https://www.cambridge.org/highereducation/books/machine-learning-refined/0A64B2370C2F7CE3ACF535835E9D7955#overview)
"""

# ╔═╡ ea08837f-3535-4807-92e5-8091c3911948
md"""
## Mini-batch SGD


We can also use mini-batch instead of just one observation

* *e.g.* with a batch size ``5``
* a trade-off between SGD and full batch GD



**Mini-batch stochastic gradient descent**

-----

* random guess ``\large\mathbf{w}_0``

* for each *epoch*
  * split the data into equal batches ``\{B_1, B_2,\ldots, B_m\}``
  * for each batch `b` in ``\{B_1, B_2,\ldots, B_m\}``
    * ``\large\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma_t \frac{1}{|\mathtt{b}|} \sum_{i\in \mathtt{b}}\nabla L^{(i)}(\mathbf{w}_{t-1})``
-----



"""

# ╔═╡ 9522421a-1f8e-494d-a61f-f5f8dff72c83
# md"""

# ## SGD variants summary
# """

# ╔═╡ 39082958-54ed-4120-a7a1-f40f3dc9d965
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/sgd_compares.png
# ' width = '800' /></center>"

# ╔═╡ 3d0c370e-d3e5-40ec-8e66-745e4f990e18
# md"""
# [Source](avichawla.substack.com)
# """

# ╔═╡ 3583f790-bcc3-466b-95e8-e99a080e5658
begin
	Random.seed!(321)
	true_ww = [1., 1] 
	nobs = 100
	xxs = range(-2, 2, nobs)
	Xs = [ones(length(xxs)) xxs]
	Ys = rand(nobs) .< logistic.(Xs * true_ww)
end;

# ╔═╡ 06a80959-58de-4e21-bfdf-5b06caf157f1
w00 = [-10, 20];

# ╔═╡ eb5b710c-70a2-4237-8913-cd69e34b8e50
md"""

## Demonstration


A simulated logistic regression dataset
* true ``\mathbf{w} = [1, 1]^\top``
* learning rate ``\lambda = 0.25``
* ``n=100`` training observations
* ``100`` epochs (full data passes)
"""

# ╔═╡ ba0397a7-f514-42f8-b094-2ce5bd95b081
batch_loss, batch_wws=let
	# a very bad starting point: completely wrong prediction function
	ww = w00
	γ = 0.25
	iters = 100
	losses = zeros(iters+1)
	wws = Matrix(undef, length(ww), iters+1)
	losses[1] = logistic_loss(ww, Xs, Ys)
	wws[:, 1] = ww 
	nobs = length(Ys)
	for i in 1:iters
		gw = ∇logistic_loss(ww, Xs, Ys)/nobs
		# Flux.Optimise.update!(opt, ww, -gt)
		ww = ww - γ * gw
		wws[:, i+1] = ww 
		losses[i+1] = logistic_loss(ww, Xs, Ys)
	end
	losses, wws
end;

# ╔═╡ e1659838-ecf2-4205-b23c-227c586f6cc3
sgd_loss, sgd_wws=let
	# a very bad starting point: completely wrong prediction function
	ww = w00
	γ = 0.25
	iters = 100
	losses = zeros(iters+1)
	wws = Matrix(undef, length(ww), iters+1)
	losses[1] = logistic_loss(ww, Xs, Ys)
	wws[:, 1] = ww 
	for i in 1:iters
		ts = shuffle(1:length(Ys))
		for t in ts
			gw = ∇logistic_loss(ww, Xs[t, :]', Ys[t])
			ww = ww - γ * gw
		end
		wws[:, i+1] = ww 
		losses[i+1] = logistic_loss(ww, Xs, Ys)
	end
	losses, wws
end;

# ╔═╡ 1039a13b-6665-4bb1-82f8-11b740c3b5ba
let
	gr()
	plot(-12:0.1:10, -10:0.1:25, (x,y) ->logistic_loss([x,y], Xs, Ys), st=:contour, levels=10, framestyle=:none, colorbar=false, title="SGD with constant learning rate", legend=:outerbottom, size=(500, 450))
	step = 1
	# plot!(batch_wws[1,1:step:end], batch_wws[2, 1:step:end];  marker=(:diamond,3), label="Batch gradient descent", ls=:dashdot)

	plot!(sgd_wws[1,1:step:end], sgd_wws[2, 1:step:end];  c=3, marker=(:x, 5), la=0.5, ls=:dashdot, label="Stochastic gradient descent")

	# plot!(mini_b_sgd_wws[1,1:step:end], mini_b_sgd_wws[2, 1:step:end];marker=(:circle,5), ls=:dash, label="Minibatch SGD")
	plot!([w00[1]], [w00[2]], st=:scatter, markershape=:diamond, mc=:black, markersize=10, label="")
	annotate!([w00[1]], [w00[2] + .9], text("Random start", 12,  :bottom, :"Computer Modern"))
end

# ╔═╡ bf7cd9d6-c7a4-4715-a57c-d0acaef1e7d8
mini_b_sgd_loss, mini_b_sgd_wws = let
	ww = w00
	γ = 0.25
	iters = 100
	losses = zeros(iters+1)
	wws = Matrix(undef, length(ww), iters+1)
	losses[1] = logistic_loss(ww, Xs, Ys)
	wws[:, 1] = ww 
	train_loader = DataLoader((data=Xs', label=Ys), batchsize=5, shuffle=true);
	for i in 1:iters
		for (xs, ys) in train_loader
			gw = ∇logistic_loss(ww, xs', ys)/length(ys)
			ww = ww - γ * gw
		end
		wws[:, i+1] = ww 
		losses[i+1] = logistic_loss(ww, Xs, Ys)
	end
	losses, wws

end;

# ╔═╡ 2aa229e0-f578-4ff6-80a3-4897e2ad187f
let
	gr()
	plot(-12:0.1:10, -10:0.1:25, (x,y) ->logistic_loss([x,y], Xs, Ys), st=:contour, levels=10, framestyle=:none, colorbar=false, title="Batch GD vs SGD vs MiniBatch", legend=:outerbottom, size=(550, 600))


	step = 4
	plot!(batch_wws[1,1:step+2:end], batch_wws[2, 1:step+2:end];  marker=(:diamond,4, 0.5), label="Batch gradient descent", ls=:dashdot)

	plot!(sgd_wws[1,1:step:end], sgd_wws[2, 1:step:end];  marker=(:cross,4), ls=:dashdot, label="Stochastic gradient descent")

	plot!(mini_b_sgd_wws[1,1:step:end], mini_b_sgd_wws[2, 1:step:end];marker=(:circle,5), ls=:dash, label="Minibatch SGD")
	plot!([w00[1]], [w00[2]], st=:scatter, markershape=:diamond, mc=:black, markersize=10, label="")

	# plot!([true_ww[1]], [true_ww[1]], st=:scatter, markershape=:circle, mc=:black, markersize=2, label="")
	annotate!([w00[1]], [w00[2] + .9], text("random start", 12,  :bottom, :"Computer Modern"))


	annotate!([true_ww[1]], [true_ww[2] - 0.5], text("true w", 12,  :top, :"Computer Modern"))
end

# ╔═╡ 0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
md"""

## Appendix
"""

# ╔═╡ 82d513f2-be95-4c3a-8eb4-18c48a7d0b44
begin
	function ∇σ(w, x) 
		wx = w' * x
		logistic(wx) * (1-logistic(wx)) * x
	end



	function ∇h(w, x)
		x
	end
end

# ╔═╡ 862eeec0-bb67-451e-b2f9-bc633075d154
TwoColumn(let
	gr()
	w₀ = 0
	w₁, w₂ = wv_[1], wv_[2]
	plot(-5:0.5:5, -5:0.5:5, (x, y) -> (w₀+ w₁* x + w₂ * y), st=:contourf, c=:jet, colorbar=false, alpha=0.25, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="contour and gradient: "* L"h(x) =w^\top x", ratio=1, size=(350,360), titlefontsize=12)
	α = 0.5
	xs_, ys_ = meshgrid(range(-5, 5, length=10), range(-5, 5, length=10))
	∇f_d(x, y) = ∇h([1, x, y], [w₀, w₁, w₂])[2:end] * α
	quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
end, let
	gr()
	w₀ = 0
	w₁, w₂ = wv_[1], wv_[2]
	plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:contourf, c=:jet, colorbar=false, alpha=0.25, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="contour and gradient: "* L"σ(w^\top x)", ratio=1, size=(350,360), titlefontsize=12)
	α = 2
	xs_, ys_ = meshgrid(range(-5, 5, length=15), range(-5, 5, length=15))
	∇f_d(x, y) = ∇σ([1, x, y], [w₀, w₁, w₂])[2:end] * α
	quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
end
)

# ╔═╡ 4ac43cab-40fc-47b3-bbfa-13b98bcb0a47
TwoColumn(md"""
\
\
\
\

#### Where the gradient vanishes ? 

""", let
	gr()
	w₀ = 0
	w₁, w₂ = wv_[1], wv_[2]
	plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:contourf, c=:jet, colorbar=false, alpha=0.25, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="contour and gradient: "* L"σ(w^\top x)", ratio=1, size=(400,400))
	α = 2
	xs_, ys_ = meshgrid(range(-5, 5, length=15), range(-5, 5, length=15))
	∇f_d(x, y) = ∇σ([1, x, y], [w₀, w₁, w₂])[2:end] * α
	quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
end
)

# ╔═╡ 3a083374-afd6-4e64-95bd-d7e6385ab403
md"""

### Data generation
"""

# ╔═╡ 6cd96390-8565-499c-a788-fd0070331f25
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

# ╔═╡ 67dae6d0-aa32-4b46-9046-aa24295aa117
plt_binary_2d = let
	gr()
	plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], st=:scatter, label=L"y^{(i)} = 1", xlabel=L"x_1", ylabel=L"x_2", title="Binary classification example", c=2, size=(400,300))
	plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], st=:scatter, c=1, framestyle=:origin, label=L"y^{(i)} = 0", xlim=[-8, 8], ylim=[-6, 6])
end;

# ╔═╡ e77f6f3b-f18d-4260-ab7a-db61e7b4131d
TwoColumn(md"""
\
\
\

**input** features: ``\mathbf{x}^{(i)} \in \mathbb{R}^m``
* *e.g.* the right is a two-dimensional, *i.e.* ``m=2``


**output** label: ``y^{(i)} \in \{0,1\}``


""", plt_binary_2d)

# ╔═╡ b640fc38-2ada-4c35-82f0-53fd801d14e1
TwoColumn(plot(plt_binary_2d, title="2-d view", size=(300,300)), let
	gr()
	plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)), st=:scatter, ms=2, label=L"\tilde{y}^{(i)} = 1", zlim=[-.1, 1.1], xlabel = L"x_1", ylabel=L"x_2", c=2, size=(400,300))
	plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], 0* ones(sum(targets_D₂ .== 0)), st=:scatter,  ms=2,framestyle=:origin, label=L"\tilde{y}^{(i)} = 0", xlim=[-8, 8], ylim=[-6, 6], title="3-d view", c=1, camera=(45,30))
end)

# ╔═╡ 867507ad-1ca8-4d12-a9c4-646197937547
begin
	plotly()
	scatter(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)), zlim=[-0.9, 1.9], label="y=1", c=2)
	scatter!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], 0 * ones(sum(targets_D₂ .== 0)), label="y=0", c=1, framestyle=:zerolines)
	w_d₂ = linear_reg(D₂, targets_D₂ ; λ=0.0)
	plot!(-5:1:5, -5:1:5, (x,y) -> w_d₂[1] + w_d₂[2] * x + w_d₂[3] * y, alpha =0.9, st=:surface, colorbar=false, c=:jet, title="First attempt: least square fit")
end

# ╔═╡ 7b4502d1-3847-4472-b706-65cb68413927
plot_ls_class=let
	gr()
	plt = plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], st=:scatter, label="class 1", ratio=1, c=2, legend=:topleft)
	plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], st=:scatter, framestyle=:origin, label="class 2", xlim=[-8, 8], ylim=[-6, 6], c=1)
	plot!(-6:1:6, (x) -> - w_d₂[1]/w_d₂[3] - w_d₂[2]/w_d₂[3] * x, lw=4, lc=:gray, label="Decision bounary: "*L"h(\mathbf{x}) =0", title="Least square classifier")

	plot!([0, w_d₂[2]*3], [0, w_d₂[3]*3], line = (:arrow, 3), c=2, label="")

	minx, maxx = minimum(D₂[:,2])-2, maximum(D₂[:,2])+2
	miny, maxy = minimum(D₂[:,3])-2, maximum(D₂[:,3])+2
	xs, ys = meshgrid(minx:0.2:maxx, miny:0.2:maxy)
	colors = [dot(w_d₂ , [1, x, y]) > 0.0   for (x, y) in zip(xs, ys)] .+ 1
	scatter!(xs, ys, color = colors,
            markersize = 1.5, label = nothing, markerstrokewidth = 0, markeralpha=0.5, xlim= [minx, maxx], ylim =[miny, maxy], framestyle=:origin)
	plt
end;

# ╔═╡ cbbcf999-1d31-4f53-850e-d37a28cff849
begin
	plotly()
	scatter(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)),  label="ỹ=1", c=2)
	scatter!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], 0 *ones(sum(targets_D₂ .== 0)), label="ỹ=0", framestyle=:zerolines, c=1)
	# w = linear_reg(D₂, targets_D₂;λ=0.0)
	plot!(-10:1:10, -50:1:50, (x,y) -> w_d₂[1] + w_d₂[2] * x + w_d₂[3] * y, alpha =0.8, st=:surface, colorbar=false,c=:jet, title="First attempt: unbounded prediction")
end

# ╔═╡ f298ffd1-b1dd-472f-b761-c5e07a9121c0
let
	plotly()
	scatter(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)),  label="y=1", c=2)
	scatter!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], 0 *ones(sum(targets_D₂ .== 0)), label="y=0", framestyle=:zerolines, c=1)
	# w = linear_reg(D₂, targets_D₂;λ=0.0)
	plot!(-10:.1:10, -10:0.1:10, (x,y) -> logistic(0 + 2.5 * x + 2.5 * y), alpha =0.8, st=:surface, colorbar=false,c=:jet, title="First attempt: unbounded prediction")
end

# ╔═╡ 79a75f04-3aa9-4a8b-9f88-1fe8c123680b
let
	gr()
	# gr()
	bias = 0.0;

	p0 = plot(-8:0.1:10, -10:0.1:10, (w1, w2) -> zeroone_loss([bias, w1, w2], D₂, targets_D₂; agg=sum), st=:surface, c=:jet,camera = (45, 35), colorbar=false, title="1/0 error loss")

	p1 = plot(-8:0.1:10, -10:0.1:10, (w1, w2) -> logistic_mse_loss([bias, w1, w2], D₂, targets_D₂; agg=sum), st=:surface, c=:jet,camera = (45, 35), colorbar=false, title="Squared error loss")


	plot(p0, p1, layout=(1,2))

end

# ╔═╡ ae5324a5-0cf6-4485-aead-86c3d8f3041a
let
	gr()
	bias = 0.0;

	p0 = plot(-8:0.1:10, -5:0.1:10, (w1, w2) -> zeroone_loss([bias, w1, w2], D₂, targets_D₂; agg=sum), st=:contourf, c=:jet, colorbar=false, title="1/0 error loss", ratio=1, xlabel=L"w_1", ylabel=L"w_2")

	p1 = plot(-8:0.1:10, -5:0.1:10, (w1, w2) -> logistic_mse_loss([bias, w1, w2], D₂, targets_D₂; agg=sum), st=:contourf, c=:jet, colorbar=false, title="Squared error loss",  ratio=1,  xlabel=L"w_1", ylabel=L"w_2")


	
	# p2 = plot(-8:0.1:10, -5:0.1:10, (w1, w2) -> logistic_loss([bias, w1, w2], D₂, targets_D₂; agg=sum), st=:contourf, c=:jet, colorbar=false, title="Cross entropy loss",  ratio=1)

	plot(p0, p1, layout=(1,2), size=(750,360), ylim =(-5,10))

end

# ╔═╡ a0fa5474-3cb7-4829-9e2e-2bf53ca00d94
logistic_mse_loss_grad(x; bias= 0.0) = Zygote.withgradient((w) -> logistic_mse_loss([bias, w...], D₂, targets_D₂), x);

# ╔═╡ 7d830f79-e361-4de1-b3fd-dee08fd1cc06
begin

	function produce_anim(w0; bias= 0, X= D₂, y =targets_D₂,  grad= logistic_mse_loss_grad, lossf = logistic_mse_loss, γ = 0.5, maxiters = 1000, xrange =-8:0.5:10, yrange=-5:0.5:10 )
		gr()
		ws = [w0]
		loss = []
		for i in 1:maxiters
			li, gradw = grad(w0)
			w0 -= γ * gradw[1]
			push!(ws, w0)
			push!(loss, li)
		end
		ws = hcat(ws...)
		traces = ws
		plt = plot(xrange, yrange, (w1, w2) -> lossf([bias, w1, w2], X, y; agg=sum), st=:contourf, c=:jet, colorbar=false,  ratio=1, alpha=0.8, xlim =(xrange |> extrema), ylim = (yrange |> extrema))
		wt = traces[:, 1]
		scatter!([wt[1]], [wt[2]], markershape=:xcross, label="start", markerstrokewidth=4, mc=3, markersize=6)
		anim = @animate for t in 2:20:maxiters
			plot!([traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
			plot!([wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="", title="Iteration $(t); loss = $(round(loss[t];digits=2))")
			wt = traces[1:2, t]
		end 
		return anim
	end
	# gif(anim, fps=5)
end

# ╔═╡ dfe4234a-d2a3-4b59-9578-64bd6e8a5c33
gif(produce_anim([-4.5, -4.5]); fps=10)

# ╔═╡ a24b03a3-b2a0-4b12-b2a8-25c8cf7f843c
TwoColumn(gif(produce_anim([0, -4.5]); fps=10), gif(produce_anim([-4.5, 0]); fps=10))

# ╔═╡ fed8b7b2-34c4-445c-bf11-21b5161c8d28
let
	gr()
	bias = 0.0;

	p0 = plot(-8:0.1:10, -10:0.1:10, (w1, w2) -> zeroone_loss([bias, w1, w2], D₂, targets_D₂; agg=mean), st=:surface, c=:jet,camera = (75, 35), colorbar=false, title="1/0 error loss")

	p1 = plot(-8:0.1:10, -10:0.1:10, (w1, w2) -> logistic_mse_loss([bias, w1, w2], D₂, targets_D₂; agg=mean), st=:surface, c=:jet,camera = (75, 35), colorbar=false, title="Squared error loss")


	
	p2 = plot(-8:0.1:10, -10:0.1:10, (w1, w2) -> logistic_loss([bias, w1, w2], D₂, targets_D₂; agg=mean), st=:surface, c=:jet, camera = (75, 35),colorbar=false, title="Cross entropy loss")

	plot(p0, p1, p2, layout=(1,3), titlefontsize=10,zticks=false)

end

# ╔═╡ cb9cc32f-a12b-43ec-83c8-3ed26a96992f
let
	gr()
	bias = 0.0;

	p0 = plot(-8:0.1:10, -5:0.1:10, (w1, w2) -> zeroone_loss([bias, w1, w2], D₂, targets_D₂; agg=sum), st=:contourf, c=:jet, colorbar=false, title="1/0 error loss", ratio=1)

	p1 = plot(-8:0.1:10, -5:0.1:10, (w1, w2) -> logistic_mse_loss([bias, w1, w2], D₂, targets_D₂; agg=sum), st=:contourf, c=:jet, colorbar=false, title="Squared error loss",  ratio=1)


	
	p2 = plot(-8:0.1:10, -5:0.1:10, (w1, w2) -> logistic_loss([bias, w1, w2], D₂, targets_D₂; agg=sum), st=:contourf, c=:jet, colorbar=false, title="Cross entropy loss",  ratio=1)

	plot(p0, p1, p2, layout=(1,3), size=(750,270), ylim =(-5,10))

end

# ╔═╡ 8fe5631a-1f10-4af4-990d-5a23c96fb73b
begin
	D1 = [
	    7 4;
	    5 6;
	    8 6;
	    9.5 5;
	    9 7
	]

	# D1 = randn(5, 2) .+ 2
	
	D2 = [
	    2 3;
	    3 2;
	    3 6;
	    5.5 4.5;
	    5 3;
	]

	# D2 = randn(5,2) .- 2

	D = [D1; D2]
	D = [ones(10) D]
	targets = [ones(5); zeros(5)]
	AB = [1.5 8; 10 1.9]
end;

# ╔═╡ 619df17d-9f75-463f-afe5-6cbffb0762d5
begin
	n3_ = 30
	extraD = randn(n3_, 2)/2 .+ [2 -6]
	D₃ = [copy(D₂); [ones(n3_) extraD]]
	targets_D₃ = [targets_D₂; zeros(n3_)]
	targets_D₃_ = [targets_D₂; -ones(n3_)]
end

# ╔═╡ a270b1b3-7bb4-4612-9e57-3ea9b7daedc0
losses, wws=let
	# a very bad starting point: completely wrong prediction function
	ww = [0, -5, -5]
	γ = 0.008
	iters = 2000
	losses = zeros(iters+1)
	wws = Matrix(undef, 3, iters+1)
	losses[1] = logistic_loss(ww, D₃, targets_D₃)
	wws[:, 1] = ww 
	for i in 1:iters
		gw = ∇logistic_loss(ww, D₃, targets_D₃)
		# Flux.Optimise.update!(opt, ww, -gt)
		ww = ww - γ * gw
		wws[:, i+1] = ww 
		losses[i+1] = logistic_loss(ww, D₃, targets_D₃)
	end
	losses, wws
end;

# ╔═╡ 872aa766-2ed8-4cb3-a029-4a2bb42c90a8
let
	gr()
	plot(losses[1:50], label="loss", xlabel="Epoch",ylabel="loss", title="Batch gradient descent's loss vs epoch")
end

# ╔═╡ daf19517-4123-49a8-affb-5d869e08480a
anim_logis=let
	gr()
	# xs_, ys_ = meshgrid(range(-5, 5, length=20), range(-5, 5, length=20))

	anim = @animate for t in [1:25;26:10:200;]
		plot(D₃[targets_D₃ .== 1, 2], D₃[targets_D₃ .== 1, 3], st=:scatter, label="class 1", c=2)
		plot!(D₃[targets_D₃ .== 0, 2], D₃[targets_D₃ .== 0, 3], st=:scatter, framestyle=:origin, label="class 2", xlim=[-8, 8], legend=:topleft, c=1)
		w₀, w₁, w₂ = (wws[:, t])
		plot!(-5:0.1:5, -5:0.1:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:contour, levels=6, c=:jet, colorbar=false, alpha=0.5, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="Epoch: "*L"%$(t);"*" loss: " *L"%$(round(losses[t]; digits=1))", ratio=1)
		quiver!([0.0], [0.0], quiver=([w₁- 0.0], [w₂-0.0]), c=2, lw=2)
	end

	anim
	# ∇f_d(x, y) = ∇σ([1, x, y], [w₀, w₁, w₂])[2:end] * 1
	# quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
	# w_d₃ = linear_reg(D₃, targets_D₃_;λ=0.0)
	# w_d₂
	# plot!(-6:1:6, (x) -> - w_d₃[1]/w_d₃[3] - w_d₃[2]/w_d₃[3] * x, lw=4, lc=:gray, label="Decision boundary: "*L"h(\mathbf{x}) =0", title="Least square classifier fails")
end;

# ╔═╡ 03840aa7-4258-47f7-b802-92ab93c92911
gif(anim_logis, fps=5)

# ╔═╡ 8398cb20-b9f4-427c-8b10-2a572f8a5632
anim_logis2=let

	gr()

	anim = @animate for t in [1:25...]
		plot(D₃[targets_D₃ .== 1, 2], D₃[targets_D₃ .== 1, 3], ones(sum(targets_D₃ .== 1)), st=:scatter, label="class 1", c=2)
		plot!(D₃[targets_D₃ .== 0, 2], D₃[targets_D₃ .== 0, 3], zeros(sum(targets_D₃ .== 0)), st=:scatter, framestyle=:origin, label="class 2", xlim=[-8, 8], legend=:topleft, c=1)
		w₀, w₁, w₂ = (wws[:, t])
		plot!(-5:0.1:5, -5:0.1:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.5, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="Epoch: "*L"%$(t);"*" loss: " *L"%$(round(losses[t]; digits=1))", ratio=1)
	end

	anim
	# ∇f_d(x, y) = ∇σ([1, x, y], [w₀, w₁, w₂])[2:end] * 1
	# quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
	# w_d₃ = linear_reg(D₃, targets_D₃_;λ=0.0)
	# w_d₂
	# plot!(-6:1:6, (x) -> - w_d₃[1]/w_d₃[3] - w_d₃[2]/w_d₃[3] * x, lw=4, lc=:gray, label="Decision boundary: "*L"h(\mathbf{x}) =0", title="Least square classifier fails")
end;

# ╔═╡ 12991c72-6b19-47ed-9ebe-90e3d615f985
gif(anim_logis2, fps=2)

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

# ╔═╡ 8076e8af-35b9-4113-ba89-c3e5bb25812f
let
	plotly()
	w₀ = 0
	w₁, w₂ = wv_[1], wv_[2]
	p1 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> w₀+ w₁* x + w₂ * y, st=:surface, c=:jet, colorbar=false, alpha=0.8, framestyle=:zerolines, xlabel="x₁", ylabel="x₂", title="h(x) = wᵀ x")

	plot!(-5:0.5:5, -5:0.5:5, (x, y) -> 0, st=:surface, c=:gray, alpha=0.5)
	arrow3d!([0], [0], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
	x0s = -5:0.5:5
	if w₂ ==0
		x0s = range(-w₀/w₁-eps(1.0) , -w₀/w₁+eps(1.0), 20)
		y0s = range(-5, 5, 20)
	else
		y0s = (- w₁ * x0s .- w₀) ./ w₂
	end
	plot!(x0s, y0s, zeros(length(x0s)), lc=:gray, lw=4, label="")
	
	p2 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.8, zlim=[-0.1, 1.1],  xlabel="x₁", ylabel="x₂", title="σ(wᵀx)", framestyle=:zerolines)
	plot!(-5:0.5:5, -5:0.5:5, (x, y) -> 0.5, st=:surface, c=:gray, alpha=0.75)
	arrow3d!([0], [0], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
	plot!(x0s, y0s, .5 * ones(length(x0s)), lc=:gray, lw=4, label="")
	plot(p1, p2)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Implicit3DPlotting = "d997a800-832a-4a4c-b340-7dddf3c1ad50"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
Distributions = "~0.25.107"
HypertextLiteral = "~0.9.5"
Implicit3DPlotting = "~0.3.0"
LaTeXStrings = "~1.3.1"
Latexify = "~0.16.1"
LogExpFunctions = "~0.3.26"
MLUtils = "~0.4.4"
Plots = "~1.40.0"
PlutoTeachingTools = "~0.2.14"
PlutoUI = "~0.7.55"
StatsBase = "~0.34.2"
StatsPlots = "~0.15.6"
Zygote = "~0.6.69"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "97e7205a0242463482c333a6110fd49a0a1965c4"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractLattices]]
git-tree-sha1 = "222ee9e50b98f51b5d78feb93dd928880df35f06"
uuid = "398f06c4-4d28-53ec-89ca-5b2656b7603d"
version = "0.3.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "c278dfab760520b8bb7e9511b968bf4ba38b7acc"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.3"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cde29ddf7e5726c9fb511f340244ea3481267608"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

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
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "bbec08a37f8722786d87bedf84eae19c020c4efa"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.7.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "588e0d680ad1d7201d4c6a804dcb1cd9cba79fbb"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.0.3"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "7aa7ad1682f3d5754e3491bb59b8103cae28e3a3"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.40"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1f03a9fa24271160ed7e73051fba3c1a759b53f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.4.0"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bonito]]
deps = ["Base64", "CodecZlib", "Colors", "Dates", "Deno_jll", "HTTP", "Hyperscript", "LinearAlgebra", "Markdown", "MsgPack", "Observables", "RelocatableFolders", "SHA", "Sockets", "Tables", "ThreadPools", "URIs", "UUIDs", "WidgetsBase"]
git-tree-sha1 = "eb7ffbde27b8f321aaac74891dcddcd5676eeae9"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f8"
version = "3.0.4"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "32abd86e3c2025db5172aa182b982debed519834"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.1"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "213f001d1233fd3b8ef007f50c8cab29061917d8"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.61.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1287e3872d646eed95198457873249bd9f0caed2"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.20.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "9ebb045901e9bbf58767a9f34ff89831ed711aae"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.7"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "c0216e792f518b39b22212127d4a84dc31e4e386"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.5"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "9b1ca1aa6ce3f71b3d1840c538a8210a043625eb"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

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
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "75bd5b6fc5089df449b5d35fa501c846c9b6549b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.12.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

    [deps.CompositionsBase.weakdeps]
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "8cfa272e8bdedfa88b6aefbbca7c19f1befac519"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "ac67408d9ddf207de5cfa9a97e114352430f01ed"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.16"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelaunayTriangulation]]
deps = ["DataStructures", "EnumX", "ExactPredicates", "Random", "SimpleGraphs"]
git-tree-sha1 = "26eb8e2331b55735c3d305d949aabd7363f07ba7"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "0.8.11"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Deno_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cd6756e833c377e0ce9cd63fb97689a255f12323"
uuid = "04572ae6-984a-583e-9378-9577a1c2574d"
version = "1.33.4+0"

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
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7c302d7a5fec5214eb8a5a4c466dcf7a51fcf169"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.107"

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

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "e8b8c949551f417e040f16e5c431b6e83e306e54"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.7"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.Extents]]
git-tree-sha1 = "2140cd04483da90b2da7f99b2add0750504fc39c"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.2"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "c5c28c245101bd59154f649e19b038d15901b5dc"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.2"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "73d1214fec245096717847c62d389a5d2ac86504"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.22.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "055626e1a35f6771fe99060e835b72ca61a52621"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW]]
deps = ["GLFW_jll"]
git-tree-sha1 = "35dbc482f0967d8dceaa7ce007d16f9064072166"
uuid = "f7f18e0c-5ee9-5ccd-a5bf-e8befd85ed98"
version = "3.4.1"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

[[deps.GLMakie]]
deps = ["ColorTypes", "Colors", "FileIO", "FixedPointNumbers", "FreeTypeAbstraction", "GLFW", "GeometryBasics", "LinearAlgebra", "Makie", "Markdown", "MeshIO", "ModernGL", "Observables", "PrecompileTools", "Printf", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "501d116cbd5af531bdca20a6f4fb705ffd61784f"
uuid = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
version = "0.9.6"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "85d7fb51afb3def5dcb85ad31c3707795c8bccc1"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "9.1.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "3458564589be207fa6a77dbbf8b97674c9836aab"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "77f81da2964cc9fa7c0127f941e8bce37f7f1d70"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.2+0"

[[deps.GenericLinearAlgebra]]
deps = ["LinearAlgebra", "Printf", "Random", "libblastrampoline_jll"]
git-tree-sha1 = "02be7066f936af6b04669f7c370a31af9036c440"
uuid = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
version = "0.3.11"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "d4f85701f569584f2cff7ba67a137d03f0cfb7d0"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.3"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "5694b56ccf9d15addedc35e9a4ba9c317721b788"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.10"

[[deps.GeometryTypes]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "d796f7be0383b5416cd403420ce0af083b0f9b28"
uuid = "4d00f742-c7ba-57c2-abde-4428a4b178cb"
version = "0.8.5"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "af13a277efd8a6e716d79ef635d5342ccb75be61"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.10.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "abbbb9ec3afd783a7cbd82ef01dcd088ea051398"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.1"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

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
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "5d8c5713f38f7bc029e26627b687710ba406d0dd"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.12"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "b2a7eaa169c13f5bcae8131a83bc30eff8f71be0"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.2"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.Implicit3DPlotting]]
deps = ["GLMakie", "Meshing", "Polyhedra", "WGLMakie"]
git-tree-sha1 = "370e3b1c92cd5decb01cb0ec6ff78ab7b6b53f66"
uuid = "d997a800-832a-4a4c-b340-7dddf3c1ad50"
version = "0.3.0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5fdf2fe6724d8caabf43b557b84ce53f3b7e2f6b"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.0.2+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "RoundingEmulator"]
git-tree-sha1 = "c274ec586ea58eb7b42afd0c5d67e50ff50229b5"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.5"
weakdeps = ["DiffRules", "RecipesBase"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

[[deps.IntervalSets]]
deps = ["Dates", "Random"]
git-tree-sha1 = "3d8866c029dd6b16e69e0d4a939c4dfcb98fac47"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.8"
weakdeps = ["Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60b1194df0a3298f460063de985eae7b01bc011a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.1+0"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "5036b4cf6d85b08d80de09ef65b4d951f6e68659"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.19.0"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "04663b9e1eb0d0eabf76a6d0752e0dac83d53b36"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.28"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "Requires", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "4e0cb2f5aad44dcfdc91088e85dee4ecb22c791c"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.16"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "fee018a29b60733876eb557804b5b109dd3dd8a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.8"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "cb4619f7353fc62a1a22ffa3d7ed9791cfb47ad8"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.4.2"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "98eaee04d96d973e79c25d49167668c5c8fb50e2"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.27+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "3a994404d3f6709610701c7dabfc03fed87a81f8"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.1"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearAlgebraX]]
deps = ["LinearAlgebra", "Mods", "Primes", "SimplePolynomials"]
git-tree-sha1 = "d76cec8007ec123c2b681269d40f94b053473fcf"
uuid = "9b3f67b0-2d00-526e-9884-9e4938f8fb88"
version = "0.2.7"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "20ce1091ba18bcdae71ad9b71ee2367796ba6c48"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.4.4"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "72dc3cf284559eb8f53aa593fe62cb33f83ed0c0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.0.0+0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "b45738c2e3d0d402dffa32b2c1654759a2ac35a4"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "StableHashTraits", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun"]
git-tree-sha1 = "5458e49833d178c071dd6ab6938ac5320f37bc98"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.20.5"

[[deps.MakieCore]]
deps = ["Observables", "REPL"]
git-tree-sha1 = "248b7a4be0f92b497f7a331aed02c1e9a878f46b"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.7.3"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "8b40681684df46785a0012d352982e22ac3be59e"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.25.2"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "96ca8a313eb6437db5ffe946c457a401bbb8ce1d"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.5.7"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MeshIO]]
deps = ["ColorTypes", "FileIO", "GeometryBasics", "Printf"]
git-tree-sha1 = "8be09d84a2d597c7c0c34d7d604c039c9763e48c"
uuid = "7269a6da-0436-5bbc-96c2-40638cbb6118"
version = "0.4.10"

[[deps.Meshing]]
deps = ["GeometryBasics", "GeometryTypes", "StaticArrays"]
git-tree-sha1 = "a8a908f8512d3c05ba20e4ecf0699e6114150d7c"
uuid = "e6723b4c-ebff-59f1-b4b7-d97aa5274f73"
version = "0.6.0"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.ModernGL]]
deps = ["Libdl"]
git-tree-sha1 = "b76ea40b5c0f45790ae09492712dd326208c28b2"
uuid = "66fc600b-dfda-50eb-8b99-91cfa97b1301"
version = "1.1.7"

[[deps.Mods]]
git-tree-sha1 = "924f962b524a71eef7a21dae1e6853817f9b658f"
uuid = "7475f97c-0381-53b1-977b-4c60186c8d62"
version = "2.2.4"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.Multisets]]
git-tree-sha1 = "8d852646862c96e226367ad10c8af56099b4047e"
uuid = "3b2b4ff1-bcff-5658-a3ee-dbcf1ce5ac09"
version = "0.4.4"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "806eea990fb41f9b36f1253e5697aa645bf6a9f8"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.4.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "d2811b435d2f571bdfdfa644bb806a66b458e186"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.11"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ded64ff6d4fdd1cb68dfcbb818c69e144a5b2e4c"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.16"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "6a731f2b5c03157418a20c12195eb4b74c8f8621"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.13.0"
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
version = "0.3.23+2"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60e3045590bd104a16fefb12836c00c0ef8c7f8c"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "MathOptInterface", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "47fea72de134f75b105a5d4a1abe5c6aec89d390"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.1"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "ec3edfe723df33528e085e632414499f26650501"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Permutations]]
deps = ["Combinatorics", "LinearAlgebra", "Random"]
git-tree-sha1 = "eb3f9df2457819bf0a9019bd93cc451697a0751e"
uuid = "2ae35dd2-176d-5d53-8349-f30d82d94d4f"
version = "0.4.20"

[[deps.PikaParser]]
deps = ["DocStringExtensions"]
git-tree-sha1 = "d6ff87de27ff3082131f31a714d25ab6d0a88abf"
uuid = "3bbf5609-3e7b-44cd-8549-7c69f321e792"
version = "0.6.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "862942baf5663da528f66d24996eb6da85218e76"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "38a748946dca52a622e79eea6ed35c6737499109"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.0"

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
git-tree-sha1 = "89f57f710cc121a7f32473791af3d6beefc59051"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.14"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "68723afdb616445c6caaef6255067a8339f91325"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.55"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Polyhedra]]
deps = ["GenericLinearAlgebra", "GeometryBasics", "JuMP", "LinearAlgebra", "MutableArithmetics", "RecipesBase", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "e7b1e266cc9f3cb046d6c8d2c3aefc418a53428d"
uuid = "67491407-f73d-577b-9b50-8179a7c68029"
version = "0.7.6"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase", "Setfield", "SparseArrays"]
git-tree-sha1 = "a9c7a523d5ed375be3983db190f6a5874ae9286d"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.6"
weakdeps = ["ChainRulesCore", "FFTW", "MakieCore", "MutableArithmetics"]

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "1d05623b5952aed1307bf8b43bec8b8d1ef94b6e"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.5"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "00099623ffee15972c16111bcf84c58a0051257c"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.9.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

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
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "3fe4e5b9cdbb9bbc851c57b149e516acc07f8f72"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.13"

[[deps.RingLists]]
deps = ["Random"]
git-tree-sha1 = "f39da63aa6d2d88e0c1bd20ed6a3ff9ea7171ada"
uuid = "286e9d63-9694-5540-9e3c-4e6708fa07b2"
version = "0.2.8"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "79123bc60c5507f035e6d1d9e563bb2971954ec8"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleGraphs]]
deps = ["AbstractLattices", "Combinatorics", "DataStructures", "IterTools", "LightXML", "LinearAlgebra", "LinearAlgebraX", "Optim", "Primes", "Random", "RingLists", "SimplePartitions", "SimplePolynomials", "SimpleRandom", "SparseArrays", "Statistics"]
git-tree-sha1 = "f65caa24a622f985cc341de81d3f9744435d0d0f"
uuid = "55797a34-41de-5266-9ec1-32ac4eb504d3"
version = "0.8.6"

[[deps.SimplePartitions]]
deps = ["AbstractLattices", "DataStructures", "Permutations"]
git-tree-sha1 = "e9330391d04241eafdc358713b48396619c83bcb"
uuid = "ec83eff0-a5b5-5643-ae32-5cbf6eedec9d"
version = "0.3.1"

[[deps.SimplePolynomials]]
deps = ["Mods", "Multisets", "Polynomials", "Primes"]
git-tree-sha1 = "7063828369cafa93f3187b3d0159f05582011405"
uuid = "cc47b68c-3164-5771-a705-2bc0097375a0"
version = "0.2.17"

[[deps.SimpleRandom]]
deps = ["Distributions", "LinearAlgebra", "Random"]
git-tree-sha1 = "3a6fb395e37afab81aeea85bae48a4db5cd7244a"
uuid = "a6525b86-64cd-54fa-8f65-62fc48bdc0e8"
version = "0.3.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableHashTraits]]
deps = ["Compat", "PikaParser", "SHA", "Tables", "TupleTools"]
git-tree-sha1 = "662f56ffe22b3985f3be7474f0aecbaf214ecf0f"
uuid = "c5dd0088-6c3f-4803-b00e-f31a60c170fa"
version = "1.1.6"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "7b0e9c14c624e435076d19aea1e5cbdec2b9ca37"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.2"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "9115a29e6c2cf66cf213ccc17ffd61e27e743b24"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.6"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "1b0b1205a56dc288b71b1961d48e351520702e24"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.17"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

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

[[deps.ThreadPools]]
deps = ["Printf", "RecipesBase", "Statistics"]
git-tree-sha1 = "50cb5f85d5646bc1422aa0238aa5bfca99ca9ae7"
uuid = "b189fb0b-2eb5-4ed4-bc0c-d34c51242431"
version = "2.1.1"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "34cc045dd0aaa59b8bbe86c644679bc57f1d5bd0"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.8"

[[deps.TranscodingStreams]]
git-tree-sha1 = "54194d92959d8ebaa8e26227dbe3cdefcdcd594f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.3"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "3064e780dbb8a9296ebb3af8f440f787bb5332af"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.80"

    [deps.Transducers.extensions]
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.TupleTools]]
git-tree-sha1 = "155515ed4c4236db30049ac1495e2969cc06be9d"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.4.3"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "3c793be6df9dd77a0cf49d80984ef9ff996948fa"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.19.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "323e3d0acf5e78a56dfae7bd8928c989b4f3083e"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.WGLMakie]]
deps = ["Bonito", "Colors", "FileIO", "FreeTypeAbstraction", "GeometryBasics", "Hyperscript", "LinearAlgebra", "Makie", "Observables", "PNGFiles", "PrecompileTools", "RelocatableFolders", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "e3a8288662383160da449eccf78fd377632e277c"
uuid = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"
version = "0.9.5"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WidgetsBase]]
deps = ["Observables"]
git-tree-sha1 = "30a1d631eb06e8c868c559599f915a62d55c2601"
uuid = "eead4739-05f7-45a1-878c-cee36b57321c"
version = "0.1.4"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "5f24e158cf4cee437052371455fe361f526da062"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.6"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "801cbe47eae69adc50f36c3caec4758d2650741b"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.2+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522b8414d40c4cbbab8dee346ac3a09f9768f25d"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.5+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

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
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "4ddb4470e47b0094c93055a3bcae799165cc68f1"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.69"

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
git-tree-sha1 = "27798139afc0a2afa7b1824c206d5e87ea587a00"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.5"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "93284c28274d9e75218a416c65ec49d0e0fcdf3d"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.40+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

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
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─9f90a18b-114f-4039-9aaf-f52c77205a49
# ╟─86ac8000-a595-4162-863d-8720ff9dd3bd
# ╟─ece21354-0718-4afb-a905-c7899f41883b
# ╟─99b2a872-7d59-4a7e-ab7e-79e825563f0c
# ╟─646cafb0-e9c3-4f85-8b38-ae75fafb7c61
# ╟─3e2e1ea8-3a7d-462f-ac38-43a087907a14
# ╟─7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
# ╟─bc96a33d-9011-41ec-a19e-d472cbaafb70
# ╟─1d1c759c-9b59-4f6e-a61f-02bd5e279f68
# ╟─bcc3cea5-0564-481b-883a-a45a1b870ba3
# ╟─e77f6f3b-f18d-4260-ab7a-db61e7b4131d
# ╟─67dae6d0-aa32-4b46-9046-aa24295aa117
# ╟─2d70884d-c922-45e6-853e-5608affdd860
# ╟─b640fc38-2ada-4c35-82f0-53fd801d14e1
# ╟─867507ad-1ca8-4d12-a9c4-646197937547
# ╟─8765a8e8-6f73-4ad4-9604-6a9eb5991241
# ╟─8c69c448-c104-4cae-9352-10d4cec512a9
# ╟─68a0247b-9346-42fc-b2d2-4373e70a1789
# ╟─bf83caa9-d9f9-48a2-9563-152fcbb8afdc
# ╟─c78fbd4c-1e5f-4307-9ab3-9303b0744de0
# ╟─7b4502d1-3847-4472-b706-65cb68413927
# ╟─ea55660a-090b-453c-a141-b6867670ba48
# ╟─ffb7f8f1-739c-43a5-b13e-4045b50c9e05
# ╟─3315ef07-17df-4fe1-a7a5-3e8f039c0cc1
# ╟─c936fe77-827b-4245-93e5-d239d467bffd
# ╟─86420ad6-3995-424c-9bdd-21214bf544df
# ╟─cdccf955-19f3-45bb-8dfe-2e15addcdd32
# ╟─4d474d9f-50dc-4c9f-9505-65e024c83f38
# ╟─331571e5-f314-42db-ae56-66e64e485b85
# ╟─cbbcf999-1d31-4f53-850e-d37a28cff849
# ╟─c8e55a60-0829-4cc7-bc9b-065809ac791c
# ╟─5ed6a7a3-7871-467f-9ff0-779e2c403bac
# ╟─7ad891af-18d2-48d9-a9a6-f322fdd30bed
# ╟─15c50511-3c4a-46b4-8712-fecbdd181bfd
# ╟─ffb51b97-d873-4142-bc99-d48c9a8c390a
# ╟─53e69939-09b2-41b5-a5a1-21b075275f8c
# ╟─5d15c0a7-1038-4d44-8dc4-986809b5d45e
# ╟─e11f7ee9-e50a-4c7c-a5aa-acc6f712a063
# ╟─5997601f-5ee4-4ade-91f6-3bbdaa46f17c
# ╟─4cac5129-3b3d-42c4-839c-254ac26199f1
# ╠═137441e2-2f6c-43bb-b1a2-b240ee3ab6e7
# ╟─246b8623-2f47-4ac1-b562-75d50889e6d8
# ╠═206b39d0-8a4b-43ed-9e56-d09e4da59883
# ╠═3deafee4-92a0-4879-b618-20d58ec731d1
# ╟─6ba6bed9-d913-4511-9829-787fe8a09fa7
# ╟─a078a211-e390-4158-9076-d154921bf8b4
# ╠═22b15086-cde5-4e07-90e2-0cfd82c34889
# ╟─8076e8af-35b9-4113-ba89-c3e5bb25812f
# ╟─862eeec0-bb67-451e-b2f9-bc633075d154
# ╟─4ac43cab-40fc-47b3-bbfa-13b98bcb0a47
# ╟─324ea2b8-c350-438d-8c8f-6404045fc19f
# ╟─206c82ee-389e-4c92-adbf-9578f7125418
# ╟─1a062043-8cfe-4946-8eb0-688d3896229e
# ╟─88093782-d180-4422-bd92-357c716bfc89
# ╟─91a6f3de-01c6-4215-9928-1ecce113adc1
# ╟─f298ffd1-b1dd-472f-b761-c5e07a9121c0
# ╟─2459a2d9-4f48-46ab-82e5-3968e713f15f
# ╟─32eb5c4f-8fde-44c1-a748-905de6aaf364
# ╟─79a75f04-3aa9-4a8b-9f88-1fe8c123680b
# ╟─ae5324a5-0cf6-4485-aead-86c3d8f3041a
# ╟─16d84881-d40a-4694-9812-6896b336accc
# ╟─dd0a4446-65ed-4b3d-87a6-ba950457aa72
# ╟─72697e34-6f63-4d6b-a9ff-ae23a00d4ce2
# ╟─dfe4234a-d2a3-4b59-9578-64bd6e8a5c33
# ╟─593eb726-47a5-4783-8872-03a75c6cfb89
# ╟─a24b03a3-b2a0-4b12-b2a8-25c8cf7f843c
# ╟─27385a11-e8de-4d88-898a-8b9bd0a73ee2
# ╟─e7dac1a5-dc63-4364-a7d6-13ab9808c9c6
# ╟─7d830f79-e361-4de1-b3fd-dee08fd1cc06
# ╟─2588893b-7978-45af-ac71-20cb2af34b10
# ╟─a0fa5474-3cb7-4829-9e2e-2bf53ca00d94
# ╟─52ff5315-002c-480b-9a4b-c04124498277
# ╟─8fd7703b-db80-4aa4-977f-c7e3ad1f9fb6
# ╟─0469b52f-ce98-4cfa-abf3-be53630b30dd
# ╟─4f884fee-4934-46c3-8352-0105652f8537
# ╟─9dbf5502-fa44-404f-88ae-be3488e3e41c
# ╟─5dc4531b-cc10-41de-b055-2ed4270ae2cc
# ╟─73d71e0a-f957-426c-a356-3238032603d9
# ╟─a67f4cb1-6558-466d-9ebc-df21dd83ce96
# ╟─479fb04e-c95f-4522-9195-6e6e9648565b
# ╟─3e980014-daf7-4d8b-b9e6-14a5d078e3b6
# ╟─5d2f56e8-21b2-4aa9-b450-40f7881489e0
# ╟─6cbddc5d-ae3f-43ac-9b7a-bbc779739353
# ╟─64a5e292-14b4-4df0-871d-65d9fec6201d
# ╟─96b18c60-87c6-4c09-9882-fbbc5a53f046
# ╟─35d5f991-5194-4174-badc-324ad7d15ac3
# ╟─f693814d-9639-4abb-a5b4-e83fe7a28a77
# ╟─67f7449f-19b8-4607-9e32-0f8a16a806c0
# ╟─fb0361a6-c967-4dcd-9002-55ae25e225a5
# ╟─5c0eaab3-de6d-457f-a0c3-8ea6b5da2c88
# ╟─de93ac5e-bec6-4764-ac6d-f84076ff20ee
# ╟─a50cc950-45ca-4745-9f47-a3fbb28db782
# ╟─1421c748-083b-4977-8a88-6a39c9b9906d
# ╟─82baa991-6df5-4561-9643-52db96c5e99b
# ╟─9a02f4f8-38d6-44a4-9118-1440bfc4d271
# ╟─fed8b7b2-34c4-445c-bf11-21b5161c8d28
# ╟─cb9cc32f-a12b-43ec-83c8-3ed26a96992f
# ╟─ce822f87-d2c4-434b-9b5d-b629962d2df2
# ╟─e67259ac-e6a9-476d-9795-520e4b6a7b9b
# ╟─3abbf43d-a288-46e6-989f-d92a0bef8b18
# ╟─4927a13b-3c3a-401a-a772-8f1602ea5129
# ╟─ebf67e2b-ac81-4232-b000-209b09e378a8
# ╟─85fe861d-f9fe-4416-aa3a-f03fb1d6b393
# ╟─4bf3d6c3-1778-400e-8133-6e51a053537e
# ╟─23e726a1-1620-464b-a6d5-3eca16179534
# ╟─cbe1feda-ae3f-4e4d-a43b-101cc35dfeb1
# ╟─5c0bc1cb-d330-45dc-87f2-d33c3a2d41e7
# ╟─2ad9f96c-01ee-4410-9940-1feb5df3ba0a
# ╟─11bc7b24-2c5e-49b6-a153-e1d128d52b4a
# ╟─a836ddca-e9f0-430f-948b-bc4683c48245
# ╟─824baded-9755-4ff1-8e26-bf7df96ba0e5
# ╟─9d25c603-07b3-4758-9b28-0d2041e18569
# ╟─929916d2-9abf-40b2-9399-85c3ba05989b
# ╟─8743242e-7f7f-4f54-b27c-1f08d1a110e2
# ╟─aaaadaa8-d2e9-4647-82de-5547b2d6ddb4
# ╟─6e62b53d-68d8-45de-9a0f-8223b5d2df92
# ╟─7ed0de94-b513-40c1-9f83-6ed75fcd4cdd
# ╟─188e9981-9597-483f-b1e3-689e26389b61
# ╟─50d9a624-20d8-4feb-a565-4efcaf22fe27
# ╟─ec78bc4f-884b-4525-8d1b-138b37274ee7
# ╟─e88db991-bf54-4548-80cb-7cd307300ec9
# ╟─99c3f5ee-63d0-4f6f-90d9-2e524a7e945a
# ╟─a270b1b3-7bb4-4612-9e57-3ea9b7daedc0
# ╟─872aa766-2ed8-4cb3-a029-4a2bb42c90a8
# ╟─12991c72-6b19-47ed-9ebe-90e3d615f985
# ╟─03840aa7-4258-47f7-b802-92ab93c92911
# ╟─daf19517-4123-49a8-affb-5d869e08480a
# ╟─8398cb20-b9f4-427c-8b10-2a572f8a5632
# ╟─ff397d54-42b3-40af-bf36-4716fd9a4419
# ╟─fcc49f3c-6219-40b8-a5d0-85fa70738a8d
# ╟─6dc74312-0e3b-4ead-b369-f9c2b70ab3d3
# ╟─e0f3cee1-b6ee-4399-8e4f-c0d70b94723e
# ╟─1039a13b-6665-4bb1-82f8-11b740c3b5ba
# ╟─aa5b5ee6-b5fe-4245-80fd-ab3ab3963d59
# ╟─819d01e1-4b81-4210-82de-eaf838b6a337
# ╟─4d4a45b5-e221-4713-b3fa-3eb36a813385
# ╟─dc70f9a4-9b52-490a-9a94-95fe903401ce
# ╟─f7499d8d-e511-4540-ab68-84454b3d9cd9
# ╟─0cf4d7db-5545-4b1c-ba6c-d9a6a3501e0e
# ╟─ea08837f-3535-4807-92e5-8091c3911948
# ╟─9522421a-1f8e-494d-a61f-f5f8dff72c83
# ╟─39082958-54ed-4120-a7a1-f40f3dc9d965
# ╟─3d0c370e-d3e5-40ec-8e66-745e4f990e18
# ╟─3583f790-bcc3-466b-95e8-e99a080e5658
# ╟─06a80959-58de-4e21-bfdf-5b06caf157f1
# ╟─eb5b710c-70a2-4237-8913-cd69e34b8e50
# ╟─2aa229e0-f578-4ff6-80a3-4897e2ad187f
# ╟─ba0397a7-f514-42f8-b094-2ce5bd95b081
# ╟─e1659838-ecf2-4205-b23c-227c586f6cc3
# ╟─1aa0cc79-e9ca-44bc-b9ab-711eed853f00
# ╟─bf7cd9d6-c7a4-4715-a57c-d0acaef1e7d8
# ╟─0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
# ╠═82d513f2-be95-4c3a-8eb4-18c48a7d0b44
# ╟─3a083374-afd6-4e64-95bd-d7e6385ab403
# ╠═6cd96390-8565-499c-a788-fd0070331f25
# ╠═8fe5631a-1f10-4af4-990d-5a23c96fb73b
# ╠═619df17d-9f75-463f-afe5-6cbffb0762d5
# ╠═8687dbd1-4857-40e4-b9cb-af469b8563e2
# ╠═fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
