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
	default(dpi=300)
	using StatsPlots
	# import PlotlyBase
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

# ╔═╡ 1b8abb8b-f53e-4bde-b649-1557b8e7906c
begin

	using MLDatasets
	using Images
	ENV["DATADEPS_ALWAYS_ACCEPT"] = true
	mnist_train_X, mnist_train_ys = MNIST(split=:train)[:];
	mnist_test_X, mnist_test_ys = MNIST(split=:test)[:];
	# begin
	# ENV["DATADEPS_ALWAYS_ACCEPT"] = true
	# mnist_train_X, mnist_train_ys = MNIST(split=:train)[:];
	# mnist_test_X, mnist_test_ys = MNIST(split=:test)[:];

end;

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

# CS5014 Machine Learning 


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


* **Stochastic gradient descent** [_An Alternative View: When Does SGD Escape Local Minima?_ by Robert Kleinberg, Yuanzhi Li, Yang Yuan ICML](https://arxiv.org/pdf/1802.06175.pdf)

"""

# ╔═╡ bcc3cea5-0564-481b-883a-a45a1b870ba3
md"""
## Binary classification



"""

# ╔═╡ 2d70884d-c922-45e6-853e-5608affdd860
md"""

## A first attempt -- use regression


"""

# ╔═╡ 7b4502d1-3847-4472-b706-65cb68413927
# plot_ls_class=let
# 	gr()
# 	plt = plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], st=:scatter, label="class 1", ratio=1, c=2, legend=:topleft)
# 	plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], st=:scatter, framestyle=:origin, label="class 2", xlim=[-8, 8], ylim=[-6, 6], c=1)
# 	plot!(-6:1:6, (x) -> - w_d₂[1]/w_d₂[3] - w_d₂[2]/w_d₂[3] * x, lw=4, lc=:gray, label="Decision bounary: "*L"h(\mathbf{x}) =0", title="Least square classifier")

# 	plot!([0, w_d₂[2]*3], [0, w_d₂[3]*3], line = (:arrow, 3), c=2, label="")

# 	minx, maxx = minimum(D₂[:,2])-2, maximum(D₂[:,2])+2
# 	miny, maxy = minimum(D₂[:,3])-2, maximum(D₂[:,3])+2
# 	xs, ys = meshgrid(minx:0.2:maxx, miny:0.2:maxy)
# 	colors = [dot(w_d₂ , [1, x, y]) > 0.0   for (x, y) in zip(xs, ys)] .+ 1
# 	scatter!(xs, ys, color = colors,
#             markersize = 1.5, label = nothing, markerstrokewidth = 0, markeralpha=0.5, xlim= [minx, maxx], ylim =[miny, maxy], framestyle=:origin)
# 	plt
# end;

# ╔═╡ 3315ef07-17df-4fe1-a7a5-3e8f039c0cc1
linear_reg(X, y; λ = 1) = (X' * X + λ *I) \ X' * y;

# ╔═╡ 331571e5-f314-42db-ae56-66e64e485b85
md"""

## Problem with the model

#### The prediction function is unbounded

```math
\Large
\begin{align}
\hat{y}^{(i)} &= \mathbf{w}^{\top} \mathbf{x}^{(i)} \\
&\in (-\infty, \infty)
\end{align}
```


#### But, the target ``y`` is *binary*
  * *e.g.* ``\hat{y}^{(i)} = 100``, or ``-100`` does not make sense


"""

# ╔═╡ c8e55a60-0829-4cc7-bc9b-065809ac791c
md"""

## `Logistic` or `Sigmoid`

"""

# ╔═╡ 5ed6a7a3-7871-467f-9ff0-779e2c403bac
TwoColumn(md"""


#### `logistic` (*aka* `sigmoid`)

```math
\large
\texttt{logistic}(x) \triangleq \sigma(x) = \frac{1}{1+e^{-x}}.
``` 

* ``\sigma`` shorthand notation for the activation


* ##### it squeezes a line to 0 and 1
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
	gr()
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
md"""

Add ``\sigma(h(x))``: $(@bind add_sigma_h CheckBox(default=false))
"""

# ╔═╡ e11f7ee9-e50a-4c7c-a5aa-acc6f712a063
md"Slope: ``w_1`` $(@bind w₁_ Slider([(0.0:0.00001:4.)...; (4.0:0.0001:10)...], default=1.0, show_value=true)),"

# ╔═╡ 5997601f-5ee4-4ade-91f6-3bbdaa46f17c
let
	w₀_ = 0
	k, b= w₁_, w₀_
	gr()
	plot(-15:0.1:15, (x) -> logistic( x ), xlabel=L"x", label=L"\sigma(1x + 0.0)", legend=:topleft, lw=3, size=(800,800), c=1, ratio=10)
	plot!(-15:0.1:15, (x) -> k * x+ b, ylim=[-1., 2], xlim =[-15, 15],label=L"h(x) =%$(round(w₁_; digits=2)) x + %$(w₀_)", lw=3, lc=7, ls=:dash,  framestyle=:origin, legend=:outerbottom, legendfontsize=14, labelfontsize=14, ytickfontsize=14)
	
	x₀_ = -b/k
	if add_sigma_h
		plot!(-15:0.1:15, (x) -> logistic(k * x + b), xlabel=L"x", label=L"\sigma(h(x))=\sigma(%$(round(w₁_; digits=2)) x + %$(w₀_))", lw=3, lc=7)
		plot!([x₀_, x₀_], [0, .5], lw=2, ls=:dash, lc=:gray, st=:line, label="")
		plot!([.5, x₀_], [.5, .5], lw=2, ls=:dash, lc=:gray, st=:line, label="")
	end


	plot!( x -> 1, c=:gray, lw=1, label="", ls=:dash)
	plot!( x -> 0, c=:gray, lw=1, label="", ls=:dash)
	
end

# ╔═╡ 4cac5129-3b3d-42c4-839c-254ac26199f1
md"""

## Logistic function 
##### -- higher-dimensional


##### It *squeezes* a *hyperplane* (or multiple lines) instead of a *line*

```math
\large
h(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + w_0\tag{a hyper-plane}
```


```math
\large
\sigma(h(\mathbf{x})) = σ(\mathbf{w}^\top \mathbf{x} + w_0) = \frac{1}{1+e^{-\mathbf{w}^\top \mathbf{x} - w_0}}\tag{a ``S" plane}
```
"""

# ╔═╡ 137441e2-2f6c-43bb-b1a2-b240ee3ab6e7
ww = [1, 1]

# ╔═╡ 246b8623-2f47-4ac1-b562-75d50889e6d8
md"Add lines: $(@bind add_p_lines CheckBox(default=false)), show transformed: $(@bind show_sigmoid CheckBox(false))"

# ╔═╡ b146a786-ee54-45d8-ade1-474af127451f
TwoColumn(
	let
	plotly()
	w₀ = 0
	f(x1, x2) = dot(ww, [x1, x2]) + w₀
	x_center = [0, 0]
	plt = plot(-15:2:15, -15:2:15, (x1, x2) -> f(x1, x2), st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.8, framestyle=:zerolines, c=:coolwarm, colorbar=false, camera=(-5, 30), size=(350,400))

	# min_z, max_z = extrema([f(x1, x2) for x1 in -15:2:15 for x2 in -15:2:15])
	if add_p_lines
		x00= [x_center..., f(x_center...)]
		wv = [ww..., ww'*ww]
		ts = -14:2:14
		# xyzs = x00 .+ wv .* ts'
		vv = ww[2] == 0 ? [0, 1] : [1, -ww[1]/ww[2]]
		vv = vv/norm(vv)
		vvs = x00[1:2] .+ vv * range(-15, 15, 15)'
		vvs =[vvs; f(x_center...) .+ zeros(size(vvs)[2])']

		for v0 in eachcol(vvs)
			xyzs = v0 .+ wv .* ts'
			path3d!(xyzs[1,:], xyzs[2,:], xyzs[3,:], lw=3,label="")
		end
	end
	


	plt
end
	
	, 
let
	if show_sigmoid 
		plotly()
		w₀ = 0
		f(x1, x2) = dot(ww, [x1, x2]) + w₀
		x_center = [0, 0]
		plt = plot(-15:0.2:15, -15:0.2:15, (x1, x2) -> logistic(f(x1, x2)), st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.8, framestyle=:zerolines, c=:coolwarm, colorbar=false, camera=(-5, 30), size=(350,400))
		if add_p_lines
			x00 = [x_center..., f(x_center...)]
			wv = [ww..., ww'*ww]
			ts = -14:0.2:14
			vv = ww[2] == 0 ? [0, 1] : [1, -ww[1]/ww[2]]
			vv = vv/norm(vv)
			vvs = x00[1:2] .+ vv * range(-15, 15, 15)'
			vvs =[vvs; f(x_center...) .+ zeros(size(vvs)[2])']
	
			for v0 in eachcol(vvs)
				xyzs = v0 .+ wv .* ts'
				newzs = [logistic(f(xy[1:2]...)) for xy in eachcol(xyzs)]
				path3d!(xyzs[1,:], xyzs[2,:], newzs, lw=3,label="")
			end
		end
		plt
	else
		md"""

		"""
	end
end

)

# ╔═╡ 6ba6bed9-d913-4511-9829-787fe8a09fa7
md"""
## Logistic function (cont.)
##### -- higher-dimensional

##### The gradients are 

```math
\Large
\nabla h(\mathbf{x}) = \mathbf{w};\;\;\; \nabla \sigma(h(\mathbf{x})) = \sigma'(h(\mathbf{x}))\cdot \mathbf{w} 
```

"""

# ╔═╡ 22b15086-cde5-4e07-90e2-0cfd82c34889
wv_ = [1, 1] * 1;

# ╔═╡ a078a211-e390-4158-9076-d154921bf8b4
md"##### With ``\mathbf{w}=``$(latexify_md(wv_)), and ``w_0 =0``"

# ╔═╡ 324ea2b8-c350-438d-8c8f-6404045fc19f
# md"""

# ## Logistic function

# """

# ╔═╡ 206c82ee-389e-4c92-adbf-9578f7125418
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.

# ╔═╡ 01d6f011-da4f-4f6e-a75a-53123bfeb9fd
md"""

## 


#### If we change $\mathbf{w} = \begin{bmatrix}-1 \\ -1 \end{bmatrix}$ and keep $w_0=0$, 
"""

# ╔═╡ 1a062043-8cfe-4946-8eb0-688d3896229e
md"""

## Logistic regression -- as a graph


```math
\Large 
\sigma(\mathbf{x}) = \frac{1}{1+ e^{- \mathbf{w}^\top\mathbf{x}}}
``` 

* ##### ``z = \mathbf{w}^\top \mathbf{x}`` is called the *logit* value
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

## The squared error loss -- visualisation


#### _Note this is the parameter space_

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

## Gradient descent: get stuck!
"""

# ╔═╡ 593eb726-47a5-4783-8872-03a75c6cfb89
md"""

## Gradient descent: if you are lucky 
"""

# ╔═╡ 78dafb40-f1c7-47e4-b043-170f98588124
md"""

## Aside: convexity

#### Convex ``\textcolor{blue}{f(\mathbf{x})}``: if ``\mathbf{x}_1, \mathbf{x}_2 \in \text{dom} f``, and ``\theta \in [0,1]``, we have 

```math
\Large
f(\theta \mathbf{x}_1 +(1-\theta) \mathbf{x}_2) \leq \theta f(\mathbf{x}_1) + (1-\theta) f(\mathbf{x}_2)
```

"""

# ╔═╡ 89ebe16c-845a-4123-bb74-68654327aa05
html"""<center><img src="https://tisp.indigits.com/_images/convex_function.png" width = "500"/></center>"""

# ╔═╡ 27385a11-e8de-4d88-898a-8b9bd0a73ee2
md"""


## The squared error loss is not convex



> #### _Or_ it has too many *flat* areas!

* ##### the *consequence*: gradient descent gets stuck


* ##### DO NOT use squared error for classification!


> #### What loss to use then?
> * ##### hint: use likelihood ``p(y|\mathbf{x}, \mathbf{w})``
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



##### Since ``y^{(i)} \in \{0,1\}`` is binary,
* ##### a natural choice of ``p(y^{(i)}|\ldots)`` is Bernoulli !

> ```math
> \large
> 
> p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) = \texttt{Bernoulli}(\sigma^{(i)}) =\begin{cases}\sigma^{(i)} & y^{(i)} =1
> \\
> 1-\sigma^{(i)} & y^{(i)} = 0   \end{cases}
> ```

* ##### short-hand notation ``\sigma^{(i)} =\sigma(\mathbf{w}^\top\mathbf{x}^{(i)})``


"""

# ╔═╡ 4f884fee-4934-46c3-8352-0105652f8537
md"""


## Probabilistic model for logistic regression (cont.)



##### Since ``y^{(i)} \in \{0,1\}`` is binary,
* ##### a natural choice of *likelihood* is Bernoulli !



> ```math
> \large
> 
> p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) = \texttt{Bernoulli}(\sigma^{(i)}) =\begin{cases}\sigma^{(i)} & y^{(i)} =1
> \\
> 1-\sigma^{(i)} & y^{(i)} = 0   \end{cases}
> ```

* ##### short-hand notation ``\sigma^{(i)} =\sigma(\mathbf{w}^\top\mathbf{x}^{(i)})``

##### The probabilistic generative model therefore is

---

##### for each ``\mathbf{x}^{(i)}``
  * ##### compute bias ``\sigma^{(i)} =\sigma(\mathbf{w}^\top \mathbf{x}^{(i)})``
  * ##### *toss a coin with bias* ``y^{(i)} \sim \texttt{Bernoulli}(\sigma^{(i)})``

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

# ╔═╡ fb52e9ee-b685-4d8f-8bab-b5328f300472
md"""

## Proper Loss for binary classification



> ```math
> \large
> 
> p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) = \texttt{Bernoulli}(\sigma^{(i)}) =\begin{cases}\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} =1
> \\
> 1-\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} = 0   \end{cases}
> ```



#### Recall Bernoulli likelihood can be written in one-line

```math
\large
p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) =  (\sigma^{(i)})^{y^{(i)}} (1-\sigma^{(i)})^{1- y^{(i)}}
```


"""

# ╔═╡ 96b18c60-87c6-4c09-9882-fbbc5a53f046
md"""

## Proper Loss for binary classification



> ```math
> \large
> 
> p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) = \texttt{Bernoulli}(\sigma^{(i)}) =\begin{cases}\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} =1
> \\
> 1-\sigma(\mathbf{w}^\top\mathbf{x}^{(i)}) & y^{(i)} = 0   \end{cases}
> ```



#### Recall Bernoulli likelihood can be written in one-line

```math
\large
p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) =  (\sigma^{(i)})^{y^{(i)}} (1-\sigma^{(i)})^{1- y^{(i)}}
```


- #### the log-likelihood is


```math
\large
\ln p(y^{(i)}|\mathbf{w}, \mathbf{x}^{(i)}) =  {y^{(i)}} \ln \sigma^{(i)} + (1- y^{(i)}) \ln (1-\sigma^{(i)})
```

- #### the loss is just the _negative_ log-likelihood (known as binary cross entropy loss)
  $$\Large
  \boxed{\ell^{(i)}(\mathbf{w}) = - {y^{(i)}} \ln \sigma^{(i)}- (1- y^{(i)}) \ln (1-\sigma^{(i)})}$$


"""

# ╔═╡ b6d6471a-e5dc-44cd-8c3b-24b150e1ccad
md"""


## Cross entropy loss

##### -- implementation of the loss



#### Since $y^{(i)} \in \{0, 1\}$ are binary, the loss

$$\Large
  \boxed{\ell^{(i)}(\mathbf{w}) = - {y^{(i)}} \ln \sigma^{(i)}- (1- y^{(i)}) \ln (1-\sigma^{(i)})}$$

#### Reduces to

```math
\Large
\ell^{(i)}(\mathbf{w}) = \begin{cases}
-\ln (1-\sigma^{(i)})& y^{(i)} = 0\\
-\ln(\sigma^{(i)})& y^{(i)} = 1
\end{cases}
```


* ##### this is the preferred approach to implement

* ##### when $y^{(i)}=1, \sigma^{(i)}=1$ (the prediction is perfect, the loss should be 0), the naive implementation:

  $$\Large\begin{align}

   \ell^{(i)}(\mathbf{w}) &= - 1 \ln 1- 0 \ln 0 = \texttt{NaN}\\
  \end{align}$$

  * ###### since ``0\ln(0) = \texttt{NaN}``
"""

# ╔═╡ 862b33ee-e1b2-4261-be22-47fb7122a3ac
0 * log(0)

# ╔═╡ 35d5f991-5194-4174-badc-324ad7d15ac3
md"""

## Cross entropy `Loss` 


#### The joint loss is the mean

```math
\Large
\begin{align}
\ell(\mathbf{w}) &= \frac{1}{n}\sum_{i=1}^n \ell^{(i)}(\mathbf{w}) \\
&= - \frac{1}{n}\sum_{i=1}^n{y^{(i)}} \ln \sigma^{(i)}+ (1- y^{(i)}) \ln (1-\sigma^{(i)})

\end{align}
```

* ##### so the gradient size does not vary with the training data size
"""

# ╔═╡ a50cc950-45ca-4745-9f47-a3fbb28db782
md"""


## `1/0` loss (classification accuracy)
"""

# ╔═╡ 1421c748-083b-4977-8a88-6a39c9b9906d
TwoColumn(md"""


##### For classification, a more *natural loss* is 1/0 classification accuracy:


```math
\Large
\begin{align} \ell_{1/0}({y}, \sigma) = \mathbb{1}(y \neq \hat{y})
\end{align}
```

* #### where ``\hat{y} = \mathbb{1}(\sigma > 0.5)`` is the prediction label
\

* ##### however, it is *not* differentiable 
  * the gradients are zero _everywhere_

""", let
	gr()

	plot(0:0.001:1, (x) -> x < .5, lw=4, xlabel="Predicted prob. "* L"\sigma^{(i)}= p(y^{(i)}=1|x^{(i)})", ylabel="loss", label=L"1/0"* " loss", title="When the class label is 1", ylim =[-0.2, 4], size=(350,350))

	quiver!([0.25], [2], quiver=([0.75- 0.25], [2-2]), c=:green, la=0.5, lw=3)
	quiver!([0.75], [3], quiver=([0.25- 0.75], [3-3]), c=:red, lw=3)
	annotate!(0.5, 2.1, text("better", "Computer Modern", :green, :bottom))
	annotate!(0.5, 3.1, text("worse", "Computer Modern", :red, :bottom))
end)

# ╔═╡ f693814d-9639-4abb-a5b4-e83fe7a28a77
md"""

## Cross-entropy loss -- how ?


"""

# ╔═╡ 67f7449f-19b8-4607-9e32-0f8a16a806c0
TwoColumn(md"""

\


#### When ``y^{(i)} = 1``, *i.e.* the true label is 1, the CE loss 


```math
\Large
\begin{align}
 \ell^{(i)}= - \ln (\sigma^{(i)})
\end{align}
```


* ##### the prediction is correct,  the loss ``\rightarrow`` zero
* ##### the prediction is wrong, the loss ``\rightarrow`` `inf`

""", let
	gr()
	plot(0:0.005:1, (x) -> -log(x), lw=2, xlabel="Predicted prob. "* L"\sigma^{(i)}= p(y^{(i)}=1|x^{(i)})", ylabel="loss", label=L"-\ln \sigma^{(i)}", title=L"y^{(i)}=1"* ": i.e. class label is 1", annotate = [(1, 0.9, text("perfect pred", "Computer Modern", :right, rotation = 270 ,:green, 12)), (0.11, 3.5, text("worst pred", :right, "Computer Modern", rotation = 270 ,:red, 12))], size=(350,350))


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



#### When ``y^{(i)} = 0`` (the true label is 0)


```math
\Large
\begin{align}
 \ell^{(i)}(\mathbf{w}) =- \ln (1-\sigma^{(i)})
\end{align}
```
""", let
	gr()
	plot(0:0.005:1, (x) -> -log(1-x), lw=2, xlabel="Predicted prob. "* L"\sigma^{(i)}= p(y^{(i)}=1|x^{(i)})",  ylabel="loss", label=L"-\ln(1-\sigma^{(i)})", title=L"y^{(i)}=0"* ": the true class label is 0", size=(350,350))


	# quiver!([0], [0.8], quiver=([1-1], [0-0.8]), c=:red, lw=3)
	
	# quiver!([0.07], [5], quiver=([0.0- 0.06], [5-5]), c=:green, lw=3)

	quiver!([0.25], [2], quiver=([0.75- 0.25], [2-2]), c=:red, lw=3)
	quiver!([0.75], [3], quiver=([0.25- 0.75], [3-3]), c=:green, lw=3)
	annotate!(0.5, 2.1, text("worse", "Computer Modern", :red, :bottom))
	annotate!(0.5, 3.1, text("better", "Computer Modern", :green, :bottom))
end)

# ╔═╡ 82baa991-6df5-4561-9643-52db96c5e99b
md"""


## Surrogate loss comparison

"""

# ╔═╡ 9a02f4f8-38d6-44a4-9118-1440bfc4d271
let
	gr()

	plot(0:0.001:1, (x) -> x < .5, lw=2, xlabel="Predicted prob. "* L"\sigma^{(i)}= p(y^{(i)}=1|x^{(i)})",  ylabel="loss", label=L"1/0"* " loss", title="When the class label is 1", ylim =[-0.2, 5], size=(550,500), legendfontsize=12, labelfontsize=14)

	plot!(0:0.005:1, (x) -> -log(x), lw=2,  ylabel="loss", label="Cross Entropy loss")

	plot!(0:0.005:1, (x) -> (x-1)^2, lw=2, lc=4, label="Squared error loss")

	# plot!(0:0.005:1, (x) -> (1-x)^2+ (x-1)^2, lw=2, lc=4, ls=:dash, label="Bier loss")
	# quiver!([1], [0.8], quiver=([1-1], [0-0.8]), c=:green, lw=3)
	# quiver!([0.07], [5], quiver=([0.0- 0.06], [5-5]), c=:red, lw=3)

	quiver!([0.25], [2], quiver=([0.75- 0.25], [2-2]), c=:green, la=0.5, lw=3)
	quiver!([0.75], [3], quiver=([0.25- 0.75], [3-3]), c=:red, lw=3)
	annotate!(0.5, 2.1, text("better", "Computer Modern", :green, :bottom))
	annotate!(0.5, 3.1, text("worse", "Computer Modern", :red, :bottom))
end

# ╔═╡ 0f692780-e79c-4aec-a0f4-29c4efbe5390
md"""

## Loss comparison
"""

# ╔═╡ 57cce3be-b168-425a-bd36-79b9ed2327ea
# logistic_mse_loss = 1

# ╔═╡ e67259ac-e6a9-476d-9795-520e4b6a7b9b
md"""

## Learning -- gradient descent



```math
\large
\hat{\mathbf{w}}_{\text{MLE}} \leftarrow \arg\min_{\mathbf{w}} \underbrace{\frac{1}{n}\sum_{i=1}^n-{y^{(i)}} \ln \sigma^{(i)}- (1- y^{(i)}) \ln (1-\sigma^{(i)})}_{\text{CE Loss: } \ell(\mathbf{w})}
```



**Bach gradient descent algorithm:**

-----

* ##### random guess ``\large\mathbf{w}_0``

* ##### while ``\|\nabla \ell(\mathbf{w}_{t-1})\|_2 > \texttt{tol}``
  * ###### ``\large\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma \nabla \ell(\mathbf{w}_{t-1})``
-----



"""

# ╔═╡ 3abbf43d-a288-46e6-989f-d92a0bef8b18
md"""
## Gradient derivation


```math
\Large
\begin{align}
\ell^{(i)}(\mathbf{w}) 
&= - {y^{(i)}} \ln \sigma^{(i)}- (1- y^{(i)}) \ln (1-\sigma^{(i)})
\end{align}
```

* #### where

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
show_img("/logistic_grad_flow1_new.svg", w=750)

# ╔═╡ 85fe861d-f9fe-4416-aa3a-f03fb1d6b393
md"""

## Gradient derivation

#### Chain rule by graph

\


```math
\Large
\nabla\ell^{(i)} = \frac{\partial \ell^{(i)}}{\partial \sigma^{(i)}} \frac{\partial \sigma^{(i)}}{\partial z^{(i)}}\frac{\partial z^{(i)}}{\partial \mathbf{w}}
```
"""

# ╔═╡ 4bf3d6c3-1778-400e-8133-6e51a053537e
show_img("/logistic_grad_flow2_new.svg", w=750)

# ╔═╡ 23e726a1-1620-464b-a6d5-3eca16179534
md"""

## Gradient derivation

#### Chain rule by graph

"""

# ╔═╡ cbe1feda-ae3f-4e4d-a43b-101cc35dfeb1
show_img("/logistic_grad_flow3_new.svg", w=750)

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
show_img("/logistic_grad_flow4_new.svg", w=750)

# ╔═╡ a836ddca-e9f0-430f-948b-bc4683c48245
md"""

##### Multiply the three together and tidy up


```math
\large
\begin{align}
\nabla{\ell}^{(i)}(\mathbf{w}) &= \left(-\frac{y^{(i)}}{\sigma^{(i)}} + \frac{1-y^{(i)}}{1-\sigma^{(i)}} \right)  \left(\sigma^{(i)} (1-\sigma^{(i)} ) \right) \mathbf{x}^{(i)}\\
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
\nabla \ell(\mathbf{w})  = -\sum_{i=1}^n \underbrace{(y^{(i)} - \sigma^{(i)})}_{\text{pred. error for }i} \cdot \mathbf{x}^{(i)} 
```

##### The _gradient_ for *linear regression* is


```math
\large
\nabla \ell(\mathbf{w})  = -\sum_{i=1}^n \underbrace{(y^{(i)} - \mathbf{w}^{\top} \mathbf{x}^{(i)})}_{\text{pred. error for }i} \cdot \mathbf{x}^{(i)} 
```

##### The same idea: the gradient is proportional to the prediction quality
* prediction is perfect: gradient is zero
* otherwise, adjust accordingly
"""

# ╔═╡ aaaadaa8-d2e9-4647-82de-5547b2d6ddb4
md"""

## Gradient -- matrix notation

#### The gradient is


```math
\large
\nabla\ell(\mathbf{w})  = - \frac{1}{n}\sum_{i=1}^n \underbrace{(y^{(i)} - \sigma^{(i)})}_{\text{pred. error for }i} \cdot \mathbf{x}^{(i)} 
```

#### In matrix notation, 

```math
\large
\nabla \ell(\mathbf{w}) =-\frac{1}{n} \mathbf{X}^\top (\mathbf{y} - \boldsymbol{\sigma})
```

* ##### where 

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

# ╔═╡ 7ed0de94-b513-40c1-9f83-6ed75fcd4cdd
md"""

## Learing algorithm -- Implementation


"""

# ╔═╡ 188e9981-9597-483f-b1e3-689e26389b61
md"""


**Implementation in Python**


```python
def logistic(x):    
    return 1/ (1 + np.exp(-x))
```


```python
for i in range(max_iters):
	yhats = logistic(Xs@w) # predictions (btw 0 and 1)
    grad =  Xs.T@(yhats - ys) / np.size(ys) # the gradient
    w = w - gamma * grad  # gradient descent
```


"""

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

# ╔═╡ b64f9f8c-6bf7-44d1-a9a2-c51d35f39491
TwoColumn(show_img("logisticgif1.gif", w=450), show_img("logisticgif2.gif", w=450))

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

# ╔═╡ 1aa0cc79-e9ca-44bc-b9ab-711eed853f00
# using MLUtils

# ╔═╡ b7de7efa-43ef-485c-af2f-8868c7665cc1
md"""

## Case study: MNIST


#### **Binary classification**: classify zeros from non-zeros

* ##### `0`s (the positive case) from 

* ##### non-zeros `{1,2,…,9}` (negative cases)

\


#### Flatten the input ``28 \times 28`` to a long ``28^2`` long vector 
* ##### +1: dummy one for the intercept
* ##### use logistic regression to train the classifier
"""

# ╔═╡ bd5e2a40-6258-4ecd-a926-6d408cb02387
function flatten_images(X) # flatten the matrix input to a vector
	vcat([X[:,:,i][:]' for i in 1:size(X)[3]]...)
end;

# ╔═╡ 90d0b85d-80d9-421b-a22b-fde00b2a9180
mnist_zeros_xs_train, mnist_zeros_ys_train, mnist_zeros_xs_test, mnist_zeros_ys_test = let #prepare the dataset
	Random.seed!(123)
	train_size = 500
	zeros_ids = mnist_train_ys .== 0
	zeros_indices = shuffle!(findall(zeros_ids))[1:train_size]
	non_zero_indices = shuffle!(findall(.!zeros_ids))[1: 5*train_size]
	x_zeros = mnist_train_X[:, :, zeros_indices]
	X_zeros = flatten_images(x_zeros)
	x_nzeros = mnist_train_X[:, :, non_zero_indices]
	X_nzeros = flatten_images(x_nzeros)
	Xs_zeros = [X_zeros; X_nzeros]
	ys_zeros = [ones(Bool, length(zeros_indices)); zeros(Bool, length(non_zero_indices))]
	Xs_zeros_ = [ones(size(Xs_zeros)[1]) Xs_zeros]
	X_zeros_test = [ones(length(mnist_test_ys)) flatten_images(mnist_test_X)]
	ys_zeros_test = mnist_test_ys .== 0
	Xs_zeros_, ys_zeros, X_zeros_test, ys_zeros_test
end;

# ╔═╡ b4b2ff74-758d-4b7f-84b7-9c5565c39161
md"""
#### Implementation: gradient descent
"""

# ╔═╡ 2179b785-7869-4a60-9875-14bd59af965f
losses_, ww_ = let
	X = mnist_zeros_xs_train
	y = mnist_zeros_ys_train
	ww = zeros(28^2+1)
	γ = 0.1
	iters = 2500
	losses = zeros(iters+1)
	losses[1] = logistic_loss(ww, X, y)
	for i in 1:iters
		gw = ∇logistic_loss(ww, X, y)/length(y)
		ww = ww - γ * gw 
		losses[i+1] = logistic_loss(ww, X, y)
	end
	losses, ww
end;

# ╔═╡ 9ad301a5-62b2-4dae-97e2-400d4af6831e
md"""

#### Check the learning curve
"""

# ╔═╡ 83369d85-0f0c-4ffe-b1a5-0bacb330f7f7
let
	gr()
	plot(losses_, title="MNIST binary classification loss", xlabel="iteration", ylabel="loss", label="",lw=2)
end

# ╔═╡ b0133033-09bf-4b45-94a6-ec9d172133f8
accuracy(y, ŷ) = mean(y .== ŷ);

# ╔═╡ db594b4a-d22a-402a-a9df-6ca7a1f3cfa9
predict(w, X) = logistic.(X * w) .> 0.5;

# ╔═╡ 93e7a7e8-c8b4-47f1-a410-e87e8434b98d
md"""

## Classification accuracy

"""

# ╔═╡ 717afb21-baa7-4381-9e07-00829df1ac59
train_acc = accuracy(mnist_zeros_ys_train, predict(ww_, mnist_zeros_xs_train))

# ╔═╡ 53edc301-e9a7-4200-bb69-2159e0b3a23d
md"""

### Training accuracy is $(round(train_acc; digits=3))
"""

# ╔═╡ 3bc95495-fccc-4aac-aa13-554ef999e770
test_acc = accuracy(mnist_zeros_ys_test, predict(ww_, mnist_zeros_xs_test))

# ╔═╡ 1a5a22dd-89bf-483b-b7f2-e477512f84ac
md"""

### Testing accuracy is $(test_acc)

"""

# ╔═╡ ac842d64-8127-4b09-94eb-839f68692e5b
md"""
## Visualise the parameter ``\mathbf{w}``


"""

# ╔═╡ fc11e0a8-e658-4e73-b4a7-569be7927014
plt_mnist_w=plot(reshape(ww_[2:end], 28, 28)', st=:heatmap, r=1, c=:jet, size=(380,400));

# ╔═╡ 77636d53-5426-4a52-9e61-430119e881c6
TwoColumn(md"""
\


##### ``\Large\hat{\mathbf{w}} \in \mathbb{R}^{28\times 28}`` can be viewed an image 

\

##### Recall the model

$\large p(y \text{ is a 0}|\mathbf{x}) = \sigma(\hat{\mathbf{w}}^\top\mathbf{x} + b)$

\

##### Does it make sense?

* when $\mathbf{w}^\top\mathbf{x}$ is maximised?
""", plt_mnist_w)

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

# ╔═╡ 0ca92827-2538-405c-a9e1-2ac53bd70453
TwoColumn(let
	gr()
	w₀ = 0
	wv_ = - [1,1 ]
	w₁, w₂ = wv_[1], wv_[2]
	plot(-5:0.5:5, -5:0.5:5, (x, y) -> (w₀+ w₁* x + w₂ * y), st=:contourf, c=:jet, colorbar=false, alpha=0.25, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="contour and gradient: "* L"h(x) =w^\top x", ratio=1, size=(350,360), titlefontsize=12)
	α = 0.5
	xs_, ys_ = meshgrid(range(-5, 5, length=10), range(-5, 5, length=10))
	∇f_d(x, y) = ∇h([1, x, y], [w₀, w₁, w₂])[2:end] * α
	quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
end, let
	gr()
	w₀ = 0
	wv_ = - [1,1 ]
	w₁, w₂ = wv_[1], wv_[2]
	plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:contourf, c=:jet, colorbar=false, alpha=0.25, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="contour and gradient: "* L"σ(w^\top x)", ratio=1, size=(350,360), titlefontsize=12)
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

##### *input* features: ``\mathbf{x}^{(i)} \in \mathbb{R}^m``
* ``m=2`` here


##### *output* label: ``y^{(i)} \in \{0,1\}``


""", plt_binary_2d)

# ╔═╡ b640fc38-2ada-4c35-82f0-53fd801d14e1
TwoColumn(plot(plt_binary_2d, title="2-d view", size=(300,300)), let
	gr()
	plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)), st=:scatter, ms=2, label=L"{y}^{(i)} = 1", zlim=[-.1, 1.1], xlabel = L"x_1", ylabel=L"x_2", c=2, size=(400,300))
	plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], 0* ones(sum(targets_D₂ .== 0)), st=:scatter,  ms=2,framestyle=:origin, label=L"{y}^{(i)} = 0", xlim=[-8, 8], ylim=[-6, 6], title="3-d view", c=1, camera=(45,30))
end)

# ╔═╡ 867507ad-1ca8-4d12-a9c4-646197937547
begin
	plotly()
	scatter(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)), zlim=[-0.9, 1.9], label="y=1", c=2)
	scatter!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], 0 * ones(sum(targets_D₂ .== 0)), label="y=0", c=1, framestyle=:zerolines)
	w_d₂ = linear_reg(D₂, targets_D₂ ; λ=0.0)
	plot!(-5:1:5, -5:1:5, (x,y) -> w_d₂[1] + w_d₂[2] * x + w_d₂[3] * y, alpha =0.9, st=:surface, colorbar=false, c=:jet, title="First attempt: least square fit")
end

# ╔═╡ cbbcf999-1d31-4f53-850e-d37a28cff849
begin
	plotly()
	scatter(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)),  label="y=1", c=2)
	scatter!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], 0 *ones(sum(targets_D₂ .== 0)), label="y=0", framestyle=:zerolines, c=1)
	# w = linear_reg(D₂, targets_D₂;λ=0.0)
	plot!(-10:1:10, -50:1:50, (x,y) -> w_d₂[1] + w_d₂[2] * x + w_d₂[3] * y, alpha =0.8, st=:surface, colorbar=false,c=:jet, title="First attempt: unbounded prediction")
end

# ╔═╡ f298ffd1-b1dd-472f-b761-c5e07a9121c0
let
	gr()
	plt = scatter(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)),  label="y=1", c=2)
	scatter!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], 0 *ones(sum(targets_D₂ .== 0)), label="y=0", framestyle=:zerolines, c=1)
	# w = linear_reg(D₂, targets_D₂;λ=0.0)
	f̂(x, y) = logistic(0 + .5 * x + .5 * y)
	plot!(-8:.5:8, -8:0.5:8, (x,y) -> f̂(x, y), alpha =0.8, st=:surface, colorbar=false,c=:jet, title="Least square curve fitting", camera=(58,20), xlabel=L"x_1", ylabel=L"x_2", zlabel=L"y")

	for (i, xi) in enumerate(eachrow(D₂[:, 2:3]))
		if targets_D₂[i] == 1
			plot!([xi[1], xi[1]], [xi[2], xi[2]], [1, f̂(xi...)], c =2, label="")

		else
			plot!([xi[1], xi[1]], [xi[2], xi[2]], [0, f̂(xi...)], c=1, label="")
		end
	end
	plt
end

# ╔═╡ 79a75f04-3aa9-4a8b-9f88-1fe8c123680b
let
	gr()
	# gr()
	bias = 0.0;

	# p0 = plot(-8:0.1:10, -10:0.1:10, (w1, w2) -> zeroone_loss([bias, w1, w2], D₂, targets_D₂; agg=sum), st=:surface, c=:jet,camera = (45, 35), colorbar=false, title="1/0 error loss")

	p1 = plot(-20:0.5:20, -20:0.1:20, (w1, w2) -> logistic_mse_loss([bias, w1, w2], D₂, targets_D₂; agg=sum), st=:surface, c=:jet, camera = (75, 35), colorbar=false, title="Squared error loss", xlabel=L"w_1", ylabel=L"w_2", zlabel="loss")

	p2 = plot(-20:0.1:20, -20:0.1:20, (w1, w2) -> logistic_mse_loss([bias, w1, w2], D₂, targets_D₂; agg=sum), st=:contour, c=:jet, colorbar=false, title="Squared error loss", ratio=1, ylim =[-20,20], fill=true, alpha=0.5, xlabel=L"w_1", ylabel=L"w_2")
	plot(p1, p2, layout=(1,2))

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

# ╔═╡ d9a62218-ac20-4fab-9786-a7afdcb48945
let
	gr()
	w₀ = 0
	wv_ = - wv_
	w₁, w₂ = wv_[1], wv_[2]
	p1 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> w₀+ w₁* x + w₂ * y, st=:surface, c=:jet, colorbar=false, alpha=0.8, framestyle=:zerolines, xlabel=L"x_1", ylabel=L"x_2", title=L"h(x) = w^\top x")

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
	
	p2 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.8, zlim=[-0.1, 1.1],  xlabel=L"x_1", ylabel=L"x_2", title=L"σ(w^\top x)", framestyle=:zerolines)
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
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
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
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
Distributions = "~0.25.107"
HypertextLiteral = "~0.9.5"
Images = "~0.26.1"
LaTeXStrings = "~1.3.1"
Latexify = "~0.16.1"
LogExpFunctions = "~0.3.26"
MLDatasets = "~0.7.18"
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

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "ad826630e0e9e3b73d3934023e83d05a67a3f7c5"

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

[[deps.AtomsBase]]
deps = ["LinearAlgebra", "PeriodicTable", "Preferences", "Printf", "Requires", "StaticArrays", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "922c2469c526996566dbabd273d15701ed2aacfe"
uuid = "a963bdd2-2df7-4f54-a1ee-49d51e6be12a"
version = "0.5.2"

    [deps.AtomsBase.extensions]
    AtomsBaseAtomsViewExt = "AtomsView"

    [deps.AtomsBase.weakdeps]
    AtomsView = "ee286e10-dd2d-4ff2-afcb-0a3cd50c8041"

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

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "3b642331600250f592719140c60cf12372b82d66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.5.1"

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

[[deps.BufferedStreams]]
git-tree-sha1 = "6863c5b7fc997eadcabdbaf6c5f201dc30032643"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.2.2"

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

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

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

[[deps.Chemfiles]]
deps = ["AtomsBase", "Chemfiles_jll", "DocStringExtensions", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "8d4217428ee7c64605d1217a8ea810436fd03742"
uuid = "46823bd8-5fb3-5f92-9aa0-96921f3dd015"
version = "0.10.43"

[[deps.Chemfiles_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f3743181e30d87c23d9c8ebd493b77f43d8f1890"
uuid = "78a364fa-1a3c-552a-b4bb-8fa0f9c1fcca"
version = "0.10.4+0"

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

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "980f01d6d3283b3dbdfd7ed89405f96b7256ad57"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "2.0.1"

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

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

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

[[deps.DataDeps]]
deps = ["HTTP", "Libdl", "Reexport", "SHA", "Scratch", "p7zip_jll"]
git-tree-sha1 = "8ae085b71c462c2cb1cfedcb10c3c877ec6cf03f"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.13"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d8928e9169ff76c6281f39a659f9bca3a573f24c"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

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

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

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
git-tree-sha1 = "8ddb438e956891a63a5367d7fab61550fc720026"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.2.6"
weakdeps = ["JLD2"]

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

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

[[deps.GZip]]
deps = ["Libdl", "Zlib_jll"]
git-tree-sha1 = "0085ccd5ec327c077ec5b91a5f937b759810ba62"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.6.2"

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

[[deps.Glob]]
git-tree-sha1 = "97285bbd5230dd766e9ef6749b80fc617126d496"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.1"

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
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "e94f84da9af7ce9c6be049e9067e511e17ff89ec"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.6+0"

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

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "3d468106a05408f9f7b6f161d9e7715159af247b"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.12.2+0"

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
git-tree-sha1 = "cffa21df12f00ca1a365eb8ed107614b40e8c6da"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.6"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "783b70725ed326340adf225be4889906c96b8fd1"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.7"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "7196039573b6f312864547eb7a74360d6c0ab8e6"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.9.0"

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

[[deps.InternedStrings]]
deps = ["Random", "Test"]
git-tree-sha1 = "eb05b5625bc5d821b8075a77e4c421933e20c76b"
uuid = "7d512f48-7fb1-5a58-b986-67e6dc259f01"
version = "0.7.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "65d505fa4c0d7072990d659ef3fc086eb6da8208"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.2"
weakdeps = ["ForwardDiff", "Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

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
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues", "TranscodingStreams"]
git-tree-sha1 = "d97791feefda45729613fafeccc4fbef3f539151"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.5.15"
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

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "411eccfe8aba0814ffa0fdf4860913ed09c34975"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.3"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

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

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "277779adfedf4a30d66b64edc75dc6bb6d52a16e"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.6"

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
git-tree-sha1 = "83c617e9e9b02306a7acab79e05ec10253db7c87"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.38"

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

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "ce8614210409eaa54ed5968f4b50aa96da7ae543"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.4.4"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8e76807afb59ebb833e9b131ebf1a8c006510f33"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.38+0"

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
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

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

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "e24491cb83551e44a69b9106c50666dea9d953ab"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.4.4"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "1d2dd9b186742b0f317f2530ddcbf00eebb18e96"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.7"

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
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "a772d8d1987433538a5c226f79393324b55f7846"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.8"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "9341048b9f723f2ae2a72a5269ac2f15f80534dc"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.3.2+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "e214f2a20bdd64c04cd3e4ff62d3c9be7e969a59"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.4+0"

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
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

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

[[deps.NPZ]]
deps = ["FileIO", "ZipFile"]
git-tree-sha1 = "60a8e272fe0c5079363b28b0953831e2dd7b7e6f"
uuid = "15e1cf62-19b3-5cfa-8e77-841668bca605"
version = "0.4.3"

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

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "ec764453819f802fc1e144bfe750c454181bd66d"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.8+0"

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

[[deps.PeriodicTable]]
deps = ["Base64", "Unitful"]
git-tree-sha1 = "238aa6298007565529f911b734e18addd56985e1"
uuid = "7b2266bf-644c-5ea3-82d8-af4bbd25a884"
version = "1.2.1"

[[deps.Pickle]]
deps = ["BFloat16s", "DataStructures", "InternedStrings", "Mmap", "Serialization", "SparseArrays", "StridedViews", "StringEncodings", "ZipFile"]
git-tree-sha1 = "b10600c3a4094c9a35a81c4d109ad5da8a99875f"
uuid = "fbb45041-c46e-462f-888f-7c521cafbc2c"
version = "0.3.6"

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

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "bfe839e9668f0c58367fb62d8757315c0eac8777"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.20"

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
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "6b8e2f0bae3f678811678065c09571c1619da219"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.1.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

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

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "b7e5b731326a99431517b0b4c1f3902e842103a2"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.12.0"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

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

[[deps.StridedViews]]
deps = ["LinearAlgebra", "PackageExtensionCompat"]
git-tree-sha1 = "425158c52aa58d42593be6861befadf8b2541e9b"
uuid = "4db3bf67-4bd7-4b4e-b153-31dc3fb37143"
version = "0.4.1"

    [deps.StridedViews.extensions]
    StridedViewsCUDAExt = "CUDA"
    StridedViewsPtrArraysExt = "PtrArrays"

    [deps.StridedViews.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    PtrArrays = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "b765e46ba27ecf6b44faf70df40c57aa3a547dcb"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.7"

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
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

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

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "6258d453843c466d84c17a58732dda5deeb8d3af"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.24.0"
weakdeps = ["ConstructionBase", "ForwardDiff", "InverseFunctions", "Printf"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    PrintfExt = "Printf"

[[deps.UnitfulAtomic]]
deps = ["Unitful"]
git-tree-sha1 = "903be579194534af1c4b4778d1ace676ca042238"
uuid = "a7773ee8-282e-5fa2-be4e-bd808c38a91a"
version = "1.0.0"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "af305cc62419f9bd61b6644d19170a4d258c7967"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.7.0"

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

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

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

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

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

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "4909eb8f1cbf6bd4b1c30dd18b2ead9019ef2fad"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.18.1+0"

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

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "f492b7fe1698e623024e873244f10d89c95c340a"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.1"

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
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "8462a20f0fd85b4ef4a1b7310d33e7475d2bb14f"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.77"

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
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1aa23f01927b2dac46db77a56b31088feee0a491"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.4+0"

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
# ╟─7b4502d1-3847-4472-b706-65cb68413927
# ╟─3315ef07-17df-4fe1-a7a5-3e8f039c0cc1
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
# ╟─137441e2-2f6c-43bb-b1a2-b240ee3ab6e7
# ╟─246b8623-2f47-4ac1-b562-75d50889e6d8
# ╟─b146a786-ee54-45d8-ade1-474af127451f
# ╟─6ba6bed9-d913-4511-9829-787fe8a09fa7
# ╟─a078a211-e390-4158-9076-d154921bf8b4
# ╟─22b15086-cde5-4e07-90e2-0cfd82c34889
# ╟─8076e8af-35b9-4113-ba89-c3e5bb25812f
# ╟─862eeec0-bb67-451e-b2f9-bc633075d154
# ╟─4ac43cab-40fc-47b3-bbfa-13b98bcb0a47
# ╟─324ea2b8-c350-438d-8c8f-6404045fc19f
# ╟─206c82ee-389e-4c92-adbf-9578f7125418
# ╟─01d6f011-da4f-4f6e-a75a-53123bfeb9fd
# ╟─d9a62218-ac20-4fab-9786-a7afdcb48945
# ╟─0ca92827-2538-405c-a9e1-2ac53bd70453
# ╟─1a062043-8cfe-4946-8eb0-688d3896229e
# ╟─88093782-d180-4422-bd92-357c716bfc89
# ╟─91a6f3de-01c6-4215-9928-1ecce113adc1
# ╟─f298ffd1-b1dd-472f-b761-c5e07a9121c0
# ╟─2459a2d9-4f48-46ab-82e5-3968e713f15f
# ╟─32eb5c4f-8fde-44c1-a748-905de6aaf364
# ╟─79a75f04-3aa9-4a8b-9f88-1fe8c123680b
# ╟─16d84881-d40a-4694-9812-6896b336accc
# ╟─dd0a4446-65ed-4b3d-87a6-ba950457aa72
# ╟─72697e34-6f63-4d6b-a9ff-ae23a00d4ce2
# ╟─dfe4234a-d2a3-4b59-9578-64bd6e8a5c33
# ╟─593eb726-47a5-4783-8872-03a75c6cfb89
# ╟─a24b03a3-b2a0-4b12-b2a8-25c8cf7f843c
# ╟─78dafb40-f1c7-47e4-b043-170f98588124
# ╟─89ebe16c-845a-4123-bb74-68654327aa05
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
# ╟─fb52e9ee-b685-4d8f-8bab-b5328f300472
# ╟─96b18c60-87c6-4c09-9882-fbbc5a53f046
# ╟─b6d6471a-e5dc-44cd-8c3b-24b150e1ccad
# ╠═862b33ee-e1b2-4261-be22-47fb7122a3ac
# ╟─35d5f991-5194-4174-badc-324ad7d15ac3
# ╟─a50cc950-45ca-4745-9f47-a3fbb28db782
# ╟─1421c748-083b-4977-8a88-6a39c9b9906d
# ╟─f693814d-9639-4abb-a5b4-e83fe7a28a77
# ╟─67f7449f-19b8-4607-9e32-0f8a16a806c0
# ╟─fb0361a6-c967-4dcd-9002-55ae25e225a5
# ╟─5c0eaab3-de6d-457f-a0c3-8ea6b5da2c88
# ╟─de93ac5e-bec6-4764-ac6d-f84076ff20ee
# ╟─82baa991-6df5-4561-9643-52db96c5e99b
# ╟─9a02f4f8-38d6-44a4-9118-1440bfc4d271
# ╟─0f692780-e79c-4aec-a0f4-29c4efbe5390
# ╟─fed8b7b2-34c4-445c-bf11-21b5161c8d28
# ╟─cb9cc32f-a12b-43ec-83c8-3ed26a96992f
# ╠═57cce3be-b168-425a-bd36-79b9ed2327ea
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
# ╟─aaaadaa8-d2e9-4647-82de-5547b2d6ddb4
# ╟─7ed0de94-b513-40c1-9f83-6ed75fcd4cdd
# ╟─188e9981-9597-483f-b1e3-689e26389b61
# ╟─ec78bc4f-884b-4525-8d1b-138b37274ee7
# ╟─e88db991-bf54-4548-80cb-7cd307300ec9
# ╟─99c3f5ee-63d0-4f6f-90d9-2e524a7e945a
# ╟─b64f9f8c-6bf7-44d1-a9a2-c51d35f39491
# ╟─3583f790-bcc3-466b-95e8-e99a080e5658
# ╟─06a80959-58de-4e21-bfdf-5b06caf157f1
# ╟─1aa0cc79-e9ca-44bc-b9ab-711eed853f00
# ╟─b7de7efa-43ef-485c-af2f-8868c7665cc1
# ╠═1b8abb8b-f53e-4bde-b649-1557b8e7906c
# ╟─bd5e2a40-6258-4ecd-a926-6d408cb02387
# ╟─90d0b85d-80d9-421b-a22b-fde00b2a9180
# ╟─b4b2ff74-758d-4b7f-84b7-9c5565c39161
# ╠═2179b785-7869-4a60-9875-14bd59af965f
# ╟─9ad301a5-62b2-4dae-97e2-400d4af6831e
# ╟─83369d85-0f0c-4ffe-b1a5-0bacb330f7f7
# ╟─b0133033-09bf-4b45-94a6-ec9d172133f8
# ╟─db594b4a-d22a-402a-a9df-6ca7a1f3cfa9
# ╟─93e7a7e8-c8b4-47f1-a410-e87e8434b98d
# ╟─53edc301-e9a7-4200-bb69-2159e0b3a23d
# ╠═717afb21-baa7-4381-9e07-00829df1ac59
# ╟─1a5a22dd-89bf-483b-b7f2-e477512f84ac
# ╠═3bc95495-fccc-4aac-aa13-554ef999e770
# ╟─ac842d64-8127-4b09-94eb-839f68692e5b
# ╟─77636d53-5426-4a52-9e61-430119e881c6
# ╟─fc11e0a8-e658-4e73-b4a7-569be7927014
# ╟─0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
# ╟─82d513f2-be95-4c3a-8eb4-18c48a7d0b44
# ╟─3a083374-afd6-4e64-95bd-d7e6385ab403
# ╟─6cd96390-8565-499c-a788-fd0070331f25
# ╟─8fe5631a-1f10-4af4-990d-5a23c96fb73b
# ╟─619df17d-9f75-463f-afe5-6cbffb0762d5
# ╟─8687dbd1-4857-40e4-b9cb-af469b8563e2
# ╟─fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
