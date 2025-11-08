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

# ╔═╡ f1ecdbf7-cc9d-4d4d-a57b-d3c1bd1d2628
begin
	# using LinearAlgebra
	using PlutoUI
	using PlutoTeachingTools
	using LaTeXStrings
	using Latexify
	# using Random
	# using Statistics
	using HypertextLiteral
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
end

# ╔═╡ df4d7b58-3e52-428c-9fe6-d3138793e9e6
using LinearAlgebra

# ╔═╡ 436ecc71-fc90-4c73-a27f-29d800bcd189
using StatsBase:mean

# ╔═╡ e0b5bd62-cb0d-4e65-bdaa-f6d1963e89bd
using LogExpFunctions

# ╔═╡ a54a36f8-ceb2-4c82-8ee6-8f6b140fd6a4
using Zygote

# ╔═╡ ce9fb355-5483-4850-a06d-938c08ca8c10
begin
	using Random
	Random.seed!(123)
	nobs = 500
	Xs = rand(nobs, 2) .* 6 .- 3
	Ys = rand(nobs) .< exp.(-(sum(Xs, dims=2)/2).^2)[:]
end;

# ╔═╡ 3b27bab2-4548-406c-9ee3-bd9e5e926b17
TableOfContents()

# ╔═╡ b62f2103-fc79-4bdf-8a21-5238c132ec5e
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ cf92f290-12ad-4a78-9f90-85b755ddcd31
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

# ╔═╡ 9ed57ba6-c18d-436f-9d1d-6f7a4798d87f
ChooseDisplayMode()

# ╔═╡ bde3675d-e349-4498-83e8-2fa90511f568
md"""

# CS5014 Machine Learning


#### Tutorial 2 solution
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 12863468-865b-4aae-8349-884698a78ed4
md"""


## Question 1


###### Consider a ``\mathbb{R}^2 \rightarrow \mathbb{R}`` function ``h(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + w_0``, and ``\mathbf{w} = [1, 2]^\top`` is known, assume we are at ``\mathbf{x}_0 = [2,3]^\top``; 
* which direction to follow if we want to increase the function the most ?
* which direction to follow if we want to decrease the function the most ?
* provide a direction such that the function stays the same 
* are the descent/ascent directions unique?
* is ``w_0`` relevant at all (when it comes to answering the above questions)?


"""

# ╔═╡ 3b85a854-400c-464b-bbe3-e8c4b43e009a
md"Add directions: $(@bind add_directions CheckBox(false))"

# ╔═╡ 88627363-e62f-48c1-886c-aa741fb3754f
md"``\color{red}\mathbf{u}`` (red): $(@bind utheta Slider(range(-π, π, 100), default=0)); Make the move along ``\color{red}\mathbf{u}``: $(@bind add_xnew CheckBox(false))"

# ╔═╡ c381ba06-2e70-4223-924a-4c1946fa632a
md"Add gradient ``\nabla h(\mathbf{x})=\mathbf{w}`` (green): $(@bind add_grad_vec CheckBox(false))"

# ╔═╡ 3a39807b-5625-4a1b-8ac6-b81588e74a7f
function perp_square(origin, vx, vy; δ=0.1) 
	x = δ * vx/sqrt(norm(vx))
	y = δ * vy/sqrt(norm(vy))
	xyunit = origin+ x + y
	xunit = origin + x
	yunit = origin +y
	Shape([origin[1], xunit[1], xyunit[1], yunit[1]], [origin[2], xunit[2], xyunit[2], yunit[2]])
end

# ╔═╡ 9971851e-ddae-4096-92ef-373bfe9ba5a5
let
	gr()
	xlims = [0, 6]
	ylims = [0, 6]
	w = [1, 2]
	x0 = [2, 3.0]
	plt = plot(range(xlims..., 20), range(ylims..., 20), (x1, x2) -> dot([x1, x2], w) + 2, st=:contourf, c=:coolwarm,  colorbar=false, title=L"h(\mathbf{x})",  ratio=1, framestyle=:origins, xlim =xlims, ylim =xlims, xlabel=L"w_1", ylabel=L"w_1", alpha=0.3, size=(350,350), titlefontsize=12)


	# x0 = [0.13, 0.13]

	# yhats = logistic.(D₂ * [0, w0[1], w0[2]])
	# ∇li = (D₂ .* (yhats - targets_D₂))[:, 2:end] #ignore intercept w₀
	θs = range(0, 2π, 12)
	∇l = w
	levelv = [1, -∇l[1]/∇l[2]]
	levelv = (levelv ./ norm(levelv))
	ts = -1:0.1:1
	levelset = x0 .+ ((ts)' .* levelv)
	r = 1.0
	gx_new = r * ∇l + x0

	plot!([x0[1], gx_new[1]], [x0[2], gx_new[2]], line = (:arrow, 1.0, :blue), label="")
	
	
	plot!(levelset[1,:], levelset[2,:], lw=2, lc=:purple, label="", st=:path)
	# # plot!(range(w0[1]-0.5, w0[1]+0.5, 5), x-> , lw=1, lc=:gray, label="")
	plot!(perp_square(x0, levelv, ∇l ; δ=0.3), lw=1, lc=:purple, label="", fillcolor=false)
	
	∇li = [r * cos.(θs) r* sin.(θs)]

	sgx_new = r * ∇li .+ x0'

	# first_neg = true
	# # first_pos = true
	
	for x in eachrow(sgx_new)
		if (gx_new - x0)' * (x - x0) < 0 
			plot!([x0[1], x[1]], [x0[2], x[2]], line = (:arrow, 0.3, :blue), label="")
	
		else
			plot!([x0[1], x[1]], [x0[2], x[2]], line = (:arrow, 0.3, :red), label="")
			
		end
	end

	plt 
end

# ╔═╡ b38c1389-4c16-4e42-be66-7700910f1ed9
md"""

## Solution

#### Short answer: _gradient direction points to the greatest ascent direction_


* and the gradient of $h(\mathbf{x})$ is 

$$\nabla_{\mathbf{x}} h = \mathbf{w}$$
* therefore, the direction to increase the function the most is the gradient direction at $\mathbf{x}_0$ is $\mathbf{w} = [1, 2]^\top$ and it is a constant direction: the greatest ascent direction doesn't change w.r.t where we are;

* the direction to decrease the function the most is the negative/opposite gradient direction, *i.e.* $-\nabla_{\mathbf{x}} h = - \mathbf{w} = [-1, -2]^\top$


#### Long answer
Assume we are at $\mathbf{x}_0$, and denote the direction to follow from $\mathbf{x}_0$ as $\mathbf{u}\in \mathbb{R}^2$ and $\|\mathbf{u}\|_2 = 1$. The function value after taking the direction $\mathbf{u}$ from $\mathbf{x}_0$ is 

$$h(\mathbf{x}_0 + \mathbf{u}) = \mathbf{w}^\top (\mathbf{x}_0 + \mathbf{u}) +w_0 = \mathbf{w}^\top \mathbf{x}_0 + \mathbf{w}^\top\mathbf{u} + w_0,$$

where $\mathbf{w}, w_0, \mathbf{x}_0$ are all given therefore constant. The above function becomes a function of $\mathbf{u}$ only, 

$$h(\mathbf{x}_0 + \mathbf{u}) =\mathbf{w}^\top\mathbf{u} + C,$$

where $C= w_0 + \mathbf{w}^\top \mathbf{x}_0.$ 

Recall the definition of inner product, 

$$h(\mathbf{x}_0 + \mathbf{u}) =\mathbf{w}^\top\mathbf{u} + C= \|\mathbf{w}\|\cdot \|\mathbf{u}\| \cos(\theta_{\mathbf{u}, \mathbf{w}})+C,$$

where $\theta_{\mathbf{u}, \mathbf{w}}$ denotes the angle between the two vectors. $\cos(\theta)$ takes its maximum, $1$ ,when $\theta=0$, *i.e.* when the two vectors point to the same direction; and its minimum, $-1$, when $\theta=\pi$ (or 180 degree).


Therefore, to maximise $h$, $\mathbf{u}$ has to point to the same direction as $\mathbf{w}$, therefore the greatest ascent direction is $$\mathbf{w}$$; to decrease the function the most, the direction has to be the opposite direction of $\mathbf{w}$: $-\mathbf{w}$.




"""

# ╔═╡ 2895ef81-b716-4c84-ba8c-cdf48ad52447
md"""

#### level direction

When $\mathbf{u}$ is orthogonal to $\mathbf{w}$, $\mathbf{w} \perp \mathbf{u}$, then $\cos(\theta_{\mathbf{u}, \mathbf{w}})=0$, therefore, 

$$h(\mathbf{x}_0 + \mathbf{u}) =\mathbf{w}^\top\mathbf{u} + C=C,$$

such a $\mathbf{u}$ does not increase nor decrease the function. And it is called level set direction (it proves why gradient is always perpendicular to the level curve). There are infinite number of directions that are perpendicular/orthogonal to $\mathbf{w}$, or $\mathbf{w}^\top\mathbf{u} = 0$. But we can find one of them by fixing $u_1=1$ or any other number, 

$$\begin{align}\mathbf{w}^\top\mathbf{u} &=w_1u_1 + w_2 u_2= 0\\
\Rightarrow & w_1 + w_2 u_2 = 0\\
\Rightarrow & u_2 = -\frac{w_1}{w_2} = -\frac{1}{2},
\end{align}$$

Therefore, a level set direction is $\mathbf{u} =[1, -\frac{1}{2}]^\top$
"""

# ╔═╡ cf24ff17-5d3f-45ef-90c7-3903aa3478bc
md"""

#### The ascent/descent directions are not unique

Ascent directions are those with $\mathbf{w}^\top\mathbf{u}= \|\mathbf{w}\|\cdot \|\mathbf{u}\| \cos(\theta_{\mathbf{u}, \mathbf{w}})>0$. Note that $\|\mathbf{w}\|\cdot \|\mathbf{u}\| >0$, therefore $\cos(\theta_{\mathbf{u}, \mathbf{w}})>0 \Leftrightarrow h(\mathbf{x}_0+\mathbf{u}) > h(\mathbf{x}_0)$ ; it happens if and only if the angle is acute or $\theta_{\mathbf{u}, \mathbf{w}} < \frac{\pi}{2}$; 

Descent directions are those with $\mathbf{w}^\top\mathbf{u}= \|\mathbf{w}\|\cdot \|\mathbf{u}\| \cos(\theta_{\mathbf{u}, \mathbf{w}})<0$, then $h(\mathbf{x}_0+\mathbf{u}) < h(\mathbf{x}_0)$; it happens if and only if the angle is obtuse or $\theta_{\mathbf{u}, \mathbf{w}} > \frac{\pi}{2}$; 
"""

# ╔═╡ 88994483-36ab-46be-87f1-5cf10b9fe26e
md"""

### $w_0$ is irrelevent

Since the gradient does not change w.r.t $w_0$; it only lifts/lower the hyperplane: the greatest ascent/descent directions do not change.

But for a non-linear transformation of hyperplanes, $w_0$ matters. For example, $g(\mathbf{x}; \mathbf{w}, w_0) = e^{-\mathbf{w}^\top\mathbf{x}-w_0}$, its gradient is 

$$\nabla_\mathbf{x} \sigma = -e^{-\mathbf{w}^\top\mathbf{x}-w_0}\cdot  \mathbf{w},$$

(you should derive the gradient by yourself to check whether it is correct!) The gradient changes with different $w_0$ values (or is not constant w.r.t $w_0$). 

"""

# ╔═╡ f03b4f96-a10a-4623-a0c0-9d1a8de0ba81
md"""


## Question 2
###### Consider logistic regression's prediction function
```math
\sigma(\mathbf{x}) = \frac{1}{1+e^{-h(\mathbf{x}; \mathbf{w}, w_0)}}
```

###### where $h(\mathbf{x};\mathbf{w}, w_0)= \mathbf{w}^\top\mathbf{x} +w_0$ is a hyperplane;

* find an expression for the binary classification's decision boundary and show that the boundary is linear (hint: the decision boundary is defined as *i.e.* ``\sigma(\mathbf{x}) =0.5``)


* show that the decision boundary does not change when we scale the parameters:  ``\mathbf{w}\leftarrow k{\mathbf{w}}`` and ``w_0 \leftarrow k {\mathbf{w}}_0`` when ``k\neq 0``; what is the effect of setting ``k=0``?

###### Explain the effect of the following changes. Does the decision boundary change? How the predictions change?

* set ``{w}_0 \leftarrow 0``, ``w_0 \leftarrow w_0 + 5``, ``w_0 \leftarrow w_0 -5``

* set ``\mathbf{w} \leftarrow -\mathbf{w}, w_0 \leftarrow - w_0``

* set ``\mathbf{w}  \leftarrow -\mathbf{w}``
"""

# ╔═╡ aa7dcff1-b1b5-4b87-b1d6-0f0d79ae3d3b
md"""

### Solution
The decision boundary is  defined as 
$$\sigma(\mathbf{x}) = \frac{1}{1+e^{-h(\mathbf{x}; \mathbf{w}, w_0)}} =0.5,$$ which implies


$$\frac{1}{1+e^{-h(\mathbf{x}; \mathbf{w}, w_0)}} =0.5 \Longleftrightarrow e^{-h(\mathbf{x}; \mathbf{w}, w_0)} = 1.0 \Longleftrightarrow h(\mathbf{x}; \mathbf{w}, w_0) = 0$$

which implies

$$\mathbf{w}^\top\mathbf{x} + w_0 = 0.$$

If we scale both $\mathbf{w}, w_0$ by a constant $k\neq 0$, the decision boundary becomes

$$(k\mathbf{w})^\top\mathbf{x} + k\cdot w_0 = 0 \Longleftrightarrow k\cdot \mathbf{w}^\top\mathbf{x} + k\cdot w_0 = 0$$

dividing $k\neq 0$ both sides, the decision boundary remain the same

$$(k\mathbf{w})^\top\mathbf{x} + k\cdot w_0 = 0 \Longleftrightarrow \mathbf{w}^\top\mathbf{x} + w_0 = 0$$

Although the decision boundary does not change, the logistic function approaches to a step function when $k\rightarrow \infty$. That's why we need to usually add regularisation to logistic regression models.

If we set $k=0$, we have $h(\mathbf{x}) = - 0\mathbf{w}^\top\mathbf{x} - 0w_0 =0$ everywhere, which implies $\sigma(\mathbf{x}) = \frac{1}{1+e^0}= 0.5$ everywhere.

"""

# ╔═╡ bdb33912-b145-4ea9-98c4-32a1b52efa24
md"""``k``=$(@bind kk Slider(0.01:0.05:5, default=1))"""

# ╔═╡ 566628e5-862d-49f1-a896-603251d94b1a
w0 = 1 * kk;

# ╔═╡ b8147db6-6f7f-4d1d-bfa6-757245e52fe8
wv_ = [1, 1] * kk;

# ╔═╡ 3da8e967-7b62-4130-b9b6-3d919894fdfe
md"""
#### change $w_0$

* set ``{w}_0 \leftarrow 0``, ``w_0 \leftarrow w_0 + 5``, ``w_0 \leftarrow w_0 -5``


Changing the bias $w_0$ term alone lifts the hyperplane; therefore it also changes where the plane intersects with the $xy$-plane, which means the decision boundary also will change accordingly.

"""

# ╔═╡ 0a4b683c-1f83-4e24-9f6f-3919d9b63490
@bind w0_ Slider(-10:10, default=0)

# ╔═╡ 2c6bd19a-b732-4030-8ad2-615ef3980362
# let
# 	plotly()
# 	w₀ = w0
# 	w₁, w₂ = wv_[1], wv_[2]
# 	p1 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> w₀+ w₁* x + w₂ * y, st=:surface, c=:jet, colorbar=false, alpha=0.8, framestyle=:zerolines, xlabel="x₁", ylabel="x₂", title="h(x) = wᵀ x")

# 	plot!(-5:0.5:5, -5:0.5:5, (x, y) -> 0, st=:surface, c=:gray, alpha=0.5)
# 	arrow3d!([0], [0], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
# 	x0s = -5:0.5:5
# 	if w₂ ==0
# 		x0s = range(-w₀/w₁-eps(1.0) , -w₀/w₁+eps(1.0), 20)
# 		y0s = range(-5, 5, 20)
# 	else
# 		y0s = (- w₁ * x0s .- w₀) ./ w₂
# 	end
# 	plot!(x0s, y0s, zeros(length(x0s)), lc=:gray, lw=4, label="")
	
# 	p2 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.8, zlim=[-0.1, 1.1],  xlabel="x₁", ylabel="x₂", title="σ(wᵀx)", framestyle=:zerolines)
# 	plot!(-5:0.5:5, -5:0.5:5, (x, y) -> 0.5, st=:surface, c=:gray, alpha=0.75)
# 	arrow3d!([0], [0], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
# 	plot!(x0s, y0s, .5 * ones(length(x0s)), lc=:gray, lw=4, label="")
# 	plot(p1, p2)
# end

# ╔═╡ 9ef7deeb-deab-41b8-89c7-609f5b9a387f
# TwoColumn(let
# 	gr()
# 	w₀ = 0
# 	w₁, w₂ = wv_[1], wv_[2]
# 	plot(-5:0.5:5, -5:0.5:5, (x, y) -> (w₀+ w₁* x + w₂ * y), st=:contourf, c=:jet, colorbar=false, alpha=0.25, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="contour and gradient: "* L"h(x) =w^\top x", ratio=1, size=(350,360), titlefontsize=12)
# 	α = 0.5
# 	xs_, ys_ = meshgrid(range(-5, 5, length=10), range(-5, 5, length=10))
# 	∇f_d(x, y) = ∇h([1, x, y], [w₀, w₁, w₂])[2:end] * α
# 	quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
# end, let
# 	gr()
# 	w₀ = 0
# 	w₁, w₂ = wv_[1], wv_[2]
# 	plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:contourf, c=:jet, colorbar=false, alpha=0.25, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="contour and gradient: "* L"σ(w^\top x)", ratio=1, size=(350,360), titlefontsize=12)
# 	α = 2
# 	xs_, ys_ = meshgrid(range(-5, 5, length=15), range(-5, 5, length=15))
# 	∇f_d(x, y) = ∇σ([1, x, y], [w₀, w₁, w₂])[2:end] * α
# 	quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
# end
# )

# ╔═╡ a76c5b02-c0f4-4128-8ef8-0441b856d03a
begin
	∇σ(x, w) = logistic(dot(w, x)) * (1- logistic(dot(w,x))) * w
	∇h(x, w) = w
end

# ╔═╡ 2e5fb5b8-9526-4f6b-91f1-a4ef6bf2a2e7
# TwoColumn(let
# 	gr()
# 	w₀ = 0
# 	wv_ = [1, 2]
# 	w₁, w₂ = wv_[1], wv_[2]
# 	plot(-5:0.5:5, -5:0.5:5, (x, y) -> (w₀+ w₁* x + w₂ * y), st=:contourf, c=:coolwarm, colorbar=false, alpha=0.25, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="contour and gradient: "* L"h(x) =w^\top x", ratio=1, size=(350,360), titlefontsize=12)
# 	α = 0.3
# 	xs_, ys_ = meshgrid(range(-5, 5, length=10), range(-5, 5, length=10))
# 	# ∇h([1])
# 	∇f_d(x, y) = wv_ * α
# 	quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
# end, let
# 	gr()
# 	w₀ = 0
# 	wv_ = [1, 2]
# 	w₁, w₂ = wv_[1], wv_[2]
# 	# logistic(z?)
# 	plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:contourf, c=:coolwarm, colorbar=false, alpha=0.25, xlim=[-5, 5], ylim=[-5, 5],  xlabel=L"x_1", ylabel=L"x_2", title="contour and gradient: "* L"σ(w^\top x)", ratio=1, size=(350,360), titlefontsize=12)
# 	α = 1
# 	xs_, ys_ = meshgrid(range(-5, 5, length=15), range(-5, 5, length=15))
# 	∇f_d(x, y) = ∇σ([1, x, y], [w₀, w₁, w₂])[2:end] * α
# 	quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
# end
# )

# ╔═╡ 12d54350-c152-443a-a780-f08946210523
md"""

## Question 3


##### Sketch the following functions 
* ``h(x) = \exp(-x^2)`` (to help you get started, the function is plotted below; but you need to understand why before moving on!)
* ``h(x) = \exp(- (wx +w_0)^2)`` (hint: where the linear function intersects with zero?)

* ``h(\mathbf{x}) = \exp(- (\mathbf{w}^\top \mathbf{x} + w_0)^2)``; where ``\mathbf{x}, \mathbf{w} \in\mathbb{R}^n``  (hint: where the hyperplane intersects with the zero plane ?)
"""

# ╔═╡ c4b24fc6-a94f-4a86-823e-a062c9344218
plot(x -> exp(-x^2), label="", title=L"h(x) = \exp(-x^2)", lw=2, framestyle=:zerolines, size=(350,200))

# ╔═╡ eaacfdb5-e34a-4d50-bfae-c75208906180
md"""
## Solution

"""

# ╔═╡ 5a64e355-9682-4bd8-900b-4da1ada119ee
@bind id Slider(1:4)

# ╔═╡ c27db17d-b2f0-485a-b87b-a35e85dfbaac
plts1 = let
	plts = []
	plt1 = plot(x ->x,  lw=2, framestyle=:origin, ylim =(-1,1.3), label=L"x",legendfontsize=15)

	# for f in [x -> x^2, x -> -x^2, ]
	plts = [plt1]
	plt = deepcopy(plts[end])
	plot!(plt, x-> x^2, label=L"x^2", lw=2, legendfontsize=15)

	push!(plts, plt)
	plt = deepcopy(plts[end])
	plot!(plt, x-> -x^2, label=L"-x^2", lw=2, legendfontsize=15)

	push!(plts, plt)
	plt = deepcopy(plts[end])

	plot!(plt, x -> exp(-x^2), label="", title=L"h(x) = \exp(-x^2)", lw=2, legendfontsize=15)

	push!(plts, plt)
end;

# ╔═╡ 930ad37f-7802-46a7-9a65-c934196b8816
begin
	plts1[id]
end

# ╔═╡ 8360e6f6-4417-4fbe-8513-d69b85d02e85
@bind id2 Slider(1:4)

# ╔═╡ 26080a33-c7db-44ef-98ee-f2b8e443d6ad
plts2 = let
	plts = []

	k = 2
	b = -2
	plt1 = plot(-3:0.1:3, x ->k * x + b,  lw=2, framestyle=:origin, ylim =(-1,1.3), label=L"%$(k)x %$(b)", legendfontsize=15)

	# for f in [x -> x^2, x -> -x^2, ]
	plts = [plt1]
	plt = deepcopy(plts[end])
	plot!(plt, -3:0.01:3,x-> (k * x +b)^2, lw=2, label=L"(%$(k)x %$(b))^2", legendfontsize=15)

	push!(plts, plt)
	plt = deepcopy(plts[end])
	plot!(plt, -3:0.01:3,x->  -(k * x +b)^2, label=L"-(%$(k)x %$(b))^2", lw=2, legendfontsize=15)

	push!(plts, plt)
	plt = deepcopy(plts[end])

	plot!(plt, -3:0.01:3, x -> exp(-(k * x +b)^2), label=L"\exp\{-(%$(k)x %$(b))^2\}", title=L"h(x) = \exp(-(wx+w_0)^2)", lw=2,legendfontsize=15)


	push!(plts, plt)
end;

# ╔═╡ 1284949e-4d1a-4559-97cd-d58287816229
begin
	plts2[id2]
end

# ╔═╡ 0d862000-fb6a-4805-8bf4-5ea8e488517d
ww = [1, 1] * 0.25

# ╔═╡ 3aded33d-b29b-47bf-8403-12b7195511eb
md"Add lines: $(@bind add_p_lines CheckBox(default=false)), show transformed: $(@bind show_sigmoid CheckBox(false))"

# ╔═╡ 9de1c4ad-68da-4353-b237-8fe26cd4e48f
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
		ts = -18/norm(wv):2:18/norm(wv)
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
		ff(x) = exp(- f(x[1], x[2])^2)
		# ff(x1, x2) = exp(- )
		plt = plot(-10:0.1:10, -10:0.1:10, (x1, x2) -> ff([x1, x2]), st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.8, framestyle=:zerolines, c=:coolwarm, colorbar=false, camera=(-5, 30), size=(350,400))
		if add_p_lines
			x00 = [x_center..., f(x_center...)]
			wv = [ww..., ww'*ww]
			ts = -14/norm(wv):1:14/norm(wv)
			vv = ww[2] == 0 ? [0, 1] : [1, -ww[1]/ww[2]]
			vv = vv/norm(vv)
			vvs = x00[1:2] .+ vv * range(-15, 15, 15)'
			vvs =[vvs; f(x_center...) .+ zeros(size(vvs)[2])']
	
			for v0 in eachcol(vvs)
				xyzs = v0 .+ wv .* ts'
				newzs = [exp(- f(xy[1:2]...)^2) for xy in eachcol(xyzs)]
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

# ╔═╡ 7216b051-c9fd-47ee-9892-72cf4f862f92
md"""

##### Consider the following binary classification problem


* ###### the data you are given is no longer linearly separable, but "banded", and the data is noisy

* ###### provide a probabilistic model for the problem, *i.e.* define a suitable ``p(y^{(i)}|\mathbf{x}^{(i)})`` (hint: check first question above, you will need the results from the question above)
"""

# ╔═╡ 1e483e31-c1c1-4ffe-900d-3b419d8b0122
TwoColumn(show_img("banded.png", w=300), show_img("banded2.png", w=300))

# ╔═╡ faf13435-546f-4b7f-aeee-f54adafa78d3
md"
> Disclaimer*: the question is adapted/copied from a past Machine Learning exam question at Cambridge. The link to the original question is  [https://www.cl.cam.ac.uk/teaching/exams/pastpapers/y2021p8q10.pdf](https://www.cl.cam.ac.uk/teaching/exams/pastpapers/y2021p8q10.pdf))
"

# ╔═╡ fd398edb-ca31-444b-9477-bc6389c84c9f
md"""

### Solution

$$p(y^{(i)}|\mathbf{x}^{(i)}, \mathbf{w}) = \begin{cases} \exp(- (\mathbf{w}^\top \mathbf{x} + w_0)^2) & y^{(i)} =1 \\

1- \exp(- (\mathbf{w}^\top \mathbf{x} + w_0)^2) & y^{(i)} = 0
\end{cases}$$

Just Bernoulli model with bias equals to $h(\mathbf{x})$.

Then the loss is just the negative log likelihood becomes 

$$\ell(\mathbf{w}) = - \frac{1}{n}\sum_{i=1}^n\ln p(y^{(i)}|\mathbf{x}^{(i)})$$

$$\ell^{(i)} (\mathbf{w})= \begin{cases}-\ln \hat{y}^{(i)}  & y^{(i)} = 1\\ - \ln (1-\hat{y}^{(i)}) & y^{(i)}=0 \end{cases}$$

"""

# ╔═╡ 5ef4cd0d-840d-4799-8691-90b54046a07a
md"""
##### Outline and implement a learning algorithm to train your model 

The following simulated dataset (of ``n=500`` training instances) should be used. And the data (stored in variables `Xs, Ys`) is plotted below.


hint: should you need gradients in your training algorithm, you may use `Zygote.jl`
"""

# ╔═╡ c53f1dc7-4498-41d5-a6de-28b574a5b9b8
Xs, Ys

# ╔═╡ e943a212-3780-46c9-b866-2a206c40dbe0
function exp_1msq(z)
	exp(-z^2)
end

# ╔═╡ 9804c882-54f7-4400-8621-7c8fa576342a
losses, w_fit = let
	## implement your algorithm here
	Random.seed!(123)
	w0 = randn(3)
	Xs_ = [ones(size(Xs)[1]) Xs]
	losses = []

	lr = 0.1
	for _ in 1:100
		loss, gw = Zygote.withgradient(w0) do ww
			y_preds = exp_1msq.(Xs_ * ww)
		# Xs_ * w0
	 		loss = sum(-log.(y_preds)[Ys])
			loss += sum(-log.(1 .- y_preds)[.!Ys])
			loss = loss / size(Xs_)[1]
		end
		push!(losses, loss)
		w0 -= lr * gw[1]
	end
	losses, w0 
end

# ╔═╡ 62e0abc8-a998-48c7-958e-9f12700ab009
let
	gr()
	plt1 = scatter(Xs[Ys, 1], Xs[Ys, 2], framestyle=:semi, xlabel=L"x_1", ylabel=L"x_2", label="class 1", ms=3, alpha=0.9)
	scatter!(Xs[.!Ys, 1], Xs[.!Ys, 2], framestyle=:semi, xlabel=L"x_1", ylabel=L"x_2", label="class 0",ratio=1, xlim =(-3,3), ms=3, alpha=0.5)
	plot!(-3:0.1:3, -3:0.1:3, (x, y) -> exp_1msq([1, x, y]' * w_fit), st=:contourf, c=cgrad(:coolwarm, rev=true), alpha=0.6, fillalpha=0.1, xlim =(-3,3), ylim =(-3, 3), title="Prediction contour")

	plt2 = scatter(Xs[Ys, 1], Xs[Ys, 2], framestyle=:semi, xlabel=L"x_1", ylabel=L"x_2", label="class 1", ms=3, alpha=0.9)
	scatter!(Xs[.!Ys, 1], Xs[.!Ys, 2], framestyle=:semi, xlabel=L"x_1", ylabel=L"x_2", label="class 0",ratio=1, xlim =(-3,3), ms=3, alpha=0.5)
	plot!(-3:0.02:3, -3:0.02:3, (x, y) -> exp_1msq([1, x, y]' * w_fit) > 0.5, st=:contour, c=cgrad(:coolwarm, rev=true), alpha=0.9, xlim =(-3,3), ylim =(-3, 3), title="Decision boundary")

	plot(plt1, plt2, size=(700,300))
end

# ╔═╡ 9db15d52-1761-4103-b0f0-bd02e37d5d75
md"""

## Question 4 


"""

# ╔═╡ 3528e5a8-0127-4379-81e2-a997e4c769ca
md"""

##### Now consider an extension of the banded problem (I name it "multiple banded" data) 
(the bandwidths and the noise scales are assumed to be fixed across the input domain)

* ###### define a suitable generative model ``p(y^{(i)}|\mathbf{x}^{(i)})`` (hint: use ``\sin(x)``)

"""

# ╔═╡ 64b94612-a446-4cb4-a10d-5aa2f55d4103
show_img("multibanded.svg", w=500)

# ╔═╡ 52d8483c-d9b4-4401-85e3-1e2a7bd791fd
md"""

## Appendix
"""

# ╔═╡ bc9ab53c-58af-473c-9e05-100248f264aa
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.

# ╔═╡ 7959520f-c599-44e3-9f3e-cac9ce410926
let
	gr()
	w₀ = 0
	wv_ = [1, 2]
	w₁, w₂ = wv_[1], wv_[2]
	plot(0:0.5:6, 0:0.5:6, (x, y) -> (w₀+ w₁* x + w₂ * y), st=:contourf, c=:coolwarm, colorbar=false, alpha=0.25, xlim=[0, 6], ylim=[0, 6],  xlabel=L"x_1", ylabel=L"x_2", title="contour and gradient: "* L"h(x) =w^\top x", ratio=1, size=(350,360), titlefontsize=12)
	α = 0.2
	xs_, ys_ = meshgrid(range(0, 6, length=10), range(0, 6, length=10))
	# ∇h([1])
	∇f_d(x, y) = wv_ * α
	quiver!(xs_, ys_, quiver = ∇f_d, c=:black)
end

# ╔═╡ 76781718-9c80-4a67-a5a2-4abaa5f2d6e5
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

# ╔═╡ c3c3b39f-c421-4d87-b0bb-2b6c7df0ad89
let
	gr()
	w = [1, 2]
	x0 = [2, 3.0]
	b=w
	w₀ = 0
	f(x1, x2) = dot(b, [x1, x2]) + w₀
	xlims = [-1,5]
	ylims = [-1, 5]
	plt = plot(range(xlims..., 10), range(ylims..., 10), (x1, x2) -> dot(b, [x1, x2])+w₀, st=:surface, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"h", alpha=0.8, framestyle=:origin, c=:coolwarm, colorbar=false, camera=(60, 25), xlim = xlims, ylims = ylims)
	min_z, max_z = extrema([f(x1, x2) for x1 in -15:2:15 for x2 in -15:2:15])
	x = range( xlims[1],stop=xlims[2],length=10)
	y = range( ylims[1],stop= ylims[2],length=10)	

	plot!([x0[1], x0[1]], [x0[2], x0[2]], [f(x0...), 0], lw=2, lc=:black, ls=:dash, label="")
	scatter!([x0[1]], [x0[2]], [f(x0...)], ms=2, markershape=:circle, label="", mc=:white, msc=:gray, msw=2, alpha=0.9)
	scatter!([x0[1]], [x0[2]], [0], label=L"\mathbf{x}_0", ms =3, mc=:white, msc=:gray, msw=2, alpha=1.0)
	r = 1
	if add_directions
		for theta ∈ range(-π, π, 12)
			arrow3d!([x0[1]], [x0[2]], [0], [r * cos(theta)], [r * sin(theta)], [0]; as = 0.2, lc=1, la=0.5, lw=1, scale=:identity)
		end
	end
	uu = [r* cos(utheta), r * sin(utheta)]
	dd_ = (uu / norm(uu)) * r
	arrow3d!([x0[1]], [x0[2]], [0], [dd_[1]], [dd_[2]], [0]; as = 0.3, lc=2, la=1, lw=2.5, scale=:identity)

	if add_grad_vec
		# gd= 
		gd = b / norm(b) * r
		arrow3d!([x0[1]], [x0[2]], [0], [gd[1]], [gd[2]], [0]; as = 0.3, lc=3, la=1, lw=2.5, scale=:identity)
	end
	if add_xnew
		x_new = x0 + dd_
		scatter!([x_new[1]], [x_new[2]], [0], label=L"\mathbf{x}_{new}", ms =3, mc=:white, msc=2, msw=2, alpha=1.0)
		scatter!([x_new[1]], [x_new[2]], [f(x_new...)], ms=2, markershape=:circle, label="", mc=2, msc=:gray, msw=2, alpha=0.9)
		plot!([x_new[1], x_new[1]], [x_new[2], x_new[2]], [f(x_new...), 0], lw=2, lc=2, ls=:dash, label="")
	end
	plt
end

# ╔═╡ b4f85271-2983-4bd3-8cc6-1e4e3324db82
let
	gr()
	w = [1, 2]
	x0 = [2, 3.0]
	b=w
	w₀ = 0
	f(x1, x2) = dot(b, [x1, x2]) + w₀
	xlims = [-1,5]
	ylims = [-1, 5]
	plts= []
	anim = @animate for utheta in range(0, 2π, 20)
	plt = plot(range(xlims..., 10), range(ylims..., 10), (x1, x2) -> dot(b, [x1, x2])+w₀, st=:surface, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"h", alpha=0.8, framestyle=:origin, c=:coolwarm, colorbar=false, camera=(60, 25), xlim = xlims, ylims = ylims, size=(500,500))
	min_z, max_z = extrema([f(x1, x2) for x1 in -15:2:15 for x2 in -15:2:15])
	x = range( xlims[1],stop=xlims[2],length=10)
	y = range( ylims[1],stop= ylims[2],length=10)	

	plot!([x0[1], x0[1]], [x0[2], x0[2]], [f(x0...), 0], lw=2, lc=:black, ls=:dash, label="")
	scatter!([x0[1]], [x0[2]], [f(x0...)], ms=2, markershape=:circle, label="", mc=:white, msc=:gray, msw=2, alpha=0.9)
	scatter!([x0[1]], [x0[2]], [0], label=L"\mathbf{x}_0", ms =3, mc=:white, msc=:gray, msw=2, alpha=1.0)
	r = 1

	for theta ∈ range(-π, π, 12)
		arrow3d!([x0[1]], [x0[2]], [0], [r * cos(theta)], [r * sin(theta)], [0]; as = 0.2, lc=1, la=0.5, lw=1, scale=:identity)
	end

	uu = [r* cos(utheta), r * sin(utheta)]
	dd_ = (uu / norm(uu)) * r
	arrow3d!([x0[1]], [x0[2]], [0], [dd_[1]], [dd_[2]], [0]; as = 0.3, lc=2, la=1, lw=2.5, scale=:identity)
	annotate!([x0[1]+dd_[1]], [x0[2]+dd_[2]], [0], text(L"\mathbf{u}", 10, :red, :top))

		# gd= 
		gd = b / norm(b) * r
		arrow3d!([x0[1]], [x0[2]], [0], [gd[1]], [gd[2]], [0]; as = 0.3, lc=3, la=1, lw=2.5, scale=:identity)

		annotate!([x0[1]+gd[1]], [x0[2]+gd[2]], [0], text(L"\nabla h", 10, :green, :left))

		x_new = x0 + dd_
		scatter!([x_new[1]], [x_new[2]], [0], label=L"\mathbf{x}_{new}", ms =3, mc=:white, msc=2, msw=2, alpha=1.0)
		scatter!([x_new[1]], [x_new[2]], [f(x_new...)], ms=2, markershape=:circle, label="", mc=2, msc=:gray, msw=2, alpha=0.9)
		plot!([x_new[1], x_new[1]], [x_new[2], x_new[2]], [f(x_new...), 0], lw=2, lc=2, ls=:dash, label="")
		push!(plts, plt)
	end
	gif(anim, fps=5)
end

# ╔═╡ cb9d4012-68c4-43e0-b6c3-9f2b435192f4
let
	gr()
	w₀ = w0
	wv_ =  wv_
	w₁, w₂ = wv_[1], wv_[2]
	p1 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> w₀+ w₁* x + w₂ * y, st=:surface, c=:jet, colorbar=false, alpha=0.8, framestyle=:origin, xlabel=L"x_1", ylabel=L"x_2", title=L"h(x) = %$(kk) w^\top x + %$(kk)w_0", ratio=1, zlim =[-30,30])

	
	xprime = [1, (-w₀ - wv_[1]) / wv_[2]]
	wprime = wv_ ./ norm(wv_)
	xorigin = dot(xprime, wprime) * wprime
	plot!(-5:0.5:5, -5:0.5:5, (x, y) -> 0, st=:surface, c=:gray, alpha=0.5)
	arrow3d!([xorigin[1]], [xorigin[2]], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)

	x0s = -5:0.5:5
	if w₂ ==0
		x0s = range(-w₀/w₁-eps(1.0) , -w₀/w₁+eps(1.0), 20)
		y0s = range(-5, 5, 20)
	else
		y0s = (- w₁ * x0s .- w₀) ./ w₂
	end
	plot!(x0s, y0s, zeros(length(x0s)), lc=:gray, lw=4, label="")
	
	p2 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.8, zlim=[-0.1, 1.1],  xlabel=L"x_1", ylabel=L"x_2", title=L"σ(%$(kk) \cdot w^\top x + %$(kk)w_0)", framestyle=:zerolines)
	plot!(-5:0.5:5, -5:0.5:5, (x, y) -> 0.5, st=:surface, c=:gray, alpha=0.75)
	arrow3d!([xorigin[1]], [xorigin[2]], [0.5], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
	plot!(x0s, y0s, .5 * ones(length(x0s)), lc=:gray, lw=4, label="")
	plot(p1, p2)
end

# ╔═╡ fb623571-df57-41fe-a025-7b9672dba242
let
	gr()
	w₀ = w0_
	wv_ =  [1,0] * 3
	w₁, w₂ = wv_[1], wv_[2]
	p1 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> w₀+ w₁* x + w₂ * y, st=:surface, c=:jet, colorbar=false, alpha=0.8, framestyle=:origin, xlabel=L"x_1", ylabel=L"x_2", title=L"h(x) = %$(kk) w^\top x + %$(kk)w_0", ratio=1, zlim =[-35,35])
	xprime = [1, (-w₀ - wv_[1]) / wv_[2]]
	wprime = wv_ ./ norm(wv_)
	xorigin = dot(xprime, wprime) * wprime
	plot!(-5:0.5:5, -5:0.5:5, (x, y) -> 0, st=:surface, c=:gray, alpha=0.5)
	arrow3d!([xprime[1]], [xprime[2]], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
	x0s = -5:0.5:5
	if w₂ ==0
		x0s = range(-w₀/w₁-eps(1.0) , -w₀/w₁+eps(1.0), 20)
		y0s = range(-5, 5, 20)
	else
		y0s = (- w₁ * x0s .- w₀) ./ w₂
	end
	plot!(x0s, y0s, zeros(length(x0s)), lc=:gray, lw=4, label="")
	
	p2 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.8, zlim=[-0.1, 1.1],  xlabel=L"x_1", ylabel=L"x_2", title=L"σ(%$(kk) \cdot w^\top x + %$(kk)w_0)", framestyle=:zerolines)
	plot!(-5:0.5:5, -5:0.5:5, (x, y) -> 0.5, st=:surface, c=:gray, alpha=0.75)
	arrow3d!([xprime[1]], [xprime[2]], [0.5], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
	plot!(x0s, y0s, .5 * ones(length(x0s)), lc=:gray, lw=4, label="")
	plot(p1, p2)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.1"
Latexify = "~0.16.1"
LogExpFunctions = "~0.3.27"
Plots = "~1.39.0"
PlutoTeachingTools = "~0.2.14"
PlutoUI = "~0.7.55"
StatsBase = "~0.34.2"
Zygote = "~0.6.69"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "47b8c07efa6f8df0e39529f6046d381fc4bea6f9"

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

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

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

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

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
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

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

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

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
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

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

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "4bf4b400a8234cff0f177da4a160a90296159ce9"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.41"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

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
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

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

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "cc0a5deefdb12ab3a096f00a6d42133af4560d71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

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
version = "0.8.1+4"

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
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

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
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

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

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

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
git-tree-sha1 = "e3be13f448a43610f978d29b7adf78c76022467a"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.12"
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

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

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

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "ee6f41aac16f6c9a8cab34e2f7a200418b1cc1e3"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+0"

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
# ╟─f1ecdbf7-cc9d-4d4d-a57b-d3c1bd1d2628
# ╟─3b27bab2-4548-406c-9ee3-bd9e5e926b17
# ╟─cf92f290-12ad-4a78-9f90-85b755ddcd31
# ╟─b62f2103-fc79-4bdf-8a21-5238c132ec5e
# ╟─9ed57ba6-c18d-436f-9d1d-6f7a4798d87f
# ╟─bde3675d-e349-4498-83e8-2fa90511f568
# ╟─df4d7b58-3e52-428c-9fe6-d3138793e9e6
# ╟─12863468-865b-4aae-8349-884698a78ed4
# ╟─3b85a854-400c-464b-bbe3-e8c4b43e009a
# ╟─88627363-e62f-48c1-886c-aa741fb3754f
# ╟─c381ba06-2e70-4223-924a-4c1946fa632a
# ╟─c3c3b39f-c421-4d87-b0bb-2b6c7df0ad89
# ╟─b4f85271-2983-4bd3-8cc6-1e4e3324db82
# ╟─436ecc71-fc90-4c73-a27f-29d800bcd189
# ╟─3a39807b-5625-4a1b-8ac6-b81588e74a7f
# ╟─e0b5bd62-cb0d-4e65-bdaa-f6d1963e89bd
# ╟─7959520f-c599-44e3-9f3e-cac9ce410926
# ╟─9971851e-ddae-4096-92ef-373bfe9ba5a5
# ╟─b38c1389-4c16-4e42-be66-7700910f1ed9
# ╟─2895ef81-b716-4c84-ba8c-cdf48ad52447
# ╟─cf24ff17-5d3f-45ef-90c7-3903aa3478bc
# ╟─88994483-36ab-46be-87f1-5cf10b9fe26e
# ╟─f03b4f96-a10a-4623-a0c0-9d1a8de0ba81
# ╟─aa7dcff1-b1b5-4b87-b1d6-0f0d79ae3d3b
# ╟─566628e5-862d-49f1-a896-603251d94b1a
# ╟─b8147db6-6f7f-4d1d-bfa6-757245e52fe8
# ╟─bdb33912-b145-4ea9-98c4-32a1b52efa24
# ╟─cb9d4012-68c4-43e0-b6c3-9f2b435192f4
# ╟─3da8e967-7b62-4130-b9b6-3d919894fdfe
# ╟─0a4b683c-1f83-4e24-9f6f-3919d9b63490
# ╟─fb623571-df57-41fe-a025-7b9672dba242
# ╟─2c6bd19a-b732-4030-8ad2-615ef3980362
# ╟─9ef7deeb-deab-41b8-89c7-609f5b9a387f
# ╟─a76c5b02-c0f4-4128-8ef8-0441b856d03a
# ╟─2e5fb5b8-9526-4f6b-91f1-a4ef6bf2a2e7
# ╟─12d54350-c152-443a-a780-f08946210523
# ╟─c4b24fc6-a94f-4a86-823e-a062c9344218
# ╟─eaacfdb5-e34a-4d50-bfae-c75208906180
# ╟─5a64e355-9682-4bd8-900b-4da1ada119ee
# ╟─930ad37f-7802-46a7-9a65-c934196b8816
# ╟─c27db17d-b2f0-485a-b87b-a35e85dfbaac
# ╟─8360e6f6-4417-4fbe-8513-d69b85d02e85
# ╟─1284949e-4d1a-4559-97cd-d58287816229
# ╟─26080a33-c7db-44ef-98ee-f2b8e443d6ad
# ╟─0d862000-fb6a-4805-8bf4-5ea8e488517d
# ╟─3aded33d-b29b-47bf-8403-12b7195511eb
# ╟─9de1c4ad-68da-4353-b237-8fe26cd4e48f
# ╟─7216b051-c9fd-47ee-9892-72cf4f862f92
# ╟─1e483e31-c1c1-4ffe-900d-3b419d8b0122
# ╟─faf13435-546f-4b7f-aeee-f54adafa78d3
# ╟─fd398edb-ca31-444b-9477-bc6389c84c9f
# ╟─5ef4cd0d-840d-4799-8691-90b54046a07a
# ╠═c53f1dc7-4498-41d5-a6de-28b574a5b9b8
# ╠═e943a212-3780-46c9-b866-2a206c40dbe0
# ╠═a54a36f8-ceb2-4c82-8ee6-8f6b140fd6a4
# ╠═9804c882-54f7-4400-8621-7c8fa576342a
# ╟─62e0abc8-a998-48c7-958e-9f12700ab009
# ╟─9db15d52-1761-4103-b0f0-bd02e37d5d75
# ╟─3528e5a8-0127-4379-81e2-a997e4c769ca
# ╟─64b94612-a446-4cb4-a10d-5aa2f55d4103
# ╟─52d8483c-d9b4-4401-85e3-1e2a7bd791fd
# ╠═bc9ab53c-58af-473c-9e05-100248f264aa
# ╠═76781718-9c80-4a67-a5a2-4abaa5f2d6e5
# ╟─ce9fb355-5483-4850-a06d-938c08ca8c10
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
