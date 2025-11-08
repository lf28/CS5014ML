### A Pluto.jl notebook ###
# v0.19.46

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

# ╔═╡ 9ca39f56-5eea-11ed-03e1-81eb0fa2a37d
begin
	using PlutoTeachingTools
	using PlutoUI
	# using Plots
	using LinearAlgebra
	# using StatsPlots
	using LogExpFunctions
	using StatsBase
	using LaTeXStrings
	using Latexify
	using Random
	using HypertextLiteral
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
end

# ╔═╡ dcacb71b-bf45-4b36-a2d2-477839f52411
using Logging

# ╔═╡ 83282ceb-ad43-462a-8c28-88191ac49471
using ForwardDiff

# ╔═╡ f3cbf4df-9704-41f6-9bdf-5ed3e0edd250
begin
	using Zygote
end

# ╔═╡ f9736ecc-053f-444a-9ef5-cdbe85795fce
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ 8d096119-f6db-4f62-a091-6f00372468ec
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

# ╔═╡ d7a55322-0d9f-44e8-a2c6-4f0cead25f9d
Logging.disable_logging(Logging.Info) ; # or e.g. Logging.Info

# ╔═╡ 5bb178a2-f119-405f-a65b-ec6d59a112e0
TableOfContents()

# ╔═╡ 799ead95-2f68-4bcb-aae8-dd8b0b00560c
ChooseDisplayMode()

# ╔═╡ afb99b70-a418-4379-969b-355fbcfe8f14
md"""

# CS5014 Machine Learning


#### Gradient descent

\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ ee92f14a-784a-407c-8955-9ea1dd9cf942
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

# ╔═╡ 2c063e21-faec-479d-a452-db411a7c9adc
md"""

## Notations


"""

# ╔═╡ 694d5df4-b073-4d74-b929-108c4ea6d646
TwoColumn(md"""

##### Super-index with bracket: ``\large \mathbf{x}^{(i)}``

* ###### index observations/data/rows
* *e.g.* ``y^{(i)}`` the ``i``-th target
* *e.g.* ``\mathbf{x}^{(i)}`` the ``i``-th observation's features
* usually use ``n``: number of observations

""", html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/iindex.svg" width = "400"/></center>""")

# ╔═╡ b49fd692-c36f-40de-be10-55a65ba13dba
TwoColumn(md"""

##### Sub-index: ``\large \mathbf{x}_j``

* ###### index feature/columns
* *e.g.* ``\mathbf{x}^{(3)}_2``: the second feature of ``i``-th observation
* usually use letter ``m``: number of features 

""", html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/jindex.svg" width = "400"/></center>""")

# ╔═╡ c49a6965-727e-419b-b66e-57dc61415edf
md"""

# Gradient descent


"""

# ╔═╡ e7447991-fd5f-403f-8ea8-61df1e076ae7
md"""

## Reading & references

##### Essential reading 


* [_Understanding deep learning_ by _Simon Prince._: Chapter 6.1](https://github.com/udlbook/udlbook/releases/download/v.1.20/UnderstandingDeepLearning_16_1_24_C.pdf)

"""

# ╔═╡ 4b757c23-d30a-445c-a9d7-1b733150cacf
md"""

## Recap: linear approximation
"""

# ╔═╡ a92344a8-83a0-4d8a-825e-b546b53d6292
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/linearappxmulti.svg" width = "450"/></center>"""

# ╔═╡ cbb348f6-25e6-4db5-9dce-e5e7d5e07931
md"""Move me ``x_1``: $(@bind x01 Slider(-1.8:0.1:2.8; default= 0)), ``x_2``: $(@bind x02 Slider(-1.8:0.1:2.8; default= 0))"""

# ╔═╡ 07c7e0e4-bbfe-4b98-80a9-d952b6705997
md"""Add ``\nabla(\mathbf{x})=\mathbf{0}``: $(@bind add_zeros CheckBox(false)); Move me: $(@bind angi Slider(-30:90; default = 45)); $(@bind angi2 Slider(-45:60; default = 30))"""

# ╔═╡ 996fcbcb-1f30-4138-85cf-7b3e9a708505
l0s = [[0, 0], [2, -1], [2, 2], [-1, 2], [-1, -1], [2, 0], [0, -1], [0, 2], [-1, 0]];

# ╔═╡ f3f2c04a-c50d-49cd-907a-07fa301df236
md"""



## Recap: facts about _gradient_  (1)

!!! important "First fact"
	#### ``\nabla f(\mathbf{x}): \mathbb{R}^n \rightarrow \mathbb{R}^n`` : is a vector to vector function
	



"""

# ╔═╡ 7534d726-45cc-453c-904e-166db6c11410
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/gradient.svg" width = "400"/></center>"""

# ╔═╡ 496bed39-98e5-437f-906f-413be5e07e77
md"""



## Recap: facts about _gradient_  (2)




!!! important "Second fact"
	#### ``\nabla f(\mathbf{x})``: the output points to the *greatest ascent direction* 
	* #### _locally_ at ``\mathbf{x}``



"""

# ╔═╡ cc9be8be-7230-4873-884c-6ac119c0bdb3
html"""<center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/3d-gradient-cos.svg/2560px-3d-gradient-cos.svg.png" width = "350"/></center>"""

# ╔═╡ 65eea27a-0df8-409f-840d-1ff0bd740339
md"""
## Optimisation 

#### To find the optimal ``\mathbf{w}``, we can simply solve 


```math
\Large
\nabla l(\mathbf{w}) = \mathbf{0} \;\;\; \# \texttt{solve it for w}
```


"""

# ╔═╡ 5921c8dc-25d6-4acb-9f60-1bcf8644ee12
md"""
## Optimisation 

#### To find the optimal ``\mathbf{w}``, we can simply solve 


```math
\Large
\nabla l(\mathbf{w}) = \mathbf{0} \;\;\; \# \texttt{solve it for w}
```

### But...

```math
\Large
	\boxed{\nabla l(\mathbf{w}) = \mathbf{0} \;\;\; \# \texttt{very hard to solve}}
``` 

* #### *CAN NOT* be solved for _99.99% problems_
  * ##### linear regression is a rare exception

"""

# ╔═╡ 49781399-7f9e-4237-ab23-e52fdb87ed15
md"""
## How to optimise then?

> ### *Gradient Descent*

* ### this is the _Bread-and-Butter_ of CS5014

"""

# ╔═╡ 7d4e0c2e-0f2b-4e06-a265-14e276ba9387
md"""

## Gradient descent



### _The key idea_: 
> #### _use_ local linear approximations to guide us _at each step_



"""

# ╔═╡ 002c9cac-6bc1-4288-903f-159afdcceef8
md"Add tangent: $(@bind add_approx CheckBox(false)), Add gradient: $(@bind add_grad CheckBox(false)), Add negative grad: $(@bind add_neggd CheckBox(false)), Add next: $(@bind add_nextstep CheckBox(false))"

# ╔═╡ 035032e4-fde2-4067-b08c-477346d4c5af
f1(x) = .5 * x^2; # you can change this function!

# ╔═╡ ffeb64b4-db59-4f20-9332-b502904802b5
x_pos = let
	x_pos = Float64[-3.5]
	x = x_pos[1]
	λ = 0.15
	for i in 1:30
		xg = Zygote.gradient(f1, x)[1]
		x = x - λ * xg
		push!(x_pos, x)
	end

	x_pos
end;

# ╔═╡ 478d8fe1-9548-4fab-bc60-10c1b87d4e37
md"Move me: $(@bind x̂ Slider(x_pos))"

# ╔═╡ a14618d8-8416-4a1d-aee4-9b8d55fa4204
plt_linear_approx = let
	gr()
	f = f1
    # Plot function
    xs = range(-4, 4, 200)
    ymin, ymax = extrema(f.(xs))
    p = plot(
        xs,
        f;
        label="",
        xlabel=L"x",
        legend=:outerbottom,
        ylims = (ymin - .5, ymax + .5),
        legendfontsize=10,
		lw = 2,
		ratio = .7,
		framestyle=:zerolines
    )
	scatter!([x̂], [f(x̂)], label="", mc=:white, msc=:gray, msw=2, alpha=0.5)
	annotate!([x̂], [0.1], text(L"x_t"))
    # Obtain the function 𝒟fₓ̃ᵀ
    ŷ, 𝒟fₓ̂ᵀ = Zygote.pullback(f, x̂)
    # Plot Dfₓ̃(x)
    # plot!(p, xs, w -> 𝒟fₓ̂ᵀ(w)[1]; label=L"Derivative $\mathcal{D}f_\tilde{x}(x)$")
    # Show point of linearization
    vline!(p, [x̂]; style=:dash, c=:gray, label="")
    # Plot 1st order Taylor series approximation
    taylor_approx(x) = f(x̂) + 𝒟fₓ̂ᵀ(x - x̂)[1] # f(x) ≈ f(x̃) + 𝒟f(x̃)(x-x̃)
	if add_approx 
    	plot!(p, range(x̂ -2, x̂+2, 10), taylor_approx; label="", lc=:gray,  lw=1.5)
	end

	xg = Zygote.gradient(f, x̂)[1]
	if add_grad
		xg = x̂ + xg
		plot!([x̂, xg], [f(x̂), f(x̂)], lc=:gray, arrow=Plots.Arrow(:close, :head, 1, 1),  st=:path, label="")
		annotate!(.5 * [x̂ + xg], [f(x̂)], text(L"f'(x_0)=%$(xg)", 10, :bottom))
	end
	λ = 0.15
	x_new = x̂ -λ * xg
	if add_neggd
		plot!([x̂, x_new], [f(x̂), f(x̂)], lc=:black, lw=2, arrow=Plots.Arrow(:close, :head, 1, 1),  st=:path, label="")
		annotate!(.5 * [x̂ + x_new], [f(x̂)], text(L"-\gamma f'(x_0)", 10, :bottom))
	end

	if add_nextstep
		# x_new = x̂ + xg_neg
		scatter!([x_new], [f(x_new)], label="", mc=:white, msc=:black, msw=2, alpha=1.0)
		vline!([x_new]; style=:dash, c=:black, label="")
		annotate!([x_new], [0.1], text(L"x_{new}"))
	end

	p
end;

# ╔═╡ da573528-649b-4fbe-97f9-5d82fe0cc716
plt_linear_approx

# ╔═╡ a21a6912-18a6-49b8-8ba8-18df9006578a
f2(x) = sin(2.3x) + 0.3*x^2; # you can change this function!

# ╔═╡ 35bad274-8618-48c1-b9ea-a232d6fcf23c
# f2(x) = .5 * x^2; # 

# ╔═╡ 67d19b65-ffe6-4d53-9087-f4062311309d
md"""
## Demonstration in ``1``-dim
"""

# ╔═╡ 063a28e7-4e1d-4583-8ff9-90930dab6396
@bind x_start Select([-3.5 => "x1", -2.0 => "x2", 0.5=> "x3", 1.0 => "x4", 3.0=> "x5"])

# ╔═╡ c5ed4d55-ccc5-4855-8b04-203a4205ff5c
let
	gr()
	f = f2
	x0 = Float64(x_start)
    # Plot function
    xs = range(-3.7, 3.7, 100)
    ymin, ymax = extrema(f.(xs))
    p = plot(
        xs,
        f;
        label="",
        xlabel=L"x",
		ylabel=L"f(x)",
        legend=:outerbottom,
        ylims = (ymin - .5, ymax + .5),
        legendfontsize=10,
		lw = 2,
		c=:gray,
		framestyle=:zerolines
    )
	xs = [x0]
	γ = 0.025
	for i in 1:25
		x0 = x0 - γ * ForwardDiff.derivative(f, x0)
		push!(xs, x0)
	end
	plts = []
	for i in 2:length(xs)
		xss = xs[1:i] 
		plt = deepcopy(p)
		scatter!(xss, f.(xss), c=:coolwarm, zcolor = range(0, stop = 0.5, length=length(xss)), ms=5,  colorbar=false, label="", title="Iteration: $(i-1)")
		scatter!(xss, zeros(length(xss)), c=:coolwarm, zcolor = range(0, stop = 0.5, length=length(xss)), ms=5,  colorbar=false, label="")
		push!(plts, plt)
	end

	
	anim = Animation()
		
	[frame(anim, p) for p in plts]

	gif(anim, fps=8)
end

# ╔═╡ 36d0cea4-4d17-4493-bc45-d5f1f15a67f8
let
	gr()
	f = f2
	# x0 = Float64(x_start)
    # Plot function
    xs = range(-3.7, 3.7, 100)
    ymin, ymax = extrema(f.(xs))
    p = plot(
        xs,
        f;
        label="",
        xlabel=L"x",
		ylabel=L"f(x)",
        legend=:outerbottom,
        ylims = (ymin - .5, ymax + .5),
        legendfontsize=10,
		lw = 2,
		c=:gray,
		framestyle=:zerolines
    )
	x0s = [-3.5, -2.5, -2, 0.4, 1, 3]
	scatter!(x0s, f.(x0s), c=:gray, ms=6, m=:x, markerstrokewidth=5, label="")
	scatter!(x0s, zeros(length(x0s)), c=:gray, ms=6, mc=:white, msc=:gray, msw=2, alpha=2.0, label="")
	γ = 0.04
	xss = zeros(length(x0s) , 20)
	xss[:, 1] = x0s
	plts = []
	for i in 2:20
		x0s = x0s .- γ * ForwardDiff.derivative.(f, x0s)
		# push!(xss, x0s)
		xss[:, i] = x0s
	end

	# xss

	for j in 3:size(xss)[2]
		plt = deepcopy(p)
		for i in 1:size(xss)[1]
			xij = xss[i, 2:j]
			scatter!(xij, f.(xij), c=:coolwarm, zcolor = range(0, stop = 0.5, length=length(xij)), ms=3,  colorbar=false, label="", title="Iteration: $(j-1)")
			scatter!(xij, zeros(length(xij)), c=:jet, zcolor = range(0, stop = 0.5, length=length(xij)), ms=3,  colorbar=false, label="")
		end
		push!(plts, plt)
	end


	
	anim = Animation()
		
	[frame(anim, p) for p in plts]

	gif(anim, fps=4)
end

# ╔═╡ b06cb19a-4b0e-4a45-aa6e-285b8f47a06d
md"""

## Gradient descent in multi-dimension
"""

# ╔═╡ 0541ccc4-b05b-44d3-9ea7-00c27b644c04
md"Add linear approx: $(@bind add_linear_app CheckBox(false)), add directions: $(@bind add_where_to_go CheckBox(false)), add gradient: $(@bind add_gradient_local CheckBox(false)), add negative gradient: $(@bind add_neg_gradient_local CheckBox(false)), Move: $(@bind add_show_the_move CheckBox(false))"

# ╔═╡ 9caa90a1-103e-4268-83a7-875dcdd6c488
md"Show where to go: $(@bind add_where_go_values CheckBox(false)); ``\mathbf{u}`` (orange): $(@bind utheta Slider(range(-π, π, 100), default=0))"

# ╔═╡ 1d5970d6-95d5-44a1-9790-0138c0d3f1a4
md"""

## Demonstration
"""

# ╔═╡ db883f69-f516-432d-81f6-2bdfd2f0b40c
@bind x0_st Select([[0.1, 0.1] * 0.9 => "x1", [0.1,  1.9] =>"x2", [3, 3] * 0.98 => "x3", [0.1, 3] * 0.98 => "x4"])

# ╔═╡ 4814ab50-3ed2-4197-b116-719ccbb28939
function taylorApprox(f, x0, order = 2)
	gx0 = ForwardDiff.gradient(f, x0)
	hx0 = ForwardDiff.hessian(f, x0)
	if order == 1	
		# tf(x) = f(x0) + gx0' * (x-x0) this is a bug; need to use anonymouse function instead
		(x) ->  f(x0) + gx0' * (x-x0)
	else

		(x) -> f(x0) + gx0' * (x-x0)  + 0.5 *(x-x0)' * hx0 * (x-x0)
	end
end;

# ╔═╡ b6fd9d29-b5ac-4940-90d8-fcf0d8063003
begin
	f_demo(w₁, w₂) = 1/4 * (w₁^4 + w₂^4) - 1/3 *(w₁^3 + w₂^3) - w₁^2 - w₂^2 + 4
	f_demo(w::Vector{T}) where T <: Real = f_demo(w...)
	∇f_demo(w₁, w₂) = [w₁^3 - w₁^2 - 2 * w₁, w₂^3 - w₂^2 - 2 * w₂]
	∇f_demo(w::Vector{T}) where T <: Real = ∇f_demo(w...)
end;

# ╔═╡ c9a9640d-dbf5-4e51-9c6e-3c859b61f593
let
	gr()
	f = f_demo
	x1_ = range(-2., stop =3.0, length=60)
	x2_ = range(-2., stop =3.0, length=60)
	p1_ = plot(x1_, x2_, (a, b) -> f([a, b]), st=:surface, xlabel =L"x_1", ylabel=L"x_2", zlabel=L"f(\mathbf{x})", colorbar=false, color=:jet, title="Solve " * L"\nabla L{(\mathbf{x})} = \mathbf{0}", camera=(angi, angi2))
	len = 0.7
	l0 = [x01, x02]
	tf = taylorApprox(f, l0, 1)
	plot!(range(l0[1] - 0.8, l0[1]+0.8, 5), range(l0[2] - 0.8, l0[2]+0.8, 5), (a, b) -> tf([a, b]), st=:surface, alpha=0.15, display_option=Plots.GR.OPTION_Z_SHADED_MESH)
	scatter!([l0[1]], [l0[2]], [f(l0)], ms=5, label=L"\mathbf{x}_0", mc=:white, msc=:gray, msw=2, alpha=2.0)
	if add_zeros
		for (li, l0) in enumerate(l0s)
			tf = taylorApprox(f, l0, 1)
			plot!(range(l0[1] - len, l0[1]+len, 5), range(l0[2] - len, l0[2]+len, 5), (a, b) -> tf([a, b]), st=:surface, alpha=0.3,  display_option=Plots.GR.OPTION_Z_SHADED_MESH)
			label = li == 1 ? L"\mathbf{x};\; \texttt{s.t.} \nabla L(\mathbf{x}) = 0" : ""
			scatter!([l0[1]], [l0[2]], [f(l0)], ms=3, label=label)
		end
	end
	p1_
end

# ╔═╡ f477c82e-f4d6-4918-85d1-19ce21e90b5b
more_ex_surface = let
	gr()
	plot(-2:0.1:3, -2:0.1:3, f_demo, st=:surface, color=:jet, colorbar=false, aspect_ratio=1.0, xlabel=L"w_1", ylabel=L"w_2")
end;

# ╔═╡ f4a1f7db-0bcb-45b6-be9d-1c57dd6e2b99
function gradient_descent(f, ∇f; w_init= zeros(2), max_iters = 200, γ = 0.01)
	w₀ = w_init
	losses = Vector{Float64}(undef, max_iters+1)
	traces = zeros(length(w₀), max_iters+1)
	losses[1], traces[:, 1] = f(w₀), w₀
	for t in 1:max_iters
		# calculate the gradient at t
		∇w = ∇f(w₀)
		# follow a small gradient step
		w₀ = w₀ - γ * ∇w
		losses[t+1], traces[:, t+1] = f(w₀), w₀ # book keeping for visualisation
	end
	return losses, traces
end;

# ╔═╡ c2cd3414-7d70-467b-b881-799c20a8489c
md"""
## The algorithm



"""

# ╔═╡ 20f57c9d-3543-4cb6-8844-30b30e3b08ec
md"""


```math
\Large
\boxed{\mathbf{w}_{new} \leftarrow \mathbf{w}_{old} - \underbrace{\colorbox{lightgreen}{$\gamma$}}_{{\color{green}\small \rm learning\; rate}} \nabla f(\mathbf{w}_{old})}
```


* ##### the algorithm needs
  * ##### ``\nabla f(\mathbf{w})``, the gradient function
  * ##### ``\colorbox{lightgreen}{$\gamma$}``, a positive _learning rate_ *e.g.* 0.01


"""

# ╔═╡ c2bebba9-9a03-43dd-81e4-93da08e1118c
md"""

##

----
* #### random guess ``\large\mathbf{w}_0``

* #### while *not converge*
  * #### ``\large\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma \nabla f(\mathbf{w}_{t-1})``
-----


"""

# ╔═╡ 43ac689e-c16a-47cc-9c21-477fb2a6989b
md"""
##

##### _Show me the code_

```python
w0 = np.random.randn(d) # random guess; d is the dimension of w
lr = 0.1 # learning rate 
while True:
	gradw = evaluate_grad(w0) # compute the gradient at w0
	w0 = w0 - lr * gradw # follow the gradient by a little
	... # break if converge
```
"""

# ╔═╡ a5bfaa1e-9d40-4f2e-b0d5-73ab9bbceddf
md"""

## Maximisation: gradient _ascent_

#### To *maximise* a function: gradient ascent


```math
\LARGE
\mathbf{w}_{new} \leftarrow \mathbf{w}_{old}\;  \colorbox{orange}{$+$} \;\gamma \nabla f(\mathbf{w}_{old})
```



#### The two problems are exchangeable

```math
\Large \arg\max f(\mathbf{w}) \Leftrightarrow \arg\min \{-f(\mathbf{w})\}
```


* ##### we only consider minimisation problem onwards



"""

# ╔═╡ ace9c646-743c-494f-a35b-0acc7107b8a9
md"""

## Properties of GD: self-tuning steps

!!! note ""
	#### Self-tuning steps 
	* ##### it implies it will converge (even left running forever) provided
	  * ##### _there is a local minimum_
	  * ##### _and the learning rate is ``<\infty``_

"""

# ╔═╡ 3b1c8be8-fb5e-48d7-9436-cd23a715f223
let
	gr()
	f = f2
	x0 = Float64(x_start)
    # Plot function
    xs = range(-3.7, 3.7, 100)
    ymin, ymax = extrema(f.(xs))
    p = plot(
        xs,
        f;
        label="",
        xlabel=L"x",
		ylabel=L"f(x)",
        legend=:outerbottom,
        ylims = (ymin - .5, ymax + .5),
        legendfontsize=10,
		lw = 2,
		c=:gray,
		framestyle=:zerolines,
		size = (800,400),
		title="Self-tuning step sizes"
    )

	xs = range(-3.7, 3.7, 45)
	λ = 0.19
	plot!(xs, f.(xs), st=:scatter, ms=2, mc=1,label="")
	for x̂ in xs
		xg = ForwardDiff.derivative(f, x̂)
		x_new = x̂ -λ * xg
		plot!([x̂, x_new], [f(x̂), f(x̂)], lc=:black, lw=1,arrow=true,  arrowsize=1, st=:path, label="")
		# annotate!(.5 * [x̂ + x_new], [f(x̂)], text(L"-\lambda f'(x_0)", 10, :bottom))
	end
	
	p
end

# ╔═╡ f177f82c-fb9b-45a4-8944-6d2d0c540572
md"""

#### Why? 
* #### the gradient ``\nabla f(\mathbf{w}_t) \rightarrow \mathbf{0}`` as ``t\rightarrow \infty``!

 * ##### therefore, ``\large\mathbf{w}_t \leftarrow \underbrace{\mathbf{w}_{t-1} - \gamma \nabla f(\mathbf{w}_{t-1})}_{\mathbf{w}_{t-1}} ``
"""

# ╔═╡ 00dac405-aef9-420f-b019-12d325901fd9
let

	gr()
	w0 = [0.5, 0.5]
	γ = 0.1
	_, traces = gradient_descent(f_demo, ∇f_demo; w_init = w0, max_iters = 40, γ = γ)
	plt=plot(-0.1:0.05:3, -0.1:0.05:3, f_demo, st=:contour, color=:jet, colorbar=false, aspect_ratio=1.0, framestyle=:none)
	wt = w0
	anim = @animate for t in 1:1:size(traces)[2]
		plot!([traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3, title="Iteration: $(t)")
		plot!([wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="")
		wt = traces[1:2, t]
	end

	gif(anim, fps=5)
end

# ╔═╡ 46dd39e0-95bf-4079-a982-4f93b16c3a5e
md"""

## Properties of GD: local optimum


!!! note ""
	#### Gradient descent only converge to a local optimum

	* #### depending on the starting location ``\mathbf{w}_0``

"""

# ╔═╡ 095f95fa-75e1-488f-a1c6-afcaaad30baa
TwoColumn(let
	gr()
	w0 = [-3,-2]
	_, traces = gradient_descent(f_demo, ∇f_demo; w_init = w0)
	plt=plot(-2.:0.05:3, -2.:0.05:3, f_demo, st=:contour, color=:jet, colorbar=false, aspect_ratio=1.0, framestyle=:none)
	wt = w0
	anim = @animate for t in 1:5:size(traces)[2]
		plot!([traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
		plot!([wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="")
		wt = traces[1:2, t]
	end

	gif(anim, fps=5)
end, let
	gr()
	w0 = [3.5,-2]
	_, traces = gradient_descent(f_demo, ∇f_demo; w_init = w0)
	plt=plot(-2.:0.05:3, -2.:0.05:3, f_demo, st=:contour, color=:jet, colorbar=false, aspect_ratio=1.0, framestyle=:none)
	wt = w0
	anim = @animate for t in 1:5:size(traces)[2]
		plot!([traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
		plot!([wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="")
		wt = traces[1:2, t]
	end

	gif(anim, fps=5)
end)

# ╔═╡ 5d52d818-21b2-4e8f-8efa-486274b68e57
fs, traces = gradient_descent(f_demo, ∇f_demo; w_init=[3.5,2.5]);

# ╔═╡ b732f3da-7be0-4470-9a5b-a389c1d1c166
anim_demo = let
	gr()

	plt = plot(-2.:0.05:3, -2.:0.05:3, f_demo, st=:contour, color=:jet, colorbar=false, aspect_ratio=1.0, framestyle=:none)
	wt = [3.5, 2.5]
	anim = @animate for t in 1:5:size(traces)[2]
		plot!([traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
		plot!([wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="")
		wt = traces[1:2, t]
	end
end;

# ╔═╡ f15c498f-a4af-4cd5-a2db-2ea42516fb0f
TwoColumn(gif(anim_demo, fps=5), let
	gr()
	w0 = [-2.5, 3]
	_, traces = gradient_descent(f_demo, ∇f_demo; w_init = w0)
	plt=plot(-2.:0.05:3, -2.:0.05:3, f_demo, st=:contour, color=:jet, colorbar=false, aspect_ratio=1.0, framestyle=:none)
	wt = w0
	anim = @animate for t in 1:5:size(traces)[2]
		plot!([traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
		plot!([wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="")
		wt = traces[1:2, t]
	end

	gif(anim, fps=5)
end)

# ╔═╡ 538a3ded-2ca0-4f0c-a931-fb7e43e5c24f
md"""

## Propterties of GD: learning rate


##### The learning rate ``\gamma`` should be tuned

* ##### set by trial and error
* ##### ``\gamma`` can also be adaptive: ``\gamma_t`` (_this is optional_)

"""

# ╔═╡ cd2a8e1e-d174-4c62-9391-beca7272a15d
md"""

## Learning rate too large
"""

# ╔═╡ f9bc87f7-0e29-481d-97a8-24f62364dc5a
# let
# 	gr()
# 	f_demo = f_demo_2
# 	∇f_demo = ∇f_demo_2
# 	# w0 = []
# 	γ = 1.0
# 	losses, traces = gradient_descent(f_demo, ∇f_demo; w_init=[0,-1], γ = γ)
# 	# plt1 = plot(-3.:0.05:1, -2.:0.05:1, f_demo, st=:contour, color=:jet, colorbar=false,  xlim=[-3, 1], ylim=[-2, 1],  size=(600,600), xlabel=L"w_1", ylabel=L"w_2")

# 	# plt2 = plot(xlabel="iteration", ylabel="Loss", title="Learning rate : "* L"\gamma = 0.1")
# 	# wt = [0, -1]
# 	# anim = @animate for t in 1:5:size(traces)[2]
# 	# 	plot!(plt1, [traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
# 	# 	plot!(plt1, [wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="", title="Learning rate : "* L"\gamma = %$(γ);"*" Iteration: $(t)")
# 		# wt = traces[1:2, t]

# 		plt2 = plot(losses .+ 5, xlabel="iteration", ylabel="Loss", title="Loss trajectory with learning rate "* L"\gamma", label="", framestyle=:origins)

# 	# 	plot(plt1, plt2, layout = grid(2, 1, heights=[0.7 ,0.3]))
# 	# end

# 	# gif(anim, fps=5)
# end

# ╔═╡ 8afbc273-c62e-49a8-98f6-893b0d8a3696
md"""

## Learning rate too small
"""

# ╔═╡ 726545fd-fbbc-4ce9-a35d-30e7e23ff3f9
begin
	f_demo_2(w₁, w₂) = sin(w₁ + w₂) + cos(w₁)^2
	∇f_demo_2(w₁, w₂) = [cos(w₁ + w₂) - 2*cos(w₁)*sin(w₁),  cos(w₁ + w₂)]
end;

# ╔═╡ 78c47b3f-9641-4ea4-9f3d-fadaa0ae004c
begin
	f_demo_2(w) = f_demo_2(w[1], w[2]) #overload the function with vector input
	∇f_demo_2(w) = ∇f_demo_2(w[1], w[2])  #overload with vector input
end;

# ╔═╡ df7c5704-a493-499e-b75d-550606e04edd
let
	gr()
	f_demo = f_demo_2
	∇f_demo = ∇f_demo_2
	# w0 = []
	γ = 0.1
	losses, traces = gradient_descent(f_demo, ∇f_demo; w_init=[0,-1], γ = γ)
	plt1 = plot(-3.:0.05:1, -2.:0.05:1, f_demo, st=:contour, color=:jet, colorbar=false,  xlim=[-3, 1], ylim=[-2, 1],  size=(600,600), xlabel=L"w_1", ylabel=L"w_2")

	# plt2 = plot(xlabel="iteration", ylabel="Loss", title="Learning rate : "* L"\gamma = 0.1")
	wt = [0, -1]
	anim = @animate for t in 1:5:size(traces)[2]
		plot!(plt1, [traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
		plot!(plt1, [wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="", title="Learning rate : "* L"\gamma = %$(γ);"*" Iteration: $(t)")
		wt = traces[1:2, t]

		plt2 = plot(losses[1:t], xlabel="iteration", ylabel="Loss", title="Loss trajectory with learning rate : "* L"\gamma = %$(γ)", label="")

		plot(plt1, plt2, layout = grid(2, 1, heights=[0.7 ,0.3]))
	end

	gif(anim, fps=5)
end

# ╔═╡ 17c84d3e-aba7-4c66-87fc-8c0289d0b51d
let
	gr()
	f_demo = f_demo_2
	∇f_demo = ∇f_demo_2
	# w0 = []
	γ = 1.0
	losses, traces = gradient_descent(f_demo, ∇f_demo; w_init=[0,-1], γ = γ)
	plt1 = plot(-3.:0.05:1, -2.:0.05:1, f_demo, st=:contour, color=:jet, colorbar=false,  xlim=[-3, 1], ylim=[-2, 1],  size=(600,600), xlabel=L"w_1", ylabel=L"w_2")

	# plt2 = plot(xlabel="iteration", ylabel="Loss", title="Learning rate : "* L"\gamma = 0.1")
	wt = [0, -1]
	anim = @animate for t in 1:5:size(traces)[2]
		plot!(plt1, [traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
		plot!(plt1, [wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="", title="Learning rate : "* L"\gamma = %$(γ);"*" Iteration: $(t)")
		wt = traces[1:2, t]

		plt2 = plot(losses[1:t], xlabel="iteration", ylabel="Loss", title="Loss trajectory with learning rate : "* L"\gamma = %$(γ)", label="")

		plot(plt1, plt2, layout = grid(2, 1, heights=[0.7 ,0.3]))
	end

	gif(anim, fps=5)
end

# ╔═╡ 1c1fb3aa-0dd6-4ee7-98af-81d384260685
let
	gr()
	f_demo = f_demo_2
	∇f_demo = ∇f_demo_2
	# w0 = []
	γ = 1e-3
	losses, traces = gradient_descent(f_demo, ∇f_demo; w_init=[0,-1], γ = γ)
	plt1 = plot(-3.:0.05:1, -2.:0.05:1, f_demo, st=:contour, color=:jet, colorbar=false,  xlim=[-3, 1], ylim=[-2, 1],  size=(600,600), xlabel=L"w_1", ylabel=L"w_2")

	# plt2 = plot(xlabel="iteration", ylabel="Loss", title="Learning rate : "* L"\gamma = 0.1")
	wt = [0, -1]
	anim = @animate for t in 1:5:size(traces)[2]
		plot!(plt1, [traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3)
		plot!(plt1, [wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="", title="Learning rate : "* L"\gamma = %$(γ);"*" Iteration: $(t)")
		wt = traces[1:2, t]

		plt2 = plot(losses[1:t], xlabel="iteration", ylabel="Loss", title="Loss trajectory with learning rate : "* L"\gamma = %$(γ)", label="")

		plot(plt1, plt2, layout = grid(2, 1, heights=[0.7 ,0.3]))
	end

	gif(anim, fps=5)
end

# ╔═╡ c9baffcf-c1cc-42d6-96b5-1cc9a8801113
md"""



## Gradient check

#### Gradient derivation is *error*-prone
* ##### _derivation error_
* ##### _or implementation error_


#### We test software carefully; we should also test gradients

* ##### we validate gradients by finite-difference method




"""

# ╔═╡ 40af31cc-1bc6-436c-ae6b-d13e2300779a
md"## Derivative -- finite difference method"

# ╔═╡ 5063f1be-5f18-43f6-b428-a0464e8fb339
md"

```math
\Large
\frac{d f({x})}{d x} \approx \frac{f({x}+ \epsilon) - f({x})}{\epsilon}
```
"

# ╔═╡ c0acfd8e-c152-4991-80a9-65d69c3bda69
md"""

```python
## python
def df(x, f, eps)
	df = f(x+eps) - f(x)
	df/eps
```
"""

# ╔═╡ 1ae4f029-cb95-4897-8029-903767babf7b
function df(x; f, ϵ)  
	(f(x+ϵ) - f(x))/ϵ
end;

# ╔═╡ 4f176ca1-0326-4576-ae18-af2e1b697655
md"

##### As an example, 

```math
\large
f(x)= \sin(x),\; f'(x) = \cos(x)
```
"

# ╔═╡ a676ed26-1420-478f-a25c-fe99ab94c0a5
md"""The small constant ``ϵ``: $(@bind ϵ_ Slider(1e-7:1e-7:.5, default=0.5, show_value=true))"""

# ╔═╡ 6e25724e-79f7-4050-b3ff-fc51bfc852b5
let
	gr()
	x₀ = 0.0
	Δx = ϵ_
	xs = -1.2π : 0.1: 1.2π
	f, ∇f = sin, cos
	# anim = @animate for Δx in π:-0.1:0.0
	# Δx = 1.3
	plot(xs, sin, label=L"\sin(x)", ylim = [-1.5, 1.5], xlabel=L"x", lw=2, legend=:outerbottom, framestyle=:semi, title="Derivative at "*L"x=0", legendfontsize=10)
		df = f(x₀ + Δx)-f(x₀)
		k = Δx == 0 ? ∇f(x₀) : df/Δx
		b = f(x₀) - k * x₀ 
		# the approximating linear function with Δx 
		plot!(xs, (x) -> k*x+b, label="", lw=2)
		# the location where the derivative is defined
		scatter!([x₀], [f(x₀)], ms=3, label=L"x_0,\; \sin(x_0)")
		scatter!([x₀+Δx], [f(x₀+Δx)], ms=3, label=L"x_0+Δx,\; \sin(x_0+Δx)")
		plot!([x₀, x₀+Δx], [f(x₀), f(x₀)], lc=:gray, label="")
		plot!([x₀+Δx, x₀+Δx], [f(x₀), f(x₀+Δx)], lc=:gray, label="")
		font_size = Δx < 0.8 ? 12 : 14
		annotate!(x₀+Δx, 0.5 *(f(x₀) + f(x₀+Δx)), text(L"Δf", font_size, :top, rotation = 90))
		annotate!(0.5*(x₀+x₀+Δx), 0, text(L"Δx", font_size,:top))
		annotate!(-.6, 1, text(L"\frac{Δf}{Δx}=%$(round(k, digits=6))", 15,:top))
end

# ╔═╡ 0be36e2d-817b-4951-8a3d-4ab44ce761ce
md"
The difference:
"

# ╔═╡ 94899a98-1565-4129-a264-1c4e1855982b
let
	𝒻(x) = sin(x)
	∇𝒻(x) = cos(x)
	x₁ = 1.0
	df(x₁; f=𝒻, ϵ = ϵ_) - ∇𝒻(x₁)
end

# ╔═╡ 4bd343a1-e6b9-439b-b156-8830687bb9e6
md"""
## Gradient check -- finite difference method

##### By gradient's definition, each entry is a partial:

```math
\Large
\frac{\partial f(\mathbf{x})}{\partial x_i} \approx \frac{f(\mathbf{x}+ \epsilon \cdot \mathbf{e}_i) - f(\mathbf{x})}{\epsilon}
```

* ##### ``\mathbf{e}_i`` is the i-th standard basis vector

  * as an example, for ``i=1``
```math
\large
\mathbf{x} + \epsilon\cdot  \mathbf{e}_1 =\begin{bmatrix}x_1  \\ x_2 \\\vdots \\ x_m \end{bmatrix} + \epsilon \begin{bmatrix}1  \\ 0 \\\vdots \\ 0 \end{bmatrix}= \begin{bmatrix}x_1 + \epsilon \\ x_2 \\\vdots \\ x_m \end{bmatrix}
```

* ##### *idiot-proof* and can be computed for all ``f`` 
	  
"""

# ╔═╡ afd2cf78-ca7a-4662-9350-75f857ef528b
md"""

## Finite difference -- central method

#### In practice, we instead use the `central` approximation
* mote accurate for the same ``\Delta x``

```math
\Large
\frac{d f({x})}{d x} \approx \frac{f({x}+ \epsilon) - f({x} -\epsilon)}{2\epsilon}
```

"""

# ╔═╡ 90257a92-c845-4595-abe9-33d6d9c22c2c
md"Move me: $(@bind ϵ2_ Slider(0:1e-6:2, default=.25, show_value=true))"

# ╔═╡ 4ba59af5-3ebc-42b4-a928-3c4b17024f8e
let
	gr()
	x₀ = 0.0
	Δx = ϵ2_
	xs = -1.2π : 0.1: 1.2π
	f, ∇f = sin, cos
	# anim = @animate for Δx in π:-0.1:0.0
	# Δx = 1.3
	plot(xs, sin, label=L"\sin(x)", ylim = [-1.5, 1.5], xlabel=L"x", lw=2, legend=:outerbottom, framestyle=:semi, title="Derivative at "*L"x=0", legendfontsize=10)
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

# ╔═╡ 1643d5a5-8925-45b7-a4ff-a39165207ec1
md"""

## Multi-variate gradient - finite differences



$$\large \nabla f(\mathbf{w}) = \begin{bmatrix}\frac{\partial f(\mathbf{w})}{\partial w_1}\\ \frac{\partial f(\mathbf{w})}{\partial w_2}\\ \vdots \\ \frac{\partial f(\mathbf{w})}{\partial w_m}\end{bmatrix} \approx \frac{1}{2\epsilon} \begin{bmatrix}f(\mathbf{w} +\epsilon\mathbf{e}_1) - f(\mathbf{w} -\epsilon\mathbf{e}_1)\\ f(\mathbf{w} +\epsilon\mathbf{e}_2) - f(\mathbf{w} -\epsilon\mathbf{e}_2)\\ \vdots \\ f(\mathbf{w} +\epsilon\mathbf{e}_m) - f(\mathbf{w} -\epsilon\mathbf{e}_m)\end{bmatrix}$$
"""

# ╔═╡ a6e494dc-2eea-4a0d-930e-3ef59182b730
md"""

#### Python `numpy` code -- gradient check
"""

# ╔═╡ 8a5a3a3d-d341-48e9-bcba-a16094f048c4
md"""

```python
def gradient_finite_difference(f, initial, eps=1e-6):
  initial = np.array(initial, dtype=float)
  n = len(initial)
  output = np.zeros(n)
  for i in range(n):
    ei = np.zeros(n)
    ei[i] = 1
    f1 = f(initial + eps * ei)
    f2 = f(initial - eps * ei)
    output[i] = (f1-f2)/(2*eps)
  output = output.reshape(n,1)
  return output

```

"""

# ╔═╡ 5a9c85ce-6c22-40da-ae6d-ed216ebfa175
md"""
#### `Julia` -- gradient finite difference
"""

# ╔═╡ 258131b3-3f64-4acc-a69f-e095ee48d42b
function gradient_finite_difference(f, x₀, ϵ = 1e-6)
	n = length(x₀)
	∇fx₀ = zeros(n)
	for i in 1:n
		eᵢ = zeros(n)
		eᵢ[i] = 1.0
		f₁ = f(x₀ + ϵ * eᵢ)
		f₂ = f(x₀ - ϵ * eᵢ)
		∇fx₀[i] = (f₁ - f₂) / (2 * ϵ)
	end
	return ∇fx₀
end;

# ╔═╡ 5f544e6e-2f92-4e12-b208-162b17b343d7
gradient_finite_difference(x -> x' * x, [1.0,2,3,4,5]) ## *e.g.* finding the gradient of x'x at [1, 2, 3, 4, 5]; should be 2x = [2,4,6,8,10]

# ╔═╡ 466d0d89-bf16-40f9-8eb2-2dfe70a8b227
md"""

# Case study: linear regression


"""

# ╔═╡ d94a06dc-77fb-4e27-a313-6589b5641519
md"""

## Demonstration:  toy dataset

"""

# ╔═╡ 1cd4daa9-4683-4b01-a4ed-7c55d769c861
md"""
## Gradient descent demonstration


"""

# ╔═╡ 730b641f-00bc-4b27-a832-166d1e2d75c9
md"""


## 


##### *Gradient descent* converges to almost the **same** result as normal equation's solution
"""

# ╔═╡ 8d9edfd5-6f97-4c4e-9464-3841078cc7c4
linear_reg_normal_eq(X, y) = X \ y;

# ╔═╡ 4788ccd2-62a9-423f-95f8-57507ccdd6e3
function ∇linear_reg(w; X, y) 
	-X' * (y - X* w) / length(y)
end;

# ╔═╡ 4b5ec74e-b89c-4bbf-9ac1-fa7d5b709f57
function loss(w, X, y) 
	error = X * w - y
	.5 * dot(error, error)/length(y)
end;

# ╔═╡ 69b9cca5-be80-434f-8e92-299bc12d5258
md"""

## What gradient descent is doing?


##### Let's dig deeper


* ##### consider sample ``i``'s gradient 
```math
\large
\begin{align}
\nabla l^{(i)}(\mathbf{w}_{old}) &=  -\underbrace{(y^{(i)} - \mathbf{w}_{old}^\top\mathbf{x}^{(i)})}_{\text{prediction diff: } e^{(i)}} \cdot \mathbf{x}^{(i)} \\
&= - e^{(i)}\cdot \mathbf{x}^{(i)}

\end{align}
```

* ##### gradient descent step

```math
\large
\mathbf{w}_{new} \leftarrow \mathbf{w}_{old} -\gamma  (-e^{(i)}\cdot \mathbf{x}^{(i)})
``` 

"""


# ╔═╡ e95f09c8-9364-4cf2-91a7-4e2dcbcd1053
md"""

## What gradient descent is doing?


##### Let's dig deeper


* ##### consider sample ``i``'s gradient 
```math
\large
\begin{align}
\nabla l^{(i)}(\mathbf{w}_{old}) &=  -\underbrace{(y^{(i)} - \mathbf{w}_{old}^\top\mathbf{x}^{(i)})}_{\text{prediction diff: } e^{(i)}} \mathbf{x}^{(i)} \\
&= - e^{(i)}\cdot \mathbf{x}^{(i)}

\end{align}
```

* ##### gradient descent step

```math
\large
\mathbf{w}_{new} \leftarrow \mathbf{w}_{old} -\gamma  (-e^{(i)}\cdot \mathbf{x}^{(i)})
``` 

```math
\Large
\boxed{\mathbf{w}_{new} \leftarrow \mathbf{w}_{old} + \gamma\cdot e^{(i)} \mathbf{x}^{(i)}\;\; \# \texttt{gradient step}}
```
* ##### where ``e^{(i)} = y^{(i)} - \mathbf{w}_{old}^\top\mathbf{x}^{(i)}`` is prediction error
"""


# ╔═╡ 9ccef028-d31d-45d6-a37f-77cd31f772dc
aside(tip(md"``\large e`` here stands for prediction **error** or difference."))

# ╔═╡ 42e01d0b-a093-44f9-8b1b-ae76b6e67b7d
md"""

## Interpretation 

```math
\Large
\boxed{\mathbf{w}_{new} \leftarrow \mathbf{w}_{old} + \gamma\cdot e\, \mathbf{x}\;\; \# \texttt{gradient step}}
```


* ##### where ``e = y - \mathbf{w}_{old}^\top\mathbf{x}`` is current prediction error

#### After the update, the *new prediction* 

```math
\Large
\hat{h}_{new}=\mathbf{w}_{new}^\top \mathbf{x} 
```

"""

# ╔═╡ 12b78d04-2cdb-44c0-8350-110c957e9757
md"""

## Interpretation 

```math
\Large
\boxed{\mathbf{w}_{new} \leftarrow \mathbf{w}_{old} + \gamma\cdot e\, \mathbf{x}\;\; \# \texttt{gradient step}}
```


* ##### where ``e = y - \mathbf{w}_{old}^\top\mathbf{x}`` is current prediction error

#### After the update, the *new prediction* 

```math
\Large
\begin{align}
\hat{h}_{new}&=\mathbf{w}_{new}^\top \mathbf{x} \\
&= (\mathbf{w}_{old} + \gamma\cdot e\, \mathbf{x})^\top \mathbf{x} 
\end{align}
```

"""

# ╔═╡ 46121e9d-3bef-48fd-a709-2b7613d0f8be
md"""

## Interpretation 

```math
\Large
\boxed{\mathbf{w}_{new} \leftarrow \mathbf{w}_{old} + \gamma\cdot e\, \mathbf{x}\;\; \# \texttt{gradient step}}
```


* ##### where ``e = y - \mathbf{w}_{old}^\top\mathbf{x}`` is current prediction error

#### After the update, the *new prediction* 

```math
\Large
\begin{align}
\hat{h}_{new}&=\mathbf{w}_{new}^\top \mathbf{x} \\
&= (\mathbf{w}_{old} + \gamma\cdot e\, \mathbf{x})^\top \mathbf{x} \\
&= \mathbf{w}_{old}^\top\mathbf{x} + \gamma \cdot e \cdot \mathbf{x}^\top\mathbf{x}
\end{align}
```

"""

# ╔═╡ 4488af42-7f4a-4c52-b8ca-1d86aeea9e69
md"""

## Interpretation 

```math
\Large
\boxed{\mathbf{w}_{new} \leftarrow \mathbf{w}_{old} + \gamma\cdot e\, \mathbf{x}\;\; \# \texttt{gradient step}}
```


* ##### where ``e = y - \mathbf{w}_{old}^\top\mathbf{x}`` is current prediction error

#### After the update, the *new prediction* 

```math
\Large
\begin{align}
\hat{h}_{new}&=\mathbf{w}_{new}^\top \mathbf{x} \\
&= (\mathbf{w}_{old} + \gamma\cdot e\, \mathbf{x})^\top \mathbf{x} \\
&= \mathbf{w}_{old}^\top\mathbf{x} + \gamma \cdot e \cdot \mathbf{x}^\top\mathbf{x} \\
&= \boxed{\hat{h}_{old} + \gamma \cdot e \cdot \mathbf{x}^\top\mathbf{x}}
\end{align}
```

"""

# ╔═╡ 37eb85ca-a07e-4409-a8ab-e27819552355
md"""

## Interpretation (cont.)

#### After the update, the *new prediction* 

```math
\Large
\boxed{\hat{h}_{new}= \hat{h}_{old} + e\cdot \gamma\|\mathbf{x}\|^2_2}
```

* ##### note that ``\gamma\|\mathbf{x}\|^2_2 > 0`` is positive

#### when error ``e \approx 0``,  the current model is perfect 

$$\Large \boxed{\hat{h}_{new}\approx \hat{h}_{old}}$$



"""

# ╔═╡ 1c5c58b6-8567-4ce0-af7a-fa84f4879d39
md"""

## Interpretation (cont.)

#### After the update, the *new prediction* 

```math
\Large
\boxed{\hat{h}_{new}= \hat{h}_{old} + e\cdot \gamma\|\mathbf{x}\|^2_2}
```

* ##### note that ``\gamma\|\mathbf{x}\|^2_2 > 0`` is positive


#### when error ``e \approx 0``,  the current model is perfect 

$$\Large \boxed{\hat{h}_{new}\approx \hat{h}_{old}}$$

\

#### when error ``e > 0``, the current model under predicts 

$$\Large \boxed{\hat{h}_{new}= \hat{h}_{old} + |\epsilon|}$$

* where ``|\epsilon| \triangleq |e| \cdot \gamma \|\mathbf{x}\|_2^2 >0``
\

#### when error ``e < 0``, the current model over predicts 

$$\Large \boxed{\hat{h}_{new}= \hat{h}_{old} - |\epsilon|}$$


* where ``|\epsilon| \triangleq |e| \cdot \gamma \|\mathbf{x}\|_2^2 >0``
"""

# ╔═╡ 61263a4c-f756-439e-a2bf-b7d7bc6d5b66
# md"""

# ## A different loss?


# Auto-differentiation package becomes very useful for customised loss


# ```math
# \text{loss}_p(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n |y^{(i)} - \mathbf{w}^{\top} \mathbf{x}^{(i)}|^p 
# ```

# * for example train a regression with ``p=4``
# * one only needs to define the loss and leave the gradient derivation to auto-diff

# Note that the optimisation cannot be solved analytically anymore
# * but we can apply gradient descent as usual
# """

# ╔═╡ de62b0cb-0ca8-4b6c-9edb-409328bcf540
begin
	Random.seed!(123)
	nobs_new = 20
	x_train_new = randn(nobs_new, 2)
	y_train_new = x_train_new * ones(2) + randn(nobs_new)/5
	x_train_new_bad = copy(x_train_new)
	x_train_new_bad[:, end] = 10 * x_train_new[:, end]
	# x_train_new \ y_train_new
end;

# ╔═╡ 0ed56488-7e32-474a-a631-3e42f3481cf2
md"""

## The need of data pre-processing 


#### Let's inflate second feature ``\times 100``

```math
\Large
\mathbf{x}_2 \leftarrow 100 \times \mathbf{x}_2 
```

* ##### scale the second feature (second column) by 100

"""

# ╔═╡ 57b7e86d-38d8-4508-a410-465a3f14d6f5
TwoColumn(
md"""


#### Original training data ``\mathbf{X}``


$(latexify_md(round.(x_train_new[1:10, :], digits=3)))
""",

	md"""
#### Changed data set ``\mathbf{X}_{scaled}``
$(latexify_md(round.(x_train_new_bad[1:10, :], digits=2)))
	"""

	
)

# ╔═╡ 3813e242-cef7-4c8f-a97f-842f67ba86ab
md"Add scaled dataset: $(@bind add_scaled_data CheckBox(false))"

# ╔═╡ 6ae54ba5-2d2d-4270-9617-51061c68271a
let
	plotly()
	plt = plot(x_train_new[:, 1], x_train_new[:, 2], y_train_new, st=:scatter3d, ms=3, label="original data")
	# plot!(-1:0.1:1, -1:0.1:1, (x,y) -> dot([x, y], ones(2)), st=:surface)
	surface!(-3:0.5:3, -3:0.5:3.0, (x1, x2) -> dot([x1, x2], [1,1]), c=1,  colorbar=false, xlabel="x1", ylabel="x2", zlabel="y", alpha=0.5, title="Original training data", legendfontsize=12)


	if add_scaled_data
		plot!(x_train_new_bad[:, 1], x_train_new_bad[:, 2], y_train_new, st=:scatter3d, ms=3, mc=2, label="scaled data", title="Original & scaled training")
	
		ww = x_train_new_bad \ y_train_new
		surface!(-3:0.5:3, range(extrema(x_train_new_bad[:, 2]).+ (-2, 2)..., 10), (x1, x2) -> dot([x1, x2], ww), c=2,  colorbar=false, xlabel="x1", ylabel="x2", zlabel="y", alpha=0.5)
	end
	
	plt
end

# ╔═╡ bd23fdab-0651-47fd-bcd0-68db16275e32
md"""

## Implication on loss 
"""

# ╔═╡ b6946dae-9b8d-4666-9616-00edcc1f524f
let
	gr()
	plt1 = plot(range(-3,5, 100), range(-3, 5, 100), (x,y) -> loss([x, y], x_train_new, y_train_new), c=:jet, colorbar=false, nlevels=25, st=:surface, ratio=1,xlabel=L"w_1", ylabel=L"w_2", title="Original data loss surface")
	
	plt2 = plot(range(-5,3*5, 100), range(-3, 3, 100), (x,y) -> loss([x, y], x_train_new_bad, y_train_new), c=:jet, colorbar=false, nlevels=25, st=:surface, ratio=1,xlabel=L"w_1", ylabel=L"w_2", title="Scaled data loss surface")

	plot(plt1, plt2)
end

# ╔═╡ 2e704205-791d-46c7-9a01-c1e77e178717
let
	gr()
	plt1 = plot(range(-3,5, 100), range(-3, 5, 100), (x,y) -> loss([x, y], x_train_new, y_train_new), c=:jet, colorbar=false, nlevels=25, st=:contour, ratio=1,xlabel=L"w_1", ylabel=L"w_2", title="Original data loss")
	
	plt2 = plot(range(-10,3*5, 100), range(-3, 3, 100), (x,y) -> loss([x, y], x_train_new_bad, y_train_new), c=:jet, colorbar=false, nlevels=25, st=:contour, ratio=1,xlabel=L"w_1", ylabel=L"w_2", title="Scaled data loss")

	plot(plt1, plt2, size=(800,400), labelfontsize=15)
end

# ╔═╡ da8ee92e-275f-47bb-a60e-855ce686c38d
md"""

## Gradient descent struggles


##### One size ``\gamma`` does not fit all


```math
\begin{bmatrix}
w_0\\
w_1\\
\vdots\\

w_m
\end{bmatrix} = \begin{bmatrix}
w_0\\
w_1\\
\vdots\\

w_m
\end{bmatrix} - \colorbox{orange}{$\LARGE \gamma$} \begin{bmatrix}
\frac{\partial f}{\partial w_0}(\mathbf{w})\\
\frac{\partial f}{\partial w_1}(\mathbf{w})\\
\vdots\\

\frac{\partial f}{\partial w_m}(\mathbf{w})
\end{bmatrix}
```
"""

# ╔═╡ 42e7bc1a-7380-47f6-aa2d-c49cdaff0b08
md"""

## Solution: _feature normalisation_


#### For each feature (columns), normalise it 

$$\Large
\begin{align}
\Large
\text{scaled } \tilde{x}^{(i)}_j&=\frac{x^{(i)}_j-\mu_j}{\sigma_j}\end{align}$$


* ##### where 

$$\Large
\begin{align}
{\mu_j} &= \frac{1}{n}\sum_{i=1}^n{x^{(i)}_j}
\\
\Large
\sigma_j &= \sqrt{\frac{1}{n}\sum_{i=1}^n{(x^{(i)}_j - \mu_j)^2}}\end{align}$$
"""

# ╔═╡ 487e5448-8565-440e-a5ae-56dbe3c5cadf
md"""Zero centered the feature: $(@bind add_zero_centered CheckBox(false)), Standardise the feature: $(@bind add_stand CheckBox(false)) """

# ╔═╡ 35ddd086-900a-4f44-9aae-59a88d27774f
let
	gr()
	Random.seed!(234)

	x_train_bad = randn(30, 2) .* [1 5]
	L = [cos(π/4) -sin(π/4); sin(π/4) cos(π/4)]
	x_train_bad = x_train_bad *  L'  .+ [10 10]                                                                                                                                                                                                                                                            
	plt = scatter(x_train_bad[:, 1], x_train_bad[:,2], xlim = [-10, maximum(x_train_bad[:, 1])+2], ylim =[-10, maximum(x_train_bad[:, 2]) +2], label="Initial training data", ratio=0.5, xlabel=L"x_1", ylabel=L"x_2", framestyle =:zerolines, legendfontsize=10, alpha=0.7, r=1)
	center = mean(x_train_bad, dims=1)
	scatter!([center[1]], [center[2]], ms=8, mc=1, m=:x, markerstrokewidth=6,label="")

	if add_zero_centered
		# plt = scatter(x_train_bad[:, 1], x_train_bad[:,2], xlim = [-5, maximum(x_train_bad[:, 1])+2], ylim =[-5, maximum(x_train_bad[:, 2]) +2], label="Initial training data", ratio=0.5, xlabel=L"x_1", ylabel=L"x_2", framestyle =:zerolines, legendfontsize=10, alpha=0.7)
		# center = mean(x_train_bad, dims=1)
		scatter!([center[1]], [center[2]], ms=8, mc=1, m=:x, markerstrokewidth=6,label="")
		x_train_new_t = x_train_bad .- mean(x_train_bad, dims=1)

		# new_center = mean(x_train_bad, dims=1)
		scatter!(x_train_new_t[:, 1] ,c=2, alpha=0.5, x_train_new_t[:,2], label="Zero centered data")
		scatter!([0], [0], ms=8, mc=2, m=:x, markerstrokewidth=6,label="")

		if add_stand
		
			x_train_new_t = x_train_new_t ./ std(x_train_bad, dims=1)
			# # scatter!()
			scatter!(x_train_new_t[:, 1] , c=3, ms=5, x_train_new_t[:,2], label="Normalised data")
		end
	end
	
	
	plt
end

# ╔═╡ fe721301-1e77-4c89-8592-17d8c88776a7
md"""

## After the transform: effect on `Loss` 


"""

# ╔═╡ fbf9a928-fdc7-4f38-9b93-4bf74c20ad5b
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/loss5.png" width = "800"/></center>""" 

# ╔═╡ b76084a8-7722-4db3-9795-e1ce03c50b9d
md"""

## Predict ``h(\mathbf{x}_{test})``


### To predict ``\mathbf{x}_{test}``, you need to apply the _same transform_


$$\Large
\begin{align}
\Large
\tilde{\mathbf{x}}^{(i)}_{test}&=\frac{\mathbf{x}_{test}^{(i)}-\boldsymbol{\mu}}{\boldsymbol{\sigma}}\end{align}$$


* *i.e.* the same ``\mu, \sigma`` obtained in the training data
"""

# ╔═╡ 4602e84e-435e-425e-9fe0-a2764bd7ae47
md"""

# Auto-diff: compute gradient efficiently


#### Apart from 

* #### *method 1*: manual derivation
  * ##### error-prone & impractical

#### There are a few options

* #### *method 2*: finite difference method (just discussed)
  * ##### inaccurate (rounding errors) and slow

#### The go-to method

* #### *method 3*: auto-differentiation
"""

# ╔═╡ d3d2f0f0-1a16-495c-b3e5-7b0eec73ada5
md"""

## What is Auto-differentiation ?

\

#### Programs that differentiates functions automatically

* #### there are two modes
   * #### forward mode 
   * #### reverse mode (*e.g.* backpropagation)

\

#### We will discuss auto-diff later in the course; today is a preview 
"""

# ╔═╡ fb5aa103-3c24-43c9-a2b4-0c0665a5476e
md"""

## Auto-diff in Python


### A lot of frameworks out there

* #### `Autograd` (or `Jax`) (we will see this today)


* #### `PyTorch`


* #### `TensorFlow`


* ##### and so on
"""

# ╔═╡ 58f37e36-755e-4f5a-af96-183a9a227118
md"""

## Python's `Autograd` example


#### Import the package

```python
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever
```


"""

# ╔═╡ 508e8a33-8b94-4807-9118-0a3f6d4cc1f5
md"""

## Python's `Autograd` example


#### Import the package

```python
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever
```

#### Define the loss

```math
\large
l(\mathbf{w})= \frac{1}{2n}\sum_{i=1}^n (y^{(i)} - \mathbf{w}^\top\mathbf{x}^{(i)})^2 = \frac{1}{2n}(\mathbf{y} -\mathbf{Xw})^\top(\mathbf{y} -\mathbf{Xw})
```


```python
def loss(w, X, y):
	y_pred = X @ w
	error = y - y_pred
	return 0.5 / len(y) * np.sum(error ** 2)
```


"""

# ╔═╡ f43a8744-81f7-4da7-ae76-d9203d91e81a
md"""

## Python's `Autograd` example


#### Import the package

```python
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever
```

#### Define the loss

```math
\large
l(\mathbf{w})= \frac{1}{2n}\sum_{i=1}^n (y^{(i)} - \mathbf{w}^\top\mathbf{x}^{(i)})^2 = \frac{1}{2n}(\mathbf{y} -\mathbf{Xw})^\top(\mathbf{y} -\mathbf{Xw})
```

```python
def loss(w, X, y):
	y_pred = X @ w
	error = y - y_pred
	return 0.5 / len(y) * np.sum(error ** 2)
```



#### Compute the gradient ``\nabla \text{loss}(\mathbf{w})``


```python
gw = grad(loss, 0)(w0, Xtrain, ytrain) # dloss/dw
```

* ``0``: the first argument of ``\texttt{loss}``

"""

# ╔═╡ 93d3d44c-5072-4412-b1c6-adc810f20235
md"""

## Python's `Autograd` example


#### Import the package

```python
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever
```

#### Define the loss

```math
\large
l(\mathbf{w})= \frac{1}{2n}\sum_{i=1}^n (y^{(i)} - \mathbf{w}^\top\mathbf{x}^{(i)})^2 = \frac{1}{2n}(\mathbf{y} -\mathbf{Xw})^\top(\mathbf{y} -\mathbf{Xw})
```

```python
def loss(w, X, y):
	y_pred = X @ w
	error = y - y_pred
	return 0.5 / len(y) * np.sum(error ** 2)
```



#### Compute the gradient ``\nabla \text{loss}(\mathbf{w})``


```python
gw = grad(loss, 0)(w0, Xtrain, ytrain) # dloss/dw
```

* ``0``: the first argument of ``\texttt{loss}``


```python
gX = grad(loss, 1)(w0, Xtrain, ytrain) # dloss/dX
gy = grad(loss, 2)(w0, Xtrain, ytrain) # dloss/dy
```

* ``1,2`` index the second/third (we rarely need their gradients though)

"""

# ╔═╡ 04131640-f043-4582-b143-8c7cd214d52a
md"""

## Auto-diff in `Julia`




### A lot of options as well

* #### `Zygote.jl` 


* #### `ForwardDiff.jl`


* #### `ReverseDiff.jl`


* ##### and so on
"""

# ╔═╡ c72fa154-96cf-4079-aea5-e1b4aa8e2cf2
md"""

## `Zygote.jl` example

"""

# ╔═╡ b7236763-ae9e-427c-9a0f-94cadbc34b8b
md"""

### `Using` the package
"""

# ╔═╡ c9324c6b-9ca5-48fd-9ee5-e2f1ec744c60
md"""

### Implement the loss
"""

# ╔═╡ bc86509b-2164-46d1-809a-7cadc6d5d53d
function ℓ(w, X, y)
	ŷ = X * w
	error = y -ŷ
	n = length(y)
	0.5 * sum(error.^2)/n
end

# ╔═╡ f5988f97-50ad-4f32-9fb5-69fc0ed4bc73
md"""
### Compute the gradient ``\ell(\mathbf{w}_0)`` 

* at ``\mathbf{w}_0= [1,2,3]``
"""

# ╔═╡ 373b1b1d-1bd8-4750-86c5-52810456aeed
w₀ = [1, 2 ,3]

# ╔═╡ 97ed5d31-7832-404f-907f-779260456c57
md"""
#### Let's check it

* ##### recall the gradient is


```math
\large
\nabla \ell(\mathbf{w}) = -\frac{1}{n}\mathbf{X}^\top (\mathbf{y} - \mathbf{Xw}) 
```
"""

# ╔═╡ 75510fd8-c1a9-4eda-98b3-61dedb243b78
md"""

# Another case study : non-convex optimisation
"""

# ╔═╡ 4c1865c6-a044-4c65-8388-09380ee86844
md"""

## Gabor function 



```math
\Large
f(x; \mathbf{w}) = \sin(w_0 + 0.06 \cdot w_1 x) \cdot \exp\left \{-\frac{(w_0 + 0.06 \cdot w_1 x)^2}{32.0} \right \}
```

* ##### the parameters: ``w_0`` and ``w_1``

"""

# ╔═╡ 2095d604-e9d5-46f3-b152-e8e1f767db60
begin
	function gabor_f(x; w)
		z = w[1] .+ 0.06 * w[2] * x 
		@. sin(z) * exp(- z^2 /32.0) 
	end

end;

# ╔═╡ 1f0a9fae-e314-401d-967d-ab7f5b9990f9
let
	gr()
	xs = range(-15, 15, 500)
	ws = [[-5, 25], [20,40], 10 * [-5, 25], [-20,40]]
	plts = []
	lw = 1.5
	for (i, w) in enumerate(ws)
		if i == 1
			plt = plot(xs, (x)-> gabor_f(x; w=w), lw=lw, title=L"[{w}_0= %$(w[1]),\;\; w_1=%$(w[2])]", xlabel=L"x", ylabel=L"f(x)", titlefontsize=10, label=L"f(x;\mathbf{w})", labelfontsize=8, tickfontsize=6)
			
			push!(plts, plt)
		else
			plt = plot(xs, (x)-> gabor_f(x; w=w), lw=lw, title=L"[{w}_0= %$(w[1]), w_1=%$(w[2])]", titlefontsize=10, xlabel=L"x", ylabel=L"f(x)", label=L"f(x;\mathbf{w})", labelfontsize=8, tickfontsize=6)
			push!(plts, plt)
		end

	end

	plot(plts..., layout=(2,2), size=(700,400))
	# plot((x) -> gabor_f(x; w=w))
end

# ╔═╡ e0d43282-8197-414b-aa9c-d83cdb8b9899
md"""

## Gabor function regression

"""

# ╔═╡ 88d3904c-b915-4396-be48-ae0f2bf0cc6b
begin
	gr()
	Random.seed!(111)
	nobs_gab = 50
	x_train_gab = rand(50) * 30 .- 15 
	true_w_gab = [0.0, 16.6]
	σ_noise_gab = 0.05
	y_train_gab = gabor_f(x_train_gab; w=true_w_gab) + σ_noise_gab * randn(50)

end;

# ╔═╡ 3f89f4ae-56de-4906-b760-fdb2b4a22a88
let
	true_w = true_w_gab
	x_train = x_train_gab
	y_train = y_train_gab
	gr()
	plot(-15:0.1:15, (x) -> gabor_f(x; w=true_w), lw=1.5, label="true function", xlabel=L"x", ylabel=L"y")

	annotate!([-10], [.75], text("true "*L"\mathbf{w} = %$((true_w))^\top", :blue, :"Computer Modern", 13))
	scatter!(x_train, y_train, ms=3, c=1, label="training data", title="Regression dataset with a Gabor function")
end

# ╔═╡ 995c0495-3feb-4a68-8da6-e2733890abff
md"""

## Gabor function regression

"""

# ╔═╡ 632be70e-beb8-44b6-864b-d02de7b904c6
TwoColumnWideRight(md"""


#### Learning: 
##### - minimise MSE loss
\

```math
\begin{align}
\large
\hat{\mathbf{w}} 
&\leftarrow \arg\min_{\mathbf{w}} \underbrace{\frac{1}{2n} \sum_{i=1}^n (\hat{y}^{(i)}- y^{(i)})^2}_{\text{loss: }L(\mathbf{w})}
\end{align}
```

##### _where_ 
  * ##### ``\hat{y}^{(i)} = f(x^{(i)};\mathbf{w})``
  * ##### parameter: ``\mathbf{w}``

""", let
	x_train = x_train_gab
	y_train = y_train_gab
	true_w = true_w_gab
	gr()
	Random.seed!(123)
	ww = true_w /1.5
	plt = plot(-15:0.1:15, (x) -> gabor_f(x; w=ww), lw=1.5, lc=:gray, label="Prediction function "*L"f(x;\mathbf{w})", legend=:outerbottom, size=(450,350), titlefontsize=14, legendfontsize=8)
	# step = 2
	plot_size = 35
	x_idx = 1:(min(plot_size, length(x_train)))
	ŷs = gabor_f(x_train; w=ww)

	mseloss = .5 * mean((ŷs - y_train).^2)
	scatter!(x_train[x_idx], y_train[x_idx], ms=4, c=1, label="training data", title="Regression loss: "*L"L(\mathbf{w})"*L"=%$(round(mseloss, digits=2))", yaxis=false, framestyle=:origin)
	for i in x_idx
		plot!([x_train[i], x_train[i]], [y_train[i], ŷs[i]], lw=1.5,alpha=0.5, ls=:dash, arrow=arrow(:head, :simple, 0.05, 0.05), lc=:gray,  label="")
		
	end
	plt

	# annotate!([-10], [.75], text("true "*L"\mathbf{w} = %$((true_w))^\top", :blue, :"Computer Modern", 13))
	# 
end)

# ╔═╡ f2a85f4b-7e6d-46d3-97dc-c148a412886a
md"""

## Gradient descent


#### The optimisation *CANNOT* be solved analytically
```math
\Large
\begin{align}
\hat{\mathbf{w}} 
&\leftarrow \arg\min_{w_0, w_1} \underbrace{\frac{1}{2n} \sum_{i=1}^n (\hat{y}^{(i)}- y^{(i)})^2}_{L(\mathbf{w})}
\end{align}
```
* ##### where ``\hat{y}^{(i)} =  \sin(w_0 + 0.06 \cdot w_1 x^{(i)}) \cdot \exp\left \{-\frac{(w_0 + 0.06 \cdot w_1 x^{(i)})^2}{32.0} \right \}``

##### _In other words_, the following system of equations

```math
\Large
\nabla_{\mathbf{w}} L(\mathbf{w}) =\mathbf{0}
```

* ##### _has no closed-form analytical solution!_


## The loss surface
"""

# ╔═╡ 89064ca7-6aff-40b4-a414-aaa23b3c9d1e
md"""



##### There are *multiple local minimums* 
* ##### _the darker the smaller the loss_
"""

# ╔═╡ e5898a5a-dc97-449c-96b0-72c2d5f9ad85
md"""

## Aside: convexity

#### Convex ``f(\mathbf{x})``: if ``\mathbf{x}_1, \mathbf{x}_2 \in \text{dom} f``, and ``\theta \in [0,1]``, we have 

```math
\Large
f(\theta \mathbf{x}_1 +(1-\theta) \mathbf{x}_2) \leq \theta f(\mathbf{x}_1) + (1-\theta) f(\mathbf{x}_2)
```

"""

# ╔═╡ 65de1dd5-d6d6-42a8-b41c-53498abe0c9f
html"""<center><img src="https://tisp.indigits.com/_images/convex_function.png" width = "500"/></center>"""

# ╔═╡ 1a9ad655-027e-4f59-a0e6-48d12a4abe17
md"""

```math
\Large
\text{Convex} \Rightarrow \textit{local } \text{minimums are } \textit{global } \text{minimums}
```
"""

# ╔═╡ c5368d27-7710-402e-97d1-047f3949298a
md"""
## Gradient descent



### But we can resort to *gradient descent*!


##### _Show me the code_

```python
w0 = np.random.randn(2) # random guess
gradw = evaluate_grad(w0, x_train, y_train) # gradient
lr = 0.1 # learning rate 
while norm(gradw) > epsilon:
	gradw = evaluate_grad(w0, x_train, y_train)
	w0 = w0 - lr * gradw
	...
```
"""

# ╔═╡ 5dbc7275-7e06-4c07-b751-ccc52ad3e168
md"""
## Demonstration

##### Random start with ``\mathbf{w}_0= [0, 8]^\top``
"""

# ╔═╡ db83f381-cb61-4752-91d8-6ef19b6a3dca
md"""
## Demonstration

##### Random start with ``\mathbf{w}_0= [2.8, 12]^\top``
"""

# ╔═╡ 0b4d6238-e21b-4998-9825-e7da2001cb03
md"""
## Demonstration

##### Random start with ``\mathbf{w}_0= [-7.5, 12]^\top``
"""

# ╔═╡ cfb74a46-252c-4c71-b569-0b901ba3d615
md"""
## Demonstration

##### Random start with ``\mathbf{w}_0= [-5, 5]^\top``
"""

# ╔═╡ c5f788a4-4b6a-4fa0-90c4-dca98af59f8f
md"""
## Demonstration

##### Random start with ``\mathbf{w}_0= [6, 10]^\top``
"""

# ╔═╡ 2b975333-8b48-4848-a261-4a15c6589658
md"""

## Loss trajectories _vs_ local minimums


#### Note the true parameter is ``\mathbf{w}_{\text{true}} = [0, 16.5]``
* ##### where it converges depends on ``\mathbf{w}_0``

"""

# ╔═╡ 8014a3e8-19b6-4086-8fce-bcfe16a16991
md"""


## A remedy: ``L_2`` regularisation


#### Apply a penalty term to regularise the loss 

```math
\Large
\ell(\mathbf{w}) = \underbrace{L(\mathbf{w}) }_{\normalsize\texttt{old loss}}+ \underbrace{\lambda \mathbf{w}^\top\mathbf{w}}_{\normalsize L_2\texttt{ penalty}}
```

* ####  it smoothes the loss surface (less bumpy)
  * ##### minimums shrinks towards ``\mathbf{0}``
  * ##### some local minimums disappear
"""

# ╔═╡ 536f04f3-045a-445d-81ae-2644f78dc1eb
function forward_loss(w, xs, ys)
	# forward pass
	ŷs = gabor_f(xs; w=w)

	.5 * mean((ŷs - ys).^2)
end

# ╔═╡ b67461c2-6a96-4e49-9fcc-c5604b14fdbb
let
	x_train = x_train_gab
	y_train = y_train_gab
	true_w = true_w_gab
	# gr()
	plotly()
	plt1 = plot(-10:0.1:10, 2:0.1:25, (w0, w1) -> forward_loss([w0, w1], x_train, y_train), st=:contour, levels=20, fill=true, alpha=0.99, xlabel="w0", ylabel="w1", colorbar=true, title="" )

	plt2=plot(-10:0.1:10, 2:0.1:25, (w0, w1) -> forward_loss([w0, w1], x_train, y_train), st=:surface, levels=20, fill=true, alpha=0.99, xlabel="x0", ylabel="w1", colorbar=false, title="")
	plot(plt1, plt2, size=(720,350))
end

# ╔═╡ 8c6bf500-5481-422d-bdfc-c0738be80bc3
let
	x_train = x_train_gab
	y_train = y_train_gab
	true_w = true_w_gab

	gr()
	plt = plot(-10:0.1:10, 2:0.1:25, (w0, w1) -> forward_loss([w0, w1], x_train, y_train), st=:contour, levels=20, fill=true, alpha=0.9, xlabel=L"w_0", ylabel=L"w_1", colorbar=false, title="Loss")
λ = 0.0011
plt_loss = plot(-10:0.5:10, -10:0.5:10, (w0, w1) -> 0.5 * λ * (w0^2+w1^2), st=:contour, levels=20, fill=true, alpha=0.9, xlabel=L"w_0", ylabel=L"w_1", colorbar=false, title=L"L_2"*" penalty")
	
	# plt2 = plot(-2:0.1:10, 2:0.1:25, (w0, w1) -> forward_loss([w0, w1], x_train, y_train) + 0.5 * λ * (w0^2+w1^2) , levels=12,st=:contour, fill=true, alpha=0.9, xlabel=L"w_0", ylabel=L"w_1", colorbar=false, title=L"L_2"*" regularised loss")
	# l = @layout [a{0.5w} b{0.5w} ; c{0.5w}]
	plt1 = plot(plt, plt_loss, layout=(1,2), size=(700,350))

end

# ╔═╡ dfcbda48-4bde-418d-83ec-5d8dc1b5ab94
function plot_rst(w; true_w = true_w_gab, x_train = x_train_gab, y_train=y_train_gab, add_truef=true)
	# plt = plot(-15:0.1:15, (x) -> gabor_f(x; w = true_w), lw=1, label="true function")
	plt = plot(-15:0.1:15, (x) -> gabor_f(x; w = w), lw=2, lc=2,label="fitted function")
	if add_truef
		plot!(-15:0.1:15, (x) -> gabor_f(x; w = true_w),lc=1, lw=1.5, label="true function")
	end
	scatter!(x_train, y_train, ms=3, c=1, label="training data")
	return plt
end;

# ╔═╡ 6d4600f7-baa4-41e3-bf0a-8c5c13e55f64
begin 

	function train(max_iters; w_init=zeros(2), γ = 0.1, x_train=x_train_gab, y_train=y_train_gab, λ = 0.0)
		ws = zeros(2, max_iters+1)
		ws[:, 1] = w₀ = copy(w_init)
		losses = []
		for i in 1:max_iters
			loss, grad = withgradient(w₀) do w
				ŷs = gabor_f(x_train; w = w)
				l2_loss = 0.5 * λ * sum(w .^2)
				.5 * mean((ŷs - y_train).^2) + l2_loss 
			end
			push!(losses, loss)
			w₀ .-= γ * grad[1]
			ws[:, i+1] = w₀
			# println(w₀)
		end
		return losses, ws
	end
end

# ╔═╡ 19b1a466-13ba-4b79-97c4-36ea007ad219
ws_l2, w_l2, ws_mle, w_mle = let
	x_train = x_train_gab
	y_train = y_train_gab
	true_w = true_w_gab
	_, ws_l2 = train(5000; w_init=true_w, λ = 0.0011)
	w_l2 = ws_l2[:, end];

	_, ws_mle = train(5000; w_init=true_w, λ = 0.0)
	w_mle = ws_mle[:, end];

	ws_l2, w_l2, ws_mle, w_mle
end;

# ╔═╡ e00b4813-2df0-48fd-9322-80ccb3d0f115
let
	x_train = x_train_gab
	y_train = y_train_gab
	true_w = true_w_gab
	gr()
	ms = 5
	plt = plot(-10:0.1:10, 2:0.1:25, (w0, w1) -> forward_loss([w0, w1], x_train, y_train), st=:contour, levels=20,alpha=0.9, fill=true, xlabel=L"w_0", ylabel=L"w_1", colorbar=false, title="Loss")

	scatter!([w_mle[1]], [w_mle[2]], ms = ms, mc=3, markershape=:diamond,label="minimum", xlim = (-10, 10), ylim = (2, 25))
	
	λ = 0.0011
	
	plt2 = plot(-10:0.1:10, 2:0.1:25, (w0, w1) -> forward_loss([w0, w1], x_train, y_train) + 0.5 * λ * (w0^2+w1^2) , levels=12,st=:contour, fill=true, alpha=0.8, xlabel=L"w_0", ylabel=L"w_1", colorbar=false, title="Loss"*L"\,+\,L_2"*" penalty")
	# scatter!([w_mle[1]], [w_mle[2]], xs = 8, mc=3, markershape=:diamond,label="minimum", xlim = (-10, 10), ylim = (2, 25))
	plot!([w_mle[1], w_l2[1]], [w_mle[2], w_l2[2]], lw=1, ls=:dash, arrow=arrow(:head, :simple), lc=:gray,  label="", xlim = (-10, 10), ylim = (2, 25))
	scatter!([w_mle[1]], [w_mle[2]], ms = ms, mc=3, markershape=:diamond,label="minimum")
	scatter!([w_l2[1]], [w_l2[2]],  mc=4, ms=ms, markershape=:diamond,label=L"L_2"*" regularised minimum", xlim = (-10, 10), ylim = (2, 25))
	vv = w_l2 - w_mle
	# quiver!([w_mle[1]], [w_mle[2]], quiver= ([vv[1]], [vv[2]]))
	plot(plt, plt2, layout=(1,2), size=(700,350))

end

# ╔═╡ 1e4f6557-c9f5-49af-8a88-140d721b70c5
function produce_anim(w_traces, losses; step=200, x_train = x_train_gab, y_train = y_train_gab, true_w=true_w_gab)
	gr()
	plt = plot(-10:0.1:10, 2.5:0.1:25, (w0, w1) -> forward_loss([w0, w1], x_train, y_train), st=:contour, fill=false, alpha=0.9, c=:jet, xlabel=L"w_0", ylabel=L"w_1", colorbar=false)
	scatter!(plt, true_w[1:1], true_w[2:2], label="true w", markershape=:x, markerstrokewidth=2, markersize=6, xlim=[-10,10], ylim =[2.5, 25])
	traces = w_traces
	wt = traces[:, 1]
	plt2 = plot_rst(wt; add_truef=true)

	anim = @animate for t in 2:step:size(traces)[2]
		plot!(plt, [traces[1, t]], [traces[2, t]], st=:scatter, color=2, label="", markersize=3, title="Iteration: $(t)"*"; loss=$(round(losses[t];digits=2))")
		# plot!(plt, [wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow,1, :gray), label="", title="Iteration: $(t)")

		# plot!(plt, [wt[1], traces[1, t]], [wt[2], traces[2, t]], seriestype = :path, arrow=(:arrow, :head, 0.01, 0.01), linealpha=1, linecolor=:gray, label="", title="Iteration: $(t)")
		plt2 = plot_rst(wt; add_truef=true)

		title!(plt2, "Iteration: $(t)"*"; loss=$(round(losses[t];digits=2))")
		plot!(plt2, xlabel=L"x", ylabel=L"y")
		plot(plt, plt2, layout=(1,2), size=(800,380))
		wt = traces[1:2, t]
	end
	# wt = traces[:, 1]
	
	# anim2 = @animate for t in 2:step:size(traces)[2]
	# 	plot_rst(wt; add_truef=true)

	# 	title!("Iteration: $(t)")
	# 	# plot!(-15:0.1:15, (x) -> gabor_f(x; w = wt), lw=2, lc=2,label="")

	# 	wt = traces[1:2, t]
	# end

	return anim
end

# ╔═╡ c1411cf0-8982-4e0f-a01a-ae4ba22e16b2
anims, losses_all, weights_gab = let
	# x_train = x_train_gab
	# y_train = y_train_gab
	# true_w = true_w_gab
	## run GD multiple times from different initial locations
	losses_all = Dict();

	ws_init = [[0.0, 8], [2.8, 12], [-7.5, 12], [-5.0, 5], [6.0, 10]] 

	max_iters = 5_000
	anims = []
	for w in ws_init
		losses_, ws_trace = train(max_iters; w_init= w)
		anim = produce_anim(ws_trace, losses_)
		push!(anims, anim)
		losses_all[[w, ws_trace[:, end]]] = losses_
	end

	weights = (losses_all |> keys) |> collect
	# gif(anim1, fps=5)
	anims, losses_all, weights
end;

# ╔═╡ ad5fdaf4-21ee-4328-aee2-a5c73b1a5c2a
let
	# w_init = [0.0, 8]
	# losses_, ws_trace = train(5000; w_init= w_init)
	# anim1 = produce_anim(ws_trace, losses_)

	# losses_all[[w_init, ws_trace[:, end]]] = losses_
	gif(anims[1], fps=5)
	# gif(anim2, fps=5)
end

# ╔═╡ e833ae14-1bc4-49b7-a9cd-eea0554bb47f
let
	# w_init = Float64.([2.8, 12])
	# losses_, ws_trace = train(5000; w_init= w_init)
	# anim1 = produce_anim(ws_trace, losses_)
	# losses_all[[w_init, ws_trace[:, end]]] = losses_
	gif(anims[2], fps=5)
	# gif(anim2, fps=5)
end

# ╔═╡ a6427ed6-7c4e-4165-a819-7d14be18f513
let
	# w_init = Float64.([-7.5, 12])
	# losses_, ws_trace = train(5000; w_init= w_init)
	# anim1 = produce_anim(ws_trace, losses_)
	# losses_all[[w_init, ws_trace[:, end]]] = losses_
	# gif(anim1, fps=5)

	gif(anims[3], fps=5)
	# gif(anim2, fps=5)
end

# ╔═╡ 57c0ac44-60ee-425f-8189-c231fb1cd700
let
	# w_init = Float64.([-5.0, 5])
	# losses_, ws_trace = train(5000; w_init= w_init)
	# anim1 = produce_anim(ws_trace, losses_)
	# losses_all[[w_init, ws_trace[:, end]]] = losses_
	# gif(anim1, fps=5)
	# gif(anim2, fps=5)
	gif(anims[4], fps=5)
end

# ╔═╡ f17b7b3c-150a-48b2-842b-66ffbb36f2e6
let
	# w_init = Float64.([6, 10])
	# losses_, ws_trace = train(5000; w_init= w_init)
	# anim1 = produce_anim(ws_trace, losses_)
	# losses_all[[w_init, ws_trace[:, end]]] = losses_
	# gif(anim1, fps=5)
	# gif(anim2, fps=5)
	gif(anims[5], fps=5)
end

# ╔═╡ ebd84273-2610-4ad4-810d-664d205c6591
# highlight_i = 1:((losses_all |> keys ) |> length);
md"Select curve : $(@bind highlight_i Select(1:((losses_all |> keys ) |> length)))"

# ╔═╡ 0552fef1-2a56-4041-956f-6692ca0be446
TwoColumn(let
	gr()
	plt_losses = plot(size=(350,400), legend=:outerbottom) 

	cs = 2:(length(losses_all)+1)
	for (i, w) in enumerate(keys(losses_all))
		wid = (i == highlight_i) ?  3 : 1
		plot!(losses_all[w][1:10:end], xscale=:identity, label=L"\mathbf{w}_0=%$(round.(w[1]; digits=1));"*"  "*L"\mathbf{w}_{\texttt{end}}=%$(round.(w[2]; digits=1))", lw=wid, lc= cs[i], xlabel="Iteration "*L"\times\, 10", ylabel="loss", legendfontsize=8, titlefontsize=9, labelfontsize=8)
	end

	title!("Different training runs")
	plt_losses
end, let
	weights = weights_gab
	x_train = x_train_gab
	y_train = y_train_gab
	true_w = true_w_gab
	w = weights[highlight_i][2]
	plt = plot(-15:0.1:15, (x) -> gabor_f(x; w = w), lw=3, lc=highlight_i+1, label="fitted function", xlabel=L"x", ylabel=L"y")
	# if add_truef
	plot!(-15:0.1:15, (x) -> gabor_f(x; w = true_w),lc = 1, lw=1, label="true function")
	# end
	loss_end = losses_all[weights[highlight_i]][end]
	scatter!(x_train, y_train, ms=3, c=1, label="training data")
	plot!(plt, size=(350,374), title="Learnt function: "*L"\hat{\mathbf{w}} =%$(round.(w; digits=2));", legend=:outerbottom,legendfontsize=8, titlefontsize=10, labelfontsize=8)

	annotate!([-9], [-0.8], text("Loss: "*L"L=%$(round(loss_end;digits=3))", 10, :blue, "Computer Modern"))
end)

# ╔═╡ b6e7802f-3ce2-4663-b6a7-432d5295547d
md"""

# Appendix
"""

# ╔═╡ b6fca076-0639-490a-b77c-7a0155c25f39
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.

# ╔═╡ d96be66d-d960-4402-872e-14161144a7df
md"""

### Simulate data for the linear regression example
"""

# ╔═╡ 48e8aa1f-9a19-456a-bb6d-cdbfe0f5403b
begin
	Random.seed!(123)
	num_features = 2
	num_data = 50
	true_w = rand(num_features+1) * 3
	# simulate the design matrix or input features
	X_train = [ones(num_data) rand(num_data, num_features)]
	# generate the noisy observations
	y_train = X_train * true_w + randn(num_data)
end;

# ╔═╡ 43be7d18-4417-4683-aae8-c35ffc9fb55f
let
	plotly()
	# plot(X_train[:,2], y_train, st=:scatter, label="Observations")
	scatter(X_train[:, 2], X_train[:,3], y_train, markersize=1.5, label="observations", title="Linear regression toy dataset", xlabel="x₁", ylabel="x₂", zlabel="y")
	# surface!(0:0.5:1, 0:0.5:1.0, (x1, x2) -> dot([1, x1, x2], true_w),  colorbar=false, xlabel="x₁", ylabel="x₂", zlabel="y")
end

# ╔═╡ 3e229fb0-62a3-443f-b864-848a0da18711
X_train, y_train;

# ╔═╡ 5155785f-0cda-43de-9501-db8f2d03fcb2
w_normal_eq = linear_reg_normal_eq(X_train, y_train);

# ╔═╡ 47bf5f18-23c0-4780-815d-0fed28f2bc57
w_normal_eq

# ╔═╡ 09c731d4-3b14-49c4-a609-4ab046513078
ws_history, losses = let
	∇l(x) = ∇linear_reg(x; X= X_train, y=y_train)
	max_iters = 2000
	losses = []
	# random starting point
	w₀ = zeros(num_features+1)
	push!(losses, loss(w₀, X_train, y_train))
	ws_history = zeros(num_features+1, max_iters+1)
	ws_history[:, 1] = w₀
	γ = 0.1
	for i in 1:max_iters
		w₀ = w₀ - γ * ∇l(w₀)
		push!(losses, loss(w₀, X_train, y_train)) # book keeping; optional
		ws_history[:, i+1] = w₀ # book keeping; optional
	end
	ws_history, losses
end;

# ╔═╡ 8cca1036-a57f-48e0-826c-c81c4fe19085
anim=let
	gr()
	anim = @animate for i in 1:20
		# plot(1:10)
		scatter(X_train[:, 2], X_train[:,3], y_train, markersize=1.5, label="")
		w0 = ws_history[:, i]
		surface!(0:0.5:1, 0:0.5:1.0, (x1, x2) -> dot([1, x1, x2], w0), c=:jet,  colorbar=false, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"y", alpha=0.5, title="Iteration "*string(i))
	end
end;

# ╔═╡ ea4a2ce4-5b20-4ad3-a39f-51b54be5f3d3
gif(anim, fps=3)

# ╔═╡ fccd6463-ed05-4a7d-89e6-c87981d08248
ws_history[:, end] # gradient descent result;

# ╔═╡ 2a74d131-8af9-4fdc-a5c6-b426d624c505
ws_hist_bad, losses_bad=let
	ws_history, losses = let
	∇l(x) = ∇linear_reg(x; X= x_train_new_bad, y=y_train_new)
	max_iters = 50
	losses = []
	# random starting point
	w₀ = [14.5, 1.5]
	push!(losses, loss(w₀, x_train_new_bad, y_train_new))
	ws_history = zeros(num_features, max_iters+1)
	ws_history[:, 1] = w₀
	γ = 0.023
	for i in 1:max_iters
		w₀ = w₀ - γ * ∇l(w₀)
		push!(losses, loss(w₀, x_train_new_bad, y_train_new)) # book keeping; optional
		ws_history[:, i+1] = w₀ # book keeping; optional
	end
	ws_history, losses
end;
end;

# ╔═╡ 8e411922-1037-4e05-a0de-7e6e728632ba
let
	gr()
	w0 = ws_hist_bad[:, 1]
	losses = losses_bad
	traces = ws_hist_bad
	plt = plot(range(-10,3*5, 100), range(-3, 3, 100), (x,y) -> loss([x, y], x_train_new_bad, y_train_new), c=:jet, colorbar=false, nlevels=25, st=:contour, ratio=1,xlabel=L"w_1", ylabel=L"w_2")
	scatter!([w0[1]], [w0[2]], c=1, label="")
	wt = w0
	anim = @animate for t in 1:1:min(15, size(traces)[2])
		plot!(plt, [traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3, title="Scaled data: Iteration $(t)")
		plot!(plt, [wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="")
		wt = traces[1:2, t]
		plt2 = plot(losses[1:t], xlabel="iteration", ylabel="Loss", title="Loss trajectory", label="", lw=2)

		plot(plt, plt2, layout = grid(2, 1, heights=[0.7 ,0.3]))
	end

	gif(anim, fps=3)
end

# ╔═╡ 592bd479-6d7d-403b-ab43-502aab4d36cb
ws_hist_good, losses_good=let
	ws_history, losses = let
	∇l(x) = ∇linear_reg(x; X= x_train_new, y=y_train_new)
	max_iters = 50
	losses = []
	# random starting point
	w₀ = [5., 5]
	push!(losses, loss(w₀, x_train_new, y_train_new))
	ws_history = zeros(num_features, max_iters+1)
	ws_history[:, 1] = w₀
	γ = 0.2
	for i in 1:max_iters
		w₀ = w₀ - γ * ∇l(w₀)
		push!(losses, loss(w₀, x_train_new, y_train_new)) # book keeping; optional
		ws_history[:, i+1] = w₀ # book keeping; optional
	end
	ws_history, losses
end;
end;

# ╔═╡ d6c05060-67af-45d9-83ec-a35f92cf412a
let
	gr()
	w0 = ws_hist_good[:, 1]
	losses = losses_good
	traces = ws_hist_good
	plt1 = plot(range(-3,5, 100), range(-3, 5, 100), (x,y) -> loss([x, y], x_train_new, y_train_new), c=:jet, colorbar=false, nlevels=25, st=:contour, ratio=1,xlabel=L"w_1", ylabel=L"w_2", framestyle=:semi)

	plt2 = plot()
	scatter!([w0[1]], [w0[2]], c=1, label="")
	wt = w0
	anim = @animate for t in 1:1:min(15, size(traces)[2])
		plot!(plt1, [traces[1, t]], [traces[2, t]], st=:scatter, color=1, label="", markersize=3, title="Orginal data: Iteration $(t)")
		plot!(plt1, [wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:arrow, 0.8, :gray), label="")
		wt = traces[1:2, t]

		plt2 = plot(losses[1:t], xlabel="iteration", ylabel="Loss", title="Loss trajectory", label="", lw=2)

		plot(plt1, plt2, layout = grid(2, 1, heights=[0.7 ,0.3]))
	end

	gif(anim, fps=3)
end

# ╔═╡ 1d6b5309-8bbd-49b0-bbe6-74b071946166
begin
	X_train_new = copy(X_train)
	X_train_new[:, end] = X_train[:, end] * 100
end;

# ╔═╡ 13fe1511-fa89-473c-9c24-c56423a091b9
gw = let
	# gw = Zygote.gradient(w -> ℓ(w, X_train, y_train), w0)[1] # alternatively
	gw, gX, gy = Zygote.gradient(ℓ, w₀, X_train, y_train)
	gw
end

# ╔═╡ 65309501-ceff-4d8c-8297-6194b8289f04
gw # auto-diff result

# ╔═╡ 9bb33492-ea60-4b1b-b622-d5fb5456d1f6
gw_hand_deriv = - (1/ length(y_train)) * X_train' * (y_train - X_train * w₀) # hand derived 

# ╔═╡ 6cd33331-71f5-4bbc-a0ab-9b9bcd968ade
gw_finite_diff = gradient_finite_difference(w -> ℓ(w, X_train, y_train), w₀) ## finite difference method

# ╔═╡ 961ac176-6ade-4bf8-b8e9-d6d78ce664a0
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

# ╔═╡ fd8ae815-892c-4375-9a2f-13e5bd8c6ecf
let
	gr()
	f(x) = f_demo(x) + 3
	x0 = [0.6, 0.6]
	tf(x) = f(x0) + ∇f_demo(x0)' * (x-x0)
	x1_ = range(-0.1, stop =3.0, length=20)
	x2_ = range(-0.1,  stop =3.0, length=20)
	p1_ = plot(x1_, x2_, (a, b) -> f([a, b]), st=:surface, xlim=[-0.2, 3],ylim=[-0.2, 3],  zlim = [-1, 8+3], xlabel =L"w_1", ylabel=L"w_2", zlabel=L"f(\mathbf{w})", colorbar=false, color=:jet, size=(600,550), framestyle=:zerolines)
	scatter!([x0[1]], [x0[2]], [f(x0)],  label="", mc=1, msw=2, alpha=1.0)
	scatter!([x0[1]], [x0[2]], [0], label=L"\mathbf{w}_0", ms =4, mc=:white, msc=:gray, msw=2, alpha=1.0)
	plot!([x0[1], x0[1]], [x0[2], x0[2]], [f(x0), 0], lw=2, lc=:black, ls=:dash, label="")
	γ = 0.4
	if add_where_to_go
		for theta ∈ range(-π, π, 12)
			arrow3d!([x0[1]], [x0[2]], [0], [γ * cos(theta)], [γ * sin(theta)], [0]; as = 0.1, lc=1, la=0.5, lw=1, scale=:identity)
		end
	end




	gd = ∇f_demo(x0)
	gd = gd/norm(gd) * γ
	xlen, ylen = 0.5, 0.5
	if add_linear_app
		plot!(range(x0[1] - xlen, x0[1] + xlen, 4), range(x0[2] - ylen, x0[2]+ylen, 4), (a, b) -> tf([a, b]), st=:surface, alpha=0.4,c=:gray)
	end

	if add_gradient_local
		if add_neg_gradient_local
			gdd = -gd
		else
			gdd = gd
		end
		arrow3d!([x0[1]], [x0[2]], [0], [gdd[1]], [gdd[1]], [0]; as = 0.2, lc=3, la=1, lw=3, scale=:identity)
		if add_where_go_values
			x_new = x0 + gdd
			scatter!([x_new[1]], [x_new[2]], [0], label="", ms =3, mc=:white, msc=2, msw=2, alpha=1.0)
			scatter!([x_new[1]], [x_new[2]], [f(x_new)], ms =4, mc=:white, msc=2, msw=2, alpha=1.0, label="")
			plot!([x_new[1], x_new[1]], [x_new[2], x_new[2]], [f(x_new), 0], lw=2, lc=2, ls=:dash, label="")
		end
	else
		dd = [γ * cos(utheta), γ * sin(utheta)]

		if add_where_go_values
			arrow3d!([x0[1]], [x0[2]], [0], [dd[1]], [dd[2]], [0]; as = 0.2, lc=2, la=1, lw=3, scale=:identity)
			x_new = x0 + dd
			scatter!([x_new[1]], [x_new[2]], [0], ms =3, mc=:white,label="", msc=2, msw=2, alpha=1.0)
			scatter!([x_new[1]], [x_new[2]], [f(x_new)], ms=3, markershape=:circle, label="", mc=2,  msw=2, alpha=0.9)
			plot!([x_new[1], x_new[1]], [x_new[2], x_new[2]], [f(x_new), 0], lw=2, lc=2, ls=:dash, label="")
		end
	end

	

	if add_show_the_move
		x_new = x0 - gd
		scatter!([x_new[1]], [x_new[2]], [0], label=L"\mathbf{x}_{new}", ms =4, mc=:white, msc=3, msw=2, alpha=1.0)
		scatter!([x_new[1]], [x_new[2]], [f(x_new)], mc=1, msw=2, alpha=2.0, label="")
		plot!([x_new[1], x_new[1]], [x_new[2], x_new[2]], [f(x_new), 0], lw=2, lc=3, ls=:dash, label="")
		# scatter!([x_new[1]], [x_new[2]], [0], label=L"\mathbf{x}_{new}", ms =3, mc=:white, msc=2, msw=2, alpha=1.0)
		# scatter!([x_new[1]], [x_new[2]], [f(x_new)], ms=2, markershape=:circle, label="", mc=2, msc=:gray, msw=2, alpha=0.9)
		# plot!([x_new[1], x_new[1]], [x_new[2], x_new[2]], [f(x_new), 0], lw=2, lc=2, ls=:dash, label="")
	end

		
	p1_
end

# ╔═╡ d66070cf-796a-4589-b2cb-933146c13592
let
	plotly()
	x0 = x0_st
	γ = 0.2
	xs = []
	push!(xs, x0)
	eps = 0.2
	c = 4
	f(x) = f_demo(x) + c
	x1_ = range(-0.1, stop =3.0, length=30)
	x2_ = range(-0.1,  stop =3.0, length=30)
	p1_ = plot(x1_, x2_, (a, b) -> f([a, b]) , st=:surface, xlim=[-0.2, 3],ylim=[-0.2, 3],  zlim =[-1, 8 + c], xlabel =
	"w1", ylabel="w2", zlabel="f(w)", colorbar=false, color=:jet, alpha=0.75, size=(600,550), framestyle=:zerolines)
	msize = 3
	scatter!([x0[1]], [x0[2]], [f(x0) + eps],  label="", mc=:gray, m=:x, ms=msize)
	scatter!([x0[1]], [x0[2]], [0],  label="", mc=:white, msc=:gray, msw=2, alpha=2.0, ms=msize)
	plot!([x0[1], x0[1]], [x0[2], x0[2]], [f(x0), 0], lw=1, lc=:black, ls=:dash, label="")
	for i in 1:20
		gd = ForwardDiff.gradient(f, x0)
		xnew = x0 - γ * gd
		scatter!([xnew[1]], [xnew[2]], [f(xnew)+eps],  label="", mc = 1, ms=msize)
		scatter!([xnew[1]], [xnew[2]], [0],  label="", mc=:white, msc=:gray, msw=2,ms=msize, alpha=2.0)
		plot!([xnew[1], xnew[1]], [xnew[2], xnew[2]], [f(xnew), 0], lw=1, lc=:black, ls=:dash, label="")
		plot!([x0[1], xnew[1]], [x0[2], xnew[2]], [f(x0),f(xnew)], line = (:solid, 2, 0.9, :blue), label="")
		df = xnew - x0
		arrow3d!([x0[1]],[x0[2]],[0],[df[1]],[df[2]],[0]; as=0.3, lc=2, lw=2)
		x0 = xnew
		push!(xs, x0)
		
	end
	
	xs
	p1_
end

# ╔═╡ dfb3736b-10a1-429a-a561-f942b3ac7549
plts = let
	gr()
	x0 = [0.3, 0.3]
	γ = 0.2
	xs = [x0]
	for i in 1:10
		x0 = x0 - γ * ∇f_demo(x0)
		push!(xs, x0)
	end
	f(x) = f_demo(x) + 3
	
	# xs
	plts = []
	xlen, ylen = 0.5, 0.5
	x1_ = range(-0.1, stop =3.0, length=25)
	x2_ = range(-0.1,  stop =3.0, length=25)
	for (i, xi) in enumerate(xs[1:end-1])	
		plt = plot(x1_, x2_, (a, b) -> f([a, b]), st=:surface, xlim=[-0.2, 3],ylim=[-0.2, 3],  zlim = [-1, 8+3], xlabel =L"w_1", ylabel=L"w_2", zlabel=L"f(\mathbf{w})", colorbar=false, color=:jet, framestyle=:zerolines, title="Iteration: $(i)")
		scatter!([xi[1]], [xi[2]], [f(xi)],  label="", c=1, alpha=1.0)
		scatter!([xi[1]], [xi[2]], [0], label=L"\mathbf{w}_t", ms =3, mc=:white, msc=:gray, msw=2, alpha=1.0)
		plot!([xi[1], xi[1]], [xi[2], xi[2]], [f(xi), 0], lw=2, lc=:gray, ls=:dash, label="")
		tf = taylorApprox(f, xi, 1)
		plot!(range(xi[1] - xlen, xi[1] + xlen, 4), range(xi[2] - ylen, xi[2]+ylen, 4), (a, b) -> tf([a, b]), st=:surface, alpha=0.4,c=:gray)

		push!(plts, plt)
		# scatter!([xi[1]], [xi[2]], [f(xi)],  label="", mc=:white, msc=:gray, msw=2, alpha=2.0)
		plt_new = deepcopy(plt)
		x_new = xs[i+1]
		gd = x_new - xi
		arrow3d!([xi[1]], [xi[2]], [0], [gd[1]], [gd[1]], [0]; as = 0.2, lc=2, la=1, lw=3, scale=:identity)


		push!(plts, plt_new)

		plt_new2 = deepcopy(plt)

		scatter!([x_new[1]], [x_new[2]], [0], label=L"\mathbf{w}_{t+1}", ms =3, mc=:white, msc=2, msw=2, alpha=1.0)
		scatter!([x_new[1]], [x_new[2]], [f(x_new)], ms=2, markershape=:circle, label="", mc=2, msc=:gray, msw=2, alpha=0.9)
		plot!([x_new[1], x_new[1]], [x_new[2], x_new[2]], [f(x_new), 0], lw=2, lc=2, ls=:dash, label="")
		push!(plts, plt_new2)
	end

	plts
end;

# ╔═╡ 96d87bd0-bdee-45e2-8e67-397a95ea85c0
begin 
	anim_new = Animation()
	for p in plts
		frame(anim_new, p)
	end

	gif(anim_new, fps=3)
end

# ╔═╡ 7d5a7f03-74b0-42f8-a5c8-19e077db26bc
md"""

## Explain gradient descent

Let's consider univariate function first

A differentiable function ``f(x)`` can be approximated locally at ``x_0`` by a **linear function**


```math

f(x) \approx f(x_0) + f'(x_0) ( x- x_0)
```


* slope: ``b_1= f'(x_0)``
* intercept: ``b_0 = f(x_0) - x_0 f'(x_0)``


The idea can be generalised to multivariate function


```math

f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top ( \mathbf{x}- \mathbf{x}_0)
```


* derivative ``\rightarrow`` gradient

""";

# ╔═╡ 7c6e1fd4-aa87-45c0-9c49-6dab2de90043
linear_approx_f(x; f, ∇f, x₀) = f(x₀) + dot(∇f(x₀), (x-x₀));

# ╔═╡ e54c9f2d-ae25-46f1-8fcc-a75184a3610b
begin
	A = Matrix(I, 2, 2)
	f(x) = dot(x, A, x)
	∇f(x) = 2* A* x
end;

# ╔═╡ dc7bbdc2-d3df-42a8-865b-74c42096cb57
md"Expansion location:";

# ╔═╡ ed7f77ac-0cc5-4141-b88b-c5a86749ddd6
x₀ = [-5, 5];

# ╔═╡ e1219a00-b1cd-4aba-a3ab-cfec9380fcf3
let
	plotly()
	plot(-15:1.0:15, -15:1.0:15, (x1, x2) -> f([x1, x2]), st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.8, framestyle=:zerolines, ratio=1, colorbar=false)
	plot!(-15:1:15, -15:1:15, (x1, x2) -> linear_approx_f([x1, x2]; f=f, ∇f= ∇f, x₀), st=:surface)
end;

# ╔═╡ 55b788aa-e846-421e-b214-ef99e24fd97e
md"""

## Explain gradient descent



```math

f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top ( \mathbf{x}- \mathbf{x}_0)
```


* the linear approximation is accurate when ``\mathbf{x}`` is close to ``\mathbf{x}_0``

**Gradient descent**
* at each iteration, find a local linear approximation
* and follow the (opposite) direction of the hyperplane 
* small learning rate: the approximation is correct only locally
""";

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
ForwardDiff = "~0.10.36"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.1"
Latexify = "~0.15.17"
LogExpFunctions = "~0.3.18"
Plots = "~1.39.0"
PlutoTeachingTools = "~0.2.5"
PlutoUI = "~0.7.48"
StatsBase = "~0.34.2"
Zygote = "~0.6.68"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.0"
manifest_format = "2.0"
project_hash = "8876df1a3f77290d4c7eb8319093c4898ffd8acf"

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
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

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
git-tree-sha1 = "be227d253d132a6d57f9ccf5f67c0fb6488afd87"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.71.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
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
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b5278586822443594ff615963b0c09755771b3e0"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.26.0"

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
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

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
git-tree-sha1 = "ea32b83ca4fefa1768dc84e504cc0a94fb1ab8d1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.2"

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
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

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
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Formatting]]
deps = ["Logging", "Printf"]
git-tree-sha1 = "fb409abab2caf118986fc597ba84b50cbaf00b87"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.3"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "532f9126ad901533af1d4f5c198867227a7bb077"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+1"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "62ee71528cca49be797076a76bdc654a170a523e"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "10.3.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "ec632f177c0d990e64d955ccc1b8c04c485a0950"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.6"

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
git-tree-sha1 = "674ff0db93fffcd11a3573986e550d66cd4fd71f"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "d1d712be3164d61d1fb98e7ce9bcbc6cc06b45ed"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.8"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "401e4f3f30f43af2c8478fc008da50096ea5240f"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.3.1+0"

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
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "39d64b09147620f5ffbf6b2d3255be3c901bec63"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.8"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "f389674c99bfcde17dc57454011aa44d5a260a40"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "25ee0be4d43d0269027024d75a24c24d6c6e590c"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.4+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "2984284a8abcfcc4784d95a9e2ea4e352dd8ede7"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.36"

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
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "4ad43cb0a4bb5e5b1506e1d1f48646d7e0c80363"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.1.2"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "05a8bd5a42309a9ec82f700876903abce1017dd3"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.34+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+1"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "8c57307b5d9bb3be1ff2da469063628631d4d51e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.21"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    DiffEqBiologicalExt = "DiffEqBiological"
    ParameterizedFunctionsExt = "DiffEqBase"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e"
    DiffEqBiological = "eb300fae-53e8-50a0-950c-e21f52c2b7e0"
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
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "9fd170c4bbfd8b935fdc5f8b7aa33532c991a673"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.11+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fbb1f2bef882392312feb1ede3615ddc1e9b99ed"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.49.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

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
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "96d2a4a668f5c098fb8a26ce7da53cde3e462a80"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.0.3"

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
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

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
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e127b609fb9ecba6f201ba7ab753d5a605d53801"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.54.1+0"

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
git-tree-sha1 = "6e55c6841ce3411ccb3457ee52fc48cb698d6fb0"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.2.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

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
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "2d4e5de3ac1c348fd39ddf8adbef82aa56b65576"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.6.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

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
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

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
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "f4dc295e983502292c4c3f951dbb4e985e35b3be"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.18"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = "GPUArraysCore"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d95fe458f26209c66a187b1114df96fd70839efd"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.21.0"

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
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "1165b0443d0eca63ac1e32b8c0eb69ed2f4f8127"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ac88fb95ae6447c8dda6a5503f3bafd496ae8632"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.6+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "bcd466676fef0878338c61e655629fa7bbc69d8e"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+0"

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
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "f2f85ad73ca67b5d3c94239b0fde005e0fe2d900"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.71"

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
git-tree-sha1 = "936081b536ae4aa65415d869287d43ef3cb576b2"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.53.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

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
git-tree-sha1 = "b70c870239dc3d7bc094eb2d6be9b73d27bef280"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+0"

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
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─9ca39f56-5eea-11ed-03e1-81eb0fa2a37d
# ╟─f9736ecc-053f-444a-9ef5-cdbe85795fce
# ╟─8d096119-f6db-4f62-a091-6f00372468ec
# ╟─dcacb71b-bf45-4b36-a2d2-477839f52411
# ╟─83282ceb-ad43-462a-8c28-88191ac49471
# ╟─d7a55322-0d9f-44e8-a2c6-4f0cead25f9d
# ╟─5bb178a2-f119-405f-a65b-ec6d59a112e0
# ╟─799ead95-2f68-4bcb-aae8-dd8b0b00560c
# ╟─afb99b70-a418-4379-969b-355fbcfe8f14
# ╟─ee92f14a-784a-407c-8955-9ea1dd9cf942
# ╟─2c063e21-faec-479d-a452-db411a7c9adc
# ╟─694d5df4-b073-4d74-b929-108c4ea6d646
# ╟─b49fd692-c36f-40de-be10-55a65ba13dba
# ╟─c49a6965-727e-419b-b66e-57dc61415edf
# ╟─e7447991-fd5f-403f-8ea8-61df1e076ae7
# ╟─4b757c23-d30a-445c-a9d7-1b733150cacf
# ╟─a92344a8-83a0-4d8a-825e-b546b53d6292
# ╟─cbb348f6-25e6-4db5-9dce-e5e7d5e07931
# ╟─07c7e0e4-bbfe-4b98-80a9-d952b6705997
# ╟─c9a9640d-dbf5-4e51-9c6e-3c859b61f593
# ╟─996fcbcb-1f30-4138-85cf-7b3e9a708505
# ╟─f3f2c04a-c50d-49cd-907a-07fa301df236
# ╟─7534d726-45cc-453c-904e-166db6c11410
# ╟─496bed39-98e5-437f-906f-413be5e07e77
# ╟─cc9be8be-7230-4873-884c-6ac119c0bdb3
# ╟─65eea27a-0df8-409f-840d-1ff0bd740339
# ╟─5921c8dc-25d6-4acb-9f60-1bcf8644ee12
# ╟─49781399-7f9e-4237-ab23-e52fdb87ed15
# ╟─7d4e0c2e-0f2b-4e06-a265-14e276ba9387
# ╟─478d8fe1-9548-4fab-bc60-10c1b87d4e37
# ╟─002c9cac-6bc1-4288-903f-159afdcceef8
# ╟─da573528-649b-4fbe-97f9-5d82fe0cc716
# ╟─035032e4-fde2-4067-b08c-477346d4c5af
# ╟─ffeb64b4-db59-4f20-9332-b502904802b5
# ╟─a14618d8-8416-4a1d-aee4-9b8d55fa4204
# ╟─a21a6912-18a6-49b8-8ba8-18df9006578a
# ╟─35bad274-8618-48c1-b9ea-a232d6fcf23c
# ╟─67d19b65-ffe6-4d53-9087-f4062311309d
# ╟─063a28e7-4e1d-4583-8ff9-90930dab6396
# ╟─c5ed4d55-ccc5-4855-8b04-203a4205ff5c
# ╟─36d0cea4-4d17-4493-bc45-d5f1f15a67f8
# ╟─b06cb19a-4b0e-4a45-aa6e-285b8f47a06d
# ╟─0541ccc4-b05b-44d3-9ea7-00c27b644c04
# ╟─9caa90a1-103e-4268-83a7-875dcdd6c488
# ╟─fd8ae815-892c-4375-9a2f-13e5bd8c6ecf
# ╟─1d5970d6-95d5-44a1-9790-0138c0d3f1a4
# ╟─96d87bd0-bdee-45e2-8e67-397a95ea85c0
# ╟─db883f69-f516-432d-81f6-2bdfd2f0b40c
# ╟─d66070cf-796a-4589-b2cb-933146c13592
# ╟─dfb3736b-10a1-429a-a561-f942b3ac7549
# ╟─4814ab50-3ed2-4197-b116-719ccbb28939
# ╟─f477c82e-f4d6-4918-85d1-19ce21e90b5b
# ╟─b6fd9d29-b5ac-4940-90d8-fcf0d8063003
# ╟─f4a1f7db-0bcb-45b6-be9d-1c57dd6e2b99
# ╟─c2cd3414-7d70-467b-b881-799c20a8489c
# ╟─20f57c9d-3543-4cb6-8844-30b30e3b08ec
# ╟─c2bebba9-9a03-43dd-81e4-93da08e1118c
# ╟─43ac689e-c16a-47cc-9c21-477fb2a6989b
# ╟─a5bfaa1e-9d40-4f2e-b0d5-73ab9bbceddf
# ╟─ace9c646-743c-494f-a35b-0acc7107b8a9
# ╟─3b1c8be8-fb5e-48d7-9436-cd23a715f223
# ╟─f177f82c-fb9b-45a4-8944-6d2d0c540572
# ╟─00dac405-aef9-420f-b019-12d325901fd9
# ╟─46dd39e0-95bf-4079-a982-4f93b16c3a5e
# ╟─f15c498f-a4af-4cd5-a2db-2ea42516fb0f
# ╟─095f95fa-75e1-488f-a1c6-afcaaad30baa
# ╟─5d52d818-21b2-4e8f-8efa-486274b68e57
# ╟─b732f3da-7be0-4470-9a5b-a389c1d1c166
# ╟─538a3ded-2ca0-4f0c-a931-fb7e43e5c24f
# ╟─df7c5704-a493-499e-b75d-550606e04edd
# ╟─cd2a8e1e-d174-4c62-9391-beca7272a15d
# ╟─17c84d3e-aba7-4c66-87fc-8c0289d0b51d
# ╟─f9bc87f7-0e29-481d-97a8-24f62364dc5a
# ╟─8afbc273-c62e-49a8-98f6-893b0d8a3696
# ╟─1c1fb3aa-0dd6-4ee7-98af-81d384260685
# ╟─726545fd-fbbc-4ce9-a35d-30e7e23ff3f9
# ╟─78c47b3f-9641-4ea4-9f3d-fadaa0ae004c
# ╟─c9baffcf-c1cc-42d6-96b5-1cc9a8801113
# ╟─40af31cc-1bc6-436c-ae6b-d13e2300779a
# ╟─5063f1be-5f18-43f6-b428-a0464e8fb339
# ╟─c0acfd8e-c152-4991-80a9-65d69c3bda69
# ╟─1ae4f029-cb95-4897-8029-903767babf7b
# ╟─4f176ca1-0326-4576-ae18-af2e1b697655
# ╟─a676ed26-1420-478f-a25c-fe99ab94c0a5
# ╟─6e25724e-79f7-4050-b3ff-fc51bfc852b5
# ╟─0be36e2d-817b-4951-8a3d-4ab44ce761ce
# ╟─94899a98-1565-4129-a264-1c4e1855982b
# ╟─4bd343a1-e6b9-439b-b156-8830687bb9e6
# ╟─afd2cf78-ca7a-4662-9350-75f857ef528b
# ╟─90257a92-c845-4595-abe9-33d6d9c22c2c
# ╟─4ba59af5-3ebc-42b4-a928-3c4b17024f8e
# ╟─1643d5a5-8925-45b7-a4ff-a39165207ec1
# ╟─a6e494dc-2eea-4a0d-930e-3ef59182b730
# ╟─8a5a3a3d-d341-48e9-bcba-a16094f048c4
# ╟─5a9c85ce-6c22-40da-ae6d-ed216ebfa175
# ╠═258131b3-3f64-4acc-a69f-e095ee48d42b
# ╠═5f544e6e-2f92-4e12-b208-162b17b343d7
# ╟─466d0d89-bf16-40f9-8eb2-2dfe70a8b227
# ╟─d94a06dc-77fb-4e27-a313-6589b5641519
# ╟─43be7d18-4417-4683-aae8-c35ffc9fb55f
# ╟─1cd4daa9-4683-4b01-a4ed-7c55d769c861
# ╟─8cca1036-a57f-48e0-826c-c81c4fe19085
# ╟─ea4a2ce4-5b20-4ad3-a39f-51b54be5f3d3
# ╟─730b641f-00bc-4b27-a832-166d1e2d75c9
# ╠═fccd6463-ed05-4a7d-89e6-c87981d08248
# ╠═47bf5f18-23c0-4780-815d-0fed28f2bc57
# ╟─3e229fb0-62a3-443f-b864-848a0da18711
# ╟─8d9edfd5-6f97-4c4e-9464-3841078cc7c4
# ╟─5155785f-0cda-43de-9501-db8f2d03fcb2
# ╟─4788ccd2-62a9-423f-95f8-57507ccdd6e3
# ╟─4b5ec74e-b89c-4bbf-9ac1-fa7d5b709f57
# ╟─09c731d4-3b14-49c4-a609-4ab046513078
# ╟─69b9cca5-be80-434f-8e92-299bc12d5258
# ╟─e95f09c8-9364-4cf2-91a7-4e2dcbcd1053
# ╟─9ccef028-d31d-45d6-a37f-77cd31f772dc
# ╟─42e01d0b-a093-44f9-8b1b-ae76b6e67b7d
# ╟─12b78d04-2cdb-44c0-8350-110c957e9757
# ╟─46121e9d-3bef-48fd-a709-2b7613d0f8be
# ╟─4488af42-7f4a-4c52-b8ca-1d86aeea9e69
# ╟─37eb85ca-a07e-4409-a8ab-e27819552355
# ╟─1c5c58b6-8567-4ce0-af7a-fa84f4879d39
# ╟─61263a4c-f756-439e-a2bf-b7d7bc6d5b66
# ╟─de62b0cb-0ca8-4b6c-9edb-409328bcf540
# ╟─0ed56488-7e32-474a-a631-3e42f3481cf2
# ╟─57b7e86d-38d8-4508-a410-465a3f14d6f5
# ╟─3813e242-cef7-4c8f-a97f-842f67ba86ab
# ╟─6ae54ba5-2d2d-4270-9617-51061c68271a
# ╟─bd23fdab-0651-47fd-bcd0-68db16275e32
# ╟─b6946dae-9b8d-4666-9616-00edcc1f524f
# ╟─2e704205-791d-46c7-9a01-c1e77e178717
# ╟─2a74d131-8af9-4fdc-a5c6-b426d624c505
# ╟─592bd479-6d7d-403b-ab43-502aab4d36cb
# ╟─da8ee92e-275f-47bb-a60e-855ce686c38d
# ╟─d6c05060-67af-45d9-83ec-a35f92cf412a
# ╟─8e411922-1037-4e05-a0de-7e6e728632ba
# ╟─1d6b5309-8bbd-49b0-bbe6-74b071946166
# ╟─42e7bc1a-7380-47f6-aa2d-c49cdaff0b08
# ╟─487e5448-8565-440e-a5ae-56dbe3c5cadf
# ╟─35ddd086-900a-4f44-9aae-59a88d27774f
# ╟─fe721301-1e77-4c89-8592-17d8c88776a7
# ╟─fbf9a928-fdc7-4f38-9b93-4bf74c20ad5b
# ╟─b76084a8-7722-4db3-9795-e1ce03c50b9d
# ╟─4602e84e-435e-425e-9fe0-a2764bd7ae47
# ╟─d3d2f0f0-1a16-495c-b3e5-7b0eec73ada5
# ╟─fb5aa103-3c24-43c9-a2b4-0c0665a5476e
# ╟─58f37e36-755e-4f5a-af96-183a9a227118
# ╟─508e8a33-8b94-4807-9118-0a3f6d4cc1f5
# ╟─f43a8744-81f7-4da7-ae76-d9203d91e81a
# ╟─93d3d44c-5072-4412-b1c6-adc810f20235
# ╟─04131640-f043-4582-b143-8c7cd214d52a
# ╟─c72fa154-96cf-4079-aea5-e1b4aa8e2cf2
# ╟─b7236763-ae9e-427c-9a0f-94cadbc34b8b
# ╠═f3cbf4df-9704-41f6-9bdf-5ed3e0edd250
# ╟─c9324c6b-9ca5-48fd-9ee5-e2f1ec744c60
# ╠═bc86509b-2164-46d1-809a-7cadc6d5d53d
# ╟─f5988f97-50ad-4f32-9fb5-69fc0ed4bc73
# ╠═373b1b1d-1bd8-4750-86c5-52810456aeed
# ╠═13fe1511-fa89-473c-9c24-c56423a091b9
# ╟─97ed5d31-7832-404f-907f-779260456c57
# ╠═65309501-ceff-4d8c-8297-6194b8289f04
# ╠═9bb33492-ea60-4b1b-b622-d5fb5456d1f6
# ╠═6cd33331-71f5-4bbc-a0ab-9b9bcd968ade
# ╟─75510fd8-c1a9-4eda-98b3-61dedb243b78
# ╟─4c1865c6-a044-4c65-8388-09380ee86844
# ╟─1f0a9fae-e314-401d-967d-ab7f5b9990f9
# ╟─2095d604-e9d5-46f3-b152-e8e1f767db60
# ╟─e0d43282-8197-414b-aa9c-d83cdb8b9899
# ╟─3f89f4ae-56de-4906-b760-fdb2b4a22a88
# ╟─88d3904c-b915-4396-be48-ae0f2bf0cc6b
# ╟─995c0495-3feb-4a68-8da6-e2733890abff
# ╟─632be70e-beb8-44b6-864b-d02de7b904c6
# ╟─f2a85f4b-7e6d-46d3-97dc-c148a412886a
# ╟─89064ca7-6aff-40b4-a414-aaa23b3c9d1e
# ╟─b67461c2-6a96-4e49-9fcc-c5604b14fdbb
# ╟─e5898a5a-dc97-449c-96b0-72c2d5f9ad85
# ╟─65de1dd5-d6d6-42a8-b41c-53498abe0c9f
# ╟─1a9ad655-027e-4f59-a0e6-48d12a4abe17
# ╟─c5368d27-7710-402e-97d1-047f3949298a
# ╟─c1411cf0-8982-4e0f-a01a-ae4ba22e16b2
# ╟─5dbc7275-7e06-4c07-b751-ccc52ad3e168
# ╟─ad5fdaf4-21ee-4328-aee2-a5c73b1a5c2a
# ╟─db83f381-cb61-4752-91d8-6ef19b6a3dca
# ╟─e833ae14-1bc4-49b7-a9cd-eea0554bb47f
# ╟─0b4d6238-e21b-4998-9825-e7da2001cb03
# ╟─a6427ed6-7c4e-4165-a819-7d14be18f513
# ╟─cfb74a46-252c-4c71-b569-0b901ba3d615
# ╟─57c0ac44-60ee-425f-8189-c231fb1cd700
# ╟─c5f788a4-4b6a-4fa0-90c4-dca98af59f8f
# ╟─f17b7b3c-150a-48b2-842b-66ffbb36f2e6
# ╟─2b975333-8b48-4848-a261-4a15c6589658
# ╟─ebd84273-2610-4ad4-810d-664d205c6591
# ╟─0552fef1-2a56-4041-956f-6692ca0be446
# ╟─8014a3e8-19b6-4086-8fce-bcfe16a16991
# ╟─8c6bf500-5481-422d-bdfc-c0738be80bc3
# ╟─e00b4813-2df0-48fd-9322-80ccb3d0f115
# ╟─19b1a466-13ba-4b79-97c4-36ea007ad219
# ╟─536f04f3-045a-445d-81ae-2644f78dc1eb
# ╟─dfcbda48-4bde-418d-83ec-5d8dc1b5ab94
# ╟─6d4600f7-baa4-41e3-bf0a-8c5c13e55f64
# ╟─1e4f6557-c9f5-49af-8a88-140d721b70c5
# ╟─b6e7802f-3ce2-4663-b6a7-432d5295547d
# ╟─b6fca076-0639-490a-b77c-7a0155c25f39
# ╟─d96be66d-d960-4402-872e-14161144a7df
# ╟─48e8aa1f-9a19-456a-bb6d-cdbfe0f5403b
# ╠═961ac176-6ade-4bf8-b8e9-d6d78ce664a0
# ╟─7d5a7f03-74b0-42f8-a5c8-19e077db26bc
# ╟─7c6e1fd4-aa87-45c0-9c49-6dab2de90043
# ╟─e54c9f2d-ae25-46f1-8fcc-a75184a3610b
# ╟─dc7bbdc2-d3df-42a8-865b-74c42096cb57
# ╟─ed7f77ac-0cc5-4141-b88b-c5a86749ddd6
# ╟─e1219a00-b1cd-4aba-a3ab-cfec9380fcf3
# ╟─55b788aa-e846-421e-b214-ef99e24fd97e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
