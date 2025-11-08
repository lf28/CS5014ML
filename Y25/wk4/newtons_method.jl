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
	using Zygote
end

# ╔═╡ dcacb71b-bf45-4b36-a2d2-477839f52411
using Logging

# ╔═╡ 83282ceb-ad43-462a-8c28-88191ac49471
using ForwardDiff

# ╔═╡ 30a9cc96-092d-4122-8101-1878fa83d1cb
using Distributions

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



#### Beyond gradient descent

\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ ee92f14a-784a-407c-8955-9ea1dd9cf942
# md"""

# ## Notations


# #### Scalars: normal case letters
# * ``x,y,\beta,\gamma``


# #### Vectors: **Bold-face** smaller case
# * ``\mathbf{x},\mathbf{y}, \boldsymbol{\beta}``
# * ``\mathbf{x}^\top``: row vector

# #### Matrices: **Bold-face** capital case
# * ``\mathbf{X},\mathbf{A}, \boldsymbol{\Gamma}``  

# #### Tensors: *sans serif* font
# * ``\mathsf{X, A, \Gamma}``

# """

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

# More optimisation algorithm

## Today
### Two more optimisation algorithms

* #### Stochastic gradient descent


* #### Newton's method


"""

# ╔═╡ 3f3eae53-262d-4ef9-bda3-466b870af972
md"""

## Reading & references

##### Essential reading 


* **Stochastic gradient descent** [_Understanding Deep Learning_ by _Simon Prince_: Chapter 6.2](https://github.com/udlbook/udlbook/releases/download/v2.00/UnderstandingDeepLearning_28_01_24_C.pdf)


* **Newton's method** [_Deep Learning_ by _Ian Goodfellow and Yoshua Bengio and Aaron Courville_ Chapter 4.3.1](https://www.deeplearningbook.org/contents/numerical.html)


##### Suggested reading


* **Stochastic gradient descent** [_An Alternative View: When Does SGD Escape Local Minima?_ by Robert Kleinberg, Yuanzhi Li, Yang Yuan ICML](https://arxiv.org/pdf/1802.06175.pdf)

"""

# ╔═╡ a0ca254a-8de0-4693-b2b3-d7620e22c843
md"""

## Batch gradient is expensive compute


#### _For most ML problems,_ the loss is 

```math
\Large
\ell(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^n \ell^{(i)}(\mathbf{w})
```
* ##### average loss of all training samples' losses






"""

# ╔═╡ 2fa021dd-0040-438c-91bf-3a4359d9a760
md"Add all stochastic gradients ``\nabla \ell^{(i)}``: $(@bind add_all_sgds CheckBox(false))"

# ╔═╡ 7f6641c5-9052-46be-9926-e3531d4ebab9
TwoColumn(md"""
#### And the gradient is


```math
\Large
\nabla \ell(\mathbf{w})  = \boxed{\frac{1}{n}\sum_{i=1}^n \nabla \ell^{(i)}(\mathbf{w})}_{\text{too expensive!}}
```

* ##### average direction of individual gradient directions
* ##### needs to store and go through all ``n`` training data



""", let
	gr()
	vv = [.5, .5] *2
	plt = plot([0, vv[1]], [0, vv[2]], lc=:blue, arrow=Plots.Arrow(:closed, :head, 10, 10),  st=:path, lw=2, c=:red,  xlim=[-.2, 2], ylim=[-.2, 2], ratio=1, label="",framestyle=:none, legend=:bottomright, size=(320,320), legendfontsize=12)
	annotate!([vv[1] + .1], [vv[2] + .05], (L"\nabla \ell",14, :blue))
	Random.seed!(345678902)

	for j in 1: 3
		v = vv + randn(2) ./ 2.5
		annotate!([v[1] ], [v[2] ], (L"\nabla \ell^{(%$(j))}", 8, :gray, :bottom))
		plot!([0, v[1]], [0, v[2]], lc=:gray, arrow=Plots.Arrow(:open, :head,1,1),  st=:path, lw=.3, label="")
	end
	v = vv + randn(2) /2
	annotate!([v[1] ], [v[2] ], (L"\nabla \ell^{(i)}", 12, :gray, :bottom))
	plot!([0, v[1]], [0, v[2]], lc=:gray, arrow=Plots.Arrow(:open, :head,1,1),  st=:path, lw=.3, label="")
	if add_all_sgds
		Random.seed!(123)
		for i in 1:20
			v = vv + randn(2) ./ 2.5
			plot!([0, v[1]], [0, v[2]], lc=:gray, arrow = Plots.Arrow(:open, :head,1,1),  st=:path, lw=.3, label="")
		end
	end
	plt
end)

# ╔═╡ 8e78770a-536b-4555-a312-0ba7006be5b0
md"""


## Demonstration -- logistic regression example

##### -- classification data (left) & its cross entropy loss (right)
"""

# ╔═╡ 9601b450-c167-4c14-b627-f6b229e93788
md"""
## Stochastic gradient descent

#### *Idea*: descent with *one observation*'s gradient

```math
\Large
\begin{align}
\nabla \ell(\mathbf{w}) 
&\approx \nabla \ell^{(i)}(\mathbf{w})
\end{align}
```

* ###### ``\nabla \ell^{(i)}`` are noisy version of the batch gradient ``\nabla \ell``

* ###### note that some of the ``\nabla\ell^{(i)}``s may not even be a descent direction!

"""

# ╔═╡ 521cda12-8e0f-4df9-81e8-a6d9178203cb
begin
	function logistic_loss(w, X, y; agg=sum)
		σ = logistic.(X * w)
		# deal with boundary cases such as σ = 0 or 1, log(0) gracefully
		# sum(y .* log.(σ) + (1 .- y).* log.(1 .- σ))
		# rather you should use xlogy and xlog1py
		(-(xlogy.(y, σ) + xlog1py.(1 .-y, -σ))) |> agg
	end
end;

# ╔═╡ 66768384-cb86-4aa4-856f-bf70a7c2bf9a
md"""

##  Why "noisy" gradients?


* ##### it helps the algorithm jump out of _local minimums_

"""

# ╔═╡ 7c3a0677-6c78-44ec-a3be-aca9f3a59197
show_img("sgdlocaloptim.png", w=900)

# ╔═╡ 10b08e9c-ad61-46ad-b650-c839159727aa
md"""
##
"""

# ╔═╡ f62458b2-8837-435d-801f-0bf9689086c7
show_img("sgd_twitter.png", w = 400)

# ╔═╡ 1d38cea7-7e0c-4d2b-bd30-ce307fed0131
md"""

## Stochastic gradient descent (SGD)




-----

* ##### random guess ``\large\mathbf{w}_0``

* ##### for each *epoch*
  * ###### randomly shuffle the data
  * ###### for each ``i \in 1\ldots n``
    * ###### ``\large\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma_{t-1} \nabla \ell^{(i)}(\mathbf{w}_{t-1})``
-----



"""

# ╔═╡ b913d5a9-6949-4a5c-9741-2395d965a779
show_img("sgd_logis.svg", w=450)

# ╔═╡ c2c7e41e-ca23-43b0-a9ee-77e15510bb02

md"""
## Learning rate

##### To ensure convergence, the learning rate ``\gamma_t`` usually is decaying 

```math 
\large \gamma_t = \frac{\gamma_0}{\sqrt{t}}\;\; \text{or}\;\; \gamma_t = \frac{\gamma_0}{1\, +\, \eta\, \cdot\, t}
```

* ##### ``\gamma_0``: initial learning rate
* ##### but constant also works well 
* ##### as the stochastic gradient can be noisy at the end
"""

# ╔═╡ 4e4ec9fd-c61f-422b-8cc4-89facf5aab50

TwoColumn(show_img("sgddelay.svg", w=390), show_img("sgddecay2.svg", w=350)  )

# ╔═╡ 00610b54-a7a5-4b57-ae1e-1fe9c263a46e
md"""

## Random shuffle

##### Three variants:

* ###### SGD: without shuffling


* ###### SS: single shuffle



* ###### RR: repeated random shuffle
"""

# ╔═╡ 36c4908a-fb2b-4c66-bebd-49d79c0b78a8
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/sgdcompare_logist.png
' width = '800' /></center>"

# ╔═╡ e3150185-e218-4b97-bc13-0d9965523025
md"""

\* Koloskova *et al* (2023) *Shuffle SGD is Always Better than SGD: Improved Analysis of SGD with Arbitrary Data Orders*
"""

# ╔═╡ 9e1f8587-af1f-4c32-a3a9-e2ef85ced356
md"""

## GD vs Mini-batch
"""

# ╔═╡ f8223f05-b3c4-4b8b-94ed-8f853e594bed
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/ml_stochastic.png
' width = '800' /></center>"

# ╔═╡ 3c8fbd4b-1ee3-4a93-8bfa-e26d28ad0da0
md"""
[*Watt, Borhani, and Katsaggelos (2023), Machine Learning Refined
Foundations, Algorithms, and Applications](https://www.cambridge.org/highereducation/books/machine-learning-refined/0A64B2370C2F7CE3ACF535835E9D7955#overview)
"""

# ╔═╡ cf207982-9a38-42f4-abc7-bdad2f9edf37
md"""
## Mini-batch SGD


-----

* ##### random guess ``\large\mathbf{w}_0``

* ##### for each *epoch*
  * ###### split the data into equal batches ``\{B_1, B_2,\ldots, B_m\}``
  * ###### for each batch `b` in ``\{B_1, B_2,\ldots, B_m\}``
    * ###### ``\large\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \gamma_{t-1} \frac{1}{|\mathtt{b}|} \sum_{i\in \mathtt{b}}\nabla \ell^{(i)}(\mathbf{w}_{t-1})``
-----




* ###### *e.g.* with a batch size ``5``
* ###### a trade-off between SGD and full batch GD



"""

# ╔═╡ cac6464a-21b2-4e9c-a54d-75159a907e06
begin
	Random.seed!(321)
	true_ww = [1., 1] 
	nobs = 100
	xxs = range(-2, 2, nobs)
	Xs = [ones(length(xxs)) xxs]
	Ys = rand(nobs) .< logistic.(Xs * true_ww)
end;

# ╔═╡ 03a478a1-d345-49d4-94aa-84376dd1055d
w00 = [-10, 20];

# ╔═╡ 151ece0b-c0ac-42ed-8a2b-2c25665df221
md"""

## Demonstration


##### A simulated logistic regression dataset
* ###### ``n=100`` training observations
* ###### ``100`` epochs (full data passes)
"""

# ╔═╡ 60713d75-dccb-407f-9657-599ed4553ece
show_img("gdvssgd.svg", w=500)

# ╔═╡ b5dadd63-f02d-490a-b57a-733268e98723
md"""

# Newton's method
"""

# ╔═╡ 7d4e0c2e-0f2b-4e06-a265-14e276ba9387
md"""

## Gradient descent: recap


> #### Gradient descent: 
> * ##### _use_ local linear approximations to guide the search _at each step_



"""

# ╔═╡ 002c9cac-6bc1-4288-903f-159afdcceef8
md"Add ``x_t``: $(@bind add_x0 CheckBox(false)), Add tangent: $(@bind add_approx CheckBox(false)), Add gradient: $(@bind add_grad CheckBox(false)), Add negative grad: $(@bind add_neggd CheckBox(false)), Add next: $(@bind add_nextstep CheckBox(false))"

# ╔═╡ 996fcbcb-1f30-4138-85cf-7b3e9a708505
# l0s = [[0, 0], [2, -1], [2, 2], [-1, 2], [-1, -1], [2, 0], [0, -1], [0, 2], [-1, 0]];

# ╔═╡ 757997c9-e461-4396-a60e-1c9626adfab7
md"""

## Gradient descent can be inefficient


##### -- Especially, when the loss surface is relatively flat

"""

# ╔═╡ e265f1b4-7017-405f-a3a2-7cb7253e6b47
md"""
## Gradient descent in multi-``d`` -- inefficient

##### -- the same applies to higher dimensional

"""

# ╔═╡ 6788a06d-fb80-4648-add2-4d279b397a48
md"""

## How about quadratic approximation?
"""

# ╔═╡ 61abede5-a616-437a-a1e4-2f4e04371c37
md"""

> ##### _Quadratic_ approximation (1-`dim`)
> ```math
> \Large
> \begin{align}
> \hat{f}(x) &= c + f'(x_0)\cdot (x-x_0) + \frac{1}{2}{{f''(x_0)}}\cdot(x-x_0)^2
> \end{align}
> ```

"""

# ╔═╡ 9793393a-a97d-407e-82e9-66ba3d4d3db3
md"""``x_0``: $(@bind x̂2 Slider(-4:0.1:3.5, default=2.6))"""

# ╔═╡ 91ace246-53a0-4a9d-a309-9a9a7fcf17c2
f3(x) = sin(1.95*x) + 0.5*x^2; # you can change this function!

# ╔═╡ a9fc4bad-9e77-49ec-8181-ff4826b72c8f
plt_quad_approx = let
	λ_gd_demo = 0.05
	x̂ = x̂2
	gr()
	f = f3
	λ = λ_gd_demo
    # Plot function
    xs = range(-4, 3.5, 200)
    ymin, ymax = extrema(f.(xs))
    p = plot(
        xs,
        f;
        label="",
        xlabel=L"x",
        legend=:bottomleft,
        ylims = (ymin - 2.5, ymax + .5),
		xlims = (-6, 4.5),
        legendfontsize=14,
		titlefontsize=16,
		lw = 2,
		ratio = .5,
		framestyle=:zerolines,
		size=(800,500)
    )
	scatter!([x̂], [f(x̂)], label="", mc=:white, msc=:gray, msw=2, alpha=0.5)

	if add_x0
		annotate!([x̂], [0.1], text(L"x_t"))
	end
    # Obtain the function 𝒟fₓ̃ᵀ
    # ŷ, 𝒟fₓ̂ᵀ = Zygote.pullback(f, x̂)
	fprime = ForwardDiff.derivative(f, x̂)
	fprimep = ForwardDiff.derivative(x -> ForwardDiff.derivative(f, x), x̂)
    function taylor_approx(x; x̂, order = 1) 
		fx = f(x̂) + fprime * (x - x̂)
		if order > 1
			fx += .5 * fprimep * (x-x̂)^2	
		end# f(x) ≈ f(x̃) + 𝒟f(x̃)(x-x̃)
		return fx
	end
	if true
    	plot!(p, range(x̂ - 1, x̂ +1, 3), (x) -> taylor_approx(x; x̂=x̂); label=L"linear approx. at $x_0$", lc=2,  lw=2, ls=:dash, title="Linear approximation")
	end
	if true
		x_center = x̂
		if (abs(fprimep) > 1e-5) && abs(fprime) < 1e-3
			x_center = (fprimep * x̂ -fprime)/ fprimep
		end
		plot!(p, range(x_center -4, x_center +4, 80), (x) -> taylor_approx(x; x̂=x̂, order=2); label=L"quadratic approx. at $x_0$: "*L"\hat{f}(x)", lc=3,  lw=2, ls=:solid, title="Quadratic approximation: "*L"\hat{f}(x)")
		# fpptxt = ""
		# if fprimep > 0
		# 	fpptxt = Plots.text(L"f^{''}(x)>0", 20, :green)
		# elseif abs(fprimep) < 1e-5
		# 	fpptxt = Plots.text(L"f^{''}(x)=0", 20,:green)

		# else
		# 	fpptxt = Plots.text(L"f^{''}(x)<0", 20,:green)
		# end
		# annotate!([x̂], [f(x̂)+0.9], fpptxt)

	end


	# if add_grad
	# 	xg = Zygote.gradient(f, x̂)[1]
	# 	xg = x̂ + xg
	# 	plot!([x̂, xg], [f(x̂), f(x̂)], lc=:gray, arrow=Plots.Arrow(:close, :head, 1, 1),  st=:path, label="")
	# 	annotate!(.5 * [x̂ + xg], [f(x̂)], text(L"f'(x_0)", 10, :bottom))
	# end


	# if add_neggd
	# 	xg = Zygote.gradient(f, x̂)[1]
	# 	x_new = x̂ - λ * xg
	# 	plot!([x̂, x_new], [f(x̂), f(x̂)], lc=:black, lw=2, arrow=Plots.Arrow(:close, :head, 1, 1),  st=:path, label="")
	# 	annotate!(.5 * [x̂ + x_new], [f(x̂)], text(L"-\gamma f'(x_0)", 8, :bottom))
	# end

	# if add_nextstep
	# 	xg = Zygote.gradient(f, x̂)[1]
	# 	x_new = x̂ - λ * xg
	# 	# x_new = x̂ + xg_neg
	# 	scatter!([x_new], [f(x_new)], label="", mc=:white, msc=:black, msw=2, alpha=1.0)
	# 	vline!([x_new]; style=:dash, c=:black, label="")
	# 	annotate!([x_new], [0.1], text(L"x_{new}", 10))
	# end

	p
end;

# ╔═╡ 16dc340d-bce6-4e0e-bbe8-a79202e6bd30
plt_quad_approx

# ╔═╡ 35f87da6-636b-48a1-9ee6-cbbf829063c7
md"""
## Quadratics are easy to optimise!

"""

# ╔═╡ 0ea8278a-9a88-4a18-8dd5-3bb403b61c16
md"""

> ##### _Quadratic_ approximation
> ```math
> \Large
> \begin{align}
> \hat{f}(x) &= c + g\cdot (x-x_0) + \frac{1}{2}{{h}}\cdot(x-x_0)^2
> \end{align}
> ```
* define ``c\triangleq f(x_0)``, ``g\triangleq f'(x_0)``, and ``h\triangleq f''(x_0)``
"""

# ╔═╡ 4c41c0fe-5a20-4ea9-b67d-034ce03ec4d7
md"""

##### Quadratic functions are easy to optimise!
```math
\Large
x_{new} \leftarrow \arg\min_x \hat{f}(x);\; \text{this is easy!}
```
```math
\large
\hat{f}'(x) = g  + \frac{1}{\cancel{2}}\cancel{2}h (x-x_0) = 0
```
	
```math
\large

h (x-x_0) = -g
```

```math
\large
h x=hx_0  -g
```

```math
\large

\boxed{x= x_0  -\frac{g}{h}}
```
"""

# ╔═╡ e34df139-7003-4eff-b948-8331af75d107
md"""

## Newton's method


> #### Newton's method
> * ##### _use_ local quadratic approximations to guide the search _at each step_

"""

# ╔═╡ 3d917876-92de-4e14-bb35-9028555b65ee
md"""$(@bind reset0 Button("Restart")) 
$(@bind next0 Button("Next step"))"""

# ╔═╡ a9931cea-a3bf-4965-9941-68bcefeb3286
begin
	reset0
	t3 = [1]
end;

# ╔═╡ bb684f2d-499d-4d4a-a95d-9b4f440046cc
# md"""

# ## Newton's method (`1-dim`)

# \


# -----

# * ##### random guess ``{w}_0``

# * ##### for ``t=1,2,\ldots``
#   * ###### ``\large g \leftarrow \ell'(w_t)``  # compute derivative
#   * ###### ``\large h \leftarrow \ell''(w_t)`` # compute second order derivative
#   * ###### ``\large w_{t} \leftarrow w_{t-1} - h^{-1}\cdot g`` # optimise the quadratic approx.
# -----


# """

# ╔═╡ cde0173f-d0c2-4331-9794-fbf8ddf4ac2a
function newton_method_univar(f, x₀; maxiters=10)
	xs = [x₀]
	losses = [f(x₀)]
	∇f(x) = ForwardDiff.derivative(f, x)
	∇∇f(x) = ForwardDiff.derivative(xx -> ForwardDiff.derivative(f, xx), x)
	for t in 1:maxiters
		g = ∇f(x₀)
		h = ∇∇f(x₀)
		x₀ = x₀ - g/h
		push!(xs, x₀)
		push!(losses, f(x₀))
	end
	xs, losses
end;

# ╔═╡ 34efc8ac-b761-48f3-922f-07a936f987bc
md"""

## Multi-dim quadratic
"""

# ╔═╡ bc860dc9-aa46-4139-9b53-300018580011
md"""

> ##### Local _quadratic_ approximation
> ```math
> \Large
> \begin{align}
> \hat{f}(\mathbf{x}) &= c + \mathbf{g}^{\top}\cdot (\mathbf{x}-\mathbf{x}_0) + \frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top\mathbf{H}(\mathbf{x}-\mathbf{x}_0)
> \end{align}
> ```
* where ``c \triangleq f(\mathbf{x}_0)``, ``\mathbf{g}\triangleq \nabla f(\mathbf{x}_0)`` is the gradient, and ``\mathbf{H} =\nabla^2f(\mathbf{x}_0)`` is the Hessian
"""

# ╔═╡ fda8f865-e2da-4311-b440-8e6eb5aa568f
md"""

#### Optimise it

```math
\large
\nabla_{\mathbf{x}} \hat{f}(\mathbf{x}) = \mathbf{g} + \frac{1}{\cancel{2}} \cancel{2} \mathbf{H}\cdot (\mathbf{x} -\mathbf{x}_0) =\mathbf{0}
```

```math
\large
\begin{align}
&\Rightarrow \mathbf{H}\cdot (\mathbf{x} -\mathbf{x}_0) =-\mathbf{g}\\
&\Rightarrow \mathbf{H}\mathbf{x}  =\mathbf{Hx}_0 -\mathbf{g}\\
&\Rightarrow \mathbf{x}  =\mathbf{H}^{-1}\mathbf{Hx}_0 -\mathbf{H}^{-1}\mathbf{g}\\
&\Rightarrow \mathbf{x} = \mathbf{x}_0 -\mathbf{H}^{-1}\mathbf{g}
\end{align}
```

"""

# ╔═╡ cfb159c5-bbd9-42a1-a555-4238da3787c9
md"""

## Newton's method (`multi-dim`)

\


-----

* ##### random guess ``\mathbf{w}_0``

* ##### for ``t=1,2,\ldots``
  * ###### ``\large \mathbf{g} \leftarrow \nabla \ell(\mathbf{w}_t)\;\;\;``  # compute gradient
  * ###### ``\large \mathbf{H} \leftarrow \nabla^2 \ell(\mathbf{w}_t)\;\;\;`` # compute Hessian
  * ###### ``\large \mathbf{w}_{t} \leftarrow \mathbf{w}_{t-1} - \mathbf{H}^{-1}\mathbf{g}\;\;\;`` # optimise the quadratic approx.
-----


"""

# ╔═╡ 63fdf6c8-a19e-4aea-aa49-f43833ddc3de
md"""

## Implementation details


##### In practice, we do not invert ``\mathbf{H}``, i.e. compute ``\mathbf{H}^{-1}``

* ###### instead, we can use matrix left divide operator `H \ g` to find the direction


-----

* ##### random guess ``\mathbf{w}_0``

* ##### for ``t=1,2,\ldots``
  * ###### ``\large \mathbf{g} \leftarrow \nabla \ell(\mathbf{w}_t)\;\;\;``  # compute gradient
  * ###### ``\large \mathbf{H} \leftarrow \nabla^2 \ell(\mathbf{w}_t)\;\;\;`` # compute Hessian
  * ###### ``\mathbf{d} = \mathbf{H}`` \ ``\mathbf{g}\;\;\;`` # solve ``\mathbf{H}\mathbf{d}=\mathbf{g}`` for ``\mathbf{d}``
  * ###### ``\large \mathbf{w}_{t} \leftarrow \mathbf{w}_{t-1} - \mathbf{d}\;\;\;`` # optimise the quadratic approx.
-----


"""

# ╔═╡ f306a611-2e71-429e-a2ad-404fac9f5768
md"""

## Demonstration
"""

# ╔═╡ 1952793d-8ad6-4820-984c-225a4d8fccc4
md"""$(@bind reset1 Button("Restart")) 
$(@bind next1 Button("Next step"))"""

# ╔═╡ ecd513ef-eb64-4202-9380-ed69d6f1e803
begin
	reset1
	t2 = [1]
end;

# ╔═╡ 7c31e0fb-298c-408d-9415-8904a639b715
md"""

## More demonstration
"""

# ╔═╡ 169db43d-88f6-4cfa-9bf0-cac033df6204
md"""
## Hessian matrix -- "second order derivative"*


> ##### The *Hessian* matrix is defined as 
> ```math
> \Large
>
> \mathbf{H} = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} &  \frac{\partial^2 f}{\partial x_1 \partial x_2} & \ldots & \frac{\partial^2 f}{\partial  x_1\partial x_n} \\
> \frac{\partial^2 f}{\partial x_2\partial x_1} & \frac{\partial^2 f}{\partial x_2^2} &  \ldots & \frac{\partial^2 f}{\partial x_2\partial x_n}\\
> \vdots & \vdots & \ddots & \vdots\\
> \frac{\partial^2 f}{\partial x_n\partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} &  \ldots & \frac{\partial^2 f}{\partial x_n^2}
> \end{bmatrix}
>
> ```

* ##### fact: _Hessian_ is symmetric
"""

# ╔═╡ b657e76e-dbd0-403a-ae5a-603ce9d22472
md"""
## What are ``\frac{\partial^2 f}{\partial x_i\partial x_j}``?


```math
\Large
\frac{\partial^2 f}{\partial x_1^2} ,\; \frac{\partial^2 f}{\partial x_i\partial x_j}
```
* #### it means taking _second order_ partial derivative (or take partial twice)

```math
	\frac{\partial^2 f}{\partial x_1^2} = \frac{\partial }{\partial x_1}\left (\frac{\partial f}{\partial x_1}\right)
```

```math
\frac{\partial^2 f}{\partial x_2 \partial x_1} = \frac{\partial }{\partial x_2}\left (\frac{\partial f}{\partial x_1}\right )
```
"""

# ╔═╡ e4f6b60f-714e-472d-9dc1-087c48e1b92f
md"""

## Example: ``f(\mathbf{x}) = \mathbf{x}^\top \mathbf{x}``



```math
\Large 
f(\mathbf{x}) =\mathbf{x}^\top \mathbf{x} = x_1^2 + x_2^2
```

* ##### The gradient is 

$$
\large
\nabla f(\mathbf{x}) = \begin{bmatrix}\frac{\partial f(\mathbf{x})}{\partial x_1}\\ \frac{\partial f(\mathbf{x})}{\partial x_2}\end{bmatrix} = \begin{bmatrix} 2 x_1\\ 2x_2\end{bmatrix} = 2 \mathbf{x}$$


* ##### The Hessian is

$$
\Large
\mathbf{H}(\mathbf{x}) = \begin{bmatrix}\frac{\partial^2 f(\mathbf{x})}{\partial x_1^2} & \frac{\partial^2 f(\mathbf{x})}{\partial x_1\partial x_2}\\ \frac{\partial^2 f(\mathbf{x})}{\partial x_2\partial x_1} & \frac{\partial f^2(\mathbf{x})}{\partial x_2^2}\end{bmatrix} =$$


"""

# ╔═╡ d03d7e28-3145-4d35-ab75-52991c65dfe7
md"""

## Example: ``f(\mathbf{x}) = \mathbf{x}^\top \mathbf{x}``


```math
\Large 
f(\mathbf{x}) =\mathbf{x}^\top \mathbf{x} = x_1^2 + x_2^2
```

* ##### The gradient is 

$$
\Large
\nabla f(\mathbf{x}) = \begin{bmatrix}\frac{\partial f(\mathbf{x})}{\partial x_1}\\ \frac{\partial f(\mathbf{x})}{\partial x_2}\end{bmatrix} = \begin{bmatrix} 2 x_1\\ 2x_2\end{bmatrix} = 2 \mathbf{x}$$


* ##### The Hessian is

$$
\Large
\mathbf{H}(\mathbf{x}) = \begin{bmatrix}\frac{\partial^2 f(\mathbf{x})}{\partial x_1^2} & \frac{\partial^2 f(\mathbf{x})}{\partial x_1\partial x_2}\\ \frac{\partial^2 f(\mathbf{x})}{\partial x_2\partial x_1} & \frac{\partial^2 f(\mathbf{x})}{\partial x_2^2}\end{bmatrix} = \begin{bmatrix}\textcolor{red}{\frac{\partial}{\partial x_1}\left (\frac{\partial (x_1^2+x_2^2)}{\partial x_1}\right )=\frac{\partial(2x_1)}{\partial x_1}} & \cdot \\ \cdot  & \cdot \end{bmatrix}$$


"""

# ╔═╡ 923c4081-5be8-4c6f-b3a2-79179db1b843
md"""

## Example: ``f(\mathbf{x}) = \mathbf{x}^\top \mathbf{x}``


```math
\Large 
f(\mathbf{x}) =\mathbf{x}^\top \mathbf{x} = x_1^2 + x_2^2
```

* ##### The gradient is 

$$
\Large
\nabla f(\mathbf{x}) = \begin{bmatrix}\frac{\partial f(\mathbf{x})}{\partial x_1}\\ \frac{\partial f(\mathbf{x})}{\partial x_2}\end{bmatrix} = \begin{bmatrix} 2 x_1\\ 2x_2\end{bmatrix} = 2 \mathbf{x}$$


* ##### The Hessian is

$$
\Large
\mathbf{H}(\mathbf{x}) = \begin{bmatrix}\frac{\partial^2 f(\mathbf{x})}{\partial x_1^2} & \frac{\partial^2 f(\mathbf{x})}{\partial x_1\partial x_2}\\ \frac{\partial^2 f(\mathbf{x})}{\partial x_2\partial x_1} & \frac{\partial^2 f(\mathbf{x})}{\partial x_2^2}\end{bmatrix} = \begin{bmatrix}\textcolor{red}{\frac{\partial(2x_1)}{\partial x_1}=2} & \cdot \\ \cdot  & \cdot \end{bmatrix}$$


"""

# ╔═╡ a9f2cdc1-ca80-4778-8374-70838f09930e
md"""

## Example: ``f(\mathbf{x}) = \mathbf{x}^\top \mathbf{x}``


```math
\Large 
f(\mathbf{x}) =\mathbf{x}^\top \mathbf{x} = x_1^2 + x_2^2
```

* ##### The gradient is 

$$
\large
\nabla f(\mathbf{x}) = \begin{bmatrix}\frac{\partial f(\mathbf{x})}{\partial x_1}\\ \frac{\partial f(\mathbf{x})}{\partial x_2}\end{bmatrix} = \begin{bmatrix} 2 x_1\\ 2x_2\end{bmatrix} = 2 \mathbf{x}$$


* ##### The Hessian is

$$
\Large
\mathbf{H}(\mathbf{x}) = \begin{bmatrix}\frac{\partial f^2(\mathbf{x})}{\partial x_1^2} & \frac{\partial f^2(\mathbf{x})}{\partial x_1\partial x_2}\\ \frac{\partial f(\mathbf{x})}{\partial x_2\partial x_1} & \frac{\partial f^2(\mathbf{x})}{\partial x_2^2}\end{bmatrix} = \begin{bmatrix}2 & \textcolor{red}{\frac{\partial}{\partial x_1} \left(\frac{\partial (x_1^2+x_2^2)}{\partial x_2}\right )=\frac{\partial(2x_2)}{\partial x_1}} \\ \cdot  & \cdot \end{bmatrix}$$


"""

# ╔═╡ 9005698a-28ee-4e6d-9648-41de3eb5e822
md"""

## Example: ``f(\mathbf{x}) = \mathbf{x}^\top \mathbf{x}``



```math
\Large 
f(\mathbf{x}) =\mathbf{x}^\top \mathbf{x} = x_1^2 + x_2^2
```

* ##### The gradient is 

$$
\large
\nabla f(\mathbf{x}) = \begin{bmatrix}\frac{\partial f(\mathbf{x})}{\partial x_1}\\ \frac{\partial f(\mathbf{x})}{\partial x_2}\end{bmatrix} = \begin{bmatrix} 2 x_1\\ 2x_2\end{bmatrix} = 2 \mathbf{x}$$


* ##### The Hessian is

$$
\Large
\mathbf{H}(\mathbf{x}) = \begin{bmatrix}\frac{\partial^2 f(\mathbf{x})}{\partial x_1^2} & \frac{\partial^2 f(\mathbf{x})}{\partial x_1\partial x_2}\\ \frac{\partial^2 f(\mathbf{x})}{\partial x_2\partial x_1} & \frac{\partial^2 f(\mathbf{x})}{\partial x_2^2}\end{bmatrix} = \begin{bmatrix}2 & 0\\ 0  & 2 \end{bmatrix} = 2\cdot \mathbf{I}$$


"""

# ╔═╡ 3ccab9e0-9d2c-4621-8e2e-064f1724154e
md"""

## More generally


```math
\Large 
f(\mathbf{x}) =\mathbf{x}^\top\mathbf{A} \mathbf{x} 
```

* ##### the Hessian: ``\mathbf{H}= 2\mathbf{A}``

* ##### similar to univariate result: ``f(x) =a x^2``, and ``f''(x) = 2a``
"""

# ╔═╡ bba249ff-9e05-4fc4-adc5-1654aafd0dfe
md"""

## Compute Hessian --auto-diff


##### `Julia` (`Zygote.jl`): `Zygote.hessian()`
```julia
Zygote.hessian(f, x0)
```


"""

# ╔═╡ e3cc868f-68df-422d-8094-8a4fa7af41f1
Zygote.hessian(x-> x'x, zeros(2))

# ╔═╡ f3b33961-033d-4bc8-bd98-c532db378e65
md"""

##### `Python` (`JAX`): `jax.hessian(.)` or (autograd) `autograd.hessian(.)`

```pyton
f = lambda x: sum(x**2)
jax.hessian(f)(jax.numpy.array([0., 0.])))
```

"""

# ╔═╡ 4d58697c-a812-420f-be40-d3a899093e96
md"""

## Issues with Newton's method



* ##### ``\mathbf{H}`` can be expensive to compute and invert


"""

# ╔═╡ 30521ba1-0146-4968-9d23-01ee9d2f6388
md"""

## Issues with Newton's method



* ##### ``\mathbf{H}`` can be expensive to compute and invert

\

* ##### it might accidentally maximise the loss, if loss is non-convex!
  * ###### it depends on starting point and local Hessian (negative definite, or curving down)
"""

# ╔═╡ 906238dd-21ba-42d9-a0f6-be9b4c603660
# produce_gif(produce_anim_newton_1d(f1, -3.5, 5), 5)

# ╔═╡ 2facfd77-7085-4931-9579-510d559a39ff
md"""

## Combine gradient descent & Newton's method

"""

# ╔═╡ 2f0efc3e-dc38-4232-b811-ad9a437450be
md"""

* ##### start with gradient descent then speed up with Newton's method

"""

# ╔═╡ b6e7802f-3ce2-4663-b6a7-432d5295547d
md"""

# Appendix
"""

# ╔═╡ a772e13c-cfc8-4ab8-ac82-512371b972cf
function perp_square(origin, vx, vy; δ=0.1) 
	x = δ * vx/sqrt(norm(vx))
	y = δ * vy/sqrt(norm(vy))
	xyunit = origin+ x + y
	xunit = origin + x
	yunit = origin +y
	Shape([origin[1], xunit[1], xyunit[1], yunit[1]], [origin[2], xunit[2], xyunit[2], yunit[2]])
end

# ╔═╡ 6968886a-b001-4dd5-8df9-e86db865f1c9
 function taylor_approx_univar(x, x̂, fx0, fp, fpp=0, order = 2) 
	fx = fx0 + fp * (x - x̂)
	if order > 1
		fx += .5 * fpp * (x - x̂)^2	
	end# 
	return fx
end

# ╔═╡ 35823e40-4c97-47bd-aba1-2936e455ad62
function produce_gif(plts, fps=5)
	anim = Animation()

	[frame(anim, plt) for plt in plts]

	gif(anim; fps=fps)
end

# ╔═╡ 768802ed-8c30-4497-943e-6f2c8fd0be9c
function produce_anim_newton_1d(f, x0, maxiters=10; xlims = (-4.5, 4.5), ylims = (-.8, .5), ratio=0.4, t0 = 0)
	my_colors = [cgrad(:BuPu, 0.2:0.1:10, rev=false, scale=:exp10)[z] for z ∈ range(0.3, 1.0, length = maxiters+1)]
	xxs, losses = newton_method_univar(f, x0; maxiters =maxiters)
	gr()
    xs = range(xlims..., 200)
    ymin, ymax = extrema(f.(xs))
    p = plot(xs, f; label="", xlabel=L"x",
        legend=:outerbottom,
        ylims = (ymin +ylims[1], ymax + ylims[2]),
		xlims = xlims, 
		legendfontsize=10, lw = 2,
		ratio = ratio, framestyle=:zerolines,title="Iteration $((t0+0))"
    )
	scatter!([x0], [f(x0)], label="", mc = my_colors[1], msc = my_colors[1], msw=2, alpha = 0.9)
	vline!([x0], lc = my_colors[1], label="", ls=:dash)
	annotate!([x0], [0], text(L"x_{%$(t0+0)}", 15, :top))
	plts = [p]
	for (t, x) in enumerate(xxs[1:end-1])
		plt = deepcopy(plts[end])
		fprime = ForwardDiff.derivative(f, x)
		fprimep = ForwardDiff.derivative(xx -> ForwardDiff.derivative(f, xx), x)
		x_center = x
		if (abs(fprimep) > 1e-5) && abs(fprime) < 1e-3
			x_center = (fprimep * x -fprime)/ fprimep
		end
		plot!(plt, range(x_center -4, x_center + 4, 200), (xx) -> taylor_approx_univar(xx, x, f(x), fprime, fprimep); label="", lc = my_colors[t+1],  lw=0.8, ls=:solid)	
		push!(plts, plt)
		plt2 = deepcopy(plts[end])
		xnext = xxs[t+1]
		scatter!(plt2, [xnext], [f(xnext)], label="", mc = my_colors[t+1], msc = my_colors[t+1], msw=2, alpha=0.9)
		vline!(plt2, [xnext], lc = my_colors[t+1], label="", ls=:dash)
		annotate!(plt2, [xnext], [0], text(L"x_{%$(t + t0)}", 15, :top), title = "Iteration $((t0+t))")
		push!(plts, plt2)
	end	
	plts
end

# ╔═╡ 03b6adf4-b0f9-4d6f-aeac-f74cb584fd48
plts_ = produce_anim_newton_1d(x -> x^4, 3.7, xlims =(-4., 4.0), ratio=0.015);

# ╔═╡ ba99be29-4b40-4229-bb34-91dd7c249549
let
	next0
	plt = plts_[t3[end]]
	t3[end] += 1
	if t3[end] > length(plts_)
		t3[end] = length(plts_)
	end
	plt
end

# ╔═╡ 6aac849d-17f5-4a46-ba0c-ec8d42034c93
function produce_anim_gd_1d(f, x0, maxiters=10, γ = 0.1; xlims = (-4.5, 4.5), ylims = (-.8, .5), ratio=0.4)
	my_colors = [cgrad(:BuPu, 0.2:0.1:10, rev=false, scale=:exp10)[z] for z ∈ range(0.3, 1.0, length = maxiters+1)]
	# xxs, losses = newton_method_univar(f, x0; maxiters =maxiters)
	xxs = [x0]
	losses = [f(x0)]
	x0_ = x0
	for _ in 1:maxiters
		x0_ = x0_ - γ * ForwardDiff.derivative(f, x0_)
		push!(xxs, x0_)
		push!(losses, f(x0_))
	end
	gr()
    xs = range(xlims..., 200)
    ymin, ymax = extrema(f.(xs))
    p = plot(xs, f; label="", xlabel=L"x",
        legend=:outerbottom,
        ylims = (ymin +ylims[1], ymax + ylims[2]),
		xlims = xlims, 
		legendfontsize = 10, lw = 2,
		ratio = ratio, framestyle=:zerolines,title="Iteration $(0)"
    )
	scatter!([x0], [f(x0)], label="", mc = my_colors[1], msc = my_colors[1], msw=2, alpha = 0.9)
	vline!([x0], lc = my_colors[1], label="", ls=:dash)
	annotate!([x0], [0], text(L"x_{%$(0)}", 15, :top))
	plts = [p]
	for (t, x) in enumerate(xxs[1:end-1])
		plt = deepcopy(plts[end])
		fprime = ForwardDiff.derivative(f, x)
		x_center = x
		plot!(plt, range(x_center -1.5, x_center + 1.5, 4), (xx) -> taylor_approx_univar(xx, x, f(x), fprime, 0, 1); label="", lc = my_colors[t+1],  lw=0.8, ls=:solid)	
		push!(plts, plt)
		plt2 = deepcopy(plts[end])
		xnext = xxs[t+1]
		scatter!(plt2, [xnext], [f(xnext)], label="", mc = my_colors[t+1], msc = my_colors[t+1], msw=2, alpha=0.9,  title = "Iteration $(t)")
		vline!(plt2, [xnext], lc = my_colors[t+1], label="", ls=:dash)
		if (t < 3) || (t == length(xxs[1:end-1]))
			annotate!(plt2, [xnext], [0], text(L"x_{%$(t)}", 15, :top))
		end
		push!(plts, plt2)
	end	
	plts
end

# ╔═╡ 7db906d6-b978-463f-a833-41e3e9baf2c3
anims_gd_1d = produce_anim_gd_1d(x -> x^4, 3.5, 10, 0.002; xlims =(-4,4), ratio=0.015);

# ╔═╡ 3da328c6-85cd-4c96-a849-e4db6600440d
@bind t1 Slider(1:length(anims_gd_1d))

# ╔═╡ 8902e2c1-7cd7-4088-9423-ddd4e3a0a48e
anims_gd_1d[t1]

# ╔═╡ a21a6912-18a6-49b8-8ba8-18df9006578a
f1(x) = sin(2.3x) + 0.3*x^2; # you can change this function!

# ╔═╡ 60e59cf7-d666-44c7-8ec1-dd95740f945f
x_pos = let
	x_pos = Float64[3.0]
	x = x_pos[1]
	λ = 0.1
	for i in 1:30
		xg = Zygote.gradient(f1, x)[1]
		x = x - λ * xg
		push!(x_pos, x)
	end

	x_pos
end;

# ╔═╡ 478d8fe1-9548-4fab-bc60-10c1b87d4e37
md"Move me: $(@bind x̂ Slider(x_pos))"

# ╔═╡ 91e1d742-9581-4b0e-bb80-0b444019dfe5
plt_linear_approx = let
	gr()
	f = f1
    # Plot function
    xs = range(-4, 4, 200)
    ymin, ymax = extrema(f.(xs))
	λ = 0.1

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
    vline!(p, [x̂]; style=:dash, c=:gray, label="")
    # Plot 1st order Taylor series approximation
    taylor_approx(x) = f(x̂) + 𝒟fₓ̂ᵀ(x - x̂)[1] # f(x) ≈ f(x̃) + 𝒟f(x̃)(x-x̃)
	if add_approx 
    	plot!(p, range(x̂ -1, x̂+1, 10), taylor_approx; label="", lc=2,  lw=1.2, ls=:solid)
	end

	if add_grad
		xg = Zygote.gradient(f, x̂)[1]
		xnew_g = x̂ + xg
		plot!([x̂, xnew_g], [f(x̂), f(x̂)], lc=:gray, arrow=Plots.Arrow(:close, :head, 1, 1),  st=:path, label="")
		annotate!(.5 * [x̂ + xnew_g], [f(x̂)], text(L"f'(x_0)=%$(round(xg;digits=2))", 10, :bottom))
	end
	if add_neggd
			xg = Zygote.gradient(f, x̂)[1]

		x_new = x̂ - λ * xg
		plot!([x̂, x_new], [f(x̂), f(x̂)], lc=:black, lw=2, arrow=Plots.Arrow(:close, :head, 1, 1),  st=:path, label="")
		annotate!(.5 * [x̂ + x_new], [f(x̂)], text(L"-\gamma f'(x_0)", 10, :bottom))
	end

	if add_nextstep
			xg = Zygote.gradient(f, x̂)[1]

		x_new = x̂ - λ * xg
		# x_new = x̂ + xg_neg
		scatter!([x_new], [f(x_new)], label="", mc=:white, msc=:black, msw=2, alpha=1.0)
		vline!([x_new]; style=:dash, c=:black, label="")
		annotate!([x_new], [0.1], text(L"x_{new}"))
	end

	p
end;

# ╔═╡ da573528-649b-4fbe-97f9-5d82fe0cc716
plt_linear_approx

# ╔═╡ a216e7ea-61f2-4165-8ec9-3ad9c38d4e7c
produce_gif(produce_anim_newton_1d(f1, -1.9, 5), 5)

# ╔═╡ 41c6fd98-843a-4f81-bb26-06da4cdacdc7
begin

	gamma_gd = 0.08
	n_steps = 5
	anims_gd = produce_anim_gd_1d(f1, -1.9, n_steps, gamma_gd)
	
end;

# ╔═╡ a74c430e-847b-4690-873c-bb0d26db325e
x_gd_newton = let
	x0 = -1.9

	for t in 1:n_steps
		g = ForwardDiff.derivative(f1, x0)

		x0 -= gamma_gd * g
	end
	x0
end;

# ╔═╡ c890f828-5131-4d5e-9264-d03776e7f338
anims_newtons = produce_anim_newton_1d(f1, x_gd_newton, 3; t0 = n_steps+1);

# ╔═╡ b9310442-2cda-4a9e-9c9e-238e06dcf11b
produce_gif([anims_gd..., anims_newtons...], 4)

# ╔═╡ fda7641b-38e9-493d-9aff-cf9ad646151c
f2= f1;

# ╔═╡ b4a17b0d-e26f-4c7e-8a4f-18fbd4407535
f4(x) = sum(x.^4) 

# ╔═╡ c75bb824-a7be-49b4-853d-1c39e9384e73
function produce_anim_gd(f, x0, maxiters=10, γ = 0.1; x1lims = (-0.1, 3.0), x2lims = (-0.1, 3.0), ratio=0.4, size=(600,500), camera =(30,30) ,keepappx = false)
	my_colors = [cgrad(:BuPu, 0.2:0.1:10, rev=false, scale=:exp10)[z] for z ∈ range(0.3, 1.0, length = maxiters+1)]
	xxs = [x0]
	losses = [f(x0)]
	x0_ = x0
	for _ in 1:maxiters
		x0_ = x0_ - γ * ForwardDiff.gradient(f, x0_)
		push!(xxs, x0_)
		push!(losses, f(x0_))
	end
	gr()
	x1_ = range(x1lims..., length=20)
	x2_ = range(x2lims..., length=20)
    zmin, zmax = extrema([f([x,y]) for x in x1_ for y in x2_])
	p = plot(x1_, x2_, (a, b) -> f([a, b]), st=:surface, xlim=x1lims .+ (-0.1, .1), ylim = x2lims .+ (-0.1, 0.1),  zlim = (min(0, zmin-0.5), zmax+0.5), xlabel =L"w_1", ylabel=L"w_2", zlabel=L"f(\mathbf{w})", colorbar=false, color=:jet, size = size, framestyle=:zerolines, camera=camera)
	scatter!([x0[1]], [x0[2]], [f(x0)],  label="", mc=my_colors[1], msw=2, alpha=1.0)
	scatter!([x0[1]], [x0[2]], [0], label=L"\mathbf{w}_0", ms =4, mc=my_colors[1], msc = my_colors[1], msw=2, alpha=1.0)
	plot!([x0[1], x0[1]], [x0[2], x0[2]], [f(x0), 0], lw=1, lc = my_colors[1], ls=:dash, label="")
	plts = [p]
	xlen, ylen = 0.5, 0.5
	for (t, x) in enumerate(xxs[1:end-1])
		plt = deepcopy(plts[end])
		∇f = ForwardDiff.gradient(f, x)
		tf(xx) = f(x) + ∇f' * (xx - x)
		if keepappx
			plot!(plt, range(x[1] - xlen, x[1] + xlen, 4), range(x[2] - ylen, x[2]+ylen, 4), (a, b) -> tf([a, b]), st=:surface, alpha = 0.2, c=my_colors[t])	
		else
			plot!(range(x[1] - xlen, x[1] + xlen, 4), range(x[2] - ylen, x[2]+ylen, 4), (a, b) -> tf([a, b]), st=:surface, alpha = 0.2, c=my_colors[t])
		end
		
		push!(plts, plt)
		plt2 = deepcopy(plts[end])
		xnext = xxs[t+1]
		scatter!(plt2, [xnext[1]], [xnext[2]], [f(xnext)],  label="", mc=my_colors[t+1], msc = my_colors[t+1], msw=2, alpha=0.9, title = "Iteration $(t)")
		scatter!([xnext[1]], [xnext[2]], [0], label="", ms =4, mc=my_colors[t+1], msc = my_colors[t+1], msw=2, alpha=1.0)
		plot!([xnext[1], xnext[1]], [xnext[2], xnext[2]], [f(xnext), 0], lw=1, lc = my_colors[t+1], ls=:dash, label="")
		push!(plts, plt2)
	end	
	plts
end

# ╔═╡ ed0a186a-b00f-45a1-bd40-a7c4e7eb5625
anim_gd2 = produce_anim_gd(x -> f4(x) + 10 , [2.0, 2.0], 15, 0.005; x1lims =(-2,2),x2lims=(-2,2), camera=(50,20));

# ╔═╡ cacbc107-8578-419d-9cff-31d3b4573954
produce_gif(anim_gd2, 2)

# ╔═╡ d72ea49f-c9c7-4d4b-ad91-c4417fe3994d
function newton_method(f, x₀; maxiters=10, ϵ = 1e-6, α = 1.0)
	xs = [x₀]
	losses = [f(x₀)]
	∇f(x) = ForwardDiff.gradient(f, x)
	∇∇f(x) = ForwardDiff.hessian(f, x)
	for t in 1:maxiters
		g = ∇f(x₀)
		H = ∇∇f(x₀)
		x₀ = x₀ - α * (H + ϵ * I)\g 
		push!(xs, x₀)
		push!(losses, f(x₀))
	end
	xs, losses
end

# ╔═╡ a463e924-9dbd-40ef-9f98-6e881778aabe
function produce_anim_newton(f, x0, maxiters=5, γ = 0.1; x1lims = (-0.1, 3.0), x2lims = (-0.1, 3.0), ratio=0.4, size=(600,500), keepappx = false)
	my_colors = [cgrad(:BuPu, 0.2:0.1:10, rev=false, scale=:exp10)[z] for z ∈ range(0.3, 1.0, length = maxiters+1)]
	xxs, losses = newton_method(f, x0; maxiters=maxiters)
	gr()
	x1_ = range(x1lims..., length=20)
	x2_ = range(x2lims..., length=20)
    zmin, zmax = extrema([f([x,y]) for x in x1_ for y in x2_])
	p = plot(x1_, x2_, (a, b) -> f([a, b]), st=:surface, xlim=x1lims .+ (-0.1, .1), ylim = x2lims .+ (-0.1, 0.1),  zlim = (min(0, zmin-0.5), zmax+0.5), xlabel =L"w_1", ylabel=L"w_2", zlabel=L"f(\mathbf{w})", colorbar=false, color=:jet, size = size, framestyle=:zerolines)
	scatter!([x0[1]], [x0[2]], [f(x0)],  label="", mc=my_colors[1], msw=2, alpha=1.0,  title = "Iteration 0")
	scatter!([x0[1]], [x0[2]], [0], label=L"\mathbf{w}_0", ms =4, mc=my_colors[1], msc = my_colors[1], msw=2, alpha=1.0)
	plot!([x0[1], x0[1]], [x0[2], x0[2]], [f(x0), 0], lw=1, lc = my_colors[1], ls=:dash, label="")
	plts = [p]
	xlen, ylen = 2.5, 2.5
	for (t, x) in enumerate(xxs[1:end-1])
		plt = deepcopy(plts[end])
		∇f = ForwardDiff.gradient(f, x)
		∇∇f = ForwardDiff.hessian(f, x)
		tf(xx) = f(x) + ∇f' * (xx - x) + 0.5 * (xx-x)'*∇∇f*(xx-x)
		if keepappx
			plot!(plt, range(x[1] - xlen, x[1] + xlen, 20), range(x[2] - ylen, x[2]+ylen, 20), (a, b) -> tf([a, b]), st=:surface, alpha = 0.2, c=my_colors[t])
		else
			plot!(range(x[1] - xlen, x[1] + xlen, 20), range(x[2] - ylen, x[2]+ylen, 20), (a, b) -> tf([a, b]), st=:surface, alpha = 0.2, c=my_colors[t])
		end
		push!(plts, plt)
		plt2 = deepcopy(plts[end])
		xnext = xxs[t+1]
		scatter!(plt2, [xnext[1]], [xnext[2]], [f(xnext)],  label="", mc=my_colors[t+1], msc = my_colors[t+1], msw=2, alpha=0.9, title = "Iteration $(t)")
		scatter!([xnext[1]], [xnext[2]], [0], label="", ms =4, mc=my_colors[t+1], msc = my_colors[t+1], msw=2, alpha=1.0)
		plot!([xnext[1], xnext[1]], [xnext[2], xnext[2]], [f(xnext), 0], lw=1, lc = my_colors[t+1], ls=:dash, label="")
		push!(plts, plt2)
	end	
	plts
end

# ╔═╡ 4567ad32-ee69-4c96-b345-62d86c73450f
anim_newt2 = produce_anim_newton(x -> f4(x) + 10 , [2.0, 2.0], 8;x1lims =(-2,2),x2lims=(-2,2),  keepappx=false);

# ╔═╡ d64c3a9d-e924-4a2b-a11c-b1e9f1c61a66
produce_gif(anim_newt2, 2)

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

# ╔═╡ 6384880a-0d63-4c5b-bedf-146437a02c22
anim_newt = produce_anim_newton(x -> f_demo(x) +3 , [2.9, 2.9]; keepappx=true);

# ╔═╡ 7339be91-1400-4122-abe6-3cdb2957b137
let
	next1
	plt = anim_newt[min(t2[end], length(anim_newt))]
	t2[end] = t2[end] + 1
	plt
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

# ╔═╡ de62b0cb-0ca8-4b6c-9edb-409328bcf540
# begin
# 	Random.seed!(123)
# 	nobs_new = 20
# 	x_train_new = randn(nobs_new, 2)
# 	y_train_new = x_train_new * ones(2) + randn(nobs_new)/5
# 	x_train_new_bad = copy(x_train_new)
# 	x_train_new_bad[:, end] = 10 * x_train_new[:, end]
# 	# x_train_new \ y_train_new
# end;

# ╔═╡ dfcbda48-4bde-418d-83ec-5d8dc1b5ab94
# function plot_rst(w; true_w = true_w_gab, x_train = x_train_gab, y_train=y_train_gab, add_truef=true)
# 	# plt = plot(-15:0.1:15, (x) -> gabor_f(x; w = true_w), lw=1, label="true function")
# 	plt = plot(-15:0.1:15, (x) -> gabor_f(x; w = w), lw=2, lc=2,label="fitted function")
# 	if add_truef
# 		plot!(-15:0.1:15, (x) -> gabor_f(x; w = true_w),lc=1, lw=1.5, label="true function")
# 	end
# 	scatter!(x_train, y_train, ms=3, c=1, label="training data")
# 	return plt
# end;

# ╔═╡ b6fca076-0639-490a-b77c-7a0155c25f39
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.

# ╔═╡ dbf14863-d497-447c-b50f-9f39812c71ec
D₂, targets_D₂, targets_D₂_=let
	Random.seed!(123)
	D_class_1 = rand(MvNormal(zeros(2), Matrix(1.2*[1 0.8; 0.8 1.0])), 20)' .+2
	D_class_2 = rand(MvNormal(zeros(2), Matrix(1.2*[1 0.8; 0.8 1.0])), 20)' .-2
	data₂ = [D_class_1; D_class_2]
	D₂ = [ones(size(data₂)[1]) data₂]
	targets_D₂ = [ones(size(D_class_1)[1]); zeros(size(D_class_2)[1])]
	targets_D₂_ = [ones(size(D_class_1)[1]); -1 *ones(size(D_class_2)[1])]
	D₂, targets_D₂,targets_D₂_
end;

# ╔═╡ 0af43292-6302-4e47-b7e8-95f2a84a0145
md"""Choose observation ``i``: $(@bind sgidx Slider(shuffle(1:size(D₂)[1]), show_value=true) )"""

# ╔═╡ 85260ea4-3946-4229-b8a1-2fd7b6737ea9
TwoColumn(
let
	plot(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], st=:scatter, c=1, label=L"y=0", ms=3, alpha=0.5, size=(350,350), titlefontsize=12, xlabel=L"x_1", ylabel=L"x_2")
	plot!(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], st=:scatter, c=2, label=L"y=1", ratio=1, framestyle=:zerolines, legendfontsize=12, title="Training data",  ms=3, alpha=0.5)
	cc = targets_D₂[sgidx] == 0 ? :blue : :red
	plot!([D₂[sgidx, 2]], [D₂[sgidx, 3]], st=:scatter, ms=5, label="", c = Int(targets_D₂[sgidx] +1), series_annotations = Plots.text(L"\mathbf{x}^{(i)}", cc, :bottom))

end , let
	gr()
	xlims = (-0.15, 1)
	# ylims =
	plt = plot(range(xlims..., 20), range(xlims..., 20), (w1, w2) -> logistic_loss([0, w1, w2], D₂, targets_D₂; agg = mean), st=:contourf, c=:coolwarm,  colorbar=false, title=L"-\nabla \ell"* " vs " * L"-{\nabla \ell^{(i)}}",  ratio=1, framestyle=:origins, xlim =xlims, ylim =xlims, xlabel=L"w_1", ylabel=L"w_1", alpha=0.3, size=(350,350), titlefontsize=12)


	w0 = [0.13, 0.13]

	yhats = logistic.(D₂ * [0, w0[1], w0[2]])
	∇li = (D₂ .* (yhats - targets_D₂))[:, 2:end] #ignore intercept w₀
	∇l = mean(∇li, dims=1)[:] 
	levelv = [1, -∇l[1]/∇l[2]]
	levelv = (levelv ./ norm(levelv))
	ts = -.2:0.1:0.2
	levelset = w0 .+ ((ts)' .* levelv)
	plot!(levelset[1,:], levelset[2,:], lw=2, lc=:purple, label="", st=:path)
	# plot!(range(w0[1]-0.5, w0[1]+0.5, 5), x-> , lw=1, lc=:gray, label="")
	plot!(perp_square(w0, levelv, -∇l ; δ=0.05), lw=1, lc=:purple, label="", fillcolor=false)
	r = 0.6
	sgw_new = w0' .- r * ∇li
	gw_new = w0 - r * ∇l
	first_neg = true
	# first_pos = true
	
	for w in eachrow(sgw_new)
		if (gw_new - w0)' * (w - w0) > 0 
			plot!([w0[1], w[1]], [w0[1], w[2]], line = (:arrow, 0.15, :blue), label="")
	
		else
			plot!([w0[1], w[1]], [w0[1], w[2]], line = (:arrow, 0.8, :red), label="")
			
		end
	end
	δy = (∇li * ∇l) 
	ci = δy[sgidx] > 0 ? :blue : :red
	plot!([w0[1], sgw_new[sgidx, 1]], [w0[1], sgw_new[sgidx,2]], line = (:arrow, 1, ci), label="")
	annotate!([sgw_new[sgidx,1]], [sgw_new[sgidx, 2]], text(L"-\nabla \ell^{(i)}", 15, ci, :bottom))
	
	imax = argmax(δy)

	imin = argmin(δy)

	plot!([w0[1], gw_new[1]], [w0[1], gw_new[2]], line = (:arrow, 3, :black), label="")

	annotate!([gw_new[1]], [gw_new[2]], text(L"-\nabla \ell(\mathbf{w})", :left))

	plt 
end)

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

# ╔═╡ 7c6e1fd4-aa87-45c0-9c49-6dab2de90043
linear_approx_f(x; f, ∇f, x₀) = f(x₀) + dot(∇f(x₀), (x-x₀));

# ╔═╡ e54c9f2d-ae25-46f1-8fcc-a75184a3610b
begin
	A = Matrix(I, 2, 2)
	f(x) = dot(x, A, x)
	∇f(x) = 2* A* x
end;

# ╔═╡ ed7f77ac-0cc5-4141-b88b-c5a86749ddd6
x₀ = [-5, 5];

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
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
Distributions = "~0.25.107"
ForwardDiff = "~0.10.36"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.1"
Latexify = "~0.15.17"
LogExpFunctions = "~0.3.18"
Plots = "~1.39.0"
PlutoTeachingTools = "~0.2.5"
PlutoUI = "~0.7.48"
StatsBase = "~0.34.2"
Zygote = "~0.6.69"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "637a19d25c7629417a84651e8514fa8e23373c1a"

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
git-tree-sha1 = "545a177179195e442472a1c4dc86982aa7a1bef0"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.7"

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

[[deps.Formatting]]
deps = ["Logging", "Printf"]
git-tree-sha1 = "fb409abab2caf118986fc597ba84b50cbaf00b87"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.3"

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
git-tree-sha1 = "ed6834e95bd326c52d5675b4181386dfbe885afb"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.55.5+0"

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

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

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
# ╟─3f3eae53-262d-4ef9-bda3-466b870af972
# ╟─a0ca254a-8de0-4693-b2b3-d7620e22c843
# ╟─2fa021dd-0040-438c-91bf-3a4359d9a760
# ╟─7f6641c5-9052-46be-9926-e3531d4ebab9
# ╟─8e78770a-536b-4555-a312-0ba7006be5b0
# ╟─0af43292-6302-4e47-b7e8-95f2a84a0145
# ╟─85260ea4-3946-4229-b8a1-2fd7b6737ea9
# ╟─9601b450-c167-4c14-b627-f6b229e93788
# ╟─521cda12-8e0f-4df9-81e8-a6d9178203cb
# ╟─66768384-cb86-4aa4-856f-bf70a7c2bf9a
# ╟─7c3a0677-6c78-44ec-a3be-aca9f3a59197
# ╟─10b08e9c-ad61-46ad-b650-c839159727aa
# ╟─f62458b2-8837-435d-801f-0bf9689086c7
# ╟─1d38cea7-7e0c-4d2b-bd30-ce307fed0131
# ╟─b913d5a9-6949-4a5c-9741-2395d965a779
# ╟─c2c7e41e-ca23-43b0-a9ee-77e15510bb02
# ╟─4e4ec9fd-c61f-422b-8cc4-89facf5aab50
# ╟─00610b54-a7a5-4b57-ae1e-1fe9c263a46e
# ╟─36c4908a-fb2b-4c66-bebd-49d79c0b78a8
# ╟─e3150185-e218-4b97-bc13-0d9965523025
# ╟─9e1f8587-af1f-4c32-a3a9-e2ef85ced356
# ╟─f8223f05-b3c4-4b8b-94ed-8f853e594bed
# ╟─3c8fbd4b-1ee3-4a93-8bfa-e26d28ad0da0
# ╟─cf207982-9a38-42f4-abc7-bdad2f9edf37
# ╟─cac6464a-21b2-4e9c-a54d-75159a907e06
# ╟─03a478a1-d345-49d4-94aa-84376dd1055d
# ╟─151ece0b-c0ac-42ed-8a2b-2c25665df221
# ╟─60713d75-dccb-407f-9657-599ed4553ece
# ╟─b5dadd63-f02d-490a-b57a-733268e98723
# ╟─7d4e0c2e-0f2b-4e06-a265-14e276ba9387
# ╟─478d8fe1-9548-4fab-bc60-10c1b87d4e37
# ╟─002c9cac-6bc1-4288-903f-159afdcceef8
# ╟─da573528-649b-4fbe-97f9-5d82fe0cc716
# ╟─91e1d742-9581-4b0e-bb80-0b444019dfe5
# ╟─996fcbcb-1f30-4138-85cf-7b3e9a708505
# ╟─60e59cf7-d666-44c7-8ec1-dd95740f945f
# ╟─757997c9-e461-4396-a60e-1c9626adfab7
# ╟─7db906d6-b978-463f-a833-41e3e9baf2c3
# ╟─3da328c6-85cd-4c96-a849-e4db6600440d
# ╟─8902e2c1-7cd7-4088-9423-ddd4e3a0a48e
# ╟─e265f1b4-7017-405f-a3a2-7cb7253e6b47
# ╟─cacbc107-8578-419d-9cff-31d3b4573954
# ╟─ed0a186a-b00f-45a1-bd40-a7c4e7eb5625
# ╟─6788a06d-fb80-4648-add2-4d279b397a48
# ╟─61abede5-a616-437a-a1e4-2f4e04371c37
# ╟─9793393a-a97d-407e-82e9-66ba3d4d3db3
# ╟─16dc340d-bce6-4e0e-bbe8-a79202e6bd30
# ╟─91ace246-53a0-4a9d-a309-9a9a7fcf17c2
# ╟─a9fc4bad-9e77-49ec-8181-ff4826b72c8f
# ╟─35f87da6-636b-48a1-9ee6-cbbf829063c7
# ╟─0ea8278a-9a88-4a18-8dd5-3bb403b61c16
# ╟─4c41c0fe-5a20-4ea9-b67d-034ce03ec4d7
# ╟─e34df139-7003-4eff-b948-8331af75d107
# ╟─3d917876-92de-4e14-bb35-9028555b65ee
# ╟─a9931cea-a3bf-4965-9941-68bcefeb3286
# ╟─ba99be29-4b40-4229-bb34-91dd7c249549
# ╟─03b6adf4-b0f9-4d6f-aeac-f74cb584fd48
# ╟─bb684f2d-499d-4d4a-a95d-9b4f440046cc
# ╟─cde0173f-d0c2-4331-9794-fbf8ddf4ac2a
# ╟─34efc8ac-b761-48f3-922f-07a936f987bc
# ╟─bc860dc9-aa46-4139-9b53-300018580011
# ╟─fda8f865-e2da-4311-b440-8e6eb5aa568f
# ╟─cfb159c5-bbd9-42a1-a555-4238da3787c9
# ╟─63fdf6c8-a19e-4aea-aa49-f43833ddc3de
# ╟─f306a611-2e71-429e-a2ad-404fac9f5768
# ╟─1952793d-8ad6-4820-984c-225a4d8fccc4
# ╟─7339be91-1400-4122-abe6-3cdb2957b137
# ╟─ecd513ef-eb64-4202-9380-ed69d6f1e803
# ╟─6384880a-0d63-4c5b-bedf-146437a02c22
# ╟─7c31e0fb-298c-408d-9415-8904a639b715
# ╟─d64c3a9d-e924-4a2b-a11c-b1e9f1c61a66
# ╟─4567ad32-ee69-4c96-b345-62d86c73450f
# ╟─169db43d-88f6-4cfa-9bf0-cac033df6204
# ╟─b657e76e-dbd0-403a-ae5a-603ce9d22472
# ╟─e4f6b60f-714e-472d-9dc1-087c48e1b92f
# ╟─d03d7e28-3145-4d35-ab75-52991c65dfe7
# ╟─923c4081-5be8-4c6f-b3a2-79179db1b843
# ╟─a9f2cdc1-ca80-4778-8374-70838f09930e
# ╟─9005698a-28ee-4e6d-9648-41de3eb5e822
# ╟─3ccab9e0-9d2c-4621-8e2e-064f1724154e
# ╟─bba249ff-9e05-4fc4-adc5-1654aafd0dfe
# ╠═e3cc868f-68df-422d-8094-8a4fa7af41f1
# ╟─f3b33961-033d-4bc8-bd98-c532db378e65
# ╟─4d58697c-a812-420f-be40-d3a899093e96
# ╟─30521ba1-0146-4968-9d23-01ee9d2f6388
# ╟─a216e7ea-61f2-4165-8ec9-3ad9c38d4e7c
# ╟─906238dd-21ba-42d9-a0f6-be9b4c603660
# ╟─2facfd77-7085-4931-9579-510d559a39ff
# ╟─2f0efc3e-dc38-4232-b811-ad9a437450be
# ╟─b9310442-2cda-4a9e-9c9e-238e06dcf11b
# ╟─41c6fd98-843a-4f81-bb26-06da4cdacdc7
# ╟─a74c430e-847b-4690-873c-bb0d26db325e
# ╟─c890f828-5131-4d5e-9264-d03776e7f338
# ╟─b6e7802f-3ce2-4663-b6a7-432d5295547d
# ╟─a772e13c-cfc8-4ab8-ac82-512371b972cf
# ╟─6968886a-b001-4dd5-8df9-e86db865f1c9
# ╟─35823e40-4c97-47bd-aba1-2936e455ad62
# ╟─768802ed-8c30-4497-943e-6f2c8fd0be9c
# ╟─6aac849d-17f5-4a46-ba0c-ec8d42034c93
# ╟─a21a6912-18a6-49b8-8ba8-18df9006578a
# ╟─fda7641b-38e9-493d-9aff-cf9ad646151c
# ╟─b4a17b0d-e26f-4c7e-8a4f-18fbd4407535
# ╟─c75bb824-a7be-49b4-853d-1c39e9384e73
# ╟─a463e924-9dbd-40ef-9f98-6e881778aabe
# ╟─d72ea49f-c9c7-4d4b-ad91-c4417fe3994d
# ╟─dfb3736b-10a1-429a-a561-f942b3ac7549
# ╟─4814ab50-3ed2-4197-b116-719ccbb28939
# ╟─f477c82e-f4d6-4918-85d1-19ce21e90b5b
# ╟─b6fd9d29-b5ac-4940-90d8-fcf0d8063003
# ╟─f4a1f7db-0bcb-45b6-be9d-1c57dd6e2b99
# ╟─de62b0cb-0ca8-4b6c-9edb-409328bcf540
# ╟─dfcbda48-4bde-418d-83ec-5d8dc1b5ab94
# ╟─b6fca076-0639-490a-b77c-7a0155c25f39
# ╟─30a9cc96-092d-4122-8101-1878fa83d1cb
# ╟─dbf14863-d497-447c-b50f-9f39812c71ec
# ╟─961ac176-6ade-4bf8-b8e9-d6d78ce664a0
# ╟─7c6e1fd4-aa87-45c0-9c49-6dab2de90043
# ╟─e54c9f2d-ae25-46f1-8fcc-a75184a3610b
# ╟─ed7f77ac-0cc5-4141-b88b-c5a86749ddd6
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
