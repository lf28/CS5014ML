### A Pluto.jl notebook ###
# v0.19.36

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

# ╔═╡ 275a2ccb-254d-473c-bb67-50d80cdf590c
begin

using NonlinearSolve

function f(u, p)
    u .* u .- p
end
u0 = [1.0, 1.0]
p = 2.0
prob = NonlinearProblem(f, u0, p)
sol = solve(prob, NewtonRaphson())
end

# ╔═╡ f79bd8ab-894e-4e7b-84eb-cf840baa08e4
using Logging

# ╔═╡ cb72ebe2-cea8-4467-a211-5c3ac7af74a4
TableOfContents()

# ╔═╡ f9023c9e-c529-48a0-b94b-31d822dd4a11
ChooseDisplayMode()

# ╔═╡ d11b231c-3d4d-4fa2-8b1c-f3dd742f8977
md"""

# CS5014 Machine Learning


#### Optimization basics & Gradient
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 36d19f22-cada-446e-a7df-30be835cc373
md"""

## Reading & references

##### Essential reading 


* [_Deep Learning_ by _Ian Goodfellow et al._: Chapter 4.3](https://www.deeplearningbook.org/contents/numerical.html)


##### Suggested reading 
* [_Linear algebra review and reference_ by Zico Kolter](https://studres.cs.st-andrews.ac.uk/CS5014/0-General/cs229-linalg.pdf) section 3.11 and 4
"""

# ╔═╡ c23db844-4f23-47db-8189-ed5f3035692a
md"""

# Univariate calculus refresher
"""

# ╔═╡ 463a8681-9989-4d90-897d-c8df3a328274
md"""

## Recap: linear function



```math
\huge
f(x) = b\cdot x+ c
```

* ##### ``c``: intercept
* ##### ``b``: slope
"""

# ╔═╡ a03ae249-b644-4269-b695-0ce8bb13a276
TwoColumn(let
	gr()
	b₁, b₀ = 1.5, 0
	plt = plot(  title="Effect of intercept: "*L"c", size=(350, 500))

	bbs = [[b₁, b₀]  for b₀ in -3:3]
	for (b₁, b₀) in bbs
		if b₀ < 0 
			anno_text = L"f(x) = %$(b₁)x %$(b₀)"
		else
			anno_text = L"f(x) = %$(b₁)x + %$(b₀)"
		end
		plot!(-1:0.1:3, (x) -> b₁*x+b₀, framestyle=:origin, label=anno_text, legend=:outerbottom, lw=2, ylim =[-4, 10])
	end
	plt
end, let
	gr()
	a, b = 0, 2
	plt = plot( legend=:outerbottom, title="Effect of slope: "*L"b", size=(350, 500))

	abs = [(-2, b),  (-1.5, b), (-1, b), (-0.5, b),  (0,b), (.5, b), (1, b), (1.5, b), (2, b)]
	for (a, b) in abs
		plot!(-2:0.1:3, (x) -> a*x+b, framestyle=:origin, label=L"f(x) = %$(a)x + %$(b)",  lw=2, xlim =[-2,3], ylim = [-4,12])
	end
	plt
end)

# ╔═╡ bfcab553-2f25-4481-9b4c-1e5466100428
md"""

## Recap: quadratic function


```math
\huge 
f(x) = ax^2 + b x+ c, \;\; a\neq 0

```
"""

# ╔═╡ 2afb3251-4471-496d-a4c5-aff3df1b5ae6
pltapos, pltaneg=let
	gr()
	b, c = 0, 0
	plt = plot( legend=:outerbottom, title="Effect of "*L"a>0", size=(300,400))
	plt2 = plot( legend=:outerbottom, title="Effect of "*L"a<0", size=(300,400))
	
	ass = [0.1, 1,2,3,4,6]
	for a in ass
		plot!(plt, -5:0.2:5, (x) -> a* x^2 + b* x+ c, framestyle=:origin, label=L"f(x) = %$(a)x^2", lw=2)
		plot!(plt2, -5:0.2:5, (x) -> -a * x^2 + b* x+ c, framestyle=:origin, label=L"f(x) = -%$(a)x^2", lw=2)
	end


	plt, plt2
end;

# ╔═╡ ecb880b2-242a-4a0c-a727-caa2ef09f89c
TwoColumn(md"

#### when `` a > 0``


The function has a **minimum**

$(pltapos)
", 
	
	
md" #### when `` a<0``


The function has a **maximum**


$(pltaneg)
")

# ╔═╡ 7b3cf4f1-1f86-4fec-9ca3-4f1f5e977512
md"""

## Differential calculus -- the big picture


#### Given a non-linear ``f``
* #### but if zoom-in, what do you observe?

"""

# ╔═╡ 992344fe-1f96-41e4-abbd-cae8501a19b5
Foldable("Zoom-in observation", md"""

(``f\Rightarrow`` a straight line)
""")

# ╔═╡ 80492c19-3263-4824-8fe6-49789d341d4e
f(x) = x * sin(x^2) + 1; # you can change this function!

# ╔═╡ 728c0b5e-5bf5-4c1d-8b7b-c7a41bae4f7e
let
	plotly()
    # Plot function
    xs = range(-2, 3, 2000)
    ymin, ymax = extrema(f.(xs))
    p = plot(
        xs,
        f;
        # label=L"$f(x)$",
        # xlabel=L"x",
        legend=:outerbottom,
        ylims = (ymin - .5, ymax + .5),
        legendfontsize=10,
		lw = 3,
		ratio = .7,
		label="",
		framestyle=:zerolines,
		size=(700,400)
    )

  
end

# ╔═╡ 35ddd576-28fe-4a08-90cc-ec977a3da9b3
md"""

### _Indeed_, 
> ### `differentiable` ``\Leftrightarrow`` _can be linearly approximated_


> ### `differential calculus` ``\Leftrightarrow`` _linearization_
"""

# ╔═╡ ef2ce0c0-887f-4d7c-be7b-74be3e1d3ac7
md"""

## Essense of differential calculus 


#### Approximate _non-linear_ ``f(x)`` with _simpler (linear) functions_
"""

# ╔═╡ fdb084c6-e5e7-4be4-bc52-6d23fc65c84e
md"""
 
> #### Non-linear ``f(x)`` can be approximated locally by
> * ##### a _linear function_ 

"""

# ╔═╡ c4290f2d-ee0a-475a-909a-7876bc32a147
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/linearappx.svg" width = "450"/></center>"""

# ╔═╡ 6d415e5a-ffbd-45d3-badc-13741a009438
Foldable("More formally*", md"""


> ```math
> f(x) = f(x_0) + f'(x_0)(x-x_0)  + o(|x-x_0|)
> ```

where the small ``o`` denotes that the function is an order of magnitude smaller around 𝑥0 than the function ``|x -x_0|``.

""")

# ╔═╡ 856b97e5-88a7-4399-b449-1fef8aae58e6
md"Show ``x_0``: $(@bind show_x0 CheckBox(false))"

# ╔═╡ fe6a4422-eebb-40db-b3ef-5ab4736798da
md"Add linear approx.: $(@bind add_linear CheckBox(false)); Move me ``x_0``: $(@bind x̂_ Slider(-2:0.005:3, default=-1.355, show_value=true))"

# ╔═╡ 3cdfb6b3-18c0-49f1-95d7-d21a0e9891b5
let
	gr()
	x̂ = x̂_
    # Plot function
    xs = range(-2, 3, 200)
    ymin, ymax = extrema(f.(xs))
    p = plot(
        xs,
        f;
        label=L"$f(x)$",
        xlabel=L"x",
        legend=:outerbottom,
        ylims = (ymin - .5, ymax + .5),
        legendfontsize=15,
		lw = 3,
		ratio = .7,
		framestyle=:zerolines,
		size=(800,600)
    )

	if show_x0
		scatter!([x̂],[f(x̂)], label = "", mc=:white, msc=:gray, msw=2, alpha=2.0, series_annotations=(text(L"x_0", :top, 25)))
		# annotate!()
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
	if add_linear
    	plot!(p, xs, (x) -> taylor_approx(x; x̂=x̂); label=L"linear approx. at $x_0$", lc=2,  lw=2, title="Linear approximation")
	end
	# if add_quadratic
	# 	plot!(p, xs, (x) -> taylor_approx(x; x̂=x̂, order=2); label=L"quadratic approx. at $x_0$", lc=3,  lw=3, title="Quadratic approximation")

	# end

	p

	# xq = x̂ + 0.9
	# annotate!([xq], [taylor_approx(xq; x̂=x̂, order=2)], text("Quadratic approx"))

end

# ╔═╡ d412c90d-f170-42b9-83ff-d4959f043dbf
md"""

## Derivative -- `f'(x)`



* ##### `dirivative`: *limit* of _change ratio_
"""

# ╔═╡ 77c1a618-055a-449a-a737-12789624ed7f
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/diriv_.svg" width = "900"/></center>"""

# ╔═╡ 1d6d6565-c8e3-4f65-a581-29a6a5d8da5e
md"``\Delta x``: $(@bind Δx Slider(1.5:-0.1:0, default=1.5)), Add approx area: $(@bind add_neighbour CheckBox(default=false))"

# ╔═╡ acacb324-8eb2-427e-85fe-354a56fd0df9
let
	gr()
	x₀ = 0.0
	xs = -1.2π : 0.1: 1.2π
	f, ∇f = sin, cos
	# anim = @animate for Δx in π:-0.1:0.0
	# Δx = 1.3
	plt = plot(xs, sin, label=L"\sin(x)", ylim = [-1.5, 1.5], xlabel=L"x", lw=2, legend=:outerbottom, framestyle=:semi, title="Derivative at "*L"x=0", legendfontsize=10, ylabel=L"f")


	df = f(x₀ + Δx)-f(x₀)
	k = Δx == 0 ? ∇f(x₀) : df/Δx
	b = f(x₀) - k * x₀ 
	# the approximating linear function with Δx 
	plot!(xs, (x) -> k*x+b, label="", lw=2, lc=2)
	# the location where the derivative is defined
	scatter!([x₀], [f(x₀)], ms=3, label=L"x_0,\; \sin(x_0)")
	scatter!([x₀+Δx], [f(x₀+Δx)], ms=3, label=L"x_0+Δx,\; \sin(x_0+Δx)")
	plot!([x₀, x₀+Δx], [f(x₀), f(x₀)], lc=:gray, label="")
	plot!([x₀+Δx, x₀+Δx], [f(x₀), f(x₀+Δx)], lc=:gray, label="")
	font_size = Δx < 0.8 ? 12 : 14
	annotate!(x₀+Δx, 0.5 *(f(x₀) + f(x₀+Δx)), text(L"Δf", font_size, :top, rotation = 90))
	annotate!(0.5*(x₀+x₀+Δx), 0, text(L"Δx", font_size,:top))
	annotate!(-.6, 1, text(L"\frac{Δf}{Δx}=%$(round(k, digits=2))", 15,:top))
	if add_neighbour
		vspan!([-0.5, .5], ylim =(-1.5,1.5),  alpha=0.5, c=:gray, label="")
	end
	plt
end

# ╔═╡ 99054046-a379-4c9e-b93e-5822d2ea00dd
md"""

## Derivative
"""

# ╔═╡ a49a52c9-f6ca-4dfc-afa1-8af648763ef5
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/05-example-monotonicity-derivatives.png" width = "500"/></center>"""

# ╔═╡ c05f97ed-c92c-4895-a44e-bfc497a8dc78
md"""


## Optimisation


"""

# ╔═╡ e432d785-7107-490f-8a4d-00893077e764
TwoColumn(md"""

#### To optimise ``f``

```math
\Large 
x_{\text{min}} \leftarrow \arg\min_x f(x)
```
##### _or_ 

```math
\Large 
x_{\text{max}} \leftarrow \arg\min_x f(x)
```
##### Solve for ``x``

```math
\Large
\frac{\mathrm{d}f}{\mathrm{d}x}(x) = 0
```


""", md"
![](https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/05-example-monotonicity-derivatives.png)
")

# ╔═╡ e7de66c8-c332-48fc-b4ca-57584d1aa364
md"""

## Essense of differential calculus 

#### - higher order differentials

#### _further more_, approximate _non-linear_ ``f(x)`` by _quadratic_ functions
"""

# ╔═╡ 80b3ef8f-42d2-4c11-a22e-71b8b1462173
Foldable("Second order approximation", md"""

> ##### Non-linear ``f(x)`` can be approximated by a _quadratic function_
> ```math
> \Large
> \begin{align}
> f(x) &\approx f(x_0) + f'(x_0)(x-x_0) + \frac{1}{2}\underbrace{\boxed{f^{''}(x_0)}}_{\text{\small second order derivative}}(x-x_0)^2
> \end{align}
> ```

""")

# ╔═╡ 2a5b2096-7c06-4173-9bbe-cb684de63bc3
md"Show ``x_0``: $(@bind show_x02 CheckBox(false)); Add linear approx.: $(@bind add_linear2 CheckBox(false)); Add quadratic approx.: $(@bind add_quadratic2 CheckBox(false)); Move me ``x_0``: $(@bind x̂2 Slider(-2:0.005:3, default=-1.355, show_value=true))"

# ╔═╡ 8b6fd167-eaca-44ac-9fab-868e7f639a4e
let
	gr()
	x̂ = x̂2
    # Plot function
    xs = range(-2, 3, 200)
    ymin, ymax = extrema(f.(xs))
    p = plot(
        xs,
        f;
        label=L"$f(x)$",
        xlabel=L"x",
        legend=:outerbottom,
        ylims = (ymin - .5, ymax + .5),
        legendfontsize=15,
		lw = 3,
		ratio = .7,
		framestyle=:zerolines,
		size=(800,600)
    )

	if show_x02
		scatter!([x̂],[f(x̂)], label = "", mc=:white, msc=:gray, msw=2, alpha=2.0, series_annotations=(text(L"x_0", :top, 25)))
		# annotate!()
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
	if add_linear2
    	plot!(p, xs, (x) -> taylor_approx(x; x̂=x̂); label=L"linear approx. at $x_0$", lc=2,  lw=2, title="Linear approximation")
	end
	if add_quadratic2
		plot!(p, xs, (x) -> taylor_approx(x; x̂=x̂, order=2); label=L"quadratic approx. at $x_0$", lc=3,  lw=3, title="Quadratic approximation")

	end

	p

	# xq = x̂ + 0.9
	# annotate!([xq], [taylor_approx(xq; x̂=x̂, order=2)], text("Quadratic approx"))

end

# ╔═╡ 631125ee-4c81-4c43-bb51-0e9c63826028
md"""
## Optimisation: max/min test

#### - implication of second order approx
"""

# ╔═╡ a2613587-64bb-4d2a-95a4-a21ce6f0a77e
TwoColumn(md"""

#### To optimise ``f``
* ``\arg\max_x f(x)`` or ``\arg\min_x f(x)``

#### Solve for ``x``

```math
\Large
\frac{\mathrm{d}f}{\mathrm{d}x}(x) = 0
```
#### Then max/min test
* ##### maximum: ``f''(x) <0``

* ##### minimum: ``f''(x) > 0``

""", md"
![](https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/05-example-monotonicity-derivatives.png)
")

# ╔═╡ 7f8094e3-64c3-4b41-87e4-9bf676595339
md"[Figure source](https://tivadardanka.com/book)"

# ╔═╡ b5d8084c-9cb3-4ff6-b61a-176b454a695e
let
	gr()
	x̂ = x̂2
	# x_max = 
    # Plot function
    xs = range(-2, 3, 200)
    ymin, ymax = extrema(f.(xs))
    p = plot(
        xs,
        f;
        label=L"$f(x)$",
        xlabel=L"x",
        legend=:outerbottom,
        ylims = (ymin - .5, ymax + .5),
        legendfontsize=15,
		lw = 3,
		ratio = .7,
		framestyle=:zerolines,
		size=(800,600)
    )

	if show_x02
		scatter!([x̂],[f(x̂)], label = "", mc=:white, msc=:gray, msw=2, alpha=2.0, series_annotations=(text(L"x_0", :top, 25)))
		# annotate!()
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
	if add_linear2
    	plot!(p, xs, (x) -> taylor_approx(x; x̂=x̂); label=L"linear approx. at $x_0$", lc=2,  lw=2, title="Linear approximation")
	end
	if add_quadratic2
		plot!(p, xs, (x) -> taylor_approx(x; x̂=x̂, order=2); label=L"quadratic approx. at $x_0$", lc=3,  lw=3, title="Quadratic approximation")

	end

	p

	# xq = x̂ + 0.9
	# annotate!([xq], [taylor_approx(xq; x̂=x̂, order=2)], text("Quadratic approx"))

end

# ╔═╡ 1f94cba0-6bbb-480b-a62b-81bbbe0ed3ed
md"""
## What if ``f''(x) =0``?
"""

# ╔═╡ 90a78fd7-c6d6-49b9-b8fb-d056de180b9b
TwoColumn(
md"""
\
\


#### *_Saddle point_*:  an exception

* derivative vanishes
* but **neither** maximum **or** minimum
* ``a=0``: degenerative quadratic approximation
"""
,let
	bias = 10
	plot(-5:0.1:5, x-> x^3 +bias,size=(250,350), lw=2, label="", framestyle=:origin, title="Saddle point", titlefontsize=12)
	scatter!([0], [bias], label="", c=:black)

	plot!(-3:0.5:3, x -> 10, c=:black, lw=1.5, label="")
end
)

# ╔═╡ 23133bc6-dc27-4d09-85c1-0e0ad188a038
# md"""

# ## Qudratic approximation


# > If ``f: \mathbb R \rightarrow \mathbb R`` is differentiable at ``x_0``, then
# > 
# > ``f(x)`` can be locally approximated by a quadratic function
# > ```math
# > \Large
# > \begin{align}
# > f(x) &\approx f(x_0) + f'(x_0)(x-x_0) + \frac{1}{2}\underbrace{\boxed{f^{''}(x_0)}}_{\text{\small second order derivative}}(x-x_0)^2
# > \end{align}
# > ```


# """

# ╔═╡ fad92d3e-04ea-40b1-ae7e-35c5b3c37be1
# md"Add approx. $(@bind add_quadratic CheckBox()); Move me ``x_0``: $(@bind x̂_ Slider(-2:0.2:3, default=-1.5, show_value=true))"

# ╔═╡ 5794b474-5541-4625-b433-f30b9c6a857a
# plt_linear_approx = begin
#     # Plot function
#     xs = range(-2, 3, 200)
#     ymin, ymax = extrema(f.(xs))
#     p = plot(
#         xs,
#         f;
#         label=L"$f(x)$",
#         xlabel=L"x",
#         legend=:topleft,
#         ylims = (ymin - .5, ymax + .5),
#         legendfontsize=10,
# 		lw = 2,
# 		ratio = .7,
# 		framestyle=:zerolines
#     )

#     # Obtain the function 𝒟fₓ̃ᵀ
#     ŷ, 𝒟fₓ̂ᵀ = Zygote.pullback(f, x̂)

#     # Plot Dfₓ̃(x)
#     # plot!(p, xs, w -> 𝒟fₓ̂ᵀ(w)[1]; label=L"Derivative $\mathcal{D}f_\tilde{x}(x)$")
#     # Show point of linearization
#     vline!(p, [x̂]; style=:dash, c=:gray, label=L"x_0")
#     # Plot 1st order Taylor series approximation
#     taylor_approx(x) = f(x̂) + 𝒟fₓ̂ᵀ(x - x̂)[1] # f(x) ≈ f(x̃) + 𝒟f(x̃)(x-x̃)
#     plot!(p, xs, taylor_approx; label=L"Linear approx. at $x_0$", lc=2,  lw=2)
# end;

# ╔═╡ 97193fb0-1b92-4c45-be35-e5a61e7f58ed
md"""
# Multivariate calculus

## Essense of (multi-var) calculus

#### for _multi-variate_ differential function 

```math
\Large f: \mathbb{R}^n \rightarrow \mathbb{R}
```

* ###### difficult to study it globally!
"""

# ╔═╡ a0d30c40-b64a-4953-8e1a-3315c0cf16d0
md"""

## Essense of (multi-var) calculus

### But we _can_

* #### _approximate ``f(\mathbf{x})``_ locally  
  * ###### line => (hyper-)planes
  * ###### quadratic parabola => (hyper-)paraboloid
"""

# ╔═╡ fa17d5c4-26e0-4fe6-aea8-614d7e0dc099
md"Add ``\mathbf{x}_0``: $(@bind add_x0_ CheckBox(default=false)), Add linear: $(@bind add_linear_app CheckBox(default=false)), Add quadratic : $(@bind add_quadratic_app CheckBox(default=false))"

# ╔═╡ 791534b7-f284-495f-9710-32719656b661
md"""Move me ``x_1``: $(@bind x01_ Slider(-1.8:0.1:2.8; default= 0)), ``x_2``: $(@bind x02_ Slider(-1.8:0.1:2.8; default= 0))"""

# ╔═╡ afa0f2a8-0fbf-4783-b788-f481f6eaa691
md"""

## _Therefore_, we need to _first_
"""

# ╔═╡ a6314fbe-2491-4d8f-afd7-5b5669df94cb
md"""


##### Before we tackle multi-variate calculus, we need some tools


* ##### multi-variate linear 

```math
\large
f(\mathbf{x}) = \mathbf{b}^\top\mathbf{x} + c
```

* ##### and multivariate quadratic functions (paraboloid)

```math
\large
f(\mathbf{x}) = \mathbf{x}^\top\mathbf{A}\mathbf{x} + \mathbf{b}^\top\mathbf{x} + c
```
"""

# ╔═╡ b0e577be-fc2d-40a4-81a5-ae6530d93939
md"""
## Linear function: ``\mathbb R^n \rightarrow \mathbb R``


```math
\Large 
\begin{align}
f(\mathbf{x}) &=   c + b_1 x_1 + b_2 x_2 + \ldots  b_n x_n\\
	&= c + \mathbf{b}^\top \mathbf{x} 
\end{align}
```

* where ``\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top`` 
* ``\mathbf{b} = [w_1, w_2, \ldots, w_n]^\top``


"""

# ╔═╡ 87a63a6b-5fcc-4d79-982b-90e31fcd4887
aside(tip(md"
Recall ``\mathbf{b}^\top \mathbf{x}  = b_1 x_1 + b_2 x_2 + \ldots  b_n x_n``
"))

# ╔═╡ 8e91558a-0d80-4fcf-940d-47fdc90812ec
md"""
##

```math
\Large 
\begin{align}f(x) = c + b\cdot x\; \xRightarrow{\color{blue}\text{generalisation} } 
\; f(\mathbf{x})
	= c+ \mathbf{b}^\top \mathbf{x}
\end{align}
```
* ##### _direct generalisation_ of the linear function

* ##### _line_ to _(hyper-)plane_
"""

# ╔═╡ 193a34da-e52c-4625-83c9-7e942496023d
md"""




## Examples

##### The most boring function: when ``c=1, \mathbf{b} =\mathbf{0}``


```math
\large
\begin{array}{c}
f(\mathbf{x})
	= \underbrace{1}_{\normalsize=c}+ {\underbrace{\begin{bmatrix} 0 & 0\end{bmatrix}}_{\normalsize=\mathbf{b}}}^\top  {\begin{bmatrix}x_1 \\ x_2 \end{bmatrix}}_{\normalsize = \mathbf{x}}  \\
\Downarrow \\
\LARGE f(\mathbf{x}) = 1  
\end{array}
```

* generalisation of horizontal line ``f(x)=1``

"""

# ╔═╡ bc3a7134-ee9a-4f98-97e2-f3141addb37d
TwoColumn(let
	gr()
	plot(-10:1:10, x -> 1, xlabel=L"x_1", ylabel=L"f",lw=2, c=:blue, alpha=0.8, framestyle=:zerolines, size=(280,300), ylim =[-0.0, 2], label="",  title=L"f(x)=1")
end,
	let
	gr()
	plot(-10:1:10, -10:1:10, (x1, x2) -> 1, st=:surface, zlim = [-0.0, 2], xlabel=L"x_1", ylabel=L"x_2", zlabel=L"f", alpha=0.8, c=:coolwarm, colorbar=false, framestyle=:zerolines,  display_option=Plots.GR.OPTION_Z_SHADED_MESH, size=(370,350), title=L"f(\mathbf{x})=1"
	)
end

)

# ╔═╡ dddfbde0-7710-41de-a829-5a85cbd57544
md"""

##

#### A less boring function: (_hyper-_)plane

```math
\Large 
\begin{align}
f(\mathbf{x}) &= 10 + \begin{bmatrix}1& 0\end{bmatrix} \begin{bmatrix}x_1\\ x_2\end{bmatrix}\\
&= 10 + x_1 
\end{align}
```


* *i.e.* ``\mathbf{b} = [1,0]^\top``

* ##### verying ``x_2`` has no impact on ``f`` (as ``b_2=0``)
"""

# ╔═╡ eed66f8f-361d-4a18-8c30-c407202c95ea
b = [1, 0]

# ╔═╡ 33c93e1c-ba0d-4082-8c8f-fe2a9826334c
md"Add ``x_2=c`` vertical plane: $(@bind add_x2plane CheckBox(default=false)), add more planes: $(@bind add_more_x2_planes CheckBox(default=false))"

# ╔═╡ 6bea0190-b914-4397-844a-90a8699a6123
md"Add ``x_1=c`` vertical plane: $(@bind add_x1plane CheckBox(default=false)), add more planes: $(@bind add_more_x1_planes CheckBox(default=false))"

# ╔═╡ 32184644-ec26-4fbb-b9c7-c4f436338840
let
	plotly()
	# b = b
	w₀ = 10
	f(x1, x2) = dot(b, [x1, x2]) + w₀
	plt = plot(-15:2:15, -15:2:15, (x1, x2) -> dot(b, [x1, x2])+w₀, st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.8, framestyle=:zerolines, c=:coolwarm, colorbar=false)

	min_z, max_z = extrema([f(x1, x2) for x1 in -15:2:15 for x2 in -15:2:15])
	x = range( -15,stop=15,length=10)
	y = range( -15,stop= 15,length=10)
	
	x0 = [0, 0.0]
	eps_x = range(x0[1] -1*eps() ,stop=x0[1]  +eps(),length=10) #create a very small range
	eps_y = range(x0[2]-1*eps(),stop=x0[2] +eps(),length=10)
	X = repeat(range(min_z, max_z, length=10),1,length(y))
	Y = repeat(range(min_z, max_z, length=10)',length(x),1)
	if add_x2plane
		surface!(x, eps_y, X, surftype=(surface = true, mesh = true), c=2, alpha=0.35)  
		xs = -15:1:15
		ys = x0[2] * ones(length(xs))
		zs = [f(xs[i], ys[i]) for i in 1:length(ys)]
		path3d!(xs, ys, zs, lw=5, label="", c=2)

		if add_more_x2_planes
			for x2_ in range(-15, 15, 7)
				eps_y_ = range(x2_-1*eps(),stop=x2_ +eps(),length=10)
				surface!(x, eps_y_, X, surftype=(surface = true, mesh = true), c=2, alpha=0.35)  
				ys = x2_ * ones(length(xs))
				zs = [f(xs[i], ys[i]) for i in 1:length(ys)]
				path3d!(xs, ys, zs, lw=5, label="", c=2)
			end
		end
	end

	if add_x1plane
		surface!(eps_x, y, Y, surftype=(surface = true, mesh = true), c=3, alpha=0.35)  
		ys = -15:1:15
		xs = x0[1] * ones(length(ys))
		zs = [f(xs[i], ys[i]) for i in 1:length(ys)]
		path3d!(xs, ys, zs, lw=5, label="", c=3)

		if add_more_x1_planes
			for x1_ in range(-15, 15, 7)
				eps_x_ = range(x1_-1*eps(),stop=x1_ +eps(),length=10)
				surface!(eps_x_, y, Y, surftype=(surface = true, mesh = true), c=3, alpha=0.35) 
				xs = x1_ * ones(length(ys))
				zs = [f(xs[i], ys[i]) for i in 1:length(ys)]
				path3d!(xs, ys, zs, lw=5, label="", c=3)
			end
		end

	end
	# plot!(-15:2:15, -10:2:10, (x1, x2) -> 10.0, st=:surface, alpha=0.8, title="Level set")
	# x

	plt
end

# ╔═╡ 629b6fbf-e1af-469b-9543-71642562b3a3
md"""

## Multi-var. _quadratic_ function: ``\mathbb{R}^n \rightarrow \mathbb{R}``

#### Recall single-variate quadratic ``f``

```math
\Large
f(x) = x^2
```
"""

# ╔═╡ 871229d9-28f8-4245-99e9-5dd11e186a83
md"""


* ###### it returns the _squared distance_ between ``x`` and ``0``

```math
\large
x^2 = |x-0|^2
```

* ``|x-0|``: abs. difference/error between ``x`` and ``0``

"""

# ╔═╡ f2ece3eb-b582-44b6-9bf7-84d3b7bd806a
@bind x0 Slider(-5:0.5:5; default=0, show_value=true)

# ╔═╡ 084eb288-a774-4369-882d-bd86a7c6f572
let
	gr()
	# a, b, c = qa_, qb_, qc_
	# x₀ = (a ==0) ? 0 : -b/(2*a)
	# plt = plot((x₀-1.5*abs(2a)):0.1:(x₀+1.5*abs(2a)), (x) -> a*x^2+b*x + c, framestyle=:origin, label=L"f(x) = %$(a)x^2 + %$(b)x + %$(c)", legend=:outerright, lw=2)

	# # abcs = [(-2, 0, -2), (-2, 3, 1)]
	# for (a, b, c) in abcs
	# 	plot!(-4:0.1:4, (x) -> a*x^2+b*x+c, framestyle=:origin, lw =2, label=L"f(x) = %$(a)x^2 + %$(b)x + %$(c)", legend=:outerright)
	# end


	plot((x)->x^2, label=L"f(x) = x^2", lw=2, framestyle=:origin, title=L"f(x)=x^2", size=(650,300), legendfontsize=12, legend=:outerright)
	x_ = x0
	plot!([x_, x_], [0, x_^2], ls=:dash, lc=:gray, lw=2, label="")

	annotate!([x_], [x_^2/2], L"x^2= %$(x_^2)", :right)
	# plt
end

# ╔═╡ 8b7fc5db-7e7c-42fb-8a64-2d4530cb53c3
md"""

## Multi-var. _quadratic_ function: ``\mathbb{R}^n \rightarrow \mathbb{R}``

##### Its two variate counter part with ``\mathbf{x} =[x_1, x_2]^\top``:

```math
\Large
f(x) = x^2 \textcolor{blue}{\xRightarrow{\rm generalisation}} f(\mathbf{x}) = x_1^2 + x_2^2 
```
"""

# ╔═╡ a0b18f94-39f0-4f84-86e5-d772cd2206e1
md"""

## Multi-var. _quadratic_ function: ``\mathbb{R}^n \rightarrow \mathbb{R}``

#### Its multi-variate counter part for ``\mathbf{x}\in \mathbb{R}^n``

```math
\Large
f(x) = x^2 \textcolor{blue}{\xRightarrow{\rm generalisation}} f(\mathbf{x})= \mathbf{x}^\top\mathbf{x}
```
"""

# ╔═╡ 81725e2b-1e8d-4af5-80dd-4ba017573c5d
Foldable("", md"""

Remember 

```math
\Large
f(\mathbf{x}) = x_1^2 + x_2^2 = \begin{bmatrix} x_1 & x_2\end{bmatrix} \begin{bmatrix}x_1\\ x_2 \end{bmatrix}
```

""")

# ╔═╡ 7c893c0d-3a9a-45b6-9e08-0224892bf385
md"""

* ##### still the _squared distance_ between ``\mathbf{x}`` and ``\mathbf{0}`` (or squared length of ``\mathbf{x}``)

```math
\Large
\mathbf{x}^\top \mathbf{x} = (\mathbf{x} -\mathbf{0})^\top (\mathbf{x} -\mathbf{0})
```


"""

# ╔═╡ 538b7908-3d95-461c-a24d-b3b05a914ddb
md"move me: $(@bind x0_ Slider(-6:0.1:6, default=0.8))"

# ╔═╡ 9c08d9dd-769b-4c39-a59b-db6924ea1c11
v0 = [1, 1];

# ╔═╡ 1fd7978a-77f9-47ca-be6b-d2a05ec82f64
let
	gr()
	x0 = [0, 0]
	A = Matrix(I,2,2)
	μ = [0,0]
	x_ = x0_ * v0 / (norm(v0))
	f(x1, x2) = dot([x1, x2]- μ, A, [x1, x2]-μ)
	plt = plot(μ[1]-5:0.5:μ[1]+5, μ[2]-5:0.5:μ[2]+5, f, st=:surface, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"f",  alpha=0.5, color=:jet, framestyle=:zerolines, ratio=1, colorbar=false, xlim=[-6, 6], ylim = [-6, 6], title="Qudratic function " * L"\mathbf{x}^\top\mathbf{x}=%$(round(f(x_...);digits=4))")	
	ys = -5:.5:5
	xs = x0[1] * ones(length(ys))
	zs = [dot([xs[i], ys[i]]- μ, A, [xs[i], ys[i]]-μ) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]]- μ, A, [ys[i], xs[i]]-μ) for i in 1:length(ys)]	
	plot!([x_[1], x_[1]], [x_[2], x_[2]], [f(x_...), 0], lw=2, lc=:black, ls=:dash,label="")
	scatter!([x_[1]], [x_[2]], [0], ms=3, markershape=:circle, label=L"x",  mc=:white, msc=:gray, msw=2, alpha=1.0)
	scatter!([x_[1]], [x_[2]], [f(x_...)], ms=3, alpha=0.5, markershape=:circle, label="")
	
end

# ╔═╡ e4f5077c-4ed8-4305-b12b-0a7639844fd6
md"""

##

### Question: _how to sketch_ ``f(\mathbf{x}) = x_1^2 +x_2^2 =\mathbf{x}^\top\mathbf{x}`` ?


* ##### _where_ $\mathbf{x} =\begin{bmatrix}x_1 \\ x_2 \end{bmatrix}$


"""

# ╔═╡ 3250ecdf-b115-43cb-8602-f5814aec79c1
md"Add ``x_1=0`` plane: $(@bind add_x1_plane CheckBox(default=false)), Add ``x_2=0`` plane: $(@bind add_x2_plane CheckBox(default=false)), Add ``f=c`` plane: $(@bind add_z_plane CheckBox(default=false))"

# ╔═╡ 6adcf3c5-a30b-406a-a3f2-4d1c8e8e0fa2
md"Move ``f=c`` height: $(@bind z_height Slider(0:1:45, default=30))"

# ╔═╡ e5ab945d-8d15-4979-bd5b-5aef9c36c85c
let
	x0 = [0,0.]
	plotly()
	A = Matrix(I,2,2)
	μ = [0,0]
	f(x1, x2) = dot([x1, x2]- μ, A, [x1, x2]-μ) + .5
	plt = plot(μ[1]-5:0.8:μ[1]+5, μ[2]-5:0.8:μ[2]+5, f, st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.5, color=:jet, framestyle=:zerolines, ratio=1, colorbar=false, xlim=[-6,6], ylim=[-6, 6], zlim =[-1, 50])
	res = 20
	x = range( -6,stop=40,length=10)
	y = range( -6,stop= 40,length=10)
	eps_x = range(x0[1] -1*eps() ,stop=x0[1]  +eps(),length=10) #create a very small range
	eps_y = range(x0[2]-1*eps(),stop=x0[2] +eps(),length=10)
	X = repeat(x, 1, length(y))
	Y = repeat(y',length(x),1)
	if add_x2_plane
		surface!(x, eps_y, X, surftype=(surface = true, mesh = true), c=2, alpha=0.35)  #create "verticle"
		xs = -5:0.5:5
		ys = x0[2] * ones(length(xs))
		zs = [f(xs[i], ys[i]) for i in 1:length(ys)]
		path3d!(xs, ys, zs, lw=3, label="", c=2)
	end
	if add_x1_plane
		# arrow3d!([x0[1]], [x0[2]], [0], [0], [3], [0]; as = 0.4, lc=1, la=1, lw=2, scale=:identity)
		ys = -5:0.5:5
		xs = x0[1] * ones(length(ys))
		zs = [f(xs[i], ys[i]) for i in 1:length(ys)]
		path3d!(xs,ys,zs, lw=3, label="", c=1)
		surface!(eps_x,y,Y, surftype=(surface = true, mesh = true), c=1, alpha=0.35) #create "verticle" 
	end


	if add_z_plane
		surface!(-5:0.5:5,-5:.5:5, (x, y) -> z_height, surftype=(surface = true, mesh = true), c=3, alpha=0.55) 
		r = sqrt(z_height)
		xs = r * sin.(range(-π, π, 30))
		ys = r * cos.(range(-π, π, 30))
		zs = [z_height for i in 1:length(ys)]
		path3d!(xs, ys, zs, lw=3, label="", c=1)
	end
	plt
end

# ╔═╡ 38caa279-0e78-4f64-9dfb-782f229d9e94
md"""

## Multi-var quadratic function: ``\mathbb{R}^n \rightarrow \mathbb{R}``



```math
\Large
f(x) = a x^2 + bx + c \textcolor{blue}{\xRightarrow{\rm generalisation}} f(\mathbf{x})= \mathbf{x}^\top\mathbf{A}\mathbf{x} + \mathbf{b}^\top\mathbf{x} + c
```


\

* #### ``\mathbf{x}^\top \mathbf{A}\mathbf{x}``: is called _quadratic form_

"""

# ╔═╡ 38c0f8fb-ce5d-4e78-9cd3-50a0f317ed12
md"""

## Example

```math
\Large
\begin{align}
&\;\;\;\;\begin{bmatrix}x_1 & x_2\end{bmatrix}_{1\times 2}\begin{bmatrix}A_{11} & A_{12} \\ A_{21} & A_{22}\end{bmatrix}_{2\times 2}\begin{bmatrix}x_1 \\ x_2 \end{bmatrix}_{2\times 1}
\end{align}
```


* ##### the result is a scalar!

* ##### the result is ``\sum_i\sum_j A_{ij} x_i x_j``

"""

# ╔═╡ 1ef66d06-5981-4e94-9e1e-6472f1e35aaf
Foldable("Details", md"""


```math
\Large
\begin{align}
&\;\;\;\;\begin{bmatrix}x_1 & x_2\end{bmatrix}\begin{bmatrix}A_{11} & A_{12} \\ A_{21} & A_{22}\end{bmatrix}\begin{bmatrix}x_1 \\ x_2 \end{bmatrix}\\
&= \begin{bmatrix}x_1 & x_2\end{bmatrix}\begin{bmatrix}A_{11}x_1 + A_{12}x_2 \\ A_{21}x_1 + A_{22}x_2 \end{bmatrix}\\
&= x_1 (A_{11}x_1 + A_{12}x_2) + x_2 (A_{21}x_1 + A_{22}x_2)\\
&= A_{11}x_1 x_1 + A_{12}x_1x_2+ A_{21}x_2 x_1 + A_{22}x_2 x_2
\end{align}
```

If ``\mathbf{A}`` is symmetric, ``A_{21} = A_{12}``, the result is 


$A_{11}x_1^2 + 2A_{12}x_1x_2 + A_{22}x_2^2$
""")

# ╔═╡ e5fe477d-2694-4803-a101-b2fd9cad9726
md"""
## Univariate max/min test



##### The quadratic coefficient ``\Huge a`` determines: 

* #### maximum or minimum
"""

# ╔═╡ 9cb7368a-0cfa-46cd-8e76-fba25abe4ed0
TwoColumn(md"

#### when `` a > 0``


The function has a **minimum**

$(pltapos)
", 
	
	
md" #### when `` a<0``


The function has a **maximum**


$(pltaneg)
")

# ╔═╡ 77188a12-c136-483b-841a-280f53034d7b
md"""

## Multivariate max/min test -- minimum 

#### _Positive definite_: ``\mathbf{A}``

```math
\Large
f(\mathbf{x})= \mathbf{x}^\top\mathbf{A}\mathbf{x} 
```

* ##### when ``\mathbf{A}`` is _positive definite_, *i.e.*
$\text{\large positive definite: }\;\;\Large \mathbf{x}^\top\mathbf{A}\mathbf{x} > 0\; \text{for all } \mathbf{x\in \mathbb{R}}^n$



"""

# ╔═╡ 7f93a9b2-4485-4ae1-931c-46ae6c994eff
Foldable("Cross reference univariate result", md"""



* ##### _cross reference_ the single variate case

$\large \text{positive } a>0,\; \text{then } \underbrace{a\cdot x^2}_{xax} > 0\; \text{for all } {x\in \mathbb{R}}$

""")

# ╔═╡ 82034449-2245-4387-99d1-15e49623b0ad
md"""

##

> #### *Interpretation*:  _for all directions ``\mathbf{x}\in \mathbb{R}^n``, ``f`` *faces UP*_
> $\text{\large positive definite: }\;\;\Large \mathbf{x}^\top\mathbf{A}\mathbf{x} > 0\; \text{for all } \mathbf{x\in \mathbb{R}}^n$
> * ####  then ``f`` has a *minimum*
> * *e.g.*
> ```math 
> \large
> \mathbf{A} = \begin{bmatrix}1 & 0 \\0 & 1\end{bmatrix}
> ```


"""

# ╔═╡ 3bb46ea9-94ee-4be2-b0ec-bfce14d521a2
md"""

## Max/min test: maximum

#### _Negative definite_ ``\mathbf{A}``

```math
\Large
f(\mathbf{x})= \mathbf{x}^\top\mathbf{A}\mathbf{x} 
```

* ##### when ``\mathbf{A}`` is _*negative definite*_, *i.e.* 

$\text{negative definite: }\large \mathbf{x}^\top\mathbf{A}\mathbf{x} < 0\; \text{for all } \mathbf{x}\in \mathbb{R}^n$

"""

# ╔═╡ fc733c15-01bf-463c-a3e7-1bcd20c1a5f1
Foldable("Cross reference univariate result", md"""


* ##### cross reference for singular variate case 

$\large\text{negative}\; a<0, \;\text{then }\underbrace{a\; x^2}_{xax} < 0\; \text{for all } {x\in \mathbb{R}}$
""")

# ╔═╡ 9fab28fa-d158-4cd6-b706-817618be4e4c
md"""
##

> #### *Interpretation*:  _for all directions ``\mathbf{x}\in \mathbb{R}^n``, ``f`` *faces DOWN*_
> $\text{negative definite: }\large \mathbf{x}^\top\mathbf{A}\mathbf{x} < 0\; \text{for all } \mathbf{x}\in \mathbb{R}^n$
> * ##### then ``f`` has a *maximum*
> * *e.g.*
> ```math
> \mathbf{A} = \begin{bmatrix}-1 & 0 \\0 & -1\end{bmatrix}
> ```


"""

# ╔═╡ dfa4fc88-6cfc-4d87-ba7c-87bb2e91bae0
md"""

## Max/min test : indefinite 
```math
\Large
f(\mathbf{x})= \mathbf{x}^\top\mathbf{A}\mathbf{x} 
```

* ##### when ``\mathbf{A}`` is _*indefinite*_, *i.e.* 

$\large \mathbf{x}^\top\mathbf{A}\mathbf{x} < 0;\; \text{ for some }\mathbf{x}$

$\large\mathbf{x}^\top\mathbf{A}\mathbf{x} > 0;\; \text{ for some }\mathbf{x}$

* e.g. when
```math
\mathbf{A} = \begin{bmatrix}1 & 0 \\0 & -1\end{bmatrix}
```

"""

# ╔═╡ ccf4b05b-c5ca-41f2-80d3-7482d465467c
Foldable("Cross reference univariate case", md"""

when ``a=0``, ``a x^2 + bx + c`` reduces to  line ``bx+c``; if the derivative (``b=0``) is zero, then the second order approximation is a horizontal line.
""")

# ╔═╡ 0fba1c83-7f68-4f66-8f28-92c382f16ac9
md"""

#### Interpretation


* ##### neither _maximum_ nor _minimum_, a _saddle_ surface

"""

# ╔═╡ a80e21da-90f4-40d2-8ca3-eda504180295
md"""
## Partial derivative

#### The *partial derivative* w.r.t. $x_i$ is

$$\large \frac{\partial f}{\partial \textcolor{red}{x_i}}(\mathbf{x}) = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, \textcolor{red}{x_i+h}, \ldots, x_n) - f(x_1, \ldots, \textcolor{red}{x_i}, \ldots, x_n)}{h}$$

* ##### _change_ one dimension (``i``- th dimension) *while keeing* all ``x_{j\neq i}`` constant


"""

# ╔═╡ 76239279-b34f-441d-9a21-184a24345637
md"""

## Partial derivative

#### The *partial derivative* w.r.t. $x_i$ is

$$
\large \frac{\partial f}{\partial \textcolor{red}{x_i}}(\mathbf{x}) = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, \textcolor{red}{x_i+h}, \ldots, x_n) - f(x_1, \ldots, \textcolor{red}{x_i}, \ldots, x_n)}{h}$$

* ##### _change_ one dimension (``i``- th dimension) *while keeing* all ``x_{j\neq i}`` constant



##### Note that this is the same as
```math
\large
\frac{\partial f}{\partial \textcolor{red}{x_i}}(\mathbf{x}) = \lim_{h \rightarrow 0} \frac{f(\mathbf{x}+ h \cdot \textcolor{red}{\mathbf{e}_i}) - f(\mathbf{x})}{h}

```

* ``\mathbf{e}_i`` is the i-th standard basis vector, as an example, for ``i=1``
```math
\mathbf{x} + h\cdot  \mathbf{e}_1 =\begin{bmatrix}x_1  \\ x_2 \\\vdots \\ x_n \end{bmatrix} + h \begin{bmatrix}1  \\ 0 \\\vdots \\ 0 \end{bmatrix}= \begin{bmatrix}x_1 + h \\ x_2 \\\vdots \\ x_n \end{bmatrix}
```



- change rate (slope) along the ``i``-th standard basis (``\textcolor{red}{\mathbf{e}_i}``) direction 


"""

# ╔═╡ 5be1eead-5c0a-477e-9227-31dd0e7a000f
md"Add ``\mathbf{e}_1``: $(@bind add_e1 CheckBox(default=false)), add ``\mathbf{e}_2``: $(@bind add_theother CheckBox(default=false))"

# ╔═╡ 5c06f638-f429-4f53-a971-f0adc68ef6a1
xx = [2,2]

# ╔═╡ 95247ac3-7058-4042-8c84-f52b19770313
md"""

## More example/exercise


```math
\Large 
f(\mathbf{x}) = c + b_1 x_1 + b_2 x_2 = c + \mathbf{b}^\top\mathbf{x}
```


* ##### ``\large \frac{\partial f}{\partial x_1}=?``
\

* ##### ``\large \frac{\partial f}{\partial x_2}=?``
"""

# ╔═╡ a4a4b442-a0fb-4193-b82e-38cfe463d38e
md"""

## More example/exercise


```math
\Large 
f(\mathbf{x}) = c + b_1 x_1 + b_2 x_2 = w_0 + \mathbf{b}^\top\mathbf{x}
```


* ##### ``\large \frac{\partial f}{\partial x_1}= b_1``
\

* ##### ``\large \frac{\partial f}{\partial x_2}=?``
"""

# ╔═╡ a519be30-5065-4de3-a4d7-d0eb6261b549
md"""

## More example/exercise


```math
\Large 
f(\mathbf{x}) = c + b_1 x_1 + b_2 x_2 = w_0 + \mathbf{b}^\top\mathbf{x}
```


* ##### ``\large \frac{\partial f}{\partial x_1}= b_1``
\

* ##### ``\large \frac{\partial f}{\partial x_2}=b_2``
"""

# ╔═╡ f45b1035-7d34-4069-bca5-600a4ba43a7e
md"""

## More example/exercise


```math
\Large 
f(\mathbf{x}) = x_1^2\cdot  x_2
```


* ##### ``\large \frac{\partial f}{\partial x_1}=?``
\

* ##### ``\large \frac{\partial f}{\partial x_2}=?``
"""

# ╔═╡ c5874a8c-b37a-45a9-8bf0-d32f39085374
md"""

## More example/exercise


```math
\Large 
f(\mathbf{x}) = x_1^2\cdot  x_2
```


* ##### ``\large \frac{\partial f}{\partial x_1}=2 x_2 x_1``
\

* ##### ``\large \frac{\partial f}{\partial x_2}=?``
"""

# ╔═╡ affa56cf-e919-48f6-a39f-ee534f966a7f
md"""

## More example/exercise


```math
\Large 
f(\mathbf{x}) = x_1^2\cdot  x_2
```


* ##### ``\large \frac{\partial f}{\partial x_1}=2 x_2 x_1``
\

* ##### ``\large \frac{\partial f}{\partial x_2}=x_1^2``
"""

# ╔═╡ 3c39ca05-4856-48c6-af66-ad0f99c2ef19
md"""

## Gradient -- ``\nabla f(\mathbf{x})``
"""

# ╔═╡ b86f4f23-ee0c-42e9-97f5-ee343a391d1e
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/grrad.svg" width = "650"/></center>"""

# ╔═╡ e161ad2e-8df1-4757-badd-ec83bc1de986
md"""
## First key fact about _gradient_



"""

# ╔═╡ 8d5d9759-2d18-4eea-9ddc-f4ae357363ea
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/gradient.svg" width = "400"/></center>"""

# ╔═╡ b8dcfce1-881b-4f0f-8e35-c8831b8f8c52

md"""
- #### *gradient* itself is a function: _vector_ to _vector_ function
  - ##### *input*:  a vector ``\mathbf{x} \in \mathbb{R}^n`` (interpreted as an *location*) 
  - ##### *output*: a vector ``\nabla f(\mathbf{x}) \in \mathbb{R}^n`` (interpreted as a *direction*)
"""

# ╔═╡ 44a399ba-1ad7-4e4d-8ff3-ae30929e97bb
md"""

## Gradient example: ``f(\mathbf{x}) = \mathbf{b}^\top \mathbf{x} + c``



```math
\Large
f(\mathbf{x}) =\mathbf{b}^\top \mathbf{x} + c = b_1 x_1 + b_2 x_2 + c
```

#### The gradient is 

$$\large

\nabla f(\mathbf{x}) = \begin{bmatrix}\frac{\partial f(\mathbf{x})}{\partial x_1}\\ \frac{\partial f(\mathbf{x})}{\partial x_2}\end{bmatrix} = \begin{bmatrix} b_1\\ b_2\end{bmatrix} = \mathbf{b}$$



* ##### the gradient is *constant* direction ``\mathbf{b}`` 
  * ###### for all locations ``\mathbf{x} \in \mathbb R^2``

"""

# ╔═╡ 92c18f58-aa3e-4844-ab23-f01fd2705fb3
md"""

!!! note "Hint"
	##### Recall univariate linear function and its derivative:
	
	```math
	\Large
	f(x) = bx+c,\;\; f'(x) = b
	```


"""

# ╔═╡ 338e662e-57b2-4bc0-a4ea-dc7bf6a790f4
md"""

## Gradient example: ``f(\mathbf{x}) = \mathbf{x}^\top \mathbf{x}``



```math
\Large
f(\mathbf{x}) = \mathbf{x}^\top \mathbf{x} = x_1^2 + x_2^2
```

#### The gradient is ?




"""

# ╔═╡ 6e79f928-93eb-4a75-9e7b-f3666d6a71e2
md"""

$$\large
\nabla f(\mathbf{x}) = \begin{bmatrix}\frac{\partial f(\mathbf{x})}{\partial x_1}\\ \frac{\partial f(\mathbf{x})}{\partial x_2}\end{bmatrix} = \begin{bmatrix} 2 x_1\\ 2 x_2\end{bmatrix} = 2 \mathbf{x}$$


"""

# ╔═╡ ee135ef9-5be5-4a4e-80fe-d04179215cab
md"""

!!! note "Hint"
	##### Recall univariate linear function and its derivative:
	
	
	```math
	\large
	f(x) = x^2 = x * x,\;\; f'(x) = 2x
	```


"""

# ╔═╡ fd86950f-4478-4b18-8bc3-94cf908eac32
md"""

## How about ``f(\mathbf{x}) = \mathbf{x}^\top \mathbf{Ax} + \mathbf{b}^\top \mathbf{x} + c``


!!! question "Question"

	##### What's the gradient of the quadratic function?

	```math
	\large
	f(\mathbf{x}) = \mathbf{x}^\top \mathbf{Ax} + \mathbf{b}^\top \mathbf{x} + c
	```
	* ###### assume ``\mathbf{A}`` is symmetric


!!! note "Hint"
	##### Recall univariate quadratic function and its derivative:
	
	```math
	\large
	f(x) = ax^2+bx+c,\;\; f'(x) = 2ax + b
	```

#### Just guess?

"""

# ╔═╡ fc377a9a-92b2-4444-b5bf-0402f0b39e88
Foldable("Answer", md"

```math
\large
\nabla f(\mathbf{x}) = (\mathbf{A} +\mathbf{A}^\top)\mathbf{x} + \mathbf{b}
```


For symmetric ``\mathbf{A}``, the result is

```math
\large
\boxed{\nabla f(\mathbf{x}) = 2 \mathbf{A}\mathbf{x} + \mathbf{b}}
```

Check appendix for full justification. 
")

# ╔═╡ faba77c4-afe2-434b-9ce7-3bb1740cbb0f
md"""

## Why _gradient_ matters ? 
##### -- local linear approximation!
\

> ##### If ``f: \mathbb R^n \rightarrow \mathbb R`` is differentiable at ``\mathbf{x}_0``, then
> 
> ##### ``f(\mathbf{x})`` can be approximated by a linear function (locally at ``\mathbf{x}_0``)

"""

# ╔═╡ ce95d775-1234-4a61-86bf-bf82d9051c38
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/linearappxmulti.svg" width = "450"/></center>"""

# ╔═╡ fdab2182-1a2f-497d-b560-5f4b62ea0554
Foldable("Uni-variate result", 


	html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/linearappx.svg" width = "400"/></center>"""

)

# ╔═╡ dcaa1293-b7f9-4f10-b7c3-a6b328cfd890
md"""Move me ``x_1``: $(@bind x01 Slider(-1.8:0.1:2.8; default= 0)), ``x_2``: $(@bind x02 Slider(-1.8:0.1:2.8; default= 0))"""

# ╔═╡ 1ba6cdc1-0612-49f7-b9c7-833877e8b80f
md"""



## Second fact about _gradient_ : important!

#### ``\nabla f(\mathbf{x})``: points to the *greatest ascent direction* 

* #### _locally_ at ``\mathbf{x}``



"""

# ╔═╡ d54951a4-2379-4b85-960d-fdaea320b376
html"""<center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/3d-gradient-cos.svg/2560px-3d-gradient-cos.svg.png" width = "350"/></center>"""

# ╔═╡ 025a340c-21e4-48b3-a76e-4acae68c40ac
md"[source](https://en.wikipedia.org/wiki/Gradient)"

# ╔═╡ 5d182ed5-74b0-4e42-a381-1464c62d876a
md"""

## But why? 

#### We only need to consider linear function case
(because non-linear ``f`` can be approximated well locally by a linear function.) 

```math
\Large
f(\mathbf{x}) = c+ \mathbf{w}^\top \mathbf{x}
```


* #### assume we are at ``\mathbf{x}_0``, which _direction_ to choose?
"""

# ╔═╡ dde45286-689b-458e-a19f-225bd0e64cbf
md"Add directions: $(@bind add_directions CheckBox(false))"

# ╔═╡ e8144237-53e6-4455-b198-03f60caae667
md"""
## But why?

```math
\Large
f(\mathbf{x}) = c+ \mathbf{w}^\top \mathbf{x}
```


"""

# ╔═╡ 73aafe0e-1112-4b6b-a87b-3f60cd2f4d03
md"
#### _First_, the gradient is

```math
\large
\nabla f(\mathbf{x}) = \begin{bmatrix}\frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \end{bmatrix} = \begin{bmatrix}w_1 \\ w_2 \end{bmatrix} =\mathbf{w}
```

"

# ╔═╡ 535ee2f4-64ae-4799-9a4f-fbc6fdd7c9d6
md"""



#### Denote _unit_ directional vector: ``\mathbf{u}``

* if we move away from ``\mathbf{x}_0`` by ``\mathbf{u}``, the new location is 

```math
\mathbf{x}_{new} = \mathbf{x}_0 + \mathbf{u}
```

* then ``f(\mathbf{x}_{new})`` becomes
```math
\large
\begin{align}
f(\mathbf{x}_0 +\mathbf{u}) &= c + \mathbf{w}^\top (\mathbf{x}_0+\mathbf{u})\\
&=\underbrace{c + \mathbf{w}^\top \mathbf{x}_0}_{\text{constant!}} + \mathbf{w}^\top\mathbf{u}
\end{align}
```


* for what direction ``\mathbf{u}``, ``f`` is maximised?
"""

# ╔═╡ eafcde85-cc83-4903-aef0-04e24f34d169
Foldable("", md"""

#### The gradient ``\nabla f(\mathbf{x}) =\mathbf{w}`` and ``\mathbf{u}`` point to the same direction!

```math
\max_{\mathbf{u}} \mathbf{w}^\top\mathbf{u} = \|\mathbf{w}\| \|\mathbf{u}\| \cos\theta = \|\mathbf{w}\| \cos\theta
```

* ``\|\mathbf{u}\| = 1``: unit vector
* when ``\theta = 0``!
""")

# ╔═╡ b0a301fa-5b5f-4dcd-90dc-506ca860d950
md"``\color{red}\mathbf{u}`` (red): $(@bind utheta Slider(range(-π, π, 100), default=0)); Make the move along ``\color{red}\mathbf{u}``: $(@bind add_xnew CheckBox(false))"

# ╔═╡ bee80039-b7f6-4d43-ae23-8bf20629e3d9
md"Add gradient ``\nabla f(\mathbf{x})=\mathbf{w}`` (green): $(@bind add_grad_vec CheckBox(false))"

# ╔═╡ 022193c3-d6bb-4292-82c4-85b94df7775c
md"""

## Gradient visualisation: ``f(\mathbf{x}) = \mathbf{x}^\top \mathbf{x}``


```math
\Large 
f(\mathbf{x}) =\mathbf{x}^\top \mathbf{x} = x_1^2 + x_2^2
```

##### The gradient is 

$$
\large
\nabla f(\mathbf{x}) = \begin{bmatrix}\frac{\partial f(\mathbf{x})}{\partial x_1}\\ \frac{\partial f(\mathbf{x})}{\partial x_2}\end{bmatrix} = \begin{bmatrix} 2 x_1\\ 2x_2\end{bmatrix} = 2 \mathbf{x}$$


* the gradient vanishes when ``\mathbf{x} \rightarrow \mathbf{0}``
* the gradients ``\nabla f(\mathbf{x})`` point outwardly (again, the greatest ascend direction)
"""

# ╔═╡ aecf467e-5bb1-4429-b62a-e34156eb5f83
TwoColumn(let
	x0 = [0, 0]
	plotly()
	A = Matrix(I,2,2)
	μ = [0,0]
	f(x1, x2) = dot([x1, x2]- μ, A, [x1, x2]-μ)
	plot(μ[1]-5:1:μ[1]+5, μ[2]-5:1:μ[2]+5, f, st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.5, color=:jet, framestyle=:zerolines, ratio=1, colorbar=false, xlim=[-6, 6], ylim = [-6, 6], size=(300,300))
	scatter!([x0[1]], [x0[2]], [0], ms=1, markershape=:cross, label="")
	scatter!([x0[1]], [x0[2]], [f(x0...)], ms=1, label="")
	vs = 0:0.1:f(x0...)
	plot!(2 * ones(length(vs)), 2 * ones(length(vs)), vs, lw=3, ls=:dash, lc=:gray, label="")

end, let
	gr()
	A = Matrix(I, 2, 2)
	f(x₁, x₂) = dot([x₁, x₂], A, [x₁, x₂])
	∇f(x₁, x₂) = 2 * A* [x₁, x₂] / 5
	xs = -20:0.5:20
	ys= -20:0.5:20
	cont = contour(xs, ys, (x, y)->f(x,y), c=:jet, xlabel=L"x_1", ylabel=L"x_2", framestyle=:origin, title="Gradient field plot", ratio=1, size=(300,300))
	# for better visualisation
	meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.
	xs_, ys_ = meshgrid(range(-15, 15, length=8), range(-15, 15, length=6))
	quiver!(xs_, ys_, quiver = ∇f, c=:green)
end)

# ╔═╡ a34bedc7-1f89-4535-8fe7-98097e905695
md"""

## Gradient visualisation: more example
"""

# ╔═╡ 1f2cc5d7-5fd7-42d0-963f-24aa0de076eb
md"""


```math
\large
f(w_1, w_2) = \frac{1}{4} (w_1^4 + w_2^4) -\frac{1}{3} (w_1^3 +w_2^3) - w_1^2 -w_2^2 +4
```
"""

# ╔═╡ 474aaa5d-2f26-4e3a-adee-fb5dbac67c15
md"""

## Derive the gradient



```math
\large
f(w_1, w_2) = \frac{1}{4} (w_1^4 + w_2^4) -\frac{1}{3} (w_1^3 +w_2^3) - w_1^2 -w_2^2 +4
```

!!! hint "Question: derive the gradient"
	```math
	\large
		\nabla f(\mathbf{w}) = \begin{bmatrix} \frac{\partial f}{\partial w_1} \\ \frac{\partial f}{\partial w_2}\end{bmatrix} =\begin{bmatrix} w_1^3 - w_1^2 - 2w_1 \\ w_2^3 - w_2^2 - 2w_2\end{bmatrix}
	```
"""

# ╔═╡ 63e09a57-44a6-4a5b-b9ce-b09dbbcf0f46
meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x))) # helper function to create a quiver grid.;

# ╔═╡ df7e37ee-dde4-483c-bf2f-66a219077b35
md"""
## Visualisation of the gradients

##### Four local minimums: ``[2,2], [2,-1],[-1,-1], [-1, 2]``

* #### where *gradients vanish*


"""

# ╔═╡ cfa000a1-f350-442f-8a40-82e4b8ee0fef
begin
	
	∇f_demo(w₁, w₂) = [w₁^3 - w₁^2 - 2 * w₁, w₂^3 - w₂^2 - 2 * w₂]
	∇f_demo(w::Vector{T}) where T <: Real = ∇f_demo(w...)
end;

# ╔═╡ 76ea84a3-3013-4a6e-ae33-e439c5f16d31
begin
	f_demo(w₁, w₂) = 1/4 * (w₁^4 + w₂^4) - 1/3 *(w₁^3 + w₂^3) - w₁^2 - w₂^2 + 4
	f_demo(w::Vector{T}) where T <: Real = f_demo(w...)
end;

# ╔═╡ 766bf632-e1c0-4c5f-8f71-4304ec16a1cb
let
	gr()
	f = f_demo
	# l0 = [x01_, x02_]
	# tf = taylorApprox(f, l0, 1)
	x1_ = range(-2., stop =3.0, length=60)
	x2_ = range(-2., stop =3.0, length=60)
	p1_ = plot(x1_, x2_, (a, b) -> f([a, b]), st=:surface, xlabel =L"x_1", ylabel=L"x_2", zlabel=L"f(\mathbf{x})", colorbar=false, color=:jet)
	# plot!(range(l0[1] - 1, l0[1]+1, 5), range(l0[2] - 1, l0[2]+1, 5), (a, b) -> tf([a, b]), st=:surface, alpha=0.4, display_option=Plots.GR.OPTION_Z_SHADED_MESH)
	# scatter!([l0[1]], [l0[2]], [f(l0)], ms=2, label=L"\mathbf{x}")
	p1_
end

# ╔═╡ dfba25eb-0316-464f-8ff6-b4acc8d86c1e
more_ex_surface = let
	gr()
	plot(-2:0.1:3, -2:0.1:3, f_demo, st=:surface, color=:jet, colorbar=false, aspect_ratio=1.0, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"f(x)", title="A "*L"\mathbb{R}^2\rightarrow \mathbb{R}"*" function", size=(300,300))
end;

# ╔═╡ 38d3f4ec-c8be-4b19-9484-f8748fbd0df7
TwoColumn(plot(more_ex_surface, xlabel=L"w_1", ylabel=L"w_2", title="", size=(350,350)), let
	gr()
	plot(-2:0.05:3, -2:0.05:3, f_demo, st=:contour, color=:jet, colorbar=false, aspect_ratio=1.0, framestyle=:box,xlabel=L"w_1", ylabel=L"w_2", xlim=[-2.1,3.1], size=(350,350))

end)

# ╔═╡ a9177e2e-b2f8-46cc-ac64-504d2ec896a4
TwoColumn(plot(more_ex_surface, xlabel=L"w_1", ylabel=L"w_2", title="", size=(250,250)), let
	gr()
	α = 0.3
	plt = plot(-2:0.05:3, -2:0.05:3, f_demo, st=:contour, color=:jet, alpha=1, colorbar=false, aspect_ratio=1.0, levels=22, xlim=[-2, 3], ylim=[-2, 3], xlabel=L"w_1", ylabel=L"w_2")
	xs_, ys_ = meshgrid(range(-2, 3, length=25), range(-2, 3, length=25))
	∇f_d(x, y) = ∇f_demo(x, y) * α
	quiver!(xs_, ys_, quiver = ∇f_d, c=3)
	xs = [2 2; 2 -1; -1 -1; -1 2]
	for x in eachrow(xs)
		scatter!([x[1]], [x[2]], label="", markershape=:x, c= 1, markerstrokewidth=4, ms=5, xlim=(-2,3), ylim=(-2,3))
	end
	plt
end)

# ╔═╡ ade67f75-7eb7-418b-b5f5-ddbb8f1a6cc3
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

# ╔═╡ 4fae417c-d0a1-4ee0-baab-e0d27e84e8cf
let
	gr()
	f = f_demo
	l0 = [x01_, x02_]
	tf = taylorApprox(f, l0, 1)
	x1_ = range(-2., stop =3.0, length=30)
	x2_ = range(-2., stop =3.0, length=30)
	p1_ = plot(x1_, x2_, (a, b) -> f([a, b]), st=:surface, xlabel =L"x_1", ylabel=L"x_2", zlabel=L"f(\mathbf{x})", colorbar=false, color=:jet, size=(600,550))
	if add_x0_
		scatter!([l0[1]], [l0[2]], [f(l0)],  label=L"\mathbf{x}_0", mc=:white, msc=:gray, msw=2, alpha=2.0)
	end
	if add_linear_app
		plot!(range(l0[1] - 1, l0[1]+1, 4), range(l0[2] - 1, l0[2]+1, 4), (a, b) -> tf([a, b]), st=:surface, alpha=0.4, display_option=Plots.GR.OPTION_Z_SHADED_MESH, title="Linear approximation")
	end

	if add_quadratic_app
		tf2 = taylorApprox(f, l0, 2)
		surface!(p1_, range(l0[1] - 1.2, l0[1]+1.2, 30), range(l0[2] - 1.2, l0[2]+1.2, 30), (x, y) -> tf2([x, y]), label="", alpha=0.5, display_option=Plots.GR.OPTION_SHADED_MESH,zlim =(-1,9), title="Quadratic approximation")
	end

	p1_
end

# ╔═╡ 9441ff29-4df7-41cf-a85d-3219ec9ad0f0
let
	gr()
	f = f_demo
	l0 = [x01, x02]
	tf = taylorApprox(f, l0, 1)
	x1_ = range(-2., stop =3.0, length=60)
	x2_ = range(-2., stop =3.0, length=60)
	p1_ = plot(x1_, x2_, (a, b) -> f([a, b]), st=:surface, xlabel =L"x_1", ylabel=L"x_2", zlabel=L"f(\mathbf{x})", colorbar=false, color=:jet)
	plot!(range(l0[1] - 0.8, l0[1]+0.8, 5), range(l0[2] - 0.8, l0[2]+0.8, 5), (a, b) -> tf([a, b]), st=:surface, alpha=0.4, display_option=Plots.GR.OPTION_Z_SHADED_MESH)
	scatter!([l0[1]], [l0[2]], [f(l0)], ms=5, label=L"\mathbf{x}_0", mc=:white, msc=:gray, msw=2, alpha=2.0)
	p1_
end

# ╔═╡ 7451193a-6a5c-424f-8ffd-14550ce6320c
l0s = [[0, 0], [2, -1], [2, 2], [-1, 2], [-1, -1], [2, 0], [0, -1], [0, 2], [-1, 0]];

# ╔═╡ 479b34fb-d1d1-443a-8ee9-949e35a378c3
md"""
## Multivariate optimisation



#### Often, we want to optimise a ``\mathbb R^n \rightarrow \mathbb R`` function ``f(\mathbf{x})``


```math
\Large
\hat{\mathbf{x}} \leftarrow \arg\min_{\mathbf{x}} f(\mathbf{x})
```


* ##### _firstly_, find the gradient ``\nabla f(\mathbf{x})``

* ##### _secondly_, solve for ``\nabla f(\mathbf{x}) = \mathbf{0}``
```math
\Large
\nabla f(\mathbf{x}) = \mathbf{0}
```

* ###### gradient vanishes ``\Leftrightarrow`` horizontal linear approximation




"""

# ╔═╡ a9573110-2ebf-4fb5-89a4-f36926068fc4
md"""

## Optimisation example -- an easy problem


###### Optimise quadratic function ``f(x) = ax^2+bx+c`` and ``f'(x) = 2ax+b``


* the solution is

```math
\large
f'(x) = 2ax+b =0 \Rightarrow \boxed{x' = -\frac{1}{2}a^{-1}b}
```

	"""

# ╔═╡ 9426b55b-d6fb-4bed-b518-6cf22d78fcb6
md"""

## Optimisation example -- an easy problem



##### Similarly, for multivariate quadratic function with symmetric ``\mathbf{A}``:

```math
\large
f(\mathbf{x}) = \mathbf{x}^\top \mathbf{Ax} + \mathbf{b}^\top \mathbf{x} + c
```

* #### its gradient is

```math
\large
\nabla f(\mathbf{x}) = 2 \mathbf{A}\mathbf{x} + \mathbf{b}
```


* #### the solution of ``\nabla f(\mathbf{x}) = \mathbf{0}`` is


```math
\large
\nabla f(\mathbf{x}) = 2 \mathbf{A}\mathbf{x} + \mathbf{b} =\mathbf{0} \Rightarrow \boxed{\mathbf{x}'= -\frac{1}{2}\mathbf{A}^{-1}\mathbf{b} }
```


"""

# ╔═╡ 3f53c70a-b62a-4266-aca2-ca8570a58207
bv_, cv_= [10, 10], 0.0;

# ╔═╡ ee9565f8-9e54-4f5a-8df7-f4983cfe3c76
begin
	A_ = Matrix(1I, 2, 2) # positive definite
	# A_ = - Matrix(1I, 2, 2) # negative definite
	# A_ = Matrix([1 0; 0 -1]) # neither definite
end;

# ╔═╡ 01049ef2-6364-478f-b756-410681796879
qform(x; A=A_, b=bv_, c=c_) = x' * A * x + b'* x + c # quadratic form function

# ╔═╡ 2f21c8d6-9a96-47d0-9150-5cc937085769
let
	gr()
	plot(-5:0.25:5, -5:0.25:5, (x,y) -> qform([x,y]; A= -Matrix(I,2,2), b=zeros(2), c=0), st=:surface, colorbar=false, color=:coolwarm, title="A is negative definite; maximum", display_option=Plots.GR.OPTION_SHADED_MESH, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"f", framestyle=:semi)
end

# ╔═╡ de792554-a433-45ba-b397-22bd027a54e8
TwoColumn(let 
	plotly()
	A = [1 0; 0 -1]
	plot(-5:0.5:5, -5:0.5:5, (x,y) -> qform([x,y]; A= A, b=zeros(2), c=0), st=:surface, colorbar=false, color=:jet, title="A is not definite; not determined", size=(400,400))
	ys = -5:.5:5
	xs = zeros(length(ys))
	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
	path3d!(xs, ys,zs, lw=8, label="", c=1)
	xs = -5:.5:5
	ys = zeros(length(xs))
	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
	path3d!(xs, ys,zs, lw=8, label="", c=2)
end, html"""<center><img src="https://saddlemania.com/wp-content/uploads/2022/04/parts-of-saddle.jpeg" height = "300"/></center>""" )

# ╔═╡ 7384f349-ec08-41bf-91b6-d80ffdda59b5
xmin_ = -0.5 * A_^(-1) * bv_;

# ╔═╡ 668074ee-fcc3-4e78-82fe-5f1c83e8ebfc
md"""

## Demonstration

"""

# ╔═╡ 6549244a-679c-4e12-9f60-f77c25afecaa
TwoColumn(md"""
``\mathbf{A}=``$(latexify_md(A_)) 

""", md"""

``\mathbf{b}=``$(latexify_md(bv_))
""")

# ╔═╡ 3a059c7c-38b3-4aa4-b9e6-88e4235a4f4b
md"""

``\mathbf{x}_{min} = -\frac{1}{2} \mathbf{A}^{-1}\mathbf{b}=``$(latexify_md(xmin_))

"""

# ╔═╡ 429a7da7-ce5d-4207-af56-cee550112335
let
	plotly()
	xmin_0 = xmin_
	xs = range(xmin_0[1]-8, xmin_0[1]+8, 100)
	plot(xs, range(xmin_0[2]-8, xmin_0[2]+8, 100), (x1, x2) -> qform([x1, x2]; A=A_, b= bv_, c= cv_), st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.8, framestyle=:zerolines, ratio=1, c=:jet, colorbar=false)

	scatter!([xmin_0[1]],[xmin_0[2]], [qform(xmin_0; A=A_, b= bv_, c= cv_)], label="x': min/max/station", c=:black, ms=3.5)

	lim_plane = 6
	plot!(range(xmin_0[1]-lim_plane, xmin_0[1]+lim_plane, 10),  range(xmin_0[2]-lim_plane, xmin_0[2]+lim_plane, 10), (x, y) -> qform(xmin_0; A=A_, b= bv_, c= cv_), st=:surface, c=:gray, alpha=0.8)
	# scatter!([xmin_0[1]],[xmin_0[2]], [0], label="")
end

# ╔═╡ e6825073-f73e-409b-9944-bc0258598003
md"""

## Optimisation -- more challenging case


#### Solve ``\nabla L(\mathbf{x}) = \mathbf{0}``

* ##### _multiple_ solutions!


#### _But_ how to do max/min test?

* ##### need second order approximation!
"""

# ╔═╡ f598a0af-f235-49b9-b85a-3e7a841b70d0
md""" Move me: $(@bind angi Slider(-30:90; default = 45)); $(@bind angi2 Slider(-45:60; default = 30))"""

# ╔═╡ cbf8e293-8d7b-4420-8631-58b04e3725f3
let
	gr()
	f = f_demo
	x1_ = range(-2., stop =3.0, length=60)
	x2_ = range(-2., stop =3.0, length=60)
	p1_ = plot(x1_, x2_, (a, b) -> f([a, b]), st=:surface, xlabel =L"x_1", ylabel=L"x_2", zlabel=L"f(\mathbf{x})", colorbar=false, color=:jet, title="Solve " * L"\nabla L{(\mathbf{x})} = \mathbf{0}", camera=(angi, angi2))
	len = 0.7
	for (li, l0) in enumerate(l0s)
		tf = taylorApprox(f, l0, 1)
		plot!(range(l0[1] - len, l0[1]+len, 5), range(l0[2] - len, l0[2]+len, 5), (a, b) -> tf([a, b]), st=:surface, alpha=0.3,  display_option=Plots.GR.OPTION_Z_SHADED_MESH)
		label = li == 1 ? L"\mathbf{x};\; \texttt{s.t.} \nabla L(\mathbf{x}) = 0" : ""
		scatter!([l0[1]], [l0[2]], [f(l0)], ms=3, label=label)
	end
	p1_
end

# ╔═╡ 7ff064f9-5e99-4fc5-a0e9-60768d2553b2
md"""

## Recall: quadratic approximation 
##### for univariate ``f: \mathbb{R}\rightarrow \mathbb{R}``
"""

# ╔═╡ ceed5776-0fdc-4457-8064-a735e7652761
md"""

> ##### It can be approximated locally by a _quadratic function_
> ```math
> \Large
> \begin{align}
> f(x) &\approx f(x_0) + f'(x_0)(x-x_0) + \frac{1}{2}\underbrace{\boxed{\color{red}f^{''}(x_0)}}_{\text{\small second order derivative}}(x-x_0)^2
> \end{align}
> ```

"""

# ╔═╡ 64188019-bcdd-4a7a-ac15-529223737274
md"Add quadratic approx.: $(@bind add_quadratic2_ CheckBox(false)); Move me ``x_0``: $(@bind x̂2_ Slider(-2:0.005:3, default=-1.355, show_value=true))"

# ╔═╡ 8965aa00-0832-4e76-91d4-e68a51f41518
let
	gr()
	x̂ = x̂2_
    # Plot function
    xs = range(-2, 3, 200)
    ymin, ymax = extrema(f.(xs))
    p = plot(
        xs,
        f;
        label=L"$f(x)$",
        xlabel=L"x",
        legend=:outerbottom,
        ylims = (ymin - .5, ymax + .5),
        legendfontsize=15,
		lw = 3,
		ratio = .7,
		framestyle=:zerolines,
		size=(800,600)
    )

	# if show_x02
		scatter!([x̂],[f(x̂)], label = "", mc=:white, msc=:gray, msw=2, alpha=2.0, series_annotations=(text(L"x_0", :top, 25)))
		# annotate!()
	# end
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
	# if add_linear2
    	plot!(p, xs, (x) -> taylor_approx(x; x̂=x̂); label=L"linear approx. at $x_0$", lc=2,  lw=1, title="Linear approximation")
	# end
	if add_quadratic2_
		plot!(p, xs, (x) -> taylor_approx(x; x̂=x̂, order=2); label=L"quadratic approx. at $x_0$", lc=3,  lw=3, title="Quadratic approximation")

	end

	p

	# xq = x̂ + 0.9
	# annotate!([xq], [taylor_approx(xq; x̂=x̂, order=2)], text("Quadratic approx"))

end

# ╔═╡ 9b045278-a171-4c4c-af85-a018f60641a2
md"""

## Qudratic approximation


> If ``f: \mathbb R \rightarrow \mathbb R`` is differentiable at ``x_0``, then
> 
> ``f(x)`` can be locally approximated by a **quadratic** function
> ```math
> \Large
> \begin{align}
> f(x) &\approx f(x_0) + f'(x_0)(x-x_0) + \frac{1}{2}\underbrace{f^{''}(x_0)(x-x_0)^2}_{(x-x_0)f^{''}(x_0)(x-x_0)}
> \end{align}
> ```

##### The multivariate *generalisation*

> If ``f: \mathbb R^n \rightarrow \mathbb R`` is differentiable at ``\mathbf{x}_0``, then
> 
> ``f(\mathbf{x})`` can be locally approximated by a **quadratic** function
> ```math
> \Large
> \begin{align}
> f(\mathbf{x}) &\approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top(\mathbf{x}- \mathbf{x}_0) + \\
> &\;\;\;\;\;\;\frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H}(\mathbf{x}_0)(\mathbf{x}-\mathbf{x}_0)
> \end{align}
> ```
"""

# ╔═╡ fc5934c2-7dd2-432e-afef-8b8ca237e90d
md"""
## Hessian matrix -- "second order derivative"

##### Function ``f: \mathbb{R}^n \rightarrow \mathbb{R}``, the equivalence of ``f''(x)`` is called *Hessian* matrix

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

# ╔═╡ f2d4c4f3-ae87-4aa2-84c7-3697babe6852
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

# ╔═╡ 75fc265d-f8c5-40b3-8843-786166b5120c
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

# ╔═╡ e274104b-a0a3-48e3-8ea9-b657c0477604
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

# ╔═╡ 8265eacc-4cc6-4da2-82ef-9b87d860384f
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
\mathbf{H}(\mathbf{x}) = \begin{bmatrix}\frac{\partial f^2(\mathbf{x})}{\partial x_1^2} & \frac{\partial f^2(\mathbf{x})}{\partial x_1\partial x_2}\\ \frac{\partial f(\mathbf{x})}{\partial x_2\partial x_1} & \frac{\partial f^2(\mathbf{x})}{\partial x_2^2}\end{bmatrix} = \begin{bmatrix}2 & \textcolor{red}{\frac{\partial}{\partial x_1} \left(\frac{\partial (x_1^2+x_2^2)}{\partial x_1 \partial x_2}\right )=\frac{\partial(2x_1)}{\partial x_2}} \\ \cdot  & \cdot \end{bmatrix}$$


"""

# ╔═╡ 25448984-7d6c-4bff-802a-45d52dc57971
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
\mathbf{H}(\mathbf{x}) = \begin{bmatrix}\frac{\partial^2 f(\mathbf{x})}{\partial x_1^2} & \frac{\partial^2 f(\mathbf{x})}{\partial x_1\partial x_2}\\ \frac{\partial^2 f(\mathbf{x})}{\partial x_2\partial x_1} & \frac{\partial^2 f(\mathbf{x})}{\partial x_2^2}\end{bmatrix} = \begin{bmatrix}2 & 0\\ 0  & 2 \end{bmatrix}$$


"""

# ╔═╡ 60f7019b-c72a-45b7-8d25-b1456dea0a5f
md"""

## 

#### Question: will I be asked to compute ``\mathbf{H}`` in the exam?


##### Answer: _No_! Understanding how it is used in optimisation is more important!
"""

# ╔═╡ a9015dc1-487c-4b10-9b5e-6c7b967ba912
md"""

## Multivariate max/min test
"""

# ╔═╡ a160a210-2c3b-460a-aed4-54fbabe72706
md"Show min: $(@bind show_mins CheckBox(default=true)); Show max: $(@bind show_maxs CheckBox(default=true)); Show saddles: $(@bind show_saddles CheckBox(default=true))"

# ╔═╡ 7c4ccb8b-d6fe-4042-86b9-4a57eaf49669
md"Check Hessians therefore max/min: $(@bind add_quadratic_app_ CheckBox(false))"

# ╔═╡ f20bd1ac-7cb5-4105-91d0-cbb206960033
let
	plotly()
	f = f_demo
	l0s_min = [[2, -1], [2, 2], [-1, 2], [-1, -1]]
	l0s_max = [[0, 0]]
	l0s_saddle = [[2, 0], [0, -1], [0, 2], [-1, 0]]
	
	x1_ = range(-2., stop =3.0, length=60)
	x2_ = range(-2., stop =3.0, length=60)
	p1_ = plot(x1_, x2_, (a, b) -> f([a, b]), st=:surface, xlabel ="x1", ylabel="x2", zlabel="f(x)", colorbar=false, color=:jet, title="")
	len = 1.2
	if show_mins
		for (li, l0) in enumerate(l0s_min)
			tf = taylorApprox(f, l0, 1)
			plot!(range(l0[1] - len, l0[1]+len, 5), range(l0[2] - len, l0[2]+len, 5), (a, b) -> tf([a, b]), st=:surface, alpha=0.4, display_option=Plots.GR.OPTION_Z_SHADED_MESH)
			label = li == 1 ? "x_min" : ""
			scatter!([l0[1]], [l0[2]], [f(l0)], ms=2, label="")
			if add_quadratic_app_
				tf2 = taylorApprox(f, l0, 2)
				surface!(p1_, range(l0[1] - 1.2, l0[1]+1.2, 30), range(l0[2] - 1.2, l0[2]+1.2, 30), (x, y) -> tf2([x, y]), label="", alpha=0.8, display_option=Plots.GR.OPTION_SHADED_MESH,zlim =(-1,9))
			end
		end
	end

	if show_maxs
		for (li, l0) in enumerate(l0s_max)
			tf = taylorApprox(f, l0, 1)	
			plot!(range(l0[1] - len, l0[1]+len, 5), range(l0[2] - len, l0[2]+len, 5), (a, b) -> tf([a, b]), st=:surface, alpha=0.4, display_option=Plots.GR.OPTION_Z_SHADED_MESH)
			label = li == 1 ? "x_max" : ""
			scatter!([l0[1]], [l0[2]], [f(l0)], ms=2, label="")
			if add_quadratic_app_
				tf2 = taylorApprox(f, l0, 2)
				surface!(p1_, range(l0[1] - 1.2, l0[1]+1.2, 30), range(l0[2] - 1.2, l0[2]+1.2, 30), (x, y) -> tf2([x, y]), label="", alpha=0.8, display_option=Plots.GR.OPTION_SHADED_MESH,zlim =(-1,9))
			end
		end
	end

	if show_saddles
		for (li, l0) in enumerate(l0s_saddle)
			tf = taylorApprox(f, l0, 1)
			plot!(range(l0[1] - len, l0[1]+len, 5), range(l0[2] - len, l0[2]+len, 5), (a, b) -> tf([a, b]), st=:surface, alpha=0.4, display_option=Plots.GR.OPTION_Z_SHADED_MESH)
			label = li == 1 ? "x_saddle" : ""
			scatter!([l0[1]], [l0[2]], [f(l0)], ms=2, label="")
			if add_quadratic_app_
				tf2 = taylorApprox(f, l0, 2)
				surface!(p1_, range(l0[1] - 1.2, l0[1]+1.2, 30), range(l0[2] - 1.2, l0[2]+1.2, 30), (x, y) -> tf2([x, y]), label="", alpha=0.8, display_option=Plots.GR.OPTION_SHADED_MESH,zlim =(-1,9))
			end
		end
	end
	p1_
end

# ╔═╡ 174d22a6-0bdc-4ae2-a789-c5b0901385dc
md"""

## Reading & references

##### Essential reading 


* [_Deep Learning_ by _Ian Goodfellow et al._: Chapter 4.3](https://www.deeplearningbook.org/contents/numerical.html)


##### Suggested reading 
* [_Linear algebra review and reference_ by Zico Kolter](https://studres.cs.st-andrews.ac.uk/CS5014/0-General/cs229-linalg.pdf) section 3.11 and 4
"""

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

# ╔═╡ b5b9e8ce-8d1f-43c9-8b29-5e23652a68e5
let
	plotly()
	f(x) = qform(x; A=Matrix(I,2,2), b=zeros(2), c=0) +5
	plt = plot(-5:0.5:5, -5:0.5:5, (x,y) -> f([x,y]), st=:surface, colorbar=false, color=:jet,alpha=0.8, xlim=[-5, 5] , ylim=[-5, 5],zlim =[0, 55], title=L"\mathbf{A}" * "is positive definite; minimum")
	θs = range(0, 2π, 15)
	length = 4
	for (ind, θ) in enumerate(θs)
		x, y = cos(θ) * length, sin(θ)* length	
		arrow3d!([0], [0], [0], [x], [y], [0]; as=0.1, lc=ind, la=0.9, lw=2, scale=:identity)
		v = [cos(θ), sin(θ)]
		xs = range(-5, 5, 50)
		k = v[2]/v[1]
		ys = k .* xs
		zs = [f([x, ys[i]]) for (i, x) in enumerate(xs)]
		path3d!(xs, ys, zs, lw=3, label="", c=ind)
	end
	plt
end

# ╔═╡ 736cdab9-a5f3-4f85-889a-86abbc5357e6
let
	x0 = xx
	plotly()
	A = Matrix(I,2,2)
	μ = [0,0]
	f(x1, x2) = dot([x1, x2]- μ, A, [x1, x2]-μ) + .5
	plt = plot(μ[1]-6:0.8:μ[1]+6, μ[2]-6:0.8:μ[2]+6, f, st=:surface, xlabel="x₁", ylabel="x₂", zlabel="f",  alpha=0.5, color=:jet, framestyle=:zerolines, ratio=1, colorbar=false, xlim=[-7.5, 7.5], ylim=[-6.8, 6.8], zlim =[-1, 60])
	scatter!([x0[1]], [x0[2]], [0], ms=1, markershape=:x, label="x'")
	scatter!([x0[1]], [x0[2]], [f(x0...)], ms=1, label="")
	vs = 0:0.1:f(x0...)
plot!(x0[1] * ones(length(vs)), x0[2] * ones(length(vs)), vs, lw=3, ls=:dash, lc=:gray, label="")
	res = 20
	x = range( -6,stop=40,length=10)
	y = range( -6,stop= 40,length=10)
	eps_x = range(x0[1] -1*eps() ,stop=x0[1]  +eps(),length=10) #create a very small range
	eps_y = range(x0[2]-1*eps(),stop=x0[2] +eps(),length=10)
	X = repeat(x,1,length(y))
	Y = repeat(y',length(x),1)
	if add_e1
			
	arrow3d!([x0[1]], [x0[2]], [0], [3], [0], [0]; as = 0.4, lc=2, la=1, lw=2, scale=:identity)
		surface!(x, eps_y, X, surftype=(surface = true, mesh = true), c=2, alpha=0.35)  #create "verticle"
		# surface!(eps_x,y,Y, surftype=(surface = true, mesh = true), c=:gray, alpha=0.5) #create "verticle" 
		xs = -5:0.5:5
		ys = x0[2] * ones(length(xs))
		zs = [f(xs[i], ys[i]) for i in 1:length(ys)]
		path3d!(xs, ys, zs, lw=3, label="", c=2)
	end
	if add_theother
		arrow3d!([x0[1]], [x0[2]], [0], [0], [3], [0]; as = 0.4, lc=1, la=1, lw=2, scale=:identity)

		ys = -5:0.5:5
		xs = x0[1] * ones(length(ys))
		zs = [f(xs[i], ys[i]) for i in 1:length(ys)]
		path3d!(xs,ys,zs, lw=3, label="", c=1)
		surface!(eps_x,y,Y, surftype=(surface = true, mesh = true), c=1, alpha=0.35) #create "verticle" 

	end
	plt
end

# ╔═╡ 63a348a3-fedc-4baf-83cf-313e82d18684
let
	gr()
	b = [1, 0]
	w₀ = 10
	f(x1, x2) = dot(b, [x1, x2]) + w₀
	plt = plot(-15:2:15, -15:2:15, (x1, x2) -> dot(b, [x1, x2])+w₀, st=:surface, xlabel=L"x_1", ylabel=L"x_2", zlabel="f", alpha=0.8, framestyle=:zerolines, c=:coolwarm, colorbar=false, camera=(15, 25))
	min_z, max_z = extrema([f(x1, x2) for x1 in -15:2:15 for x2 in -15:2:15])
	x = range( -15,stop=15,length=10)
	y = range( -15,stop= 15,length=10)	
	x0 = [6, 6.0]
	plot!([x0[1], x0[1]], [x0[2], x0[2]], [f(x0...), 0], lw=2, lc=:black, ls=:dash, label="")
	scatter!([x0[1]], [x0[2]], [f(x0...)], ms=2, markershape=:circle, label="", mc=:white, msc=:gray, msw=2, alpha=0.9)
	scatter!([x0[1]], [x0[2]], [0], label=L"\mathbf{x}_0", ms =3, mc=:white, msc=:gray, msw=2, alpha=1.0)
	r = 4.5
	if add_directions
		for theta ∈ range(-π, π, 12)
			arrow3d!([x0[1]], [x0[2]], [0], [r * cos(theta)], [r * sin(theta)], [0]; as = 0.2, lc=1, la=1, lw=1, scale=:identity)
		end
	end
	plt
end

# ╔═╡ 866f2f3b-8ca5-4b10-9384-065850007007
let
	gr()
	b = [1, 0]
	w₀ = 10
	f(x1, x2) = dot(b, [x1, x2]) + w₀
	plt = plot(-15:2:15, -15:2:15, (x1, x2) -> dot(b, [x1, x2])+w₀, st=:surface, xlabel=L"x_1", ylabel=L"x_2", zlabel="f", alpha=0.8, framestyle=:zerolines, c=:coolwarm, colorbar=false, camera=(15, 25))
	min_z, max_z = extrema([f(x1, x2) for x1 in -15:2:15 for x2 in -15:2:15])
	x = range( -15,stop=15,length=10)
	y = range( -15,stop= 15,length=10)	
	x0 = [6, 6.0]
	plot!([x0[1], x0[1]], [x0[2], x0[2]], [f(x0...), 0], lw=2, lc=:black, ls=:dash, label="")
	scatter!([x0[1]], [x0[2]], [f(x0...)], ms=2, markershape=:circle, label="", mc=:white, msc=:gray, msw=2, alpha=0.9)
	scatter!([x0[1]], [x0[2]], [0], label=L"\mathbf{x}_0", ms =3, mc=:white, msc=:gray, msw=2, alpha=1.0)
	r = 6
	if add_directions
		for theta ∈ range(-π, π, 12)
			arrow3d!([x0[1]], [x0[2]], [0], [r * cos(theta)], [r * sin(theta)], [0]; as = 0.2, lc=1, la=0.2, lw=1, scale=:identity)
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

# ╔═╡ 238e7b56-fb3a-4e9b-9c31-09c1f4a1df2a
# begin
# 	Random.seed!(111)
# 	num_features = 2
# 	num_data = 25
# 	true_w = rand(num_features+1) * 10
# 	# simulate the design matrix or input features
# 	X_train = [ones(num_data) rand(num_data, num_features)]
# 	# generate the noisy observations
# 	y_train = X_train * true_w + randn(num_data)
# end;

# ╔═╡ c9b5e47c-e0f1-4496-a342-e37df85d6de9
begin
	# define a function that returns a Plots.Shape
	rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
end;

# ╔═╡ 8deb1b8c-b67f-4d07-8986-2333dbadcccc
# md"""
# ![](https://allisonhorst.github.io/palmerpenguins/reference/figures/lter_penguins.png)"""

# ╔═╡ af622189-e504-4633-9d9e-ab16c7293f82
# df_penguin = let
# 	Logging.disable_logging(Logging.Warn)
# 	df_penguin = DataFrame(CSV.File(download("https://gist.githubusercontent.com/slopp/ce3b90b9168f2f921784de84fa445651/raw/4ecf3041f0ed4913e7c230758733948bc561f434/penguins.csv"), types=[Int, String, String, [Float64 for _ in 1:4]..., String, Int]))
# 	df_penguin[completecases(df_penguin), :]
# end;

# ╔═╡ 9267c4a4-74d1-4515-95d6-acc3b12e5ed6
# first(df_penguin, 5)

# ╔═╡ 76cc8ca7-d17e-4cd7-a6de-4d606e0a0985
# @df df_penguin scatter(:flipper_length_mm, :body_mass_g, group = (:species), legend=:topleft, xlabel="Flipper length", ylabel="Body mass");

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
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
CSV = "~0.10.10"
DataFrames = "~1.5.0"
ForwardDiff = "~0.10.36"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.21"
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

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "443f6ff241872543f19e89759ecb840260df7743"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "2b301c2388067d655fe5e4ca6d4aa53b61f895b4"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.31"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

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

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "54b00d1b93791f8e19e31584bd30f2cb6004614b"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.38"

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

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.BufferedStreams]]
git-tree-sha1 = "bb065b14d7f941b8617bc323063dbe79f55d16ea"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.1.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "ed28c86cbde3dc3f53cf76643c2e9bc11d56acc7"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.10"

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
git-tree-sha1 = "0aa0a3dd7b9bacbbadf1932ccbdfa938985c5561"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.58.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.Chemfiles]]
deps = ["Chemfiles_jll", "DocStringExtensions"]
git-tree-sha1 = "6951fe6a535a07041122a3a6860a63a7a83e081e"
uuid = "46823bd8-5fb3-5f92-9aa0-96921f3dd015"
version = "0.10.40"

[[deps.Chemfiles_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f3743181e30d87c23d9c8ebd493b77f43d8f1890"
uuid = "78a364fa-1a3c-552a-b4bb-8fa0f9c1fcca"
version = "0.10.4+0"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "a6e6ce44a1e0a781772fc795fb7343b1925e9898"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.2"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "d730914ef30a06732bdd9f763f6cc32e92ffbff1"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "be6ab11021cd29f0344d5c4357b163af05a48cba"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.21.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"
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
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "96d823b94ba8d187a6d8f0826e731195a74b90e9"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "738fec4d684a9a6ee9598a8bfee305b26831f28c"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.2"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataDeps]]
deps = ["HTTP", "Libdl", "Reexport", "SHA", "p7zip_jll"]
git-tree-sha1 = "bc0a264d3e7b3eeb0b6fc9f6481f970697f29805"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.10"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "aa51303df86f8626a962fccb878430cdb0a97eee"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.5.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

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
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "49eba9ad9f7ead780bfb7ee319f962c811c6d3b2"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.8"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "c72970914c8a21b36bbc244e9df0ed1834a0360b"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.95"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

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

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "f9818144ce7c8c41edf5c4c179c684d92aa4d9fe"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.6.0"

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
git-tree-sha1 = "299dc33549f68299137e51e6d49a13b5b1da9673"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "ed569cb9e7e3590d5ba884da7edc50216aac5811"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.1.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.FoldsThreads]]
deps = ["Accessors", "FunctionWrappers", "InitialValues", "SplittablesBase", "Transducers"]
git-tree-sha1 = "eb8e1989b9028f7e0985b4268dabe94682249025"
uuid = "9c68100b-dfe1-47cf-94c8-95104e173443"
version = "0.1.1"

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

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

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
git-tree-sha1 = "8b8a2fd4536ece6e554168c21860b6820a8a83db"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.7"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "19fad9cd9ae44847fe842558a744748084a722d1"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.7+0"

[[deps.GZip]]
deps = ["Libdl"]
git-tree-sha1 = "039be665faf0b8ae36e089cd694233f5dee3f7d6"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.5.1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Glob]]
git-tree-sha1 = "97285bbd5230dd766e9ef6749b80fc617126d496"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.1"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "d61890399bc535850c4bf08e4e0d3a7ad0f21cbd"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "c73fdc3d9da7700691848b78c61841274076932a"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.16.15"

[[deps.HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "4cc2bb72df6ff40b055295fdef6d92955f9dede8"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.2+2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e77dbf117412d4f164a464d610ee6050cc75272"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.6"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "84204eae2dd237500835990bcade263e27674a93"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.16"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "8aa91235360659ca7560db43a7d57541120aa31d"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "ce28c68c900eed3cdbfa418be66ed053e54d4f56"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.7"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0cb9352ef2e01574eeebdb102948a58740dcaf83"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2023.1.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InternedStrings]]
deps = ["Random", "Test"]
git-tree-sha1 = "eb05b5625bc5d821b8075a77e4c421933e20c76b"
uuid = "7d512f48-7fb1-5a58-b986-67e6dc259f01"
version = "0.7.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "6667aadd1cdee2c6cd068128b3d226ebc4fb0c67"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.9"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "42c17b18ced77ff0be65957a591d34f4ed57c631"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.31"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "SnoopPrecompile", "StructTypes", "UUIDs"]
git-tree-sha1 = "84b10656a41ef564c39d2d477d7236966d2b5683"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.12.0"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "6a125e6a4cb391e0b9adbd1afa9e771c2179f8ef"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.23"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "47be64f040a7ece575c2b5f53ca6da7b548d69f4"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.4"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "90442c50e202a5cdf21a7899c66b240fdef14035"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.7"

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
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "26a31cdd9f1f4ea74f649a7bf249703c687a953d"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "5.1.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "09b7505cc0b1cee87e5d4a26eea61d2e1b0dcd35"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.21+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

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
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "6eff5740c8ab02c90065719579c7aa0eb40c9f69"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.4"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MLDatasets]]
deps = ["CSV", "Chemfiles", "DataDeps", "DataFrames", "DelimitedFiles", "FileIO", "FixedPointNumbers", "GZip", "Glob", "HDF5", "ImageShow", "JLD2", "JSON3", "LazyModules", "MAT", "MLUtils", "NPZ", "Pickle", "Printf", "Requires", "SparseArrays", "Tables"]
git-tree-sha1 = "498b37aa3ebb4407adea36df1b244fa4e397de5e"
uuid = "eb30cadb-4394-5ae3-aed4-317e484a6458"
version = "0.7.9"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "FoldsThreads", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "ca31739905ddb08c59758726e22b9e25d0d1521b"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.2"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

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

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "99e6dbb50d8a96702dc60954569e9fe7291cc55d"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.20"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"

[[deps.NPZ]]
deps = ["FileIO", "ZipFile"]
git-tree-sha1 = "60a8e272fe0c5079363b28b0953831e2dd7b7e6f"
uuid = "15e1cf62-19b3-5cfa-8e77-841668bca605"
version = "0.4.3"

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
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+2"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a5aef8d4a6e8d81f171b2bd4be5265b01384c74c"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.10"

[[deps.Pickle]]
deps = ["DataStructures", "InternedStrings", "Serialization", "SparseArrays", "Strided", "StringEncodings", "ZipFile"]
git-tree-sha1 = "e6a34eb1dc0c498f0774bbfbbbeff2de101f4235"
uuid = "fbb45041-c46e-462f-888f-7c521cafbc2c"
version = "0.3.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ad59edfb711a4751e0b8271454df47f84a47a29e"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.14"

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
git-tree-sha1 = "88222661708df26242d0bfb9237d023557d11718"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.11"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "b478a748be27bd2f2c73a7690da219d0844db305"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.51"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "259e206946c293698122f63e2b513a7c99a244e8"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "213579618ec1f42dea7dd637a42785a608b1ea9c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.4"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "6d7bb727e76147ba18eed998700998e17b8e4911"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.4"
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
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "feafdc70b2e6684314e188d95fe66d116de834a7"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.2"

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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "77d3c4726515dca71f6d80fbb5e251088defe305"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.18"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

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

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

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
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "8982b3607a212b070a5e46eea83eb62b4744ae12"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.25"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "75ebe04c5bed70b91614d684259b661c9e6274a4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.0"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "14ef622cf28b05e38f8af1de57bc9142b03fbfe3"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.5"

[[deps.Strided]]
deps = ["LinearAlgebra", "TupleTools"]
git-tree-sha1 = "a7a664c91104329c88222aa20264e1a05b6ad138"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "1.2.3"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "33c0da881af3248dafefb939a21694b97cfece76"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.6"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "0a3db38e4cce3c54fe7a71f831cd7b6194a54213"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.16"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ca4bccb03acf9faaf4137a9abc1881ed1841aa70"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.10.0"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

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

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "25358a5f2384c490e98abd565ed321ffae2cbb37"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.76"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "ba4aa36b2d5c98d6ed1f149da916b3ba46527b2b"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.14.0"
weakdeps = ["InverseFunctions"]

    [deps.Unitful.extensions]
    InverseFunctionsUnitfulExt = "InverseFunctions"

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
git-tree-sha1 = "ea37e6066bf194ab78f4e747f5245261f17a7175"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.2"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

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
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "30c1b8bfc2b3c7c5d8bba7cd32e8b6d5f968e7c3"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.68"

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

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

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

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

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
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─17a3ac47-56dd-4901-bb77-90171eebc8c4
# ╟─a26e482a-f925-48da-99ba-c23ad0a9bed6
# ╟─29998665-0c8d-4ba4-8232-19bd0de71477
# ╟─cb72ebe2-cea8-4467-a211-5c3ac7af74a4
# ╟─f9023c9e-c529-48a0-b94b-31d822dd4a11
# ╟─d11b231c-3d4d-4fa2-8b1c-f3dd742f8977
# ╟─36d19f22-cada-446e-a7df-30be835cc373
# ╟─c23db844-4f23-47db-8189-ed5f3035692a
# ╟─463a8681-9989-4d90-897d-c8df3a328274
# ╟─a03ae249-b644-4269-b695-0ce8bb13a276
# ╟─bfcab553-2f25-4481-9b4c-1e5466100428
# ╟─ecb880b2-242a-4a0c-a727-caa2ef09f89c
# ╟─2afb3251-4471-496d-a4c5-aff3df1b5ae6
# ╟─7b3cf4f1-1f86-4fec-9ca3-4f1f5e977512
# ╟─992344fe-1f96-41e4-abbd-cae8501a19b5
# ╠═80492c19-3263-4824-8fe6-49789d341d4e
# ╟─728c0b5e-5bf5-4c1d-8b7b-c7a41bae4f7e
# ╟─35ddd576-28fe-4a08-90cc-ec977a3da9b3
# ╟─ef2ce0c0-887f-4d7c-be7b-74be3e1d3ac7
# ╟─fdb084c6-e5e7-4be4-bc52-6d23fc65c84e
# ╟─c4290f2d-ee0a-475a-909a-7876bc32a147
# ╟─6d415e5a-ffbd-45d3-badc-13741a009438
# ╟─856b97e5-88a7-4399-b449-1fef8aae58e6
# ╟─fe6a4422-eebb-40db-b3ef-5ab4736798da
# ╟─3cdfb6b3-18c0-49f1-95d7-d21a0e9891b5
# ╟─d412c90d-f170-42b9-83ff-d4959f043dbf
# ╟─77c1a618-055a-449a-a737-12789624ed7f
# ╟─1d6d6565-c8e3-4f65-a581-29a6a5d8da5e
# ╟─acacb324-8eb2-427e-85fe-354a56fd0df9
# ╟─99054046-a379-4c9e-b93e-5822d2ea00dd
# ╟─a49a52c9-f6ca-4dfc-afa1-8af648763ef5
# ╟─c05f97ed-c92c-4895-a44e-bfc497a8dc78
# ╟─e432d785-7107-490f-8a4d-00893077e764
# ╟─e7de66c8-c332-48fc-b4ca-57584d1aa364
# ╟─80b3ef8f-42d2-4c11-a22e-71b8b1462173
# ╟─2a5b2096-7c06-4173-9bbe-cb684de63bc3
# ╟─8b6fd167-eaca-44ac-9fab-868e7f639a4e
# ╟─631125ee-4c81-4c43-bb51-0e9c63826028
# ╟─a2613587-64bb-4d2a-95a4-a21ce6f0a77e
# ╟─7f8094e3-64c3-4b41-87e4-9bf676595339
# ╠═275a2ccb-254d-473c-bb67-50d80cdf590c
# ╠═b5d8084c-9cb3-4ff6-b61a-176b454a695e
# ╟─1f94cba0-6bbb-480b-a62b-81bbbe0ed3ed
# ╟─90a78fd7-c6d6-49b9-b8fb-d056de180b9b
# ╟─23133bc6-dc27-4d09-85c1-0e0ad188a038
# ╟─fad92d3e-04ea-40b1-ae7e-35c5b3c37be1
# ╟─5794b474-5541-4625-b433-f30b9c6a857a
# ╟─97193fb0-1b92-4c45-be35-e5a61e7f58ed
# ╟─766bf632-e1c0-4c5f-8f71-4304ec16a1cb
# ╟─a0d30c40-b64a-4953-8e1a-3315c0cf16d0
# ╟─fa17d5c4-26e0-4fe6-aea8-614d7e0dc099
# ╟─791534b7-f284-495f-9710-32719656b661
# ╟─4fae417c-d0a1-4ee0-baab-e0d27e84e8cf
# ╟─dfba25eb-0316-464f-8ff6-b4acc8d86c1e
# ╟─afa0f2a8-0fbf-4783-b788-f481f6eaa691
# ╟─a6314fbe-2491-4d8f-afd7-5b5669df94cb
# ╟─b0e577be-fc2d-40a4-81a5-ae6530d93939
# ╟─87a63a6b-5fcc-4d79-982b-90e31fcd4887
# ╟─8e91558a-0d80-4fcf-940d-47fdc90812ec
# ╟─193a34da-e52c-4625-83c9-7e942496023d
# ╟─bc3a7134-ee9a-4f98-97e2-f3141addb37d
# ╟─dddfbde0-7710-41de-a829-5a85cbd57544
# ╠═eed66f8f-361d-4a18-8c30-c407202c95ea
# ╟─33c93e1c-ba0d-4082-8c8f-fe2a9826334c
# ╟─6bea0190-b914-4397-844a-90a8699a6123
# ╟─32184644-ec26-4fbb-b9c7-c4f436338840
# ╟─629b6fbf-e1af-469b-9543-71642562b3a3
# ╟─871229d9-28f8-4245-99e9-5dd11e186a83
# ╟─f2ece3eb-b582-44b6-9bf7-84d3b7bd806a
# ╟─084eb288-a774-4369-882d-bd86a7c6f572
# ╟─8b7fc5db-7e7c-42fb-8a64-2d4530cb53c3
# ╟─a0b18f94-39f0-4f84-86e5-d772cd2206e1
# ╟─81725e2b-1e8d-4af5-80dd-4ba017573c5d
# ╟─7c893c0d-3a9a-45b6-9e08-0224892bf385
# ╟─538b7908-3d95-461c-a24d-b3b05a914ddb
# ╟─9c08d9dd-769b-4c39-a59b-db6924ea1c11
# ╟─1fd7978a-77f9-47ca-be6b-d2a05ec82f64
# ╟─e4f5077c-4ed8-4305-b12b-0a7639844fd6
# ╟─3250ecdf-b115-43cb-8602-f5814aec79c1
# ╟─6adcf3c5-a30b-406a-a3f2-4d1c8e8e0fa2
# ╟─e5ab945d-8d15-4979-bd5b-5aef9c36c85c
# ╟─38caa279-0e78-4f64-9dfb-782f229d9e94
# ╟─38c0f8fb-ce5d-4e78-9cd3-50a0f317ed12
# ╟─1ef66d06-5981-4e94-9e1e-6472f1e35aaf
# ╟─e5fe477d-2694-4803-a101-b2fd9cad9726
# ╟─9cb7368a-0cfa-46cd-8e76-fba25abe4ed0
# ╟─77188a12-c136-483b-841a-280f53034d7b
# ╟─7f93a9b2-4485-4ae1-931c-46ae6c994eff
# ╟─82034449-2245-4387-99d1-15e49623b0ad
# ╟─b5b9e8ce-8d1f-43c9-8b29-5e23652a68e5
# ╟─01049ef2-6364-478f-b756-410681796879
# ╟─3bb46ea9-94ee-4be2-b0ec-bfce14d521a2
# ╟─fc733c15-01bf-463c-a3e7-1bcd20c1a5f1
# ╟─9fab28fa-d158-4cd6-b706-817618be4e4c
# ╟─2f21c8d6-9a96-47d0-9150-5cc937085769
# ╟─dfa4fc88-6cfc-4d87-ba7c-87bb2e91bae0
# ╟─ccf4b05b-c5ca-41f2-80d3-7482d465467c
# ╟─0fba1c83-7f68-4f66-8f28-92c382f16ac9
# ╟─de792554-a433-45ba-b397-22bd027a54e8
# ╟─a80e21da-90f4-40d2-8ca3-eda504180295
# ╟─76239279-b34f-441d-9a21-184a24345637
# ╟─5be1eead-5c0a-477e-9227-31dd0e7a000f
# ╠═5c06f638-f429-4f53-a971-f0adc68ef6a1
# ╟─736cdab9-a5f3-4f85-889a-86abbc5357e6
# ╟─95247ac3-7058-4042-8c84-f52b19770313
# ╟─a4a4b442-a0fb-4193-b82e-38cfe463d38e
# ╟─a519be30-5065-4de3-a4d7-d0eb6261b549
# ╟─f45b1035-7d34-4069-bca5-600a4ba43a7e
# ╟─c5874a8c-b37a-45a9-8bf0-d32f39085374
# ╟─affa56cf-e919-48f6-a39f-ee534f966a7f
# ╟─3c39ca05-4856-48c6-af66-ad0f99c2ef19
# ╟─b86f4f23-ee0c-42e9-97f5-ee343a391d1e
# ╟─e161ad2e-8df1-4757-badd-ec83bc1de986
# ╟─8d5d9759-2d18-4eea-9ddc-f4ae357363ea
# ╟─b8dcfce1-881b-4f0f-8e35-c8831b8f8c52
# ╟─44a399ba-1ad7-4e4d-8ff3-ae30929e97bb
# ╟─92c18f58-aa3e-4844-ab23-f01fd2705fb3
# ╟─338e662e-57b2-4bc0-a4ea-dc7bf6a790f4
# ╟─6e79f928-93eb-4a75-9e7b-f3666d6a71e2
# ╟─ee135ef9-5be5-4a4e-80fe-d04179215cab
# ╟─fd86950f-4478-4b18-8bc3-94cf908eac32
# ╟─fc377a9a-92b2-4444-b5bf-0402f0b39e88
# ╟─faba77c4-afe2-434b-9ce7-3bb1740cbb0f
# ╟─ce95d775-1234-4a61-86bf-bf82d9051c38
# ╟─fdab2182-1a2f-497d-b560-5f4b62ea0554
# ╟─dcaa1293-b7f9-4f10-b7c3-a6b328cfd890
# ╟─9441ff29-4df7-41cf-a85d-3219ec9ad0f0
# ╟─1ba6cdc1-0612-49f7-b9c7-833877e8b80f
# ╟─d54951a4-2379-4b85-960d-fdaea320b376
# ╟─025a340c-21e4-48b3-a76e-4acae68c40ac
# ╟─5d182ed5-74b0-4e42-a381-1464c62d876a
# ╟─dde45286-689b-458e-a19f-225bd0e64cbf
# ╟─63a348a3-fedc-4baf-83cf-313e82d18684
# ╟─e8144237-53e6-4455-b198-03f60caae667
# ╟─73aafe0e-1112-4b6b-a87b-3f60cd2f4d03
# ╟─535ee2f4-64ae-4799-9a4f-fbc6fdd7c9d6
# ╟─eafcde85-cc83-4903-aef0-04e24f34d169
# ╟─b0a301fa-5b5f-4dcd-90dc-506ca860d950
# ╟─bee80039-b7f6-4d43-ae23-8bf20629e3d9
# ╟─866f2f3b-8ca5-4b10-9384-065850007007
# ╟─022193c3-d6bb-4292-82c4-85b94df7775c
# ╟─aecf467e-5bb1-4429-b62a-e34156eb5f83
# ╟─a34bedc7-1f89-4535-8fe7-98097e905695
# ╟─1f2cc5d7-5fd7-42d0-963f-24aa0de076eb
# ╟─38d3f4ec-c8be-4b19-9484-f8748fbd0df7
# ╟─474aaa5d-2f26-4e3a-adee-fb5dbac67c15
# ╠═63e09a57-44a6-4a5b-b9ce-b09dbbcf0f46
# ╟─df7e37ee-dde4-483c-bf2f-66a219077b35
# ╟─a9177e2e-b2f8-46cc-ac64-504d2ec896a4
# ╟─cfa000a1-f350-442f-8a40-82e4b8ee0fef
# ╟─76ea84a3-3013-4a6e-ae33-e439c5f16d31
# ╟─ade67f75-7eb7-418b-b5f5-ddbb8f1a6cc3
# ╟─7451193a-6a5c-424f-8ffd-14550ce6320c
# ╟─479b34fb-d1d1-443a-8ee9-949e35a378c3
# ╟─a9573110-2ebf-4fb5-89a4-f36926068fc4
# ╟─9426b55b-d6fb-4bed-b518-6cf22d78fcb6
# ╟─3f53c70a-b62a-4266-aca2-ca8570a58207
# ╟─7384f349-ec08-41bf-91b6-d80ffdda59b5
# ╟─ee9565f8-9e54-4f5a-8df7-f4983cfe3c76
# ╟─668074ee-fcc3-4e78-82fe-5f1c83e8ebfc
# ╟─6549244a-679c-4e12-9f60-f77c25afecaa
# ╟─3a059c7c-38b3-4aa4-b9e6-88e4235a4f4b
# ╟─429a7da7-ce5d-4207-af56-cee550112335
# ╟─e6825073-f73e-409b-9944-bc0258598003
# ╟─f598a0af-f235-49b9-b85a-3e7a841b70d0
# ╟─cbf8e293-8d7b-4420-8631-58b04e3725f3
# ╟─7ff064f9-5e99-4fc5-a0e9-60768d2553b2
# ╟─ceed5776-0fdc-4457-8064-a735e7652761
# ╟─64188019-bcdd-4a7a-ac15-529223737274
# ╟─8965aa00-0832-4e76-91d4-e68a51f41518
# ╟─9b045278-a171-4c4c-af85-a018f60641a2
# ╟─fc5934c2-7dd2-432e-afef-8b8ca237e90d
# ╟─f2d4c4f3-ae87-4aa2-84c7-3697babe6852
# ╟─75fc265d-f8c5-40b3-8843-786166b5120c
# ╟─e274104b-a0a3-48e3-8ea9-b657c0477604
# ╟─8265eacc-4cc6-4da2-82ef-9b87d860384f
# ╟─25448984-7d6c-4bff-802a-45d52dc57971
# ╟─60f7019b-c72a-45b7-8d25-b1456dea0a5f
# ╟─a9015dc1-487c-4b10-9b5e-6c7b967ba912
# ╟─a160a210-2c3b-460a-aed4-54fbabe72706
# ╟─7c4ccb8b-d6fe-4042-86b9-4a57eaf49669
# ╟─f20bd1ac-7cb5-4105-91d0-cbb206960033
# ╟─174d22a6-0bdc-4ae2-a789-c5b0901385dc
# ╟─974f1b58-3ec6-447a-95f2-6bbeda43f12f
# ╟─f1261f00-7fc6-41bb-8706-0b1973d72955
# ╟─238e7b56-fb3a-4e9b-9c31-09c1f4a1df2a
# ╟─c9b5e47c-e0f1-4496-a342-e37df85d6de9
# ╟─8deb1b8c-b67f-4d07-8986-2333dbadcccc
# ╟─f79bd8ab-894e-4e7b-84eb-cf840baa08e4
# ╟─af622189-e504-4633-9d9e-ab16c7293f82
# ╟─9267c4a4-74d1-4515-95d6-acc3b12e5ed6
# ╟─76cc8ca7-d17e-4cd7-a6de-4d606e0a0985
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
