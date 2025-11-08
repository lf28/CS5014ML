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

# ╔═╡ 6bdbcbc6-2203-46f9-b633-e742a6884a11
begin
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	using PlutoTeachingTools
	using PlutoUI
	# using Plots
	using LinearAlgebra
	# using StatsPlots
	# using LogExpFunctions
	# using StatsBase
	using LaTeXStrings
	using Latexify
	using Random
	using HypertextLiteral
	
	using ForwardDiff
end

# ╔═╡ 45f4c200-b0bb-4d15-a6c9-b519a25afcba
f(x) = x * sin(x^2) + 1; # you can change this function!

# ╔═╡ 23ad8369-85dc-49cd-baba-cff6c6979e14
ChooseDisplayMode()

# ╔═╡ 522d01e5-a19c-402f-89f3-b6e6fb98dad6
TableOfContents()

# ╔═╡ fd748bc2-305a-4b93-ba83-2b825f901516
md"""

# TL;DR guide of calculus


\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ b1b5bebc-79ec-432b-809f-f37c0f2065a1
md"""

## Reading & references

##### Essential reading 


* [_Deep Learning_ by _Ian Goodfellow et al._: Chapter 4.3](https://www.deeplearningbook.org/contents/numerical.html)


##### Suggested reading 
* [_Linear algebra review and reference_ by Zico Kolter](https://studres.cs.st-andrews.ac.uk/CS5014/0-General/cs229-linalg.pdf) section 3.11 and 4
"""

# ╔═╡ e5709c4a-c838-4888-ac52-645b3e6230fc
md"""

## Recap: linear function



```math
\huge
f(x) = b\cdot x+ c
```

* ##### ``c``: intercept
* ##### ``b``: slope
"""

# ╔═╡ c52dd91c-9ce4-4d2e-970e-57ee3c2684f5
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

# ╔═╡ 0dd38584-b6b0-463f-8e21-ef625fb97277
md"""

## Recap: quadratic function


```math
\huge 
f(x) = ax^2 + b x+ c, \;\; a\neq 0

```
"""

# ╔═╡ f8e1cdae-4929-4e8a-a58f-8e9637eea406
pltapos, pltaneg=let
	gr()
	b, c = 0, 0
	plt = plot( legend=:outerbottom, title="Effect of "*L"a>0", size=(300,400))
	plt2 = plot( legend=:outerbottom, title="Effect of "*L"a<0", size=(300,400))
	
	ass = [0.1, 1,2,3,4,6]
	for a in ass
		plot!(plt, -5:0.2:5, (x) -> a* x^2 + b* x+ c, framestyle=:origin, label=L"f(x) = %$(a)x^2", legendfontsize=10, lw=2)
		plot!(plt2, -5:0.2:5, (x) -> -a * x^2 + b* x+ c, framestyle=:origin, label=L"f(x) = -%$(a)x^2", legendfontsize=10, lw=2)
	end


	plt, plt2
end;

# ╔═╡ c45d6b24-026e-4936-a12e-f3f1ec3794e3
TwoColumn(md"

#### when `` a > 0``
##### _the function_ has a *minimum*

$(pltapos)
", 
	
	
md" #### when `` a<0``


##### _the function_ has a *maximum*


$(pltaneg)
")

# ╔═╡ f78d8e73-8e66-4432-93ad-61e064f437d3
md"""

## Differential calculus -- the big picture


#### Given a non-linear ``f``
* #### if we _**zoom-in**_, what do you observe?

"""

# ╔═╡ 904e6749-2832-48ff-b4e3-a546901a6fda
Foldable("Zoom-in observation", md"""

(``f\Rightarrow`` a straight line)
""")

# ╔═╡ 576ef6a5-8507-452a-9e72-9a48e9ad2842
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

# ╔═╡ f276290b-da38-4267-ab17-c49a05adfa1a
md"""
## `TL;DR` of univariate calculus


#### A differentiable function ``f(x)``  can be approximated locally by a _linear function_

"""

# ╔═╡ 1b19c622-7208-4711-887a-62335934c0d5
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/linearappx.svg" width = "380"/></center>"""

# ╔═╡ c2423b8f-7cd5-48a9-9a27-ff11060e1dd2
let
	gr()
	x̂ = 1
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
        legendfontsize=10,
		lw = 3,
		ratio = .7,
		framestyle=:zerolines,
		size=(700,400)
    )


	if true
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
	if true
    	plot!(p, range(x̂ - 3, x̂ +3, 3), (x) -> taylor_approx(x; x̂=x̂); label=L"f_{linear}(x) = f(x_0) + f'(x_0)(x-x_0)", lc=:black,  lw=2, title="Linear approximation")
	end

	p

	# xq = x̂ + 0.9
	# annotate!([xq], [taylor_approx(xq; x̂=x̂, order=2)], text("Quadratic approx"))

end

# ╔═╡ b5fc7959-87e6-4f83-aeab-8f5818a45b90
md"""
##

#### If ``f(x)`` is twice-differentiable, then it can be further approximated by a _quadratic function_
```math
\Large
\begin{align}
f(x) &\approx f(x_0) + f'(x_0)(x-x_0) + \boxed{\frac{1}{2}f^{''}(x_0)(x-x_0)^2}
\end{align}
```

"""

# ╔═╡ 77b81e32-3c64-4ad5-b16d-66bbbf534f02
md"Show ``x_0``: $(@bind show_x02 CheckBox(false)); Add linear approx.: $(@bind add_linear2 CheckBox(false)); Add quadratic approx.: $(@bind add_quadratic2 CheckBox(false)); Move me ``x_0``: $(@bind x̂2 Slider(-2:0.005:3, default=-1.355, show_value=true))"

# ╔═╡ e2b926ab-49ba-4862-9b0f-b396441f708d
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
    	plot!(p, range(x̂ - 1, x̂ +1, 3), (x) -> taylor_approx(x; x̂=x̂); label=L"linear approx. at $x_0$", lc=2,  lw=2, title="Linear approximation")
	end
	if add_quadratic2
		x_center = x̂
		if (abs(fprimep) > 1e-5) && abs(fprime) < 1e-3
			x_center = (fprimep * x̂ -fprime)/ fprimep
		end
		plot!(p, range(x_center -1, x_center +1, 30), (x) -> taylor_approx(x; x̂=x̂, order=2); label=L"quadratic approx. at $x_0$", lc=3,  lw=3, ls=:dash, title="Quadratic approximation")
		fpptxt = ""
		if fprimep > 0
			fpptxt = Plots.text(L"f^{''}(x)>0", 20, :green)
		elseif abs(fprimep) < 1e-5
			fpptxt = Plots.text(L"f^{''}(x)=0", 20, :green)

		else
			fpptxt = Plots.text(L"f^{''}(x)<0", 20, :green)
		end
		annotate!([x̂], [f(x̂)+1], fpptxt)

	end

	p

	# xq = x̂ + 0.9
	# annotate!([xq], [taylor_approx(xq; x̂=x̂, order=2)], text("Quadratic approx"))

end

# ╔═╡ 14bfdb25-2015-44cc-991b-536b221d388e
md"""

## `TL;DR` -- cont.

#### The _linear/quadratic_ approxs can be used to find  max/min ``f(x)``
"""

# ╔═╡ ece30c5c-b579-4968-8bd3-137e46605218
md"Show quadratic: $(@bind show_qapproxs CheckBox(false)), Show local mins: $(@bind show_local_mins CheckBox(false)), Show local max: $(@bind show_local_maxs CheckBox(false)), Show saddles: $(@bind show_local_saddles CheckBox(false))"

# ╔═╡ ae484ad1-bbe6-4892-9932-27f105224877
TwoColumn(
md"""

#### To optimise ``f``
* ##### ``\large \arg\max_x f(x)`` or ``\large \arg\min_x f(x)``

#### We solve 

```math
\Large
f'(x) = 0
```
#### To test max/min 
* ##### maximum: ``f''(x) <0``

* ##### minimum: ``f''(x) > 0``

* ##### neither: ``f''(x) =0``

""",
	
let
	fmins = [-1.3552 , 2.1945]
	fmaxs = [1.3552, 2.81373]
	fsaddle = [0]
	gr()
    xs = range(-2, 3, 200)
    ymin, ymax = extrema(f.(xs))
    p = plot(
        xs,
        f;
        label="",
        xlabel=L"x",
        legend=:outerbottom,
        ylims = (ymin - .9, ymax + .9),
        legendfontsize=15,
		lw = 3,
		ratio = .7,
		framestyle=:zerolines,
		size=(400,400)
    )
	fprimef(x_) = ForwardDiff.derivative(f, x_)
	fprimepf(x_)= ForwardDiff.derivative(x -> ForwardDiff.derivative(f, x), x_)

	function taylor_approx(x; x̂, fp, fpp=0.0, order = 1) 
		fx = f(x̂) + fp * (x - x̂)
		if order > 1
			fx += .5 * fpp * (x-x̂)^2	
		end# f(x) ≈ f(x̃) + 𝒟f(x̃)(x-x̃)
		return fx
	end 

	for (i, xs) ∈ enumerate([fmins, fmaxs, fsaddle])
		for x̂ in xs
			fprime = fprimef(x̂)
			# fprimep = fprimepf(x̂) 
			scatter!([x̂],[f(x̂)], label = "", mc=:white, msc=:gray, msw=2, alpha=2.0)
	    	plot!(p, range(x̂ -0.5, x̂+0.5, 10), (x) -> taylor_approx(x; x̂=x̂, fp= 				fprime); label="", lc=:gray,  lw=2, title="")
			if show_qapproxs
				fprimep = fprimepf(x̂) 
				lc = 4
				if i == 1
					lc = 3
				elseif i ==2
					lc=2
				end
				plot!(p, range(x̂ -0.4, x̂+0.4, 20), (x) -> taylor_approx(x; x̂=x̂, fp= fprime, fpp=fprimep, order=2); label="", lc=lc,  lw=2.5,alpha=0.8, title="")
			end
		end
	end
	
	if show_local_mins
		for (i, x̂) ∈ enumerate(fmins) 	
			# anno = i == 1 ? "local min" : "global min"
			scatter!([x̂],[f(x̂)], label = "", mc=:white, msc=:gray, msw=2, alpha=2.0, series_annotations=(text(L"x_{min}", :blue, :top, 18)))
			# scatter!([x̂],[f(x̂)], label = "", mc=:white, msc=:gray, msw=2, alpha=2.0, series_annotations=(text(L"f''(x_{min})>0", :blue, :bottom, 20)))
			annotate!([x̂], [f(x̂)-.8], Plots.text(L"f^{''}(x_{min}) >0",:blue, 10))
		end
	end

	
	if show_local_maxs
		for x̂ ∈ fmaxs 	
			scatter!([x̂],[f(x̂)], label = "", mc=:white, msc=:gray, msw=2, alpha=2.0, series_annotations=(text(L"x_{max}", :red, :bottom, 20)))
			annotate!([x̂], [f(x̂)+1], Plots.text(L"f^{''}(x_{max}) <0",:red, 10))

			# fprime = fprimef(x̂)
			# fprimep = fprimepf(x̂) 
			# plot!(p, range(x̂ -0.4, x̂+0.4, 20), (x) -> taylor_approx(x; x̂=x̂, fp= fprime, fpp=fprimep, order=2); label="", lc=2,  lw=2.5, alpha=0.8, title="")
		end
	end


	if show_local_saddles
		for x̂ ∈ fsaddle 	
			scatter!([x̂],[f(x̂)], label = "", mc=:white, msc=:gray, msw=2, alpha=2.0, series_annotations=(text(L"x_{\texttt{saddle}}", :purple, :bottom, 20)))

			annotate!([x̂], [f(x̂)-0.2], Plots.text(L"f^{''}(x_{min}) =0",:purple,:top, 10))

			# fprime = fprimef(x̂)
			# fprimep = fprimepf(x̂) 
			# plot!(p, range(x̂ -0.4, x̂+0.4, 20), (x) -> taylor_approx(x; x̂=x̂, fp= fprime, fpp=fprimep, order=2); label="", lc=4,  lw=2.5, alpha=0.8, title="")
		end
	end


	p


end
	
)

# ╔═╡ d44a2ce7-bf3e-4da5-8c53-a241524a891a
l0s = [[0, 0], [2, -1], [2, 2], [-1, 2], [-1, -1], [2, 0], [0, -1], [0, 2], [-1, 0]];

# ╔═╡ 47f1ce87-8e8c-4ecc-b9e0-e2338189f2e6
md"""

## Derivative -- `f'(x)`

"""

# ╔═╡ 954be22e-e3d1-42eb-a7e6-712fb1e3e0ed
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/diriv_.svg" width = "900"/></center>"""

# ╔═╡ ae4072fb-5686-470b-afb5-7fd537df6603
md"``\Delta x``: $(@bind Δx Slider(1.5:-0.1:0, default=1.5)), Add approx area: $(@bind add_neighbour CheckBox(default=false))"

# ╔═╡ eee41566-4d5d-43c8-9899-13f3cf9e8ba0
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

# ╔═╡ bb06c5ee-d08c-4343-869c-6807d3a8ac21
md"""

# Multi-variate differential calculus
"""

# ╔═╡ 89c876f2-71bf-471d-ae75-6d3291583995
md"""
## (Multi-variate) Linear function: ``\mathbb R^n \rightarrow \mathbb R``


```math
\Large 
\begin{align}
f(\mathbf{x}) &=   c + b_1 x_1 + b_2 x_2 + \ldots  b_n x_n\\
	&= c + \mathbf{b}^\top \mathbf{x} 
\end{align}
```

* where ``\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top`` 
* ##### "`slopes`": ``\mathbf{b} = [w_1, w_2, \ldots, w_n]^\top``


"""

# ╔═╡ d4e093f7-edda-4a29-bed5-fb908c764ba0
aside(tip(md"
Recall ``\mathbf{b}^\top \mathbf{x}  = b_1 x_1 + b_2 x_2 + \ldots  b_n x_n``
"))

# ╔═╡ 7cfadac5-2041-4af7-abed-920120701f90
md"""


```math
\Large 
\begin{align}f(x) = c + b\cdot x\; \xRightarrow{\color{blue}\text{generalisation} } 
\; f(\mathbf{x})
	= c+ \mathbf{b}^\top \mathbf{x}
\end{align}
```
* #### _direct generalisation_ of the linear function

* #### _line_ to _(hyper-)plane_
"""

# ╔═╡ 33e6e2eb-6ff5-4a92-ad28-f9f57e3ccb39
md"""




## Hyperplane example

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

* ##### _generalisation_ of  ``f(x)=1``

"""

# ╔═╡ 16aa467e-dbb5-4fdd-be97-617590a7f3e8
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

# ╔═╡ ac42b3a9-6744-49ae-b55f-48fb184bedaa
md"""

## More hyperplane examples

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

# ╔═╡ cb58cc20-172a-4c2c-81f3-e3ccd7bcd8ab
b = [1, 0]

# ╔═╡ 49d64d9d-d077-4b7c-8a90-4c1d7ba7cdc6
md"Add ``x_2=c`` vertical plane: $(@bind add_x2plane CheckBox(default=false)), add more planes: $(@bind add_more_x2_planes CheckBox(default=false))"

# ╔═╡ f905afd1-bf33-4d9a-8b73-c2dfa79995ff
md"Add ``x_1=c`` vertical plane: $(@bind add_x1plane CheckBox(default=false)), add more planes: $(@bind add_more_x1_planes CheckBox(default=false))"

# ╔═╡ 14149895-394f-4c58-a135-12366f2a873e
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

# ╔═╡ 7d17349b-1e3a-4011-954d-483c1493078f
md"""

## `TL;DR` -- cont.

#### The same idea applies to multivariate function
* ##### a differentiable ``f(\mathbf{x})`` approximated by a hyper-plane
"""

# ╔═╡ 2acd37fc-3b78-4c33-9ec9-c405f1527aca
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/linearappxmulti.svg" width = "450"/></center>"""

# ╔═╡ 47d3302c-ad34-4a82-8c1c-46f74ce10440
md"Add ``\mathbf{x}_0``: $(@bind add_x0_ CheckBox(default=false)), Add linear: $(@bind add_linear_app CheckBox(default=false)), Add quadratic : $(@bind add_quadratic_app CheckBox(default=false))"

# ╔═╡ a21182db-3f9d-4dca-8cdd-08dc38453231
md"""Move me ``x_1``: $(@bind x01_ Slider(-1.8:0.1:2.8; default= 0)), ``x_2``: $(@bind x02_ Slider(-1.8:0.1:2.8; default= 0))"""

# ╔═╡ 0932bac0-921f-40d2-bfcf-7af2d8f1a268
md"""

## Gradient -- ``\nabla f(\mathbf{x})``
"""

# ╔═╡ 3240adaf-f984-4caf-95a9-742a40897f46
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/grrad.svg" width = "650"/></center>"""

# ╔═╡ 7f76686c-21dc-414e-9526-f12e98e57bb2
md"""
## Partial derivative

#### The *partial derivative* w.r.t. $x_i$ is

$$\large \frac{\partial f}{\partial \textcolor{red}{x_i}}(\mathbf{x}) = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, \textcolor{red}{x_i+h}, \ldots, x_n) - f(x_1, \ldots, \textcolor{red}{x_i}, \ldots, x_n)}{h}$$

* ##### _change_ one dimension (``i``- th dimension) *while keeing* all ``x_{j\neq i}`` constant


"""

# ╔═╡ fb08d3fa-35a5-4afa-88d6-7587c1bfa339
md"Add ``\mathbf{e}_1``: $(@bind add_e1 CheckBox(default=false)), add ``\mathbf{e}_2``: $(@bind add_theother CheckBox(default=false))"

# ╔═╡ b2113f74-c2ad-46c1-8092-fd8190649504
xx = [2,2]

# ╔═╡ 0452ca78-9ed3-4355-afea-e3819ef79afd
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

# ╔═╡ 5d68596e-856f-49b8-a376-f4449bd949cd
md"""
## Optimisation

##### To optimise ``f(\mathbf{x})``, we find where the gradient vanishes
* ##### or _equivalently_, when the _local approx. plane_ is flat (which also implies ``\nabla f(\mathbf{x}) =\mathbf{0}``)

```math
\Large
\nabla f(\mathbf{x}) = \mathbf{0}
```
"""

# ╔═╡ 6f8e97e8-1b75-4eb4-a233-76c72177d56e
md""" Move me: $(@bind angi Slider(-30:90; default = 45)); $(@bind angi2 Slider(-45:60; default = 30))"""

# ╔═╡ 202bd4a9-7e47-47ff-9e7d-04d03b2d92e5
md"""

## Partial derivative examples


```math
\Large 
f(\mathbf{x}) = c + b_1 x_1 + b_2 x_2 = c + \mathbf{b}^\top\mathbf{x}
```


* ##### ``\large \frac{\partial f}{\partial x_1}=?``
  * hint: treat ``x_2`` as a constant

\

* ##### ``\large \frac{\partial f}{\partial x_2}=?``
  * hint: treat ``x_1`` as a constant
"""

# ╔═╡ d05d2a0d-1a95-467e-8359-2dd651046c7a
md"""

## Partial derivative examples


```math
\Large 
f(\mathbf{x}) = c + b_1 x_1 + b_2 x_2 = c + \mathbf{b}^\top\mathbf{x}
```


* ##### ``\large \frac{\partial f}{\partial x_1}= b_1``
\

* ##### ``\large \frac{\partial f}{\partial x_2}=?``
"""

# ╔═╡ d4ce1024-bb1e-46a6-af83-0e00aac59ba3
md"""

## Partial derivative examples


```math
\Large 
f(\mathbf{x}) = c + b_1 x_1 + b_2 x_2 = c + \mathbf{b}^\top\mathbf{x}
```


* ##### ``\large \frac{\partial f}{\partial x_1}= b_1``
\

* ##### ``\large \frac{\partial f}{\partial x_2}=b_2``

#### Therefore, ``\nabla f(\mathbf{x}) = \mathbf{b}``
"""

# ╔═╡ ba6f08d0-6504-46b1-91ed-32a939c6676f
md"""

## Gradient of ``f(\mathbf{x}) = \mathbf{b}^\top \mathbf{x} + c``



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

# ╔═╡ 18b5f0f8-0af8-49a9-8e4d-a665a64ab3dc
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

# ╔═╡ 2e5c777e-cf27-4ee3-a9be-47b3e5ace5c7
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

# ╔═╡ fd743eb9-8e07-4db8-a391-d6184ca686e3
md"""

## More example/exercise


```math
\Large 
f(\mathbf{x}) = x_1^2\cdot  x_2
```


* ##### ``\large \frac{\partial f}{\partial x_1}=2 x_2 x_1``
\

* ##### ``\large \frac{\partial f}{\partial x_2}=x_1^2``

#### The gradient is 

$$\nabla f = \begin{bmatrix}2x_2x_1 \\ x_1^2 \end{bmatrix}$$
"""

# ╔═╡ c2522516-deb1-425c-b4dc-d9de117203d5
md"""



## Key fact about _gradient_ : important!


!!! important ""
	### ``\nabla f(\mathbf{x})``: points to the *greatest ascent direction* 
	* #### _locally_ at ``\mathbf{x}``



"""

# ╔═╡ 3a2d6f1f-c762-45b1-8e31-8266d42d1703
html"""<center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/3d-gradient-cos.svg/2560px-3d-gradient-cos.svg.png" width = "350"/></center>"""

# ╔═╡ 8a0a45c7-485e-4a64-895a-d3bb63af29c9
md"""

## Gradient example: ``f(\mathbf{x}) = \mathbf{x}^\top \mathbf{x}``


```math
\Large 
f(\mathbf{x}) =\mathbf{x}^\top \mathbf{x} = x_1^2 + x_2^2
```

##### The gradient is 

$$
\large
\nabla f(\mathbf{x}) = \begin{bmatrix}\frac{\partial f(\mathbf{x})}{\partial x_1}\\ \frac{\partial f(\mathbf{x})}{\partial x_2}\end{bmatrix} = \begin{bmatrix} 2 x_1\\ 2x_2\end{bmatrix} = 2 \mathbf{x}$$


* the gradients ``\nabla f(\mathbf{x})`` point outwardly (the greatest ascend direction)

* the gradient vanishes when ``\mathbf{x} \rightarrow \mathbf{0}``
"""

# ╔═╡ 561d026d-b349-433e-b865-21e55f7ffd77
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

# ╔═╡ 36454a35-67a7-4c94-9a5b-bec1447bde4b
md"""

## To further test max/min 
#### -- we need quadratic approximation
\
 

##### If ``f: \mathbb R^n \rightarrow \mathbb R`` is twice differentiable at ``\mathbf{x}_0``, then
> 
> ```math
> \Large
> \begin{align}
> f(\mathbf{x}) &\approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top(\mathbf{x}- \mathbf{x}_0) + \\
> &\;\;\;\;\;\;\boxed{\frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H}(\mathbf{x}_0)(\mathbf{x}-\mathbf{x}_0)}
> \end{align}
> ```

* ``\mathbf{H}(\mathbf{x}_0)`` is a square matrix called _Hessian_ matrix, serving the same purpose of ``f''(x_0)``
* ##### the max/min is determined by ``\mathbf{H}``: the quadratic coefficient
"""

# ╔═╡ de61d887-5c67-43f5-99e8-182934e8e464
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

# ╔═╡ e6994df8-e1d4-4efa-a7a1-0e4750394b9e
begin
	f_demo(w₁, w₂) = 1/4 * (w₁^4 + w₂^4) - 1/3 *(w₁^3 + w₂^3) - w₁^2 - w₂^2 + 4
	f_demo(w::Vector{T}) where T <: Real = f_demo(w...)
	∇f_demo(w₁, w₂) = [w₁^3 - w₁^2 - 2 * w₁, w₂^3 - w₂^2 - 2 * w₂]
	∇f_demo(w::Vector{T}) where T <: Real = ∇f_demo(w...)
end;

# ╔═╡ 32c452cb-b4d5-488f-8996-14f53dcc8ca4
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

# ╔═╡ cbc0e2f6-5417-403c-b348-ebb7deb9126e
let
	gr()
	f = f_demo
	x1_ = range(-2., stop =3.0, length=60)
	x2_ = range(-2., stop =3.0, length=60)
	p1_ = plot(x1_, x2_, (a, b) -> f([a, b]), st=:surface, xlabel =L"x_1", ylabel=L"x_2", zlabel=L"f(\mathbf{x})", colorbar=false, color=:jet, title="", camera=(angi, angi2))
	len = 0.7
	for (li, l0) in enumerate(l0s)
		tf = taylorApprox(f, l0, 1)
		plot!(range(l0[1] - len, l0[1]+len, 5), range(l0[2] - len, l0[2]+len, 5), (a, b) -> tf([a, b]), st=:surface, alpha=0.3,  display_option=Plots.GR.OPTION_Z_SHADED_MESH)
		label = li == 1 ? L"\mathbf{x};\; \texttt{s.t.} \nabla f(\mathbf{x}) = 0" : ""
		scatter!([l0[1]], [l0[2]], [f(l0)], ms=3, label=label)
	end
	p1_
end

# ╔═╡ a019b313-d46a-42e5-90d5-fbd239a1546a
let
	plotly()
	f = f_demo
	l0s_min = [[2, -1], [2, 2], [-1, 2], [-1, -1]]
	l0s_max = [[0, 0]]
	l0s_saddle = [[2, 0], [0, -1], [0, 2], [-1, 0]]
	x1_ = range(-2., stop =3.0, length=60)
	x2_ = range(-2., stop =3.0, length=60)
	p1_ = plot(x1_, x2_, (a, b) -> f([a, b]), st=:surface, xlabel ="x1", ylabel="x2", zlabel="f(x)", colorbar=false, color=:jet, title="Minimums", alpha=0.8)
	len = 1.2

	for (li, l0) in enumerate(l0s_min)
			tf = taylorApprox(f, l0, 1)
			plot!(range(l0[1] - len, l0[1]+len, 5), range(l0[2] - len, l0[2]+len, 5), (a, b) -> tf([a, b]), st=:surface, alpha=0.4, display_option=Plots.GR.OPTION_Z_SHADED_MESH)
			label = li == 1 ? "x_min" : ""
			scatter!([l0[1]], [l0[2]], [f(l0)], ms=2, label="")

				tf2 = taylorApprox(f, l0, 2)
				surface!(p1_, range(l0[1] - 1., l0[1]+1., 30), range(l0[2] - 1., l0[2]+1., 30), (x, y) -> tf2([x, y]), label="", alpha=0.8, c=:gray, zlim =(-2,9))
	end

	p1_
end

# ╔═╡ 3edda8b6-f7ef-48ea-b24a-03d832cee646
let
	plotly()
	f = f_demo
	l0s_min = [[2, -1], [2, 2], [-1, 2], [-1, -1]]
	l0s_max = [[0, 0]]
	l0s_saddle = [[2, 0], [0, -1], [0, 2], [-1, 0]]
	x1_ = range(-2., stop =3.0, length=60)
	x2_ = range(-2., stop =3.0, length=60)
	p1_ = plot(x1_, x2_, (a, b) -> f([a, b]), st=:surface, xlabel ="x1", ylabel="x2", zlabel="f(x)", colorbar=false, color=:jet, title="Maximum", alpha=0.8)
	len = 1.2
	for (li, l0) in enumerate(l0s_max)
			tf = taylorApprox(f, l0, 1)	
			plot!(range(l0[1] - len, l0[1]+len, 5), range(l0[2] - len, l0[2]+len, 5), (a, b) -> tf([a, b]), st=:surface, alpha=0.7, display_option=Plots.GR.OPTION_Z_SHADED_MESH)
			label = li == 1 ? "x_max" : ""
			scatter!([l0[1]], [l0[2]], [f(l0)], ms=2, label="")

				tf2 = taylorApprox(f, l0, 2)
				surface!(p1_, range(l0[1] - 1.2, l0[1]+1.2, 30), range(l0[2] - 1.2, l0[2]+1.2, 30), (x, y) -> tf2([x, y]), label="", alpha=0.8, c=:pink,zlim =(-2,9))

		end
	p1_
end

# ╔═╡ 124e68a7-c6c7-4d7e-b3b1-0f3e293e4f22
let
	plotly()
	f = f_demo
	l0s_min = [[2, -1], [2, 2], [-1, 2], [-1, -1]]
	l0s_max = [[0, 0]]
	l0s_saddle = [[2, 0], [0, -1], [0, 2], [-1, 0]]
	x1_ = range(-2., stop =3.0, length=60)
	x2_ = range(-2., stop =3.0, length=60)
	p1_ = plot(x1_, x2_, (a, b) -> f([a, b]), st=:surface, xlabel ="x1", ylabel="x2", zlabel="f(x)", colorbar=false, color=:jet, title="Saddle points", alpha=0.75)
	len = 1.2
	

	
	for (li, l0) in enumerate(l0s_saddle)
		tf = taylorApprox(f, l0, 1)
		plot!(range(l0[1] - len, l0[1]+len, 5), range(l0[2] - len, l0[2]+len, 5), (a, b) -> tf([a, b]), st=:surface, alpha=0.8, display_option=Plots.GR.OPTION_Z_SHADED_MESH)
		label = li == 1 ? "x_saddle" : ""
		scatter!([l0[1]], [l0[2]], [f(l0)], ms=2, label="")

			tf2 = taylorApprox(f, l0, 2)
			surface!(p1_, range(l0[1] - 1.2, l0[1]+1.2, 30), range(l0[2] - 1.2, l0[2]+1.2, 30), (x, y) -> tf2([x, y]), label="", alpha=1.0, c=:pink, zlim =(-2,9))

	end

	p1_
end

# ╔═╡ f3030d80-2db0-4385-ab90-d380196cd915
md"""

## Multivariate quadratic functions

#### A (twice-) differentiable function 
```math
\large
\begin{align}
 f(\mathbf{x}) &\approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top(\mathbf{x}- \mathbf{x}_0) + \frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H}(\mathbf{x}_0)(\mathbf{x}-\mathbf{x}_0)
 \end{align}
```


* ##### the max/min is determined by ``\mathbf{H}``: the quadratic coefficient
\
\

#### We need to investigate multivariate quadratics further 

* ##### they are _more flexible_ than its univariate counterpart

* ##### but still simple enough to grasp
"""

# ╔═╡ 60e22889-c4a9-4adc-9ff6-ef461b760c5b
md"""

## Quadratic function 


```math
\LARGE
f(x)=ax^2\;\; \Rightarrow \;\;f(\mathbf{x}) = \mathbf{x}^\top\mathbf{A}\mathbf{x}
```

\

* ##### multivariate quadratic function is direct generalisation 


* ##### note that ``\mathbf{A}`` serves the same purpose as _Hessian_ ``\mathbf{H}``


* ##### we ignore the linear and constant terms 
  * they do not affect the shape but just re-locates and elevates the function



"""

# ╔═╡ a016c888-3679-48df-9629-8ee38b28c59d
md"""

## Example

```math
\Large
\mathbf{x}^\top \mathbf{A}\mathbf{x} = \begin{align}
&\;\;\;\;\begin{bmatrix}x_1 & x_2\end{bmatrix}_{1\times 2}\begin{bmatrix}A_{11} & A_{12} \\ A_{21} & A_{22}\end{bmatrix}_{2\times 2}\begin{bmatrix}x_1 \\ x_2 \end{bmatrix}_{2\times 1}
\end{align}
```


* ##### the result is a scalar!

* ##### the result is ``\sum_i\sum_j A_{ij} x_i x_j``

"""

# ╔═╡ e0ba2117-9ebc-4c57-b34e-e74914369f8b
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

# ╔═╡ a0e4ceba-c028-4940-84be-d1c346f93078
md"""
## Gradient of quadratic


#### What is the gradient $\nabla f(\mathbf{x})$ ? 
```math
\LARGE
f(\mathbf{x}) = \mathbf{x}^\top\mathbf{A}\mathbf{x}
```


* ##### assume $\mathbf{A}$ is symmetric, i.e. $A_{12} = A_{21}$

"""

# ╔═╡ 5bf79563-bbf2-41cb-b8b0-9bd0e6558927
Foldable("Details", md"""

#### Since ``\mathbf{A}`` is symmetric, ``A_{21} = A_{12}``, 


$$\large \mathbf{A} = \begin{bmatrix}A_{11} & A_{12}  \\ A_{12} & A_{22}\end{bmatrix}$$


#### the quadratic form is

$\Large f(\mathbf{x}) = A_{11}x_1^2 + 2A_{12}x_1x_2 + A_{22}x_2^2$


$$\Large \begin{align}\nabla f(\mathbf{x}) &= \begin{bmatrix}2A_{11}x_1+ 2 A_{12}x_2 \\ 2A_{12}x_1 + 2A_{22}x_2 \end{bmatrix} = 2 \begin{bmatrix}A_{11} & A_{12} \\ A_{12} & A_{22} \end{bmatrix}  \begin{bmatrix} x_1 \\ x_2\end{bmatrix} \\

&= 2\mathbf{Ax}\end{align}$$
""")

# ╔═╡ 32e1940d-d0c5-4775-beb8-597b62437d9b
md"""

## More generally ``f(\mathbf{x}) = \mathbf{x}^\top \mathbf{Ax} + \mathbf{b}^\top \mathbf{x} + c``


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

# ╔═╡ 7cfb26a6-8d71-4349-a9ec-f6c18b3f4c80
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

# ╔═╡ 0b71cdd1-8697-4a5a-91e0-7bbfe5cb7abf
md"""

## Univariate _quadratic_ function 


```math
\Large
f(x) = ax^2 
```
* ##### it only has three possible cases
"""

# ╔═╡ 8e76c380-3021-4515-be65-a99d7a36bbc8
begin
	gr()
	plt1 = plot(x -> x^2, xlim =[-3,3], ylim =[-1, 3], title="",c=1, framestyle=:zerolines, lw=3, label="", size=(200, 250), titlefontsize=20)
	plt2 = plot(x -> -x^2, xlim =[-3,3], ylim =[-3, 1], title="",c=2, framestyle=:zerolines, lw=3, label="", size=(200, 250), titlefontsize=20)
	plt3 = plot(x -> 0, xlim =[-3,3], ylim =[-1, 1], title="",c=3, framestyle=:zerolines, lw=3, label="", size=(200, 250), titlefontsize=20)

	# plot(plt1, plt2, plt3, layout = (1, 3),ratio=1.5, framestyle=:zerolines, lw=5, size=(800, 400), titlefontsize=25)
end;

# ╔═╡ f8dbbd5d-95f1-4baf-97f4-bc2ae19a50b6
ThreeColumn(md"""
##### Positive case ``a>0``
```math
\large
ax^2 >0; \text{ for all } x\in \mathbb{R}
```

$(begin
	gr()
	plt1
end)

""", md"""
##### ``\;\;`` Negative case ``a<0``
```math
\large
ax^2 <0; \text{ for all } x\in \mathbb{R}
```
$(begin
	gr()
	plt2
end)

""", md"""
#####  ``\;``Degenerate case ``a=0``
```math
\large
ax^2 =0; \text{ for all } x\in \mathbb{R}
```

$(begin
	gr()
	plt3
end)
""")

# ╔═╡ d2ce46e1-7f16-4ea2-a1a8-9cc34df59d14
md"""

## Multivariate _quadratic_ function 


```math
\Large
f(x) = \mathbf{x}^\top\mathbf{Ax}
```

* ##### it actually has 5 cases 

"""

# ╔═╡ 74d2671d-d3f6-4648-b9fd-7684d93af76c
qform(x; A=A_, b=bv_, c=c_) = x' * A * x + b'* x + c # quadratic form function

# ╔═╡ df28bf21-f37d-46f4-b1f5-42dbb3c67e5f
md"""

## Positive definite 

> $\text{\Large positive definite: }\;\;\Large \mathbf{x}^\top\mathbf{A}\mathbf{x} > 0\; \text{for all } \mathbf{x\in \mathbb{R}}^n$
> * *e.g.*
> ```math 
> \large
> \mathbf{A} = \begin{bmatrix}1 & 0 \\0 & 1\end{bmatrix}
> ```

> #### *Interpretation*:  _for all directions ``\mathbf{x}\in \mathbb{R}^n``, the function *faces UP*_
> * ##### then ``f`` has a *maximum*
"""

# ╔═╡ 802c12c2-7c5c-4e15-8c12-a173c33e3f31
md"""
## Negative definite

> $\text{\large Negative definite: }\large \mathbf{x}^\top\mathbf{A}\mathbf{x} < 0\; \text{for all } \mathbf{x}\in \mathbb{R}^n$
> * *e.g.*
> ```math
> \mathbf{A} = \begin{bmatrix}-1 & 0 \\0 & -1\end{bmatrix}
> ```

> #### *Interpretation*:  _for all directions ``\mathbf{x}\in \mathbb{R}^n``, function *faces DOWN*_
> * ##### then ``f`` has a *maximum*



"""

# ╔═╡ 5d1e3cc8-daa9-4454-9e6a-18f4339af281
md"""

## Indefinite case -- saddle point
"""

# ╔═╡ f5c2a280-30c8-4992-a404-c5a711cfa461
md"""

## Optimise quadratic function


#### Quadratics are easy to optimise
* ##### there are closed-form solutions


##### _For univariate quadratic_,  when ``a\neq 0``

```math
\Large 
f(x) = ax^2 + bx + c
```

* ##### _the max/min is_

```math
\large
f'(x) = 2ax + b = 0 \Rightarrow \boxed{\Large x = -\frac{1}{2}a^{-1}b}
```


##### _For multivariate quadratic_,  when ``\mathbf{A}\succ 0`` (positive definite) or ``\mathbf{A} \prec 0``(negative definite)

```math
\Large 
f(\mathbf{x}) = \mathbf{x}^\top \mathbf{Ax} + \mathbf{b}^\top\mathbf{x} + c
```

* ##### _the max/min is_

```math
\large
\nabla f(\mathbf{x}) = 2\mathbf{Ax} + \mathbf{b} = \mathbf{0} \Rightarrow \boxed{\Large \mathbf{x} = -\frac{1}{2} \mathbf{A}^{-1}\mathbf{b}}
```

* ##### when ``\mathbf{A}`` is semi-definite (``\mathbf{A}^{-1}`` does not exist), it implies there is no unique max or min
"""

# ╔═╡ 514a5bc2-214e-4a65-a24a-df4e89e8d385
As = [Matrix(I, 2, 2) => "Positive definite", -Matrix(I,2,2) => "Negative definite", [1 0; 0 -1] => "Indefine", [1 0; 0 0] => "Positive semi-definite", -[1 0; 0 0] => "Negative semi-definite"];

# ╔═╡ 1945329b-7449-4033-ab6f-fec954274c02
# @bind Ai Select(1:5)

# ╔═╡ fea055fe-1fe4-4d39-835b-32462337b61a
# let
# 	gr()
	
# 	A_ = As[Ai]
# 	plot(-5:0.5:5, -5:0.5:5, (x, y) -> qform([x, y],A_.first), st=:surface, colorbar=false, c=:coolwarm, alpha=0.6, title = A_.second)
# 	ys = -5:.5:5
# 	xs = zeros(length(ys))
# 	A = A_.first
# 	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
# 	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
# 	path3d!(xs, ys,zs, lw=5, label="", c=1)
# 	xs = -5:.5:5
# 	ys = zeros(length(xs))
# 	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
# 	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
# 	path3d!(xs, ys,zs, lw=3, label="", c=2)
# end

# ╔═╡ 9e9c94d8-e550-4ebf-9285-ffc56edc8997
qform(x, A) = x' * A * x;

# ╔═╡ f400334e-9325-49a0-8e0a-b3171a188d5d
let
	gr()
	plot(-5:0.25:5, -5:0.25:5, (x,y) -> qform([x,y]; A= -Matrix(I,2,2), b=zeros(2), c=0), st=:surface, colorbar=false, color=:coolwarm, title="A is negative definite; maximum", display_option=Plots.GR.OPTION_SHADED_MESH, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"f", framestyle=:semi)
end

# ╔═╡ eb7327b2-0c0d-411d-bddd-0fdb38060932
TwoColumn(let 
	gr()
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

# ╔═╡ 8be4395d-e4f3-4d5e-aa22-9b2a82c33292
plt_pd = let
  	Amat = As[1]
	A = Amat.first
	a_text = Amat.second
	plot(-5:0.5:5, -5:0.5:5, (x, y) -> qform([x, y],A), st=:surface, colorbar=false, c=:coolwarm, alpha=0.6, title = "", size=(230,250), framestyle=:zerolines)
	ys = -5:.5:5
	xs = zeros(length(ys))
	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
	path3d!(xs, ys,zs, lw=5, label="", c=3)
	xs = -5:.5:5
	ys = zeros(length(xs))
	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
	path3d!(xs, ys,zs, lw=3, label="", c=2)

end;

# ╔═╡ 7544c835-b587-499c-b761-a54f5005e07c
plt_nd = let
  	Amat = As[2]
	A = Amat.first
	a_text = Amat.second
	plot(-5:0.5:5, -5:0.5:5, (x, y) -> qform([x, y],A), st=:surface, colorbar=false, c=:coolwarm, alpha=0.6, title = "", size=(230,250), framestyle=:zerolines)
	ys = -5:.5:5
	xs = zeros(length(ys))
	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
	path3d!(xs, ys,zs, lw=5, label="", c=3)
	xs = -5:.5:5
	ys = zeros(length(xs))
	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
	path3d!(xs, ys,zs, lw=3, label="", c=2)

end;

# ╔═╡ 5f7d79f1-8763-4901-87a8-1534cd78e3b7
plt_id = let
  	Amat = As[3]
	A = Amat.first
	a_text = Amat.second
	plot(-5:0.5:5, -5:0.5:5, (x, y) -> qform([x, y],A), st=:surface, colorbar=false, c=:coolwarm, alpha=0.6, title = "", size=(230,250), framestyle=:zerolines)
	ys = -5:.5:5
	xs = zeros(length(ys))
	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
	path3d!(xs, ys,zs, lw=5, label="", c=3)
	xs = -5:.5:5
	ys = zeros(length(xs))
	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
	path3d!(xs, ys,zs, lw=3, label="", c=2)

end;

# ╔═╡ e982bd9d-1387-4131-92a7-43d69b84c685
ThreeColumn(md"""
##### Positive definite ``\mathbf{A}\succ 0``
\

```math
\mathbf{x}^\top\mathbf{Ax} >0;\text{ for all } \mathbf{x}\in \mathbb{R}^n
```



$(begin
	gr()
	plt_pd
end)

""", md"""
#####  Negative definite ``\mathbf{A}\prec 0``
\

```math
\mathbf{x}^\top\mathbf{Ax} <0;\text{ for all } \mathbf{x}\in \mathbb{R}^n
```
$(begin
	gr()
	plt_nd
end)

""", md"""
#####  ``\;\;\;\;\;\;\;\;``In-definite ``\mathbf{A}``
```math
\begin{align}
\mathbf{x}^\top\mathbf{Ax} <0;\text{ for some } \mathbf{x}\in \mathbb{R}^n \\
\mathbf{x}^\top\mathbf{Ax} >0;\text{ for some } \mathbf{x}\in \mathbb{R}^n
\end{align}
```

$(begin
	gr()
	plt_id
end)
""")

# ╔═╡ 34051bb5-d66f-4739-a54e-39a6dba5398b
plt_spd = let
  	Amat = As[4]
	A = Amat.first
	a_text = Amat.second
	plot(-5:0.5:5, -5:0.5:5, (x, y) -> qform([x, y],A), st=:surface, colorbar=false, c=:coolwarm, alpha=0.6, title = "", size=(230,250), framestyle=:zerolines)
	ys = -5:.5:5
	xs = zeros(length(ys))
	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
	path3d!(xs, ys,zs, lw=5, label="", c=3)
	xs = -5:.5:5
	ys = zeros(length(xs))
	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
	path3d!(xs, ys,zs, lw=3, label="", c=2)

end;

# ╔═╡ 490c2a7d-e8c6-4838-ab2c-b5c049757b43
plt_snd = let
  	Amat = As[5]
	A = Amat.first
	a_text = Amat.second
	plot(-5:0.5:5, -5:0.5:5, (x, y) -> qform([x, y],A), st=:surface, colorbar=false, c=:coolwarm, alpha=0.6, title = "", size=(230,250), framestyle=:zerolines)
	ys = -5:.5:5
	xs = zeros(length(ys))
	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
	path3d!(xs, ys,zs, lw=5, label="", c=3)
	xs = -5:.5:5
	ys = zeros(length(xs))
	zs = [dot([xs[i], ys[i]], A, [xs[i], ys[i]]) for i in 1:length(ys)]
	zs2 = [dot([ys[i], xs[i]], A, [ys[i], xs[i]]) for i in 1:length(ys)]
	path3d!(xs, ys,zs, lw=3, label="", c=2)

end;

# ╔═╡ 093db468-8c99-4149-a30f-a296a06d4223
TwoColumn(md"""
##### Positive semi-definite ``\mathbf{A}\succeq 0``
\

```math
\mathbf{x}^\top\mathbf{Ax} \geq 0;\text{ for all } \mathbf{x}\in \mathbb{R}^n
```



$(begin
	gr()
	plot(plt_spd, size=(300,250))
end)

""", md"""
#####  Negative semi-definite ``\mathbf{A}\preceq 0``
\

```math
\mathbf{x}^\top\mathbf{Ax} \leq 0;\text{ for all } \mathbf{x}\in \mathbb{R}^n
```
$(begin
	gr()
	plot(plt_snd, size=(300,250))
end)

""")

# ╔═╡ 74869b94-1612-4b2c-bb99-af67da45e016
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

# ╔═╡ fd372fc1-0ea1-4a13-9e6e-02c7291e3e79
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

# ╔═╡ 4680d25c-158e-4276-9aeb-b807bb9e4696
let
	gr()
	f(x) = qform(x; A=Matrix(I,2,2), b=zeros(2), c=0) +10
	plt = plot(-5:0.5:5, -5:0.5:5, (x,y) -> f([x,y]), st=:surface, colorbar=false, color=:gray,alpha=0.3, xlim=[-5, 5] , ylim=[-5, 5], zlim =[-2, 55], title=L"\mathbf{A}" * "is positive definite; minimum")
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
		path3d!(xs, ys, zs, lw=1.5, label="", c=ind)
	end
	plt
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
ForwardDiff = "~0.10.36"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
Latexify = "~0.16.1"
Plots = "~1.40.0"
PlutoTeachingTools = "~0.2.14"
PlutoUI = "~0.7.55"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "bb35e546c0af58912c89b8a019b56e9fc98e4fd7"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

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

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+4"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

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

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "f36e5e8fdffcb5646ea5da81495a5a7566005127"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.3"

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

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

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

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "9bf00ba4c45867c86251a7fd4cb646dcbeb41bf0"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.12"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "36d5430819123553bf31dfdceb3653ca7d9e62d7"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.12+0"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

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
git-tree-sha1 = "a729439c18f7112cbbd9fcdc1771ecc7f071df6a"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.39"

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

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "fe891aea7ccd23897520db7f16931212454e277e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.1"

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
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

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

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

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

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

    [deps.Revise.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"

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

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

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

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

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

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "622cf78670d067c738667aaa96c553430b65e269"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+0"

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
# ╟─6bdbcbc6-2203-46f9-b633-e742a6884a11
# ╟─45f4c200-b0bb-4d15-a6c9-b519a25afcba
# ╟─23ad8369-85dc-49cd-baba-cff6c6979e14
# ╟─522d01e5-a19c-402f-89f3-b6e6fb98dad6
# ╟─fd748bc2-305a-4b93-ba83-2b825f901516
# ╟─b1b5bebc-79ec-432b-809f-f37c0f2065a1
# ╟─e5709c4a-c838-4888-ac52-645b3e6230fc
# ╟─c52dd91c-9ce4-4d2e-970e-57ee3c2684f5
# ╟─0dd38584-b6b0-463f-8e21-ef625fb97277
# ╟─c45d6b24-026e-4936-a12e-f3f1ec3794e3
# ╟─f8e1cdae-4929-4e8a-a58f-8e9637eea406
# ╟─f78d8e73-8e66-4432-93ad-61e064f437d3
# ╟─904e6749-2832-48ff-b4e3-a546901a6fda
# ╟─576ef6a5-8507-452a-9e72-9a48e9ad2842
# ╟─f276290b-da38-4267-ab17-c49a05adfa1a
# ╟─1b19c622-7208-4711-887a-62335934c0d5
# ╟─c2423b8f-7cd5-48a9-9a27-ff11060e1dd2
# ╟─b5fc7959-87e6-4f83-aeab-8f5818a45b90
# ╟─77b81e32-3c64-4ad5-b16d-66bbbf534f02
# ╟─e2b926ab-49ba-4862-9b0f-b396441f708d
# ╟─14bfdb25-2015-44cc-991b-536b221d388e
# ╟─ece30c5c-b579-4968-8bd3-137e46605218
# ╟─ae484ad1-bbe6-4892-9932-27f105224877
# ╟─d44a2ce7-bf3e-4da5-8c53-a241524a891a
# ╟─47f1ce87-8e8c-4ecc-b9e0-e2338189f2e6
# ╟─954be22e-e3d1-42eb-a7e6-712fb1e3e0ed
# ╟─ae4072fb-5686-470b-afb5-7fd537df6603
# ╟─eee41566-4d5d-43c8-9899-13f3cf9e8ba0
# ╟─bb06c5ee-d08c-4343-869c-6807d3a8ac21
# ╟─89c876f2-71bf-471d-ae75-6d3291583995
# ╟─d4e093f7-edda-4a29-bed5-fb908c764ba0
# ╟─7cfadac5-2041-4af7-abed-920120701f90
# ╟─33e6e2eb-6ff5-4a92-ad28-f9f57e3ccb39
# ╟─16aa467e-dbb5-4fdd-be97-617590a7f3e8
# ╟─ac42b3a9-6744-49ae-b55f-48fb184bedaa
# ╟─cb58cc20-172a-4c2c-81f3-e3ccd7bcd8ab
# ╟─49d64d9d-d077-4b7c-8a90-4c1d7ba7cdc6
# ╟─f905afd1-bf33-4d9a-8b73-c2dfa79995ff
# ╟─14149895-394f-4c58-a135-12366f2a873e
# ╟─7d17349b-1e3a-4011-954d-483c1493078f
# ╟─2acd37fc-3b78-4c33-9ec9-c405f1527aca
# ╟─47d3302c-ad34-4a82-8c1c-46f74ce10440
# ╟─a21182db-3f9d-4dca-8cdd-08dc38453231
# ╟─32c452cb-b4d5-488f-8996-14f53dcc8ca4
# ╟─0932bac0-921f-40d2-bfcf-7af2d8f1a268
# ╟─3240adaf-f984-4caf-95a9-742a40897f46
# ╟─7f76686c-21dc-414e-9526-f12e98e57bb2
# ╟─fb08d3fa-35a5-4afa-88d6-7587c1bfa339
# ╟─b2113f74-c2ad-46c1-8092-fd8190649504
# ╟─fd372fc1-0ea1-4a13-9e6e-02c7291e3e79
# ╟─0452ca78-9ed3-4355-afea-e3819ef79afd
# ╟─5d68596e-856f-49b8-a376-f4449bd949cd
# ╟─cbc0e2f6-5417-403c-b348-ebb7deb9126e
# ╟─6f8e97e8-1b75-4eb4-a233-76c72177d56e
# ╟─202bd4a9-7e47-47ff-9e7d-04d03b2d92e5
# ╟─d05d2a0d-1a95-467e-8359-2dd651046c7a
# ╟─d4ce1024-bb1e-46a6-af83-0e00aac59ba3
# ╟─ba6f08d0-6504-46b1-91ed-32a939c6676f
# ╟─18b5f0f8-0af8-49a9-8e4d-a665a64ab3dc
# ╟─2e5c777e-cf27-4ee3-a9be-47b3e5ace5c7
# ╟─fd743eb9-8e07-4db8-a391-d6184ca686e3
# ╟─c2522516-deb1-425c-b4dc-d9de117203d5
# ╟─3a2d6f1f-c762-45b1-8e31-8266d42d1703
# ╟─8a0a45c7-485e-4a64-895a-d3bb63af29c9
# ╟─561d026d-b349-433e-b865-21e55f7ffd77
# ╟─36454a35-67a7-4c94-9a5b-bec1447bde4b
# ╟─a019b313-d46a-42e5-90d5-fbd239a1546a
# ╟─3edda8b6-f7ef-48ea-b24a-03d832cee646
# ╟─124e68a7-c6c7-4d7e-b3b1-0f3e293e4f22
# ╟─de61d887-5c67-43f5-99e8-182934e8e464
# ╟─e6994df8-e1d4-4efa-a7a1-0e4750394b9e
# ╟─f3030d80-2db0-4385-ab90-d380196cd915
# ╟─60e22889-c4a9-4adc-9ff6-ef461b760c5b
# ╟─a016c888-3679-48df-9629-8ee38b28c59d
# ╟─e0ba2117-9ebc-4c57-b34e-e74914369f8b
# ╟─a0e4ceba-c028-4940-84be-d1c346f93078
# ╟─5bf79563-bbf2-41cb-b8b0-9bd0e6558927
# ╟─32e1940d-d0c5-4775-beb8-597b62437d9b
# ╟─7cfb26a6-8d71-4349-a9ec-f6c18b3f4c80
# ╟─0b71cdd1-8697-4a5a-91e0-7bbfe5cb7abf
# ╟─f8dbbd5d-95f1-4baf-97f4-bc2ae19a50b6
# ╟─8e76c380-3021-4515-be65-a99d7a36bbc8
# ╟─d2ce46e1-7f16-4ea2-a1a8-9cc34df59d14
# ╟─e982bd9d-1387-4131-92a7-43d69b84c685
# ╟─093db468-8c99-4149-a30f-a296a06d4223
# ╟─74d2671d-d3f6-4648-b9fd-7684d93af76c
# ╟─df28bf21-f37d-46f4-b1f5-42dbb3c67e5f
# ╟─4680d25c-158e-4276-9aeb-b807bb9e4696
# ╟─802c12c2-7c5c-4e15-8c12-a173c33e3f31
# ╟─f400334e-9325-49a0-8e0a-b3171a188d5d
# ╟─5d1e3cc8-daa9-4454-9e6a-18f4339af281
# ╟─eb7327b2-0c0d-411d-bddd-0fdb38060932
# ╟─f5c2a280-30c8-4992-a404-c5a711cfa461
# ╟─514a5bc2-214e-4a65-a24a-df4e89e8d385
# ╟─1945329b-7449-4033-ab6f-fec954274c02
# ╟─fea055fe-1fe4-4d39-835b-32462337b61a
# ╟─9e9c94d8-e550-4ebf-9285-ffc56edc8997
# ╟─8be4395d-e4f3-4d5e-aa22-9b2a82c33292
# ╟─7544c835-b587-499c-b761-a54f5005e07c
# ╟─5f7d79f1-8763-4901-87a8-1534cd78e3b7
# ╟─34051bb5-d66f-4739-a54e-39a6dba5398b
# ╟─490c2a7d-e8c6-4838-ab2c-b5c049757b43
# ╟─74869b94-1612-4b2c-bb99-af67da45e016
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
