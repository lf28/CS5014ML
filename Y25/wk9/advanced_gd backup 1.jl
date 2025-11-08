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

# ╔═╡ a533cf6c-22d8-4659-813a-f1e12984326b
using Optimisers

# ╔═╡ d400c959-0a96-49bf-9239-de12d5e39de1
using Zygote

# ╔═╡ 30e2c8a3-0bc1-42d3-bd27-d7e493f4c164
using Logging

# ╔═╡ e1e8c5ca-874c-4a30-9211-8c29ee226c3b
Logging.disable_logging(Logging.Info) ; # or e.g. Logging.Info

# ╔═╡ 3e2e1ea8-3a7d-462f-ac38-43a087907a14
TableOfContents()

# ╔═╡ 52dcd4b9-9ef7-4128-a81d-d7e454cae9d6
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ 19ebad08-8461-46fc-90bf-fcb1fa30d833
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

# ╔═╡ 7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
ChooseDisplayMode()

# ╔═╡ bc96a33d-9011-41ec-a19e-d472cbaafb70
md"""

# CS5914 Machine Learning Algorithms


#### Advanced gradient descent algorithms
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 7091d2cf-9237-45b2-b609-f442cd1cdba5
md"""

## Topics to cover
	
"""

# ╔═╡ 0a7f37e1-51bc-427d-a947-31a6be5b765e
aside((md"""$(@bind next1 Button("next")) 
$(@bind init1 Button("init"))
	"""))

# ╔═╡ 595a5ef3-4f54-4502-a943-ace4146efa31
begin
	init1
	next_idx = [0];
end;

# ╔═╡ a696c014-2070-4041-ada3-da79f50c9140
begin
	next1
	topics = ["Limitations of gradient descent", "Momentum", "Resilient propagation (RProp)", "RMSProp & Adam"]
	@htl "<ul>$([@htl("""<li>$b</li><br>""") for b in topics[1:min(next_idx[1], length(topics))]])</ul>"
end

# ╔═╡ bc1ee08d-9376-44d7-968c-5e114b09a5e0
let
	next1
	next_idx[1] += 1
end;

# ╔═╡ 84c68c61-7168-44c5-a606-b1b252377229
md"""

## Recap: Gradient descent 


Given objective function ``\ell(\mathbf{w})``, and  ``\mathbf{w} =\begin{bmatrix} w_1  \\  w_2 \\ \vdots \\  w_n \end{bmatrix}``



###### *Gradient descent* aims at optimising/minimising the objective function

```math
\boxed{\hat{\mathbf{w}}\leftarrow \arg\min_{\mathbf{w}} \ell(\mathbf{w})}
```



## Recap: Gradient descent 


Given objective function ``\ell(\mathbf{w})``, and  ``\mathbf{w} =\begin{bmatrix} w_1  \\  w_2 \\ \vdots \\  w_n \end{bmatrix}``



###### *Gradient descent* aims at optimising/minimising the objective function

```math
\boxed{\hat{\mathbf{w}}\leftarrow \arg\min_{\mathbf{w}} \ell(\mathbf{w})}
```

The algorithm starts with a random guess of ``\mathbf{w}^{(0)}`` and then iteratively update:

```math
\large
\mathbf{w}^{(t)} \leftarrow \mathbf{w}^{(t-1)} - γ\, \mathbf{g}^{(t-1)}
```

* ``\gamma > 0``: learning rate

To simplify the notation, ``\mathbf{g}`` denotes the gradient:

```math
\large
\mathbf{g} \triangleq \nabla\ell(\mathbf{w}) = \begin{bmatrix} \frac{\partial}{\partial w_1} \ell(\mathbf{w}) \\ \frac{\partial}{\partial w_2} \ell(\mathbf{w})  \\ \vdots \\ \frac{\partial}{\partial w_m} \ell(\mathbf{w}) \end{bmatrix}
```
"""

# ╔═╡ d32970e9-1ec5-4074-92a9-1989e86234f7
md"""
## An example

The objective function is a simple _quadratic_ function

```math
\ell(\mathbf{w}) = \mathbf{w}^\top \mathbf{A}\,\mathbf{w}
```

* where ``\mathbf{A} = \begin{bmatrix}.5 & 0 \\ 0 &   9.75\end{bmatrix}`` 

* the minimum is ``\hat{\mathbf{w}} = [0, 0]^\top``
* the starting point ``\mathbf{w}^{(0)} = [10, 1]^\top``
"""

# ╔═╡ 65c48015-fcfb-41c7-b3c7-79411b9721b9
md"""
## Example -- gradient descent

"""

# ╔═╡ fd636237-ebc5-4d8c-9703-f3c93adfda5b
md"""

###### The convergence is pretty slow when ``\gamma = 0.05``
"""

# ╔═╡ 59b80abc-e696-4870-9cf8-0aaafcbbd6e0
md"""

## Limitations of gradient descent (1) 

#### -- step length & zig-zag


"""

# ╔═╡ e5b02c45-9b49-4ceb-8f88-2a25229ff62b
md"""

###### The convergence is a bit better when ``\gamma = 0.1``
"""

# ╔═╡ 0af4a76f-578b-4195-826b-84cdf7b65afe
md"""

## Limitations of gradient descent (1) 

#### -- step length & zig-zag


"""

# ╔═╡ 791cb069-a6a9-4c33-a4fd-677b7a4ceb6f
md"""

###### The convergence is a bit better when ``\gamma = 0.1``
"""

# ╔═╡ 6071849b-08f7-4618-a3b5-1c6ba67c6396
md"""



###### _However,_ the *Zig-zag* behaviour is also very inefficient 
* _no convergence_ after 25 iterations
"""

# ╔═╡ b2b0ef75-7192-4d2c-9fcc-3bc4e49da8cc
md"""


## Aside: on-line sample mean algorithm


!!! question "Compute average on-line"
	Given a stream of data (and we do not know the final size in advance)
	```math
		\mathcal{D} = \{d_1, d_2, \ldots\},
	```
	How to compute the mean online?
	```math
		\mu^{(t)} = \frac{1}{t}\sum_{i=1}^t d_i
	```
	* for ``t \geq 1``
    * *e.g.* ``\mu^{(1)} = d_1``, ``\mu^{(2)} =\frac{1}{2}(d_1+d_2)``

##### Why _online_?

* we do not want to save the data ``\mathcal{D}`` in memory

* we always have the latest ``\mu^{(t)}`` available
"""

# ╔═╡ bebe066b-c212-402f-af15-6c329e42941e
md"""

## A recursive formula for ``\mu^{(t)}``

Given the mean of the first ``t-1`` data, ``\mu^{(t-1)}``:


```math
\large
\mu^{(t-1)} = \frac{1}{t-1} \sum_{i=1}^{t-1} d_i
```

Then

```math
\large
(t-1) \cdot \mu^{(t-1)} = \sum_{i=1}^{t-1} d_i
```



"""

# ╔═╡ 749db615-65c5-41fb-9232-f0c936f2c811
md"""

## A recursive formula for ``\mu^{(t)}``

Given the mean of the first ``t-1`` data, ``\mu^{(t-1)}``:


```math
\large
\mu^{(t-1)} = \frac{1}{t-1} \sum_{i=1}^{t-1} d_i
```

Then

```math
\large
(t-1) \cdot \mu^{(t-1)} = \sum_{i=1}^{t-1} d_i
```


We can therefore establish the following _recursive_ formula to **update** ``\mu^{(t)}``:

```math
\large
\begin{align}
\mu^{(t)} &= \frac{1}{t}\left[\;\overbrace{\underbrace{(t-1)\;\cdot\;\mu^{(t-1)}}_{d_1+d_2+\ldots+d_{t-1} }\;+\; d_t}^{\text{sum of first } t \text{ data}}\;\right ]
\end{align}
```

"""

# ╔═╡ 27457b37-3939-4b86-b29c-240cd214c94d
md"""

## A recursive formula for ``\mu^{(t)}``

Given the mean of the first ``t-1`` data, ``\mu^{(t-1)}``:


```math
\large
\mu^{(t-1)} = \frac{1}{t-1} \sum_{i=1}^{t-1} d_i
```

_therefore,_

```math
\large
(t-1) \times \mu^{(t-1)} = \sum_{i=1}^{t-1} d_i
```


We can therefore establish the following _recursive_ formula to **update** ``\mu^{(t)}``:

```math
\large
\begin{align}
\mu^{(t)} &= \frac{1}{t}\left[\;\overbrace{\underbrace{(t-1)\;\cdot\;\mu^{(t-1)}}_{d_1+d_2+\ldots+d_{t-1} }\;+\; d_t}^{\text{sum of first } t \text{ data}}\;\right ]\\
&= \frac{t-1}{t}\, \mu^{(t-1)} + \frac{1}{t}\, d_t 
\end{align}
```

"""

# ╔═╡ c3f2a18c-8f4d-46c5-a326-3bfdba99934b
md"""

## A recursive formula for ``\mu^{(t)}``

Given the mean of the first ``t-1`` data, ``\mu^{(t-1)}``:


```math
\large
\mu^{(t-1)} = \frac{1}{t-1} \sum_{i=1}^{t-1} d_i
```

Then

```math
\large
(t-1) \cdot \mu^{(t-1)} = \sum_{i=1}^{t-1} d_i
```


We can therefore establish the following _recursive_ formula to **update** ``\mu^{(t)}``:

```math
\large
\begin{align}
\mu^{(t)} &= \frac{1}{t}\left[\;\overbrace{\underbrace{(t-1)\;\cdot\;\mu^{(t-1)}}_{d_1+d_2+\ldots+d_{t-1} }\;+\; d_t}^{\text{sum of first } t \text{ data}}\;\right ]\\
&= \frac{t-1}{t} \mu^{(t-1)} + \frac{1}{t}d_t \quad \quad \# \text{ note } \tfrac{t-1}{t} + \tfrac{1}{t} =1\\
&= \colorbox{lightgreen}{$\beta_t\, \mu^{(t-1)} + (1-\beta_t)\, d_t$}
\end{align}
```
* where ``\beta_t = \frac{t-1}{t}``

* an online update formula
"""

# ╔═╡ 3e68c117-2016-4da6-931a-912309499a7d
md"""


## A recursive on-line algorithm

In pseudo code:

----

**Initialisation**: ``\mu^{(0)} = 0``


*for* ``t= 1,2,\ldots``

```math
\begin{align}
\beta_t &\leftarrow \frac{t-1}{t}\quad \quad \# \texttt{a discount factor}\\
\mu^{(t)} &\leftarrow \beta_t\, \mu^{(t-1)} + (1-\beta_t)\, d_t \\
\end{align}
```

---


"""

# ╔═╡ f48c2156-79d6-43c7-b53b-0b3fc77b0b12
begin
	Random.seed!(123)
	ndata = 100

	true_f(x) = @. sin(0.5 * x) * 5 + x
	# data2 = sin.(0.5 * range(0, 20 * π, ndata)) * 5 + 1 * range(0, 20, ndata)
	data2 = true_f(range(0, 15 * π, ndata))
	data2 .+= randn(length(data2)) * 3


	data1 = randn(ndata)
end;

# ╔═╡ 23c4f897-765e-44a9-8f06-83e1d0b6a34f
md"""
## Problem of simple average


When ``t \rightarrow \infty`` large,

```math
\frac{1}{t} \rightarrow 0;\quad  \frac{t-1}{t} \rightarrow 1
```


Then 

```math
\begin{align}
\mu^{(t)} &= \frac{t-1}{t}\, \mu^{(t-1)} + \frac{1}{t}\, d_t \\
&\approx 1 \cdot \mu^{(t-1)} + 0\cdot\, (d_t -\mu^{(t-1)})\\

&= \mu^{(t-1)}
\end{align}
```

* the mean ``\mu^{(t)}`` converges and 
* ``d_t``'s contribution vanishes as ``t`` increases
"""

# ╔═╡ 9d7c9641-1653-428a-b366-ab74d9c7c78d
md"""

## An example
"""

# ╔═╡ 352eeeac-b7a2-4b28-be35-be05d044281a
begin
	plot(data2, st=:scatter, alpha=0.8, xlabel=L"t", ylabel="data", label="", title="A dataset with shifting mean")
	# plot!(true_f)
end

# ╔═╡ a00ddb3b-6036-4ff7-ae4d-2969cac6ce21
md"""

## An example
"""

# ╔═╡ ee40107d-41f8-400a-b86d-0ac092e6bb80
gif_2, plt_mean2=let

	data = data2
	
	# plt = plot(data, st=:scatter, c=:gray, alpha=0.1, label="data", xlabel=L"t")
	plt = plot(xlabel=L"t", ylabel=L"d",ylim = extrema(data) .+ (-4, +4))

	μt = 0
	anim = @animate for t in 1:length(data)
		scatter!([t], [data[t]], label="", c=1, title="Online simple average "*L"\mu^{(%$(t))}")
		βt = (t-1)/t
		newμt =  βt * μt + (1-βt) * data[t]
		lbl = (t == 1) ? L"\mu_{(t)}" : ""
		plot!([t-1, t], [μt, newμt], c =4, lw=2, st=:path, label=lbl)
		μt = newμt
	end

	gif(anim, fps=10), plt

end;

# ╔═╡ b7ffb0f7-1ca8-48ac-89d0-63768a48932b
gif_2

# ╔═╡ a8cb91a2-9c1d-43ef-aaae-f2d6638c5423
plt_mean2

# ╔═╡ fd004056-68b2-4cd5-8719-bf8aed131483
gif_1, plt_mean1=let

	data = data1
	
	# plt = plot(data, st=:scatter, c=:gray, alpha=0.1, label="data", xlabel=L"t")
	plt = plot(xlabel=L"t", ylabel=L"d",ylim = extrema(data) .* 1.2)

	μt = 0
	anim = @animate for t in 1:length(data)
		scatter!([t], [data[t]], label="", c=1, title="Online simple average "*L"\mu^{(%$(t))}")
		βt = (t-1)/t
		newμt =  βt * μt + (1-βt) * data[t]
		lbl = (t == 1) ? L"\mu_{(t)}" : ""
		plot!([t-1, t], [μt, newμt], c =4, lw=2, st=:path, label=lbl)
		μt = newμt
	end

	gif(anim, fps=10), plt

end;

# ╔═╡ 5832e04f-e5e0-4c55-a9aa-1be7e9fe1a91
gif_1

# ╔═╡ 465b942d-d8af-4314-b84c-643125172eaf
plt_mean1

# ╔═╡ a22221c1-9a94-406d-96e3-6d584b9187cd
md"""

## Solution: Exponentially Moving Weighted Average (EMWA)



###### We can also simply set ``\beta_t \in [0, 1]`` a constant 


----

**Initialisation**: ``\mu^{(0)} = 0``

Choose fixed discount factor *e.g.* ``\beta \leftarrow 0.9``

*for* ``t= 1,2,\ldots``

```math
\begin{align}

\mu^{(t)} &\leftarrow \beta \, \mu^{(t-1)} + (1-\beta)\, d_t \\
\end{align}
```

---
"""

# ╔═╡ 00a01580-a254-40b3-ab2f-b1d745e67ac8
md"""

## Solution: Exponentially Moving Weighted Average (EMWA)



###### We can also simply set ``\beta_t \in [0, 1]`` a constant 


----

**Initialisation**: ``\mu^{(0)} = 0``

Choose fixed discount factor *e.g.* ``\beta \leftarrow 0.9``

*for* ``t= 1,2,\ldots``

```math
\begin{align}

\mu^{(t)} &\leftarrow \beta \, \mu^{(t-1)} + (1-\beta)\, d_t \\
\end{align}
```

---
"""

# ╔═╡ 14c5ff65-4e01-48d7-af58-01a78ec60788
md"""

Let's assume ``\beta=0.9``, then the new ``\mu^{(t)}`` is a weighted average
```math
\mu^{(t)} = 0.9\cdot \mu^{(t-1)} + 0.1\cdot d_t
```
* the new data ``d_t`` is given a *constant* weight ``0.1``, which does not diminish over the time
"""

# ╔═╡ 9cf9bc30-479f-4686-a22d-998e619bc865
md"""

## Why _exponentially weighted_?


It can be proved that the weights assigned are **exponentially weighted**


```math
\large
\mu^{(t+1)} \propto (\beta^t \, d_1 + \beta^{t-1}\, d_2 + \ldots + \beta^0 d_{t+1})
```


* the most recent observation is given weight ``\beta^0 = 1``
* earlier observations are discounted exponentially ``\beta^{t-i+1}``
"""

# ╔═╡ 2ffeee86-af16-408f-b819-4bb7ac3219a4
let
	n = 20
	βs = [0.9, 0.3]

	plt = plot( xlabel="data index", ylabel="Weights")
	plot!(1:n, ones(n) * 1/n, markershape=:circle, ms=1,st=:sticks, label="", lw=1, c=1)
	plot!(1:n,c=1, ones(n) * 1/n, fill=true, alpha=0.5, label="simple average")
	for (i, β) in enumerate(βs) 	
		weights = (1-β) * [β^(t-1) for t in n:-1:1]
	# plot()
		plot!(1:n, weights, markershape=:circle, ms=1,st=:sticks, label="", lw=1, c=i+1)

		plot!(weights, c=i+1, fill=true, alpha=0.5, label="EWMA "*L"\beta= %$(β)")
	end
	plt
end

# ╔═╡ adb336ba-fcc0-415a-891f-7c52fb068312
gif_3, plt_mean3=let
	β = 0.9
	data = data2
	
	# plt = plot(data, st=:scatter, c=:gray, alpha=0.1, label="data", xlabel=L"t")
	plt = plot(xlabel=L"t", ylabel=L"d",ylim = extrema(data) .+ (-4, +4))
	βt = β
	μt = 0
	anim = @animate for t in 1:length(data)
		scatter!([t], [data[t]], label="", c=1, title="EWMA with "*L"\beta=%$(β);\;\; \mu^{(%$(t))}")
		
		newμt =  βt * μt + (1-βt) * data[t]
		lbl = (t == 1) ? L"\mu_{(t)}" : ""
		plot!([t-1, t], [μt, newμt], c =4, lw=2, st=:path, label=lbl)
		μt = newμt
	end

	gif(anim, fps=5), plt

end;

# ╔═╡ 848054d5-c484-432e-be1f-1744510f73a2
gif_3

# ╔═╡ ff84798b-f6c1-47b9-946a-91eb4196ff44
md"""

## Effect of ``\beta``




"""

# ╔═╡ 53d98475-4c75-4399-96b2-ee4623121309
gif_4, plt_mean4= let
	β = 0.4
	data = data2
	
	# plt = plot(data, st=:scatter, c=:gray, alpha=0.1, label="data", xlabel=L"t")
	plt = plot(xlabel=L"t", ylabel=L"d",ylim = extrema(data) .+ (-4, +4))
	βt = β
	μt = 0
	anim = @animate for t in 1:length(data)
		scatter!([t], [data[t]], label="", c=1, title="EWMA with "*L"\beta=%$(β);\;\; \mu^{(%$(t))}")
		
		newμt =  βt * μt + (1-βt) * data[t]
		lbl = (t == 1) ? L"\mu_{(t)}" : ""
		plot!([t-1, t], [μt, newμt], c =4, lw=2, st=:path, label=lbl)
		μt = newμt
	end

	gif(anim, fps=5), plt

end;

# ╔═╡ 0675960b-0d73-493b-8091-be0a8c37ca14
TwoColumn(md"""
##### Larger ``\beta``: smoother curve; 
* historical data given more weights
* the curve slow to react to the trend 
$(plot(plt_mean3, size=(320,300), titlefontsize=12))
"""
	
	, md"""
##### Smaller ``\beta``: wiggly curve
* more sensitive to the new data
* the curve closely follow the trend
	
$(plot(plt_mean4, size=(320,300), titlefontsize=12))
	""")

# ╔═╡ 267ba8d6-74ef-4ccf-a5f5-8571ef9891b3
md"""

## `Momentum` method


A simple solution is called `Momentum`


* the idea: use **the average of** some recent **past gradients**



```math
\large
\boxed{\begin{align}
\mathbf{d}^{(t-1)} &\leftarrow \beta\,\mathbf{d}^{(t-2)} + (1-\beta)\, \mathbf{g}^{(t-1)}\quad \# \text{EW moving average}\\

\mathbf{w}^{(t)} &\leftarrow \mathbf{w}^{(t-1)} - \gamma\, \mathbf{d}^{(t-1)}
\end{align}}
```

* we can initialise with ``\mathbf{d}^{(0)} = \mathbf{0}`` or ``\mathbf{g}^{(1)}``
"""

# ╔═╡ 64a95557-9f91-4d5d-9f5d-0d8d912e0f20
md"""

## Demonstration: `Momentum`
"""

# ╔═╡ 3b16f5dc-d9ad-4f19-9b80-b2e6bbcea2c6
md"""

## Gradient descent _vs_ `Momentum`
"""

# ╔═╡ 6518a6f9-761f-4535-a52f-83e1c4143d87
md"""

## Limitation of Gradient descent 2 
#### -- plateau in-resilient
"""

# ╔═╡ e80debf7-8f26-40a4-b17e-6184325e5827
md"""

## Gradient descent struggles

**Gradient descent** fails miserably


"""

# ╔═╡ 86dd6aca-312e-4fee-b471-d7168c86ce7d
md"""

## `Momentum` also struggles
"""

# ╔═╡ 0ee47753-5c87-45bc-bf55-dc4c419e6934
md"""

## The problem ?



"""

# ╔═╡ aebaf029-7b5f-4156-9340-80f644c117e9
md"""

## Towards `RProp`



##### Solution: instead of using ``\mathbf{g}^{(t)}``, use its sign!
 

```math
\texttt{sign}(x) =\begin{cases}1  & \text{if } x \geq 0 \\
-1 & \text{if } x<0
\end{cases}
```

Apply ``\texttt{sign}`` element-wisely to each ``\frac{\partial }{\partial w_j} \ell(\mathbf{w})``, for all ``t``

```math
\mathbf{d}^{(t-1)} = \texttt{sign}(\mathbf{g}^{(t-1)}) = \begin{bmatrix} \texttt{sign}\left (\frac{\partial}{\partial w_1} \ell(\mathbf{w}^{(t-1)})\right ) \\ \texttt{sign}\left (\frac{\partial}{\partial w_2} \ell(\mathbf{w}^{(t-1)}) \right ) \\ \vdots \\ \texttt{sign}\left (\frac{\partial}{\partial w_n} \ell(\mathbf{w}^{(t-1)}) \right )\end{bmatrix}
```

"""

# ╔═╡ c3140e0e-61ad-43ac-b3de-988b0bee4f15
md"""

#### How it works?
"""

# ╔═╡ 98408363-dd47-4aa7-a62c-71d7e92a9ed8
md"""

Add RProp negative gradient ``-\texttt{sign}(\mathbf{g})``: $(@bind add_rpg CheckBox(default=false))
"""

# ╔═╡ b3c61071-0f25-41e6-99f9-e193f40777b2
# md"""

# ## Towards `RProp`



# ##### Idea: instead of using ``\mathbf{g}^{(t)}``, use its sign!
 

# ```math
# \texttt{sign}(x) =\begin{cases}1  & \text{if } x \geq 0 \\
# -1 & \text{if } x<0
# \end{cases}
# ```

# Apply ``\texttt{sign}`` element-wisely to each ``\frac{\partial }{\partial w_j} \ell(\mathbf{w})``, for all ``t``

# ```math
# \mathbf{d}^{(t-1)} = \texttt{sign}(\mathbf{g}^{(t-1)}) = \begin{bmatrix} \texttt{sign}\left (\frac{\partial}{\partial w_1} \ell(\mathbf{w}^{(t-1)})\right ) \\ \texttt{sign}\left (\frac{\partial}{\partial w_2} \ell(\mathbf{w}^{(t-1)}) \right ) \\ \vdots \\ \texttt{sign}\left (\frac{\partial}{\partial w_n} \ell(\mathbf{w}^{(t-1)}) \right )\end{bmatrix}
# ```

# """

# ╔═╡ 72960519-0f21-4fe5-9363-61175e0c5063
md"""

## `RProp` 

##### `RProp`: _**R**esilient back**Prop**agation_

----

for each ``t``

* apply ``\texttt{sign}`` element-wisely to each ``\frac{\partial }{\partial w_j} \ell(\mathbf{w})``, for all ``t``

```math
\mathbf{d}^{(t-1)} = \texttt{sign}(\mathbf{g}^{(t-1)}) = \begin{bmatrix} \texttt{sign}\left (\frac{\partial}{\partial w_1} \ell(\mathbf{w}^{(t-1)})\right ) \\ \texttt{sign}\left (\frac{\partial}{\partial w_2} \ell(\mathbf{w}^{(t-1)}) \right ) \\ \vdots \\ \texttt{sign}\left (\frac{\partial}{\partial w_n} \ell(\mathbf{w}^{(t-1)}) \right )\end{bmatrix}
```
```math
\large
\mathbf{w}^{(t)} \leftarrow \mathbf{w}^{(t-1)} - γ\, \mathbf{d}^{(t-1)}
```
----


or I'd rather just call it
* ##### `sign(∇ℓ)` descent 
"""

# ╔═╡ 1a57df9e-63e4-4496-850e-ed6c592fc12a
md"""

## Demonstration
"""

# ╔═╡ 9424354c-7a8e-449c-8c64-5916882458b3
md"""

## Gradient descent *vs* `Momentum` *vs* `RProp`
"""

# ╔═╡ e9f81ad3-29f3-466f-9bed-ef06dfe64b27
md"""

## An alternative formula for `RProp`

Note that 

```math
\begin{align}
\texttt{sign}(x) &= \begin{cases}1  & \text{if } x \geq 0 \\
-1 & \text{if } x<0
\end{cases}  \\
&= \frac{x}{ |x|} \quad \# \text{absolute value}\\
&=\frac{x}{ \sqrt{x^2}}
\end{align}
```


Therefore, `RProp` can also be implemented as 

```math
\begin{align}
\mathbf{d}^{(t-1)} &= \texttt{sign}(\mathbf{g}^{(t-1)}) = \begin{bmatrix}\frac{\mathbf{g}^{(t-1)}_1}{ \sqrt{\left (\mathbf{g}^{(t-1)}_1\right )^2}}  \\ 
\frac{\mathbf{g}^{(t-1)}_2}{ \sqrt{\left (\mathbf{g}^{(t-1)}_2\right )^2}} 
\\ \vdots \\ \frac{\mathbf{g}^{(t-1)}_m}{ \sqrt{\left (\mathbf{g}^{(t-1)} \right )^2}} \end{bmatrix} \\
\mathbf{w}^{(t)} &\leftarrow \mathbf{w}^{(t-1)} - γ\, \mathbf{d}^{(t-1)}
\end{align}
```
"""

# ╔═╡ 60887b8f-9ba2-4aaf-8760-d92a66a75cc1
md"""

## An alternative formula for `RProp`

Note that 

```math
\begin{align}
\texttt{sign}(x) &= \begin{cases}1  & \text{if } x \geq 0 \\
-1 & \text{if } x<0
\end{cases}  \\
&= \frac{x}{ |x|} \quad \# \text{absolute value}\\
&=\frac{x}{ \sqrt{x^2}}
\end{align}
```


Therefore, `RProp` can also be implemented as 

```math
\begin{align}
\mathbf{d}^{(t-1)} &\leftarrow \begin{bmatrix}\frac{\mathbf{g}^{(t-1)}_1}{ \sqrt{\left (\mathbf{g}^{(t-1)}_1\right )^2}+\epsilon}  \\ 
\frac{\mathbf{g}^{(t-1)}_2}{ \sqrt{\left (\mathbf{g}^{(t-1)}_2\right )^2+\epsilon}} 
\\ \vdots \\ \frac{\mathbf{g}^{(t-1)}_m}{ \sqrt{\left (\mathbf{g}^{(t-1)} \right )^2}+\epsilon} \end{bmatrix} \\
\mathbf{w}^{(t)} &\leftarrow \mathbf{w}^{(t-1)} - γ\, \mathbf{d}^{(t-1)}
\end{align}
```

* in practice, we add a small constant ``\epsilon`` to avoid dividing by zero problem
"""

# ╔═╡ 4da613dc-4d03-4d94-b963-90892c004af2
md"""

## An alternative formula for `RProp`

Note that 

```math
\large
\begin{align}
\texttt{sign}(x) &= \begin{cases}1  & \text{if } x \geq 0 \\
-1 & \text{if } x<0
\end{cases}  \\
&= \frac{x}{ |x|} \quad \# \text{absolute value}\\
&=\frac{x}{ \sqrt{x^2}}
\end{align}
```


Therefore, `RProp` can be specified as (here we introduce ``\mathbf{v}``: gradient's square)

```math
\begin{align}
\color{red}{\mathbf{v}^{(t-1)}} &\color{red}{\leftarrow (\mathbf{g}^{(t-1)})^2} = \small \begin{bmatrix} \left (\mathbf{g}^{(t-1)}_1\right )^2 \\ 
\left (\mathbf{g}^{(t-1)}_2\right )^2
\\ \vdots \\ \left (\mathbf{g}^{(t-1)}_m\right )^2 \end{bmatrix}  \\
\mathbf{d}^{(t-1)} &\leftarrow  \frac{\mathbf{g}^{(t-1)}}{\sqrt{\mathbf{v}^{(t-1)}}+\epsilon} \quad \# \text{sign}(\nabla\ell)\\
\mathbf{w}^{(t)} &\leftarrow \mathbf{w}^{(t-1)} - γ\, \mathbf{d}^{(t-1)}
\end{align}
```

"""

# ╔═╡ c337818e-a45f-48a1-b1f0-f02ba0f264f9
md"""

## `RMSProp`

##### -- minibatches version `RProp`
\


**Root Mean Squared Propogation** (`RMSProp`) is a variant of `RProp`

* stochastic gradient descent with **mini-batches**'s gradient can be noisy

* we use **exponentially weighted moving average** to smooth out the gradient square ``\mathbf{v}^{(t-1)}`` between the batches

\


"""

# ╔═╡ 9a79612a-e8dd-4cb0-8776-9e22e00c299d
TwoColumn(md"""
Resilient backprop (`RProp`)


---
_for each_ ``t``

```math
\small
\begin{align}
\mathbf{v}^{(t-1)} &\leftarrow \left (\mathbf{g}^{(t-1)}\right )^2  \\
\mathbf{d}^{(t-1)} &\leftarrow  \frac{\mathbf{g}^{(t-1)}}{\sqrt{\mathbf{v}^{(t-1)}}+\epsilon} \\
\mathbf{w}^{(t)} &\leftarrow \mathbf{w}^{(t-1)} - γ\, \mathbf{d}^{(t-1)}
\end{align}
```
---


""", md"""
**Root Mean Squared Propogation** (`RMSProp`)


----
_for each_ ``t``

```math
\small
\begin{align}
\color{red}{\mathbf{v}^{(t-1)}} & \color{red}\leftarrow \beta\, \mathbf{v}^{(t-2)} + (1-\beta)\left (\mathbf{g}^{(t-1)}\right )^2  \quad \# \text{EWMA of } \mathbf{v}  \\
\mathbf{d}^{(t-1)} &\leftarrow  \frac{\mathbf{g}^{(t-1)}}{\sqrt{\mathbf{v}^{(t-1)}}+\epsilon} \\
\mathbf{w}^{(t)} &\leftarrow \mathbf{w}^{(t-1)} - γ\, \mathbf{d}^{(t-1)}
\end{align}
```

----


* ``\beta = 0.9`` is a good choice (between 0.5  and 0.9)

""")

# ╔═╡ 77f1314c-bfcb-4d56-b1bb-01afadcf0517
aside(tip(md"""


We name ``\mathbf{g}^2`` as ``\mathbf{v}`` on purpose

* ``\mathbf{v}`` here stands for **variance**


It can be shown that 

```math
\bar{\mathbf{g}} =\mathbb{E}[\mathbf{g}] =\mathbf{0}
```

* *i.e.* the expected value (or long term average) of the gradient is zero (this is known as the *score* function in frequentist statistical inference jargon)

* the random variables here are sampling distrbution of the data 


Therefore, the variance 

```math
\begin{align}
\mathbb{V}[\mathbf{g}] &= \mathbb{E}[(\mathbf{g} -\mathbb{E}[\mathbf{g}])  (\mathbf{g} -\mathbb{E}[\mathbf{g}])^\top ] \\
&=\mathbb{E}[\mathbf{g}\mathbf{g}^\top ]
\end{align}
```

If we assume the variance is a diagonal matrix, then

$\begin{align}
\mathbf{v} &= \text{diag}(\mathbb{V}[\mathbf{g}]) = \mathbb{E}[\mathbf{g} \odot \mathbf{g}]\\
&= \mathbb{E} [\mathbf{g}^2]
\end{align}$
"""))

# ╔═╡ 9bbd144e-72fd-43b8-bbb1-46f7d198063a
md"""

## Comparison
"""

# ╔═╡ d5507c0a-a924-4e29-a415-7c031a6cae63
md"""

## `Adam`  -- a more adaptive alternative

#### -- `Momentum` with `RMSProp`


\

**Adaptive Moment Estimation** (`Adam`) is another variant 


* it introduces EWMA estimation of ``\mathbf{g}`` together with ``\mathbf{v}``

* combination of all above flavours: `RProp`, `Momentum` and `RMSProp`
"""

# ╔═╡ 4a20ceb4-d318-42c3-8f7e-27ebe1d75a80
TwoColumn(md"""
**Root Mean Squared Propogation** (`RMSProp`)


---
_for each_ ``t``

```math
\small
\begin{align}
\mathbf{v}^{(t-1)} &\leftarrow \beta\, \mathbf{v}^{(t-2)} + (1-\beta)\left (\mathbf{g}^{(t-1)}\right )^2    \\
\mathbf{d}^{(t-1)} &\leftarrow  \frac{\mathbf{g}^{(t-1)}}{\sqrt{\mathbf{v}^{(t-1)}}+\epsilon} \\
\mathbf{w}^{(t)} &\leftarrow \mathbf{w}^{(t-1)} - γ\, \mathbf{d}^{(t-1)}
\end{align}
```
---


""", md"""
**Adaptive Moment Estimation** (`Adam`) 


---
_for each_ ``t``

```math
\small
\begin{align}
\color{red}\mathbf{m}^{(t-1)} &\color{red}{\leftarrow \beta_1\, \mathbf{m}^{(t-2)} + (1-\beta_1) \mathbf{g}^{(t-1)} } \quad \# \text{EWMA of } \mathbf{g}  \\
\mathbf{v}^{(t-1)} &\leftarrow \beta_2\, \mathbf{v}^{(t-2)} + (1-\beta_2)\left (\mathbf{g}^{(t-1)}\right )^2  \quad \# \text{EWMA of } \mathbf{v}  \\
\mathbf{d}^{(t-1)} &\leftarrow  \frac{\mathbf{m}^{(t-1)}}{\sqrt{\mathbf{v}^{(t-1)}}+\epsilon} \\
\mathbf{w}^{(t)} &\leftarrow \mathbf{w}^{(t-1)} - γ\, \mathbf{d}^{(t-1)}
\end{align}
```
---

* ``\beta_1 = 0.9, \beta_2 =0.99`` is commonly used

""")

# ╔═╡ d5369f75-b6ea-4e37-b362-6099971cadb1
md"""

## Comparison
"""

# ╔═╡ 07af2fa0-0c01-428f-88cd-fef8ffd402e9
md"""

## Comparison
"""

# ╔═╡ a0b0df2e-5fe7-4b5e-a41a-71ce0ed4e7f3
md"""

## So what method to use?




"""

# ╔═╡ a862a5d1-4ccf-4d29-b3c6-c5ab48c264ad
md"""


###### _No easy answer to this question_

* note that the above is just one hand-crafted example

* usually `Adam` (and also `RMSProp`) works well straight out of the box
  
* you should just do trial and error


###### The effect of optimiser is usually not _that_ significant

* even vanilla gradient descent should work reasonably well

* if training fails, it is more likely your code is wrong rather than the fault of the optimiser!
"""

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

# ╔═╡ fc523f31-ed47-464d-8fdc-5410f0e483e3
function g1(w; A = diagm([.5, 9.75]), b =zeros(2), c=0)
	# forward 
	l = w' * A * w + w'*b + c
	# backward 
	# gw = (A + A') * w + b
	return l
end;

# ╔═╡ 1996275f-0aa5-4ca4-a613-7f6998d46d15
begin
	gr()
	plot(-1:0.1:10.5, -2:0.1:2, (w1, w2) -> g1([w1, w2]), st=:contourf, c=:coolwarm, colorbar=false, ratio=1.2, ylim=(-2,2), size=(600,310), xlabel=L"w_1", ylabel=L"w_2", title="An example")

	scatter!([0],[0], ms=4, markershape=:x, markerstrokewidth=5, label="minimum", ylim=(-2,2), xlim =(-1, 10.5))
	scatter!([10], [1], ms=6,  markerstrokewidth=5, c=4, markershape=:cross, label="starting "*L"\mathbf{w}_0")
	# plot(-2:0.1:10.5, -2:0.1:2, (w1, w2) -> g1([w1, w2]), st=:surface, c=:coolwarm, colorbar=false, ratio=1.2, ylim=(-2,2), size=(700,320), xlabel=L"w_1", ylabel=L"w_2")
end

# ╔═╡ 8f3dd723-2662-4f80-b772-c4047b68f0d3
function g2(w)
	w1, w2 = w[1], w[2]
	max(0, tanh(4*w1 + 4*w2)) + abs(0.5*w1) 
end;

# ╔═╡ 0777d82d-320b-454d-a476-2cb895194456
let
	gr()
	plot(-3:0.1:3, -2:0.1:4, (x, y) -> g2([x, y]), st=:contourf, c=:coolwarm, size=(400,350), colorbar=false)
	scatter!([0],[-1.5], ms=5, markershape=:x,c=:red,  markerstrokewidth=5, label="minimum", ylim=(-2,2), xlim =(-1, 10.5))
	scatter!([2], [2], ms=8,  markerstrokewidth=5, c=:blue, markershape=:cross, label="starting "*L"\mathbf{w}_0", xlim = (-3,3), ylim =(-2,4))
end

# ╔═╡ 72493456-a531-47be-b799-ff9d7dfa488e
let
	plotly()
	plot(-3:0.1:3, -2:0.1:4, (x, y) -> g2([x, y]), st=:surface, c=:jet, size=(400,400), colorbar=false)
end

# ╔═╡ c4432aa9-85ac-4833-99d9-48e3fe34f50d
gt = gradient(g2, [2., 2.])[1];

# ╔═╡ 76839145-2fd2-4f32-bbc0-09624b4df361
TwoColumn(md"""
```math
\mathbf{g} \triangleq \nabla\ell(\mathbf{w}) = \begin{bmatrix} \frac{\partial}{\partial w_1} \ell(\mathbf{w}) \\ \frac{\partial}{\partial w_2} \ell(\mathbf{w}) \end{bmatrix} \approx  \begin{bmatrix} 0.5\\ 2.04 \times 10^{-13}\end{bmatrix}
```
or $(latexify_md(round.(gt; digits=2)))
* the partial of ``w_2`` is very small

* ``w_2`` **barely gets any update**

```math
\begin{bmatrix} w_1^{(t)} \\ w_2^{(t)} \end{bmatrix} = \begin{bmatrix} w_1^{(t-1)} \\ w_2^{(t-1)} \end{bmatrix} -\gamma \begin{bmatrix} 0.5\\ 0.0 \end{bmatrix}
```

""", let
	gr()
	plt = plot(-3:0.1:3, -2:0.1:4, (x, y) -> g2([x, y]), st=:contourf, c=:coolwarm, size=(310,310), colorbar=false)
	scatter!([0],[-1.5], ms=5, markershape=:x,c=:red,  markerstrokewidth=5, label="minimum", ylim=(-2,2), xlim =(-1, 10.5))
	scatter!([2], [2], ms=5,  markerstrokewidth=5, c=:blue, markershape=:x, label="starting "*L"\mathbf{w}_0", xlim = (-3,3), ylim =(-2,4))

	w0 = [2, 2.]
	g1 = gradient(g2, w0)[1] * 1.5
	vg = w0 - g1
	# quiver!([w0[1]], [w0[2]], quiver=([vg[1] - w0[1]], [vg[2] - w0[2]]), label="grad")

	plot!([w0[1], vg[1]], [w0[2],vg[2]], line = (:arrow, 2, :blue), label = "")

	annotate!([vg[1]], [vg[2]+0.2], text(L"\mathbf{g}", 12, :blue, :right))

	
	plt
end)

# ╔═╡ a294e093-60a1-42f4-a732-1c2fae5c46ba
let
	gt = gradient(g2, [2., 2.])[1]
	gt, sign.(gt)
end

# ╔═╡ a2b85d68-5320-40d3-b91c-ab30323dc605
let
	gr()
	plt = plot(-3:0.1:3, -2:0.1:4, (x, y) -> g2([x, y]), st=:contourf, c=:coolwarm, size=(500,400), colorbar=false)
	scatter!([0],[-1.5], ms=5, markershape=:x,c=:red,  markerstrokewidth=5, label="minimum", ylim=(-2,2), xlim =(-1, 10.5))
	scatter!([2], [2], ms=5,  markerstrokewidth=5, c=:blue, markershape=:x, label="starting "*L"\mathbf{w}_0", xlim = (-3,3), ylim =(-2,4))

	w0 = [2, 2.]
	g1 = gradient(g2, w0)[1]
	vg = w0 - g1
	# quiver!([w0[1]], [w0[2]], quiver=([vg[1] - w0[1]], [vg[2] - w0[2]]), label="grad")

	plot!([w0[1], vg[1]], [w0[2],vg[2]], line = (:arrow, 3, :blue), label = "")

	annotate!([vg[1]], [vg[2]+0.2], text(L"-\mathbf{g}", 12, :blue, :right))

	if add_rpg
		pg = .8 * sign.(g1)
		vpg = w0 - pg
		plot!([w0[1], vpg[1]], [w0[2],vpg[2]], line = (:arrow, 3, :red), label = "")
	
		annotate!([vpg[1]], [vpg[2]+0.2], text(L"-\texttt{sign}(\mathbf{g})", 12, :red, :right))
	end
	plt
end

# ╔═╡ 6b405fbb-c7e7-4bca-a1ab-dc2c37b97961
function produce_anim(f, ws, losses; method="Gradient descent", color=:thermal, ms=5, xlims = (-1.0, 10.5), ylims = (-2.0, 2.0), steps = 100, ratio = 1.2, fig_size=(700, 320), minimum = nothing)
	gr()
	# range_num = 200
	plt = plot(range(xlims..., steps), range(ylims..., steps), (w1, w2) -> f([w1, w2]), st=:contourf, c=:coolwarm, colorbar=false, ratio=ratio, ylim=ylims, xlim=xlims, size=fig_size, xlabel = L"w_1", ylabel=L"w_2")
	# plt = plot()
	traces = ws
	cscheme = cgrad(color, rev=true)
	anim = []
	losses_ = losses / maximum(losses)
	if !isnothing(minimum)
		scatter!([minimum[1]], [minimum[2]], markersize=6, mc = get(cscheme, 0), markershape=:x, label="minimum", markerstrokewidth=3)
	end
	wt = traces[:, 1]
	scatter!([wt[1]], [wt[2]], color = get(cscheme, losses_[1]), markersize = ms, label="")
	
	anim = @animate for t in 2:(size(traces)[2])
		plot!(plt, [traces[1, t]], [traces[2, t]], st=:scatter, color=get(cscheme, losses_[t]), label="", markersize=ms, title=method*"; Iteration: $(t)")
		plot!(plt, [wt[1], traces[1, t]], [wt[2], traces[2, t]], line = (:path, 1, :gray), label="")
		wt = traces[1:2, t]
	end

	# title!(plt, method*" trajectory")
	return anim, plt
end

# ╔═╡ 78d7da50-6937-4834-baa5-ade61aad4756
function train_optimiser(f; max_iters =25, w_init=[10, 1.], optimiser = Descent(0.1))
	maxiters = max_iters
	model = (w = w_init,)
	optim = Optimisers.setup(optimiser, model)
	losses = []
	ws = zeros(2, maxiters+1)
	ws[:, 1] = model.w
	for i in 2:(maxiters+1)
		li, grad = withgradient(model) do m
			f(m.w)[1]
		end
		push!(losses, li)
		Optimisers.update!(optim, model, grad[1])
		ws[:, i] = model.w
	end
	push!(losses, f(model.w)[1])

	return losses, ws
end

# ╔═╡ 485edefb-caec-47c7-b67c-0aa26f1d6152
begin

	optims_g1 = [Descent(0.05), Descent(0.1), Momentum(0.1, 0.2), Momentum(0.05, 0.6)]
	names_g1 = ["Gradient descent "*L"\gamma=0.05", "Gradient descent "*L"\gamma=0.1", "Momentum "*L"\beta=0.2", "Momentum "*L"\beta = 0.6"]
	losses_g1 = []
	anims_g1 = []
	plts_g1 = []
	for (i, opt) in enumerate(optims_g1)
		loss, ws = train_optimiser(g1; max_iters =25, optimiser = opt)
		anim, plt = produce_anim(g1, ws, loss; method=names_g1[i], fig_size=(600,300), minimum =zeros(2))
		push!(losses_g1, loss)
		push!(anims_g1, anim)
		push!(plts_g1, plt)
	end

	# losses_g1, anims_g1

end;

# ╔═╡ e446b537-8cf8-4838-b8a1-51cec0cd8402
gif(anims_g1[1], fps=3)

# ╔═╡ b077d3c7-6afe-43cc-a4e3-8f41e9567ba0
let
	gif(anims_g1[2], fps=3)
end

# ╔═╡ 305c6931-ba8d-4e5f-baea-7d844edfb71d
let
	plt = plot(xlabel="Iteration", ylabel="Loss", size=(550,300), title="Gradient descent with different learning rates", titlefontsize=12, legendfontsize=10)
	for (i,l) in enumerate(losses_g1[1:2])
		plot!(l, label=names_g1[i], lw=1.8)
	end
	plt
end

# ╔═╡ 2067bfe9-20ad-42d6-a0bb-dc003a2ba0e5
let
	gif(anims_g1[2], fps=3)
end

# ╔═╡ 42e8b5be-f834-4799-a28c-d07103ad9e70
gif(anims_g1[3], fps=3)

# ╔═╡ e8588c68-1f60-449b-9092-2146e7d10463
gif(anims_g1[4], fps=3)

# ╔═╡ 8ccdb56f-5a0b-4f32-a77e-ec0bc7f72f0d
plot(plts_g1..., layout=(2,2), titlefontsize=8, size=(800, 450))

# ╔═╡ 2bcf770b-da47-4102-abb2-c9bf7576706c
let
	plt = plot(xlabel="Iteration", ylabel="Loss")
	for (i,l) in enumerate(losses_g1)
		plot!(l, label=names_g1[i], lw=1.5)
	end
	plt
end

# ╔═╡ 09b50288-7d4c-47c4-af3b-5a3d8d904c8a
begin

	optims = [Descent(0.3), Momentum(0.3, 0.7), Rprop(0.3), RMSProp(0.3), Adam(0.3)]
	names = ["Gradient descent","Momentum", "Rprop", "RMSProp", "Adam"]
	losses_g2 = []
	anims_g2 = []
	plts_g2 = []
	for (i, opt) in enumerate(optims)
		loss, ws = train_optimiser(g2; max_iters =50, w_init=[2, 2.], optimiser = opt)
		anim, plt = produce_anim(g2, ws, loss; xlims =(-3., 4.), ylims =(-3., 4.), fig_size=(400,400), ratio=1, ms=3, method=names[i])
		push!(losses_g2, loss)
		push!(anims_g2, anim)
		push!(plts_g2, plt)
	end

	# losses_g2, anims_g2
end

# ╔═╡ 945120a5-1e92-4a46-a3be-3f1b5d765df8
let
	Logging.disable_logging(Logging.Info) # or e.g. Logging.Info
	TwoColumn(gif(anims_g2[1]; fps=4), md"")
end

# ╔═╡ 1816987f-f14a-4aae-9aa9-c41e4a5c06ac
let
	Logging.disable_logging(Logging.Info) # or e.g. Logging.Info
	TwoColumn(gif(anims_g2[1]; fps=4), gif(anims_g2[2]; fps=4))
end

# ╔═╡ c9ad9b05-d3a1-425f-94ca-b5a394cfdad0
let
	Logging.disable_logging(Logging.Info) # or e.g. Logging.Info
	TwoColumn(gif(anims_g2[2]; fps=15), gif(anims_g2[3]; fps=15))
	# gif(anims_g2[3]; fps=15)
end

# ╔═╡ 2350cfa1-e873-465e-a837-ef436aff4530
plot(plts_g2[[1,2,3]]..., layout=(1,3),titlefontsize=10, size=(800,350))

# ╔═╡ a6c54a00-b89b-41f7-9de9-c5e2f835da7e
let
	Logging.disable_logging(Logging.Info) # or e.g. Logging.Info
	TwoColumn(gif(anims_g2[3]; fps=4), gif(anims_g2[4]; fps=3))
end

# ╔═╡ d3133877-ba2e-4faa-91c9-0798760858d3
let
	Logging.disable_logging(Logging.Info) # or e.g. Logging.Info
	TwoColumn(gif(anims_g2[4]; fps=10), gif(anims_g2[5]; fps=10))
end

# ╔═╡ 624cde95-7e4a-419a-b859-499f8919b6aa
plot(plts_g2[[1; 3:end]]..., layout=(2,2), titlefontsize=10, size=(800, 800))

# ╔═╡ e4798d52-9435-4f23-a8fc-732a556a61fd
let
	plt = plot(xlabel="Iteration", ylabel="Loss", title="Comparison between different optimisers")
	for (i,l) in enumerate(losses_g2)
		plot!(l, label=names[i], lw=1.5)
	end
	plt
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.1"
Latexify = "~0.16.1"
Optimisers = "~0.3.1"
Plots = "~1.39.0"
PlutoTeachingTools = "~0.2.14"
PlutoUI = "~0.7.54"
Zygote = "~0.6.68"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "1eedf585aee45312e1d8dcc3b64f4cb45d744587"

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

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "60a0339f28a233601cb74468032b5c302d5067de"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.2"

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

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "c9ff5c686240c31eb8570b662dd1f66f4b183116"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.3.4"

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
# ╠═9f90a18b-114f-4039-9aaf-f52c77205a49
# ╟─a533cf6c-22d8-4659-813a-f1e12984326b
# ╟─d400c959-0a96-49bf-9239-de12d5e39de1
# ╟─30e2c8a3-0bc1-42d3-bd27-d7e493f4c164
# ╟─e1e8c5ca-874c-4a30-9211-8c29ee226c3b
# ╟─3e2e1ea8-3a7d-462f-ac38-43a087907a14
# ╟─52dcd4b9-9ef7-4128-a81d-d7e454cae9d6
# ╟─19ebad08-8461-46fc-90bf-fcb1fa30d833
# ╟─7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
# ╟─bc96a33d-9011-41ec-a19e-d472cbaafb70
# ╟─7091d2cf-9237-45b2-b609-f442cd1cdba5
# ╟─0a7f37e1-51bc-427d-a947-31a6be5b765e
# ╟─a696c014-2070-4041-ada3-da79f50c9140
# ╟─595a5ef3-4f54-4502-a943-ace4146efa31
# ╟─bc1ee08d-9376-44d7-968c-5e114b09a5e0
# ╟─84c68c61-7168-44c5-a606-b1b252377229
# ╟─d32970e9-1ec5-4074-92a9-1989e86234f7
# ╟─1996275f-0aa5-4ca4-a613-7f6998d46d15
# ╟─65c48015-fcfb-41c7-b3c7-79411b9721b9
# ╟─fd636237-ebc5-4d8c-9703-f3c93adfda5b
# ╟─e446b537-8cf8-4838-b8a1-51cec0cd8402
# ╟─59b80abc-e696-4870-9cf8-0aaafcbbd6e0
# ╟─e5b02c45-9b49-4ceb-8f88-2a25229ff62b
# ╟─b077d3c7-6afe-43cc-a4e3-8f41e9567ba0
# ╟─305c6931-ba8d-4e5f-baea-7d844edfb71d
# ╟─0af4a76f-578b-4195-826b-84cdf7b65afe
# ╟─791cb069-a6a9-4c33-a4fd-677b7a4ceb6f
# ╟─6071849b-08f7-4618-a3b5-1c6ba67c6396
# ╟─2067bfe9-20ad-42d6-a0bb-dc003a2ba0e5
# ╟─485edefb-caec-47c7-b67c-0aa26f1d6152
# ╟─b2b0ef75-7192-4d2c-9fcc-3bc4e49da8cc
# ╟─bebe066b-c212-402f-af15-6c329e42941e
# ╟─749db615-65c5-41fb-9232-f0c936f2c811
# ╟─27457b37-3939-4b86-b29c-240cd214c94d
# ╟─c3f2a18c-8f4d-46c5-a326-3bfdba99934b
# ╟─3e68c117-2016-4da6-931a-912309499a7d
# ╟─f48c2156-79d6-43c7-b53b-0b3fc77b0b12
# ╟─5832e04f-e5e0-4c55-a9aa-1be7e9fe1a91
# ╟─465b942d-d8af-4314-b84c-643125172eaf
# ╟─23c4f897-765e-44a9-8f06-83e1d0b6a34f
# ╟─9d7c9641-1653-428a-b366-ab74d9c7c78d
# ╟─352eeeac-b7a2-4b28-be35-be05d044281a
# ╟─a00ddb3b-6036-4ff7-ae4d-2969cac6ce21
# ╟─b7ffb0f7-1ca8-48ac-89d0-63768a48932b
# ╟─a8cb91a2-9c1d-43ef-aaae-f2d6638c5423
# ╟─ee40107d-41f8-400a-b86d-0ac092e6bb80
# ╟─fd004056-68b2-4cd5-8719-bf8aed131483
# ╟─a22221c1-9a94-406d-96e3-6d584b9187cd
# ╟─00a01580-a254-40b3-ab2f-b1d745e67ac8
# ╟─14c5ff65-4e01-48d7-af58-01a78ec60788
# ╟─848054d5-c484-432e-be1f-1744510f73a2
# ╟─9cf9bc30-479f-4686-a22d-998e619bc865
# ╟─2ffeee86-af16-408f-b819-4bb7ac3219a4
# ╟─adb336ba-fcc0-415a-891f-7c52fb068312
# ╟─ff84798b-f6c1-47b9-946a-91eb4196ff44
# ╟─0675960b-0d73-493b-8091-be0a8c37ca14
# ╟─53d98475-4c75-4399-96b2-ee4623121309
# ╟─267ba8d6-74ef-4ccf-a5f5-8571ef9891b3
# ╟─64a95557-9f91-4d5d-9f5d-0d8d912e0f20
# ╟─42e8b5be-f834-4799-a28c-d07103ad9e70
# ╟─e8588c68-1f60-449b-9092-2146e7d10463
# ╟─3b16f5dc-d9ad-4f19-9b80-b2e6bbcea2c6
# ╟─8ccdb56f-5a0b-4f32-a77e-ec0bc7f72f0d
# ╟─2bcf770b-da47-4102-abb2-c9bf7576706c
# ╟─6518a6f9-761f-4535-a52f-83e1c4143d87
# ╟─0777d82d-320b-454d-a476-2cb895194456
# ╟─72493456-a531-47be-b799-ff9d7dfa488e
# ╟─e80debf7-8f26-40a4-b17e-6184325e5827
# ╟─945120a5-1e92-4a46-a3be-3f1b5d765df8
# ╟─86dd6aca-312e-4fee-b471-d7168c86ce7d
# ╟─1816987f-f14a-4aae-9aa9-c41e4a5c06ac
# ╟─0ee47753-5c87-45bc-bf55-dc4c419e6934
# ╟─76839145-2fd2-4f32-bbc0-09624b4df361
# ╟─c4432aa9-85ac-4833-99d9-48e3fe34f50d
# ╟─aebaf029-7b5f-4156-9340-80f644c117e9
# ╟─c3140e0e-61ad-43ac-b3de-988b0bee4f15
# ╟─a294e093-60a1-42f4-a732-1c2fae5c46ba
# ╟─98408363-dd47-4aa7-a62c-71d7e92a9ed8
# ╟─a2b85d68-5320-40d3-b91c-ab30323dc605
# ╟─b3c61071-0f25-41e6-99f9-e193f40777b2
# ╟─72960519-0f21-4fe5-9363-61175e0c5063
# ╟─1a57df9e-63e4-4496-850e-ed6c592fc12a
# ╟─c9ad9b05-d3a1-425f-94ca-b5a394cfdad0
# ╟─9424354c-7a8e-449c-8c64-5916882458b3
# ╟─2350cfa1-e873-465e-a837-ef436aff4530
# ╟─e9f81ad3-29f3-466f-9bed-ef06dfe64b27
# ╟─60887b8f-9ba2-4aaf-8760-d92a66a75cc1
# ╟─4da613dc-4d03-4d94-b963-90892c004af2
# ╟─c337818e-a45f-48a1-b1f0-f02ba0f264f9
# ╟─9a79612a-e8dd-4cb0-8776-9e22e00c299d
# ╟─77f1314c-bfcb-4d56-b1bb-01afadcf0517
# ╟─09b50288-7d4c-47c4-af3b-5a3d8d904c8a
# ╟─9bbd144e-72fd-43b8-bbb1-46f7d198063a
# ╟─a6c54a00-b89b-41f7-9de9-c5e2f835da7e
# ╟─d5507c0a-a924-4e29-a415-7c031a6cae63
# ╟─4a20ceb4-d318-42c3-8f7e-27ebe1d75a80
# ╟─d5369f75-b6ea-4e37-b362-6099971cadb1
# ╟─d3133877-ba2e-4faa-91c9-0798760858d3
# ╟─07af2fa0-0c01-428f-88cd-fef8ffd402e9
# ╟─624cde95-7e4a-419a-b859-499f8919b6aa
# ╟─e4798d52-9435-4f23-a8fc-732a556a61fd
# ╟─a0b0df2e-5fe7-4b5e-a41a-71ce0ed4e7f3
# ╟─a862a5d1-4ccf-4d29-b3c6-c5ab48c264ad
# ╟─0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
# ╟─8687dbd1-4857-40e4-b9cb-af469b8563e2
# ╟─fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
# ╠═fc523f31-ed47-464d-8fdc-5410f0e483e3
# ╠═8f3dd723-2662-4f80-b772-c4047b68f0d3
# ╠═6b405fbb-c7e7-4bca-a1ab-dc2c37b97961
# ╠═78d7da50-6937-4834-baa5-ade61aad4756
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
