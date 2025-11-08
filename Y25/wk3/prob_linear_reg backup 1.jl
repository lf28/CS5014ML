### A Pluto.jl notebook ###
# v0.20.18

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

# ╔═╡ 94a408c8-64fe-11ed-1c46-fd85dc9f96de
begin
	using PlutoTeachingTools
	using PlutoUI
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	using LinearAlgebra
	using StatsPlots
	using LogExpFunctions
	using StatsBase
	using LaTeXStrings
	using Latexify
	using Random
	using Distributions
	using HypertextLiteral
end

# ╔═╡ 0a8a31fe-6220-4eaa-8a51-1e8452db01b3
using Zygote

# ╔═╡ 8fd91a2b-9569-46df-b0dc-f29841fd2015
TableOfContents()

# ╔═╡ e13a748a-61aa-4f3d-9d1c-98a27d6a8d0a
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ 82216880-0550-4dc8-abce-783d39e404a9
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

# ╔═╡ cde253e9-724d-4d2b-b82a-e5919240ddd3
ChooseDisplayMode()

# ╔═╡ d5a707bb-d921-4e13-bad0-8b8d03e1852a
md"""

# CS5014 Machine Learning


#### Probabilistic linear regression 
###### and extensions
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 6df53306-f00a-4d7e-9e77-0b717f016f06
# md"""

# ## Notations


# Superscript--index with brackets ``.^{(i)}``: ``i \in \{1,2,\ldots, n\}`` index observations/data
# * ``n`` total number of observations
# * *e.g.* ``y^{(i)}`` the i-th observation's label
# * ``\mathbf{x}^{(i)}`` the i-th observation's predictor vector

# Subscript--index: feature index ``j \in \{1,2,,\ldots, m\} ``
# * ``m`` total number of features
# * *e.g.* ``\mathbf{x}^{(i)}_2``: the second element/feature of ``i``--th observation


# Vectors: **Bold--face** smaller case:
# * ``\mathbf{x},\mathbf{y}, \boldsymbol{\beta}``
# * ``\mathbf{x}^\top``: row vector

# Matrices: **Bold--face** capital case: 
# * ``\mathbf{X},\mathbf{A}, \boldsymbol{\Gamma}``  


# Scalars: normal letters
# * ``x,y,\beta,\gamma``

# """

# ╔═╡ 50debe1f-dd65-42a2-a276-9814b43b6882
md"""


# Probabilistic linear regression
"""

# ╔═╡ 29997b58-b32b-455e-83d6-42ba628694f6
md"""

## Reading & references

##### Essential reading 


* **MLE & Loss function** [_Understanding Deep Learning_ by _Simon Prince_: Chapter 5.1-5.3](https://github.com/udlbook/udlbook/releases/download/v2.00/UnderstandingDeepLearning_28_01_24_C.pdf)




"""

# ╔═╡ de783288-05d7-47f0-a5ae-8c8e668015ac
md"""


## Recap: Gaussian


Random variable ``y`` is Gaussian distributed if

$$\large p(y|\mu, \sigma^2) = \mathcal{N}(y; \mu, \sigma^2) = \frac{1}{ \sqrt{2\pi\sigma^2}} e^{-\frac{1}{2} \frac{(y-\mu)^2}{\sigma^2} }$$

* ``\mu``: mean parameter, the central location of the distribution
* ``\sigma^2``: variance, controls the spread




"""

# ╔═╡ 54bf4146-3653-4caa-9544-e7d95fae9ee5
plt_gs_μ=let

	μs = [-3, 0, 3]
	σ²s = [1 , 2 , 5]
	plt_gaussian = Plots.plot(title="Gaussian distributions", xlabel=L"y", ylabel=L"p(y)")
	for i in 1:3 
		plot!(plt_gaussian, Normal(μs[i], sqrt(σ²s[i])), fill=true, alpha=0.5, label=L"\mathcal{N}(μ=%$(μs[i]), σ^2=%$(σ²s[i]))")
		vline!([μs[i]], color=i, label="", linewidth=2)
	end
	plt_gaussian
end;

# ╔═╡ 986c6c21-220e-4ccf-ba60-c30baf46e3c2
plt_gs_ss=let

	# μs = [0, 0, 0, 0]
	σ²s = [1 , 2 , 5, 10, 25]
	plt_gaussian = Plots.plot(title="Effects of "*L"\sigma^2", xlabel=L"y", ylabel=L"p(y)")
	for i in 1:length(σ²s)
		μ = 0
		plot!(plt_gaussian, Normal(μ, sqrt(σ²s[i])), fill=true, alpha=0.5, label=L"\mathcal{N}(μ=%$(μ), σ^2=%$(σ²s[i]))")
		# vline!([μ], color=i, label="", linewidth=2)
	end
	plt_gaussian
end;

# ╔═╡ 1ca89046-70dc-4850-b863-47f99ce6684d
plot(plt_gs_μ, plt_gs_ss, size=(800,400))

# ╔═╡ 96a21111-92e4-42ed-8afc-9f39042a5759
md"""


## Gaussian: log likelihood


##### Gaussian distribution

$$\large p(y|\mu,\sigma^2) = \mathcal{N}(y; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp \left \{-\frac{1}{2} \frac{(y-\mu)^2}{\sigma^2} \right \}$$


##### The log density is

!!! note ""
	$$\large \ln p(y|\mu,\sigma^2) = -\frac{1}{2}\ln 2\pi  -\frac{1}{2}\ln \sigma^2-\frac{1}{2\sigma^2} (y - \mu)^2$$

"""

# ╔═╡ 94b6d065-d774-4035-bc35-107e30feba63
md"""

## Least square's loss



"""

# ╔═╡ 9ad2f1ce-cd76-4e33-ac87-e847d1841bfb
md"""



## Probabilistic model for linear regression

##### Probabilistic regression: a probabilistic model
  * ##### we tell a generative story about how ``y^{(i)}`` being generated
  * ##### in other words, we specify ``p(y^{(i)}| \cdots)``


## Probabilistic model for linear regression


##### We assume each $y^{(i)}$ is _generated_ by

```math
\large
y^{(i)} = \underbrace{h(\mathbf{x}^{(i)})}_{\text{signal}} + \underbrace{\epsilon^{(i)}}_{\text{noise}}, \;\; \epsilon^{(i)} \sim  \mathcal{N}(0, \sigma^2)
```

* ##### ``y^{(i)}`` is a noisy observation of ``h(\mathbf{x}^{(i)})``



## Probabilistic model for linear regression (cont.)


##### _Algorithmically_, the "story" for ``y^{(i)}`` is


---
##### Given fixed ``\{\mathbf{x}^{(i)}\}``, which are assumed fixed and non-random


##### for each ``\mathbf{x}^{(i)}``
  * ###### *true signal* ``h(\mathbf{x}^{(i)}) =\mathbf{w}^\top \mathbf{x}^{(i)}``
  * ###### *sample* a Gaussian noise ``\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)``
  * ###### *generate* ``y^{(i)} = h(\mathbf{x}^{(i)}) + \epsilon^{(i)}``

---

##### _Probabilistically_, it means ``y^{(i)}`` is

$\large \begin{align}p(y^{(i)}|\mathbf{x}^{(i)}, \mathbf{w}, \sigma^2) &= \mathcal{N}(y^{(i)};  \mathbf{w}^\top \mathbf{x}^{(i)} , \sigma^2)\end{align}$
"""

# ╔═╡ 2c7f2bad-10f2-4705-9dc8-549752713a03
md"Add true function ``h(x; \mathbf{w})``: $(@bind add_h CheckBox(default=false)),
Add ``p(y^{(i)}|x^{(i)})``: $(@bind add_pyi CheckBox(default=false)),
Add ``y^{(i)}\sim p(y^{(i)}|x^{(i)})``: $(@bind add_yi CheckBox(default=false))
"

# ╔═╡ d1e601fd-62d9-4b37-8fdf-352dd82ca9d6
md"""

## Probabilistic linear regression model (anim.)
"""

# ╔═╡ 3c508c8e-04ae-481c-8407-a77884479aa1
begin
	Random.seed!(123)
	n_obs = 20
	# the input x is fixed; non-random
	xs = range(-0.5, 1; length = n_obs)
	true_w = [1.0, 1.0]
	true_σ² = 0.05
	ys = zeros(n_obs)
	for (i, xⁱ) in enumerate(xs)
		hⁱ = true_w' * [1, xⁱ]
		ys[i] = hⁱ + rand(Normal(0, sqrt(true_σ²)))
	end
end

# ╔═╡ 33f90867-39e0-4b09-8e33-eaccbc5f6e62
md"
Select ``x^{(i)}``: $(@bind add_i Slider(1:length(xs); show_value=true))
"

# ╔═╡ 0415f7b6-2717-4ca7-920f-03a202dac6a2
md"""
## Probabilistic linear regression model


> $\large \begin{align}p(y^{(i)}|\mathbf{x}^{(i)}, \mathbf{w}, \sigma^2) &= \mathcal{N}(y^{(i)};  \mathbf{w}^\top \mathbf{x}^{(i)} , \sigma^2)\\ &= \frac{1}{\sqrt{2\pi\sigma^2}}\text{exp}\left(-\frac{(y^{(i)}-{\mathbf{w}}^\top\mathbf{x}^{(i)})^2}{2\sigma^2}\right)\end{align}$

* ``y^{(i)}`` is a univariate Gaussian with mean $\mathbf{w}^\top \mathbf{x}^{(i)}$ and variance $\sigma^2$ 


"""

# ╔═╡ d456e90d-467a-4063-a97f-113677646e78
md"``x_i`` $(@bind xᵢ0 Slider(-0.5:0.1:1, default=0.15));	``\sigma^2`` $(@bind σ²0 Slider(0.005:0.01:0.15, default=0.05))"

# ╔═╡ 40506d38-e87a-4411-ac9d-2ce932ec3dd2
md"Input $x^{(i)}=$ $(xᵢ0); and ``\sigma^2=`` $(σ²0)"

# ╔═╡ 9aa580fe-f42b-4600-bc88-4a22057f30cc
begin
	Random.seed!(123)
	β0 = true_w
	n0 = 100
	xx = range(-0.5, 1; length = n0)
	yy = β0[1] .+ xx * β0[2] + sqrt(σ²0) * randn(n0)
end;

# ╔═╡ fc0f92ba-784b-4be6-84c6-c7faa4551ee7
let

	plt = plot(xs, zeros(length(xs)), st=:scatter, framestyle=:origin, labels=L"x", color=:black, ms=5, markershape=:x, xlabel=L"x", ylim=[-0.2, 2.5], ylabel=L"y", legend=:outerbottom)

	if add_h
		plot!(-0.6:0.1:1.1, (x) -> true_w[1] + true_w[2]*x, lw=2, label="the true signal: " * L"h(x)")
	end
	σ²0 = true_σ²
	xis = xs
	i = add_i
	if add_pyi
		for j in 1:i
			x = xis[j]
			μi = dot(β0, [1, x])
			σ = sqrt(σ²0)
			xs_ = μi- 4 * σ :0.05:μi+ 4 * σ
			ys_ = pdf.(Normal(μi, σ), xs_)
			ys_ = 0.1 *ys_ ./ maximum(ys_)
			if j == i
				scatter!([x],[μi], markerstrokewidth =1, markershape = :diamond, c=2, label=L"\mu=h(x)", markersize=5)
			end
			lw_ = j == i ? 2 : 0.5 
			plot!(ys_ .+x, xs_, c=:grey, label="", linewidth=lw_)
		end
	end


	if add_yi
		scatter!([xis[1:i]],[ys[1:i]], markershape = :circle, label="observation: "*L"y^{(i)}", c=1, markersize=4)
	end

	plt
end

# ╔═╡ 10104f42-424d-41c1-b5f9-1a5a63cd97bc
let

	plt = plot(xs, zeros(length(xs)), st=:scatter, framestyle=:origin, labels=L"x", color=:black, ms=5, markershape=:x, xlabel=L"x", ylim=[-0.2, 2.5], ylabel=L"y", legend=:outerbottom)

	plot!(-0.6:0.1:1.1, (x) -> true_w[1] + true_w[2]*x, lw=2, label="the true signal: " * L"h(x)")

	# σ²0 = true_σ²
	xis = xs


	anim = @animate for i in 1:length(xis)
		x = xis[i]
		μi = dot(β0, [1, x])
		σ = sqrt(true_σ²)
		xs_ = μi- 4 * σ :0.05:μi+ 4 * σ
		ys_ = pdf.(Normal(μi, sqrt(true_σ²)), xs_)
		ys_ = 0.1 *ys_ ./ maximum(ys_)
		# scatter!([x],[μi], markerstrokewidth =1, markershape = :diamond, c=:grey, label="", markersize=3)
		plot!(ys_ .+x, xs_, c=:grey, label="", linewidth=.5)
		scatter!([xis[i]],[ys[i]], markershape = :circle, label="", c=1, markersize=4)
	end

	gif(anim; fps=4)
end

# ╔═╡ 441b0b82-6c1c-4d51-ad88-c65857d11618
let
	gr()
	# b_1 = 3.0
	p_lr = plot(title="Linear regression's probabilistic model",legend=:bottomright)
	plot!(xx, yy, st=:scatter, ylim=[0, 3],framestyle=:origin, label="observations", legend=:topleft)
	plot!(-0.5:0.1:1.0, x->β0[1]+β0[2]*x, c= 1, linewidth=5, label="",  ylim=[-1, 3],framestyle=:origin)
	# xis = [-0.35, -0.2, 0, 0.25, 0.5, 0.75, 0.99, xᵢ0]
	xis = [range(-0.5, 1.0, 8)...]
	push!(xis, xᵢ0)
	for i in 1:length(xis)
		x = xis[i]
		μi = dot(β0, [1, x])
		σ = sqrt(σ²0)
		xs_ = μi- 4*σ :0.05:μi+ 4 *σ
		ys_ = pdf.(Normal(μi, σ), xs_)
		ys_ = 0.1 *ys_ ./ maximum(ys_)
		if i == length(xis)
			scatter!([x],[μi], markerstrokewidth =2, markershape = :diamond, c=:red, label=L"h(x)", markersize=6)
			plot!(ys_ .+x, xs_, c=:red, label="", linewidth=3)
		else
			plot!(ys_ .+x, xs_, c=:gray, label="", linewidth=1)
			# scatter!([x],[μi], markerstrokewidth =2, markershape = :diamond, label="μ @ x="*string(x))
		end
		
	end
	p_lr	
end

# ╔═╡ 33851131-9889-4d5f-80be-c91b062f6eae
md"""

## The likelihood function

##### The likelihood function is

```math
\large
p(\mathcal{D}|\mathbf{w}, \sigma^2, \{\mathbf{x}^{(i)}\}) = \prod_{i=1}^n p(y^{(i)}|\mathbf{w}, \sigma^2, \mathbf{x}^{(i)})
```

##### As usual, its log transformation is 

```math
\large
\ell(\mathbf{w}, \sigma^2) = \ln p(\mathcal{D}|\mathbf{w}, \sigma^2, \{\mathbf{x}^{(i)}\}) = \sum_{i=1}^n \ln p(y^{(i)}|\mathbf{w}, \sigma^2, \mathbf{x}^{(i)})
```
* ##### it is a function of ``\mathbf{w}, \sigma^2`` !
"""

# ╔═╡ cd48a047-f803-4df1-9b9f-1d422579591e
md"""

## The likelihood function


##### Recall the log-transformed likelihood for ``y^{(i)}`` is

$$\large \ln p({y}^{(i)}| \mathbf{w}, \sigma^2, \mathbf{x}^{(i)}) = -\frac{1}{2} \ln 2\pi\sigma^2 -\frac{1}{2\sigma^2}({y}^{(i)}-\mathbf{w}^\top \mathbf{x}^{(i)})^2$$


##### The log-likelihood therefore is 

$$\large \begin{align}\ell (\mathbf{w}, \sigma^2) &= \ln p(\mathcal{D}|\mathbf{w}, \sigma^2) = \sum_{i=1}^n \ln p(y^{(i)}|\mathbf{w}, \sigma^2, \mathbf{x}^{(i)})\\
&=  -\frac{n}{2} \ln 2\pi\sigma^2 -\frac{1}{2\sigma^2} \underbrace{\colorbox{pink}{$\sum_{i=1}^n({y}^{(i)}-\mathbf{w}^\top \mathbf{x}^{(i)})^2$}}_{\text{sum of squared error loss!}}
\end{align}$$



## MLE is LSE

##### *Maximising* ``\ell(\mathbf{w})`` is the same as minimising its **negation** 


```math
\Large
\begin{align}
&\;\;\;\;\;\,{\mathbf{w}}_{\text{MLE}} \leftarrow \arg\max_{\mathbf{w}} \ell (\mathbf{w}, \sigma^2) \\
&\;\;\;\;\;\,{\mathbf{w}}_{\text{MLE}} \leftarrow \arg\min_{\mathbf{w}} \colorbox{lightgreen}{$-$}\ell (\mathbf{w}, \sigma^2) 
\end{align}
```


"""

# ╔═╡ c406b7a7-6917-4936-92a1-bd52049fec68
md"""
## MLE is LSE

##### *Maximising* ``\ell(\mathbf{w}, \sigma^2)`` is the same as minimising its *negation*



```math
\large
\begin{align}
&\;\;\;\;\;\,{\mathbf{w}}_{\text{MLE}} \leftarrow \arg\max_{\mathbf{w}} \ell (\mathbf{w}, \sigma^2) \\
&\Rightarrow{\mathbf{w}}_{\text{MLE}} \leftarrow \arg\min_{\mathbf{w}} \colorbox{lightgreen}{$-$}\ell (\mathbf{w}, \sigma^2) \\

&\Rightarrow {\mathbf{w}}_{\text{MLE}} \leftarrow \arg\min_{\mathbf{w}} \left (\underbrace{\frac{n}{2} \ln 2\pi \sigma^2}_{\text{const.}} +\frac{1}{2\underbrace{\sigma^2}_{\rm const.}} \sum_{i=1}^n({y}^{(i)}-\mathbf{w}^\top \mathbf{x}^{(i)})^2\right )
\end{align}
```



* ##### MLE is just the same as least square 

"""

# ╔═╡ a62ec2a1-50b2-468d-bfc0-79b998a2f6fe
md"""


## Observation noise variance ``\sigma^2_{\text{MLE}}`` 


##### We can also estimate $\sigma^2$ 

```math
\large
\begin{align}
&\;\;\;\;\;\,{\mathbf{w}}_{\text{MLE}},{\sigma^2}_{\text{MLE}} \leftarrow \arg\min_{\mathbf{w}, \sigma^2} -\ell (\mathbf{w}, \sigma^2) 
\end{align}
```


##### Take the derivative w.r.t. ``\sigma^2`` and solve it,


```math
\large
{\sigma^2}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^n (y^{(i)} -\mathbf{w}_{\text{MLE}}^\top\mathbf{x}^{(i)})^2
```
"""

# ╔═╡ f32ba09b-dbce-4bab-a2a0-7b935b56ab22
md"""

##

!!! fact "Summary"
	Gaussian likelihood-based probabilistic regression's MLE are
	```math
		\large
		\begin{align}
		{\mathbf{w}}_{\text{MLE}} &= {\mathbf{w}}_{\text{LSE}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}\\

		\sigma^2_{\text{MLE}} &= \frac{1}{n} \sum_{i=1}^n (y^{(i)} -\mathbf{w}_{\text{MLE}}^\top\mathbf{x}^{(i)})^2
		\end{align}
	```

"""

# ╔═╡ 6d4d6876-5e7a-469a-86d4-cd7a2bc237dc
# md"""

# # Probabilistic regression extensions*
# """

# ╔═╡ 2d11f0fa-3174-4f2a-80aa-3b0eaa5eee57
md"""

## Why bother MLE?


##### The likelihood provides a *principled approach* to define loss

```math
\Large
\text{loss}(\mathbf{w}) = - \ln p(y^{(i)}|\mathbf{x}^{(i)}) 
```


* ##### different probability models ``\Longleftrightarrow`` different loss
* ##### it saves us from engineering specific losses 


"""

# ╔═╡ c1614569-e9ab-40a3-b34b-cf2242eb5ffd
# md"""

# ## Dealing with outliers
# """

# ╔═╡ ec77faee-0082-4b77-ad62-f9aac4ccb77e
# plot(plt_out, size=(600,450), legend=:outerbottom)

# ╔═╡ 0cfdb61c-016a-4b44-a15c-9f6d47e2f469
# plt_out, xs_out, ys_out = let
# 	Random.seed!(123)
# 	xs_new = [collect(range(-0.55, -0.4, length=5))..., collect(range(0.85, 0.95, length=5))...]
# 	ys_outliers = [ones(5) * 2 + randn(5)/5..., ones(5) * 0 + randn(5)/8...]
# 	# append!(ys_outliers, ones(length(xs_new)) * 0.1 .+ randn(5)/5)
# 	xs_out = [xx; xs_new]
# 	ys_out = [yy; ys_outliers]
# 	outlier_plt = plot(xx, yy, st=:scatter, framestyle=:origin, c=1, alpha=.3, ms=3, label="observations")
# 	plot!(xs_new, ys_outliers, st=:scatter, c=1, label="outliers")
# 	w_out = [ones(length(xs_out)) xs_out] \ ys_out
# 	ys_true = true_w[1] .+ true_w[2] * (-0.5:0.1:1.0)
# 	plot!(-0.5:0.1:1.0, ys_true, lw=4, lc=1, label="the true signal: " * L"h(x)", size=(300,300),  title="Outlier observations")

# 	ys_gaussians = w_out[1] .+ w_out[2] * (-0.5:0.1:1.0)
# 	plot!(-0.5:0.1:1.0, ys_gaussians, label="Gaussian MLE",  lw=4, legend=:topleft)
# 	plot!(-0.5:0.1:1.0, ys_gaussians, c=4,  fillrange = ys_true, fillstyle=:| , fillalpha=0.9, label="",  lw=4, legend=:topleft)
# 	# plot!(fillrange = y, fillstyle = :/)
# 	outlier_plt, xs_out, ys_out
# end;

# ╔═╡ 3ebaa9fd-994b-4606-a0ed-e2b303cf6d6b
# md"""

# ## Recap: Gaussian -- "_68-95-99 rule_"

# """

# ╔═╡ 5f9fe4e9-db6e-4f48-861e-362dfff220c6
# TwoColumn(md""" 

# ##### The probability mass centered at ``\mu``

# $\mathbb{P}( \mu-2\sigma \leq x\leq  \mu + 2\sigma) \approx 95\%$

# * ##### *almost impossible* to oberve data ``> \pm 3\times\sigma``  from ``\mu``

# """, @htl """<center><img src='https://tikz.net/files/gaussians-002.png'  width = '350' /></center>""")

# ╔═╡ a810b24a-94b1-48a5-b425-0de998504d5c
# md"""

# ## Laplace _vs_ Gaussian

# ```math
# \large
# \texttt{Laplace}(y; \mu, \sigma) = \frac{1}{2\sigma} \exp \left (- \frac{|y-\mu|}{\sigma}\right )
# ```
# """

# ╔═╡ 3354023d-9328-4708-9834-9ab1029743c7
# md"Add `Gaussian` $(@bind add_gaussian CheckBox(default=false)),
# Add `Laplace` samples $(@bind add_lap_s CheckBox(default=false)),
# Add `Gaussian` samples $(@bind add_gauss_s CheckBox(default=false))
# "

# ╔═╡ 6b5aaf90-aa6a-4409-aebb-78a6771b0064
# let
# 	# gr()
# 	# μs = [ 0, 3]
# 	# σ²s = [1 , 2 , 5]
# 	Random.seed!(123456)
# 	nn = 400
# 	ys = rand(Laplace(), nn)
# 	plt_gaussian = plot(title="Gaussian/Laplace distributions", xlabel=L"y",ylabel=L"p(y)", framestyle=:origin)


# 	plot!(plt_gaussian, -4.5:0.1:4.5, (x) -> pdf(Laplace(0, sqrt(1)), x), fill=true, alpha=0.5, label=L"\texttt{Laplace}(μ=0, σ=1)", c=1, legendfontsize=12)

# 	if add_gaussian
# 		plot!(plt_gaussian, -4.5:0.1:4.5, (x) -> pdf(Normal(0, sqrt(1)), x), fill=true, alpha=0.5, label=L"\mathcal{N}(μ=0, σ^2=1)", c=2)
# 	end


# 	if add_lap_s
# 		scatter!(ys, zeros(length(ys)) .-0.05, c=1,  markersize=3, alpha=.4, label="Laplace samples")
# 	end

# 	if add_gauss_s
# 		ys_gaus = randn(nn)
# 		scatter!(ys_gaus, zeros(length(ys)) .-0.025, c=2,  markersize=3, alpha=.4, label="Gaussian samples")
# 	end

# 	plt_gaussian
# 	# # end
# 	# plt_gaussian
# end

# ╔═╡ 8be44433-4791-4e5d-bf6c-6c4727822a00
# md"""

# ## Robust regression model


# > $\large \begin{align}p(y^{(i)}|\mathbf{x}^{(i)}, \mathbf{w}, \sigma^2) &= \mathcal{N}(y^{(i)};  \mathbf{w}^\top \mathbf{x}^{(i)} , \sigma^2)\end{align}$
# > $\Downarrow$
# > $\large \begin{align}p(y^{(i)}|\mathbf{x}^{(i)}, \mathbf{w}, \sigma) &= \texttt{Laplace}(y^{(i)};  \mathbf{w}^\top \mathbf{x}^{(i)} , \sigma) \end{align}$

# """

# ╔═╡ 244d3782-bcc2-4f07-971d-12c2bb748463
# md"Make the change: $(@bind add_laplace_liks CheckBox(false))"

# ╔═╡ 06223805-b9c7-454b-9e75-31364bf50dfd
# let
# 	gr()
# 	# b_1 = 3.0
# 	xs = xs_out
# 	ys = ys_out
# 	title_ = add_laplace_liks ? "Probabilistic regression with Laplace likelihood" : "Probabilistic regression with Gaussian likelihood"
# 	p_lr = plot(title=title_,legend=:bottomright)
# 	plot!(xs, ys, st=:scatter, ylim=[0, 3],framestyle=:origin, alpha=0.8, ms=3, label="observations", legend=:topleft)
# 	plot!(-0.5:0.1:1.0, x->β0[1]+β0[2]*x, c= 2, linewidth=3, label="",  ylim=[-1, 3],framestyle=:origin)
# 	# xis = [-0.35, -0.2, 0, 0.25, 0.5, 0.75, 0.99, xᵢ0]
# 	xis = [range(-0.5, 0.99, 8)...]
# 	# push!(xis, xᵢ0)
# 	for i in 1:length(xis)
# 		x = xis[i]
# 		μi = dot(β0, [1, x])
# 		σ = sqrt(true_σ²)
		
# 		if add_laplace_liks
# 			xs_ = μi- 6 * σ :0.01:μi+ 6 *σ
# 			ys_ = pdf.(Laplace(μi, 2*σ), xs_)
# 			ys_ = 0.1 * ys_ ./ maximum(ys_)
# 			plot!([x, x], [μi - 4 * σ , μi + 4*σ], st=:path, label="", c=:gray, lw=1, arrow=(:both))
# 			plot!(ys_ .+x, xs_, c=:gray, label="", linewidth=1.5)
# 		else
# 			xs_ = μi- 4*σ :0.05:μi+ 4 *σ
# 			ys_ = pdf.(Normal(μi, σ), xs_)
# 			ys_ = 0.1 *ys_ ./ maximum(ys_)
# 			plot!([x, x], [μi - 2*σ , μi + 2*σ], st=:path, label="", c=:gray, arrow=(:both))
# 			plot!(ys_ .+x, xs_, c=:gray, label="", linewidth=1)
# 		end
		
		
# 		# if i == length(xis)
# 			# scatter!([x],[μi], markerstrokewidth =2, markershape = :diamond, c=:red, label=L"\mu", markersize=6)
# 			# plot!(ys_ .+x, xs_, c=:red, label="", linewidth=3)
# 		# else
# 			scatter!([x],[μi], markerstrokewidth =2, markershape = :diamond, c=:red, label="", markersize=3)

# 			# scatter!([x],[μi], markerstrokewidth =2, markershape = :diamond, label="μ @ x="*string(x))
# 		# end
		
# 	end
# 	p_lr	
# end

# ╔═╡ 4628c200-f72a-45ff-8759-6dd60593d45b
# md"""

# ## Robust loss

# #### The `Laplace` log-likelihood is

# $$\Large \begin{align}\ell (\mathbf{w}, \sigma) &= \ln p(\mathcal{D}|\mathbf{w}, \sigma) \\
# &= -n \ln 2\sigma -\frac{1}{\sigma} {\sum_{i=1}^n|{y}^{(i)}-\mathbf{w}^\top \mathbf{x}^{(i)}|}
# \end{align}$$


# #### The loss is *negative* log-likelihood

# $$\Large \begin{align}\mathcal{L} (\mathbf{w}, \sigma) 
# &=- \ell(\mathbf{w}, \sigma) \\
# &= n \ln 2\sigma +\frac{1}{\sigma} \underbrace{\sum_{i=1}^n|{y}^{(i)}-\mathbf{w}^\top \mathbf{x}^{(i)}|}_{\text{sum of absolute deviations!}}
# \end{align}$$

# * ##### we can use gradient descent to learn it
#   * ###### gradient descent works even loss is not differentiable everywhere
# """

# ╔═╡ 53c16406-4db3-490d-9313-699503d55a58
# md"""

# ## _Absolute deviation_ vs _Squared error_
# """

# ╔═╡ cb44fd76-286d-47a8-9b24-c793e7144977
# begin

# 	plot(-4:0.1:4, x -> abs(x), framestyle=:zerolines, lw=2, label=L"|x|", xlabel="error", legendfontsize =15, ratio=0.35)

# 	plot!(-4:0.1:4, x-> x^2, label=L"x^2", ylabel="Loss",lw=2, title="Squared vs absolute error", legend=:outerbottom)

# end

# ╔═╡ 07cdc6e2-376e-432f-a7d9-a8729ccad3b3
# begin
# 	function laplace_reg_loss(w, X, y) 
# 		diff = sum(abs.(X * w - y))
# 		return diff
#  	end
# end;

# ╔═╡ 42577101-7eb0-4c94-8624-52d8ad8073a5
# w_laplace, losses_laplace = let
# 	max_iters = 100
# 	wₜ = zeros(2)
# 	Xs_out = [ones(length(xs_out)) xs_out]
# 	∇l(w) = Zygote.gradient((x) -> laplace_reg_loss(x, Xs_out, ys_out), w)[1]
# 	γ = 0.05
# 	loss = []
# 	for t in 1:max_iters
# 		γ = γ / (1+ 0.5 * γ * t)
# 		∇wₜ = ∇l(wₜ)
# 		wₜ = wₜ - γ * ∇wₜ
# 		push!(loss, laplace_reg_loss(wₜ, Xs_out, ys_out))
# 	end
# 	wₜ, loss
# end;

# ╔═╡ 1dbd6027-626d-46aa-b25a-1b26ba9270ae
# md"""

# ## Learning -- Gradient descent*

# """

# ╔═╡ 67cf29ea-45c7-4454-9e36-4efa316f5a57
# md"""

# #### Consider the ``i``-th loss only


# ```math
# \large
# \begin{align}
# 	&l^{(i)}(\mathbf{w}) = |{y}^{(i)}-\mathbf{w}^\top \mathbf{x}^{(i)}|\\

# &\nabla l^{(i)}(\mathbf{w})= ?
# \end{align}
# ```

# * the total loss (and gradient) is just the sum (of the individual gradients)

# #### Note that

# ```math
# \large
# l^{(i)}(\mathbf{w}) = \begin{cases} {y}^{(i)}-\mathbf{w}^\top \mathbf{x}^{(i)} & \text{if}\; y^{(i)} > \mathbf{w}^\top \mathbf{x}^{(i)}\\ -{y}^{(i)}+\mathbf{w}^\top\mathbf{x}^{(i)}  & \text{if}\; y^{(i)} < \mathbf{w}^\top \mathbf{x}^{(i)}\end{cases} 
# ```



# """

# ╔═╡ a1045431-6d95-49bb-b5a8-81758df2b883
# aside(tip(plot(abs, size=(275,200),lw=2, title=L"f(x)=|x|", label="")))

# ╔═╡ dd2066bd-650d-463b-97b2-b8abf15bf589
# md"""

# ## Learning -- Gradient descent*

# """

# ╔═╡ 5c7cdb6e-8218-446f-93dc-de313a0db997
# md"""

# #### Consider the ``i``-th loss only


# ```math
# \large
# \begin{align}
# 	&l^{(i)}(\mathbf{w}) = |{y}^{(i)}-\mathbf{w}^\top \mathbf{x}^{(i)}|\\

# &\nabla l^{(i)}(\mathbf{w})= ?
# \end{align}
# ```

# * the total loss (and gradient) is just the sum (of the individual gradients)


# #### Note that

# ```math
# \large
# l^{(i)}(\mathbf{w}) = \begin{cases} {y}^{(i)}-\mathbf{w}^\top \mathbf{x}^{(i)} & \text{if}\; y^{(i)} > \mathbf{w}^\top \mathbf{x}^{(i)}\\ -{y}^{(i)}+\mathbf{w}^\top\mathbf{x}^{(i)}  & \text{if}\; y^{(i)} < \mathbf{w}^\top \mathbf{x}^{(i)}\end{cases} 
# ```


# #### Its gradient therefore is


# ```math
# \large
# \nabla l^{(i)}(\mathbf{w}) = \begin{cases} - \mathbf{x}^{(i)} & \text{if}\; y^{(i)} > \mathbf{w}^\top \mathbf{x}^{(i)}\\ \mathbf{x}^{(i)}  & \text{if}\; y^{(i)} < \mathbf{w}^\top \mathbf{x}^{(i)}\end{cases}= \boxed{- {\normalsize\texttt{sign}}(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)})\cdot \mathbf{x}^{(i)}}
# ```
# * `sign`: `sign(-2) = -1`, `sign(2)=1`

# """

# ╔═╡ 1b470ba6-8f0e-4bde-af9c-0c4685fb2bd2
# md"""
# ## Learning -- Gradient descent (cont.)*
# """

# ╔═╡ f7c4dd1d-8df8-4600-97c6-42939ebf1eb8
# md"""


# #### The final gradient is the sum of the individual gradients

# ```math
# \large
# \begin{align}
# \nabla l(\mathbf{w}) &= \sum_{i=1}^n\nabla l^{(i)}(\mathbf{w}) = - \sum_{i=1}^n  \texttt{sign}(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)})\cdot \mathbf{x}^{(i)} \\
# &= -\mathbf{X}^\top \cdot \mathbf{\texttt{sign}}(\mathbf{y} - \mathbf{Xw})
# \end{align}
# ```
# """

# ╔═╡ 922f3ff1-b74e-47a1-8ee1-fa6de5015e4f
# md"""

# **Implementation of the gradient in Python**
# ```python
# 	# implementation in Python and Numpy
# 	# @: matrix multiplication
# 	# np.sign: Numpy provides the sign function
# 	grad = - Xs.T @ np.sign(ys - (Xs @ w))

# ```

# """

# ╔═╡ e37002c5-b0a8-41bb-86dd-6c19db385661
# md"""
# **Implementation of the gradient in Julia**

# ```julia
# 	# much cleaner implementation in Julia
# 	# ".": broadcasting
# 	∇l = - Xs' * sign.(ys - Xs * w)
# ```

# """

# ╔═╡ ee8ca729-8011-49b6-96fa-89c873261ca0
# md"""
# ## Demonstration
# """

# ╔═╡ b402fb18-9157-4099-8ed3-6d50f0238614
# md"Add robust regrssion fit: $(@bind add_robust CheckBox(default=false))"

# ╔═╡ 58932bb0-b94b-4fbd-8170-0717fb24b7b2
# let
# 	gr()
# 	plt = plot(xs_out, ys_out, st=:scatter, framestyle=:origin, label="observations", legend=:topleft, alpha=.5)
# 	w_out = [ones(length(xs_out)) xs_out] \ ys_out
# 	plot!(-0.5:0.1:1.0, (x) -> true_w[1] + true_w[2]*x, lw=3, lc=1, label="the true signal: " * L"h(x)", legend=:topright)

# 	plot!(-0.5:0.1:1.0, (x) -> w_out[1] + w_out[2]*x, lw=3, lc=4, label="Gaussian MLE " * L"\hat{h}(x)", title="Robust regression with Laplace likelihood")
# 	if add_robust
# 		plot!(-0.5:0.1:1.0, (x) -> w_laplace[1] + w_laplace[2]*x, lw=3, lc=2,  label="Laplace MLE " * L"\hat{h}(x)")
# 	end
# 	plot(plt, legend=:outerbottom, size=(600,500), legendfontsize=12)
# end

# ╔═╡ 83c226e6-7b59-4f75-beee-09eb8af39429
# md"""


# ## Non-constant noise
# """

# ╔═╡ b85a28d3-f221-4c02-b4d6-fa2ae0492600
# plot(plt_heter1, size=(600,400), legendfontsize=12)

# ╔═╡ 2839d19e-5fd9-4baf-bb0a-8eaa99e100b6
# plt_heter2, xs_hetero, ys_hetero = let
# 	Random.seed!(123)
# 	n0 = 200
# 	xs = range(-0.5, 1; length = n0)
# 	true_w = [1, 3]
# 	true_σ = [0, 1]
# 	lnσ = true_σ[1] .+ true_σ[2] * xs 
# 	ys = true_w[1] .+ 3 * xs +  exp.(lnσ).* randn(n0)
# 	plt = plot(xs, ys, st=:scatter, framestyle=:origin, label="observations", legend=:topleft, size=(300,300), ms=3, alpha=.5)

# 	plot!(-0.5:0.1:1.0, (x) -> true_w[1] + true_w[2]*x, lw=3, lc=1, label="the true signal: " * L"h(x)", title="Heterogeneous noise")

# 	xis = range(-.5, 1; length = 8)
# 	for i in 1:length(xis)
# 		x = xis[i]
# 		μi = dot(true_w, [1, x])
# 		σ = exp(true_σ[1] .+ true_σ[2] * x)
# 		xs_ = μi- 4 * σ :0.1:μi+ 4 * σ
# 		ys_ = pdf.(Normal(μi, σ), xs_)
# 		ys_ = 0.1 *ys_ ./ maximum(ys_)
# 		plot!(ys_ .+x, xs_, c=:grey, label="", linewidth=1)
# 		# scatter!([xis[i]],[ys[i]], markershape = :circle, label="", c=1, markersize=4)
# 	end
# 	plt, xs, ys
# end;

# ╔═╡ e7cd15e6-b73e-4528-b2d4-c30933e352b5
# plt_heter1 = let
# 	Random.seed!(123)
# 	n0 = 200
# 	xs = range(-0.5, 1; length = n0)
# 	true_w = [1, 3]
# 	true_σ = [0, 1]
# 	lnσ = true_σ[1] .+ true_σ[2] * xs 
# 	ys = true_w[1] .+ 3 * xs +  exp.(lnσ).* randn(n0)
# 	plt = plot(xs, ys, st=:scatter, framestyle=:origin, label="observations", legend=:topleft, size=(300,300), ms=3, alpha=.5)
# 	# plot!(xs_new, ys_outliers, st=:scatter, framestyle=:origin, label="outliers", legend=:topright)
# 	# w_out = [ones(length(xs_out)) xs_out] \ ys_out
# 	plot!(-0.5:0.1:1.0, (x) -> true_w[1] + true_w[2]*x, lw=3, lc=1, label="the true signal: " * L"h(x)", title="Heterogeneous noise")
# 	plt
# end;

# ╔═╡ 6fbe1186-ba88-4d37-8def-41eae484cbfc
# md"""

# ## How to specify the loss?
# ##### -- just specify a different likelihood ``p(y^{(i)}|x^{(i)})``

# """

# ╔═╡ 51e878bc-0f82-459e-a95c-f197c732efc5
# show_img("hetero.png", w=480)

# ╔═╡ e71aadbe-150a-4d6e-b9b3-f6e747adefe3
# @bind x0i_hetero Slider(-0.5:0.05:1)

# ╔═╡ d1428f56-4d6a-4435-9835-eb944941d369
# let
# 	Random.seed!(123)
# 	n0 = 200
# 	xs = range(-0.5, 1; length = n0)
# 	true_w = [1, 3]
# 	true_σ = [0, 1]
# 	lnσ = true_σ[1] .+ true_σ[2] * xs 
# 	ys = true_w[1] .+ 3 * xs +  exp.(lnσ).* randn(n0)
# 	plt = plot(xs, ys, st=:scatter, framestyle=:origin, label="observations", legend=:topleft, size=(600,400), ms=2, alpha=.5)
# 	plot!(-0.6:0.1:1.1, (x) -> true_w[1] + true_w[2]*x, lw=3, lc=2, label=L"\mu(x)", title="Heterogeneous noise", legendfontsize=11)
# 	xis = collect(range(-.5, 1; length = 8))
# 	push!(xis, x0i_hetero)
# 	for i in 1:length(xis)
# 		x = xis[i]
# 		μi = dot(true_w, [1, x])
# 		σ = exp(true_σ[1] .+ true_σ[2] * x)
# 		xs_ = μi- 4 * σ :0.1:μi+ 4 * σ
# 		ys_ = pdf.(Normal(μi, σ), xs_)
# 		ys_ = 0.1 *ys_ ./ maximum(ys_)
# 		plot!(ys_ .+x, xs_, c=:grey, label="", linewidth=0.5)
# 		plot!([x+0.04, x+0.04], [μi - 2*σ , μi + 2*σ], st=:path, lw=1, label="", c=:gray, arrow=(:both))
# 		if i == length(xis)
# 			scatter!([x],[μi], markerstrokewidth =1, markershape = :circle, c=2, label="", series_annotations = text(L"\mu^{(i)}", 12, :purple, :bottom),markersize=4)
# 			plot!(ys_ .+x, xs_, c=:grey, label="", linewidth=2)
# 		plot!([x+0.04, x+0.04], [μi - 2*σ , μi + 2*σ], st=:path, label="", c=:green, lw=2, arrow=(:both))
# 			annotate!([x+0.06], [μi - 2*σ], text(L"\sigma^{(i)}", 12, :green, :top))
# 			# annotate!([x], [μi - σ], text(L"\sigma^{(i)}", :black, :left))
# 			# plot!(ys_ .+x, xs_, c=:red, label="", linewidth=3)
# 		end
# 	end

	
# 	plt
# end

# ╔═╡ eef1d109-e6ba-496a-a362-7a10030cb220
# md"""

# ## More specifically

# ```math
# \Large
# p(y^{(i)}|x^{(i)}) = \mathcal{N}\left (\mu(x^{(i)}),  (\sigma(x^{(i)}) )^2\right )
# ```

# * ##### where 
# $$\Large \begin{cases}  
# \mu(x) = w_1 x + w_0\\
# \sigma(x) = v_1 x + v_0 
# \end{cases}$$


# * ##### parameters: ``\mathbf{w} = [w_0, w_1]^\top`` and ``\mathbf{v} = [v_0, v_1]^\top``

# """

# ╔═╡ 53ec0760-2567-4260-9edf-25bf40824a9e
# TwoColumn(
# md"""
# ##### _Constant noise_ 
	
# $(begin
# show_img("homo_graph.svg",  w=300)
# end)

# """, md"""
# ##### _Changing noise_ 
	
# $(begin
# show_img("hetero_graph.svg",  w=320)
# end)
# """)

# ╔═╡ c4927acc-bcd5-41fa-84eb-4f03eebd53fb
# md"""

# ## However, it does not work (yet)!

# ```math
# \Large
# p(y^{(i)}|x^{(i)}) = \mathcal{N}\left (\mu(x^{(i)}),  (\sigma(x^{(i)}) )^2\right )
# ```

# * #### where 
# $$\Large \begin{cases}  
# \mu(x) = w_1 x + w_0\\
# \sigma(x) = v_1 x + v_0 
# \end{cases}$$


# * #### parameters: ``\mathbf{w} = [w_0, w_1]^\top`` and ``\mathbf{v} = [v_0, v_1]^\top``

# * ### BUT, note that ``\sigma>0``
# ```math
# \Large
# \sigma(x)= v_0+v_1x\; \text{ is not always positive!}
# ``` 
# """

# ╔═╡ 0813d6e2-e61d-471a-b8a2-2e11867e5f12
# md"""

# ## Second attempt

# ```math
# \Large
# p(y^{(i)}|x^{(i)}) = \mathcal{N}\left (\mu(x^{(i)}),  (\sigma(x^{(i)}) )^2\right )
# ```

# * #### where 
# $$\Large \begin{cases}  
# \mu(x) = w_1 x + w_0\\
# \sigma(x) = \colorbox{pink}{$\exp$}{(v_1 x + v_0)}
# \end{cases}\;\;\;\; \text{or}\;\;\;\begin{cases}  
# \mu(x) = w_1 x + w_0\\
# \sigma(x) = \colorbox{pink}{softplus}{(v_1 x + v_0)}
# \end{cases}$$

# * #### apply a ``\mathbb{R}\rightarrow \mathbb{R}^+`` activation function, 
#   * ##### *e.g.* exponential ``e^x`` or softplus ``\texttt{softplus}(x) = \ln(1+e^x)``



# """

# ╔═╡ d4daaa75-46e5-4ea1-b95f-5299c2d37d89
# let

# 	plt1 = plot(-3:0.1:3, exp, framestyle=:origin, lw=2, label=L"e^x", xlabel="x",ratio=0.5, title="Exponential function "*L"e^x", ylim =[0, 20])

# 	plt2 = plot(-10:0.1:10, softplus,lw=2,label="softplus(x)",  ratio=2, xlabel="x", framestyle=:origin, title="Softplus "*L"\ln(1+e^x)", ylim =[0, 10])

# 	plot(plt1, plt2, size=(600,300))
# end

# ╔═╡ 8364684c-fda9-4d87-bf82-20e2838f54c7
# md"""

# ## Loss & Learning



# #### The loss naturally emerges again, just `nll`

# ```math
# \Large
# \begin{align}
# \mathcal{L}(\mathbf{w}, \mathbf{v}) 
# &=- \sum_{i=1}^n \ln \mathcal{N}(y^{(i)}; \mu^{(i)}, (\sigma^{(i)})^2)
# \end{align}
# ```

# * #### where ``\mu^{(i)} = \mathbf{w}^\top\mathbf{x}^{(i)}``, ``\sigma^{(i)}=\exp\{ \mathbf{v}^\top\mathbf{x}^{(i)}\}`` 

# * #### learning: gradient descent
# """

# ╔═╡ d10fe9b3-ba7c-43cd-bfe9-f1d2f3043d70
# let
# 	Random.seed!(123)
# 	n0 = 200
# 	xs = range(-0.5, 1; length = n0)
# 	true_w = [1, 3]
# 	true_σ = [0, 1]
# 	lnσ = true_σ[1] .+ true_σ[2] * xs 
# 	ys = true_w[1] .+ 3 * xs +  exp.(lnσ).* randn(n0)
# 	plt = plot(xs, ys, st=:scatter, framestyle=:origin, label="observations", legend=:topleft, size=(600,400), ms=2, alpha=.5)
# 	xs_new = -0.52:0.05:1.05

# 	μs = true_w[1] .+ true_w[2] * xs_new
# 	σs = exp.(true_σ[1] .+ true_σ[2] * xs_new )
	
# 	plot!(xs_new, μs, lw=3, lc=1, label="fitted: " * L"\mu(x)", ribbon = 2*σs, c=1, fillalpha=0.2, title="Fitted model")
	

# 	plot!(xs_new, μs + 2*σs, lw=1, lc=1, label=L"\mu(x) +2 \sigma(x)")

# 	plot!(xs_new, μs - 2*σs, lw=1, lc=1, label=L"\mu(x) -2 \sigma(x)")
# 	# xis = collect(range(-.5, 1; length = 8))
# 	# push!(xis, x0i_hetero)
# 	# for i in 1:length(xis)
# 		# x = xis[i]
# 		# μi = dot(true_w, [1, x])
# 		# σ = exp(true_σ[1] .+ true_σ[2] * x)
# 		# xs_ = μi- 4 * σ :0.1:μi+ 4 * σ
# 		# ys_ = pdf.(Normal(μi, σ), xs_)
# 		# ys_ = 0.1 *ys_ ./ maximum(ys_)
# 		# plot!(ys_ .+x, xs_, c=:grey, label="", linewidth=0.5)
# 		# plot!([x+0.04, x+0.04], [μi - 2*σ , μi + 2*σ], st=:path, lw=1, label="", c=:gray, arrow=(:both))
# 		# if i == length(xis)
# 		# 	scatter!([x],[μi], markerstrokewidth =1, markershape = :circle, c=2, label="", series_annotations = text(L"\mu^{(i)}", 12, :purple, :bottom),markersize=4)
# 		# 	plot!(ys_ .+x, xs_, c=:grey, label="", linewidth=2)
# 		# plot!([x+0.04, x+0.04], [μi - 2*σ , μi + 2*σ], st=:path, label="", c=:green, lw=2, arrow=(:both))
# 		# 	annotate!([x+0.06], [μi - 2*σ], text(L"\sigma^{(i)}", 12, :green, :top))
# 		# 	# annotate!([x], [μi - σ], text(L"\sigma^{(i)}", :black, :left))
# 		# 	# plot!(ys_ .+x, xs_, c=:red, label="", linewidth=3)
# 		# end
# 	# end

	
# 	plt
# end

# ╔═╡ 09fc8d85-1e52-4339-806c-8de89031bc13
# begin

# 	xs_ = 0- 4  :0.1:0+ 4
# 	ys_ = pdf.(Normal(0, 1), xs_)
# 	# ys_ = 0.1 *ys_ ./ maximum(ys_)
# 	plot(-ys_ , xs_, c=:grey, label="", linewidth=1, fill=true, alpha=0.4, framestyle=:none, ratio=0.2)

# 	max_x = pdf(Normal(0, 1), 0)
# 	plot!([-max_x-0.05, 0], [0, 0], ls=:dash, lc=:green, lw=1.5, label="")


# 	plot!( [- max_x *2 /3, - max_x *2 /3], [0-1.5, 0+1.5], st=:path, lc=1, lw=1.5, arrow=:both, label="")


# 	annotate!([- max_x *2 /3], [-1.5], text(L"\sigma", 22,  :blue, :top))

# 	annotate!([- max_x-0.1], [0], text(L"\mu", 22,  :green, :right), size=(120, 300))
# end

# ╔═╡ 4c7e30b8-5332-4bf3-a23b-4e5c49580ed4
md"""

# Appendix
"""

# ╔═╡ 53a10970-7e9b-4bd4-bd65-b22028b66835
begin
	# define a function that returns a Plots.Shape
	rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
end;

# ╔═╡ 5a847577-6ea2-4346-8303-348d12dafb35
TwoColumn(md"""

**Least squared method**'s loss is sum of squared errors (SSE)


$$\large \text{loss}(\mathbf{w}) = \frac{1}{2}\sum_{i=1}^n (y^{(i)}- \mathbf{w}^\top \mathbf{x}^{(i)})^2$$


> Do we have any **other** _justification_ for the loss? 

* yes! *probabilistic linear regression model*


""", let
	gr()
	Random.seed!(123)
	n = 10
	w0, w1 = 1, 1
	Xs = range(-2, 2, n)
	ys = (w0 .+ Xs .* w1) .+ randn(n)/1
	Xs = [Xs; 0.5]
	ys = [ys; 3.5]
	plt = plot(Xs, ys, st=:scatter, markersize=3, alpha=0.5, label="", xlabel=L"x", ylabel=L"y", ratio=1, title="SSE loss")
	plot!(-2.9:0.1:2.9, (x) -> w0 + w1 * x , xlim=[-3, 3], lw=2, label="", legend=:topleft, framestyle=:axes, size=(300,300))
	ŷs = Xs .* w1 .+ w0
	for i in 1:length(Xs)
		plot!([Xs[i], Xs[i]], [ys[i], ŷs[i] ], lc=:gray, lw=1.5, label="")
		iidx = i
			if (ys[iidx] -  ŷs[iidx]) > 0 
		li = -(ŷs[iidx] - ys[iidx] )
		plot!(rectangle(li, li, Xs[iidx], ŷs[iidx]), lw=2, color=:gray, opacity=.5, label="")
	else
		li = -(ŷs[iidx] - ys[iidx] )
		plot!(rectangle(abs(li), li, Xs[iidx], ŷs[iidx]), lw=2, color=:gray, opacity=.5, label="")
		# annotate!(.5*(Xs[iidx] + abs(li)), 0.5*(ys[iidx] + ŷs[iidx]), text(L"(y^i - h(x^{(i)}))^2", 10, :black ))

	end
	end


	plt
end)

# ╔═╡ 73d51e22-3e40-4cbb-b649-65b719036647
# md"""

# ## (Independent) multivariate Gaussian


# Consider a ``2\times 1`` random vector 

# ```math
# \mathbf{x} = \begin{bmatrix} x_1\\ x_2 \end{bmatrix}
# ```


# If we assume each element is a zero mean univariate Gaussian *i.e.*

# ```math
# x_1 \sim \mathcal{N}(0, \sigma_1^2)\;\;x_2 \sim \mathcal{N}(0, \sigma_2^2)
# ```

# If we further assume they are independent, the joint probability distribution ``p(\mathbf{x})`` is

# $$\begin{align}p(\mathbf{x}) =p(x_1)p(x_2) 
# &= \underbrace{\frac{1}{\sqrt{2\pi}\sigma_1}\exp\left [-\frac{1}{2} \frac{(x_1-0)^2}{\sigma_1^2}\right ]}_{{p(x_1)}} \cdot \underbrace{\frac{1}{\sqrt{2\pi}\sigma_2}\exp\left [-\frac{1}{2} \frac{(x_2-0)^2}{\sigma_2^2}\right ]}_{p(x_2)} \\
# &= \frac{1}{(\sqrt{2\pi})^2 \sigma_1 \sigma_2} \exp{\left \{ -\frac{1}{2} \left (\frac{x_1^2}{\sigma_1^2}+\frac{x_2^2}{\sigma_2^2}\right ) \right\}}
# \end{align}$$


# Generalise the idea to ``n`` dimensional ``\mathbf{x} \in R^n``

# $$\begin{align}p(\mathbf{x}) 
# &= \frac{1}{\left (\sqrt{2\pi} \right )^n \prod_i \sigma_i} \exp{\left \{ -\frac{1}{2} \sum_{i=1}^n\frac{1}{\sigma_i^2} x_i^2 \right\}}
# \end{align}$$
# """

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
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
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
Distributions = "~0.25.107"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.1"
Latexify = "~0.16.1"
LogExpFunctions = "~0.3.26"
Plots = "~1.39.0"
PlutoTeachingTools = "~0.2.14"
PlutoUI = "~0.7.54"
StatsBase = "~0.34.2"
StatsPlots = "~0.15.6"
Zygote = "~0.6.69"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.7"
manifest_format = "2.0"
project_hash = "de4a5b39c61b66f478c5422fd11e046d6bffb671"

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

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

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

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "0f14a5456bdc6b9731a5682f439a672750a09e48"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.0.4+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

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

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

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

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "cc0a5deefdb12ab3a096f00a6d42133af4560d71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.2"

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
version = "0.8.5+0"

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

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
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

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

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
# ╠═94a408c8-64fe-11ed-1c46-fd85dc9f96de
# ╟─8fd91a2b-9569-46df-b0dc-f29841fd2015
# ╟─e13a748a-61aa-4f3d-9d1c-98a27d6a8d0a
# ╟─82216880-0550-4dc8-abce-783d39e404a9
# ╟─cde253e9-724d-4d2b-b82a-e5919240ddd3
# ╟─d5a707bb-d921-4e13-bad0-8b8d03e1852a
# ╟─6df53306-f00a-4d7e-9e77-0b717f016f06
# ╟─50debe1f-dd65-42a2-a276-9814b43b6882
# ╟─29997b58-b32b-455e-83d6-42ba628694f6
# ╟─de783288-05d7-47f0-a5ae-8c8e668015ac
# ╟─1ca89046-70dc-4850-b863-47f99ce6684d
# ╟─54bf4146-3653-4caa-9544-e7d95fae9ee5
# ╟─986c6c21-220e-4ccf-ba60-c30baf46e3c2
# ╟─96a21111-92e4-42ed-8afc-9f39042a5759
# ╟─94b6d065-d774-4035-bc35-107e30feba63
# ╟─5a847577-6ea2-4346-8303-348d12dafb35
# ╟─9ad2f1ce-cd76-4e33-ac87-e847d1841bfb
# ╟─2c7f2bad-10f2-4705-9dc8-549752713a03
# ╟─33f90867-39e0-4b09-8e33-eaccbc5f6e62
# ╟─fc0f92ba-784b-4be6-84c6-c7faa4551ee7
# ╟─d1e601fd-62d9-4b37-8fdf-352dd82ca9d6
# ╟─10104f42-424d-41c1-b5f9-1a5a63cd97bc
# ╟─3c508c8e-04ae-481c-8407-a77884479aa1
# ╟─0415f7b6-2717-4ca7-920f-03a202dac6a2
# ╟─40506d38-e87a-4411-ac9d-2ce932ec3dd2
# ╟─d456e90d-467a-4063-a97f-113677646e78
# ╟─441b0b82-6c1c-4d51-ad88-c65857d11618
# ╟─9aa580fe-f42b-4600-bc88-4a22057f30cc
# ╟─33851131-9889-4d5f-80be-c91b062f6eae
# ╟─cd48a047-f803-4df1-9b9f-1d422579591e
# ╟─c406b7a7-6917-4936-92a1-bd52049fec68
# ╟─a62ec2a1-50b2-468d-bfc0-79b998a2f6fe
# ╟─f32ba09b-dbce-4bab-a2a0-7b935b56ab22
# ╟─6d4d6876-5e7a-469a-86d4-cd7a2bc237dc
# ╟─2d11f0fa-3174-4f2a-80aa-3b0eaa5eee57
# ╟─c1614569-e9ab-40a3-b34b-cf2242eb5ffd
# ╟─ec77faee-0082-4b77-ad62-f9aac4ccb77e
# ╟─0cfdb61c-016a-4b44-a15c-9f6d47e2f469
# ╟─3ebaa9fd-994b-4606-a0ed-e2b303cf6d6b
# ╟─5f9fe4e9-db6e-4f48-861e-362dfff220c6
# ╟─a810b24a-94b1-48a5-b425-0de998504d5c
# ╟─3354023d-9328-4708-9834-9ab1029743c7
# ╟─6b5aaf90-aa6a-4409-aebb-78a6771b0064
# ╟─8be44433-4791-4e5d-bf6c-6c4727822a00
# ╟─244d3782-bcc2-4f07-971d-12c2bb748463
# ╟─06223805-b9c7-454b-9e75-31364bf50dfd
# ╟─4628c200-f72a-45ff-8759-6dd60593d45b
# ╟─53c16406-4db3-490d-9313-699503d55a58
# ╟─cb44fd76-286d-47a8-9b24-c793e7144977
# ╟─07cdc6e2-376e-432f-a7d9-a8729ccad3b3
# ╟─0a8a31fe-6220-4eaa-8a51-1e8452db01b3
# ╟─42577101-7eb0-4c94-8624-52d8ad8073a5
# ╟─1dbd6027-626d-46aa-b25a-1b26ba9270ae
# ╟─67cf29ea-45c7-4454-9e36-4efa316f5a57
# ╟─a1045431-6d95-49bb-b5a8-81758df2b883
# ╟─dd2066bd-650d-463b-97b2-b8abf15bf589
# ╟─5c7cdb6e-8218-446f-93dc-de313a0db997
# ╟─1b470ba6-8f0e-4bde-af9c-0c4685fb2bd2
# ╟─f7c4dd1d-8df8-4600-97c6-42939ebf1eb8
# ╟─922f3ff1-b74e-47a1-8ee1-fa6de5015e4f
# ╟─e37002c5-b0a8-41bb-86dd-6c19db385661
# ╟─ee8ca729-8011-49b6-96fa-89c873261ca0
# ╟─b402fb18-9157-4099-8ed3-6d50f0238614
# ╟─58932bb0-b94b-4fbd-8170-0717fb24b7b2
# ╟─83c226e6-7b59-4f75-beee-09eb8af39429
# ╟─b85a28d3-f221-4c02-b4d6-fa2ae0492600
# ╟─2839d19e-5fd9-4baf-bb0a-8eaa99e100b6
# ╟─e7cd15e6-b73e-4528-b2d4-c30933e352b5
# ╟─6fbe1186-ba88-4d37-8def-41eae484cbfc
# ╟─51e878bc-0f82-459e-a95c-f197c732efc5
# ╟─e71aadbe-150a-4d6e-b9b3-f6e747adefe3
# ╟─d1428f56-4d6a-4435-9835-eb944941d369
# ╟─eef1d109-e6ba-496a-a362-7a10030cb220
# ╟─53ec0760-2567-4260-9edf-25bf40824a9e
# ╟─c4927acc-bcd5-41fa-84eb-4f03eebd53fb
# ╟─0813d6e2-e61d-471a-b8a2-2e11867e5f12
# ╟─d4daaa75-46e5-4ea1-b95f-5299c2d37d89
# ╟─8364684c-fda9-4d87-bf82-20e2838f54c7
# ╟─d10fe9b3-ba7c-43cd-bfe9-f1d2f3043d70
# ╟─09fc8d85-1e52-4339-806c-8de89031bc13
# ╟─4c7e30b8-5332-4bf3-a23b-4e5c49580ed4
# ╟─53a10970-7e9b-4bd4-bd65-b22028b66835
# ╟─73d51e22-3e40-4cbb-b649-65b719036647
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
