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
	using StatsPlots
	using LogExpFunctions
end

# ╔═╡ 50752620-a604-442c-bf92-992963b1dd7a
using Images

# ╔═╡ 3b4a2f77-587b-41fd-af92-17e9411929c8
using GaussianProcesses

# ╔═╡ 1afdb42f-6bce-4fb1-860a-820d98df0f9d
using Distributions

# ╔═╡ ef112987-74b4-41fc-842f-ebf1c901b59b
using StatsBase

# ╔═╡ f5cf6163-ea7b-4e47-98c8-f862e3a2ebd7
using DataFrames

# ╔═╡ 3e2e1ea8-3a7d-462f-ac38-43a087907a14
TableOfContents()

# ╔═╡ 12d6f93e-24c7-470f-a54b-4947a56480ec
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ 7e53d524-bffc-4648-ba36-22a3f4dd1eee
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

# CS5014 Machine Learning


#### Probability theory & Maximum Likelihood Estimation
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 7091d2cf-9237-45b2-b609-f442cd1cdba5
md"""

## This lecture



* #### Probability theory
  * ###### random variable
  * ###### probability distribution
\


* #### Maximum Likelihood Estimation: 
  $$\Large\hat{h}_{MLE} \leftarrow \arg\max_h \ln P(\mathcal{D}|h)$$
  * ###### a principled approach to define loss

"""

# ╔═╡ adacdf8a-bc6a-4805-9372-724eb73e7620
md"""

## Reading & references

##### Essential reading 


* **Probability Theory** [_Deep Learning_ by _Ian Goodfellow and Yoshua Bengio and Aaron Courvill_: Chapter 3.1-3.3; 3.9](https://www.deeplearningbook.org/contents/prob.html)


* **MLE** [_Deep Learning_ by _Ian Goodfellow and Yoshua Bengio and Aaron Courvill_: Chapter 5.5](https://www.deeplearningbook.org/contents/ml.html)


"""

# ╔═╡ fbfb86e2-3a6d-4127-9dc9-823bbd7aa26b
md"""

## Why probability theory?



!!! note ""
	##### ``\;\;\;\;\;\;\;\;\;\;`` _*Uncertainty*_ is _everywhere_ in Machine Learning





##### (Some) _sources of uncertainty_


"""

# ╔═╡ 3d99ee38-80cc-47f4-8b70-a6253c78e3bd
TwoColumn(
md"""



* Data is **_Noisy_** 


* _**Model**_ is uncertain



"""
, md"""

* **_Prediction_** is uncertain


* _**Changing**_ environment


* and more ...

"""

	
)

# ╔═╡ 5bd15469-e888-4505-a53d-49fef3329ea4
md"Add data ``y``: $(@bind add_ys CheckBox(false)), Highlight noise: $(@bind add_noises CheckBox(false)), Add linear ``h(x)``: $(@bind add_lin CheckBox(default=false)),
Add other fits: $(@bind add_gp CheckBox(default=false)),
Add prediction uncertainty: $(@bind add_intv CheckBox(default=false))"

# ╔═╡ 9e27864c-f5cb-4780-bcd3-e3d29a69742a
Foldable("Prediction with a distribution rather than a point estimate.", md"""

Ideally, we want to predict with a probabilistic distribution:

```math
\large
p(y_{test}|x_{test})
```
""")

# ╔═╡ c9e0eaae-b340-434e-bdc9-dfdbc747221e
let
	Random.seed!(111)
	# Generate random data for Gaussian process
	nobs = 6
	x = [0.5, 1.0, 1.5, 4.5, 5.6, 6]
	f(x) =  .75 * x + sin(x)
	y = f.(x) + 0.01 * rand(nobs) + randn(nobs)/1.3

	plt = plot()
	
	xs = 0:0.05:2π
	plot!(xs, x -> f(x), color=:blue, lw=2, xlabel="x", ylabel="target", label="true function: "*L"h(x)", legendfontsize=13)
	if add_ys
		plot!(x, y, st=:scatter, label="", markershape=:circle, markersize= 7,  c=1)
		if add_noises
			[plot!([xi, xi], [f.(xi), yi], label="", c=:gray, ls=:dash, lw=2, st=:path) for (xi, yi) in zip(x, y)]
		end
	end
	# Set-up mean and kernel
	se = SE(0.0, 0.0)
	m = MeanZero()
	
	# Construct and plot GP
	gp = GP(x, y, m, se, log(1/2.5))

	# plot(gp;  xlabel=L"x", ylabel=L"y", title="Gaussian process", legend=false, xlim =[0, 2π])
	
	samples = rand(gp, xs, 10)
	w0, w1 = [ones(length(x)) x] \ y

	if add_lin
		plot!(xs, (x) -> w0 + w1*x, lw=2, lc=:green, label="linear fit")
	end
	if add_gp
		plot!(xs, samples, lw=1.5, ls=:dash, label="", alpha=0.8)

		if add_intv
			plot!(gp; obsv=false, lw=1.5, label="mean estimation of: "*L"h(x)")
		end
	end



	# μ, σ² = predict_y(gp, xs);

	# plot!(xs, μ, ribbon=σ²,fillalpha=.5)

	plt
end

# ╔═╡ 1e927760-4dc6-4ecf-86ec-8fed16dbb0d6
md"""

## Why probability theory?


!!! important ""
	##### ``\;\;\;\;\;\;\;\;\;\;`` Without _Probability Theory_, ML is beyond _useless_! 

#### _As an example,_


"""

# ╔═╡ eb8af12f-ca58-4c51-a434-0925c099ffa3
show_img("airp1.svg")

# ╔═╡ 5ae8261b-283f-4bdf-b188-07f62f4c339e
md"""

## Why probability theory? (cont.)


!!! important ""
	##### ``\;\;\;\;\;\;\;\;\;\;`` Without _Probability Theory_, ML is beyond _useless_! 

#### _As an example,_


"""

# ╔═╡ 95c88b47-f702-44b0-bbb2-f305132a2db0
show_img("airp2.svg")

# ╔═╡ 1d62b70b-c30c-458a-9d19-835685d38d12
md"""

## Why probability theory? (cont.)


!!! important ""
	##### ``\;\;\;\;\;\;\;\;\;\;`` Without _Probability Theory_, ML is beyond _useless_! 

#### _As an example,_ 

* ###### _none of the prediction would be 100% accurate_


"""

# ╔═╡ ba8fac78-31b1-427c-8f78-bc545751be41
show_img("airp3.svg")

# ╔═╡ dd2c51d5-2a49-44db-801b-4c37f32ce001
md"""

## Why probability theory? (cont.)


!!! important ""
	##### ``\;\;\;\;\;\;\;\;\;\;`` Without _Probability Theory_, ML is beyond _useless_! 

#### _As an example,_ 

* ###### _none of the prediction would be 100%_
* ###### _but would be useful if we attach prediction uncertainty_

```math
\large
p(y_{T+h} |\mathbf{y}_{1:T}): \;\; \text{Prediction with a distribution}
```

"""

# ╔═╡ 1ea7dad6-1a8d-40e4-8248-069e18d6e646
show_img("airp4.svg")

# ╔═╡ 59a38e32-c2f3-465f-928d-c05f8d69f496
md"""

## Prediction is uncertain -- classification

##### Instead of predicting a _point estimator_ ``\hat{y}_{test} \in \{\texttt{cat}, \texttt{dog}\}``, better to have
```math
\large
P(y_{test}|\mathbf{x}_{test}) = \begin{bmatrix}
\cdot \\
\cdot\\
\cdot\\
\end{bmatrix}
\begin{array}{l}
\texttt{cat}\\
\texttt{dog}\\
\texttt{others}\\
\end{array}
```

"""

# ╔═╡ a7a24713-a29a-4f0c-996b-f98305bac09c
md"""

## Prediction is uncertain -- classification


```math

P(y_{test}|\mathbf{x}_{test}) = \begin{bmatrix}
\cdot \\
\cdot\\
\cdot\\
\end{bmatrix}
\begin{array}{l}
\texttt{cat}\\
\texttt{dog}\\
\texttt{others}\\
\end{array}
```
"""

# ╔═╡ 2ce6c56b-733c-42e8-a63b-d774cb6c199c
md"""

## Prediction is uncertain -- classification


```math

P(y_{test}|\mathbf{x}_{test}) = \begin{bmatrix}
\cdot \\
\cdot\\
\cdot\\
\end{bmatrix}
\begin{array}{l}
\texttt{cat}\\
\texttt{dog}\\
\texttt{others}\\
\end{array}
```
"""

# ╔═╡ 2e4df75b-0778-4ed4-840a-417da2d65204
md"""

## Prediction is uncertain -- classification


```math

P(y_{test}|\mathbf{x}_{test}) = \begin{bmatrix}
\cdot \\
\cdot\\
\cdot\\
\end{bmatrix}
\begin{array}{l}
\texttt{cat}\\
\texttt{dog}\\
\texttt{others}\\
\end{array}
```
"""

# ╔═╡ c5be7eb8-e0b6-48cc-8dbe-788fa6624999
Hs_catdogs = ["Cat", "Dog", "Others"];

# ╔═╡ 81ab9972-07bc-4ce9-9138-3359d4e34025
plt1, plt2, plt3=let
	ps = [.9, .05, .05]
	texts = [Plots.text(L"%$(p)", 10) for (i, p) in enumerate(ps)]
	plt_cat = plot(ps, fill = true, st=:bar, xticks=(1:3, Hs_catdogs),  ylim =[0, 1.0], label="", title="", color =:orange,  texts = texts,size=(200,200))


	ps = [.05, .9, .05]
	texts = [Plots.text(L"%$(p)", 10) for (i, p) in enumerate(ps)]
	plt_dog=plot(ps, fill = true, st=:bar, xticks=(1:3, Hs_catdogs),  ylim =[0, 1.0], label="", title="", color =:orange,  texts = texts,size=(200,200))


	ps = [.25, .25, .5]
	texts = [Plots.text(L"%$(p)", 10) for (i, p) in enumerate(ps)]
	plt_dontknow=plot(ps, fill = true, st=:bar, xticks=(1:3, Hs_catdogs),  ylim =[0, 1.0], label="", title="", color =:orange,  texts = texts,size=(200,200))

	plt_cat, plt_dog, plt_dontknow
end;

# ╔═╡ e8fd61f1-33a6-43d8-8056-fb7cf97291b5
ThreeColumn(md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/cat1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```
$(plt1)

""", md"""

""",
	md"""

""")

# ╔═╡ fc9e9bb6-2287-46c8-8518-c9d0804c094e
ThreeColumn(md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/cat1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt1)
""", md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/dog1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt2)
""",
	md"""


""")

# ╔═╡ 8730b9a2-a1b4-456c-974c-ecd8880e6834
ThreeColumn(md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/cat1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt1)
""", md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/dog1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt2)
""",
	md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/catdog1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{???}
```
$(plt3)

""")

# ╔═╡ dc8a3e36-2021-42dd-bc49-0eb6ab784fac
ThreeColumn(md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/cat1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt1)
""", md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/dog1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt2)
""",
	md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/catdog1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{???}
```

$(plot([.0, .0, .0], fill = true, st=:bar, xticks=(1:3, Hs_catdogs),  ylim =[0, 1.0], label="", title="", size=(200,200)))
""")

# ╔═╡ 2f7003a0-8dc8-4c2b-a3f3-0ff4a089e8f7
md"""
## One formula for all ML

```math
\LARGE

p(y_{test}|x_{test})
```


#### I hope you appreciate how _neat_ it is

* ##### no need to differentiate _classification_, _regression_ any more
* ##### even no need to differentiate _supervised_ or _unsupervised_ learning

##

!!! note ""
	 #### _Probability theory_ provides us with 
	 * ##### _a unified approach to do (almost all) ML_
	 * ##### in other words, _probability theory explains (almost) all ML practices_
	 > #### _Probability Theory_ is the **soul** of Machine Learning
	
"""

# ╔═╡ bce5c041-be39-4ed1-8935-c389293400bc
penny_image = load(download("https://www.usacoinbook.com/us-coins/lincoln-memorial-cent.jpg"));

# ╔═╡ db6eb97c-558c-4206-a112-6ab3b0ad04c8
begin
	head = penny_image[:, 1:end÷2]
	tail = penny_image[:, end÷2:end]
end;

# ╔═╡ ff61cd9d-a193-44b3-a715-3c372ade7f79
md"# Probability theory"

# ╔═╡ 128b6ad8-aa21-4d0a-8124-2cf433bc79c4
# md"""

# ## Random variable 


# ### Let's consider ``\cancel{\textbf{random}} \textbf{variable}``  first, 

# > ##### _e.g._ a variable ``\large X= 5``

# * ##### ``X``: a _deterministic_ variable
# * ##### *with ``100\%`` certainty*, ``X`` takes the value of 5


# ##


# #### ``{\textbf{Random}}\; \textbf{variable}``: the value is _random_ 


# * ##### *probability distribution* ``P(X)``: to quantify the uncertainty of ``X``
# ```math
# \large
# \begin{equation}  \begin{array}{c|cccccc} 
#  & X = 1 & X = 2 & X = 3 & X = 4 & X = 5 & X = 6 \\
# \hline
# P(X) & 1/6 & 1/6 & 1/6 & 1/6 & 1/6 & 1/6

# \end{array} \end{equation} 

# ```

# * ``P(X=x) > 0,\; \forall x\;\; \text{and}\;\; \sum_x{P(X=x)}=1``
# """

# ╔═╡ 4d99c216-e32f-43a3-a122-ccbb697711fc
md"""
## Random variable





"""

# ╔═╡ f0028ab9-593b-4d25-a59a-e93aecd14ad4
TwoColumn(md"""


#### *Discrete random variables*

* ##### ``R:`` raining or not ?
  * ``\Omega =\{t, f\}``


* ##### ``T:`` temperature hot or cold?
  * ``\Omega =\{hot, cold\}``

""" , md"""

#### *Continuous random variables*
* ##### ``W \in [0, +\infty)`` weight of a dog
  * ``\Omega = [0, \infty)``


* ##### ``T \in (-\infty, +\infty)`` temperature in Celcius
  * ``\Omega = [-\infty, \infty)``

""")

# ╔═╡ 403af436-d7f2-43c0-803a-8104ba69fcfd
md"""
## Probability distributions 

#### -- _discrete random variable_


#### *Probability distributions*: requirement

$$\large P(X=x) \geq 0,\; \forall x\;\; \text{and}\;\; \sum_x{P(X=x)}=1$$
* ##### non-negative and sum to one





"""

# ╔═╡ 68742b9e-e94e-40fb-a120-0c83f99847ce
TwoColumn(md"""
**Temperature** $T: P(T)$

```math

\begin{equation}  \begin{array}{c|c} 
T & P(T)\\
\hline
hot & 0.5 \\
cold & 0.5
\end{array} \end{equation} 
```

""", md"""

**Weather** $W: P(W)$

```math

\begin{equation}  \begin{array}{c|c} 
W & P(W)\\
\hline
sun & 0.6 \\
rain & 0.1 \\
fog & 0.2 \\
snow & 0.1
\end{array} \end{equation} 
```

""")

# ╔═╡ 5b500acf-7029-43ff-9835-a26d8fe05194
# md"""
# ## Notation

# - Capital letter $X,Y, \texttt{Pass}, \texttt{Weather}$ are random variables


# - Smaller letters $x,y, +x, -y, \texttt{true, false, cloudy, sunny}$ are particular values r.v.s can take  


# - Notation: $P(x)$  is a shorthand notation for $P(X=x)$


# - So ``P(X)`` is assumed to be a distribution, but ``P(x)`` is a number

# Therefore, $P(W)$ means a full distribution vector

# ```math

# \begin{equation}  \begin{array}{c|c} 
# W & P(W)\\
# \hline
# sun & 0.6 \\
# rain & 0.1 \\
# fog & 0.2 \\
# snow & 0.1
# \end{array} \end{equation} 
# ```

# But ``P(sum)`` is a number

# ```math
# P(sun) = P(W=sum) = 0.6 
# ```


# """

# ╔═╡ 4bf768de-833f-45bf-9429-4820ff61553f
# md"""

# ## Examples of r.v.s

# | Variable  | Discrete or continous| Domain ``\, \Omega`` |
# | :---|:---:|:---:|
# | Toss of a coin | Discrete | ``\{0,1\}`` |
# | Roll of a die | Discrete |``\{1,2,\ldots, 6\}`` |
# | Outcome of a court case | Discrete |``\{0,1\}`` |
# | Number of heads of 100 coin tosses| Discrete|``\{0,1, \ldots, 100\}`` |
# | Number of covid cases | Discrete|``\{0,1,\ldots\}`` |
# | Height of a human | Continuous |``\mathbb{R}^+=(0, +\infty)`` |
# | The probability of coin's bias ``\theta``| Continuous|``[0,1]`` |
# | Measurement error of people's height| Continuous|``(-\infty, \infty)`` |
# """

# ╔═╡ 656da51f-fd35-4e89-9af5-b5f0fdf8618f
md"""
##  Discrete r.v. -- Bernoulli 




"""

# ╔═╡ 80038fee-b922-479d-9687-771e7e258fcf
md"Model parameter ``\theta``: $(@bind θ Slider(0:0.1:1, default=0.5; show_value=true))"

# ╔═╡ 7c03a15f-9ac1-465f-86a4-d2a6087e5970
TwoColumn(md"""
#### *Bernoulli*: ``X`` is binary

```math
\large
P(X ) =\begin{cases}\theta & x= 1 \\ 1-\theta & x=0 \end{cases}
```

* ##### for example, coin tossing 

##### The parameter:
```math
\Large 0\leq \theta \leq 1
```


##### Note that 
* ``0 \leq P(x) \leq 1; \text{and}\; \sum_{x=0,1}P(x) = \theta + 1-\theta = 1``


""", 

	begin
		bar(Bernoulli(θ), xticks=[0,1], xlabel=L"X", ylabel=L"P(X)", label="", ylim=[0,1.0], size=(250,300), title="Bernoulli dis.")
	end
)

# ╔═╡ e28e8089-f52b-440a-9861-895f9c378c84
md"""
## Discrete r.v. -- Bernoulli 

#### Probability distribution in one--line

```math
\Large 
\begin{align}
P(X=x) &=\begin{cases}\theta & x= 1 \\ 1-\theta & x=0 \end{cases} \\

&=\boxed{ \theta^{x} (1-\theta)^{1-x}}
\end{align}
```

##### ``\text{for}\; x\in \{0, 1\}``
* ##### ``x=0``: ``P(X=0) = \theta^{0} (1-\theta)^{1-0} = \underbrace{\theta^0}_{1}\cdot (1-\theta)= 1-\theta``
* ##### ``x=1``: ``P(X=1) = \theta^{1} (1-\theta)^{1-1} = \theta\cdot (1-\theta)^0= \theta``

"""

# ╔═╡ 1e52d388-1e8d-4c20-b6e7-bcdd674ea406
md"""
## Discrete r.v. -- Categorical (Multinoulli)




"""

# ╔═╡ 176afbd7-fb8c-4710-9e7d-bde3c16e9a2d
md"Choose ``k``: $(@bind kk Slider(1:6)) ; Model parameter ``\theta_k``: $(@bind θk Slider(0:0.001:1, default=1/6; show_value=true))"

# ╔═╡ c2d14bf1-f16c-4573-a862-4d2dc73d39c0
begin
	θs = ones(6) * (1 - θk)/5
		θs[kk] = θk
	plt_categorical=bar(Categorical(θs), xlabel=L"X", ylabel=L"P(X)", label="", ylim=[0,1.0], size=(330,300), title="Categorical dis.")

end;

# ╔═╡ 0623a15a-4361-4dda-86b6-782a2c9bc609
TwoColumn(md"""
#### *Categorical* ``X`` (``K`` categories)

```math
\large
P(X ) =\begin{cases}\theta_1 & x= 1 \\ \theta_2 & x=2 \\
\vdots \\

\theta_K & x=K
\end{cases}
```

* ##### for example, dice rolling

##### The parameter:
```math
\boldsymbol{\theta}= [\theta_1, \theta_2, \ldots, \theta_K]
```



* ``\theta_k \geq 0`` and ``\sum_k \theta =1``




""", 

	begin
		plt_categorical
	end
)

# ╔═╡ c2f2b12f-39a1-47aa-839f-f181b0b30236
md"``\boldsymbol{\theta}=``$(latexify_md(round.(θs; digits=3)))"

# ╔═╡ d0801b01-062c-4fcd-921b-84cf37ddc447
md"""
## Discrete r.v. -- Categorical  

#### Probability distribution in one--line

```math
\large
\begin{align}
P(X ) =\begin{cases}\theta_1 & x= 1 \\ \theta_2 & x=2 \\
\vdots \\

\theta_K & x=K
\end{cases}\;\;\Longleftrightarrow \;\; P(X =x) = \Large \prod_{k=1}^K\, \theta_k^{\mathbb{1}(x = k)}
\end{align}
```


* ##### _indicator function_: ``\mathbb{1}(\texttt{true}) = 1``, ``\mathbb{1}(\texttt{false}) = 0``
  * *e.g.* ``\mathbb{1}(1=2) = 0``, ``\mathbb{1}(1=1)=1``

## Discrete r.v. -- Categorical  

#### Probability distribution in one--line

```math
\large
\begin{align}
P(X ) =\begin{cases}\theta_1 & x= 1 \\ \theta_2 & x=2 \\
\vdots \\

\theta_K & x=K
\end{cases}\;\;\Longleftrightarrow \;\; P(X =x) = \Large \prod_{k=1}^K\, \theta_k^{\mathbb{1}(x = k)}
\end{align}
```


* ##### _indicator function_: ``\mathbb{1}(\texttt{true}) = 1``, ``\mathbb{1}(\texttt{false}) = 0``
  * *e.g.* ``\mathbb{1}(1=2) = 0``, ``\mathbb{1}(1=1)=1``

##### As an example, for ``K=6``, we have

```math
\large
P(X=x) = \theta_1^{\mathbb{1}(x =1)} \cdot \theta_2^{\mathbb{1}(x =2)}  \cdot \theta_3^{\mathbb{1}(x =3)} \cdot \theta_4^{\mathbb{1}(x =4)} \cdot \theta_5^{\mathbb{1}(x =5)}\cdot \theta_6^{\mathbb{1}(x =6)}
```


* ##### to verify it, *e.g.* ``x= 2``

```math
\large
\begin{align}
P(X=2) &= \theta_1^{\mathbb{1}(2 =1)} \cdot \theta_2^{\mathbb{1}(2 =2)}  \cdot \theta_3^{\mathbb{1}(2 =3)} \cdot \theta_4^{\mathbb{1}(2 =4)} \cdot \theta_5^{\mathbb{1}(2 =5)}\cdot \theta_6^{\mathbb{1}(2 =6)}\\
\end{align}
```


"""

# ╔═╡ 21a88f9c-5153-4c65-80fd-80aee753971c
md"""

## Discrete r.v. -- Categorical  

#### Probability distribution in one--line

```math
\large
\begin{align}
P(X ) =\begin{cases}\theta_1 & x= 1 \\ \theta_2 & x=2 \\
\vdots \\

\theta_K & x=K
\end{cases}\;\;\Longleftrightarrow \;\; P(X=x ) = \Large \prod_{k=1}^K\, \theta_k^{\mathbb{1}(x = k)}
\end{align}
```

* ##### _indicator function_: ``\mathbb{1}(\texttt{true}) = 1``, ``\mathbb{1}(\texttt{false}) = 0``
  * *e.g.* ``\mathbb{1}(1=2) = 0``, ``\mathbb{1}(1=1)=1``

##### for ``K=6``, we have

```math
\large
P(X=x) = \theta_1^{\mathbb{1}(x =1)} \cdot \theta_2^{\mathbb{1}(x =2)}  \cdot \theta_3^{\mathbb{1}(x =3)} \cdot \theta_4^{\mathbb{1}(x =4)} \cdot \theta_5^{\mathbb{1}(x =5)}\cdot \theta_6^{\mathbb{1}(x =6)}
```


* ##### to verify it, *e.g.* ``x= 2``

```math
\large
\begin{align}
P(X=2) &= \theta_1^{\mathbb{1}(2 =1)} \cdot \theta_2^{\mathbb{1}(2 =2)}  \cdot \theta_3^{\mathbb{1}(2 =3)} \cdot \theta_4^{\mathbb{1}(2 =4)} \cdot \theta_5^{\mathbb{1}(2 =5)}\cdot \theta_6^{\mathbb{1}(2 =6)}\\

&= \theta_1^{0} \cdot \theta_2^{1}  \cdot \theta_3^{0} \cdot \theta_4^{0} \cdot \theta_5^{0}\cdot \theta_6^{0}\\
\end{align}
```


"""

# ╔═╡ 21fe71dc-6f9d-4945-9c41-db4ec252980d
md"""

## Discrete r.v. -- Categorical  

#### Probability distribution in one--line

```math
\large
\begin{align}
P(X ) =\begin{cases}\theta_1 & x= 1 \\ \theta_2 & x=2 \\
\vdots \\

\theta_K & x=K
\end{cases}\;\;\Longleftrightarrow \;\; P(X =x) = \Large \prod_{k=1}^K\, \theta_k^{\mathbb{1}(x = k)}
\end{align}
```



* ##### _indicator function_: ``\mathbb{1}(\texttt{true}) = 1``, ``\mathbb{1}(\texttt{false}) = 0``
  * *e.g.* ``\mathbb{1}(1=2) = 0``, ``\mathbb{1}(1=1)=1``

##### for ``K=6``, we have

```math
\large
P(X=x) = \theta_1^{\mathbb{1}(x =1)} \cdot \theta_2^{\mathbb{1}(x =2)}  \cdot \theta_3^{\mathbb{1}(x =3)} \cdot \theta_4^{\mathbb{1}(x =4)} \cdot \theta_5^{\mathbb{1}(x =5)}\cdot \theta_6^{\mathbb{1}(x =6)}
```


* ##### we can verify *e.g.* ``x= 2``

```math
\large
\begin{align}
P(X=2) &= \theta_1^{\mathbb{1}(2 =1)} \cdot \theta_2^{\mathbb{1}(2 =2)}  \cdot \theta_3^{\mathbb{1}(2 =3)} \cdot \theta_4^{\mathbb{1}(2 =4)} \cdot \theta_5^{\mathbb{1}(2 =5)}\cdot \theta_6^{\mathbb{1}(2 =6)}\\

&= \theta_1^{0} \cdot \theta_2^{1}  \cdot \theta_3^{0} \cdot \theta_4^{0} \cdot \theta_5^{0}\cdot \theta_6^{0}\\
&= 1\, \cdot \,\theta_2 \cdot \, 1 \, \, \cdot \, 1 \,\cdot \, 1 \, \,\cdot \, 1\\
&= \theta_2
\end{align}
```


"""

# ╔═╡ 946f34ae-d8c2-444f-b2d7-042d8c258e33
md"""
## Probability distributions 

#### -- _continuous random variable_


#### *Probability distributions*: requirement

$$\Large p(x) \geq 0,\; \forall x\;\; \text{and}\;\; \int{p(x)\,dx}=1$$
* ##### non-negative and _integrate_ to one

* ##### it means that _the area under the curve_ ``p(x)`` is ``1``



"""

# ╔═╡ 5fdd4820-2cd3-40bb-b1b1-9647af7156ca
md"""Add integration: $(@bind add_intg CheckBox(false))"""

# ╔═╡ 6271dfaf-9499-4561-a06e-dcf63cac33be
let

	μs = [0]
	σ²s = [1]
	plt_gaussian = Plots.plot(title="A continuous "*L"p(x)", xlabel=L"X", ylabel=L"p(X)", size=(700,300), framestyle=:zerolines)
	# for i in 1:3 
	i=1
	fill_ = false
	if add_intg
		fill_ = true
		annotate!([0], [0.18], text(L"\int p(x)dx=1", :blue, 20))
	end
	plot!(plt_gaussian, Normal(μs[i], sqrt(σ²s[i])), fill=fill_, alpha=0.5, lw=5, label=L"p(x)", legendfontsize=15)
		# vline!([μs[i]], color=i, label="", linewidth=2)
		
	
	# end
	plt_gaussian
end

# ╔═╡ a4980317-32aa-44b8-97a8-8887c0e65bb4
md"""

## 	Continuous r.v. -- Uniform


"""

# ╔═╡ 197e2d17-fd19-46b1-8f51-0fa2748340e5
TwoColumn(md"""

#### ``X``: *uniform distribution* btw ``[a, b]``

```math
\Large
p(x) = \begin{cases}\frac{1}{b-a} & a \leq x \leq b \\ 0 & \text{otherwise}\end{cases}
```

* ##### no preference over the range between ``a`` and ``b``


##### *Example*: ``a=0, b=1``, then
* ###### ``p(x) = 1\cdot \mathbb{1}(0<x<1)``



""", 
let
	a, b = 0, 1
	plot(Uniform(0,1), fill=true, alpha= 0.5, lw=2, xlim =[-0.2, 1.2], ylim=[0, 2], xlabel=L"X", ylabel=L"p(X)", label="", title=L"p(X) = \texttt{Uniform}(0,1)", size=(300,300))
end
)

# ╔═╡ 765ee957-c9fc-4144-b893-992e389d273d
md"""

```python
## To sample from a uniform distribution in Python
np.random.rand()
## in Julia
rand()
```
"""

# ╔═╡ 2bad0f9a-2b21-4686-aa41-2b430c354454
# md"""

# ## Probability with p.d.f




# """

# ╔═╡ 9594d76f-3274-48fa-b833-f0c26daa229a
# TwoColumn(md"""

# \
# \
# \


# !!! question "Question"
# 	##### What is the probability that ``X \in [0.5, 1.0]``?
# """, let
# 	a, b = 0, 1
# 	plot(Uniform(0,1), fill=true, alpha= 0.5, lw=2, xlim =[-0.2, 1.2], ylim=[0, 2], xlabel=L"X", ylabel=L"p(X)", label="")
# 	c = 0.5
# 	plot!(.5:0.1:1.0, (x)-> 1.0, fill=true, alpha=0.5, label="",title=L"p(X) = \texttt{Uniform}(0,1)", size=(300,300))
# end)

# ╔═╡ ce1d7bad-179a-48c0-86c7-2de82c55a96d
# md"""
# ##
# """

# ╔═╡ 09f78f45-3790-4218-847f-b9ea1e61176a
# TwoColumn(md"""

# ##### For continuous r.v., we compute *probability* by *integration* 


# ```math
# \begin{align}
# P(X \in [0.5, 1.0]) &= \int_{0.5}^{1} p(x) \mathrm{d}x \\
# &= \int_{0.5}^{1}1 \mathrm{d}x = 1 \cdot 0.5 = 0.5
# \end{align}
# ```
# * ##### interpretation: the shaded area is 0.5

# """, 
	
# let
# 	a, b = 0, 1
# 	plot(Uniform(0,1), fill=true, alpha= 0.5, lw=2, ylim=[0, 2], xlabel=L"X", xlim =[-0.2, 1.2], ylabel=L"p(X)", label="")
# 	c = 0.5
# 	plot!(.5:0.1:1.0, (x)-> 1.0, fill=true, alpha=0.5, label=L"\mathbb{P}(X\in [0.5, 1.0])=0.5", size=(300,300))
# end)

# ╔═╡ b89ac105-597e-44ac-9b58-c1c3c5ac59e9
md"""
## Continuous r.v. -- Gaussian



"""

# ╔═╡ 2b66deba-7144-496e-b9f2-ecf96b0e42ba
show_img("/CS5914/gaussian_eq_1d.png", w=560)

# ╔═╡ 355c71f8-975e-4a65-b1a6-7e27a67b41d8
md"""

#### Parameters: 
* ##### _location_: ``\mu`` mean
* ##### _scale_: ``\sigma`` the standard deviation


"""

# ╔═╡ 5f7bd93c-4210-49ba-8191-5167a5936a3c
let

	μs = [-3, 0, 3]
	σ²s = [1 , 2 , 5]
	plt_gaussian = Plots.plot(title="Gaussian distributions", xlabel=L"X", ylabel=L"p(X)", size=(700,350))
	for i in 1:3 
		plot!(plt_gaussian, Normal(μs[i], sqrt(σ²s[i])), fill=true, alpha=0.5, label=L"\mathcal{N}(μ=%$(μs[i]), σ^2=%$(σ²s[i]))")
		vline!([μs[i]], color=i, label="", linewidth=2)
	end
	plt_gaussian
end

# ╔═╡ c0d0cdc4-5c25-4f5d-b92f-9facc999e53e
md"""

```python
## To sample from a standard Normal distribution in Python
np.random.randn()
## in Julia
randn()
```
"""

# ╔═╡ d3f51b03-384c-428c-b7e4-bdc1508e6a02
md"""

## Dissect Gaussian

```math
\boxed{\colorbox{lightblue}{$\left(\frac{x -\mu}{\sigma}\right)^2$}} \Longrightarrow -\frac{1}{2}{\left(\frac{x -\mu}{\sigma}\right)^2} \Longrightarrow  \large{e^{-\frac{1}{2}{\left(\frac{x -\mu}{\sigma}\right)^2}} } \Longrightarrow {\frac{1}{\sigma \sqrt{2\pi}}} e^{-\frac{1}{2} {\left(\frac{x -\mu}{\sigma}\right)^2}}
```

* ``\colorbox{lightblue}{$(\frac{x-\mu}{\sigma})^2$}``: the `kernel` measures how far away ``x`` is from ``\mu``
  * measured w.r.t measurement unit ``\sigma``
  * how many ``\sigma`` units away 

"""

# ╔═╡ 875a06b7-eb90-451e-a888-3e4f13832053
md"``\mu``: $(@bind μ1_ Slider(-5:.1:5.0, default=0.0, show_value=true)),
``x``: $(@bind x1_ Slider(-5:.1:5.0, default=2.0, show_value=true))"

# ╔═╡ c2497681-0729-451a-ab5f-43937bc9e100
let

	μ = μ1_
	σ = 1.0

	f1(x) = ((x - μ)/σ )^2

	plot(range(μ-10, μ+10, 100), (x) -> f1(x), lw=2, label=L"\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= \left(({x-\mu})/{\sigma}\right)^2", framestyle=:origin, xlim =[-20, 20], ratio=0.2)

	x_ = x1_
	plot!([x_, x_], [0, f1(x_)], ls=:dot, lc=1, lw=2, label="")
	annotate!([x_], [f1(x_)], text(L"f(x)=%$(round(f1(x_); digits=2))", :blue,:right))
	annotate!([x_], [0], text(L"x_0", :blue,:top))
	annotate!([μ], [0], text(L"\mu", :red, :top))
	vline!([μ], lc=:red, lw=1.5, label=L"\mu")
end

# ╔═╡ 00c8c5a4-c58f-4a62-ba88-ca3f590977d7
md"""

## Dissect Gaussian

```math
\left(\frac{x -\mu}{\sigma}\right)^2 \Longrightarrow \boxed{\colorbox{orange}{$-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2$} }\Longrightarrow  \large{e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2} } \Longrightarrow {\frac{1}{\sigma \sqrt{2\pi}}} e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2}
```


* ``\colorbox{orange}{$-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2$} ``: ``p(x)`` is negative correlated with the distance
  * further away ``x`` is from ``\mu``, ``p(x)`` is smaller
"""

# ╔═╡ 06e178a8-bcd1-4646-8e51-1b90a2e09784
md"``\mu``: $(@bind μ2_ Slider(-5:.1:5.0, default=0.0, show_value=true)),
``x``: $(@bind x2_ Slider(-8:.1:8.0, default=2.0, show_value=true))"

# ╔═╡ ab5612b9-9681-4984-b58e-3783c0c0c6e4
let

	μ = μ2_
	σ = 1

	f1(x) = ((x - μ)/σ )^2
	f2(x) = -0.5* f1(x)
	plot(range(μ -10, μ+10, 100), (x) -> f1(x), lw=2, label=L"\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= \left(({x-\mu})/{\sigma}\right)^2", framestyle=:origin, xlim =[-10, 10], ylim = [-30, 30], ratio=0.3, legend=:outerright, legendfontsize=8)
	plot!(f2, lw=2, label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2}\left(({x-\mu})/{\sigma}\right)^2")


	x_ = x2_
	plot!([x_, x_], [0, f2(x_)], ls=:dot, lc=:red, lw=3, label="")
	annotate!([x_], [f2(x_)], text(L"f(x)=%$(round(f2(x_); digits=2))", 10, :red, :top))

	annotate!([x_], [0], text(L"x_0", :red,:top))
	annotate!([μ], [0], text(L"\mu", :red, :top))

	vline!([μ], lc=3, lw=2, label="")
end

# ╔═╡ 6e7ace1b-6c6f-44e4-8377-dd7804f94ee0
md"""

## Dissect Gaussian

```math
\left(\frac{x -\mu}{\sigma}\right)^2 \Longrightarrow -\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2 \Longrightarrow  \large{ \boxed{\colorbox{lightgreen}{$e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2} $}}} \Longrightarrow {\frac{1}{\sigma \sqrt{2\pi}}} e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2}
```


* "``\exp``": the exponential function makes sure ``p(x)>0`` for all ``x``

"""

# ╔═╡ cb3f15a1-3d04-447a-a5a2-50c66f356922
md"``\mu``: $(@bind μ3_ Slider(-5:.1:5.0, default=0.0, show_value=true)),
``x``: $(@bind x3_ Slider(-5:.1:5.0, default=2.0, show_value=true))"

# ╔═╡ 43f6f92c-fe29-484f-ad1b-18a674574ef2
let

	μ = μ3_
	σ = 1

	f1(x) = ((x - μ)/σ )^2
	f2(x) = -0.5* f1(x)
	f3(x) = exp(f2(x))
	# plot(f1, lw=2, label=L"\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= \left(({x-\mu})/{\sigma}\right)^2")
	plot(range(μ -5, μ+5, 100), (x) -> f2(x), lw=2, lc=2,label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2}\left(({x-\mu})/{\sigma}\right)^2", framestyle=:origin)
	plot!(f3, lw=2, lc=3, label=L"e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}", title=L"f(x)= e^{-\frac{1}{2}\left(({x-\mu})/{\sigma}\right)^2}", ylim=[-2,1.5])


	x_ = x3_
	plot!([x_, x_], [0, f3(x_)], ls=:dot, lc=:green, lw=2, label="")
	annotate!([x_], [f3(x_)], text(L"f(x)=%$(round(f3(x_); digits=2))",:green, :bottom))

	vline!([μ], lc=:red, lw=2, label=L"\mu")
end

# ╔═╡ 72af797b-5340-482e-be00-2cda375dd734
md"""

## Dissect Gaussian

```math
\left(\frac{x -\mu}{\sigma}\right)^2 \Longrightarrow -\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2 \Longrightarrow  \large{ e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2} } \Longrightarrow \boxed{\colorbox{pink}{${\frac{1}{\sigma \sqrt{2\pi}}} e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2}$}}
```


* ``\frac{1}{\sigma \sqrt{2\pi}}``: normalising constant, a contant from ``x``'s perspective
  * it normalise the density such that $$\int p(x)\mathrm{d}x = 1$$

"""

# ╔═╡ 723365e7-1fad-4899-8ac1-fb8674e2b9a7
md"``\mu``: $(@bind μ4_ Slider(-5:.1:5.0, default=0.0, show_value=true))"

# ╔═╡ a862e9d6-c31d-4b21-80c0-e359a5435b6b
let
	μ = μ4_
	σ = 1
	f1(x) = ((x - μ)/σ )^2
	f2(x) = -0.5* f1(x)
	f3(x) = exp(f2(x))

	f4(x) = 1/(σ * sqrt(2π)) *exp(f2(x))
	# plot(f1, lw=2, label=L"\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= \left(({x-\mu})/{\sigma}\right)^2")
	plot(range(μ -5, μ+5, 100), (x) -> f2(x), lw=2, lc=2,label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2}\left(({x-\mu})/{\sigma}\right)^2", framestyle=:origin)
	plot!(f3, lw=2, lc=3, label=L"e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}", title=L"f(x)= e^{-\frac{1}{2}\left(({x-\mu})/{\sigma}\right)^2}", ylim=[-2,1.5])

	plot!(f4, lw=2, lc=4, label=L"\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}", title=L"f(x)= \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(({x-\mu})/{\sigma}\right)^2}", ylim=[-2,1.5])

end

# ╔═╡ ce53b75a-3b3c-4ca9-87ec-10f6ec2d38b2
md"""

## Summary
"""

# ╔═╡ 1659e22a-716a-40c3-9efd-839cb94658c3
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gaussian_eq_1d_.png' width = '580' /></center>"

# ╔═╡ bfe99268-6097-494a-886f-58709e33cd01
md"""
## Laplace -- another location-scale distribution 

"""

# ╔═╡ 58192d79-796b-42ef-8ed0-09b0b0473066
TwoColumn(md"""

```math
\large
p(x; \mu, \sigma) = \frac{1}{2\sigma} \exp \left (- \frac{|x-\mu|}{\sigma}\right )
```
##### _Parameters_: 
* ###### _location_: ``\mu`` mean
* ###### _scale_: ``\sigma`` 


""", let

	μs = [-4, 0, 5]
	σ²s = [1 , 2 , 5]
	plt_lap = Plots.plot(title="Laplace distributions", xlabel=L"X", ylabel=L"p(X)", size=(300,300))
	for i in 1:3 
		plot!(plt_lap, Laplace(μs[i], sqrt(σ²s[i])), fill=true, alpha=0.5, label=L"\mathcal{L}(μ=%$(μs[i]), σ^2=%$(σ²s[i]))")
		vline!([μs[i]], color=i, label="", linewidth=2)
	end
	plt_lap
end)

# ╔═╡ 89ec84d5-6a21-4760-a482-2768624baec7
Foldable("", html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/laplace_eq.png' width = '550' /></center>")

# ╔═╡ 729746ec-9503-4598-ab4a-6794b5fa4192
begin
	f1(x; μ=0, σ=1) = abs(x-μ)/σ
	f2(x; μ=0, σ=1) = - f1(x; μ=μ, σ=σ)
	f3(x; μ=0, σ=1) = exp(f2(x; μ=μ, σ=σ))
	f4(x; μ=0, σ=1) = 1/(2σ) *exp(f2(x; μ=μ, σ=σ))
end;

# ╔═╡ 1a6b3ec2-b086-4f62-9d9b-5956f67f6c9d
ThreeColumn(let
	μ = μ4_
	σ = 1
	maxy = f4(μ; μ=μ, σ=σ) 
	plt = plot(range(μ -4, μ+4, 100), (x) -> f2(x; μ=μ, σ=σ), lw=2, lc=2, title = "", framestyle=:origin, ylim=[-2, max(maxy + 0.1, 1.5)], size=(230,250), label=L"\;- \left(\frac{|x-\mu|}{{\sigma}}\right)", legendfontsize=10, legend=:outertop)
	vline!([μ], label="", ls=:dash, lw=2, lc=2, la=0.5)
	plt
end, let
	μ = μ4_
	σ = 1
	maxy = f4(μ; μ=μ, σ=σ) 
	plt = plot(range(μ -4, μ+4, 100), (x) -> f2(x; μ=μ, σ=σ), lw=1., lc=2,label="" , ls=:dash, framestyle=:origin, ylim=[-2,max(maxy + 0.1, 1.5)], size=(230,250))
	
	plot!(x -> f3(x; μ=μ, σ=σ), lw=2, lc=3, title="", ylim=[-2,max(maxy + 0.1, 1.5)], titlefontsize=15, label=L"\exp{\left(-\frac{|x-\mu|}{{\sigma}}\right)}", legendfontsize=8, legend=:outertop)

	vline!([μ], label="", ls=:dash, lw=2, lc=2, la=0.5)
	plt
end, let
	μ = μ4_
	σ = 1
	maxy = f4(μ; μ=μ, σ=σ) 
	plt = plot(range(μ -4, μ+4, 100), (x) -> f2(x; μ=μ, σ=σ), lw=1, ls=:dash, lc=2,label="", title="", framestyle=:origin, ylim=[-2,max(maxy + 0.1, 1.5)],size=(230,250))
	plot!(x -> f3(x; μ=μ, σ=σ), lw=1, ls=:dash, lc=3, label="", title="")
	# end
	vline!([μ], label="", ls=:dash, lw=2, lc=2, la=0.5)
	plot!(x -> f4(x; μ=μ, σ=σ), lw=3, lc=4, title="", label=L"\frac{1}{2\sigma} \exp{\left(-\frac{|x-\mu|}{{\sigma}}\right)}", legendfontsize=8, legend=:outertop)

	plt
end)

# ╔═╡ bcec824a-c3ca-4041-a5c1-ad42abfe2f99
md"""

## Gaussian -- "_68-95-99 rule_"

"""

# ╔═╡ c1cec1ca-48c5-46ce-9017-74376fc34c98
TwoColumn(md""" 

##### The probability mass centered at ``\mu``

$\mathbb{P}( \mu-2\sigma \leq x\leq  \mu + 2\sigma) \approx 95\%$

* ######  Gaussians have short tails: very unlikely to observe _outliers_

* *e.g.* **almost impossible** to oberve ``x`` that are ``3\times`` standard deviation from the mean ``\mu``

""", @htl """<center><img src='https://tikz.net/files/gaussians-002.png'  width = '350' /></center>""")

# ╔═╡ 6f756993-89fd-4e90-a551-68f4c7cb5ced
md"""Add Laplace: $(@bind add_laplace CheckBox(false))"""

# ╔═╡ a91eedec-0549-4c17-b973-13ac1770d808
begin
	Random.seed!(1234)
	x_gaussians = randn(2_000)
	x_laplaces = rand(Laplace(), 2_000)
	plot(Normal(0,1), fill=true, fillalpha=0.1, lw=2, ratio=10, framestyle=:zerolines, label="Normal(0,1)", size=(700,300), legendfontsize=12, ylim =[-0.1, 0.55])

	if add_laplace
		plot!(Laplace(), lw=2, fill=true, alpha=0.2,  label="Laplace(0,1)")
		scatter!(x_laplaces, zeros(length(x_laplaces)) .+ 0.05, m=:vline, c=2, ms=5, label="") 
	end
	scatter!(x_gaussians, zeros(length(x_gaussians)), m=:vline, c=:blue, ms=5, label="") 
	

	plot!([-2, 2], [-0.05, -0.05], st=:path, label="", arrows=:both, lc=:blue)

	annotate!([0], [-0.05], Plots.text(L"\mu -2\sigma \;\;\;\;\;\;\;\;\;\;\;\; \mu+2\sigma", 10, :top, :blue))

end

# ╔═╡ 8cacc332-fb7f-4266-b49d-280692e87891
md"""

## Exponential distribution

##### Not all distributions are _location & scale_, *e.g.* _Exponential_ distribution

```math
\LARGE
p(x|\theta) = \frac{1}{\theta} e^{-\frac{x}{\theta}}
```


* ##### a distribution for strictly positive random variable ``x>0`` 
"""

# ╔═╡ 6b338615-8f0c-4dc2-bc27-b88642db489a
md"Add interpretation: $(@bind add_exp CheckBox(false))"

# ╔═╡ d52dfe3e-46d5-460b-b4de-3661212b9f58
begin
	plot(Exponential(), xlabel=L"x", fill=true, lw=3,alpha=0.5, ylabel=L"p(x)", label=L"\texttt{Exp}(x; \theta=1)", legendfontsize=12)

	plot!(Exponential(2),  fill=true, lw=3,alpha=0.5, label=L"\texttt{Exp}(x; \theta=2)")
	if add_exp

		title!("MMS coursework submission time distribution")

		xlabel!("time before deadline")

		plot!(labelfontsize=18)
	end


	plot!(Exponential(3),  fill=true, lw=3,alpha=0.5, label=L"\texttt{Exp}(x; \theta=3)", xlim =[-0.5, 15])
end

# ╔═╡ e557ad8b-9e4f-4209-908f-2251e2e2cde9
# md"""
# ## Joint distribution


# A **joint distribution** over a set of random variables: ``X_1, X_2, \ldots, X_n`` 


# ```math
# \large
# \begin{equation} P(X_1= x_1, X_2=x_2,\ldots, X_n= x_n) = P(x_1, x_2, \ldots, x_n) \end{equation} 
# ```

# * the joint event ``\{X_1=x_1, X_2=x_2, \ldots, X_n=x_n\}``'s distribution

# * must still statisfy

# $P(x_1, x_2, \ldots, x_n) \geq 0\;\; \text{and}\;\;  \sum_{x_1, x_2, \ldots, x_n} P(x_1, x_2, \ldots, x_n) =1$ 

# ## Joint distribution -- examples


# For example, joint distribution of temperature (``T``) and weather (``W``): ``P(T,W)``
# ```math
# \begin{equation}
# \begin{array}{c c |c} 
# T & W & P\\
# \hline
# hot & sun & 0.4 \\
# hot & rain & 0.1 \\
# cold & sun  & 0.2\\
# cold & rain & 0.3\end{array} \end{equation} 
# ```



# """

# ╔═╡ 62627b47-5ec9-4d7d-9e94-2148ff198f66
# md"""
# ## Independence: genuine coin toss

# > Two sequences of 300 “coin flips” (H for heads, T for tails). 
# > 
# > * which one is the genuine **independent** coin tosses?

# **Sequence 1**

# >	TTHHTHTTHTTTHTTTHTTTHTTHTHHTHHTHTHHTTTHHTHTHTTHTHHTTHTHHTHTTTHHTTHHTTHHHTHHTHTTHTHTTHHTHHHTTHTHTTTHHTTHTHTHTHTHTTHTHTHHHTTHTHTHHTHHHTHTHTTHTTHHTHTHTHTTHHTTHTHTTHHHTHTHTHTTHTTHHTTHTHHTHHHTTHHTHTTHTHTHTHTHTHTHHHTHTHTHTHHTHHTHTHTTHTTTHHTHTTTHTHHTHHHHTTTHHTHTHTHTHHHTTHHTHTTTHTHHTHTHTHHTHTTHTTHTHHTHTHTTT

# **Sequence 2**

# >	HTHHHTHTTHHTTTTTTTTHHHTTTHHTTTTHHTTHHHTTHTHTTTTTTHTHTTTTHHHHTHTHTTHTTTHTTHTTTTHTHHTHHHHTTTTTHHHHTHHHTTTTHTHTTHHHHTHHHHHHHHTTHHTHHTHHHHHHHTTHTHTTTHHTTTTHTHHTTHTTHTHTHTTHHHHHTTHTTTHTHTHHTTTTHTTTTTHHTHTHHHHTTTTHTHHHTHHTHTHTHTHHHTHTTHHHTHHHHHHTHHHTHTTTHHHTTTHHTHTTHHTHHHTHTTHTTHTTTHHTHTHTTTTHTHTHTTHTHTHT

# * both of them have ``N_h = 148`` and ``N_t= 152``
# """

# ╔═╡ fc09b97a-13c9-4721-83ca-f7caa5f55079
begin
	seq1="TTHHTHTTHTTTHTTTHTTTHTTHTHHTHHTHTHHTTTHHTHTHTTHTHHTTHTHHTHTTTHHTTHHTTHHHTHHTHTTHTHTTHHTHHHTTHTHTTTHHTTHTHTHTHTHTTHTHTHHHTTHTHTHHTHHHTHTHTTHTTHHTHTHTHTTHHTTHTHTTHHHTHTHTHTTHTTHHTTHTHHTHHHTTHHTHTTHTHTHTHTHTHTHHHTHTHTHTHHTHHTHTHTTHTTTHHTHTTTHTHHTHHHHTTTHHTHTHTHTHHHTTHHTHTTTHTHHTHTHTHHTHTTHTTHTHHTHTHTTT"
	seq2 = "HTHHHTHTTHHTTTTTTTTHHHTTTHHTTTTHHTTHHHTTHTHTTTTTTHTHTTTTHHHHTHTHTTHTTTHTTHTTTTHTHHTHHHHTTTTTHHHHTHHHTTTTHTHTTHHHHTHHHHHHHHTTHHTHHTHHHHHHHTTHTHTTTHHTTTTHTHHTTHTTHTHTHTTHHHHHTTHTTTHTHTHHTTTTHTTTTTHHTHTHHHHTTTTHTHHHTHHTHTHTHTHHHTHTTHHHTHHHHHHTHHHTHTTTHHHTTTHHTHTTHHTHHHTHTTHTTHTTTHHTHTHTTTTHTHTHTTHTHTHT"
	sequence1=map((x) -> x=='H' ? 1 : 2,  [c for c in seq1])
	sequence2=map((x) -> x=='H' ? 1 : 2,  [c for c in seq2])
end;

# ╔═╡ 95779ca4-b743-43f1-af12-6b14c0e28f0b
# md"""

# ## Independence: genuine coin toss (cont.)


# Recall **independence**'s definition

# ```math
# \large
# P(X_{t+1}|X_{t}) = P(X_{t+1})
# ```

# * ``X_{t}``: the tossing result at ``t``
# * ``X_{t+1}``: the next tossing result at ``t+1``


# And the conditional distribution should be (due to independence)

# ```math
# \large
# P(X_{t+1}=\texttt{h}|X_{t}=\texttt{h}) = P(X_{t+1}=\texttt{h}|X_{t}=\texttt{t}) =P(X_{t+1}=\texttt{h}) = 0.5
# ```

# """

# ╔═╡ 0f280847-2404-4211-8221-e30418cf4d42
# md"""

# ##


# **Sequence 1**

# >	TTHHTHTTHTTTHTTTHTTTHTTHTHHTHHTHTHHTTTHHTHTHTTHTHHTTHTHHTHTTTHHTTHHTTHHHTHHTHTTHTHTTHHTHHHTTHTHTTTHHTTHTHTHTHTHTTHTHTHHHTTHTHTHHTHHHTHTHTTHTTHHTHTHTHTTHHTTHTHTTHHHTHTHTHTTHTTHHTTHTHHTHHHTTHHTHTTHTHTHTHTHTHTHHHTHTHTHTHHTHHTHTHTTHTTTHHTHTTTHTHHTHHHHTTTHHTHTHTHTHHHTTHHTHTTTHTHHTHTHTHHTHTTHTTHTHHTHTHTTT

# The joint frequency table is

# ```math

# \begin{equation}  \begin{array}{c|cc} 
# n(X_{t}, X_{t+1}) & X_{t+1} = \texttt h & X_{t+1} =\texttt t \\
# \hline
# X_t =\texttt h & 46 & 102 \\ 

# X_t= \texttt t & 102 & 49 \\ 

# \end{array} \end{equation} 

# ```

# * ``P(X_{t+1}=\texttt h|X_t=\texttt h) =\frac{46}{46+102} \approx 0.311 \ll 0.5``
# * ``P(X_{t+1}=\texttt h|X_t=\texttt t) =\frac{102}{102+49} \approx 0.675 \gg 0.5``

# """

# ╔═╡ b682cc8d-4eeb-4ecd-897c-e15a3e40f76d
# md"""

# ##

# **Sequence 2**

# >	HTHHHTHTTHHTTTTTTTTHHHTTTHHTTTTHHTTHHHTTHTHTTTTTTHTHTTTTHHHHTHTHTTHTTTHTTHTTTTHTHHTHHHHTTTTTHHHHTHHHTTTTHTHTTHHHHTHHHHHHHHTTHHTHHTHHHHHHHTTHTHTTTHHTTTTHTHHTTHTTHTHTHTTHHHHHTTHTTTHTHTHHTTTTHTTTTTHHTHTHHHHTTTTHTHHHTHHTHTHTHTHHHTHTTHHHTHHHHHHTHHHTHTTTHHHTTTHHTHTTHHTHHHTHTTHTTHTTTHHTHTHTTTTHTHTHTTHTHTHT

# ```math

# \begin{equation}  \begin{array}{c|cc} 
# n(X_{t}, X_{t+1}) & X_{t+1} = \texttt h & X_{t+1} =\texttt t \\
# \hline
# X_t =\texttt h & 71 & 77 \\ 

# X_t= \texttt t & 76 & 75 \\ 

# \end{array} \end{equation} 

# ```

# * ``\hat P(X_{t+1}=\texttt h|X_t=\texttt h) =\frac{71}{71+77} \approx 0.48 \approx 0.5``
# * ``\hat P(X_{t+1}=\texttt h|X_t=\texttt t) =\frac{76}{76+75} \approx 0.503 \approx 0.5``
# """

# ╔═╡ 183721ea-de2d-44fd-b429-d75d655a9094
md"""

# Maximum Likelihood Estimation
"""

# ╔═╡ 9b77f11e-3a8f-4b05-b02a-291fae630583
md"""


## _Likelihood_ function ``P(\texttt{Data}| \texttt{hypothesis})``
"""

# ╔═╡ fd83e1c0-0ef9-4887-b1e7-3ca2868ed5b8
show_img("mle2.png", h=190)

# ╔═╡ 2b4d26bd-8636-4f8c-9152-fd80b7d73b87
md"""

* #### data ``\mathcal{D}`` is observed, therefore fixed
* #### hypothesis ``h`` is unknown, it can vary
"""

# ╔═╡ 2773dab9-f6ca-40b9-8b52-9deb9ae23557
md"""


## _Likelihood_ function ``P(\texttt{Data}| \texttt{hypothesis})``
"""

# ╔═╡ 65de12b9-1977-4c2d-9911-9284df73cf44
show_img("mle1.png", h=160)

# ╔═╡ 1be833fd-8ae7-4f2c-852c-824e91ddeb9a
md"""


* ##### ``\ell(h)`` measures the "matchness" of the observed data with a verying ``h``  
  * larger ``\ell(h)``: the data is **more** likely to have been generated by ``h``
  * lower ``\ell(h)``: the data is **less** likely to have been generated by ``h``


* ##### ``\ell(h)`` is **NOT** a probability (distribution) of ``h``
  * *i.e.* ``\sum_h \ell(h)\neq 1`` usually
"""

# ╔═╡ d9dfcc42-e0e5-4410-8c2c-0ccbd8aaa3b5
md"""
## Maximum Likelihood Estimation (MLE)



```math
\Large
\hat{h}_{\text{MLE}} \leftarrow \arg\max_{h}\; P(\mathcal D|h)
```

* ##### ``\hat h_{\text{MLE}}`` is called *maximum likelihood estimator*



* ##### "``\arg\max``": find the variable ``h`` that maximise ``P(\mathcal D|h)``
  * ###### ``\hat{h}_{\text{MLE}}`` matches ``\mathcal{D}`` the most
  * ###### ``\hat{h}_{\text{MLE}}`` most likely to have been used to generate  ``\mathcal{D}``


"""

# ╔═╡ 990ddc33-c000-46ea-83a0-746e946f28dc
md"""


## An example

!!! note "Cancer or cold"
	#### Someone has coughed. Has he got  
	* ##### _stage 4 lung cancer_,  _common cold_, or *healthy* ?


"""

# ╔═╡ 90f00871-97c5-4e43-93e0-37b99b488d37
show_img("coughmle.png", h=160)

# ╔═╡ 0a58b41e-9b5d-4954-9c24-af993a3e2f00
md"""

* ##### observed: ``\mathcal{D} = \texttt{coughed}``
* ##### hypothesis: ``h \in \{\texttt{cancer} , \texttt{cold}, \texttt{healthy}\}``


"""

# ╔═╡ ed2017ad-04a5-43e5-9690-5d92b5fb92de
md"""## Likelihood function example (cont.)"""

# ╔═╡ 5dd53b59-4f79-4fbb-9a17-9705189a9cbf
TwoColumn(md"""

##### The likelihood function is
\


$$\large \begin{align} \ell(h)&= P(\texttt{coughed}|h) \\
&= \begin{cases}\color{blue} 0.05 & \colorbox{lightblue}{h = \texttt{hthy}} \\
\color{red}0.45 & \colorbox{orange}{h = \texttt{cold}} \\
\color{green}0.55 & \colorbox{lightgreen}{h = \texttt{cancer}}
\end{cases} 
\end{align}$$

* ##### it is a function of ``h``
* BUT it is **NOT** a probability distribution of ``h``


""",

	show_img("coughlik.svg", w=300)
)

# ╔═╡ daac7901-5e1c-409e-81f0-816888cfe34b
md"""

## MLE 
"""

# ╔═╡ 6dfc3afd-412e-4890-804f-3951d19470a6
TwoColumn(md"""
$$\large \begin{align} \ell(h)&= P(\texttt{coughed}|h) \\
&= \begin{cases}\color{blue} 0.05 & \colorbox{lightblue}{h = \texttt{hthy}} \\
\color{red}0.45 & \colorbox{orange}{h = \texttt{cold}} \\
\color{green}0.55 & \colorbox{lightgreen}{h = \texttt{cancer}}
\end{cases} 
\end{align}$$
\

```math
\Large
\begin{align}
\hat{h}_{MLE} &= \arg\max_h \ell(h)\\
&=\boxed{\texttt{cancer}}
\end{align}
```

* #### *very bizarre* answer: this is known as over-fitting
""",

	show_img("coughlik.svg", w=300)
)

# ╔═╡ 86369522-1eaf-4f43-ace5-7d991ef78564
begin
	Hs = ["healthy", "cold", "cancer"]
	prior_p = [0.89, 0.1, 0.01]
	liks = [0.05, 0.45, 0.54]
	x_= 1
	post_p_unorm = prior_p .* liks
	marg_lik = sum(post_p_unorm)
	post_p = post_p_unorm/marg_lik
end;

# ╔═╡ 4e2f3597-6ea3-4006-8dda-5151ebac5ba7
# like_plt = plot(liks, fill=true, st=:bar, xticks=(1:3, Hs), markershape=:circ, ylim =[0, 0.6], c=1:3, label="",xlabel=L"h",xtickfontsize=12, title="Likelihood: "*L"P(\texttt{coughed}|h)",lw=2, size=(300,300));

# ╔═╡ 3a689042-ce56-427c-b567-4c71b74dc77f
md"""

## Another example: coin toss


> #### Toss a coin ``n=10`` times independently:
>
> $$\Large\mathcal{D} =\{1,0,0,1,1,1,1,1,1,1\},$$
> *i.e.* 8 of 10 are heads. 
> #### Find the _bias_ of the coin?

* Remember the bias ``\theta \in[0,1]`` is the probability of observing a head when you toss a coin.
"""

# ╔═╡ 9c53de83-7552-4d72-8f02-bfb82230e440
md"""

## B.t.w. this is a valid ML problem



"""

# ╔═╡ 8beaa73a-87c6-4042-9968-4e881c04a033
md"""


## Another example: coin toss (cont.)

##### The unknown _hypothesis_ (a continuous hypothesis space): 

```math
\Large
\theta \in [0,1]
```

##### The observed data: 

```math
\Large
	\mathcal{D} =\{Y_1, Y_2,\ldots, Y_{10}\}
```

## Another example: coin toss (cont.)


##### The unknown _hypothesis_ (a continuous hypothesis space): 

```math
\Large
\theta \in [0,1]
```


##### The observed data: 

```math
\Large
	\mathcal{D} =\{Y_1, Y_2,\ldots, Y_{10}\}
```


##### The *likelihood function*

```math
\large
P(\mathcal{D}|\theta) = P(y^{(1)}, y^{(2)}, \ldots, y^{(10)}|\theta)= \prod_{i=1}^{10} P(y^{(i)}|\theta)
```

* ##### *independent* therefore we multiply all together
"""

# ╔═╡ fc5bc22c-7bd4-4b76-b93f-453806bc5cd6
md"""
##

##### Recall each ``P(y^{(i)}|\theta)`` is a _Bernoulli_ distribution

```math
\large
\begin{align}
P(y^{(i)}|\theta) = \begin{cases} 1-\theta & y^{(i)} = 0 \\ \theta & y^{(i)} = 1\end{cases}, \;\; \text{or}\;\; P(y^{(i)}|\theta) = \theta^{y^{(i)}} (1-\theta)^{1- y^{(i)}}

\end{align}
```


"""

# ╔═╡ 8cf468ad-ef03-47fb-8e7a-155f12ad0133
md"""

## Another example: coin toss (cont.)


##### The unknown _hypothesis_ (a continuous hypothesis space): 

```math
\Large
\theta \in [0,1]
```


##### The observed data: 

```math
\Large
	\mathcal{D} =\{Y_1, Y_2,\ldots, Y_{10}\}
```


##### The *likelihood function*

```math
\large
\begin{align}
P(\mathcal{D}|\theta) &= P(y^{(1)}, y^{(2)}, \ldots, y^{(n)}|\theta)\\
&= \prod_{i=1}^{n} P(y^{(i)}|\theta)\\
&= \prod_{i=1}^{n} \theta^{y^{(i)}} (1-\theta)^{1- y^{(i)}} \\
&= \theta^{n⁺} (1- \theta)^{n- n⁺}
\end{align}
```


* ``n^+ = \sum_{i=1}^n y^{(i)}``: the total number of heads 
* ``n``: total tosses


"""

# ╔═╡ 1be5a5b4-4c14-4b6f-9c07-3ffe82dfb849
aside(tip(md"""
``P(y^{(i)}|\theta)`` is a _Bernoulli_ distribution

```math
\large
\begin{align}
P(y^{(i)}|\theta) &= \begin{cases} 1-\theta & y^{(i)} = 0 \\ \theta & y^{(i)} = 1\end{cases}\\
&= \theta^{y^{(i)}} (1-\theta)^{1- y^{(i)}}
\end{align}
```



"""))

# ╔═╡ 3f368884-9979-4cea-b462-ba67bc47d517
begin
	𝒟 = [1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
end;

# ╔═╡ 66d1a512-5476-4dda-a383-b5ced3f99514
TwoColumn(md"""
\

#### A _binary classification_ problem 

* ##### we want to classify ``y^{(i)} \in \{0, 1\}``


* ##### with no discriminative feature ``x``; 


* ##### alternatively, you may assume ``x^{(i)}=1`` for all ``i=1,\ldots, n`` (dummy ones)

""", 
md"""

###### Training data
$(begin
trainingdata = DataFrame(x = ones(10), target = 𝒟)
end)
""")

# ╔═╡ 71bc71e7-ddf2-4861-ba17-3279950f1f93
md"""

## Another example: coin toss (cont.)



"""

# ╔═╡ 6698f6de-52e4-4f0e-a7d6-818fa1288e44
md"""

## Log-likelihood


##### A common practice is to maximise the *log-* likelihood

```math
\Large
\hat{\theta}_{\text{MLE}} \leftarrow \arg\max_{\theta'}\; \underbrace{\ln p(\mathcal{D}|\theta')}_{\ell(\theta)}
```


##### Log function (``e`` based) is a monotonically increasing function

* ##### the log transform *does not* change ``\hat{\theta}_{\text{MLE}}``

"""

# ╔═╡ 3c88504f-4f4e-4c51-98c4-b27e3f6d9f61
aside(tip(plot(log, framestyle=:zerolines, size=(250,150), ylim=[-5,3],label="", lw=2, title="Log function", titlefontsize=10)))

# ╔═╡ 99334132-f0ee-4c62-8e1b-fc3136700ae5
function ℓ(θ, 𝒟; logprob=false)
	N = length(𝒟)
	N⁺ = sum(𝒟)
	ℓ(θ, N, N⁺; logprob=logprob)
end;

# ╔═╡ e6529b7b-ea95-44be-84c1-609dfd3c52a7
function ℓ(θ, n, n⁺; logprob=false)
	# logL = n⁺ * log(θ) + (n-n⁺) * log(1-θ)
	# use xlogy(x, y) to deal with the boundary case 0*log(0) case gracefully, i.e. θ=0, n⁺ = 0
	logL = xlogy(n⁺, θ) + xlogy(n-n⁺, 1-θ)
	logprob ? logL : exp(logL)
end;

# ╔═╡ ad9b4a60-c1aa-46bb-a3e2-8e9eee0119ef
TwoColumn(md"""


#### The *likelihood function*

```math
\Large
\begin{align}
P(\mathcal{D}|\theta) 
= \theta^{n⁺} (1- \theta)^{n- n⁺}
\end{align}
```


* ##### ``n^+ = 8``: number of heads 
* ##### ``n=10``: number of tosses

* ##### It is STILL a function of the hypothesis ``\theta``
""", let
	pl1 = plot(0:0.01:1.0, θ -> ℓ(θ, 𝒟), color=1, lw=2, xlabel=L"θ", ylabel=L"p(\mathcal{D}|θ)", label="", title="Likelihood: "*L"p(\mathcal{D}|\theta)", size=(300,300))

	vline!([mean(𝒟)], lw=2, lc=1, label="MLE = 0.8")
	pl1
end)

# ╔═╡ 95149606-f17a-4e4b-9d44-6cbd68509d86
let

θs = 0:0.01:1.0
	pl1 = plot(θs, θ -> ℓ(θ, 𝒟), color=1,  xlabel=L"θ", ylabel=L"p(\mathcal{D}|θ)", label="", lw=2, title="Likelihood: "*L"p(\mathcal{D}|\theta)")

	vline!([mean(𝒟)], c=:grey, label="", lw=2, ls=:dash)

	pl2 = plot(θs, θ -> ℓ(θ, 𝒟; logprob=true), color=1,  lw=2, xlabel=L"θ", ylabel=L"p(\mathcal{D}|θ)", label="", title="Log likelihood: "*L"\ln p(\mathcal{D}|\theta)")
	vline!([mean(𝒟)], c=:grey, label="", lw=2, ls=:dash)

	plot(pl1, pl2, size=(600,300))
end

# ╔═╡ d5303044-c534-4b37-a126-cec8cfd97ec9
md"""
## Why take  ``\ln``  ?


"""

# ╔═╡ a2d9d8f7-85e1-4fbb-8683-95a301031da5
TwoColumn(md"""
##### Log-likelihood is more *numerically stable*

* ###### the likelihood underflows easily when ``n`` is large 
  
  $\large\begin{align}
  P(\mathcal{D}|\theta) 
  = \theta^{n⁺} (1- \theta)^{n- n⁺}
  \end{align}$
  * ###### it underflows: ``.5^{100} \times (1-.5)^{100}``

* ###### but *log* likelihood is more stable

```math
\begin{align}
\ln P(\mathcal{D}|\theta) 
= {n⁺}\ln\theta+ (n- n⁺)\ln(1- \theta)
\end{align}
```

""", md"")

# ╔═╡ 6fb13b48-bbaa-469e-b327-c64c7e2d4bfd
md"""
## Why take ``\ln`` ?


"""

# ╔═╡ 3fc3b46a-39fc-448b-b2ca-eb7a787bb480
TwoColumn(md"""
##### Log-likelihood is more *numerically stable*

* ###### the likelihood underflows easily when ``n`` is large 
  
  $\begin{align}
  P(\mathcal{D}|\theta) 
  = \theta^{n⁺} (1- \theta)^{n- n⁺}
  \end{align}$
  * *e.g.* it underflows ``.5^{100} \times (1-.5)^{100}``

* ###### but *log* likelihood is more stable

```math
\begin{align}
\ln P(\mathcal{D}|\theta) 
= {n⁺}\ln\theta+ (n- n⁺)\ln(1- \theta)
\end{align}
```


""", md"##### Log-likelihood is more *convenient*

* ###### e.g. Gaussian density is
```math
\begin{align}
p(x) 
= {\frac{1}{ \sqrt{2\pi\sigma^2}}} e^{-\frac{1}{2} {\left(\frac{x -\mu}{\sigma}\right)^2}}
\end{align}
```

* ##### but *log* likelihood is much simpler 
  * ``e`` and ``\ln`` cancels

```math
\begin{align}
\ln p(x) 
= -\frac{1}{2}\ln 2\pi\sigma^2 -\frac{1}{2\sigma^2} (x - \mu)^2
\end{align}
```")

# ╔═╡ 697e2973-ee2a-4227-bc19-f34fc7b1bcb9
md"""
## Another example: coin toss (cont.)




"""

# ╔═╡ 0e452e81-91b8-4724-becb-ce96cb3f1b8a
TwoColumn(md"""

\


#### The log-likelihood function becomes:


```math
\Large

\begin{align}
\ell(\theta) &=\ln p(\mathcal{D}|\theta)  \\
&= \ln \{\theta^{n⁺} (1- \theta)^{n- n⁺}\}\\
&={n⁺}\ln \theta+ (n- n⁺)\ln (1- \theta)

\end{align}
```""", begin plot(0:0.01:1, θ -> ℓ(θ, 𝒟; logprob=true), color=1,  lw=2, xlabel=L"θ", ylabel=L"p(\mathcal{D}|θ)", label="", title="Log-likelihood: "*L"\ell(\theta)", size=(300,300))
end
)

# ╔═╡ 8a00508d-c275-4ef5-8568-67e1e3d1187d
md"""
$(begin

aside(tip(md" Log identities:

```math 
\ln(a \cdot b) = \ln a + \ln b
```

```math
e^{\ln x} = x;\;\; \ln (e^x) = x;\;\; 
```

```math 
\ln(a^b) = b \ln a
```

```math 
\ln\left (\prod_i p_i\right ) = \sum_{i} \ln p_i
```
"))
end)
"""

# ╔═╡ 15b8b453-0010-4bb6-99f7-6e5723759cd1
md"""

## MLE: good old calculus

#### To optimise the likelihood function; 

```math
\large
\theta_{\text{MLE}} \leftarrow \arg\max_{\theta}\ell(\theta)
```

#### Take derivative and set to zero

```math
\large
\frac{\mathrm{d} \ell{(\theta)}}{\mathrm{d}\theta} =0

```


* the MLE is just the observed frequency, or sample mean 

```math
\large
\boxed{
\hat{\theta}_{\text{MLE}} = \frac{n⁺}{n}}
```

"""

# ╔═╡ 821ea80c-a058-41e2-8cb5-f6c4519fc662
Foldable("Details", md"""
Take derivative w.r.t ``\theta``

```math
\begin{align}
\frac{\mathrm{d}\ell(\theta)}{\mathrm{d}\theta} &= n^+ \frac{1}{\theta} + (n-n^+) \frac{1}{1-\theta} (-1)
\end{align}
```

Set the derivative to zero and solve it 

```math
\begin{align}
n^+ \frac{1}{\theta} &+ (n-n^+) \frac{1}{1-\theta} (-1) =0 \\
&\Rightarrow n^+ (1-\theta) = (n-n^+) \theta \\
&\Rightarrow n^+ - n^+ \theta = n \theta - n^+ \theta\\
&\Rightarrow \theta = \frac{n^+}{n}
\end{align}
```

""")

# ╔═╡ 3b1ca12b-ed16-42c3-b3e2-377052fa2d37
md"""
## MLE: overfitting

> ##### What if we only toss a coin twice and observe 
> ```math
> \Large \mathcal{D}=\{0,0\}?
> ``` 
> * ##### ``n⁺ =0, n=2``




"""

# ╔═╡ 1ecc24bf-6263-4279-9549-aeb75b805294
TwoColumn(md"""
##### The MLE is 

$$\large
\hat{\theta}_{\text{MLE}} =\frac{n^+}{n} = \frac{\textcolor{red}0}{2} =0$$


* ##### MLE overfits (esp when data is sparse)!


##### Why?
* ###### MLE tries to fit the current observed data (even only two observations)

""", let
	gr()
	𝒟 = [0,0]
	like_plt_seller = plot(0:0.01:1., θ -> ℓ(θ, 𝒟; logprob=false), color=1,  xlabel=L"θ", ylabel=L"p(\{0,0\}|θ)", label="",lw=2, title="Likelihood: "*L"p(\{0,0\}|\theta)", size=(300,300))
	vline!([mean(𝒟)], c=:grey, label="", lw=2, ls=:dash)
end)

# ╔═╡ c475b487-2546-4bd7-b3e5-b931e5db6277
md"""

## "Learning" and MLE


"""

# ╔═╡ d79c7f60-2e00-4ba4-9e47-edbc9154c1d3
TwoColumn(md"""
\

##### Recall machine "*learning*"


* ##### needs some *goodness* measure of ``h``
  

* ##### ``\ln p(\mathcal{D}|h)`` serves this purpose!
  * ``\theta \in [0,1]`` of the Bernoulli example
  * ``\mathcal{D}`` the tossing results
""", html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/mlgoodness.png' width = '250' /></center>")

# ╔═╡ 50eacaac-ddf9-4f54-8c5a-d947cb51d7ad
md"""


## Gaussian MLE


#### Find the MLE of a Gaussian given observations

```math
\Large
\mathcal{D} = \{y^{(1)}, y^{(2)}, \ldots, y^{(n)}\}
```


"""

# ╔═╡ f73d838c-f2dc-4540-9171-df8d188d14ff
md"""


##### The joint log-likelihood is 


```math
\large 
\ln p(\mathcal{D}|\mu, \sigma) = \ln \left \{\prod_{i=1}^n \mathcal{N}(y^{(i)};\mu, \sigma^2)\right \}
```


* ##### parameters: ``\mu, \sigma^2 >0``
"""

# ╔═╡ 30c37f2b-8dbf-4146-99c9-fc16e8bbe700
@bind add_gauss_fit CheckBox(false)

# ╔═╡ f721bc14-42d7-4f0d-971f-2fc2d4b85818
md"""

## What MLE aims to do ?


```math
\Large
\hat{\mu}, \hat{\sigma^2} \leftarrow \arg\min_{\mu, \sigma} -\ell(\mu, \sigma)
```

* it minimises the negative log-likelihood (or "mis-matchness")
"""

# ╔═╡ 697bc4e0-d74d-459a-9d22-1b5716b8a11d
begin
	gaussian_D = randn(50) * sqrt(2.0) .+ 3 
end;

# ╔═╡ 85ec65ef-c7c7-400b-bbb5-bbdfb8875c60
let
	# Random.seed!(1234)
	x_gaussians = gaussian_D
	mm = 2
	ss = 1.5
	gauss = Normal(mm, ss)
	plt = plot(x_gaussians, 0.05 * ones(length(x_gaussians)), st=:sticks, m=:circle, c=1, ms=4, markerstrokewidth=1, label="data: "*L"\{y^{(i)}\}", framestyle=:zerolines, yaxis=false, ylim =[-0.01, 0.06], xlabel="",  legendfontsize=10, size=(700,200)) 
	height = pdf(gauss, mm)
	if add_gauss_fit
		plot!(gauss, fill=true, fillalpha=0.1, lw=2, legendfontsize=12, ylim =[-0.005, height+0.1], label=L"\mathcal{N}(\mu, \sigma^2)")
	
		plot!([mm-2*ss  , mm + 2*ss], [height/3, height/3], st=:path, label="", arrows=:both, lc=2, size=(700,310))
	end
	plt
end

# ╔═╡ 3235ea89-44cc-42b9-b6ed-be0dd2d0ef08
md"""``\mu``: $(@bind μ_ Slider(-2.1:0.1:6; default = mean(gaussian_D))), ``\sigma``: $(@bind s_ Slider(0.45:0.01:3; default= std(gaussian_D)))"""

# ╔═╡ 27df7f09-4a0f-40eb-826e-90635e996dc8
let
	# Random.seed!(1234)
	x_gaussians = gaussian_D
	# x_laplaces = rand(Laplace(), 2_000)
	mm = μ_
	ss = s_
	gauss = Normal(mm, ss)
	plot(x_gaussians, 0.05 * ones(length(x_gaussians)), st=:sticks, m=:circle, c=1, ms=4, markerstrokewidth=1, label="data: "*L"\{y^{(i)}\}", framestyle=:zerolines, yaxis=false,  xlabel="y", legendfontsize=10) 
	height = pdf(gauss, mm)
	plot!(gauss, fill=true, fillalpha=0.1, lw=2, ratio=10, framestyle=:zerolines, size=(700,325), legendfontsize=12, xlim = mean(x_gaussians) .+ (-6.5,6.5), ylim =[-0.005, height+0.1], label=L"\mathcal{N}(\mu, \sigma)")
	
	plot!([mm-2*ss  , mm + 2*ss], [height/3, height/3], st=:path, label="", arrows=:both, lc=2)

	like = loglikelihood(gauss, x_gaussians)
	plot!([mm, mm], [0, height], st=:path, ls=:dash, c=2, lw=2, label="")
	annotate!([mm], [height/3], Plots.text(L"\sigma = %$(round(2*ss, digits=2))", 12, :bottom, :red), title="Negative log-likelihood "*L"-\ell(\mu, \sigma)= %$(round(-like;digits=2))")

	annotate!([mm], [height], text(L"\mu = %$(round(mm; digits=2))", :red, :bottom))

end

# ╔═╡ 4b7c5bc1-88c3-4352-8ad3-91abd2f03597
begin
	mle_m = mean(gaussian_D)
	mle_sigma = mean((gaussian_D .- mle_m).^2) |> sqrt
	mle_value = -exp(loglikelihood(Normal(mle_m, mle_sigma), gaussian_D)/50)
	surfaceplt = plot(range(mle_m .+ (-5.0, 5.0)..., 100), range(0.5, mle_sigma+2, 100), (m, s) -> -exp(loglikelihood(Normal(m, s), gaussian_D)/50) ,c=:jet, st=:surface, xlabel=L"\mu", ylabel=L"\sigma", alpha=0.8, colorbar=false, title="Negative likelihood: "*L"-p(\mathcal{D}|\mu, \sigma)", framestyle=:zerolines, zlim =[mle_value-0.001, 0.05], camera=(20,45))

	likeh = -exp(loglikelihood(Normal(μ_, s_), gaussian_D)/50)
	scatter!([μ_], [s_],[0.0],  ms=5, m=:x, markerstrokewidth=3, label="current hypothesis")

	scatter!([mle_m], [mle_sigma],[0.0],  ms=5, m=:x, c=:blue, markerstrokewidth=5, label="MLE")
	
	scatter!([μ_], [s_],[likeh],  ms=3, mc=4,  m=:circle, markerstrokewidth=1, label="")

	
	scatter!([mle_m], [mle_sigma],[mle_value],  ms=3, mc=4,  m=:x, markerstrokewidth=2, label="")

	plot!([μ_, μ_] , [s_, s_], [0, likeh], st=:path, lw=2, ls=:dash, lc=:gray, label="")

	plot!([mle_m, mle_m] , [mle_sigma, mle_sigma], [0, mle_value], st=:path, lw=1, ls=:dash, lc=:blue, label="")
	contplt = plot(range(mle_m .+ (-5.0, 5.0)..., 100), range(0.5, mle_sigma+2, 100), (m, s) -> -exp(loglikelihood(Normal(m, s), gaussian_D)/50) ,c=:jet, st=:contour, xlabel=L"\mu", ylabel=L"\sigma", alpha=0.8, colorbar=false, title="Negative likelihood: contour")
	scatter!([μ_], [s_], ms=5, m=:x, markerstrokewidth=3, label="current hypothesis")

	scatter!([mle_m], [mle_sigma], ms=5, m=:x,c=:blue, markerstrokewidth=5, label="MLE")
	plot(surfaceplt, contplt, size=(800,400))
end

# ╔═╡ 831c0440-06a2-429c-b2e9-1dc09467523f
md"""
## Gaussian MLE (conti.)
##### Recall that Gaussian likelihood is


```math
\large 
p(y^{(i)}|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left \{ -\frac{1}{2\sigma^2} (y^{(i)} -\mu)^2 \right \}
```


##### Take the log, we have 

```math
\large 
\ln p(y^{(i)}|\mu, \sigma) 
= -\frac{1}{2}\ln 2\pi -\frac{1}{2}\ln \sigma^2 -\frac{1}{2\sigma^2} (y^{(i)} - \mu)^2
```


"""

# ╔═╡ c5c4a329-875a-4eba-a95e-97f4b2702f66
md"""
## Gaussian MLE (conti.)
##### Recall that Gaussian likelihood is


```math
\large 
p(y^{(i)}|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left \{ -\frac{1}{2\sigma^2} (y^{(i)} -\mu)^2 \right \}
```


##### Take the log, we have 

```math
\large 
\ln p(y^{(i)}|\mu, \sigma) 
= -\frac{1}{2}\ln 2\pi -\frac{1}{2}\ln \sigma^2 -\frac{1}{2\sigma^2} (y^{(i)} - \mu)^2
```


##### Therefore, the total joint log-likelihood is


```math
\large
\begin{align}
\ell(\mu,\sigma^2) &= \ln p(\mathcal{D}|\mu, \sigma)= \sum_{i=1}^{n}\ln p(y^{(i)}|\mu, \sigma) \\
\end{align}
```
"""

# ╔═╡ bcf33704-2f9d-4b6b-a7c9-6c1ad0349273
md"""
## Gaussian MLE (conti.)
##### Recall that Gaussian likelihood is


```math
\large 
p(y^{(i)}|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left \{ -\frac{1}{2\sigma^2} (y^{(i)} -\mu)^2 \right \}
```


##### Take the log, we have 

```math
\large 
\ln p(y^{(i)}|\mu, \sigma) 
= -\frac{1}{2}\ln 2\pi -\frac{1}{2}\ln \sigma^2 -\frac{1}{2\sigma^2} (y^{(i)} - \mu)^2
```


##### Therefore, the total joint log-likelihood is


```math
\large
\begin{align}
\ell(\mu,\sigma^2) &= \ln p(\mathcal{D}|\mu, \sigma)= \sum_{i=1}^{n}\ln p(y^{(i)}|\mu, \sigma) \\
&= \sum_{i=1}^{n} \left \{\cancel{-\frac{1}{2}\ln2\pi} -\frac{1}{2}\ln\sigma^2 -\frac{1}{2\sigma^2} (y^{(i)} - \mu)^2\right \} 
\end{align}
```
"""

# ╔═╡ ef7c0dc7-d5d6-4dc1-91fa-2508363b3931
# md"""
# ## Gaussian MLE (conti.)
# ##### Recall that Gaussian likelihood is


# ```math
# \large 
# p(y^{(i)}|\mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left \{ -\frac{1}{2\sigma^2} (y^{(i)} -\mu)^2 \right \}
# ```


# ##### Take the log, we have 

# ```math
# \large 
# \ln p(y^{(i)}|\mu, \sigma) 
# = -\frac{1}{2}\ln 2\pi -\frac{1}{2}\ln \sigma^2 -\frac{1}{2\sigma^2} (y^{(i)} - \mu)^2
# ```


# ##### Therefore, the total joint log-likelihood is


# ```math
# \large
# \begin{align}
# \ell(\mu,\sigma^2) &= \ln p(\mathcal{D}|\mu, \sigma)= \sum_{i=1}^{n}\ln p(y^{(i)}|\mu, \sigma) \\
# &= \sum_{i=1}^{n} \left \{\cancel{-\frac{1}{2}\ln2\pi} -\frac{1}{2}\ln\sigma^2 -\frac{1}{2\sigma^2} (y^{(i)} - \mu)^2\right \} \\
# &= -\sum_{i=1}^{n}\frac{1}{2}\ln\sigma^2 -\sum_{i=1}^n\frac{1}{2\sigma^2} (y^{(i)} - \mu)^2 + C\\
# &= - \frac{n}{2}\ln\sigma^2-\frac{1}{2\sigma^2}\sum_{i=1}^n (y^{(i)} - \mu)^2 + C
# \end{align}
# ```
# """

# ╔═╡ 2be8f43f-b13e-4b9b-a22f-171c59dd2933
md"""

## Gaussian MLE (cont.)



```math
\Large
\begin{align}
\hat{\mu}, \hat{\sigma}^2\leftarrow \arg\max_{\mu, \sigma^2}\; \ell(\mu,\sigma^2)
\end{align}
```

##### To find MLE, take derivative and solve it for zero


##### And the results are 


```math
\Large
\hat{\mu} = \frac{1}{n} \sum_{i=1}^n y^{(i)};\;\; \hat{\sigma^2} = \frac{1}{n}\sum_{i=1}^n (y^{(i)} - \hat{\mu})^2

```

* *i.e.* the MLE are just sample mean and sample variance
"""

# ╔═╡ 47d289dc-61a0-4a90-828d-5f8ed711b245
md"""

## Exercise 


!!! question ""
	##### Find the derivative 
	```math
	\large
		\frac{\partial \ell(\mu,\sigma^2)}{\partial \mu}, \frac{\partial \ell(\mu,\sigma^2)}{\partial \sigma^2}
	```
	* ##### set them to zero and verify the MLE are sample mean/variance
	* ##### and recall that 
	```math
		\ell(\mu,\sigma^2) = - \frac{n}{2}\ln\sigma^2-\frac{1}{2\sigma^2}\sum_{i=1}^n (y^{(i)} - \mu)^2
	```

"""

# ╔═╡ 7d72d5cd-97bc-4aec-a9bf-ddf11d162bc2
# md"""

# ## MLE properties



# > ##### Consistency:  
# > $$\Large n\rightarrow \infty,\;\; \hat{\theta}_{MLE} \rightarrow \theta_{true}$$


# * ###### as training sample size ``n`` gets large, ``\hat{\theta}_{MLE}`` converge to the true value
# * under certain regularity conditions*
# """

# ╔═╡ 1cb2ad7b-f3fc-4ab0-a2e8-ed0f7f0b617f
# let
# 	Random.seed!(1111)
# 	n_exps = 20
# 	nn = Int(1e4*3)
# 	# sample_sizes = Int[10, 100, 10000, nn]
# 	samples = randn(nn)
# 	nsizes = Int.(floor.(range(1, nn, 300) |> collect))
# 	mles = [] 
# 	for n in nsizes
# 		push!(mles, mean(samples[1:n]))
# 	end

# 	plt = plot(mles, c=1, marker=:circle, ms=1.5, label="MLE: " * L"\hat{\theta}", legendfontsize=12, xlabelfontsize=15, ylims = (-0.8, 0.8), xscale=:log2, framestyle=:origins, xaxis=false, xlabel="Sample size: n")

# 	# if show_more
# 		# for i in 1:n_exps
# 		# 	samples_ = randn(nn)
# 		# 	mles = [] 
# 		# 	for n in nsizes
# 		# 		push!(mles, mean(samples_[1:n]))
# 		# 	end
# 		# 	plot!(plt, mles, c=i+1, marker=:circle, ms=1.5, legendfontsize=12, xlabelfontsize=15,  label="", xscale=:log2)
# 		# end
# 	# end

# 	hline!([0], lw=2, lc=1, label="True "*L"\theta")
# 	plt
# end

# ╔═╡ b86f814e-ccba-437b-8b8d-430b55f2c0ce
# md"""

# ## MLE properties


# > ##### Efficiency:  
# > $$\Large n\rightarrow \infty,\;\; \text{var}[\hat{\theta}_{MLE}] \rightarrow 0$$

# * ###### MLE not only correct, but also reliable
# * ###### MLE is the most efficient, the variance reduces to zero faster than any other alternatives

# """

# ╔═╡ 431021e7-a0eb-46ef-84d1-5ae946e087d2
# md"Repeat the process: $(@bind show_more CheckBox(false))"

# ╔═╡ 7481b9d3-1f15-4097-9d10-2076dca0fe78
# let
# 	Random.seed!(1111)
# 	n_exps = 20
# 	nn = Int(1e4*3)
# 	samples = randn(nn)
# 	nsizes = Int.(floor.(range(1, nn, 300) |> collect))
# 	mles = [] 
# 	for n in nsizes
# 		push!(mles, mean(samples[1:n]))
# 	end

# 	plt = plot(mles, c=1, marker=:circle, ms=1.5, label="MLE: " * L"\hat{\theta}", legendfontsize=12, xlabelfontsize=15, ylims = (-0.8, 0.8), xscale=:log10, framestyle=:origins, xaxis=false, xlabel="Sample size: n")

# 	if show_more
# 		for i in 1:n_exps
# 			samples_ = randn(nn)
# 			mles = [] 
# 			for n in nsizes
# 				push!(mles, mean(samples_[1:n]))
# 			end
# 			plot!(plt, mles, c=i+1, marker=:circle, ms=1.5, legendfontsize=12, xlabelfontsize=15,  label="", xscale=:log10)
# 		end
# 	end

# 	hline!([0], lw=2, lc=1, label="True "*L"\theta")
# 	plt
# end

# ╔═╡ 6de7f8fd-d564-46ac-982b-dd9ea75f8a67
# md"""sample size: $(@bind nn_size Slider(10:10:5_0000)), add alternative estimate: $(@bind add_alternative CheckBox(false))"""

# ╔═╡ 17dded4c-d2d8-44d8-bd13-6438861e7a51
# let
# 	Random.seed!(123)
# 	nn = nn_size
# 	ws = exp.(randn(5_0000) * 2)
# 	samples = randn(5_0000, 100)
# 	plt = histogram(mean(samples[1:nn, :], dims=1)[:], nbins=20, xlims=[-1.0, 1.0], label="MLE estimators", xlabel=L"\hat{\theta}", ylabel="Count", title="Sample size $(nn)", alpha=0.5, legendfontsize=10)
# 	# density!(mean(samples[1:nn, :], dims=1)[:])

# 	if add_alternative
	
# 		# ws = ws/sum(ws)
# 		histogram!([mean(ss[1:nn], weights(ws[1:nn])) for ss in eachcol(samples)], nbins=20, xlims=[-0.8, 0.8], label="Non-MLE estimators", alpha=0.5)
# 	end

# 	plt
# end

# ╔═╡ 0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
md"""

# Appendix
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

# ╔═╡ 7af755d1-2d86-49c6-a58f-afc55a1c1631
# md"""
# ## Joint distribution


# #### A random variable set ``\mathbf{X}= \{X_1, X_2, \ldots, X_n\}`` 


# ```math
# \Large
# \begin{equation} P(X_1= x_1, X_2=x_2,\ldots, X_n= x_n)
# \end{equation} 
# ```

# * ##### joint event ``(X_1=x_1)\wedge (X_2=x_2)\wedge \ldots``'s distribution

# * ##### it still must statisfy

# $\large P(x_1, x_2, \ldots, x_n) \geq 0\;\; \text{and}\;\;  \sum_{x_1, x_2, \ldots, x_n} P(x_1, x_2, \ldots, x_n) =1$ 

# #### for example

# * ##### joint distribution: temperature (``T``) and weather (``W``)
# ```math
# \large
# \begin{equation}
# \begin{array}{c c |c} 
# T & W & P(T, W)\\
# \hline
# hot & sun & 0.4 \\
# hot & rain & 0.1 \\
# cold & sun  & 0.2\\
# cold & rain & 0.3\end{array} \end{equation} 
# ```



# """

# ╔═╡ a1c40faa-0f7c-47d9-bffc-8f5c9cee1994
# md"""
# ## Two probability rules


# > #### Rule 1: *sum rule* 

# > #### Rule 2: *product rule* (*a.k.a* chain rule)




# """

# ╔═╡ 2a6fa3e6-9f50-4a76-9b19-bdb058e7cbea
# md"""
# ## Probability rule 1: sum rule



# > $$\large P(X_1) = \sum_{x_2}P(X_1, X_2=x_2);\;\; P(X_2) = \sum_{x_1}P(X_1=x_1, X_2),$$


# * ##### ``P(X_1), P(X_2)`` are called *marginal probability distribution*

# """

# ╔═╡ 05b5ab16-bd58-4769-9311-47609c536867
# md"""
# ## Example -- sum rule

# """

# ╔═╡ 72298cca-eb8d-467a-8122-0509d364aa89
# Resource(figure_url * "/figs4CS5010/sumrule_.png", :width=>800, :align=>"left")

# ╔═╡ 94be177e-d81e-4c5a-ba72-a86a088e8726
# md"""

# ## Conditional probability

# #### Conditional probability

# $$\Large P(A=a|B=b) = \frac{P(A=a, B=b)}{P(B=b)}$$

# * ##### read: *probability of ``A`` given ``B``*
# * ##### the probability of $A=a$ given $B=b$ is true



# """

# ╔═╡ 273a7468-5b7b-4226-86ca-8035750adf02
# Resource(figure_url * "/figs4CS5010/condiprob.png", :width=>800, :align=>"left")

# ╔═╡ 3c236cae-f522-4e9d-b87c-99c67b13c16a
# md"""
# ##
# #### _For example_,

# """

# ╔═╡ 2379d2af-7d0f-4dbf-8dab-0f2e9018e7a1
# TwoColumn(md"""
# ##### Give joint
# ```math
# \large
# \begin{equation}
# \begin{array}{c c |c} 
# T & W & P(T, W)\\
# \hline
# hot & sun & 0.4 \\
# hot & rain & 0.1 \\
# cold & sun  & 0.2\\
# cold & rain & 0.3\end{array} \end{equation} 
# ```

# """, md"""
# ##### what is ``P(T=hot|W=sun)``?


# ```math
# \begin{align}
# P(T=h|W=sun) &= \frac{P(T=h, W=s)}{P(W=s)}\\
# &=\frac{0.4}{ 0.4 + 0.2}
# \end{align}
# ```
# """)

# ╔═╡ e19950e2-1505-4695-8617-bcb9119b28b5
# md"""

# ## Independence

# ##### If random variable $X,Y$ are *independent*, then

# ```math
# \Large
# P(X,Y) = P(X)P(Y)
# ```


# ##### For _multiple_ independent random variables, 

# ```math
# \Large
# P(X_1, X_2, \ldots, X_n) = \prod_{i=1}^n P(X_i)
# ```
# """

# ╔═╡ 5f21100b-3698-456d-91cf-16756be54356
# md"""
# ## Probability rule 2: product rule


# > $$\Large P(X, Y) = P(X)P(Y|X);\;\; P(X, Y) = P(Y)P(X|Y)$$


# * ##### the general joint factor formula
# * ##### the chain order doesn't matter



# """

# ╔═╡ fa444cca-2da6-45c7-bbe7-32c8a2334ecc
# md"""
# ## Example -- product rule

# > $$\large P(D, W) = P(W)P(D|W)$$

# """

# ╔═╡ ac90ae8d-bbc5-4fc5-9a2d-8d0d521fa335
# Resource(figure_url * "/figs4CS5010/prodrule2.png", :width=>800, :align=>"left")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GaussianProcesses = "891a1506-143c-57d2-908e-e1f8e92e6de9"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
DataFrames = "~1.6.1"
Distributions = "~0.25.107"
GaussianProcesses = "~0.12.5"
HypertextLiteral = "~0.9.5"
Images = "~0.26.0"
LaTeXStrings = "~1.3.1"
Latexify = "~0.16.1"
LogExpFunctions = "~0.3.26"
Plots = "~1.40.0"
PlutoTeachingTools = "~0.2.14"
PlutoUI = "~0.7.55"
StatsBase = "~0.33.21"
StatsPlots = "~0.15.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "002befbf3c880770e9afcecbe3a1ca43f761ba5e"

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
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
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
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

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
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "5a97e67919535d6841172016c9530fd69494e5ec"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.6"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

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
git-tree-sha1 = "b5278586822443594ff615963b0c09755771b3e0"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.26.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

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

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

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
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

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

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

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

[[deps.ElasticArrays]]
deps = ["Adapt"]
git-tree-sha1 = "75e5697f521c9ab89816d3abeea806dfc5afb967"
uuid = "fdbdab4c-e67f-52f5-8c3f-e7b388dad3d4"
version = "1.2.12"

[[deps.ElasticPDMats]]
deps = ["LinearAlgebra", "MacroTools", "PDMats"]
git-tree-sha1 = "03ec11d0151e8a772b396aecd663e1c76fc8edcf"
uuid = "2904ab23-551e-5aed-883f-487f97af5226"
version = "0.2.3"

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

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

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

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "58d83dd5a78a36205bdfddb82b1bb67682e64487"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "0.4.9"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

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

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "f089ab1f834470c525562030c8cfde4025d5e915"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.27.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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

[[deps.GaussianProcesses]]
deps = ["Distances", "Distributions", "ElasticArrays", "ElasticPDMats", "FastGaussQuadrature", "ForwardDiff", "LinearAlgebra", "Optim", "PDMats", "Printf", "ProgressMeter", "Random", "RecipesBase", "ScikitLearnBase", "SpecialFunctions", "StaticArrays", "Statistics", "StatsFuns"]
git-tree-sha1 = "31749ff6868caf6dd50902eec652a724071dbed3"
uuid = "891a1506-143c-57d2-908e-e1f8e92e6de9"
version = "0.12.5"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43ba3d3c82c18d88471cfd2924931658838c9d8f"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.0+4"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1dc470db8b1131cfc7fb4c115de89fe391b9e780"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.12.0"

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

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

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
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "acf614720ef026d38400b3817614c45882d75500"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.4"

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
git-tree-sha1 = "3447781d4c80dbe6d71d239f7cfb1f8049d4c84f"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.6"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "437abb322a41d527c197fa800455f79d414f0a3c"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.8"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d65554bad8b16d9562050c67e7223abf91eaba2f"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.13+0"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "6f0a801136cb9c229aebea0df296cdcd471dbcd1"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.5"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "783b70725ed326340adf225be4889906c96b8fd1"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.7"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "44664eea5408828c03e5addb84fa4f916132fc26"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.8.1"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "e0884bdf01bbbb111aea77c348368a86fb4b5ab6"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.10.1"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageBinarization", "ImageContrastAdjustment", "ImageCore", "ImageCorners", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "12fdd617c7fe25dc4a6cc804d657cc4b2230302b"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.26.1"

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

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "b842cbff3f44804a84fda409745cc8f04c029a20"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.6"

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

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
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
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "Requires", "TranscodingStreams"]
git-tree-sha1 = "a0746c21bdc986d0dc293efa6b1faee112c37c28"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.53"

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

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

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

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll"]
git-tree-sha1 = "fa7fd067dca76cadd880f1ca937b4f387975a9f5"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.16.0+0"

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

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "8084c25a250e00ae427a379a5b607e7aed96a2dd"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.171"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

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

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

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

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "1130dbe1d5276cb656f6e1094ce97466ed700e5a"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.2"

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

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

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
git-tree-sha1 = "0a41c2d8e204a3ad713242139628e01a29556967"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.5.3+0"

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

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "c1f51f704f689f87f28b33836fd460ecf9b34583"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.11.0"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

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

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

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

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "7d1d27896cadf629b9a8f0c2541cca215b958dc0"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.15"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

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

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

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
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
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
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e60724fd3beea548353984dc61c943ecddb0e29a"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.3+0"

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
git-tree-sha1 = "fea870727142270bdf7624ad675901a1ee3b4c87"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.1"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "456f610ca2fbd1c14f5fcf31c6bfadc55e7d66e0"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.43"

[[deps.ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

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

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "4b33e0e081a825dbfaf314decf58fa47e53d6acb"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.4.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

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
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

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

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "87d51a3ee9a4b0d2fe054bdd3fc2436258db2603"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.1.1"

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
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5950925ff997ed6fb3e985dcce8eb1ba42a0bbe7"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.18"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a6b1675a536c5ad1a60e5a5153e1fee12eb146e3"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.0"

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

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "38f139cc4abf345dd4f22286ec000728d5e8e097"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.10.2"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

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
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "4ab62a49f1d8d9548a1c8d1a75e5f55cf196f64e"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.71"

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

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

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
# ╟─9f90a18b-114f-4039-9aaf-f52c77205a49
# ╟─50752620-a604-442c-bf92-992963b1dd7a
# ╟─3e2e1ea8-3a7d-462f-ac38-43a087907a14
# ╟─12d6f93e-24c7-470f-a54b-4947a56480ec
# ╟─7e53d524-bffc-4648-ba36-22a3f4dd1eee
# ╟─7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
# ╟─bc96a33d-9011-41ec-a19e-d472cbaafb70
# ╟─7091d2cf-9237-45b2-b609-f442cd1cdba5
# ╟─adacdf8a-bc6a-4805-9372-724eb73e7620
# ╟─3b4a2f77-587b-41fd-af92-17e9411929c8
# ╟─fbfb86e2-3a6d-4127-9dc9-823bbd7aa26b
# ╟─3d99ee38-80cc-47f4-8b70-a6253c78e3bd
# ╟─5bd15469-e888-4505-a53d-49fef3329ea4
# ╟─9e27864c-f5cb-4780-bcd3-e3d29a69742a
# ╟─c9e0eaae-b340-434e-bdc9-dfdbc747221e
# ╟─1e927760-4dc6-4ecf-86ec-8fed16dbb0d6
# ╟─eb8af12f-ca58-4c51-a434-0925c099ffa3
# ╟─5ae8261b-283f-4bdf-b188-07f62f4c339e
# ╟─95c88b47-f702-44b0-bbb2-f305132a2db0
# ╟─1d62b70b-c30c-458a-9d19-835685d38d12
# ╟─ba8fac78-31b1-427c-8f78-bc545751be41
# ╟─dd2c51d5-2a49-44db-801b-4c37f32ce001
# ╟─1ea7dad6-1a8d-40e4-8248-069e18d6e646
# ╟─59a38e32-c2f3-465f-928d-c05f8d69f496
# ╟─e8fd61f1-33a6-43d8-8056-fb7cf97291b5
# ╟─81ab9972-07bc-4ce9-9138-3359d4e34025
# ╟─a7a24713-a29a-4f0c-996b-f98305bac09c
# ╟─fc9e9bb6-2287-46c8-8518-c9d0804c094e
# ╟─2ce6c56b-733c-42e8-a63b-d774cb6c199c
# ╟─dc8a3e36-2021-42dd-bc49-0eb6ab784fac
# ╟─2e4df75b-0778-4ed4-840a-417da2d65204
# ╟─8730b9a2-a1b4-456c-974c-ecd8880e6834
# ╟─c5be7eb8-e0b6-48cc-8dbe-788fa6624999
# ╟─2f7003a0-8dc8-4c2b-a3f3-0ff4a089e8f7
# ╟─bce5c041-be39-4ed1-8935-c389293400bc
# ╟─db6eb97c-558c-4206-a112-6ab3b0ad04c8
# ╟─1afdb42f-6bce-4fb1-860a-820d98df0f9d
# ╟─ff61cd9d-a193-44b3-a715-3c372ade7f79
# ╟─128b6ad8-aa21-4d0a-8124-2cf433bc79c4
# ╟─4d99c216-e32f-43a3-a122-ccbb697711fc
# ╟─f0028ab9-593b-4d25-a59a-e93aecd14ad4
# ╟─403af436-d7f2-43c0-803a-8104ba69fcfd
# ╟─68742b9e-e94e-40fb-a120-0c83f99847ce
# ╟─5b500acf-7029-43ff-9835-a26d8fe05194
# ╟─4bf768de-833f-45bf-9429-4820ff61553f
# ╟─656da51f-fd35-4e89-9af5-b5f0fdf8618f
# ╟─7c03a15f-9ac1-465f-86a4-d2a6087e5970
# ╟─80038fee-b922-479d-9687-771e7e258fcf
# ╟─e28e8089-f52b-440a-9861-895f9c378c84
# ╟─c2d14bf1-f16c-4573-a862-4d2dc73d39c0
# ╟─1e52d388-1e8d-4c20-b6e7-bcdd674ea406
# ╟─0623a15a-4361-4dda-86b6-782a2c9bc609
# ╟─c2f2b12f-39a1-47aa-839f-f181b0b30236
# ╟─176afbd7-fb8c-4710-9e7d-bde3c16e9a2d
# ╟─d0801b01-062c-4fcd-921b-84cf37ddc447
# ╟─21a88f9c-5153-4c65-80fd-80aee753971c
# ╟─21fe71dc-6f9d-4945-9c41-db4ec252980d
# ╟─946f34ae-d8c2-444f-b2d7-042d8c258e33
# ╟─5fdd4820-2cd3-40bb-b1b1-9647af7156ca
# ╟─6271dfaf-9499-4561-a06e-dcf63cac33be
# ╟─a4980317-32aa-44b8-97a8-8887c0e65bb4
# ╟─197e2d17-fd19-46b1-8f51-0fa2748340e5
# ╟─765ee957-c9fc-4144-b893-992e389d273d
# ╟─2bad0f9a-2b21-4686-aa41-2b430c354454
# ╟─9594d76f-3274-48fa-b833-f0c26daa229a
# ╟─ce1d7bad-179a-48c0-86c7-2de82c55a96d
# ╟─09f78f45-3790-4218-847f-b9ea1e61176a
# ╟─b89ac105-597e-44ac-9b58-c1c3c5ac59e9
# ╟─2b66deba-7144-496e-b9f2-ecf96b0e42ba
# ╟─355c71f8-975e-4a65-b1a6-7e27a67b41d8
# ╟─5f7bd93c-4210-49ba-8191-5167a5936a3c
# ╟─c0d0cdc4-5c25-4f5d-b92f-9facc999e53e
# ╟─d3f51b03-384c-428c-b7e4-bdc1508e6a02
# ╟─875a06b7-eb90-451e-a888-3e4f13832053
# ╟─c2497681-0729-451a-ab5f-43937bc9e100
# ╟─00c8c5a4-c58f-4a62-ba88-ca3f590977d7
# ╟─06e178a8-bcd1-4646-8e51-1b90a2e09784
# ╟─ab5612b9-9681-4984-b58e-3783c0c0c6e4
# ╟─6e7ace1b-6c6f-44e4-8377-dd7804f94ee0
# ╟─cb3f15a1-3d04-447a-a5a2-50c66f356922
# ╟─43f6f92c-fe29-484f-ad1b-18a674574ef2
# ╟─72af797b-5340-482e-be00-2cda375dd734
# ╟─723365e7-1fad-4899-8ac1-fb8674e2b9a7
# ╟─a862e9d6-c31d-4b21-80c0-e359a5435b6b
# ╟─ce53b75a-3b3c-4ca9-87ec-10f6ec2d38b2
# ╟─1659e22a-716a-40c3-9efd-839cb94658c3
# ╟─bfe99268-6097-494a-886f-58709e33cd01
# ╟─58192d79-796b-42ef-8ed0-09b0b0473066
# ╟─89ec84d5-6a21-4760-a482-2768624baec7
# ╟─1a6b3ec2-b086-4f62-9d9b-5956f67f6c9d
# ╟─729746ec-9503-4598-ab4a-6794b5fa4192
# ╟─bcec824a-c3ca-4041-a5c1-ad42abfe2f99
# ╟─c1cec1ca-48c5-46ce-9017-74376fc34c98
# ╟─6f756993-89fd-4e90-a551-68f4c7cb5ced
# ╟─a91eedec-0549-4c17-b973-13ac1770d808
# ╟─8cacc332-fb7f-4266-b49d-280692e87891
# ╟─6b338615-8f0c-4dc2-bc27-b88642db489a
# ╟─d52dfe3e-46d5-460b-b4de-3661212b9f58
# ╟─e557ad8b-9e4f-4209-908f-2251e2e2cde9
# ╟─62627b47-5ec9-4d7d-9e94-2148ff198f66
# ╟─fc09b97a-13c9-4721-83ca-f7caa5f55079
# ╟─ef112987-74b4-41fc-842f-ebf1c901b59b
# ╟─95779ca4-b743-43f1-af12-6b14c0e28f0b
# ╟─0f280847-2404-4211-8221-e30418cf4d42
# ╟─b682cc8d-4eeb-4ecd-897c-e15a3e40f76d
# ╟─183721ea-de2d-44fd-b429-d75d655a9094
# ╟─9b77f11e-3a8f-4b05-b02a-291fae630583
# ╟─fd83e1c0-0ef9-4887-b1e7-3ca2868ed5b8
# ╟─2b4d26bd-8636-4f8c-9152-fd80b7d73b87
# ╟─2773dab9-f6ca-40b9-8b52-9deb9ae23557
# ╟─65de12b9-1977-4c2d-9911-9284df73cf44
# ╟─1be833fd-8ae7-4f2c-852c-824e91ddeb9a
# ╟─d9dfcc42-e0e5-4410-8c2c-0ccbd8aaa3b5
# ╟─990ddc33-c000-46ea-83a0-746e946f28dc
# ╟─90f00871-97c5-4e43-93e0-37b99b488d37
# ╟─0a58b41e-9b5d-4954-9c24-af993a3e2f00
# ╟─ed2017ad-04a5-43e5-9690-5d92b5fb92de
# ╟─5dd53b59-4f79-4fbb-9a17-9705189a9cbf
# ╟─daac7901-5e1c-409e-81f0-816888cfe34b
# ╟─6dfc3afd-412e-4890-804f-3951d19470a6
# ╟─86369522-1eaf-4f43-ace5-7d991ef78564
# ╟─4e2f3597-6ea3-4006-8dda-5151ebac5ba7
# ╟─3a689042-ce56-427c-b567-4c71b74dc77f
# ╟─f5cf6163-ea7b-4e47-98c8-f862e3a2ebd7
# ╟─9c53de83-7552-4d72-8f02-bfb82230e440
# ╟─66d1a512-5476-4dda-a383-b5ced3f99514
# ╟─8beaa73a-87c6-4042-9968-4e881c04a033
# ╟─fc5bc22c-7bd4-4b76-b93f-453806bc5cd6
# ╟─8cf468ad-ef03-47fb-8e7a-155f12ad0133
# ╟─1be5a5b4-4c14-4b6f-9c07-3ffe82dfb849
# ╟─3f368884-9979-4cea-b462-ba67bc47d517
# ╟─71bc71e7-ddf2-4861-ba17-3279950f1f93
# ╟─ad9b4a60-c1aa-46bb-a3e2-8e9eee0119ef
# ╟─6698f6de-52e4-4f0e-a7d6-818fa1288e44
# ╟─95149606-f17a-4e4b-9d44-6cbd68509d86
# ╟─3c88504f-4f4e-4c51-98c4-b27e3f6d9f61
# ╟─99334132-f0ee-4c62-8e1b-fc3136700ae5
# ╟─e6529b7b-ea95-44be-84c1-609dfd3c52a7
# ╟─d5303044-c534-4b37-a126-cec8cfd97ec9
# ╟─a2d9d8f7-85e1-4fbb-8683-95a301031da5
# ╟─6fb13b48-bbaa-469e-b327-c64c7e2d4bfd
# ╟─3fc3b46a-39fc-448b-b2ca-eb7a787bb480
# ╟─697e2973-ee2a-4227-bc19-f34fc7b1bcb9
# ╟─0e452e81-91b8-4724-becb-ce96cb3f1b8a
# ╟─8a00508d-c275-4ef5-8568-67e1e3d1187d
# ╟─15b8b453-0010-4bb6-99f7-6e5723759cd1
# ╟─821ea80c-a058-41e2-8cb5-f6c4519fc662
# ╟─3b1ca12b-ed16-42c3-b3e2-377052fa2d37
# ╟─1ecc24bf-6263-4279-9549-aeb75b805294
# ╟─c475b487-2546-4bd7-b3e5-b931e5db6277
# ╟─d79c7f60-2e00-4ba4-9e47-edbc9154c1d3
# ╟─50eacaac-ddf9-4f54-8c5a-d947cb51d7ad
# ╟─f73d838c-f2dc-4540-9171-df8d188d14ff
# ╟─30c37f2b-8dbf-4146-99c9-fc16e8bbe700
# ╟─85ec65ef-c7c7-400b-bbb5-bbdfb8875c60
# ╟─f721bc14-42d7-4f0d-971f-2fc2d4b85818
# ╟─3235ea89-44cc-42b9-b6ed-be0dd2d0ef08
# ╟─27df7f09-4a0f-40eb-826e-90635e996dc8
# ╟─697bc4e0-d74d-459a-9d22-1b5716b8a11d
# ╟─4b7c5bc1-88c3-4352-8ad3-91abd2f03597
# ╟─831c0440-06a2-429c-b2e9-1dc09467523f
# ╟─c5c4a329-875a-4eba-a95e-97f4b2702f66
# ╟─bcf33704-2f9d-4b6b-a7c9-6c1ad0349273
# ╟─ef7c0dc7-d5d6-4dc1-91fa-2508363b3931
# ╟─2be8f43f-b13e-4b9b-a22f-171c59dd2933
# ╟─47d289dc-61a0-4a90-828d-5f8ed711b245
# ╟─7d72d5cd-97bc-4aec-a9bf-ddf11d162bc2
# ╟─1cb2ad7b-f3fc-4ab0-a2e8-ed0f7f0b617f
# ╟─b86f814e-ccba-437b-8b8d-430b55f2c0ce
# ╟─431021e7-a0eb-46ef-84d1-5ae946e087d2
# ╟─7481b9d3-1f15-4097-9d10-2076dca0fe78
# ╟─6de7f8fd-d564-46ac-982b-dd9ea75f8a67
# ╟─17dded4c-d2d8-44d8-bd13-6438861e7a51
# ╟─0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
# ╟─8687dbd1-4857-40e4-b9cb-af469b8563e2
# ╟─fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
# ╟─7af755d1-2d86-49c6-a58f-afc55a1c1631
# ╟─a1c40faa-0f7c-47d9-bffc-8f5c9cee1994
# ╟─2a6fa3e6-9f50-4a76-9b19-bdb058e7cbea
# ╟─05b5ab16-bd58-4769-9311-47609c536867
# ╟─72298cca-eb8d-467a-8122-0509d364aa89
# ╟─94be177e-d81e-4c5a-ba72-a86a088e8726
# ╟─273a7468-5b7b-4226-86ca-8035750adf02
# ╟─3c236cae-f522-4e9d-b87c-99c67b13c16a
# ╟─2379d2af-7d0f-4dbf-8dab-0f2e9018e7a1
# ╟─e19950e2-1505-4695-8617-bcb9119b28b5
# ╟─5f21100b-3698-456d-91cf-16756be54356
# ╟─fa444cca-2da6-45c7-bbe7-32c8a2334ecc
# ╟─ac90ae8d-bbc5-4fc5-9a2d-8d0d521fa335
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
