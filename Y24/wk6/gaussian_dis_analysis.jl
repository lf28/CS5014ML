### A Pluto.jl notebook ###
# v0.19.38

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

# ╔═╡ 120a282a-91c1-11ec-346f-25d56e50d38c
begin
	using Distributions, StatsBase, Clustering
	using StatsPlots
	using PalmerPenguins, DataFrames
	using LogExpFunctions:logsumexp
	using Flux
end

# ╔═╡ 5aa0adbe-7b7f-49da-9bab-0a78108912fd
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

# ╔═╡ fc9cffe8-b447-4ea0-bd0b-cdc811305620
begin
	using Logging
	Logging.disable_logging(Logging.Info); # or e.g. Logging.Info
end;

# ╔═╡ 22e1fbc9-f0bd-4159-b92f-11c412a660e6
using MLJLinearModels

# ╔═╡ 654ecc9d-731a-4ce7-ac45-f6d9d229c59e
using Zygote

# ╔═╡ 7d3f2a61-fd98-4925-a932-04bec1ba1c3c
using DistributionsAD

# ╔═╡ 7b3003cc-797c-48f5-b8e9-6c2aba9f82da
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ e9e06a13-5307-4e62-b797-d2d2e1f8ac70
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

# ╔═╡ 646dd3d8-6092-4435-aee9-01fa6a281bdc
ChooseDisplayMode()

# ╔═╡ 093c4c78-6179-4196-8d94-e548621df69b
TableOfContents()

# ╔═╡ 16497eaf-3593-45e0-8e6a-6783198663c3
md"""

# CS5014 Machine Learning


#### Gaussian discriminant analysis
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ f7e989fd-d955-4323-bdef-57d9ffbe5a18
begin
	ENV["DATADEPS_ALWAYS_ACCEPT"] = true
	table = PalmerPenguins.load()
 	df = DataFrame(table)
end;

# ╔═╡ e136edb5-3e98-4355-83e2-55761eb8b15c
md"""
## Why probabilistic approach

A key message I hope to convey here
!!! correct ""
	**Probabilistic models** unify most *interesting* machine learning models 
    * supervised learning
	* and also **unsupervised learning**

In other words: **from probabilistic models' eyes, they are the same**


**Machine learning** are just **probabilistic inferences**:

$$P(y|x)$$

* assume different $P(\cdot)$ and plug in different $x$ and $y$ for different problems/situations
  * e.g. regression: $P$ is Gaussian;
  * and classification: $P$ is Bernoulli or Multinoulli 
* we will come back to this key message at the end of next lecture

""";

# ╔═╡ c76aa603-f295-45f3-b55f-1d2d97b03c59
md"""

## Reading & references


##### Essential reading 


* **Multivariate Gaussian** [_Probabilistic Machine Learning_ by _Kevin Murphy_: Chapter 3.2.1-3.2.2](https://probml.github.io/pml-book/book1.html)


* **GDA** [_Probabilistic Machine Learning_ by _Kevin Murphy_: Chapter 9.2.1 to 9.2.5](https://probml.github.io/pml-book/book1.html)


##### Suggested reading 


* **Multivariate Gaussian** [_Pattern recognition and Machine Learning_ by _Christopher Bishop_: Chapter 2.3](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)




"""

# ╔═╡ b537d88b-5a40-4369-9208-c865be19125b
md"""

# Probabilistic generative models 
"""

# ╔═╡ 32aff8ba-a316-4756-9014-d2b194979acc
md"""

## Probabilistic `discriminative` models

#### -- _e.g._ softmax regression
"""

# ╔═╡ e5730f69-c3a6-4ac3-bead-b5e135efe70d
show_img("pdis_model.svg", w=500)

# ╔═╡ a114ac63-e38c-44c9-ae9b-a3c0c247b90c
md"""

##

#### `Discriminative` 
  * ##### it requires `discriminative` training instances to work 
  * ##### by _drawing decision boundaries_ btw the opposite classes
      * ###### *e.g.* if we only had one class, it won't work
"""

# ╔═╡ 17f05a30-11a0-4288-b28c-7ee33e271913
show_img("discrim_model_.svg", w=550)

# ╔═╡ db248681-0938-4762-80f6-7bf61f914128
md"""

## `discriminative` models limitations

#### How softmax regression classify ``\mathbf{x}_{test} =[15, 12]^\top``?

```math
\Large
p(y_{test}|\mathbf{x}_{test} = [15, 12]^\top,\mathbf{W})=?
```
"""

# ╔═╡ 661e876c-1966-466a-9295-04a2a2e37bca
show_img("discrim_model_problem.svg", w=400)

# ╔═╡ 84902be5-49c7-4c05-96f0-4f1bd818e2ad
md"""

## `discriminative` models limitations


#### How to classify this?

```math
\Large
p(y_{test} = \textit{chinstrap}|\mathbf{x}_{test} = [15, 12]^\top,\mathbf{W})=100\%
```

* ##### it will classify all given ``\mathbf{x}_{test}`` without questioning its validity
"""

# ╔═╡ d5bb961f-8f21-4282-a1e8-cd8a892cb33c
show_img("discrim_model_problem.svg", w=400)

# ╔═╡ e7d8fe14-40f6-499a-963c-f4342e84d964
md"""
## Probabilistic `generative` models 


"""

# ╔═╡ 1356c175-1e62-4465-85b2-10017667f29c
show_img("pgen_model.svg", w=500)

# ╔═╡ cdd0006b-f45f-4ee1-b17d-82262d304226
md"""


* ##### both ``\mathbf{x}`` and ``y`` are random with a joint distribution
"""

# ╔═╡ cc9d9e19-c464-4ff7-ae1f-bba4111075b3
md"""
## Probabilistic `generative` models 

##### (Gaussian discriminant analysis (GDA)) 



"""

# ╔═╡ c51a2ec7-3270-4dbe-92fd-520d78db2bf7
show_img("gaussiandis.svg", w=500)	

# ╔═╡ 3c2e0b13-f43b-4cfe-b6ce-49a2be0f7519
TwoColumnWideLeft(md"""
\


#### Both ``\mathbf{x}^{(i)}`` and ``y^{(i)}`` *random*

* ##### based on product rule ``p(X,Y) = P(Y)P(X|Y)``

* ##### its graphical model representation (right)

""", html"""<br/><center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gda_bnalone.svg" width = "250"/></center>
""")

# ╔═╡ 200b1531-2a94-472a-9fd2-b90b04877581
md"""
## Aside: probabilistic graphical models*


#### It specifies the joint distribution

```math
\Large
p(X_1, \ldots, X_N) = \prod_{i=1}^N p(X_i|\text{parent}(X_i))
```
"""

# ╔═╡ a4d6230d-9f6d-4ff3-aa0e-e77bc63873b2
TwoColumn(md"""
\
\
\


```math
\Large
p(y^{(i)},\mathbf{x}^{(i)}) = p(y^{(i)}) p(\mathbf{x}^{(i)}|y^{(i)})
```
""",
	html"""<br/><center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gda_bnalone.svg" width = "220"/></center>
""")

# ╔═╡ 25251600-dc00-48de-9509-35117c319e09
md"""

## Aside: probabilistic graphical models
#### -- node types
"""

# ╔═╡ 7dd25028-281f-4130-af09-881ae5015309
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/notations.png' width = '200' /></center>"

# ╔═╡ c4260ef4-d521-4d23-9495-41741940f128
# html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gda_bn.svg" height = "280"/></center>
# """

# ╔═╡ 9938af3d-0cad-4005-b2f6-db7fcf84b89a
md"""

## GDA example
"""

# ╔═╡ f033fbb8-55b3-40a7-a2e1-db8777daffc6
md"""
* #####  ``y^{(i)} \in \{\text{Adelie}, \text{Chinstrap}, \text{Gentoo}\}`` categorical
  * ##### ``p(y)``: the prior distribution over the three species

* #####  ``\mathbf{x}^{(i)}``: the bill length and depth measurements of an animal given the species ``y^{(i)}``
  * ##### *e.g.*``p(\mathbf{x}|y=\text{Chinstrap})``: a Gaussian (red cluster)
"""

# ╔═╡ d2af46c5-4795-497f-8f86-70b11cb95cb0
md"""




## Probabilistic generative models - GDA


#### -- data generating process ``p(y^{(i)}, \mathbf{x}^{(i)})``
"""

# ╔═╡ 6b27d9f7-e74c-404a-8248-c67cbbfe8342
gif_qda_gen = let
	# data_d1 = data₁[:, 1]
	nn = 20
	trueμs = [[-3.0, 2.0], [3.0, 2.0], [0., -2]]
	trueπs = [0.15, 0.7, 0.15]
	trueπs = 1/3 * ones(3)
	# mvns = [Normal(trueμs[k], trueσs[k]) for k in 1:3]
	# truezs = zeros(Int, nn)
	data = zeros(2, nn)
	Random.seed!(123)
	mvns = [MvNormal(trueμs[k], 1) for k in 1:3]
	truezs = rand(Categorical(trueπs), nn)
	anim_gen = @animate for (i, zi) in enumerate(truezs)
		plot(-7:0.1:7, -6:0.1:6, (x, y) -> pdf(mvns[zi], [x, y]), st=:surface, alpha=0.25, label="", c=zi, colorbar=false, framestyle=:none, size=(500,500), zaxis=false, xlabel=L"x_1", ylabel=L"x_2")
		data[:, i] = di = rand(mvns[zi])
		scatter!([di[1]], [di[2]], [0],markershape=:x, markerstrokewidth=5, c= zi, ms=8, label="sample "*L"y^{(%$i)}\sim \mathcal{Cat}(\pi)\;" * "; "*"sample "*L"x^{(%$i)}\sim \mathcal{N}_{%$(zi)}", legendfontsize=12, legend=:top)
		
		for c_ in 1:3
			zs_ = findall(truezs[1:i-1] .== c_)
			nc = length(zs_)
			if nc > 0
				scatter!(data[1, zs_], data[2, zs_], zeros(nc), markershape=:circle, ms=4, c= c_, alpha=0.5, label="")
			end
		end
	# 	# plot!((x) -> trueπs[2] * pdf(mvns[2], x),lc=(zi == 2 ? 2 : :gray), lw= (zi == 2 ? 2 : 1), label="")
	# 	# plot!((x) -> trueπs[3] * pdf(mvns[3], x),lc=(zi == 3 ? 3 : :gray), lw= (zi == 3 ? 2 : 1), label="")

	end

	# # truezs
	
	# # density(data_d1)
	gif(anim_gen, fps=1)
end;

# ╔═╡ 1a7f5b2f-c245-4c6e-854e-e8ce5ccf3f72
TwoColumnWideRight(md"""

\
\
\
\


----
#### for ``i = 1,2,..., n``
```math
\large
\begin{align}
y^{(i)} \;&\;\sim\; \mathcal{Cat}(\boldsymbol\pi) \\
\mathbf{x}^{(i)}|y^{(i)}=c \; &\;\sim\; \mathcal{N}(\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)
\end{align}
```
----



""", gif_qda_gen)

# ╔═╡ f1128d41-541d-49de-b0e3-644769c6e59f
md"""


## Probabilistic generative models - GDA


#### -- the full model ``p(y^{(i)}, \mathbf{x}^{(i)}|\mathbf{\Phi})``
"""

# ╔═╡ b334263b-211d-4188-b6a6-b629de7b8744
md"""

* #### the parameters $\mathbf{\Phi} = \left \{\boldsymbol{\pi},  \{\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c \}_{c=1}^C\right \}$
"""

# ╔═╡ ce6f3828-65b0-403e-bec0-6faa85899c27
TwoColumn(md"""


$$\large p(y^{(i)}|\boldsymbol{\pi})=\begin{cases}\pi_1 & y^{(i)}=1 \\ \pi_2 & y^{(i)} =2 \\\vdots & \vdots \\ \pi_C & y^{(i)} =C\end{cases}$$


\


$$\large 
\begin{align}p(\mathbf{x}^{(i)}&|y^{(i)} , \{\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c \}_{c=1}^C)\\
&=\begin{cases}\mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) & y^{(i)}=1 \\ \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)& y^{(i)} =2 \\\vdots & \vdots \\ \mathcal{N}(\boldsymbol{\mu}_C, \boldsymbol{\Sigma}_C) & y^{(i)} =C\end{cases}
\end{align}$$
""", show_img("qda_full.svg"))

# ╔═╡ a9e078c2-1768-4184-bc13-9aa885c0c645
md"""

## Recall: `choice to product trick`

#### -- Discrete choice function in one-liner

```math
\large
\begin{align}
f(X ) =\begin{cases}\theta_1 & x= 1 \\ \theta_2 & x=2 \\
\vdots \\

\theta_K & x=K
\end{cases}\;\;\Longleftrightarrow \;\; f(X =x) = \Large \prod_{k=1}^K\, \theta_k^{\mathbb{1}(x = k)}
\end{align}
```



* ##### _indicator function_: ``\mathbb{1}(\texttt{true}) = 1``, ``\mathbb{1}(\texttt{false}) = 0``




"""

# ╔═╡ 12c87fe2-4e5a-4a65-a06a-b6a59a274c3a
Foldable("Explanation", md"""

###### for example, for a ``K=3`` choice example

```math
\large
f(X=x) = \theta_1^{\mathbb{1}(x =1)} \cdot \theta_2^{\mathbb{1}(x =2)}  \cdot \theta_3^{\mathbb{1}(x =3)}
```


* ###### we can verify *e.g.* ``x= 2``

```math
\large
\begin{align}
f(X=2) &= \theta_1^{\mathbb{1}(2 =1)} \cdot \theta_2^{\mathbb{1}(2 =2)}  \cdot \theta_3^{\mathbb{1}(2 =3)}\\

&= \theta_1^{0} \cdot \theta_2^{1}  \cdot \theta_3^{0} \\
&= 1\, \cdot \,\theta_2 \cdot \, 1 \\
&= \theta_2
\end{align}
```

""")

# ╔═╡ ad768230-58cd-466f-b6ba-b8a0ca769277
md"""


## Probabilistic generative models - GDA


#### -- the full model ``p(y^{(i)}, \mathbf{x}^{(i)}|\mathbf{\Theta})`` in one-liners
"""

# ╔═╡ e7d9217d-b25b-43d5-9503-b82e76776d2a
TwoColumn(md"""

\
\


$$\large p(y^{(i)}|\boldsymbol{\pi})=\prod_{c=1}^C \left(\pi_c\right)^{\mathbb{1}(y^{(i)} =c)}$$


\


$$\large 
\begin{align}p(\mathbf{x}^{(i)}&|y^{(i)} , \{\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c \}_{c=1}^C)\\
&=\prod_{c=1}^C \mathcal{N}(\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)^{\mathbb{1}(y^{(i)} =c)}
\end{align}$$
""", show_img("qda_full.svg"))

# ╔═╡ 3a747ab4-8dc1-48ef-97ae-a736448020b3
md"""

# Multi-variate Gaussian
"""

# ╔═╡ 1f34d131-71e6-4ee6-96f2-124ca31ba531
let


	plt1 = plot(Normal(), title="Univariate Gaussian", label="", framestyle=:origins, lw=2, ylabel=L"p(x)", ratio=12, xlim =(-4,4), ylim =(0., 0.5), xlabel=L"x")

	plt2 = plot(-4:0.1:4, -4:0.1:4, (x, y) -> pdf(MvNormal(zeros(2), Matrix(I,2,2)), [x, y]),st=:surface, c=:coolwarm, title="Multivariate Gaussian", label="", framestyle=:origins, lw=2, colorbar=false, ylabel=L"x_2", xlabel=L"x_1", zlabel=L"p(\mathbf{x})")

	plot(plt1, plt2, size=(700,350))
end

# ╔═╡ a5e253b5-0869-49fe-8e8e-b7d12df45265
md"""


## Recap: Euclidean distance _btw_ ``\boldsymbol{\mu}`` _and_ ``\mathbf{x}``



"""

# ╔═╡ 1da36dc8-1ec6-4541-b2a0-d47e2cfd39f3
TwoColumn(md"""

#### Given ``\mathbf{x}, \boldsymbol{\mu} \in \mathbb{R}^d``,
\


$$\LARGE\mathbf{x} -\boldsymbol{\mu}$$

* ##### is a new vector


""",let
	μ = [1.0, 0.5]
	# plot(μ[1] .+ cos.(t), μ[2] .+ sin.(t), ratio=1,  xlim = μ .+ [-1.5, 1.5],lw=2, label="", framestyle=:origin,  size=(300,300))

	plt = scatter([μ[1]], [μ[2]], label="", marker=:x, markerstrokewidth=0, markersize=0, size=(300,300), ratio=1,c=2,  xlim = [-0.5, 2], ylim =  [-0.5, 2.2], framestyle=:origin)
	# θ = π/4

	quiver!([0], [0], quiver=([μ[1]], [μ[2]]), lw=1.5, lc=2)
	v = [1.5, 1.8]
	ii = 3
	scatter!([v[1]], [v[2]], c=1,  markersize=0, label="")
	quiver!([0], [0], quiver=([v[1]], [v[2]]), lw=1.5, lc=1)
	# plot!([μ[1], v[1]],  [μ[2], v[2]], lc=:gray, arrow=Plots.Arrow(:close, :both, 1,1),  lw=0.6, st=:path, label="")
	xmu = - v + μ 
	quiver!([μ[1]], [μ[2]], quiver=([-xmu[1]], [-xmu[2]]), lw=3, lc=3)	
	annotate!(μ[1], μ[2], text(L"μ", 20, :red,:top))
	annotate!(v[1], v[2], text(L"\mathbf{x}", 20, :blue,:bottom))
	
	θ = acos(dot(-xmu, [1,0])/ norm(xmu)) * 180/π
	annotate!(.5 * (μ[1] + v[1]), .5 * (μ[2] + v[2]),text(L"\mathbf{x}-\mu", 20, :green, :top, rotation=θ))

		# annotate!(0.5*b[1], 0.5*b[2], text(L"\sqrt{\mathbf{b}^\top\mathbf{b}}", 18, :bottom, rotation = θb ))
	plt

end )

# ╔═╡ 63a102df-1e17-488c-9381-3fbf16df0103
md"""


## Recap: Euclidean distance _btw_ ``\boldsymbol{\mu}`` _and_ ``\mathbf{x}``


"""

# ╔═╡ f3ec5dd2-0192-4a5f-a993-dc84f08ea98f
TwoColumn(md"""

#### Given ``\mathbf{x}, \boldsymbol{\mu} \in \mathbb{R}^d``,

$$\LARGE\mathbf{x} -\boldsymbol{\mu}$$


##### The Euclidean distance is its norm

$$\large\begin{align}\|\mathbf{x}&-\boldsymbol{\mu}\|_2^2=\sum_{j=1}^d (x_j - \mu_{j})^2\\
&=\begin{bmatrix}x_1- \mu_{1}& \ldots & x_d- \mu_{d}\end{bmatrix}\begin{bmatrix} x_1-\mu_{1} \\ \vdots \\ x_d-\mu_{d}\end{bmatrix} \\
&= {(\mathbf{x} - \boldsymbol\mu)^\top} {(\mathbf{x} - \boldsymbol\mu)}
\end{align}$$



""",let
	μ = [1.0, 0.5]
	# plot(μ[1] .+ cos.(t), μ[2] .+ sin.(t), ratio=1,  xlim = μ .+ [-1.5, 1.5],lw=2, label="", framestyle=:origin,  size=(300,300))

	plt = scatter([μ[1]], [μ[2]], label="", marker=:x, markerstrokewidth=0, markersize=0, size=(300,300), ratio=1,c=2,  xlim = [-0.5, 2], ylim =  [-0.5, 2.2], framestyle=:origin)
	# θ = π/4

	quiver!([0], [0], quiver=([μ[1]], [μ[2]]), lw=1.5, lc=2)
	v = [1.5, 1.8]
	ii = 3
	scatter!([v[1]], [v[2]], c=1,  markersize=0, label="")
	quiver!([0], [0], quiver=([v[1]], [v[2]]), lw=1.5, lc=1)
	# plot!([μ[1], v[1]],  [μ[2], v[2]], lc=:gray, arrow=Plots.Arrow(:close, :both, 1,1),  lw=0.6, st=:path, label="")
	xmu = - v + μ 
	quiver!([μ[1]], [μ[2]], quiver=([-xmu[1]], [-xmu[2]]), lw=3, lc=3)	
	annotate!(μ[1], μ[2], text(L"μ", 20, :red,:top))
	annotate!(v[1], v[2], text(L"\mathbf{x}", 20, :blue,:bottom))
	
	θ = acos(dot(-xmu, [1,0])/ norm(xmu)) * 180/π
	annotate!(.5 * (μ[1] + v[1]), .5 * (μ[2] + v[2]),text(L"\mathbf{x}-\mu", 20, :green, :top, rotation=θ))

		# annotate!(0.5*b[1], 0.5*b[2], text(L"\sqrt{\mathbf{b}^\top\mathbf{b}}", 18, :bottom, rotation = θb ))
	plt

end )

# ╔═╡ 7f3ef771-7db3-43f9-a7c5-aef1448741bc
md"""


## Recap: Euclidean distance _btw_ ``\boldsymbol{\mu}`` _and_ ``\mathbf{x}``


"""

# ╔═╡ 7acf4734-ccdb-4b2b-a94a-2362989b7b26
TwoColumn(md"""

#### Euclidean distance 



$$\Large\begin{align}\|\mathbf{x}&-\boldsymbol{\mu}\|_2^2\\
&= {(\mathbf{x} - \boldsymbol\mu)^\top} {(\mathbf{x} - \boldsymbol\mu)}\\ 
&= \boxed{(\mathbf{x} - \boldsymbol\mu)^\top  \mathbf{I}^{-1}  (\mathbf{x} - \boldsymbol\mu)}\end{align}$$

* ##### *quadratic form*, multi-var of ``(x-\mu)1^{-1} (x-\mu)``
* where

  $\mathbf{I} = \begin{bmatrix} 1 &0& \ldots & 0 \\
  0 &1 & \ldots & 0  \\
  \vdots & \vdots & \vdots & \vdots \\
  0 & 0 & \ldots & 1
  \end{bmatrix}$

""",
	
	let
	μ = [1.0, 0.5]
	plt = scatter([μ[1]], [μ[2]], label="", marker=:x, markerstrokewidth=0, markersize=0, size=(300,300), ratio=1,c=2,  xlims = (μ[1] -1.3, μ[1] +1.5), ylims = (μ[2] -1.2, μ[2] +1.5), framestyle=:origin)
	# θ = π/4

	quiver!([0], [0], quiver=([μ[1]], [μ[2]]), lw=0.9, ls=:dash, lc=2)
	v = [1.5, 1.8]*9/10
	ii = 3
	scatter!([v[1]], [v[2]], c=1,  markersize=0, label="")
	quiver!([0], [0], quiver=([v[1]], [v[2]]), lw=0.9, ls=:dash, lc=1)

	xmu = - v + μ 
	quiver!([μ[1]], [μ[2]], quiver=([-xmu[1]], [-xmu[2]]), lw=3, lc=3)	
	annotate!(μ[1], μ[2], text(L"μ", 20, :red,:top))
	annotate!(v[1], v[2], text(L"\mathbf{x}", 20, :blue,:bottom))
	annotate!([[0]], [[0]], text(L"\mathbf{0}", 14, :black,:top))

	θ = acos(dot(-xmu, [1,0])/ norm(xmu)) * 180/π
	annotate!(.5 * (μ[1] + v[1]), .5 * (μ[2] + v[2]),text(L"\mathbf{x}-\mu", 20, :green, :top, rotation=θ))

	r = norm(xmu)

	ts = range(0, 2π, 60)
	xs = cos.(ts) * r .+ μ[1]
	ys = sin.(ts) * r .+ μ[2]

	plot!(xs, ys, st=:path, c=3, ms=1,ls=:dash, lc=3, label="")

	# hline!([μ[2]], lw=1, lc=:black, label="")
	# vline!([μ[1]], lw=1, lc=:black, label="")


	plt

 

end )

# ╔═╡ 7635232b-2db3-4eaa-a09e-e22e744e8c6c
md"""


## Recap: Euclidean distance _btw_ ``\boldsymbol{\mu}`` _and_ ``\mathbf{x}``


"""

# ╔═╡ 98d5d365-1f48-4e3a-885f-e2331f5c5241
TwoColumn(md"""

#### Euclidean distance 



$$\Large\begin{align}&\|\mathbf{x}-\boldsymbol{\mu}\|_2^2\\
&= \boxed{(\mathbf{x} - \boldsymbol\mu)^\top  \mathbf{I}^{-1}  (\mathbf{x} - \boldsymbol\mu)}\end{align}$$


#### Comparing with 
$\large\|\mathbf{x}-\boldsymbol{0}\|_2^2 = ({\mathbf{x}-\mathbf{0})^\top  \mathbf{I}^{-1} (\mathbf{x} -\mathbf{0)}}$
* ##### it shifts `origin` from ``\mathbf{0}`` to ``\boldsymbol{\mu}``

""",
	
	let
	μ = [1.0, 0.5]
	plt = scatter([μ[1]], [μ[2]], label="", marker=:x, markerstrokewidth=0, markersize=0, size=(300,300), ratio=1,c=2,  xlims = (μ[1] -1.3, μ[1] +1.5), ylims = (μ[2] -1.2, μ[2] +1.5), framestyle=:none)
	# θ = π/4

	quiver!([0], [0], quiver=([μ[1]], [μ[2]]), lw=2, ls=:solid, lc=2)
	v = [1.5, 1.8]*9/10
	ii = 3
	scatter!([v[1]], [v[2]], c=1,  markersize=0, label="")
	quiver!([0], [0], quiver=([v[1]], [v[2]]), lw=0.3, ls=:dash, lc=1)

	xmu = - v + μ 
	quiver!([μ[1]], [μ[2]], quiver=([-xmu[1]], [-xmu[2]]), lw=3, lc=3)	
	annotate!(μ[1], μ[2], text(L"μ", 20, :red,:top))
	annotate!(v[1], v[2], text(L"\mathbf{x}", 20, :blue,:bottom))
	annotate!([[0]], [[0]], text(L"\mathbf{0}", 14, :black,:top))

	θ = acos(dot(-xmu, [1,0])/ norm(xmu)) * 180/π
	annotate!(.5 * (μ[1] + v[1]), .5 * (μ[2] + v[2]),text(L"\mathbf{x}-\mu", 20, :green, :top, rotation=θ))

	r = norm(xmu)

	ts = range(0, 2π, 60)
	xs = cos.(ts) * r .+ μ[1]
	ys = sin.(ts) * r .+ μ[2]

	plot!(xs, ys, st=:path, c=3, ms=1,ls=:dash, lc=3, label="")


	# xs = cos.(ts) * r 
	# ys = sin.(ts) * r 

	# plot!(xs, ys, st=:path, c=3, ms=1,ls=:dash, lc=3, lw=0.5, label="")

	hline!([μ[2]], lw=1, lc=:black, label="")
	vline!([μ[1]], lw=1, lc=:black, label="")

	hline!([0], lw=.3, lc=:black, label="")
	vline!([0], lw=.3, lc=:black, label="")

	plt

 

end )

# ╔═╡ 15e64dfe-b46d-4bca-889d-e4ff639c89cc
md"""


## Recap: quadratic form


#### Recall that the quadratic form's formula


```math
\Large
\mathbf{x}^\top\mathbf{A}\mathbf{x} = \sum_{ij} A_{ij} x_i x_j
```
"""

# ╔═╡ 16790822-bb9d-4933-92d8-5dbf4dded74e
md"""

#### For example

```math
\Large
\begin{align}
&\;\;\;\;\begin{bmatrix}x_1 & x_2\end{bmatrix}_{1\times 2}\begin{bmatrix}A_{11} & 0 \\ 0 & A_{22}\end{bmatrix}_{2\times 2}\begin{bmatrix}x_1 \\ x_2 \end{bmatrix}_{2\times 1} 
\end{align}
```

* ##### the result is ``\sum_i\sum_j A_{ij} x_i x_j= A_{11}x_1^2 + A_{22}x_2^2``

"""

# ╔═╡ 8162c635-81a8-4d60-905f-f73522c14efb
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

""")

# ╔═╡ 76a4705c-7a2d-4916-abb5-fa24e3818ebb
md"""
## Recap: univariate Gaussian



"""

# ╔═╡ 65bc9927-4a46-439b-be41-38ae3c6379d0
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gaussian_eq_1d.png' width = '600' /></center>"

# ╔═╡ d421195e-abd3-4e01-8380-04b51fd25fe9
md"""

## Dissect Gaussian

```math
\Large
{\color{darkorange}{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2}} \Longrightarrow  { \color{green}{ e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2} }} \Longrightarrow  {\color{purple}{\frac{1}{\sigma \sqrt{2\pi}}} e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2}}
```

"""

# ╔═╡ 37bdd1bd-5742-49f3-aa57-5f0c998322fb
md"``\mu``: $(@bind μ4_ Slider(-5:.1:5.0, default=1.0, show_value=true)),
``\sigma``: $(@bind σ4_ Slider(0.1:.1:2, default=1.0, show_value=true))"

# ╔═╡ ba4a6076-bbf6-446c-bb97-41825cfa2659
md"
Add ``-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2``: $(@bind add_kernel CheckBox(default=false)), 
Add exp ``e^{-\frac{1}{2}\left(\frac{x -\mu}{\sigma}\right)^2}``: $(@bind add_ekernel CheckBox(default=false)), 
Add final ``p(x)``: $(@bind add_px CheckBox(default=false))
"

# ╔═╡ 2a111f88-2187-4fdf-acc5-4a3572956d36
begin
	f1(x; μ=0, σ=1) = ((x - μ)/σ )^2
	f2(x; μ=0, σ=1) = -0.5 * f1(x; μ=μ, σ=σ)
	f3(x; μ=0, σ=1) = exp(f2(x; μ=μ, σ=σ))
	f4(x; μ=0, σ=1) = 1/(σ * sqrt(2π)) *exp(f2(x; μ=μ, σ=σ))
end;

# ╔═╡ bcdda8a3-c1ab-4e95-b07e-299a506abb91
let
	μ = μ4_
	σ = σ4_
	# f1(x) = ((x - μ)/σ )^2
	# f2(x) = -0.5* f1(x)
	# f3(x) = exp(f2(x))
	maxy = f4(μ; μ=μ, σ=σ) 
	# f4(x) = 1/(σ * sqrt(2π)) *exp(f2(x))
	# plot(f1, lw=2, label=L"\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= \left(({x-\mu})/{\sigma}\right)^2")
	if add_kernel
		plt = plot(range(μ -5, μ+5, 100), (x) -> f2(x; μ=μ, σ=σ), lw=1.5, lc=2,label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2} \left(({x-\mu})/{\sigma}\right)^2", framestyle=:origin, ylim=[-2, max(maxy + 0.1, 1.5)])
	else
		plt = plot(framestyle=:origin, ylim=[-2,1.5], xlim=[-2,2])
	end
	if add_ekernel
		plot!((x) -> f3(x; μ=μ, σ=σ), lw=1.5, lc=3, label=L"e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}", title=L"f(x)= e^{-\frac{1}{2}\left(({x-\mu})/{\sigma}\right)^2}")
	end
	vline!([μ], label="", ls=:dash, lw=2, lc=2, la=0.5)
	if add_px
		plot!(x -> f4(x; μ=μ, σ=σ), lw=3, lc=4, label=L"\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}", title="Gaussian prob. density")
	end
	plt
end

# ╔═╡ 2581cbd3-5aa4-43ce-bd32-5ad32fff58e2
md"""


## Multivariate Gaussian
"""

# ╔═╡ c5f80244-0c5e-4a60-845b-1a46ccc4e7d6
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gaussian_eq_md_alt.png' width = '900' /></center>"

# ╔═╡ 6352141c-92dc-4ca7-ae7b-37f5f01b1d32
md"""
#### Below is uni-variate Gaussian
"""

# ╔═╡ 544fd97b-3239-49c0-9814-064feebbc108
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gaussian_eq_1d_alt.png' width = '640' /></center>"

# ╔═╡ 16dcdf35-0d33-4061-9d5c-722465cc2c4f
md"""
## Multivariate Gaussian -- mean

#### Mean: ``\boldsymbol\mu \in \mathbb{R}^d`` (a vector)


```math
\large
\boldsymbol{\mu} =\begin{bmatrix}\mu_1 \\ \mu_2 \\\vdots \\ \mu_d \end{bmatrix}
```


"""

# ╔═╡ 4590bcf8-9c34-4fec-b4d0-093b45f10801
md"""

## Effect of ``\boldsymbol\mu``

"""

# ╔═╡ 2bbfc4a2-e099-4ad4-9668-5c3b8ede04d4
let
	gr()
	Sigma = [2 0; 0 1]

	μ = [3, 3]
	mvn = MvNormal(μ, Sigma)
	plt1 = plot(range(0-6, 0+6, 50),range(0-6, 0+6, 50), (x, y) -> pdf(MvNormal(zeros(2), Sigma), [x, y]), levels=8, ratio=1, framestyle=:origin ,c=:coolwarm, st=:contourf, colorbar=false, alpha=0.6, xlim =(-6, 6), ylim =(-6, +6), title=L"{\mu}=\mathbf{0}")

	# quiver!([0], [0], 	quiver=([μ[1]], [μ[2]]), c=:black, lw=1.5)
	# annotate!([μ[1]*2/3], [μ[2]*2/3], text(L"{\mu}", :top))
	plt2 = plot(range(0-6, 0+6, 50),range(0-6, 0+6, 50), (x, y) -> pdf(mvn, [x, y]), levels=8, ratio=1, framestyle=:origin ,c=:coolwarm, st=:contourf, colorbar=false, alpha=0.6,  xlim =(-6, 6), ylim =(0-6, 6), title=L"{\mu}=[3, 3]^\top")

	quiver!([0], [0], 	quiver=([μ[1]], [μ[2]]), c=:black, lw=1.5)
	annotate!([μ[1]*2/3], [μ[2]*2/3], text(L"{\mu}", :top))
	plot(plt1, plt2, size=(700,350), xlabel=L"x_1", ylabel=L"x_2")
end

# ╔═╡ 5b238aa9-df84-4dd8-894c-3ad1dcd62c08
md"""

## Multivariate Gaussian -- variance

#### Variance: ``\boldsymbol\Sigma`` a $d \times d$ matrix
```math
\large
\boldsymbol{\Sigma} =\begin{bmatrix}\sigma_1^2 & \sigma_{12} & \ldots & \sigma_{1d} \\ \sigma_{21}  & \sigma_2^2 & \ldots & \sigma_{2d}\\\vdots & \vdots & \vdots & \vdots \\ \sigma_{d1} &  \sigma_{d2} & \ldots & \sigma_{d}^2\end{bmatrix}
```

* ##### *symmetric*: ``\boldsymbol{\Sigma}^\top = \boldsymbol{\Sigma}``, therefore ``\sigma_{ij} =\sigma_{ji}``


* ##### _positive definite_: 

  $$\Large\mathbf{x}^\top \boldsymbol{\Sigma} \mathbf{x} > 0;\;\; \mathbf{x}^\top \boldsymbol{\Sigma}^{-1} \mathbf{x} > 0; \;\text{for all } \mathbf{x}\neq \mathbf{0} \in \mathbb{R}^d$$ 



* ##### what does `positive definite` imply here?

"""

# ╔═╡ 4cec5c11-1db1-478d-b6e3-c500149ae238
md"""

## Multivariate Gaussian 
"""

# ╔═╡ 0a9fee13-536c-4053-8f0b-eabcba1fc62f
md"""

## The distance metric


"""

# ╔═╡ 3009f053-9636-428b-9612-75696d785983
md"""



* #### this is known as the *mahalanobis distance*

  $\Large d_{\boldsymbol{\Sigma}}(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}$


* ##### it still follows `68-95-99` rule, *e.g.*

  $\large \mathbb{P}\left (d_{\boldsymbol{\Sigma}}(\mathbf{x}, \boldsymbol{\mu}) \leq 2\right) \approx 95\%$
  
"""

# ╔═╡ 11562cf7-afc1-46ec-8b69-29f4bb503546
md"""

## Dissect multivariate Gaussian
"""

# ╔═╡ ccb7b38a-2509-48c9-aee1-97b5b6a8d599
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gaussian_eq_md_alt.png' width = '900' /></center>"

# ╔═╡ 56abb43a-715f-4aec-8ab2-cde254a8b94c
md"
Add ``-\frac{1}{2}(x -\mu){\Sigma}^{-1}(x -\mu)``: $(@bind add_kernel2 CheckBox(default=true)), 
Add exp ``e^{-\frac{1}{2}(x -\mu){\Sigma}^{-1}(x -\mu)}``: $(@bind add_ekernel2 CheckBox(default=false)), 
Add final ``p(x)``: $(@bind add_px2 CheckBox(default=false))
"

# ╔═╡ c35f9ae2-0f88-4e34-888e-89399c0b1a89
let
	plotly()
	μ = [0, 0]
	Σ = Matrix(I, 2, 2) / 10
	Σinv = inv(Σ)
	f1(x) = (x -μ)' * Σinv * (x -μ)
	f2(x) = -0.5 * f1(x)
	f3(x) = exp(f2(x))
	f4(x) = pdf(MvNormal(μ, Σ), x)
	xlen = 0.9
	ylen = 0.9
	plt = plot(range(μ[1] -xlen, μ[1]+xlen, 50), range(μ[2] -ylen, μ[2]+ylen, 50), (x, y) -> f1([x, y]), st=:surface, alpha=0.55, label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2} ({x-\mu})\Sigma^{-1}(x-\mu)", framestyle=:zerolines, c=:coolwarm,  colorbar=false, ratio=1)
	if add_kernel2
		
	
		plt = plot(range(μ[1] -xlen, μ[1]+xlen, 50), range(μ[2] -ylen, μ[2]+ylen, 50), (x, y) -> f1([x, y]), st=:surface, c=:coolwarm, alpha=0.3,  label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2} ({x-\mu})\Sigma^{-1}(x-\mu)", framestyle=:zerolines,   colorbar=false, ratio=1)

		plot!(range(μ[1] -xlen, μ[1]+xlen, 50), range(μ[2] -ylen, μ[2]+ylen, 50), (x, y) -> f2([x, y]), st=:surface, alpha=0.55,c=:coolwarm, label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2} ({x-\mu})\Sigma^{-1}(x-\mu)",colorbar=false, ratio=1)
	end
	if add_ekernel2
		
		maxz = f3(μ)
		plt=plot(range(μ[1] -xlen, μ[1]+xlen, 50), range(μ[2] -ylen, μ[2]+ylen, 50), (x, y) -> f3([x,y]), st=:surface,  label=L"e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}", title=L"f(x)= e^{-\frac{1}{2} ({x-\mu})\Sigma^{-1}(x-\mu)}", zlim=[-3,maxz+0.5],c=:jet,  alpha=0.8, colorbar=false)

		plot!(range(μ[1] -xlen, μ[1]+xlen, 50), range(μ[2] -ylen, μ[2]+ylen, 50), (x, y) -> f2([x, y]), st=:surface, alpha=0.3, label=L"-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2", title=L"f(x)= -\frac{1}{2} ({x-\mu})\Sigma^{-1}(x-\mu)", framestyle=:zerolines, c=:jet, zlim=[-2,maxz+0.5],  colorbar=false, ratio=1)
	end
	# vline!([μ], label=L"\mu", ls=:dash, lw=2, lc=:gray, la=0.5)
	if add_px2
		maxz = f4(μ)
		plt = plot(range(μ[1] -xlen, μ[1]+xlen, 80), range(μ[2] -ylen, μ[2]+ylen, 80), (x, y) -> f4([x,y]), label=L"p(x)", alpha=0.7, title="Gaussian prob. density", st=:surface, zlim=[-2, maxz+0.5], c=:coolwarm, colorbar=false)

		plot!(range(μ[1] -xlen, μ[1]+xlen, 50), range(μ[2] -ylen, μ[2]+ylen, 50), (x, y) -> f3([x,y]), st=:surface,  alpha=0.4, colorbar=false)

		plot!(range(μ[1] -xlen, μ[1]+xlen, 50), range(μ[2] -ylen, μ[2]+ylen, 50), (x, y) -> f2([x, y]), st=:surface, alpha=0.3)
	end
	plt
end

# ╔═╡ 665bc7b6-53b3-4e76-83af-c6eda83faafa
md"""

## Aside: matrix `determinant` ``|\mathbf{\Sigma}|``

"""

# ╔═╡ c98836a1-6107-4c8e-b53d-98c502010d80
md"""


#### Square matrix ``\mathbf{\Sigma}``,  its `determinant` is written as

```math
\Large
\text{det}(\mathbf{\Sigma})=|\mathbf{\Sigma}| 
``` 


* #### it is a scalar: ``|\mathbf{\Sigma}| \in \mathbb{R}``

* #### it measures the `signed volume` of the sapce formed by the column vectors of ``\mathbf{\Sigma}``


### *Fact*: `determinant` of a _diagonal/triangular_ matrix ``\mathbf{L}``

```math
\Large
\mathbf{L} =\begin{bmatrix}\colorbox{pink}{$\ell_{11}$} & 0 & \ldots & 0 \\
\ell_{21} & \colorbox{pink}{$\ell_{22}$} & \ldots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
\ell_{n1} & \ell_{n2} & \ldots & \colorbox{pink}{$\ell_{nn}$}
\end{bmatrix};\;\;

|\mathbf{L}| = \prod_{i=1}^n \colorbox{pink}{$\ell_{ii}$}
```
"""

# ╔═╡ cf0c1e8c-3339-4a11-a445-7bc4d8bc81dc
md"""
## Effect of  $\boldsymbol{\Sigma}$ 

$\Large \boldsymbol{\Sigma}=\mathbf{I}= \begin{bmatrix} 1 & 0 \\0 & 1\end{bmatrix}$

* #### the distance kernel is Euclidean distance

$(\mathbf{x} - \boldsymbol{\mu})^\top \mathbf I^{-1}(\mathbf{x}-\boldsymbol{\mu})=(\mathbf{x} - \boldsymbol{\mu})^\top(\mathbf{x}-\boldsymbol{\mu})= (x_{1} -\mu_1)^2+ (x_2 - \mu_2)^2$

"""

# ╔═╡ 9d3bbf42-9f4a-45cc-9e96-421799329900
function mdistance(x, mvn)
	μ, Σ = mvn.μ, mvn.Σ
	return sqrt((dot(x - μ, Σ^(-1) * (x - μ))))
end 

# ╔═╡ 8a68b9cb-c814-4ba6-b701-e5a02e4bfb75
md"""

## Effect of ``\boldsymbol{\Sigma}``: diagonal 

$\Large \boldsymbol\Sigma =  \begin{bmatrix} \sigma_1^2 & 0 \\0 & \sigma_2^2\end{bmatrix} \Longrightarrow \Large \boldsymbol\Sigma^{-1} = \begin{bmatrix} 1/\sigma_1^2 & 0 \\0 & 1/\sigma_2^2\end{bmatrix}$


#### The distance measure is *axis aligned* ellipses


$\large (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})= \underbrace{\frac{(x_{1} -\mu_1)^2}{\sigma_1^2}+ \frac{(x_2 - \mu_2)^2}{\sigma_2^2}}_{\text{analytical form of an ellipse}}$

"""

# ╔═╡ db6febd1-9780-4d16-8a17-0304272d7269
begin
	Random.seed!(123)
	mvnsample = randn(2, 500)
end;

# ╔═╡ 6054fdf1-928b-4d1b-8ef8-23da27b9631c
let
	gr()
	Random.seed!(234)
	μ₁ = [2, 2]
	Σ = Matrix(1.0I, 2,2)
	mvn1 = 	MvNormal(μ₁, Σ )
	spl1 = mvnsample .+ μ₁
	x₁s = μ₁[1]-3:0.1:μ₁[1]+3
	x₂s = μ₁[2]-3:0.1:μ₁[2]+3	
	mvnplt₁ = scatter(spl1[1,:], spl1[2,:], ratio=1, label="", xlabel=L"x_1", ylabel=L"x_2", alpha=0.3, framestyle=:origin, ms=3, xlim = (-1.5, 5.5), ylim = (-2,5.5))	
	scatter!([μ₁[1]], [μ₁[2]], label=L"{\mu}=%$(μ₁)", markershape = :x, markerstrokewidth=5, markersize=6)	
	plot!(x₁s, x₂s, (x1, x2)-> mdistance([x1, x2], mvn1), levels=[1, 2, 3], linewidth=3, c=cgrad(:jet, 5; categorical=true, rev=true), st=:contour, colorbar=false)	
	μ = μ₁
	plot!([μ[1]-1, μ[1]+1], [3.8, 3.8], st=:path, lw=1.5, lc=:black, label="", arrows=:both)

	annotate!([μ[1]], [3.8], text(L"\sigma_1=1", :bottom))
	plot!([3.8, 3.8], [μ[2]-1, μ[2]+1], st=:path, lw=1.5, lc=:black, label="", arrows=:both)
	annotate!([3.8], [μ[2]], text(L"\sigma_2=1", :bottom, rotation=270))

	mvnplt₂ = surface(x₁s, x₂s, (x1, x2)->pdf(mvn1, [x1, x2]), color=:coolwarm, st=:surface, title="", colorbar=false, xlabel=L"x_1", ylabel=L"x_2", zlabel=L"p(\mathbf{x})")
	plot(mvnplt₁, mvnplt₂)
end

# ╔═╡ 2f9d23f7-8476-47e1-93f6-71dc36110b03
plt_diag1, plt_diag2=let
	gr()
	μ = [2,2]
	Σ = [1.0 0; 0 2]
	L = cholesky(Σ).L
	mvn = 	MvNormal(μ, Σ)
	spl = μ.+ L * mvnsample
	x₁s = μ[1]-3:0.1:μ[1]+3
	x₂s = μ[2]-4.5:0.1:μ[2]+4.8	
	plt1 = scatter(spl[1,:], spl[2,:], ratio=1, label="", xlabel=L"x_1", ylabel=L"x_2", alpha=0.15, framestyle=:origin, ms=2)	
	scatter!([μ[1]], [μ[2]], ratio=1, label=L"\mu=%$(μ)", markershape = :x, markerstrokewidth=5, markersize=4)	
	plot!(x₁s, x₂s, (x1, x2)->mdistance([x1, x2], mvn), levels=[1, 2, 3], linewidth=3,  c=cgrad(:jet, 5; categorical=true, rev=true), st=:contour, colorbar=false, ylim = extrema(spl[2,:]) .+ (-.5, .5))	
	plot!([μ[1]-1, μ[1]+1], [4.0, 4.0], st=:path, lw=1.5, lc=:black, label="", arrows=:both)

	annotate!([μ[1]], [4.0], text(L"\sigma_1=1", :bottom))
	plot!([3.5, 3.5], [μ[2]-2, μ[2]+2], st=:path, lw=1.5, lc=:black, label="", arrows=:both)
	annotate!([3.5], [μ[2]], text(L"\sigma_2=2", :bottom, rotation=270))

	
	Σ = [2.0 0; 0 1]
	mvn = 	MvNormal(μ, Σ)
	L = cholesky(Σ).L
	spl = μ.+ L * mvnsample
	# x₁s = range(extrema(spl[1,:]) .+ (-1.01, 1.01)... , 50)

	plt2 = scatter(spl[1,:], spl[2,:], ratio=1, label="", xlabel=L"x_1", ylabel=L"x_2", alpha=0.3, framestyle=:origin, ms=2)	

	x₁s = μ[1]-4.5:0.1:μ[1]+4.5
	x₂s = μ[2]-3:0.1:μ[2]+3	
	plot!(x₁s, x₂s, (x1, x2)->mdistance([x1, x2], mvn), levels=[1, 2, 3], linewidth=3, c=cgrad(:jet, 5; categorical=true, rev=true),  st=:contour, colorbar=false, ylim = extrema(spl[2,:]) .+ (-.5, .5))
	scatter!([μ[1]], [μ[2]], ratio=1, label=L"\mu=%$(μ)", markershape = :x, markerstrokewidth=5, markersize=4, c=2)	

	plot!([μ[1]-2, μ[1]+2], [3.8, 3.8], st=:path, lw=1.5, lc=:black, label="", arrows=:both)

	annotate!([μ[1]], [3.8], text(L"\sigma_1=2", :bottom))
	plot!([4.0, 4.0], [μ[2]-1, μ[2]+1], st=:path, lw=1.5, lc=:black, label="", arrows=:both)
	annotate!([4.0], [μ[2]], text(L"\sigma_2=1", :bottom, rotation=270))

	# scatter!([xis[i]],[ys[i]], markershape = :circle, label="", c=1, markersize=4)
	plt1, plt2
end;

# ╔═╡ b2dee8c3-c98b-4c57-95ca-a41895b5318c
TwoColumn(md"""
$\large\boldsymbol\Sigma =  \begin{bmatrix} 1 & 0 \\0 & 2\end{bmatrix}$

$(plot(plt_diag1, size=(300,300)))

""", md"""
$\large\boldsymbol\Sigma =  \begin{bmatrix} 2 & 0 \\0 & 1\end{bmatrix}$


$(plot(plt_diag2, size=(300,300)))

""")

# ╔═╡ ff04483d-4590-4ee1-817b-d492b897523b
md"""

## Diagonal $\boldsymbol \Sigma$ ``\Rightarrow`` `independence`



#### Diagonal ``\mathbf{\Sigma}`` $\Rightarrow$ *Independent* marginals


```math
\Large 
p(\mathbf{x}) = p(x_1) p(x_2)= \mathcal{N}(x_1;\mu_1, \sigma_1^2)\cdot \mathcal{N}(x_2;\mu_2, \sigma_2^2) 
```
"""

# ╔═╡ b47aef56-fbec-49cd-a33e-6dff4dd1c543
md"""

## Why _independence_ ?*

##### When $\boldsymbol\Sigma =  \begin{bmatrix} \sigma_1^2 & 0 \\0 & \sigma_2^2\end{bmatrix}$ is diagonal, then

$\large(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})= \frac{(x_{1} -\mu_1)^2}{\sigma_1^2}+ \frac{(x_2 - \mu_2)^2}{\sigma_2^2}$

#### The joint probability distribution ``p(\mathbf{x})``

$$\large\begin{align}p(\mathbf{x}) 
&= \frac{1}{(\sqrt{2\pi})^2 \left |\begin{bmatrix}\sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{bmatrix}\right |^{\frac{1}{2}}} \exp{\left \{ -\frac{1}{2} \left (\frac{(x_1-\mu_1)^2}{\sigma_1^2}+\frac{(x_2-\mu_2)^2}{\sigma_2^2}\right ) \right\}}\\
 &=\underbrace{\frac{1}{\sqrt{2\pi}\sigma_1}\exp\left [-\frac{1}{2} \frac{(x_1-\mu_1)^2}{\sigma_1^2}\right ]}_{{p(x_1)}} \cdot \underbrace{\frac{1}{\sqrt{2\pi}\sigma_2}\exp\left [-\frac{1}{2} \frac{(x_2-\mu_2)^2}{\sigma_2^2}\right ]}_{p(x_2)} \\
&= p(x_1)p(x_2) 
\end{align}$$


"""

# ╔═╡ 71683f0a-9c71-48b5-a760-e8847eee82ce
aside(tip(md"""

```math
e^{a+b} = e^a\cdot e^b
```
"""))

# ╔═╡ 7ba00267-cd92-4aba-90cb-c3357ed0770f
md"""

##

#### The idea generalises to ``d`` dimensional ``\mathbf{x} \in \mathbb{R}^d``

$$\Large\begin{align}p\left(\mathbf{x}; \mathbf{\Sigma} = \begin{bmatrix} \sigma_1^2 &  &  \\ & \ddots &  \\ & &  \sigma^2_d
\end{bmatrix}\right)  = \prod_{i=1}^d p(x_i; \mu_i, \sigma_i^2)
\end{align}$$

* Hint: the determinant of diagonal variance matrix is: ``|\mathbf{\Sigma}| =\prod_{i=1}^d \sigma_{i}^2``
"""

# ╔═╡ cc5c8774-6727-48e9-b8bf-d038343f894d
md"""

##  Example (diagonal $\boldsymbol \Sigma$)

#### $\boldsymbol\Sigma =  \begin{bmatrix} \sigma_1^2 & 0 \\0 & \sigma_2^2\end{bmatrix}$ implies $\Rightarrow$ *independent* ``p(\mathbf{x}) = p(x_1) p(x_2)``

* ##### therefore,

```math
\Large 
p({x}_2|x_1) = p(x_2)\;\;\; p({x}_1|x_2) = p(x_1) 
```
"""

# ╔═╡ ada51855-a39f-4727-8b9c-b95ff2b179c9
md"""Add ``p(x_2|X_1= x_1)``: $(@bind add_px2x1 CheckBox(default=false)), ``X_1=`` $(@bind x1_ Slider(-1.5:0.1:5.5))
"""

# ╔═╡ 53456b95-5f92-4e2a-81ca-17f2a7ef2885
md"""Add ``p(x_1|X_2= x_2)``: $(@bind add_px2x1_ CheckBox(default=false)), ``X_2=`` $(@bind x2_ Slider(-1:0.1:4))
"""

# ╔═╡ 4d1ead3c-00b0-44ac-b541-21323d9c03ea
let
	gr()
	layout = @layout [a            _
	                  b{0.8w,0.8h} c]
	μ₁ = [2,2]
	# x₁s = μ₁[1]-3:0.1:μ₁[1]+3
	# x₂s = μ₁[2]-3:0.1:μ₁[2]+3	
	gr()
	μ₂ = [2,2]
	σ₁² = 2
	σ₂² = 1
	Σ₂ = [σ₁² 0; 0 σ₂²]
	L₂ = [sqrt(σ₁²) 0; 0 sqrt(σ₂²)]
	mvn₂ = 	MvNormal(μ₂, Σ₂)
	spl₂ = μ₂.+ L₂ * mvnsample
	x₁s_ = μ₂[1]-6:0.1:μ₂[1]+6	
	x₂s_ = μ₂[2]-6:0.1:μ₂[2]+6	
	plt_joint = scatter(spl₂[1,:], spl₂[2,:], label="", xlabel=L"x_1", ylabel=L"x_2", alpha=0.3, ms=2, framestyle=:origin, ylim = (-1.5,5.5), xlim =(-2.8, 7.2))	
	scatter!([μ₂[1]], [μ₂[2]], label=L"\mu=%$(μ₂)", markershape = :x, markerstrokewidth=5, markersize=6)	
	plot!(x₁s_, x₂s_, (x1, x2)-> mdistance([x1, x2], mvn₂), levels = [1, 2, 3], linewidth=3,  c=cgrad(:jet, 5; categorical=true, rev=true)[0.2:0.1:1.5], st=:contour,  colorbar=:false)	
	marg_py = plot(Normal(μ₂[2], sqrt(σ₂²)),permute=(:y, :x),  label="", title=L"p(x_2)", lc=1, lw=2)
	marg_px = plot(Normal(μ₂[1], sqrt(σ₁²)), label="", title=L"p(x_1)", lc=2, lw=2)
	# plt2 = plot()
	if add_px2x1
		vline!(plt_joint, [x1_], lw=2, ls=:dash, label="", lc=1)
		annotate!(plt_joint, [x1_], [-1], text(L"x_1=%$(round(x1_;digits=2))", :blue))
		annotate!(plt_joint, [x1_+2], [2], text(L"p(x_2|x_1=%$(round(x1_;digits=2)))", :green, rotation = 270))
		
		# plot!(marg_py, Normal(μ₂[2], sqrt(σ₂²)), permute=(:y, :x), lw=3, lc=3,  alpha=0.2, label="")
		
		x = x1_
		μi = μ₂[2] + Σ₂[2, 1] * Σ₂[1,1]^(-1) * (x - μ₂[1])
		σ = Σ₂[2,2] - Σ₂[2, 1] * Σ₂[1,1]^(-1) * Σ₂[1,2]
		xs_ = μi- 3 * sqrt(σ) :0.05:μi+ 3 * sqrt(σ)
		ys_ = pdf.(Normal(μi, sqrt(σ)), xs_) * 4
		# plot!(plt_joint, (x) -> μ₂[2] + Σ₂[2, 1] * Σ₂[1,1]^(-1) * (x - μ₂[1]), c=2, lw=3, ls=:dash, label="")
		# plt2 = 
		plot!(plt_joint, ys_ .+x, xs_, c=2, lw=2, lc=3,label="")
		
	end


	if add_px2x1_
		hline!(plt_joint, [x2_], lw=2, ls=:dash, label="", lc=4)
		annotate!(plt_joint, [5.9], [x2_], text(L"x_2=%$(round(x2_;digits=2))", :purple, :top))
		annotate!(plt_joint, [2], [x2_+ 2], text(L"p(x_1|x_2=%$(round(x2_;digits=2)))", :purple, :top, rotation = 0))
		
		# plot!(marg_py, Normal(μ₂[2], sqrt(σ₂²)), permute=(:y, :x), lw=3, lc=3,  alpha=0.2, label="")
		
		y = x2_
		μi = μ₂[1] + Σ₂[1, 2] * Σ₂[2,2]^(-1) * (y - μ₂[2])
		σ = Σ₂[1,1] - Σ₂[1, 2] * Σ₂[2,2]^(-1) * Σ₂[2,1]
		xs_ = μi- 3 * sqrt(σ) :0.05:μi+ 3 * sqrt(σ)
		ys_ = pdf.(Normal(μi, sqrt(σ)), xs_) * 5
		# # plot!(plt_joint, (x) -> μ₂[2] + Σ₂[2, 1] * Σ₂[1,1]^(-1) * (x - μ₂[1]), c=2, lw=3, ls=:dash, label="")
		# # plt2 = 
		plot!(plt_joint, xs_,  ys_.+y,  c=2, lw=2, lc=4,label="")
		
	end


	# marg_px = bar(sum(joint_P, dims=2)[:], yticks=1:size(joint_P)[1], xlim =(0,0.6), orientation=:h ,yflip=false, label="", title=L"P(X)", ylabel=L"X", bar_width = 0.5)

	plot(marg_px, plt_joint, marg_py, layout = layout)	
end

# ╔═╡ 09f35be2-3678-401a-a12d-784790397f48
md"""

##  Effect of ``\boldsymbol{\Sigma}`` (full covariance)


#### The distance measure: (rotated) *ellipses*


$\large (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})= \underbrace{\frac{(\mathbf{v}_1^\top(\mathbf{x}-\boldsymbol\mu))^2}{\lambda_1} + \frac{(\mathbf{v}_2^\top(\mathbf{x}-\boldsymbol\mu))^2}{\lambda_2}}_{\text{still an analytical form of ellipse}}$

* ##### ``\mathbf{v}_1`` and ``\mathbf{v}_2`` are the **eigen vectors** of $\boldsymbol\Sigma$; and ``\lambda_1, \lambda_2`` are the **eigen values**
* ##### *i.e.* the rotated ellipse's basis (the $\textcolor{red}{\text{red vectors}}$ in the plot below)

"""

# ╔═╡ 1b4fe792-c0c8-4101-b2b8-8213ae573dd3
md"""

## Example (full covriance $\boldsymbol{\Sigma}$)


$\Large \boldsymbol\Sigma =  \begin{bmatrix} \sigma_1^2 & \sigma_{12} \\ \sigma_{21} & \sigma_2^2\end{bmatrix}$


#### *covariance*: $\sigma_{21}=\sigma_{12} = \mathbb{E}[(X_1 -\mu_1)(X_2 -\mu_2)]$

* ##### ``\sigma_{12} >0``: positively correlated
* ##### ``\sigma_{12} < 0``: negatively correlated


"""

# ╔═╡ 551e073e-a697-4a3a-9135-2dc6571d7e8c
@bind σ₁₂ Slider(-0.99:0.02:0.99, default=0.6)

# ╔═╡ 3d653c13-42e2-4192-b5ea-afdf850ca5e3
let
	gr()
	# mvnsample = randn(2, 500);
	μ₂ = [2,2]
	x₁s = range(μ₂[1] -3, μ₂[1]+3, 50)
	x₂s = range(μ₂[2] -3, μ₂[2]+3, 50)
	# Σ₂ = [1 0; 0 2]
	Σ₃ = [1 σ₁₂; σ₁₂ 1]
	# cholesky decomposition of Σ (only to reuse the random samples)
	L₃ = cholesky(Σ₃).L
	mvn₃ = 	MvNormal(μ₂, Σ₃)
	# μ + L * MvNormal(0, I) = MvNormal(μ, LLᵀ)
	spl₃ = μ₂.+ L₃ * mvnsample
	# x₁s_ = μ₂[1]-6:0.1:μ₂[1]+6	
	# x₂s_ = μ₂[2]-6:0.1:μ₂[2]+6	
	plt_gau3 = scatter(spl₃[1,:], spl₃[2,:], ratio=1, label="", xlabel=L"x_1", alpha=0.5, ylabel=L"x_2", framestyle=:origin)	
	scatter!([μ₂[1]], [μ₂[2]], ratio=1, label=L"\mu", markershape = :x, markerstrokewidth=5, markersize=8)	
	


	# if true
	# 	x = x1_2
	# 	# μi = dot(true_w, [1, x])
	# 	μi = μ₂[2] + Σ₃[2, 1] * Σ₃[1,1]^(-1) * (x - μ₂[1])
	# 	σ = Σ₃[2,2] - Σ₃[2, 1] * Σ₃[1,1]^(-1) * Σ₃[1,2]
	# 	xs_ = μi - 4 * sqrt(σ) :0.05:μi+ 4 * sqrt(σ)
	# 	ys_ = pdf.(Normal(μi, sqrt(σ)), xs_)
	# 	# ys_ = ys_ ./ maximum(ys_)
	# 		# scatter!([x],[μi], markerstrokewidth =1, markershape = :diamond, c=:grey, label="", markersize=3)
	# 	plot!(ys_ .+x, xs_, c=2, lw=2, label=L"p(X_2|x_1)")

	# 	plot!((x) -> μ₂[2] + Σ₃[2, 1] * Σ₃[1,1]^(-1) * (x - μ₂[1]), c=2, lw=3, ls=:dash, label="")

	# else
	λs, vs = eigen(Σ₃)
	v1 = (vs .* λs')[:,1] * 2
	v2 = (vs .* λs')[:,2] * 2
	quiver!([μ₂[1]], [μ₂[2]], quiver=([v1[1]], [v1[2]]), linewidth=4, color=:red, alpha=0.8)
	quiver!([μ₂[1]], [μ₂[2]], quiver=([v2[1]], [v2[2]]), linewidth=4, color=:red, alpha=0.8)
	plot!(x₁s, x₂s, (x1, x2)-> mdistance([x1, x2], mvn₃), levels = [1, 2, 3], linewidth=3,  c=cgrad(:jet, 5; categorical=true, rev=true)[0.2:0.1:1.5], st=:contour,  colorbar=:false)
		
	# end
	plt_gau3
end

# ╔═╡ eb074796-52d5-4cbf-8499-3f53206b3d99
md"``\sigma_{12}=\sigma_{21}``=$(σ₁₂)"

# ╔═╡ 5a539af7-0daa-44c1-8a11-4ba71f840d76
md"""Add ``p(x_2|X_1= x_1)``: $(@bind add_px2x1_2 CheckBox(default=false)), ``X_1=`` $(@bind x1_2 Slider(-2:0.1:6))
"""

# ╔═╡ dd1411b2-c0de-4755-b377-446acfcbb257
md"""Add ``p(x_1|X_2= x_2)``: $(@bind add_px2x1_2_ CheckBox(default=false)), ``X_2=`` $(@bind x2_2 Slider(-1.5:0.1:4.5))
"""

# ╔═╡ 5ab978c5-c668-4f7c-9899-16bba89549da
let
	gr()
	layout = @layout [a            _
	                  b{0.8w,0.8h} c]
	# μ₁ = [2,2]

	gr()
	μ₂ = [2,2]
	x₁s = μ₂[1]-3:0.1:μ₂[1]+3
	x₂s = μ₂[2]-3:0.1:μ₂[2]+3	
	σ₁² = 1
	σ₂² = 1
	Σ₂ = [σ₁² σ₁₂; σ₁₂ σ₂²]
	L₂ = cholesky(Σ₂).L
	mvn₂ = 	MvNormal(μ₂, Σ₂)
	spl₂ = μ₂.+ L₂ * mvnsample
	# x₁s_ = μ₂[1]-6:0.1:μ₂[1]+6	
	# x₂s_ = μ₂[2]-6:0.1:μ₂[2]+6	
	plt_joint = scatter(spl₂[1,:], spl₂[2,:], label="", xlabel=L"x_1", ylabel=L"x_2", alpha=0.1, ms=2, framestyle=:origin, ylim = (-1.5,5.5), xlim =(-2, 6))	
	scatter!([μ₂[1]], [μ₂[2]], label=L"\mu=%$(μ₂)", markershape = :x, markerstrokewidth=3, markersize=6)	
	plot!(x₁s, x₂s, (x1, x2)->pdf(mvn₂, [x1, x2]), levels=4, lw=2, alpha=0.5, c=:jet, st=:contour, colorbar=:false)	
	marg_py = plot(Normal(μ₂[2], sqrt(σ₂²)),permute=(:y, :x),  label="", title="", lc=1, lw=2)

	annotate!(marg_py,  [pdf(Normal(μ₂[2], sqrt(σ₂²)), μ₂[2])], [ μ₂[2]],text(L"p(x_2)", 12, :blue, :top, rotation=270))
	marg_px = plot(Normal(μ₂[1], sqrt(σ₁²)), label="", title="", lc=2, lw=2)

	annotate!(marg_px,  [ μ₂[1]], [pdf(Normal(μ₂[1], sqrt(σ₁²)), μ₂[1])] ,text(L"p(x_1)", 12, :red, :top) )
	# plt2 = plot()
	if add_px2x1_2
		x1_ = x1_2
		vline!(plt_joint, [x1_], lw=2, ls=:dash, label="", lc=3)
		annotate!(plt_joint, [x1_], [-1], text(L"x_1=%$(round(x1_;digits=2))", :green))
		
		x = x1_
		μi = μ₂[2] + Σ₂[2, 1] * Σ₂[1,1]^(-1) * (x - μ₂[1])
		σ = Σ₂[2,2] - Σ₂[2, 1] * Σ₂[1,1]^(-1) * Σ₂[1,2]
		xs_ = μi- 3 * sqrt(σ) :0.05:μi+ 3 * sqrt(σ)
		ys_ = pdf.(Normal(μi, sqrt(σ)), xs_) * 3
		plot!(plt_joint, (x) -> μ₂[2] + Σ₂[2, 1] * Σ₂[1,1]^(-1) * (x - μ₂[1]), c=3, lw=2, ls=:dash, label="")
		# plt2 = 
		plot!(plt_joint, ys_ .+x, xs_, c=2, lw=2, lc=3,label="")
		plot!(marg_py, Normal(μi, σ), c=2, lw=2,permute=(:y, :x), lc=3,label="")
		annotate!(plt_joint, [pdf(Normal(μi, σ), μi)+x1_], [μi], text(L"p(x_2|x_1=%$(round(x1_;digits=2)))", :green, rotation = 270, :bottom))

		scatter!(plt_joint, [x1_], [μi], label="", markershape = :x, markerstrokewidth=3, markersize=6, c=3)	
		annotate!(marg_py,  [pdf(Normal(μi, σ), μi)], [μi],text(L"p(x_2|x_1=%$(round(x1_;digits=2)))", 12, :green, :top, rotation=270))
	end


	if add_px2x1_2_
		x2_ = x2_2
		hline!(plt_joint, [x2_], lw=2, ls=:dash, label="", lc=4)
		annotate!(plt_joint, [5.9], [x2_], text(L"x_2=%$(round(x2_;digits=2))", :purple, :top))
		annotate!(plt_joint, [2], [x2_+ 2], text(L"p(x_1|x_2=%$(round(x2_;digits=2)))", :purple, :top, rotation = 0))		
		y = x2_
		μi = μ₂[1] + Σ₂[1, 2] * Σ₂[2,2]^(-1) * (y - μ₂[2])
		σ = Σ₂[1,1] - Σ₂[1, 2] * Σ₂[2,2]^(-1) * Σ₂[2,1]
		xs_ = μi- 3 * sqrt(σ) :0.05:μi+ 3 * sqrt(σ)
		ys_ = pdf.(Normal(μi, sqrt(σ)), xs_) * 5
		# # plot!(plt_joint, (x) -> μ₂[2] + Σ₂[2, 1] * Σ₂[1,1]^(-1) * (x - μ₂[1]), c=2, lw=3, ls=:dash, label="")
		# # plt2 = 
		plot!(plt_joint, xs_,  ys_.+y,  c=2, lw=2, lc=4,label="")
		plot!(marg_px, Normal(μi, σ),  lw=2,lc=4,label="")
		scatter!(plt_joint, [μi], [x2_], label="", markershape = :x, markerstrokewidth=3, markersize=6, c=:purple)	
		annotate!(marg_px,  [μi], [pdf(Normal(μi, σ),μi)] ,text(L"p(x_1|x_2=%$(round(x2_;digits=2)))", 12, :purple, :top) )
	end


	# marg_px = bar(sum(joint_P, dims=2)[:], yticks=1:size(joint_P)[1], xlim =(0,0.6), orientation=:h ,yflip=false, label="", title=L"P(X)", ylabel=L"X", bar_width = 0.5)

	plot(marg_px, plt_joint, marg_py, layout = layout, size=(650,650))	
end

# ╔═╡ c8528433-4b95-44f3-ae66-cf11ea4f3b99
md"""

## MLE estimation of Multivariate Gaussian



#### The MLE ``\boldsymbol\mu, \boldsymbol\Sigma`` given $$\mathcal{D} =\{\mathbf{x}^{(i)}\}_{i=1}^n$$ are


$$\large\boxed{\begin{align}\hat{\boldsymbol{\mu}} &= \frac{1}{n}{\sum_{i=1}^n \mathbf x^{(i)}}\\
\hat{\boldsymbol{\Sigma}} &= \frac{1}{n} \sum_{i=1}^n (\mathbf x^{(i)}-\hat{\boldsymbol{\mu}})(\mathbf x^{(i)}-\hat{\boldsymbol{\mu}})^\top
\end{align}}$$


* ##### again very straighforward: sample mean and sample variance
"""

# ╔═╡ 2dba181a-05be-4e1a-b004-858573eda215
md"""


##### Cross-reference the MLE result for univariate Gaussian 


$$\large \hat{{\mu}} = \frac{1}{n}{\sum_{i=1}^n  x^{(i)}}$$

$$\large \hat{{\sigma}}^2 = \frac{1}{n} \sum_{i=1}^n ( x^{(i)}-\hat{{\mu}})( x^{(i)}-\hat{{\mu}})$$


"""

# ╔═╡ 2496221f-70f9-4187-b329-35fbaf03a480
md"""

## Summary of multi-variate Gaussians


##### -- ``\mathbf\Sigma`` leads to different distance measures
"""

# ╔═╡ c05d44bd-1564-430a-b633-b57cca1f5526
Σs = [Matrix(1.0I, 2,2), [2 0; 0 0.5],  [.5 0; 0 2] , [1 0.9; 0.9 1], [1 -0.9; -0.9 1]];

# ╔═╡ fd0a8334-07e6-4678-bf28-d334d81fc67e
plts_mvns=let
	Random.seed!(123)
	nobs= 250
	plts = []

	for Σ in Σs
		mvn = MvNormal(zeros(2), Σ)
		data = rand(mvn, nobs)
	  	# scatter(data[1,:], data[2,:])
		plt = plot(-3:0.1:3, -3:0.1:3, (x, y) -> pdf(mvn, [x,y]), st=:contour, c=:jet, clabels=false, ratio=1, lw=2, levels=5, colorbar=false, framestyle=:origin)

		# plot!(-3:0.1:3, -3:0.1:3, (x, y) -> pdf(mvn, [x,y]), st=:contour, c=:jet, ratio=1, lw=3, levels=5, colorbar=false, framestyle=:origin)
		# scatter!(data[1,:], data[2,:], c=1, alpha=0.5, ms=2, label="")
		push!(plts, plt)
	end

	# color=:turbo, clabels=true,
	plts
end;

# ╔═╡ 9ecd708a-0585-4a68-b222-702e8de02abb
ThreeColumn(
md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[1]))

$(plot(plts_mvns[1], size=(220,220)))
	


"""	,
md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[2]))

$(plot(plts_mvns[2], size=(220,220)))
	



"""
	,


md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[3]))


$(plot(plts_mvns[3], size=(220,220)))
	


"""
)

# ╔═╡ 133e4cb5-9faf-49b8-96ef-c9fc5c8e0b94
TwoColumn(

md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[4]))



$(plot(plts_mvns[4], size=(320,220)))
	

"""
	,


md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[5]))

$(plot(plts_mvns[5], size=(320,220)))
	



"""
)

# ╔═╡ bbfba1d5-c280-43e1-9721-d0ab2b9226ed
begin
	gr()
	plt2 = @df df scatter(:bill_length_mm, :bill_depth_mm, group =:species, framestyle=:origins,  xlabel="bill length (mm)", ylabel="bill depth (g)");


end;

# ╔═╡ cc9d8ae5-9ed3-407c-b63c-e8c4ea1cd472
# md"""

# ## Demonstration
# """

# ╔═╡ 9a0ea270-10d8-44f5-98a1-f6324572548e
# TwoColumn(
# 	md"""

# Assume a ``C=3``, *i.e.* three class problems, 

# **The prior** (uniform prior)

# $p(y) = \mathcal{Cat}(\boldsymbol\pi) = \begin{cases}\colorbox{lightblue}{1/3} & y=1\\ \colorbox{lightsalmon}{1/3} & y=2\\ \colorbox{lightgreen}{1/3} &y=3 \end{cases}$

# **The likelihood**: ``p(\mathbf{x}|y)``, *i.e.* Gaussian components:

# ```math
# \begin{align}
# &\colorbox{lightblue}{$p(x|y=1) = \mathcal{N}_1(-3, 1)$}\; \\
# &\colorbox{lightsalmon}{$p(x|y=2) = \mathcal{N}_2(0, 1)$}\; \\
# &\colorbox{lightgreen}{$p(x|y=3) = \mathcal{N}_3(3, 1)$}
# \end{align}
# ```
# """
	
# 	,let
# 	go
# 	if !isempty(zs_samples)
# 		i = length(zs_samples)
# 		zi = zs_samples[end]
# 		qda_plot =	plot((x) -> pdf(mvns[zi], x), fill=true, alpha=0.5, lc = zi , c=zi, lw= 2, label="sample "*L"y^{(%$i)}\sim \mathcal{Cat}(\pi);\;"*L" y^{(%$i)}=%$(zi)",  xlim =[-6, 6], framestyle=:zerolines, yaxis=false, title="", size=(300,350))
# 		for k in 1:3
# 			plot!((x) -> pdf(mvns[k], x), lc=k, lw= 1, fill=false, ls=:dash, label="")
# 		end

# 		scatter!([ys_samples[end]], [0], markershape=:x, markerstrokewidth=5, c= zi, ms=8, label="sample "*L"x^{(%$i)}\sim \mathcal{N}_{%$(zi)}", legendfontsize=12, legend=:outertop)
			
# 		for c_ in 1:3
# 			zs_ = zs_samples .== c_
# 			nc = sum(zs_)
# 			if nc > 0
# 				scatter!(ys_samples[zs_], zeros(nc), markershape=:diamond, ms=6, c= c_, alpha=0.5, label="")
# 			end
# 		end
# 	else
# 		qda_plot =	plot((x) -> pdf(mvns[1], x), lc=1, lw= .5, fill=true, c=1, alpha=0.5, ls=:dash, xlim =[-6, 6], label="", framestyle=:zerolines, yaxis=false, title="Probabilistic generative model (3 classes)", titlefontsize=8, size=(300,350))
# 		plot!((x) -> pdf(mvns[2], x),lc=2, lw=.5,fill=true, c=2, alpha=0.5,  ls=:dash, label="")
# 		plot!((x) -> pdf(mvns[3], x), lc=3, lw=.5, fill=true, c=3, alpha=0.5, ls=:dash, label="")
# 	end
# 	qda_plot
# end)

# ╔═╡ 17c07bab-5d5a-4480-8e60-94ffc4d891ef
# @bind init_reset Button("restart")

# ╔═╡ 9e21c264-1175-479f-a0bd-51b21c67ce36
# md"""
# start the simulation: $(@bind start CheckBox(default=false)),
# $(@bind go Button("step"))

# """

# ╔═╡ e0473be1-1ee0-42fe-95a1-cdd6c948fb35
# begin
# 	go
# 	if start 
# 		iters[1] += 1
# 		zi = sample(Weights(trueπs))
# 		push!(zs_samples, zi)
# 		di = rand(mvns[zi])
# 		push!(ys_samples, di)
# 		zi, di
# 	end
# end;

# ╔═╡ 0ca6a1e1-6f91-42fa-84b5-7c3b9170e56a
# begin
# 	gr()
# 	init_reset
# 	trueμs = [-3, 0, 3.]
# 	trueσs = [1 , 1, 1]
# 	trueπs = 1/3 * ones(3)
# 	mvns = [Normal(trueμs[k], trueσs[k]) for k in 1:3]
# 	zs_samples = []
# 	# qda_plot = Plots.plot(lc=:gray, lw= .5, ls=:dash, xlim =[-6, 6], ylim =[0, 1.5], label="", framestyle=:zerolines, yaxis=false, title="")
# 	iters = [-1]
# 	ys_samples = []
# end;

# ╔═╡ 77aaff69-13bb-4ffd-ad63-62993e13f873
# md"""

# ## An one-dimensional example 
# ##### (generative model)


# \


# ----
# ```math
# \begin{align}
# y \;&\;\sim\; \mathcal{Cat}(\boldsymbol\pi) \tag{generate $y$}\\
# x|y=c \; &\;\sim\; \mathcal{N}_c(\mu_c, \sigma^2_c) \tag{generate $x$ conditional on $y$}
# \end{align}
# ```
# ----

# * ``c=1\ldots 3``, *i.e.* three-class classification

# """

# ╔═╡ e5a23ba6-7859-4212-8854-86b238332eef
# let
# 	# data_d1 = data₁[:, 1]
# 	nn = 30
# 	trueμs = [-3, 0, 3.]
# 	trueσs = [1 , 1, 1]
# 	# trueπs = [0.15, 0.7, 0.15]
# 	trueπs = 1/3 * ones(3)
# 	mvns = [Normal(trueμs[k], trueσs[k]) for k in 1:3]
# 	truezs = zeros(Int, nn)
# 	data = zeros(nn)
# 	Random.seed!(123)
# 	anim_gen = @animate for i in 1:nn
# 		truezs[i] = zi = sample(Weights(trueπs))
# 		plot((x) -> pdf(mvns[1], x), lc=:gray, lw= .5, ls=:dash, xlim =[-6, 6], label="", framestyle=:zerolines, yaxis=false, title="")
# 			# title="Sample "*L"y^{(%$i)}\sim \mathcal{Cat}(%$(round.(trueπs, digits=2)))\;" * "; "*L"y^{(%$(i))} = %$(zi)")
# 		plot!((x) -> pdf(mvns[2], x),lc=:gray, lw=.5, ls=:dash, label="")
# 		plot!((x) -> pdf(mvns[3], x), lc=:gray, lw=.5, ls=:dash, label="")
# 		# ci = zi == 1 ? :gray : 1 
# 		plot!((x) -> pdf(mvns[zi], x), fill=true, alpha=0.5, lc = zi , c=zi, lw= 2, label="")

# 		data[i] = di = rand(mvns[zi])
# 		scatter!([di], [0], markershape=:x, markerstrokewidth=5, c= zi, ms=8, label="sample "*L"y^{(%$i)}\sim \mathcal{Cat}(\pi)\;" * "; "*"sample "*L"x^{(%$i)}\sim \mathcal{N}_{%$(zi)}", legendfontsize=12, legend=:outertop)
		
# 		for c_ in 1:3
# 			zs_ = truezs[1:i-1] .== c_
# 			nc = sum(zs_)
# 			if nc > 0
# 				scatter!(data[1:i-1][zs_], zeros(nc), markershape=:diamond, ms=6, c= c_, alpha=0.5, label="")
# 			end
# 		end
# 	end

# 	gif(anim_gen, fps=1)
# end

# ╔═╡ 85eeec0d-f69f-4cf9-a718-df255a948433
WW = let
	nn = 100
	trueμs = [-3, 0, 3.]
	trueσs = [1 , 1, 1]
	# trueπs = 1/3 * ones(3)
	mvns = [Normal(trueμs[k], trueσs[k]) for k in 1:3]
	labels = Int[]
	data = Float64[]
	for k in 1:3
		append!(data, rand(mvns[k], nn))
		append!(labels, k * ones(Int, nn))
	end
	λ = 0.0
	mlr = MultinomialRegression(λ) # you can also just use LogisticRegression
	theta = MLJLinearModels.fit(mlr, Matrix(reshape(data, :, 1)), labels)
	p = 2
	c = 3
	W = reshape(theta, p, c)
end;

# ╔═╡ 46d30b0c-0cf5-4369-a81f-6c234043a8ea
# md"""

# ## Why generative model ?

# !!! note "Discriminative model"
# 	```math
# 	\large p(y|\mathbf{x})
# 	```

# !!! note "Generative model"
# 	```math
# 	\large p(y, \mathbf{x}) = p(y) p(\mathbf{x}|y)
# 	```


# * note that ``\large p(\mathbf{x}|y)`` is unique to the generative model

# ##### Applications of the generative model ``\large p(\mathbf{x}, y)``:

# 1. outlier detection; 

# ```math
# \mathbf{x}\text{ is an outlier if: }\;P(\mathbf{x}|y) < \epsilon; \; \text{where }\epsilon \text{ is a small constant}
# ```
# 2. deal with missing data 

# ```math
# \mathbf{x}=[\texttt{missing}, x_2]^\top;\;p(\mathbf{x}|y) = \sum_{x_1} p(X_1= x_1, x_2|y)
# ```
# 3. simulate pseudo/fake data
# ```math
# \mathbf{x}\sim p(\mathbf{x}|y=c)\;\; \text{or}\;\; \mathbf{x}\sim p(\mathbf{x})=\sum_{c=1}^C p(y=c)p(\mathbf{x}|y=c)
# ```
# """

# ╔═╡ 870ff42f-d903-4ccf-a538-43bbf9ec978b
md"""

# Classification with Bayes' rule
"""

# ╔═╡ 67ba35e4-9250-4ac5-b1f4-94bbdab42258
md"""

## Classification with generative models



##### Classify ``y^{(i)}`` by computing the _posterior_


```math
\Large
p(y^{(i)}|\mathbf{x}^{(i)})=\begin{cases} ? & y^{(i)} = \texttt{Adelie} \\
? & y^{(i)} = \texttt{Chinstrap}
\\
? & y^{(i)} = \texttt{Gento}
\end{cases}
```
"""

# ╔═╡ c75c6c55-ef2d-410d-9ff2-6647b228dc29
md"""Classify $(@bind add_decision_boundary CheckBox(false)), ``\mathbf{x}^{(i)}`` $(@bind text_idx Select(([1:2;324:325;147:148;])))"""

# ╔═╡ 76c05a93-698c-4ee1-9411-8a518c4e56b0
# md"""

# ## Recap: Bayes' rule
# """

# ╔═╡ 21803313-fa45-48ea-bd3f-a30bc7696215
# TwoColumn(md""" 
# ```math
# \begin{align}
# &\text{Hypothesis: }h \in \{\texttt{healthy}, \texttt{cold}, \texttt{cancer}\}\\
# &\text{Observation: }\mathcal{D} = \{\texttt{cough} = \texttt{true}\}
# \end{align}
# ```

# Apply Bayes' rule

# $$\begin{align}P(h|\texttt{cough}) &\propto P(h) P(\texttt{cough}|h) \\
# &= \begin{cases} 0.321 & h=\texttt{healthy} \\
# 0.614 & h=\texttt{cold}\\ 
# 0.065 & h=\texttt{cancer}\end{cases}
# \end{align}$$

# $\hat{h}_{\rm MAP} =\arg\max_h p(h|\texttt{cough})=\texttt{cold}$

# """, html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/cough_bn.svg" height = "210"/></center>
# """)

# ╔═╡ 2acf2c33-bd3b-4369-857a-714d0d1fc600
# md"""

# ## Bayes' rule


# #### Bayes' rule allows to compute the  probability reversely from

# ```math
# \Large
# p(H) p(\mathcal{D}|H) \Rightarrow p(H|\mathcal{D})
# ```
# """

# ╔═╡ 6775b1df-f9dd-423e-a6ef-5c9b345e5f0f
# html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/bayes.png' width = '400' /></center>"

# ╔═╡ abd46485-03c9-4077-8ace-34f9b897bb04
md"""

## Classification rule (Bayes' rule)

#### To classify ``y`` given ``\mathbf{x}``, apply *Bayes' rule*


```math
\Large
p({y}|\mathbf{x}) = \frac{p(y) p(\mathbf{x}|y)}{p(\mathbf{x})}
```


* ##### ``p(y=c) = \pi_c``, the prior 

* ##### ``p(\mathbf{x}|y=c) = \mathcal{N}_c(\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)``, Gaussian likelihood


## Classification rule (Bayes' rule)

#### To classify ``y`` given ``\mathbf{x}``, apply *Bayes' rule*


```math
\Large
p({y}|\mathbf{x}) = \frac{p(y) p(\mathbf{x}|y)}{p(\mathbf{x})}
```


* ##### ``p(y=c) = \pi_c``, the prior 

* ##### ``p(\mathbf{x}|y=c) = \mathcal{N}_c(\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)``, Gaussian likelihood

#### The _normalising constant_ for the posterior is

```math
\Large
p(\mathbf{x}) = \sum_{c=1}^C p(y=c) p(\mathbf{x}|y=c)
```


## Classification rule -- summary


#### Therefore, for GDA

```math
\Large
p({y}=c|\mathbf{x}) = \frac{\pi_c\cdot \mathcal{N}(x; \mu_c, \Sigma_c)}{\sum_c \pi_c\cdot \mathcal{N}(x; \mu_c, \Sigma_c)}
```
"""

# ╔═╡ dd915c43-b3cc-4608-87b0-852b2d533b15
md"""


## Demonstration

```math
\Large
p({y}=c|\mathbf{x}) = \frac{\pi_c\cdot \mathcal{N}(x; \mu_c, \Sigma_c)}{\sum_c \pi_c\cdot \mathcal{N}(x; \mu_c, \Sigma_c)}
```
* ##### note that ``p(y|x)`` always sum to one; 

"""

# ╔═╡ 0bab5178-5cbd-467e-959d-1f09b496d2af
md"Select ``x``: $(@bind x00 Slider(-4.5:0.1:4.5, default=0, show_value=true))"

# ╔═╡ b10c879d-98c8-4617-91de-fba1d9499ba4
let
	# data_d1 = data₁[:, 1]
	nn = 30
	trueμs = [-1.5, 0, 1.5]
	trueσs = [1 , 1, 1]
	# trueπs = [0.15, 0.7, 0.15]
	trueπs = 1/3 * ones(3)
	mvns = [Normal(trueμs[k], trueσs[k]) for k in 1:3]
	truezs = zeros(Int, nn)
	data = zeros(nn)
	Random.seed!(123)

	zs = rand(Categorical(trueπs), nn)
	# x0 =-3
	py = trueπs .* pdf.(mvns, x00)
	y0 = argmax(py)
	plt = plot(xlim =[-6, 6], label="", framestyle=:origins, yaxis=true, ylabel=L"p(y,x)", title="", legend=:outerbottom)
	for c in 1:3
		alpha = 0.4
		lw = 2
		label = L"{π_{%$c} \mathcal{N}(x; \mu_{%$c}, \sigma^2_{%$c})= %$(round(py[c];digits=3))}"
		plot!((x) -> trueπs[c] * pdf(mvns[c], x), lc=c, fill=true, alpha=alpha, lw=0.5, c=c, ls=:dash, label=label, legendfontsize = 12)
		# if showall
		plot!([x00], [py[c]], label="", c=:gray, lw=lw, alpha=1, lc=c, ls=:dash, arrow=(:close, :end, 1,1), markersize=1, st=:sticks)
		# else
			# plot!([x0], [py[class]], label="", c=:gray, lw=lw, alpha=1, lc=class, ls=:solid, arrow=(:close, :end, 1,1), markersize=1, st=:sticks)
		# end
	end
	py_ = py/sum(py)
	
	scatter!([x00], [0], label="", c=:gray, markershape=:x, markerstrokewidth=3, markersize=6,  title=L"p(y|x) = %$(round.(py_;digits=2))")


	# end

	plt
	
end

# ╔═╡ 57c4e40c-6760-4c88-896d-7ea8faf134e0
md"""

## Classification shortcut (MAP estimator)

#### To classify ``y``, if we just want to find the *most likely class*

> ```math
> \Large
> \begin{align}
> \hat{y}_{MAP} 
> &= \arg\max_c\;  \frac{\pi_c\cdot \mathcal{N}(x; \mu_c, \Sigma_c)}{\boxed{\sum_{c'} \pi_{c'}\cdot \mathcal{N}(x; \mu_{c'}, \Sigma_{c'})}}
> \end{align}
> ```

* #### this is known as maximum aposteriori estimator



* #### to find the MAP, we do not need to compute the denominator



> ```math
> \Large
> \begin{align}
> \hat{y}_{MAP} 
> &= \arg\max_c\; \pi_c \cdot \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_c,  \boldsymbol{\Sigma}_c)
> \end{align}
> ```


"""

# ╔═╡ 1a6cb45d-1b26-4efa-bd40-f7a8e3bbd770
md"""
## Demonstration
"""

# ╔═╡ c85b688c-fc8d-4dfa-98bd-9e43dd0b79d5
md"Select ``x``: $(@bind x0 Slider(-5.5:0.1:5.5, default=0, show_value=true))"

# ╔═╡ ee10a243-052f-4c0f-8f0d-e16ad6ceb611
md"Select ``y``: $(@bind class Select(1:3)),
Show all ``y``: $(@bind showall CheckBox(default=false)) ,
Add descision boundary $(@bind add_db CheckBox(default=false))
"

# ╔═╡ df0719cb-fc54-456d-aac7-48237a96cbdd
let
	# data_d1 = data₁[:, 1]
	nn = 30
	trueμs = [-1.5, 0, 1.5]
	trueσs = [1 , 1, 1]
	# trueπs = [0.15, 0.7, 0.15]
	trueπs = 1/3 * ones(3)
	mvns = [Normal(trueμs[k], trueσs[k]) for k in 1:3]
	truezs = zeros(Int, nn)
	data = zeros(nn)
	Random.seed!(123)

	zs = rand(Categorical(trueπs), nn)
	x1_ = (trueμs[1:2]) |> mean
	x2_ = (trueμs[2:end]) |> mean
	# x0 =-3
	py = trueπs .* pdf.(mvns, x0)
	y0 = argmax(py)
	plt = plot(xlim =[-6, 6], label="", framestyle=:origins, yaxis=true, ylabel=L"p(y,x)", title="", legend=:outerbottom)
			# title="Sample "*L"y^{(%$i)}\sim \mathcal{Cat}(%$(round.(trueπs, digits=2)))\;" * "; "*L"y^{(%$(i))} = %$(zi)")
	for c in 1:3
		alpha = c == class ? 0.5 : 0.05
		lw=2
		if showall
			alpha = 0.25
			lw = 2
		end
		label = L"{π_{%$c} \mathcal{N}(x; \mu_{%$c}, \sigma^2_{%$c})= %$(round(py[c];digits=3))}"
		plot!((x) -> trueπs[1] * pdf(mvns[c], x), lc=c, fill=true, alpha=alpha, lw=0.5, c=c, ls=:dash, label=label, legendfontsize = 12)
		if showall
			plot!([x0], [py[c]], label="", c=:gray, lw=c, alpha=1, lc=c, ls=:dash, arrow=(:close, :end, 1,1), markersize=1, st=:sticks)
		else
			plot!([x0], [py[class]], label="", c=:gray, lw=lw, alpha=1, lc=class, ls=:solid, arrow=(:close, :end, 1,1), markersize=1, st=:sticks)
		end
	end
	
	scatter!([x0], [0], label="", c=:gray, markershape=:x, markerstrokewidth=3, markersize=6,  title=L"\hat{y} = \arg\max_c\; \{\pi \cdot \mathcal{N}(x)\}=%$(y0)")


	if add_db
		plot!([x1_], [trueπs[1] * pdf(mvns[1], x1_)], label="", lc=1, lw=2.5, alpha=1,  ls=:dash, markersize=1, st=:sticks)
		plot!([x2_], [trueπs[2] * pdf(mvns[2], x2_)], label="", lc=2, lw=2.5, alpha=1, ls=:dash, markersize=1, st=:sticks)
	end

	# end

	plt
	
	# plot!((x) -> pdf(mvns[3], x), lc=:gray, lw=.5, ls=:dash, label="")
	# anim_gen = @animate for i in 1:nn
	# 	truezs[i] = zi = sample(Weights(trueπs))
	# 	plot((x) -> pdf(mvns[1], x), lc=:gray, lw= .5, ls=:dash, xlim =[-6, 6], label="", framestyle=:zerolines, yaxis=false, title="")
	# 		# title="Sample "*L"y^{(%$i)}\sim \mathcal{Cat}(%$(round.(trueπs, digits=2)))\;" * "; "*L"y^{(%$(i))} = %$(zi)")
	# 	plot!((x) -> pdf(mvns[2], x),lc=:gray, lw=.5, ls=:dash, label="")
	# 	plot!((x) -> pdf(mvns[3], x), lc=:gray, lw=.5, ls=:dash, label="")
	# 	# ci = zi == 1 ? :gray : 1 
	# 	plot!((x) -> pdf(mvns[zi], x), fill=true, alpha=0.5, lc = zi , c=zi, lw= 2, label="")

	# 	data[i] = di = rand(mvns[zi])
	# 	scatter!([di], [0], markershape=:x, markerstrokewidth=5, c= zi, ms=8, label="sample "*L"y^{(%$i)}\sim \mathcal{Cat}(\pi)\;" * "; "*"sample "*L"x^{(%$i)}\sim \mathcal{N}_{%$(zi)}", legendfontsize=12, legend=:outertop)
		
	# 	for c_ in 1:3
	# 		zs_ = truezs[1:i-1] .== c_
	# 		nc = sum(zs_)
	# 		if nc > 0
	# 			scatter!(data[1:i-1][zs_], zeros(nc), markershape=:diamond, ms=6, c= c_, alpha=0.5, label="")
	# 		end
	# 	end
	# end

	# gif(anim_gen, fps=1)
end

# ╔═╡ 2ad600a2-4e5d-4af6-a18c-caaa516a542d
begin
	trueπss_ = [[1,1,1],[1, 4, 1], [4, 1, 1], [1, 1, 4]]
	trueπss_ = [trueπss_[k]/sum(pi) for (k, pi) in enumerate(trueπss_)]
end;

# ╔═╡ b4d619a1-8741-4902-86f8-cd8e84c9d785
md"""

## Effect of prior ``p(y) = \boldsymbol{\pi}``
"""

# ╔═╡ e243fe55-ee3e-47dc-9a7a-4319e0e86f8e
@bind trueπs_ Select(trueπss_)

# ╔═╡ c63369ed-58ed-4dd3-9292-c6c265ad52ba
md"Select ``x``: $(@bind x0_ Slider(-5.5:0.1:5.5, default=0, show_value=true))"

# ╔═╡ 30cf6d78-c541-4d2d-b455-cb365d52b5cd
let
	# data_d1 = data₁[:, 1]
	nn = 30
	trueμs = [-1.5, 0, 1.5]
	trueσs = [1 , 1, 1]
	# trueπs = [0.15, 0.7, 0.15]
	trueπs = trueπs_
	mvns = [Normal(trueμs[k], trueσs[k]) for k in 1:3]
	truezs = zeros(Int, nn)
	data = zeros(nn)
	Random.seed!(123)

	zs = rand(Categorical(trueπs), nn)
	x0 = x0_
	py = trueπs .* pdf.(mvns, x0)
	y0 = argmax(py)
	plt = plot(xlim =[-6, 6], label="", framestyle=:origins, yaxis=true, ylabel=L"p(y,x)", title="", legend=:outerbottom)
			# title="Sample "*L"y^{(%$i)}\sim \mathcal{Cat}(%$(round.(trueπs, digits=2)))\;" * "; "*L"y^{(%$(i))} = %$(zi)")
	for c in 1:3
		alpha =  c == y0 ? 0.4 : 0.2 
		lw=2
		label = L"{π_{%$c} \mathcal{N}(x; \mu_{%$c}, \sigma^2_{%$c})= %$(round(py[c];digits=3))}"
		plot!((x) -> trueπs[c] * pdf(mvns[c], x), lc=c, fill=true, alpha=alpha, lw=0.5, c=c, ls=:dash, label=label, legendfontsize = 12)
		# if showall
		plot!([x0], [py[c]], label="", c=:gray, lw=c, alpha=1, lc=c, ls=:dash, arrow=(:close, :end, 1,1), markersize=1, st=:sticks)
		# else
		# 	plot!([x0], [py[class]], label="", c=:gray, lw=lw, alpha=1, lc=class, ls=:solid, arrow=(:close, :end, 1,1), markersize=1, st=:sticks)
		# end
	end
	
	scatter!([x0], [0], label="", c=:gray, markershape=:x, markerstrokewidth=3, markersize=6,  title=L"\hat{y} = \arg\max_c\; \pi \cdot \mathcal{N}(x)=%$(y0)")

	j, k = 1, 2
	x1_ = 0.5 * (trueμs[j] + trueμs[k]) - (log(trueπs[j]) -log(trueπs[k]))/(trueμs[j] - trueμs[k])
	j, k = 2, 3
	x2_ = 0.5 * (trueμs[j] + trueμs[k]) - (log(trueπs[j]) -log(trueπs[k]))/(trueμs[j] - trueμs[k])
	# if add_db
	plot!([x1_], [trueπs[1] * pdf(mvns[1], x1_)], label="", lc=1, lw=3, alpha=1, ls=:dash, markersize=1, st=:sticks)
	plot!([x2_], [trueπs[2] * pdf(mvns[2], x2_)], label="", lc=2, lw=3, alpha=1, ls=:dash, markersize=1, st=:sticks)
	# end

	plt
	# x1_
	
end

# ╔═╡ e33b07c9-fb73-405d-8ee0-6e6e88e32bab
md"""

## Classification in practice (use ``\ln``)


#### Recall ``\ln``-transform (a monotonic function) does not change ``\arg\max`` result

> ```math
> \Large
> \begin{align}
> \hat{y}_{MAP} 
> &= \arg\max_c\; \pi_c \cdot \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_c,  \boldsymbol{\Sigma}_c)
> \end{align}
> ```




> ```math
> \Large
> \begin{align}
> \hat{y}_{MAP} 
> &= \arg\max_c\; \left\{\ln \pi_c + \ln \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_c,  \boldsymbol{\Sigma}_c)\right \}
> \end{align}
> ```

* ##### ``\ln`` does not change ``\arg\max`` result but simplifies computation



"""

# ╔═╡ 809cf0f6-dece-4294-9624-80a6dc19ee46
md"""
## Classification in practice (use ``\ln``)

#### Sub-in Gaussian's log-likelihood:


```math
\large
\begin{align}
\hat{y} 
&= \arg\max_c\, \left \{\ln \pi_c + \ln  \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c) \right \}\\
&= \arg\max_c\, \left \{\ln \pi_c - \frac{1}{2}\ln |\boldsymbol\Sigma_c|-\frac{1}{2}\colorbox{pink}{$(\mathbf{x} -\boldsymbol{\mu}_c)^\top\boldsymbol{\Sigma}_c^{-1}(\mathbf{x} -\boldsymbol{\mu}_c)$} \underbrace{\cancel{- \frac{d}{2}\ln 2\pi}}_{\rm constant} \right \}
\end{align}
```


* ##### it shows that GDAs are distance/metric based classifier 
"""

# ╔═╡ eb14fb68-6950-4374-adef-35b583bf99fb
md"""

## Classification example


### Assume $$\large\boldsymbol{\Sigma}_c = \mathbf{I}$$ for all class



```math
\Large
\begin{align}
\hat{y} 
&= \arg\max_c\, \left \{\ln \pi_c -\frac{1}{2}\underbrace{\colorbox{lightgreen}{$(\mathbf{x} -\boldsymbol{\mu}_c)^\top(\mathbf{x} -\boldsymbol{\mu}_c)$} }_{\text{Eucliean distance } \mathbf{x} \text{ and } \boldsymbol{\mu}_c}\right \}
\end{align}
```

* ##### note that ``|\mathbf{I}| =1``, therefore ``\ln |\mathbf{I}| =0``
"""

# ╔═╡ e28dda3f-fbcd-47f0-8f99-4f3a2087905d
md"Add boundary: $(@bind add_db_1 CheckBox(default=false))"

# ╔═╡ c363c7d2-5749-49cb-8643-57a1b9dda8eb
md"""

## Linear discriminant analysis


> #### Linear discriminant analysis (LDA) assumes for all ``c``, 
> 
> $\Large \boldsymbol{\Sigma}_c = \boldsymbol{\Sigma}\;\; \text{for all }c=1\ldots C$

#### The classification rule reduces to


```math
\large
\begin{align}
\hat{y} 
&= \arg\max_c\, \left \{\ln \pi_c - \underbrace{\cancel{\frac{1}{2}\ln |\boldsymbol\Sigma|}}_{\rm constant}-\frac{1}{2}(\mathbf{x} -\boldsymbol{\mu}_c)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_c) \right \} \\


&= \arg\max_c\, \left \{\ln \pi_c -\frac{1}{2}\underbrace{\colorbox{lightgreen}{$(\mathbf{x} -\boldsymbol{\mu}_c)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_c)$} }_{\text{distance btw } \mathbf{x} \text{ and } \boldsymbol{\mu}_c}\right \}
\end{align}
```

* ##### *i.e.* classify based on distance to centre ``\boldsymbol{\mu}_c`` (adjusted with the prior)

* ##### the decision boundary is therefore **linear**

"""

# ╔═╡ 51b1572f-5965-41a8-b6d6-061c48f9af0c
# md"""

# ## Linear discriminant analysis (LDA)


# **Linear discriminant analysis (LDA)** model assumes for all ``c``, 


# $\large \large \boldsymbol{\Sigma}_c = \boldsymbol{\Sigma}\;\; \text{for all }c=1\ldots C$

# Then the decision rule reduces to

# ```math
# \large
# \begin{align}
# \hat{y} 
# &= \arg\max_c\, \left \{\ln \pi_c - \underbrace{\cancel{\frac{1}{2}\ln |\boldsymbol\Sigma|}}_{\rm constant}-\frac{1}{2}(\mathbf{x} -\boldsymbol{\mu}_c)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_c) \right \} \\


# &= \arg\max_c\, \left \{\ln \pi_c -\frac{1}{2}\underbrace{\colorbox{lightgreen}{$(\mathbf{x} -\boldsymbol{\mu}_c)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_c)$} }_{\text{distance btw } \mathbf{x} \text{ and } \boldsymbol{\mu}_c}\right \}
# \end{align}
# ```

# * ###### *i.e.* classify ``y`` based on its distance to the centers (adjusted with the prior)

# * the decision boundary is therefore **linear**
# """

# ╔═╡ 2c3f2b50-0a95-4577-add8-8bf72580a44f
# let
# 	Random.seed!(123)
# 	K₁ =3
# 	n₁ = 600
# 	# D₁ = zeros(n₁, 2)
# 	# 200 per cluster
# 	truezs₁ = repeat(1:K₁; inner=200)
# 	trueμs₁ = zeros(2, K₁)
# 	trueμs₁[:,1] = [2.0, 2.0]
# 	trueμs₁[:,2] = [-2.0, 2]
# 	trueμs₁[:,3] = [0., -1.5]
# 	LL = cholesky([1 0; 0. 1]).L
# 	data₁ = trueμs₁[:,1]' .+ randn(200, 2) * LL
# 	data₁ = vcat(data₁, trueμs₁[:,2]' .+ randn(200, 2)* LL)
# 	data₁ = vcat(data₁, trueμs₁[:,3]' .+ randn(200, 2)* LL)
# 	plt₁ = plot(ratio=1, framestyle=:origin)
# 	truemvns = [MvNormal(trueμs₁[:, k], Matrix(I,2,2)) for k in 1:K₁]

# 	xs_qda = minimum(data₁[:,1])-0.1:0.1:maximum(data₁[:,1])+0.1
# 	ys_qda = minimum(data₁[:,2])-0.1:0.1:maximum(data₁[:,2])+0.1
# 	for k in 1:K₁
# 		scatter!(data₁[truezs₁ .== k,1], data₁[truezs₁ .==k,2], c=k, label="Class"*string(k), alpha=0.8) 
# 		contour!(xs_qda, ys_qda, (x,y)-> pdf(truemvns[k], [x,y]), levels=5, colorbar = false, lw=2, alpha=0.7, c=:jet) 
# 	end
# 	plt₁
# end

# ╔═╡ ec66c0b1-f75c-41e5-91b9-b6a358cd9c3c
# md"""

# ## LDA example

# The covariance *e.g* is identity matrix
# ```math
# \large\boldsymbol{\Sigma}_c = \mathbf{I}
# ```
# Then, the decision rule reduces to 

# ```math
# \large
# \begin{align}
# \hat{y} 
# &= \arg\max_c\, \left \{\ln \pi_c -\frac{1}{2}\underbrace{\colorbox{lightgreen}{$(\mathbf{x} -\boldsymbol{\mu}_c)^\top(\mathbf{x} -\boldsymbol{\mu}_c)$} }_{\text{Eucliean distance } \mathbf{x} \text{ and } \boldsymbol{\mu}_c}\right \}
# \end{align}
# ```
# * assign ``\hat{y}`` based on the Euclidean distances to the centres
# * note the decision boundary is linear !
# """

# ╔═╡ ea4a783e-6a18-4d4e-b0a2-bf4fd8070c7a
md"""

## Demonstration 

#### The covariance *e.g* is shared 
```math
\large\boldsymbol{\Sigma}_1 = \large\boldsymbol{\Sigma}_2 =\large\boldsymbol{\Sigma}_3= \begin{bmatrix}1 & 0.8 \\\ 0.8 & 1.0 \end{bmatrix}
```

"""

# ╔═╡ 3b937208-c0bb-42e2-99a2-533113a0d4e0
md"Add boundary: $(@bind add_db_2 CheckBox(default=false))"

# ╔═╡ 2ab95efd-4aca-412b-998f-f777b151d05e
md"""

## Why linear?


#### For LDA, the log-posterior for class ``c`` is:

```math
\Large
\begin{align}
\ln p(y=c|\mathbf{x}) &= \ln \pi_c -\frac{1}{2} (\mathbf{x} -\boldsymbol{\mu}_c)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_c) + C\\
&=\mathbf{w}_c^\top \mathbf{x} + b_{c} + C
\end{align}
```

* ##### where ``\mathbf{w}_c = \mathbf{\Sigma}^{-1}\boldsymbol{\mu}_c`` and ``b_{c} = \ln(\pi_c)  -\frac{1}{2} \boldsymbol{\mu}_c^\top\mathbf{\Sigma}^{-1}\boldsymbol{\mu}``

* #### this is the same as softmax regression! 
  * (apply ``\exp`` on both sides and normalise!)

```math
\Large
\begin{align}
p(y=c|\mathbf{x}) &= e^{C}\cdot{e^{(\mathbf{w}_c^\top \mathbf{x} + b_{c} )}}\\

&\propto {e^{(\mathbf{w}_c^\top \mathbf{x} + b_{c} )}}\\
&= \frac{e^{(\mathbf{w}_c^\top \mathbf{x} + b_{c})}}{\sum_{c'} e^{(\mathbf{w}_{c'}^\top \mathbf{x} + b_{c'} )}}
\end{align}
```


"""

# ╔═╡ 2e7c7e4c-83ec-4555-9b50-9a308354bab2
md"""
## LDA _vs_ Softmax regression
### Palma Penguine dataset: very similar classification 
"""

# ╔═╡ 4b4d7507-2a07-4c3c-95ce-5446a9ba2362
# md"""

# ## Why linear?


# #### *Classification boundary* between class 1 and 2 is defined

# ```math
# \Large
# \ln p(y=1|\mathbf{x}) = \ln p(y=2|\mathbf{x})\;\; \text{for }  \mathbf{x}\in \mathbb{R}^d
# ```

# * #### which is the same as
# ```math
# \large
# \ln \pi_1 -\frac{1}{2}\underbrace{\colorbox{lightgreen}{$(\mathbf{x} -\boldsymbol{\mu}_1)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_1)$} }_{\text{distance btw } \mathbf{x} \text{ and } \boldsymbol{\mu}_1}= \ln \pi_2 -\frac{1}{2}\underbrace{\colorbox{lightpink}{$(\mathbf{x} -\boldsymbol{\mu}_2)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_2)$} }_{\text{distance btw } \mathbf{x} \text{ and } \boldsymbol{\mu}_2}
# ```

# * #### we just need to show the above equation is linear

# """

# ╔═╡ 89bfebb4-ee0d-46c8-956a-6a5088599ae6
# md"""

# ## Why linear? -- details


# The **decision boundary** between class 1 and 2 is



# ```math
# \ln \pi_1 -\frac{1}{2}\underbrace{\colorbox{lightgreen}{$(\mathbf{x} -\boldsymbol{\mu}_1)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_1)$} }_{\text{distance btw } \mathbf{x} \text{ and } \boldsymbol{\mu}_1}= \ln \pi_2 -\frac{1}{2}\underbrace{\colorbox{lightpink}{$(\mathbf{x} -\boldsymbol{\mu}_2)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_2)$} }_{\text{distance btw } \mathbf{x} \text{ and } \boldsymbol{\mu}_2}
# ```

# Multiply ``2`` on both sides,
# ```math
# \begin{align}
# 2\ln \pi_1 -(\mathbf{x} -\boldsymbol{\mu}_1)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_1)= 2\ln \pi_2 -(\mathbf{x} -\boldsymbol{\mu}_2)^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} -\boldsymbol{\mu}_2) \tag{$\times 2$}
# \end{align}
# ```

# Then expand the quadratic terms,

# ```math
# \begin{align}
# 2\ln \pi_1 -(\cancel{\mathbf{x}^\top\boldsymbol{\Sigma}^{-1}\mathbf{x} }-&2\boldsymbol{\mu}_1^\top\boldsymbol{\Sigma}^{-1}\mathbf{x} + \boldsymbol{\mu}_1^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1) = \\
# &2\ln \pi_2 -(\cancel{\mathbf{x}^\top\boldsymbol{\Sigma}^{-1}\mathbf{x}} -2\boldsymbol{\mu}_2^\top\boldsymbol{\Sigma}^{-1}\mathbf{x} + \boldsymbol{\mu}_2^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2) 
# \end{align}

# ```

# ```math
# \begin{align}
# &\Rightarrow\;\; 2\ln \pi_1 + 2\boldsymbol{\mu}_1^\top\boldsymbol{\Sigma}^{-1}\mathbf{x}-\boldsymbol{\mu}_1^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1 = 2\ln \pi_2 + 2\boldsymbol{\mu}_2^\top\boldsymbol{\Sigma}^{-1}\mathbf{x}-\boldsymbol{\mu}_2^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2\\
# &\Rightarrow\;\; \underbrace{2(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)^\top\boldsymbol{\Sigma}^{-1}}_{\mathbf{w}^\top}\,\mathbf{x} = \underbrace{2\ln \frac{\pi_2}{\pi_1} +\boldsymbol{\mu}_1^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1-\boldsymbol{\mu}_2^\top\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_2 }_{w_0}
# \end{align}
# ```

# Therefore, we have the following form

# ```math
# \mathbf{w}^\top \mathbf{x} - w_0 =0
# ```
# * which is a hyper-plane (linear function)
# """

# ╔═╡ 1747e403-c5d6-471a-9f6e-fcf7ee8063d1
# let
# 	class = 3
# 	gr()
# 	data = data₁
# 	# K = K₃
# 	zs = truezs₁
# 	μs, Σ, πs = LDA_fit(data, zs)
# 	mvns = [MvNormal(μs[:, k], Symmetric(Σ)) for k in 1:size(μs)[2]]
# 	# mvns = []
# 	plt=plot(-7:.05:7, -6:0.05:6, (x,y) -> e_step([x, y]', mvns, πs)[1][:] |> argmax, c=:black, lw=1, alpha=0.7, title="Decision boundary by LDA", st=:contour, colorbar=false, ratio=1, framestyle=:grid)
# 	for k in [1,3,2]
# 		if k ==1
# 			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="")
# 		else
# 			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="")
# 		end
# 	end
# 	plt
# end

# ╔═╡ bcc88b1b-4e37-4d68-8af2-9b3923634dfd
md"""

## GDA -- quadratic discriminant analysis


> #### *Quadratic discriminant analysis (QDA)* model assumes for all ``c``, 
> 
> $\Large \boldsymbol{\Sigma}_1\neq\boldsymbol{\Sigma}_2 \neq \ldots \neq \boldsymbol{\Sigma}_c$
> * #### all covariances are not the same
> * ##### the decision boundary is *quadratic*

"""

# ╔═╡ 84a4326b-2482-4b5f-9801-1a09d8d20f5b
md"""

## QDA example

$\Large \boldsymbol{\Sigma}_1\neq\boldsymbol{\Sigma}_2 \neq \boldsymbol{\Sigma}_3$

"""

# ╔═╡ 5c8ebc31-6e76-44fa-9b05-455030defcfb
md"Add boundary: $(@bind add_db_qda CheckBox(default=false))"

# ╔═╡ 716ff72f-2299-472c-84b8-17315e8edc48
md"Choose class $(@bind class_d3 Select(1:3))"

# ╔═╡ a6b8e8e6-8af3-42e2-ae8a-d891739f0317
# md"""Compare classification boundary: $(@bind add_fits_ CheckBox(false))"""

# ╔═╡ 8336bd73-5b07-4c2a-b773-2df78de81bb2
md"""

## QDA vs LDA
"""

# ╔═╡ 6094abc7-96bb-4c3f-9156-b5de5d7873f6
md"""

## Naive Bayes (with Gaussian likelihood)


#### Naive Bayes assumption: _conditional independent_ ``\mathbf{x}``
"""

# ╔═╡ 8ca2a156-e0f8-453f-a02c-daf23681bf89
md"""


> ```math
> \large 
> \begin{align}
> p(\mathbf{x}|y=c) &= p(x_1|y)p(x_2|y) \ldots p(x_d|y)\\
> &= \prod_{i=1}^d p(x_i|y=c) = \mathcal{N}\left (\mathbf{x}; \boldsymbol{\mu}_c, \mathbf{\Sigma}_c = \begin{bmatrix} \sigma_{c;1}^2 &  &  \\ & \ddots &  \\ & &  \sigma^2_{c;d}\end{bmatrix}\right)
> \end{align}
> ```


* ##### which simply corresponds to diagonal covariance
"""

# ╔═╡ d4acae8a-8f27-4a2a-93d5-63d6b6b6db20
ThreeColumn(
md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[1]))

$(plot(plts_mvns[1], size=(220,220)))
	


"""	,
md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[2]))

$(plot(plts_mvns[2], size=(220,220)))
	



"""
	,


md"""

$$\mathbf \Sigma=$$
$(latexify_md(Σs[3]))


$(plot(plts_mvns[3], size=(220,220)))
	


"""
)

# ╔═╡ 102aeb0e-fc0f-4099-9e71-fdfc36863e2e
# let
# 	gr()
# 	K₂ = 3
# 	trueμs₂ = zeros(2,K₂)
# 	trueΣs₂ = zeros(2,2,K₂)
# 	trueμs₂[:,1], trueΣs₂[:,:,1] = [-2.0, 0.0], Matrix([0.5 .5; .5 1])
# 	trueμs₂[:,2], trueΣs₂[:,:,2] = [2.5, 0.0], Matrix([0.5 .5; .5 1])
# 	trueμs₂[:,3], trueΣs₂[:,:,3] = [0., -1],  Matrix([0.5 .5; .5 1])
# 	trueπs₂ = [0.3, 0.3, 0.4]
# 	truemvns₂ = [MvNormal(trueμs₂[:,k], trueΣs₂[:,:,k]) for k in 1:K₂]
# 	n₂= 600
# 	truezs₂ = rand(Categorical(trueπs₂), n₂)
# 	data₂= vcat([rand(truemvns₂[z])' for z in truezs₂]...)

# 	plt1 = plot_clusters(data₂, truezs₂, 3)
# 	# plot(plt1, xlim=(-5,5), size=(300, 300))


# 	K₂ = 3
# 	trueμs₂ = zeros(2,K₂)
# 	trueΣs₂ = zeros(2,2,K₂)
# 	trueμs₂[:,1], trueΣs₂[:,:,1] = [-2.0, 0.0], Matrix([0.5 -.4; -.4 1])
# 	trueμs₂[:,2], trueΣs₂[:,:,2] = [2.5, 0.0], Matrix([1 .8; .8 1])
# 	trueμs₂[:,3], trueΣs₂[:,:,3] = [0., -1],  Matrix([2 .8; .8 1])
# 	trueπs₂ = [0.3, 0.3, 0.4]
# 	truemvns₂ = [MvNormal(trueμs₂[:,k], trueΣs₂[:,:,k]) for k in 1:K₂]
# 	n₂= 800
# 	truezs₂ = rand(Categorical(trueπs₂), n₂)
# 	data₂= vcat([rand(truemvns₂[z])' for z in truezs₂]...)

# 	plt2 = plot_clusters(data₂, truezs₂, 3)



# 	K₂ = 3
# 	trueμs₂ = zeros(2,K₂)
# 	trueΣs₂ = zeros(2,2,K₂)
# 	trueμs₂[:,1], trueΣs₂[:,:,1] = [-2.5, 0.0], Matrix([0.3 0; 0 2])
# 	trueμs₂[:,2], trueΣs₂[:,:,2] = [2.5, 1.0], Matrix([2 0; 0 .3])
# 	trueμs₂[:,3], trueΣs₂[:,:,3] = [0., -1],  Matrix([2.5 0; 0 .3])
# 	trueπs₂ = [0.3, 0.3, 0.4]
# 	truemvns₂ = [MvNormal(trueμs₂[:,k], trueΣs₂[:,:,k]) for k in 1:K₂]
# 	n₂= 800
# 	truezs₂ = rand(Categorical(trueπs₂), n₂)
# 	data₂= vcat([rand(truemvns₂[z])' for z in truezs₂]...)

# 	plt3 = plot_clusters(data₂, truezs₂, 3)

# 	plot(plt1, plt2, plt3, layout=(1,3), size=(700, 280), xlabel=L"x_1", ylabel=L"x_2", ylim = (-5.5, 5.5), xlim = (-5.5, 5.5), title=["Dataset 1" "Dataset 2" "Dataset 3"])
# 	# xs = -6:0.1:6
# 	# ys = -6:0.1:6
# 	# 	# if add_db_2
# 	# πs = 1/2 * ones(2)

# 	# if false
# 	# 	plot!(xs, ys, (x,y) -> e_step([x, y]', truemvns₄, trueπs₄)[1][1], c=cgrad(:coolwarm; rev=true, scale=:log2), lw=2, alpha=0.4, title="QDA boundary", st=:contourf, fillalpha=0.1, colorbar=false, xlim =(-6, 6), ylim =(-6, 6), levels =0:0.1:1.0, framestyle=:origin)
	
# 	# 	for k in 1:2
# 	# 		plot!(xs, ys, (x,y)-> pdf(mvns_qda[k], [x,y]), levels=25,  st=:contour, colorbar = false,  color=:jet, linewidth=2) 
# 	# 		scatter!([mvns_qda[k].μ[1]], [mvns_qda[k].μ[2]], color = k, label = "", markersize = 5, markershape=:star4, markerstrokewidth=3, alpha=0.5,  framestyle=:origin)
# 	# 	end

# 	# end

# 	# μs, Σs, πs = LDA_fit(data₄, truezs₄)
# 	# mvns = [MvNormal(μs[:, i], Σs) for i in 1:2]
# 	# plt2 = plot_clustering_rst(data₄,  2, truezs₄, mvns, πs, title="LDA fit")

# 	# if false
# 	# 	plot!(plt2, xs, ys, (x,y) -> e_step([x, y]', mvns, πs)[1][1] ,c = cgrad(:coolwarm; rev=true, scale=:log2), lw=2, alpha=.5, title="LDA boundary", levels = 0:0.1:1.0,st=:contourf, fillalpha=0.1, colorbar=false, xlim =(-6, 6), ylim =(-6, 6))
	
# 	# 	for k in 1:2
# 	# 		plot!(xs, ys, (x,y)-> pdf(mvns[k], [x,y]), levels=30,  st=:contour, colorbar = false,  color=:jet, linewidth=2) 
# 	# 		scatter!([mvns[k].μ[1]], [mvns[k].μ[2]], color = k, label = "", markersize = 5, markershape=:star4, markerstrokewidth=3, alpha=0.5,  framestyle=:origin)
# 	# 	end

# 	# end
# 	# plot(plt, plt2, legend=false, ylim =(-5.5, 6), size=(700,350))
# end

# ╔═╡ cc60e789-02d1-4944-8ad5-718ede99669c
begin
	K₂ = 3
	trueμs₂ = zeros(2,K₂)
	trueΣs₂ = zeros(2,2,K₂)
	trueμs₂[:,1], trueΣs₂[:,:,1] = [-2.0, 0.0], 0.5 * Matrix(1.0I, 2,2)
	trueμs₂[:,2], trueΣs₂[:,:,2] = [2.0, 0.0], 0.5 * Matrix(1.0I, 2,2)
	trueμs₂[:,3], trueΣs₂[:,:,3] = [0., 0],  Matrix([0.5 0; 0 2])
	trueπs₂ = [0.2, 0.2, 0.6]
	truemvns₂ = [MvNormal(trueμs₂[:,k], trueΣs₂[:,:,k]) for k in 1:K₂]
	n₂= 800
	truezs₂ = rand(Categorical(trueπs₂), n₂)
	data₂= vcat([rand(truemvns₂[z])' for z in truezs₂]...)
	# data₂, truezs₂ = sampleMixGaussian(n₂, truemvns₂, trueπs₂)
end;

# ╔═╡ 629061d5-bf97-4ccf-af28-f1c5cd36b34c
# let
# 	class = 3
# 	gr()
# 	data = data₂
# 	# K = K₃
# 	zs = truezs₂
# 	# μs, Σs, πs = QDA_fit(data, zs)
# 	μs, Σs, πs = trueμs₂, trueΣs₂, trueπs₂
# 	mvns = [MvNormal(μs[:, k], Symmetric(Σs[:,:, k])) for k in 1:size(μs)[2]]
# 	plt=plot(-5:.05:5, -5:0.05:5, (x,y) -> e_step([x, y]', mvns, πs)[1][class], levels=6, c=:coolwarm, lw=1, alpha=1.0, title="Decision boundary by QDA: "*L"p(y=%$(class)|\mathbf{x})", st=:contour, colorbar=true, ratio=1, framestyle=:origin)
# 	for k in [1,3,2]
# 		# if k ==1
# 			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, alpha=0.7, ms=3, label="class $(k)")
# 		# else
# 		# 	scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="class $(k)")
# 		# end
# 	end
# 	plt
# end

# ╔═╡ b4bfb1ba-73f2-45c9-9303-baee4345f8f6
# let
# 	gr()
# 	zs= truezs₂
# 	data = data₂
# 	μs, Σs, πs = trueμs₂, trueΣs₂, trueπs₂
# 	mvns = [MvNormal(μs[:, k], Symmetric(Σs[:,:, k])) for k in 1:size(μs)[2]]
# 	# ws, _ = e_step(data, mvns, πs)
# 	# yŝ = argmax.(eachrow(ws))
# 	# wrong_idx = findall(zs .!= yŝ)
# 	plt = plot(-5:.05:5, -5:0.05:5, (x,y) -> decisionBdry(x,y, mvns, πs)[2], c=1:3, alpha=0.5, ratio=1, framestyle=:origin, title="Decision boundary by QDA", st=:heatmap, colorbar=false)
# 	for k in [1,3,2]
# 		zks = findall(zs .== k)
# 		scatter!(data[zks, 1], data[zks, 2], c = k, alpha=0.3, label="")
# 	end
# 	plt
# end

# ╔═╡ 6be1c156-8dcc-48e3-b684-e89c4a0a7863
# TwoColumn(md"""The true parameters are known:

# $\boldsymbol\pi = [0.2, 0.6, 0.2]$

# $\boldsymbol\mu_1 = [1.5 , 1.5];\,\boldsymbol \Sigma_1 = \begin{bmatrix}2, -1.5\\-1.5, 2\end{bmatrix}$
# $\boldsymbol\mu_2 = [0.0 , 0.0];\, \boldsymbol\Sigma_2 = \begin{bmatrix}2, 1.5\\1.5, 2\end{bmatrix}$
# $\boldsymbol\mu_3 = [-1.5 , -1.5];\,\boldsymbol \Sigma_3 = \begin{bmatrix}2, -1.5\\-1.5, 2\end{bmatrix}$


# """, let
# 	# gr()
# 	# plt₃_= plot_clustering_rst(data₃,  K₃, truezs₃, truemvns₃, trueπs₃)
# 	# title!(plt₃_, "More QDA example")
# 	plot(plt₃_, size=(350,350))
# end)

# ╔═╡ b5732a82-e951-475a-b6f2-d8584c07c7a9
# let
# 	gr()
# 	plt₃_= plot_clustering_rst(data₃,  K₃, truezs₃, truemvns₃, trueπs₃)
# 	title!(plt₃_, "More QDA example")
# 	plot(plt₃_, size=(350,350))
# end

# ╔═╡ 552da9d4-5ab9-427d-9287-feb1eca57183
# md"""


# ## Model assumptions


# It can be a disaster if we apply **LDA** on data with non-linear decision boundaries

# * or when the co-variances are **very different** (the **QDA** assumption)
# """

# ╔═╡ 55f2aacf-a342-4d4c-a1b7-60c5c29ab340
# md"Highlight mis-classified instances: $(@bind add_mis CheckBox(default=false))"

# ╔═╡ 3f869ae4-14b1-4b5a-b5c9-bc437bfc99da
# TwoColumn(plot(plt_d3_true_bd, size=(330,330), title="QDA on non-linear data", framestyle=:origins, titlefontsize=10), let
# 	gr()
# 	data = data₃
# 	zs = truezs₃
# 	μs, Σ, πs = LDA_fit(data, zs)
# 	mvnsₘₗ = [MvNormal(μ, Symmetric(Σ)) for μ in eachcol(μs)]
# 	ws, _ = e_step(data, mvnsₘₗ, πs)
# 	yŝ = argmax.(eachrow(ws))
# 	plt=plot(-7:.02:7, -7:0.02:7, (x,y) -> decisionBdry(x, y, mvnsₘₗ, πs)[2], c=1:3, alpha=0.6, title="LDA on a non-linear data", st=:heatmap, colorbar=false, ratio=1, framestyle=:origins, size=(330,330),titlefontsize=10)
# 	wrong_idx = findall(zs .!= yŝ)
# 	for k in [1,3,2]
# 		zks = findall(zs .== k)
# 		scatter!(data[zks, 1], data[zks, 2], c = k, alpha=0.6, label="")

# 	end
# 	acc = length(wrong_idx) / length(zs)

# 	if add_mis
# 		wrongzk = wrong_idx
# 		scatter!(data[wrongzk, 1], data[wrongzk, 2], c = :black, markersize=6, label="mis-classified",alpha=0.8, marker=:x, markerstrokewidth=2, title="LDA on non-linear data; accuracy = "*L"%$(acc)", titlefontsize=10)
# 	end

# 	plt
# end)

# ╔═╡ 5a963c3d-46cd-4697-8991-5d5e1bb9a9e5
# let
# 	gr()
# 	data = data₃
# 	zs = truezs₃
# 	μs, Σ, πs = LDA_fit(data, zs)
# 	mvnsₘₗ = [MvNormal(μ, Symmetric(Σ)) for μ in eachcol(μs)]
# 	ws, _ = e_step(data, mvnsₘₗ, πs)
# 	yŝ = argmax.(eachrow(ws))
# 	plt=plot(-6:.02:6, -6:0.02:6, (x,y) -> decisionBdry(x, y, mvnsₘₗ, πs)[2], c=1:3, alpha=0.6, title="LDA on a non-linear data", st=:heatmap, colorbar=false, ratio=1, framestyle=:zerolines)
# 	wrong_idx = findall(zs .!= yŝ)
# 	for k in [1,3,2]
# 		zks = findall(zs .== k)
# 		scatter!(data[zks, 1], data[zks, 2], c = k, alpha=0.6, label="")

# 	end
# 	acc = length(wrong_idx) / length(zs)

# 	if add_mis
# 		wrongzk = wrong_idx
# 		scatter!(data[wrongzk, 1], data[wrongzk, 2], c = :black, markersize=6, label="mis-classified",alpha=0.8, marker=:x, markerstrokewidth=2, title="LDA on a non-linear data; accuracy = "*L"%$(acc)")
# 	end

# 	plt
# end

# ╔═╡ 1b6cb7eb-814e-4df8-89ed-56ebc8f06a4a
# md"""


# ## Model assumptions


# On the contrary, if we apply **QDA** on data with linear decision boundaries

# * or when the co-variances are **shared** (the **LDA** assumption)
# * it is not a big issue if there are enough training data or the dimension is small; otherwise it will overfit
# """

# ╔═╡ d5ec14e2-fa45-4232-9ae8-06b84bf48525
# let
# 	class = 3
# 	gr()
# 	data = data₁
# 	# K = K₃
# 	zs = truezs₁
# 	μs, Σs, πs = QDA_fit(data, zs)
# 	mvns = [MvNormal(μs[:, k], Symmetric(Σs[:,:, k])) for k in 1:size(μs)[2]]
# 	# mvns = []
# 	plt=plot(-7:.02:7, -6:0.02:6, (x,y) -> e_step([x, y]', mvns, πs)[1][:] |> argmax, c=1:3, lw=1, alpha=0.7, title="Decision boundary by QDA", st=:heatmap, colorbar=false, ratio=1, framestyle=:origins)
# 	for k in [1,3,2]
# 		if k ==1
# 			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="")
# 		else
# 			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="")
# 		end
# 	end
# 	plt
# end

# ╔═╡ c4ab9540-3848-483c-9ba5-e795913f844a
# let
# 	class = 3
# 	gr()
# 	data = data₁
# 	# K = K₃
# 	zs = truezs₁
# 	μs, Σs, πs = QDA_fit(data, zs)
# 	mvns = [MvNormal(μs[:, k], Symmetric(Σs[:,:, k])) for k in 1:size(μs)[2]]
# 	# mvns = []
# 	plt=plot(-5:.05:5, -5:0.05:5, (x,y) -> e_step([x, y]', mvns, πs)[1][class], levels=6, c=:coolwarm, lw=1, alpha=1.0, title="Decision boundary by supervised learning QDA", st=:contour, colorbar=false, ratio=1, framestyle=:grid)
# 	for k in [1,3,2]
# 		if k ==1
# 			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="")
# 		else
# 			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="")
# 		end
# 	end
# 	plt
# end

# ╔═╡ cba8b537-de68-4b2f-bf7a-d0bdd3aded7a
md"""

# MLE learning of GDA
"""

# ╔═╡ a0cd59ef-6468-4622-ab0e-0f7239eba0a8
md"""

## GDA Learning 
"""

# ╔═╡ fb217fd9-b929-47d9-ab69-b4637c894205
TwoColumn(md"""

\

#### In practice, we do not know the model parameters 

```math
\Large
\boldsymbol\Phi = \{\boldsymbol\pi, \{\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c\}_{c=1}^C\}
```


* #### but we have the observed ``\large \mathcal{D}=\{\mathbf{x}^{(i)}, y^{(i)}\}_{i=1}^n``


* #### (log-)likelihood is available 

```math
\Large
\ell(\Phi) = \ln p(\mathcal{D}|\mathbf{\Phi})
```
""", 

show_img("gda_learning.svg")
)

# ╔═╡ 4e038980-c531-4f7c-9c51-4e346eccc0aa
md"""



## MLE



#### We use maximum likelihood estimation (MLE) again

$$\Large\hat{\boldsymbol{{\Phi}}} \leftarrow \arg\min_{\boldsymbol\Phi}\, \underbrace{-\ln p(\mathcal{D}|\boldsymbol\Phi)}_{\mathcal{L}(\boldsymbol\Phi)}$$


* #### *Learning*: estimate a model's parameters given *observed* ``\mathcal{D}=\{\mathbf{x}^{(i)}, y^{(i)}\}_{i=1}^n``


* #### Big picture: good old calculus, solve

$\Large \frac{\partial \mathcal L}{\partial \boldsymbol\pi} =\mathbf 0; \; \frac{\partial \mathcal L}{\partial \boldsymbol\mu_c} =\mathbf 0;\; \frac{\partial \mathcal L}{\partial \boldsymbol\Sigma_c} =\mathbf 0$


"""

# ╔═╡ 359b407f-6371-4e8a-b822-956173e89a47
md"""

## QDA learning: MLE 

#### The optimisation has closed form analytical solution:

> $$\Large \hat \pi_c = \frac{n_c}{n}$$
> $$\Large \hat{\boldsymbol{\mu}}_c = \frac{1}{n_c}\, {\sum_{i=1}^n \mathbb{1}(y^{(i)}=c)\cdot\mathbf x^{(i)}}$$
> $$\Large \hat{\boldsymbol{\Sigma}}_c = \frac{1}{n_c} \sum_{i=1}^n \mathbb{1}(y^{(i)}=c) (\mathbf x^{(i)}-\hat{\boldsymbol{\mu}}_c)(\mathbf x^{(i)}-\hat{\boldsymbol{\mu}}_c)^\top$$

* where  $n_c = \sum_{i=1}^n \mathbb{1}(y^{(i)} = c)$
* ``\hat{\boldsymbol{\pi}}``: frequency of labels belong to each class 
* ``\hat{\boldsymbol{\mu}}_c, \hat{\boldsymbol{\Sigma}}_c``: the sample mean and covariance of the datasets belong to each class $c$
"""

# ╔═╡ e4176cf0-d5b7-4b9a-a4cd-b25f0f5a987f
function QDA_fit(XX, Y)
	n, d = size(XX)
	K = length(unique(Y))
	μs = zeros(d, K)
	Σs = zeros(d,d,K)
	ns = zeros(Int, K)
	# for each class c
	for c in (unique(Y)|>sort)
		# compute nc
		ns[c] = sum(Y .==c)
		# find all X such that y=c
		xc = XX[Y .== c, :]
		# compute the mean
		μs[:, c] = μc = mean(xc, dims=1)[:]
		# compute the variance matrix
		error = (xc .- μc')
		Σs[:,:,c] = error' * error/ns[c] ## XᵀX = ∑ xixiᵀ
	end
	μs, Σs, ns/n
end

# ╔═╡ dd7db7a6-51c0-4421-8308-2c822107a370
TwoColumn(
html"<br/><br/><center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gda_bnalone.svg
' width = '240' /></center>"

, 
	
let
	plt2_pen_qda = @df df scatter(:bill_length_mm, :bill_depth_mm, group =:species, framestyle=:origins,  alpha=0.5,xlabel="bill length (mm)", ylabel="bill depth (mm)", size=(330,330));
	df_ = df |> dropmissing
	ys_peng = df_[:, 1] 
	peng_map = Dict((s,i) for (i, s) in enumerate(unique(df_[:, 1])|>sort))
	ys_peng = [peng_map[s] for s in ys_peng]
	Xs_peng =  df_[:, 3:4]  |> Matrix
	pen_μs, pen_Σs, pen_πs = QDA_fit(Xs_peng, ys_peng)
	pen_mvns = [MvNormal(pen_μs[:,k], pen_Σs[:,:,k]) for k in 1:3]
	
	for k in 1:3
		plot!(plt2_pen_qda, 32:0.1:60, 13:0.1:22, (x,y)-> pdf(pen_mvns[k], [x,y]), levels=10,  st=:contour, colorbar = false, alpha=0.6, color=k, linewidth=3) 
		scatter!(pen_μs[1:1, k], pen_μs[2:2, k], c=k, label="", markershape=:x, markerstrokewidth=5, markersize=6)
	end

	plt2_pen_qda
end)

# ╔═╡ af373899-ed4d-4d57-a524-83b04063abf3
# md"""

# ## What's MLE of ``\boldsymbol\pi``?

# #### The prior ``\large p(y)``'s parameter ``\boldsymbol{\pi}``


# $\large p(y^{(i)})= \mathcal{Cat}(y^{(i)}; \boldsymbol\pi)=\begin{cases}\pi_1 & y^{(i)}=1 \\ \pi_2 & y^{(i)} =2 \\\vdots & \vdots \\ \pi_C & y^{(i)} =C\end{cases}$

# * #### ``\pi_c``: the prior proportion of ``y^{(i)} =c``


# """

# ╔═╡ 6331b0f5-94be-426d-b055-e1369eb2a962
md"""

## What's MLE of ``\boldsymbol\pi``?



"""

# ╔═╡ 06eebe92-bbab-449f-acb9-0e31ad2bfaa8
TwoColumnWideLeft(md"""

#### *Question*: there are total 2620 M&Ms

| Red | Orange | Yellow | Green | Blue | Brown |
| --- | ------ | ------ | ------| -----| ----- |
| 372 | 544    | 369    | 483   | 481  |  371  |

> ##### How to estimate the probability of each color ``\boldsymbol\pi``? 

""", html"<center><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Plain-M%26Ms-Pile.jpg/500px-Plain-M%26Ms-Pile.jpg' width = '200' /></center>")

# ╔═╡ 7c48e850-fdd9-4e77-87fa-a5800a26a77b
md"""

#### *Maximum Likelihood Estimation (MLE)* is just *relative frequency*


$$\Large \hat{\pi}_c = \frac{n_c}{n} = \frac{\sum_{i=1}^n \mathbb{1}(y^{(i)} = c)}{n}$$

* ``n`` is the total number of observations, that is 2620
* ``n_c`` is the number/count of ``c`` events, which is defined as 


**For example,**

$$\hat{\pi}_{red} = \frac{372}{2620}$$
"""


# ╔═╡ 95e63027-6434-4fa7-a9e1-bcca754b9601
begin

	df_ = df |> dropmissing
	ys_peng = df_[:, 1] 
	peng_map = Dict((s,i) for (i, s) in enumerate(unique(df_[:, 1])|>sort))
	ys_peng = [peng_map[s] for s in ys_peng]
	Xs_peng =  df_[:, 3:4]  |> Matrix
	pen_μs, pen_Σs, pen_πs = QDA_fit(Xs_peng, ys_peng)
	pen_mvns = [MvNormal(pen_μs[:,k], pen_Σs[:,:,k]) for k in 1:3]

end;

# ╔═╡ 35ad62e8-1212-4480-9b6f-d4c82da72c2e
plt_sfm = let
	Xs_peng_ = [ones(size(Xs_peng)[1]) Xs_peng]
	Xs_peng_[:, 2:3] = Xs_peng_[:, 2:3] .- mean(Xs_peng, dims=1)
	peng_mu =  mean(Xs_peng, dims=1)
	peng_std = std(Xs_peng, dims=1)
	Xs_peng_[:, 2:3] = Xs_peng_[:, 2:3] ./ std(Xs_peng, dims=1)
	ys = Matrix(I, 3, 3)[:, ys_peng]
	WW = zeros(3, 3)
	γ = 1.0
	losses = []
	for _ in 1:500
		loss, gw = Zygote.withgradient(WW) do W
			lnŷ = logsoftmax(W * Xs_peng_')
			sum(-lnŷ .* ys)/size(Xs_peng_)[1]
		end
		WW -= γ * gw[1]
		push!(losses, loss)
	end

	
	transform(v) = (v .- peng_mu) ./ peng_std

	
	plt = @df df scatter(:bill_length_mm, :bill_depth_mm, group =:species, framestyle=:origins,  alpha=0.5,xlabel="bill length (mm)", ylabel="bill depth (mm)")
	plot!(32:.1:60, 13:.1:22, (x, y) -> WW * [1, transform([x y])...] |> argmax, lw=1.0,  st=:heatmap, c=1:3, alpha=0.5, colorbar=false, size=(330,330), title="Softmax regression")

end;

# ╔═╡ 62f07c1e-4226-4a35-8d3a-198e41e10354
md"""

## Demonstration -- estimation of ``\pi``
"""

# ╔═╡ 2f7b3bf2-ce1a-4755-af3b-a82f02fb7752
let
	plt = @df df_ scatter(:bill_length_mm, :bill_depth_mm, group =:species,alpha=0.2,  framestyle=:origins,  xlabel="bill length (mm)", ylabel="bill depth (g)");
	# for k in 1:3
	k = 1
	nks = counts(ys_peng, 1:3)
		# plot!(32:0.1:60, 13:0.1:22, (x,y)-> pdf(pen_mvns[k], [x,y]), levels=6,  st=:contour, colorbar = false, alpha=0.5, color=k, linewidth=3) 
	title = ""
	anim=@animate for k in 1:3
		title = title * " "*L"n_{%$k} = %$(nks[k]);"
		scatter!(Xs_peng[ys_peng .==k, 1], Xs_peng[ys_peng .==k, 2], c=k, label="", title=title)
	end
	# end
	# @animate for k in 1:K
	# 	scatter!()
	# end
	gif(anim, fps=1)
end

# ╔═╡ a3ea595b-7b3e-4b97-bf1f-21f9a07fdd0d
md"""

## Recall:  MLE of multivariate  Gaussian



#### The MLE estimators for multi-variate Gaussian ``\boldsymbol\mu, \boldsymbol\Sigma``


$$\large \hat{\boldsymbol{\mu}} = \frac{1}{n}{\sum_{i=1}^n \mathbf x^{(i)}}$$

$$\large \hat{\boldsymbol{\Sigma}} = \frac{1}{n} \sum_{i=1}^n (\mathbf x^{(i)}-\hat{\boldsymbol{\mu}})(\mathbf x^{(i)}-\hat{\boldsymbol{\mu}})^\top$$


* ##### again very straighforward: sample mean and sample variance
"""

# ╔═╡ 8e4324d5-2a88-41d3-b229-43e9f41d4191
# md"""

# ## 

# Note that for uni-variate Gaussian, the MLE are just the sample average and variance


# $$\large \hat{{\mu}} = \frac{1}{n}{\sum_{i=1}^n  x^{(i)}}$$

# $$\large \hat{{\sigma}}^2 = \frac{1}{n} \sum_{i=1}^n ( x^{(i)}-\hat{{\mu}})( x^{(i)}-\hat{{\mu}}) = \frac{1}{n} \sum_{i=1}^n( x^{(i)}-\hat{{\mu}})^2$$


# """

# ╔═╡ 9c7bbd1f-cf1c-4eae-a4c5-36324c5aff0a
nks = counts(ys_peng, 1:3);

# ╔═╡ de5879af-c979-4b3b-a444-db264c30297b
md"""The counts are $(latexify_md(nks)), therefore

```math
\Large
\hat{\boldsymbol\pi} = \begin{bmatrix}\frac{146}{333} \\ \frac{68}{333} \\ \frac{119}{333} \end{bmatrix}
```
"""

# ╔═╡ c1b120cb-36ec-49b9-af55-13e98630b6db
md"""

## Demonstration -- estimation of ``\mu``

$$\Large\hat{\boldsymbol{\mu}}_c = \frac{1}{n_c}\, {\sum_{i=1}^n \mathbb{1}(y^{(i)}=c)\cdot\mathbf x^{(i)}}$$
"""

# ╔═╡ ed986bfb-1582-4dcb-b39f-565657cfa59c
md"""

## Demonstration -- estimation of ``\boldsymbol{\mu}, \boldsymbol{\Sigma}``



$$\large\hat{\boldsymbol{\Sigma}}_c = \frac{1}{n_c} \sum_{i=1}^n \mathbb{1}(y^{(i)}=c) (\mathbf x^{(i)}-\hat{\boldsymbol{\mu}}_c)(\mathbf x^{(i)}-\hat{\boldsymbol{\mu}}_c)^\top$$

"""

# ╔═╡ 170fc849-2f28-4d31-81db-39fbcc6ac6e4
let
	plt = @df df_ scatter(:bill_length_mm, :bill_depth_mm, group =:species,alpha=0.2,  framestyle=:origins,  xlabel="bill length (mm)", ylabel="bill depth (g)");
	# for k in 1:3
	k = 1
	nks = counts(ys_peng, 1:3)
		# plot!(32:0.1:60, 13:0.1:22, (x,y)-> pdf(pen_mvns[k], [x,y]), levels=6,  st=:contour, colorbar = false, alpha=0.5, color=k, linewidth=3) 
	title = "Estimate"
	anim=@animate for k in 1:3
		title = title * " "*L"\{\mu_{%$k}, \Sigma_{%$(k)}\};"
		scatter!(Xs_peng[ys_peng .==k, 1], Xs_peng[ys_peng .==k, 2], c=k, alpha=0.6, label="", title=title)

		scatter!(pen_μs[1:1, k], pen_μs[2:2, k], markershape=:x, markerstrokewidth=5, markersize=8,label="", c= k)

		plot!(32:0.1:60, 13:0.1:22, (x,y)-> pdf(pen_mvns[k], [x,y]), levels=6,  st=:contour, colorbar = false, alpha=0.5, color=k, linewidth=3) 
	end

	gif(anim, fps=0.8)
end

# ╔═╡ 92e04df6-153d-402d-a7fe-f708390c1185
# md"""

# ## Another dataset


# MLE are very efficient when the model assumptions are correct

# * it means the estimates converge to the true parameters well
# """

# ╔═╡ f2d0e1e0-a8ef-4ac5-8593-892e4a5ac67c
md"""

## Demonstration
"""

# ╔═╡ 05820b6f-45e9-4eaa-b6ba-c52813b5fe46
md"""

## LDA learning: MLE 



#### The *maximum likelihood estimators* are 

> $$\large\hat{\boldsymbol{\Sigma}} = \frac{1}{n}\sum_{i=1}^n\sum_{c=1}^C  \mathbb{1}(y^{(i)}=c) (\mathbf x^{(i)}-\boldsymbol{\mu}_c)(\mathbf x^{(i)}-\boldsymbol{\mu}_c)^\top$$


* ##### ``\large \hat{\boldsymbol{\Sigma}}``: the pooled covariance of datasets 
"""

# ╔═╡ 1d08c5f5-cbff-40ef-bcb8-971637931e20
function LDA_fit(XX, Y)
	n, d = size(XX)
	sse = zeros(d, d)
	K = length(unique(Y))
	μs = zeros(d, K)
	ns = zeros(Int, K)
	for c in (unique(Y)|>sort)
		ns[c] = sum(Y .==c)
		xc = XX[Y .== c, :]
		μs[:, c] = μc = mean(xc, dims=1)[:]
		error = (xc .- μc')
		sse += error' * error
	end
	μs, sse/n, ns/n
end

# ╔═╡ af868b9b-130d-4d4f-8fc6-ff6d9b6f604f
# md"True $\mu_1 = [-2 , 1];\mu_2 = [2 , 1]; \mu_3 = [0 , -1]$; 

# Estimated ``\hat{\boldsymbol{\mu}}=``"

# ╔═╡ 5b980d00-f159-49cd-b959-479cd3b1a444
# latexify((μ_ml[:, 1]); fmt=FancyNumberFormatter(2)),latexify((μ_ml[:, 2]); fmt=FancyNumberFormatter(2)), latexify((μ_ml[:, 3]); fmt=FancyNumberFormatter(2))

# ╔═╡ 08eb8c76-c19a-431f-b5ad-a14a38b18946
# md"True $\Sigma_1 = \begin{bmatrix}.5, 0\\0, .5\end{bmatrix},  \Sigma_2 = \begin{bmatrix}.5, 0\\0, .5\end{bmatrix}, \Sigma_3 = \begin{bmatrix}0.5, 0\\0, 2\end{bmatrix}$;

# Estimated ``\hat{\boldsymbol{\Sigma}}=``"

# ╔═╡ 928a1491-3695-4bed-b346-b983f389a26f
# latexify((Σsₘₗ[:,:, 1]); fmt=FancyNumberFormatter(2)), latexify((Σsₘₗ[:,:, 2]); fmt=FancyNumberFormatter(2)), latexify((Σsₘₗ[:,:, 3]); fmt=FancyNumberFormatter(2))

# ╔═╡ bc04175a-f082-46be-a5ee-8d16562db340
# md"True $\boldsymbol\pi = [0.2, 0.2, 0.6]$ ; 

# Estimated ``\hat{\boldsymbol{\pi}}=`` $(latexify_md(πₘₗ))"

# ╔═╡ b0e16123-df7e-429c-a795-9e5ba788171a
πₘₗ = counts(truezs₂)/length(truezs₂);

# ╔═╡ a0465ae8-c843-4fc0-abaf-0497ada26652
md"""

# Appendix

"""

# ╔═╡ cdf72ed6-0d70-4901-9b8f-a12ceacd359d
md"""
## MLE -- further details



Due to independence, the likelihood is

$p(\mathcal{D}|\Phi) = \prod_{i=1}^n p(y^{(i)}, x^{(i)}) = \prod_{i=1}^n p(y^{(i)})p(x^{(i)}|y^{(i)})$

Take log 

$$\mathcal{L}(\Phi) = \ln p(\mathcal{D}|\Phi) = \sum_{i=1}^n \ln p(y^{(i)})+ \sum_{i=1}^n \ln p(x^{(i)}|y^{(i)})$$




Write down the distribution with $\mathbb{1}(\cdot)$ notation (you should verify they are the same)

$$p(y^{(i)}) = \prod_{k=1}^K \pi_k^{\mathbb{1}(y^{(i)}=k)}$$ and also

$$p(x^{(i)}|y^{(i)}) = \prod_{k=1}^K \mathcal{N}(x^{(i)}|\mu_k,\Sigma_k)^{\mathbb{1}(y^{(i)}=k)}$$

Their logs are 

$$\ln p(y^{(i)}) = \sum_{k=1}^K {\mathbb{1}(y^{(i)}=k)} \cdot \pi_k;\;\; \ln p(x^{(i)}|y^{(i)}) =  \sum_{k=1}^K {\mathbb{1}(y^{(i)}=k)} \cdot \ln \mathcal{N}(x^{(i)}|\mu_k,\Sigma_k)$$

Then

$$\mathcal L(\Phi) = \sum_{i=1}^n \sum_{k=1}^K {\mathbb{1}(y^{(i)}=k)} \cdot \ln \pi_k+ \sum_{i=1}^n \sum_{k=1}^K {\mathbb{1}(y^{(i)}=k)} \cdot \ln \mathcal{N}(x^{(i)}|\mu_k,\Sigma_k)$$

Therefore, we can isolate the terms and write $\mathcal L$ as a function of $\mu_k, \Sigma_k$:

$\mathcal L(\mu_k,\Sigma_k) = \sum_{i=1}^n {\mathbb{1}(y^{(i)}=k)} \cdot \ln \mathcal{N}(x^{(i)}|\mu_k,\Sigma_k) +C$

which justifies why the MLE for $\mu_k, \Sigma_k$ are the pooled MLE for the k-th class's observations!


The first term is ordinary Multinoulli log-likelihood, its MLE is relative frequency (need to use Lagrange multiplier as $\sum_{k} \pi_k =1$).
"""

# ╔═╡ e03a111a-8cf9-40af-842d-8f8ca8a197fd
md"""

## Gradient descent*

You can also do gradient but it is significantly more complex due to the fact the parameters are constained. For example, the prior parameter should sum to one ``\sum_c \pi_c=1``; and the covariance matrix should be positive definite.

That said, it is still doable by reparameterise the parameters. For example, apply `softmax` to reparameterise ``\pi``, and optimise ``\phi`` instead:


```math
\boldsymbol{\pi} = \texttt{softmax}(\boldsymbol{\phi})
```

* see below for a demonstration
"""

# ╔═╡ 66440f3a-65b4-4f4e-8158-d886138044e3
let
	Random.seed!(123)
	## simulate 500 observations from a categorical distribution with true parameter 0.1, 0.2, 0.7
	ys = sample(1:3, Weights([0.1, 0.2, 0.7]), 500)
	ϕ = randn(3)
	γ = 0.5
	for i in 1:100
		gs = Zygote.gradient(ϕ) do ϕ 
			logπ = logsoftmax(ϕ)
			- mean(logπ[ys])
		end
		ϕ .-= γ * gs[1]
	end

	## transform the parameter back; MLE of π
	softmax(ϕ)
end

# ╔═╡ 976711c5-009d-4feb-bc14-d262e250fdc5
# md"""
# #### reparameterize positive definite matrix 

# The covariance matrix is a bit more involved, but doable, note that given a lower triangular matrix ``\mathbf{L}``, ``\mathbf{LL}^\top`` is always positive definite. To see this, 

# ```math
# \mathbf{x}^\top\mathbf{LL}^\top \mathbf{x} =( \mathbf{L}^\top \mathbf{x})^\top (\mathbf{L}^\top\mathbf{x}) =\|\mathbf{L}^\top\mathbf{x}\|_2^2 > 0
# ```

# """

# ╔═╡ bb58e7e4-6e7f-4966-af6b-e51cb405c470
begin
	function expdiag(array)
		buf = Zygote.Buffer(array)
		buf[:, :] = tril(array)
	    for i in 1:size(array)[1]
	        buf[i, i] = exp(array[i, i])
	    end
	    return copy(buf)
	end
	
	
	function unpackΣ(L)
		LL = expdiag(L)
		LL * LL'
	end
end

# ╔═╡ e954e6e8-e593-4cc9-9318-2f88ee8ae261
function gda_negative_loglik(data, ys, ϕ, μs, Ls; LDA=false)
	logπ = logsoftmax(ϕ)
	nll = - mean(logπ[ys])
	K = length(ϕ)
	if LDA
		Σ = unpackΣ(Ls)
	end
	for k in 1:K
		μ = μs[:, k]
		if LDA == false 
			Σ = unpackΣ(Ls[:, :, k])
		end
		mvn = MvNormal(μ, Σ)
		nk = sum(ys .== k)
		if nk > 0
			nll -= loglikelihood(mvn, data[ys .== k, :]')/nk
		end
		# Σ
	end
	return nll
end

# ╔═╡ e8d5b5f2-8e69-4279-8ea7-c6f5d78ac828
# let

# 	ss = ones(2, 2)
# 	L = tril(ss)
# 	diagidx = diagind(L)
# 	L[diagidx] .= exp.(L[diagidx])
# 	Σ = L * L'
# end

# ╔═╡ 634bd495-a611-4ee8-9d3c-4e17c536ac6d
# let
# 	LL = rand(2,2)
# 	buf = Zygote.Buffer(LL)
# 	buf[:, :] = tril(LL)

	
# 	buf, LL
# end

# ╔═╡ 71029d31-4689-44af-aa8c-d0a154edd8c4
truemvns₂, trueπs₂

# ╔═╡ 02002d81-0b36-4b55-87b7-5778f6942014
md"""
### Gradient descent based learning of GDA
"""

# ╔═╡ ca675aba-8031-47d7-a2b2-2aa8d37cda00
let
	data₂, truezs₂
	ϕ = randn(3)
	μs = randn(2, 3)
	Ls = zeros(2, 2, 3)
	[Ls[:,:,k] .= tril(Ls[:,:,k]) for k in 1:3]
	# Ls

	γ = 0.5
	losses = []
	for _ in 1:50
		nll, g = Zygote.withgradient(ϕ, μs, Ls) do s, m, L 
			gda_negative_loglik(data₂, truezs₂, s, m, L)
		end
	
		ϕ .-= γ * g[1]
		μs .-= γ * g[2]
		Ls .-= γ * g[3]
		push!(losses, nll)
	end

	plot(losses), softmax(ϕ), μs, [unpackΣ(Ls[:,:,k]) for k in 1:3 ]
	# g[3]
end

# ╔═╡ 70d3f9fa-c6d4-4cf8-927b-540f0b2fd00e
let
	data₂, truezs₂
	ϕ = randn(3)
	μs = randn(2, 3)
	Ls = tril(zeros(2, 2))
	# [Ls[:,:,k] .= tril(Ls[:,:,k]) for k in 1:3]
	# Ls

	γ = 0.3
	losses = []
	for _ in 1:50
		nll, g = Zygote.withgradient(ϕ, μs, Ls) do s, m, L 
			gda_negative_loglik(data₂, truezs₂, s, m, L; LDA=true)
		end
	
		ϕ .-= γ * g[1]
		μs .-= γ * g[2]
		Ls .-= γ * g[3]
		push!(losses, nll)
	end

	plot(losses), softmax(ϕ), μs, unpackΣ(Ls)
	# g[3]
end

# ╔═╡ a3ef0d5d-1ca4-47b5-a54f-501431461a5f
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

# ╔═╡ a5427cf3-440d-4684-ac5a-cba131398b80
let
	gr()
	Sigma = [2 0; 0 1]

	μ = [3, 3]
	mvn = MvNormal(μ, Sigma)
	plt1 = plot(range(0-6, 0+6, 50),range(0-6, 0+6, 50), (x, y) -> pdf(MvNormal(zeros(2), Sigma), [x, y]), levels=8, ratio=1, framestyle=:origin ,c=:coolwarm, st=:surface, colorbar=false, alpha=0.6, xlim =(-6, 6), ylim =(-6, +6), title=L"{\mu}=\mathbf{0}")

	# quiver!([0], [0], 	quiver=([μ[1]], [μ[2]]), c=:black, lw=1.5)
	# annotate!([μ[1]*2/3], [μ[2]*2/3], text(L"{\mu}", :top))
	plt2 = plot(range(0-6, 0+6, 50),range(0-6, 0+6, 50), (x, y) -> pdf(mvn, [x, y]), levels=8, ratio=1, framestyle=:origin ,c=:coolwarm, st=:surface, colorbar=false, alpha=0.6,  xlim =(-6, 6), ylim =(0-6, 6), title=L"{\mu}=[3, 3]^\top")

	arrow3d!([0], [0], [0],	[μ[1]], [μ[2]], [0], lw=1.5)
	# annotate!([μ[1]*2/3], [μ[2]*2/3], text(L"{\mu}", :top))
	plot(plt1, plt2, size=(700,300), xlabel=L"x_1", ylabel=L"x_2", zlabel=L"p(\mathbf{x})")
end

# ╔═╡ 6ec93f8b-3e26-4a30-ad40-4f922911487d
TwoColumn(md"""


#### What does `positive definite` imply here?

$$\Large(\mathbf{x} -\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} -\boldsymbol{\mu}) >0$$ 
  * #####  the  quadratic bowl faces up


  * ##### $$(\mathbf{x} -\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} -\boldsymbol{\mu})$$ is a valid distance metric btw ``\mathbf{x}`` and ``\boldsymbol{\mu}`` (since distance can't be negative!)
""", let
	# cgrad(mycmap, 3, categorical=true, rev=true)

	μ = [2,2]
	plt = plot(range(μ[1]-5, μ[1]+5, 50), range(μ[2]-5, μ[2]+5, 50), (x, y) -> norm([x, y] -μ)^2+10, st=:surface, zlim =(-1, 60), c = cgrad(:jet, rev=false), ratio=1, xlim =(μ[1]-5, μ[1]+5), ylim =(μ[2]-5, μ[2]+5), colorbar=false, framestyle=:zerolines, size=(330,300))

	# cs = cgrad(:jet, range())
	zs = cgrad(:jet)[range(0.1, 0.8, 5)]
	for (i,r) in enumerate(range(1, 5, 5))
		ts = range(0, 2π, 50)
		xs = μ[1] .+ cos.(ts) * r
		ys = μ[2] .+ sin.(ts) * r
		path3d!(xs, ys, zeros(length(xs)), c= zs[i], label="")
	end


	xx = [1, 100]
	xx = (xx ./ norm(xx)) * 3
	xx = xx + μ
	arrow3d!([μ[1]], [μ[2]], [0], [xx[1] -μ[1]], [xx[2]-μ[2]], [0]; as=0.1, lc=1, la=1, lw=2, scale=:identity)

	annotate!([xx[1]+2], [xx[2]-0.2], [0], text(L"x-\mu", :blue, :top))
	
	scatter3d!([μ[1]], [μ[2]], [0], label=L"\mu", m=:circle, c=:red)
	plt
	# cgrad(:jet, 30, rev=false)|>length
end)

# ╔═╡ 2f8e92fc-3f3f-417f-9171-c2c755d5e0f0
begin
	μ_ml, Σ_ml = zeros(2,K₂), zeros(2,2,K₂)
	for k in 1:K₂
		data_in_ck = data₂[truezs₂ .==k,:]
		μ_ml[:,k] = mean(data_in_ck, dims=1)
		Σ_ml[:,:, k] = cov(data_in_ck)
	end
end

# ╔═╡ 58663741-fa05-4804-8734-8ccb1fa90b2d
Σsₘₗ = Σ_ml;

# ╔═╡ 5d28e09c-891d-44c0-98a4-ef4cf3a235f1
μsₘₗ = μ_ml;

# ╔═╡ dafd1a68-715b-4f06-a4f2-287c123761f8
begin
	function sampleMixGaussian(n, mvns, πs)
		d = size(mvns[1].Σ)[1]
		samples = zeros(n, d)
		# sample from the multinoulli distribution of cⁱ
		cs = rand(Categorical(πs), n)
		for i in 1:n
			samples[i,:] = rand(mvns[cs[i]])
		end
		return samples, cs
	end
end;

# ╔═╡ e0cfcb9b-794b-4731-abf7-5435f67ced42
begin
	K₃ = 3
	trueπs₃ = [0.2, 0.6, 0.2]
	trueμs₃ = [[3.5, 0] [0.0, 0] [-3.5, 0]]
	trueΣs₃ = zeros(2,2,K₃)
	trueΣs₃[:, :, 1] = [2 -1.5; -1.5 3]
	trueΣs₃[:, :, 3] = [2 1.5; 1.5 3]
	trueΣs₃[:,:,2] = [2 0; 0 3.]
	truemvns₃ = [MvNormal(trueμs₃[:,k], trueΣs₃[:,:,k]) for k in 1:K₃]
	n₃ = 300* K₃
	data₃, truezs₃ = sampleMixGaussian(n₃, truemvns₃, trueπs₃)
	data₃test, truezs₃test = sampleMixGaussian(100, truemvns₃, trueπs₃)
	xs₃ = (minimum(data₃[:,1])-1):0.1: (maximum(data₃[:,1])+1)
	ys₃ = (minimum(data₃[:,2])-1):0.1: (maximum(data₃[:,2])+1)
end;

# ╔═╡ 6d750ce4-26b8-4af4-916d-921e6bd9b16c
data₄, truezs₄, trueπs₄, trueμs₄, trueΣs₄, truemvns₄ = let
	K₄ = 2
	trueπs₄ = [0.5, 0.5]
	trueμs₄ = [[-.5, -1] [0., 0.]]
	trueΣs₄ = zeros(2,2,K₄)
	trueΣs₄[:, :, 1] = [3. 0.5; 0.5 1]
	# trueΣs₃[:, :, 3] = [2 1.5; 1.5 3]
	trueΣs₄[:,:,2] = [1 -1.5; -1.5 3.]
	truemvns₄ = [MvNormal(trueμs₄[:,k], trueΣs₄[:,:,k]) for k in 1:K₄]
	# n = 200 * K₃
	data₄, truezs₄ = sampleMixGaussian(300 * K₄, truemvns₄, trueπs₄)
	# data₃test, truezs₃test = sampleMixGaussian(100, truemvns₄, trueπs₄)
	# xs₃ = (minimum(data₃[:,1])-1):0.1: (maximum(data₃[:,1])+1)
	# ys₃ = (minimum(data₃[:,2])-1):0.1: (maximum(data₃[:,2])+1)
	data₄, truezs₄, trueπs₄, trueμs₄, trueΣs₄, truemvns₄
end;

# ╔═╡ 93b4939f-3406-4e4f-9e31-cc25c23b0284
begin
	Xs = dropmissing(df)[:, 3:6] |> Matrix
	Ys = dropmissing(df)[:, 1]
	Ys_onehot  = Flux.onehotbatch(Ys, unique(Ys))
end;

# ╔═╡ 620789b7-59bc-4e17-bcfb-728a329eed0f
qdform(x, S) = dot(x, S, x);

# ╔═╡ d66e373d-8443-4810-9332-305d9781a21a
md"""

Functions used to plot and produce the gifs

"""

# ╔═╡ acfb80f0-f4d0-4870-b401-6e26c1c99e45
function plot_clusters(D, zs, K, loss=nothing, iter=nothing, name ="class ")
	title_string = ""
	if !isnothing(iter)
		title_string ="Iteration: "*string(iter)*";"
	end
	if !isnothing(loss)
		title_string *= " L = "*string(round(loss; digits=2))
	end
	plt = plot(title=title_string, ratio=1)
	for k in 1:K
		scatter!(D[zs .==k,1], D[zs .==k, 2], label=name*string(k), ms=3, alpha=0.5)
	end
	return plt
end;

# ╔═╡ e091ce93-9526-4c7f-9f14-7634419bfe57
# plot clustering results: scatter plot + Gaussian contours
function plot_clustering_rst(data, K, zs, mvns, πs= 1/K .* ones(K); title="")
	xs = (minimum(data[:,1])-0.5):0.1: (maximum(data[:,1])+0.5)
	ys = (minimum(data[:,2])-0.5):0.1: (maximum(data[:,2])+0.5)
	_, dim = size(data)
	# if center parameters are given rather than an array of MvNormals
	if mvns isa Matrix{Float64}
		mvns = [MvNormal(mvns[:,k], Matrix(1.0I, dim, dim)) for k in 1:K]
		πs = 1/K .* ones(K)
	end
	if ndims(zs) >1
		zs = [c[2] for c in findmax(zs, dims=2)[2]][:]
	end
	p = plot_clusters(data, zs, K)
	for k in 1:K 
		plot!(xs, ys, (x,y)-> pdf(mvns[k], [x,y]), levels=5,  st=:contour, colorbar = false, ratio=1, color=:jet, linewidth=2) 
		scatter!([mvns[k].μ[1]], [mvns[k].μ[2]], color = k, label = "", markersize = 5, markershape=:star4, markerstrokewidth=3, alpha=0.5,  framestyle=:origin)
	end
	title!(p, title)
	return p
end;

# ╔═╡ 9d2e3f26-253e-4bba-b70f-fc3b5c4617d8
begin
	gr()
	plt2_= plot_clustering_rst(data₂,  K₂, truezs₂, truemvns₂, trueπs₂)
	title!(plt2_, "QDA example dataset")
	plt2_
end;

# ╔═╡ 0434dd27-4349-4731-80d5-b71ab99b53e2
begin
	gr()
	plt₃_= plot_clustering_rst(data₃,  K₃, truezs₃, truemvns₃, trueπs₃)
	title!(plt₃_, "More QDA example")
end;

# ╔═╡ 889093e8-5e14-4211-8807-113adbac9a46
let
	gr()

	pltqda =  plot_clustering_rst(data₂,  K₂, truezs₂, truemvns₂, trueπs₂)
	# title!(plt2_, "QDA example dataset")
	# mvnsₘₗ = [MvNormal(μsₘₗ[:,k], Σsₘₗ[:,:,k]) for k in 1:K₂]

	μs, Σs, πs = QDA_fit(data₂, truezs₂)

	mvnsₘₗ = [MvNormal(μs[:, k], Symmetric(Σs[:,:, k])) for k in 1:size(μs)[2]]
	# pltqdaₘₗ = plot(title="QDA MLE params", ratio=1)
	# for k in 1:K₂
	# 	scatter!(data₂[truezs₂ .==k,1], data₂[truezs₂ .==k, 2], label="", c= k )
	# 	scatter!([mvnsₘₗ[k].μ[1]], [mvnsₘₗ[k].μ[2]], color = k, label = "", markersize = 10, markershape=:diamond, markerstrokewidth=3)
	# 	contour!(-5:.05:5, -5:0.05:5, (x,y)-> pdf(mvnsₘₗ[k], [x,y]), levels=5, colorbar = false, ratio=1,lw=3, c=:jet) 
	# end
	pltqdaₘₗ = plot_clustering_rst(data₂,  K₂, truezs₂, mvnsₘₗ, πs)
	title!(pltqda, "QDA Truth")
	title!(pltqdaₘₗ, "QDA MLE")
	plot(pltqda, pltqdaₘₗ, layout=(1,2))
end

# ╔═╡ d44526f4-3051-47ee-8b63-f5e694c2e609
function e_step(data, mvns, πs)
	K = length(mvns)
	# logLiks: a n by K matrix of P(dᵢ|μₖ, Σₖ)
	logLiks = hcat([logpdf(mvns[k], data') for k in 1:K]...)
	# broadcast log(P(zᵢ=k)) to each row 
	logPost = log.(πs') .+ logLiks
	# apply log∑exp to each row to find the log of the normalising constant of p(zᵢ|…)
	logsums = logsumexp(logPost, dims=2)
	# normalise in log space then transform back to find the responsibility matrix
	ws = exp.(logPost .- logsums)
	# return the responsibility matrix and the log-likelihood
	return ws, sum(logsums)
end;

# ╔═╡ 83a5408d-3747-4015-90bc-30e91ef5d7a7
if add_decision_boundary
	md"``p(y^{(i)}|\mathbf{x}^{(i)})=``$(latexify_md(round.(e_step(Xs_peng[text_idx:text_idx, 1:2], pen_mvns, pen_πs)[1][:]; digits=2)))"
else 
	md""
end

# ╔═╡ 30845782-cdd0-4c2e-b237-f331ee28db99
plt2_pen_qda = let
	plt2_pen_qda = @df df scatter(:bill_length_mm, :bill_depth_mm, group =:species, framestyle=:origins,  alpha=0.5,xlabel="bill length (mm)", ylabel="bill depth (mm)");
	for k in 1:3
		plot!(plt2_pen_qda, 32:0.1:60, 13:0.1:22, (x,y)-> pdf(pen_mvns[k], [x,y]), levels=10,  st=:contour, colorbar = false, alpha=0.6, color=k, linewidth=3) 
	end
	if add_decision_boundary
		plot!(32:.5:60, 13:.5:22, (x, y) -> e_step([x, y]', pen_mvns, pen_πs)[1][:] |> argmax, lw=1.0,  st=:contourf, c=1:3, alpha=0.05, colorbar=false)
		idx = text_idx
		scatter!(Xs_peng[idx:idx, 1], Xs_peng[idx:idx, 2], m=:x, mc=:black, label="", markersize=8, markerstrokewidth=3)
	end
	plt2_pen_qda
end;

# ╔═╡ 8dbda402-2130-41e4-a8b2-74e8a535364f
TwoColumn(
	if add_decision_boundary
		show_img("gda_classify.svg", h=280)
	else
		show_img("/CS5914/gda_bnalone.svg", h=260)
	end
,  
begin
	if add_decision_boundary
		plot(plt2_pen_qda, size=(330,330))
	else
		show_img("qda_penguine1.svg", w=330)
	end
end)

# ╔═╡ 901004da-2ac9-45fd-9643-8ce1cc819aa8
let
	Random.seed!(123)
	K₁ =3
	n₁ = 600

	πs = 1/K₁ * ones(K₁)

	truezs₁ = repeat(1:K₁; inner=200)
	trueμs₁ = zeros(2, K₁)
	trueμs₁[:,1] = [2.0, 2.0]
	trueμs₁[:,2] = [-2.0, 2]
	trueμs₁[:,3] = [0., -1.5]
	LL = cholesky([1 0; 0. 1]).L
	data₁ = trueμs₁[:,1]' .+ randn(200, 2) * LL
	data₁ = vcat(data₁, trueμs₁[:,2]' .+ randn(200, 2)* LL)
	data₁ = vcat(data₁, trueμs₁[:,3]' .+ randn(200, 2)* LL)
	plt₁ = plot(ratio=1, framestyle=:origin)
	mvns = [MvNormal(trueμs₁[:, k], Matrix(I,2,2)) for k in 1:K₁]
	xs_qda = minimum(data₁[:,1])-0.1:0.1:maximum(data₁[:,1])+0.1
	ys_qda = minimum(data₁[:,2])-0.1:0.1:maximum(data₁[:,2])+0.1
	
	for k in 1:K₁
		scatter!(data₁[truezs₁ .== k,1], data₁[truezs₁ .==k,2], c=k, label="Class"*string(k), alpha=0.5) 
		contour!(-5:.02:5, -5:0.02:5,  (x,y)-> mdistance([x,y], mvns[k]), levels=[.5, 1.0, 2.0], colorbar = false, lw=2, alpha=0.7, c=cgrad(:jet, 5; categorical=true, rev=true)[0.3:0.1:3])
	end

	if add_db_1
		plot!(-5:.02:5, -5:0.02:5, (x,y) -> e_step([x, y]', mvns, πs)[1][:] |> argmax,  c=:black, lw=2, alpha=1.0, title="Decision boundary by LDA", st=:contour, colorbar=false, ratio=1)
	end
	plt₁
end

# ╔═╡ 3ea57a8e-8d15-4f41-acfb-e3bd1d65e585
begin
	K₁ =3
	n₁ = 600
	# D₁ = zeros(n₁, 2)
	# 200 per cluster
	truezs₁ = repeat(1:K₁; inner=200)
	trueμs₁ = zeros(2, K₁)
	trueμs₁[:,1] = [0.0, 2.0]
	trueμs₁[:,2] = [2.0, 1]
	trueμs₁[:,3] = [0., -1.5]
	LL = cholesky([1 0.8; 0.8 1]).L
	Random.seed!(123)
	data₁ = trueμs₁[:,1]' .+ randn(200, 2)*LL'
	data₁ = vcat(data₁, trueμs₁[:,2]' .+ randn(200, 2)* LL')
	data₁ = vcat(data₁, trueμs₁[:,3]' .+ randn(200, 2)* LL')
	plt₁ = plot(ratio=1, framestyle=:origin)
	truemvns₁ = [MvNormal(trueμs₁[:, k], LL * LL') for k in 1:K₁]
	xs_qda = minimum(data₁[:,1])-0.2:0.05:maximum(data₁[:,1])+0.2
	ys_qda = minimum(data₁[:,2])-1:0.05:maximum(data₁[:,2])+0.5
	for k in 1:K₁
		scatter!(data₁[truezs₁ .== k,1], data₁[truezs₁ .==k,2], c=k, ms=3, alpha=0.5, label="Class "*string(k)) 
		contour!(xs_qda, ys_qda, (x,y)-> mdistance([x,y], truemvns₁[k]), levels=[0.5, 1.0, 1.5], colorbar = false, lw=2, alpha=0.9,  c=cgrad(:jet, 5; categorical=true, rev=true)[0.3:0.1:7])
	end
	title!(plt₁, "Classification (dataset with labels)")
	if add_db_2
		πs = 1/K₁ * ones(K₁)
		plot!(xs_qda, ys_qda, (x,y) -> e_step([x, y]', truemvns₁, πs)[1][:] |> argmax, c=:black, lw=2, alpha=0.7, title="Decision boundary by LDA", st=:contour, colorbar=false)
	end
	plt₁
end

# ╔═╡ 03fb6688-dcc4-4ae4-ac6c-77000cbf90c3
begin
	## batch qda classify XX: N * d matrix
	function qda_classify(XX, π, μs, Σs)
		C = length(π)
		# logLiks: a N by C matrix of ln P(xᵢ|μc, Σc)
		logLiks = hcat([logpdf(MvNormal(μs[:,c], Σs[:,:,c]), XX') for c in 1:C]...)
		# broadcast ln(π) to each row (each observation)
		logPost = log.(πs') .+ logLiks
		# apply argmax to each row/xi
		return argmax(eachrow(logPost))
	end

end

# ╔═╡ d319d77e-fdf1-4971-a4e0-23bd0ad573cd
plt2_pen_lda = let
	plt2_pen_lda = @df df scatter(:bill_length_mm, :bill_depth_mm, group =:species, framestyle=:origins,  alpha=0.5,xlabel="bill length (mm)", ylabel="bill depth (mm)");
	μs, Σs, πs = LDA_fit(Xs_peng, ys_peng)
	mvns = [MvNormal(μs[:, i], Σs) for i in 1:3]
	# for k in 1:3
	# 	plot!(plt2_pen_lda, 32:0.1:60, 13:0.1:22, (x,y)-> pdf(mvns[k], [x,y]), levels=4, st=:contour, colorbar = false, alpha=0.6, color=k, linewidth=3) 
	# end
	# if add_decision_boundary
	plot!(32:.1:60, 13:.1:22, (x, y) -> e_step([x, y]', mvns, πs)[1][:] |> argmax, lw=1.0,  st=:heatmap, c=1:3, alpha=0.5, colorbar=false, title="LDA classifier", size=(280,280))

	plt2_pen_lda
end;

# ╔═╡ 760dfb53-fc7e-4314-998f-cd627be4b8e6
plot(plt_sfm, plt2_pen_lda, size=(700,350))

# ╔═╡ 665c5bbc-9a5c-4e4a-930d-725bc2c9c883
let
	gr()
	data = data₃
	K = K₃
	zs = truezs₃
	# μs, Σs, πs = QDA_fit(data, zs)
	μs, Σs, πs = trueμs₃, trueΣs₃, trueπs₃
	mvns = [MvNormal(μs[:, k], Symmetric(Σs[:,:, k])) for k in 1:size(μs)[2]]
	# mvns = []
	plt = plot(-6.5:.05:6.5, -6.5:0.05:6.5, (x,y) -> e_step([x, y]', mvns, πs)[1][class_d3], levels=10, c=:coolwarm, lw=1, title="Class-wise boundary: "*L"p(y=%$(class_d3)|\mathbf{x})", st=:contourf, xlim =(-6.5,6.5), ylim =(-6.5,6.5),colorbar=true, ratio=1, alpha=0.5, framestyle=:origin)
	for k in [1,3,2]
		# if k ==1
		# 	scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="")
		# else
			scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, ms=3, alpha=0.8, label="")
		# end
	end
	plt
end

# ╔═╡ 5edefee7-9c68-4f4a-84c3-4a98207f890c
let
	gr()

	μs, Σs, πs = QDA_fit(data₄, truezs₄)
	mvns_qda = [MvNormal(μs[:, i], Σs[:,:,i]) for i in 1:2]

	plt = plot_clustering_rst(data₄,  2, truezs₄, mvns_qda, πs, title="QDA fit")

	xs = -6:0.1:6
	ys = -6:0.1:6
		# if add_db_2
	πs = 1/2 * ones(2)

	if false
		plot!(xs, ys, (x,y) -> e_step([x, y]', truemvns₄, trueπs₄)[1][1], c=cgrad(:coolwarm; rev=true, scale=:log2), lw=2, alpha=0.4, title="QDA boundary", st=:contourf, fillalpha=0.1, colorbar=false, xlim =(-6, 6), ylim =(-6, 6), levels =0:0.1:1.0, framestyle=:origin)
	
		for k in 1:2
			plot!(xs, ys, (x,y)-> pdf(mvns_qda[k], [x,y]), levels=25,  st=:contour, colorbar = false,  color=:jet, linewidth=2) 
			scatter!([mvns_qda[k].μ[1]], [mvns_qda[k].μ[2]], color = k, label = "", markersize = 5, markershape=:star4, markerstrokewidth=3, alpha=0.5,  framestyle=:origin)
		end

	end

	# plot!(plt, )


	# plt2 = plot_clustering_rst(data₄,  2, truezs₄, truemvns₄, trueπs₄)

	# xs = (minimum(data₄[:,1])):0.1: (maximum(data₄[:,1]))
	# ys = (minimum(data₄[:,2])):0.1: (maximum(data₄[:,2]))
	
		# if add_db_2
	μs, Σs, πs = LDA_fit(data₄, truezs₄)
	mvns = [MvNormal(μs[:, i], Σs) for i in 1:2]
	plt2 = plot_clustering_rst(data₄,  2, truezs₄, mvns, πs, title="LDA fit")

	if false
		plot!(plt2, xs, ys, (x,y) -> e_step([x, y]', mvns, πs)[1][1] ,c = cgrad(:coolwarm; rev=true, scale=:log2), lw=2, alpha=.5, title="LDA boundary", levels = 0:0.1:1.0,st=:contourf, fillalpha=0.1, colorbar=false, xlim =(-6, 6), ylim =(-6, 6))
	
		for k in 1:2
			plot!(xs, ys, (x,y)-> pdf(mvns[k], [x,y]), levels=30,  st=:contour, colorbar = false,  color=:jet, linewidth=2) 
			scatter!([mvns[k].μ[1]], [mvns[k].μ[2]], color = k, label = "", markersize = 5, markershape=:star4, markerstrokewidth=3, alpha=0.5,  framestyle=:origin)
		end

	end
	plot(plt, plt2, legend=false, ylim =(-5.5, 6), size=(700,350))
end

# ╔═╡ e95dd200-9eb3-44a9-b810-c20b50812a90
let
	plotly()

	μs, Σs, πs = QDA_fit(data₄, truezs₄)
	xs = -6:0.1:6
	ys = -6:0.1:6
		# if add_db_2
	πs = 1/2 * ones(2)
	plt= plot(xs, ys, (x,y) -> (e_step([x, y]', truemvns₄, trueπs₄)[1][:]|> argmax) -1, c=cgrad(:coolwarm; rev=false, scale=:log2), lw=2, alpha=0.9, title="QDA boundary", st=:surface,  colorbar=false, xlim =(-6, 6), ylim =(-6, 6), zlim =(-0.1,1.1),  framestyle=:zerolines)
	plot!(data₄[truezs₄ .== 2, 1], data₄[truezs₄ .== 2, 2], ones(sum(truezs₄ .== 2)), st=:scatter, ms=2, alpha=0.5, c=2)
	plot!(data₄[truezs₄ .== 1, 1], data₄[truezs₄ .== 1, 2], zeros(sum(truezs₄ .== 1)), st=:scatter, ms=2, alpha=0.5, c=1)


	μs, Σs, πs = LDA_fit(data₄, truezs₄)
	mvns = [MvNormal(μs[:, i], Σs) for i in 1:2]

	# # if add_fits_
	plt2 = plot(xs, ys, (x,y) -> (e_step([x, y]', mvns, πs)[1][:]|>argmax)-1 ,c = cgrad(:coolwarm; rev=false, scale=:log2), lw=2, alpha=.9, title="LDA boundary", st=:surface,  colorbar=false, xlim =(-6, 6), ylim =(-6, 6),  zlim =(-0.1,1.1), framestyle=:zerolines)
	
	plot!(data₄[truezs₄ .== 2, 1], data₄[truezs₄ .== 2, 2], ones(sum(truezs₄ .== 2)), st=:scatter, ms=2, alpha=0.5, c=2)
	plot!(data₄[truezs₄ .== 1, 1], data₄[truezs₄ .== 1, 2], zeros(sum(truezs₄ .== 1)), st=:scatter, ms=2, alpha=0.5, c=1)

	# # end
	plot(plt, plt2, legend=false, size=(700,350))
end

# ╔═╡ e6947cf6-7982-4280-98cb-82053b70372f
plt₁_new= let
	gr()
	K₁ =3
	n₁ = 600
	# D₁ = zeros(n₁, 2)
	# 200 per cluster
	truezs₁ = repeat(1:K₁; inner=200)
	trueμs₁ = zeros(2, K₁)
	trueμs₁[:,1] = [0.0, 2.0]
	trueμs₁[:,2] = [2.0, 1]
	trueμs₁[:,3] = [0., -1.5]
	LL = cholesky([1 0.8; 0.8 1]).L
	Random.seed!(123)
	data₁ = trueμs₁[:,1]' .+ randn(200, 2)*LL'
	data₁ = vcat(data₁, trueμs₁[:,2]' .+ randn(200, 2)* LL')
	data₁ = vcat(data₁, trueμs₁[:,3]' .+ randn(200, 2)* LL')
	plt₁ = plot(ratio=1, framestyle=:origin)
	truemvns₁ = [MvNormal(trueμs₁[:, k], LL * LL') for k in 1:K₁]
	xs_qda = minimum(data₁[:,1])-0.2:0.05:maximum(data₁[:,1])+0.2
	ys_qda = minimum(data₁[:,2])-1:0.05:maximum(data₁[:,2])+0.5
	for k in 1:K₁
		scatter!(data₁[truezs₁ .== k,1], data₁[truezs₁ .==k,2], c=k, ms=3, alpha=0.5, label="Class "*string(k)) 
		contour!(xs_qda, ys_qda, (x,y)-> pdf(truemvns₁[k], [x,y]), levels=5, colorbar = false, lw=2, alpha=0.9, c=:jet) 
	end
	title!(plt₁, "Classification (dataset with labels)")
	if false
		πs = 1/K₁ * ones(K₁)
		plot!(xs_qda, ys_qda, (x,y) -> (e_step([x, y]', truemvns₁, πs)[1][:] |> argmax) /3, c=:black, lw=2, alpha=0.7, title="Decision boundary by LDA", st=:contour, colorbar=false)
	end
	plt₁
end;

# ╔═╡ 18a5208c-4a01-4784-a69d-0bd5e3bb9faf
md"""

## Summary 




| Model | Assumption | Example|
| :---|  :---:| :---:|
| LDA |  $\mathbf{\Sigma} =\mathbf\Sigma_1=\ldots \mathbf\Sigma_C$| $(plot(plt₁_new, size=(220,220), title="", legend=false))|
| Naive Bayes | Diagonal ``\mathbf{\Sigma}_c = \text{diag}(\boldsymbol{\sigma}_c)``|$(plot(plt2_, size=(220,220), title="", legend=false))|
|  QDA  | $\mathbf{\Sigma}_1 \neq \mathbf\Sigma_2\neq\ldots \mathbf\Sigma_C$ |$(plot(plt₃_, size=(220,220), title="", legend=false))|
"""

# ╔═╡ 7b47cda6-d772-468c-a8f3-75e3d77369d8
begin
	# decision boundary function of input [x,y] 
	function decisionBdry(x, y, mvns, πs)
		z, _ = e_step([x,y]', mvns, πs)
		findmax(z[:])
	end

end;

# ╔═╡ eae40715-9b33-494f-8814-8c6f967aeade
plt_d3_true_bd=let
	gr()
	data = data₃
	K = K₃
	truezs = truezs₃

	# μs, Σs, πs = QDA_fit(data, truezs)
	μs, Σs, πs = trueμs₃, trueΣs₃, trueπs₃
	mvns = [MvNormal(μs[:, k], Symmetric(Σs[:,:, k])) for k in 1:size(μs)[2]]
	# mvns = []
	plt=plot(-7:.05:7, -7:0.05:7, (x,y) -> decisionBdry(x, y, mvns, πs)[2], c=1:3, alpha=0.6, title="Decision boundary by supervised learning QDA", st=:heatmap, colorbar=false, ratio=1, framestyle=:origin)
	for k in [1,3,2]
		if k ==1
			scatter!(data[truezs .==k, 1], data[truezs .==k, 2], c=k, label="")
		else
			scatter!(data[truezs .==k, 1], data[truezs .==k, 2], c=k, label="")
		end
	end
	plt
end;

# ╔═╡ 3393e09c-a474-47da-a21f-0d8ec9017979
let
	gr()
	plt = plot_clustering_rst(data₃,  K₃, truezs₃, truemvns₃, trueπs₃)
	title!(plt2_, "QDA example dataset")

	if add_db_qda

		plt = plot(plt_d3_true_bd)
	end
	# plt2_
	plt
end

# ╔═╡ 27755688-f647-48e5-a939-bb0fa70c95d8
function m_step(data, ws)
	_, d = size(data)
	K = size(ws)[2]
	ns = sum(ws, dims=1)
	πs = ns ./ sum(ns)
	# weighted sums ∑ wᵢₖ xᵢ where wᵢₖ = P(zᵢ=k|\cdots)
	ss = data' * ws
	# the weighted ML for μₖ = ∑ wᵢₖ xᵢ/ ∑ wᵢₖ
	μs = ss ./ ns
	Σs = zeros(d, d, K)
	for k in 1:K
		error = (data .- μs[:,k]')
		# weighted sum of squared error
		# use Symmetric to remove floating number numerical error
		Σs[:,:,k] =  Symmetric((error' * (ws[:,k] .* error))/ns[k])
	end
	# this is optional: you can just return μs and Σs
	mvns = [MvNormal(μs[:,k], Σs[:,:,k]) for k in 1:K]
	return mvns, πs[:]
end;

# ╔═╡ 99ed2b50-2ea4-465a-9d4f-4517e74d1216
function produce_gif(plts, fps=5)
	anim = Animation()

	[frame(anim, plt) for plt in plts]

	gif(anim; fps=fps)
end

# ╔═╡ 6328dc99-9419-4ce0-9c76-ed2cadd8e2f3
let

	# for k in 1:3
	k = 1
	nks = counts(ys_peng, 1:3)
		# plot!(32:0.1:60, 13:0.1:22, (x,y)-> pdf(pen_mvns[k], [x,y]), levels=6,  st=:contour, colorbar = false, alpha=0.5, color=k, linewidth=3) 
	title = ""

	plts = []
	anim = @animate for k in 1:3
		plt1 = @df df_ scatter(:bill_length_mm, :bill_depth_mm, group =:species,alpha=0.1,  framestyle=:origins,  xlabel="bill length (mm)", ylabel="bill depth (g)");
		scatter!(Xs_peng[ys_peng .==k, 1], Xs_peng[ys_peng .==k, 2], c=k, alpha=0.4, label="", title=title)
		push!(plts, plt1)

		plt2 = @df df_ scatter(:bill_length_mm, :bill_depth_mm, group =:species,alpha=0.05,  framestyle=:origins,  xlabel="bill length (mm)", ylabel="bill depth (g)");
		title = title * " "*L"\hat{\mu}_{%$k} = %$(round.(pen_μs[:, k];digits=1));"
		scatter!(Xs_peng[ys_peng .==k, 1], Xs_peng[ys_peng .==k, 2], c=k, alpha=0.3, label="")
		scatter!(plt2, pen_μs[1:1, k], pen_μs[2:2, k], markershape=:x, markerstrokewidth=12, markersize=14,label="", c= k, title=title)
		push!(plts, plt2)
	end

	# gif(anim, fps=0.8)

	produce_gif(plts, 1.0)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Clustering = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
DistributionsAD = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
MLJLinearModels = "6ee0df7b-362f-4a72-a706-9e79364fb692"
PalmerPenguins = "8b842266-38fa-440a-9b57-31493939ab85"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
Clustering = "~0.15.6"
DataFrames = "~1.6.0"
Distributions = "~0.25.107"
DistributionsAD = "~0.6.54"
Flux = "~0.14.0"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.1"
Latexify = "~0.16.1"
LogExpFunctions = "~0.3.26"
MLJLinearModels = "~0.10.0"
PalmerPenguins = "~0.1.4"
Plots = "~1.39.0"
PlutoTeachingTools = "~0.2.14"
PlutoUI = "~0.7.51"
StatsBase = "~0.34.2"
StatsPlots = "~0.15.5"
Zygote = "~0.6.68"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "c4eb460861d0d527dd99bcbb8f4fd0b755c5ad0c"

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
git-tree-sha1 = "c278dfab760520b8bb7e9511b968bf4ba38b7acc"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.3"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cde29ddf7e5726c9fb511f340244ea3481267608"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.2"
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

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables"]
git-tree-sha1 = "e28912ce94077686443433c2800104b061a827ed"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.39"

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
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "679e69c611fff422038e9e21e270c4197d49d918"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.12"

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
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "2118cb2765f8197b08e5958cdd17c165427425ee"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.19.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "407f38961ac11a6e14b2df7095a2577f7cb7cb1b"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.6"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "c0216e792f518b39b22212127d4a84dc31e4e386"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

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

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "886826d76ea9e72b35fcd000e535588f7b60f21d"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

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
git-tree-sha1 = "6e8d74545d34528c30ccd3fa0f3c00f8ed49584c"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.11"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

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

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "Distributions", "FillArrays", "LinearAlgebra", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "060a19f3f879773399a7011676eb273ccc265241"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.54"

    [deps.DistributionsAD.extensions]
    DistributionsADForwardDiffExt = "ForwardDiff"
    DistributionsADLazyArraysExt = "LazyArrays"
    DistributionsADReverseDiffExt = "ReverseDiff"
    DistributionsADTrackerExt = "Tracker"

    [deps.DistributionsAD.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

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
git-tree-sha1 = "ec22cbbcd01cba8f41eecd7d44aac1f23ee985e3"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.2"

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

[[deps.Flux]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "2827339fbc2291d541a9c62ffbf28da7f3621ae4"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.14.9"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxCUDAcuDNNExt = ["CUDA", "cuDNN"]
    FluxMetalExt = "Metal"

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

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
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9a68d75d466ccc1218d0552a8e1631151c569545"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

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
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

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
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

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

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "b435d190ef8369cf4d79cc9dd5fba88ba0165307"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.3"

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

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60b1194df0a3298f460063de985eae7b01bc011a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.1+0"

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
git-tree-sha1 = "653e0824fc9ab55b3beec67a6dbbe514a65fb954"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.15"

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

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9df2ab050ffefe870a09c7b6afdb0cde381703f2"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.11.1"
weakdeps = ["ChainRulesCore", "SparseArrays", "Statistics"]

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

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
git-tree-sha1 = "38756922d32476c8f41f73560b910fc805a5a103"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.4.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "72dc3cf284559eb8f53aa593fe62cb33f83ed0c0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.0.0+0"

[[deps.MLJLinearModels]]
deps = ["DocStringExtensions", "IterativeSolvers", "LinearAlgebra", "LinearMaps", "MLJModelInterface", "Optim", "Parameters"]
git-tree-sha1 = "7f517fd840ca433a8fae673edb31678ff55d969c"
uuid = "6ee0df7b-362f-4a72-a706-9e79364fb692"
version = "0.10.0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "0cd3514d865b928e6a36f03497f65b5b1dee38c1"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.9.4"

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
git-tree-sha1 = "b211c553c199c111d998ecdaf7623d1b89b69f93"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.12"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

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

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "900a11b3a2b02e36b25cb55a80777d4a4670f0f6"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.10"

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

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "5e4029759e8699ec12ebdf8721e51a659443403c"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.4"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

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
git-tree-sha1 = "cc6e1927ac521b659af340e0ca45828a3ffc748f"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.12+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "01f85d9269b13fedc61e63cc72ee2213565f7a72"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.8"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "34205b1204cc83c43cd9cfe53ffbd3b310f6e8c5"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.3.1"

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

[[deps.PalmerPenguins]]
deps = ["CSV", "DataDeps"]
git-tree-sha1 = "e7c581b0e29f7d35f47927d65d4965b413c10d90"
uuid = "8b842266-38fa-440a-9b57-31493939ab85"
version = "0.1.4"

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
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

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
git-tree-sha1 = "862942baf5663da528f66d24996eb6da85218e76"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.0"

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
git-tree-sha1 = "89f57f710cc121a7f32473791af3d6beefc59051"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.14"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "bd7c69c7f7173097e7b5e1be07cee2b8b7447f51"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.54"

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

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "88b895d13d53b5577fd53379d913b9ab9ac82660"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9ebcd48c498668c7fa0e97a9cae873fbee7bfee1"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

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

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

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

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "4e17a790909b17f7bf1496e3aec138cf01b60b3b"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.0"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

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

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "0a3db38e4cce3c54fe7a71f831cd7b6194a54213"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.16"

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

[[deps.TranscodingStreams]]
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
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
git-tree-sha1 = "5f24e158cf4cee437052371455fe361f526da062"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.6"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

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
git-tree-sha1 = "9d749cd449fb448aeca4feee9a2f4186dbb5d184"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.4"

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
# ╟─120a282a-91c1-11ec-346f-25d56e50d38c
# ╟─5aa0adbe-7b7f-49da-9bab-0a78108912fd
# ╟─fc9cffe8-b447-4ea0-bd0b-cdc811305620
# ╟─7b3003cc-797c-48f5-b8e9-6c2aba9f82da
# ╟─e9e06a13-5307-4e62-b797-d2d2e1f8ac70
# ╟─646dd3d8-6092-4435-aee9-01fa6a281bdc
# ╟─093c4c78-6179-4196-8d94-e548621df69b
# ╟─16497eaf-3593-45e0-8e6a-6783198663c3
# ╟─f7e989fd-d955-4323-bdef-57d9ffbe5a18
# ╟─e136edb5-3e98-4355-83e2-55761eb8b15c
# ╟─c76aa603-f295-45f3-b55f-1d2d97b03c59
# ╟─b537d88b-5a40-4369-9208-c865be19125b
# ╟─32aff8ba-a316-4756-9014-d2b194979acc
# ╟─e5730f69-c3a6-4ac3-bead-b5e135efe70d
# ╟─a114ac63-e38c-44c9-ae9b-a3c0c247b90c
# ╟─17f05a30-11a0-4288-b28c-7ee33e271913
# ╟─db248681-0938-4762-80f6-7bf61f914128
# ╟─661e876c-1966-466a-9295-04a2a2e37bca
# ╟─84902be5-49c7-4c05-96f0-4f1bd818e2ad
# ╟─d5bb961f-8f21-4282-a1e8-cd8a892cb33c
# ╟─e7d8fe14-40f6-499a-963c-f4342e84d964
# ╟─1356c175-1e62-4465-85b2-10017667f29c
# ╟─cdd0006b-f45f-4ee1-b17d-82262d304226
# ╟─cc9d9e19-c464-4ff7-ae1f-bba4111075b3
# ╟─c51a2ec7-3270-4dbe-92fd-520d78db2bf7
# ╟─3c2e0b13-f43b-4cfe-b6ce-49a2be0f7519
# ╟─200b1531-2a94-472a-9fd2-b90b04877581
# ╟─a4d6230d-9f6d-4ff3-aa0e-e77bc63873b2
# ╟─25251600-dc00-48de-9509-35117c319e09
# ╟─7dd25028-281f-4130-af09-881ae5015309
# ╟─c4260ef4-d521-4d23-9495-41741940f128
# ╟─9938af3d-0cad-4005-b2f6-db7fcf84b89a
# ╟─dd7db7a6-51c0-4421-8308-2c822107a370
# ╟─f033fbb8-55b3-40a7-a2e1-db8777daffc6
# ╟─d2af46c5-4795-497f-8f86-70b11cb95cb0
# ╟─1a7f5b2f-c245-4c6e-854e-e8ce5ccf3f72
# ╟─6b27d9f7-e74c-404a-8248-c67cbbfe8342
# ╟─f1128d41-541d-49de-b0e3-644769c6e59f
# ╟─b334263b-211d-4188-b6a6-b629de7b8744
# ╟─ce6f3828-65b0-403e-bec0-6faa85899c27
# ╟─a9e078c2-1768-4184-bc13-9aa885c0c645
# ╟─12c87fe2-4e5a-4a65-a06a-b6a59a274c3a
# ╟─ad768230-58cd-466f-b6ba-b8a0ca769277
# ╟─e7d9217d-b25b-43d5-9503-b82e76776d2a
# ╟─3a747ab4-8dc1-48ef-97ae-a736448020b3
# ╟─1f34d131-71e6-4ee6-96f2-124ca31ba531
# ╟─a5e253b5-0869-49fe-8e8e-b7d12df45265
# ╟─1da36dc8-1ec6-4541-b2a0-d47e2cfd39f3
# ╟─63a102df-1e17-488c-9381-3fbf16df0103
# ╟─f3ec5dd2-0192-4a5f-a993-dc84f08ea98f
# ╟─7f3ef771-7db3-43f9-a7c5-aef1448741bc
# ╟─7acf4734-ccdb-4b2b-a94a-2362989b7b26
# ╟─7635232b-2db3-4eaa-a09e-e22e744e8c6c
# ╟─98d5d365-1f48-4e3a-885f-e2331f5c5241
# ╟─15e64dfe-b46d-4bca-889d-e4ff639c89cc
# ╟─16790822-bb9d-4933-92d8-5dbf4dded74e
# ╟─8162c635-81a8-4d60-905f-f73522c14efb
# ╟─76a4705c-7a2d-4916-abb5-fa24e3818ebb
# ╟─65bc9927-4a46-439b-be41-38ae3c6379d0
# ╟─d421195e-abd3-4e01-8380-04b51fd25fe9
# ╟─37bdd1bd-5742-49f3-aa57-5f0c998322fb
# ╟─ba4a6076-bbf6-446c-bb97-41825cfa2659
# ╟─bcdda8a3-c1ab-4e95-b07e-299a506abb91
# ╟─2a111f88-2187-4fdf-acc5-4a3572956d36
# ╟─2581cbd3-5aa4-43ce-bd32-5ad32fff58e2
# ╟─c5f80244-0c5e-4a60-845b-1a46ccc4e7d6
# ╟─6352141c-92dc-4ca7-ae7b-37f5f01b1d32
# ╟─544fd97b-3239-49c0-9814-064feebbc108
# ╟─16dcdf35-0d33-4061-9d5c-722465cc2c4f
# ╟─4590bcf8-9c34-4fec-b4d0-093b45f10801
# ╟─2bbfc4a2-e099-4ad4-9668-5c3b8ede04d4
# ╟─a5427cf3-440d-4684-ac5a-cba131398b80
# ╟─5b238aa9-df84-4dd8-894c-3ad1dcd62c08
# ╟─4cec5c11-1db1-478d-b6e3-c500149ae238
# ╟─6ec93f8b-3e26-4a30-ad40-4f922911487d
# ╟─0a9fee13-536c-4053-8f0b-eabcba1fc62f
# ╟─3009f053-9636-428b-9612-75696d785983
# ╟─11562cf7-afc1-46ec-8b69-29f4bb503546
# ╟─ccb7b38a-2509-48c9-aee1-97b5b6a8d599
# ╟─56abb43a-715f-4aec-8ab2-cde254a8b94c
# ╟─c35f9ae2-0f88-4e34-888e-89399c0b1a89
# ╟─665bc7b6-53b3-4e76-83af-c6eda83faafa
# ╟─c98836a1-6107-4c8e-b53d-98c502010d80
# ╟─cf0c1e8c-3339-4a11-a445-7bc4d8bc81dc
# ╟─6054fdf1-928b-4d1b-8ef8-23da27b9631c
# ╟─9d3bbf42-9f4a-45cc-9e96-421799329900
# ╟─8a68b9cb-c814-4ba6-b701-e5a02e4bfb75
# ╟─b2dee8c3-c98b-4c57-95ca-a41895b5318c
# ╟─db6febd1-9780-4d16-8a17-0304272d7269
# ╟─2f9d23f7-8476-47e1-93f6-71dc36110b03
# ╟─ff04483d-4590-4ee1-817b-d492b897523b
# ╟─b47aef56-fbec-49cd-a33e-6dff4dd1c543
# ╟─71683f0a-9c71-48b5-a760-e8847eee82ce
# ╟─7ba00267-cd92-4aba-90cb-c3357ed0770f
# ╟─cc5c8774-6727-48e9-b8bf-d038343f894d
# ╟─ada51855-a39f-4727-8b9c-b95ff2b179c9
# ╟─53456b95-5f92-4e2a-81ca-17f2a7ef2885
# ╟─4d1ead3c-00b0-44ac-b541-21323d9c03ea
# ╟─09f35be2-3678-401a-a12d-784790397f48
# ╟─3d653c13-42e2-4192-b5ea-afdf850ca5e3
# ╟─1b4fe792-c0c8-4101-b2b8-8213ae573dd3
# ╟─eb074796-52d5-4cbf-8499-3f53206b3d99
# ╟─551e073e-a697-4a3a-9135-2dc6571d7e8c
# ╟─5a539af7-0daa-44c1-8a11-4ba71f840d76
# ╟─dd1411b2-c0de-4755-b377-446acfcbb257
# ╟─5ab978c5-c668-4f7c-9899-16bba89549da
# ╟─c8528433-4b95-44f3-ae66-cf11ea4f3b99
# ╟─2dba181a-05be-4e1a-b004-858573eda215
# ╟─2496221f-70f9-4187-b329-35fbaf03a480
# ╟─9ecd708a-0585-4a68-b222-702e8de02abb
# ╟─133e4cb5-9faf-49b8-96ef-c9fc5c8e0b94
# ╟─c05d44bd-1564-430a-b633-b57cca1f5526
# ╟─fd0a8334-07e6-4678-bf28-d334d81fc67e
# ╟─bbfba1d5-c280-43e1-9721-d0ab2b9226ed
# ╟─cc9d8ae5-9ed3-407c-b63c-e8c4ea1cd472
# ╟─9a0ea270-10d8-44f5-98a1-f6324572548e
# ╟─17c07bab-5d5a-4480-8e60-94ffc4d891ef
# ╟─9e21c264-1175-479f-a0bd-51b21c67ce36
# ╟─e0473be1-1ee0-42fe-95a1-cdd6c948fb35
# ╟─0ca6a1e1-6f91-42fa-84b5-7c3b9170e56a
# ╟─77aaff69-13bb-4ffd-ad63-62993e13f873
# ╟─e5a23ba6-7859-4212-8854-86b238332eef
# ╟─22e1fbc9-f0bd-4159-b92f-11c412a660e6
# ╟─85eeec0d-f69f-4cf9-a718-df255a948433
# ╟─46d30b0c-0cf5-4369-a81f-6c234043a8ea
# ╟─870ff42f-d903-4ccf-a538-43bbf9ec978b
# ╟─67ba35e4-9250-4ac5-b1f4-94bbdab42258
# ╟─83a5408d-3747-4015-90bc-30e91ef5d7a7
# ╟─c75c6c55-ef2d-410d-9ff2-6647b228dc29
# ╟─8dbda402-2130-41e4-a8b2-74e8a535364f
# ╟─30845782-cdd0-4c2e-b237-f331ee28db99
# ╟─76c05a93-698c-4ee1-9411-8a518c4e56b0
# ╟─21803313-fa45-48ea-bd3f-a30bc7696215
# ╟─2acf2c33-bd3b-4369-857a-714d0d1fc600
# ╟─6775b1df-f9dd-423e-a6ef-5c9b345e5f0f
# ╟─abd46485-03c9-4077-8ace-34f9b897bb04
# ╟─dd915c43-b3cc-4608-87b0-852b2d533b15
# ╟─0bab5178-5cbd-467e-959d-1f09b496d2af
# ╟─b10c879d-98c8-4617-91de-fba1d9499ba4
# ╟─57c4e40c-6760-4c88-896d-7ea8faf134e0
# ╟─1a6cb45d-1b26-4efa-bd40-f7a8e3bbd770
# ╟─c85b688c-fc8d-4dfa-98bd-9e43dd0b79d5
# ╟─ee10a243-052f-4c0f-8f0d-e16ad6ceb611
# ╟─df0719cb-fc54-456d-aac7-48237a96cbdd
# ╟─2ad600a2-4e5d-4af6-a18c-caaa516a542d
# ╟─b4d619a1-8741-4902-86f8-cd8e84c9d785
# ╟─e243fe55-ee3e-47dc-9a7a-4319e0e86f8e
# ╟─c63369ed-58ed-4dd3-9292-c6c265ad52ba
# ╟─30cf6d78-c541-4d2d-b455-cb365d52b5cd
# ╟─e33b07c9-fb73-405d-8ee0-6e6e88e32bab
# ╠═03fb6688-dcc4-4ae4-ac6c-77000cbf90c3
# ╟─809cf0f6-dece-4294-9624-80a6dc19ee46
# ╟─eb14fb68-6950-4374-adef-35b583bf99fb
# ╟─e28dda3f-fbcd-47f0-8f99-4f3a2087905d
# ╟─901004da-2ac9-45fd-9643-8ce1cc819aa8
# ╟─c363c7d2-5749-49cb-8643-57a1b9dda8eb
# ╟─51b1572f-5965-41a8-b6d6-061c48f9af0c
# ╟─2c3f2b50-0a95-4577-add8-8bf72580a44f
# ╟─ec66c0b1-f75c-41e5-91b9-b6a358cd9c3c
# ╟─ea4a783e-6a18-4d4e-b0a2-bf4fd8070c7a
# ╟─3b937208-c0bb-42e2-99a2-533113a0d4e0
# ╟─3ea57a8e-8d15-4f41-acfb-e3bd1d65e585
# ╟─2ab95efd-4aca-412b-998f-f777b151d05e
# ╟─2e7c7e4c-83ec-4555-9b50-9a308354bab2
# ╟─760dfb53-fc7e-4314-998f-cd627be4b8e6
# ╟─d319d77e-fdf1-4971-a4e0-23bd0ad573cd
# ╟─35ad62e8-1212-4480-9b6f-d4c82da72c2e
# ╟─4b4d7507-2a07-4c3c-95ce-5446a9ba2362
# ╟─89bfebb4-ee0d-46c8-956a-6a5088599ae6
# ╟─1747e403-c5d6-471a-9f6e-fcf7ee8063d1
# ╟─bcc88b1b-4e37-4d68-8af2-9b3923634dfd
# ╟─e0cfcb9b-794b-4731-abf7-5435f67ced42
# ╟─84a4326b-2482-4b5f-9801-1a09d8d20f5b
# ╟─5c8ebc31-6e76-44fa-9b05-455030defcfb
# ╟─3393e09c-a474-47da-a21f-0d8ec9017979
# ╟─716ff72f-2299-472c-84b8-17315e8edc48
# ╟─665c5bbc-9a5c-4e4a-930d-725bc2c9c883
# ╟─a6b8e8e6-8af3-42e2-ae8a-d891739f0317
# ╟─8336bd73-5b07-4c2a-b773-2df78de81bb2
# ╟─5edefee7-9c68-4f4a-84c3-4a98207f890c
# ╟─e95dd200-9eb3-44a9-b810-c20b50812a90
# ╟─6d750ce4-26b8-4af4-916d-921e6bd9b16c
# ╟─6094abc7-96bb-4c3f-9156-b5de5d7873f6
# ╟─8ca2a156-e0f8-453f-a02c-daf23681bf89
# ╟─d4acae8a-8f27-4a2a-93d5-63d6b6b6db20
# ╟─e6947cf6-7982-4280-98cb-82053b70372f
# ╟─102aeb0e-fc0f-4099-9e71-fdfc36863e2e
# ╟─18a5208c-4a01-4784-a69d-0bd5e3bb9faf
# ╟─cc60e789-02d1-4944-8ad5-718ede99669c
# ╟─9d2e3f26-253e-4bba-b70f-fc3b5c4617d8
# ╟─629061d5-bf97-4ccf-af28-f1c5cd36b34c
# ╟─b4bfb1ba-73f2-45c9-9303-baee4345f8f6
# ╟─6be1c156-8dcc-48e3-b684-e89c4a0a7863
# ╟─b5732a82-e951-475a-b6f2-d8584c07c7a9
# ╟─0434dd27-4349-4731-80d5-b71ab99b53e2
# ╟─eae40715-9b33-494f-8814-8c6f967aeade
# ╟─552da9d4-5ab9-427d-9287-feb1eca57183
# ╟─55f2aacf-a342-4d4c-a1b7-60c5c29ab340
# ╟─3f869ae4-14b1-4b5a-b5c9-bc437bfc99da
# ╟─5a963c3d-46cd-4697-8991-5d5e1bb9a9e5
# ╟─1b6cb7eb-814e-4df8-89ed-56ebc8f06a4a
# ╟─d5ec14e2-fa45-4232-9ae8-06b84bf48525
# ╟─c4ab9540-3848-483c-9ba5-e795913f844a
# ╟─cba8b537-de68-4b2f-bf7a-d0bdd3aded7a
# ╟─a0cd59ef-6468-4622-ab0e-0f7239eba0a8
# ╟─fb217fd9-b929-47d9-ab69-b4637c894205
# ╟─4e038980-c531-4f7c-9c51-4e346eccc0aa
# ╟─359b407f-6371-4e8a-b822-956173e89a47
# ╠═e4176cf0-d5b7-4b9a-a4cd-b25f0f5a987f
# ╟─af373899-ed4d-4d57-a524-83b04063abf3
# ╟─6331b0f5-94be-426d-b055-e1369eb2a962
# ╟─06eebe92-bbab-449f-acb9-0e31ad2bfaa8
# ╟─7c48e850-fdd9-4e77-87fa-a5800a26a77b
# ╟─95e63027-6434-4fa7-a9e1-bcca754b9601
# ╟─62f07c1e-4226-4a35-8d3a-198e41e10354
# ╟─2f7b3bf2-ce1a-4755-af3b-a82f02fb7752
# ╟─de5879af-c979-4b3b-a444-db264c30297b
# ╟─a3ea595b-7b3e-4b97-bf1f-21f9a07fdd0d
# ╟─8e4324d5-2a88-41d3-b229-43e9f41d4191
# ╟─9c7bbd1f-cf1c-4eae-a4c5-36324c5aff0a
# ╟─c1b120cb-36ec-49b9-af55-13e98630b6db
# ╟─6328dc99-9419-4ce0-9c76-ed2cadd8e2f3
# ╟─ed986bfb-1582-4dcb-b39f-565657cfa59c
# ╟─170fc849-2f28-4d31-81db-39fbcc6ac6e4
# ╟─92e04df6-153d-402d-a7fe-f708390c1185
# ╟─f2d0e1e0-a8ef-4ac5-8593-892e4a5ac67c
# ╟─889093e8-5e14-4211-8807-113adbac9a46
# ╟─05820b6f-45e9-4eaa-b6ba-c52813b5fe46
# ╠═1d08c5f5-cbff-40ef-bcb8-971637931e20
# ╟─af868b9b-130d-4d4f-8fc6-ff6d9b6f604f
# ╟─5b980d00-f159-49cd-b959-479cd3b1a444
# ╟─08eb8c76-c19a-431f-b5ad-a14a38b18946
# ╟─928a1491-3695-4bed-b346-b983f389a26f
# ╟─bc04175a-f082-46be-a5ee-8d16562db340
# ╟─b0e16123-df7e-429c-a795-9e5ba788171a
# ╟─58663741-fa05-4804-8734-8ccb1fa90b2d
# ╟─5d28e09c-891d-44c0-98a4-ef4cf3a235f1
# ╟─a0465ae8-c843-4fc0-abaf-0497ada26652
# ╟─cdf72ed6-0d70-4901-9b8f-a12ceacd359d
# ╟─e03a111a-8cf9-40af-842d-8f8ca8a197fd
# ╠═654ecc9d-731a-4ce7-ac45-f6d9d229c59e
# ╠═66440f3a-65b4-4f4e-8158-d886138044e3
# ╟─976711c5-009d-4feb-bc14-d262e250fdc5
# ╟─7d3f2a61-fd98-4925-a932-04bec1ba1c3c
# ╟─bb58e7e4-6e7f-4966-af6b-e51cb405c470
# ╟─e954e6e8-e593-4cc9-9318-2f88ee8ae261
# ╟─e8d5b5f2-8e69-4279-8ea7-c6f5d78ac828
# ╟─634bd495-a611-4ee8-9d3c-4e17c536ac6d
# ╟─71029d31-4689-44af-aa8c-d0a154edd8c4
# ╟─02002d81-0b36-4b55-87b7-5778f6942014
# ╟─ca675aba-8031-47d7-a2b2-2aa8d37cda00
# ╟─70d3f9fa-c6d4-4cf8-927b-540f0b2fd00e
# ╟─a3ef0d5d-1ca4-47b5-a54f-501431461a5f
# ╟─2f8e92fc-3f3f-417f-9171-c2c755d5e0f0
# ╟─dafd1a68-715b-4f06-a4f2-287c123761f8
# ╟─93b4939f-3406-4e4f-9e31-cc25c23b0284
# ╟─620789b7-59bc-4e17-bcfb-728a329eed0f
# ╟─7b47cda6-d772-468c-a8f3-75e3d77369d8
# ╟─d66e373d-8443-4810-9332-305d9781a21a
# ╟─acfb80f0-f4d0-4870-b401-6e26c1c99e45
# ╟─e091ce93-9526-4c7f-9f14-7634419bfe57
# ╟─d44526f4-3051-47ee-8b63-f5e694c2e609
# ╟─27755688-f647-48e5-a939-bb0fa70c95d8
# ╠═99ed2b50-2ea4-465a-9d4f-4517e74d1216
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
