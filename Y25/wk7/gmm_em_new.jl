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

# ╔═╡ 120a282a-91c1-11ec-346f-25d56e50d38c
begin
	using Distributions,Random, StatsBase, Clustering, LinearAlgebra
	using PlutoUI
	using StatsPlots
	# using PlutoTeachingTools
end

# ╔═╡ 06d45497-b465-4370-8411-9651e33e70e6
begin
	# using LinearAlgebra
	# using PlutoUI
	using PlutoTeachingTools
	using LaTeXStrings
	using Latexify
	# using Random
	using Statistics
	using LogExpFunctions
	using HypertextLiteral
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	
end

# ╔═╡ adaf23c3-1643-41c5-84a7-e2b73af048d6
begin
	using Logging
	Logging.disable_logging(Logging.Info); # or e.g. Logging.Info
end;

# ╔═╡ 527963b1-ee7a-4cad-a2b7-b6a7789bfbbe
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ 0f826d7b-9628-4962-9c9e-db3ea287954a
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

# ╔═╡ 6f051fad-2c4b-4a9e-9361-e9b62ba189c5
TableOfContents()

# ╔═╡ be9bcfcb-7ec7-4851-bd3f-24d4c29462fe
md"""

# CS5014 Machine Learning


#### Unsupervised learning 
###### EM algorithm
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 6148a7c6-8c6d-48f7-88aa-4657841882b7
# md"""

# ## Clustering 

# """

# ╔═╡ f69c85dd-6227-4d63-9ad8-8e6a6610ef84
# TwoColumn(begin
# 	gr()
# 	# plt_qda = plot_clustering_rst(data₂, K₂, truezs₂,  truemvns₂, trueπs₂; add_gaussian_contours=true)
# 	scatter(data₂[:, 1], data₂[:,2], ms=4, alpha=0.5, framestyle=:origin, label="x", ratio=1, xlabel=L"x_1", ylabel=L"x_2", title="Unsupervised learning data", size=(350,350), titlefontsize=10)

# end, let 

# plot(plt_qda, size=(350,350), xlabel=L"x_1", ylabel=L"x_2", title="Clustering objective", titlefontsize=10)
# end)

# ╔═╡ 7db71b2e-caae-415f-b87b-f87665fd8d5e
md"""

# Gaussian mixture models
"""

# ╔═╡ 83ee2491-d08d-4eea-8a3a-b8f9715c9351
md"""

## Classification _vs_ Clustering


"""

# ╔═╡ f12c11a2-f548-4409-ab06-e6a7cb8dfb06
TwoColumn(md"""


> ##### Classification 
> * #####  *known* labels $\{y^{(1)}, y^{(2)}, \ldots, y^{(n)}\}$

""", md"""



> ##### Clustering 
> * #####  *unknown* labels $\{y^{(i)} = \texttt{missing}\}$



""")

# ╔═╡ 4bfd08f3-5bdb-48a8-bacd-0d38ad674b74
md"""

## Probabilistic generative model 

##### -- no fundamental difference

"""

# ╔═╡ e0a0ef50-cb98-44a9-a2d8-10191a30e521
TwoColumn(md"""


> ##### Supervised learning 
> * #####  labels are observed $\{y^{(1)}, y^{(2)}, \ldots, y^{(n)}\}$

""", md"""



> ##### Unsupervised learning 
> * #####  labels are hidden $\{y^{(i)} = \texttt{missing}\}$



""")

# ╔═╡ 7c8bfb91-4f3c-4e56-88b1-cb822983b3bc
TwoColumnWideLeft(html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/qdavsgmm.svg
' width = '550' /></center>", html"<br/><br/><center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/notations.png' width = '110' /></center>")

# ╔═╡ fb6ed0e0-a21d-44a5-a0fb-53cdd318628a
# md"""

# ## Gaussian mixture model (GMM)


# ##### Gaussian mixture model (GMM) 

# * shares **the same underlying model** as QDA

# * the **unsupervised learning** version of QDA
#   * the labels ``z^{(i)}`` are **not observed**

# """

# ╔═╡ cb88a7dd-0abf-4229-b29a-ba0b963b63fc
# md"""

# ## Gaussian mixture model (GMM)


# """

# ╔═╡ 5acf48ee-9b2f-4727-8a6b-fc3d0d7f68ed
md"""

## Gaussian mixture model (GMM)


"""

# ╔═╡ e0196ba3-314e-417d-9a36-5a6b8b0c556a
TwoColumnWideLeft(md"""

#### Since ``z^{(i)}`` are *hidden*, we can sum it out

```math
\large
\begin{align}
p(\mathbf{x}^{(i)}) &= \sum_{z^{(i)}} p(z^{(i)}, \mathbf{x}^{(i)})\;\;\;\;\;\;\text{sum rule}\\
&= \sum_{k=1}^K p(z^{(i)}=k)p(\mathbf{x}^{(i)}|z^{(i)}=k)\\
&= \underbrace{\sum_{k=1}^K \pi_k\, \mathcal{N}(\mathbf{x}^{(i)}; \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)}_{\text{Gaussian mixture model's density}}
\end{align}
```


""", html"<br/><br/><br/><center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gmm_bn.svg
' width = '340' /></center>")

# ╔═╡ 5d820b63-c9c7-4ec2-9c53-966ff6fd70df
# TwoColumnWideLeft(md"""

# The **full** probabilistic model for GMM is

# ```math
# \large
# p(z^{(i)}, \mathbf{x}^{(i)}) = p(z^{(i)}) p(\mathbf{x}^{(i)}|z^{(i)})
# ```

# where  

# * prior for $z^{(i)}$: ``p(z^{(i)})``: how *popular* that class is in apriori

# $$p(z^{(i)}=k) = \pi_k$$

# * likelihood for $\mathbf{x}^{(i)}$: ``p(\mathbf{x}^{(i)}|z^{(i)}=k)``:  given knowing the label, how likely to see a observation $\mathbf{x}^{(i)}$:

# $$p(\mathbf{x}^{(i)}|z^{(i)}=k) = \mathcal{N}(\mathbf{x}^{(i)}; \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)$$

# """, html"<br/><br/><br/><center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/gmm_bn.svg
# ' width = '240' /></center>")

# ╔═╡ 6c2164fe-c544-4c42-bed4-9168e1ab049a
md"""
## Why sum-out ``z`` ?

#### Because $z$s are not observed

* ###### the observed for clustering problem is $\mathcal{D}=\{\mathbf{x}^{(i)}\}$ only!


#### And *likelihood* is: *the conditional probability* of the _observed data_

$\large p(\mathcal{D}|\theta) = p(\{\mathbf{x}^{(i)}\}|\theta) = \prod_{i=1}^n \left(\sum_{k=1}^K \pi_k \cdot \mathcal{N}(\mathbf{x}^{(i)}; \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)\right )$
* without marginalisation, we do not have a likelihood model to train the model


#### *In comparison*, QDA's likelihood model is 

$\large \begin{align}p(\mathcal{D}|\theta) &= p(\{\mathbf{x}^{(i)}, z^{(i)}\}|\theta) = \prod_{i=1}^n p(z^{(i)})p(\mathbf{x}^{(i)}|z^{(i)}) \\
&= \prod_{i=1}^n 
 \pi_{z^{(i)}} \mathcal{N}(\mathbf{x}^{(i)}; \boldsymbol{\mu}_{z^{(i)}}, \mathbf{\Sigma}_{z^{(i)}})
\end{align}$

"""

# ╔═╡ 7b51f61c-7635-4d79-97a4-9e5210c827cf
md"""

## GMM visialisation (1-dimensional)

```math
\large
\begin{align}
p({x}) 
&= \sum_{k=1}^K \pi_k\, \mathcal{N}({x}; {\mu}_k, {\sigma}^2_k)
\end{align}
```

* ``p(x)``: just super-imposed ``K`` Gaussians

"""

# ╔═╡ 03f6cd09-9c78-4592-b3e4-379cbdff40a9
md" ``\pi_1\propto`` $(@bind n₁0_ Slider(1:0.5:10, default=1));	``\pi_2\propto`` $(@bind n₂0_ Slider(1:0.5:10, default=1)); ``\pi_3\propto`` $(@bind n₃0_ Slider(1:0.5:10, default=1))"

# ╔═╡ 2ab75abc-a685-4a5e-becf-6976ed439068
begin
	πs0_ = [n₁0_, n₂0_, n₃0_]
	πs0_ = πs0_/sum(πs0_)
end;

# ╔═╡ 702c96a8-439a-4b02-9125-06767d363e71
md"""
``\boldsymbol{\pi}=`` $(latexify_md(round.(πs0_; digits=3))); 

The three univariate Gaussians are $\mathcal{N}(-3,1), \mathcal{N}(0, 1), \mathcal{N}(3, 1)$

"""

# ╔═╡ 0b7dfe99-e93d-4203-81fd-b04c38105daa
let
	gr()
	trueμs = [-3, 0, 3]
	trueσs = [1 , 1, 1]
	# trueπs = [0.15, 0.7, 0.15]
	trueπs = πs0_
	mvns = [Normal(trueμs[k], trueσs[k]) for k in 1:3]
	plt = plot(xlim =[-6, 6], label="", framestyle=:origins, yaxis=true, ylabel=L"p(x)", title="", legend=:outerbottom)
		mixn = MixtureModel(mvns, trueπs)

	plot!((x) -> pdf(mixn, x), lw=2, label=L"p(x) = \sum_{k} \pi_k \mathcal{N}(\mu_{k}, \sigma^2_{k})", legendfontsize=10)
	for (k, nn) in enumerate(mvns)
		plot!((x) -> trueπs[k] * pdf(nn, x), label=L"\pi_{%$(k)}\, \mathcal{N}(\mu_{%$k}, \sigma^2_{%$k})", lw=1, ls=:dash)
	end
	
	plt
end

# ╔═╡ 6a9df385-4261-4adb-80d3-32a02808f0f0
md"""

## GMM Visualisation (multi-d)

```math
\large
\begin{align}
p(\mathbf{x}) 
&= \sum_{k=1}^K \pi_k\, \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)
\end{align}
```

"""

# ╔═╡ e7c6725d-74d3-4fd4-9abe-38716693f2bb
md" ``\pi_1\propto`` $(@bind n₁0 Slider(1:0.5:10, default=1));	``\pi_2\propto`` $(@bind n₂0 Slider(1:0.5:10, default=1)); ``\pi_3\propto`` $(@bind n₃0 Slider(1:0.5:10, default=1))"

# ╔═╡ 07613525-1f06-4f78-bbfa-4486fc0cf121
begin
	πs0 = [n₁0, n₂0, n₃0]
	πs0 = πs0/sum(πs0)
end;

# ╔═╡ 5d0816c3-0f4a-4a0d-b0a8-1cd5644e8eba
md"""
``\boldsymbol{\pi}=`` $(latexify_md(round.(πs0; digits=3)))
"""

# ╔═╡ e431bdb7-fd7d-4ded-9ec5-993518d89381
md"""

# Learning of GMM (EM algorithm)
"""

# ╔═╡ 09ca554e-e5c6-400e-9a1c-11b60b8fd338
md"""

## Learning 

#### Learning: find the MLE of the parameters based on $\mathcal{D}$

```math
\Large
\hat{\boldsymbol{\Theta}}  \leftarrow  \arg\max_{\boldsymbol\Theta}\; \ln p(\mathcal{D}|\mathbf{\Theta})
```


* ##### parameters: ``\boldsymbol\Theta = \{\boldsymbol\pi, \{\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c\}_{c=1}^C\}``



"""

# ╔═╡ 983bc7fa-6b95-4c09-bf2f-85152febfa36
TwoColumn(
md"""

#### Supervsied learning

* ##### the observed ``\mathcal{D}=\{\mathbf{x}^{(i)}, y^{(i)}\}``

$(show_img("gda_learning.svg", w=330))
"""

, 

	md"""

#### Unsupervsied learning

* ##### the observed ``\mathcal{D}=\{\mathbf{x}^{(i)}\}``

	
$(show_img("emlearning.svg", w=330))
"""
)

# ╔═╡ 5e469163-3e8b-46be-a218-b608f01f75cf
# md"""

# ## Learning of GMM

# In practice, we do not know the model parameters 

# ```math
# \large
# \boldsymbol\Theta = \{\boldsymbol\pi, \{\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^K\}
# ```



# * the observed data ``\large \mathcal{D}=\{\mathbf{x}^{(i)}\}_{i=1}^n`` only



# **Learning** or **training**: estimate a model's parameters given *observed* data by MLE

# $$\large\begin{align}\hat{\boldsymbol{{\Theta}}} &= \arg\max_{\boldsymbol\Theta}\, \underbrace{\ln p(\mathcal{D}|\boldsymbol\Theta)}_{\mathcal{L}(\boldsymbol\Theta)}
# = \arg\max_{\boldsymbol\Theta}\, \ln \left\{\prod_{i=1}^n \left(\sum_{k=1}^K \pi_k \cdot \mathcal{N}(\mathbf{x}^{(i)}; \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)\right )\right \}\\
# &= \arg\max_{\boldsymbol\Theta}\sum_{i=1}^n \ln \left(\sum_{k=1}^K \pi_k \cdot \mathcal{N}(\mathbf{x}^{(i)}; \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)\right )\end{align}$$

# * however, it is not easy to directly optimise it when ``\{z^{(i)}\}`` are hidden

# * ``\boldsymbol{\pi}, \mathbf{\Sigma}_k`` are all contrained, not easy to apply gradient descent

# * the ``\ln \sum_k`` is in general not easy to compute, note that ``\ln \sum_k \neq \sum_k \ln``
# """

# ╔═╡ 1751787f-b7d6-4078-903a-decc0804ce54
md"""

## The supervised is fairly straightforward 

##### -- *i.e.* when the labels ``\{z^{(i)}\}`` are observed
"""

# ╔═╡ fbc2977f-c02c-4ac1-ace3-b0ad8ee90149
md"""

##### The supervised learning rules

> $$\large \hat \pi_k =\frac{\sum_{i=1}^n \mathbb{1}(z^{(i)} = k)}{n} =\frac{n_k}{n}$$
> $$\large \hat{\boldsymbol{\mu}}_k = \frac{1}{n_k}\, {\sum_{i=1}^n \mathbb{1}(z^{(i)}=k)\cdot\mathbf x^{(i)}}$$
> $$\large \hat{\boldsymbol{\Sigma}}_k = \frac{1}{n_k} \sum_{i=1}^n \mathbb{1}(z^{(i)}=k) (\mathbf x^{(i)}-\boldsymbol{\mu}_k)(\mathbf x^{(i)}-\boldsymbol{\mu}_c)^\top$$
"""

# ╔═╡ 0dbc7c7d-4dea-48c5-a086-0a8666b732ef
md"""

## Supervised learning of QDA
"""

# ╔═╡ d692526e-fbe5-4982-b217-88b0565b96bc
md"""

## Towards EM algorithm 



##### _Unsupervised learning_ have no observed ``\{z^{(i)}\}`` 

##### _But_ the probabilistic model can be used to compute their estimates

$\large{p(z^{(i)}=k|\mathbf{x}^{(i)})} = r_{ik}$

* ###### ``r_{ik}``: the responsibility of cluster ``k`` towards ``\mathbf{x}^{(i)}``

"""

# ╔═╡ 45299ff7-3fc7-4277-9d46-75ba525f73f5
md"""

* note that ``p(z^{(i)}=k|\mathbf{x}^{(i)})`` can be viewed as an **estimate** of ``\mathbb{1}(z^{(i)} =k)``
  
  $p(z^{(i)}=k|\mathbf{x}^{(i)}) = \hat{\mathbb{1}}(z^{(i)} =k)$

* the same idea as softmax and one-hot encoding vector, *e.g. assume ``z^{(i)}=2``*

$p(z^{(i)}|\mathbf{x}^{(i)}) = \begin{bmatrix}0.01_{=r_{i1}} \\ \colorbox{pink}{0.99}_{=r_{i2}} \\ \vdots \\
0_{=r_{iK}} \end{bmatrix};\;\; \mathbb{1}(z^{(i)}) = \begin{bmatrix}0 \\ \colorbox{pink}1\\ \vdots \\
0\end{bmatrix}$
"""

# ╔═╡ 4ebf3dfc-0df7-45a2-8e81-d9b7225a218e
aside(tip(md"""
Actually, $p(z^{(i)}=k|\mathbf{x}^{(i)}) = \mathbb{E}[z^{(i)}=k |\mathbf{x}^{(i)}]$

* the expectation of the random variable ``\mathbb{1}(z^{(i)} =k)``
"""))

# ╔═╡ e5fa6fe9-af47-487c-9c30-e35b6da8d5bd
md"""

## Computing ``p(z|\mathbf{x})``: Bayes' rule

```math
\large
p({z}=k|\mathbf{x}; \mathbf\Theta) = \frac{\pi_k\cdot \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)}{\sum_{j=1}^K \pi_k\cdot \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_j, \mathbf{\Sigma}_j)}
```

* ###### note that it depends on ``\mathbf\Theta = \{\boldsymbol{\pi}, \{\boldsymbol{\mu}_k, \mathbf{\Sigma}_k\}\}`` !
"""

# ╔═╡ 8fb18689-3f90-4971-95d4-ef622af4d8dd
md"""

## An egg chicken dilemma


"""

# ╔═╡ c88de029-9a12-4a77-9a24-5942fc99a06f
TwoColumn(md"""
\


##### *Egg* and *Chicken* depends on each other 

* to have eggs: we need chicken
* to have chicken: we need eggs
* *i.e.* they are coupled 


""", 

html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/eggchicken.jpeg
' height = '200' /></center>"
)

# ╔═╡ fd5b729e-e51e-44b2-a992-f68852313797
md"""

## An egg chicken dilemma


"""

# ╔═╡ 1c380bc6-24a5-462c-bddf-5e0bcd7ea855
TwoColumn(md"""
\


##### *Egg* and *Chicken* depends on each other 

* to have eggs: we need chicken
* to have chicken: we need eggs
* *i.e.* they are coupled 


""", 

html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/eggchicken.jpeg
' height = '200' /></center>"
)

# ╔═╡ 188f7d65-3c87-4229-a58f-f7d8a6a21d32
TwoColumn(md"""
\


##### ``\{z^{(i)}\}`` and ``\mathbf{\Theta}`` are in the same dilemma 

* to estimate ``p(z|\mathbf{x}, \mathbf{\Theta})``: we need ``\mathbf{\Theta}``
* to estimate ``\mathbf{\Theta}``: we need ``\{z^{(i)}\}``
* *i.e.* they are coupled 


""", 

html"<br/><center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/em_dilimma.png
' height = '150' /></center>"
)

# ╔═╡ 780190ed-b34f-414b-b778-a5891deb3a8f
# md"""

# ## Revisit K-means

# K-means is a specific case of EM with the following assumptions

# **Model wise**, the prior 

# $$p(z^{(i)}=k) = \pi_k = 1/K$$
# * and covariances are tied but also fixed to be identity matrix $\mathbf{\Sigma}_k = \mathbf{I}$, which explains the Euclidean distance used

# **assignment step** is just a **hard E step** (winner takes all)

# $$r_{ik} \leftarrow \begin{cases} 1, \text{ if } k=\arg\max_{k'} p(z^{(i)}=k'|\mathbf{x}^{(i)})& \\ 0, \text{ otherwise} \end{cases}$$ 
# $$\begin{align*}
#   \arg\max_{k'} p(z^{(i)}=k'|{x}^{(i)}) &=\arg\max_{k'}\frac{\bcancel{\tfrac{1}{K}} \mathcal{N}(x^{(i)}; {\mu}_{k'}, {I})}{\sum_{j=1}^K \bcancel{\tfrac{1}{K}} \mathcal{N}(x^{(i)}; {\mu}_j, {I})} \\
#   &= \arg\max_{k'} \frac{1}{(2\pi)^{d/2}}\cdot \exp\left (-\frac{1}{2}(x^{(i)}-\mu_{k'})^\top(x^{(i)}-\mu_{k'})\right )\\
#   &= \arg\min_{k'} (x^{(i)}-\mu_{k'})^\top(x^{(i)}-\mu_{k'}) \\
#   &= \arg\min_{k'}\|{x}^{(i)}-{\mu}_{k'}\|_2^2
#   \end{align*}$$

# **update step** is the M-step following the above hard assignment 
#   * only update the mean $\mu_k$ based on the assignment
#   * as $\pi$ and $\Sigma_k$ are assumed known or fixed

# """

# ╔═╡ 8d03c7ee-2db1-46c2-bf5a-d55a59dfe863
md"""

## Revisit K-means


"""

# ╔═╡ f11eac0e-59d4-4594-9535-d2ad75fb456f
TwoColumn(md"""
##### K-means makes _simplifying_ assumptions

* ###### uniform prior  
$$p(z^{(i)}=k) = \pi_k = 1/K$$

* ###### identity covs: $$\boldsymbol{\Sigma}_k =\mathbf{I}$$

* ###### and *hard* posterior (winner takes all)

$$r_{ik} \leftarrow \begin{cases} 1, \text{ if } k=\arg\max_{k'} p(z^{(i)}=k'|\mathbf{x}^{(i)})& \\ 0, \text{ otherwise} \end{cases}$$ 


""", md"""
\

$(show_img("kmeans_dag_.svg", w=300))

"""
)

# ╔═╡ c81ad0d3-1bd5-480e-829f-ce31a9d6e907
show_img("kmeans_circ.svg", w=500)

# ╔═╡ 53757539-9ce7-4ba8-84d7-994d9830d41f
md"""

## EM algorithm for GMMs

"""

# ╔═╡ c258c3b1-c841-49b6-8d71-19213aa6e6d6
TwoColumn(md"""



#### Similarly, EM for GMM:

* ##### E step: update ``p(z|\mathbf{x}, \mathbf{\Theta}^{(t)})`` conditioned on ``\mathbf{\Theta}^{(t)}``
* ##### M step: re-estimate ``\mathbf{\Theta}^{(t+1)}`` conditioned on ``\mathbf{r}=p(z|\mathbf{x}, \mathbf{\Theta}^{(t)})``


""", 

html"<br/><center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/em_dilimma2.png
' height = '150' /></center>"
)

# ╔═╡ 3b3359f4-b500-40bc-a7fd-c80e19023f73
md"""

## EM  vs QDA learning


!!! note "Supervised Learning of QDA"
	$\small
	\begin{align}\hat \pi_k &=\frac{1}{n}\sum_{i=1}^n \mathbb{1}(z^{(i)} = k)\\
	\hat{\boldsymbol{\mu}}_k &= \frac{1}{\sum_{i=1}^n \mathbb{1}(z^{(i)} = k)}\, {\sum_{i=1}^n \mathbb{1}(z^{(i)}=k)\cdot\mathbf x^{(i)}}
	\\
	\hat{\boldsymbol{\Sigma}}_k&= \frac{1}{\sum_{i=1}^n \mathbb{1}(z^{(i)} = k)} \sum_{i=1}^n \mathbb{1}(z^{(i)}=k) (\mathbf x^{(i)}-\boldsymbol{\mu}_k)(\mathbf x^{(i)}-\boldsymbol{\mu}_c)^\top\end{align}$



##### The *Maximisation step* (M step) of *EM* re-estimates the average log-likelihood


```math
\large
{\Theta}^{(t)} \leftarrow \mathbb{E}_{{z}^{(i)} \sim p({z}^{(i)}|\Theta^{(t-1)}, \mathbf{x}^{(i)})}\left [\ln p(\{z^{(i)}, \mathbf{x}^{(i)}\}|\Theta) \right ]
```

* by replacing $\mathbb{1}(z^{(i)}=k) \Rightarrow \underbrace{p(z^{(i)}=k|\mathbf{x}^{(i)}, \Theta^{(t-1)})}_{r_{ik}}$

!!! note "M-step of EM algorithm"
	$\begin{align}\pi_k^{(t)} &\leftarrow \frac{1}{n}\sum_{i=1}^n r_{ik}\\
	\boldsymbol{\mu}_{k}^{(t)} &\leftarrow \frac{1}{\sum_{i=1}^n r_{ik}} \sum_{i=1}^n r_{ik}\mathbf{x}^{(i)}\\
	{\mathbf{\Sigma}}_{k}^{(t)} &\leftarrow \frac{1}{\sum_{i=1}^n r_{ik}} \sum_{i=1}^n r_{ik} (\mathbf{x}^{(i)}-{\boldsymbol{\mu}}_{k})(\mathbf{x}^{(i)}-{\boldsymbol{\mu}}_{k})^\top
	\end{align}$


"""

# ╔═╡ 9ac9bb50-0419-42c2-a90f-c3995ff72df5
# md"""

# ## EM algorithm estimates MLE



# #### It can be shown that the (log-)likelihood 

# $\large \ln p(\mathcal{D}|\mathbf\Theta^{(\text{iter}-1)}) < \ln p(\mathcal{D}|\mathbf\Theta^{(\text{iter})})$ 

# * both E step and M step improves the log likelihood

# * the algorithm will finally converge to a **(local) maximum** (depends on initialisation)
# * monitor the (log) likelihood for debugging and convergence check
# """

# ╔═╡ 1d3dc35d-f9f7-4b17-962d-0430d5a1cfea
md"""

## EM algorithm for GMMs




"""

# ╔═╡ 49e641e7-977c-4f34-94dd-8db4f31939d0
md"""

----


##### *Initilisation*: randomly guess $\mathbf\Theta^{(0)} =\{{\pi_k}^{(0)}, \boldsymbol\mu_k^{(0)}, \mathbf\Sigma_k^{(0)}\}_{k=1}^K$

##### *Repeat* until converge

* ##### E step: for $i= 1\ldots n,\; k= 1\ldots K$
$$r_{ik} \leftarrow p(z^{(i)}=k|\mathbf{x}^{(i)}) = \frac{\pi_k^{(t)} \mathcal{N}(\mathbf{x}^{(i)}; \boldsymbol{\mu}_k^{(t)}, \mathbf{\Sigma}_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} \mathcal{N}(\mathbf{x}^{(i)}; \boldsymbol{\mu}_j^{(t)}, \mathbf{\Sigma}_j^{(t)})}$$


* ##### M step: update $\mathbf{\Theta}^{(t)}$, for $k=1\ldots K$

$\pi_k^{(t)} \leftarrow \frac{1}{n}\sum_{i=1}^n r_{ik}$

$\boldsymbol{\mu}_{k}^{(t)} \leftarrow \frac{1}{\sum_{i=1}^n r_{ik}} \sum_{i=1}^n r_{ik}\mathbf{x}^{(i)}$

${\mathbf{\Sigma}}_{k}^{(t)} \leftarrow \frac{1}{\sum_{i=1}^n r_{ik}} \sum_{i=1}^n r_{ik} (\mathbf{x}^{(i)}-{\boldsymbol{\mu}}_{k})(\mathbf{x}^{(i)}-{\boldsymbol{\mu}}_{k})^\top$

$t\leftarrow t+1$


----

"""

# ╔═╡ 4d4d467a-8dab-4111-b950-afd405580d65
md"""

## EM algorithm -- "ME" algorithm

#### We can swap the order of E and M steps

----


##### *Initilisation*: random guess the responsibilities $\mathbf{r}_k$

##### *Repeat* until converge

* ##### M step: update $\mathbf{\Theta}^{(t)}$, for $k=1\ldots K$



* ##### E step: update $r_{ik} \leftarrow p(z^{(i)}=k|\mathbf{x}^{(i)})$



----

"""

# ╔═╡ 629c24b5-0028-47b8-9490-0c3d8a8665b7
md"""

## Visualise E step: ``p(z=k|\mathbf{x}, \Theta)``


#### We can visualise the responsibity vectors ``\mathbf{r}_k``


```math
\Large
p({z}^{(i)}=k|\mathbf{x}^{(i)}; \mathbf\Theta) = r_{ik}, \;\; \text{
for }i=1\ldots n
```

```math
\Large
\mathbf{r}_k = [r_{1k}, r_{2k}, \ldots, r_{nk}]^\top
```

* #### circle size ``\propto r_{ik}``
  * therefore, more likely the ``i``-th observation to belong cluster ``k``, the larger circle

"""

# ╔═╡ 8f9757f3-4f8d-470b-9964-65812c8d86be
md"""

#### ``\mathbf{R}`` matrix example
* ``n=500`` rows: *i.e.* ``500`` observations
* ``K=3`` columns: *i.e.* ``3`` clusters
* each row is non-negative and sum to one: which is the posterior $p(z^{(i)}|\mathbf{x}^{(i)})$ for the $i$-th observation
"""

# ╔═╡ 1f7cbf66-3c75-486f-bd55-09930fb50579
md"""

## E step implementation*

#### In E-step, we need to compute

$$\large r_{k} = p(z=k|\mathbf{x}) = \frac{\pi_k \ell_k}{\sum_{k'=1}^K \pi_{k'} \ell_{k'}}$$

* where $\ell_k = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)$ is the likelihood; also note that $\sum_k r_k=1$

#### We usually compute probability in log space: 

$$\large \ln r_{k} = \ln \pi_k  + \ln\ell_k -\underbrace{\ln\left \{ \sum_{k'}\exp\{\ln \pi_{k'} + \ln \ell_{k'}\}\right\}}_{\text{logsumexp}}$$



* one should use `logsumexp` instead of naive implementation to compute the log evidence

$$\large \ln r_{k} = \ln \pi_k  + \ln\ell_k -\texttt{logsumexp}(\{\ln \pi_k+ \ln \ell_k\}_{k=1}^K)$$



"""

# ╔═╡ 7ad702d7-7ceb-48ee-9258-ea78de864b0d
Foldable("Marginal log-likelihood", md"""

* also note that the log evidence is the log marginal likelihood (the optimisation objective)!

$$\large\ln p(x) = \ln \sum_{k'} \pi_{k'}\ell_{k'}=\texttt{logsumexp}(\{\ln \pi_k+ \ln \ell_k\}_{k=1}^K)$$

""")

# ╔═╡ 5f6adb33-a20a-47ca-84fd-9d5440f30749
md"""


#### To find the normalised probability, we apply `exp` back

$$\large r_{k} = \texttt{exp}(\ln r_k)$$
"""

# ╔═╡ 017b7b44-1b9b-4627-bb9d-2d9743ab0774
md"## Demonstration

Suppose we are normalising the unnormalised log posterior vector: ``\ln \mathbf{r}=``$(latexify_md([-1e3, -1e3, -1e3])),

The correct ouput after normalisation should be: ``\mathbf{r}=``$(latexify_md(round.([1/3, 1/3, 1/3]; digits=2)))
"

# ╔═╡ 28b3b5fd-2e8d-4a90-b708-710ac1cbae0e
[-1e3, -1e3, -1e3]

# ╔═╡ 1ef1de8d-73e7-44ec-bdea-03d763d6332e
let
	logpost = [-1e3, -1e3, -1e3]
	## naive implementation: e^{-1e3} = 0! underflow to 0!
	sum(exp.(logpost))
	log(sum(exp.(logpost))), exp.(logpost .- log(sum(exp.(logpost))))
end

# ╔═╡ f1b2db15-a9ff-4fbb-a2f7-4ca99cc44637
let
	logpost = [-1e3, -1e3, -1e3] 
	## alterantively, you can also just use softmax
	logsumexp(logpost), exp.(logpost .- logsumexp(logpost)), softmax(logpost)
end

# ╔═╡ 6a168c4f-19e8-4335-8b51-34c2a42dadac
md"""


## Visualise M-step: ``\boldsymbol{\pi}``



#### The re-estimation of ``\pi_k`` are for ``k =1\ldots K``


$\large \begin{align}
	{\pi}_{k}^{(t)} &\leftarrow \frac{\sum_{i=1}^n r_{ik}}{\sum_{k'=1}^{K}\sum_{i=1}^n r_{ik'}} = \frac{n_k}{\sum_{k'=1}^K n_{k'}}
	\end{align}$

* ``n_k = \sum_{i=1}^n {r}_{ik}``

* ``\boldsymbol{\pi}^{(t)} \propto \begin{bmatrix}71.9, 127.6, 300.5\end{bmatrix} \approx \begin{bmatrix}0.14, 0.26, 0.6\end{bmatrix}``
"""

# ╔═╡ 4d8bbaa8-57dd-4e7f-8bf6-d82e4ffabe64
md"""


## Visualise M-step: ``\boldsymbol{\mu}_k``


#### The re-estimation of ``\boldsymbol\mu_k`` are for ``k =1\ldots K``


$\begin{align}
	\boldsymbol{\mu}_{k}^{(t)} &\leftarrow \frac{1}{\sum_{i=1}^n r_{ik}} \sum_{i=1}^n r_{ik}\cdot \mathbf{x}^{(i)}
	\end{align}$

* ##### *weighted average*; the weights are the responsibility vector ``\mathbf{r}_k``
"""

# ╔═╡ 0e782555-a8e5-4c4a-9afa-3e7b8de143ca
md"""


## Visualise M-step: ``\boldsymbol{\Sigma}_k``



#### The re-estimation of ``\boldsymbol\mu_k, \mathbf{\Sigma}_k`` are the same idea for ``k =1\ldots K``


$\begin{align}
	\boldsymbol{\mu}_{k}^{(t)} &\leftarrow \frac{1}{\sum_{i=1}^n r_{ik}} \sum_{i=1}^n r_{ik}\mathbf{x}^{(i)}\\
	{\mathbf{\Sigma}}_{k}^{(t)} &\leftarrow \frac{1}{\sum_{i=1}^n r_{ik}} \sum_{i=1}^n r_{ik} (\mathbf{x}^{(i)}-{\boldsymbol{\mu}}_{k}^{(t)})(\mathbf{x}^{(i)}-{\boldsymbol{\mu}}_{k}^{(t)})^\top
	\end{align}$

* ##### weighted average and the weights are the responsibility vector ``\mathbf{r}_k``
"""

# ╔═╡ a2781145-89bd-4bd5-af50-095410ebc6a5
md"""
## Demon of EM

"""

# ╔═╡ 5a0be82c-a669-4134-b064-f4363661f439
# begin
# 	gr()
# 	mixAnims = mixGaussiansDemoGif(data₂, K₂, 100; init_step="m", add_contour=true, seed=222)
# end;

# ╔═╡ 619387f3-e0c8-43d8-90f9-b319f3c849dd
# gif(mixAnims[1], fps=10)

# ╔═╡ 453a4215-6ff5-4b45-ad23-ed6710389d1f
show_img("emgmm1.gif")

# ╔═╡ 5a372f09-1b13-443c-ad81-4337612669aa
md"""

## Demon of EM (another dataset)
"""

# ╔═╡ e04d2344-4787-4eb5-97a4-8d02dad09b88
# begin
# 	gr()
# 	mixAnims₃ = mixGaussiansDemoGif(data₃, K₃, 100; init_step="m", add_contour=true, seed=123)
# end;

# ╔═╡ 69321cbd-f325-4da8-8674-a90f616b9ee2
# gif(mixAnims₃[1], fps=5)

# ╔═╡ ea1efb75-3af9-49c5-9c1d-8bc298c6eaad
show_img("emgmm2.gif")

# ╔═╡ 801732eb-b50a-40c1-8199-576cdd06ce5e
md"""

## Demon of EM


#### The log-likelihood only increases over the iteration
* ##### very useful debug tool
"""

# ╔═╡ ab538ef3-0ebf-4a4c-839a-eea19e1920d8
begin

	Random.seed!(123)
	nobs = 1200
	ts = rand(nobs) * 2 .- 1 

	ys = @. 0.45 * sin(π * ts * 4.5) - 0.5 * ts + randn() * 0.15
	data₄ = [ys ts]
end;

# ╔═╡ c00e00f2-a1d6-407a-8326-15e6cf125d56
md"""

## Demon of EM (another dataset)
"""

# ╔═╡ 9b27b240-9e2a-4b49-a643-83739d468c5b
begin
	plot(ys, ts, st=:scatter, ratio =1, label="data", xlabel=L"x", ylabel=L"y", title="Dataset without ground truth", xlim =[-1.2, 1.2], markersize =4, alpha=0.5)
end

# ╔═╡ 1e8f4f6e-42fe-442e-92c0-d44699904462
# begin
# 	gr()
# 	kmAnim₄ = kmeansDemoGif(data₄, 9, 25; init_step="u", add_contour=false, seed=234);
# end;

# ╔═╡ 74557480-f940-4c60-8b12-8255a307c310
md"""

### K-means result
"""

# ╔═╡ 09643fce-8711-4fe9-bd65-7e2cf2494ee1
show_img("kmeans.gif")

# ╔═╡ 99deb886-deea-4528-967c-7b9330da648d
# begin
# 	gr()
# 	# mixGaussiansDemoGif(data, K, iters = 10; init_step="e", add_contour=false, seed=123)
# 	emAnim₄ = mixGaussiansDemoGif(data₄, 9, 100; init_step="m", add_contour=true, seed=3456, every = 5);
# end;

# ╔═╡ 44222f14-d4c3-4556-88b0-5021091e64a8
md"""

### GMM with EM result
"""

# ╔═╡ 1c9b4d6f-63d1-42a2-9511-550048b67bf4
# gif(emAnim₄[1], fps=5)

# ╔═╡ 58764f2b-e4e9-4f96-8f96-03f4041a5dd8
show_img("emgmm3.gif")

# ╔═╡ 0572eb7c-e6d3-4c28-8531-4619720e7592
md"""
## Demon of EM  (local optimum)



##### EM might converge to a local optimum
"""

# ╔═╡ bc075093-632f-4554-a100-15b43e6d679b
# gif(mixAnims_localmin[1], fps=10)

# ╔═╡ b6a5836b-d14c-4168-a342-31af6acf0a55
show_img("emgmm4.gif")

# ╔═╡ fce33a12-a40a-4cc3-9bf0-d5c8d388b649
md"""
## Local optimum

#### *EM* might get trapped at local optimums 
* *e.g.* one extreme initialisation example: all $z^{(i)}=1$; i.e. all data assigned to one cluster
  * mixture collapses to one singular Gaussian
* no improvement can be made even in the first iteration
  
#### *Solution*: repeat the algorithm a few times with different random initialisations
* and use likelihood as a guide to find the best model


"""

# ╔═╡ 05c2d1ac-d447-4942-940d-4f4052e66eeb
# begin
# 	gr()
# 	# change the seed to 
# 	mixAnims_localmin = mixGaussiansDemoGif(data₂, K₂, 100; init_step="e", add_contour=true, seed=999123)
# end;

# ╔═╡ 23ebe05e-e8b1-47bb-b918-ac390e21fd0b
md"""

## *Implementation

"""

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
	rs = exp.(logPost .- logsums)
	# return the responsibility matrix and the log-likelihood
	return rs, sum(logsums)
end

# ╔═╡ 27755688-f647-48e5-a939-bb0fa70c95d8
function m_step(data, rs)
	_, d = size(data)
	K = size(rs)[2]
	ns = sum(rs, dims=1)
	πs = ns ./ sum(ns)
	# weighted sums ∑ rᵢₖ xᵢ where rᵢₖ = P(zᵢ=k|\cdots)
	ss = data' * rs
	# the weighted ML for μₖ = ∑ rᵢₖ xᵢ/ ∑ rᵢₖ
	μs = ss ./ ns
	Σs = zeros(d, d, K)
	for k in 1:K
		error = (data .- μs[:,k]')
		# weighted sum of squared error
		# use Symmetric to remove floating number numerical error
		Σs[:,:,k] =  Symmetric((error' * (rs[:,k] .* error))/ns[k])
	end
	# this is optional: you can just return μs and Σs
	mvns = [MvNormal(μs[:,k], Σs[:,:,k]) for k in 1:K]
	return mvns, πs[:]
end

# ╔═╡ 2a539d0d-2bdc-4af0-b96f-52676393b458
md"""
## Choosing $K$


#### As $K \rightarrow \infty$, the likelihood will increase (out of bound)

* when $K =n$, each observation "occupies" a Gaussian with ``0`` variance
  * the likelihood is infinite
* no surprise: likelihood based method favours complicated models, i.e. overfitting

"""

# ╔═╡ 16d4218c-7158-447d-8e3d-440a5d323801
md"""
## Choosing $K$ -- BIC


#### *Bayesian information criteria (BIC)*

$$\large\text{BIC}(\mathcal M) = \ln p(\mathcal{D}|\Theta_{\text{MLE}}, \mathcal M) -\frac{\text{dim}}{2}\ln n$$

* ``\mathcal M``: model under consideration, *e.g.* $K=1,2,3\ldots$ for mixture
* ``\text{dim}``: the total number of parameters
  * therefore complicated models are penalised
* ``n``: number of training samples





"""

# ╔═╡ ceb4d9cc-2e32-4f7d-b832-2de8951feb1d
function bic_mix_gaussian(logL, K, d, n)
	# dim(π): K-1; 
	# dim(μ) *K: d*K; 
	# dim(Σ) *K : (d+1)*d/2, a symmetric matrix
	dim = (K-1) + (d + d*(d+1)/2) * K
	logL - dim/2 * log(n)
end

# ╔═╡ 97355933-69bb-4e68-9ba1-956fd3684d0f
md"""
## General mixture model



"""

# ╔═╡ 0b07fda3-8c56-4224-b685-763ac06558ae
TwoColumn(md"""
#### We do not have to assume Gaussian

$\Large p(x)= \sum_{k=1}^K \pi_k \cdot \underbrace{p(x|z=k,\phi_k)}_{\text{can be any dist.}}$

* ##### ``p(x|z=k,\phi_k)`` can be any distribution
  * Bernoullis, von-Mises Fisher, 
  * even linear regresion & neural nets

* ##### all the other assumptions are the same
* in P2, you are going to implement an EM algorithm for mixture of Bernoullis


""", 

md"""
\
\
\
$(show_img("gmmix.svg", w=220))


""")

# ╔═╡ 5bc9d85c-27c0-42e2-a233-15b58e9bebe8
md"""

## EM algorithm for general mixture

#### *E step*

$\large r_{ik} \leftarrow {p(z^{(i)}=k|x^{(i)})}$

#### *M step*

$$\large \hat \pi_k \leftarrow \frac{\sum_{i=1}^n r_{ik}}{n},\;\; \hat \phi_k \leftarrow \arg\max_{\phi} \sum_{i=1}^n {r_{ik}} \cdot \ln p(x^{(i)}|\phi)$$


* ##### solving *weighted MLE* for $\phi_k$

"""

# ╔═╡ 6d919076-958f-4ba8-9089-92df4a27d030
# md"""

# ## *Towards unsupervised learning of mixture

# Let's consider **supervised learning** first, i.e. assume we had the labels, **maximum likelihood estimation** aims at optimising

# $$\hat \theta = \arg\max_{\theta} P(D|\theta)$$

# * observed data: ``D=\{x^i, z^i\}_{i=1}^n``, labels are observed
# * model parameters: ``\theta = \{\pi_k, \phi_k\}_{k=1}^K``

# The likelihood becomes

# $P(D|\theta) = \prod_{i=1}^n p(z^i, x^i) = \prod_{i=1}^n p(z^i)p(x^i|z^i)$

# Take log and after some algebra, it can be shown that (check appendix for details)

# $$\mathcal L(\theta) = \ln P(D|\theta)=\sum_{i=1}^n \sum_{k=1}^K {I(z^i=k)} \cdot \ln \pi_k+ \sum_{i=1}^n \sum_{k=1}^K {I(z^i=k)} \cdot \ln p(x^i|\phi_k)$$


# To optimise $\phi_k$, i.e. find $$\frac{\partial \mathcal L}{\partial \phi_k}$$, we isolate the terms of $\phi_k$ only

# $$\mathcal{L}(\phi_k) = \sum_{i=1}^n {I(z^i=k)} \cdot \ln p(x^i|\phi_k) + \text{const.}$$

# Therefore, the MLE for $\phi_k$ is 

# $\hat \phi_k \leftarrow \arg\max_{\phi} \sum_{i=1}^n {I(z^i=k)} \cdot \ln p(x^i|\phi)= \arg\max_{\phi} \ln P(D_k|\phi)$

# * ``D_k =\{ x^i| z^i=k, \text{ for } i \in 1,\ldots, n\}``
# * the MLE of the those data belong to the k-th class!


# In summary, for **supervised learning**, **maximum likelihood estimators** for mixture model is 

# $$\hat \pi_k = \frac{\sum_{i=1}^n I(z^i= k)}{n}, \hat \phi_k \leftarrow \arg\max_{\phi} \sum_{i=1}^n {I(z^i=k)} \cdot \ln p(x^i|\phi)$$

# """

# ╔═╡ 9d083ce9-72ef-4283-b757-ee331d7b89f2
# md"Input $x^{(i)}=$ $(xᵢ0); and ``\sigma^2=`` $(σ²0)"

# ╔═╡ af80e90a-380e-427d-b6d4-dfd791628b6f
md"""
## Recap: probabilistic linear regression model


"""

# ╔═╡ 0e79dfba-30eb-4eb0-9036-b59d0a66e52b
TwoColumn(md"""

> $\large\begin{align}p(y^{(i)}|\mathbf{x}^{(i)}, \mathbf{w}, \sigma^2) &= \mathcal{N}(y^{(i)};  \mathbf{w}^\top \mathbf{x}^{(i)} , \sigma^2)\end{align}$
> * ``y^{(i)}`` is a univariate Gaussian with mean $\mathbf{w}^\top \mathbf{x}^{(i)}$ and variance $\sigma^2$ 

""", 
	md"""

	
$(show_img("problinreg.svg", w=330))
	"""
)

# ╔═╡ f20a3dd2-c34c-44bf-8921-b3a736e1d433
md"``x_i`` $(@bind xᵢ0 Slider(-0.5:0.1:1, default=0.15));	``\sigma^2`` $(@bind σ²0 Slider(0.005:0.01:0.15, default=0.05))"

# ╔═╡ b9cd7a0f-f9ee-4da2-92d9-f47f8bc4ce28
let
	gr()
	Random.seed!(123)
	true_w = [1.0, 1.0]
	β0 = true_w
	n0 = 100
	xx = range(-0.5, 1; length = n0)
	yy = β0[1] .+ xx * β0[2] + sqrt(σ²0) * randn(n0)

	# b_1 = 3.0
	p_lr = plot(title="Probabilistic linear regression model",legend=:bottomright)
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
			scatter!([x],[μi], markerstrokewidth =2, markershape = :diamond, c=:red, label=L"\mu(x)", markersize=6)
			plot!(ys_ .+x, xs_, c=:red, label="", linewidth=3)
		else
			plot!(ys_ .+x, xs_, c=:gray, label="", linewidth=1)
			# scatter!([x],[μi], markerstrokewidth =2, markershape = :diamond, label="μ @ x="*string(x))
		end
		
	end
	p_lr	
end

# ╔═╡ 39d00409-608a-4b13-8131-4f157999d774
md"""

## Example- mixture of regressions

##### What if your data looks like this ?
* ##### one linear regression fit is bad
* ##### quite likely: $z^{(i)}\in 1,\ldots,K$ are missing when collecting the data
  * _e.g._ each observation's gender, or some other categorical feature
  * or due to privacy issue, cannot be collected at all!
"""

# ╔═╡ 0722858a-e2e1-4031-8537-729f4f578175
begin
	nₘₗᵣ = 600
	trueβs = [[-3, 4.0] [4.0, -6.0] [-1, 8.0]]
	trueσ²s = [0.25, 0.25, 1.0]
	X = [ones(nₘₗᵣ) rand(nₘₗᵣ)]
	trueπsₘₗᵣ = [0.4, 0.4, 0.2]	
	truezsₘₗᵣ = rand(Categorical(trueπsₘₗᵣ), nₘₗᵣ)	
	Y = zeros(nₘₗᵣ)
	Kₘₗᵣ = 3
	for i in 1:nₘₗᵣ
		zᵢ = truezsₘₗᵣ[i]
		Y[i] = rand(Normal(X[i,:]' * trueβs[:, zᵢ], sqrt(trueσ²s[zᵢ])))
	end
	dataₘₗᵣ = [X Y]
end;

# ╔═╡ 9659b077-a7fc-4511-922a-7e5dd0a09b38
plot(X[: ,2], Y, st=:scatter, label="", title="Dataset ", xlabel=L"x", ylabel=L"y")

# ╔═╡ d99c9fe9-0444-4a56-988a-ef805307b4f9
md"""
## Example- mixture of regression (conti.)



"""

# ╔═╡ a7aaf298-3d86-4913-a430-fc7dd22efd1a
TwoColumn(md"""
#### Finite mixture of regressions 

$$\large\begin{align} p(y^{(i)}|\mathbf{x}^{(i)}) 
&=\sum_{k=1}^K \pi_k\cdot  {\mathcal{N}(y^{(i)}; \mathbf{w}_k^\top \mathbf{x}^{(i)}, \sigma_k^2)}\end{align}$$

- ``y^{(i)}`` can take one of ``K`` possible regression models  
- ``\phi_k=\{\mathbf{w}_k, \sigma_k^2 \}`` are the ``K`` regression components' parameters
  

""", md"""
\

$(show_img("mreg_dag.svg", w=300))
""")

# ╔═╡ 0a960801-802d-44f3-a3ff-6c3ac3436b5d
let
	p_mlr = plot(title="Mixture of linear regression with true zs", xlabel=L"x", ylabel=L"y", framestyle=:origins)
	for k in 1:Kₘₗᵣ
		plot!(X[truezsₘₗᵣ .== k ,2], Y[truezsₘₗᵣ .==k], st=:scatter, c=k,alpha=0.5, ms=2,  label="")
		plot!([0,1], x->trueβs[1,k]+trueβs[2,k]*x, c=k, linewidth=4, label="")
	end
	ii = 120
	xis_ = [X[ii, 2]]
	for k in 1:Kₘₗᵣ
		x = xis_[1]
		μi = dot(trueβs[:, k], [1, x])
		xs_ = μi-3:0.01:μi+3
		ys_ = pdf.(Normal(μi, 3*sqrt(σ²0)), xs_)
		ys_ = 0.1 *ys_ ./ maximum(ys_)
		scatter!([x],[μi], markerstrokewidth =3, markershape = :diamond, c=k, label=L"μ_{%$k}", markersize=4)
		plot!(ys_ .+x, xs_, c=k, label="", linewidth=2)
	end
	scatter!([xis_], [Y[ii]], markersize = 4, markershape=:xcross, markerstrokewidth=3, c= :black, label=L"y^{(i)}")
	p_mlr
end

# ╔═╡ 79a87663-0a9c-454a-bd83-3750562043c6
# md"""

# ## EM algorithm for general mixture

# Initilisation: random guess ${\Theta} \leftarrow \{{\pi_k}, \phi_k\}_{k=1}^K$


# * Expectation step (E step): for $i= 1\ldots n,\; k= 1\ldots K$
# $$r_{ik} \leftarrow p(z^{(i)}=k|{x}^{(i)}) = \frac{\pi_k \cdot p(x^{(i)}|\phi_k)}{\sum_{j=1}^K \pi_j \cdot p(x^{(i)}| \phi_j)}$$


# * Maximisation step (M step): update ${\theta}$, for $k=1\ldots K$

# $\pi_k \leftarrow \frac{1}{n}\sum_{i=1}^n r_{ik}$

# ${\phi}_{k} \leftarrow \arg\max_{\phi} \sum_{i=1}^n r_{ik} \ln p(x^{(i)}|\phi)$


# Repeat above two steps until converge

# """

# ╔═╡ a8c88ad6-b805-461d-bf75-f5f6dbc76fd8
# md"""

# * E-step is almost the same for all mixture model (you only need to change the likelihood)
# * M-step is **weighted maximum likelihood** estimation
#   * some weighted MLE have closed form solution (like Gaussian)
#   * if not, we need to apply gradient descent in the M step
#     * e.g. mixture of logistic regression 

# """

# ╔═╡ 96c5c8f8-df01-439f-9827-aad090437a31
md"""
## EM algorithm for finite mixture of regression

##### Initilisation: random guess ${\Theta} \leftarrow \{{\pi_k}, \mathbf{w}_k, \sigma^2_k\}_{k=1}^K$


##### Repeat until converge


* ##### Expectation step (E step): for $i= 1\ldots n,\; k= 1\ldots K$
$$\large\begin{align}r_{ik} \leftarrow p(z^{(i)}=k|y^{(i)}, \mathbf{x}^{(i)}) 
&\propto {\pi_k \cdot {\mathcal{N}(y^{(i)}; \mathbf{w}_k^\top \mathbf{x}^{(i)},    \sigma_k^2)}}
\end{align}$$


* ##### Maximisation step (M step): update ${\theta}$, for $k=1\ldots K$

$\large\pi_k \leftarrow \frac{1}{n}\sum_{i=1}^n r_{ik}$

$\large\mathbf{w}_{k},\sigma_k^2 \leftarrow \arg\max_{\mathbf{w}, \sigma^2} \sum_{i=1}^n r_{ik} \ln {\mathcal{N}(y^{(i)}; \mathbf{w}^\top \mathbf{x}^{(i)},    \sigma^2)}$



"""

# ╔═╡ 7b82382a-e3fb-4736-a07a-4b92ab002c74
Foldable("M step details", 

md"""

##### The solution to the weighted MLE is (check this week's tutorial)

$$\large\boxed{{\mathbf{w}_{k} \leftarrow (\mathbf{X}^\top \texttt{diag}(\mathbf{r}_k)\, \mathbf{X})^{-1}\cdot \mathbf{X}^\top  \texttt{diag}(\mathbf{r}_k)\, \mathbf{y}}}$$
$$\large\boxed{\sigma_k^2 \leftarrow \frac{1}{n_k} \sum_{i=1}^n r_{ik} \cdot ({y}^{(i)} - \mathbf{w}_k^\top \mathbf{x}^{(i)})^2}$$ 

* where  
$\texttt{diag}(\mathbf{r}_k) = \begin{bmatrix} r_{1k} & 0 & \ldots & 0 \\
0 & r_{2k} & \ldots & 0 \\
\vdots & \vdots & \vdots & \vdots \\ 
0 & 0 & \ldots & r_{nk}
\end{bmatrix}_{n\times n}$





""")

# ╔═╡ ed4c725d-c7bd-4231-adee-0153c626eb69
Foldable("Sanity check", md"""




* It makes sense, if we set all weights to 1  *i.e.* $r_{ik}=1$, or $\texttt{diag}(\mathbf{r}_k)=\mathbf{I}$  then, we **recover** the normal unweighted MLE for linear regression!




$$\mathbf{w}=(\mathbf{X}^\top \texttt{diag}(\mathbf{r}_k) \mathbf{X})^{-1} \mathbf{X}^\top  \texttt{diag}(\mathbf{r}_k) \mathbf{y}=(\mathbf{X}^\top \mathbf{I} \mathbf{X})^{-1} \mathbf{X}^\top  \mathbf{I} \mathbf{y} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top  \mathbf{y}$$

""")

# ╔═╡ 539554be-70f1-4985-a88d-35a7f201792e
# wsₘₗᵣ[iₜₕ,:]

# ╔═╡ 65a6081d-2d5a-4dbf-8040-7156ecd2c649
md"""
## Demonstration of E-step

"""

# ╔═╡ 223371fa-aa87-481f-b483-8854fbeb1ff2
md"Choose i-th observation: $(@bind iₜₕ Slider(1:length(truezsₘₗᵣ), default=1))"

# ╔═╡ ee7991d5-496d-4274-9f8d-7c0e8686822a
xx_idx_orders = sortperm(X[:, 2]);

# ╔═╡ cf6f3ff3-4209-49f9-8b78-1cc1307a7c78
begin
	gr()
	p_mlr_ = plot(title="E-step of mixture of regression", legend=:topleft, xlabel=L"x", ylabel=L"y", xlim =[-0.02, 1.11], ylim =[-6, 8.9])
	for k in 1:3
		plot!(X[truezsₘₗᵣ .== k ,2], Y[truezsₘₗᵣ .==k], st=:scatter, alpha=0.3, ms=3,c=k, label="")
		plot!([0,1], x->trueβs[1,k]+trueβs[2,k]*x, c=k, linewidth=4, label="")
	end
	xis_ = [X[xx_idx_orders[iₜₕ], 2]]
	for k in 1:Kₘₗᵣ
		x = xis_[1]
		μi = dot(trueβs[:, k], [1, x])
		xs_ = μi-3:0.01:μi+3
		ys_ = pdf.(Normal(μi, 2*sqrt(σ²0)), xs_)
		ys_ = 0.1 *ys_ ./ maximum(ys_)
		scatter!([x],[μi], markerstrokewidth =3, markershape = :diamond, c=k, label=L"μ_{%$k}", markersize=4)
		plot!(ys_ .+x, xs_, c=k, label="", linewidth=2)
	end
	scatter!([xis_], [Y[xx_idx_orders[iₜₕ]]], markersize = 6, markershape=:xcross, markerstrokewidth=3, c= :black, label=L"y^{(i)}")
	p_mlr_
end

# ╔═╡ b087392e-48ba-43b0-8385-9348a01391c0
md"""

## EM for mixture regressions -- summary
"""

# ╔═╡ 830240dc-393a-4ac1-82a2-ac9041847146
md"""
----

##### Initilisation: random guess ${\Theta} \leftarrow \{{\pi_k}, \mathbf{w}_k, \sigma^2_k\}_{k=1}^K$


##### Repeat until converge


* ##### Expectation step (E step): for $i= 1\ldots n,\; k= 1\ldots K$
$$\begin{align}r_{ik} \leftarrow \frac{\pi_k \cdot {\mathcal{N}(y^{(i)}; \mathbf{w}_k^\top \mathbf{x}^{(i)},    \sigma_k^2)}}{\sum_{j=1}^K \pi_j \cdot {\mathcal{N}(y^{(i)}; \mathbf{w}_j^\top \mathbf{x}^{(i)},   \sigma_j^2)}}
\end{align}$$


* ##### Maximisation step (M step): update ${\Theta}$, for $k=1\ldots K$

$\pi_k \leftarrow \frac{1}{n}\sum_{i=1}^n r_{ik}$

$${\mathbf{w}_{k} \leftarrow (\mathbf{X}^\top \texttt{diag}(\mathbf{r}_k)\, \mathbf{X})^{-1}\cdot \mathbf{X}^\top  \texttt{diag}(\mathbf{r}_k)\, \mathbf{y}}$$
$${\sigma_k^2 \leftarrow \frac{1}{n_k}\left (\sum_{i=1}^n r_{ik} ({y}^{(i)} - \mathbf{w}_k^\top \mathbf{x}^{(i)})^2\right ) }$$

----
"""

# ╔═╡ 6ec8901b-03fe-42c7-b302-2bfb984d2360
md"""
## Implementation in Julia

"""

# ╔═╡ 2aea1227-6155-4bf9-85bb-4d8a0c7f999b
md"##### E-step is almost the same, only differ in the likelihood part"

# ╔═╡ da784e69-0524-44cc-a726-70b576c5a570
function e_step_mix_reg(data, Ws, σ²s, πs)
	X = data[:, 1:end-1]
	y = data[:, end]
	K = length(πs)
	# this is the only line that is different from EM mixture Gaussians
	logLiks = hcat([logpdf.(Normal.(X * Ws[:,k], sqrt(σ²s[k])), y) for k in 1:K]...)
	logPost = log.(πs') .+ logLiks
	logsums = logsumexp(logPost, dims=2)
	ws = exp.(logPost .- logsums)
	return ws, sum(logsums)/length(y)
end

# ╔═╡ 74ee7c89-d53c-45b5-a01c-44f91a061745
wsₘₗᵣ,_ = e_step_mix_reg(dataₘₗᵣ, trueβs, trueσ²s, trueπsₘₗᵣ) ;

# ╔═╡ 1ec00e74-5a6d-4352-93a2-e54e786dd969
md"``p(z^{(i)}|y^{(i)}, \mathbf{x}^{(i)})=``$(latexify_md(round.(wsₘₗᵣ[xx_idx_orders[iₜₕ],:]; digits=4))) for the ``i=`` $(iₜₕ)th observation is"

# ╔═╡ 541a1722-fd87-4217-961a-48dd271a3b1f
md"##### Direct translation of the M step to code"

# ╔═╡ ab41aa0a-a034-47bb-8ed3-c3224d7ae20a
function m_step_mix_reg(data, rs)
	n, K = size(rs)
	# the design matrix
	X = data[:, 1:end-1]
	_, d = size(X)
	# the targets 
	y = data[:, end]
	Ws = zeros(d,K)
	σ²s = ones(K)
	ns = sum(rs, dims=1)
	# compute π
	πs = ns ./sum(ns)
	for k in 1:K
		Rₖ = diagm(rs[:,k])
		# weighted least square estimations
		Ws[:, k] = wₖ =(X'* Rₖ * X) \ (X' * Rₖ * y)
		σ²s[k] = sum(rs[:,k] .* (y - X * wₖ).^2) / ns[k]
	end
	return πs[:], Ws, σ²s
end

# ╔═╡ 661ab8e4-f223-4361-91ca-27741e7fde5e
md"""

## Demonstration
"""

# ╔═╡ 01dfacdc-b59d-4382-b451-db6035949c36
begin
	gr()
	plEMₘₗᵣ = []
	zs0 = rand(1:Kₘₗᵣ, size(dataₘₗᵣ)[1])
	rs0 = Matrix(I, Kₘₗᵣ, Kₘₗᵣ)[zs0,:]
	l_ = Inf
	anim = @animate for iter in 1:30
		πs0, Ws0, σ²s0 = m_step_mix_reg(dataₘₗᵣ, rs0)
		p = plot(title="Iteration: "*string(iter)*L"\,;\; L=%$(round(l_, digits=2))")
		for k in 1:Kₘₗᵣ
			plot!(X[zs0 .== k ,2], Y[zs0 .==k], st=:scatter, c=k, label="", alpha=0.65)
			plot!([0,1], x->Ws0[1,k]+Ws0[2,k]*x, c=k, linewidth=4, label="")
		end
		
		rs0, l_ = e_step_mix_reg(dataₘₗᵣ, Ws0, σ²s0, πs0)
		zs0 = argmax.(eachrow(rs0))
		push!(plEMₘₗᵣ, p)
	end
end

# ╔═╡ 12f42d6e-1d67-4de7-9527-4d638eaac7e3
gif(anim, fps=5)

# ╔═╡ 90d26dc8-8eb3-425c-8d5b-ba150265f796
md"""

# Why EM works?*
"""

# ╔═╡ 0b79e6d4-7c5e-49be-b780-9ea4295c7029
md"""

## What is expectation?


##### Assume we have a unit-length stick
* ##### we break it up randomly 
* ##### what is the average length of the first half?
"""

# ╔═╡ 873230c7-13a9-4fb0-b4f9-fce7a7a4397d
md"""Break it: $(@bind break_it CheckBox(false)), Mark first half: $(@bind add_first_half CheckBox(false))"""

# ╔═╡ 21913109-795c-4a91-868a-41fa0de38a12
let
	Random.seed!(122334)
	n = 1
	plt = plot([0, 1], [0, 0], framestyle=:none, label="", c=:black, lw=5, alpha=0.2, size=(700,100), ylim =[-0.5, n+1], title="Unit stick")
	
	len_1 = rand()
	if break_it

		scatter!([len_1], [0], marker=:x, markersize=8, c=:red, markerstrokewidth=8, label="")
		if add_first_half
			plot!([0, len_1], [0, 0], label="", c=1, lw=3, alpha=0.5,arrow=Plots.arrow(:both))
			annotate!([0.5 * (len_1)], [0+0.8], text(L"X_{\texttt{first\;\; half}}", 15,:blue,  rotation=0))
		end
	end
	
	plt
end

# ╔═╡ 0d3dfdb4-280a-4316-a425-3f0a1ab32c67
md"""
## What is expectation?


#### Let random variable $X$: the length of the first half

$$\Large p(X) = \begin{cases} 1 & 0\leq X\leq 1 \\ 0 & \text{otherwise}\end{cases}$$

* ##### its average length is "expectation"

* ##### first, by Monte Carlo, (simulate $M$ stick-breakings and compute average)
$$\Large x^{(m)} \sim p(X) = \begin{cases} 1 & 0\leq X\leq 1 \\ 0 & \text{otherwise}\end{cases}$$
$$\Large\mathbb{E}[X] \simeq \frac{1}{M} \sum_{m=1}^M x^{(m)} =\frac{1}{2};\;\; M \rightarrow \infty$$
"""

# ╔═╡ 6b1563f6-226f-4539-8cc9-0d7d7b084999
let
	n = 20
	plt = plot([0, 0], [0, 1], framestyle=:none, label="", c=:black, lw=5, alpha=0.2, size=(700,250), xlim =[-0.5, n+1])
	len_1 = rand()
	plot!([0, 0], [0, len_1], label="", c=1, lw=3, alpha=0.5,arrow=Plots.arrow(:both))
	scatter!([0], [len_1], marker=:x, markersize=8, c=:red, markerstrokewidth=4, label="")

	for i in 1:(n-1)
		# if i != floor(n/2)+1
			plot!([i, i], [0, 1], framestyle=:none, label="", c=:black, lw=5, alpha=0.2)
			len_i = rand()
			plot!([i, i], [0, len_i], label="", c=1, lw=3, alpha=0.5,arrow=Plots.arrow(:both))
			scatter!([i], [len_i], marker=:x, markersize=6, c=:red, markerstrokewidth=4, label="")
		# end
	end

	annotate!([n + .5], [0.5], text(L"\ldots", 40))
	# plot!([0, 0], [len_1, 1], label="", c=2, alpha=0.5, lw=3, arrow=false)
	plt
end

# ╔═╡ 4197312e-cec7-41d4-aebf-a7ba4c7053ce
let
	Random.seed!(123)
	mean(rand(10000)) ## simulate 10000 stick breakings
end

# ╔═╡ e518420b-ec3a-4a77-9ca8-cdbace6c3721
md"""
## What is expectation ?
#### More concretely, the "expectation" is computed as


$$\Large\mathbb{E}[X] = \int_{0}^1 p(X=x)\cdot x \,\mathrm{d}x =\int_{0}^1 x\, \mathrm{d}x = \frac{1}{2}x^2\Big\vert_{0}^1=\frac{1}{2}$$


* a weighted average of stick lengths $x\in [0,1]$ and the weights are $p(X=x)$
"""

# ╔═╡ f338604d-a58f-4888-ace8-718c52ff3275
md"""

## Expectation

#### More generally, the expectation of $f(X)$ is defined as 

$\Large \mathbb{E}_{X\sim p(X)}[f(X)] = \sum_{x} f(x) \cdot p(X=x)$

* basically, it is a weighted average of $f(X)$ and the weights are $p(X=x)$

"""

# ╔═╡ b7ec3a9a-d171-4fd5-9534-4a82903c575e
md"""

## Example


##### Assume we have a unit-length stick (break randomly)
* ##### what is the average length of the _longer half_?
* it should definitely be greater than $\Large\frac{1}{2}$!
"""

# ╔═╡ ad34bb36-c919-4fe9-845b-5e4d5e329c3e
md"Add longer halfs: $(@bind longer_halfs CheckBox(false))"

# ╔═╡ fa67c6b6-a14d-497e-a6f1-b99be84b6ffe
let
	Random.seed!(1234)
	n = 20
	plt = plot(framestyle=:none, label="", c=:black, lw=5, alpha=0.2, size=(700,250), xlim =[-0.5, n+1])
	# len_1 = rand()
	# plot!([0, 0], [0, len_1], label="", c=1, lw=3, alpha=0.5,arrow=Plots.arrow(:both))
	# scatter!([0], [len_1], marker=:x, markersize=8, c=:red, markerstrokewidth=4, label="")

	for i in 0:(n)
		plot!([i, i], [0, 1], framestyle=:none, label="", c=:black, lw=5, alpha=0.2)
		len_i = rand()
		scatter!([i], [len_i], marker=:x, markersize=6, c=:red, markerstrokewidth=4, label="")
		if longer_halfs
			if len_i ≥ 0.5
				plot!([i, i], [0, len_i], label="", c=1, lw=3, alpha=0.5,arrow=Plots.arrow(:both))
				
			else
				plot!([i, i], [len_i, 1], label="", c=1, lw=3, alpha=0.5,arrow=Plots.arrow(:both))
			end
		end
	end

	annotate!([n + 1], [0.5], text(L"\ldots", 40))
	# plot!([0, 0], [len_1, 1], label="", c=2, alpha=0.5, lw=3, arrow=false)
	plt
end

# ╔═╡ b3e448ec-97ea-435b-9790-faa78bf94c63
md"""

#### The length of the longer half is

$$\Large f(X) = \begin{cases}X & X \geq 0.5 \\ 1-X & X < 0.5\end{cases}$$



"""

# ╔═╡ 14ba281d-057a-4485-9794-32466388cee0
md"""

##### Again, let's compute it first by simulation

$$\Large\mathbb{E}_{X\sim p(X)}[f(X)] \simeq \frac{1}{M} \sum_{m=1}^M f(x^{(m)}) =\frac{3}{4};\;\; M \rightarrow \infty$$
"""

# ╔═╡ 7a0654e2-f01b-43d9-b500-b4a7a95eb428
let
	Random.seed!(123)
	## Monte Carlo estimate
	sticks = rand(10000) 
	sticks[sticks .< 0.5] = 1 .- sticks[sticks .< 0.5]
	mean(sticks)
end

# ╔═╡ fecf3e10-ba13-4fea-8fa8-d10097280bab
md"""
#### More formally, by expectation's definition,
"""

# ╔═╡ a2631364-95a2-4d2e-832b-696b54aba50d
Foldable("The expectation", md"""


$$\Large\begin{align}\mathbb{E}[f(X)] &= \int_{0}^1 p(X=x) f(x) \mathrm{d}x \\
&=\int_{0}^{0.5} (1-x)\mathrm{d}x  + \int_{0.5}^1 x\mathrm{d}x  \\
&= (x - \frac{1}{2}x^2)\Big\vert_{0}^{0.5} + \frac{1}{2}x^2\Big\vert_{0.5}^1\\
&=\frac{3}{4}
\end{align}$$



""")

# ╔═╡ bcb78ed1-eb28-4165-89f2-5fba14cf4eb3
# let

# plot(0:0.1:1.0, x -> x ≥ 0.5 ? x : 1-x , xlabel="x", title="", label=L"f(x)",ylabel="", ratio=1, framestyle=:semi, size=(300,300), lw=2, ylim =[0.45, 1.0])
# end

# ╔═╡ 8422af99-c189-4606-a30d-465f0c0f560d
md"""
## Expectation -- summary

#### Expectation of $f(X)$ is defined as 

$\large \mathbb{E}[f(X)] = \sum_{x} f(x) \cdot p(X=x)$

* basically, it is a weighted average


##### Example 1, find the expectation of a Bernoulli r.v. $X$ with bias $p$:

$$\mathbb{E}[X] = 1 \cdot p(X=1) + 0 \cdot p(X=0) = 1\cdot p + 0 \cdot (1-p) = p$$

* ``f`` is a identity function

##### Example 2, assume $p(z=k) = r_k$, i.e. $z$ is a categorical r.v. then

$\large\begin{align}\mathbb{E}[\mathbb{1}(z=k)] &= \sum_{k'=1}^K \mathbb{1}(z=k) \cdot p(z=k') \\
&=0\cdot r_1 + \ldots+ 1 \cdot r_k + \ldots 0 \cdot r_{K}\\
&= r_k
\end{align}$

* the expected value of $\mathbb{1}(z=k)$ is just $r_k$
"""

# ╔═╡ 7bd7f0ef-9060-47da-89ed-14b964c85354
md"""

## Expectation -- summary (cont.)

#### Facts about expectations

* expectation of a constant is constant itself
$$\mathbb{E}[c] =c$$

* expectation is a linear operator

$$\mathbb{E}[aX_1 + bX_2] =a \mathbb{E}[X_1] +b  \mathbb{E}[X_2]$$



$$\mathbb{E}\left[\sum_{i=1}r_i X_i\right] =r_i \sum_{i=1}^n \mathbb{E}[X_i]$$

"""

# ╔═╡ 46921d8b-6111-44fa-a1d5-607e6d95f0d6
md"""

## EM algorithm



#### The marignal log-likelihood is hard to compute

$$\Large\mathcal{L}(\theta) =\ln p(X |\theta) = \ln \int_Z p(X, Z|\theta) \mathrm{d}Z\;\;\;\;\; \text{hard to compute!}$$



#### The _complete_ log-likelihood is easy to compute

$$\large\ln p(X, Z|\theta) = \ln \left (\prod_{i=1}^n p(X_i|\text{parent}(X_i))\right )\;\;\;\;\; \text{easy to compute!}$$



"""

# ╔═╡ 70926550-6e20-41dc-bba2-3ecd390e69cf
md"""

## EM algorithm



#### The marignal log-likelihood is hard to compute

$$\Large\mathcal{L}(\theta) =\ln p(X |\theta) = \ln \int_Z p(X, Z|\theta) \mathrm{d}Z\;\;\;\;\; \text{hard to compute!}$$



#### The complete log-likelihood is easy to compute

$$\large\ln p(X, Z|\theta) = \ln \left (\prod_{i=1}^n p(X_i|\text{parent}(X_i))\right )\;\;\;\;\; \text{easy to compute!}$$



#### EM optimises the "average" of the complete log-likelihood 

$$\Large Q(\theta, \theta_t) = \mathbb{E}_{Z\sim p(Z|X, \theta_t)}\left[\ln p(X, Z|\theta)\right]$$


$$\Large \theta_{t+1} \leftarrow \arg\max_{\theta}Q(\theta, \theta_{t})$$

* ``\ln p(X, Z|\theta)`` is a function of $Z$
* Expectation depends on ``\theta_t``: the current posterior ``p(Z|X, \theta_t)``
"""

# ╔═╡ 4d942d63-eed1-4302-91d8-04a2444b05b1
md"""

## Why $Q(\theta; \theta_t)$ ?


#### It can be proved that 

$$\Large\ln p(X |\theta) \geq Q(\theta; \theta_t) + C$$


* ##### ``Q`` lower-bounds the true log likelihood for all $\theta$

* ##### and when $\theta =\theta_t$, the two functions kiss: $\ln p(X |\theta_t) = Q(\theta_t; \theta_t) +C$



"""

# ╔═╡ 44175f60-6bde-4d4e-b977-318239e7eaa8
show_img("em_illu.svg", w=500)

# ╔═╡ c7a6c2a3-cffc-4cf2-82cb-7264ab67db05
md"""
##

### Optimise $$Q$$ such as

$$\Large \theta_{t+1} \leftarrow \arg\max_{\theta}Q(\theta, \theta_{t})+C$$

* due to $\arg\max$, the new $Q$ should be no worse than the current: 

$\large Q(\theta_{t+1}, \theta_{t}) + C\geq Q(\theta_{t}, \theta_{t}) +C$

##### It garantees that 

$$\large\ln p(X |\theta_{t+1}) \geq \ln p(X |\theta_t)$$
  * **proof by contradiction**: if not, then $\ln p(X |\theta_{t+1}) < \ln p(X |\theta_t)$, which implies 

$$\ln p(X |\theta_{t+1}) < \ln p(X |\theta_t) = Q(\theta_t,\theta_t)+C \leq Q(\theta_{t+1}, \theta_{t})+C$$
* ``Q`` won't be a lower bound anymore since $\ln p(X |\theta_{t+1}) < Q(\theta_{t+1}, \theta_{t})+C$. **Contradiction!**

"""

# ╔═╡ b5c598bd-ed60-4e00-a26c-975faf71c265
md"""

## Proof of the lower bound*

#### We prove the lower bound property here, that is for all $\theta$,

$$\Large\ln p(X |\theta) \geq Q(\theta; \theta_{t}) + C$$


#### Proof:

$$\begin{align}
\ln p(X|\theta)&= \ln \int p(X, Z|\theta) dZ\\
&= \ln \int \frac{p(X, Z|\theta)}{p(Z|X, \theta_t)} p(Z|X, \theta_t) dZ \\
&= \ln \mathbb{E}_{Z\sim p(Z|X, \theta_t)}\left[ \frac{p(X, Z|\theta)}{p(Z|X, \theta_t)}\right] \\
&\geq \mathbb{E}_{Z\sim p(Z|X, \theta_t)} \left [\ln \frac{p(X, Z|\theta)}{p(Z|X, \theta_t)} \right] \\
&= \mathbb{E}_{Z\sim p(Z|X, \theta_t)} \left [\ln {p(X, Z|\theta)} \right] - \mathbb{E}_{Z\sim p(Z|X, \theta_t)} \left [\ln{p(Z|X, \theta_t)}\right ]\\
&= Q(\theta, \theta_t) + C({\theta_t})
\end{align}$$

* line 4 has used Jensen's equality, that is for all $f(Z)$, (here $f(Z)=\frac{p(X, Z|\theta)}{p(Z|X, \theta_t)}$)

$$\ln \mathbb{E}[f(Z)] \geq \mathbb{E} \left[ \ln f(Z) \right ]$$

* line 5 has used $\ln \frac{a}{b} =\ln a-\ln b$

* where the constant $C$, depending on $\theta_t$, is the entropy of $p(Z|X, \theta_t)$; it is a constant from $\theta$'s perspective
"""

# ╔═╡ f0fd83b4-9b15-4ce7-b220-60c3ff0bfea0
md"""

#### Exercise: prove that when $\theta=\theta_t$, the two functions "kiss", *i.e.*

$$\Large\ln p(X |\theta_t) = Q(\theta_t; \theta_t) + C$$

"""

# ╔═╡ 2af8276a-3aa7-4e43-8e76-694ac5131e72
Foldable("Solution", md"""

Recall the definition of $Q(\theta, \theta_t)$:

$$Q(\theta, \theta_t) = \mathbb{E}_{Z\sim p(Z|X, \theta_t)}\left[\ln p(X, Z|\theta)\right],$$ and recall from the previous slide: the constant gap $$C = - \mathbb{E}_{Z\sim p(Z|X, \theta_t)} \left [\ln{p(Z|X, \theta_t)}\right ]$$


$$Q(\theta, \theta_t) + C = \mathbb{E}_{Z\sim p(Z|X, \theta_t)}\left[\ln p(X, Z|\theta)\right] - \mathbb{E}_{Z\sim p(Z|X, \theta_t)} \left [\ln{p(Z|X, \theta_t)}\right ]$$


When $\theta=\theta_t$, we have

$$\begin{align}Q(\theta_t, \theta_t) + C &= \mathbb{E}_{Z\sim p(Z|X, \theta_t)} \left [\ln \frac{p(X, Z|\theta_t)}{p(Z|X, \theta_t)} \right] \\
&= \mathbb{E}_{Z\sim p(Z|X, \theta_t)} \left [\ln \frac{\cancel{p(Z|X, \theta_t)}p(X|\theta_t)}{\cancel{p(Z|X, \theta_t)}} \right] \\
&=\mathbb{E}_{Z\sim p(Z|X, \theta_t)} \left [\ln p(X|\theta_t) \right]\\
&=\ln p(X|\theta_t)
\end{align}$$


* line two has used product rule for the numerator

* the last line has used the fact that a constant's expectation is itself: $$\mathbb{E}[c] =c$$; note that $\ln p(X|\theta_t)$ is a normalising constant which is independent from $Z$
""")

# ╔═╡ 71e543d9-5be8-4f78-9097-f4aad7885d5a
md"""
## Revisit EM for general mixture models*

Here, we wrap up EM algorithm by revisiting the algorithm for general mixture models. We are going to justify why the EM algorithm for a general mixture should follow the E and M steps as discussed.

"""

# ╔═╡ 2356d8e1-8854-440b-b861-90b71685a258
md"""
First of all, note that by using the cases to product trick, we have

$$p(z^{(i)}) = \prod_{k=1}^K \pi_k^{\mathbb{1}(z^{(i)}=k)}; \;\; p(x^{(i)}|z^{(i)}) = \prod_{k=1}^K p(x^{(i)}|\phi_k)^{\mathbb{1}(z^{(i)}=k)}$$


Then the complete log-likelihood becomes

$$\begin{align}\ln p(\{{x}^{(i)}, z^{(i)}\}_{i=1}^n|\Theta) &=  \ln \prod_{i=1}^np({x}^{(i)}, z^{(i)}|\{\phi_k\}, \pi)\\
&=\ln \prod_{i=1}^n p(z^{(i)}|\pi)p({x}^{(i)}| z^{(i)},\{\phi_k\})\\
&= \sum_{i=1}^n \ln p(z^{(i)}|\pi) + \ln p({x}^{(i)}| z^{(i)},\{\phi_k\})\\
&= \sum_{i=1}^n \sum_{k=1}^K {\mathbb{1}(z^{(i)}=k)} \cdot \ln \pi_k+ \sum_{i=1}^n \sum_{k=1}^K {\mathbb{1}(z^{(i)}=k)} \cdot \ln p(x^{(i)}|\phi_k)
\end{align}$$

EM optimises the surrogate $Q(\Theta, \Theta^{(t)})$, which is the expectation of the  complete log likelihood

$\begin{align}Q(\Theta, \Theta^{(t)}) &= \mathbb{E}\left[ \ln p(\{{x}^{(i)}, z^{(i)}\}_{i=1}^n|\Theta)\right ]\\
&= \mathbb{E}\left[ \sum_{i=1}^n \sum_{k=1}^K {\mathbb{1}(z^{(i)}=k)} \cdot \ln \pi_k+ \sum_{i=1}^n \sum_{k=1}^K {\mathbb{1}(z^{(i)}=k)} \cdot \ln p(x^{(i)}|\phi_k)\right ]\\
&= \sum_{i=1}^n \sum_{k=1}^K \mathbb{E}[{\mathbb{1}(z^{(i)}=k)}] \cdot \ln \pi_k+ \sum_{i=1}^n \sum_{k=1}^K \mathbb{E}[{\mathbb{1}(z^{(i)}=k)}] \cdot \ln p(x^{(i)}|\phi_k)\\
&=\sum_{i=1}^n \sum_{k=1}^K r_{ik} \cdot \ln \pi_k+ \sum_{i=1}^n \sum_{k=1}^K r_{ik}\cdot \ln p(x^{(i)}|\phi_k)
\end{align}$



* to optimise $\phi_k$, we isolate the terms with $\phi_k$ only:

$\begin{align}Q(\Theta, \Theta^{(t)}) 
&=\sum_{i=1}^n r_{ik}\cdot \ln p(x^{(i)}|\phi_k) + C
\end{align}$


* optimising above proves the EM algorithm: it optimises the weighted MLE

$$\phi_k^{(t+1)} \leftarrow \arg\max_{\phi}\sum_{i=1}^n r_{ik}\cdot \ln p(x^{(i)}|\phi_k)$$

* to optimise $\pi$, we need to add a lagrange multiplier (since it is constrained), it can be shown that 

$$\pi^{(t)}_k = \frac{\sum_{i=1}r_{ik}}{n}$$

"""

# ╔═╡ 64d31497-9009-49f2-b132-07a81331ac2f
md"""

## Suggested reading

Machine learning: a probabilistic approach by Kevin Murphy
* 4.2: Gaussian discriminant analysis
* 11.2 and 11.4: mixture of Gaussians 


"""

# ╔═╡ a0465ae8-c843-4fc0-abaf-0497ada26652
md"""

## Appendix

Utility functions
"""

# ╔═╡ 3ea57a8e-8d15-4f41-acfb-e3bd1d65e585
begin
	Random.seed!(123)
	K₁ =3
	n_each = 100
	# D₁ = zeros(n₁, 2)
	# 200 per cluster
	truezs₁ = repeat(1:K₁; inner=n_each)
	trueμs₁ = zeros(2, K₁)
	trueμs₁[:,1] = [-3.0, 2.0]
	trueμs₁[:,2] = [3.0, 2.0]
	trueμs₁[:,3] = [0., -2]
	data₁ = trueμs₁[:,1]' .+ randn(n_each, 2)
	data₁ = vcat(data₁, trueμs₁[:,2]' .+ randn(n_each, 2))
	data₁ = vcat(data₁, trueμs₁[:,3]' .+ randn(n_each, 2))
	# plt₁ = plot(ratio=1, framestyle=:origin)
	# for k in 1:K₁
	# 	scatter!(data₁[truezs₁ .== k,1], data₁[truezs₁ .==k,2], label="Class"*string(k)) 
	# end
	# title!(plt₁, "Supervised learning: classification")
end;

# ╔═╡ f60afe8c-1067-485c-ae3c-b510960bf01a
begin

	plt₁ = plot(ratio=1, framestyle=:origin)
	for k in 1:K₁
		scatter!(data₁[truezs₁ .== k,1], data₁[truezs₁ .==k,2], label="Class"*string(k)) 
	end
	title!(plt₁, "Classification")
end;

# ╔═╡ 077cf7c8-e4a3-4352-b435-36f1b9e44c1b
plt_cluster  = let
	plt = plot(ratio=1, framestyle=:origin, title="Clustering")
	# for k in 1:K₁
	scatter!(data₁[:,1], data₁[:,2], label="data") 
	# end
end;

# ╔═╡ 4cf3cde7-f95c-443d-b8fa-7c3a3ca4ff4d
TwoColumn(plot(plt₁, size=(300,300)), plot(plt_cluster, size=(300,300)))

# ╔═╡ a414e554-3a8c-472d-af82-07c2f0843627
begin
	function assignment_step(D, μs)
		_, K = size(μs)
		distances = hcat([sum((D .- μs[:,k]').^2, dims=2) for k in 1:K]...)
		min_dis, zs_ = findmin(distances, dims=2)
		# zs_ is a cartesian tuple; retrieve the min k for each obs.
		zs = [c[2] for c in zs_][:]
		return min_dis[:], zs
	end

	function update_step(D, zs, K)
		_, d = size(D)
		μs = zeros(d,K)
		# update
		for k in 1:K
			μₖ = mean(D[zs.==k,:], dims=1)[:]
			μs[:,k] = μₖ
		end
		return μs
	end
end;

# ╔═╡ 33cc44b4-bd32-4112-bf33-6807ae53818c
function kmeans(D, K=3; tol= 1e-6, maxIters= 100, seed= 123)
	Random.seed!(seed)
	# initialise
	n, d = size(D)
	zs = rand(1:K, n)
	μs = D[rand(1:n, K),:]'
	loss = zeros(maxIters)
	i = 1
	while i <= maxIters
		# assigment
		min_dis, zs = assignment_step(D, μs)
		# update
		μs = update_step(D, zs, K)
		
		loss[i] = sum(min_dis)

		if i > 1 && abs(loss[i]-loss[i-1]) < tol
			i = i + 1
			break;
		end
		i = i + 1
	end
	return loss[1:i-1], zs, μs
end;

# ╔═╡ 5f9ad998-c410-4358-925b-66e5d3b2f9e9
begin
	Random.seed!(123)
	K₂ = 3
	trueμs₂ = zeros(2,K₂)
	trueΣs₂ = zeros(2,2,K₂)
	trueμs₂[:,1], trueΣs₂[:,:,1] = [-2.0, 0.0], 0.5 * Matrix(1.0I, 2,2)
	trueμs₂[:,2], trueΣs₂[:,:,2] = [2.0, 0.0], 0.5 * Matrix(1.0I, 2,2)
	trueμs₂[:,3], trueΣs₂[:,:,3] = [0., 0],  Matrix([0.5 0; 0 2])
	trueπs₂ = [0.2, 0.2, 0.6]
	truemvns₂ = [MvNormal(trueμs₂[:,k], trueΣs₂[:,:,k]) for k in 1:K₂]
	n₂= 500
	truezs₂ = rand(Categorical(trueπs₂), n₂)
	data₂= vcat([rand(truemvns₂[z])' for z in truezs₂]...)
	# data₂, truezs₂ = sampleMixGaussian(n₂, truemvns₂, trueπs₂)
end;

# ╔═╡ c17c74a0-2b24-4d27-97cb-5407a47bbab1
ns_d2 = counts(truezs₂, 1:3);

# ╔═╡ c0e777da-c3be-42d3-8fa5-a540da94714c
md"Select observation ``i=`` $(@bind idx_ve Slider(1:2:size(data₂)[1], show_value=true))
"

# ╔═╡ b8362a65-1944-470a-9185-09335cd2d94b
let
	gr()
	idx = idx_ve
	# class_d3 =3
	data = data₂
	K = K₂
	zs = truezs₂
	# μs, Σs, πs = QDA_fit(data, zs)
	μs, Σs, πs = trueμs₂, trueΣs₂, trueπs₂
	mvns = [MvNormal(μs[:, k], Symmetric(Σs[:,:, k])) for k in 1:size(μs)[2]]
	# plt = plot(-6:.05:6, -6:0.05:6, (x,y) -> e_step([x, y]', mvns, πs)[1][:] |> argmax,  c=1:K, lw=1, alpha=0.8, title="Visualise E step", st=:heatmap, colorbar=true, ratio=1, framestyle=:origin)

	plt = plot(ratio=1, framestyle=:origin)

	for k = 1:K
		# plot!(-6:.05:6, -6:0.05:6, (x,y)-> qdform([x,y]-mvns[k].μ, inv(mvns[k].Σ)), levels=[2.0],  st=:contour, colorbar = false, ratio=1, color=k, linewidth=3) 
		scatter!([mvns[k].μ[1]], [mvns[k].μ[2]], color = k, label = "cluster "*string(k), markersize = 1, markershape=:circle, markerstrokewidth=0.1)
	end
	qz = e_step(data, mvns, πs)[1]
	colors = [RGB(r,g,b) for (b,r,g) in eachrow(qz)]
	scatter!(data[:, 1], data[:, 2], c=colors, ms=4, alpha=0.9, label="")
	# if add_bd
	# 	plot!(-6:.05:6, -6:0.05:6, (x,y) -> e_step([x, y]', mvns, πs)[1][:] |> argmax,  c=1:K, lw=1, alpha=0.4, st=:heatmap, colorbar=false, xlim=[-6,6])
	# end
	scatter!(data[idx:idx, 1], data[idx:idx, 2], c=:black, ms=8, markershape=:x, markerstrokewidth=4, alpha=1, label=L"x^{(i)}", title=L"p(z|\mathbf{x}) = %$(round.(qz[idx, :], digits=2))", titlefontsize=18)


	# plot!(-3.5:.05:3.5, -4:0.05:4, (x,y) -> e_step([x, y]', mvns, πs)[1][3],  c=:jet, levels=4, lw=1, alpha=0.5, title="Visualise E step", st=:contour, fill=false, colorbar=false, ratio=1, framestyle=:origin)
	# plot!(-6:.05:6, -6:0.05:6, (x,y) -> e_step([x, y]', mvns, πs)[1][2],  c=1:K, lw=1, alpha=0.8, title="Visualise E step", st=:contour, fill=true, colorbar=false, ratio=1, framestyle=:origin)
	# plot!(-6:.05:6, -6:0.05:6, (x,y) -> e_step([x, y]', mvns, πs)[1][3],  c=1:K, lw=1, alpha=0.8, title="Visualise E step", st=:contour, fill=true, colorbar=false, ratio=1, framestyle=:origin)

end

# ╔═╡ c75fd76e-8581-4dcf-808a-17c133b6fadc
md"Select cluster ``k=`` $(@bind k_idx Slider(1:K₂, show_value=true)),
Sum ``\mathbf{r}_k = \sum_i r_{ik}``: $(@bind add_sum CheckBox(default=false))
"

# ╔═╡ fdf3dda3-afc3-43c6-b58e-45e1ebf39369
let
	gr()
	data = data₂
	K = 3
	dim = 2
	# ms = trueμs₂
	# mvns = [MvNormal(ms[:,k], trueΣs₂[:,:,k]) for k in 1:K]
	Random.seed!(123)
	ms = trueμs₂ .+ randn(dim, K)/2
	mvns = [MvNormal(ms[:,k], 2*trueΣs₂[:,:,k]) for k in 1:K]
	qz, l = e_step(data, mvns, trueπs₂)
	nks = sum(qz, dims=1)
	k = k_idx
	# colors = [RGB(r,g,b) for (b,r,g) in eachrow(qz)]
	title = add_sum ? "Visualise "*L"\mathbf{r}_{%$(k)}"*";"*L"\;\sum {r}_{i%$(k)} = %$(round(nks[k]; digits=2))" : "Visualise "*L"\mathbf{r}_{%$(k)}"
	plt = plot(ratio=1, framestyle=:origin, title=title)
	# scatter!(data[:, 1], data[:, 2], c=truezs₂, ms=4, alpha=0.1, label="")
	
	scatter!(data[:, 1], data[:, 2], c=k, ms=qz[:, k] *10, alpha=0.8, label="")

	# for k in 1:3

		# ctr = ms[:, k]
		# for (i, x) in enumerate(eachrow(data))
		# 	plot!([x[1], ctr[1]],  [x[2], ctr[2]], lc = k, lw = qz[i, k] *1., st=:path, label="")
		# end
	# end
	plt
end

# ╔═╡ 2b5be103-c5b9-4326-9282-4598da2e11f9
e_step(data₂, [MvNormal(trueμs₂[:,k], 2*trueΣs₂[:,:,k]) for k in 1:3], trueπs₂)[1]

# ╔═╡ 608a2278-8823-4760-8f59-3aebd08ab65b
let
	gr()
	data = data₂
	K = 3
	dim = 2
	# ms = trueμs₂
	# mvns = [MvNormal(ms[:,k], trueΣs₂[:,:,k]) for k in 1:K]
	Random.seed!(123)
	ms = trueμs₂ .+ randn(dim, K)/2
	mvns = [MvNormal(ms[:,k], 2*trueΣs₂[:,:,k]) for k in 1:K]
	qz, l = e_step(data, mvns, trueπs₂)
	nks = sum(qz, dims=1)
	# k = k_idx
	plt = plot(ratio=1, framestyle=:origin, titlefontsize=12)

	title = ""
	# colors = [RGB(r,g,b) for (b,r,g) in eachrow(qz)]

	for k in 1:K
		title = title * L"{n}_{%$(k)}=\sum\!_{i} {r}_{i%$(k)} = %$(round(nks[k]; digits=1));\;\;"
	# scatter!(data[:, 1], data[:, 2], c=truezs₂, ms=4, alpha=0.1, label="")
	
		scatter!(data[:, 1], data[:, 2], c = k, ms=qz[:, k] * 8, alpha = 0.9, label="", title="")
	end

	title!(title)
	# for k in 1:3

		# ctr = ms[:, k]
		# for (i, x) in enumerate(eachrow(data))
		# 	plot!([x[1], ctr[1]],  [x[2], ctr[2]], lc = k, lw = qz[i, k] *1., st=:path, label="")
		# end
	# end
	plt
end

# ╔═╡ 2cabb54f-55e7-43ae-97e4-ad4ead73a16f
let
	gr()
	data = data₂
	K = 3
	dim = 2
	Random.seed!(123)
	ms = trueμs₂ .+ randn(dim, K)/2
	# ms = trueμs₂ ./ 2
	# ms = trueμs₂ + randn(dim, K)
	mvns = [MvNormal(ms[:,k], trueΣs₂[:,:,k]) for k in 1:K]
	# mvns = [MvNormal(ms[:,k], Matrix(I,2,2)) for k in 1:K]

	qz, l = e_step(data, mvns, 1/K * ones(K))
	newmvns, newπ= m_step(data, qz)
	# newctr = newmvns[k].μ
	# scatter!([newctr[1]], [newctr[2]], c=k, ms=10, markershape=:star ,alpha=1, label="")
	# zs_ = [c[2] for c in findmax(zs, dims=2)[2]][:]
	# zs
	title = "M step:"
	anim=@animate for k in 1:3
		# colors = [RGB(r,g,b) for (b,r,g) in eachrow(qz)]
		colors = [RGB(r,g,b) for (b,r,g) in eachrow(qz)]
		plt = plot(ratio=1, framestyle=:origin, titlefontsize =15)
		scatter!(data[:, 1], data[:, 2], c=truezs₂, ms=4, alpha=0.05, label="")
	
		scatter!(data[:, 1], data[:, 2], c=k, ms=qz[:,k] *10, alpha=qz[:,k], label="")
		newctr = newmvns[k].μ
		title = title * " "*L"\hat{\mu}_{%$k} = %$(round.(newctr;digits=1));"
		scatter!([newctr[1]], [newctr[2]], c=k, ms=12, markershape=:star ,markerstrokewidth=3, alpha=1, label="", title=title)
	end

	gif(anim, fps=0.5)
end

# ╔═╡ ebb2e81c-534d-48d6-8d49-44c01eb13edc
let
	gr()
	data = data₂
	K = 3
	dim = 2
	Random.seed!(123)
	# ms = trueμs₂ .+ randn(dim, K)/1.5
	# ms = trueμs₂ ./ 2
	ms = trueμs₂ + randn(dim, K)/2
	mvns = [MvNormal(ms[:,k], trueΣs₂[:,:,k]) for k in 1:K]
	# mvns = [MvNormal(ms[:,k], Matrix(I,2,2)) for k in 1:K]

	qz, l = e_step(data, mvns, 1/K * ones(K))
	newmvns, newπ= m_step(data, qz)
	title = "M step:"
	anim=@animate for k in 1:3
		# colors = [RGB(r,g,b) for (b,r,g) in eachrow(qz)]
		colors = [RGB(r,g,b) for (b,r,g) in eachrow(qz)]
		plt = plot(ratio=1, framestyle=:origin, titlefontsize =15)
		scatter!(data[:, 1], data[:, 2], c=truezs₂, ms=4, alpha=0.05, label="")
	
		scatter!(data[:, 1], data[:, 2], c=k, ms=qz[:,k]*10, alpha = 0.8*qz[:,k], label="")
		newctr = newmvns[k].μ
		title = title * " "*L"\{\mu_{%$k}, \Sigma_{%$(k)}\};"
		scatter!([newctr[1]], [newctr[2]], c=k, ms=10, markershape=:star , markerstrokewidth=3, alpha=1, label="", title=title)
		plot!(-5:0.1:5, -3:0.1:3, (x,y)-> pdf(newmvns[k], [x,y]), levels=6,  st=:contour, colorbar = false, alpha=1, color=k, linewidth=3) 
	end

	gif(anim, fps=1)
end

# ╔═╡ 3dd729e6-c33a-4279-a3bc-0e82b217b588
begin
	Random.seed!(4321)
	# K₂ = 3
	# trueμs₂ = zeros(2,K₂)
	# trueΣs₂ = zeros(2,2,K₂)
	# trueμs₂[:,1], trueΣs₂[:,:,1] = [-2.0, 0.0], 0.5 * Matrix(1.0I, 2,2)
	# trueμs₂[:,2], trueΣs₂[:,:,2] = [2.0, 0.0], 0.5 * Matrix(1.0I, 2,2)
	# trueμs₂[:,3], trueΣs₂[:,:,3] = [0., 0],  Matrix([0.5 0; 0 2])
	# trueπs₂ = [0.2, 0.2, 0.6]
	# truemvns₂ = [MvNormal(trueμs₂[:,k], trueΣs₂[:,:,k]) for k in 1:K₂]
	n₂_= 2000
	truezs₂_ = rand(Categorical(trueπs₂), n₂_)
	data₂_ = vcat([rand(truemvns₂[z])' for z in truezs₂]...)
	# data₂, truezs₂ = sampleMixGaussian(n₂, truemvns₂, trueπs₂)
end;

# ╔═╡ bbd25bdf-e82d-4f65-bfad-7d8e8e9cca18
struct data_set
	X
	labels
	mvns
	πs
end

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
end

# ╔═╡ e0cfcb9b-794b-4731-abf7-5435f67ced42
begin
	Random.seed!(123)
	K₃ = 3
	trueπs₃ = [0.25, 0.5, 0.25]
	trueμs₃ = [[1, 1] [0.0, 0] [-1, -1]]
	trueΣs₃ = zeros(2,2,K₃)
	trueΣs₃ .= [1 -0.9; -0.9 1]
	trueΣs₃[:,:,2] = [1 0.9; 0.9 1]
	truemvns₃ = [MvNormal(trueμs₃[:,k], trueΣs₃[:,:,k]) for k in 1:K₃]
	n₃ = 200* K₃
	data₃, truezs₃ = sampleMixGaussian(200, truemvns₃, trueπs₃)
	data₃test, truezs₃test = sampleMixGaussian(100, truemvns₃, trueπs₃)
	xs₃ = (minimum(data₃[:,1])-1):0.1: (maximum(data₃[:,1])+1)
	ys₃ = (minimum(data₃[:,2])-1):0.1: (maximum(data₃[:,2])+1)
	# dataset3
	dataset3 = data_set(data₃, truezs₃, truemvns₃, trueπs₃)
end;

# ╔═╡ c7fd532d-d72a-439a-9e71-e85392c66f8c
_, zskm₃, ms₃ = kmeans(data₃, K₃) ;

# ╔═╡ 76859d4c-f3e2-4576-b4d6-b637e9c99877
function QDA_fit(data, labels)
	n, d = size(data)
	# sse = zeros(d, d)
	K = length(unique(labels))
	μs = zeros(d, K)
	Σs = zeros(d,d,K)
	ns = zeros(Int, K)
	for k in (unique(labels)|>sort)
		ns[k] = sum(labels .==k)
		datak = data[labels .== k, :]
		μs[:, k] = μk = mean(datak, dims=1)[:]
		error = (datak .- μk')
		Σs[:,:,k] = error'*error/ns[k]
	end
	μs, Σs, ns/n
end

# ╔═╡ fc07f268-3f21-41e7-8d2b-dac341c226e2
qda_d2_μ,qda_d2_σ, qda_d2_π = QDA_fit(data₂, truezs₂);

# ╔═╡ a737c382-e0ac-4a98-a32d-9407a54c1b48
TwoColumn(md"""

> $$\hat \pi_k =\frac{\sum_{i=1}^n \mathbb{1}(z^{(i)} = k)}{n} =\frac{n_k}{n}$$

\

``\hat{\boldsymbol{\pi}}\propto`` $(latexify_md(round.(ns_d2))) ``=``  $(latexify_md(round.(qda_d2_π; digits=3)))
""", let
	gr()
	data = data₂
	zs = truezs₂
	K = 3
	dim = 2
	# ms = trueμs₂
	# mvns = [MvNormal(ms[:,k], trueΣs₂[:,:,k]) for k in 1:K]
	# Random.seed!(123)
	# ms = trueμs₂ .+ randn(dim, K)/2
	# mvns = [MvNormal(ms[:,k], 2*trueΣs₂[:,:,k]) for k in 1:K]
	# qz, l = e_step(data, mvns, trueπs₂)
	# colors = [RGB(r,g,b) for (b,r,g) in eachrow(qz)]
	plt = plot(ratio=1, framestyle=:origin, title="")
	# scatter!(data[:, 1], data[:, 2], c=truezs₂, ms=4, alpha=0.1, label="")
	# scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, ms= 8, label="")
	for k_ in 1:K
		scatter!(data[zs .==k_, 1], data[zs .==k_, 2], c=k_, ms= 4, alpha= 0.2, label="")
	end

	k = 1
	nks = counts(zs, 1:K)
	title = ""
	anim = @animate for k in 1:3
		title = title * " "*L"n_{%$k} = %$(nks[k]);"
		scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="", title=title)
	end
	gif(anim, fps=1)
end)

# ╔═╡ 507ffd07-cf87-4fa6-9bf3-4ccc9d4f3887
TwoColumn(md"""
> for ``k = 1\ldots K``:
>
> $$\small\hat{\boldsymbol{\mu}}_k = \frac{1}{n_k}\, {\sum_{i=1}^n \mathbb{1}(z^{(i)}=k)\cdot\mathbf x^{(i)}}$$
> $$\small\hat{\boldsymbol{\Sigma}}_k = \frac{1}{n_k} \sum_{i=1}^n \mathbb{1}(z^{(i)}=k) (\mathbf x^{(i)}-\boldsymbol{\mu}_k)(\mathbf x^{(i)}-\boldsymbol{\mu}_k)^\top$$

""", let
	gr()
	data = data₂
	zs = truezs₂
	K = 3
	dim = 2
	
	plt = plot(ratio=0.8, framestyle=:origin, title="")
	mvns = [MvNormal(qda_d2_μ[:,k], qda_d2_σ[:,:,k]) for k in 1:K]
	for k_ in 1:K
		scatter!(data[zs .==k_, 1], data[zs .==k_, 2], c=k_, ms= 4, alpha= 0.2, label="")
	end

	k = 1
	nks = counts(zs, 1:K)
	title = "QDA estimate"
	anim = @animate for k in 1:3
		title = title * " "*L"\{\mu_{%$k}, \Sigma_{%$(k)}\};"
		scatter!(data[zs .==k, 1], data[zs .==k, 2], c=k, label="", title=title)
		plot!(-3.5:0.1:3.5, -3:0.1:3, (x,y)-> pdf(mvns[k], [x,y]), levels=6,  st=:contour, colorbar = false, alpha=1, color=k, linewidth=3) 
	end
	gif(anim, fps=1)
end)

# ╔═╡ 620789b7-59bc-4e17-bcfb-728a329eed0f
qdform(x, S) = dot(x, S, x)

# ╔═╡ 7b47cda6-d772-468c-a8f3-75e3d77369d8
begin
# decision boundary function of input [x,y] 
function decisionBdry(x,y, mvns, πs)
	z, _ = e_step([x,y]', mvns, πs)
	findmax(z[:])
end

end

# ╔═╡ 8d0c6fdc-4717-4203-b933-4b37fe60d512
function logLikMixGuassian(x, mvns, πs, logLik=true) 
	l = logsumexp(log.(πs) .+ [logpdf(mvn, x) for mvn in mvns])
	logLik ? l : exp(l)
end

# ╔═╡ 054b5889-c133-4cd0-b930-33962c559d8f
TwoColumn(
let
	gr()
	logPx = false
	xs = range(extrema(data₂[:, 1])..., 100)
	ys = range(extrema(data₂[:, 2])..., 100)
	truemvns = deepcopy(truemvns₂[[1, 3, 2]])
	plt_mix_contour = plot(xs, ys, (x,y) -> logLikMixGuassian([x,y], truemvns, πs0, logPx), st=:contour,fill = true, c=:jet,  colorbar=false, title="contour plot "*L"p(\mathbf{x})", size=(350,350), xlabel=L"x_1", ylabel=L"x_2")
end

,
let
	plotly()
	logPx = false
	xs = range(extrema(data₂[:, 1])..., 100)
	ys = range(extrema(data₂[:, 2])..., 100)
	truemvns = deepcopy(truemvns₂[[1, 3, 2]])
	plt_mix_surface=plot(xs, ys, (x,y) -> logLikMixGuassian([x,y], truemvns, πs0, logPx), st=:surface, fill = true, color =:jet, colorbar=false, title="density plot p(x)",size=(350,350))
end
)

# ╔═╡ 8d06ce32-2c8d-4317-8c38-108ec0e7fe23
function em_mix_gaussian(data, K=3; maxIters= 100, tol= 1e-4, init_step="e", seed=123)
	Random.seed!(seed)
	# initialisation
	n,d = size(data)
	if init_step == "e"
		zᵢ = rand(1:K, n)
		μs = zeros(d, K)
		[μs[:,k] = mean(data[zᵢ .== k,:], dims=1)[:] for k in 1:K] 
	elseif init_step == "m"
		μs = data[rand(1:n, K), :]'
	else
		μs = randn(d,K)
		μs .+= mean(data, dims=1)[:] 
	end
	Σs = zeros(d,d,K)
	Σs .= Matrix(1.0I, d,d)
	mvns = [MvNormal(μs[:,k], Σs[:,:,k]) for k in 1:K]
	πs = 1/K .* ones(K)
	zs = zeros(n,K)
	logLiks = Array{Float64,1}()
	i = 1
	for i in 1:maxIters
		# E-step
		zs, logLik = e_step(data, mvns, πs)
		# M-step
		mvns, πs = m_step(data, zs)
		push!(logLiks, logLik)
		# be nice, let it run at least three iters
		if i>2 && abs(logLiks[end] - logLiks[end-1])< tol
			break;
		end
	end
	return logLiks, mvns, πs, zs
end

# ╔═╡ 605c727b-d5c8-418e-8d15-b19fc59acaef
ll, _, _, _=em_mix_gaussian(data₃, K₃; init_step="m", seed=123);

# ╔═╡ ccefec6c-df1b-4a4e-9155-2c757105fcce
begin
	gr()
	plot(ll, xlabel="iteration", ylabel="log likelihood", label="", lw=2)
end

# ╔═╡ 27265853-be33-4756-8322-fe0e7db76506
begin
	lls = []
	zs = []
	KK = 8
	for k in 1:KK
		logLiks, _, _ , zs_ = em_mix_gaussian(data₂_, k)
		push!(zs, zs_)
		push!(lls,logLiks[end])
	end
end

# ╔═╡ f9d63463-283a-42b0-bcc9-37c35bf7c87c
let
	bics = bic_mix_gaussian.(lls, 1:KK, 2, n₂_)
	plot(bics, title="Choose K via BIC", xlabel="K", ylabel="BIC", label="BIC", lw=2, lc=2, legend=:bottomright)
	plot!(lls, label="Likelihood", lc=1, lw=1.5, alpha=0.5)
	maxK = argmax(bics)
	scatter!([maxK], [bics[maxK]], ms=10, mc=2, markershape=:circle, alpha=0.8, label="")
end

# ╔═╡ d66e373d-8443-4810-9332-305d9781a21a
md"""

Functions used to plot and produce the gifs

"""

# ╔═╡ acfb80f0-f4d0-4870-b401-6e26c1c99e45
function plot_clusters(D, zs, K, loss=nothing, iter=nothing,  framestyle=:origin; title_string=nothing, alpha=0.5)
	if isnothing(title_string)
		title_string = ""
		if !isnothing(iter)
			title_string ="Iteration: "*string(iter)*";"
		end
		if !isnothing(loss)
			title_string *= " L = "*string(round(loss; digits=2))
		end
	end
	plt = plot(title=title_string, ratio=1, framestyle=framestyle)
	for k in 1:K
		scatter!(D[zs .==k,1], D[zs .==k, 2], label="cluster "*string(k), ms=3, alpha=alpha)
	end
	return plt
end

# ╔═╡ 0cdc7493-8bce-4272-9f1a-bd24326298ef
# # plot clustering results: scatter plot + Gaussian contours
# function plot_clustering_rst(data, K, zs, mvns, πs= 1/K .* ones(K); title="", add_gaussian_contours= false, add_contours=true)
# 	xs = (minimum(data[:,1])-0.5):0.1: (maximum(data[:,1])+0.5)
# 	ys = (minimum(data[:,2])-0.5):0.1: (maximum(data[:,2])+0.5)
# 	_, dim = size(data)
# 	# if center parameters are given rather than an array of MvNormals
# 	if mvns isa Matrix{Float64}
# 		mvns = [MvNormal(mvns[:,k], Matrix(1.0I, dim, dim)) for k in 1:K]
# 		πs = 1/K .* ones(K)
# 	end
# 	if ndims(zs) >1
# 		zs = [c[2] for c in findmax(zs, dims=2)[2]][:]
# 	end
# 	p = plot_clusters(data, zs, K)
# 	for k in 1:K 
# 		if add_gaussian_contours
# 			plot!(xs, ys, (x,y)-> pdf(mvns[k], [x,y]), levels=5,  st=:contour, colorbar = false, ratio=1, color=:jet, linewidth=2) 
# 		elseif add_contours
# 			plot!(xs, ys, (x,y)-> qdform([x,y]-mvns[k].μ, inv(mvns[k].Σ)), levels=[2.0],  st=:contour, colorbar = false, ratio=1, color=k, linewidth=3) 
# 		end
		
# 		scatter!([mvns[k].μ[1]], [mvns[k].μ[2]], color = k, label = "", markersize = 10, markershape=:star4, markerstrokewidth=3)
# 	end
# 	title!(p, title)
# 	return p
# end

# ╔═╡ e091ce93-9526-4c7f-9f14-7634419bfe57
# plot clustering results: scatter plot + Gaussian contours
function plot_clustering_rst(data, K, zs, mvns, πs= 1/K .* ones(K); title="", add_gaussian_contours= false, add_contours = true, lw=2)
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
		if add_gaussian_contours
			plot!(xs, ys, (x,y)-> pdf(mvns[k], [x,y]), levels=5,  st=:contour, colorbar = false, ratio=1, color=:jet, linewidth=lw) 
		elseif add_contours
			plot!(xs, ys, (x,y)-> qdform([x,y]-mvns[k].μ, inv(mvns[k].Σ)), levels=[1.0],  st=:contour, colorbar = false, ratio=1, color=k, linewidth=lw) 
		end
		
		scatter!([mvns[k].μ[1]], [mvns[k].μ[2]], color = k, label = "", markersize = 10, markershape=:star4, markerstrokewidth=3)
	end
	title!(p, title)
	return p
end

# ╔═╡ ca3040cd-35f1-451c-a26c-3a09a9c3a4c0
let
	_, zskm₄, ms₄ = kmeans(data₄, 9; seed= 222);
	plt = plot_clustering_rst(data₄, 9, zskm₄, ms₄; add_gaussian_contours=false, add_contours=false, title= "K-means result for dataset 4")
end

# ╔═╡ 1781a7b5-8dea-41c9-9cee-5f6c87819aa4
let
	KK =9
	_, mvns, πs , zs = em_mix_gaussian(data₄, KK;  maxIters= 400, tol= 1e-6, init_step="m", seed=3456)
	# plt₁₀ = plot_clusters(data₂_, argmax.(eachrow(zs[KK])), KK)
	plt₁₀ = plot_clustering_rst(data₄, KK, argmax.(eachrow(zs)), mvns, πs; title="", add_gaussian_contours= false, lw=3)
	title!(plt₁₀, "EM fit with "*L"K = %$(KK)")
end

# ╔═╡ 7cada25f-0ef0-4b8e-9ba9-3dfed9e0cdfd
TwoColumn(plot(lls, xlabel=L"K", ylabel="Log-likelihood", label="Likelihood", lc=1, lw=2, legend=:bottomright, title="Likelihood overfits with K", size=(330,300)), let
	KK =10
	_, mvns, πs , zs = em_mix_gaussian(data₂, KK)
	# plt₁₀ = plot_clusters(data₂_, argmax.(eachrow(zs[KK])), KK)
	plt₁₀ = plot_clustering_rst(data₂, KK, argmax.(eachrow(zs)), mvns, πs; title="", add_gaussian_contours= false, lw=3)
	title!(plt₁₀, "EM fit with "*L"K = %$(KK)")
	plot(plt₁₀, size=(350, 300), legend=:bottomright)
end)

# ╔═╡ 6569c4e1-5d62-42ad-94c0-927dd6b6f504
begin
	gr()
	plt₃_= plot_clustering_rst(data₃,  K₃, truezs₃, truemvns₃, trueπs₃)
	title!(plt₃_, "Overlapping dataset")
	plot(plt₃_, size=(300,300), titlefontsize=10)
end;

# ╔═╡ 5a8cdbe7-6abe-4f07-8bcc-89dd71fc35f7
function kmeansDemoGif(data, K, iters = 10; init_step="a", add_contour=false, seed=123)
	Random.seed!(seed)
	# only support 2-d
	anims = [Animation() for i in 1:3]
	dim =2 
	# initialise by random assignment
	if init_step == "a"
		zs = rand(1:K, size(data)[1])
		l = Inf
	# initialise by randomly setting the centers 
	else
		ridx = sample(1:size(data)[1], K)
		# ms = reshape(repeat(mean(data, dims=1)', K), (dim,K))
		# ms .+= randn(dim,K)
		ms = data[ridx, :]'
		ls, zs = assignment_step(data, ms)
		l = sum(ls)
	end
	# xs = (minimum(data[:,1])-0.1):0.1:(maximum(data[:,1])+0.1)
	xs = range(minimum(data[:,1])-0.1, maximum(data[:,1])+0.1, 100)
	# ys = (minimum(data[:,2])-0.1):0.1:(maximum(data[:,2])+0.1)
	ys = range(minimum(data[:,2])-0.1, maximum(data[:,2])+0.1, 100)

	# cs = cgrad(:lighttest, K+1, categorical = true)
	ps = 1/K .* ones(K)
	for iter in 1:iters
		ms = update_step(data, zs, K)
		# animation 1: classification evolution
		p1 = plot_clusters(data, zs, K, l, iter)
		# if add_contour
			for k in 1:K 
				if add_contour
				plot!(xs, ys, (x,y)-> sum((ms[:, k] - [x,y]).^2), levels=[5],  st=:contour, colorbar = false, ratio=1, color=k, linewidth=3)  
				end
				scatter!([ms[1,k]], [ms[2,k]], color = k, label = "", markersize = 10, markershape=:star4, markerstrokewidth=2)
			end
		# end
		frame(anims[1], p1)
		# animation 2: decision boundary
		mvns = [MvNormal(ms[:,k], Matrix(1.0I, dim, dim)) for k in 1:K]
		p2 = plot(xs, ys, (x,y) -> decisionBdry(x,y, mvns, ps)[2],  leg=:none, title="Iteration: "*string(iter)*"; L="*string(round(l; digits=2)), ratio=1, framestyle=:origin, c=1:K, alpha=0.5, st=:heatmap)
		for k in 1:K
			scatter!(data[zs .==k, 1], data[zs .==k, 2], c= k, ms=3, alpha=0.5)
			scatter!([ms[1,k]], [ms[2,k]], color = k, label = "", markersize = 10, markershape=:star4, markerstrokewidth=2)
			# plot!(xs, ys, (x,y)-> sum((ms[:, k] - [x,y]).^2), levels=[1.5],  st=:contour, colorbar = false, ratio=1, color=k, linewidth=3)  
		end
		frame(anims[2], p2)

		# animation 3: contour evolution
		# animation 3: contour plot
		# p3 = plot_clusters(data, zs, K, l, iter)
		p3 = plot(xs, ys, (x,y) -> logLikMixGuassian([x,y], mvns, ps), st=:contour, fill=true, colorbar=false, ratio=1, title="Iteration: "*string(iter)*"; L="*string(round(l; digits=2)))
		# for k in 1:K
		# 	scatter!(data[zs .==k, 1], data[zs .==k, 2], c= cs[k], label="")
		# end
		frame(anims[3], p3)
		
		ls,zs = assignment_step(data, ms)
		l = sum(ls)
	end

	return anims
end

# ╔═╡ c46e0b36-c3fd-4b7f-8f31-25c3315bb10c
# plot type: cl: classification; db: decision boundary; ct: contour
function mixGaussiansDemoGif(data, K, iters = 10; init_step="e", add_contour=false, seed=123, every = 1)
	Random.seed!(seed)
	# only support 2-d
	dim = 2 
	anims = [Animation() for i in 1:3]
	if init_step == "e"
		zs_ = rand(1:K, size(data)[1])
		zs = Matrix(I,K,K)[zs_,:]
		l = Inf
	else
		# ms = reshape(repeat(mean(data, dims=1)', K), (dim,K))
		# ms .+= randn(dim,K)

		ridx = sample(1:size(data)[1], K)
		# ms = reshape(repeat(mean(data, dims=1)', K), (dim,K))
		# ms .+= randn(dim,K)
		ms = data[ridx, :]'
		mvns = [MvNormal(ms[:,k], Matrix(1.0I,dim,dim)) for k in 1:K]
		zs, l = e_step(data, mvns, 1/K .* ones(K))
		zs_ = [c[2] for c in findmax(zs, dims=2)[2]][:]
	end
	# xs = (minimum(data[:,1])-0.1):0.1:(maximum(data[:,1])+0.1)
	# ys = (minimum(data[:,2])-0.1):0.1:(maximum(data[:,2])+0.1)
	xs = range(minimum(data[:,1])-0.1, maximum(data[:,1])+0.1, 100)
	ys = range(minimum(data[:,2])-0.1, maximum(data[:,2])+0.1, 100)
	# cs = cgrad(:lighttest, K+1, categorical = true)
	for iter in 1:iters
		# M step
		mvns, ps  = m_step(data, zs)
		# animation 1: classification evolution 
		p1 = plot_clusters(data, zs_, K, l, iter)
		for k in 1:K 
			if add_contour
				plot!(xs, ys, (x,y)-> qdform([x,y]-mvns[k].μ, inv(mvns[k].Σ)), levels=[2.0],  st=:contour, colorbar = false, ratio=1, color=k, linewidth=3) 
			end
			
			scatter!([mvns[k].μ[1]], [mvns[k].μ[2]], color = k, label = "", markersize = 10, markershape=:star4, markerstrokewidth=2)
		end
		
		# frame(anims[1], p1)
		# animation 2: decision boundary evolution 
		p2 = heatmap(xs, ys, (x,y) -> decisionBdry(x,y, mvns, ps)[2], c=1:K, leg=:none, title="Iteration: "*string(iter)*"; L="*string(round(l; digits=2)), ratio=1, framestyle=:origin)
		for k in 1:K
			scatter!(data[zs_ .==k, 1], data[zs_ .==k, 2], c= k)
		end
		# frame(anims[2], p2)

		# animation 3: contour plot
		# p3 = plot_clusters(data, zs_, K, l, iter)
		p3 = plot(xs, ys, (x,y) -> logLikMixGuassian([x,y], mvns, ps), st=:contour, fill=true, colorbar=false, title="Iteration: "*string(iter)*"; L="*string(round(l; digits=2)), ratio=1, framestyle=:origin)
		# for k in 1:K
		# 	scatter!(data[zs_ .==k, 1], data[zs_ .==k, 2], c= cs[k], label="")
		# end

		if (every == 1) || (every > 1 && (iter % every == 1))
			frame(anims[1], p1)
			frame(anims[2], p2)
			frame(anims[3], p3)
		end
		# E step
		zs, l = e_step(data, mvns, ps)
		zs_ = [c[2] for c in findmax(zs, dims=2)[2]][:]
	end
	return anims
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Clustering = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
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
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Clustering = "~0.15.6"
Distributions = "~0.25.107"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
Latexify = "~0.16.1"
LogExpFunctions = "~0.3.26"
Plots = "~1.40.9"
PlutoTeachingTools = "~0.2.14"
PlutoUI = "~0.7.54"
Statistics = "~1.11.1"
StatsBase = "~0.34.2"
StatsPlots = "~0.15.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "440890e64138ecb4e19e77121156035747e94b50"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+2"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "9ebb045901e9bbf58767a9f34ff89831ed711aae"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.7"

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
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

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
git-tree-sha1 = "3101c32aab536e7a27b1763c0797dba151b899ad"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.113"

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

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+1"

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
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

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

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "ee28ddcd5517d54e417182fec3886e7412d3926f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f31929b9e67066bee48eec8b03c0df47d31a74b3"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.8+0"

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
git-tree-sha1 = "1336e07ba2eb75614c99496501a8f4b233e9fafe"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.10"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "401e4f3f30f43af2c8478fc008da50096ea5240f"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.3.1+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "b1c2585431c382e3fe5805874bda6aea90a95de9"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.25"

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

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

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
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

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
git-tree-sha1 = "36bdbc52f13a7d1dcb0f3cd694e01677a515655b"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.0+0"

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
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c6ce1e19f3aec9b59186bdf06cdf3c4fc5f5f3e6"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.50.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "61dfdba58e585066d8bce214c5a51eaa0539f269"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "b404131d06f7886402758c9ce2214b636eb4d54a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.0+0"

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
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "260dc274c1bc2cb839e758588c63d9c8b5e639d1"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.0.5"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

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

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

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
git-tree-sha1 = "1a27764e945a152f7ca7efa04de513d473e9542e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.1"
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

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

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

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

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
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

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
git-tree-sha1 = "7f4228017b83c66bd6aa4fddeb170ce487e53bc7"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.6.2"

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

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "d0553ce4031a081cc42387a9b9c8441b7d99f32d"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.7"

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

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
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
git-tree-sha1 = "777657803913ffc7e8cc20f0fd04b634f871af8f"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.8"
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
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

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

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "6a451c6f33a176150f315726eba8b92fbfdb9ae7"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.4+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "15e637a697345f6743674f1322beefbc5dcd5cfc"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.3+0"

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
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─120a282a-91c1-11ec-346f-25d56e50d38c
# ╟─06d45497-b465-4370-8411-9651e33e70e6
# ╟─adaf23c3-1643-41c5-84a7-e2b73af048d6
# ╟─527963b1-ee7a-4cad-a2b7-b6a7789bfbbe
# ╟─0f826d7b-9628-4962-9c9e-db3ea287954a
# ╟─646dd3d8-6092-4435-aee9-01fa6a281bdc
# ╟─6f051fad-2c4b-4a9e-9361-e9b62ba189c5
# ╟─be9bcfcb-7ec7-4851-bd3f-24d4c29462fe
# ╟─6148a7c6-8c6d-48f7-88aa-4657841882b7
# ╟─f69c85dd-6227-4d63-9ad8-8e6a6610ef84
# ╟─f60afe8c-1067-485c-ae3c-b510960bf01a
# ╟─077cf7c8-e4a3-4352-b435-36f1b9e44c1b
# ╟─7db71b2e-caae-415f-b87b-f87665fd8d5e
# ╟─83ee2491-d08d-4eea-8a3a-b8f9715c9351
# ╟─4cf3cde7-f95c-443d-b8fa-7c3a3ca4ff4d
# ╟─f12c11a2-f548-4409-ab06-e6a7cb8dfb06
# ╟─4bfd08f3-5bdb-48a8-bacd-0d38ad674b74
# ╟─e0a0ef50-cb98-44a9-a2d8-10191a30e521
# ╟─7c8bfb91-4f3c-4e56-88b1-cb822983b3bc
# ╟─fb6ed0e0-a21d-44a5-a0fb-53cdd318628a
# ╟─cb88a7dd-0abf-4229-b29a-ba0b963b63fc
# ╟─5acf48ee-9b2f-4727-8a6b-fc3d0d7f68ed
# ╟─e0196ba3-314e-417d-9a36-5a6b8b0c556a
# ╟─5d820b63-c9c7-4ec2-9c53-966ff6fd70df
# ╟─6c2164fe-c544-4c42-bed4-9168e1ab049a
# ╟─7b51f61c-7635-4d79-97a4-9e5210c827cf
# ╟─702c96a8-439a-4b02-9125-06767d363e71
# ╟─2ab75abc-a685-4a5e-becf-6976ed439068
# ╟─03f6cd09-9c78-4592-b3e4-379cbdff40a9
# ╟─0b7dfe99-e93d-4203-81fd-b04c38105daa
# ╟─6a9df385-4261-4adb-80d3-32a02808f0f0
# ╟─5d0816c3-0f4a-4a0d-b0a8-1cd5644e8eba
# ╟─07613525-1f06-4f78-bbfa-4486fc0cf121
# ╟─e7c6725d-74d3-4fd4-9abe-38716693f2bb
# ╟─054b5889-c133-4cd0-b930-33962c559d8f
# ╟─e431bdb7-fd7d-4ded-9ec5-993518d89381
# ╟─09ca554e-e5c6-400e-9a1c-11b60b8fd338
# ╟─983bc7fa-6b95-4c09-bf2f-85152febfa36
# ╟─5e469163-3e8b-46be-a218-b608f01f75cf
# ╟─1751787f-b7d6-4078-903a-decc0804ce54
# ╟─fbc2977f-c02c-4ac1-ace3-b0ad8ee90149
# ╟─0dbc7c7d-4dea-48c5-a086-0a8666b732ef
# ╟─a737c382-e0ac-4a98-a32d-9407a54c1b48
# ╟─fc07f268-3f21-41e7-8d2b-dac341c226e2
# ╟─c17c74a0-2b24-4d27-97cb-5407a47bbab1
# ╟─507ffd07-cf87-4fa6-9bf3-4ccc9d4f3887
# ╟─d692526e-fbe5-4982-b217-88b0565b96bc
# ╟─45299ff7-3fc7-4277-9d46-75ba525f73f5
# ╟─4ebf3dfc-0df7-45a2-8e81-d9b7225a218e
# ╟─e5fa6fe9-af47-487c-9c30-e35b6da8d5bd
# ╟─c0e777da-c3be-42d3-8fa5-a540da94714c
# ╟─b8362a65-1944-470a-9185-09335cd2d94b
# ╟─8fb18689-3f90-4971-95d4-ef622af4d8dd
# ╟─c88de029-9a12-4a77-9a24-5942fc99a06f
# ╟─fd5b729e-e51e-44b2-a992-f68852313797
# ╟─1c380bc6-24a5-462c-bddf-5e0bcd7ea855
# ╟─188f7d65-3c87-4229-a58f-f7d8a6a21d32
# ╟─780190ed-b34f-414b-b778-a5891deb3a8f
# ╟─8d03c7ee-2db1-46c2-bf5a-d55a59dfe863
# ╟─f11eac0e-59d4-4594-9535-d2ad75fb456f
# ╟─c81ad0d3-1bd5-480e-829f-ce31a9d6e907
# ╟─53757539-9ce7-4ba8-84d7-994d9830d41f
# ╟─c258c3b1-c841-49b6-8d71-19213aa6e6d6
# ╟─3b3359f4-b500-40bc-a7fd-c80e19023f73
# ╟─9ac9bb50-0419-42c2-a90f-c3995ff72df5
# ╟─1d3dc35d-f9f7-4b17-962d-0430d5a1cfea
# ╟─49e641e7-977c-4f34-94dd-8db4f31939d0
# ╟─4d4d467a-8dab-4111-b950-afd405580d65
# ╟─629c24b5-0028-47b8-9490-0c3d8a8665b7
# ╟─c75fd76e-8581-4dcf-808a-17c133b6fadc
# ╟─fdf3dda3-afc3-43c6-b58e-45e1ebf39369
# ╟─8f9757f3-4f8d-470b-9964-65812c8d86be
# ╟─2b5be103-c5b9-4326-9282-4598da2e11f9
# ╟─1f7cbf66-3c75-486f-bd55-09930fb50579
# ╟─7ad702d7-7ceb-48ee-9258-ea78de864b0d
# ╟─5f6adb33-a20a-47ca-84fd-9d5440f30749
# ╟─017b7b44-1b9b-4627-bb9d-2d9743ab0774
# ╠═28b3b5fd-2e8d-4a90-b708-710ac1cbae0e
# ╠═1ef1de8d-73e7-44ec-bdea-03d763d6332e
# ╠═f1b2db15-a9ff-4fbb-a2f7-4ca99cc44637
# ╟─6a168c4f-19e8-4335-8b51-34c2a42dadac
# ╟─608a2278-8823-4760-8f59-3aebd08ab65b
# ╟─4d8bbaa8-57dd-4e7f-8bf6-d82e4ffabe64
# ╟─2cabb54f-55e7-43ae-97e4-ad4ead73a16f
# ╟─0e782555-a8e5-4c4a-9afa-3e7b8de143ca
# ╟─ebb2e81c-534d-48d6-8d49-44c01eb13edc
# ╟─a2781145-89bd-4bd5-af50-095410ebc6a5
# ╟─5a0be82c-a669-4134-b064-f4363661f439
# ╟─619387f3-e0c8-43d8-90f9-b319f3c849dd
# ╟─453a4215-6ff5-4b45-ad23-ed6710389d1f
# ╟─5a372f09-1b13-443c-ad81-4337612669aa
# ╟─e04d2344-4787-4eb5-97a4-8d02dad09b88
# ╟─69321cbd-f325-4da8-8674-a90f616b9ee2
# ╟─ea1efb75-3af9-49c5-9c1d-8bc298c6eaad
# ╟─801732eb-b50a-40c1-8199-576cdd06ce5e
# ╟─ccefec6c-df1b-4a4e-9155-2c757105fcce
# ╟─ab538ef3-0ebf-4a4c-839a-eea19e1920d8
# ╟─c00e00f2-a1d6-407a-8326-15e6cf125d56
# ╟─9b27b240-9e2a-4b49-a643-83739d468c5b
# ╟─1e8f4f6e-42fe-442e-92c0-d44699904462
# ╟─74557480-f940-4c60-8b12-8255a307c310
# ╟─09643fce-8711-4fe9-bd65-7e2cf2494ee1
# ╟─ca3040cd-35f1-451c-a26c-3a09a9c3a4c0
# ╟─99deb886-deea-4528-967c-7b9330da648d
# ╟─44222f14-d4c3-4556-88b0-5021091e64a8
# ╟─1c9b4d6f-63d1-42a2-9511-550048b67bf4
# ╟─58764f2b-e4e9-4f96-8f96-03f4041a5dd8
# ╟─1781a7b5-8dea-41c9-9cee-5f6c87819aa4
# ╟─605c727b-d5c8-418e-8d15-b19fc59acaef
# ╟─0572eb7c-e6d3-4c28-8531-4619720e7592
# ╟─bc075093-632f-4554-a100-15b43e6d679b
# ╟─b6a5836b-d14c-4168-a342-31af6acf0a55
# ╟─fce33a12-a40a-4cc3-9bf0-d5c8d388b649
# ╟─05c2d1ac-d447-4942-940d-4f4052e66eeb
# ╟─23ebe05e-e8b1-47bb-b918-ac390e21fd0b
# ╠═d44526f4-3051-47ee-8b63-f5e694c2e609
# ╠═27755688-f647-48e5-a939-bb0fa70c95d8
# ╟─2a539d0d-2bdc-4af0-b96f-52676393b458
# ╟─7cada25f-0ef0-4b8e-9ba9-3dfed9e0cdfd
# ╟─27265853-be33-4756-8322-fe0e7db76506
# ╟─16d4218c-7158-447d-8e3d-440a5d323801
# ╟─ceb4d9cc-2e32-4f7d-b832-2de8951feb1d
# ╟─f9d63463-283a-42b0-bcc9-37c35bf7c87c
# ╟─3dd729e6-c33a-4279-a3bc-0e82b217b588
# ╟─97355933-69bb-4e68-9ba1-956fd3684d0f
# ╟─0b07fda3-8c56-4224-b685-763ac06558ae
# ╟─5bc9d85c-27c0-42e2-a233-15b58e9bebe8
# ╟─6d919076-958f-4ba8-9089-92df4a27d030
# ╟─9d083ce9-72ef-4283-b757-ee331d7b89f2
# ╟─af80e90a-380e-427d-b6d4-dfd791628b6f
# ╟─0e79dfba-30eb-4eb0-9036-b59d0a66e52b
# ╟─f20a3dd2-c34c-44bf-8921-b3a736e1d433
# ╟─b9cd7a0f-f9ee-4da2-92d9-f47f8bc4ce28
# ╟─39d00409-608a-4b13-8131-4f157999d774
# ╟─9659b077-a7fc-4511-922a-7e5dd0a09b38
# ╟─0722858a-e2e1-4031-8537-729f4f578175
# ╟─d99c9fe9-0444-4a56-988a-ef805307b4f9
# ╟─a7aaf298-3d86-4913-a430-fc7dd22efd1a
# ╟─0a960801-802d-44f3-a3ff-6c3ac3436b5d
# ╟─79a87663-0a9c-454a-bd83-3750562043c6
# ╟─a8c88ad6-b805-461d-bf75-f5f6dbc76fd8
# ╟─96c5c8f8-df01-439f-9827-aad090437a31
# ╟─7b82382a-e3fb-4736-a07a-4b92ab002c74
# ╟─ed4c725d-c7bd-4231-adee-0153c626eb69
# ╟─74ee7c89-d53c-45b5-a01c-44f91a061745
# ╟─539554be-70f1-4985-a88d-35a7f201792e
# ╟─65a6081d-2d5a-4dbf-8040-7156ecd2c649
# ╟─1ec00e74-5a6d-4352-93a2-e54e786dd969
# ╟─223371fa-aa87-481f-b483-8854fbeb1ff2
# ╟─cf6f3ff3-4209-49f9-8b78-1cc1307a7c78
# ╟─ee7991d5-496d-4274-9f8d-7c0e8686822a
# ╟─b087392e-48ba-43b0-8385-9348a01391c0
# ╟─830240dc-393a-4ac1-82a2-ac9041847146
# ╟─6ec8901b-03fe-42c7-b302-2bfb984d2360
# ╟─2aea1227-6155-4bf9-85bb-4d8a0c7f999b
# ╠═da784e69-0524-44cc-a726-70b576c5a570
# ╟─541a1722-fd87-4217-961a-48dd271a3b1f
# ╠═ab41aa0a-a034-47bb-8ed3-c3224d7ae20a
# ╟─661ab8e4-f223-4361-91ca-27741e7fde5e
# ╟─01dfacdc-b59d-4382-b451-db6035949c36
# ╟─12f42d6e-1d67-4de7-9527-4d638eaac7e3
# ╟─90d26dc8-8eb3-425c-8d5b-ba150265f796
# ╟─0b79e6d4-7c5e-49be-b780-9ea4295c7029
# ╟─873230c7-13a9-4fb0-b4f9-fce7a7a4397d
# ╟─21913109-795c-4a91-868a-41fa0de38a12
# ╟─0d3dfdb4-280a-4316-a425-3f0a1ab32c67
# ╟─6b1563f6-226f-4539-8cc9-0d7d7b084999
# ╟─4197312e-cec7-41d4-aebf-a7ba4c7053ce
# ╟─e518420b-ec3a-4a77-9ca8-cdbace6c3721
# ╟─f338604d-a58f-4888-ace8-718c52ff3275
# ╟─b7ec3a9a-d171-4fd5-9534-4a82903c575e
# ╟─ad34bb36-c919-4fe9-845b-5e4d5e329c3e
# ╟─fa67c6b6-a14d-497e-a6f1-b99be84b6ffe
# ╟─b3e448ec-97ea-435b-9790-faa78bf94c63
# ╟─14ba281d-057a-4485-9794-32466388cee0
# ╟─7a0654e2-f01b-43d9-b500-b4a7a95eb428
# ╟─fecf3e10-ba13-4fea-8fa8-d10097280bab
# ╟─a2631364-95a2-4d2e-832b-696b54aba50d
# ╟─bcb78ed1-eb28-4165-89f2-5fba14cf4eb3
# ╟─8422af99-c189-4606-a30d-465f0c0f560d
# ╟─7bd7f0ef-9060-47da-89ed-14b964c85354
# ╟─46921d8b-6111-44fa-a1d5-607e6d95f0d6
# ╟─70926550-6e20-41dc-bba2-3ecd390e69cf
# ╟─4d942d63-eed1-4302-91d8-04a2444b05b1
# ╟─44175f60-6bde-4d4e-b977-318239e7eaa8
# ╟─c7a6c2a3-cffc-4cf2-82cb-7264ab67db05
# ╟─b5c598bd-ed60-4e00-a26c-975faf71c265
# ╟─f0fd83b4-9b15-4ce7-b220-60c3ff0bfea0
# ╟─2af8276a-3aa7-4e43-8e76-694ac5131e72
# ╟─71e543d9-5be8-4f78-9097-f4aad7885d5a
# ╟─2356d8e1-8854-440b-b861-90b71685a258
# ╟─64d31497-9009-49f2-b132-07a81331ac2f
# ╟─a0465ae8-c843-4fc0-abaf-0497ada26652
# ╠═3ea57a8e-8d15-4f41-acfb-e3bd1d65e585
# ╠═a414e554-3a8c-472d-af82-07c2f0843627
# ╠═33cc44b4-bd32-4112-bf33-6807ae53818c
# ╠═5f9ad998-c410-4358-925b-66e5d3b2f9e9
# ╠═6569c4e1-5d62-42ad-94c0-927dd6b6f504
# ╠═c7fd532d-d72a-439a-9e71-e85392c66f8c
# ╠═e0cfcb9b-794b-4731-abf7-5435f67ced42
# ╠═bbd25bdf-e82d-4f65-bfad-7d8e8e9cca18
# ╟─dafd1a68-715b-4f06-a4f2-287c123761f8
# ╟─76859d4c-f3e2-4576-b4d6-b637e9c99877
# ╟─620789b7-59bc-4e17-bcfb-728a329eed0f
# ╟─7b47cda6-d772-468c-a8f3-75e3d77369d8
# ╟─8d0c6fdc-4717-4203-b933-4b37fe60d512
# ╟─8d06ce32-2c8d-4317-8c38-108ec0e7fe23
# ╟─d66e373d-8443-4810-9332-305d9781a21a
# ╟─acfb80f0-f4d0-4870-b401-6e26c1c99e45
# ╟─0cdc7493-8bce-4272-9f1a-bd24326298ef
# ╟─e091ce93-9526-4c7f-9f14-7634419bfe57
# ╟─5a8cdbe7-6abe-4f07-8bcc-89dd71fc35f7
# ╟─c46e0b36-c3fd-4b7f-8f31-25c3315bb10c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
