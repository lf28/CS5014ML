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
	using Statistics
	using HypertextLiteral
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
end

# ╔═╡ 29998665-0c8d-4ba4-8232-19bd0de71477
begin
	using DataFrames, CSV
	using MLDatasets
	# using Images
end

# ╔═╡ 47050d18-06c3-4e9c-8c9a-6488cd765453
using GLMNet

# ╔═╡ 6e00611c-6403-431a-a331-347b62cf2c69
using Convex,SCS

# ╔═╡ 5335a2ec-2a18-4958-88b2-8b8c5b8cdbe5
using MLJBase

# ╔═╡ f79bd8ab-894e-4e7b-84eb-cf840baa08e4
using Logging

# ╔═╡ cb72ebe2-cea8-4467-a211-5c3ac7af74a4
TableOfContents()

# ╔═╡ e141fcb3-3912-48cd-a42d-5a777332031d
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ c39974a0-744c-478a-bfed-0ef97412f0c0
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

# ╔═╡ 358dc59c-8d06-4272-9a13-6886cdaf3dd9
ChooseDisplayMode()

# ╔═╡ 9bd2e7d6-c9fb-4a67-96ef-049f713f4d53
md"""

# CS5014 Machine Learning 


#### Regularisation & Hyperparameter tuning

\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 882da1c1-1358-4a40-8f69-b4d7cbc9387e
# md"""

# ## An example dataset


# The true function is a **4**-th power polynomial:

# ```math
# \large
# h(x) = -5 x + 5 x^2 + 2 x^3 - 2x^4
# ```
# """

# ╔═╡ 987ea5cb-b656-46f1-ae64-02e47fe32c9e
md"""



## Reading & references

##### Essential reading 


* **Regularisation** [_Pattern recognition and Machine Learning_ by _Christopher Bishop_: Chapter 3.1.4](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)


* **Overfitting/Cross validation** [_Deep Learning_ by _Ian Goodfellow and Yoshua Bengio and Aaron Courville_ Chapter 4.3.1](https://www.deeplearningbook.org/contents/ml.html)



"""

# ╔═╡ 75bc5b4c-7935-494f-ab11-c0908774800f
md"""

# `Ridge` regression
"""

# ╔═╡ c91fc381-613c-4c9b-98b6-f4dd897a69a5
md"""
## Training and Testing 

```math
\Large
\text{Avg train loss} = \frac{1}{n_{train}}\sum_{({x}^{(i)}, y^{(i)}) \in \mathcal{D}_{train}} (y^{(i)} - \hat{h}(\mathbf{x}^{(i)}))^2
```

```math
\Large
\text{Avg test loss} = \frac{1}{n_{test}}\sum_{({x}^{(i)}, y^{(i)}) \in \mathcal{D}_{test}} (y^{(i)} - \hat{h}(\mathbf{x}^{(i)}))^2
```

* ##### ``\mathcal{D}_{test}``: unseen or reserved dataset
"""

# ╔═╡ 726a1008-26e6-417d-a73a-57e32cb224b6
md"""

## Underfitting 

"""

# ╔═╡ fc2206f4-fd0f-44f8-ae94-9dae77022fab
md"""

#### *Underfitting*: the fitted models are overly _simple_
* ##### the *true P-order* is **4**
* ##### symptom: the training loss is large 
"""

# ╔═╡ 05837852-8838-40e8-b8de-1b01b5c57994
md" Add true function: $(@bind add_true CheckBox(default=false))"

# ╔═╡ d771fc78-09ae-4ce3-9ea3-514addbfd8a0
md"""

## Overfitting


"""

# ╔═╡ 2e9e4586-3e05-4ecf-9d56-3fa0bf2cdb70
md"""

#### *Overfitting*: the fitted models are overly _complicated_

* ##### the fitted model learns spurious patterns
* ##### symptom: training loss is small but testing loss is large (generalisation gap)
"""

# ╔═╡ 0f34bfca-89f2-494f-9991-db0f13e2c0a3
md" Add true function: $(@bind add_true_overfit CheckBox(default=false))"

# ╔═╡ db7d9963-b613-4350-be0b-30a479da2682
# md"""

# ## Training loss


# ```math
# \Large
# \text{Avg training loss} = \frac{1}{n} \sum_{(x^{(i)}, y^{(i)})\in \mathcal{D}_\text{train}} (y^{(i)} - \hat{y}^{(i)})^2
# ```

# """

# ╔═╡ afd07c02-3538-4bb4-a072-d9a66734bd6f
# let
# 	gr()
# 	poly_order = [0, 1, 2, 3, 7, 15]
# 	ylim = [extrema(y_poly)...;]
# 	ylim += [-1.0, 1.0]
# 	plots_ =[]
# 	for p in poly_order
# 		plt = plot(x_poly, y_poly, st=:scatter, label="")
# 		w, loss = poly_reg(x_poly, y_poly; order=p)
# 		plot!(-2:0.02:2, (x) -> poly_fun(x, w), lw=2, lc=:red,  label="Order: "*L"p=%$(p)", legend=:outerbottom, ylim=ylim, title="Training loss: "*L"%$(round(loss; digits=2))", legendfontsize=15)
# 		plot!(-2:0.05:2, (x) -> poly_fun(x, w_poly), label="", lw=2, lc=1)
# 		push!(plots_, plt)
# 	end
# 	plot(plots_..., size=(900, 700))
# end

# ╔═╡ 21b6b547-66ed-4231-830b-1a09adaf400c
# let
# 	gr()
# 	poly_order = [0, 1, 2, 4, 7, 15]


# 	plots_ =[]
# 	for p in poly_order
# 		plt = plot(x_poly, y_poly, st=:scatter, ms=3, mc=1,alpha=0.5, label="train data")
# 		plot!(x_poly_test, y_poly_test, st=:scatter, ms=3, mc=2,alpha=0.5,  label="test data")
# 		w, loss = poly_reg(x_poly, y_poly; order=p)
# 		loss_test = norm([poly_fun(x, w) for x in x_poly_test] - y_poly_test)/length(y_poly_test)
# 		if p == 4
# 		plot!(-2:0.01:2, (x) -> poly_fun(x, w), lw=2, lc=:red,  label="Order: "*L"p=%$(p)", legend=:outerbottom, ylim=[-7, 7], title="test loss: "*L"{%$(round(loss_test; digits=2))}")
# 		else
# 		plot!(-2:0.01:2, (x) -> poly_fun(x, w), lw=2, lc=:red,  label="Order: "*L"p=%$(p)", legend=:outerbottom, ylim=[-7, 7], title="test loss: "*L"%$(round(loss_test; digits=2))")
# 		end
# 		push!(plots_, plt)
# 	end
# 	plot(plots_..., size=(900, 600), legendfontsize=10)
# end

# ╔═╡ d7ca4793-2595-426c-a324-d85fec9feffd
md"""
## The trade-off
"""

# ╔═╡ c6cd66f8-f921-4a0c-be5c-94daf87862c8
show_img("training_cv.png", w=650)

# ╔═╡ 6fa3941b-5975-463c-9427-2ebad15f78eb
begin
	rbf(s=1) = (z) -> exp(-.5 * z^2/s)
	relu(s=1) = (z) -> max(z, 0)
	sigmoid(s=1) = (z) -> logistic(z/s)
end;

# ╔═╡ 7c6520b2-0446-44c0-a0b5-5aacff87cf84
function basis_expansion(xs, μs, σ::Function; intercept=true)
	n = length(xs)
	p = length(μs) 
	Φ = zeros(n, p)
	for (j, μ) in enumerate(μs)
		Φ[:, j] = σ.(xs .- μ)
	end
	if intercept 
		return [ones(n) Φ]
	else
		Φ
	end
end;

# ╔═╡ 17db14e7-b62c-432e-b33f-8f0b20b67a99
# begin
# 	Random.seed!(111)
# 	n_fixed_basis = n_basis
# 	μs = range((extrema(xs_q4) .+ (-.5, .5))..., n_fixed_basis+2)[2:end-1]
# end;

# ╔═╡ 5965c389-2d9e-4d0a-8ae0-9868b360e12e
linear_reg(Φ, y; λ = 1e-10) =  Φ \ y;

# ╔═╡ 3fafc04c-c46a-4a12-ba95-8dad1ff879e0
md"""
## Bias-variance trade-off

#### Error decomposition


$$\Large \text{test error} = \textbf{Irreducible Error} + \textbf{Bias}^2 + \textbf{Variance}$$


"""

# ╔═╡ bc756099-80e2-400a-97cb-c664f593c964
html"<center><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Bias_and_variance_contributing_to_total_error.svg/2880px-Bias_and_variance_contributing_to_total_error.svg.png
' width = '500' /></center>"

# ╔═╡ 9ed65430-fe12-4e83-8368-e6a6c62715f4
md"""
* ##### Irreducible error: the part of $y$ not predictable by anything in ``x`` 


* ##### `Bias²`: the averaged difference (squared) between the fitted model and the true model


* ##### `Var`: how sensitive the model is to another training data ``\mathcal{D}\sim P(x, y)`` 
"""

# ╔═╡ 98d86dad-19a2-4d7a-b593-69a390ef8d68
md"""
## Irreducible error



#### The error that is intrinsic in the data generating process: e.g.


$$\Large y = h_{true}(x_0) + \epsilon; \;\; \epsilon \sim \mathcal{N}(0, \sigma^2)$$


* ##### ``\epsilon``s are the irreducible error 

* ##### usually out of our control therefore irreducible
"""

# ╔═╡ 9b39a0b0-11df-4772-bb2e-a82fe219929c
md"""

## `Bias`



$$\large\begin{align}\text{Bias}(x_0) &= \mathbb{E}_{\mathcal{D}\sim P(x, y)}\left [ \hat{h}_{\mathcal{D}}(x_0)- h_{true}(x_0)\right] \\ 
&\approx \frac{1}{N} \sum_{n=1}^N \hat{h}_{\mathcal{D}^{(n)}}(x_0) - h_{true}(x_0)\end{align}$$


* ##### assume ``\mathcal{D}^{(n)} \sim P(x, y)`` is another training dataset (generated based on ``p(x, y)``)


* ##### for each $\mathcal{D}^{(n)}$, we fit $\hat{h}_{\mathcal{D}^{(n)}}$

* ##### `bias` is the error between the average fit and the ground truth

"""

# ╔═╡ 19d2b878-18e5-47ee-9458-c18b2e29341e
@bind refit Button("Repeat: with another dataset")

# ╔═╡ cb9e6893-eb31-4a32-8764-700b472e5606
md"""Number of basis: $(@bind n_basis_ Slider([0; 2:1:18])), show ``x_0``: $(@bind show_x0 CheckBox(default=false)); Add fit $(@bind add_fit_ CheckBox(default=false)), show all fits $(@bind show_fit_ensemble_ CheckBox(default=false))"""

# ╔═╡ 24d7b69b-3b58-49e2-93a7-42ad0cafe587
f(x) = log(x+1.2) + sin(x*5) + x + 1;

# ╔═╡ b372df13-9963-4df8-a0da-37485b3775b8
# f(x) = sin(x * 2π) + x

# ╔═╡ 2c44745b-6561-42ae-ba0b-13b5cd0aeaa3
begin
	refit
	n_train = 40
	noise_scale = 0.25
	xtrain = sort(rand(n_train) * 2 .- 1)
	ytrain = f.(xtrain) + randn(n_train) * noise_scale
end;

# ╔═╡ a07cf1fa-1b04-4767-9f7a-671096a04501
normal_eq_fit(X, y; λ = 1e-15) = (X'X +λ *I) \ (X' * y);

# ╔═╡ 854f6ec9-daff-4660-9513-d5f8c709cf8d
md"""

## `Variance`



$$\large\begin{align}\text{Var}(x_0) &= \mathbb{V}_{\mathcal{D}\sim P(x, y)}\left [ \hat{h}_{\mathcal{D}}(x_0)\right ] \\ 
&= \mathbb{E}_{\mathcal{D}\sim P(x, y)}\left [ \left(\hat{h}_{\mathcal{D}}(x_0) - \mathbb{E}_\mathcal{D}\left [\hat{h}_{\mathcal{D}}(x_0)\right]\right)^2 \right ] \\
&\approx \frac{1}{N} \left (\hat{h}_{\mathcal{D}}(x_0) - \frac{1}{N}\sum_{n=1}^N \hat{h}_{\mathcal{D}^{(n)}}(x_0)\right )^2\end{align}$$


* ##### assume ``\mathcal{D}^{(n)} \sim P(x, y)`` are training datasets (generated based on ``p(x, y)``)


* ##### `Var` is variance of the predictions at $x_0$

"""

# ╔═╡ 59e25758-84fb-48d4-8654-c7c87a51fdee
@bind xtest Slider(-0.95:0.1:0.95)

# ╔═╡ be68ba6a-21e2-4a4e-8f37-9391442242f7
let
	Random.seed!(123)
	n_train = 15
	noise_scale = 1.5
	xtrain = sort(rand(n_train) * 2 .- 1)
	ytrain = f.(xtrain) + randn(n_train) * noise_scale

	# n_basis = 2
	x0_ = xtest
	s_scale = 0.25
	plt = plot(framestyle=:origin)
	plot!(-1:0.01:1., f, lw=3, lc=1, ylim =[-3.5, 5.5], label="true "*L"h(x)", legendfontsize=12, legend=:topleft, size=[900,600])
	plot!([xtrain[1], xtrain[1]], [f.(xtrain[1]), ytrain[1]], c=:gray, ls=:dash, lw=2, st=:path, label=L"\epsilon")
	plot!(xtrain, ytrain, st=:scatter, c=3, alpha=0.5, ms=6, label="training data")
	[plot!([xi, xi], [f.(xi), yi], c=:gray, ls=:dash, lw=2, st=:path, label="") for (xi, yi) in zip(xtrain, ytrain)]

	
	plt
end

# ╔═╡ 510b7bd3-c4a8-4b9a-9f64-9bf02bf14d22
md"""Number of basis: $(@bind n_basis Slider([0; 2:1:20])), Add fit $(@bind add_fit CheckBox(default=false)), show all fits $(@bind show_fit_ensemble CheckBox(default=false))"""

# ╔═╡ a7ee5e77-75ea-45f3-bc5d-a289a7f8c30a
begin
	Random.seed!(111)
	n_times = 400
	X_train = rand(n_train * n_times) *2 .- 1
	Y_train = f.(X_train) + randn(length(X_train)) * noise_scale
	Y_train = reshape(Y_train, :, n_times)
end;

# ╔═╡ a519f36e-3055-4ddd-a8ac-c37fc3a7e2bd
function rbf(x, μ, s)
	exp(-1* (x-μ)^2 / (2*s^2))
end

# ╔═╡ a0150656-3bde-477d-957e-f50b206cf6b1
ϕ(x, xs, s) = [1.0, rbf.(x, xs, s)...]

# ╔═╡ d625f158-f0fb-43c8-92eb-a447b7a23d2b
let
	n_basis = n_basis_
	xtest = x0_ = -0.25
	x0s = range(-1.0, 1.0, n_basis)
	s_scale = 0.25
	Φ = ones(length(X_train))
	if n_basis > 0
		Φ_ = hcat([rbf.(X_train, x0, s_scale) for x0 in x0s]...)
		Φ = [Φ Φ_]
	end
	# # Φ
	Φs = reshape(Φ', :, n_train, n_times) 
	Φs = permutedims(Φs, [2, 1, 3])
	ws = []
	for i in 1:n_times
		Φ, yt = Φs[:, :, i], Y_train[:, i]
		wi = normal_eq_fit(Φ, yt)
		push!(ws, wi)
	end
	ws = hcat(ws...)
	yhats = ws' * ϕ(xtest, x0s, s_scale)
	plt = plot(framestyle=:origin)
	plot!(-1:0.01:1., f, lw=3, lc=1, ylim =[-3.5, 5.5], label="true "*L"h(x)", legendfontsize=12, legend=:topleft, size=[900,600])

	if show_x0
		vline!([x0_], lc=:gray, lw=2, ls=:dash, label="")
		annotate!([x0_], [-1], text(L"\mathbf{x}_0", :right, 20))
		
		scatter!([x0_], [f(x0_)], c=1, m=:x, ms=8, markerstrokewidth=4, label="true "* L"h(x_0)")
	end
	if show_fit_ensemble_
		ids = 1:25		
		for i in ids
			w_mle = ws[:, i]
			label_ = i == 1 ?  "fitted "*L"\hat{h}(x)" : ""
			plot!(-1:0.01:1, x ->  dot(w_mle, ϕ(x, x0s, s_scale)), label=label_, lw=0.8, ls=:dot, la=0.5, lc=3)
		end
		# fhat_x0 = 
		plot!(-1.:0.01:1.0, x -> mean(ws' * ϕ(x, x0s, s_scale)) , label="average fit", lw=2, ls=:solid, la = 0.9, lc=3)
		scatter!(repeat([x0_], length(ids)), yhats[ids], c=3, m=:+, ms=7, markerstrokewidth=2, label="")

		scatter!([x0_], [mean(yhats)], c=3, m=:x, ms=8, markerstrokewidth=4, label="average "* L"\hat{h}(x_0)")
		
		
	else
		plot!(xtrain, ytrain, st=:scatter, c=3, alpha=0.5, ms=6, label="training data")
		if add_fit_
			if n_basis > 0
				Φ = [ones(n_train) hcat([rbf.(xtrain, x0, s_scale) for x0 in x0s]...)]
			else
				Φ = ones(n_train)
			end
			w_mle = normal_eq_fit(Φ, ytrain)
			plot!(-1:0.01:1, x ->  dot(w_mle, ϕ(x, x0s, s_scale)), label="fitted "*L"\hat{h}(x)", lw=2, lc=3)
		end
	end
	plt
end

# ╔═╡ 189fb02d-ab82-4580-8f03-cfde475e3b8e
begin
	x0s = range(-1.0, 1.0, n_basis)
	s_scale = 0.2
	Φ = ones(length(X_train))
	if n_basis > 0
		Φ_ = hcat([rbf.(X_train, x0, s_scale) for x0 in x0s]...)
		Φ = [Φ Φ_]
	end
	# # Φ
	Φs = reshape(Φ', :, n_train, n_times) 
	Φs = permutedims(Φs, [2, 1, 3])
	ws = []
	for i in 1:n_times
		Φ, yt = Φs[:, :, i], Y_train[:, i]
		wi = normal_eq_fit(Φ, yt)
		push!(ws, wi)
	end
	ws = hcat(ws...)
	yhats = ws' * ϕ(xtest, x0s, s_scale)
end;

# ╔═╡ b16feb9a-740e-473d-a7b8-1f0c761271de
let
	
	# n_basis = 2
	x0_ = xtest
	# s_scale = 0.25
	plt = plot(framestyle=:origin)
	plot!(-1:0.01:1., f, lw=3, lc=1, ylim =[-3.5, 5.5], label="true "*L"h(x)", legendfontsize=12, legend=:topleft, size=[900,600])
	# if show_basis
	# 	for x0 in x0s
	# 		plot!(-1.05:0.05:1.05, x -> rbf(x, x0, s_scale), label="", c=:gray, ls=:dash, la=0.5)
	# 	end
	# end
	if show_fit_ensemble
		ids = 1:25		
		for i in ids
			w_mle = ws[:, i]
			label_ = i == 1 ?  "fitted "*L"\hat{h}(x)" : ""
			plot!(-1:0.01:1, x ->  dot(w_mle, ϕ(x, x0s, s_scale)), label=label_, lw=0.8, ls=:dot, la=0.5, lc=3)
		end

		plot!(-1.:0.01:1.0, x -> mean(ws' * ϕ(x, x0s, s_scale)) , label="average fit", lw=2, ls=:solid, la = 0.9, lc=3)
		scatter!(repeat([x0_], length(ids)), yhats[ids], c=3, m=:+, ms=7, markerstrokewidth=2, label="")
		vline!([x0_], lc=:gray, lw=2, ls=:dash, label="")
		annotate!([x0_], [-1], text(L"\mathbf{x}_0", :right, 20))
		scatter!([x0_], [f(x0_)], c=1, m=:x, ms=8, markerstrokewidth=4, label="true "* L"h(x_0)")
	else
		plot!(xtrain, ytrain, st=:scatter, c=3, alpha=0.5, ms=6, label="training data")
		if add_fit
			if n_basis > 0
				Φ = [ones(n_train) hcat([rbf.(xtrain, x0, s_scale) for x0 in x0s]...)]
			else
				Φ = ones(n_train)
			end
			w_mle = normal_eq_fit(Φ, ytrain)
			plot!(-1:0.01:1, x ->  dot(w_mle, ϕ(x, x0s, s_scale)), label="fitted "*L"\hat{h}(x)", lw=2, lc=3)
		end
	end
	plt
end

# ╔═╡ 3e45d793-aefc-4151-b9f2-6d3557f834a2
begin
	scatter(yhats, zeros(length(yhats)) .+0.15, m=:vline, c=3, ms=10, label=L"\hat{h}_n(x_0)", xlabel=L"y")
	density!(yhats, fill=true, alpha=0.5, c=3, bandwidth = 0.3, trim=false, xlim = mean(yhats) .+ (-4, 4), ylim = [0, 1.4], framestyle=:zerelines, yaxis=false, label="", ratio=2, legendfontsize=10, legend=:outerbottom)
	vline!([f(xtest)], lc=1, lw=2, ls=:dash, label=L"h_{true}(x_0)")
	vline!([mean(yhats)], lc=3, lw=2, ls=:dash, label="Average "*L"\hat{h}(x_0)")

	plot!([f(xtest), mean(yhats)], [0.7, 0.7], lc=2, lw=2, arrow=Plots.Arrow(:close, :both, 1, 1),  st=:path, label="")

	annotate!(.5 * [f(xtest) + mean(yhats)], [0.7], text(L"\texttt{Bias}", 15, :bottom, :darkorange))

	std_ = std(yhats)
	plot!([mean(yhats) - 3 * std_, mean(yhats) + 3 * std_], [0.3, 0.3], lc=3, lw=2, arrow=Plots.Arrow(:close, :both, 1, 1),  st=:path, label="")


	annotate!([mean(yhats)], [0.3], text(L"\texttt{Var}", 15, :bottom, :darkgreen))
end

# ╔═╡ ce8cd98f-da70-476c-817c-026365b982c0
md"""

## Regularisation



#### One technique to avoid *overfitting* is *regularisation*

\

#### The idea: add a *penalty* term 

```math
\Large
\mathcal{L}_{ridge}(\mathbf{w}) = \underbrace{\frac{1}{2n}\sum_{i=1}^n (y^{(i)} - \hat{y}^{(i)})^2}_{\text{mse loss}} + \boxed{\frac{\lambda}{2} \sum_{j=1}^{m} w_j^2}_{\text{penalty term}}
```
* ##### where ``\lambda \geq 0`` is a hyperparameter
* ##### intuitively:  large ``\mathbf{w}`` implies wiggly function
  * ##### large ``w_j``, therefore, are penalised


> #### And this is called *`Ridge` regression*



"""

# ╔═╡ 6a9e0bd9-8a2a-40f0-aaac-bf32d39ffac8
md"""
## `Ridge` regression

#### In matrix notation

```math
\Large
\mathbf{w}_{ridge} \leftarrow \arg\min_{\mathbf{w}} \frac{1}{2n}  \|\mathbf{y}-\mathbf{Xw}\|_2^2 + \boxed{\frac{\lambda}{2} \mathbf{w}^\top \mathbf{w}}
```

* ##### this is still a quadratic function! 
* why? sum of quadratics is still quadratic: ``(3w-2)^2  + \lambda w^2`` is still quadratic

#### Its gradient is 


```math
\Large
\nabla \mathcal{L}_{ridge}(\mathbf{w}) = \frac{1}{n}\mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) + \lambda\mathbf{w}
```



"""

# ╔═╡ 8436c5ce-d7be-4c02-9e89-7da701303263
aside(tip(md"""

Recall 

$$\|\mathbf{w}\|_2^2 = \mathbf{w}^\top\mathbf{w}$$

```math
\nabla_{\mathbf{w}} \mathbf{w}^\top\mathbf{w} = 2 \mathbf{w}
```

"""))

# ╔═╡ 44508906-0740-401e-bbb0-ebb3c0158f4f
md"""

## `Ridge` regression -- closed-form solution details

#### Set the gradient to zero, and solve it


```math
\begin{align}
	\nabla \mathcal{L}(\mathbf{w}) &= \frac{1}{n}\mathbf{X}^\top(\mathbf{Xw}- \mathbf{y} ) + \lambda\mathbf{w} = \mathbf{0} \\
	&\Rightarrow \mathbf{X}^\top\mathbf{Xw}- \mathbf{X}^\top\mathbf{y}  + n\lambda\mathbf{w}= \mathbf{0} \tag{expand matrix}\\
	&\Rightarrow (\mathbf{X}^\top\mathbf{X} +n\lambda\mathbf{I})\mathbf{w} = \mathbf{X}^\top\mathbf{y} \tag{$\lambda \mathbf{w} = \lambda \mathbf{Iw}$}\\
	&\Rightarrow \mathbf{w} = (\mathbf{X}^\top\mathbf{X} +n\lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}\tag{multiply inverse}
\end{align}
```


!!! note "Ridge regression solution"
	```math
	\large
		\mathbf{w}_{ridge} = (\mathbf{X}^\top\mathbf{X} +n\lambda \mathbf{I})^{-1} \mathbf{X}^\top\mathbf{y}
	```

"""

# ╔═╡ df62e796-d711-4ea1-a0c2-e3ec6c501a78
# md"""

# ## `Ridge` regression -- closed form solution



# !!! note "Ridge regression solution"
# 	```math
# 	\Large
# 		\mathbf{w}_{ridge} = (\mathbf{X}^\top\mathbf{X} +n\lambda \mathbf{I})^{-1} \mathbf{X}^\top\mathbf{y}
# 	```

# """

# ╔═╡ 8493d307-9158-4821-bf3b-c368d9cd5fc5
ridge_reg(X, y; λ = 1) = (X' * X + λ *I) \ (X' * y);

# ╔═╡ 439810a2-7d65-4ace-96a8-5969be0ca6aa
md"""

## Implementation 

#### Similar to linear regression, `Ridge` regression aims to solve 

```math
\Large
(\mathbf{X}^\top\mathbf{X} +n\lambda \mathbf{I})\cdot \mathbf{w}_{ridge} =  \mathbf{X}^\top\mathbf{y}
```
 


#### Julia (matrix left divide `\`)

```julia
w_ridge = (X'X + (n*λ)I) \ X'y
```
\


#### Python (least square solve)


```python
w_ridge = numpy.linalg.lstsq(X.T@X + n * λ * np.eye(d), X.T@y)[0]
```

"""

# ╔═╡ 8e3b6b19-d209-4dd7-a58e-f4246c897b66
md"""
## `Ridge` regression -- gradient descent

#### Recall the gradient is:

```math
\Large
\nabla \mathcal{L}_{ridge}(\mathbf{w}) = \nabla \mathcal{L}(\mathbf{w}) + \lambda\mathbf{w}
```


#### We can also use *gradient descent*, or weight decay:


```math
\Large
\mathbf{w}_{new}  \leftarrow \mathbf{w}_{old} - \gamma\cdot \left (\nabla \mathcal{L}(\mathbf{w}_{old})\, \; +\, \lambda\mathbf{w}_{old}\right)
```

* ##### the penaly's gradient ``\boxed{-\lambda\mathbf{w}}`` is called **weight decay**
* ##### it *shrinks* ``\mathbf{w}`` closer to ``\mathbf{0}``




"""

# ╔═╡ 8b73ae97-1c85-406f-830c-11a5079e934d
md"""
## Show me the code

#### *Implementation in Python*

\

```python
for i in range(0, max_iters):
		ypreds = Xs@w
		error = ypreds - ys
        grad =  Xs.T@error/len(ys) + lam0 * w ## gradient + weight decay
        # gradient descent
        w = w - gamma * grad
		loss = np.sum(error ** 2) / (2* len(ys)) # original mse loss
		loss += lam0 / 2 * np.sum(w ** 2) # L2 penalty
        print("Iteration %d | Loss: %f" % (i, loss)) # optional 
```


"""

# ╔═╡ ebdf7644-3383-46c2-bfdd-203a838daabf
md"""

## Effect of ``\lambda``

```math
\Large
\mathcal{L}_{ridge}(\mathbf{w}) = \frac{1}{2n}  (\mathbf{y}-\hat{\mathbf{y}})^\top (\mathbf{y}-\hat{\mathbf{y}}) + \boxed{\frac{\lambda}{2} \mathbf{w}^\top \mathbf{w}}
```


* ##### ``\lambda \rightarrow 0``, ``\mathbf{w}_{ridge} \rightarrow \mathbf{w}_{lsq}``

* ##### ``\lambda \rightarrow \infty``, ``\mathbf{w}_{ridge} \rightarrow \mathbf{0}``
"""

# ╔═╡ 570a250d-3aec-4c5f-9089-57b308998f6d
md"Select ``\lambda``: $(@bind λ_ Slider([0:0.01:100; 1e20], default= 0)), add ridge fit: $(@bind add_ridge CheckBox())"

# ╔═╡ 03edf15e-d365-44d2-8b4c-0c8a5bbe3432
md"""

## `Ridge` logistic regression
"""

# ╔═╡ ccb21e90-b473-4caf-aa8e-81259feeb15b
md"""

#### Logistic regression overfitting : esp when data is linearly separable


* ##### ``\mathbf{w} \rightarrow \infty``, ``\text{loss} \rightarrow 0``
* ##### black and white prediction
"""

# ╔═╡ 19fbdee1-1775-4c5c-b09f-324b24890a99
show_img("logis_overfitting.svg", w=700)

# ╔═╡ eda59f0e-de36-4bed-9cf0-698f45a17211
md"""

## Overfitting -- logistic regression

#### -- _predictions_ are black and white & lack of nuance
"""

# ╔═╡ ed66e944-993d-4a43-8191-b8ee708413fb
show_img("logistic_overfitting2.svg", w=600)

# ╔═╡ 45ad2754-1f97-4223-8289-284a182d0970
md"""


## `Ridge` regularisation 

#### -- no closed-form solution, apply gradient descent


"""

# ╔═╡ 5ba6caf3-23a7-485e-a530-2a3ece0c3e71
show_img("logistic_ridge.svg", w=600)

# ╔═╡ aaceb96b-0928-4ae0-b65e-c23344673967
md"""

## Comparison

"""

# ╔═╡ f8e4c992-8d31-4f6f-9a9e-a12d0efde1b1
show_img("logistic_regu_predictions.svg", w=600)

# ╔═╡ 763da9a9-5949-4bcf-a703-803fa2c2e82c
md"""

# `Ridge` _vs_ `Lasso`
"""

# ╔═╡ 0585ce12-41a3-43cd-8e73-96b349baf9eb
md"""
## What `Ridge` regression does?*



#### More generally, ridge regression's loss can be defined as


```math
\large
\mathcal{L}_{ridge}(\mathbf{w}) = \frac{1}{2n}  (\mathbf{y}-\mathbf{Xw})^\top (\mathbf{y}-\mathbf{Xw}) + {\frac{\lambda}{2} (\mathbf{w}-\mathbf{w}_0)^\top (\mathbf{w}-\mathbf{w}_0)}
```



* ##### where ``\mathbf{w}_0 = \mathbf{0}``

"""


# ╔═╡ aa697720-0662-4810-9957-b4489d698a83
md"""
## What `Ridge` regression does?* (cont.)


#### More generally, ridge regression's loss can be defined as


```math
\large
\mathcal{L}_{ridge}(\mathbf{w}) = \frac{1}{2n}  (\mathbf{y}-\mathbf{Xw})^\top (\mathbf{y}-\mathbf{Xw}) + {\frac{\lambda}{2} (\mathbf{w}-\mathbf{w}_0)^\top (\mathbf{w}-\mathbf{w}_0)}
```



* ##### where ``\mathbf{w}_0 = \mathbf{0}``

#### _Optimise it_ (set the gradient to zero and solve it)


```math
\Large
\begin{align}
\nabla &\mathcal{L}_{ridge}(\mathbf{w}) = \frac{1}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} -\mathbf{y}) + \lambda (\mathbf{w}-\mathbf{w}_0) = \mathbf{0} \\
&\Rightarrow \hat{\mathbf{w}}_{ridge} = (\mathbf{X}^\top\mathbf{X} +\lambda_n\mathbf{I})^{-1}(\mathbf{X}^\top\mathbf{y} +\lambda_n \mathbf{w}_0)
\end{align}
```

* ##### where ``\lambda_n = n\lambda``
"""


# ╔═╡ ea3b6e91-26af-4079-a8be-c3e90ba5b265
Foldable("Details", md"""

```math
\Large
\begin{align}
\nabla &\mathcal{L}_{ridge}(\mathbf{w}) = \frac{1}{n}\mathbf{X}^\top(\mathbf{X}\mathbf{w} -\mathbf{y}) + \lambda (\mathbf{w}-\mathbf{w}_0) = \mathbf{0} \\
&\Rightarrow  \mathbf{X}^\top\mathbf{X}\mathbf{w} -  \mathbf{X}^\top\mathbf{y} + n\lambda \mathbf{w} -n\lambda \mathbf{w}_0 =\mathbf{0} \\
&\Rightarrow (\mathbf{X}^\top\mathbf{X} +n\lambda \mathbf{I})\mathbf{w} = \mathbf{X}^\top\mathbf{y} +n\lambda \mathbf{w}_0\\
&\Rightarrow \hat{\mathbf{w}}_{ridge} = (\mathbf{X}^\top\mathbf{X} +n\lambda \mathbf{I})^{-1}(\mathbf{X}^\top\mathbf{y} +n\lambda \mathbf{w}_0)
\end{align}
```


""")

# ╔═╡ 5ebaf2c4-e23d-4835-9411-f4973a3e1ef8
md"""

## What `Ridge` regression does?* (cont.)


```math
\Large
\boxed{\begin{align}
\hat{\mathbf{w}}&_{ridge} = (\mathbf{X}^\top\mathbf{X} +\lambda_n \mathbf{I})^{-1}(\mathbf{X}^\top\mathbf{y} +\lambda_n \mathbf{w}_0) 
\end{align}}
```

* #### where ``\lambda_n \triangleq n\lambda``

```math
\Large
\begin{align}
 \hat{\mathbf{w}}&_{ridge} = (\mathbf{X}^\top\mathbf{X} +\lambda_n \mathbf{I})^{-1}(\mathbf{X}^\top\mathbf{y} +\lambda_n \mathbf{w}_0) \\
&= (\mathbf{X}^\top\mathbf{X} +\lambda_n \mathbf{I})^{-1}(\underbrace{\mathbf{X}^\top\mathbf{X}\cdot (\mathbf{X}^\top\mathbf{X})^{-1}}_{\texttt{identity } \mathbf{I}}\mathbf{X}^\top\mathbf{y} +\lambda_n \mathbf{w}_0)
\end{align}
```


## What `Ridge` regression does?* (cont.)


```math
\Large
\begin{align}
 \hat{\mathbf{w}}&_{ridge} = (\mathbf{X}^\top\mathbf{X} +\lambda_n \mathbf{I})^{-1}(\mathbf{X}^\top\mathbf{y} +\lambda_n \mathbf{w}_0) \\
&= (\mathbf{X}^\top\mathbf{X} +\lambda_n \mathbf{I})^{-1}(\mathbf{X}^\top\mathbf{X}\cdot \underbrace{ (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}}_{\text{least square est.}} +\lambda_n \mathbf{w}_0)
\end{align}
```




"""

# ╔═╡ fe020885-dd23-4e75-9df1-3e3b8fdef64b
md"""


## What `Ridge` regression does?* (cont.)


```math
\Large
\begin{align}
 \hat{\mathbf{w}}_{ridge} &= (\mathbf{X}^\top\mathbf{X} +\lambda_n \mathbf{I})^{-1}(\mathbf{X}^\top\mathbf{y} +\lambda_n \mathbf{w}_0) \\
&= (\mathbf{X}^\top\mathbf{X} +\lambda_n \mathbf{I})^{-1}(\mathbf{X}^\top\mathbf{X}\cdot \underbrace{(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}} +\lambda_n \mathbf{Iw}_0) \\
&=(\mathbf{X}^\top\mathbf{X} +\lambda_n \mathbf{I})^{-1}(\mathbf{X}^\top\mathbf{X}\cdot \hat{\mathbf{w}}_{lsq} +\lambda_n \mathbf{Iw}_0)
\end{align}
```

* ##### note that ``\hat{\mathbf{w}}_{lsq}`` is the least squared estimator (or unregularised)



* ##### if we assume all are scalars, we have (it is a weighted average)


```math
\Large
\begin{align}
 \hat{{w}}_{ridge} 
&=(x^2 +\lambda_n )^{-1}(x^2\cdot \hat{{w}}_{lsq} +\lambda_n {w}_0)\\
&= \frac{x^2}{x^2 +\lambda_n }\hat{{w}}_{lsq} +  \frac{\lambda_n}{x^2 +\lambda_n }{w}_{0}
\end{align}
```

## What `Ridge` regression does?* (cont.)


```math
\Large
\begin{align}
 \hat{\mathbf{w}}_{ridge} 
&=(\mathbf{X}^\top\mathbf{X} +\lambda_n \mathbf{I})^{-1}(\mathbf{X}^\top\mathbf{X}\cdot \hat{\mathbf{w}}_{lsq} +\lambda_n \mathbf{Iw}_0)
\end{align}
```

* ##### note that ``\hat{\mathbf{w}}_{lsq}`` is the least squared estimator (or unregularised)



* ##### if we assume all are scalars, we have


```math
\Large
\begin{align}
 \hat{{w}}_{ridge} 
&=(x^2 +\lambda_n )^{-1}(x^2\cdot \hat{{w}}_{lsq} +\lambda_n {w}_0)\\
&= \frac{x^2}{x^2 +\lambda_n }\hat{{w}}_{lsq} +  \frac{\lambda_n}{x^2 +\lambda_n }{w}_{0}
\end{align}
```

> #### _So what's ridge estimator_ ``\hat{\mathbf{w}}_{ridge}`` ?

> #### `` \hat{\mathbf{w}}_{ridge}`` is a (matrix) weighted average between 
> * ##### ``\hat{\mathbf{w}}_{lsq}`` and ``\mathbf{w}_0`` (``\mathbf{0}`` for ridge regression)
> * ##### the weights are ``\mathbf{X}^\top\mathbf{X}`` and ``\lambda_n \mathbf{I}``
"""

# ╔═╡ 4fb1c0b4-4368-461e-a12c-0629d72d0379
md"""


## To make this idea more concrete




#### Assume the weight ``\mathbf{X}^\top\mathbf{X} =\mathbf{I}``
* ##### *i.e.* the columns (features) of ``\mathbf{X}`` are normed to length 1 and also orthogonal to each other

* ##### this is unknown as "whittening" of the features


## To make this idea more concrete



#### Assume the weight ``\mathbf{X}^\top\mathbf{X} =\mathbf{I}``, and sub-in ``\mathbf{w}_0=\mathbf{0}`` then



 ```math
\Large
\begin{align}
 \hat{\mathbf{w}}_{ridge} 
&=(\mathbf{X}^\top\mathbf{X} +\lambda_n \mathbf{I})^{-1}(\mathbf{X}^\top\mathbf{X}\cdot \hat{\mathbf{w}}_{lsq} +\lambda_n \mathbf{Iw}_0) \\
&=(\mathbf{I} +\lambda_n \mathbf{I})^{-1} (\mathbf{I}\cdot \hat{\mathbf{w}}_{lsq} +\lambda_n \mathbf{0}) \\
&= \frac{1}{1+\lambda_n} \hat{\mathbf{w}}_{lsq}
\end{align}
```



* ##### ``\lambda_n \rightarrow 0``, ``\hat{\mathbf{w}}_{ridge} \rightarrow \hat{\mathbf{w}}_{lsq}``


* ##### ``\lambda_n \rightarrow \infty``, ``\hat{\mathbf{w}}_{ridge} \rightarrow \mathbf{0}``


* ##### ``0<\lambda_n < \infty``, it *shrinks*/*discount* the least square estimator by some *percentage*

"""

# ╔═╡ 8d6912ae-6756-4085-b912-10754c82656b
md"""


## The regularisation path of ridge regression


#### Ridge estimators are **NOT** **sparse** 
* ##### but just some average between 0 and ``\hat{w}_{lsq}``


#### The plot shows the _regularisation path_ $\hat{\mathbf{w}}(\lambda)$
* ##### the ridge estimator $\hat{\mathbf{w}}(\lambda)$ depends on $\lambda$
"""

# ╔═╡ e7e4bd42-cb98-4aae-b027-c0bc622e6c42
md"""

## `Lasso`


#### An alternative *regularisation* is ``L_1`` penalty: `Lasso` regression



```math
\Large
\mathcal{L}_{lasso}(\mathbf{w}) =  \mathcal{L}(\mathbf{w})  + \boxed{{\lambda} \sum_{j=1}^{m} |w_j|}_{L_1 \text{ norm}}
```
* #### where ``\lambda \geq 0`` is a hyperparameter

* #### ``L_1`` norm: ``\|\mathbf{w}\|_1 = \sum_j |w_j|``

"""

# ╔═╡ 757716d0-ee70-4a8b-951f-4253ca9eda40
md"""
## Aside: ``L_p`` norm

"""

# ╔═╡ b4026ef7-f5fd-4240-9a4f-de82e7f89e9e
TwoColumn(md"""
\
\
\
\

```math
\Large
\boxed{
\| \mathbf{w} \|_p = \left (\sum_{j=1}^m |w_j|^p \right)^{1/p}}
```
""", md"""$(Resource("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Vector-p-Norms_qtl1.svg/2560px-Vector-p-Norms_qtl1.svg.png", :width=>500))""")

# ╔═╡ cf636fb4-3631-403b-9c2d-a25676d8bb1f
md"""

## `Lasso` gradient

"""

# ╔═╡ 69949573-c43f-4568-9357-8d1e71398d88
TwoColumn(md"""
\
\

$$\Large |w_j| =\begin{cases} w_j & w_j > 0 \\ -w_j & w_j < 0 \end{cases}$$


""", plot(abs, size=(275,200),lw=2, title=L"|w|", xlabel=L"w", label=""))

# ╔═╡ bbe92f12-b863-4b51-9669-0ae8b39f32b1
TwoColumn(md"""
\

$$\Large \begin{align}\frac{d |w_j|}{d w_j} &=\begin{cases} 1 & w_j > 0 \\ -1 & w_j < 0 \end{cases}\\
&= \texttt{sign}(w_j) 
\end{align}$$

* ##### `sign(-2) = -1`, `sign(2) =1`

""", plot(sign, size=(275,200),lw=2, title=L"\partial |w|/\partial w", xlabel=L"w", label=""))

# ╔═╡ 9c25741d-960c-41c7-b5c6-da2852406ca1
md"""

## `Lasso` gradient



$$\Large
\begin{align}
\nabla (\lambda \|\mathbf{w}\|_1) &= \lambda \begin{bmatrix}\frac{\partial \sum_{j'} |w_j'|}{\partial w_1}\\
\frac{\partial \sum_{j'} w_j'}{\partial w_2} \\
\vdots \\

\frac{\partial \sum_{j'} |w_j'|}{\partial w_m}
\end{bmatrix} = \lambda \begin{bmatrix}\texttt{sign}(w_1)\\
\texttt{sign}(w_2) \\
\vdots \\

\texttt{sign}(w_m)
\end{bmatrix} \\
&=\lambda\cdot  \texttt{sign}(\mathbf{w})
\end{align}$$


#### Therefore, the total gradient is
```math
\Large
\nabla \mathcal{L}_{lasso}(\mathbf{w}) = \nabla \mathcal{L}(\mathbf{w}) + \lambda\cdot \texttt{sign}(\mathbf{w})
```

* then we can use gradient descent to learn the lasso parameter
* this is known as sub-gradient, therefore some special treatment is usually required to ensure convergence

#### _Compare with ridge regression_

```math
\Large
\nabla \mathcal{L}_{ridge}(\mathbf{w}) = \nabla \mathcal{L}(\mathbf{w}) + \lambda\cdot \mathbf{w}
```
"""

# ╔═╡ d7f125aa-27e5-4484-b0d8-b38caf650a82
md"""

## `Lasso` *vs* `Ridge`


##### `Ridge` regression
* ##### shrinks the MLE estimator towards zeros



"""

# ╔═╡ 36f97ad7-7bb0-4b41-9752-384002a53e8e
md"``\lambda:`` $(@bind lambda Slider(exp.(range(-5, 4, 100)); default=0.5))"

# ╔═╡ 268a33f8-4ded-4dbc-8141-0f81ba1c0384
let
	A = [0.5 0 ; 0 1.0]
	mu =[3.5, 3.5]
	lambdas = 0:0.05:10
	qform(x, A, mu) = (x -mu)' * A * (x -mu)
	plot(-4:0.1:8, -4:0.1:7, (x, y) ->  qform([x, y], A,mu), st=:contourf, framestyle=:zerolines, nlevels=18, c=cgrad(:coolwarm, rev=true), alpha=0.15,ratio=1, xlim =(-4, 8), ylim =(-4, 8),  size=(660,510), colorbar=false, xlabel=L"w_1", ylabel=L"w_2")

	scatter!([mu[1]], [mu[2]], m=:x, ms=8, markerstrokewidth=5, label="", series_annotation = [text(L"\hat{w}_{lsq}", 25, :red, :bottom)])
	
	ts = 0:0.05:2π

	for k in range(0.8, 3.0, 3)
		r = k/sqrt(lambda)
	
		xs = cos.(ts) * r
		ys = sin.(ts) * r
	
		plot!(xs, ys, label="", c=:blue)
	end
	scatter!([0], [0], m=:x, mc=:blue, ms=8, markerstrokewidth=5, label="", series_annotation = [text(L"{w}_{0}=0", 25, :blue, :top)])
	w_new = [(A+ lam * I) \ A' * mu for lam in lambdas]

	w_new_ = hcat(w_new...)
	plot!(w_new_[1, :], w_new_[2,:], label="", lw=2, lc=:purple,ls=:dash, legendfontsize=15)

	w_ = (A+ lambda * I) \ A' * mu
	scatter!([w_[1]], [w_[2]], m=:x, ms=8, markerstrokewidth=3, mc=:purple, series_annotation = [text(L"w_{ridge}", 25, :purple, :left)], label="", title="Ridge reg with "*L"\lambda=%$(round(lambda; digits=3))", titlefontsize=18)
end

# ╔═╡ b0aad1e2-580b-480e-9be5-3ad8c9702763
md"""

## `Lasso` *vs* `Ridge`


##### `Lasso` regression 
* ##### directly shut them off to zeros


"""

# ╔═╡ d407c511-6cf6-4eb0-aabd-25a3e7f5857c
md"``\lambda:`` $(@bind lambda2 Slider(exp.(range(-4, 3, 100)), default = 1.815))"

# ╔═╡ aa36dda8-29df-4592-a3f1-e8c1ec0955dc
function LassoEN(w, Q, γ)
    # (T, K) = (size(X, 1), size(X, 2))
	K = length(w)
    b_ls = w              #LS estimate of weights, no restrictions
	# X = Matrix(I, K, K)
    # Q = [0.5 0 ; 0 1.0]
    c = Q' * w                      #c'b = Y'X*b
    b = Variable(K)              #define variables to optimize over
    L1 = quadform(b, Q)            #b'Q*b
    L2 = dot(c, b)                 #c'b
    L3 = norm(b, 1)                #sum(|b|)
    L4 = sumsquares(b)            #sum(b^2)

    # if λ > 0
    #     Sol = minimize(L1 - 2 * L2 + γ * L3 + λ * L4)      #u'u/T + γ*sum(|b|) + λ*sum(b^2), where u = Y-Xb
    # else
    Sol = minimize(.5 * L1 -   L2 + γ * L3)               #u'u/T + γ*sum(|b|) where u = Y-Xb
    # end
    solve!(Sol, SCS.Optimizer; silent_solver = true)
    Sol.status == Convex.MOI.OPTIMAL ? b_i = vec(Convex.evaluate(b)) : b_i = NaN

    return b_i, b_ls
end;

# ╔═╡ de9fac90-7b4f-4034-a62a-6ef4bd9f01df
let
	A = [0.5 0 ; 0 1.0]
	mu =[3.5, 3.5]
	lambdas = 0:0.05:10
	qform(x, A, mu) = (x -mu)' * A * (x -mu)
	plot(-4:0.1:8, -4:0.1:7, (x, y) ->  qform([x, y], A,mu), st=:contourf, framestyle=:zerolines, nlevels=16,c=cgrad(:coolwarm, rev=true), alpha=0.15, ratio=1, xlim =(-4, 8), ylim =(-4, 8),  size=(660,510), colorbar=false, xlabel=L"w_1", ylabel=L"w_2")


	scatter!([mu[1]], [mu[2]], m=:x, ms=8, markerstrokewidth=5, label="", series_annotation = [text(L"\hat{w}_{lsq}", 25, :red, :bottom)])

	# plot!(-2:0.01:2, -2:0.01:2, (x, y) -> 10 * sum(abs.([x,y])), nlevels=10, alpha=0.5, st=:contour, colorbar=false, c=:blue)

	for k in range(0.5, 3.0, 3)
		r = k/(lambda2)
		xs = range(0, r, 10)
		ys = - xs .+ r
		plot!(xs, ys, label="", c=:blue)
		plot!(-xs, ys, label="", c=:blue)
		plot!(xs, -ys, label="", c=:blue)
		plot!(-xs, -ys, label="", c=:blue)
	end
	scatter!([0], [0], m=:x, mc=:blue, ms=8, markerstrokewidth=5, label="", series_annotation = [text(L"{w}_{0}", 25, :blue, :top)])



	
	w_new = zeros(2, length(lambdas))     #results for γM[i] are in bLasso[:,i]
	for i in 1:length(lambdas)
	    sol, _ = LassoEN(mu, A, lambdas[i])
	    w_new[:, i] = sol
	end

	plot!(w_new[1, :], w_new[2,:], label="", lw=2,lc=:purple, ls=:dash, legendfontsize=15)

	w_, _ = LassoEN(mu, A, lambda2)
	scatter!([w_[1]], [w_[2]], m=:x, ms=8, markerstrokewidth=3, mc=:purple, series_annotation = [text(L"w_{lasso}", 20, :purple, :left)], title="Lasso reg with "*L"\lambda=%$(round(lambda2; digits=3))", label="", titlefontsize=18)
end

# ╔═╡ 102adf0c-72c8-4987-99fe-ff96daba0e74
md"""


## Example -- Polynomial regression (P=12)


##### -- *Lasso* is more *sparse*

"""

# ╔═╡ c4e06848-6ca5-48e5-ab6a-b8ab92ec2702
md"``\mathbf{w}_{ridge}``:"

# ╔═╡ b61511c3-1ae8-4e30-aa55-62ecc9638f42
md"``\mathbf{w}_{lasso}``:"

# ╔═╡ cacf7514-de14-402a-b85f-ff84e90f9790
md"""

## Another example


##### Consider a toy data set, ``y^{(i)}`` depends on the _first_ feature
```math
\Large
y^{(i)} = x^{(i)}_1 + \varepsilon;
\;\;\varepsilon \sim \mathcal{N}(0, \;\sigma^2=0.2^2)
```

* ##### all the ``\mathbf{x}^{(i)} \in \mathbb{R}^{100}`` features are Gaussian noises 


* ##### _i.e._ the true regression parameter is ``\mathbf{w} = [1, 0,\ldots, 0]^\top``




"""

# ╔═╡ bbd86cd2-f298-46f4-8684-c33d11277746
begin
	Random.seed!(123)
	XX = randn(100, 100)
	YY = XX[:, 1] + randn(100) * .2
	XX_test = randn(100, 100)
	YY_test = XX_test[:, 1] + randn(100) * .2

	# loss_mle = []
	# loss_ridge = []
	# loss_lasso = []

end;

# ╔═╡ 369472e6-1b58-4310-8453-4ea86b33e807
begin
# poly_order = 12
	# X_p = poly_expand(x_poly; order = poly_order)[:, 2:end]
	path_ridge_sim = glmnet(XX, YY, alpha = 0.0, lambda = exp.([-2:.3:7;]) )
	# path_l = glmnet(X_p, y_poly, alpha = 1.0,lambda = exp.([-11:.1:4;]) )
	# path_r, path_l
	path_lasso_sim = glmnet(XX, YY, alpha = 1.0, lambda = exp.([-8:.2:0;]) )
end;

# ╔═╡ cd368db5-ae59-4eac-b3d0-3847c9c95c4c
md"""

## Another example (conti.)

"""

# ╔═╡ f3bdec6f-4280-4dd0-8ded-a4a896423f2a
let

plt1 = plot((path_ridge_sim.lambda), path_ridge_sim.betas', xscale=:ln, label="", xticks= (exp.([-2:2:7;]), -2:2:7 |> collect ), labelfontsize=12, xlabel=L"\ln \lambda", ylabel=L"\mathbf{w}(\lambda)", title="Ridge regression", tickfontsize=10)

plt2 = plot((path_lasso_sim.lambda), path_lasso_sim.betas', xscale=:ln, label="",  xticks = (exp.([-8:2:0;]), -8:2:0 |> collect ), labelfontsize=12, xlabel=L"\ln \lambda", ylabel=L"\mathbf{w}(\lambda)", title="Lasso regression", tickfontsize=10)

	plot(plt1, plt2, size=(700,350), lw=1.2)
end

# ╔═╡ 3195abbb-c8ab-4698-aa7c-657b27d8b24c
md"""$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/lassoridge1.png", :width=>900))
"""

# ╔═╡ c6c85ad7-decf-462a-84a3-cbb9bdc0f745
# md"""$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/lassoridge2.png", :width=>900))"""

# ╔═╡ cf483b66-3eed-441d-ab97-13b4509cdc8f
md"""

## Regularisation path of Lasso


#### We know $\lambda_t \rightarrow \infty$, more and more ``\hat{w}_j(\lambda_t) \rightarrow 0`` 


#### To find $\hat{\mathbf{w}}(\lambda_t)$ for all $t$,  we can learn ``\hat{\mathbf{w}}(\lambda_t)`` with in an ascending order, *i.e.* $\lambda_{t-1}< \lambda_{t}$ 

* ##### and "warm start" the gradient descent of learning $\mathbf{w}(\lambda_t)$ with initialising

$\Large \mathbf{w}_0 = \hat{\mathbf{w}}(\lambda_{t-1})$


* ##### instead of cold start from a random guess
"""

# ╔═╡ d93fbb13-da70-4303-8d5a-4c592fcdd918
# md"""

# ## Overfitting examples
# """

# ╔═╡ 02334a5c-41eb-41a2-a77c-041a2fc1c66a
# md"Number of basis: $(@bind n_basis Slider(1:1:15; default=1, show_value=true))"

# ╔═╡ 56903a30-f3fc-4f93-9572-231935295d0f
# md"Basis function: $(@bind ϕ Select([rbf, relu, sigmoid])), 
# Scale parameter ``s`` (if applied): $(@bind scale_ Slider(0.01:0.01:3.0;default=1.0, show_value=true))
# "

# ╔═╡ bd4ea5c5-164e-4604-94e2-0e018848be05
# md"Add fixed basis regression: $(@bind add_fixed_fit CheckBox(default=false))"

# ╔═╡ 45572dce-33d3-4dd2-9eb8-61cb063cf0e2
# let
# 	basis_fun = ϕ(scale_)
# 	plt = plot(xs_q4, ys_q4, st=:scatter, label="training data", ylim=[-1.25, 1.25], framestyle=:origin)
# 	for (j, μ) in enumerate(μs)
# 		if j == 1
# 			plot!(-.5:0.1:2π+.5, (x) -> basis_fun(x-μ), lw=1, ls=:dash, lc=:gray, label="basis functions")
# 		else
# 			plot!(-.5:0.1:2π+.5, (x) -> basis_fun(x-μ), lw=1, ls=:dash, lc=:gray, label="")
# 		end
# 	end
	
# 	plot!(-.5:0.1:(2π+.5), true_f, label="true signal", lc =:blue, lw=1.5, title="Fixed basis expansion regression dataset", legend=:bottomleft)

# 	if add_fixed_fit
# 		Φ = basis_expansion(xs_q4, μs, basis_fun; intercept=false)
# 		w = linear_reg(Φ, ys_q4; λ = 0.0)
# 		plot!(-0.5:0.1:2π+.5, (x) -> (basis_expansion([x], μs, basis_fun;  intercept=false) * w)[1] , label="fixed basis regression", lw=3, lc=2, ylim=[-1.25, 1.25])
# 	end


# 	plt
# end

# ╔═╡ 9c27da2e-9ab6-4106-af9f-51d6ca693bdb
md"""

# How to set the hyper-parameter
"""

# ╔═╡ e45c01fe-9b48-4011-8aab-8a70396351d5
md"""
## What are hyperparameters?

### Hyper-parameters: control the capacity/flexibility

* ##### Polynormial regression's order ``p``



* ##### Basis expansion's parameter ``\mu, s`` 
```math
\phi(x, \mu, s) = \exp\{-0.5 (x-\mu)^2/s\}
```

* ##### `Lasso`/`Ridge`'s penalty parameter ``\lambda``





* and so on

"""

# ╔═╡ e2906f07-e97c-4ddb-8013-86e2fe746064
md"""

## Hyper-parameter tuning


### Hyperparameter tuning ``\Longleftrightarrow`` _Find the optimal capacity_


"""

# ╔═╡ 147f0ed7-74f4-40a8-9898-30fe62ff3df7
md"""


* #### _training  performance_ can't be used (it favours complicated models) 

* #### we need an independent dataset (validation dataset)
"""

# ╔═╡ 7fe3f38a-16bf-4d18-b495-359e2084c07c
show_img("training_cv.png", w=650)

# ╔═╡ f8730c9e-d319-4dd4-99f1-401252daf6b2
md"""
## Hyper-parameter turning -- validation dataset
"""

# ╔═╡ d0a902c5-be1c-4752-93ac-0179513d8565
html"<center><img src='https://i0.wp.com/galaxyinferno.com/wp-content/uploads/2022/06/3.png?resize=1536%2C864&ssl=1' width = '500' /></center>"

# ╔═╡ e6d9a65b-e813-42b4-ba1d-2b0a3a72b9a7
md"""




#### *Training* data: 
* ##### given hyper-parameter ``\lambda``, train the model parameter ``\mathbf{w}`` 


#### *Validation* data: 
* ##### tune the hyper-parameter ``\lambda``


#### *Test* data: 
* ##### completely unseen data (no data leakage for both parameter and hyper-parameter)
* ##### generalisation performance for unseen data
"""

# ╔═╡ 99833f6e-813d-4d74-a8ca-df4048dcf704
md"""


## Hyper-parameter tuning (w/o validation data)


### There are some other hyper-parameters 
(usually optimisater's settings)
#### -- no need to tune them with validation data

* ###### choice of gradient descent/SGD/Newton's method
* ###### *e.g.* gradient descent learning rate/batch size/decay rate etc
* ###### just set with trial and error

"""

# ╔═╡ 0434873d-ef2f-4538-a27d-8d44316cd40f
# md"""

# ## Training loss


# ```math
# \Large
# \text{Avg training loss} = \frac{1}{n} \sum_{(x^{(i)}, y^{(i)})\in \mathcal{D}_\text{train}} (y^{(i)} - \hat{y}^{(i)})^2
# ```

# """

# ╔═╡ b1d1db29-f7c1-4e27-b54a-1b8532759334
# let
# 	gr()
# 	poly_order = [0, 1, 2, 3, 7, 15]
# 	ylim = [extrema(y_poly)...;]
# 	ylim += [-1.0, 1.0]
# 	plots_ =[]
# 	for p in poly_order
# 		plt = plot(x_poly, y_poly, st=:scatter, label="")
# 		w, loss = poly_reg(x_poly, y_poly; order=p)
# 		plot!(-2:0.02:2, (x) -> poly_fun(x, w), lw=2, lc=:red,  label="Order: "*L"p=%$(p)", legend=:outerbottom, ylim=ylim, title="Training loss: "*L"%$(round(loss; digits=2))", legendfontsize=15)
# 		plot!(-2:0.05:2, (x) -> poly_fun(x, w_poly), label="", lw=2, lc=1)
# 		push!(plots_, plt)
# 	end
# 	plot(plots_..., size=(900, 700))
# end

# ╔═╡ 40a4d260-f755-4958-af71-3c9f379b09b5
# md"""
# ## Testing performance


# ```math
# \Large
# \text{Avg test loss} = \frac{1}{n_{test}}\sum_{({x}^{(i)}, y^{(i)}) \in \mathcal{D}_{test}} (y^{(i)} - h(\mathbf{x}^{(i)}))^2
# ```
# """

# ╔═╡ 59027056-dac5-417e-ba92-724c2e55d500
# let
# 	gr()
# 	poly_order = [0, 1, 2, 4, 7, 15]


# 	plots_ =[]
# 	for p in poly_order
# 		plt = plot(x_poly, y_poly, st=:scatter, ms=3, mc=1,alpha=0.5, label="train data")
# 		plot!(x_poly_test, y_poly_test, st=:scatter, ms=3, mc=2,alpha=0.5,  label="test data")
# 		w, loss = poly_reg(x_poly, y_poly; order=p)
# 		loss_test = norm([poly_fun(x, w) for x in x_poly_test] - y_poly_test)/length(y_poly_test)
# 		if p == 4
# 		plot!(-2:0.01:2, (x) -> poly_fun(x, w), lw=2, lc=:red,  label="Order: "*L"p=%$(p)", legend=:outerbottom, ylim=[-7, 7], title="test loss: "*L"{%$(round(loss_test; digits=2))}")
# 		else
# 		plot!(-2:0.01:2, (x) -> poly_fun(x, w), lw=2, lc=:red,  label="Order: "*L"p=%$(p)", legend=:outerbottom, ylim=[-7, 7], title="test loss: "*L"%$(round(loss_test; digits=2))")
# 		end
# 		push!(plots_, plt)
# 	end
# 	plot(plots_..., size=(900, 600), legendfontsize=10)
# end

# ╔═╡ d95cc8e7-0b55-44b2-a65d-69dd93e86a9a
md"""

## Data leakage 


#### Data leakage: test data is involved in the training of *parameters* or *hyper-parameters*

* ##### it leads to **misleading** generalisation performance estimate


"""

# ╔═╡ 4b22e561-0adb-4c3e-803c-d7f880324876
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/datasplit_.png' width = '520' /></center>"

# ╔═╡ fb1ac118-b887-444a-b25d-6c33f96609c5
md"""

## Data leakage 2



!!! warning "Wrong practice"
	##### (*!!! Wrong*) Preprocess the data with both **training** + **validation** sets
    * *e.g.* when normalise the features

"""

# ╔═╡ 2aefb863-aad7-45a9-b500-04479213b3a8
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/data_leakage.png' width = '550' /></center>"

# ╔═╡ c2edb58b-c0a6-4fae-9f83-3d3a069a8a3a
md"""

## Example 

#### Polynormial regression order tuning

$\Large h_p(x) = w_0 + w_1 x + w_2 x^2 +\ldots + w_p x^p$


"""

# ╔═╡ cdef3ff8-04f8-463e-86cd-5377b57a5f8b
md"""

#### Single validation dataset

* ##### may not recover the optimal model
"""

# ╔═╡ 1d76e05e-172d-4385-b4b1-2bdd09b94dac
md"""

## K fold cross validation


#### K-fold cross-validation (CV)

* ##### repeat the process systematically **K** times

"""

# ╔═╡ 52917eba-fab4-421b-a582-83c586d7e297
html"<center><img src='https://scikit-learn.org/stable/_images/grid_search_cross_validation.png' width = '500' /></center>"

# ╔═╡ 07215124-40ac-4795-9bf8-4d902d0e8137
md"""

## K-Fold example

"""

# ╔═╡ 42f94b47-0a29-4ba0-8229-88021723ab35
# let
# 	plt = errorline(0:10, hcat(vali_rst...), errorstyle=:ribbon, label="Mean", secondarycolor=:matched,  xlabel="Polynormial order: "*L"p", ylabel="Validation MSE", title="10-Fold Cross Validation Mean and Std", xticks=0:10, legend=:outerright, legendfontsize=10, size=(750,450), labelfontsize=18)
# 	for (k, rst) in enumerate(vali_rst[1:end])
# 		plot!(plt, 0:1:10, rst, label="Fold "*string(k), alpha=0.25)
# 	end

# 	plt
# end

# ╔═╡ 7a3bb30e-2d4c-46e3-ac6a-d67874cab0d9
md"""
## One standard error rule


!!! note ""
	##### We choose the simpliest model 
	* ##### whose error is **no more** than _one standard error_ above the best model
"""

# ╔═╡ 138261fa-7736-4507-b8ad-4a1918a193e6
# md"""

# ## More example -- tuning ``\lambda``

# ```math
# \large
# \mathcal{L}(\mathbf{w}; \lambda) = \frac{1}{2}\sum_{i=1}^n (y^{(i)} - \hat{y}^{(i)})^2 + \boxed{\frac{\lambda}{2} \sum_{j=1}^{12} w_j^2}_{\text{penalty term}}
# ```


# * ##### polynormial regression ``p=12``

# * ##### *objective*: tune ``\lambda``
#   * ##### 10-fold cross validation
# """

# ╔═╡ 92d13008-4aed-4633-81b5-954475034fbf
md"""
## More example - tuning ``\lambda``


#### One standard error rule

"""

# ╔═╡ 45c57c4a-721a-4f24-bfcd-dfb3812de182
# let
# 	path_ridge = path_ridge_cv
# 	plt_ridge = plot(path_ridge.lambda, path_ridge.betas',xaxis=:log, title="", xlabel=L"\lambda", ylabel=L"w"; legend=:outerright, labels=[1:size(path_ridge.betas)[1];]', lw=1.5, legendtitle=L"w_i", xflip=:true)
# 	# vline!([exp.((4-11) /2)], ls=:dash, lw=1.5, lc=:gray, label="")
# 	vline!([lambdamin(poly_cv)], lc=2, lw=2, ls=:dash, label="CV best "* L"\lambda")
# 	vline!([poly_cv.lambda[13]], lc=3, lw=2, ls=:dash, label="within 1 std error")
# end

# ╔═╡ 974f1b58-3ec6-447a-95f2-6bbeda43f12f
md"""

# Appendix
"""

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

# ╔═╡ cb02aee5-d082-40a5-b799-db6b4af557f7
# md"""
# ## More datasets


# It turns out linear correlations are more common than we expect!

# * *e.g.* the flipper size and body mass of Penguins 
# """

# ╔═╡ 8deb1b8c-b67f-4d07-8986-2333dbadcccc
# md"""
# ![](https://allisonhorst.github.io/palmerpenguins/reference/figures/lter_penguins.png)"""

# ╔═╡ af622189-e504-4633-9d9e-ab16c7293f82
df_penguin = let
	Logging.disable_logging(Logging.Warn)
	df_penguin = DataFrame(CSV.File(download("https://gist.githubusercontent.com/slopp/ce3b90b9168f2f921784de84fa445651/raw/4ecf3041f0ed4913e7c230758733948bc561f434/penguins.csv"), types=[Int, String, String, [Float64 for _ in 1:4]..., String, Int]))
	df_penguin[completecases(df_penguin), :]
end;

# ╔═╡ 9267c4a4-74d1-4515-95d6-acc3b12e5ed6
# first(df_penguin, 5)

# ╔═╡ 76cc8ca7-d17e-4cd7-a6de-4d606e0a0985
# @df df_penguin scatter(:flipper_length_mm, :body_mass_g, group = (:species), legend=:topleft, xlabel="Flipper length", ylabel="Body mass");

# ╔═╡ c4e497fc-cfbf-4d0b-9a0c-1071f2f43a98
linear_reg_normal_eq(X, y) = X \y;

# ╔═╡ 0e2dc755-57df-4d9a-b4f3-d01569c3fcde
begin
	X_housing = MLDatasets.BostonHousing().features |> Matrix |> transpose
	df_house = DataFrame(X_housing', MLDatasets.BostonHousing().metadata["feature_names"] .|> lowercase)
	df_house[!, :target] = (MLDatasets.BostonHousing().targets |> Matrix )[:]
end;

# ╔═╡ c4f42980-0e68-4943-abbe-b28f05dd3ee5
function loss(w, X, y) # in matrix notation
	error = y - X * w
	0.5 * dot(error, error)
end;

# ╔═╡ 8fbcf6c3-320c-47ae-b4d3-d710a120eb1a
function least_square_est(X, y) # implement the method here!
	X \ y
end;

# ╔═╡ ce35ddcb-5018-4cb9-b0c9-01fb4b14be40
begin
	x_room = df_house[:, :rm]
	x_room_sq = x_room.^2 # squared x_room^2
	X_room_expanded = [ones(length(x_room)) x_room x_room_sq]
end;

# ╔═╡ edc245bc-6571-4e65-a50c-0bd4b8d63b74
function poly_expand(x; order = 2) # expand the design matrix to the pth order
	n = length(x)
	return hcat([x.^p for p in 0:order]...)
end;

# ╔═╡ 4154585d-4eff-4ee9-8f33-dad0dfdd143c
function poly_reg(x, y; order = 2) # fit a polynomial regression to the input x; x is assumed a vector
	X = poly_expand(x; order=order)
	w = linear_reg_normal_eq(X, y)
	l = loss(w, X, y)
	return w, l
end;

# ╔═╡ 46180264-bddc-47d8-90a7-a16d0ea87cfe
poly_fun(x, w) = sum([x^p for p in 0:length(w)-1] .* w);

# ╔═╡ 0dff10ec-dd13-4cc2-b092-9e72454763cc
begin
	gr()
	Random.seed!(123)
	x_poly = [range(-1.8, -0.5, length=8)... range(.5, 1.8, length=8)...][:]
	x_poly_test = -2 : 0.1: 2
	w_poly = [0.0, -5, 5, 2, -2]
	y_poly = [poly_fun(x, w_poly) for x in x_poly] + randn(length(x_poly))
	y_poly_test = [poly_fun(x, w_poly) for x in x_poly_test] + randn(length(x_poly_test))
	plot(x_poly, y_poly, st=:scatter, label="training data", xlabel = L"x", ylabel=L"y")

	# plot!(x_poly_test, y_poly_test, st=:scatter, lc=2, alpha= .5, label="testing data")
	# plot!(-2:0.05:2, (x) -> poly_fun(x, w_poly), label="true "*L"h(x)", lw=2, lc=1, size=(600, 400))
end;

# ╔═╡ 11d41404-59c4-42da-b402-850341066208
path_ridge, path_lasso = let
	poly_order = 12
	X_p = poly_expand(x_poly; order = poly_order)[:, 2:end]
	path_r = glmnet(X_p, y_poly, alpha = 0.0, lambda = exp.([-11:.1:4;]) )
	path_l = glmnet(X_p, y_poly, alpha = 1.0,lambda = exp.([-11:.1:4;]) )
	path_r, path_l
end;

# ╔═╡ 397e3e8b-8247-47f6-8ae6-91b2546089d3
plt_ridge = let
	plt_ridge = plot(path_ridge.lambda, path_ridge.betas',xaxis=:log, title="Ridge regression", xlabel=L"\log \lambda", ylabel=L"w"; legend=:outerright, labels=[1:size(path_ridge.betas)[1];]', lw=1.5, legendtitle=L"w_i")
	# plot!(path_lasso.lambda, path_lasso.betas', xaxis=:log)
	# vline!([exp.((4-11) /2)], ls=:dash, lw=1.5, lc=:gray, label="")
end;

# ╔═╡ 630ac272-b9ea-4871-98ba-fb91c844a289
plt_ridge

# ╔═╡ 2b391a81-7779-4912-aadc-b513468e029b
md"Add vline: $(@bind add_vline CheckBox(default=false)), Add vline of ``\ln\lambda``: $(@bind λi Slider(1:length(path_ridge.lambda)))"

# ╔═╡ d8dcb7ab-a538-45dd-a47b-144b1b87e33f
L"%$(round.(path_ridge.betas[:, λi]; digits=3))"

# ╔═╡ 07638f73-df94-4a8e-a511-eb11f040b01b
L"%$(round.(path_lasso.betas[:, λi]; digits=3))"

# ╔═╡ dca81d78-d851-413c-91f8-439f99c4bda3
let

	plt_ridge = plot(path_ridge.lambda, path_ridge.betas',xaxis=:log, title="Ridge regression", xlabel=L"\log \lambda", ylabel=L"w"; legend=:outerright, labels=[1:size(path_ridge.betas)[1];]', lw=1.5, legendtitle=L"w_i")
	# plot!(path_lasso.lambda, path_lasso.betas', xaxis=:log)
	# vline!([exp.((4-11) /2)], ls=:dash, lw=1.5, lc=:gray, label="")
	plt_lasso = plot(path_lasso.lambda, path_lasso.betas', xaxis=:log, title="Lasso regression",  xlabel=L"\log \lambda", ylabel=L"w"; legend=:outerright, labels=[1:size(path_ridge.betas)[1];]', legendtitle=L"w_i", lw=2);
	if add_vline
		vline!(plt_lasso, [(path_ridge.lambda[λi])], ls=:dash, lw=1.5, lc=:gray, label="")
		vline!(plt_ridge, [(path_ridge.lambda[λi])], ls=:dash, lw=1.5, lc=:gray, label="")
	end
	plot(plt_ridge, plt_lasso, lw=2, size=(900,450))
end

# ╔═╡ ae8e2ab7-ffbb-4af0-a190-c30505a117dd
path_ridge_cv = let
	poly_order = 12
	X_p = poly_expand(x_poly; order = poly_order)[:, 2:end]
	path_r = glmnet(X_p, y_poly, alpha = 0.0, lambda = exp.([-10:0.5:1.5;]) )
	# path_l = glmnet(X_p, y_poly, alpha = 1.0,lambda = exp.([-11:.1:4;]) )
	path_r
end;

# ╔═╡ 8cc45465-78e7-4777-b0c7-bb842e4e51a8
let
	gr()
	poly_order = [0, 1, 2, 3]
	ylim = [extrema(y_poly)...;]
	ylim += [-1.0, 1.0]
	plots_ =[]
	for p in poly_order
		plt = plot(x_poly, y_poly, st=:scatter, label="")
		w, loss = poly_reg(x_poly, y_poly; order=p)
		plot!(-2:0.02:2, (x) -> poly_fun(x, w), lw=2, lc=:red,  label="Order: "*L"p=%$(p)", legend=:outerbottom, ylim=ylim, title="P-order: $(p);" *" training loss: "*L"%$(round(loss; digits=2))")
		if add_true
		plot!(-2:0.05:2, (x) -> poly_fun(x, w_poly), label="", lw=2, lc=1, size=(600, 400))
		end
		push!(plots_, plt)
	end
	plot(plots_..., size=(900, 600))
end

# ╔═╡ 16bae168-3411-4f95-a4ba-3d3370d6494d
let
	gr()
	poly_order = [15, 19, 22, 25]
	ylim = [extrema(y_poly)...;]
	ylim += [-1.0, 1.0]
	plots_ =[]
	for p in poly_order
		plt = plot(x_poly, y_poly, st=:scatter, label="")
		w, loss = poly_reg(x_poly, y_poly; order=p)
		plot!(-2:0.02:2, (x) -> poly_fun(x, w), lw=2, lc=:red,  label="Order: "*L"p=%$(p)", legend=:outerbottom, ylim=ylim, title="P-order: $(p);" *" training loss: "*L"%$(round(loss; digits=2))")
		if add_true_overfit
		plot!(-2:0.05:2, (x) -> poly_fun(x, w_poly), label="", lw=2, lc=1, size=(600, 400))
		end
		push!(plots_, plt)
	end
	plot(plots_..., size=(900, 600), legendfontsize=15)
end

# ╔═╡ 5751ba57-a20c-4cf0-b6ba-70c8f97025cc
let
	gr()
	poly_order = 12
	λ = λ_
	# λs = [0, 0.5, 1, 10, 20, exp(30)]
	# plots_ =[]
	# for λ in λs
	plt = plot(x_poly, y_poly, st=:scatter, ms=3, mc=1,alpha=0.5, label="train data")
	plot!(x_poly_test, y_poly_test, st=:scatter, ms=3, mc=2,alpha=0.5,  label="test data")
	x_p = poly_expand(x_poly; order = poly_order)
	w = ridge_reg(x_p, y_poly; λ = λ)
	loss_test = norm([poly_fun(x, w) for x in x_poly_test] - y_poly_test)/length(y_poly_test)
	w_lsq = ridge_reg(x_p, y_poly; λ = 0)
	plot!(-2:0.01:2, (x) -> poly_fun(x, w_lsq), lw=2, lc=:blue,  label=L"\mathbf{w}_{lsq}", ls=:dash, legend=:outerbottom, ylim=[-7, 7], title=L"\lambda=%$(round(λ; digits=1))")
	if add_ridge
		plot!(-2:0.01:2, (x) -> poly_fun(x, w), lw=2, lc=:red,  label="", legend=:outerbottom, ylim=[-7, 7], title=L"\lambda=%$(round(λ; digits=1))")
	end

	plt
		# push!(plots_, plt)
	# end
	# plot(plots_..., size=(900, 600))
end

# ╔═╡ 15f95599-a536-434b-9365-da08b333bdbe
x_poly_data, y_poly_data = let
	# gr()
	Random.seed!(123)
	x_poly = [range(-1.8, -0.5, length=8)... range(.5, 1.8, length=8)...][:]
	nobs = 80
	x_poly_data = rand(nobs) * 4 .- 2.
	w_poly = [0.0, -5, 5, 2, -2]
	y_poly = [poly_fun(x, w_poly) for x in x_poly] + randn(length(x_poly))
	y_poly_data = [poly_fun(x, w_poly) for x in x_poly_data] + randn(length(x_poly_data)) * 1.5

	x_poly_data, y_poly_data
end;

# ╔═╡ a7f6ca87-b99d-480f-815b-d764f3245cb8
begin
	Random.seed!(2345)
	x_train = poly_expand(x_poly_data; order= 12)[:, 2:end]
	poly_cv = glmnetcv(x_train, y_poly_data; alpha=0.0, nfolds=10, lambda = exp.([-10:0.5:1.5;]) )
end;

# ╔═╡ a3927c61-126b-4252-b374-ac8869a3155b
vali_rst = let
	Random.seed!(123)
	KK = 10
	cv = CV(; nfolds=KK,  shuffle=true, rng=nothing)
	splits = MLJBase.train_test_pairs(cv, 1:length(y_poly_data))
	max_p = 10
	num_models = max_p + 1
	validation_rst = []
	for splt in splits
		train_idx, vali_idx = splt
		
		losses = zeros(num_models)
		for p in 0:max_p
			# train
			w, train_loss = poly_reg(x_poly_data[train_idx], y_poly_data[train_idx]; order = p)
			# test on validation
			losses[p+1] = norm([poly_fun(x, w) for x in x_poly_data[vali_idx]] - y_poly_data[vali_idx])/length(vali_idx)
		end
		push!(validation_rst, losses)
	end
	
	validation_rst
end;

# ╔═╡ 718d7ad5-5c27-4b67-9cd1-afec7585d1cf
let

	trial_id = 2
	plt_vali = plot(0:1:10, vali_rst[trial_id], xticks=0:10, label="Single validation", legend =:outerbottom, framestyle=:semi, xlabel="Polynormial order: "*L"p", ylabel="Validation MSE", title="Single validation set to tune polynomial order", lw=2, ylim =[0.4, 1.6])
	# for (k, rst) in enumerate(vali_rst[2:end])
	# 	plot!(plt_vali, 0:1:10, rst, label="Fold "*string(k+1))
	# end
	# errorline!(0:10, hcat(vali_rst...), errorstyle=:stick, label="Ribbon")
	vline!([4], lw=2, lc=:gray, ls=:dash, label="true "*L"p")
	
	scatter!([argmin(vali_rst[trial_id])-1], [minimum(vali_rst[trial_id])], markershape=:diamond, mc=1, label="minimum order by the validation", markersize=8, legendfontsize=14, size=(800,560), labelfontsize=15)
	plt_vali
end

# ╔═╡ d66f6274-361a-476e-9075-1890ae85a528
begin
	plt_vali = plot(0:1:10, vali_rst[1], xticks=0:10, label="Fold 1", legend =:outerright, framestyle=:semi, xlabel="Polynormial order: "*L"p", ylabel="Validation MSE", title="10-Fold Cross Validation")
	scatter!([argmin(vali_rst[1])-1], [minimum(vali_rst[1])], markershape=:diamond, mc=1, label="", markersize=4, legendfontsize=10, size=(750,450), labelfontsize=18)
	for (k, rst) in enumerate(vali_rst[2:end])
		plot!(plt_vali, 0:1:10, rst, label="Fold "*string(k+1), lc=k+1, lw=2.0)
		scatter!([argmin(rst)-1], [minimum(rst)], markershape=:diamond, mc=k+1, label="", markersize=4)
	end
	# errorline!(0:10, hcat(vali_rst...), errorstyle=:stick, label="Ribbon")
	plt_vali
end

# ╔═╡ d6d0e191-d2e1-4761-9b4a-de5eb4273df5
let
	plt_vali = plot(0:1:10, vali_rst[1], xticks=0:10, label="Fold 1", legend =:outerright, framestyle=:semi, xlabel="Polynormial order: "*L"p", ylabel="CV MSE", title="10-fold Cross Validation", alpha=0.3)
	for (k, rst) in enumerate(vali_rst[2:end])
		plot!(plt_vali, 0:1:10, rst, label="Fold "*string(k+1), alpha=0.3)
	end
	sems = [sem(hcat(vali_rst...)'[:,i]) for i in 1:11]
	plot!(0:10, mean(hcat(vali_rst...)', dims=1)[:], label="Mean", lw=2)
	yerror!(0:10, mean(hcat(vali_rst...)', dims=1)[:]; yerror = sems)
	# errorline!(0:10, hcat(vali_rst...), errorstyle=:stick, label="Mean", lw=2, secondarycolor=:matched)
	vline!([4], lw=1.5, ls=:dash, label="")
	plt_vali
end

# ╔═╡ c40dbfea-6668-46ca-9ff1-01ea9ee28abe
begin
	poly_cv_losses = hcat(vali_rst...)'
	poly_cv_sems = [sem(c) for c in eachcol(poly_cv_losses)]
	plot(poly_cv.lambda, poly_cv.meanloss , xscale=:log10, legend=:outerright, yerror = poly_cv_sems, label="",  xlabel=L"\lambda", ylabel="CV MSE", xflip=true)
	vline!([lambdamin(poly_cv)], label="minimum", lw=2, ls=:dash, title="CV Ridge regression's hyperparameter")

	vline!([poly_cv.lambda[13]], lw=2, ls=:dash, label="within 1 std error")
end

# ╔═╡ bc513037-c689-4eca-865d-d94a6a8aa997
begin
	Random.seed!(123)
	true_f(x) = sin(x)
	nobs_ = 25
	xs_q4 = collect(range(0, 2π, nobs_))
	ys_q4 = true_f.(xs_q4) + randn(nobs_)/4

end;

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Convex = "f65535da-76fb-5f13-bab9-19810c17039a"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLMNet = "8d5ece8b-de18-5317-b113-243142960cc6"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Logging = "56ddb016-857b-54e1-b83d-db4d58db5568"
MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458"
MLJBase = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SCS = "c946c3f1-0d1f-5ce8-9dea-7daa1f7e2d13"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.10.12"
Convex = "~0.15.4"
DataFrames = "~1.6.1"
GLMNet = "~0.7.2"
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.3.1"
Latexify = "~0.16.1"
LogExpFunctions = "~0.3.26"
MLDatasets = "~0.7.14"
MLJBase = "~1.1.1"
Plots = "~1.39.0"
PlutoTeachingTools = "~0.2.14"
PlutoUI = "~0.7.54"
SCS = "~2.0.0"
StatsBase = "~0.34.2"
StatsPlots = "~0.15.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "4a47dbd5f9e318aba553ff36152a7a545176033b"

[[deps.AMD]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse_jll"]
git-tree-sha1 = "45a1272e3f809d36431e57ab22703c6896b8908f"
uuid = "14f7f29c-3bd6-536c-9a0b-7339e30b5a3e"
version = "0.5.3"

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

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "0ba8f4c1f06707985ffb4804fdad1bf97b233897"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.41"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

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

[[deps.ArgCheck]]
git-tree-sha1 = "680b3b8759bd4c54052ada14e52355ab69e07876"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.4.0"

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

[[deps.AtomsBase]]
deps = ["LinearAlgebra", "PeriodicTable", "Preferences", "Printf", "Requires", "StaticArrays", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "5f76425eb977584353191c41d739e7783f036b90"
uuid = "a963bdd2-2df7-4f54-a1ee-49d51e6be12a"
version = "0.5.1"

    [deps.AtomsBase.extensions]
    AtomsBaseAtomsViewExt = "AtomsView"

    [deps.AtomsBase.weakdeps]
    AtomsView = "ee286e10-dd2d-4ff2-afcb-0a3cd50c8041"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "2c7cc21e8678eff479978a0a2ef5ce2f51b63dff"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.5.0"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "e2144b631226d9eeab2d746ca8880b7ccff504ae"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.3"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "e38fbc49a620f5d0b660d7f543db1009fe0f8336"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BufferedStreams]]
git-tree-sha1 = "6863c5b7fc997eadcabdbaf6c5f201dc30032643"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.2.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"
weakdeps = ["JSON", "RecipesBase", "SentinelArrays", "StructTypes"]

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

[[deps.CategoricalDistributions]]
deps = ["CategoricalArrays", "Distributions", "Missings", "OrderedCollections", "Random", "ScientificTypes"]
git-tree-sha1 = "926862f549a82d6c3a7145bc7f1adff2a91a39f0"
uuid = "af321ab8-2d2e-40a6-b165-3d674595d28e"
version = "0.1.15"

    [deps.CategoricalDistributions.extensions]
    UnivariateFiniteDisplayExt = "UnicodePlots"

    [deps.CategoricalDistributions.weakdeps]
    UnicodePlots = "b8865327-cd53-5732-bb35-84acbb429228"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Chemfiles]]
deps = ["AtomsBase", "Chemfiles_jll", "DocStringExtensions", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "b7b6140d72877345f03ce09d2cd97c0115f43c52"
uuid = "46823bd8-5fb3-5f92-9aa0-96921f3dd015"
version = "0.10.42"

[[deps.Chemfiles_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f3743181e30d87c23d9c8ebd493b77f43d8f1890"
uuid = "78a364fa-1a3c-552a-b4bb-8fa0f9c1fcca"
version = "0.10.4+0"

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

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "TranscodingStreams"]
git-tree-sha1 = "84990fa864b7f2b4901901ca12736e45ee79068c"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.5"

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

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

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

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Convex]]
deps = ["AbstractTrees", "BenchmarkTools", "LDLFactorizations", "LinearAlgebra", "MathOptInterface", "OrderedCollections", "SparseArrays", "Test"]
git-tree-sha1 = "e84e371b9206bdd678fe7a8cf809c7dec949e88f"
uuid = "f65535da-76fb-5f13-bab9-19810c17039a"
version = "0.15.4"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataDeps]]
deps = ["HTTP", "Libdl", "Reexport", "SHA", "Scratch", "p7zip_jll"]
git-tree-sha1 = "8ae085b71c462c2cb1cfedcb10c3c877ec6cf03f"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.13"

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

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "2ec417fc319faa2d768621085cc1feebbdee686b"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.23"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GLMNet]]
deps = ["DataFrames", "Distributed", "Distributions", "Printf", "Random", "SparseArrays", "StatsBase", "glmnet_jll"]
git-tree-sha1 = "b873c384d3490304c18224b1d5554cdebaafb60b"
uuid = "8d5ece8b-de18-5317-b113-243142960cc6"
version = "0.7.4"

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

[[deps.GZip]]
deps = ["Libdl", "Zlib_jll"]
git-tree-sha1 = "0085ccd5ec327c077ec5b91a5f937b759810ba62"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.6.2"

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

[[deps.Glob]]
git-tree-sha1 = "97285bbd5230dd766e9ef6749b80fc617126d496"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.1"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "MPIPreferences", "Mmap", "Preferences", "Printf", "Random", "Requires", "UUIDs"]
git-tree-sha1 = "e856eef26cf5bf2b0f95f8f4fc37553c72c8641c"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.17.2"

    [deps.HDF5.extensions]
    MPIExt = "MPI"

    [deps.HDF5.weakdeps]
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "38c8874692d48d5440d5752d6c74b0c6b0b60739"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.2+1"

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

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "50aedf345a709ab75872f80a2779568dc0bb461b"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.11.2+3"

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

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

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

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "0f14a5456bdc6b9731a5682f439a672750a09e48"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.0.4+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InternedStrings]]
deps = ["Random", "Test"]
git-tree-sha1 = "eb05b5625bc5d821b8075a77e4c421933e20c76b"
uuid = "7d512f48-7fb1-5a58-b986-67e6dc259f01"
version = "0.7.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

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
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "Requires", "TranscodingStreams"]
git-tree-sha1 = "91d501cb908df6f134352ad73cde5efc50138279"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.5.11"

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

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "1d322381ef7b087548321d3f878cb4c9bd8f8f9b"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.1"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

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

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

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

[[deps.LDLFactorizations]]
deps = ["AMD", "LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "70f582b446a1c3ad82cf87e62b878668beef9d13"
uuid = "40e66cde-538c-5869-a4ad-c39174c6795b"
version = "0.10.1"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

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

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LearnAPI]]
deps = ["InteractiveUtils", "Statistics"]
git-tree-sha1 = "ec695822c1faaaa64cee32d0b21505e1977b4809"
uuid = "92ad9a40-7767-427a-9ee6-6e577f1266cb"
version = "0.1.0"

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

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "1d2dd9b186742b0f317f2530ddcbf00eebb18e96"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.7"

[[deps.MIMEs]]
git-tree-sha1 = "1833212fd6f580c20d4291da9c1b4e8a655b128e"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.0.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "5de60bc6cb3899cd318d80d627560fae2e2d99ae"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.0.1+1"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLDatasets]]
deps = ["CSV", "Chemfiles", "DataDeps", "DataFrames", "DelimitedFiles", "FileIO", "FixedPointNumbers", "GZip", "Glob", "HDF5", "ImageShow", "JLD2", "JSON3", "LazyModules", "MAT", "MLUtils", "NPZ", "Pickle", "Printf", "Requires", "SparseArrays", "Statistics", "Tables"]
git-tree-sha1 = "361c2692ee730944764945859f1a6b31072e275d"
uuid = "eb30cadb-4394-5ae3-aed4-317e484a6458"
version = "0.7.18"

[[deps.MLJBase]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Dates", "DelimitedFiles", "Distributed", "Distributions", "InteractiveUtils", "InvertedIndices", "LearnAPI", "LinearAlgebra", "MLJModelInterface", "Missings", "OrderedCollections", "Parameters", "PrettyTables", "ProgressMeter", "Random", "RecipesBase", "Reexport", "ScientificTypes", "Serialization", "StatisticalMeasuresBase", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "f4782ed751d4683a2858278ef2997130a82ca710"
uuid = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
version = "1.1.2"

    [deps.MLJBase.extensions]
    DefaultMeasuresExt = "StatisticalMeasures"

    [deps.MLJBase.weakdeps]
    StatisticalMeasures = "a19d573c-0a75-4610-95b3-7071388c7541"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "ceaff6618408d0e412619321ae43b33b40c1a733"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.11.0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "6963295133aaa789f5fb18a6dd276c420793cf43"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.7"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "7715e65c47ba3941c502bffb7f266a41a7f54423"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.2.3+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "70e830dab5d0775183c99fc75e4c24c614ed7142"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.1+2"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON3", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "f5f67affb675630421d169ffd5dfa00cffcc07ca"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.35.2"

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

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

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

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "9c0bc309df575c85422232eedfb74d5a9c155401"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.6.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "bdc9d30f151590aca0af22690f5ab7dc18a551cb"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.27"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NPZ]]
deps = ["FileIO", "ZipFile"]
git-tree-sha1 = "60a8e272fe0c5079363b28b0953831e2dd7b7e6f"
uuid = "15e1cf62-19b3-5cfa-8e77-841668bca605"
version = "0.4.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "cc0a5deefdb12ab3a096f00a6d42133af4560d71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

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

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ece4587683695fe4c5f20e990da0ed7e83c351e7"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.29+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "2dace87e14256edb1dd0724ab7ba831c779b96bd"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.6+0"

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

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

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

[[deps.PeriodicTable]]
deps = ["Base64", "Unitful"]
git-tree-sha1 = "238aa6298007565529f911b734e18addd56985e1"
uuid = "7b2266bf-644c-5ea3-82d8-af4bbd25a884"
version = "1.2.1"

[[deps.Pickle]]
deps = ["BFloat16s", "DataStructures", "InternedStrings", "Mmap", "Serialization", "SparseArrays", "StridedViews", "StringEncodings", "ZipFile"]
git-tree-sha1 = "e99da19b86b7e1547b423fc1721b260cfbe83acb"
uuid = "fbb45041-c46e-462f-888f-7c521cafbc2c"
version = "0.3.5"

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

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

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

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
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

[[deps.SCS]]
deps = ["MathOptInterface", "Requires", "SCS_jll", "SparseArrays"]
git-tree-sha1 = "aa3fcff53da363b4ba4b54d4ac4c9186ab00d703"
uuid = "c946c3f1-0d1f-5ce8-9dea-7daa1f7e2d13"
version = "2.0.2"

    [deps.SCS.extensions]
    SCSSCS_GPU_jllExt = ["SCS_GPU_jll"]
    SCSSCS_MKL_jllExt = ["SCS_MKL_jll"]

    [deps.SCS.weakdeps]
    SCS_GPU_jll = "af6e375f-46ec-5fa0-b791-491b0dfa44a4"
    SCS_MKL_jll = "3f2553a9-4106-52be-b7dd-865123654657"

[[deps.SCS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl", "OpenBLAS32_jll"]
git-tree-sha1 = "902cc4e42ecca21bbd74babf899b2a5b12add323"
uuid = "f4f2fc5b-1d94-523c-97ea-2ab488bedf4b"
version = "3.2.7+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "75ccd10ca65b939dab03b812994e571bf1e3e1da"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.0.2"

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
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

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

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

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

[[deps.StatisticalMeasuresBase]]
deps = ["CategoricalArrays", "InteractiveUtils", "MLUtils", "MacroTools", "OrderedCollections", "PrecompileTools", "ScientificTypesBase", "Statistics"]
git-tree-sha1 = "e4f508cf3b3253f3eb357274fe36fb3332ca9896"
uuid = "c062fc1d-0d66-479b-b6ac-8b44719de4cc"
version = "0.1.2"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "542d979f6e756f13f862aa00b224f04f9e445f11"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.4.0"

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
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

[[deps.StridedViews]]
deps = ["LinearAlgebra", "PackageExtensionCompat"]
git-tree-sha1 = "5b765c4e401693ab08981989f74a36a010aa1d8e"
uuid = "4db3bf67-4bd7-4b4e-b153-31dc3fb37143"
version = "0.2.2"

    [deps.StridedViews.extensions]
    StridedViewsCUDAExt = "CUDA"

    [deps.StridedViews.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "b765e46ba27ecf6b44faf70df40c57aa3a547dcb"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

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

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

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

[[deps.UnitfulAtomic]]
deps = ["Unitful"]
git-tree-sha1 = "903be579194534af1c4b4778d1ace676ca042238"
uuid = "a7773ee8-282e-5fa2-be4e-bd808c38a91a"
version = "1.0.0"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

    [deps.UnsafeAtomics.weakdeps]
    LLVM = "929cbde3-209d-540e-8aea-75f648917ca0"

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

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

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

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

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

[[deps.glmnet_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "31adae3b983b579a1fbd7cfd43a4bc0d224c2f5a"
uuid = "78c6b45d-5eaf-5d68-bcfb-a5a2cb06c27f"
version = "2.0.13+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0ba42241cb6809f1a278d0bcb976e0483c3f1f2d"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+1"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f5733a5a9047722470b95a81e1b172383971105c"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.3+0"

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
# ╟─17a3ac47-56dd-4901-bb77-90171eebc8c4
# ╟─29998665-0c8d-4ba4-8232-19bd0de71477
# ╟─cb72ebe2-cea8-4467-a211-5c3ac7af74a4
# ╟─e141fcb3-3912-48cd-a42d-5a777332031d
# ╟─c39974a0-744c-478a-bfed-0ef97412f0c0
# ╟─358dc59c-8d06-4272-9a13-6886cdaf3dd9
# ╟─9bd2e7d6-c9fb-4a67-96ef-049f713f4d53
# ╟─882da1c1-1358-4a40-8f69-b4d7cbc9387e
# ╟─0dff10ec-dd13-4cc2-b092-9e72454763cc
# ╟─987ea5cb-b656-46f1-ae64-02e47fe32c9e
# ╟─75bc5b4c-7935-494f-ab11-c0908774800f
# ╟─c91fc381-613c-4c9b-98b6-f4dd897a69a5
# ╟─726a1008-26e6-417d-a73a-57e32cb224b6
# ╟─fc2206f4-fd0f-44f8-ae94-9dae77022fab
# ╟─05837852-8838-40e8-b8de-1b01b5c57994
# ╟─8cc45465-78e7-4777-b0c7-bb842e4e51a8
# ╟─d771fc78-09ae-4ce3-9ea3-514addbfd8a0
# ╟─2e9e4586-3e05-4ecf-9d56-3fa0bf2cdb70
# ╟─0f34bfca-89f2-494f-9991-db0f13e2c0a3
# ╟─16bae168-3411-4f95-a4ba-3d3370d6494d
# ╟─db7d9963-b613-4350-be0b-30a479da2682
# ╟─afd07c02-3538-4bb4-a072-d9a66734bd6f
# ╟─21b6b547-66ed-4231-830b-1a09adaf400c
# ╟─d7ca4793-2595-426c-a324-d85fec9feffd
# ╟─c6cd66f8-f921-4a0c-be5c-94daf87862c8
# ╟─6fa3941b-5975-463c-9427-2ebad15f78eb
# ╟─7c6520b2-0446-44c0-a0b5-5aacff87cf84
# ╟─17db14e7-b62c-432e-b33f-8f0b20b67a99
# ╟─5965c389-2d9e-4d0a-8ae0-9868b360e12e
# ╟─3fafc04c-c46a-4a12-ba95-8dad1ff879e0
# ╟─bc756099-80e2-400a-97cb-c664f593c964
# ╟─9ed65430-fe12-4e83-8368-e6a6c62715f4
# ╟─98d86dad-19a2-4d7a-b593-69a390ef8d68
# ╟─be68ba6a-21e2-4a4e-8f37-9391442242f7
# ╟─9b39a0b0-11df-4772-bb2e-a82fe219929c
# ╟─19d2b878-18e5-47ee-9458-c18b2e29341e
# ╟─cb9e6893-eb31-4a32-8764-700b472e5606
# ╟─d625f158-f0fb-43c8-92eb-a447b7a23d2b
# ╠═24d7b69b-3b58-49e2-93a7-42ad0cafe587
# ╠═b372df13-9963-4df8-a0da-37485b3775b8
# ╟─2c44745b-6561-42ae-ba0b-13b5cd0aeaa3
# ╟─a07cf1fa-1b04-4767-9f7a-671096a04501
# ╟─854f6ec9-daff-4660-9513-d5f8c709cf8d
# ╟─59e25758-84fb-48d4-8654-c7c87a51fdee
# ╟─510b7bd3-c4a8-4b9a-9f64-9bf02bf14d22
# ╟─b16feb9a-740e-473d-a7b8-1f0c761271de
# ╟─3e45d793-aefc-4151-b9f2-6d3557f834a2
# ╟─a7ee5e77-75ea-45f3-bc5d-a289a7f8c30a
# ╟─189fb02d-ab82-4580-8f03-cfde475e3b8e
# ╟─a519f36e-3055-4ddd-a8ac-c37fc3a7e2bd
# ╟─a0150656-3bde-477d-957e-f50b206cf6b1
# ╟─ce8cd98f-da70-476c-817c-026365b982c0
# ╟─6a9e0bd9-8a2a-40f0-aaac-bf32d39ffac8
# ╟─8436c5ce-d7be-4c02-9e89-7da701303263
# ╟─44508906-0740-401e-bbb0-ebb3c0158f4f
# ╟─df62e796-d711-4ea1-a0c2-e3ec6c501a78
# ╟─8493d307-9158-4821-bf3b-c368d9cd5fc5
# ╟─439810a2-7d65-4ace-96a8-5969be0ca6aa
# ╟─8e3b6b19-d209-4dd7-a58e-f4246c897b66
# ╟─8b73ae97-1c85-406f-830c-11a5079e934d
# ╟─ebdf7644-3383-46c2-bfdd-203a838daabf
# ╟─570a250d-3aec-4c5f-9089-57b308998f6d
# ╟─5751ba57-a20c-4cf0-b6ba-70c8f97025cc
# ╟─03edf15e-d365-44d2-8b4c-0c8a5bbe3432
# ╟─ccb21e90-b473-4caf-aa8e-81259feeb15b
# ╟─19fbdee1-1775-4c5c-b09f-324b24890a99
# ╟─eda59f0e-de36-4bed-9cf0-698f45a17211
# ╟─ed66e944-993d-4a43-8191-b8ee708413fb
# ╟─45ad2754-1f97-4223-8289-284a182d0970
# ╟─5ba6caf3-23a7-485e-a530-2a3ece0c3e71
# ╟─aaceb96b-0928-4ae0-b65e-c23344673967
# ╟─f8e4c992-8d31-4f6f-9a9e-a12d0efde1b1
# ╟─763da9a9-5949-4bcf-a703-803fa2c2e82c
# ╟─0585ce12-41a3-43cd-8e73-96b349baf9eb
# ╟─aa697720-0662-4810-9957-b4489d698a83
# ╟─ea3b6e91-26af-4079-a8be-c3e90ba5b265
# ╟─5ebaf2c4-e23d-4835-9411-f4973a3e1ef8
# ╟─fe020885-dd23-4e75-9df1-3e3b8fdef64b
# ╟─4fb1c0b4-4368-461e-a12c-0629d72d0379
# ╟─8d6912ae-6756-4085-b912-10754c82656b
# ╟─630ac272-b9ea-4871-98ba-fb91c844a289
# ╟─397e3e8b-8247-47f6-8ae6-91b2546089d3
# ╟─47050d18-06c3-4e9c-8c9a-6488cd765453
# ╟─11d41404-59c4-42da-b402-850341066208
# ╟─e7e4bd42-cb98-4aae-b027-c0bc622e6c42
# ╟─757716d0-ee70-4a8b-951f-4253ca9eda40
# ╟─b4026ef7-f5fd-4240-9a4f-de82e7f89e9e
# ╟─cf636fb4-3631-403b-9c2d-a25676d8bb1f
# ╟─69949573-c43f-4568-9357-8d1e71398d88
# ╟─bbe92f12-b863-4b51-9669-0ae8b39f32b1
# ╟─9c25741d-960c-41c7-b5c6-da2852406ca1
# ╟─d7f125aa-27e5-4484-b0d8-b38caf650a82
# ╟─36f97ad7-7bb0-4b41-9752-384002a53e8e
# ╟─268a33f8-4ded-4dbc-8141-0f81ba1c0384
# ╟─b0aad1e2-580b-480e-9be5-3ad8c9702763
# ╟─d407c511-6cf6-4eb0-aabd-25a3e7f5857c
# ╟─de9fac90-7b4f-4034-a62a-6ef4bd9f01df
# ╟─6e00611c-6403-431a-a331-347b62cf2c69
# ╟─aa36dda8-29df-4592-a3f1-e8c1ec0955dc
# ╟─102adf0c-72c8-4987-99fe-ff96daba0e74
# ╟─2b391a81-7779-4912-aadc-b513468e029b
# ╟─c4e06848-6ca5-48e5-ab6a-b8ab92ec2702
# ╟─d8dcb7ab-a538-45dd-a47b-144b1b87e33f
# ╟─b61511c3-1ae8-4e30-aa55-62ecc9638f42
# ╟─07638f73-df94-4a8e-a511-eb11f040b01b
# ╟─dca81d78-d851-413c-91f8-439f99c4bda3
# ╟─cacf7514-de14-402a-b85f-ff84e90f9790
# ╟─bbd86cd2-f298-46f4-8684-c33d11277746
# ╟─369472e6-1b58-4310-8453-4ea86b33e807
# ╟─cd368db5-ae59-4eac-b3d0-3847c9c95c4c
# ╟─f3bdec6f-4280-4dd0-8ded-a4a896423f2a
# ╟─3195abbb-c8ab-4698-aa7c-657b27d8b24c
# ╟─c6c85ad7-decf-462a-84a3-cbb9bdc0f745
# ╟─cf483b66-3eed-441d-ab97-13b4509cdc8f
# ╟─d93fbb13-da70-4303-8d5a-4c592fcdd918
# ╟─02334a5c-41eb-41a2-a77c-041a2fc1c66a
# ╟─56903a30-f3fc-4f93-9572-231935295d0f
# ╟─bd4ea5c5-164e-4604-94e2-0e018848be05
# ╟─45572dce-33d3-4dd2-9eb8-61cb063cf0e2
# ╟─9c27da2e-9ab6-4106-af9f-51d6ca693bdb
# ╟─e45c01fe-9b48-4011-8aab-8a70396351d5
# ╟─e2906f07-e97c-4ddb-8013-86e2fe746064
# ╟─147f0ed7-74f4-40a8-9898-30fe62ff3df7
# ╟─7fe3f38a-16bf-4d18-b495-359e2084c07c
# ╟─f8730c9e-d319-4dd4-99f1-401252daf6b2
# ╟─d0a902c5-be1c-4752-93ac-0179513d8565
# ╟─e6d9a65b-e813-42b4-ba1d-2b0a3a72b9a7
# ╟─99833f6e-813d-4d74-a8ca-df4048dcf704
# ╟─0434873d-ef2f-4538-a27d-8d44316cd40f
# ╟─b1d1db29-f7c1-4e27-b54a-1b8532759334
# ╟─40a4d260-f755-4958-af71-3c9f379b09b5
# ╟─59027056-dac5-417e-ba92-724c2e55d500
# ╟─d95cc8e7-0b55-44b2-a65d-69dd93e86a9a
# ╟─4b22e561-0adb-4c3e-803c-d7f880324876
# ╟─fb1ac118-b887-444a-b25d-6c33f96609c5
# ╟─2aefb863-aad7-45a9-b500-04479213b3a8
# ╟─c2edb58b-c0a6-4fae-9f83-3d3a069a8a3a
# ╟─cdef3ff8-04f8-463e-86cd-5377b57a5f8b
# ╟─718d7ad5-5c27-4b67-9cd1-afec7585d1cf
# ╟─1d76e05e-172d-4385-b4b1-2bdd09b94dac
# ╟─52917eba-fab4-421b-a582-83c586d7e297
# ╟─07215124-40ac-4795-9bf8-4d902d0e8137
# ╟─d66f6274-361a-476e-9075-1890ae85a528
# ╟─42f94b47-0a29-4ba0-8229-88021723ab35
# ╟─5335a2ec-2a18-4958-88b2-8b8c5b8cdbe5
# ╟─15f95599-a536-434b-9365-da08b333bdbe
# ╟─a3927c61-126b-4252-b374-ac8869a3155b
# ╟─7a3bb30e-2d4c-46e3-ac6a-d67874cab0d9
# ╟─d6d0e191-d2e1-4761-9b4a-de5eb4273df5
# ╟─a7f6ca87-b99d-480f-815b-d764f3245cb8
# ╟─138261fa-7736-4507-b8ad-4a1918a193e6
# ╟─92d13008-4aed-4633-81b5-954475034fbf
# ╟─c40dbfea-6668-46ca-9ff1-01ea9ee28abe
# ╟─45c57c4a-721a-4f24-bfcd-dfb3812de182
# ╟─ae8e2ab7-ffbb-4af0-a190-c30505a117dd
# ╟─974f1b58-3ec6-447a-95f2-6bbeda43f12f
# ╟─238e7b56-fb3a-4e9b-9c31-09c1f4a1df2a
# ╟─cb02aee5-d082-40a5-b799-db6b4af557f7
# ╟─8deb1b8c-b67f-4d07-8986-2333dbadcccc
# ╟─f79bd8ab-894e-4e7b-84eb-cf840baa08e4
# ╟─af622189-e504-4633-9d9e-ab16c7293f82
# ╟─9267c4a4-74d1-4515-95d6-acc3b12e5ed6
# ╟─76cc8ca7-d17e-4cd7-a6de-4d606e0a0985
# ╟─c4e497fc-cfbf-4d0b-9a0c-1071f2f43a98
# ╟─0e2dc755-57df-4d9a-b4f3-d01569c3fcde
# ╟─c4f42980-0e68-4943-abbe-b28f05dd3ee5
# ╟─8fbcf6c3-320c-47ae-b4d3-d710a120eb1a
# ╟─ce35ddcb-5018-4cb9-b0c9-01fb4b14be40
# ╟─edc245bc-6571-4e65-a50c-0bd4b8d63b74
# ╟─4154585d-4eff-4ee9-8f33-dad0dfdd143c
# ╟─46180264-bddc-47d8-90a7-a16d0ea87cfe
# ╟─bc513037-c689-4eca-865d-d94a6a8aa997
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
