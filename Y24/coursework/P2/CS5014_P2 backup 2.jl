### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 23239412-28fc-445a-9616-0dd5370d4a60
begin
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # Plot's text style
	using LinearAlgebra, StatsBase, Random, Zygote
	using LogExpFunctions
	using CSV, DataFrames
end

# ╔═╡ e28e1279-d267-4401-92f1-4f490e795be7
using Clustering

# ╔═╡ e928bee6-8009-42fc-8cfa-d230b0a7e091
begin
	using MLDatasets
	ENV["DATADEPS_ALWAYS_ACCEPT"] = true
	mnist_train_X, mnist_train_ys = MNIST(split=:train)[:];
	X_mnist_train = ((reshape(mnist_train_X, :, 60000) .> .5)')[1:4000, :]
	X_mnist_test = ((reshape(mnist_train_X, :, 60000) .> .5)')[4001:5000, :]
	Y_mnist_train = mnist_train_ys[1:4000]
	Y_mnist_test = mnist_train_ys[4001:5000]
end;

# ╔═╡ a6761964-021b-4722-953c-cee1585a3d3c
begin
	using PlutoUI, PlutoTeachingTools
end

# ╔═╡ df95822e-b5af-4f9d-b2ec-3cf9b25437f9
begin
	using Images, MosaicViews, ImageView
end

# ╔═╡ 8176b67f-3664-4ed9-9705-4052124617ff
begin
	TableOfContents()
end

# ╔═╡ cda9daf0-c214-11ee-00eb-21e793f9b4be
md"""

# CS5014 Machine Learning 

#### Practical 2
###### Credits: 50% of the coursework
"""

# ╔═╡ 2b80a240-24f6-4d5f-a60f-4f6e938184f6
md"""
## Aims


The objectives of this assignment are:

* deepen your understanding of probabilistic generative models
* deepen your understanding of the EM algorithm
* gain experience in implementing unsupervised learning algorithm
"""

# ╔═╡ 57d0435f-197c-4b2a-88c5-956e0542c13f
md"""

## Set-up

You are **only allowed** to use the following imported packages for this practical. No off-the-shelf machine learning packages are allowed to implement the two core algorithms, *i.e.* the Spherical Kmeans and the EM algorithm for mixture of Bernoulli model.

"""

# ╔═╡ 5a6b9f5d-2660-4cb7-8d99-8b5aaf4f63af
md"""
In addition, you are allowed to use `Clustering.jl` 

"""

# ╔═╡ 63d02691-2a88-464f-8e90-818f5a1404aa
md"""

* to have the ordinary `K-means` as a baseline
* to evaluate clustering performance of a clustering algorithm; read further for a quick demonstration
"""

# ╔═╡ 53f52309-9424-4669-8b1a-6f21742855a9
md"""


## Question 1 (Spherical K-means)

In this question, you are going to implement a variant of K-means algorithm, which is called *Spherical K-means*. And the algorithm is a popular clustering algorithm for text data.

### The dataset & K-means baseline
The 20 Newsgourps text dataset is used for this question. The dataset comprises around 18000 posts on 20 topics. For simplicity, we are going to use documents from the following three topics
* talk, religion & misc
* compter graphics
* science & space

The documents has been transformed by `Bag of Words` followed by `Tfidf` vectorisation (more details can be found [here](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)). As a result, each document is represented by a $\mathbb{R}^{3000}$ vector, each dimension corresponds to a word in the term dictionary. And there are in total $n=2501$ documents. The data, a matrix of size ${2501\times 3000}$ together with the corresponding encoding information, is imported for you below.
* `X_news`: a $2501 \times 3000$ document-term matrix to cluster
* `news_labels`: a $2501\times 1$ vector encoding the true topic/cluster of the documents
* `news_terms`: a $3000\times 1$ vector of the terms (`string` typed)
"""

# ╔═╡ 4b38cb51-1cc5-4f12-9642-511f71224fa7
X_news, news_labels, news_terms = let
	data1url = "https://leo.host.cs.st-andrews.ac.uk/CS5014/P2/newsdata.csv"
	file = download(data1url)
	X_news = Matrix(CSV.File(file, header=false) |> DataFrame)

	labelurl = "https://leo.host.cs.st-andrews.ac.uk/CS5014/P2/newsdata_labels.csv"
	labels = Int.(Array(CSV.File(download(labelurl), header=false, types=Float32) |> DataFrame)[:])

	termurl = "https://leo.host.cs.st-andrews.ac.uk/CS5014/P2/newsdata_terms.csv"
	terms = Array(CSV.File(download(termurl), header=false) |> DataFrame)[:]
	X_news, labels, terms
end;

# ╔═╡ 4294b63a-5d31-42eb-9230-0a89106655e2
md"""

#### Apply `Kmeans` & clustering evaluation

We aim to cluster the $3000$ documents in a unsupervised manner. Such a task is known as `topic modelling` in NLP. Ideally, we want similar themed documents are clustered together to form a "topic".

The code below demonstrates  
* apply `kmeans` from `Clustering.jl` to cluster the documents
* and evaluate the clustering performance by adjusted rand index & V-measure
"""

# ╔═╡ b05e9cb0-5bc6-44ce-b30e-e95f88b55490
km_rst = let
	Random.seed!(123) ## fix a random seed so the result is reproducible
	n_cluster = 3 ## there are three topics
	repeats = 5 ## repeat the algorithm 5 times due to clustering's local optimums
	bestrst = kmeans(X_news', n_cluster)
	for _ in 2:5
		rst = kmeans(X_news', n_cluster)
		## if the clustering result is better
		if rst.totalcost < bestrst.totalcost
			bestrst = rst
		end
	end
	bestrst
end;

# ╔═╡ b83623c0-0499-46a6-927e-a46be91d6b64
begin
	kmrd = randindex(news_labels, km_rst.assignments)[1] 	
	kmvm = vmeasure(news_labels, km_rst.assignments)
	kmrd, kmvm
end

# ╔═╡ c04db08b-2097-417b-8eb3-fbf5c8f09a81
md"""
We evaluate the `Kmeans` clustering by using adjusted rand index & V-measure
* the adjusted rand index (ranging from $-0.5$ to $1$; higher the better) is $(round(kmrd;digits=4)) 
* and V-measure (ranging from $0$ and $1$; higher the better) is $(round(kmvm; digits=4)) 
* both indicate that `K-means` performs really bad; in particular, a negative adjusted rand index indicates the clustering is worse than random guess
"""

# ╔═╡ 02bf1377-6c96-4b24-bb94-ced5d9d4e171
md"""
#### The mined "topics"
We can also mine the "topics" from the clustering result. That is, for each of the cluster center $\boldsymbol{\mu}_k \in \mathbb{R}^{3000}$, we find the top $T$, say $T=15$, most popular terms.
"""

# ╔═╡ 451206ad-e461-4bd4-945d-3cb93b660486
for (k, μₖ) in enumerate(eachcol(km_rst.centers))
	print("Cluster ", k, ": ")
	topics = news_terms[partialsortperm(μₖ, 1:15; rev=true)]
	println(join(topics, ", "))
end

# ╔═╡ b525997b-1f77-4bfb-9daa-866b76685cfc
md"""
`K-means` failed to group the documents in a clean and meaningful way (recall the three topics are religious, science-space and computer graphics)
* `cluster 1` corresponds to the computer graphic topic well
* but the rest of the two topics/clusters are a bit mixed
"""

# ╔═╡ eb932347-d521-49a5-aac1-8473794133b9
md"For you convenience, you may want to use the following method to report the top `max_t=15` topics of the $K$ clutering centers. It expects a $D\times K$ matrix as the centers, where each column is one center."

# ╔═╡ b0704b5b-43e3-48db-a865-e5670a76f6de
function print_the_topics(centers; max_t = 15)
	for (ci, c) in enumerate(eachcol(centers))
		print("Cluster ", ci, ": ")
		arr = news_terms[partialsortperm(c, 1:max_t; rev=true)]
		println(join(arr, ", "))
	end
end

# ╔═╡ 9184a2ad-d53d-4b31-b999-2e762d66adc9
print_the_topics(km_rst.centers; max_t = 15);

# ╔═╡ a472c908-407f-44d6-9253-c57d0564dcb0
md"""
### Spherical K-means

Your task is to implement a variant of K-means algorithm, which is called *Spherical K-means*. The algorithm is listed below.



**Initialisation step**: Start with randomly selecting $K$ data points as the centroid of $K$ clusters. 

**Assignment step**: *Spherical K-means* assigns a data point to the closest centroid based on *cosine distance* rather than Euclidean distance; specifically, for $i=1,\ldots, n$

$$z^{(i)} \leftarrow \arg\min_{k} \left (1- \frac{\boldsymbol{\mu}_k^\top \mathbf{x}^{(i)} }{\|\boldsymbol{\mu}_k\| \cdot \|\mathbf{x}^{(i)}\|}\right ),$$ * where $\boldsymbol{\mu}_k^\top \mathbf{x}^{(i)} = \sum_{j=1}^d {\mu}_{kj} \cdot {x}^{(i)}_{j}$ denotes the inner product and $\|\mathbf{x}\|$ is $L_2$ norm of a vector $\mathbf{x}$: $\|\mathbf{x}\| = \sqrt{\mathbf{x}^\top \mathbf{x}}$.

**Update step**: *Spherical K-means* updates the centroids such that they are unit one vectors; for $k=1,\ldots, K$

$$\boldsymbol{\mu}_k \leftarrow \frac{\sum_{i=1}^n \mathbb{1}(z^{(i)} =k) \cdot  \mathbf{x}^{(i)}}{\|\sum_{i=1}^n \mathbb{1}(z^{(i)} =k) \cdot \mathbf{x}^{(i)}\|}.$$ 

* Note that after the normalisation step, the centroids $\boldsymbol{\mu}_k$ are norm-one vectors: i.e. $\|\boldsymbol{\mu}_k\| = 1$ for $k=1,\ldots, K$.

**Repeat** the above two steps **until** the total cosine distance loss converges, where the loss is defined as

$$\texttt{loss} = \sum_{i=1}^n \left (1- \frac{\boldsymbol{\mu}_{z^{(i)}}^\top \mathbf{x}^{(i)} }{\|\boldsymbol{\mu}_{z^{(i)}}\| \cdot \|\mathbf{x}^{(i)}\|}\right ).$$



"""

# ╔═╡ 06c9ec9b-5da9-4766-931a-7e9a58290350
md"""


### Task 1.1 Implementation of Spherical K-means

Implement the `sphericalKmeans` algorithm. The method `sphericalKmeans` has

"""

# ╔═╡ 98502cac-3767-4cf0-b35e-85566e0f4744
TwoColumn(
md"""

**Inputs**:
* `data`: a $n\times d$ matrix to cluster, i.e. each row of $\texttt{data}$ is one observation $\mathbf{x}^{(i)}$
* `K`: the number of the clusters
* `tol`: tolerence of error, which is used to check whether the loss has converged so the iteration can stop
* `maxIters`: the maximum number of iterations that is allowed

""",

	md"""

**Outputs**:

* `losses`: the whole trajectory of losses over the iterations
* `zs`: the clustering labels
* `us`: the learnt $K$ centroids



	"""

	
)

# ╔═╡ 30ba25c9-ea6b-4ba5-ae08-18f7c0cfd681
Foldable("Hints", md"""



**Hint**: 
* write helper methods such as `assign_step` and `update_step`; 
* do plenty of unit tests whenever you finish implementing one method
* start with something simple and correct then refine it iteratively (to improve efficiency & vectorisation);
* Vectorise your code (use `numpy` as much as possible). If you use Python, you **should avoid writing plain loops at all cost** e.g. to iterate observations one-by-one individually; Julia's loop is very fast but vectorisation should be adopted to make it even faster
  * as an extra hint, part of the update step can be implemented as matrix matrix multiplication if you zero-hot encoding assignment vector `zs` 

* efficiency is not the most important concern, but it does matter; consider caching computed results rather than recomputing them in a loop over and over again

""")

# ╔═╡ 11dcd69d-8cab-4cd4-9f66-df301bb375b9
begin
	function sphericalKmeans(data, K=3, tol = 1e-4, maxIters = 100)
    	n, d = size(data)
    	losses = []
    	# initialisation: randomly assign K observations as centroids
    	# feel free to use a different but sensible initialisation method    
    	init_us_ids = rand(1:K, n)
		# we assume each column is a center; but you can also assume the rows are the centers
	    us = data[init_us_ids, :]'
	    zs = zeros(Int, n)
	    # loop until converge 
	    # for i in 1:maxIters
	        # assignment step
	
	        # update step
	        # convergence check        
    	return losses, zs, us
	end

end

# ╔═╡ ce27dbf9-0b96-42bd-ab65-37eafb21dc96
md"""

### Task 1.2 Evaluation

Run the algorithm on the news dataset `X_news` with $K =3$. Note that like K-means, Spherical K-means also suffers from bad initialisations. To deal with that, we can run the algorithm multiple times with different random initialisations. To make your life easier, you may want to write a wrapper method that does it automatically.

Please report the following information based on your results 
* the final loss and also plot the loss trajectory
* the corresponding adjusted rand index and V-measure
* the top 15 terms of each of the three centroids

If you run multiple times, you only need to report the results for the best one.
"""

# ╔═╡ e5647088-4864-423b-8429-b239865e8e87
md"""

### Task 1.3 Conceptual question


##### Compare the performance against K-means', what do you observe? 




"""

# ╔═╡ 9c23dbab-2f7c-478d-9e35-ebfde39259f8
md"""

*__Answer__*:

"""

# ╔═╡ be36e0a5-3aa1-44fd-b38e-7162fe1681b0
md"""


##### If Spherical Kmeans performs better, what might be the reason? 
* Hint: compare cosine distance vs Euclidean distance; what happen if the data's dimension is large?
"""

# ╔═╡ b579d610-07b7-46fe-a090-497723cfbad0
md"""

*__Answer__*:

"""

# ╔═╡ d385074d-e939-46cc-be89-6cd4854b61a9
md"""

## Question 2 (EM algorithm)



In this question, we are going to implement an EM algorithm for finite mixture of Bernoullis which can be used to cluster binary valued images and generate new images (it is called generative AI nowadays). 

We are going to use the famous MNIST handwritten digit dataset. The images are $28 \times 28$ binary matricies, *i.e.* the pixels take values in $\{0, 1\}$. The spatial structure of the images are ignored, so each image/matrix is collapsed as a $784$-dimensional binary vector. 

The dataset is imported below for you.
* `X_mnist_train`: a $4000 \times 784$ binary matrix, where each row is one image and there are 4000 images in the training data
* `Y_mnist_train`: the corresponding 4000 labels of the images
* `X_mnist_test`, `Y_mnist_test`: a further $1000 \times 784$ test data and labels
"""

# ╔═╡ d0b2732c-f4bd-473d-b833-5eab483ea253
md"To visualise the data, we can `reshape` the 784 vector back to a 28 by 28 matrix; and use `Gray.()` method. Check the following code."

# ╔═╡ 9908497f-cf5e-46e5-bebf-822ebf4fdb2c
reshape([Gray.(reshape(X_mnist_train[i,:], 28, 28)') for i in 1:10], 2, 5)

# ╔═╡ ec7ea0e6-8462-43a9-b4ca-253ae01c6c5a
md"""


### Finite mixture of Bernoullis
"""

# ╔═╡ 630fb039-5bec-4670-b1e8-d6240b07d25e
md"""

## Submission
Hand in via MMS: the completed Pluto notebook (both `.jl` and `.html`). Your notebook should be reproducible.


"""

# ╔═╡ 870b0cf2-534e-4762-9d35-09bf8d59c34e
md"""

## Marking
Your submission will be marked as a whole. 

* to get a grade above 7, you are expected to finish at least Task 1.1-1.2 to a good standard
* to get a grade above 10 and up to 13, you are expected to complete Task 1.1-1.4 to a good standard
* to get a grade above 13 and up to 17, you are expected to complete all tasks except 2.3 and 2.4 to a good standard
* to achieve a grade of 17-18, you are expected to finish all tasks except Task 2.4 flawlessly 
* to get 18+, you are expected to attempt all questions flawlessly


Marking is according to the standard mark descriptors published in the Student Handbook at:

https://info.cs.st-andrews.ac.uk/student-handbook/learning-teaching/feedback.html#GeneralMarkDescriptors


You must reference any external sources used. Guidelines for good academic practice are outlined in the student handbook at https://info.cs.st-andrews.ac.uk/student-handbook/academic/gap.html

"""

# ╔═╡ 97b66779-97d3-41cf-899b-19ce62aa9351
md"""
#### `Mosaicview` of the images
"""

# ╔═╡ 3a6a7ab7-e846-43ae-a88a-02412e2749b2
function vec2imgs(vecs, ndim=28)
	if ndims(vecs) == 1
		vecs = vecs'
	end
	n = size(vecs)[1]

	output = zeros(Float64, ndim, ndim, n)
	for i in 1:n
		output[:, :, i] = reshape(vecs[i, :], ndim, ndim)'
	end
	return Gray.(output)
end

# ╔═╡ 45d99ba9-fdc3-4072-8de1-4350a0be1ab9
mosaicview(X_mnist_train[1:10, :], fillvalue=.2, nrow = 2, npad=4, rowmajor=true)

# ╔═╡ Cell order:
# ╟─8176b67f-3664-4ed9-9705-4052124617ff
# ╟─cda9daf0-c214-11ee-00eb-21e793f9b4be
# ╟─2b80a240-24f6-4d5f-a60f-4f6e938184f6
# ╟─57d0435f-197c-4b2a-88c5-956e0542c13f
# ╠═23239412-28fc-445a-9616-0dd5370d4a60
# ╟─5a6b9f5d-2660-4cb7-8d99-8b5aaf4f63af
# ╠═e28e1279-d267-4401-92f1-4f490e795be7
# ╟─63d02691-2a88-464f-8e90-818f5a1404aa
# ╟─53f52309-9424-4669-8b1a-6f21742855a9
# ╠═4b38cb51-1cc5-4f12-9642-511f71224fa7
# ╟─4294b63a-5d31-42eb-9230-0a89106655e2
# ╠═b05e9cb0-5bc6-44ce-b30e-e95f88b55490
# ╠═b83623c0-0499-46a6-927e-a46be91d6b64
# ╟─c04db08b-2097-417b-8eb3-fbf5c8f09a81
# ╟─02bf1377-6c96-4b24-bb94-ced5d9d4e171
# ╠═451206ad-e461-4bd4-945d-3cb93b660486
# ╟─b525997b-1f77-4bfb-9daa-866b76685cfc
# ╟─eb932347-d521-49a5-aac1-8473794133b9
# ╟─b0704b5b-43e3-48db-a865-e5670a76f6de
# ╠═9184a2ad-d53d-4b31-b999-2e762d66adc9
# ╟─a472c908-407f-44d6-9253-c57d0564dcb0
# ╟─06c9ec9b-5da9-4766-931a-7e9a58290350
# ╟─98502cac-3767-4cf0-b35e-85566e0f4744
# ╟─30ba25c9-ea6b-4ba5-ae08-18f7c0cfd681
# ╠═11dcd69d-8cab-4cd4-9f66-df301bb375b9
# ╟─ce27dbf9-0b96-42bd-ab65-37eafb21dc96
# ╟─e5647088-4864-423b-8429-b239865e8e87
# ╟─9c23dbab-2f7c-478d-9e35-ebfde39259f8
# ╟─be36e0a5-3aa1-44fd-b38e-7162fe1681b0
# ╟─b579d610-07b7-46fe-a090-497723cfbad0
# ╟─d385074d-e939-46cc-be89-6cd4854b61a9
# ╠═e928bee6-8009-42fc-8cfa-d230b0a7e091
# ╟─d0b2732c-f4bd-473d-b833-5eab483ea253
# ╠═9908497f-cf5e-46e5-bebf-822ebf4fdb2c
# ╟─ec7ea0e6-8462-43a9-b4ca-253ae01c6c5a
# ╟─630fb039-5bec-4670-b1e8-d6240b07d25e
# ╟─870b0cf2-534e-4762-9d35-09bf8d59c34e
# ╠═a6761964-021b-4722-953c-cee1585a3d3c
# ╟─97b66779-97d3-41cf-899b-19ce62aa9351
# ╠═df95822e-b5af-4f9d-b2ec-3cf9b25437f9
# ╠═3a6a7ab7-e846-43ae-a88a-02412e2749b2
# ╠═45d99ba9-fdc3-4072-8de1-4350a0be1ab9
