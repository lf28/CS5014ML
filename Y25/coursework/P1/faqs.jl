### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ 5b093f9e-d651-11ee-008a-8b742b93680b
md"""
## FAQs

"""

# ╔═╡ eec2a196-bec0-4e55-8ad8-222064162c56
md"""

###### For the practical, would it be okay to also use Julia's Random package to be able to shuffle the matrix for question 1.4? If not, what would you recommend me read into to learn how to shuffle the matrix data?

* yes, `Random.jl` is allowed
"""

# ╔═╡ a7f6542b-2ecf-427e-95ac-c011a47bc4bb
md"""
###### Sorry if this was answered in the lectures, but I wanted to make sure I understand what the tolerance for gradient descent is. I have seen many different ways of using it e.g. how much the loss decrease since the last epoch, the difference in the norm of the gradient, etc. How would you expect us to use it for this practical?

* We don’t use the difference in the norm of the gradient. At convergence, the gradients are close to the zero vector, subtracting zero vector from another zero vector likely to lead to rounding error. And it does not in general translate to convergence (consider ``y=x``, the difference of the gradient is always 0). You can simply use the norm of the gradient, if the norm is small enough, the gradient is close to zero vector, adding it has little effect, which indicates convergence. Checking loss also makes sense, which directly indicates convergence. Either approach is valid. When you are not sure about something in ML, you need to go back to the first principle, i.e, maths. If the maths is right, your method is provably correct (subject to numerical issues).
"""

# ╔═╡ 43a93805-0af6-4c32-a3a6-9c7a185b608a
md"""


###### I was wondering if for each input vector we should return the value of the sigmoid function or should we output the predicted labels 0 or 1 based on the value of the sigmoid function, i.e. if it is greater than 0.5 then it will be a 1, otherwise it will be a 0. 


* Either is fine. To predict the labels, one just need to add a testing rule >.5 anyways. But when reporting accuracy, you need to make sure the accuray measure is correctly calculated.

###### In the context of these two tasks, what exactly does “Report the test performance on the unseen test dataset.” mean in terms of performance? Should we report the number of matches/mismatches, accuracy, or true/false positives/negatives?

* I think accuracy should be reported. But I’d also compare negative loglikelihood as it has much nicer statistical properties. For most statistical inferences, NLL should be the preferred measure. 
"""

# ╔═╡ bb487f95-1dfc-40a4-9418-1b5dbdb3a6b0
md"""

###### ...the extension from Task 2 and I was wondering if the additional fixed basis regression model should only learn the variance and not also the mean of the observations, is that right? My idea was that since the mean and the variance of the observations are separate concepts, and because the first part of Task 2 (the one where we use radial basis functions) is basically approximating the mean of the observations, I could keep the mean paramenters fixed, and just use them when defining the loss function based on Gaussian MLE (as shown in lecture) that I will only use to learn the variance. Is this one way of properly tackling the extension?


* You should learn both at the same time. But what you have proposed is a very sensible way to optimise the model: i.e. coordinate descent, learn mean first then learn the standard deviation, and then repeat the above two steps until converge. I don’t believe the noise scale is nonlinear though.


###### So wouldn’t I be able to first use the normal equation to learn the mean and then perform an iterative process to learn the standard deviation? Or would this yield a different result from learning the mean and the variance at the same time? Also isn’t the result from the normal equation the convergence point for the mean?


* Not really. If you write down the nll, given the standard deviations the loss wrt the mean is no longer sum of squared errors but weighted sum, as the errors std are not constant. But I reckon the results won’t differ much. But weighted sum still has a closed form solution though. You do not need to derive the closed form solution if you can’t. You can simply use gradient descent. There’s no the correct way when it comes to optimisation. I may come up with 10 different optimisation routines and they all are valid. I certainly would prefer one to another. But I won’t be picky when it comes to marking an extension. Just come up with one solution and convince yourself (or me) it is correct. 
"""

# ╔═╡ Cell order:
# ╟─5b093f9e-d651-11ee-008a-8b742b93680b
# ╟─eec2a196-bec0-4e55-8ad8-222064162c56
# ╟─a7f6542b-2ecf-427e-95ac-c011a47bc4bb
# ╟─43a93805-0af6-4c32-a3a6-9c7a185b608a
# ╟─bb487f95-1dfc-40a4-9418-1b5dbdb3a6b0
