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

# ╔═╡ 9edaaf3c-2973-11ed-193f-d500cbe5d239
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

# ╔═╡ f932c963-31d2-4e0f-ad40-2ca2447f4059
TableOfContents()

# ╔═╡ 3d168eee-aa69-4bd0-b9d7-8a9808e59170
ChooseDisplayMode()

# ╔═╡ 457b21c1-6ea9-4983-b724-fb5cbb69739d
md"""

# CS5014 Machine Learning 


#### Linear regression 2
##### Normal equation and projection

\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 54b61290-e4b2-4adb-9c2d-16b365f162e6
md"""

## Reading & references

##### Suggested reading 


* [_Pattern recognition and Machine Learning_ by _Chris Bishop._: Chapter 3.1-3.2](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)


"""

# ╔═╡ 5e774ede-251e-4321-b241-9bba130526ce
md"""

# Normal equation
\



```math
\LARGE
\boxed{
\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}}
```

* #### derive the normal equation from a *geometric* perspective


"""

# ╔═╡ b82902b8-75a2-484c-9e95-0aa85e7ffe43
md"""


## Recap: matrix vector product ``\mathbf{Xw}``


#### Note that ``\mathbf{X}`` is  a collection of ``\large m`` *column vectors*

```math
\Large
\mathbf{X}  =\begin{bmatrix}
	\vert & \vert &  & \vert\\
	\mathbf{x}_1 & \mathbf{x}_2 & \ldots & \mathbf{x}_{m} \\
	\vert & \vert & & \vert 
	\end{bmatrix}\;  \text{and}\; \mathbf{w} =\begin{bmatrix}
	w_1\\
	w_2 \\
	\vdots\\
	
	w_m
	\end{bmatrix}

```



```math
\large

```


!!! important "Matrix vector: linear combo view"	
	```math
	\begin{align}
	\Large
	\mathbf{X}\mathbf{w} &=  \begin{bmatrix}
	\vert & \vert &  & \vert\\
	\mathbf{x}_1 & \mathbf{x}_2 & \ldots & \mathbf{x}_{m} \\
	\vert & \vert & & \vert 
	\end{bmatrix}  \begin{bmatrix}
	w_1\\
	w_2 \\
	\vdots\\
	w_m
	\end{bmatrix}  =  w_1\begin{bmatrix}
           \vert\\
           \mathbf{x}_{1}\\
           \vert
         \end{bmatrix} + w_2\begin{bmatrix}
           \vert\\
           \mathbf{x}_{2}\\
           \vert
         \end{bmatrix} + \ldots w_m\begin{bmatrix}
           \vert\\
           \mathbf{x}_{m}\\
           \vert
         \end{bmatrix}\\
	& = \sum_{i=1}^m w_i \mathbf{x}_i
	\end{align}
	```
	##### In summary, 
	* ##### ``\mathbf{Xw}`` is a linear combination of the column vectors of ``\mathbf{X}``, 
	* ##### _where_ ``\mathbf{w}`` are the coefficients


"""

# ╔═╡ 918813ef-46b3-48f0-a148-a2caa5dd3fc1
md"""

## Example
"""

# ╔═╡ bbe94dfe-63fb-4a94-801f-ecaa0829382b
Foldable("The column space", md"
> The column space ?
> 
>  ``\{w_1 \textcolor{red}{\mathbf{x}_1} + w_2 \textcolor{green}{\mathbf{x}_2}\}:`` the whole shaded **plane**

")

# ╔═╡ 87f1d70f-51bb-4a72-9b2f-44142f26d040
md"
Add combination $(begin @bind add_u CheckBox(default=false) end) ;
Add column space $(begin @bind add_av CheckBox(default=false) end) 
"

# ╔═╡ ff4da64b-a12e-4e9a-93e0-77de49ab882e
md"""

 
``w_1`` = $(@bind v₁_ Slider(-2:0.1:2, default=.5, show_value=true)) 
``w_2`` = $(@bind v₂_ Slider(-2:0.1:2, default=-.9, show_value=true))

"""

# ╔═╡ 779bbc46-2c08-4554-85bf-daf3b2aa05b1
md"rotate: $(@bind ang11 Slider(-90:1:90, default=20)); up/down: $(@bind ang21 Slider(-90:1:90, default=25))"

# ╔═╡ ae0cf110-2f08-4f31-b4c5-8229d19812c8
begin
	a1 = [1.5, 3, 0]
	a2 = [3, 0, 0]
end;

# ╔═╡ 69f63143-4486-4389-a6cc-ddb6408a3414
md"""
## Solve ``\mathbf{y}=\mathbf{Xw}``


!!! warning ""
	##### We are often asked to solve

	$$\Large \mathbf{y} =\mathbf{Xw}$$ **for**  ``\mathbf{w} \in \mathbb{R}^m``; 



**For example**

```math
\Large
\underbrace{\begin{bmatrix}3 \\ 1\end{bmatrix}_{2\times 1}}_{\mathbf{y}} = \underbrace{\begin{bmatrix}2 & 0 \\ 0 & 3\end{bmatrix}_{2\times 2}}_{\mathbf{X}}\underbrace{\begin{bmatrix}\columncolor{\lightsalmon}w_1 \\ w_2 \end{bmatrix}_{2\times 1}}_{\mathbf{w}} 
```





"""

# ╔═╡ be9922ff-4493-4e17-8229-dfbfadf23842
aside(tip(md"If ``\mathbf{X}`` is **invertible**, the solution is simple

```math
\large
\mathbf{w} = \mathbf{X}^{-1}\mathbf{y}
```
"))

# ╔═╡ f7040d76-d664-4ae7-92db-29ca953e7c94
md"""

## Solve ``\mathbf{y} =\mathbf{Xw}``: linear regression


#### *Linear regression* is a special case with ``\approx``


```math
\Large
{\begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(n)}\end{bmatrix}} \approx  \begin{bmatrix}  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(1)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\  \rule[.5ex]{2.5ex}{0.5pt} & (\mathbf{x}^{(2)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\\ & \vdots & \\  \rule[.5ex]{2.5ex}{0.5pt} &(\mathbf{x}^{(n)})^\top &  \rule[.5ex]{2.5ex}{0.5pt}\end{bmatrix}_{n>m} \begin{bmatrix} \vert \\ \mathbf{w} \\\vert\end{bmatrix} =\begin{bmatrix}  (\mathbf{x}^{(1)})^\top \mathbf{w}\\  (\mathbf{x}^{(2)})^\top \mathbf{w}\\  \vdots  \\ (\mathbf{x}^{(n)})^\top \mathbf{w}\end{bmatrix}
```
* ``\mathbf{X}_{n\times m}``: ``n > m``, not invertible and **no** exact solution 
* ``\Large \approx``: we want the prediction as close as possible
* ideally, we want ``\Large =`` rather than ``\Large \approx``

"""

# ╔═╡ 4a91da1c-fa3f-4926-a068-5b70a1c8e534
Foldable("Example", md"""


```math
\Large
\underbrace{\begin{bmatrix}1 & 3 \\ 3 & 0\\ 0 &0\end{bmatrix}}_{\mathbf{X}}\underbrace{\begin{bmatrix}v_1 \\ v_2 \end{bmatrix}}_{\mathbf{w}} = \underbrace{\begin{bmatrix}3 \\ 1 \\ 1\end{bmatrix}}_{\mathbf{y}}
```


```math
\Large
w_1 \begin{bmatrix}1  \\ 3 \\ 0 \end{bmatrix}_{\mathbf{x}_1} +w_2 \begin{bmatrix}3  \\ 0 \\ 0 \end{bmatrix}_{\mathbf{x}_2} = \begin{bmatrix}3 \\ 1 \\ 1\end{bmatrix}_{\mathbf{y}}
```



* ``\mathbf{y}`` does not live in the column space of ``\mathbf{X}``
* can not be solved directly
""")

# ╔═╡ c488fa1d-8741-4136-bb35-9216314b89b9
md"""
## Geometric interpretation of ``\mathbf{y}=\mathbf{Xw}``


!!! warning "Question"
	##### What is its *geometric interpretation* 

	$$\Large \mathbf{y} = \mathbf{Xw}$$ 
	##### *for*  ``\mathbf{w} \in \mathbb{R}^m`` ?

	

"""

# ╔═╡ cb09c906-27c0-4e77-8941-b0043bf4edfe
md"""
!!! answer "Answer"
	```math
	\Large
		\begin{bmatrix} \vert\\ \mathbf{y} \\ \vert\end{bmatrix} = w_1\begin{bmatrix}
           \vert\\
           \mathbf{x}_{1}\\
           \vert
         \end{bmatrix} + w_2\begin{bmatrix}
           \vert\\
           \mathbf{x}_{2}\\
           \vert
         \end{bmatrix} + \ldots w_m\begin{bmatrix}
           \vert\\
           \mathbf{x}_{m}\\
           \vert
         \end{bmatrix}  
	```
	> ##### Does ``\mathbf{y}`` lives in the column space of ``\mathbf{X}``?
	* ###### if so, there is (are) solutions(s)
	* ###### _otherwise_, there is no **exact** solution



"""

# ╔═╡ 64b14177-03ec-4b28-933c-d2a224ae1daa
md"""

## Demonstration

#### _Solving_ ``\mathbf{y} = \mathbf{Xw}``

"""

# ╔═╡ 467b7150-63c1-4d66-8ff7-174b5b108b9c
md"
Add ``\mathbf{y}`` $(begin @bind add_b CheckBox(default=false) end) ,
Add solution $(begin @bind add_sol CheckBox(default=false) end) 

rotate vertically: $(@bind ang2 Slider(-90:1:90, default=25)),
rotate horizontally: $(@bind ang1_ Slider(0:1:90, default=50))
"

# ╔═╡ e1212d95-be2f-401d-859e-9ef36896b3a3
bv = [3.,1.5,0];

# ╔═╡ 9c10528e-3738-4a19-a9c3-e8ec587c418d
let

 	A = [a1 a2]

	A * (A \ bv) ≈ bv

end;

# ╔═╡ 8ecbe257-f3c6-447c-b686-b8d3cf22c79c
md"""

## Demonstration


"""

# ╔═╡ 3416ed8b-3d4f-421c-8330-c4fbfbfeb02c
Foldable("The approx. solution is actually:", md"
* *i.e.* its **projection** 
")

# ╔═╡ 0fc1e8bf-8635-46cd-af49-827fd5e9769c
md"
Add projection $(begin @bind add_proj CheckBox(default=false) end) 
"

# ╔═╡ 8f6489bc-ac10-491c-acb2-aab36bd99d57
md"""

 
``w_1`` = $(@bind v₁ Slider(0:0.5:5, default=2)) 
``w_2`` = $(@bind v₂ Slider(0:0.5:5, default=2))
rotate: $(@bind ang1 Slider(-90:1:90, default=20))
"""

# ╔═╡ f0fbc7ac-81ba-4837-8a5e-9369a750a8f8
bp = [v₁,v₂,0];

# ╔═╡ 8407339b-d366-4062-8c72-600b0b398cdf
md"""

## Recap: simple projection


"""

# ╔═╡ e136f6c3-9e1c-4a7c-ab55-77578602d97a
md"""

## General projection


> #### The error vector should be ``\perp`` to the column space vectors
"""

# ╔═╡ a90f7143-5fa4-4799-87a8-0db93b166bd1
md"""

## General projection

##### By the very definition of _projection_:


> ##### The error vector should be ``\perp`` to the column space vectors

##### In maths,

```math
\large
(\mathbf{y} - \mathbf{Xw}) \perp \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_m\}
```

##### Or equivalently, by inner product

```math
\large
\begin{align}
(\mathbf{y} - \mathbf{Xw})^\top \mathbf{x}_1 &= 0\\
(\mathbf{y} - \mathbf{Xw})^\top \mathbf{x}_2 &= 0\\
\vdots\\
(\mathbf{y} - \mathbf{Xw})^\top \mathbf{x}_m &= 0\\
\end{align}
```


"""

# ╔═╡ 257ebc69-2f01-4154-a7d4-a67ee42c63c1
md"""

## General projection

##### By the very definition of projection, 


> ##### The error vector should be ``\perp`` to the column space vectors

##### In maths,

```math
\large
(\mathbf{y} - \mathbf{Xw}) \perp \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_m\}
```

##### Or equivalently, by inner product

```math
\large
\begin{align}
(\mathbf{y} - \mathbf{Xw})^\top \mathbf{x}_1 &= 0\\
(\mathbf{y} - \mathbf{Xw})^\top \mathbf{x}_2 &= 0\\
\vdots\\
(\mathbf{y} - \mathbf{Xw})^\top \mathbf{x}_m &= 0\\
\end{align}
```

##### *In matrix* notations, 


> ```math
> \large
> \begin{align}
> (\mathbf{y} - \mathbf{Xw})^\top_{1\times n} &
>	 \begin{bmatrix}
>	\vert & \vert &  & \vert\\
>	\mathbf{x}_1 & \mathbf{x}_2 & \ldots & \mathbf{x}_{m} \\
>	\vert & \vert & & \vert 
>	\end{bmatrix}_{n\times m} = \mathbf{0}^\top_{1\times m}
> \end{align}
> ```

##### _Or_

> ```math
> \large
> \begin{align}
> (\mathbf{y} - \mathbf{Xw})^\top \mathbf{X} = \mathbf{0}^\top
> \end{align}
> ```

"""

# ╔═╡ 60e79a25-ec65-452b-811d-9fbf4cfdbde7
md"""

## General projection ``\Leftrightarrow`` least square estimation


> ```math
> \large
>  (\mathbf{y} - \mathbf{Xw})^\top \mathbf{X} = \mathbf{0}^\top 
> ```



* ##### solve the linear equations for ``\mathbf{w}`` (also the normal equation)

```math
\Large
\boxed{
\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}}
```


"""

# ╔═╡ e1dbd92e-f10c-4bd1-85ff-79276074fd7f
Foldable("Further details", md"Based on matrix operations,

```math
\begin{align}
\Rightarrow & \mathbf{X}^\top (\mathbf{y} - \mathbf{Xw}) = \mathbf{0} \tag{apply ⊤ on both sides}\\
\Rightarrow & \mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y} \tag{rearrange}\\
\Rightarrow & \mathbf{w} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}\tag{solve for w}
\end{align}
```
")

# ╔═╡ 76ef200a-e658-4731-add3-93d9b0922384
md"""

## General projection matrix


##### The projection (of ``\mathbf{y}`` to ``\mathbf{X}``) is 


```math
\Large

\mathbf{y}_{\rm proj} = \mathbf{X}\colorbox{orange}{$\hat{\mathbf{w}}$} = \mathbf{X}\colorbox{orange}{$(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$}
```

"""

# ╔═╡ 062668b9-3d8e-44ea-949a-155b2c1a6fc6
md"""

## General projection matrix


##### The projection (of ``\mathbf{y}`` to ``\mathbf{X}``) is 


```math
\Large

\mathbf{y}_{\rm proj} = \mathbf{X}\colorbox{orange}{$\hat{\mathbf{w}}$} = \mathbf{X}\colorbox{orange}{$(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$}
```

"""

# ╔═╡ 2e51de20-c4f4-40be-9eca-28d2aa1ad04f
md"""

#### Note that the *single projection matrix* 
* ###### is a specific case (where ``\mathbf{X}`` is single column vector ``\mathbf{x}``)

```math
\large
\mathbf{y}_{\text{proj}} = \frac{\mathbf{xx}^\top}{\mathbf{x}^\top\mathbf{x}} \mathbf{y} = \mathbf{x}(\mathbf{x}^\top\mathbf{x})^{-1} \mathbf{x}^\top\mathbf{y} 
```

"""

# ╔═╡ 11632119-7792-41b3-92f2-161aa6b35447
# md"""

# ## Pseudo inverse *

# Note that we were trying to solve

# ```math
# \Large
# \mathbf{Xw} = \mathbf{y}
# ```


# * If ``\mathbf{X}`` **were inverible**, the solution is simple, i.e.

# ```math
# \large
# \mathbf{w} = \mathbf{X}^{-1}\mathbf{y}
# ```

# However, when ``\mathbf{X}`` is not invertible, here the general solution 


# > Apply **peusdo inverse** ``(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top`` instead of ``\mathbf{X}^{-1}``: 
# > ```math
# > \large
# > \hat{\mathbf{w}} = \underbrace{(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top}_{\text{pseudo inverse}}\mathbf{y}
# > ```


# """

# ╔═╡ 84ccbf53-31d4-43c7-9da4-ac1e65fd2bf8
# Foldable("Why it is called pseudo inverse?", md"""

# ```math
# \Large 
# \underbrace{(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top}_{\text{pseudo inverse}}\mathbf{X} = \mathbf{I}
# ```
# """)

# ╔═╡ 00820112-c653-44a9-bad7-042d8ae24bde
md"""

## Summary: general projection 



> #### To solve
> ```math
> \Large
> \textcolor{blue}{\mathbf{y}} = \mathbf{X}\textcolor{purple}{\mathbf{w}} 
> ```
> #### for ``\textcolor{purple}{\mathbf{w}}``



"""

# ╔═╡ 77b40da6-4577-4968-a48d-f4ba7c6d1bca
md"""

## Appendix
"""

# ╔═╡ 2cdfab17-49c0-4ac4-a199-3ba2e2d5d216
function perp_square(origin, vx, vy; δ=0.1) 
	x = δ * vx/sqrt(norm(vx))
	y = δ * vy/sqrt(norm(vy))
	xyunit = origin+ x + y
	xunit = origin + x
	yunit = origin +y
	Shape([origin[1], xunit[1], xyunit[1], yunit[1]], [origin[2], xunit[2], xyunit[2], yunit[2]])
end

# ╔═╡ ddaaf4c3-2558-4b6f-aa7a-76d3cc4daea8
plt_proj_to_a = let
	gr()
 	plt = plot(xlim=[-1,3], ylim=[-1, 3], ratio=1, framestyle=:origin, size=(300, 300))
	# quiver([0,0,0],[0,0,0],quiver=([1,1,1],[1,2,3]))
	oo = [0,0]
	a = [3,0]
	b= [2,2]
	bp = dot(a,b)/dot(a,a)*a
	quiver!([0], [0], quiver=([a[1]], [a[2]]), lc=2, lw=2)
	quiver!([0], [0],  quiver=([b[1]], [b[2]]), lc=1, lw=2)
	annotate!(0+0.3, 0+0.3, text(L"\theta", :top))
	annotate!(a[1],a[2], text(L"\mathbf{x}", :bottom))
	annotate!(b[1],b[2], text(L"\mathbf{y}", :bottom))
	plot!([b[1], bp[1]], [b[2],bp[2]], ls=:dash, lc=:gray, lw=2, label="")
	annotate!(bp[1],bp[2]-0.1, text(L"\mathbf{y}_{\texttt{proj}}", :top))
	quiver!([0], [0],  quiver=([bp[1]], [bp[2]]), lc=4, lw=2)
	plot!(perp_square([bp[1],bp[2]], a, b -bp; δ=0.1), lw=1, fillcolor=false, label="")
	plt
end;

# ╔═╡ 5b57849e-0491-4e94-9c77-d2481e643f2f
TwoColumn(md"""
\

##### "_Simple single projection_"


> #### Project ``\mathbf{y}`` to a *single vector* ``\mathbf{x}``

\

##### Now, the problem is more _general_

> ##### Project to a space formed by **multiple** vectors
> ```math 
> \large \{\mathbf{x}_1, \mathbf{x}_2, \ldots \}
> ```

""", plt_proj_to_a)

# ╔═╡ be040c96-da49-44e6-9a73-7e26a1960261
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

# ╔═╡ 7043fade-e50d-409c-b042-fdce489c5b2f
plt_av = let
	gr()
	a1 = a1
	a2 = a2
	A= hcat([a1, a2]...)
 	plt = plot(xlim=[-4,5], ylim=[-4, 5], zlim =[-1,1.5], framestyle=:zerolines, camera=(ang11,ang21), size=(400,400))
	arrow3d!([0], [0], [0], [a1[1]], [a1[2]], [a1[3]]; as=0.1, lc=2, la=1, lw=2, scale=:identity)
	arrow3d!([0], [0], [0], [a2[1]], [a2[2]], [a2[3]]; as=0.1, lc=3, la=1, lw=2, scale=:identity)
	scatter!([0], [0], [0], mc=:black, ms =1, label="")
	normv = cross(a1, a2)
	bv = v₁_ * a1 + v₂_ * a2 
	if add_u
		arrow3d!([0], [0], [0], [bv[1]], [bv[2]], [bv[3]]; as=0.1, lc=1, la=1, lw=3, scale=:identity)
		if add_av
			surface!(-4:.2:5, -4:.2:5, (x,y) -> - x * normv[1]/normv[3]- y * normv[2]/normv[3], colorbar=false, alpha=0.25)
		end
	end
	plt
end;

# ╔═╡ 82156c81-a9cd-498f-8f68-cddbf532c187
TwoColumn(md"""##### Let's consider a case ``m=2``, 

```math 
\mathbf{X} = \begin{bmatrix} \columncolor{lightsalmon}\textcolor{red}{\vert} &\columncolor{lightgreen} \textcolor{green}{ \vert} \\
\textcolor{red}{\mathbf{x}_1} & \textcolor{green}{\mathbf{x}_2}\\
\textcolor{red}{\vert} & \vert
 \end{bmatrix},\;\; \mathbf{w} = \begin{bmatrix} w_1 \\ w_2\end{bmatrix}
```

##### _for example_, ``\mathbf{Xw}`` becomes

```math
\large
	w_1\textcolor{red}{\begin{bmatrix}
	   1.5\\
	   3\\
	   0
	 \end{bmatrix}} + w_2 \textcolor{green}{\begin{bmatrix}
	  2\\
	  0\\
	  0
	 \end{bmatrix}}
```

* ``\textcolor{red}{\mathbf{x}_1}, \textcolor{green}{\mathbf{x}_2}``: the ``\textcolor{red}{\rm red}`` and ``\textcolor{green}{\rm green}`` vectors

 
>  ##### What ``\{w_1 \textcolor{red}{\mathbf{x}_1} + w_2 \textcolor{green}{\mathbf{x}_2}\}:`` is ?
>
> * this is called **column space**
""", plt_av)

# ╔═╡ 5f77775d-23cc-48e8-ad72-0d2d80f3fb40
plt_bv = let
	gr()
	a1 = [1.5,3,0]
	a2 = [3,0,0]
	A= hcat([a1, a2]...)
 	plt = plot(xlim=[-1,5], ylim=[-1, 4], zlim =[-2,4], framestyle=:zerolines, camera=(ang1_,ang2), size=(400,400))
	arrow3d!([0], [0], [0], [a1[1]], [a1[2]], [a1[3]]; as=0.1, lc=2, la=1, lw=2, scale=:identity)
	arrow3d!([0], [0], [0], [a2[1]], [a2[2]], [a2[3]]; as=0.1, lc=3, la=1, lw=2, scale=:identity)
	scatter!([0], [0], [0], mc=:black, ms =1, label="")
	normv = cross(a1, a2)
	surface!(-1:0.1:5, -1:0.1:5, (x,y) -> - x * normv[1]/normv[3]- y * normv[2]/normv[3], colorbar=false, alpha=0.25)
	if add_b

		arrow3d!([0], [0], [0], [bv[1]], [bv[2]], [bv[3]]; as=0.1, lc=1, la=1, lw=3, scale=:identity)
		if add_sol
			A = [a1 a2]
			v = A \ bv 
			va1 = v[1] * a1
			arrow3d!([0], [0], [0], [va1[1]], [va1[2]], [va1[3]]; as=0.1, lc=2, la=1, lw=3, scale=:identity)
			plot!([bv[1], va1[1]], [bv[2], va1[2]], [bv[3], va1[3]], lw=2, lc=:gray, ls=:dash, label="")
			va2 = v[2] * a2
			arrow3d!([0], [0], [0], [va2[1]], [va2[2]], [va2[3]]; as=0.1, lc=3, la=1, lw=3, scale=:identity)
			plot!([bv[1], va2[1]], [bv[2], va2[2]], [bv[3], va2[3]], lw=2, lc=:gray, ls=:dash, label="")
		end
	end
	plt
end;

# ╔═╡ a497aaf1-cce1-4a05-86d4-4eaa9d600d81
TwoColumn(md"""
\
\
\
\


##### When ``\color{blue}\mathbf{y}``: the ``\textcolor{blue}{\rm blue}`` vector 

$\large\color{blue}\mathbf{y}\in \{\mathbf{Xw}; \mathbf{w}\in \mathbb{R}^m\}$
* ###### *there is (are) solution(s)*

""", plt_bv)

# ╔═╡ 87d57c33-c85b-4564-9ea7-aced87cbafa1
plt_av_nosol = let
	gr()
	a1 = [2,4,0]
	a2 = [3,0,0]
	b = [1,1,4]
	# c = [1,1,4]
	A= hcat([a1, a2]...)
	# bp= A*inv(A'*A)*A'*b
 	plt=plot(xlim=[-1,5], ylim=[-1, 5], zlim =[-2,5], framestyle=:zerolines, camera=(ang1,15), size=(400,400))
	arrow3d!([0], [0], [0], [b[1]], [b[2]], [b[3]]; as=0.1, lc=1, la=1, lw=2, scale=:identity)
	# annotate!(c[1], c[2], c[3], text("c"))
	arrow3d!([0], [0], [0], [a1[1]], [a1[2]], [a1[3]]; as=0.1, lc=2, la=1, lw=2, scale=:identity)
	arrow3d!([0], [0], [0], [a2[1]], [a2[2]], [a2[3]]; as=0.1, lc=3, la=1, lw=2, scale=:identity)
	scatter!([0], [0], [0], mc=:black, label="")

	if add_proj
		arrow3d!([0], [0], [0], [bp[1]], [bp[2]], [bp[3]]; as=0.1, lc=4, la=0.9, lw=2, scale=:identity)
		plot!([b[1], bp[1]], [b[2], bp[2]], [b[3], bp[3]], lw=2, lc=:gray, ls=:dash, label="")
	end

	surface!(-1:0.1:5, -1:0.1:5, (x,y) -> 0, colorbar=false, alpha=0.35)
	plt
end;

# ╔═╡ c60204cb-4172-4c2a-a9a2-886092b4b790
TwoColumn(md"""
\
\
\

##### When ``\color{blue}\mathbf{y}``: the ``\textcolor{blue}{\rm blue}`` vector _sticks out_, or

```math
\Large\textcolor{blue}{\mathbf{y}} \notin \{\mathbf{Xw}, \mathbf{w}\in \mathbb{R}^m\}
```



* ##### *NO solution*
* ##### _nevertheless_, we can find an *approximated* solution
""", plt_av_nosol)

# ╔═╡ 66728d7f-3eb6-45d7-8677-ed28894244cd
TwoColumn(md"""
\
\


#### In maths,
\
\

```math
\large
-\underbrace{(\textcolor{blue}{\mathbf{y}} - \textcolor{purple}{\mathbf{Xw}})}_{\color{brown}\text{error vec}} \perp \{\textcolor{red}{\mathbf{x}_1}, \textcolor{green}{\mathbf{x}_2}, \ldots\}
```



""", let
	gr()
	a1 = [2,4,0]
	a2 = [3,0,0]
	b = [1,1,4]
	# c = [1,1,4]
	A= hcat([a1, a2]...)
	bp= A*inv(A'*A)*A'*b
 	plt = plot(xlim=[-1,5], ylim=[-1, 5], zlim =[-2,5], framestyle=:zerolines, camera=(20,15), size=(400,400))
	arrow3d!([0], [0], [0], [b[1]], [b[2]], [b[3]]; as=0.1, lc=1, la=1, lw=2, scale=:identity)
	# annotate!(c[1], c[2], c[3], text("c"))
	arrow3d!([0], [0], [0], [a1[1]], [a1[2]], [a1[3]]; as=0.1, lc=3, la=1, lw=2, scale=:identity)
	arrow3d!([0], [0], [0], [a2[1]], [a2[2]], [a2[3]]; as=0.1, lc=2, la=1, lw=2, scale=:identity)
	scatter!([0], [0], [0], mc=:black, label="")
	arrow3d!([0], [0], [0], [bp[1]], [bp[2]], [bp[3]]; as=0.1, lc=4, la=0.9, lw=2, scale=:identity)
	plot!([b[1], bp[1]], [b[2], bp[2]], [b[3], bp[3]], lw=2, lc=:gray, ls=:solid, arrow=:arrow,  label="")
	error = bp -b 
	arrow3d!(b[1], b[2], b[3], [error[1]], [error[2]], [error[3]]; as=0.1, lc=5, la=0.9, lw=3, scale=:identity)
	# arrow3d!([b[1]], [b[2]], [b[3]], [bp[1]], [bp[2]], [bp[3]]; as=0.1, lc=4, la=0.9, lw=2, scale=:identity)
	surface!(-1:0.1:5, -1:0.1:5, (x,y) -> 0, colorbar=false, alpha=0.35)
	plt
end)

# ╔═╡ aca9b735-fd85-47b9-8a6b-5bb5785025dc
TwoColumn(md"""
\
\
\



#### The solution is 

```math
\large
\hat{\mathbf{w}}  = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}
```
#### The projection is
* ###### (or the predictions)


```math
\large
{\mathbf{y}}_{\text{proj}}  = \mathbf{X}\underbrace{(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}}_{\hat{\mathbf{w}}}
```

""", let
	gr()
	a1 = [2,4,0]
	a2 = [3,0,0]
	b = [2.5,2.5, 4.]
	A= hcat([a1, a2]...)
	bp= A * (A\b)
 	plt=plot(xlim=[-1,5], ylim=[-1, 4], zlim =[-2,4], framestyle=:zerolines, camera=(20,15), size=(400,400))
	arrow3d!([0], [0], [0], [b[1]], [b[2]], [b[3]]; as=0.1, lc=1, la=1, lw=2, scale=:identity)
	# annotate!(c[1], c[2], c[3], text("c"))
	arrow3d!([0], [0], [0], [a1[1]], [a1[2]], [a1[3]]; as=0.1, lc=3, la=1, lw=2, scale=:identity)
	arrow3d!([0], [0], [0], [a2[1]], [a2[2]], [a2[3]]; as=0.1, lc=2, la=1, lw=2, scale=:identity)
	scatter!([0], [0], [0], mc=:black, label="")
	arrow3d!([0], [0], [0], [bp[1]], [bp[2]], [bp[3]]; as=0.1, lc=4, la=0.9, lw=2, scale=:identity)
	plot!([b[1], bp[1]], [b[2], bp[2]], [b[3], bp[3]], lw=2, lc=:gray, ls=:dash, label="")
	surface!(-1:0.1:5, -1:0.1:5, (x,y) -> 0, colorbar=false, alpha=0.35)

			A = [a1 a2]
			v = A \ b 
			va1 = v[1] * a1
			arrow3d!([0], [0], [0], [va1[1]], [va1[2]], [va1[3]]; as=0.1, lc=3, la=1, lw=3, scale=:identity)
			plot!([bp[1], va1[1]], [bp[2], va1[2]], [bp[3], va1[3]], lw=2, lc=:gray, ls=:dash, label="")
			va2 = v[2] * a2
			arrow3d!([0], [0], [0], [va2[1]], [va2[2]], [va2[3]]; as=0.1, lc=2, la=1, lw=3, scale=:identity)
			plot!([bp[1], va2[1]], [bp[2], va2[2]], [bp[3], va2[3]], lw=2, lc=:gray, ls=:dash, label="")
	plt
end)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
HypertextLiteral = "~0.9.5"
LaTeXStrings = "~1.4.0"
Latexify = "~0.16.1"
Plots = "~1.40.9"
PlutoTeachingTools = "~0.2.14"
PlutoUI = "~0.7.54"
Statistics = "~1.11.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "16ea99b267ef333caf5aec2db0c9bc5771f04c2b"

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

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

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

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

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
# ╟─9edaaf3c-2973-11ed-193f-d500cbe5d239
# ╟─f932c963-31d2-4e0f-ad40-2ca2447f4059
# ╟─3d168eee-aa69-4bd0-b9d7-8a9808e59170
# ╟─457b21c1-6ea9-4983-b724-fb5cbb69739d
# ╟─54b61290-e4b2-4adb-9c2d-16b365f162e6
# ╟─5e774ede-251e-4321-b241-9bba130526ce
# ╟─b82902b8-75a2-484c-9e95-0aa85e7ffe43
# ╟─918813ef-46b3-48f0-a148-a2caa5dd3fc1
# ╟─82156c81-a9cd-498f-8f68-cddbf532c187
# ╟─bbe94dfe-63fb-4a94-801f-ecaa0829382b
# ╟─87f1d70f-51bb-4a72-9b2f-44142f26d040
# ╟─ff4da64b-a12e-4e9a-93e0-77de49ab882e
# ╟─779bbc46-2c08-4554-85bf-daf3b2aa05b1
# ╟─7043fade-e50d-409c-b042-fdce489c5b2f
# ╟─ae0cf110-2f08-4f31-b4c5-8229d19812c8
# ╟─69f63143-4486-4389-a6cc-ddb6408a3414
# ╟─be9922ff-4493-4e17-8229-dfbfadf23842
# ╟─f7040d76-d664-4ae7-92db-29ca953e7c94
# ╟─4a91da1c-fa3f-4926-a068-5b70a1c8e534
# ╟─c488fa1d-8741-4136-bb35-9216314b89b9
# ╟─cb09c906-27c0-4e77-8941-b0043bf4edfe
# ╟─5f77775d-23cc-48e8-ad72-0d2d80f3fb40
# ╟─9c10528e-3738-4a19-a9c3-e8ec587c418d
# ╟─64b14177-03ec-4b28-933c-d2a224ae1daa
# ╟─a497aaf1-cce1-4a05-86d4-4eaa9d600d81
# ╟─467b7150-63c1-4d66-8ff7-174b5b108b9c
# ╟─e1212d95-be2f-401d-859e-9ef36896b3a3
# ╟─8ecbe257-f3c6-447c-b686-b8d3cf22c79c
# ╟─c60204cb-4172-4c2a-a9a2-886092b4b790
# ╟─3416ed8b-3d4f-421c-8330-c4fbfbfeb02c
# ╟─0fc1e8bf-8635-46cd-af49-827fd5e9769c
# ╟─8f6489bc-ac10-491c-acb2-aab36bd99d57
# ╟─f0fbc7ac-81ba-4837-8a5e-9369a750a8f8
# ╟─87d57c33-c85b-4564-9ea7-aced87cbafa1
# ╟─8407339b-d366-4062-8c72-600b0b398cdf
# ╟─5b57849e-0491-4e94-9c77-d2481e643f2f
# ╟─ddaaf4c3-2558-4b6f-aa7a-76d3cc4daea8
# ╟─e136f6c3-9e1c-4a7c-ab55-77578602d97a
# ╟─66728d7f-3eb6-45d7-8677-ed28894244cd
# ╟─a90f7143-5fa4-4799-87a8-0db93b166bd1
# ╟─257ebc69-2f01-4154-a7d4-a67ee42c63c1
# ╟─60e79a25-ec65-452b-811d-9fbf4cfdbde7
# ╟─e1dbd92e-f10c-4bd1-85ff-79276074fd7f
# ╟─76ef200a-e658-4731-add3-93d9b0922384
# ╟─062668b9-3d8e-44ea-949a-155b2c1a6fc6
# ╟─2e51de20-c4f4-40be-9eca-28d2aa1ad04f
# ╟─11632119-7792-41b3-92f2-161aa6b35447
# ╟─84ccbf53-31d4-43c7-9da4-ac1e65fd2bf8
# ╟─00820112-c653-44a9-bad7-042d8ae24bde
# ╟─aca9b735-fd85-47b9-8a6b-5bb5785025dc
# ╟─77b40da6-4577-4968-a48d-f4ba7c6d1bca
# ╟─2cdfab17-49c0-4ac4-a199-3ba2e2d5d216
# ╟─be040c96-da49-44e6-9a73-7e26a1960261
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
