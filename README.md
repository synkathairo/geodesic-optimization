# geodesic-optimization

Zhang and Sra describe methods for optimizing on first-order geodesically convex optimization[^1], which allow generalization to non-linear metric spaces. Also see Vishnoi's article[^2] which explains the topic at an introductory level.

## First-order methods

Consider an optimization problem of the form[^1]: 
$$\min f(x) \quad \text{subject to } x \in \mathcal{X} \subset \mathcal{M}$$
where $f: \mathcal{M} \to \mathbb{R} \cup \{\infty \}$, $f$ is $g$-convex, $\mathcal{X}$ is a geodesically convex set, and $\mathcal{M}$ is a Hadamard manifold.

### The matrix Karcher mean problem

Zhang and Sra experiment on the matrix Karcher mean problem[^1][^3], defined as $X^{\ast}$ such that 
$$X^{\ast} = \arg \min_X \sum_{i=1}^N (d(X,A_i))^2$$
where $d(X,Y) = \lVert \log(X^{-1/2} Y X^{-1/2}) \rVert_F$ is the Riemannian metric, and each $A_i$ is given a symmetric positive definite matrix.

Full gradient descent is implemented, for the $X_{s+1}$ iteration, using the update step[^1]:
$$X_{s+1} = X_s^{1/2} \exp \left( -\eta_s \sum_{i=1}^N \log (X_s^{1/2} A_i^{-1}X_s^{1/2}) \right) X_s^{1/2}$$

Or, in stochastic gradient descent, approximated using a random chosen $A_i$ where $i \in \{1,...,N\}$, as[^1]:
$$X_{s+1} = X_s^{1/2} \exp \left( -\eta_s N\log (X_s^{1/2} A_i^{-1}X_s^{1/2}) \right) X_s^{1/2}$$

### Implementation

The code is implemented as Julia functions in `src/`. An implementation for a full gradient descent update step is contained within `matrix_karcher_mean_gd_step.jl`, and a stochastic gradient descent step is implemented in `matrix_karcher_mean_sgd_step.jl`, for the Karcher mean problem. The loop iterations to test the functions are in `test/`, implemented as `testgd.jl` and `testsgd.jl` respectively. These may be run directly, e.g. by running `julia testgd.jl`.

## Riemannian online convex optimization problem

Wang et. al describe the Riemannian online convex optimization problem (R-OCO)[^4].


## Remarks

Based on course project for [CSCI-GA.2945/ MATH-GA.2012 Convex and Nonsmooth Optimization](https://cs.nyu.edu/courses/spring24/CSCI-GA.2945-002/) at New York University.

[^1]: H. Zhang and S. Sra, “First-order methods for geodesically convex optimization,” in 29th annual conference on learning theory, V. Feldman, A. Rakhlin, and O. Shamir, Eds., in Proceedings of machine learning research, vol. 49. Columbia University, New York, New York, USA: PMLR, Jun. 2016, pp. 1617–1638. \[Online\]. Available: https://proceedings.mlr.press/v49/zhang16b.html

[^2]: N. K. Vishnoi, “Geodesic Convex Optimization: Differentiation on Manifolds, Geodesics, and Convexity.” arXiv, Jun. 17, 2018. Accessed: Apr. 18, 2024. \[Online\]. Available: http://arxiv.org/abs/1806.06373

[^3]: T. Yamazaki, “A brief introduction of the Karcher mean,” 数理解析研究所講究録, vol. 1839, pp. 31–39, 2013, \[Online\]. Available: https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1839-05.pdf

[^4]: X. Wang, Z. Tu, Y. Hong, Y. Wu, and G. Shi, “Online optimization over riemannian manifolds,” Journal of Machine Learning Research, vol. 24, no. 84, pp. 1–67, 2023, \[Online\]. Available: http://jmlr.org/papers/v24/21-1308.html
