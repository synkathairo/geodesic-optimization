# geodesic-optimization

Zhang and Sra describe methods for utilizing first-order geodesically convex optimization[^1]. Also see Vishnoi's article[^2] which explains the topic at an introductory level.

## The matrix Karcher mean problem

Zhang and Sra experiment on the matrix Karcher mean problem[^1][^3], defined as $X^{\ast}$ such that 
$$X^{\ast} = \arg \min_X \sum_{i=1}^N (d(X,A_i))^2$$
where $d(X,Y) = \lVert \log(X^{-1/2} Y X^{-1/2}) \rVert_F$ is the Riemannian metric, and each $A_i$ is given a symmetric positive definite matrix.

Full gradient descent is implemented, for the $X_{s+1}$ iteration, using the update step[^1]:
$$X_{s+1} = X_s^{1/2} \exp \left( -\eta_s \sum_{i=1}^N \log (X_s^{1/2} A_i^{-1}X_s^{1/2}) \right) X_s^{1/2}$$

Or, in stochastic gradient descent, approximated using a random chosen $A_i$ where $i \in \{1,...,N\}$, as[^1]:
$$X_{s+1} = X_s^{1/2} \exp \left( -\eta_s N\log (X_s^{1/2} A_i^{-1}X_s^{1/2}) \right) X_s^{1/2}$$

## Remarks

Based on course project for [CSCI-GA.2945/ MATH-GA.2012 Convex and Nonsmooth Optimization](https://cs.nyu.edu/courses/spring24/CSCI-GA.2945-002/) at New York University.

[^1]: H. Zhang and S. Sra, “First-order methods for geodesically convex optimization,” in 29th annual conference on learning theory, V. Feldman, A. Rakhlin, and O. Shamir, Eds., in Proceedings of machine learning research, vol. 49. Columbia University, New York, New York, USA: PMLR, Jun. 2016, pp. 1617–1638. \[Online\]. Available: https://proceedings.mlr.press/v49/zhang16b.html

[^2]: N. K. Vishnoi, “Geodesic Convex Optimization: Differentiation on Manifolds, Geodesics, and Convexity.” arXiv, Jun. 17, 2018. Accessed: Apr. 18, 2024. \[Online\]. Available: http://arxiv.org/abs/1806.06373

[^3]: T. Yamazaki, “A brief introduction of the Karcher mean,” 数理解析研究所講究録, vol. 1839, pp. 31–39, 2013, \[Online\]. Available: https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1839-05.pdf

