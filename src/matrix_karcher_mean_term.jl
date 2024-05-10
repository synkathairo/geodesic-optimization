using LinearAlgebra
# using Optimization
# using Manifolds
# using Manopt

"""
    matrix_karcher_mean_loss(A_list, X)

Compute the matrix Karcher mean loss, ``f(X; \\{A_i\\}_{i=1}^N) = \\sum_{i=1}^N \\lVert \\log(X^{1/2} A_i X^{-1/2})\\rVert_F^2``

The matrix Karcher mean is the X which minimizes this for a given ``\\mathbb{A}``.

`A_list`: list of matrices ``\\mathbb{A} = (A_1, ..., A_n) \\in \\mathbb{P}_m^n``
"""
function matrix_karcher_mean_loss(A_list, X)
    sqrt_X = sqrt(X)
    inv_sqrt_X = inv(sqrt_X)
    loss_sum = sum(norm(log(sqrt_X * Ai * inv_sqrt_X), 2)^2 for Ai in A_list)
    return loss_sum
end