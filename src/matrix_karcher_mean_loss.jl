using LinearAlgebra

"""
    matrix_karcher_mean_loss(A_list::Vector{Matrix{Float64}}, X::Matrix{Float64})

Compute the matrix Karcher mean loss, ``f(X; \\{A_i\\}_{i=1}^N) = \\sum_{i=1}^N \\lVert \\log(X^{1/2} A_i X^{-1/2})\\rVert_F^2``

The matrix Karcher mean is the X which minimizes this for a given ``\\mathbb{A}``.

`A_list`: list of matrices ``\\mathbb{A} = (A_1, ..., A_n)`` such that ``A_i \\in \\mathbb{P}_m``

`X`: candidate matrix, ``X \\in \\mathbb{P}_m``
"""
function matrix_karcher_mean_loss(A_list::Vector{Matrix{Float64}}, X)
    sqrt_X = sqrt(X)
    inv_sqrt_X = inv(sqrt_X)
    loss_sum = sum(map((Ai) -> norm(log(inv_sqrt_X * Ai * inv_sqrt_X), 2)^2, A_list))
    return loss_sum
end