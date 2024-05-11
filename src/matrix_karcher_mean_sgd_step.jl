using LinearAlgebra

"""
    matrix_karcher_mean_sgd_step()

Matrix Karcher mean stochastic gradient descent method
"""
function matrix_karcher_mean_sgd_step(A_list::Vector{Matrix{Float64}}, Xs, eta_s)
    N = size(A_list)[1]
    idx = rand(1:N)
    Ai = A_list[idx]
    sqrt_Xs = sqrt(Xs)
    mid_term = -eta_s * N * log(sqrt_Xs * inv(Ai) * sqrt_Xs)
    exp_term = exp.(mid_term)
    Xsp1 = sqrt_Xs * exp_term * sqrt_Xs
    return Xsp1
end