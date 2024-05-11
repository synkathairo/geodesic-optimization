using LinearAlgebra

"""
    matrix_karcher_mean_gd_step()

Matrix Karcher mean gradient descent method
"""
function matrix_karcher_mean_gd_step(A_list::Vector{Matrix{Float64}}, Xs, eta)
    sqrt_Xs = sqrt(Xs)
    sum_term = -eta * sum(map((Ai) -> log(sqrt_Xs * inv(Ai) * sqrt_Xs), A_list))
    exp_term = exp.(sum_term)
    Xsp1 = sqrt_Xs * exp_term * sqrt_Xs
    return Xsp1
end