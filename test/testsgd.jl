using LinearAlgebra
using Plots
include("../src/random_positive_matrix.jl")
include("../src/iter_method.jl")
# include("../src/matrix_karcher_mean_gd_step.jl")
include("../src/matrix_karcher_mean_sgd_step.jl")
include("../src/matrix_karcher_mean_loss.jl")

max_iter = 100
N = 2
n = 100
Q = 100
A_list = Vector{Matrix{Float64}}(undef,N)
for i in range(1, N)
    A_i = random_positive_matrix(n, 2, Q)
    global A_list
    A_list[i] = A_i
end

# A_list = [
#     [1.0 0.2; 0.2 1.5],
#     [2.0 0.3; 0.3 2.0],
#     [1.5 0.1; 0.1 1.8]
# ]
Xs = Matrix{Float64}(I, n, n)

# TODO set eta_s according to theorem 14 of Zhang and Sra paper

eta_s = 1 / (5 * N)

km_gd_step = Xs -> matrix_karcher_mean_sgd_step(A_list, Xs, eta)
km_loss_step = Xs -> matrix_karcher_mean_loss(A_list, Xs)

f_all, x_all = iter_method(km_gd_step, km_loss_step, Xs, max_iter)

plotf = plot(f_all)
savefig(plotf,"karchersgd.png")

# plotx = plot(x_all)
# savefig(plotx, "karchergd_xs.png")