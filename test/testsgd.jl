using LinearAlgebra
using Plots
include("../src/random_positive_matrix.jl")
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

f_all = []
x_all = []

# TODO set eta_s according to theorem 14 of Zhang and Sra paper

eta_s = 1 / (5 * N)

for i in range(1, max_iter)
    global Xs
    Xs = matrix_karcher_mean_sgd_step(A_list, Xs, eta_s)
    f = matrix_karcher_mean_loss(A_list, Xs)
    global f_all
    push!(f_all, f)
    global x_all
    push!(x_all,Xs)
end

plotf = plot(f_all)
savefig(plotf,"karchersgd.png")