using Random
using LinearAlgebra

# Adoption of code from Matrix Means toolbox, version 2.3, found at 
# https://bezout.dm.unipi.it/software/mmtoolbox/mfiles/random.m

"""
    random_positive_matrix(n, alg[, par=undef])

Generates a random matrix of size `n` by `n` with different algorithms. 

`alg`: 
- `0` random positive matrix
- `1` random positive matrix with norm `1`
- `2` random positive matrix with condition number `par`
"""
function random_positive_matrix(n::Int, alg=0, par=10)
    A = nothing
    if alg == 0
        W = randn(n, n)
        global A
        A = sqrt(W * W')
        A = (A + A') / 2
    elseif alg == 1
        W = randn(n, n) - rand(n, n)
        W = W * W'
        global A
        A = W / norm(W)
        A = (A + A') / 2
    elseif alg == 2
        W = rand(n,n) - rand(n,n)
        X = W' * W
        X = X - I * eigmin(X)
        X = X / norm(X)
        X = X + I / (par - 1)
        global A
        A = X / norm(X)
    end
    # print(A)
    return A
end