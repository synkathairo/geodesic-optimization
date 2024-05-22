function iter_method(step_fun::Function, loss_fun::Function, Xs, max_iter::Int, tol=0)
    f_all = []
    x_all = []

    for i in range(1, max_iter)
        # if i >= 2 && abs(f_all[end] - f_all[end-1]) < tol
        #     break
        # end
        Xs = step_fun(Xs)
        f = loss_fun(Xs)
        push!(f_all, f)
        push!(x_all, Xs)
    end

    return f_all, x_all
end