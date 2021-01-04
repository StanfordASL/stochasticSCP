function solve_SCP(model, scp_problem, N, max_it=15, verbose=false)
    (Delta0, omega0, omegamax, epsilon,
            convergence_threshold) = get_initial_scp_parameters(model)


    μ_p,   Σ_p,   U_p   = initialize_trajectory(model, N)
    μ,     Σ,     U     = copy(μ_p), copy(Σ_p), copy(U_p)
    μ_all, Σ_all, U_all = [], [], []
    push!(μ_all, copy(μ))
    push!(Σ_all, copy(Σ))
    push!(U_all, copy(U))

    Delta = Delta0

    # GuSTO loop
    success, it = false, 1
    while (it < max_it) && 
          !(success && 
             (convergence_metric(model, μ,U, μ_p,U_p) +
              convergence_metric(model, μ_all[end-2],U_all[end-2], μ_p,U_p)) 
                    < convergence_threshold)
        if verbose
            println("-----------\nIteration $it\n-----------")
        end
        # Storing the solution at the previous step and the linearized dynamics
        μ_p, Σ_p, U_p                                        = copy(μ), copy(Σ), copy(U)
        model.b, model.b_dx, model.b_du, model.σ, model.σ_dx = compute_dynamics(model, μ_p, U_p)
        
        # Defining the convex subproblem
        reset_problem(     scp_problem, model)
        set_parameters(    scp_problem, model, μ_p, U_p, omega0, Delta)
        define_nonconvex_cost(scp_problem, model)
        define_constraints(scp_problem, model)
        
        # Solving the convex subproblem
        JuMP.optimize!(scp_problem.solver_model)
        μ_sol, Σ_sol, U_sol = JuMP.value.(scp_problem.μ), JuMP.value.(scp_problem.Σ), JuMP.value.(scp_problem.U)
        
        # -----------
        # SCP
        μ, Σ, U = copy(μ_sol), copy(Σ_sol), copy(U_sol)

        if (it > 2) # needs at least 3 iterations to check convergence
            success = true
        else
            success = false
        end

        # Collecting the solution at each iteration
        push!(μ_all,copy(μ))
        push!(Σ_all,copy(Σ))
        push!(U_all,copy(U))
        it += 1
        
        if verbose
            println("(1-step) metric = $(convergence_metric(model,μ,U,μ_p,U_p))")
        end
    end
    if verbose
        println(">>> Finished <<<")
    end

    μ_f,Σ_f,U_f,μ_fp,Σ_fp,U_fp = μ_all[end],Σ_all[end],U_all[end],μ_all[end-1],Σ_all[end-1],U_all[end-1]
    B_trust_satisfied = satisfies_trust_region_constraints(scp_problem, model, μ_f,Σ_f,U_f,μ_fp,Σ_fp,U_fp, Delta)
    if B_trust_satisfied && verbose
        println(">>>>> Satisfies trust region constraint.")
    end

    return (μ_all,Σ_all,U_all, success, it, B_trust_satisfied)
end