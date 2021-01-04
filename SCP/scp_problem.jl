# ------------------------------
# -   SCP solver environment   -
# ------------------------------



export SCPProblem



# SCP solver as a Julia class

mutable struct SCPProblem
    # Number of time-discretization steps and step-size, respectively
    N
    dt

    # Penalization weight ω and trsut-region constraints radius Δ, respectively
    omega
    Delta

    # Model class
    solver_model

    # Current mean trajectory μ, variance, and control U
    μ # (x_dim,         N )
    Σ # (x_dim, x_dim,  N )
    U # (u_dim,        N-1)

    # Trajectory μ and control U at the previous iteration
    μ_p
    Σ_p
    U_p

    # The intial constraints are defined in the model class (used for set up)
    initial_constraint
end



# Standard constructor

function SCPProblem(model, N, μ_p, Σ_p, U_p, solver=Ipopt.Optimizer)
    N     = N
    dt    = model.tf / (N-1)
    omega = model.omega0
    Delta = model.Delta0

    solver_model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))
    μ = @variable(solver_model, μ[1:model.x_dim,1:N  ])
    Σ = @variable(solver_model, Σ[1:model.x_dim,1:model.x_dim,1:N])
    U = @variable(solver_model, U[1:model.u_dim,1:N-1])

    SCPProblem(N, dt,
               omega, Delta, 
               solver_model,
               μ,   Σ,   U,
               μ_p, Σ_p, U_p,
               [])
end



# Methods that define the convex subproblem at each new SCP iteration

function reset_problem(scp_problem::SCPProblem, model, solver=Ipopt.Optimizer)
    scp_problem.solver_model = Model(with_optimizer(solver, print_level=0))
    N = scp_problem.N
    μ = @variable(scp_problem.solver_model, μ[1:model.x_dim,1:N  ])
    Σ = @variable(scp_problem.solver_model, Σ[1:model.x_dim,1:model.x_dim,1:N])
    U = @variable(scp_problem.solver_model, U[1:model.u_dim,1:N-1])
    scp_problem.μ = μ
    scp_problem.Σ = Σ
    scp_problem.U = U
    define_obs_potential_jump_NL_functions(model, scp_problem.solver_model)
end



function set_parameters(scp_problem::SCPProblem, model,
                        μ_p, U_p, omega, Delta)
    scp_problem.μ_p = μ_p
    scp_problem.U_p = U_p
    scp_problem.omega = omega
    scp_problem.Delta = Delta
end



# The following methods define the convex subproblem at each iteration in the "JuMP" framework



# Methods that define the cost

function define_nonconvex_cost(scp_problem::SCPProblem, model)
    solver_model = scp_problem.solver_model

    μ, μ_p = scp_problem.μ, scp_problem.μ_p
    U, U_p = scp_problem.U, scp_problem.U_p
    Σ, Σ_p = scp_problem.Σ, scp_problem.Σ_p

    # Control Cost
    trueNLcost = true_NL_cost(model, solver_model, μ, Σ, U, μ_p, Σ_p, U_p)

    # Obstacles
    obs1_penalty = obstacle_potential_penalties(model, solver_model,
                                                μ, Σ, U, μ_p, Σ_p, U_p, 
                                                1, "sphere")
    obs2_penalty = obstacle_potential_penalties(model, solver_model,
                                                μ, Σ, U, μ_p, Σ_p, U_p, 
                                                2, "sphere")
    obs3_penalty = obstacle_potential_penalties(model, solver_model,
                                                μ, Σ, U, μ_p, Σ_p, U_p, 
                                                3, "sphere")
    obs4_penalty = obstacle_potential_penalties(model, solver_model,
                                                μ, Σ, U, μ_p, Σ_p, U_p, 
                                                4, "sphere")

    @NLobjective(scp_problem.solver_model, Min, trueNLcost + 
                                                obs1_penalty + obs2_penalty + obs3_penalty + obs4_penalty)
end


# This method adds to the cost the linearized state constraints as penalizations
# To simplify the formalism, penalizations are derived by introducing slack control variables λ, so that every scalar linear constraint g(t,x) <= 0 is rather penalized as
#
# ... + ∫ ω*( g(t,x) - λ )^2 dt
#
# with the additional linear control constraint " λ <= 0 "

function add_penalties(scp_problem::SCPProblem, model)
    solver_model = scp_problem.solver_model
    x_dim, u_dim = model.x_dim, model.u_dim
    omega, Delta = scp_problem.omega, scp_problem.Delta

    μ, μ_p = scp_problem.μ, scp_problem.μ_p
    U, U_p = scp_problem.U, scp_problem.U_p
    Σ, Σ_p = scp_problem.Σ, scp_problem.Σ_p
    N, dt  = length(μ[1,:]), scp_problem.dt

    penalization = 0.
    
    # Contribution of obstacle-avoidance constraints

    # Non-polygonal obstacles
    Nb_obstacles = length(model.obstacles)
    if Nb_obstacles > 0
        @variable(solver_model, lambdas_obstacles[i=1:Nb_obstacles, k=1:N])
        for k = 1:N
            for i = 1:Nb_obstacles
                lambda     = lambdas_obstacles[i,k]
                constraint = obstacle_constraint_convexified(model, μ, Σ, U, μ_p, Σ_p, U_p, 
                                                                    k, i, "sphere")

                @constraint(solver_model, lambda <= 0.)
                penalization += omega*(constraint-lambda)^2
                # penalization += omega*max(0., constraint)
            end
        end
    end

    # Polygonal obstacles
    Nb_poly_obstacles = length(model.poly_obstacles)
    if Nb_poly_obstacles > 0
        @variable(solver_model, lambdas_poly_obstacles[i=1:Nb_poly_obstacles, k=1:N])
        for k = 1:N
            for i = 1:Nb_poly_obstacles
                lambda     = lambdas_poly_obstacles[i,k]
                # constraint = poly_obstacle_constraint_convexified(model, μ, U, μ_p, U_p, k, i)
                constraint = obstacle_constraint_convexified(model, μ, Σ, U, μ_p, Σ_p, U_p, 
                                                                    k, i, "poly")

                @constraint(solver_model, lambda <= 0.)
                penalization += omega*(constraint-lambda)^2
            end
        end
    end

    # Contribution of state constraints that are different from trust-region constraints and obstacle-avoidance constraints
    @variable(solver_model, lambdas_state_max_convex_constraints[i=1:x_dim, k=1:N])
    @variable(solver_model, lambdas_state_min_convex_constraints[i=1:x_dim, k=1:N])
    for k = 1:N
        for i = 1:x_dim
            lambda_max     = lambdas_state_max_convex_constraints[i,k]
            constraint_max = state_max_convex_constraints(model, μ, U, μ_p, U_p, k, i)
            lambda_min     = lambdas_state_min_convex_constraints[i,k]
            constraint_min = state_min_convex_constraints(model, μ, U, μ_p, U_p, k, i)

            @constraint(solver_model, lambda_max <= 0.)
            penalization += omega*(constraint_max-lambda_max)^2
            @constraint(solver_model, lambda_min <= 0.)
            penalization += omega*(constraint_min-lambda_min)^2
        end
    end

    return penalization
end



# Method that checks whether penalized state constraints are hardly satisfied (up to the threshold ε)

function satisfies_state_inequality_constraints(scp_problem::SCPProblem, model, μ, Σ, U, μ_p, Σ_p, U_p, Delta)
    B_satisfies_constraints = true
    x_dim, N, epsilon       = model.x_dim, scp_problem.N, model.epsilon
    T                       = (N-1) * scp_problem.dt

    # Contribution of trust-region constraints
    for k = 1:N
        constraint = trust_region_2nd_order_cone_constraints(model, μ, Σ, U, μ_p, Σ_p, U_p,
                                                                    k, 0, Delta, T)
        if constraint > -epsilon
            print("[SCP_problem.jl] - trust_region_constraint violated at i=$i and k=$k\n")
            B_satisfies_constraints = false
        end
    end

    # Contribution of obstacle-avoidance constraints
    for k = 1:N
        # Non-polygonal obstacles
        Nb_obstacles = length(model.obstacles)
        if Nb_obstacles > 0
            for i = 1:Nb_obstacles
                constraint = obstacle_constraint(model, μ, Σ, [], [], [], [], 
                                                        k, i, "sphere")
                if constraint > epsilon
                    print("[SCP_problem.jl] - obstacle_constraint violated at i=$i and k=$k, value=$constraint\n")
                    B_satisfies_constraints = false
                end
            end
        end

        # Polygonal obstacles
        Nb_poly_obstacles = length(model.poly_obstacles)
        if Nb_poly_obstacles > 0
            for i = 1:Nb_poly_obstacles
                # constraint = poly_obstacle_constraint(model, μ, U, [], [], k, i)
                constraint = obstacle_constraint(model, μ, Σ, U, μ_p, Σ_p, U_p, 
                                                        k, i, "poly")
                if constraint > epsilon
                    print("[SCP_problem.jl] - poly_obstacles_constraint violated at i=$i and k=$k\n")
                    B_satisfies_constraints = false
                end
            end
        end
    end

    # Contribution of state constraints that are different from trust-region constraints and obstacle-avoidance constraints
    for k = 1:N
        for i = 1:x_dim
            constraint_max = state_max_convex_constraints(model, μ, U, [], [], k, i)
            constraint_min = state_min_convex_constraints(model, μ, U, [], [], k, i)
            if constraint_max > epsilon || constraint_min > epsilon
                print("[SCP_problem.jl] - state_max_convex_constraints violated at i=$i and k=$k\n")
                B_satisfies_constraints = false
            end
        end
    end

    return B_satisfies_constraints
end

function satisfies_trust_region_constraints(scp_problem::SCPProblem, model, μ, Σ, U, μ_p, Σ_p, U_p, Delta)
    B_satisfies_constraints = true
    x_dim, N, epsilon       = model.x_dim, scp_problem.N, model.epsilon
    T                       = (N-1) * scp_problem.dt

    # Contribution of trust-region constraints
    for k = 1:N
        constraint = trust_region_2nd_order_cone_constraints(model, μ, Σ, U, μ_p, Σ_p, U_p,k, 0, Delta, T)
        if constraint > -epsilon
            print("[SCP_problem.jl] - trust_region_constraint violated at k=$k\n")
            B_satisfies_constraints = false
        end
    end
    return B_satisfies_constraints
end



# These methods add the remaining constraints, such as linearized dyamical, intial/final conditions constraints and control constraints



function define_constraints(scp_problem::SCPProblem, model)
    add_initial_constraints(scp_problem, model)
    add_final_constraints(scp_problem, model)
    add_dynamics_constraints(scp_problem, model)
    add_trust_region_constraints(scp_problem, model)
end



function add_initial_constraints(scp_problem::SCPProblem, model)
    solver_model = scp_problem.solver_model
    x_dim, u_dim = model.x_dim, model.u_dim
    omega, Delta = scp_problem.omega, scp_problem.Delta

    μ, μ_p = scp_problem.μ, scp_problem.μ_p
    U, U_p = scp_problem.U, scp_problem.U_p
    Σ, Σ_p = scp_problem.Σ, scp_problem.Σ_p

    constraint = state_initial_mean_constraints(model, μ, Σ, U, μ_p, Σ_p, U_p)
    scp_problem.initial_constraint = @constraint(solver_model, constraint .== 0.)
    
    constraint = state_initial_Var_constraints(model, μ, Σ, U, μ_p, Σ_p, U_p)
    @constraint(solver_model, constraint .== 0.)
end



# To improve robustness, final conditions are imposed up to a threshold ε > 0 
#   (not always necessary, but can be depending on dynamics and discretization scheme to enable reachability to xf)
function add_final_constraints(scp_problem::SCPProblem, model)
    solver_model = scp_problem.solver_model
    x_dim, u_dim = model.x_dim, model.u_dim
    omega, Delta = scp_problem.omega, scp_problem.Delta
    epsilon      = model.epsilon_xf_constraint
    μ, U, μ_p, U_p = scp_problem.μ, scp_problem.U, scp_problem.μ_p, scp_problem.U_p

    constraint = state_final_constraints(model, μ, U, μ_p, U_p)
    @constraint(solver_model,  constraint - epsilon .<= 0.)
    @constraint(solver_model, -constraint - epsilon .<= 0.)
end



function add_dynamics_constraints(scp_problem::SCPProblem, model)
    solver_model = scp_problem.solver_model
    x_dim, u_dim = model.x_dim, model.u_dim
    omega, Delta = scp_problem.omega, scp_problem.Delta

    μ, μ_p = scp_problem.μ, scp_problem.μ_p
    U, U_p = scp_problem.U, scp_problem.U_p
    Σ, Σ_p = scp_problem.Σ, scp_problem.Σ_p
    N, dt  = length(μ[1,:]), scp_problem.dt

    for k = 1:N-1
        μ_knext, Σ_knext = μ[:,k+1],           Σ[:,:,k+1]
        μ_k,  U_k,  Σ_k  = μ[:,k],   U[:,k],   Σ[:,:,k]
        μ_kp, U_kp, Σ_kp = μ_p[:,k], U_p[:,k], Σ_p[:,:,k]

        # Simple forward Euler integration method

        # Mean dynamics
        b_kp    = model.b[k]
        b_dx_kp = model.b_dx[k]
        b_du_kp = model.b_du[k]
        constraint =  μ_knext - ( μ_k + dt * (  b_kp + 
                                                b_dx_kp * (μ_k-μ_kp) + 
                                                b_du_kp * (U_k-U_kp)
                                             )
                                )
        @constraint(solver_model, constraint .== 0.)

        # Variance dynamics
        σ_kp    = model.σ[k]
        σ_dx_kp = model.σ_dx[k]

        M_k = b_dx_kp

        σdxkp_times_μkp = sum(σ_dx_kp[:,:,i] * μ_kp[i] for i in 1:x_dim)
        B_k = σ_kp - σdxkp_times_μkp

        σdxkp_times_μk = sum(σ_dx_kp[:,:,i] .* μ_k[i] for i in 1:x_dim)

        constraint =  Σ_knext - ( Σ_k + dt * 
                        (  
                            M_k * Σ_k + Σ_k*transpose(M_k) + 
                            # σ_kp * transpose(σ_kp)
                            B_k * transpose(B_k) + 
                            σdxkp_times_μk * transpose(σdxkp_times_μkp)
                        )
                              )
        @constraint(solver_model, constraint .== 0.)
    end
end


# trust region cone constraints
function add_trust_region_constraints(scp_problem::SCPProblem, model)
    solver_model = scp_problem.solver_model
    x_dim, u_dim = model.x_dim, model.u_dim
    Delta        = scp_problem.Delta

    μ, μ_p = scp_problem.μ, scp_problem.μ_p
    U, U_p = scp_problem.U, scp_problem.U_p
    Σ, Σ_p = scp_problem.Σ, scp_problem.Σ_p
    N      = scp_problem.N
    T      = (N-1) * scp_problem.dt

    for k = 1:N
        constraint = trust_region_2nd_order_cone_constraints(model, 
                                                 μ, Σ, U, μ_p, Σ_p, U_p,
                                                 k, 0, Delta, T)
        @constraint(solver_model,  constraint <= 0.)
    end
end