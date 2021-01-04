# ----------------------------
# -   Model for the Dubins5D -
# ----------------------------

using Distributions
using Random
Random.seed!(1234)

export Dubins5D



# Model Dubins5D as a Julia class

mutable struct Dubins5D
    # State (x,y,theta) and control (u) dimensions
    x_dim
    u_dim

    # Dynamics and linearized dynamics
    # Mean dynamics
    b # [N, [x_dim]]
    b_dx # [N, [x_dim, x_dim]]
    b_du # [N, [x_dim, u_dim]]
    # Variance dynamics
    σ    # [N, [x_dim, x_dim]]
    σ_dx # [N, [x_dim, x_dim, x_dim]]

    # Model constants
    α    # term multiplying positional variance
    β    # term multiplying positional variance

    # Problem settings
    x_init
    x_final
    Σ_init
    tf

    B_final_angle_constraint
    B_final_velocity_constraint

    x_min
    x_max
    u_min
    u_max

    prob

    true_cost_weight
    B_penalize_variance

    # Cylindrical obstacles (modeled by a center (x,y) and a radius r) and polygon obstacles (not used in this example)
    obstacles
    poly_obstacles
    eps_obstacles
    alpha # corresponds to lambda in the paper, penalizes obstacle avoidance

    # GuSTO parameters
    Delta0
    omega0
    omegamax
    # threshold for constraints satisfaction : constraints <= epsilon
    epsilon
    epsilon_xf_constraint
    convergence_threshold # in %
end



# The problem is set in the class constructor

function Dubins5D()
    x_dim = 5 # [x, y, theta, θ, v, ω]
    u_dim = 2 # [a_v, a_ω]

    α = 1e-1
    β = 1e-2

    x_init  = [ 0.; 0; 0; 0; 0]
    x_final = [ 2.2; 3; 0; 0; 0]
    Σ_init  = zeros((x_dim,x_dim))
    tf = 5.

    B_final_angle_constraint    = true
    B_final_velocity_constraint = true

    myInf = 1.0e6 # Adopted to detect initial and final condition-free state variables
    x_max = [100.; 100.; 2*pi]   # m     # state limits          
    u_max = [myInf]
    x_min = -x_max
    u_min = -u_max

    # Uncertainty / chance constraints
    prob     = 0.9         # probability threshold for chance constraints

    true_cost_weight = 1.
    B_penalize_variance = true

    Delta0 = 50.
    omega0 = 500.
    omegamax = 1.0e6
    epsilon = 1e-3
    epsilon_xf_constraint = 1e-3
    convergence_threshold = 0.001

    # Cylindrical obstacles in the form [(x,y),r]
    obstacles = []
    obs = [[1.5,1.9],0.2]
    push!(obstacles, obs)
    obs = [[0.6,2.0],0.3]
    push!(obstacles, obs)
    obs = [[1.15,0.4],0.3]
    push!(obstacles, obs)
    obs = [[0.35,0.75],0.25]
    push!(obstacles, obs)
    eps_obstacles = 0.13
    alpha = 500.

    # Polygonal obstacles are not used in this example
    poly_obstacles = []

    Dubins5D(x_dim, u_dim,
             [], [], [], [], [],
             α, β,
             x_init, x_final, Σ_init, tf,
             B_final_angle_constraint, B_final_velocity_constraint,
             x_min, x_max, u_min, u_max,
             prob,
             true_cost_weight, B_penalize_variance,
             obstacles, poly_obstacles, eps_obstacles, alpha,
             Delta0,
             omega0,
             omegamax,
             epsilon,
             epsilon_xf_constraint,
             convergence_threshold)
end



# Method that returns the GuSTO parameters (used for set up)

function get_initial_scp_parameters(m::Dubins5D)
    return m.Delta0, m.omega0, m.omegamax, m.epsilon, m.convergence_threshold
end



# GuSTO is intialized by zero controls and velocities, and a straight-line in position

function initialize_trajectory(model::Dubins5D, N::Int)
  x_dim,  u_dim   = model.x_dim, model.u_dim
  x_init, x_final = model.x_init, model.x_final
  
  X = hcat(range(x_init, stop=x_final, length=N)...)
  U = zeros(u_dim, N-1)
  Σ = zeros(x_dim, x_dim, N)

  rng = MersenneTwister(1234)
  X   = X + 1e-5*(rand(rng, x_dim, N).-0.5)

  return X, Σ, U
end



# Method that returns the convergence ratio between iterations (in percentage)
# The quantities X, U denote the actual solution over time, whereas Xp, Up denote the solution at the previous step over time

function convergence_metric(model::Dubins5D, μ, U, μ_p, Up)
    u_dim = model.u_dim
    N     = length(μ[1,:])
    dt    = model.tf / (N-1)

    L2_difference = 0.0
    for k in 1:(N-1)
      L2_difference += dt * norm(U[1:u_dim,k] - Up[1:u_dim,k], 2)
    end
    return L2_difference
end



# Method that returns the original cost

function true_cost(model::Dubins5D, μ, Σ, U, μ_p, Σ_p, U_p)
    cost = 0.
    for k = 1:length(U[1,:])
      cost += sum(U[u_i,k]^2 for u_i=1:model.u_dim)
    end

    if model.B_penalize_variance
      for k = 1:length(Σ[1,1,:])
        cost += 1*sum(Σ[i,i,k] for i=1:model.x_dim)
      end
    end

    return model.true_cost_weight * cost
end

function true_NL_cost(model::Dubins5D, solver_model, μ, Σ, U, μ_p, Σ_p, U_p)
    x_dim, u_dim, N = model.x_dim, model.u_dim, (length(U[1,:])+1)
    @NLexpression(solver_model, trueNLcost, (model.tf/(N-1)) * sum(U[i,k]^2 for k in 1:length(U[1,:]) for i in 1:u_dim))
    return trueNLcost
end

# The following methods return the i-th coordinate at the k-th iteration of the various constraints and their linearized versions (when needed)
# These are returned in the form " g(t,x(t),u(t)) <= 0 "


# State bounds and trust-region constraints (these are all convex constraints)

function state_max_convex_constraints(model::Dubins5D, μ, U, μ_p, Up, k, i)
    return -1
end
function state_min_convex_constraints(model::Dubins5D, μ, U, μ_p, Up, k, i)
    return -1
end

function trust_region_2nd_order_cone_constraints(model::Dubins5D, 
                                                 μ, Σ, U, μ_p, Σ_p, U_p,
                                                 k, i, Delta, T)
    x_dim = model.x_dim

    μ_k, μ_kp = μ[:,   k], μ_p[:,   k]
    Σ_k, Σ_kp = Σ[:,:, k], Σ_p[:,:, k]

    trace_Σ  = sum(Σ_k[i,i]      for i=1:x_dim)
    trace_Σp = sum(Σ_kp[i,i]     for i=1:x_dim)
    μ_μ      = sum(μ_k[i] *μ_k[i]  for i=1:x_dim)
    μ_μp     = sum(μ_k[i] *μ_kp[i]  for i=1:x_dim)
    μp_μp    = sum(μ_kp[i]*μ_kp[i] for i=1:x_dim)

    constraint = trace_Σ + trace_Σp + μ_μ - 2*μ_μp + μp_μp - Delta^2/T
    return constraint
end


# Initial and final conditions on state variables

function state_initial_mean_constraints(model::Dubins5D, μ, Σ, U, μ_p, Σ_p, U_p)
    return ( μ[:,1] - model.x_init )
end
function state_initial_Var_constraints(model::Dubins5D, μ, Σ, U, μ_p, Σ_p, U_p)
    return ( Σ[:,:,1] - model.Σ_init )
end

function state_final_constraints(model::Dubins5D, μ, U, μ_p, Up)
  B_ang = model.B_final_angle_constraint 
  B_vel = model.B_final_velocity_constraint 

  if B_ang && B_vel
    return ( μ[:,end] - model.x_final )
  elseif B_ang
    return ( μ[1:3,end] - model.x_final[1:3] )
  elseif B_vel
    idx_f = [1,2,4,5]
    return ( μ[idx_f,end] - model.x_final[idx_f] )
  else
    return ( μ[1:2,end] - model.x_final[1:2] )
  end
end



# Methods that return the cylindrical obstacle-avoidance constraint and its lienarized version
# Here, a merely classical distance function is considered

function obstacle_constraint(model::Dubins5D, μ, Σ, U, μ_p, Σ_p, U_p, 
                                               k, obs_i,
                                               obs_type::String="sphere")
    #  obs_type    : Type of obstacles, can be 'sphere' or 'poly'
    if obs_type=="sphere"
      dimObs  = 2

      p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]

      μ_k  = μ[1:dimObs,k]
      Q_k  = cquantile(Chisq(dimObs), 1-model.prob) * Σ[1:dimObs,1:dimObs,k]

      dist = norm(μ_k - p_obs, 2) 
      dir  = (μ_k - p_obs)/dist
      dist = dist - obs_radius

      if norm(Q_k)>1e-6
        constraint = -( dist - sqrt(sum(dir[i]*Q_k[i,j]*dir[j] for i=1:dimObs for j=1:dimObs)) )
      else
        constraint = -1.
      end
    else
      print("[Dubins5D.jl::obstacle_constraint] Unknown obstacle type.")
    end
    return constraint
end
function obstacle_constraint_convexified(model::Dubins5D, μ, Σ, U, μ_p, Σ_p, U_p, 
                                                           k, obs_i,
                                                           obs_type::String="sphere")
    #  obs_type    : Type of obstacles, can be 'sphere' or 'poly'
    if obs_type=="sphere"
      dimObs  = 2

      p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
      μ_k, μ_kp         = μ[1:dimObs, k],          μ_p[1:dimObs, k]
      Σ_k, Σ_kp         = Σ[1:dimObs,1:dimObs, k], Σ_p[1:dimObs,1:dimObs, k]
      deltaμk, deltaΣk  = μ_k-μ_kp, Σ_k-Σ_kp
      chi2_n            = cquantile(Chisq(dimObs), 1-model.prob)
      Q_kp              = chi2_n * Σ_kp

      # signed distance function
      dist_prev = norm(μ_kp - p_obs, 2)
      # Gradient
      dir_prev  = (μ_kp - p_obs)/dist_prev
      # Hessian
      dir_dxkp = Matrix{Float64}(I, dimObs, dimObs)
      for i = 1:dimObs
        dir_dxkp[i,:] += -(μ_kp[i]-p_obs[i])*(μ_kp-p_obs) / (dist_prev^2)
      end
      dir_dxkp /= dist_prev
      # Remove obstacle radius
      dist_prev = dist_prev - obs_radius

      # variance term of obs. avoid. chance constraint
      if norm(Q_kp)>1e-6
        dir_Qkp_dir      = sum(dir_prev[i]*Q_kp[i,j]*dir_prev[j] 
                                  for i=1:dimObs for j=1:dimObs)
        sqrt_dir_Qkp_dir = sqrt(dir_Qkp_dir)
      else
        sqrt_dir_Qkp_dir = 0.
      end

      # constraint evaluated at last iteration
      constraint  = -( dist_prev - sqrt_dir_Qkp_dir )
      # gradient of nominal term
      constraint += -( sum(dir_prev[i] * deltaμk[i] for i=1:dimObs) )
      # gradient of term due to variance
      if norm(Q_kp)>1e-6
        # gradient w.r.t. μ
        dir_Qkp_dir_dμkp_deltaμk = (1/(2*sqrt_dir_Qkp_dir)) * 
                            2*sum(dir_prev[i]*Q_kp[i,j]*dir_dxkp[j,l]*deltaμk[l]
                                  for i=1:dimObs for j=1:dimObs for l=1:dimObs)
        constraint += -( dir_Qkp_dir_dμkp_deltaμk )

        # gradient w.r.t. Σ
        dir_Qkp_dir_dQkp = (1/(2*sqrt_dir_Qkp_dir)) * (dir_prev*transpose(dir_prev))
        dir_Qkp_dir_dQkp_deltaΣk = sum(dir_Qkp_dir_dQkp[i,j]*chi2_n*deltaΣk[i,j]
                                          for i=1:dimObs for j=1:dimObs   )
        constraint += -( dir_Qkp_dir_dQkp_deltaΣk )
      end

    else
      print("[Dubins5D.jl::obstacle_constraint_convexified] Unknown obstacle type.")
    end
    return constraint
end





function obstacle_potentialSphere_penalty(Dubins5D::Dubins5D, 
                                          pos, pos_obs, radius, alpha)
    pad = model.eps_obstacles
    if norm(pos-pos_obs)^2 < (radius+pad)^2
      return alpha * (norm(pos-pos_obs)^2-(radius+pad)^2)^2
    else
      return 0.
    end
end
function obstacle_potentialSphere_penalty_grad(model::Dubins5D, 
                                               g,
                                               pos, pos_obs, radius, alpha)
    pad = model.eps_obstacles
    if norm(pos-pos_obs)^2 < (radius+pad)^2
      g[1] = alpha * 2. * (norm(pos-pos_obs)^2-(radius+pad)^2) * 2. * (pos-pos_obs)[1] 
      g[2] = alpha * 2. * (norm(pos-pos_obs)^2-(radius+pad)^2) * 2. * (pos-pos_obs)[2] 
    else
      g[1] = 0.
      g[2] = 0.
    end
    return g
end
function define_obs_potential_jump_NL_functions(model::Dubins5D, solver_model, 
                                                obs_type::String="sphere")
    if obs_type != "sphere"      
      throw(MethodError("[Dubins5D.jl::define_obs_potential_jump_NL_functions] Only spheres are supported."))
    end
    if length(model.obstacles) > 4
      throw(MethodError("[Dubins5D.jl::define_obs_potential_jump_NL_functions] Too many obstacles."))
    end
    alpha = model.alpha
    pos_obs1, radius1 = model.obstacles[1][1], model.obstacles[1][2]
    pos_obs2, radius2 = model.obstacles[2][1], model.obstacles[2][2]
    pos_obs3, radius3 = model.obstacles[3][1], model.obstacles[3][2]
    pos_obs4, radius4 = model.obstacles[4][1], model.obstacles[4][2]
    obsPotFunc1     = (p1,p2)    -> obstacle_potentialSphere_penalty(     model,    [p1,p2], pos_obs1, radius1, alpha)
    obsPotFunc2     = (p1,p2)    -> obstacle_potentialSphere_penalty(     model,    [p1,p2], pos_obs2, radius2, alpha)
    obsPotFunc3     = (p1,p2)    -> obstacle_potentialSphere_penalty(     model,    [p1,p2], pos_obs3, radius3, alpha)
    obsPotFunc4     = (p1,p2)    -> obstacle_potentialSphere_penalty(     model,    [p1,p2], pos_obs4, radius4, alpha)
    obsPotFuncGrad1 = (g, p1,p2) -> obstacle_potentialSphere_penalty_grad(model, g, [p1,p2], pos_obs1, radius1, alpha)
    obsPotFuncGrad2 = (g, p1,p2) -> obstacle_potentialSphere_penalty_grad(model, g, [p1,p2], pos_obs2, radius2, alpha)
    obsPotFuncGrad3 = (g, p1,p2) -> obstacle_potentialSphere_penalty_grad(model, g, [p1,p2], pos_obs3, radius3, alpha)
    obsPotFuncGrad4 = (g, p1,p2) -> obstacle_potentialSphere_penalty_grad(model, g, [p1,p2], pos_obs4, radius4, alpha)
    register(solver_model, :obsPotFunc1, 2, obsPotFunc1, obsPotFuncGrad1)
    register(solver_model, :obsPotFunc2, 2, obsPotFunc2, obsPotFuncGrad2)
    register(solver_model, :obsPotFunc3, 2, obsPotFunc3, obsPotFuncGrad3)
    register(solver_model, :obsPotFunc4, 2, obsPotFunc4, obsPotFuncGrad4)
end
function obstacle_potential_penalties(model::Dubins5D, solver_model, 
                                    μ, Σ, U, μ_p, Σ_p, U_p,
                                    obs_i,
                                    obs_type::String="sphere")
    N, Nb_obstacles = length(μ[1,:]), length(model.obstacles)
    if obs_i > Nb_obstacles# != 1
      throw(MethodError("[Dubins5D.jl::obstacle_potential_penalties] Incorrect number of obstacles."))
    end
    if !(obs_type=="sphere")
      print("[Dubins5D.jl::obstacle_potential_penalties] Unknown obstacle type.")
    end

    if obs_i == 1
      @NLexpression(solver_model, obscost1, (model.tf/(N-1))*sum(obsPotFunc1(μ[1,k],μ[2,k]) for k in 1:N))
      return obscost1
    elseif obs_i == 2
      @NLexpression(solver_model, obscost2, (model.tf/(N-1))*sum(obsPotFunc2(μ[1,k],μ[2,k]) for k in 1:N))
      return obscost2
    elseif obs_i == 3
      @NLexpression(solver_model, obscost3, (model.tf/(N-1))*sum(obsPotFunc3(μ[1,k],μ[2,k]) for k in 1:N))
      return obscost3
    elseif obs_i == 4
      @NLexpression(solver_model, obscost4, (model.tf/(N-1))*sum(obsPotFunc4(μ[1,k],μ[2,k]) for k in 1:N))
      return obscost4
    else
      print("obs_i=$obs_i")
      throw(MethodError("[Dubins5D.jl::obstacle_potential_penalty] Incorrect number of obstacles."))
    end
end







# The following methods return the dynamical constraints and their linearized versions
# These are returned as time-discretized versions of the constraints " x' - f(x,u) = 0 " or " x' - A(t)*(x - xp) - B(t)*(u - up) = 0 ", respectively



# Method that returns the dynamical constraints and their linearized versions all at once

function compute_dynamics(model::Dubins5D, μ_p, U_p)
    N = length(μ_p[1,:])

    b_all, b_dx_all, b_du_all = [], [], []
    σ_all, σ_dx_all           = [], []

    for k in 1:N-1
        x_k = μ_p[:,k]
        u_k = U_p[:,k]

        b_k, b_dx_k, b_du_k = b_dyn(x_k, u_k, model), b_dx(x_k, u_k, model), b_du(x_k, u_k, model)
        σ_k, σ_dx_k         = σ_dyn(x_k, u_k, model), σ_dx(x_k, u_k, model)

        push!(b_all,    b_k)
        push!(b_dx_all, b_dx_k)
        push!(b_du_all, b_du_k)
        push!(σ_all,    σ_k)
        push!(σ_dx_all, σ_dx_k)
    end

    return b_all, b_dx_all, b_du_all, σ_all, σ_dx_all
end





# These methods return the dynamics and its linearizations with respect to the state (matrix A(t)) and the control (matrix B(t)), respectively

function b_dyn(x::Vector, u::Vector, model::Dubins5D)
  θ, v, ω  = x[3:5] 
  a_v, a_ω = u

  b = zeros(model.x_dim)
  b[1] = v*cos(θ)
  b[2] = v*sin(θ)
  b[3] = ω
  b[4] = a_v
  b[5] = a_ω

  return b
end
function b_dx(x::Vector, u::Vector, model::Dubins5D)
  θ, v, ω = x[3:5] 

  A      = zeros(model.x_dim,model.x_dim)
  A[1,3] = -v*sin(θ)
  A[2,3] =  v*cos(θ)
  A[1,4] = cos(θ)
  A[2,4] = sin(θ)
  A[3,5] = 1.
  return A
end
function b_du(x::Vector, u::Vector, model::Dubins5D)
  B      = zeros(model.x_dim, model.u_dim)
  B[4,1] = 1.
  B[5,2] = 1.
  return B
end
function σ_dyn(x::Vector, u::Vector, model::Dubins5D)
  v, ω = x[4:5] 
  α, β = model.α, model.β

  sig_diag = [α*v*ω, α*v*ω, β*v*ω, 0., 0.]
  return diagm(sig_diag)
end
function σ_dx(x::Vector, u::Vector, model::Dubins5D)
  v, ω = x[4:5] 
  α, β = model.α, model.β

  mat_σ_dx = zeros(model.x_dim,model.x_dim,model.x_dim)
  mat_σ_dx[1,1, 4] = α*ω
  mat_σ_dx[1,1, 5] = α*v
  mat_σ_dx[2,2, 4] = α*ω
  mat_σ_dx[2,2, 5] = α*v
  mat_σ_dx[3,3, 4] = β*ω
  mat_σ_dx[3,3, 5] = β*v
  return mat_σ_dx
end


##########
# get sample paths Monte-Carlo
function simulate_monte_carlo(model::Dubins5D, u_traj;
                              N_MC::Int=1000)
  # Input:  u_traj: (u_dim x (N-1))
  # Output: X_MC:   (x_dim x N x N_MC)
  x0, x_dim, u_dim, N = model.x_init, model.x_dim, model.u_dim, (size(u_traj)[2]+1)
  dt = model.tf/(N-1)

  
  d = MvNormal(zeros(x_dim), dt*Matrix{Float64}(I,x_dim,x_dim))

  X_MC = zeros(x_dim, N, N_MC)
  for m_i in 1:N_MC
    x_traj = zeros(x_dim,N)
    x_traj[:,1] = x0
    for k in 1:(N-1)
      b_k   = b_dyn(x_traj[:,k], u_traj[:,k], model)
      sig_k = σ_dyn(x_traj[:,k], u_traj[:,k], model)
      w_k   = sig_k*rand(d)

      x_traj[:,k+1] = x_traj[:,k] + b_k*dt + w_k
    end
    X_MC[:,:,m_i] = x_traj
  end
  return X_MC
end



