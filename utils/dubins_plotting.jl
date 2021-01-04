# -------------------------------------------------------------------------------
# -   Plotting script for the Dubins example - T. Lew + R. Bonalli 07/2020   -
# -------------------------------------------------------------------------------

using Distributions

# Python plotting with matplotlib
using PyCall, LaTeXStrings
import PyPlot; const plt = PyPlot

include("../Models/dubins5D.jl")
include("../SCP/scp_problem.jl")


# --------------------
# Plotting with PyPlot
# --------------------
function plt_circle(ax, pos, radius; color="k", alpha=1., label="None")
    # Filled circle
    circle = plt.matplotlib.patches.Circle(pos, radius=radius,
                    color=color, fill=true, alpha=alpha)
    ax.add_patch(circle)
    # Edge of circle, with alpha 1.
    if label=="None"
        c_edge = plt.matplotlib.patches.Circle(pos, radius=radius, 
                        color=color, alpha=1, fill=false)
    else
        c_edge = plt.matplotlib.patches.Circle(pos, radius=radius, 
                        color=color, alpha=1, fill=false, label=label)
    end
    ax.add_patch(c_edge)
    return ax
end
function plot_gaussian_confidence_ellipse(ax, mu, Sigma; probability=0.9, 
                                          additional_radius=0., alpha=0.1,
                                          color="b")
    n_dofs   = size(mu)[1]
    quantile = cquantile(Chisq(n_dofs), 1-probability)
    Q        = quantile * Sigma
    return plot_ellipse(ax, mu, Q, 
                        additional_radius=additional_radius, 
                        color=color, alpha=alpha)
end
function plot_ellipse(ax, mu, Q; additional_radius=0., 
                                  color="b", alpha=0.1)
    # Compute eigenvalues and associated eigenvectors
    # vals, vecs = np.linalg.eigh(Q)
    # print("Q=",Q)
    vals = eigvals(Hermitian(Q))
    vecs = eigvecs(Hermitian(Q))
    # print("vals=",vals)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 1]
    theta = atan(y, x) * (180/pi)

    # Eigenvalues give length of ellipse along each eigenvector
    w, h    =  2. * (sqrt.(vals) .+ additional_radius)
    ellipse = plt.matplotlib.patches.Ellipse(mu, w, h, theta, 
                                             color=color, alpha=alpha)
    ax.add_patch(ellipse)
    # ellipse.set_clip_box(ax.bbox)
    # ax.add_artist(ellipse) 
    return ax
end


function plt_solutions(scp_problem::SCPProblem, model, 
                       X_all, Sigma_all, U_all;
                       idx = [1,2],
                       B_manually_set_lims=false, xlims=[-0.25,3.25], ylims=[-0.25,3.25], 
                       figsize=(8,6), B_plot_labels=true,
                       B_plot_ellipses_final_traj=true)
    if !(B_manually_set_lims)
        xlims, ylims = zeros(2), zeros(2)
        for iter = 1:length(X_all)
            X = X_all[iter]
            max_xi, min_xi = maximum(X[idx[1],:]), minimum(X[idx[1],:])
            max_yi, min_yi = maximum(X[idx[2],:]), minimum(X[idx[2],:])
            if xlims[1] > min_xi
                xlims[1] = min_xi
            end
            if xlims[2] < max_xi
                xlims[2] = max_xi
            end
            if ylims[1] > min_yi
                ylims[1] = min_yi
            end
            if ylims[2] < max_yi
                ylims[2] = max_yi
            end
        end
    end

    N = length(X_all)

    fig = plt.figure(figsize=figsize)
    ax  = plt.gca()

    # Plot SCP solutions
    plt.plot(X_all[1][idx[1],:], X_all[1][idx[2],:],
                    label="Initializer", linewidth=3, "--")
    for iter = 2:length(X_all)
        X = X_all[iter]
        ax.plot(X[idx[1],:], X[idx[2],:],
                    label="Iterate $(iter - 1)", linewidth=2)
    end

    # Plot Gaussian confidence ellipsoids of last trajectory
    if B_plot_ellipses_final_traj
        for k = 2:size(X_all[end])[2]
            mu, Sigma = X_all[end][idx,k], Sigma_all[end][idx,idx,k]
            plot_gaussian_confidence_ellipse(ax, mu, Sigma, 
                                             probability=model.prob)
        end
    end

    # Plot obstacles
    for obs_i = 1:length(model.obstacles)
        p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
        plt_circle(ax, p_obs[idx], obs_radius; color="r", alpha=0.3)
    end


    # Settings / Style / Parameters
    PyPlot.rc("text", usetex=true)
    rcParams = PyDict(plt.matplotlib["rcParams"])
    rcParams["font.size"] = 20
    rcParams["font.family"] = "Helvetica"
    plt.xlim(xlims)
    plt.ylim(ylims)
    if B_plot_labels
        # plt.title("Open-Loop Dubins Trajectories", pad=10)
        # plt.xlabel("E")
        # plt.ylabel("N")    
        plt.legend(loc="upper right", fontsize=18, 
                                      labelspacing=0.1)
    end
    plt.grid(alpha=0.3)

    plt.draw()

    return fig
end

function plot_MC(fig, X_MC; idx=[1,2])
    # Input: X_MC:   (x_dim x N x N_MC)
    # ax  = plt.gca()

    for i in 1:size(X_MC)[3]#N_MC
        x_traj = X_MC[:,:,i]
        plt.plot(x_traj[idx[1],:], x_traj[idx[2],:], 
                    linewidth=1, color="dodgerblue", alpha=0.1)
    end
    plt.draw()
end
function plot_MC_1dim(fig, X_MC; idx=1)
    # Input: X_MC:   (x_dim x N x N_MC)
    # ax  = plt.gca()

    for i in 1:size(X_MC)[3]#N_MC
        x_traj = X_MC[:,:,i]
        plt.plot(1:size(x_traj)[2], x_traj[idx,:], 
                    linewidth=1, color="dodgerblue", alpha=0.1)
    end
    plt.draw()
end

function plt_controls(U)
    fig = plt.figure()
    ax  = plt.gca()

    # Plot SCP solutions
    for u_dim = 1:size(U)[1]
        plt.plot(1:size(U)[2], U[u_dim,:], label=u_dim)
    end
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=18)

    plt.draw()

    return fig
end

function plt_all_lin_controls(U_all, B_plot_labels=true)
    N = length(U_all)

    fig = plt.figure()
    ax  = plt.gca()

    # Plot SCP solutions
    plt.plot(1:size(U_all[1])[2], U_all[1][1,:],
                    label="Initializer", linewidth=2)
    for iter = 2:length(U_all)
        U = U_all[iter]
        plt.plot(1:size(U)[2], U[1,:], 
                    label="Iterate $(iter - 1)", linewidth=2)
    end

    # Settings / Style / Parameters
    PyPlot.rc("text", usetex=true)
    rcParams = PyDict(plt.matplotlib["rcParams"])
    rcParams["font.size"] = 20
    rcParams["font.family"] = "Helvetica"
    if B_plot_labels
        plt.legend(loc="upper right", fontsize=18, 
                                      labelspacing=0.1)
    end
    plt.grid(alpha=0.3)

    plt.draw()

    return fig
end
function plt_all_ang_controls(U_all, B_plot_labels=true)
    N = length(U_all)

    fig = plt.figure()
    ax  = plt.gca()

    # Plot SCP solutions
    plt.plot(1:size(U_all[1])[2], U_all[1][2,:],
                    label="Initializer", linewidth=2)
    for iter = 2:length(U_all)
        U = U_all[iter]
        plt.plot(1:size(U)[2], U[2,:], 
                    label="Iterate $(iter - 1)", linewidth=2)
    end

    # Settings / Style / Parameters
    PyPlot.rc("text", usetex=true)
    rcParams = PyDict(plt.matplotlib["rcParams"])
    rcParams["font.size"] = 20
    rcParams["font.family"] = "Helvetica"
    if B_plot_labels
        plt.legend(loc="upper right", fontsize=18, 
                                      labelspacing=0.1)
    end
    plt.grid(alpha=0.3)

    plt.draw()

    return fig
end

function plt_lin_velocity(X)
    fig = plt.figure()
    ax  = plt.gca()

    # Plot SCP solutions
    plt.plot(1:size(X)[2], X[4,:])

    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=18)

    plt.draw()

    return fig
end
function plt_ang_velocity(X)
    fig = plt.figure()
    ax  = plt.gca()

    # Plot SCP solutions
    plt.plot(1:size(X)[2], X[5,:])

    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=18)

    plt.draw()

    return fig
end

function plt_all_lin_velocities(X_all, B_plot_labels=true)
    N = length(X_all)

    fig = plt.figure()
    ax  = plt.gca()

    # Plot SCP solutions
    plt.plot(1:size(X_all[1])[2], X_all[1][4,:],
                    label="Initializer", linewidth=2)
    for iter = 2:length(X_all)
        X = X_all[iter]
        plt.plot(1:size(X)[2], X[4,:], 
                    label="Iterate $(iter - 1)", linewidth=2)
    end

    # Settings / Style / Parameters
    PyPlot.rc("text", usetex=true)
    rcParams = PyDict(plt.matplotlib["rcParams"])
    rcParams["font.size"] = 20
    rcParams["font.family"] = "Helvetica"
    if B_plot_labels
        plt.legend(loc="upper right", fontsize=18, 
                                      labelspacing=0.1)
    end
    plt.grid(alpha=0.3)

    plt.draw()

    return fig
end
function plt_all_ang_velocities(X_all, B_plot_labels=true)
    N = length(X_all)

    fig = plt.figure()
    ax  = plt.gca()

    # Plot SCP solutions
    plt.plot(1:size(X_all[1])[2], X_all[1][5,:],
                    label="Initializer", linewidth=2)
    for iter = 2:length(X_all)
        X = X_all[iter]
        plt.plot(1:size(X)[2], X[5,:], 
                    label="Iterate $(iter - 1)", linewidth=2)
    end

    # Settings / Style / Parameters
    PyPlot.rc("text", usetex=true)
    rcParams = PyDict(plt.matplotlib["rcParams"])
    rcParams["font.size"] = 20
    rcParams["font.family"] = "Helvetica"
    if B_plot_labels
        plt.legend(loc="upper right", fontsize=18, 
                                      labelspacing=0.1)
    end
    plt.grid(alpha=0.3)

    plt.draw()

    return fig
end

function plt_final_solution(scp_problem::SCPProblem, model, X, U)
    N = length(X_all)
    idx = [1,2]

    fig = plt.figure(figsize=(4.5, 7.5))
    ax  = plt.gca()

    # Plot final solution
    plt.plot(X[idx[1],:], X[idx[2],:], "bo-", 
                linewidth=2, markersize=6)
    plt.plot(Inf*[1,1],Inf*[1,1], "b-", label="Trajectory") # for legend

    # Plot Thrust
    for k = 1:(length(X[1,:])-1)
        xk, uk =  X[:,k], U[:,k]

        uMax, mag = 23.2, 1.5

        plt.arrow(xk[idx[1]], xk[idx[2]], 
                    mag*(uk[idx[1]]/uMax), mag*(uk[idx[2]]/uMax),
                    color="g")
    end
    plt.plot(Inf*[1,1],Inf*[1,1], "g-", label="Thrust") # for legend

    # Plot obstacles
    for obs_i = 1:length(model.obstacles)
        p_obs, obs_radius = model.obstacles[obs_i][1], model.obstacles[obs_i][2]
        plt_circle(ax, p_obs[idx], obs_radius; color="r", alpha=0.4)
    end
    plt.plot(Inf*[1,1],Inf*[1,1], "r-", label="Obstacle") # for legend

    # Settings / Style / Parameters
    PyPlot.rc("text", usetex=true)
    rcParams = PyDict(plt.matplotlib["rcParams"])
    rcParams["font.size"] = 20
    rcParams["font.family"] = "Helvetica"
    plt.title("Final Quadcopter Trajectory", pad=15)
    plt.xlim([-0.5,3.])
    plt.ylim([ 0.0,6.])
    plt.xlabel("E")
    plt.ylabel("N")    
    plt.legend(loc="lower right", fontsize=18)
    plt.grid(alpha=0.3)
    
    plt.draw()

    return fig
end

function plt_final_angle_accel(scp_problem::SCPProblem, model, X, U)
    N = length(X_all)
    t_max = 2.

    times = collect(range(0,stop=(SCPproblem.N-1)*SCPproblem.dt,length=SCPproblem.N))[1:SCPproblem.N-1]
    norms_U = sqrt.(U[1,:].^2+U[2,:].^2+U[3,:].^2)

    fig = plt.figure(figsize=(5.5, 7.5))

    # -------------
    # Tilt Angle
    plt.subplot(2,1,1)
    tilt_angles = U[3,:]./norms_U
    tilt_angles = (180. / pi) * tilt_angles
    plt.plot(times, tilt_angles, "bo-", 
                linewidth=1, markersize=4)
    # max tilt angle
    theta_max = (180/pi) * (pi/3.0)
    plt.plot([0,t_max], theta_max*ones(2), 
                    color="r", linestyle="dashed", linewidth=2)
    plt.fill_between([0,t_max], theta_max*ones(2), 90, 
                        color="r", alpha=0.2)
    # Params
    plt.title("Tilt Angle", pad=10)
    plt.xlim([0, t_max])
    plt.ylim([0, 65   ])
    # plt.xlabel("Time [s]")
    plt.ylabel(L"$\theta$ [deg]")
    plt.grid(alpha=0.3)
    plt.draw()

    # -------------
    # Acceleration
    plt.subplot(2,1,2)
    fig.tight_layout(pad=2.0)

    plt.plot(times, norms_U, "bo-", 
                linewidth=1, markersize=4)

    # max/min acceleration
    a_min, a_max = 0.6, 23.2
    plt.plot([0,t_max], a_max*ones(2), 
                    color="r", linestyle="dashed", linewidth=2)
    plt.fill_between([0,t_max], a_max*ones(2), 90, 
                        color="r", alpha=0.2)
    plt.plot([0,t_max], a_min*ones(2), 
                    color="r", linestyle="dashed", linewidth=2)
    plt.fill_between([0,t_max], a_min*ones(2), -5, 
                        color="r", alpha=0.2)

    # Parameters / Settings / Labels
    PyPlot.rc("text", usetex=true)
    rcParams = PyDict(plt.matplotlib["rcParams"])
    rcParams["font.size"] = 20
    rcParams["font.family"] = "Helvetica"
    plt.title("Cmd. Acceleration", pad=10)
    plt.xlim([0, t_max])
    plt.ylim([-1, 25  ])
    plt.xlabel("Time [s]")
    plt.ylabel(L"$u [m/s^2]$")
    plt.grid(alpha=0.3)
    plt.draw()

    return fig
end