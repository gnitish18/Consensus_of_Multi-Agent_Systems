using POMDPs, QuickPOMDPs, POMDPPolicies, POMDPModelTools, POMDPSimulators
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using DiscreteValueIteration
using Parameters, Random
using LinearAlgebra
using Distributions
using StatsPlots
using StatsBase
using Colors
using Dates

@with_kw struct ConsensusProblem
	# Rewards
	r_state_change::Real = -1000
	r_interact::Real = -500
	r_interaction_consensus::Real = 100
	r_wisdom::Real = 50
	r_final_consensus::Real = 10000
end

params = ConsensusProblem()
tim = string(string(Dates.hour(Dates.now())),"-", string(Dates.minute(Dates.now())),"-", string(Dates.second(Dates.now())))
folder = string("Figures/", Dates.today(), "_", tim)
mkpath(folder)

function visualize(curr_state::Vector, curr_action::Vector, n_states, n_actions, n_agents, iter)
    θ = collect(0:2π/1000:2π)
    p = plot(cos.(θ), sin.(θ), aspect_raio=:equal, axis=nothing, label=nothing, bordercolor=:white, color=:white, linewidth=5, size=(500,500))
    ϕ = collect(1:n_agents)*2π/n_agents
    cols = distinguishable_colors(n_states+1, colorant"black")
    for agent in 1:n_agents
        if curr_action[agent] == 0
            scatter!(p, [cos(ϕ[agent])], [sin(ϕ[agent])], markersize=20, markerstrokewidth=0, markercolor=RGB(red(cols[curr_state[agent]+2]),green(cols[curr_state[agent]+2]),blue(cols[curr_state[agent]+2])), axis=nothing, label=nothing, bordercolor=:white, aspect_ratio=:equal)
        else 
            scatter!(p, [cos(ϕ[agent])], [sin(ϕ[agent])], markersize=20, markerstrokewidth=5, markercolor=RGB(red(cols[curr_state[agent]+2]),green(cols[curr_state[agent]+2]),blue(cols[curr_state[agent]+2])), axis=nothing, label=nothing, bordercolor=:white, aspect_ratio=:equal)
        end
    end
    title!("Iteration = "*string(iter))
    ylabel!("Colors -> Agent Opinions, Borders -> Interacting Agents")
    xlabel!(string("State = ", string(curr_state), ", Action = ", string(curr_action)))
    return p
end

# Agents, States and Actions
n_agents = 3
n_states = 3
n_actions = 2
# 𝒮 = {SHAPE_iₛ| for i in 1:n_agents}
# 𝒜 = {IGNOREₐ, INTERACTₐ}
# 𝒜𝒢 = {Aᵢ | i = 1:n_agents}
# Set of States and Actions in combined form
#	Eg: In case of 4 states for each agent, individual state of each agent can be 0, 1, 2, 3
# 		For 3 agents, no. of possible states = 4^3 = 64
# 		If the three agents have the states as [3, 0, 1]
# 		The corresponding combined form will be (3*4^2 + 0*4^1 + 1*$^0 + 1) = 50
#   The actions are also encoded similarly

# The combined state is decoded into individual states for each agent 
# 	Eg: 50 = (3*4^2 + 0*4^1 + 1*$^0 + 1) -> [3, 0, 1]
function decode_States(val, n_states, n_agents)
	val = val - 1
	States = zeros(Int, n_agents)
	for i in 1:n_agents
		States[n_agents - i + 1] = val % n_states
		val = val ÷ n_states
	end
	return States
end

# The combined action is decoded into individual actions for each agent 
function decode_Actions(val, n_actions, n_agents)
	val = val - 1
	Actions = zeros(Int, n_agents)
	for i in 1:n_agents
		Actions[n_agents - i + 1] = val % n_actions
		val = val ÷ n_actions
	end
	return Actions
end

# Generates a Transition matrix for the state, action, new state combination
# 	The Transition Matrix is a vector (size of action space) of matrices (size of state space x size of statespace)
# 	Transition Matrix -> Transitions[action][state, new state]
function Transition_Matrix(StateSpace, ActionSpace, n_agents)
	s = length(StateSpace) 		# number of states
	a = length(ActionSpace) 	# number of actions
	Transitions = [zeros(s,s) for k in 1:a]
	for k in 1:a
		Transitions[k][:,1] = rand(0:0.0001:1,s)
		for i in 1:s
			for j in shuffle(2:s-1)
				lim = 1 - sum(Transitions[k][i,(Transitions[k][i,:].!=0)])
				if lim > 0
					Transitions[k][i,j] = rand(0:0.0001:lim)
				else
					Transitions[k][i,j] = 0
				end	
			end
		end
		Transitions[k][:,end] = 1 .- sum(Transitions[k],dims=2)
		Transitions[k] = round.(Transitions[k], digits=3)

		for i in 1:s
			# Circular shift to move the first values to the corresponding diagonals 
			Transitions[k][i,:] = circshift(Transitions[k][i,:],i-1)
			# Normalizing the elements so that the rows sum up to 1
			Transitions[k][i,:] = normalize(Transitions[k][i,:],1)

			# The terminal states are given deterministic probability = 1 for self transition
			# The terminal states are the ones with consensus
			# 	i.e, [0,0,0] -> 0, [1,1,1] -> 22, [2,2,2] -> 43, or [3,3,3] -> 64
			state = decode_States(i, s, n_agents)
			n = length(unique(state))
			if n == 1
				Transitions[k][i,:] = zeros(s)
				Transitions[k][i,i] = 1
			end
		end
	end
	return Transitions
end

# Transition function that generates a categorical distribution for a given state and action
function T(s, a)
 	return SparseCat(StateSpace, Transitions[a][s,:])
end

# Reward function
function R(s, a, sp)
	Reward = 0
	States = decode_States(s, n_states, n_agents)
	Actions = decode_Actions(a, n_actions, n_agents)
	S̃tates = decode_States(sp, n_states, n_agents)

	# Flag for final consensus
	fl_consensus = 0	
	for i in 1:length(S̃tates)-1
		for j in i+1:length(S̃tates)
			if S̃tates[i] != S̃tates[j]
				fl_consensus = 1
				break
			end
		end
		if fl_consensus == 1
			break
		end
	end
	# Huge reward for final Concensus
	if fl_consensus == 0
		Reward += n_agents*params.r_final_consensus
	end

	# Penalty for State Change
	State_changes = findall(x -> x != 0, States - S̃tates)
	Reward += length(State_changes)*params.r_state_change

	# Finding all interacting Agents
	p = findall(x -> x == 1, Actions)	

	# Flag for consensus of interacting agents
	fl_intr_con = 0
	if length(p) == 1
		# Self-interaction Reward
		if States[p] != S̃tates[p]
			Reward += params.r_wisdom
		end
	else
		# Penalty for interaction
		Reward += length(p)*params.r_interact
		
		# Reward if interacting agents come to a Consensus
		for i in 1:length(p)-1
			for j in i+1:length(p)
				if States[p[i]] != States[p[j]]
					fl_intr_con = 1
					break
				end
			end
			if fl_intr_con == 1
				break
			end
		end
		if fl_intr_con == 0
			Reward += length(p)*params.r_interaction_consensus
		end
	end

	return Reward
end

# Sampling for Simulation based on the probability distribution of transition matrix
function gen_sample(policy, state)
	return sample(StateSpace, Weights(Transitions[action(policy, state)][state, :]))
end

# Function to check if consensus is reached
function consensus(i)
	state = decode_States(i, n_states, n_agents)
	n = length(unique(state))
	if n == 1
		return true
	end
	return false
end

function Simulation(state, policy, sim_no, type)
	# Cumulative reward
	Cum_Reward = 0
	anim = Animation()
	
	#@show decode_States(state, n_states, n_agents)
	iter = 0
	while !consensus(state)
		iter += 1

		new_state = gen_sample(policy, state)
		Cum_Reward += R(state, action(policy, state), new_state)
		state = new_state

		# Generate plots and frames for gif
		if sim_no == 1
			plt = visualize(decode_States(state, n_states, n_agents), decode_Actions(action(policy, state), n_actions, n_agents), n_states, n_actions, n_agents, iter)
			plot(plt)
			frame(anim, plt)
			st = string(folder, "/Sim_", tim, "-", string(type), "-", string(iter), "_", string(state), "_", string(action(policy,state)))
			savefig(st)
		end

		# Terminate as no convergence to consensus
		# 	Reason: Transition probabilities are such that agents are stubborn (High (or 1) self-transition probability)
		if iter > 100000
			println("Terminatins: Consensus not reached")
			break
		end
	end
	
	# Generate gif
	if sim_no == 1
		st = string(folder, "/Sim_", tim, "-", string(type),".gif")
		gif(anim, st, fps = 1)
	end
		
	return Cum_Reward
end

function compute(n_ag, n_s, n_a, tim)
	# Agents, States and Actions
	global n_agents = n_ag
	global n_states = n_s
	global n_actions = n_a
	# 𝒮 = {SHAPE_iₛ| for i in 1:n_agents}
	# 𝒜 = {IGNOREₐ, INTERACTₐ}
	# 𝒜𝒢 = {Aᵢ | i = 1:n_agents}

	# Set of States and Actions in combined form
	#	Eg: In case of 4 states for each agent, individual state of each agent can be 0, 1, 2, 3
	# 		For 3 agents, no. of possible states = 4^3 = 64
	# 		If the three agents have the states as [3, 0, 1]
	# 		The corresponding combined form will be (3*4^2 + 0*4^1 + 1*$^0 + 1) = 50
	#   The actions are also encoded similarly
	global StateSpace = [i for i in 1:n_states^n_agents]
	global ActionSpace = [i for i in 1:n_actions^n_agents]
	global folder = string("Figures/", Dates.today(), "_", tim, "/Ag-St=", string(n_agents), "_", string(n_states))
	mkpath(folder)

	# Transition Matrix
	global Transitions = Transition_Matrix(StateSpace, ActionSpace, n_agents)

	# Creating the MDP model
	global mdp = QuickMDP(
		states       = StateSpace, 		# 𝒮
		actions      = ActionSpace,		# 𝒜
		discount     = 0.95,            # γ
		transition = T,
		reward = R,
		initialstate = rand(StateSpace)
	)
		
	# Consensus states
	global Con_States = []
	for i in StateSpace
		if consensus(i)
			global Con_States = vcat(Con_States, i)
		end
	end

	# Generate a random initial state in which agents are not in consensus
	global pre_state = rand(StateSpace[filter(x->!(x in Con_States), eachindex(StateSpace))])

	# Cumulative rewards for each algorithm
	global Rand_Cum_Reward = []
	global ValIter_Cum_Reward = []

	# Types of policies
	global Types = ["Random_Policy", "Value_Iteration_Policy"]

	# Random Policy 
	global Random_policy = RandomPolicy(mdp);
		
	# Value Iteration
	global ValIter_solver = ValueIterationSolver(max_iterations=1000, belres=1e-5, verbose=true);
	global ValIter_policy = solve(ValIter_solver, mdp);

	global n_simulations = 1000
	# Simulating n_simulations times to check the performance of the algorithm
	for i in 1:n_simulations
		#println("\n Random Policy")
		global Rand_Cum_Reward = vcat(Rand_Cum_Reward, Simulation(pre_state, Random_policy, i, Types[1]))

		#println("\n ValIter Policy")
		global ValIter_Cum_Reward = vcat(ValIter_Cum_Reward, Simulation(pre_state, ValIter_policy, i, Types[2]))
	end

	# Fitting a Gaussian curve for all the n_simulations
	global fit_Random = fit(Normal, round.(Rand_Cum_Reward))
	global fit_ValIter = fit(Normal, round.(ValIter_Cum_Reward))

	# Plotting the performance metrics
	plot(fit_Random, fillrange=0, fillalpha=0.5 , fillcolor=:red, label="Random Policy", legend = :topleft)
	plot!(fit_Random, linealpha = 1, linecolor = :red, label = nothing)
	plot!(fit_ValIter, fillrange=0, fillalpha=0.5 , fillcolor=:blue, label="Value Iteration Policy", legend = :topleft)
	plot!(fit_ValIter, linealpha = 1, linecolor = :blue, label = nothing)
	xlabel!("Cumulative Reward")
	title!("Performance Comparison")
	st = string(folder, "/MDP_Performance-", tim, "-", string(n_agents), "_", string(n_states), "_", string(n_actions), ".png")
	savefig(st)
	return [fit_Random, fit_ValIter]
end

global tim = string(string(Dates.hour(Dates.now())),"-", string(Dates.minute(Dates.now())),"-", string(Dates.second(Dates.now())))

Ag_Rand_mean = []
Ag_Rand_stddev = []
Ag_ValIter_mean = []
Ag_ValIter_stddev = []
folder = string("Figures/", Dates.today(), "_", tim)

for no_of_agents in 2:5
	global data = compute(no_of_agents, 3, 2, tim)
 	global Rand_mean = vcat(Ag_Rand_mean, data[1].μ)
	global Rand_stddev = vcat(Ag_Rand_stddev, data[1].σ)
	global ValIter_mean = vcat(Ag_ValIter_mean, data[2].μ) 
	global ValIter_stddev = vcat(Ag_ValIter_stddev, data[2].σ)
end

folder = string("Figures/", Dates.today(), "_", tim)
plot([2:5], Ag_Rand_mean, label = "μ of Random Policy", linecolor = :red, linewidth = 2, legend = :bottomleft)
scatter!([2:5], Ag_Rand_mean, label = nothing, markercolor = :blue)
plot!([2:5], Ag_ValIter_mean, label = "μ of Val_Iter Policy", linecolor = :blue, linewidth = 2, legend = :bottomleft)
scatter!([2:5], Ag_ValIter_mean, label = nothing, markercolor = :blue)
xlabel!("Number of Agents")
ylabel!("Cumulative Reward")
title!("Mean")
st = string(folder, "/Mean", "-n_agents=", string(n_agents), ".png")
savefig(st)

plot([2:5], Ag_Rand_stddev, label = "σ of Random Policy", linecolor = :red, linewidth = 2, legend = :topleft)
scatter!([2:5], Ag_Rand_stddev, label = nothing, markercolor = :red)
plot!([2:5], Ag_ValIter_stddev, label = "σ of Val_Iter Policy", linecolor = :blue, linewidth = 2, legend = :topleft)
scatter!([2:5], Ag_ValIter_stddev, label = nothing, markercolor = :blue)
xlabel!("Number of Agents")
ylabel!("Cumulative Reward")
title!("Standard Deviation")
st = string(folder, "/Stddev", "-n_agents=", string(n_agents), ".png")
savefig(st)

folder = string("Figures/", Dates.today(), "_", tim)
St_Rand_mean = []
St_Rand_stddev = []
St_ValIter_mean = []
St_ValIter_stddev = []

for no_of_states in 2:5
 	global data = compute(3, no_of_states, 2, tim)
 	global St_Rand_mean = vcat(Rand_mean, data[1].μ)
	global St_Rand_stddev = vcat(Rand_stddev, data[1].σ)
	global St_ValIter_mean = vcat(ValIter_mean, data[2].μ) 
	global St_ValIter_stddev = vcat(ValIter_stddev, data[2].σ)
end

plot([2:5], St_Rand_mean, label = "μ of Random Policy", linecolor = :red, linewidth = 2, legend = :bottomleft)
scatter!([2:5], St_Rand_mean, label = nothing, markercolor = :red)
plot!([2:5], St_ValIter_mean, label = "μ of Val_Iter Policy", linecolor = :blue, linewidth = 2, legend = :bottomleft)
scatter!([2:5], St_ValIter_mean, label = nothing, markercolor = :blue)
xlabel!("Number of Agents")
ylabel!("Cumulative Reward")
title!("Mean")
st = string(folder, "/Mean", "-n_states=", string(n_states), ".png")
savefig(st)

plot([2:5], St_Rand_stddev, label = "σ of Random Policy", linecolor = :red, linewidth = 2, legend = :topleft)
scatter!([2:5], St_Rand_stddev, label = nothing, markercolor = :red)
plot!([2:5], St_ValIter_stddev, label = "σ of Val_Iter Policy", linecolor = :blue, linewidth = 2, legend = :topleft)
scatter!([2:5], St_ValIter_stddev, label = nothing, markercolor = :blue)
xlabel!("Number of Agents")
ylabel!("Cumulative Reward")
title!("Standard Deviation")
st = string(folder, "/Stddev", "-n_states=", string(n_states), ".png")
savefig(st)