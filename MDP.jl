using POMDPs, QuickPOMDPs, POMDPPolicies, POMDPModelTools, POMDPSimulators
using Parameters, Random
using StatsBase
using LinearAlgebra
using DiscreteValueIteration
using MCTS
using Distributions
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using StatsPlots

@with_kw struct ConsensusProblem
	# Rewards
	r_state_change::Real = -1000
	r_interact::Real = -500
	r_interaction_consensus::Real = 100
	r_wisdom::Real = 50
	r_final_consensus::Real = 10000
end

params = ConsensusProblem()

# Agents, States and Actions
n_agents = 3
n_states = 4
n_actions = 2
@enum Agent A B C
@enum State SHAPE_1ₛ SHAPE_2ₛ SHAPE_3ₛ SHAPE_4ₛ
@enum Action IGNOREₐ INTERACTₐ 
𝒮 = [SHAPE_1ₛ, SHAPE_2ₛ, SHAPE_3ₛ, SHAPE_4ₛ]
𝒜 = [IGNOREₐ, INTERACTₐ]
𝒜𝒢 = [A, B, C]

# Set of States and Actions in combined form
#	Eg: In case of 4 states for each agent, individual state of each agent can be 0, 1, 2, 3
# 		For 3 agents, no. of possible states = 4^3 = 64
# 		If the three agents have the states as [3, 0, 1]
# 		The corresponding combined form will be (3*4^2 + 0*4^1 + 1*$^0 + 1) = 50
#   The actions are also encoded similarly
StateSpace = [i for i in 1:n_states^n_agents]
ActionSpace = [i for i in 1:n_actions^n_agents]

# The combined state is decoded into individual states for each agent 
# 	Eg: 50 = (3*4^2 + 0*4^1 + 1*$^0 + 1) -> [3, 0, 1]
function decode_States(val)
	val = val - 1
	States = zeros(Int, n_agents)
	for i in 1:n_agents
		States[n_agents - i + 1] = val % n_states
		val = val ÷ n_states
	end
	return States
end

# The combined action is decoded into individual actions for each agent 
function decode_Actions(val)
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
function Transition_Matrix(StateSpace, ActionSpace)
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
			state = decode_States(i)
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
	States = decode_States(s)
	Actions = decode_Actions(a)
	S̃tates = decode_States(sp)

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

# Transition Matrix
Transitions = Transition_Matrix(StateSpace, ActionSpace)

# Creating the MDP model
mdp = QuickMDP(
	states       = StateSpace, 		# 𝒮
	actions      = ActionSpace,		# 𝒜
	discount     = 0.95,            # γ
	transition = T,
	reward = R,
	initialstate = rand(StateSpace)
)

# Sampling for Simulation based on the probability distribution of transition matrix
function gen_sample(policy, state)
	return sample(StateSpace, Weights(Transitions[action(policy, state)][state, :]))
end

# Function to check if consensus is reached
function consensus(i)
	state = decode_States(i)
	n = length(unique(state))
	if n == 1
		return true
	end
	return false
end

function Simulation(state, policy, sim_no, type)
	# Cumulative reward
	Cum_Reward = 0

	#@show decode_States(state)
	iter = 0
	while !consensus(state)
		iter += 1
		new_state = gen_sample(policy, state)
		Cum_Reward += R(state, action(policy, state), new_state)
		state = new_state
		#@show decode_States(state)

		# Terminate as no convergence to consensus
		# 	Reason: Transition probabilities are such that agents are stubborn (High (or 1) self-transition probability)
		if iter > 1000000
			println("Consensus not reached")
			break
		end
	end
	return Cum_Reward
end

# Consensus states
Con_States = []
for i in StateSpace
	if consensus(i)
		global Con_States = vcat(Con_States, i)
	end
end

# Generate a random initial state in which agents are not in consensus
pre_state = rand(StateSpace[filter(x->!(x in Con_States), eachindex(StateSpace))])

# Cumulative rewards for each algorithm
Rand_Cum_Reward = []
ValIter_Cum_Reward = []

# Types of policies
Types = ["Random Policy", "Value Iteration Policy"]

# Random Policy 
Random_policy = RandomPolicy(mdp);
	
# Value Iteration
ValIter_solver = ValueIterationSolver(max_iterations=1000, belres=1e-5, verbose=true);
ValIter_policy = solve(ValIter_solver, mdp);

n_simulations = 10000
# Simulating n_simulations times to check the performance of the algorithm
for i in 1:n_simulations
	#println("\n Random Policy")
	global Rand_Cum_Reward = vcat(Rand_Cum_Reward, Simulation(pre_state, Random_policy, i, Types[1]))

	#println("\n ValIter Policy")
	global ValIter_Cum_Reward = vcat(ValIter_Cum_Reward, Simulation(pre_state, ValIter_policy, i, Types[2]))
end

# Fitting a Gaussian curve for all the n_simulations
fit_Random = fit(Normal, round.(Rand_Cum_Reward))
fit_ValIter = fit(Normal, round.(ValIter_Cum_Reward))

# Plotting the performance metrics
plot(fit_Random, fillrange=0, fillalpha=0.5 , fillcolor=:red, label="Random Policy")
plot!(fit_ValIter, fillrange=0, fillalpha=0.5 , fillcolor=:blue, label="Value Iteration Policy")
xlabel!("Cumulative Reward")
title!("Performance Comparison")
savefig("Figures/MDP_Performance.png")
