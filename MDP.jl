using POMDPs, QuickPOMDPs, POMDPPolicies, POMDPModelTools, POMDPSimulators
using Parameters, Random
using StatsBase
#using SpecialFunctions
using LinearAlgebra
using DiscreteValueIteration
using MCTS
using Distributions
using Plots; default(fontfamily="Computer Modern", framestyle=:box)
using StatsPlots

@with_kw struct ConsensusProblem
	# Rewards
	r_state_change::Real = -1000			#-100
	r_interact::Real = -500					#-50
	r_interaction_consensus::Real = 100		#100
	r_wisdom::Real = 50						#50
	r_final_consensus::Real = 10000			#1000
end

params = ConsensusProblem()

# Agents, States and Actions
n_agents = 3
n_states = 3
n_actions = 2
@enum Agent A B C
@enum State SHAPE_1ₛ SHAPE_2ₛ SHAPE_3ₛ
@enum Action IGNOREₐ INTERACTₐ 
𝒮 = [SHAPE_1ₛ, SHAPE_2ₛ, SHAPE_3ₛ]
𝒜 = [IGNOREₐ, INTERACTₐ]
𝒜𝒢 = [A, B, C]

# Set of States and Actions in combined form
StateSpace = [i for i in 1:n_states^n_agents]
ActionSpace = [i for i in 1:n_actions^n_agents]

# 
function decode_States(val)
	# converts to 0 to 26 
	val = val - 1
	States = zeros(Int, n_agents)
	for i in 1:n_agents
		States[n_agents - i + 1] = val % n_states
		val = val ÷ n_states
	end
	return States
end

# takes in 1 to 8
function decode_Actions(val)
	# converts to 0 to 7 
	val = val - 1
	Actions = zeros(Int, n_agents)
	for i in 1:n_agents
		Actions[n_agents - i + 1] = val % n_actions
		val = val ÷ n_actions
	end
	return Actions
end

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
			Transitions[k][i,:] = circshift(Transitions[k][i,:],i-1)
			Transitions[k][i,:] = normalize(Transitions[k][i,:],1)
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

# 1 to n
function T(s, a)
 	return SparseCat(StateSpace, Transitions[a][s,:])

end

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

Transitions = Transition_Matrix(StateSpace, ActionSpace)

mdp = QuickMDP(
	states       = StateSpace, 		# 𝒮
	actions      = ActionSpace,		# 𝒜
	discount     = 0.95,            # γ
	transition = T,
	reward = R,
	initialstate = rand(StateSpace),
	isterminal = s -> s[1] == 26
)

function gen_sample(policy, state)
	# Sample1 = rand(T(state, action(policy, state)))
	return sample(StateSpace, Weights(Transitions[action(policy, state)][state, :]))
	# if (Transitions[action(policy, state)][state, Sample1] > Transitions[action(policy, state)][state, Sample2])
	# 	return Sample1
	# else
	 	return Sample2
	# end
end

function consensus(i)
	state = decode_States(i)
	n = length(unique(state))
	if n == 1
		return true
	end
	return false
end

function Simulation(state, policy)
	Cum_Reward = 0
	#@show decode_States(state)
	iter = 0
	while !consensus(state)
		iter += 1
		new_state = gen_sample(policy, state)
		#@show R(state, action(policy, state), new_state)
		Cum_Reward += R(state, action(policy, state), new_state)
		state = new_state
		#@show decode_States(state)
		if iter > 1000000
			println("Consensus not reached")
			break
		end
	end
	return Cum_Reward
end

Con_States = []
for i in StateSpace
	if consensus(i)
		global Con_States = vcat(Con_States, i)
	end
end
pre_state = rand(StateSpace[filter(x->!(x in Con_States), eachindex(StateSpace))])
Rand_Cum_Reward = []
ValIter_Cum_Reward = []

Random_policy = RandomPolicy(mdp);
	
ValIter_solver = ValueIterationSolver(max_iterations=1000, belres=1e-5, verbose=true);
ValIter_policy = solve(ValIter_solver, mdp);

for i in 1:10000
	#println("\n Random Policy")
	global Rand_Cum_Reward = vcat(Rand_Cum_Reward, Simulation(pre_state, Random_policy))

	#println("\n ValIter Policy")
	global ValIter_Cum_Reward = vcat(ValIter_Cum_Reward, Simulation(pre_state, ValIter_policy))
end

@show r = fit(Normal, round.(Rand_Cum_Reward))
plot(r, fillrange=0, fillalpha=0.5 , fillcolor=:red, label="Random Policy")
@show v = fit(Normal, round.(ValIter_Cum_Reward))
plot!(v, fillrange=0, fillalpha=0.5 , fillcolor=:blue, label="Value Iteration Policy")
xlabel!("Cumulative Reward")
title!("Performance Comparison")
savefig("MDP.png")
# MCTS_solver = MCTSSolver(n_iterations=50, depth=20, exploration_constant=5.0)
# MCTS_policy = solve(MCTS_solver, mdp)
# @show Simulation(pre_state, MCTS_policy)