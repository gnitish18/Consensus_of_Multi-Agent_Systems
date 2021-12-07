using POMDPs, QuickPOMDPs, POMDPPolicies, POMDPModelTools, POMDPSimulators
using Parameters, Random
using SparseArrays
using Combinatorics
using StatsBase
using SpecialFunctions
using LinearAlgebra
using DiscreteValueIteration
using Latexify
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style

@with_kw struct ConsensusProblem
	# Rewards
	r_state_change::Real = -10
	r_interact::Real = -5
	r_interaction_consensus::Real = 10
	r_wisdom::Real = 5
	r_final_consensus::Real = 100
end

params = ConsensusProblem()

n_states = 3
n_actions = 2
n_agents = 3
Transitions = 0
@enum State SHAPE_1ₛ SHAPE_2ₛ SHAPE_3ₛ
@enum Action IGNOREₐ INTERACTₐ 
@enum Agent A B C

𝒮 = [SHAPE_1ₛ, SHAPE_2ₛ, SHAPE_3ₛ]
𝒜 = [IGNOREₐ, INTERACTₐ]
𝒜𝒢 = [A, B, C]

StateSpace = [i for i in 0:n_states^n_agents-1]
ActionSpace = [i for i in 0:n_actions^n_agents-1]

function decode_States(val)
	val = val - 1
	States = zeros(Int, n_agents)
	for i in 1:n_agents
		States[n_agents - i + 1] = val % n_states
		val = val ÷ n_states
	end
	return States
end

function decode_Actions(val)
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

function T(s, a)
 	return SparseCat(StateSpace, Transitions[a+1][s+1,:])

end

function R(s, a, sp)
	Reward = 0
	States = decode_States(s)
	Actions = decode_Actions(a)
	S̃tates = decode_States(sp)

	# Huge reward for final Concensus
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
	if fl_consensus == 0
		Reward += n_agents*params.r_final_consensus
	end

	# Penalty for State Change
	State_changes = findall(x -> x != 0, States - S̃tates)
	Reward += length(State_changes)*params.r_state_change

	# Finding all interacting Agents
	p = findall(x -> x == 1, Actions)	

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

solver = ValueIterationSolver(max_iterations=1000, belres=1e-5, verbose=true)
policy = solve(solver, mdp)

function simul(policy, state)
	#display(T(state, action(policy, state)))
	Sample1 = rand(T(state, action(policy, state)))
	# Sample2 = sample(StateSpace, Weights(Transitions[action(policy, state) + 1][state+1, :]))
	# if (Transitions[action(policy, state) + 1][state+1, Sample1] > Transitions[action(policy, state) + 1][state+1, Sample2])
	# 	return Sample1
	# else
	# 	return Sample2
	# end
	return Sample1
end

pre_state = rand(StateSpace) + 1
@show decode_States(pre_state)

# for i = 1:10
# 	global pre_state = simul(policy, pre_state)
# end

function consensus(i)
	state = decode_States(i)
	n = length(unique(state))
	if n == 1
		return true
	end
	return false
end

while !consensus(pre_state)
	global pre_state = simul(policy, pre_state)
	@show decode_States(pre_state)
end
