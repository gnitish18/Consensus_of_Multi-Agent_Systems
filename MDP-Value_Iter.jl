using POMDPs, QuickPOMDPs, POMDPPolicies, POMDPModelTools
using Parameters, Random
using Combinatorics
using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style

@with_kw struct ConsensusProblem
	# Rewards
	r_state_change::Real = -100
	r_interact::Real = -500
	r_interaction_consensus::Real = 1000
	r_wisdom::Real = 500
	r_final_consensus::Real = 10000
	
	# Transition probabilities
	p_A_a0::Real = 0.3
	p_A_a1::Real = 0.8

	p_B_a0::Real = 0.7
	p_B_a1::Real = 0.2
	
	p_C_a0::Real = 0.5
	p_C_a1::Real = 0.5
end

params = ConsensusProblem()

n_states = 3
n_actions = 2
n_agents = 3
@enum State SHAPE_1ₛ SHAPE_2ₛ SHAPE_3ₛ
@enum Action IGNOREₐ INTERACTₐ 
@enum Agent A B C

𝒮 = [SHAPE_1ₛ, SHAPE_2ₛ, SHAPE_3ₛ]
𝒜 = [IGNOREₐ, INTERACTₐ]
𝒜𝒢 = [A, B, C]

StateSpace = [i for i in 0:n_states^n_agents-1]
ActionSpace = [i for i in 0:n_actions^n_agents-1]

All_States = [(i-1)*n_states^2 + (j-1)*n_states + (k-1) for i in 1:n_states for j in 1:n_states for k in 1:n_states]
All_Actions = [(i-1)*n_actions^2 + (j-1)*n_actions + (k-1) for i in 1:n_actions for j in 1:n_actions for k in 1:n_actions]

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

# println(All_States)
# for i in 1:n_states^n_agents
# 	println(All_States[i], " ", decode_States(i))
# end
# println(" \n")
# for i in 1:n_actions^n_agents
# 	print(decode_Actions(i))
# end

transition_function = function T(s, a, s̃)
	p_A[1] = params.p_A_a0
	p_A[2] = params.p_A_a1
	p_B[1] = params.p_B_a0
	p_B[2] = params.p_B_a1	
	p_C[1] = params.p_C_a0
	p_C[2] = params.p_C_a1

	for i in 1:3
		if a == IGNOREₐ
			if s == SHAPE_1ₛ
				return SparseCat([SHAPE_1ₛ, SHAPE_2ₛ, SHAPE_3ₛ], [p_A[1], (1-p_A[1])/2, (1-p_A[1])/2])
			elseif s == SHAPE_2ₛ
				return SparseCat([SHAPE_1ₛ, SHAPE_2ₛ, SHAPE_3ₛ], [(1-p_A[1])/2, p_A[1], (1-p_A[1])/2])
			elseif s == SHAPE_3ₛ
				return SparseCat([SHAPE_1ₛ, SHAPE_2ₛ, SHAPE_3ₛ], [(1-p_A[1])/2, (1-p_A[1])/2, p_A[1]])
			end
		elseif a == INTERACTₐ
			if s == SHAPE_1ₛ
				return SparseCat([SHAPE_1ₛ, SHAPE_2ₛ, SHAPE_3ₛ], [p_A[2], (1-p_A[2])/2, (1-p_A[2])/2])
			elseif s == SHAPE_2ₛ
				return SparseCat([SHAPE_1ₛ, SHAPE_2ₛ, SHAPE_3ₛ], [(1-p_A[2])/2, p_A[2], (1-p_A[2])/2])
			elseif s == SHAPE_3ₛ
				return SparseCat([SHAPE_1ₛ, SHAPE_2ₛ, SHAPE_3ₛ], [(1-p_A[2])/2, (1-p_A[2])/2, p_A[2]])
			end
		end
	end
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

#SparseCat([SHAPE_1ₛ, SHAPE_2ₛ, SHAPE_3ₛ], [0.7, 1, 0.2])

R(25, 6, 27)

mdp = QuickPOMDP(
    states       = StateSpace, 		# 𝒮
    actions      = ActionSpace,		# 𝒜
    discount     = 0.95,            # γ
	transition = T,
	reward = R,
	isterminal = s -> s[1] > 0.5
)