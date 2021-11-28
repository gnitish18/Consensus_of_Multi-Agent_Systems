using LinearAlgebra
using Plots
using JuMP
using GLPK

𝒮 = [CIRCLEₛ, SQUAREₛ, TRIANGLEₛ]
𝒜 = [IGNOREₐ, INTERACTₐ]
ℐ = [A, B, C]

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

struct POMG
    γ # discount factor
    ℐ # agents
    𝒮 # state space
    𝒜 # joint action space
    #𝒪 # joint observation space
    T # transition function
    #O # joint observation function
    R # joint reward function
end

MDP = POMG(0.95, ℐ, 𝒮, 𝒜, T, R)

function lookahead(𝒫::POMG, U, s, a)
    #𝒮, 𝒪, T, O, R, γ = 𝒫.𝒮, joint(𝒫.𝒪), 𝒫.T, 𝒫.O, 𝒫.R, 𝒫.γ
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    #u′ = sum(R(s,a,s′) + γ*T(s,a,s′)*sum(O(a,s′,o)*U(o,s′) for o in 𝒪) for s′ in 𝒮)
    u′ = sum(R(s,a,s′) + γ*T(s,a,s′)*sum(O(a,s′,o)*U(o,s′) for o in 𝒪) for s′ in 𝒮)
    return u′
end

function evaluate_plan(𝒫::POMG, π, s)
    a = Tuple(πi() for πi in π)
    U(o,s′) = evaluate_plan(𝒫, [πi(oi) for (πi, oi) in zip(π,o)], s′)
    return isempty(first(π).subplans) ? 𝒫.R(s,a) : lookahead(𝒫, U, s, a)
end

function utility(𝒫::POMG, b, π)
    u = [evaluate_plan(𝒫, π, s) for s in 𝒫.𝒮]
    return sum(bs * us for (bs, us) in zip(b, u))
end

struct POMGNashEquilibrium
    b # initial belief
    d # depth of conditional plans
end

function create_conditional_plans(𝒫, d)
    ℐ, 𝒜, 𝒪 = 𝒫.ℐ, 𝒫.𝒜, 𝒫.𝒪
    Π = [[ConditionalPlan(ai) for ai in 𝒜[i]] for i in ℐ]
    for t in 1:d
        Π = expand_conditional_plans(𝒫, Π)
    end
    return Π
end

function expand_conditional_plans(𝒫, Π)
    ℐ, 𝒜, 𝒪 = 𝒫.ℐ, 𝒫.𝒜, 𝒫.𝒪
    return [[ConditionalPlan(ai, Dict(oi => πi for oi in 𝒪[i]))

    for πi in Π[i] for ai in 𝒜[i]] for i in ℐ]
end

function solve(M::POMGNashEquilibrium, 𝒫::POMG)
    ℐ, γ, b, d = 𝒫.ℐ, 𝒫.γ, M.b, M.d
    Π = create_conditional_plans(𝒫, d)
    U = Dict(π => utility(𝒫, b, π) for π in joint(Π))
    𝒢 = SimpleGame(γ, ℐ, Π, π -> U[π])
    π = solve(NashEquilibrium(), 𝒢)
    return Tuple(argmax(πi.p) for πi in π)
end

struct POMGDynamicProgramming
    b # initial belief
    d # depth of conditional plans
end

function solve(M::POMGDynamicProgramming, 𝒫::POMG)
    ℐ, 𝒮, 𝒜, R, γ, b, d = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.γ, M.b, M.d
    Π = [[ConditionalPlan(ai) for ai in 𝒜[i]] for i in ℐ]
    for t in 1:d
        Π = expand_conditional_plans(𝒫, Π)
        prune_dominated!(Π, 𝒫)
    end
    𝒢 = SimpleGame(γ, ℐ, Π, π -> utility(𝒫, b, π))
    π = solve(NashEquilibrium(), 𝒢)
    return Tuple(argmax(πi.p) for πi in π)
end

function prune_dominated!(Π, 𝒫::POMG)
    done = false
    while !done
        done = true
        for i in shuffle(𝒫.ℐ)
            for πi in shuffle(Π[i])
                if length(Π[i]) > 1 && is_dominated(𝒫, Π, i, πi)
                    filter!(πi′ -> πi′ ≠ πi, Π[i])
                    done = false
                    break
                end
            end
        end
    end
end

function is_dominated(𝒫::POMG, Π, i, πi)
    ℐ, 𝒮 = 𝒫.ℐ, 𝒫.𝒮
    jointΠnoti = joint([Π[j] for j in ℐ if j ≠ i])
    π(πi′, πnoti) = [j==i ? πi′ : πnoti[j>i ? j-1 : j] for j in ℐ]
    Ui = Dict((πi′, πnoti, s) => evaluate_plan(𝒫, π(πi′, πnoti), s)[i]
        for πi′ in Π[i], πnoti in jointΠnoti, s in 𝒮)
    model = Model(Ipopt.Optimizer)
    @variable(model, δ)
    @variable(model, b[jointΠnoti, 𝒮] ≥ 0)
    @objective(model, Max, δ)
    @constraint(model, [πi′=Π[i]],
        sum(b[πnoti, s] * (Ui[πi′, πnoti, s] - Ui[πi, πnoti, s])
        for πnoti in jointΠnoti for s in 𝒮) ≥ δ)
    @constraint(model, sum(b) == 1)
    optimize!(model)
    return value(δ) ≥ 0
end
