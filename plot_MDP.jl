using Plots
using Colors

function visualize(curr_state::Vector, curr_action::Vector, n_states, n_actions, n_agents, iter)
    θ = collect(0:2π/10000000:2π)
    p = plot(cos.(θ), sin.(θ), aspect_raio=:equal, axis=nothing, label=nothing, bordercolor=:white, color=:white, linewidth=5, size=(500,500))
    ϕ = collect(1:n_agents)*2π/n_agents
    cols = distinguishable_colors(n_states+1, colorant"black")
    for agent in 1:n_agents
        if curr_action[agent] == 0
            scatter!(p, [cos(ϕ[agent])], [sin(ϕ[agent])], markersize=18, markerstrokewidth=0, markercolor=RGB(red(cols[curr_state[agent]+2]),green(cols[curr_state[agent]+2]),blue(cols[curr_state[agent]+2])), axis=nothing, label=nothing, bordercolor=:white, aspect_ratio=:equal)
        else 
            scatter!(p, [cos(ϕ[agent])], [sin(ϕ[agent])], markersize=18, markerstrokewidth=5, markercolor=RGB(red(cols[curr_state[agent]+2]),green(cols[curr_state[agent]+2]),blue(cols[curr_state[agent]+2])), axis=nothing, label=nothing, bordercolor=:white, aspect_ratio=:equal)
        end
    end
    title!("Iteration = "*string(iter))
    xlabel!("Colors = Agent Opinions, Borders = Interacting Agents")
    savefig(p, "visual.png");
    return p
end
#=
iter=2
n_actions = 2
n_agents = 6
n_states = 5
curr_state = rand(0:n_states-1, n_agents)
curr_action = rand(0:n_actions-1, n_agents)
plt = visualize(curr_state, curr_action, n_states, n_actions, n_agents, iter)
plot(plt)
=#





#=
a = Animation()
	
for i in 1:10
    plt = bar(1:i, ylim=(0,10), xlim=(0,10), lab="")
    frame(a, plt)
end
	
gif(a, "simulation.gif")
=#

