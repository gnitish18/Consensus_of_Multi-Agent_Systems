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

a = Animation()