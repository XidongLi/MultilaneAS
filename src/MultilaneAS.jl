module MultilaneAS

using AutomotiveSimulator
using AutomotiveVisualization
using POMDPs
using POMDPPolicies
#import StatsBase: Weights, sample
using Random
#import Base.LinAlg: diagm, chol
using Base.Iterators: repeated
import LinearAlgebra: diagm, cholesky, dot
using MultivariateStats
using ParticleFilters
using Parameters
using SpecialFunctions
import Base: +,-,*,/
using POMDPModelTools
using ARDESPOT
using MCTS
using GridInterpolations
#using LocalFunctionApproximation
#using GlobalApproximationValueIteration
#using LocalApproximationValueIteration
import StaticArrays: SVector
using Flux
using DeepQLearning
using NNlib
using BSON: @save, load
using CUDA
using CommonRLInterface: AbstractEnv, reset!, observe, act!, terminated
using BasicPOMCP
using CPUTime
using Reel
include("Physical.jl")

include("MDP_types.jl")
include("IDM_MOBIL.jl")
#include("behavior.jl")
include("copula.jl")
include("behavior_gen.jl")
include("no_crash_model.jl")
include("beliefs.jl")
include("uniform_particle_filter.jl")
include("AS_related.jl")
include("heuristics.jl")
include("Despot_related.jl")
include("relax.jl")
include("evaluation.jl")





#test
#include("run_nocrash.jl")
end # module

#=
nb_lanes = 4
pp = PhysicalParam(nb_lanes,lane_length=100.) #2.=>col_length=8
_discount = 1.
nb_cars=10

rmodel = NoCrashRewardModel()

dmodel = NoCrashIDMMOBILModel(nb_cars, pp)

mdp = NoCrashMDP{typeof(rmodel), typeof(dmodel.behaviors)}(dmodel, rmodel, _discount, true);
=#