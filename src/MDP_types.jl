#abstract type BehaviorModel end
abstract type AbstractMLRewardModel end
abstract type AbstractMLDynamicsModel end

mutable struct MLMDP{S, A, DModel<:AbstractMLDynamicsModel, RModel<:AbstractMLRewardModel}  <: MDP{S, A}
    dmodel::DModel
    rmodel::RModel
    discount::Float64
    throw::Bool
end

mutable struct MLPOMDP{S, A, O, DModel<:AbstractMLDynamicsModel, RModel<:AbstractMLRewardModel}  <: POMDP{S, A, O}
    dmodel::DModel
    rmodel::RModel
    discount::Float64
    throw::Bool
end

mutable struct MLState 
    phy_state::Scene
    inter_state::Dict{Int, DriverModel}
end

struct MLAction
    v_lat::Float64
    a_lon::Float64
end
#=
function Base.ht_keyindex(h::Dict{MLAction,Int}, key::MLAction)
    #=
    if key.a_lon < -1 
        return 10
    else
        =#
    sz = length(h.keys)
    iter = 0
    maxprobe = h.maxprobe
    index = Base.hashindex(key, sz)
    keys = h.keys

    @inbounds while true
        if Base.isslotempty(h,index)
            break
        end
        if !Base.isslotmissing(h,index) && (key === keys[index] || isequal(key,keys[index]))
            return index
        end

        index = (index & (sz-1)) + 1
        iter += 1
        iter > maxprobe && break
    
    end
    return 10
end
=#
function Base.getindex(h::Dict{MLAction,Int}, key::MLAction)
    if key.a_lon < 0 && key.a_lon != -1
        return 10
    else
        index = Base.ht_keyindex(h, key)
        @inbounds return (index < 0) ? throw(KeyError(key)) : h.vals[index]
    end
    #return Base.ht_keyindex(h, key)
end

#=
function Base.haskey(h::Dict{MLAction,Int}, key::MLAction)
    return true
end
function Base.isequal(a::MLAction,b::MLAction)
    if a.a_lon < -1 && b.a_lon < -1
        return true
    else
        return a.v_lat==b.v_lat && a.a_lon==b.a_lon  
    end
end
=#
function Base.:(==)(a::MLAction,b::MLAction)
    
    return a.v_lat==b.v_lat && a.a_lon==b.a_lon  
    
end
    
Base.length(::Type{MLAction}) = 2
Base.convert(::Type{MLAction}, v::Vector{Float64}) = MLAction(v[1], v[2])

function Base.copyto!(v::Vector{Float64}, a::MLAction)
    v[1] = a.v_lat
    v[2] = a.a_lon
    v
end
#=
struct MLObs
    phy_state::Scene
    action::MLAction
end
=#




