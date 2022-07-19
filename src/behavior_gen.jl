abstract type BehaviorGenerator end

function T_min(generator::BehaviorGenerator)
    return generator.min.mlon.T
end
function T_max(generator::BehaviorGenerator)
    return generator.max.mlon.T
end

function v_des_min(generator::BehaviorGenerator)
    return generator.min.mlon.v_des
end
function v_des_max(generator::BehaviorGenerator)
    return generator.max.mlon.v_des
end

function s_min_min(generator::BehaviorGenerator)
    return generator.min.mlon.s_min
end
function s_min_max(generator::BehaviorGenerator)
    return generator.max.mlon.s_min
end

function a_max_min(generator::BehaviorGenerator)
    return generator.min.mlon.a_max
end
function a_max_max(generator::BehaviorGenerator)
    return generator.max.mlon.a_max
end

function d_cmf_min(generator::BehaviorGenerator)
    return generator.min.mlon.d_cmf
end
function d_cmf_max(generator::BehaviorGenerator)
    return generator.max.mlon.d_cmf
end

function safe_decel_min(generator::BehaviorGenerator)
    return generator.min.mlane.safe_decel
end
function safe_decel_max(generator::BehaviorGenerator)
    return generator.max.mlane.safe_decel
end

function politeness_min(generator::BehaviorGenerator)
    return generator.min.mlane.politeness
end
function politeness_max(generator::BehaviorGenerator)
    return generator.max.mlane.politeness
end

function advantage_threshold_min(generator::BehaviorGenerator)
    return generator.min.mlane.advantage_threshold
end
function advantage_threshold_max(generator::BehaviorGenerator)
    return generator.max.mlane.advantage_threshold
end

function get_σ(generator::BehaviorGenerator)
    return generator.min.mlon.σ
end
function get_δ(generator::BehaviorGenerator)
    return generator.min.mlon.δ
end

mutable struct UniformIDMMOBIL <: BehaviorGenerator
    min::ASModel
    max::ASModel
end
function Base.rand(rng::AbstractRNG, g::UniformIDMMOBIL) 
    return ASModel(g.min, 
                    [T_min(g) + rand(rng)*(T_max(g) - T_min(g)),
                    v_des_min(g) + rand(rng)*(v_des_max(g) - v_des_min(g)),
                    s_min_min(g) + rand(rng)*(s_min_max(g) - s_min_min(g)), 
                    a_max_min(g) + rand(rng)*(a_max_max(g) - a_max_min(g)),    
                    d_cmf_min(g) + rand(rng)*(d_cmf_max(g) - d_cmf_min(g)), 
                    safe_decel_min(g) + rand(rng)*(safe_decel_max(g)   - safe_decel_min(g)),   
                    politeness_min(g) + rand(rng)*(politeness_max(g) - politeness_min(g)),    
                    advantage_threshold_min(g) + rand(rng)*(advantage_threshold_max(g) - advantage_threshold_min(g))]
                    )
end
#=
function Base.rand(rng::AbstractRNG, g::UniformIDMMOBIL, id::Int) 
    return Idm_MobilParam(
        g.min.a_max + rand(rng)*(g.max.a_max - g.min.a_max),
        g.min.d_cmf + rand(rng)*(g.max.d_cmf - g.min.d_cmf),
        g.min.T + rand(rng)*(g.max.T - g.min.T),
        g.min.v_des + rand(rng)*(g.max.v_des - g.min.v_des),
        g.min.s_min + rand(rng)*(g.max.s_min - g.min.s_min),
        4.0,
        g.min.politeness + rand(rng)*(g.max.politeness - g.min.politeness),
        g.min.safe_decel + rand(rng)*(g.max.safe_decel - g.min.safe_decel),
        g.min.advantage_threshold + rand(rng)*(g.max.advantage_threshold - g.min.advantage_threshold),
        id
    )
end
=#
function standard_uniform(phy::PhysicalParam; factor=1.0, correlation::Union{Bool,Float64}=false)
    σ = phy.vel_sigma
    ma = 1.4;    da = 0.6
    mb = 2.0;    db = 1.0
    mT = 1.5;    dT = 0.5
    mv0 = 33.35;  dv0 = 5.55  #dv0 = 5.63
    ms0 = 2.0;   ds0 = 2.0
    del = 4.0
    mp = 0.5;    dp = 0.5
    mbsafe = 2.0;dbsafe = 1.0
    mathr = 0.1; dathr = 0.1
    idm = IntelligentDriverModel(
        σ = σ, 
        T = mT + factor*dT,
        v_des = mv0 + factor*dv0,
        s_min = ms0 + factor*ds0,
        a_max = ma + factor*da,
        d_cmf = mb + factor*db
        )

    mobil = MOBIL(
        mlon = idm, 
        safe_decel = mbsafe + factor*dbsafe,
        politeness = mp + factor*dp,
        advantage_threshold = mathr + factor*dathr
        )
    max = ASModel(idm, MLLateralDriverModel(0.0,phy.vy_0), mobil)

    idm = IntelligentDriverModel(
        σ = σ, 
        T = mT - factor*dT,
        v_des = mv0 - factor*dv0,
        s_min = ms0 - factor*ds0,
        a_max = ma - factor*da,
        d_cmf = mb - factor*db
        )

    mobil = MOBIL(
        mlon = idm, 
        safe_decel = mbsafe - factor*dbsafe,
        politeness = mp - factor*dp,
        advantage_threshold = mathr - factor*dathr
        )
    min = ASModel(idm, MLLateralDriverModel(0.0,phy.vy_0), mobil)
    
    if correlation == true
        return CorrelatedIDMMOBIL(min, max)
    elseif correlation == false
        return UniformIDMMOBIL(min, max)
    else
        return CopulaIDMMOBIL(min, max, correlation)
    end
end
#=
function standard_uniform(factor=1.0; correlation::Union{Bool,Float64}=false)
    ma = 1.4;    da = 0.6
    mb = 2.0;    db = 1.0
    mT = 1.5;    dT = 0.5
    mv0 = 33.35;  dv0 = 5.55  #dv0 = 5.63
    ms0 = 2.0;   ds0 = 2.0
    del = 4.0
    mp = 0.5;    dp = 0.5
    mbsafe = 2.0;dbsafe = 1.0
    mathr = 0.1; dathr = 0.1
    max = Idm_MobilParam(
        ma + factor*da,
        mb + factor*db,
        mT + factor*dT,
        mv0 + factor*dv0,
        ms0 + factor*ds0,
        del,
        mp + factor*dp,
        mbsafe + factor*dbsafe,
        mathr + factor*dathr
        )
    min = Idm_MobilParam(
        ma - factor*da,
        mb - factor*db,
        mT - factor*dT,
        mv0 - factor*dv0,
        ms0 - factor*ds0,
        del,
        mp - factor*dp,
        mbsafe - factor*dbsafe,
        mathr - factor*dathr
        )
    
    if correlation == true
        return CorrelatedIDMMOBIL(min, max)
    elseif correlation == false
        return UniformIDMMOBIL(min, max)
    else
        return CopulaIDMMOBIL(min, max, correlation)
    end
end
=#



mutable struct CorrelatedIDMMOBIL <: BehaviorGenerator 
    min::ASModel
    max::ASModel
end
#=
mutable struct CorrelatedIDMMOBIL <: BehaviorGenerator 
    min::Idm_MobilParam
    max::Idm_MobilParam
end
=#
function Base.rand(rng::AbstractRNG, g::CorrelatedIDMMOBIL) 
    agg = rand(rng)  
    return create_model(g, agg)
end

function create_model(g::CorrelatedIDMMOBIL, agg::Float64)
    return ASModel(g.min, 
                    [T_max(g) + agg*(T_min(g) - T_max(g)), # T is lower for more aggressive
                    v_des_min(g) + agg*(v_des_max(g) - v_des_min(g)),
                    s_min_max(g) + agg*(s_min_min(g) - s_min_max(g) ),# s0 is lower for more aggressive
                    a_max_min(g) + agg*(a_max_max(g) - a_max_min(g)),    
                    d_cmf_min(g) + agg*(d_cmf_max(g) - d_cmf_min(g)), 
                    safe_decel_min(g) + agg*(safe_decel_max(g) - safe_decel_min(g)),   
                    politeness_max(g) + agg*(politeness_min(g) - politeness_max(g)), # p is lower for more aggressive
                    advantage_threshold_max(g) + agg*(advantage_threshold_min(g) - advantage_threshold_max(g) )]# a_thr is lower for more aggressive
                    )

end

function aggressiveness(gen::CorrelatedIDMMOBIL, b::ASModel) #根据当前IDM_MOBIL_参数算出aggressive
    return (b.mlon.v_des - v_des_min(gen))/(v_des_max(gen) - v_des_min(gen))
end
#=
function Base.rand(rng::AbstractRNG, g::CorrelatedIDMMOBIL, id::Int) 
    agg = rand(rng)  
    
    return create_model(g, agg, id)
end

function create_model(g::CorrelatedIDMMOBIL, agg::Float64, id::Int)
    
    return Idm_MobilParam(
        g.min.a_max + agg*(g.max.a_max - g.min.a_max),
        g.min.d_cmf + agg*(g.max.d_cmf - g.min.d_cmf),
        g.max.T + agg*(g.min.T - g.max.T), # T is lower for more aggressive
        g.min.v_des + agg*(g.max.v_des - g.min.v_des),
        g.max.s_min + agg*(g.min.s_min - g.max.s_min), # s0 is lower for more aggressive
        4.0, #公式1中的delta，默认为4,不可修改值？
        g.max.politeness + agg*(g.min.politeness - g.max.politeness),# p is lower for more aggressive
        g.min.safe_decel + agg*(g.max.safe_decel - g.min.safe_decel),
        g.max.advantage_threshold + agg*(g.min.advantage_threshold - g.max.advantage_threshold), # a_thr is lower for more aggressive
        id
    )

end

function aggressiveness(gen::CorrelatedIDMMOBIL, b::Idm_MobilParam) #根据当前IDM_MOBIL_参数算出aggressive
    return (b.v_des - gen.min.v_des)/(gen.max.v_des - gen.min.v_des)
end
=#
mutable struct CopulaIDMMOBIL <: BehaviorGenerator
    min::ASModel
    max::ASModel
    copula::GaussianCopula
end

function CopulaIDMMOBIL(min::ASModel,
    max::ASModel,
    cor::Float64)
return CopulaIDMMOBIL(min, max,
      GaussianCopula(8, cor))
end

function Base.rand(rng::AbstractRNG, g::CopulaIDMMOBIL)
    agg = rand(rng, g.copula)
    return create_model(g, agg)
end

function create_model(g::CopulaIDMMOBIL, agg::Vector{Float64})
    @assert length(agg) == 8
    
    return ASModel(g.min, 
                    [T_max(g) + agg[1]*(T_min(g) - T_max(g)), # T is lower for more aggressive
                    v_des_min(g) + agg[2]*(v_des_max(g) - v_des_min(g)),
                    s_min_max(g) + agg[3]*(s_min_min(g) - s_min_max(g) ),# s0 is lower for more aggressive
                    a_max_min(g) + agg[4]*(a_max_max(g) - a_max_min(g)),    
                    d_cmf_min(g) + agg[5]*(d_cmf_max(g) - d_cmf_min(g)), 
                    safe_decel_min(g) + agg[6]*(safe_decel_max(g)   - safe_decel_min(g)),   
                    politeness_max(g) + agg[7]*(politeness_min(g) - politeness_max(g)), # p is lower for more aggressive
                    advantage_threshold_max(g) + agg[8]*(advantage_threshold_min(g) - advantage_threshold_max(g) )]# a_thr is lower for more aggressive
                    )
end
    
function clip(b::ASModel, g::BehaviorGenerator) 
    
    return ASModel(g.min, 
        [clamp(b.mlon.T, T_min(g), T_max(g)),
        clamp(b.mlon.v_des, v_des_min(g), v_des_max(g)),
        clamp(b.mlon.s_min, s_min_min(g), s_min_max(g)),
        clamp(b.mlon.a_max, a_max_min(g), a_max_max(g)),
        clamp(b.mlon.d_cmf, d_cmf_min(g), d_cmf_max(g)),
        clamp(b.mlane.safe_decel, safe_decel_min(g), safe_decel_max(g)),
        clamp(b.mlane.politeness, politeness_min(g), politeness_max(g)),
        clamp(b.mlane.advantage_threshold, advantage_threshold_min(g), advantage_threshold_max(g))]
        )
    
end
max_accel(gen::BehaviorGenerator) = 1.5 * a_max_max(gen)


#=
mutable struct CopulaIDMMOBIL <: BehaviorGenerator
    min::Idm_MobilParam
    max::Idm_MobilParam
    copula::GaussianCopula
end

function CopulaIDMMOBIL(min::Idm_MobilParam,
    max::Idm_MobilParam,
    cor::Float64)
return CopulaIDMMOBIL(min, max,
      GaussianCopula(8, cor))
end

function Base.rand(rng::AbstractRNG, g::CopulaIDMMOBIL, id::Int)
    agg = rand(rng, g.copula)
    return create_model(g, agg, id)
end

function create_model(g::CopulaIDMMOBIL, agg::Vector{Float64}, id::Int)
    @assert length(agg) == 8
    return Idm_MobilParam(
        g.min.a_max + agg[1]*(g.max.a_max - g.min.a_max),
        g.min.d_cmf + agg[2]*(g.max.d_cmf - g.min.d_cmf),
        g.max.T + agg[3]*(g.min.T - g.max.T), # T is lower for more aggressive
        g.min.v_des + agg[4]*(g.max.v_des - g.min.v_des),
        g.max.s_min + agg[5]*(g.min.s_min - g.max.s_min), # s0 is lower for more aggressive
        4.0, #公式1中的delta，默认为4,不可修改值？
        g.max.politeness + agg[6]*(g.min.politeness - g.max.politeness),# p is lower for more aggressive
        g.min.safe_decel + agg[7]*(g.max.safe_decel - g.min.safe_decel),
        g.max.advantage_threshold + agg[8]*(g.min.advantage_threshold - g.max.advantage_threshold), # a_thr is lower for more aggressive
        id
    )
end
    
function clip(b::Idm_MobilParam, gen::BehaviorGenerator) #clip不应用于Correlated
    
    mi = gen.min
    ma = gen.max

    return Idm_MobilParam(
        max(min(b.a_max, ma.a_max), mi.a_max),
        max(min(b.d_cmf, ma.d_cmf), mi.d_cmf),
        max(min(b.T, ma.T), mi.T),
        max(min(b.v_des, ma.v_des), mi.v_des),
        max(min(b.s_min, ma.s_min), mi.s_min),
        max(min(b.del, ma.del), mi.del) , 
        max(min(b.politeness, ma.politeness), mi.politeness),
        max(min(b.safe_decel, ma.safe_decel), mi.safe_decel),
        max(min(b.advantage_threshold, ma.advantage_threshold), mi.advantage_threshold),
        b.id
        )
    
end
max_accel(gen::BehaviorGenerator) = 1.5*gen.max.a_max
=#