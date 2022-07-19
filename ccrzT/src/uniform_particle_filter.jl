# Delete car may be a problem
mutable struct MLParticleBelief{T} <: AbstractParticleBelief{T}
    pre_ego_a::MLAction
    physical::Scene #physical state length is car_number
    particles::Vector{T} #length is car_number - 1，since ego car do not need belief  
end  #T is ParticleCollection{DriverModel}

mutable struct BehaviorParticleUpdater <: Updater
    gen::BehaviorGenerator  #in updater
    problem :: NoCrashProblem
    nb_sims::Int
    p_resample_noise::Float64  #用于决定是否添加上一个belief的std到下一个belief
    params::WeightUpdateParams
    policy::Policy
    rng::AbstractRNG
end
function BehaviorParticleUpdater(problem::NoCrashProblem)
    return BehaviorParticleUpdater(
                                    problem.dmodel.behaviors,
                                    problem,
                                    1000,
                                    0.1,
                                    WeightUpdateParams(),
                                    DespotPolicy(problem),
                                    Random.GLOBAL_RNG
                                )
end
function get_gen(up::BehaviorParticleUpdater)
    return up.gen
end
function initialize_belief(up::BehaviorParticleUpdater, phy::Scene) 
    gen = get_gen(up)
    particles = Vector{ParticleCollection{DriverModel}}(undef, phy.n-1)
    for i in 1:phy.n-1
        particle = Vector{DriverModel}(undef, up.nb_sims)
        for j in 1:up.nb_sims
            particle[j] = rand(up.rng, gen)
        end
        particles[i] = ParticleCollection(particle)
    end
    return MLParticleBelief(MLAction(0, 0), phy, particles)
    
end

function mean(b::MLParticleBelief) 
    ms = Vector{Vector{Float64}}(undef, b.physical.n -1)
    for i in 1:length(ms)
        if posgx(b.physical[i+1]) == -250.0  #deleted model
            ms[i] = [0,0,0,0,0,0,0,0]
        else
            ms[i] = [0,0,0,0,0,0,0,0]
            n = 0
            for j in 1:n_particles(b.particles[i])
                if typeof(particle(b.particles[i], j)) <: ASModel
                    ms[i] += particle(b.particles[i], j) #Vector
                    n += 1
                end
            end
            ms[i] = ms[i] / n #vector{Float64}
        end
    end    
    return ms #vector{vector{Float64}}
end
#=
function mean(b::MyParticleBelief) 
    ms = Vector{Vector{Float64}}(undef, b.physical.n -1)
    for i in 1:length(ms)
        if posgx(b.physical[i]) == 0  #deleted model
            ms[i] = [0,0,0,0,0,0,0,0,0]
        else
            ms[i] = sum(particles(b.particles[i]))/n_particles(b.particles[i]) #vector{Float64}
        end
    end    
    return ms #vector{vector{Float64}}
end
=#


function stds(b::MLParticleBelief) 
    means = mean(b) #vector{vector{Float64}}
    stds = Vector{Vector{Float64}}(undef, b.physical.n - 1)
    for i in 1:length(stds)
        if posgx(b.physical[i+1]) == -250.0 #deleted model
            stds[i] = [0,0,0,0,0,0,0,0]
        else
            stds[i] = [0,0,0,0,0,0,0,0]
            n = 0
            for j in 1:n_particles(b.particles[i])
                if typeof(particle(b.particles[i], j)) <: ASModel
                    stds[i] += ( particle(b.particles[i], j) - means[i] ).^2 #Vector
                    n += 1
                end
            end 
            stds[i] = sqrt.( stds[i] / n )
        end
    end
    return stds #vector{vector{Float64}}
end
#=
function stds(b::MyParticleBelief) 
    means = mean(b) #vector{vector{Float64}}
    stds = Vector{Vector{Float64}}(undef, b.physical.n - 1)
    for i in 1:length(stds)
        if posgx(b.physical[i]) == 0 #deleted model
            stds[i] = [0,0,0,0,0,0,0,0,0]
        else
            sum_p = zeros(Float64, length(means[i]))
            for j in 1:n_particles(b.particles[i])
                sum_p += ( particles(b.particles[i])[j] - means[i] ).^2
            end
            stds[i] = sqrt.( sum_p / n_particles(b.particles[i]) )
        end
    end
    return stds #vector{vector{Float64}}
end
=#
function predict!(next_particles::Vector{MLState}, problem::POMDP, particles::Vector{MLState}, a, rng)
    for i in 1:length(next_particles)
        s = particles[i]   
        #next_particles[i] = @gen(:sp)(problem, s, a, rng)
#重写一遍trasition，因为trasition默认使用1.5s，而belief update应使用0.1s
#rewrite trasition , because in belief update should use a shorter time step 0.1s , instead of 1.5s 
        models = s.inter_state
        #models[1] = EgoModel(a)
        phy = problem.dmodel.phys_param
        vel_sigma = phy.vel_sigma
        timestep = phy.dt
        scene = s.phy_state
        next_scene = Scene(Entity{VehicleState, VehicleDef, Int64}, length(scene))  #   next_scene = Scene(Entity, length(scene))
        for (i, veh) in enumerate(scene)
            if i == 1
            else
                #set_std!(models[veh.id], vel_sigma, timestep)
                observe!(models[veh.id], scene, get_road(problem), veh.id)
                a = rand(models[veh.id])
            end
            veh_state_p  = propagate(veh, a, get_road(problem), timestep)

            push!(next_scene, Entity(veh_state_p, veh.def, veh.id))
        end
        MLcollision_check!(problem, scene, next_scene, models, timestep)
        next_particles[i] = MLState(next_scene, models)
    end
    
    
end

function rearrange(up::BehaviorParticleUpdater, b::MLParticleBelief)
    particles = Vector{MLState}(undef, up.nb_sims)
    #models = Dict{Int, DriverModel}
    for i in 1:up.nb_sims
        models = Dict{Int, DriverModel}()
        push!(models, 1 => EgoModel(b.pre_ego_a))
        for j in 1:b.physical.n-1
            push!(models, j+1 => particle(b.particles[j], i))    
        end  
        particles[i] = MLState(b.physical, models)
    end
    return particles
end
#=
function rearrange(up::BehaviorParticleUpdater, b::MyParticleBelief)
    particles = Vector{MLState}(undef, up.nb_sims)
    #models = Dict{Int, DriverModel}
    for i in 1:up.nb_sims
        models = Dict{Int, DriverModel}()
        push!(models, 1 => EgoModel(b.pre_ego_a))
        for j in 1:b.physical.n-1
            push!(models, j+1 => pa_to_Model(b.particles[j].particles[i], get_phy(up.problem)))
        end  
        particles[i] = MLState(b.physical, models)
    end
    return particles
end
=#
function reweight(particles::Vector{MLState}, o::Scene, p::WeightUpdateParams)
    #o = o.phy_state
    weights = Vector{Vector{Float64}}(undef, o.n - 1)
    sim_n = length(particles)
    for i in 1:o.n - 1
        weights[i] = Vector{Float64}(undef, sim_n)
        if posgx(o[i+1]) == -250.0 #Delete car   
            weights[i] = repeat([1.0/sim_n], sim_n)
        elseif typeof(particles[1].inter_state[i+1]) <:DeleteModel #generate new car
            weights[i] = repeat([1.0/sim_n], sim_n)
        else
            for j in 1:sim_n
                weights[i][j] = get_weight(o[i+1].state.v, particles[j].phy_state[i+1].state.v, p.std, posgy(o[i+1]), posgy(particles[j].phy_state[i+1]), p.wrong_lane_factor)
            end
        end
    end
    return weights
end
#=
function reweight(particles::Vector{MLState}, o::Scene, p::WeightUpdateParams)
    #o = o.phy_state
    weights = Vector{Vector{Float64}}(undef, o.n - 1)
    sim_n = length(particles)
    for i in 1:o.n - 1
        weights[i] = Vector{Float64}(undef, sim_n)
    end
    
    for i in 1:sim_n
        #weights[i] = Vector{Float64}(undef, sim_n)
        
        for j in 1:o.n - 1
            if posgx(o[j+1]) == 0 #Delete car   并不需要
                weights[j][i] = 1.0 / sim_n
            elseif posgx(particles[i].phy_state[j+1]) == 0 && posgx(o[j+1]) != 0 #generate new car
                weights[j][i] = 1.0 / sim_n
            else
                weights[j][i] = get_weight(o[j+1].state.v, particles[i].phy_state[j+1].state.v, p.std, posgy(o[j+1]), posgy(particles[i].phy_state[j+1]), p.wrong_lane_factor)
            end
        end
    end
    return weights
end
=#

function get_weight(o_v::Float64, b_v::Float64, std::Float64, o_y::Float64, b_y::Float64, penalize::Float64)
    proportional_likelihood = exp(-(o_v - b_v)^2/(2*std^2))
    if abs(o_y - b_y) < 0.001
        return proportional_likelihood
    else
        return penalize*proportional_likelihood
    end
end

function resample_(b::MLParticleBelief, weights::Vector{Vector{Float64}}, up::BehaviorParticleUpdater, o::Scene, a::MLAction)
    #o = o.phy_state
    #pre_ego_y = posgy(b.physical[1])
    gen = up.gen
    #min_std = 0.001 * (gen.max-gen.min) 
    #stds = max.(std(b), min_std)
    std = stds(b)
    rng = up.rng
    n = up.nb_sims
    n_p = Vector{ParticleCollection}(undef, o.n - 1)
    for i in 1:o.n - 1
        if posgx(o[i+1]) == -250.0 #Delete car
            n_p[i] = ParticleCollection(repeat([DeleteModel()],n))
        elseif posgx(b.physical[i+1]) == -250.0     #generate new car
            particle = Vector{ASModel}(undef, n)
            for j in 1:n
                particle[j] = rand(rng, gen)
            end
            n_p[i] = ParticleCollection(particle)  
        else
            n_p[i] = resample(LowVarianceResampler(n), WeightedParticleBelief(b.particles[i].particles, weights[i], sum(weights[i]), nothing), rng)
            for id in 1:n #add std
                if rand(rng) < up.p_resample_noise
                    n_p[i].particles[id] = clip(n_p[i].particles[id]+std[i].*randn(rng, 8), gen)
                end
            end
        end
    end
    return MLParticleBelief(a, o, n_p)
end
#=
function resample_(b::MyParticleBelief, weights::Vector{Vector{Float64}}, up::BehaviorParticleUpdater, o::Scene, a::MLAction)
    #o = o.phy_state
    #pre_ego_y = posgy(b.physical[1])
    gen = up.gen
    #min_std = 0.001 * (gen.max-gen.min) 
    #stds = max.(std(b), min_std)
    std = stds(b)
    rng = up.rng
    n = up.nb_sims
    n_p = Vector{ParticleCollection}(undef, o.n - 1)
    for i in 1:o.n - 1
        if posgx(o[i+1]) == 0 #Delete car
            n_p[i] = ParticleCollection(repeat([pa_Deletemodel(i+1)],n))
        elseif posgx(b.physical[i+1]) == 0 && posgx(o[i+1]) != 0 #generate new car
            particle = Vector{Idm_MobilParam}(undef, n)
            for j in 1:n
                particle[j] = rand(rng, gen, i+1)
            end
            n_p[i] = ParticleCollection(particle)  
        else
            n_p[i] = resample(LowVarianceResampler(n), WeightedParticleBelief(b.particles[i].particles, weights[i], sum(weights[i]), nothing), rng)#加方差
            for id in 1:n
                if rand(rng) < up.p_resample_noise
                    n_p[i].particles[id] = clip(n_p[i].particles[id]+std[i].*randn(rng, 9), gen)
                end
            end
        end
    end
    return MyParticleBelief(a, o, n_p)
end
=#
function update(up::BehaviorParticleUpdater, b::MLParticleBelief, a::MLAction, o::Scene)
    particles = rearrange(up, b)
    next_particles = Vector{MLState}(undef, up.nb_sims)
    predict!(next_particles, up.problem, particles, a, up.rng)
    weights = reweight(next_particles, o, up.params)
    return resample_(b, weights, up, o, a)
end

#=


mutable struct MyWeightedParticleBelief <: AbstractParticleBelief
    gen::BehaviorGenerator
    physical::Scene #physical state 长度为car_number
    particles::Vector{WeightedParticleBelief{Idm_MobilParam}} #长度为car_number - 1，因为ego car不需要belief  
end

mutable struct BehaviorParticleUpdater <: Updater
    problem :: NoCrashProblem
    nb_sims::Int
    p_resample_noise::Float64
    resample_noise_factor::Float64 
    params::WeightUpdateParams
    rng::AbstractRNG
end

function param_means(b::MyWeightedParticleBelief) #只用到了*
    ms = Vector{Vector{Float64}}(b.physical.n -1)
    for i in 1:length(ms)
        ms[i] = sum(weights(b.particles[i]).*particles(b.particles[i]))/weight_sum(b.particles[i])
        #wts = b.weights(i) 
        #ms[i] = sum(wts.*b.particles[i])/b.weight_sum[i] #vector{Float64}
    end
    return ms #vector{vector{Float64}}
end
function param_stds(b::MyWeightedParticleBelief) #只用到了-
    means = param_means(b) #vector{vector{Float64}}
    stds = Vector{Vector{Float64}}(b.physical.n - 1)
    for i in 1:length(stds)
        stds[i] = sqrt(sum(weights(b.particles[i]).*(particles(b.particles[i]).-means[i]).^2)/weight_sum(b.particles[i]))
        #wts = b.weights(i) 
        #stds[i] = sqrt(sum(wts.*(b.particles[i].-means[i]).^2)/sum(wts))
    end
    return stds #vector{vector{Float64}}
end

"""
Return a vector of states sampled using a carwise version of Thrun's Probabilistic Robotics p. 101
"""
function lv_resample(b::MyWeightedParticleBelief, up::BehaviorParticleUpdater)
    n = up.nb_sims
    rng = up.rng
    gen = b.gen
    min_std = 0.001 * (gen.max-gen.min) #用到了-和* 返回数列
    stds = max.(param_stds(b), min_std)
    ps = Vector{ParticleCollection}(b.physical.n - 1)

    for i in 1:length(ps)
        ps[i] = resample(LowVarianceResampler(n), b.particles[i], rng)
        for id in 1:n
            if rand(up.rng) < up.p_resample_noise
                particle(ps[i], id) = clip(particle(ps[i], id)+stds[i].*randn(rng, 9), gen)
            end
        end
    end
    samples = Vector{MyState}(n)
    phy = up.problem.dmodel.phys_param
    for i in 1:n
        models = Dict{Int, DriverModel}
        push!(models, 1 => EgoModel())
        for j in 1:length(ps)
            push!(models, j+1 => pa_to_Model(particle(ps[j], i),phy))
        end
        samples[i] = MyState(b.physical, models)
    end
    return samples
end

function reweight(particles::MyState, o::Scene, p::WeightUpdateParams)
    weights = Vector{Vector{Float64}}(length(particles))
    sim_n = length(particles)
    for i in 1:sim_n
        weights[i] = Vector{Float64}(length(particles[i]))
        
        for j in 1:length(particles[i])
            if posgx(o.entities[j+1]) == 0 #Delete car  
                weights[i][j] = 1.0 / sim_n
            elseif posgx(particles[i].phy_state.entities[j+1]) == 0 && posgx(o.entities[j+1]) != 0 #generate new car
                weights[i][j] = 1.0 / sim_n
            else
                weights[i][j] = get_weight(o.entities[j+1].v, particles[i].phy_state.entities[j+1].v, p.std, posgy(o.entities[j+1]), posgy(particles[i].phy_state.entities[j+1]), p.wrong_lane_factor)
        end
    end
    return weights
end


function get_weight(o_v::Float64, b_v::Float64, std::Float64, o_y::Float64, b_y::Float64, penalize::Float64)
    proportional_likelihood = exp(-(o_v - b_v)^2/(2*std^2))
    if abs(o_y - b_y) < 0.001
        return proportional_likelihood
    else
        return penalize*proportional_likelihood
    end
end

function g_belief(particles::Vector{MyState}, o::Scene, w, up::BehaviorParticleUpdater, b::MyWeightedParticleBelief)
    n_p = Vector{WeightedParticleBelief}(o.n-1)
    
    for i in 1:o.n-1
        particle = Vector{Idm_MobilParam}(length(w))
        n_w = Vector{Float64}(length(w))
        for j in 1:length(w)
            if posgx(o.entities[i+1]) == 0 #Delete car posgx(o.entities[i+1])
                particle[j] = pa_Deletemodel(i+1)
                n_w[j] = w[j][i]
            elseif posgx(particles[j].phy_state.entities[i+1]) == 0 && posgx(o.entities[i+1]) != 0 #generate new car
                particle[j] = rand(up.rng, b.gen)
                n_w[j] = 1.0 / length(w)
            else
                particle[j] = state_to_pa(particles[j].inter_state[i+1], i+1)
                n_w[j] = w[j][i]
            end
        end
        n_p[i] = WeightedParticleBelief(particle, n_w)
    end
    return n_p
end


function update(up::BehaviorParticleUpdater, b::MyWeightedParticleBelief{T}, a, o::Scene)
    samples = lv_resample(b, up)
    
    #gen = get(up.problem).dmodel.behaviors
    particles = Vector{MyState}(up.nb_sims)
    samples = lv_resample(b_old, up)
    for i in 1:up.nb_sims
        particles[i] = @gen(:sp)(up.problem, samples[i], a, up.rng)
    end

    weights = reweight(particles, o, up.params)
    n_p = g_belief(particles, o::Scene, weights, up, b)
    b_new = MyWeightedParticleBelief(b.gen, o, n_p)
    return b_new
end





function initialize_belief(up::BehaviorParticleUpdater, phy::Scene) 
    gen = up.problem.dmodel.behaviors

#=
    particles = Vector{Idm_MobilParam}(phy.n -1)
    weights = Vector{Vector{Float64}}(s.n -1)
    for i in 1:(s.n -1)
        particles[i] = Vector{Idm_MobilParam}(up.nb_sims)
        weights[i] = Vector{Float64}(up.nb_sims)
        for j in 1:up.nb_sims
            particles[i][j] = rand(up.rng, gen)
            weights[i][j] = 1.0 / up.nb_sims
        end
    end
    return MyWeightedParticleBelief{Idm_MobilParam}(gen, s, particles, weights, ones(s.n -1))
=#


    particles = Vector{WeightedParticleBelief{Idm_MobilParam}}(phy.n-1)
    for i in 1:phy.n-1
        particle = Vector{Idm_MobilParam}(up.nb_sims)
        n_w = Vector{Float64}(up.nb_sims)
        for j in 1:up.nb_sims
            particle[j] = rand(up.rng, gen, i+1)
            n_w[j] = 1.0 / up.nb_sims
        end
        particles[i] = WeightedParticleBelief(particle, n_w)
    end
    return particles
end

=#