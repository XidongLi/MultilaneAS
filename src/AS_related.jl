mutable struct MLLateralDriverModel <: LateralDriverModel{Float64}
    vy::Float64  # lateral velocity
    vy_0::Float64 # given lateral velocity
end

Base.rand(rng::AbstractRNG, model::MLLateralDriverModel)=model.vy
function Base.haskey(odict::Dict{Scene, Int}, o::Scene) #for Despot
    if length(odict) == 0
        return false
    end
    for obs_pair in odict
        if obs_pair[1] == o
            return true
        end
    end
    return false
end
function Base.get(odict::Dict{Scene, Int}, o::Scene, default::Int)  #for Despot
    #o = o.phy_state
    if length(odict) == 0
        return default
    end
    for obs_pair in odict
        if obs_pair[1] == o
            return obs_pair[2]
        end
    end
    return default
end
function Base.:(==)(a::Scene, b::Scene)   #for Despot
    #a = a.phy_state
    if length(a) == length(b)
        for i in 1:length(a)
            if posgy(a[i]) == posgy(b[i])
                continue
            else 
                return false
            end
        end
        return true
    else 
        return false
    end
end




function AutomotiveSimulator.observe!(driver::DespotModel, scene::Scene{Entity{S, D, I}}, roadway::Roadway, egoid::I) where {S, D, I}
    bu = driver.bu
    belief = driver.belief
    policy = driver.bu.policy
    particles = rearrange(bu, belief)
    driver.action = action(policy, particles)
    #=
    println("Despot选择的动作：", driver.action)
    f=open("C:/Users/Mers/Desktop/despot_record.txt","a")  #path
    write(f, "Despot选择的动作：$(driver.action)\n")
    close(f)
    f=open("C:/Users/Mers/Desktop/action_record.txt","a")  #path
    write(f, "$(driver.action)\n")
    close(f)
    =#
    #driver.action = MLAction(2.0, 0)
end

Base.rand(rng::AbstractRNG, driver::DespotModel) = driver.action

struct ReachGoalCallback # a callback that checks if ego car has reach a certain position 
end 
function AutomotiveSimulator.run_callback(
    cb::ReachGoalCallback,
    scenes::Vector{Scene{E}},
    actions::Union{Nothing, Vector{Scene{A}}},
    roadway::R,
    models::Dict{I,M},
    tick::Int,
    ) where {E<:Entity,A<:EntityAction,R,I,M<:DriverModel}
    ego_model = models[1]
    return isterminal(ego_model.pomdp, MLState(scenes[tick], models))   
end

struct MLcollision_check # a callback that checks if a collosion happen 
end 
function AutomotiveSimulator.run_callback( #add a !  ?    
    cb::MLcollision_check,
    scenes::Vector{Scene{E}},
    actions::Union{Nothing, Vector{Scene{A}}},
    roadway::R,
    models::Dict{I,M},
    tick::Int,
    ) where {E<:Entity,A<:EntityAction,R,I,M<:DriverModel}
    println("当前为：", tick)
    ego_model = models[1]
    if tick == 1
    else
        MLcollision_check!(ego_model.pomdp, scenes[tick-1], scenes[tick], models, get_dt(ego_model.pomdp))
        ego_model.cumulative_reward = ego_model.cumulative_reward + reward(ego_model.pomdp, MLState(scenes[tick-1], models), MLAction(0,0), MLState(scenes[tick], models), get_dt(ego_model.pomdp))  
        #println(ego_model.cumulative_reward)
    end
    return false
end
struct BeliefUpdate # for Belief Update
end 
function AutomotiveSimulator.run_callback(  #add a !  ?    #更新state和belief
    cb::BeliefUpdate,
    scenes::Vector{Scene{E}},
    actions::Union{Nothing, Vector{Scene{A}}},
    roadway::R,
    models::Dict{I,M},
    tick::Int,
    ) where {E<:Entity,A<:EntityAction,R,I,M<:DriverModel}
    if tick == 1
    else
        ego_model = models[1]
        a = ego_model.action
        #a = get_by_id(last(actions), 1).action
        #println("最终采取的动作",a)
        #=
        f=open("C:/Users/Mers/Desktop/despot_record.txt","a")  #path
        write(f, "action:$a\n")
        close(f)
        =#
        #@info "最终采取的动作"  a
        #o = @gen(:o)(ego_model.pomdp, MLState(scenes[tick], models), a, Random.GLOBAL_RNG)
        o = observation(ego_model.pomdp, MLState(scenes[tick], models)).self.val
        models[1].belief = update(models[1].bu, models[1].belief, a, o)
    end
    return false
end

struct RecordCallback # a callback that checks if ego car has reach a certain position 
end 
function AutomotiveSimulator.run_callback(
    cb::RecordCallback,
    scenes::Vector{Scene{E}},
    actions::Union{Nothing, Vector{Scene{A}}},
    roadway::R,
    models::Dict{I,M},
    tick::Int,
    ) where {E<:Entity,A<:EntityAction,R,I,M<:DriverModel}
    models[1].bu.policy.step = tick
    return false   
end
"""
Return the x of the Entity(global)
"""
function posgx(vehicle::Entity)
    return vehicle.state.posG.x
end
"""
Return the y of the Entity(global)
"""
function posgy(vehicle::Entity)
    return vehicle.state.posG.y
end
"""
Return the lateral direction of the velocity
"""
function get_lateral_d(model::ASModel)
    return Int(model.mlane.dir)
end
function get_lateral_d(model::DespotModel)
    return Int(sign(model.action.v_lat))
end
function get_lateral_d(model::EgoModel)
    return Int(sign(model.action.v_lat))
end
function get_lateral_d(model::DeleteModel)
    return 0
end

function track_lateral!(model::MLLateralDriverModel, lane_change_action::Int64)
    model.vy = lane_change_action*model.vy_0
    model
end

AutomotiveSimulator.VehicleState(state::VehicleState, roadway::Roadway, y::Float64) = VehicleState(VecSE2(state.posG.x, y, state.posG.θ), roadway, state.v)
AutomotiveSimulator.VehicleState(state::VehicleState, roadway::Roadway, x::Float64, v::Float64) = VehicleState(VecSE2(x, state.posG.y, state.posG.θ), roadway, v)
#AutomotiveSimulator.VehicleState(state::VehicleState, roadway::Roadway, v::Float64) = VehicleState(VecSE2(state.posG.x, state.posG.y, state.posG.θ), roadway, v)

#=
function run_callback!(pomdp::NoCrashProblem, scene::Scene, next_scene::Scene, models::Dict{Int, DriverModel}, rng::AbstractRNG = Random.GLOBAL_RNG) 
    MLcollision_check!(pomdp, scene, next_scene, models, get_dt(pomdp))
    return false  #检测是否中断
end

function simulate(
    pomdp::NoCrashProblem,
    policy::Policy,
    state::MLState,
    belief::MyParticleBelief,
    bu::Updater,
    rng::AbstractRNG = Random.GLOBAL_RNG
    ) where {E<:Entity}

    nticks = pomdp.dmodel.phys_param.sim_nb
    scenes = [Scene(Entity{VehicleState, VehicleDef, Int64}, length(state.phy_state)) for i=1:nticks+1]
    scenes[1] = state.phy_state

    
    models = state.inter_state
    
    phy = get_phy(pomdp)

    vel_sigma = phy.vel_sigma

    timestep = phy.dt

    for tick in 1:nticks
        println(tick)
        particles = rearrange(bu, belief)
        
        a = action(policy, particles)
        println(a)
        #a = action(pomdp, state)  #这里用search_time
        models[1] = EgoModel(a)
        #exchange_time!(get_phy(pomdp))
          
        #exchange_time!(get_phy(pomdp))
        for (i, veh) in enumerate(scenes[tick])
            if i == 1
            else
                set_std!(models[veh.id], vel_sigma, timestep)
                observe!(models[veh.id], scenes[tick], get_road(pomdp), veh.id, phy)
                a = rand(rng, models[veh.id])
            end
            veh_state_p  = propagate(pomdp, veh, a, get_road(pomdp), timestep)
    
            push!(scenes[tick + 1], Entity(veh_state_p, veh.def, veh.id))
            
                
        end

        #next_state = transition(pomdp, state, a)
        
        #out = @gen(:sp,:o,:r,:info)(pomdp, state, a, rng)
        
        
        #scenes[tick + 1] = next_state.phy_state
        if run_callback!(pomdp, scenes[tick], scenes[tick + 1], models)
            return scenes[1:(tick+1)]
        end
        state = MLState(scenes[tick + 1], models)

        o = observation(pomdp, state)
        belief = update(bu, belief, a, o)
    end
    return scenes[1:(nticks+1)]

end
=#
function AutomotiveSimulator.observe!(model::MOBIL, scene::Scene{Entity{S, D, I}}, roadway::Roadway, egoid::I) where {S, D, I}

    vehicle_index = findfirst(egoid, scene)
    veh_ego = scene[vehicle_index]
    v = vel(veh_ego.state)
    egostate_M = veh_ego.state

    ego_lane = get_lane(roadway, veh_ego)

    fore_M = find_neighbor(scene, roadway, veh_ego, targetpoint_ego=VehicleTargetPointFront(), targetpoint_neighbor=VehicleTargetPointRear())
    rear_M = find_neighbor(scene, roadway, veh_ego, rear=true, targetpoint_ego=VehicleTargetPointRear(), targetpoint_neighbor=VehicleTargetPointFront())

    # accel if we do not make a lane change
    accel_M_orig = rand(observe!(reset_hidden_state!(model.mlon), scene, roadway, egoid)).a
    model.dir = DIR_MIDDLE

    advantage_threshold = model.advantage_threshold

    if n_lanes_left(roadway, ego_lane) > 0

        #rear_L = find_neighbor(scene, roadway, veh_ego, 
         #                     lane=leftlane(roadway, veh_ego), 
          #                    rear=true,
           #                   targetpoint_ego=VehicleTargetPointRear(), 
            #                  targetpoint_neighbor=VehicleTargetPointFront())
        rear_L = find_neighbor(scene, roadway, veh_ego, 
                              lane=leftlane(roadway, veh_ego), 
                              rear=true,
                              targetpoint_ego=VehicleTargetPointFront(), 
                              targetpoint_neighbor=VehicleTargetPointRear())

        # candidate position after lane change is over
        footpoint = get_footpoint(veh_ego)
        lane = get_lane(roadway, veh_ego) 
        lane_L = roadway[LaneTag(lane.tag.segment, lane.tag.lane + 1)]
        roadproj = proj(footpoint, lane_L, roadway)
        frenet_L = Frenet(RoadIndex(roadproj), roadway)
        egostate_L = VehicleState(frenet_L, roadway, vel(veh_ego.state))

        Δaccel_n = 0.0
        passes_safety_criterion = true
        if rear_L.ind != nothing
            id = scene[rear_L.ind].id
            accel_n_orig = rand(observe!(reset_hidden_state!(model.mlon), scene, roadway, id)).a

            # update ego state in scene
            scene[vehicle_index] = Entity(veh_ego, egostate_L)
            accel_n_test = rand(observe!(reset_hidden_state!(model.mlon), scene, roadway, id)).a

            body = inertial2body(get_rear(scene[vehicle_index]), get_front(scene[rear_L.ind])) # project ego to be relative to target
            s_gap = body.x
            
            # restore ego state
            scene[vehicle_index] = veh_ego
            passes_safety_criterion = accel_n_test ≥ -model.safe_decel && s_gap ≥ 0
            Δaccel_n = accel_n_test - accel_n_orig
        end

        if passes_safety_criterion

            Δaccel_o = 0.0
            if rear_M.ind != nothing
                id = scene[rear_M.ind].id
                accel_o_orig = rand(observe!(reset_hidden_state!(model.mlon), scene, roadway, id)).a
                
                # update ego state in scene
                scene[vehicle_index] = Entity(veh_ego, egostate_L)
                accel_o_test = rand(observe!(reset_hidden_state!(model.mlon), scene, roadway, id)).a
                
                # restore ego state
                scene[vehicle_index] = veh_ego
                Δaccel_o = accel_o_test - accel_o_orig
            end

            # update ego state in scene
            scene[vehicle_index] = Entity(veh_ego, egostate_L)
            accel_M_test = rand(observe!(reset_hidden_state!(model.mlon), scene, roadway, egoid)).a
            # restore ego state
            scene[vehicle_index] = veh_ego

            Δaccel_M = accel_M_test - accel_M_orig

            Δaₜₕ = Δaccel_M + model.politeness*(Δaccel_n + Δaccel_o)

            if Δaₜₕ > advantage_threshold
                model.dir = DIR_LEFT
                advantage_threshold = Δaₜₕ
            end
        end
    end

    if n_lanes_right(roadway, ego_lane) > 0

        #rear_R = find_neighbor(scene, roadway, veh_ego, lane=rightlane(roadway, veh_ego), targetpoint_ego=VehicleTargetPointRear(), targetpoint_neighbor=VehicleTargetPointFront())
        rear_R = find_neighbor(scene, roadway, veh_ego, 
                              lane=rightlane(roadway, veh_ego), 
                              rear=true,
                              targetpoint_ego=VehicleTargetPointFront(), 
                              targetpoint_neighbor=VehicleTargetPointRear())
        # candidate position after lane change is over
        footpoint = get_footpoint(veh_ego)
        lane = roadway[veh_ego.state.posF.roadind.tag]
        lane_R = roadway[LaneTag(lane.tag.segment, lane.tag.lane - 1)]
        roadproj = proj(footpoint, lane_R, roadway)
        frenet_R = Frenet(RoadIndex(roadproj), roadway)
        egostate_R = VehicleState(frenet_R, roadway, vel(veh_ego.state))

        Δaccel_n = 0.0
        passes_safety_criterion = true
        if rear_R.ind != nothing
            id = scene[rear_R.ind].id
            accel_n_orig = rand(observe!(reset_hidden_state!(model.mlon), scene, roadway, id)).a

            # update ego vehicle in scene
            scene[vehicle_index] = Entity(veh_ego, egostate_R)
            accel_n_test = rand(observe!(reset_hidden_state!(model.mlon), scene, roadway, id)).a

            body = inertial2body(get_rear(scene[vehicle_index]), get_front(scene[rear_R.ind])) # project ego to be relative to target
            s_gap = body.x

            # restore ego vehicle state
            scene[vehicle_index] = veh_ego

            passes_safety_criterion = accel_n_test ≥ -model.safe_decel && s_gap ≥ 0
            Δaccel_n = accel_n_test - accel_n_orig
        end

        if passes_safety_criterion

            Δaccel_o = 0.0
            if rear_M.ind != nothing
                id = scene[rear_M.ind].id
                accel_o_orig = rand(observe!(reset_hidden_state!(model.mlon), scene, roadway, id)).a

                # update ego vehicle in scene
                scene[vehicle_index] = Entity(veh_ego, egostate_R)
                accel_o_test = rand(observe!(reset_hidden_state!(model.mlon), scene, roadway, id)).a
                # restore ego vehicle state
                scene[vehicle_index] = veh_ego

                Δaccel_o = accel_o_test - accel_o_orig
            end

            # update ego vehicle in scene
            scene[vehicle_index] = Entity(veh_ego, egostate_R)
            accel_M_test = rand(observe!(reset_hidden_state!(model.mlon), scene, roadway, egoid)).a
            # restor ego vehicle state
            scene[vehicle_index] = veh_ego
            
            Δaccel_M = accel_M_test - accel_M_orig

            Δaₜₕ = Δaccel_M + model.politeness*(Δaccel_n + Δaccel_o)
            if Δaₜₕ > advantage_threshold
                model.dir = DIR_RIGHT
                advantage_threshold = Δaₜₕ
            elseif Δaₜₕ == advantage_threshold
                # in case of tie, if we are accelerating we go left, else, right
                if Δaccel_M > model.politeness*(Δaccel_n + Δaccel_o)
                    model.dir = DIR_LEFT
                else
                    model.dir = DIR_RIGHT
                end
            end
        end
    end

    model
end
#AutomotiveSimulator.targetpoint_delta(::VehicleTargetPointFront, veh::Entity{S, D, I}) where {S,D<:AbstractAgentDefinition, I} = 0.0
#AutomotiveSimulator.targetpoint_delta(::VehicleTargetPointRear, veh::Entity{S, D, I}) where {S,D<:AbstractAgentDefinition, I} = 0.0