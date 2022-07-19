mutable struct NoCrashIDMMOBILModel <: AbstractMLDynamicsModel
    phys_param::PhysicalParam 
    initial_phy::Scene #will be used in bound(LocalApproximationValueIterationSolver)
    #models::Dict{Int, DriverModel}    #models::Dict{Int, Idm_MobilParam} 
    behaviors::BehaviorGenerator
    lane_terminate::Bool # if true, terminate the simulation when the car has reached the desired lane
    lane_change::Bool #if true, allow ego car to do the lane change
    
end



function NoCrashIDMMOBILModel(pp::PhysicalParam, behaviors::BehaviorGenerator,scene::Scene;
    lane_terminate=false,
    lane_change = true
    )       
    return NoCrashIDMMOBILModel(
        pp,
        scene,
        behaviors,
        lane_terminate, # p_appear
        lane_change
    )
end


#=
function NoCrashIDMMOBILModel(old::NoCrashIDMMOBILModel, models::Dict{Int, DriverModel})
    return NoCrashIDMMOBILModel(old.phys_param, models, old.behaviors, old.lane_terminate, old.lane_change)
end
=#
const NoCrashMDP{R<:AbstractMLRewardModel} =  MLMDP{MLState, MLAction, NoCrashIDMMOBILModel, R}

const NoCrashPOMDP{R<:AbstractMLRewardModel} =  MLPOMDP{MLState, MLAction, Scene, NoCrashIDMMOBILModel, R} 

const NoCrashProblem{R<:AbstractMLRewardModel} =  Union{NoCrashMDP{R}, NoCrashPOMDP{R}}
#=
function Initial_models(EgoModel::DespotModel)
    models = Dict{Int, DriverModel}(1 => EgoModel)
    #models = Dict{Int, DriverModel}(1 => ASModel(IDMParam("normal"), MLLateralDriverModel(0.0,2.0), MOBILParam("normal", "normal")))
    for (idx,x) in enumerate(Iterators.product(["normal","normal","normal"],["normal","normal","normal"]))
        push!(models, idx+1 => ASModel(x[1], x[2], get_phy(EgoModel.pomdp)))
        #set_std!(models[idx+1], get_vel_sigma(pomdp), get_dt(pomdp))
    end
    return models
end
=#

function Initial_models(EgoModel::DespotModel)
    models = Dict{Int, DriverModel}(1 => EgoModel)
    #models = Dict{Int, DriverModel}(1 => ASModel(IDMParam("normal"), MLLateralDriverModel(0.0,2.0), MOBILParam("normal", "normal")))
    for (idx,x) in enumerate(Iterators.product(["cautious","normal","aggressive"],["cautious","normal","aggressive"]))
        push!(models, idx+1 => ASModel(x[1], x[2], get_phy(EgoModel.pomdp)))
        #set_std!(models[idx+1], get_vel_sigma(pomdp), get_dt(pomdp))
    end
    return models
end

function Initial_models(EgoModel::DespotModel, models::Dict{Int, DriverModel})
    models[1] = EgoModel
    return models
end

#const NoCrashPOMDP{R<:AbstractMLRewardModel, G} =  MLPOMDP{MLState, LatLonAccel, MLObs, NoCrashIDMMOBILModel{G}, R}
function car_n(pomdp::NoCrashProblem)
    return pomdp.dmodel.phys_param.nb_cars
end
function get_dt(pomdp::NoCrashProblem)
    return pomdp.dmodel.phys_param.dt
end
function get_search_t(pomdp::NoCrashProblem)
    return pomdp.dmodel.phys_param.search_t
end
function get_road(pomdp::NoCrashProblem)
    return pomdp.dmodel.phys_param.road
end
function get_phy(pomdp::NoCrashProblem)
    return pomdp.dmodel.phys_param
end
function get_initial(pomdp::NoCrashProblem)
    return pomdp.dmodel.initial_phy
end
function get_behavior(pomdp::NoCrashProblem)
    return pomdp.dmodel.behaviors
end
function get_vel_sigma(pomdp::NoCrashProblem)
    return pomdp.dmodel.phys_param.vel_sigma
end
function get_target_lane(pomdp::NoCrashProblem)
    return pomdp.rmodel.target_lane
end
function get_initial_phy(pomdp::NoCrashProblem)
    return pomdp.dmodel.initial_phy
end
function get_brake_limit(pomdp::NoCrashProblem)
    return pomdp.dmodel.phys_param.brake_limit
end
#=
function get_model(pomdp::NoCrashProblem)
    return pomdp.dmodel.models
end
=#
function POMDPs.discount(pomdp::NoCrashProblem)
    return pomdp.discount
end
function POMDPs.isterminal(pomdp::NoCrashProblem, s::MLState)
    phy = get_phy(pomdp)  
    lanes = lane_number(phy, posgy(s.phy_state[1]))
    if lanes[1] == lanes[2] == phy.nb_lanes
        return true
    else
        return false
    end
end

function AutomotiveSimulator.propagate(veh::Entity{VehicleState, D, I}, action::MLAction, roadway::Roadway, ΔT::Float64) where {D, I}   #AutomotiveSimulator.
    
    lane_w = lane_width(roadway, veh)
    y = posgy(veh)
    a_lon = action.a_lon
    v_lat = action.v_lat
    
    target = target_lane(lane_w, length(roadway.segments[1].lanes), y, sign(Int(v_lat)))
    #current = occupation_lanes(phy, y, sign(Int(v_lat)))

    v = vel(veh.state)
    ΔT² = ΔT*ΔT
    Δs = v*ΔT + 0.5*a_lon*ΔT²
    Δt = v_lat*ΔT 
    v = v + a_lon*ΔT

    y_new = y+Δt
    

    

    if Int(v_lat) > 0
        if y_new > (target-1)*lane_w || abs(y_new - (target-1)*lane_w) < 0.01
            y_new = (target-1)*lane_w
        end
    elseif Int(v_lat) < 0
        if y_new < (target-1)*lane_w || abs(y_new - (target-1)*lane_w) < 0.01
            y_new = (target-1)*lane_w
        end
    end

    veh = VehicleState(VecSE2(posgx(veh)+Δs, y_new, 0.0), roadway, v)
    return veh
end

function AutomotiveSimulator.propagate(veh::Entity{VehicleState, D, I}, action::DeleteAction, roadway::Roadway, ΔT::Float64) where {D, I}   #AutomotiveSimulator.
    
    return VehicleState(VecSE2(-250.0,0.0,0.0), roadway, 31.0)
end
#const NoCrashProblem{R<:AbstractMLRewardModel,G} =  Union{NoCrashMDP{R,G}, NoCrashPOMDP{R,G}}

#=
function initial_state(mdp::NoCrashMDP, ps::MLPhysicalState, rng::AbstractRNG=Base.GLOBAL_RNG) #与旧包不同，仅用于测试，需进一步修改
    s = MLState(ps, Vector{CarState}(length(ps.cars)))
    s.cars[1] = CarState(ps.cars[1], NORMAL)
    for i in 2:length(s.cars)
        behavior = rand(rng, mdp.dmodel.behaviors)
        s.cars[i] = CarState(ps.cars[i], behavior)
    end
    s.x = 0.0
    s.t = 0.0
    return s
end
 还没有修改
function initial_state(p::NoCrashProblem, rng::AbstractRNG=Base.GLOBAL_RNG)
    @if_debug println("debugging")
    mdp = NoCrashMDP{typeof(p.rmodel), typeof(p.dmodel.behaviors)}(p.dmodel, p.rmodel, p.discount, p.throw) # make sure an MDP
    return relaxed_initial_state(mdp, 200, rng)
end
=#

mutable struct NoCrashRewardModel <: AbstractMLRewardModel
    cost_dangerous_brake::Float64 # POSITIVE NUMBER   lambda
    reward_in_target_lane::Float64 # POSITIVE NUMBER
    brake_penalty_thresh::Float64 # (POSITIVE NUMBER) if the deceleration is greater than this cost_dangerous_brake will be accured
    target_lane::Float64
end
#NoCrashRewardModel() = NoCrashRewardModel(100,10,4,9)
NoCrashRewardModel() = NoCrashRewardModel(0.01,1,2,9)
lambda(rm::NoCrashRewardModel) = rm.cost_dangerous_brake/rm.reward_in_target_lane  #???

function POMDPs.reward(mdp::NoCrashProblem, s::MLState, a::MLAction, sp::MLState)  
    r = 0.0
    if posgy(sp.phy_state[1]) == get_target_lane(mdp)  
        #f=open("C:/Users/Mers/Desktop/despot_record.txt","a")  #path
        #write(f, "reach_reward\n")
        #close(f)
        r += mdp.rmodel.reward_in_target_lane
    end
    nb_brakes = detect_braking(mdp, s, sp)
    r -= mdp.rmodel.cost_dangerous_brake*nb_brakes
    return r
end

function POMDPs.reward(mdp::NoCrashProblem, s::MLState, a::MLAction, sp::MLState, dt::Float64)  
    r = 0.0
    if posgy(sp.phy_state[1]) == get_target_lane(mdp)  
        #f=open("C:/Users/Mers/Desktop/despot_record.txt","a")  #path
        #write(f, "reach_reward\n")
        #close(f)
        r += mdp.rmodel.reward_in_target_lane
    end
    nb_brakes = detect_braking(mdp, s, sp, dt)
    r -= mdp.rmodel.cost_dangerous_brake*nb_brakes
    return r
end

function detect_braking(mdp::NoCrashProblem, s::MLState, sp::MLState, threshold::Float64) 
    nb_brakes = 0
    #nb_leaving = 0 
    dt = get_search_t(mdp)
    for (i,c) in enumerate(s.phy_state) 
        cp = sp.phy_state[i]
        if posgx(c) == -250.0 || posgx(cp) == -250.0 #检测被移除车辆
            continue
        end
        if (cp.state.v-c.state.v)/dt < -threshold
            nb_brakes += 1
        end
    end
    # @assert nb_leaving <= 5 # sanity check - can remove this if it is violated as long as it doesn't happen all the time
    return nb_brakes
end

function detect_braking(mdp::NoCrashProblem, s::MLState, sp::MLState, threshold::Float64, dt::Float64) 
    nb_brakes = 0
    #nb_leaving = 0 
    #dt = get_search_t(mdp)
    for (i,c) in enumerate(s.phy_state) 
        cp = sp.phy_state[i]
        if posgx(c) == -250.0 || posgx(cp) == -250.0 #检测被移除车辆
            continue
        end
        if (cp.state.v-c.state.v)/dt < -threshold
            nb_brakes += 1
        end
    end
    # @assert nb_leaving <= 5 # sanity check - can remove this if it is violated as long as it doesn't happen all the time
    return nb_brakes
end

detect_braking(mdp::NoCrashProblem, s::MLState, sp::MLState) = detect_braking(mdp, s, sp, mdp.rmodel.brake_penalty_thresh) 
#=
function transition(mdp::NoCrashMDP, s::Scene, a::ASAction, rng::AbstractRNG) #应该应用于NoCrashProblem ，需要对ASAction写一个observe和propogate
    for (i, veh) in enumerate(scene)
    observe!(models[veh.id], scenes[tick], roadway, veh.id)
            a = rand(rng, models[veh.id])
        
            veh_state_p  = propagate(veh, a, roadway, timestep)
end
=#
#function generate_s(mdp::NoCrashProblem, s::MLState, a::MLAction, rng::AbstractRNG)
#end

struct NoCrashActionSpace
    NORMAL_ACTIONS::Vector{MLAction} # all the actions except brake
    acceptable::Vector{Int}
    brake::MLAction # this action will be EITHER braking at half the dangerous brake threshold OR the braking necessary to prevent a collision at all time in the future
end

const NB_NORMAL_ACTIONS = 9

function NoCrashActionSpace(mdp::NoCrashProblem) 
    accels = (-mdp.dmodel.phys_param.adjustment_acceleration, 0.0, mdp.dmodel.phys_param.adjustment_acceleration)
    lane_changes = (-mdp.dmodel.phys_param.vy_0, 0.0, mdp.dmodel.phys_param.vy_0)
    NORMAL_ACTIONS = vec([MLAction(a,l) for (a,l) in Iterators.product(lane_changes, accels)]) # this should be in the problem
    return NoCrashActionSpace(NORMAL_ACTIONS, zeros(Int, 0), MLAction(0,0)) # note: brake will be calculated later based on the state
end
function LaneChangeActionSpace(mdp::NoCrashProblem, v_lat::Float64) 
    accels = (-mdp.dmodel.phys_param.adjustment_acceleration, 0.0, mdp.dmodel.phys_param.adjustment_acceleration)
    lane_changes = v_lat
    NORMAL_ACTIONS = vec([MLAction(a,l) for (a,l) in Iterators.product(lane_changes, accels)]) # this should be in the problem
    return NoCrashActionSpace(NORMAL_ACTIONS, zeros(Int, 0), MLAction(0,0)) # note: brake will be calculated later based on the state
end
#=
function POMDPs.actions(mdp::NoCrashProblem, s::MLState) 
    
    ego_model = s.inter_state[1]
    a = ego_model.action
    s = s.phy_state
    acceptable = zeros(Int, 0)
    phy = get_phy(mdp)
    ego_y = posgy(s[1])
    lane_ego = lane_number(phy, ego_y)

    brake_acc = calc_brake_acc(mdp, s)

    if lane_ego[1] != lane_ego[2] && a.v_lat != 0 #ego car is doing the lane change
        as = LaneChangeActionSpace(mdp, a.v_lat) 

        brake = MLAction(a.v_lat, brake_acc)

        action_space = (1:length(as.NORMAL_ACTIONS))
    else 
        as = NoCrashActionSpace(mdp)

        brake = MLAction(0, brake_acc)

        action_space = (1:NB_NORMAL_ACTIONS)
    end

    for i in action_space
        a = as.NORMAL_ACTIONS[i]
            # prevent going off the road
        if (ego_y <0.01 && a.v_lat < 0) || (abs(ego_y - most_left_lane(phy)) <0.01 && a.v_lat > 0.0) #可能有误差问题？
            continue
        end
        # prevent running into the person in front or to the side
        if is_safe(mdp, s, as.NORMAL_ACTIONS[i])
            append!( acceptable, i )
        end
    end
    #brake_acc = calc_brake_acc(mdp, s)
    #brake = MLAction(0, brake_acc)
    return NoCrashActionSpace(as.NORMAL_ACTIONS, acceptable, brake)
end
=#
function POMDPs.actions(pomdp::NoCrashProblem) 
    as = NoCrashActionSpace(pomdp)
    push!( as.NORMAL_ACTIONS, MLAction(0.0,-10.0) )
    return   as.NORMAL_ACTIONS
end
function Base.findfirst(action::MLAction, vector::Vector{MultilaneAS.MLAction}) 
    for (id,a) in enumerate(vector)
        if action == a
            return id
        end
    end
    return 10
end
function POMDPs.actionindex(pomdp::NoCrashProblem, a::MLAction)
    index = findfirst(a, NoCrashActionSpace(pomdp).NORMAL_ACTIONS)
    return index
end
function actionindexes(pomdp::NoCrashProblem, actions::Vector{MLAction})
    indexes = Vector{Int}()
    for a in actions
        push!(indexes, actionindex(pomdp, a))
    end
    return indexes
end
function POMDPs.actions(pomdp::NoCrashProblem, s::MLState) #单纯由phy state决定
    
    ego_model = s.inter_state[1]
    a = ego_model.action
    s = s.phy_state
    phy = get_phy(pomdp)
    ego_y = posgy(s[1])
    lane_ego = lane_number(phy, ego_y)

    brake_acc = calc_brake_acc(pomdp, s)

    if lane_ego[1] != lane_ego[2] && a.v_lat != 0 #ego car is doing the lane change
        as = LaneChangeActionSpace(pomdp, a.v_lat) 
        brake = MLAction(a.v_lat, brake_acc)
        as = as.NORMAL_ACTIONS
    else 
        as = NoCrashActionSpace(pomdp)
        brake = MLAction(0, brake_acc)
        as = as.NORMAL_ACTIONS
    end
    acceptable = zeros(Int, 0)
    for i in 1:length(as)
        a = as[i]
            # prevent going off the road
        if (ego_y <0.01 && a.v_lat < 0) || (abs(ego_y - most_left_lane(phy)) <0.01 && a.v_lat > 0.0) 
            continue
        end
        # prevent running into the person in front or to the side
        if is_safe(pomdp, s, a)
            append!( acceptable, i )
        end
    end
    if length(acceptable) == 0
        return [brake]
    else
        return as[acceptable]
    end
    #brake_acc = calc_brake_acc(mdp, s)
    #brake = MLAction(0, brake_acc)
    #return NoCrashActionSpace(as.NORMAL_ACTIONS, acceptable, brake)
end

function POMDPs.actions(pomdp::UnderlyingMDP, s::MLState) 
    return actions(pomdp.pomdp, s) 
end
function POMDPs.actions(pomdp::NoCrashProblem, s::Vector{MLState}) 
    return actions(pomdp, s[1]) 
end

function POMDPs.actions(pomdp::NoCrashProblem, belief::ScenarioBelief) 
    #if currentobs(belief) != missing
    if isequal(currentobs(belief), missing)  #root
        a = particle(belief, 1).inter_state[1].action#a = currentobs(belief).action    
        #println("上一次的动作:", a)
        #nothing
    else
        #println("是中线吗:",posgy(belief.scenarios[1][2].phy_state[1]))
        a = MLAction(0.0, 0.0) #非root时，一定为中线开始，因为在search时必定会完成一次lane change
    end
    #ego_model = get_model(pomdp)[1]
    #a = ego_model.a
    s = belief.scenarios[1][2].phy_state  #有一定误差，非root时不同particle的x值不同
    
    phy = get_phy(pomdp)
    ego_y = posgy(s[1])
    lane_ego = lane_number(phy, ego_y)

    brake_acc = calc_brake_acc(pomdp, s)

    if lane_ego[1] != lane_ego[2] && a.v_lat != 0 #ego car is doing the lane change
        as = LaneChangeActionSpace(pomdp, a.v_lat) 
        brake = MLAction(a.v_lat, brake_acc)
        as = as.NORMAL_ACTIONS
        #push!(as, brake)
        
    else 
        as = NoCrashActionSpace(pomdp)

        brake = MLAction(0, brake_acc)
        as = as.NORMAL_ACTIONS
        #push!( as, brake )
    end
    acceptable = zeros(Int, 0)
    for i in 1:length(as)
        a = as[i]
            # prevent going off the road
        if (ego_y <0.01 && a.v_lat < 0) || (abs(ego_y - most_left_lane(phy)) <0.01 && a.v_lat > 0.0) 
            continue
        end
        # prevent running into the person in front or to the side
        if is_safe(pomdp, s, a)
            append!( acceptable, i )
        end
    end
    #accels = (-pomdp.dmodel.phys_param.adjustment_acceleration, 0.0, pomdp.dmodel.phys_param.adjustment_acceleration)
    #lane_changes = (-pomdp.dmodel.phys_param.vy_0, 0.0, pomdp.dmodel.phys_param.vy_0)
    #NORMAL_ACTIONS = vec([MLAction(a,l) for (a,l) in Iterators.product(lane_changes, accels)]) # this should be in the problem
    #println("可选动作:", as[acceptable])
    #=
    f=open("C:/Users/Mers/Desktop/despot_record.txt","a")  #path
    write(f, "available_action:$(as[acceptable])\n")
    close(f)
    =#
    if length(acceptable) == 0
        return [brake]
    else
        return as[acceptable]
    end
end

calc_brake_acc(mdp::NoCrashProblem, s::Scene) = clamp(max_safe_acc(mdp,s), -10, -0.01)#min(max_safe_acc(mdp,s), 0.0)#

function POMDPs.action(mdp::NoCrashProblem, s::MLState)
    a_space = actions(mdp::NoCrashProblem, s::MLState)
    if length(a_space.acceptable) == 0 
        return a_space.brake
    else
        #return a_space.NORMAL_ACTIONS[a_space.acceptable[1]]
        return a_space.NORMAL_ACTIONS[rand(a_space.acceptable)]#a_space.NORMAL_ACTIONS[a_space.acceptable[1]]#   #
    end
        
    
end
#actions(mdp::NoCrashProblem, b::AbstractParticleBelief) = actions(mdp, MLPhysicalState(first(particles(b)))) belief相关

"""
Test whether, if the ego vehicle takes action a, it will always be able to slow down fast enough if the car in front slams on his brakes and won't pull in front of another car so close they can't stop
"""
function is_safe(mdp::NoCrashProblem, s::Scene, a::MLAction) 
    
    if a.a_lon >= max_safe_acc(mdp, s, sign(Int(a.v_lat)))
        return false
    end
    # check whether we will go into anyone else's lane so close that they might hit us or we might run into them
    phy = get_phy(mdp)
    #dt = phy.dt
    dt = 0.1
    
    ego_y = posgy(s[1])
    lanes = lane_number(phy, ego_y)
    if lanes[1] == lanes[2] || a.v_lat != 0.0  #当ego car只做减速时，不会进入当前条件，从而发生与后车的碰撞  #lanes[1] == lanes[2] && a.v_lat != 0.0
        l_car = phy.l_car
        ego_x = posgx(s[1])
        for i in 2:phy.nb_cars
            car = s[i].state  
            ego = s[1].state               
            if posgx(s[i]) < ego_x + l_car && occupation_overlap(phy, ego_y, posgy(s[i]), sign(Int(a.v_lat)), 0)  # ego is in front of car
                # New definition of safe - the car behind can brake at max braking to avoid the ego if the ego
                # slams on his brakes
                # XXX IS THIS RIGHT??
                # I think I need a better definition of "safe" here
                gap = ego_x - posgx(s[i]) - l_car
                if gap <= 2.0 #两辆车的速度差距较大时，且dt很小时，RSS model 会允许gap在很小的时候做lane_change（gap=0.16，v前=31.7，v后=29.7，dt=0.1）
                    return false
                end                                           
                n_braking_acc = nullable_max_safe_acc(gap, car.v, ego.v, phy.brake_limit/2, dt)  # 
                #n_braking_acc = max_safe_acc(gap, car.vel, ego.vel, phy.brake_limit, dt)
                #if isnull(n_braking_acc) || get(n_braking_acc) < max_accel(mdp.dmodel.behaviors)
                if length(n_braking_acc) == 0 || n_braking_acc[1] < max_accel(get_behavior(mdp))
                #if n_braking_acc < max_accel(mdp.dmodel.behaviors)
                    return false
                end
            end
        end
    end
    return true
end

"""
Calculate the maximum safe acceleration that will allow the car to avoid a collision if the car in front slams on its brakes
"""
function max_safe_acc(mdp::NoCrashProblem, s::Scene, lane_change::Int=0)
    #dt = mdp.dmodel.phys_param.dt
    dt = 0.1
    l_car = mdp.dmodel.phys_param.l_car
    bp = mdp.dmodel.phys_param.brake_limit / 2
    ego = s[1].state
    nb_cars = s.n

    car_in_front = 0
    smallest_gap = Inf
    # find car immediately in front
    for i in 2:nb_cars#nb_cars
        if posgx(s[i]) >= ego.posG.x    
            if occupation_overlap(mdp.dmodel.phys_param, ego.posG.y, posgy(s[i]), lane_change, 0) # occupying same lane
                gap = posgx(s[i]) - ego.posG.x - l_car
                if gap < smallest_gap 
                    car_in_front = i
                    smallest_gap = gap
                end
            end
        end
    end
        # calculate necessary acceleration
    if car_in_front == 0
            return Inf
        else                                                                   
            n_brake_acc = nullable_max_safe_acc(smallest_gap, ego.v, s[car_in_front].state.v, bp, dt)
            return get(n_brake_acc, 1, -mdp.dmodel.phys_param.brake_limit) 
            #return max_safe_acc(smallest_gap, ego.v, s[car_in_front].state.vel, bp, dt)
            
        end
    
    return Inf
end

"""
Return max_safe_acc or an empty Nullable if the discriminant is negative.
"""
function nullable_max_safe_acc(gap, v_behind, v_ahead, braking_limit, dt)
    bp = braking_limit
    v = v_behind
    vo = v_ahead
    g = gap
    # VVV see mathematica notebook
    discriminant = 8*g*bp + bp^2*dt^2 - 4*bp*dt*v + 4*vo^2
    result = Vector{Union{Float64, Nothing}}()
    if discriminant >= 0.0
        return append!(result , - (bp*dt + 2*v - sqrt(discriminant)) / (2*dt))
        #return Vector{Union{Float64, Nothing}}(- (bp*dt + 2*v - sqrt(discriminant)) / (2*dt))
        #return Nullable{Float64}(- (bp*dt + 2.*v - sqrt(discriminant)) / (2.*dt))
    else
        return result
        #return Nullable{Float64}()
    end
end
#=
"""
Return the maximum acceleration that the car behind can have on this step so that it won't hit the car in front if it slams on its brakes
"""
function max_safe_acc(gap, v_behind, v_ahead, braking_limit, dt)
    return 2*(v_ahead - v_behind)/dt - braking_limit - 2*gap/dt^2
end
=#
function MLcollision_check!(pomdp::NoCrashProblem, scene::Scene, next_scene::Scene, models::Dict{Int, DriverModel}, dt::Float64, rng::AbstractRNG = Random.GLOBAL_RNG)
    phy = pomdp.dmodel.phys_param
    vel_sigma = phy.vel_sigma
    gen = pomdp.dmodel.behaviors
    roadway = phy.road
    lane_width = phy.w_lane
    nb_lane = phy.nb_lanes
    #collision check

    # first prevent lane changes into each other
    changers = zeros(Int, 0)
    
    for number = 1:length(scene)       #将做lanechange的车序号放入changers
        #println(number)
        #println(models[number].mlane.dir)
        if posgx(next_scene[number]) != -250.0 #deleted car
            if get_lateral_d(models[number]) != 0
                append!( changers, number )
            end
        end
    end 
    
    if next_scene[1].state.v <= 1         
        next_scene[1] = Entity(next_scene[1], VehicleState(next_scene[1].state, roadway, posgx(next_scene[1]), 1.0))   
    end
    
    sorted_changers = sort!(changers, by=i->posgx(scene[i]), rev=true) # this might be slow because anonymous functions are slow #根据x值排序，从大到小
    # from front to back   #出现了两个已经开始变道的车辆发生了碰撞的情况 以及两个已经开始变道的车辆中后车由于与前车太近停止进行位移直到前车到位
    if length(sorted_changers) >= 2
        for i in 1:length(sorted_changers)-1
            if target_lane(phy, posgy(scene[sorted_changers[i]]), get_lateral_d(models[sorted_changers[i]])) == target_lane(phy, posgy(scene[sorted_changers[i+1]]), get_lateral_d(models[sorted_changers[i+1]]))
                                                                                                      
                x_fore = posgx(next_scene[sorted_changers[i]])
                x_rear = posgx(next_scene[sorted_changers[i+1]])
                g_1 = idm_gstar(models[sorted_changers[i+1]], next_scene[sorted_changers[i+1]].state.v, next_scene[sorted_changers[i]].state.v - next_scene[sorted_changers[i+1]].state.v)
                if x_fore-x_rear < g_1
                    lane_fore = lane_number(phy, posgy(scene[sorted_changers[i]]))
                    lane_rear = lane_number(phy, posgy(scene[sorted_changers[i+1]]))
                    if lane_fore[1] == lane_fore[2] && lane_rear[1] != lane_rear[2] #前车开始位移，后车已经在位移中
                        #目前为前车停止位移 ， 应修改为后车停止位移并回到原位，前车继续位移。 但会导致后车的碰撞检测变得复杂？
                        next_scene[sorted_changers[i]] = Entity(next_scene[sorted_changers[i]], VehicleState(next_scene[sorted_changers[i]].state, roadway, posgy(scene[sorted_changers[i]])))
                        if sorted_changers[i] == 1 #ego car 
                            models[1].action = MLAction(0, models[1].action.a_lon)
                        else
                            models[sorted_changers[i]].mlat.vy = 0.0
                            models[sorted_changers[i]].mlane.dir = 0
                        end
                    elseif (lane_fore[1] != lane_fore[2] && lane_rear[1] == lane_rear[2]) || (lane_fore[1] == lane_fore[2] && lane_rear[1] == lane_rear[2]) #后车开始位移
                        #后车停止位移
                        next_scene[sorted_changers[i+1]] = Entity(next_scene[sorted_changers[i+1]], VehicleState(next_scene[sorted_changers[i+1]].state, roadway, posgy(scene[sorted_changers[i+1]])))   
                        if sorted_changers[i+1] == 1 #ego car 
                            
                            models[1].action = MLAction(0, models[1].action.a_lon)
                        else 
                            models[sorted_changers[i+1]].mlat.vy = 0.0
                            models[sorted_changers[i+1]].mlane.dir = 0
                        end
                        
                    else #前车，后车均已开始位移
                        #后车停止位移并回位    
                        #next_scene[sorted_changers[i+1]] = Entity(next_scene[sorted_changers[i+1]], VehicleState(next_scene[sorted_changers[i+1]].state, roadway, posgy(scene[sorted_changers[i+1]])))   
                        if sorted_changers[i+1] == 1 #ego car 
                            models[1].action = MLAction(-models[1].action.v_lat, models[1].action.a_lon)
                            next_scene[1] = Entity(next_scene[1], propagate(scene[1], models[1].action, roadway, dt)) 
                            
                        else 
                            models[sorted_changers[i+1]].mlat.vy = -models[sorted_changers[i+1]].mlat.vy
                            models[sorted_changers[i+1]].mlane.dir = -models[sorted_changers[i+1]].mlane.dir
                            #next_scene[sorted_changers[i+1]] = propagate(scene[sorted_changers[i+1]], MLAction(models[sorted_changers[i+1]].mlat.vy, models[sorted_changers[i+1]].mlon.a), roadway, dt)
                            next_scene[sorted_changers[i+1]] = Entity(next_scene[sorted_changers[i+1]], propagate(scene[sorted_changers[i+1]], MLAction(models[sorted_changers[i+1]].mlat.vy, models[sorted_changers[i+1]].mlon.a), roadway, dt))
                        end   
                       
                    end
                end
            end
        end
    end

    # second, prevent cars hitting each other due to noise
#=
    sorted = sortperm(collect(posgx(c) for c in scene), rev=true) #对x值排序，从大到小，返回序号
    if length(sorted) >= 2 #something to compare
        for i in 1:length(sorted)-1
            if posgx(next_scene[sorted[i+1]]) !=0 && posgx(next_scene[sorted[i]]) != 0
                if posgx(next_scene[sorted[i+1]]) > abs(posgx(next_scene[sorted[i]]) - phy.l_car)    #abs() aviod deletemodel #同一车道上后车超越前车，发生碰撞    
                    #if occupation_overlap(phy, posgy(scene[sorted[i]]), posgy(scene[sorted[i+1]]), get_lateral_d(models[sorted[i]]), get_lateral_d(models[sorted[i+1]]))
                    if abs(posgy(scene[sorted[i]]) - posgy(scene[sorted[i+1]])) < lane_width
                        println("longitudinal collision")
                        dx = posgx(next_scene[sorted[i]]) - posgx(scene[sorted[i+1]]) - 1.01*phy.l_car
                        new_x = posgx(next_scene[sorted[i]]) - 1.01*phy.l_car  
                        new_a = 2.0*(dx - scene[sorted[i+1]].state.v*dt)/dt^2
                        new_v = scene[sorted[i+1]].state.v + new_a*dt
                        next_scene[sorted[i+1]] = Entity(next_scene[sorted[i+1]], VehicleState(next_scene[sorted[i+1]].state, roadway, new_x, new_v))
                
                    end
                end
            end
        end
    end
     =#                       
                            
                            
                
                        

    ## Dynamics and Exits ##
    #Delete old car 
    zero_car = zeros(Int, 0)
    other_car = zeros(Int, 0)
        
    for veh in next_scene
        if posgx(veh) == -250.0
            append!( zero_car, veh.id )
        else
            if abs( posgx(veh) - posgx(next_scene[1]) ) > 100  
                next_scene[veh.id] = Entity(VehicleState(VecSE2(-250.0,0.0,0.0), roadway, 31.0), VehicleDef(), veh.id)
                models[veh.id] = DeleteModel()
                append!( zero_car, veh.id )
            else
               append!( other_car, veh.id )
            end
        end
    end

    #generate new car 
    if length(zero_car) != 0 # && rand(rng) <= mdp.dmodel.p_appear  #appear在论文中并没有提到
        #models[zero_car[1]] = pa_to_Model(rand(rng, gen, zero_car[1]), phy)
        #set_std!(models[zero_car[1]], vel_sigma, dt)
        models[zero_car[1]] = rand(rng, gen)
        vel = models[zero_car[1]].mlon.v_des + randn(rng)*get_vel_sigma(pomdp)   #速度需要一个起始误差

        clearances = zeros(Float64, nb_lane) 
        
        fill!(clearances, Inf)
        closest_cars = zeros(Int, nb_lane)
        
        fill!(closest_cars, 0)
        sstar_margins = zeros(Float64, nb_lane)
        if vel > next_scene[1].state.v
            # put at back
            for other_car_id in other_car  #找出每条路中x值最小的车  
                lowlane, highlane = occupation_lanes(phy, posgy(next_scene[other_car_id]), 0)
                back = posgx(next_scene[other_car_id]) - posgx(next_scene[1]) +50  
                if back < clearances[lowlane]
                    clearances[lowlane] = back
                    closest_cars[lowlane] = other_car_id
                end
                if back < clearances[highlane]
                    clearances[highlane] = back
                    closest_cars[highlane] = other_car_id
                end
            end
            for j in 1:nb_lane
                other = closest_cars[j]
                if other == 0
                    sstar = 0.0
                else
                    sstar=idm_gstar(models[zero_car[1]], vel, vel - next_scene[other].state.v)
                end
                sstar_margins[j] = clearances[j] - sstar #clearance to the nearest car
            end
        else
            for other_car_id in other_car 
                lowlane, highlane = occupation_lanes(phy, posgy(next_scene[other_car_id]), 0)
                front = posgx(next_scene[1]) +50 - posgx(next_scene[other_car_id])   
                if front < clearances[lowlane]
                    clearances[lowlane] = front
                    closest_cars[lowlane] = other_car_id
                end
                if front < clearances[highlane]
                    clearances[highlane] = front
                    closest_cars[highlane] = other_car_id
                end
            end
            for j in 1:nb_lane
                other = closest_cars[j]
                if other == 0
                    sstar = 0
                else
                    sstar=idm_gstar(models[other], next_scene[other].state.v, next_scene[other].state.v - vel)
                end
                sstar_margins[j] = clearances[j] - sstar
            end
        end
        margin, lane = findmax(sstar_margins)
        
        if margin > 0.0
            if vel > next_scene[1].state.v
                # at back                               
                next_scene[zero_car[1]] = Entity(VehicleState(VecSE2(posgx(next_scene[1]) - 50, lane_width*(lane-1), 0.0), roadway, vel), VehicleDef(), zero_car[1])
            else
                next_scene[zero_car[1]] = Entity(VehicleState(VecSE2(posgx(next_scene[1]) + 50, lane_width*(lane-1), 0.0), roadway, vel), VehicleDef(), zero_car[1])
            end
        else
            models[zero_car[1]] = DeleteModel()#不产生新车
        end
    end
    #return MLState(next_scene, models)
end

function POMDPs.transition(pomdp::NoCrashProblem, s::MLState, a::MLAction) 
    
    models = s.inter_state
    #models = deepcopy(s.inter_state) #deep copy , to avoid change scenario_belief during calculate lower bound(if use default action)
    models[1] = EgoModel(a)
    phy = pomdp.dmodel.phys_param
    
    vel_sigma = phy.vel_sigma
    
    timestep = phy.search_t
    scene = s.phy_state
    next_scene = Scene(Entity{VehicleState, VehicleDef, Int64}, length(scene))  #   next_scene = Scene(Entity, length(scene))
    for (i, veh) in enumerate(scene)
        if i == 1
        else
            #set_std!(models[veh.id], vel_sigma, timestep)
            observe!(models[veh.id], scene, get_road(pomdp), veh.id)
            a = rand(models[veh.id])
        end
        veh_state_p  = propagate(veh, a, get_road(pomdp), timestep)

        push!(next_scene, Entity(veh_state_p, veh.def, veh.id))
    end
    MLcollision_check!(pomdp, scene, next_scene, models, timestep)
    #return Deterministic(MLState(next_scene, models))
    return Random.SamplerTrivial(Deterministic(MLState(next_scene, models)))
    #return MLState(next_scene, models)


#=
    #collision check
    # first prevent lane changes into each other
    changers = zeros(Int, 0)
    
    for number = 1:length(scene)       #将做lanechange的车序号放入changers
        #println(number)
        #println(models[number].mlane.dir)
        if posgx(next_scene[number]) != 0
            if get_lateral_d(models[number]) != 0
                append!( changers, number )
            end
        end
    end
    sorted_changers = sort!(changers, by=i->posgx(scene[i]), rev=true) # this might be slow because anonymous functions are slow #根据x值排序，从大到小
    # from front to back
    if length(sorted_changers) >= 2
        for i in 1:length(sorted_changers)-1
            if target_lane(phy, posgy(scene[sorted_changers[i]]), get_lateral_d(models[sorted_changers[i]])) == target_lane(phy, posgy(scene[sorted_changers[i+1]]), get_lateral_d(models[sorted_changers[i+1]]))
                                                                                                      
                x_fore = posgx(next_scene[sorted_changers[i]])
                x_rear = posgx(next_scene[sorted_changers[i+1]])
                g_1 = idm_gstar(models[sorted_changers[i+1]], next_scene[sorted_changers[i+1]].state.v, next_scene[sorted_changers[i]].state.v - next_scene[sorted_changers[i+1]].state.v)
                if x_fore-x_rear < g_1
                    lane_fore = lane_number(phy, posgy(scene[sorted_changers[i]]))
                    lane_rear = lane_number(phy, posgy(scene[sorted_changers[i+1]]))
                    if lane_fore[1] == lane_fore[2] && lane_rear[1] != lane_rear[2]
                        #目前为前车停止位移 ， 应修改为后车停止位移并回到原位，前车继续位移。 但会导致后车的碰撞检测变得复杂？
                        next_scene[sorted_changers[i]] = Entity(next_scene[sorted_changers[i]], VehicleState(next_scene[sorted_changers[i]].state, roadway, posgy(scene[sorted_changers[i]])))
                        if sorted_changers[i] == 1 #ego car should store action
                            models[1] = EgoModel(MLAction(0, models[1].a.a_lon))  
                        end
                    else
                        #后车停止位移
                        next_scene[sorted_changers[i+1]] = Entity(next_scene[sorted_changers[i+1]], VehicleState(next_scene[sorted_changers[i+1]].state, roadway, posgy(scene[sorted_changers[i+1]])))   
                        if sorted_changers[i+1] == 1 #ego car should store action
                            models[1] = EgoModel(MLAction(0, models[1].a.a_lon))  
                        end
                    end
                end
            end
        end
    end
    ## Dynamics and Exits ##
    #Delete old car 
    zero_car = zeros(Int, 0)
    other_car = zeros(Int, 0)
        
    for veh in next_scene
        if posgx(veh) == 0   
            append!( zero_car, veh.id )
        else
            if abs( posgx(veh) - posgx(next_scene[1]) ) > 100  
                next_scene[veh.id] = Entity(VehicleState(VecSE2(0.0,0.0,0.0), roadway, 0.0), VehicleDef(), veh.id)
                models[veh.id] = DeleteModel()
                append!( zero_car, veh.id )
            else
               append!( other_car, veh.id )
            end
        end
    end

    #generate new car 
    if length(zero_car) != 0 # && rand(rng) <= mdp.dmodel.p_appear  #appear在论文中并没有提到
        models[zero_car[1]] = pa_to_Model(rand(rng, gen, zero_car[1]), phy)
        vel = models[zero_car[1]].mlon.v_des #速度需要一个起始误差
        clearances = zeros(Float64, nb_lane) 
        
        fill!(clearances, Inf)
        closest_cars = zeros(Int, nb_lane)
        
        fill!(closest_cars, 0)
        sstar_margins = zeros(Float64, nb_lane)
        if vel > next_scene[1].state.v
            # put at back
            for other_car_id in other_car  #找出每条路中x值最小的车  
                lowlane, highlane = occupation_lanes(phy, posgy(next_scene[other_car_id]), 0)
                back = posgx(next_scene[other_car_id]) - posgx(next_scene[1]) +50  
                if back < clearances[lowlane]
                    clearances[lowlane] = back
                    closest_cars[lowlane] = other_car_id
                end
                if back < clearances[highlane]
                    clearances[highlane] = back
                    closest_cars[highlane] = other_car_id
                end
            end
            for j in 1:nb_lane
                other = closest_cars[j]
                if other == 0
                    sstar = 0.0
                else
                    sstar=idm_gstar(models[zero_car[1]], vel, vel - next_scene[other].state.v)
                end
                sstar_margins[j] = clearances[j] - sstar #clearance to the nearest car
            end
        else
            for other_car_id in other_car 
                lowlane, highlane = occupation_lanes(phy, posgy(next_scene[other_car_id]), 0)
                front = posgx(next_scene[1]) +50 - posgx(next_scene[other_car_id])   
                if front < clearances[lowlane]
                    clearances[lowlane] = front
                    closest_cars[lowlane] = other_car_id
                end
                if front < clearances[highlane]
                    clearances[highlane] = front
                    closest_cars[highlane] = other_car_id
                end
            end
            for j in 1:nb_lane
                other = closest_cars[j]
                if other == 0
                    sstar = 0
                else
                    sstar=idm_gstar(models[other], next_scene[other].state.v, next_scene[other].state.v - vel)
                end
                sstar_margins[j] = clearances[j] - sstar
            end
        end
        margin, lane = findmax(sstar_margins)
        
        if margin > 0.0
            if vel > next_scene[1].state.v
                # at back                               
                next_scene[zero_car[1]] = Entity(VehicleState(VecSE2(posgx(next_scene[1]) - 50, lane_width*(lane-1), 0.0), roadway, vel), VehicleDef(), zero_car[1])
            else
                next_scene[zero_car[1]] = Entity(VehicleState(VecSE2(posgx(next_scene[1]) + 50, lane_width*(lane-1), 0.0), roadway, vel), VehicleDef(), zero_car[1])
            end
        else
            models[zero_car[1]] = DeleteModel()#不产生新车
        end
    end
    #test record action
    #=
    f=open("C:/Users/Mers/Desktop/action_record.txt","a")  #path
    write(f, "$(models[1].a)\n")
    close(f)
    =#
    #end : test record action

    return MLState(next_scene, models)
=#
end


function POMDPs.observation(pomdp::NoCrashProblem, sp::MLState)
    return Random.SamplerTrivial(Deterministic(sp.phy_state))
    #return sp.phy_state
end
#=
function Base.rand(rng::AbstractRNG, sp::Deterministic)
    return sp.val
end
=#
#=
observation(m::POMDP, statep)
observation(m::POMDP, action, statep)
observation(m::POMDP, state, action, statep)
=#