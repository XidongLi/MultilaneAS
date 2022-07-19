mutable struct Simple{M} <: Policy #
    mdp::M
    #sweeping_up::Bool #may be used for action(p::Simple,s::Vector), but has not implemented
end
mutable struct SimpleSolver <: Solver end

#Simple(mdp) = Simple(mdp,true)
POMDPs.solve(solver::SimpleSolver, problem::MDP) = Simple(problem)
  #solve(solver::SimpleSolver, problem::EmbeddedBehaviorMDP) = Simple(problem.base)
  POMDPs.solve(solver::SimpleSolver, problem::POMDP) = Simple(problem)
  #solve(solver::SimpleSolver, problem::AggressivenessBeliefMDP) = Simple(get(problem.up.problem))
  #solve(solver::SimpleSolver, problem::BehaviorBeliefMDP) = Simple(get(problem.up.problem))
  #solve(solver::SimpleSolver, problem::QMDPWrapper) = Simple(problem.mdp)
  #solve(solver::SimpleSolver, problem::OutcomeMDP) = Simple(problem.mdp)
  #POMDPs.updater(::Simple) = POMDPToolbox.FastPreviousObservationUpdater{MLObs}()
create_policy(s::SimpleSolver, problem::MDP) = Simple(problem)
create_policy(s::SimpleSolver, problem::POMDP) = Simple(problem)
  
set_rng!(solver::SimpleSolver, rng::AbstractRNG) = nothing
#Base.srand(p::Simple, s) = p #???

function POMDPs.action(p::Simple,s::Vector)
# lane changes if there is an opportunity
    pomdp = p.mdp
    state = s[1]
    scene = state.phy_state
    veh_ego = scene[1]
    phy = get_phy(pomdp)
    roadway = phy.road
    acc = phy.adjustment_acceleration
    vy_0 = phy.vy_0


    available_actions = actions(pomdp, state)
    lane_actions = Vector{MLAction}()
    maintain_actions = Vector{MLAction}()
    if length(available_actions) == 1
        return available_actions[1]
    else
        for action in available_actions
            if action.v_lat == vy_0
                push!(lane_actions, action)
            elseif action.v_lat == 0.0
                push!(maintain_actions, action)
            end
        end
    end

    if length(lane_actions) != 0
        if length(lane_actions) == 3
            fore = find_neighbor(scene, roadway, veh_ego,
                            targetpoint_ego=VehicleTargetPointCenter(), 
                            targetpoint_neighbor=VehicleTargetPointCenter()) 
            rear = find_neighbor(scene, roadway, veh_ego, rear=true,
                                    targetpoint_ego=VehicleTargetPointCenter(), 
                                    targetpoint_neighbor=VehicleTargetPointCenter())
            if rear.ind == nothing && fore.ind == nothing
                return MLAction(vy_0, 0.)
            else
                sgn = fore.Δs <= rear.Δs ? -1 : 1
                accel = sgn * acc
                return MLAction(vy_0, accel)
            end
        else
            return lane_actions[1]#rand(lane_actions)
        end
    elseif length(maintain_actions) != 0
        if length(maintain_actions) == 3
            fore = find_neighbor(scene, roadway, veh_ego,
                            targetpoint_ego=VehicleTargetPointCenter(), 
                            targetpoint_neighbor=VehicleTargetPointCenter()) 
            rear = find_neighbor(scene, roadway, veh_ego, rear=true,
                                    targetpoint_ego=VehicleTargetPointCenter(), 
                                    targetpoint_neighbor=VehicleTargetPointCenter())
            if rear.ind == nothing && fore.ind == nothing
                return MLAction(0.0, 0.0)
            else
                sgn = fore.Δs <= rear.Δs ? -1 : 1
                accel = sgn * acc
                return MLAction(0.0, accel)
            end
        else
            return maintain_actions[1]#rand(maintain_actions)
        end
    else
        return available_actions[1]#rand(available_actions)
    end
#=
    pomdp = p.mdp
    state = s[1]
    scene = state.phy_state
    veh_ego = scene[1]
    phy = get_phy(pomdp)
    roadway = phy.road
    acc = phy.adjustment_acceleration
#if can't move towards desired lane sweep through accelerating and decelerating   
    lanes = lane_number(phy,posgy(veh_ego))
    if lanes[1] != lanes[2]
        return MLAction(phy.vy_0, 0.0)
    end
    if is_safe(pomdp, scene, MLAction(phy.vy_0, 0.0)) 
        return MLAction(phy.vy_0, 0.0)
    end
# maintain distance
    fore = find_neighbor(scene, roadway, veh_ego,
                            targetpoint_ego=VehicleTargetPointCenter(), 
                            targetpoint_neighbor=VehicleTargetPointCenter()) 
    rear = find_neighbor(scene, roadway, veh_ego, rear=true,
                            targetpoint_ego=VehicleTargetPointCenter(), 
                            targetpoint_neighbor=VehicleTargetPointCenter())
    if rear.ind == nothing && fore.ind == nothing
        return MLAction(0.,0.)
    end
    sgn = fore.Δs <= rear.Δs ? -1 : 1
    accel = sgn * acc
    if is_safe(pomdp, scene, MLAction(0., accel)) 
        return MLAction(0., accel)
    else
        return MLAction(0, calc_brake_acc(pomdp, scene))
    end  
    #return MLAction(0., accel)
=#
end

#action(p::Simple, b::BehaviorBelief) = action(p, b.physical)
#=
  function action(pol::Simple, s::QMDPState)
      if s.isstate
          return action(pol, get(s.s))
      else
          return action(pol, get(s.b))
      end
  end
  action(p::Simple, b::AbstractParticleBelief) = action(p, MLPhysicalState(first(particles(b))))
  
  mutable struct BehaviorSolver <: Solver
      b::BehaviorModel
      keep_lane::Bool
      rng::AbstractRNG
  end
  mutable struct BehaviorPolicy <: Policy
      problem::NoCrashProblem
      b::BehaviorModel
      keep_lane::Bool
      rng::AbstractRNG
  end
  solve(s::BehaviorSolver, p::NoCrashProblem) = BehaviorPolicy(p, s.b, s.keep_lane, s.rng)
  
  function action(p::BehaviorPolicy, s::MLState, a::MLAction=MLAction(0.0,0.0))
      nbhd = get_neighborhood(p.problem.dmodel.phys_param, s, 1)
      acc = gen_accel(p.b, p.problem.dmodel, s, nbhd, 1, p.rng)
      if p.keep_lane
          lc = 0.0
      else
          lc = gen_lane_change(p.b, p.problem.dmodel, s, nbhd, 1, p.rng)
      end
      return MLAction(acc, lc)
  end
  action(p::BehaviorPolicy, b::AggressivenessBelief, a::MLAction=MLAction(0.0,0.0)) = action(p, most_likely_state(b))
  action(p::BehaviorPolicy, b::BehaviorParticleBelief, a::MLAction=MLAction(0.0,0.0)) = action(p, most_likely_state(b))
  
  mutable struct IDMLaneSeekingSolver <: Solver
      b::BehaviorModel
      rng::AbstractRNG
  end
  
  mutable struct IDMLaneSeekingPolicy <: Policy
      problem::NoCrashProblem
      b::BehaviorModel
      rng::AbstractRNG
  end
  solve(s::IDMLaneSeekingSolver, p::NoCrashProblem) = IDMLaneSeekingPolicy(p, s.b, s.rng)
  
  function action(p::IDMLaneSeekingPolicy, s::MLState, a::MLAction=MLAction(0.0,0.0))
      nbhd = get_neighborhood(p.problem.dmodel.phys_param, s, 1)
      acc = gen_accel(p.b, p.problem.dmodel, s, nbhd, 1, p.rng)
      # try to positive lanechange
      # lc = problem.dmodel.lane_change_rate * !is_lanechange_dangerous(pp,s,nbhd,1,1)
      lc = p.problem.dmodel.lane_change_rate
      if is_safe(p.problem, s, MLAction(acc, lc))
          return MLAction(acc, lc)
      end
      return MLAction(acc, 0.0)
  end
  
  
  
  struct OptimisticValue end
  
  function MCTS.estimate_value(v::OptimisticValue, m::NoCrashProblem, s::Union{MLState,MLPhysicalState}, steps::Int)
      rm = m.rmodel
      if rm isa NoCrashRewardModel
          rw = rm.reward_in_target_lane
      elseif rm isa SuccessReward
          rw = 1.0
      else
          error("Unrecognized reward model.")
      end
      dm = m.dmodel
      dt = dm.phys_param.dt
      steps_away = abs(rm.target_lane - first(s.cars).y) / (dm.lane_change_rate * dt)
      if s.x + first(s.cars).vel*steps_away*dt > dm.max_dist
          return 0.0
      elseif dm.lane_terminate
          return discount(m)^steps_away*rw
      else
          gamma = discount(m)
          return gamma^steps_away*rw/(1-gamma)
      end
  end
  
  MCTS.estimate_value(v::OptimisticValue, m::MLPOMDP, s::Union{MLState,MLPhysicalState}, h, steps::Int) = estimate_value(v, m, s, steps)

=#



mutable struct Record{M} <: Policy 
    mdp::M
    actions::Vector{MLAction}
    step::Int
end
function Record(pomdp::POMDP)
    f=open(joinpath(pwd(), "action_record.txt"),"r")  #path
    actions = split(read(f, String), "\n" )
    ac = Vector{MLAction}()   
    for action in actions
        if length(action) != 0
            a = split(action, ( '(' , ',' , ')' ) )[2:3]
            a = MLAction(parse(Float64, a[1]), parse(Float64, a[2]))
            push!(ac, a)
        end
    end
    close(f)
    return Record(pomdp, ac, 1)
end
function POMDPs.action(p::Record,s::Vector)
    actions(p.mdp, s[1])
    return p.actions[p.step]
end

