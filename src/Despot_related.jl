#=
function lower(pomdp::NoCrashProblem, b::ScenarioBelief)
    return typemin(Int)
end

function upper(pomdp::NoCrashProblem, b::ScenarioBelief)
    return typemax(Int)
end

function default_action(pomdp::NoCrashProblem)
    return MLAction(0.0, 0.0)
end

function DespotPolicy(pomdp::NoCrashProblem)

    solver = DESPOTSolver(bounds=IndependentBounds(lower, upper;check_terminal=true),default_action=default_action,K=10)
    return solve(solver, pomdp)
end
=#

function DespotPolicy(pomdp::NoCrashProblem)

    solver = DESPOTSolver(bounds=IndependentBounds(lower, upper;check_terminal=true),default_action=default_action,K=500)
    return solve(solver, pomdp)
end

function DespotPolicy_dqn(pomdp::NoCrashProblem)

    solver = DESPOTSolver(bounds=IndependentBounds(simple_lower, dqn_upper;check_terminal=true),default_action=default_action,K=100)
    return solve(solver, pomdp)
end
function simple_lower(pomdp::NoCrashProblem, b::ScenarioBelief)
    #return typemin(Int)
    belief = deepcopy(b) #deep copy , to avoid change scenario_belief during calculate lower bound
    lower_bound = lbound(DefaultPolicyLB(Simple(pomdp);max_depth=90), pomdp, belief)
    #println("lower is: $lower_bound")
    return lower_bound
    #return lbound(DefaultPolicyLB(Simple(pomdp);max_depth=90), pomdp, belief)
end
function ARDESPOT.bounds(bounds::IndependentBounds, pomdp::POMDP, b::ScenarioBelief)
    if bounds.check_terminal && all(isterminal(pomdp, s) for s in particles(b))
        return (0.0, 0.0)
    end
    l = lbound(bounds.lower, pomdp, b)
    u = ubound(bounds.upper, pomdp, b, l)
    if u < l && u >= l-bounds.consistency_fix_thresh
        u = l
    end
    return (l,u)
end
ubound(f::Function, pomdp, b, l) = f(pomdp, b, l)
function dqn_upper(pomdp::NoCrashProblem, b::ScenarioBelief, l::Float64)

    available_actions = actions(pomdp, b) 
    
    DQNPolicy = load(joinpath(pwd(), "DQNPolicy.bson"), MultilaneAS)[:DQNPolicy]
    #DQNPolicy = load("C:/Users/Mers/Desktop/DQNPolicy.bson", MultilaneAS)[:DQNPolicy]
    v = zeros(Float64, length(available_actions))
    for i in 1:n_particles(b)
        indexes = actionindexes(pomdp, available_actions)
        v = v + actionvalues(DQNPolicy, particle(b, i))[indexes]
    end
    v = v/n_particles(b)
    

    #DQNPolicy = load("C:/Users/Mers/Desktop/DQNPolicy.bson", MultilaneAS)[:DQNPolicy]
    belief = deepcopy(b) #deep copy , to avoid change scenario_belief during calculate lower bound
    upper_bound = maximum(v)
    if upper_bound <= l
        return upper(pomdp, b, l)
    else
        return upper_bound
    end
    #upper_bypolicy = lbound(DefaultPolicyLB(DQNPolicy; max_depth=90), pomdp, belief)#这个是不准确的
    #println("upper bound is:$upper_bound")
    #println("upper bound from policy is $upper_bypolicy")
    #return max(upper_bound, upper_bypolicy) #改为如果大于0.99，则输出1 ？
    #return maximum(v)
    #return lbound(DefaultPolicyLB(DQNPolicy; max_depth=90), pomdp, belief)
    
end
#=
function DespotPolicy_nlglobal(pomdp::NoCrashProblem)

    solver = DESPOTSolver(bounds=IndependentBounds(simple_lower, nlglobal_upper;check_terminal=true),default_action=default_action,K=10)
    return solve(solver, pomdp)
end

function nlglobal_upper(pomdp::NoCrashProblem, b::ScenarioBelief)
    #ngfa = load("C:/Users/Mers/Desktop/nonlinear_global_Approximator.bson", MultilaneAS)[:ngfa]
    ngfa = load(joinpath(pwd(), "nonlinear_global_Approximator.bson"), MultilaneAS)[:ngfa]
    
    v = 0
    for i in 1:n_particles(b)
        v = v + compute_value(ngfa, convert_featurevector(Vector{Float64}, particle(b, i), UnderlyingMDP(pomdp))) 
    end
    v = v/n_particles(b)
    return v
end

function DespotPolicy_lnglobal(pomdp::NoCrashProblem)

    solver = DESPOTSolver(bounds=IndependentBounds(simple_lower, lnglobal_upper;check_terminal=true),default_action=default_action,K=10)
    return solve(solver, pomdp)
end

function lnglobal_upper(pomdp::NoCrashProblem, b::ScenarioBelief)
    ngfa = load(joinpath(pwd(), "linear_global_Approximator.bson"), MultilaneAS)[:lgfa]
    #ngfa = load("C:/Users/Mers/Desktop/linear_global_Approximator.bson", MultilaneAS)[:lgfa]
    v = 0
    for i in 1:n_particles(b)
        v = v + compute_value(ngfa, convert_featurevector(Vector{Float64}, particle(b, i), UnderlyingMDP(pomdp))) 
    end
    v = v/n_particles(b)
    return v
end
=#


function MCTS.default_action(f::Function, pomdp::NoCrashProblem, b, ex) #use simple policy 
    #return MLAction(0.0, 0.0)
    #return action(pomdp, b[1]) 
    return action(Simple(pomdp), [b[1]]) 
    #return MLAction(2.0, 0.0)
end

#default_action(p.sol.default_action, p.pomdp, b, ex)::actiontype(p.pomdp), info

struct DefaultPolicy <:Policy
end

function POMDPs.action(policy::DefaultPolicy, b)
    #return action(Simple(pomdp), [b[1]]) 
    return MLAction(2.0, 0.0)
end
function POMDPs.action(p::Simple, b::ScenarioBelief) #合理的，因为在root时，所有的pairticle的phy state都一样，而simple policy只取决于phy state
    return action(p, [particle(b, 1)])
end
#=
function lower(pomdp::NoCrashProblem, b::ScenarioBelief)
    return lbound(DefaultPolicyLB(Simple(pomdp);max_depth=90), pomdp, b)
end
=#
function lower(pomdp::NoCrashProblem, b::ScenarioBelief) #use MLAction(2.0, 0.0) as DefaultPolicy
    #return typemin(Int)
    belief = deepcopy(b) #deep copy , to avoid change scenario_belief during calculate lower bound
    return lbound(DefaultPolicyLB(DefaultPolicy();max_depth=90), pomdp, belief)
end

function upper(pomdp::NoCrashProblem, b::ScenarioBelief, l::Float64)
    state = particle(b, 1)
    return upper(pomdp, state)
end

function upper(pomdp::NoCrashProblem, s::MLState)
    ego_y = posgy(s.phy_state[1])
    phy = get_phy(pomdp)
    lane = minimum(lane_number(phy,ego_y))
    return discount(pomdp)^(phy.nb_lanes - 1 - lane)*pomdp.rmodel.reward_in_target_lane
end

#=
#new bound
#LocalApproximationValueIteration, can only be used in Correlated case
function Base.convert(::Type{Vector}, gen::CorrelatedIDMMOBIL, model::ASModel)
    return aggressiveness(gen, model) #根据当前IDM_MOBIL_参数算出aggressive
end

function Base.convert(::Type{Vector}, gen::CorrelatedIDMMOBIL, model::DeleteModel)
    return -1.0
end

function Base.convert(::Type{DriverModel}, gen::CorrelatedIDMMOBIL, v::Float64)
        if v == -1.0
            return DeleteModel()
        else
            return create_model(gen, v)
        end
    #end
end

#Note that SVector must be imported through StaticArrays.jl
function POMDPs.convert_s(::Type{V} where V <: AbstractVector, s::MLState, mdp::UnderlyingMDP)
    models = s.inter_state
    a = [convert(Vector, get_behavior(mdp.pomdp), models[i+1]) for i in 1:length(models)-1]
    v = SVector(Tuple(a))
    return v
end

function POMDPs.convert_s(::Type{MLState}, v::AbstractVector, mdp::UnderlyingMDP)
    phy = get_initial(mdp.pomdp)
    models = Dict{Int, DriverModel}(1 => EgoModel())
    for i in 1:length(v)
        push!(models, i+1 => convert(DriverModel, get_behavior(mdp.pomdp), v[i]))
    end
    s = MLState(phy, models)
    return s
end

function gen_Grid(VERTICES_PER_AXIS::Int, gen::CorrelatedIDMMOBIL, mdp::UnderlyingMDP)
    #VERTICES_PER_AXIS   Controls the resolutions along the grid axis
    #grid = RectangleGrid((range(min[i], stop=max[i], length=VERTICES_PER_AXIS) for i in 1:length(min))..., [0.0, 1.0])
    grid = RectangleGrid((range(0.0, stop=1.0, length=VERTICES_PER_AXIS) for i in 1:car_n(mdp.pomdp)-1)...)     
    interp = LocalGIFunctionApproximator(grid)
    approx_solver = LocalApproximationValueIterationSolver(interp, verbose=true, max_iterations=1000, is_mdp_generative=true, n_generative_samples=1)
    approx_policy = solve(approx_solver, mdp)
    return approx_policy
end
#v = value(approx_policy, s)  # returns the approximately optimal value for state s
#a = action(approx_policy, s) # returns the approximately optimal action for state s
#123
=#
#DQN
function Base.convert(::Type{Vector}, gen::BehaviorGenerator, b::EgoModel) #b = belief
    return append!(repeat([1.5],8) , sign(b.action.v_lat))   
end
function Base.convert(::Type{Vector}, gen::BehaviorGenerator, b::ASModel)
    return [
        (b.mlon.T - T_max(gen))/(T_min(gen) - T_max(gen)),# T is lower for more aggressive
        (b.mlon.v_des - v_des_min(gen))/(v_des_max(gen) - v_des_min(gen)),
        (b.mlon.s_min - s_min_max(gen))/(s_min_min(gen) - s_min_max(gen)),# s0 is lower for more aggressive
        (b.mlon.a_max - a_max_min(gen))/(a_max_max(gen) - a_max_min(gen)),
        (b.mlon.d_cmf - d_cmf_min(gen))/(d_cmf_max(gen) - d_cmf_min(gen)),
        (b.mlane.safe_decel - safe_decel_min(gen))/(safe_decel_max(gen) - safe_decel_min(gen)),
        (b.mlane.politeness - politeness_max(gen))/(politeness_min(gen) - politeness_max(gen)),# p is lower for more aggressive
        (b.mlane.advantage_threshold - advantage_threshold_max(gen))/(advantage_threshold_min(gen) - advantage_threshold_max(gen)),# a_thr is lower for more aggressive
        sign(b.mlane.dir)
    ]
end
function Base.convert(::Type{ASModel}, gen::BehaviorGenerator, vec::Vector)
    idm = IntelligentDriverModel(
        σ = gen.min.mlon.σ, 
        T = vec[1]*(T_min(gen) - T_max(gen)) + T_max(gen),
        v_des = vec[2]*(v_des_max(gen) - v_des_min(gen)) + v_des_min(gen),
        s_min = vec[3]*(s_min_min(gen) - s_min_max(gen)) + s_min_max(gen),
        a_max = vec[4]*(a_max_max(gen) - a_max_min(gen)) + a_max_min(gen),
        d_cmf = vec[5]*(d_cmf_max(gen) - d_cmf_min(gen)) + d_cmf_min(gen)
        )
    mlat = gen.min.mlat
    mobil = MOBIL(
        mlon = idm, 
        safe_decel = vec[6]*(safe_decel_max(gen) - safe_decel_min(gen)) + safe_decel_min(gen),
        politeness = vec[7]*(politeness_min(gen) - politeness_max(gen)) + politeness_max(gen),
        advantage_threshold = vec[8]*(advantage_threshold_min(gen) - advantage_threshold_max(gen)) + advantage_threshold_max(gen)
        )
    mobil.dir = vec[9]
    ASModel(idm, mlat, mobil)
end
function Base.convert(::Type{Vector}, gen::BehaviorGenerator, b::DeleteModel)
    return append!(repeat([2],8) , 0)   
end
#=
function POMDPs.convert_s(::Type{V} where V <: AbstractArray, s::Vector{MLState}, mdp::UnderlyingMDP) #use for test DQNpolicy
    return convert_s(AbstractArray, s[1], mdp)
end
=#
#POMDPs.action(policy::NNPolicy{P}, s) where {P <: MDP} = _action(policy, POMDPs.convert_s(Array{Float32}, s, policy.problem))
function POMDPs.action(policy::NNPolicy{P}, s::Vector{MLState}) where {P <: MDP} #DQNpolicy
    available_actions = actions(policy.problem.pomdp, s[1])
    if length(available_actions) == 1
        return available_actions[1]
    else
        v = zeros(Float64, length(available_actions))
        indexes = actionindexes(policy.problem.pomdp, available_actions)
        for i in 1:length(s)
            v = v + actionvalues(policy, s[i])[indexes]
        end
        best_index = argmax(v)
        return actions(policy.problem.pomdp)[indexes][best_index]
    end
#=
    value_vector = actionvalues(policy, s[1])
    for (i, state) in enumerate(s)
        value_vector = value_vector + actionvalues(policy, s[i])
    end
    value_vector = value_vector - actionvalues(policy, s[1])
    return policy.action_map[argmax(value_vector)]
=#
end
function POMDPs.action(p::NNPolicy{P}, b::ScenarioBelief) where {P <: MDP}
    return action(p, [particle(b, 1)])
end

function POMDPs.action(policy::NNPolicy{P}, s::MLState) where {P <: MDP}
    available_actions = actions(policy.problem.pomdp, s)
    if length(available_actions) == 1
        return available_actions[1]
    else
        indexes = actionindexes(policy.problem.pomdp, available_actions)
        best_index = argmax(actionvalues(policy, s)[indexes])
        return actions(policy.problem.pomdp)[indexes][best_index]
    end
        #return policy.action_map[best_index]
end

function POMDPs.action(policy::NNPolicy{P}, s::Vector) where {P <: MDP}
    s = POMDPs.convert_s(MLState, s, policy.problem)
    return action(policy, s)
end

function POMDPs.convert_s(::Type{V} where V <: AbstractArray, s::MLState, mdp::UnderlyingMDP)
    phy_pa = get_phy(mdp.pomdp)
    phy_state = s.phy_state
    models = s.inter_state
    vec = Vector{Float64}()#get phy car_num undef
    for i in 1:length(phy_state)
        x = (posgx(phy_state[i]) - posgx(phy_state[1])) / 50 #50是否设为可改变量（在phy中）？ 
        if abs(x) > 1 #deletemodel
            x = 2 #deepset
        end
        y = posgy(phy_state[i]) / ((phy_pa.nb_lanes-1)*phy_pa.w_lane)
        v = phy_state[i].state.v / 33.35
        push!(vec,x,y,v)
        #vec[i] = ***
        inter = convert(Vector, get_behavior(mdp.pomdp), models[i])
        vec = vcat(vec,inter)
    end
    return vec
end

function POMDPs.convert_s(::Type{MLState}, vec::AbstractArray, mdp::UnderlyingMDP)
    phy_pa = get_phy(mdp.pomdp)
    phy_state = get_initial_phy(mdp.pomdp)
    models = Dict{Int, DriverModel}()
    #vec = Vector{Float64}()#get phy car_num undef
    for i in 1:phy_pa.nb_cars
        
        if vec[1+(i-1)*12] == 2
            #Delete model
            phy_state[i] = Entity(VehicleState(VecSE2(-250.0,0.0,0.0), get_road(mdp.pomdp), 31.0), VehicleDef(), i)
            push!(models, i => DeleteModel())
            continue
        end
        x = vec[1+(i-1)*12] * 50 + 100
        y = vec[2+(i-1)*12] * ((phy_pa.nb_lanes-1)*phy_pa.w_lane)
        v = vec[3+(i-1)*12] * 33.35

        phy_state[i] = Entity(phy_state[i], VehicleState(VecSE2(x, y, 0), get_road(mdp.pomdp), v))
        if i == 1
            push!(models, i => EgoModel(MLAction(vec[12]*phy_pa.vy_0, 0.0)))
        else
            push!(models, i => convert(ASModel, get_behavior(mdp.pomdp), vec[4+(i-1)*12 : 12+(i-1)*12]))
        end
    end
    return MLState(phy_state, models)
end

function gen_phy(i::Int)
    if i == 1
        return 100, rand(0:0.2:8.8), rand(27.8:0.1:38.9) 
    else
        return rand(-50:50) + 100.0, rand(0:0.2:9.0), rand(27.8:0.1:38.9) 
    end
end
function gen_phy(i::Int,mode::Bool)
    if i == 1
        return 100, 0.0, 33.3 
    else
        return rand(-50:50) + 100.0, rand(0.0:3.0:9.0), rand(27.8:0.1:38.9) 
    end
end
function phy_collision(x::Vector, y::Vector, v::Vector, l_car::Float64)
    i = length(x)
    for index in 1:i-1
        if sqrt((x[i] - x[index])^2 + (y[i] - y[index])^2) <= l_car*2
            return true
        end
    end
    return false
end

function POMDPs.initialstate(pomdp::NoCrashProblem; train = true)
    #=
    initial_state = load("C:/Users/Mers/Desktop/initial_state.bson", MultilaneAS)[:initial_state]
    return Deterministic(initial_state)
    =#
    
    phy = get_phy(pomdp)
    scene = get_initial_phy(pomdp)
    models = Dict{Int, DriverModel}()
    roadway = phy.road
    x = Vector{Float64}()
    y = Vector{Float64}()
    v = Vector{Float64}()
    i = 1
    while length(x) < phy.nb_cars
        #phy_state
        if train
            x_,y_,v_ = gen_phy(i)
        else
            x_,y_,v_ = gen_phy(i,train)
        end
        push!(x,x_)
        push!(y,y_)
        push!(v,v_)
        while phy_collision(x, y, v, phy.l_car)
            if train
                x_,y_,v_ = gen_phy(i)
            else
                x_,y_,v_ = gen_phy(i,train)
            end
            x[i] = x_
            y[i] = y_
            v[i] = v_
        end
        scene[i] = Entity(VehicleState(VecSE2(x_,y_,0.0), roadway, v_), VehicleDef(), i)
        #inter_state
        if i == 1
            push!(models, i => EgoModel(phy, y_))
            if y_ % phy.w_lane != 0
                models[i].action = MLAction(rand((1, -1)) * phy.vy_0, 0.0)
                
            end
        else
            push!(models, i => rand(Random.GLOBAL_RNG, get_behavior(pomdp)) )
            if y_ % phy.w_lane != 0
                models[i].mlane.dir = rand((1, -1))
            end
        end
        #Delete_model
        if i !=1 && rand()< 0.05
            scene[i] = Entity(VehicleState(VecSE2(-250.0,0.0,0.0), roadway, 31.0), VehicleDef(), i)
            models[i] = DeleteModel()
        end
        i = i+1
    end
    return Deterministic(MLState(scene, models))
    
end

struct EpsSimplePolicy{T<:Function, R<:AbstractRNG, A} <: ExplorationPolicy
    eps::T
    rng::R
    actions::A
end

function EpsSimplePolicy(problem, eps::Function; 
                         rng::AbstractRNG=Random.GLOBAL_RNG)
    return EpsSimplePolicy(eps, rng, actions(problem))
end
function EpsSimplePolicy(problem, eps::Real; 
                         rng::AbstractRNG=Random.GLOBAL_RNG)
    return EpsSimplePolicy(x->eps, rng, actions(problem))
end
POMDPPolicies.loginfo(p::EpsSimplePolicy, k) = (eps=p.eps(k),)
function POMDPs.action(p::EpsSimplePolicy, on_policy::Policy, k, s)
    state = POMDPs.convert_s(MLState, s, on_policy.problem)
    
    if rand(p.rng) < p.eps(k)
        available_actions = actions(on_policy.problem.pomdp, state)
        return rand(p.rng, available_actions)
    else 
        return action(on_policy, state)
    end
end

function DQN_policy(mdp::UnderlyingMDP, continue_train::Bool)
    if continue_train == true
        #DQNPolicy = load("C:/Users/Mers/Desktop/DQNPolicy.bson", MultilaneAS)[:DQNPolicy]
        DQNPolicy = load(joinpath(pwd(), "DQNPolicy.bson"), MultilaneAS)[:DQNPolicy]
        model = DQNPolicy.qnetwork
        exploration = EpsSimplePolicy(mdp, 0.01)
        #exploration = EpsGreedyPolicy(mdp, 0.01)
    else
        model = Chain(
                #Dense(120, 512, sigmoid),
                #Dense(512, 256, sigmoid),
                #Dense(256, 128, sigmoid),
                Dense(120, 128, leakyrelu),
                Dense(128, 256, leakyrelu),
                #Dense(512, 512, leakyrelu),
                #Dense(512, 512, leakyrelu),  #hyperparameteroptimisation
                #Dense(32, 32, leakyrelu),
                Dense(256, length(actions(mdp)))
                ) 
        exploration = EpsSimplePolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=100000/2))
    end
    #exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=10000/2))# use a constant eps? 0.05 
    
    #exploration = EpsSimplePolicy(mdp, 0.1)
    #=
    solver = DeepQLearningSolver(qnetwork = model, max_steps=5000000, batch_size = 32, train_freq=90,
                             exploration_policy = exploration,
                             learning_rate=0.005,log_freq=900,
                             recurrence=false,double_q=false, dueling=true, prioritized_replay=true,
                             logdir = "C:/Users/Mers/Desktop")
                            =#
    solver = DeepQLearningSolver(qnetwork = model, max_steps=100000, batch_size = 32, train_freq=10,
                            eval_freq = 500, 
                            target_update_freq = 500, 
                             exploration_policy = exploration, 
                             learning_rate=0.001,train_start = 200,
                             recurrence=false,double_q=false, dueling=false, prioritized_replay=false,
                             verbose = true, log_freq = 500)
                             #logdir = "C:/Users/Mers/Desktop")
    
    DQNPolicy = solve(solver, mdp)
    #@save "C:/Users/Mers/Desktop/DQNPolicy.bson" DQNPolicy #path tensorboard
    @save joinpath(pwd(), "DQNPolicy.bson") DQNPolicy #path 
    return DQNPolicy
end

function DeepQLearning.populate_replay_buffer!(replay::PrioritizedReplayBuffer,
                                                env::AbstractEnv,
                                                action_indices;
                                                max_pop::Int64=replay.max_size, max_steps::Int64=100,
                                                policy::Policy = Simple(env.m.pomdp))
    reset!(env)
    o = observe(env)
    done = false
    step = 0
    #state = POMDPs.convert_s(MLState, o, env.m)
    for t=1:(max_pop - replay._curr_size)
        a = action(policy, [POMDPs.convert_s(MLState, o, env.m)])
        ai = action_indices[a]
        rew = act!(env, a)
        op = observe(env)
        done = terminated(env)
        exp = DQExperience(o, ai, Float32(rew), op, done)
        DeepQLearning.add_exp!(replay, exp, abs(Float32(rew))) # assume initial td error is r
        o = op
        # println(o, " ", action, " ", rew, " ", done, " ", info) #TODO verbose?
        step += 1
        if done || step >= max_steps
            reset!(env)
            o = observe(env)
            done = false
            step = 0
        end
    end
    @assert replay._curr_size >= replay.batch_size
end

#=
function DeepQLearning.create_dueling_network(network::DeepQLearning.DuelingNetwork)
    return network
end

#GlobalApproximationValueIteration
#linear
function convert_featurevector(::Type{Vector{Float64}}, s::MLState, mdp::UnderlyingMDP) #Svector
    #v = SVector(convert_s(AbstractArray, s, mdp))
    #return v
    return convert_s(AbstractArray, s, mdp)
end

function GlobalApproximationValueIteration.sample_state(mdp::UnderlyingMDP, rng::RNG=Random.GLOBAL_RNG) where {RNG <: AbstractRNG}
    return rand(initialstate(mdp.pomdp))
end

function linear_global_policy(mdp::UnderlyingMDP, continue_train::Bool)
    if continue_train == true
        lgfa = load(joinpath(pwd(), "linear_global_Approximator.bson"), MultilaneAS)[:lgfa]
        #lgfa = load("C:/Users/Mers/Desktop/linear_global_Approximator.bson", MultilaneAS)[:lgfa]
    else
        lgfa = LinearGlobalFunctionApproximator_M(zeros(101))
    end
    gfa_solver = GlobalApproximationValueIterationSolver(lgfa, num_samples=120, max_iterations=1, verbose=true, is_mdp_generative=true, n_generative_samples=1, fv_type=SVector{110, Float64})
    #max_iterations为外围循环次数(需要是一个很大的数字)，num_samples为每次循环时的数量（需要大于state维数）
    gfa_policy = solve(gfa_solver, mdp)
    return gfa_policy
end

mutable struct LinearGlobalFunctionApproximator_M{W <: AbstractArray} <: GlobalFunctionApproximator
    weights::W
end

function GlobalApproximationValueIteration.fit!(lgfa::LinearGlobalFunctionApproximator_M, dataset_input::AbstractMatrix{T},
              dataset_output::AbstractArray{T}) where T
    # TODO: Since we are ASSIGNING to weights here, does templating even matter? Does the struct even matter?
    dataset_input = hcat(dataset_input[:,2],dataset_input[:,3],dataset_input[:,12:end])
    lgfa.weights = llsq(dataset_input, dataset_output, bias=false)
    #@save "C:/Users/Mers/Desktop/linear_global_Approximator.bson" lgfa #path 
    @save joinpath(pwd(), "linear_global_Approximator.bson") lgfa #path 
    #lgfa.weights = llsq(dataset_input, dataset_output, bias=false)
end

function GlobalApproximationValueIteration.compute_value(lgfa::LinearGlobalFunctionApproximator_M, v::AbstractArray{T}) where T
    v = vcat(v[2],v[3],v[12:end])
    return dot(lgfa.weights, v)
end
=#
#=
#Nonlinear
mutable struct NonlinearGlobalFunctionApproximator_M{M,O,L} <: GlobalFunctionApproximator
    model::M
    optimizer::O
    loss::L
end

function GlobalApproximationValueIteration.fit!(ngfa::NonlinearGlobalFunctionApproximator_M, dataset_input::AbstractMatrix{T},
              dataset_output::AbstractArray{T}) where T
    # Create loss function with loss type
    loss(x, y) = ngfa.loss(ngfa.model(x), y)

    # NOTE : Minibatch update; 1 update to model weights
    # data = repeated((param(transpose(dataset_input)), param(transpose(dataset_output))), 1)
    data = repeated((transpose(dataset_input), transpose(dataset_output)), 1)
    Flux.train!(loss, params(ngfa.model), data, ngfa.optimizer)
    @save "C:/Users/Mers/Desktop/nonlinear_global_Approximator.bson" ngfa #path 
end

function GlobalApproximationValueIteration.compute_value(ngfa::NonlinearGlobalFunctionApproximator_M, state_vector::AbstractArray{T}) where T
    return ngfa.model(state_vector)[1]
end

function nonlinear_global_policy(mdp::UnderlyingMDP, continue_train::Bool)
    if continue_train == true
        #@load "C:/Users/Mers/Desktop/nonlinear_global_Approximator.bson" nonlin_gfa #path 
        ngfa = load("C:/Users/Mers/Desktop/nonlinear_global_Approximator.bson", MultilaneAS)[:ngfa]
        #load(x, init=Main)
    else
        model = Chain(
                Dense(110, 128, sigmoid),
                Dense(128, 64, sigmoid),
                Dense(64, 32, sigmoid), 
                Dense(32, 16, sigmoid),
                Dense(16, 1)
                )
        opt = ADAM(0.005)
        ngfa = NonlinearGlobalFunctionApproximator_M(model, opt, Flux.mse)
    end
    gfa_solver = GlobalApproximationValueIterationSolver(ngfa; num_samples=120, max_iterations=1, verbose=true, is_mdp_generative=true, n_generative_samples=1)
    #max_iterations为外围循环次数(需要是一个很大的数字)，num_samples为每次循环时的数量（需要大于state维数）
    gfa_policy = solve(gfa_solver, mdp)
    return gfa_policy
end

=#

#POMDPOW related
function POMDPs.pdf(a::Random.SamplerTrivial, b::Scene)
    return 1.0
end
#pdf(::Random.SamplerTrivial{POMDPModelTools.Deterministic{EntityScene{VehicleState, VehicleDef, Int64}}, EntityScene{VehicleState, VehicleDef, Int64}}, ::EntityScene{VehicleState, VehicleDef, Int64})


#MCTS
POMDPs.action(p::MCTSPlanner, s) = first(action_info(p, s[1]))


#POMCP
#=
function BasicPOMCP.search(p::POMCPPlanner, b::Vector{MLState}, t::BasicPOMCP.POMCPTree, info::Dict)
    all_terminal = true
    nquery = 0
    start_us = CPUtime_us()
    for i in 1:p.solver.tree_queries
        nquery += 1
        if CPUtime_us() - start_us >= 1e6*p.solver.max_time
            break
        end
        s = rand(p.rng, b)
        if !POMDPs.isterminal(p.problem, s)
            available_action_index = actionindexes(p.problem, actions(p.problem, s))
            t.children[1] = actionindexes(p.problem, actions(p.problem, s))
            t.n = sizehint!(zeros(Int, length(available_action_index)), 1000)
            t.v = sizehint!(zeros(Float64, length(available_action_index)), 1000)
            BasicPOMCP.simulate(p, s, POMCPObsNode(t, 1), p.solver.max_depth)
            all_terminal = false
        end
    end
    info[:search_time_us] = CPUtime_us() - start_us
    info[:tree_queries] = nquery

    if all_terminal
        throw(AllSamplesTerminal(b))
    end

    h = 1
    best_node = first(t.children[h])
    best_v = t.v[best_node]
    @assert !isnan(best_v)
    for node in t.children[h][2:end]
        if t.v[node] >= best_v
            best_v = t.v[node]
            best_node = node
        end
    end

    return t.a_labels[best_node]
end
=#