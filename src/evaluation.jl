function evaluation(pomdp::NoCrashProblem, number::Int, solvers::Vector)
    policy_number = length(solvers) +2
    average_steps = zeros(Float64, policy_number)
    steps = zeros(Float64, policy_number)
    average_rewards = zeros(Float64, policy_number)
    rewards = zeros(Float64, policy_number)
    for iter in 1:number
        initial_state = relaxed_initial_state(pomdp)
        #@save "C:/Users/Mers/Desktop/initial_state.bson" initial_state
        @save joinpath(pwd(), "initial_state.bson") initial_state
        scene = initial_state.phy_state
        pomdp.dmodel.initial_phy = scene
        updater = BehaviorParticleUpdater(pomdp)
        updater.policy = solve(MCTSSolver(), UnderlyingMDP(pomdp))
        updater.nb_sims = 1
        belief = MultilaneAS.initialize_belief(updater, scene)
        models = MultilaneAS.Initial_models(MultilaneAS.DespotModel(MultilaneAS.MLAction(0,0), pomdp, 0, belief, updater))
        for index in 1:9
            models[index+1] = initial_state.inter_state[index+1]
            belief.particles[index].particles[1]= models[index+1]
        end
        calback = (MultilaneAS.MLcollision_check(), MultilaneAS.ReachGoalCallback(), MultilaneAS.BeliefUpdate())
        nticks = pomdp.dmodel.phys_param.sim_nb
        scenes = AutomotiveSimulator.simulate(scene, get_road(pomdp), models, nticks, get_dt(pomdp); callbacks = calback)
        veh_1 = get_by_id(scene, 1)
        camera = TargetFollowCamera(1, zoom=5.)
        snapshot = AutomotiveVisualization.render([get_road(pomdp), scene], camera=camera)
        idoverlay = IDOverlay(scene=scene, color=colorant"black", font_size=20, y_off=1.)
        snapshot = AutomotiveVisualization.render([get_road(pomdp), scene, idoverlay], camera=camera)
        timestep = 0.1
        nticks = length(scenes)
        animation = roll(fps=1.0/timestep, duration=nticks*timestep) do t, dt
            i = Int(floor(t/dt)) + 1
            update_camera!(camera, scenes[i])
            idoverlay.scene = scenes[i]
            renderables = [get_road(pomdp), scenes[i], idoverlay]
            AutomotiveVisualization.render(renderables, camera=camera)
        end
        write("MCTS$iter.gif", animation) # Write to a gif file
        reward = models[1].cumulative_reward
        rewards[1] = reward
        average_rewards[1] = average_rewards[1] + reward
        steps[1] = length(scenes)
        average_steps[1] = average_steps[1] + length(scenes)


        for policy_index in 1:policy_number - 1
            initial_state = load(joinpath(pwd(), "initial_state.bson"), MultilaneAS)[:initial_state]
            scene = initial_state.phy_state
            pomdp.dmodel.initial_phy = scene
            updater = BehaviorParticleUpdater(pomdp)
            if policy_index == 1
                DQNPolicy = load(joinpath(pwd(), "DQNPolicy.bson"), MultilaneAS)[:DQNPolicy]
                updater.policy = DQNPolicy
            else
                updater.policy = solve(solvers[policy_index-1], pomdp)
            end
            updater.nb_sims = 1000#正式测时改为1000
            if policy_index == 1
                belief = MultilaneAS.initialize_belief(updater, scene)
                @save joinpath(pwd(), "belief.bson") belief
            else
                belief = load(joinpath(pwd(), "belief.bson"), MultilaneAS)[:belief]
            end
            models = MultilaneAS.Initial_models(MultilaneAS.DespotModel(MultilaneAS.MLAction(0,0), pomdp, 0, belief, updater), initial_state.inter_state)
            for index in 1:9
                models[index+1] = initial_state.inter_state[index+1]
                #belief.particles[i].particles[1]= models[i+1]
            end
            calback = (MultilaneAS.MLcollision_check(), MultilaneAS.ReachGoalCallback(), MultilaneAS.BeliefUpdate())
            nticks = pomdp.dmodel.phys_param.sim_nb
            scenes = AutomotiveSimulator.simulate(scene, get_road(pomdp), models, nticks, get_dt(pomdp); callbacks = calback)
            veh_1 = get_by_id(scene, 1)
            camera = TargetFollowCamera(1, zoom=5.)
            snapshot = AutomotiveVisualization.render([get_road(pomdp), scene], camera=camera)
            idoverlay = IDOverlay(scene=scene, color=colorant"black", font_size=20, y_off=1.)
            snapshot = AutomotiveVisualization.render([get_road(pomdp), scene, idoverlay], camera=camera)
            timestep = 0.1
            nticks = length(scenes)
            animation = roll(fps=1.0/timestep, duration=nticks*timestep) do t, dt
                i = Int(floor(t/dt)) + 1
                update_camera!(camera, scenes[i])
                idoverlay.scene = scenes[i]
                renderables = [get_road(pomdp), scenes[i], idoverlay]
                AutomotiveVisualization.render(renderables, camera=camera)
            end
            write("policy$policy_index.$iter.gif", animation) # Write to a gif file
            reward = models[1].cumulative_reward
            rewards[policy_index+1] = reward
            average_rewards[policy_index+1] = average_rewards[policy_index+1] + reward
            steps[policy_index+1] = length(scenes)
            average_steps[policy_index+1] = average_steps[policy_index+1] + length(scenes)
    
        end
        f=open(joinpath(pwd(), "reward.txt"),"a")  #path
        write(f, "reward:$rewards\n")
        write(f, "steps:$steps\n")
        close(f)
    end
    average_rewards = average_rewards / number
    average_steps = average_steps / number
    f=open(joinpath(pwd(), "reward.txt"),"a")  #path
    write(f, "average reward:$average_rewards\n")
    write(f, "average steps:$average_steps\n")
    close(f)
end

function evaluation(pomdp::NoCrashProblem, number::Int)
    average_steps = zeros(Float64, 3)
    steps = zeros(Float64, 3)
    average_rewards = zeros(Float64, 3)
    rewards = zeros(Float64, 3)
    for iter in 1:number
        for i in 1:3
            if i == 1
                initial_state = relaxed_initial_state(pomdp)
                @save joinpath(pwd(), "initial_state.bson") initial_state
            else
                initial_state = load(joinpath(pwd(), "initial_state.bson"), MultilaneAS)[:initial_state]
            end
            scene = initial_state.phy_state
            pomdp.dmodel.initial_phy = scene
            updater = BehaviorParticleUpdater(pomdp)
            if i == 1
                DQNPolicy = load(joinpath(pwd(), "DQNPolicy.bson"), MultilaneAS)[:DQNPolicy]
                updater.policy = DQNPolicy
            elseif i ==2
                updater.policy = Simple(pomdp)
            else
                updater.policy = solve(MCTSSolver(), UnderlyingMDP(pomdp))
            end
            updater.nb_sims = 1
            belief = MultilaneAS.initialize_belief(updater, scene)
            models = MultilaneAS.Initial_models(MultilaneAS.DespotModel(MultilaneAS.MLAction(0,0), pomdp, 0, belief, updater), initial_state.inter_state)
            for index in 1:9
                models[index+1] = initial_state.inter_state[index+1]
                belief.particles[index].particles[1]= models[index+1]
            end
            calback = (MultilaneAS.MLcollision_check(), MultilaneAS.ReachGoalCallback(), MultilaneAS.BeliefUpdate())
            nticks = pomdp.dmodel.phys_param.sim_nb
            scenes = AutomotiveSimulator.simulate(scene, get_road(pomdp), models, nticks, get_dt(pomdp); callbacks = calback)
            reward = models[1].cumulative_reward
            rewards[i] = reward
            average_rewards[i] = average_rewards[i] + reward
            steps[i] = length(scenes)
            average_steps[i] = average_steps[i] + length(scenes)
    
        end
        f=open(joinpath(pwd(), "reward.txt"),"a")  #path
        write(f, "reward:$rewards\n")
        write(f, "steps:$steps\n")
        close(f)
    end
    average_rewards = average_rewards / number
    average_steps = average_steps / number
    f=open(joinpath(pwd(), "reward.txt"),"a")  #path
    write(f, "average reward:$average_rewards\n")
    write(f, "average steps:$average_steps\n")
    close(f)
end

function bound_evaluate(pomdp::NoCrashProblem, number::Int, bounds::Vector{Function})
    #=
    nb_lanes = 4
    roadway = gen_straight_roadway(4, 20000.0)  
    pp = MultilaneAS.PhysicalParam(nb_lanes,roadway,lane_length=20000.0) #2.=>col_length=8
    pp.vel_sigma = 0.0 #取消随机性，用于测试
    pp.dt = 0.1
    pp.sim_nb = 1000
    _discount = 0.95
    rmodel = MultilaneAS.NoCrashRewardModel()
    behaviors = MultilaneAS.standard_uniform(pp, correlation=0.75)
    scene = Scene([                     #根据roadway的宽度给出各个车的y值 ， 并确定初速度
        Entity(VehicleState(VecSE2(310.0,0.0,0.0), roadway, 31.0), VehicleDef(), 1),
        Entity(VehicleState(VecSE2(340.0,9.0,0.0), roadway, 31.0), VehicleDef(), 2),
        Entity(VehicleState(VecSE2(340.0,3.0,0.0), roadway, 31.0), VehicleDef(), 3),
        Entity(VehicleState(VecSE2(340.0,6.0,0.0), roadway, 31.0), VehicleDef(), 4),
        Entity(VehicleState(VecSE2(280.0,0.0,0.0), roadway, 31.0), VehicleDef(), 5),
        Entity(VehicleState(VecSE2(280.0,3.0,0.0), roadway, 31.0), VehicleDef(), 6),
        Entity(VehicleState(VecSE2(280.0,6.0,0.0), roadway, 31.0), VehicleDef(), 7),
        Entity(VehicleState(VecSE2(260.0,0.0,0.0), roadway, 31.0), VehicleDef(), 8),
        Entity(VehicleState(VecSE2(310.0,3.0,0.0), roadway, 31.0), VehicleDef(), 9),
        Entity(VehicleState(VecSE2(310.0,6.0,0.0), roadway, 31.0), VehicleDef(), 10)
    ])
    dmodel = MultilaneAS.NoCrashIDMMOBILModel(pp, behaviors, scene)
    pomdp = MultilaneAS.NoCrashPOMDP{typeof(rmodel)}(dmodel, rmodel, _discount, true)
    =#
    initial_state = MultilaneAS.relaxed_initial_state(pomdp)
end
#=
function evaluate_simple_different_dqn(pomdp::NoCrashProblem, number::Int)
    
    DQNPolicy1 = load("C:/Users/Mers/Desktop/DQNPolicy4.bson", MultilaneAS)[:DQNPolicy]
    DQNPolicy2 = load("C:/Users/Mers/Desktop/DQNPolicy.bson", MultilaneAS)[:DQNPolicy]
    #v = value(DQNPolicy, initial_state)
    #simple_lower(pomdp, ScenarioBelief([1=>initial_state], Random.GLOBAL_RNG, 0, [initial_state]))
    winner = zeros(Int, 3)
    values = Vector{Float64}(undef,3)
    for i in 1:number
        initial_state = rand(initialstate(pomdp))
        pomdp.dmodel.initial_phy = initial_state.phy_state
        values[1] = simple_lower(pomdp, ScenarioBelief([1=>initial_state], MemorizingSource(500, 90, Random.GLOBAL_RNG), 0, [initial_state]))
        values[2] = value(DQNPolicy1, initial_state)
        values[3] = value(DQNPolicy2, initial_state)
        if values[2] > values[1] && values[3] > values[1]
            max_index = argmax(values)
        else
            max_index = 1
        end
        #max_index = argmax(values)
        winner[max_index] = winner[max_index] + 1
        if i % 50 == 0
            println(i)
        end
        f=open("C:/Users/Mers/Desktop/bound_record.txt","a")  #path
        write(f, "bound：$values\n")
        close(f)
    end
    println(winner)
    f=open("C:/Users/Mers/Desktop/bound_record.txt","a")  #path
    write(f, "winner is ：$winner\n")
    close(f)
end

function evaluate_simple_dqn(pomdp::NoCrashProblem, number::Int) #compare simple with dqn
    
    DQNPolicy= load("C:/Users/Mers/Desktop/DQNPolicy.bson", MultilaneAS)[:DQNPolicy]
    #v = value(DQNPolicy, initial_state)
    #simple_lower(pomdp, ScenarioBelief([1=>initial_state], Random.GLOBAL_RNG, 0, [initial_state]))
    winner = zeros(Int, 2)
    values = Vector{Float64}(undef,2)
    for i in 1:number
        initial_state = rand(initialstate(pomdp))
        pomdp.dmodel.initial_phy = initial_state.phy_state
        values[1] = simple_lower(pomdp, ScenarioBelief([1=>initial_state], MemorizingSource(500, 90, Random.GLOBAL_RNG), 0, [initial_state]))
        values[2] = value(DQNPolicy, initial_state)
        max_index = argmax(values)
        winner[max_index] = winner[max_index] + 1
        f=open("C:/Users/Mers/Desktop/bound_record.txt","a")  #path
        write(f, "bound：$values\n")
        close(f)
        if i % 50 == 0
            println(i)
        end
    end
    println(winner)
    f=open("C:/Users/Mers/Desktop/bound_record.txt","a")  #path
    write(f, "winner is ：$winner\n")
    close(f)
end

function evaluate_dqn(pomdp::NoCrashProblem, number::Int)
    
    DQNPolicy1= load("C:/Users/Mers/Desktop/DQNPolicy1.bson", MultilaneAS)[:DQNPolicy] #change path
    DQNPolicy2= load("C:/Users/Mers/Desktop/DQNPolicy2.bson", MultilaneAS)[:DQNPolicy] #change path
    
    winner = zeros(Int, 2)
    values = Vector{Float64}(undef,2)
    for i in 1:number
        initial_state = rand(initialstate(pomdp))
        values[1] = value(DQNPolicy1, initial_state)
        values[2] = value(DQNPolicy2, initial_state)
        max_index = argmax(values)
        winner[max_index] = winner[max_index] + 1
        f=open("C:/Users/Mers/Desktop/bound_record.txt","a")  #path
        write(f, "bound：$values\n")
        close(f)
        if i % 50 == 0
            println(i)
        end
    end
    println(winner)
    f=open("C:/Users/Mers/Desktop/bound_record.txt","a")  #path
    write(f, "winner is ：$winner\n")
    close(f)
end

function evaluate_simple_global(pomdp::NoCrashProblem, number::Int)
    
    ngfa = load("C:/Users/Mers/Desktop/linear_global_Approximator.bson", MultilaneAS)[:lgfa]
    winner = zeros(Int, 2)
    values = Vector{Float64}(undef,2)
    for i in 1:number
        initial_state = rand(initialstate(pomdp))
        pomdp.dmodel.initial_phy = initial_state.phy_state
        values[1] = simple_lower(pomdp, ScenarioBelief([1=>initial_state], MemorizingSource(500, 90, Random.GLOBAL_RNG), 0, [initial_state]))
        values[2] = compute_value(ngfa, convert_featurevector(Vector{Float64}, initial_state, UnderlyingMDP(pomdp)))
        max_index = argmax(values)
        winner[max_index] = winner[max_index] + 1
        f=open("C:/Users/Mers/Desktop/bound_record.txt","a")  #path
        write(f, "bound：$values\n")
        close(f)
        if i % 50 == 0
            println(i)
        end
    end
    println(winner)
    f=open("C:/Users/Mers/Desktop/bound_record.txt","a")  #path
    write(f, "winner is ：$winner\n")
    close(f)
end
=#