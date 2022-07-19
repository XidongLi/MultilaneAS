import MultilaneAS
using AutomotiveSimulator
using AutomotiveVisualization
using Reel
using POMDPs
using Random
using ParticleFilters
using Logging


nb_lanes = 4
roadway = gen_straight_roadway(4, 20000.0)  

pp = MultilaneAS.PhysicalParam(nb_lanes,roadway,lane_length=20000.0) #2.=>col_length=8
pp.vel_sigma = 0.0 #取消随机性，用于测试
pp.dt = 0.1
pp.sim_nb = 1000
#_discount = 1.0
_discount = 0.95
rmodel = MultilaneAS.NoCrashRewardModel()
behaviors = MultilaneAS.standard_uniform(pp, correlation=true)



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
initial_state = MultilaneAS.relaxed_initial_state(pomdp)
scene = initial_state.phy_state
pomdp.dmodel.initial_phy = scene

updater = MultilaneAS.BehaviorParticleUpdater(pomdp)
updater.policy = MultilaneAS.DespotPolicy_dqn(pomdp)
updater.nb_sims = 100
belief = MultilaneAS.initialize_belief(updater, scene)


models = MultilaneAS.Initial_models(MultilaneAS.DespotModel(MultilaneAS.MLAction(0,0), pomdp, 0, belief, updater), initial_state.inter_state)
for i in 1:9
    models[i+1] = initial_state.inter_state[i+1]
    
end
#initial_state = MultilaneAS.MLState(scene, models)


calback = (MultilaneAS.MLcollision_check(), MultilaneAS.ReachGoalCallback(), MultilaneAS.BeliefUpdate())

nticks = pomdp.dmodel.phys_param.sim_nb
#scenes = [Scene(Entity{VehicleState, VehicleDef, Int64}, length(initial_state.phy_state)) for i=1:nticks+1]
#scenes = [Scene(Entity, length(scene)) for i=1:nticks+1]
#scenes = Array{Scene{Entity}, 1}(undef, nticks+1)
scenes = AutomotiveSimulator.simulate(scene, roadway, models, nticks, pp.dt; callbacks = calback)

#n = simulate!(scene, roadway, models, nticks, pp.dt, scenes, repeat([action],1000), rng = Random.GLOBAL_RNG, callbacks = calback) 
#scenes = scenes[1:(n+1)]
#EntityAction

#close(io)

veh_1 = get_by_id(scene, 1)
#camera = StaticCamera(position=VecE2(100.0,0.0), zoom=4.75, canvas_height=100)
camera = TargetFollowCamera(1, zoom=5.)
snapshot = render([roadway, scene], camera=camera)
idoverlay = IDOverlay(scene=scene, color=colorant"black", font_size=20, y_off=1.)
snapshot = render([roadway, scene, idoverlay], camera=camera)

timestep = 0.1
nticks = length(scenes)
animation = roll(fps=1.0/timestep, duration=nticks*timestep) do t, dt
    i = Int(floor(t/dt)) + 1
    update_camera!(camera, scenes[i])
    idoverlay.scene = scenes[i]
    renderables = [roadway, scenes[i], idoverlay]
    render(renderables, camera=camera)
end
write("output.gif", animation) # Write to a gif file