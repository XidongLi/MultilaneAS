import MultilaneAS
using AutomotiveSimulator
using AutomotiveVisualization
using Reel
using POMDPs
using Random
using ParticleFilters
using POMDPModelTools
using GlobalApproximationValueIteration


nb_lanes = 4
roadway = gen_straight_roadway(4, 20000.0)  

pp = MultilaneAS.PhysicalParam(nb_lanes,roadway,lane_length=20000.0) #2.=>col_length=8
pp.vel_sigma = 0.0 #取消随机性，用于测试
pp.dt = 0.1
pp.sim_nb = 1000
#_discount = 1.0
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
pomdp = MultilaneAS.NoCrashPOMDP{typeof(rmodel)}(dmodel, rmodel, _discount, true);




updater = MultilaneAS.BehaviorParticleUpdater(pomdp)
updater.nb_sims = 10
belief = MultilaneAS.initialize_belief(updater, scene)


models = MultilaneAS.Initial_models(MultilaneAS.DespotModel(MultilaneAS.MLAction(0,0), pomdp, 0, belief, updater))

initial_state = MultilaneAS.MLState(scene, models)

calback = (MultilaneAS.MLcollision_check(), MultilaneAS.ReachGoalCallback(), MultilaneAS.BeliefUpdate())

nticks = pomdp.dmodel.phys_param.sim_nb

mdp = UnderlyingMDP(pomdp)
#MultilaneAS.linear_global_policy(mdp,false) #linear
#MultilaneAS.linear_global_policy(mdp,true) #linear, continue_train
#MultilaneAS.nonlinear_global_policy(mdp,false) #nonlinear
MultilaneAS.nonlinear_global_policy(mdp,true) #nonlinear, continue_train