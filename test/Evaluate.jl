import MultilaneAS
using AutomotiveSimulator
using AutomotiveVisualization
using Reel
using POMDPs
using Random
using ParticleFilters
using Logging
using ARDESPOT
using POMCPOW
using BasicPOMCP

nb_lanes = 4
roadway = gen_straight_roadway(4, 20000.0)  
pp = MultilaneAS.PhysicalParam(nb_lanes,roadway,lane_length=20000.0) 
pp.vel_sigma = 0.0 
pp.dt = 0.1
pp.sim_nb = 1000

_discount = 0.95
rmodel = MultilaneAS.NoCrashRewardModel()
behaviors = MultilaneAS.standard_uniform(pp, correlation=true) # 更改这个数值
scene = Scene([                     
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
MultilaneAS.evaluation(pomdp, 10000) #compare DQN, MCTS, Simple



pomcpow_solver = POMCPOWSolver(max_time = 1.0, criterion=MaxUCB(20.0))
pomcp_solver = POMCPSolver(max_time = 1.0)
despot_simple_solver = DESPOTSolver(bounds=IndependentBounds(MultilaneAS.simple_lower, MultilaneAS.upper;check_terminal=true),default_action=MultilaneAS.default_action,K=100)
despot_dqn_solver = DESPOTSolver(bounds=IndependentBounds(MultilaneAS.simple_lower, MultilaneAS.dqn_upper;check_terminal=true),default_action=MultilaneAS.default_action,K=100)
MultilaneAS.evaluation(pomdp, 100, [pomcpow_solver, pomcp_solver, despot_simple_solver, despot_dqn_solver])


