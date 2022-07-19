function relaxed_initial_state(pomdp::NoCrashProblem)
    initial_state = rand(POMDPs.initialstate(pomdp;train = false))
    roadway = get_phy(pomdp).road
    initial_state.phy_state[1] = Entity(initial_state.phy_state[1], VehicleState(initial_state.phy_state[1].state, roadway, 0.0))
    @save joinpath(pwd(), "initial_state.bson") initial_state
    return initial_state
end
#=
scene = Scene([                     #根据roadway的宽度给出各个车的y值 ， 并确定初速度
    Entity(VehicleState(VecSE2(60.0,0.0,0.0), roadway, 31.0), VehicleDef(), 1),
    Entity(VehicleState(VecSE2(-250.0,0.0,0.0), roadway, 31.0), VehicleDef(), 2),
    Entity(VehicleState(VecSE2(-250.0,0.0,0.0), roadway, 31.0), VehicleDef(), 3),
    Entity(VehicleState(VecSE2(-250.0,0.0,0.0), roadway, 31.0), VehicleDef(), 4),
    Entity(VehicleState(VecSE2(-250.0,0.0,0.0), roadway, 31.0), VehicleDef(), 5),
    Entity(VehicleState(VecSE2(-250.0,0.0,0.0), roadway, 31.0), VehicleDef(), 6),
    Entity(VehicleState(VecSE2(-250.0,0.0,0.0), roadway, 31.0), VehicleDef(), 7),
    Entity(VehicleState(VecSE2(-250.0,0.0,0.0), roadway, 31.0), VehicleDef(), 8),
    Entity(VehicleState(VecSE2(-250.0,0.0,0.0), roadway, 31.0), VehicleDef(), 9),
    Entity(VehicleState(VecSE2(-250.0,0.0,0.0), roadway, 31.0), VehicleDef(), 10)
])

models = Dict{Int, DriverModel}(1 => EgoidmModel())
for i in 1:length-1
    push!(models, i+1 => DeleteModel())
end

calback = MLcollision_check()
scene = AutomotiveSimulator.simulate(scene, roadway, models, nticks, pp.dt; callbacks = calback)[-1]

MLState(scene, models)
=#