import MyAS
using AutomotiveSimulator
using AutomotiveVisualization
using Reel

nb_lanes = 4
pp = MyAS.PhysicalParam(nb_lanes,lane_length=20000.0) #2.=>col_length=8
_discount = 1.
nb_cars=10

rmodel = MyAS.NoCrashRewardModel()

dmodel = MyAS.NoCrashIDMMOBILModel(pp)

mdp = MyAS.NoCrashMDP{typeof(rmodel)}(dmodel, rmodel, _discount, true)
roadway = gen_straight_roadway(4, 20000.0)  # 200m long straight roadway with 4 lane ， DEFAULT_LANE_WIDTH = 3.0，根据lateral speed 给出

scene = Scene([                     #根据roadway的宽度给出各个车的y值 ， 并确定初速度
    Entity(VehicleState(VecSE2(60.0,3.0,0.0), roadway, 31.0), VehicleDef(), 1),
    Entity(VehicleState(VecSE2(50.0,0.0,0.0), roadway, 31.0), VehicleDef(), 2),
    Entity(VehicleState(VecSE2(50.0,3.0,0.0), roadway, 31.0), VehicleDef(), 3),
    Entity(VehicleState(VecSE2(50.0,6.0,0.0), roadway, 31.0), VehicleDef(), 4),
    Entity(VehicleState(VecSE2(70.0,0.0,0.0), roadway, 31.0), VehicleDef(), 5),
    Entity(VehicleState(VecSE2(70.0,3.0,0.0), roadway, 31.0), VehicleDef(), 6),
    Entity(VehicleState(VecSE2(70.0,6.0,0.0), roadway, 31.0), VehicleDef(), 7),
])


#=
scene = Scene([                     #根据roadway的宽度给出各个车的y值 ， 并确定初速度
    Entity(VehicleState(VecSE2(100.0,0.0,0.0), roadway, 31.0), VehicleDef(), 1),
    Entity(VehicleState(VecSE2(100.0,3.0,0.0), roadway, 31.0), VehicleDef(), 2),
    Entity(VehicleState(VecSE2(100.0,6.0,0.0), roadway, 31.0), VehicleDef(), 3),
    Entity(VehicleState(VecSE2(100.0,9.0,0.0), roadway, 31.0), VehicleDef(), 4),
    Entity(VehicleState(VecSE2(150.0,0.0,0.0), roadway, 31.0), VehicleDef(), 5),
    Entity(VehicleState(VecSE2(150.0,3.0,0.0), roadway, 31.0), VehicleDef(), 6),
    Entity(VehicleState(VecSE2(150.0,6.0,0.0), roadway, 31.0), VehicleDef(), 7),
    Entity(VehicleState(VecSE2(150.0,9.0,0.0), roadway, 31.0), VehicleDef(), 8),
    Entity(VehicleState(VecSE2(50.0,0.0,0.0), roadway, 31.0), VehicleDef(), 9),
    Entity(VehicleState(VecSE2(50.0,3.0,0.0), roadway, 31.0), VehicleDef(), 10),
])
=#

models = Dict{Int, DriverModel}(
    1 => dmodel.models[1] ,
    2 => dmodel.models[2] ,
    3 => dmodel.models[3] ,
    4 => dmodel.models[4] ,
    5 => dmodel.models[5] ,
    6 => dmodel.models[6] ,
    7 => dmodel.models[7] ,
)

#=
models = Dict{Int, MyAS.ASModel}(
    1 => MyAS.ASModel(MyAS.IDMParam("normal"), MyAS.MLLateralDriverModel(0.0,2.0), MyAS.MOBILParam("normal")),
    2 => dmodel.behaviors.models[1].models ,
    3 => dmodel.behaviors.models[2].models ,
    4 => dmodel.behaviors.models[3].models ,
    5 => dmodel.behaviors.models[4].models ,
    6 => dmodel.behaviors.models[5].models ,
    7 => dmodel.behaviors.models[6].models ,
    8 => dmodel.behaviors.models[7].models ,
    9 => dmodel.behaviors.models[8].models ,
    10 => dmodel.behaviors.models[9].models ,
)
=#
#print(models)

veh_1 = get_by_id(scene, 1)
#camera = StaticCamera(position=VecE2(100.0,0.0), zoom=4.75, canvas_height=100)
camera = TargetFollowCamera(1, zoom=5.)
snapshot = render([roadway, scene], camera=camera)
idoverlay = IDOverlay(scene=scene, color=colorant"black", font_size=20, y_off=1.)
snapshot = render([roadway, scene, idoverlay], camera=camera)

timestep = 0.1
nticks = 1000
scenes = MyAS.simulate(scene, roadway, models, nticks, timestep)
animation = roll(fps=1.0/timestep, duration=nticks*timestep) do t, dt
    i = Int(floor(t/dt)) + 1
    update_camera!(camera, scenes[i])
    idoverlay.scene = scenes[i]
    renderables = [roadway, scenes[i], idoverlay]
    render(renderables, camera=camera)
end
write("output.gif", animation) # Write to a gif file




#s = initial_state(mdp::NoCrashMDP, rng)
#=
solver=BehaviorSolver(NORMAL, true, rng)
is = MLState(0.0, 0.0, MyVehicleState[MyVehicleState(50, 1.0, 31, 0.0, NORMAL, 1)]) #Normal详见test_sets
    sim = HistoryRecorder(max_steps=steps, rng=rng)
    policy = solve(solver, mdp)
    hist = simulate(sim, mdp, policy, is)
    s = last(state_hist(hist))
    s.t = 0.0
    s.x = 0.0
=#