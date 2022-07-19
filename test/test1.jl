import MyAS
using AutomotiveSimulator
using AutomotiveVisualization
using Reel
using POMDPs
using Random
using ParticleFilters

nb_lanes = 4
roadway = gen_straight_roadway(4, 20000.0)  

pp = MyAS.PhysicalParam(nb_lanes,roadway,lane_length=20000.0) #2.=>col_length=8
_discount = 1.0
rmodel = MyAS.NoCrashRewardModel()
#behaviors = MyAS.standard_uniform(correlation=0.75)
behaviors = MyAS.standard_uniform(correlation=true)

dmodel = MyAS.NoCrashIDMMOBILModel(pp, behaviors)
pomdp = MyAS.NoCrashPOMDP{typeof(rmodel)}(dmodel, rmodel, _discount, true);

scene = Scene([                     #根据roadway的宽度给出各个车的y值 ， 并确定初速度
    Entity(VehicleState(VecSE2(60.0,3.0,0.0), roadway, 31.0), VehicleDef(), 1),
    Entity(VehicleState(VecSE2(90.0,0.0,0.0), roadway, 31.0), VehicleDef(), 2),
    Entity(VehicleState(VecSE2(90.0,3.0,0.0), roadway, 31.0), VehicleDef(), 3),
    Entity(VehicleState(VecSE2(90.0,6.0,0.0), roadway, 31.0), VehicleDef(), 4),
    Entity(VehicleState(VecSE2(30.0,0.0,0.0), roadway, 31.0), VehicleDef(), 5),
    Entity(VehicleState(VecSE2(30.0,3.0,0.0), roadway, 31.0), VehicleDef(), 6),
    Entity(VehicleState(VecSE2(30.0,6.0,0.0), roadway, 31.0), VehicleDef(), 7),
    Entity(VehicleState(VecSE2(10.0,0.0,0.0), roadway, 31.0), VehicleDef(), 8),
    Entity(VehicleState(VecSE2(10.0,3.0,0.0), roadway, 31.0), VehicleDef(), 9),
    Entity(VehicleState(VecSE2(10.0,6.0,0.0), roadway, 31.0), VehicleDef(), 10)
])

models = dmodel.models

initial_state = MyAS.MLState(scene, models)
updater = MyAS.BehaviorParticleUpdater(pomdp)
belief = MyAS.initialize_belief(updater, initial_state.phy_state)
scenes = MyAS.simulate(pomdp, initial_state, belief, updater)

#=
veh_1 = get_by_id(scene, 1)
#camera = StaticCamera(position=VecE2(100.0,0.0), zoom=4.75, canvas_height=100)
camera = TargetFollowCamera(1, zoom=5.)
snapshot = render([roadway, scene], camera=camera)
idoverlay = IDOverlay(scene=scene, color=colorant"black", font_size=20, y_off=1.)
snapshot = render([roadway, scene, idoverlay], camera=camera)

timestep = 0.1
nticks = 1000
animation = roll(fps=1.0/timestep, duration=nticks*timestep) do t, dt
    i = Int(floor(t/dt)) + 1
    update_camera!(camera, scenes[i])
    idoverlay.scene = scenes[i]
    renderables = [roadway, scenes[i], idoverlay]
    render(renderables, camera=camera)
end
write("output.gif", animation) # Write to a gif file


#action_test1 = MyAS.NoCrashActionSpace(pomdp)
#action_test2 = MyAS.actions(pomdp, initial_state)



#observation(pomdp, initial_state)


#mean = MyAS.mean(belief)
#std = MyAS.stds(belief)
#MyAS.update(updater, belief, MyAS.ASAction(0.0, 0.0), scene)
#a = 1
#MyAS.MyParticleBelief{ParticleCollection{MyAS.Idm_MobilParam}}()

=#

