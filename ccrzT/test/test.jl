using AutomotiveSimulator
roadway = gen_straight_roadway(4, 2000.0) 
scene = Scene([                     #根据roadway的宽度给出各个车的y值 ， 并确定初速度
    Entity(VehicleState(VecSE2(50.0,6.0,0.0), roadway, 31.0), VehicleDef(), 1),
    Entity(VehicleState(VecSE2(60.0,3.0,0.0), roadway, 31.0), VehicleDef(), 2),
    Entity(VehicleState(VecSE2(70.0,4.6,0.0), roadway, 31.0), VehicleDef(), 3)
    
])

fore_M = find_neighbor(scene, roadway, scene[1], targetpoint_ego=VehicleTargetPointFront(), targetpoint_neighbor=VehicleTargetPointRear())
println(fore_M)