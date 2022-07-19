import MyAS
using AutomotiveSimulator
using AutomotiveVisualization
using Reel
using POMDPs
using Random
using ParticleFilters
using Statistics

nb_lanes = 4
roadway = gen_straight_roadway(4, 20000.0)  

pp = MyAS.PhysicalParam(nb_lanes,roadway,lane_length=20000.0) #2.=>col_length=8
pp.dt = 1.5
pp.sim_nb = 10
_discount = 1.0
rmodel = MyAS.NoCrashRewardModel()
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
dmodel.models[3] = MyAS.ASModel("cautious","cautious")
dmodel.models[4] = MyAS.ASModel("cautious","cautious")
dmodel.models[5] = MyAS.ASModel("normal","normal")
dmodel.models[6] = MyAS.ASModel("normal","normal")
dmodel.models[7] = MyAS.ASModel("normal","normal")
dmodel.models[8] = MyAS.ASModel("aggressive","aggressive")
dmodel.models[9] = MyAS.ASModel("aggressive","aggressive")
dmodel.models[10] = MyAS.ASModel("aggressive","aggressive")

models = dmodel.models

initial_state = MyAS.MLState(scene, models)
updater = MyAS.BehaviorParticleUpdater(pomdp)
updater.nb_sims = 10
belief = MyAS.initialize_belief(updater, initial_state.phy_state)

for i in 1:9
    
    belief.particles[i].particles[1] = MyAS.state_to_pa(models[i+1],i+1)
    
end  

belief_agg = zeros(Float64, 10,9)
for i in 1:10
    for j in 1:9
        belief_agg[i,j] = MyAS.aggressiveness(behaviors, belief.particles[j].particles[i])
    end
end    
#result = mean(belief_agg, dims=1)
println(belief_agg)

scenes = MyAS.simulate(pomdp, initial_state, belief, updater)

real_agg = zeros(Float64, 0)
for i in 2:10
    append!( real_agg, MyAS.aggressiveness(behaviors, MyAS.state_to_pa(models[i],i)) )
end
println(real_agg)

belief_agg = zeros(Float64, 10,9)
for i in 1:10
    for j in 1:9
        belief_agg[i,j] = MyAS.aggressiveness(behaviors, belief.particles[j].particles[i])
    end
end    
println(belief_agg)
result = mean(belief_agg, dims=1)
println(result)
#=
f=open("C:/Users/Mers/Desktop/belief_record.txt","a")  #path
    write(f, "$belief_agg")
    close(f)
    =#
#println(belief_agg)
#=for i in 1:1000
    belief.particles[1].particles[i].a_max = 0.0
end
belief.particles[1].particles[1].a_max = 1.0
belief.particles[1].particles[2].a_max = 2.0
belief.particles[1].particles[3].a_max = 3.0
belief.particles[1].particles[4].a_max = 4.0
belief.particles[1].particles[5].a_max = 5.0
mean01 = MyAS.mean(belief) 

std01 = MyAS.stds(belief)
=#
a = 1


