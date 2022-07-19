#abstract type AbstractNoCrashProblem end
#=
1.Multilane
2.ASModel.mlon.a_max
=#
#=mutable struct Idm_MobilParam
	a_max::Float64 #max  comfy acceleration 论文中的Max acceleration
	d_cmf::Float64 #max comfy brake speed  论文中的Desired deceleration
	T::Float64 #desired safety time headway  论文中的Desired time gap 
	v_des::Float64 #desired speed
	s_min::Float64 #minimum headway (e.g. if x is less than this then you crashed) 论文中的Jam distance g_0
	del::Float64 #'accel exponent'  公式1中的delta，默认为4？
    politeness::Float64 #politeness factor
	safe_decel::Float64 #safe braking value
	advantage_threshold::Float64 #minimum accel
    id::Int #used for belief 2-car_number
end              σ
=#
mutable struct  DespotModel <: DriverModel{MLAction}
    action::MLAction
    pomdp::MLPOMDP
    cumulative_reward::Float64
    #state::MLState
    belief::AbstractParticleBelief
    bu::Updater
end
mutable struct EgoidmModel <: DriverModel{MLAction}
    mlon::LaneFollowingDriver
    action::MLAction
end
function EgoidmModel()
    return EgoModel(IntelligentDriverModel(σ = phy.vel_sigma,T=1.5,v_des=33.3,s_min=2.0,a_max=1.4,d_cmf=2.0), MLAction(0.0, 0.0))
end
mutable struct EgoModel <: DriverModel{MLAction}
    action::MLAction
end
function EgoModel()
    return EgoModel(MLAction(0.0, 0.0))
end

function EgoModel(phy::PhysicalParam, y::Float64)
    lane_ego = lane_number(phy, y)
    if lane_ego[1] != lane_ego[2]
        return EgoModel(MLAction(2.0, 0.0))
    else
        return EgoModel(MLAction(0.0, 0.0))
    end
end
function EgoModel(model::EgoModel, a::MLAction)
    return EgoModel(a)
end

mutable struct ASModel <: DriverModel{MLAction}  
    mlon::LaneFollowingDriver 
    mlat::LateralDriverModel 
    mlane::LaneChangeModel 
end
function ASModel(model::ASModel, Param::Vector)
    
    idm = IntelligentDriverModel(
        σ = model.mlon.σ, 
        T = Param[1],
        v_des = Param[2],
        s_min = Param[3],
        a_max = Param[4],
        d_cmf = Param[5]
        )

    mobil = MOBIL(
        mlon = idm, 
        safe_decel = Param[6],
        politeness = Param[7],
        advantage_threshold = Param[8]
        )
    return ASModel(idm, model.mlat, mobil)
end

function ASModel(phy::PhysicalParam, Param::Vector)
    idm = IntelligentDriverModel(
        σ = phy.vel_sigma, 
        T = Param[1],
        v_des = Param[2],
        s_min = Param[3],
        a_max = Param[4],
        d_cmf = Param[5]
        )
    mlat = MLLateralDriverModel(0.0, phy.vy_0)
    mobil = MOBIL(
        mlon = idm, 
        safe_decel = Param[6],
        politeness = Param[7],
        advantage_threshold = Param[8]
        )
    
    return ASModel(idm, mlat, mobil)
end

#=
function pa_Deletemodel(id::Int)
    return Idm_MobilParam(0,0,0,0,0,0,0,0,0,id)
end
function state_to_pa(model::ASModel, id::Int)
    return Idm_MobilParam(model.mlon.a_max,
                            model.mlon.d_cmf,
                            model.mlon.T,
                            model.mlon.v_des,
                            model.mlon.s_min,
                            model.mlon.δ,
                            model.mlane.politeness,
                            model.mlane.safe_decel,
                            model.mlane.advantage_threshold,
                            id
                        )
    
end
function pa_to_Model(Param::Idm_MobilParam,phy::PhysicalParam)
    if Param.v_des ==0
        return DeleteModel()
    #elseif Param.id ==1
        #return EgoModel()
    else
        idm = IntelligentDriverModel(
        T=Param.T,
        v_des=Param.v_des,
        s_min=Param.s_min,
        a_max=Param.a_max,
        d_cmf=Param.d_cmf
        )

        mobil = MOBIL(
        mlon = idm, 
        safe_decel = Param.safe_decel,
        politeness = Param.politeness,
        advantage_threshold = Param.advantage_threshold
        )
        return ASModel(idm, MLLateralDriverModel(0.0,phy.vy_0), mobil)
    end
end
=#
#=
*(a::Real, b::Idm_MobilParam) = [a * b.a_max, 
                                a * b.d_cmf, 
                                a * b.T,
                                a * b.v_des,
                                a * b.s_min,
                                a * b.del,
                                a * b.politeness,
                                a * b.safe_decel,
                                a * b.advantage_threshold]
=#

*(a::Real, b::ASModel) = [a * b.mlon.T,
                            a * b.mlon.v_des,
                            a * b.mlon.s_min,
                            a * b.mlon.a_max,
                            a * b.mlon.d_cmf, 
                            a * b.mlane.safe_decel,
                            a * b.mlane.politeness,
                            a * b.mlane.advantage_threshold]

#=
-(a::Idm_MobilParam, b::Real) = [a.a_max - b, 
                                a.d_cmf - b, 
                                a.T - b,
                                a.v_des - b,
                                a.s_min - b,
                                a.del - b,
                                a.politeness - b,
                                a.safe_decel - b,
                                a.advantage_threshold - b]
=#
-(a::ASModel, b::Real) = [a.mlon.T - b,
                            a.mlon.v_des - b,
                            a.mlon.s_min - b,
                            a.mlon.a_max - b, 
                            a.mlon.d_cmf - b, 
                            a.mlane.safe_decel - b,
                            a.mlane.politeness - b,
                            a.mlane.advantage_threshold - b]
#=
/(a::Idm_MobilParam, b::Real) = [a.a_max / b, 
                                a.d_cmf / b, 
                                a.T / b,
                                a.v_des / b,
                                a.s_min / b,
                                a.del / b,
                                a.politeness / b,
                                a.safe_decel / b,
                                a.advantage_threshold / b]   
=#
/(a::ASModel, b::Real) = [a.mlon.T / b,
                            a.mlon.v_des / b,
                            a.mlon.s_min / b,
                            a.mlon.a_max / b, 
                            a.mlon.d_cmf / b, 
                            a.mlane.safe_decel / b,
                            a.mlane.politeness / b,
                            a.mlane.advantage_threshold / b]                                 
                                
#=                      
+(a::Idm_MobilParam, b::Vector) = Idm_MobilParam(a.a_max+b[1], 
                                    a.d_cmf+b[2], 
                                    a.T+b[3],
                                    a.v_des+b[4],
                                    a.s_min+b[5],
                                    a.del+b[6],
                                    a.politeness+b[7],
                                    a.safe_decel+b[8],
                                    a.advantage_threshold + b[9],
                                    a.id)
=#                                    
+(a::ASModel, b::Vector) = ASModel(a::ASModel, 
                                    [a.mlon.T + b[1],
                                    a.mlon.v_des + b[2],
                                    a.mlon.s_min + b[3],
                                    a.mlon.a_max + b[4], 
                                    a.mlon.d_cmf + b[5], 
                                    a.mlane.safe_decel + b[6],
                                    a.mlane.politeness + b[7],
                                    a.mlane.advantage_threshold + b[8] ] 
                                    )

+(b::Vector, a::ASModel) = [a.mlon.T + b[1],
                            a.mlon.v_des + b[2],
                            a.mlon.s_min + b[3],
                            a.mlon.a_max + b[4], 
                            a.mlon.d_cmf + b[5], 
                            a.mlane.safe_decel + b[6],
                            a.mlane.politeness + b[7],
                            a.mlane.advantage_threshold + b[8] ] 
                                    
#=
-(a::Idm_MobilParam, b::Vector) = [a.a_max - b[1], 
                                    a.d_cmf - b[2], 
                                    a.T - b[3],
                                    a.v_des - b[4],
                                    a.s_min - b[5],
                                    a.del - b[6],
                                    a.politeness - b[7],
                                    a.safe_decel - b[8],
                                    a.advantage_threshold - b[9]
                                    ]
=#
-(a::ASModel, b::Vector) = [a.mlon.T - b[1],
                            a.mlon.v_des - b[2],
                            a.mlon.s_min - b[3],
                            a.mlon.a_max - b[4], 
                            a.mlon.d_cmf - b[5], 
                            a.mlane.safe_decel - b[6],
                            a.mlane.politeness - b[7],
                            a.mlane.advantage_threshold - b[8]
                            ]
#=                                    
-(a::Idm_MobilParam, b::Idm_MobilParam) = Idm_MobilParam(a.a_max - b.a_max, 
                                            a.d_cmf-b.d_cmf, 
                                            a.T-b.T,
                                            a.v_des-b.v_des,
                                            a.s_min-b.s_min,
                                            a.del-b.del,
                                            a.politeness-b.politeness,
                                            a.safe_decel-b.safe_decel,
                                            a.advantage_threshold - b.advantage_threshold,
                                            0)
=#
-(a::ASModel, b::ASModel) = ASModel(a, 
                                    [a.mlon.T - b.mlon.T,
                                    a.mlon.v_des - b.mlon.v_des,
                                    a.mlon.s_min - b.mlon.s_min,
                                    a.mlon.a_max - b.mlon.a_max, 
                                    a.mlon.d_cmf - b.mlon.d_cmf, 
                                    a.mlane.safe_decel - b.mlane.safe_decel,
                                    a.mlane.politeness - b.mlane.politeness,
                                    a.mlane.advantage_threshold - b.mlane.advantage_threshold ] 
                                    )
#=
+(a::Idm_MobilParam, b::Idm_MobilParam) = Idm_MobilParam(a.a_max+b.a_max, 
                                            a.d_cmf+b.d_cmf, 
                                            a.T+b.T,
                                            a.v_des+b.v_des,
                                            a.s_min+b.s_min,
                                            a.del+b.del,
                                            a.politeness+b.politeness,
                                            a.safe_decel+b.safe_decel,
                                            a.advantage_threshold + b.advantage_threshold,
                                            0)
=#
+(a::ASModel, b::ASModel) = ASModel(a, 
                                    [a.mlon.T + b.mlon.T,
                                    a.mlon.v_des + b.mlon.v_des,
                                    a.mlon.s_min + b.mlon.s_min,
                                    a.mlon.a_max + b.mlon.a_max, 
                                    a.mlon.d_cmf + b.mlon.d_cmf, 
                                    a.mlane.safe_decel + b.mlane.safe_decel,
                                    a.mlane.politeness + b.mlane.politeness,
                                    a.mlane.advantage_threshold + b.mlane.advantage_threshold ] 
                                    )

function IDMParam(s::AbstractString, phy::PhysicalParam)#k_spd=1,δ=4(旧代码和新包的默认值),d_max还没看旧包里是多少，也没看论文里是多少
	if lowercase(s) == "cautious"
		return IntelligentDriverModel(σ = phy.vel_sigma,T=2.0,v_des=27.8,s_min=4.0,a_max=0.8,d_cmf=1.0)
	elseif lowercase(s) == "normal"
		return IntelligentDriverModel(σ = phy.vel_sigma,T=1.5,v_des=33.3,s_min=2.0,a_max=1.4,d_cmf=2.0)
	elseif lowercase(s) == "aggressive"
		return IntelligentDriverModel(σ = phy.vel_sigma,T=1.0,v_des=38.9,s_min=0.0,a_max=2.0,d_cmf=3.0)
	else
		error("No such idm phenotype: \"$(s)\"")
	end
end

function MOBILParam(idm::String,mobil::String, phy::PhysicalParam)
	if lowercase(mobil) == "cautious"
		return MOBIL(mlon=IDMParam(idm, phy),safe_decel=1.0,politeness=1.0,advantage_threshold=0.2)
	elseif lowercase(mobil) == "normal"
		return MOBIL(mlon=IDMParam(idm, phy),safe_decel=2.0,politeness=0.5,advantage_threshold=1.0)
	elseif lowercase(mobil) == "aggressive"
		return MOBIL(mlon=IDMParam(idm, phy),safe_decel=3.0,politeness=0.0,advantage_threshold=0.0)
    end
end

function idm_gstar(p::ASModel,v::Float64,dv::Float64) #used in collision_check
    p = p.mlon
    return p.s_min + p.T*v + v*dv/(2*sqrt(p.a_max*p.d_cmf))
end
function idm_gstar(p::DriverModel,v::Float64,dv::Float64) #Egocar采用了normal的参数 #used in collision_check for Ego car
    return 2.0 + 1.5*v + v*dv/(2*sqrt(1.4*2.0))
end



function ASModel(idm::String, mobil::String, phy::PhysicalParam)
    return ASModel(IDMParam(idm, phy), MLLateralDriverModel(0.0,2.0), MOBILParam(idm, mobil, phy))
end
function AutomotiveSimulator.observe!(driver::EgoidmModel, scene::Scene{Entity{S, D, I}}, roadway::Roadway, egoid::I) where {S, D, I}
    observe!(driver.mlon, scene, roadway, egoid)
end
Base.rand(driver::EgoidmModel; rng::AbstractRNG = Random.GLOBAL_RNG) = MLAction(rand(rng, driver.mlat), 0.0)
Base.rand(rng::AbstractRNG, driver::EgoidmModel) = MLAction(rand(rng, driver.mlat), 0.0)
function AutomotiveSimulator.observe!(driver::ASModel, scene::Scene{Entity{S, D, I}}, roadway::Roadway, egoid::I) where {S, D, I}
	#vehicle_index = findfirst(egoid, scene) #不需要
    ego = scene[egoid]

    
    lanes = lane_number(lane_width(roadway, ego), length(roadway.segments[1].lanes), posgy(ego))
    #lanes = lane_number(phy, posgy(ego))
    if lanes[1] == lanes[2]   #在中线上，决定是否做lane change
    #if abs(posg(scene[vehicle_index]).y/3 - round(posg(scene[vehicle_index]).y/3)) > 1e-3 #不是 0 3 6 9 在做lane change 因此不需要Mobil来决定是否位移
        observe!(driver.mlane, scene, roadway, egoid)
        #posg(scene[vehicle_index]).y % 3  > 1e-3
    #if posg(scene[vehicle_index]).y % 3 != 0  #3是路宽，也应为一个输入值 因为除不尽的原因应该为小于0.1  
    else
        
    end

    lane_change_action = rand(driver.mlane)  

    if lane_change_action.dir == DIR_MIDDLE
        target_lane = get_lane(roadway, ego)
    elseif lane_change_action.dir == DIR_LEFT
        target_lane = leftlane(roadway, ego)
    else
        target_lane = rightlane(roadway, ego)
    end
    fore_original = find_neighbor(scene, roadway, ego, 
                        lane=get_lane(roadway, ego), 
                        targetpoint_ego=VehicleTargetPointFront(), 
                        targetpoint_neighbor=VehicleTargetPointRear())
    fore_target = find_neighbor(scene, roadway, ego, 
                        lane=target_lane, 
                        targetpoint_ego=VehicleTargetPointFront(), 
                        targetpoint_neighbor=VehicleTargetPointRear())
    if fore_original.Δs >= fore_target.Δs
        fore = fore_target
    else
        fore = fore_original
    end
    track_lateral!(driver.mlat, lane_change_action.dir)
    track_longitudinal!(driver.mlon, scene, roadway, egoid, fore)
	#for test
	if egoid == 1
		print("stop")
	end

    driver
end
#syntax: optional positional arguments must occur at end
#Base.rand(rng::AbstractRNG = Random.GLOBAL_RNG, driver::ASModel) = MLAction(rand(rng, driver.mlat), rand(rng, driver.mlon).a)
Base.rand(driver::ASModel; rng::AbstractRNG = Random.GLOBAL_RNG) = MLAction(rand(rng, driver.mlat), rand(rng, driver.mlon).a)
Base.rand(rng::AbstractRNG, driver::ASModel) = MLAction(rand(rng, driver.mlat), rand(rng, driver.mlon).a)
mutable struct DeleteAction 
end
mutable struct DeleteModel <: DriverModel{DeleteAction}
    DeleteAction::DeleteAction
end
function DeleteModel()
    return DeleteModel(DeleteAction())
end
function AutomotiveSimulator.observe!(driver::DeleteModel, scene::Scene{Entity{S, D, I}}, roadway::Roadway, egoid::I) where {S, D, I}
end
Base.rand(driver::DeleteModel; rng::AbstractRNG = Random.GLOBAL_RNG) = DeleteAction()
Base.rand(rng::AbstractRNG, driver::DeleteModel) = DeleteAction()
#=
function set_std!(model::ASModel, vel_sigma::Float64, timestep::Float64)
    model.mlon.σ = vel_sigma/timestep
    
end

function set_std!(model::DeleteModel, vel_sigma::Float64, timestep::Float64)
end
=#  