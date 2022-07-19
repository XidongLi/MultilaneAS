

#=
struct IDMMOBILBehavior <: BehaviorModel
	models::ASModel
	idx::Int
end

*(a::Real, b::IDMMOBILBehavior) = IDMMOBILBehavior(a*models, 0)
function IDMMOBILBehavior(idm::String,mobil::String,id)
	return IDMMOBILBehavior(ASModel(IDMParam(idm), MLLateralDriverModel(0.0,2.0), MOBILParam(mobil)),id)
end

=#


