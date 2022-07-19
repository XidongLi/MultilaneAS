mutable struct PhysicalParam #mutable for test 
	dt::Float64 #时间
	search_t::Float64 
	w_car::Float64
	l_car::Float64
	v_0::Float64 #egocar 初速度
	w_lane::Float64 #旧库默认值为4，新库默认值为3 ， 这里用的是3
    brake_limit::Float64 # always positive b_max
    nb_lanes::Int
	lane_length::Float64 #路长
	vy_0::Float64 #given lateral velocity
	nb_cars::Int
	adjustment_acceleration::Float64   #ego car 的加速度 用于search时
	p_appear::Float64 # probability of a new car appearing if the maximum number are not on the road 用于生成新车 p_appear=0.5,
    appear_clearance::Float64 # minimum clearance for a car to appear 用于生成新车 35.0, # appear_clearance
    vel_sigma::Float64 # std of new car speed about v0 用于生成新车 0.5
	sim_nb::Int
	road::Roadway
end

function PhysicalParam(nb_lanes::Int, road::Roadway;
						dt::Float64=0.1,
						search_t::Float64 = 1.5,
						w_car::Float64=1.8, #旧库默认值为1.8
						l_car::Float64=4., #新库默认值也为4
						v_0::Float64=31.,
						w_lane::Float64=3.,
						lane_length::Float64=12.,
                        brake_limit::Float64=8., # coefficient of friction of about 0.8
						vy_0::Float64=2.0 ,
						nb_cars::Int=10 ,
						adjustment_acceleration::Float64=1.0 ,#旧包给的值1.0
						p_appear = 0.5 ,
						appear_clearance = 35.0 ,
						vel_sigma = 0.5,
						sim_nb = 1000
)

	#assert(v_fast >= v_med)
	#assert(v_med >= v_slow)
	#assert(v_fast > v_slow)
    return PhysicalParam(dt, search_t, w_car, l_car, v_0, w_lane, brake_limit, nb_lanes, lane_length, vy_0, nb_cars, adjustment_acceleration, p_appear, appear_clearance, vel_sigma, sim_nb, road)
end
#=
function exchange_time!(phy::PhysicalParam)
	time = phy.dt
	phy.dt = phy.search_t
	phy.search_t = time
end
=#
"""
Return the lateral coordinate of the the leftmost lane
"""
function most_left_lane(phy::PhysicalParam)
	return (phy.nb_lanes-1) * phy.w_lane
end

"""
Return the lane index according to lateral coordiante
"""
function lane_number(phy::PhysicalParam,y::Float64)  #没问题
	for i in 1:phy.nb_lanes
		if (y - (i-1)*phy.w_lane) < 0.01
			#println("y: ",y)
			#println(i)
			return i,i
		elseif abs(y - i*phy.w_lane) < 0.01
			#println("y: ",y)
			#println(i)
			return i+1,i+1
		elseif y < i*phy.w_lane
			#println("y: ",y)
			#println(i)
			return i,i+1
		end
	end
end

function lane_number(w_lane::Float64, nb_lanes::Int, y::Float64)  
	for i in 1:nb_lanes
		if (y - (i-1)*w_lane) < 0.01
			#println("y: ",y)
			#println(i)
			return i,i
		elseif abs(y - i*w_lane) < 0.01
			#println("y: ",y)
			#println(i)
			return i+1,i+1
		elseif y < i*w_lane
			#println("y: ",y)
			#println(i)
			return i,i+1
		end
	end
end

"""
Return a Pair{Int} of lanes that the car will occupy at some point in the time step (both lanes could be the same)

Recall that a car can occupy at most two lanes on a single time step
"""
function occupation_lanes(phy::PhysicalParam, y::Float64, lc::Int)  #没问题
	lanes = lane_number(phy, y)
	if lanes[1] == lanes[2]
		return lanes[1],lanes[1]+lc
	else
		return lanes
    end
end
#=
"""
Returns true if cars at y1 and y2 occupy the same lane
"""
function occupation_overlap(y1::Float64, y2::Float64)
    return abs(y1-y2) < 1.0 || ceil(y1) == floor(y2) || floor(y1) == ceil(y2)
end
=#

"""
Return true if cars will occupy the same lane at some time in the time step
"""
function occupation_overlap(phy::PhysicalParam, y1::Float64, y2::Float64, l1::Int, l2::Int)
    lanes1 = occupation_lanes(phy, y1, l1)
    lanes2 = occupation_lanes(phy, y2, l2)
    return lanes1[1] in lanes2 || lanes1[2] in lanes2
end

"""
Return target lane
"""
function target_lane(phy::PhysicalParam, y::Float64, l::Int)  #没问题
	lanes = lane_number(phy, y)
    if lanes[1] == lanes[2]
		return lanes[1]+l
	else
		if l>0
			return maximum(lanes)
		else
			return minimum(lanes)
		end
	end
end

function target_lane(w_lane::Float64, nb_lanes::Int, y::Float64, l::Int)  
	lanes = lane_number(w_lane, nb_lanes, y)
    if lanes[1] == lanes[2]
		return lanes[1]+l
	else
		if l>0
			return maximum(lanes)
		else
			return minimum(lanes)
		end
	end
end



