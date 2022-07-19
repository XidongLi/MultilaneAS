

@with_kw mutable struct WeightUpdateParams
    #smoothing::Float64 = 0.02 # value between 0 and 1, adds this fraction of the max to each entry in the vector
    wrong_lane_factor::Float64 = 0.1 #论文里为 0.2
    std::Float64 = 0.5
end