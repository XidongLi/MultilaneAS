import MyAS
using Random



behaviors_1 = MyAS.standard_uniform()
behaviors_2 = MyAS.standard_uniform(correlation=true)
behaviors_3 = MyAS.standard_uniform(correlation=0.75)
#behaviors_4 = MyAS.standard_uniform(correlation=1.0)

test_rand_1 = rand(Random.GLOBAL_RNG, behaviors_1, 1)
test_rand_2 = rand(Random.GLOBAL_RNG, behaviors_2, 1)
aggressiveness01 = MyAS.aggressiveness(behaviors_2, test_rand_2)
test_rand_3 = rand(Random.GLOBAL_RNG, behaviors_3, 1)


behaviors_4 = MyAS.CopulaIDMMOBIL(behaviors_3.min, behaviors_3.max, 0.999999)
test_rand_4 = rand(Random.GLOBAL_RNG, behaviors_4, 1)
a = 1


