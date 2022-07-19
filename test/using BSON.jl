using BSON: load
using MultilaneAS
ngfa = load("C:/Users/Mers/Desktop/log/qnetwork.bson", MultilaneAS)[:qnetwork]
println("a")