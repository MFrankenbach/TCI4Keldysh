using Revise
using Test
using TCI4Keldysh
using HDF5
using QuanticsTCI

# define some utility fcs for testing:
include("utils4tests.jl")

test_dir = joinpath(dirname(@__FILE__), "tests")
for test in readdir(test_dir)
    if test[1:4] == "test"
        include(joinpath(test_dir, test))
    end
end