using Revise
using Test
using TCI4Keldysh
using HDF5
using QuanticsTCI
import TensorCrossInterpolation as TCI

# define some utility fcs for testing:
include("utils4tests.jl")

# pass testing arguments with Pkg.test(;test_args::Vector{AbstractString}=ARGS)
println("==== Testing arguments: $ARGS")
args = [uppercase(a) for a in ARGS]

test_dir = joinpath(dirname(@__FILE__), "tests")

function test_broaden()
    include(joinpath(test_dir, "test_broaden.jl"))
    include(joinpath(test_dir, "test_broaden_logGauss.jl")) 
    include(joinpath(test_dir, "test_broaden_lin.jl")) 
    include(joinpath(test_dir, "test_broaden_mp.jl"))       
end

function test_utils()
    include(joinpath(test_dir, "test_utils.jl"))
    include(joinpath(test_dir, "test_TCI_utils.jl"))
end

function test_correlators()
    include(joinpath(test_dir, "test_PSF_corr_conv_pointwiseTCI.jl")) 
    include(joinpath(test_dir, "test_FullCorrelators_KF.jl"))  
    include(joinpath(test_dir, "test_PartialCorrelators_reg.jl"))     
    include(joinpath(test_dir, "test_FullCorrelators_reg.jl")) 
end

function test_SIE()
    include(joinpath(test_dir, "test_improved_estimators_TCI.jl")) 
    include(joinpath(test_dir, "test_improved_estimators.jl"))     
end

if isempty(args) || ("ALL" in args)
    for test in readdir(test_dir)
        if test[1:4] == "test"
            include(joinpath(test_dir, test))
        end
    end
elseif "POINTWISE" in args
# main test target
    test_broaden()
    test_utils()
    test_correlators()
    test_SIE()
elseif "TENSORTRAIN" in args
    test_broaden()
    test_utils()
    include(joinpath(test_dir, "test_qtci_PSF.jl"))                      
    include(joinpath(test_dir, "test_PSF_correlator_conversion_TCI.jl")) 
    include(joinpath(test_dir, "test_qtt_freqConversion.jl")) 
else
    # specific tests
    if "BROADEN" in args
        test_broaden()
    end
    if "UTIL" in args || "UTILS" in args
        test_utils()
    end
    if "SIE" in args
        include(joinpath(test_dir, "test_improved_estimators.jl"))     
    end
end

println("==== tests for $args DONE")