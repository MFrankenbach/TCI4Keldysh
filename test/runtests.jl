using Revise
using Test
using TCI4Keldysh
using HDF5
using QuanticsTCI

# define some utility fcs for testing:
include("utils4tests.jl")

# test broadening:
include("test_broaden_logGauss.jl")
include("test_broaden_lin.jl")
include("test_broaden.jl")
include("test_broaden_mp.jl")

# test correlators:
include("test_PartialCorrelators_reg.jl")
include("test_FullCorrelators_reg.jl")
include("test_FullCorrelators_KF.jl")

#test miscellaneous:
include("test_qtci_PSF.jl")
include("test_utils.jl")