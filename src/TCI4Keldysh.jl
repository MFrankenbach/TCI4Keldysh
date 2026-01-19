module TCI4Keldysh

using PrecompileTools

@recompile_invalidations begin
    using LinearAlgebra
    using LinearAlgebra.BLAS
    using Match
    using Printf
    using JLD2
    using FileIO
    #using Tullio
    using Dates
    using Quantics
    using QuanticsTCI
    using QuanticsGrids
    import TensorCrossInterpolation as TCI
    using ITensors
    using ITensorMPS

    using Lehmann

    using FFTW
    using StaticArrays
    using Combinatorics
    using MAT
    using StridedViews
    using Interpolations
end

# macro to unlock debug mode
DEBUG() = false

macro DEBUG(expr, msgs)
    esc(:(if $(@__MODULE__).DEBUG() @assert($expr, $msgs...) end))
end


# macro to unlock timing of function evaluations
TIME() = false

macro TIME(expr, msgs)
    esc(:(if $(@__MODULE__).TIME() print($msgs..., "\t"); @time($expr) else $expr end))
end

# macro to unlock timing of function evaluations
VERBOSE() = false
# from very little (0) to a lot of (3) output
const VERBOSITY = Ref(2)
function SET_VERBOSITY(level::Int)
    VERBOSITY[] = level
end
GET_VERBOSITY() = VERBOSITY[]

macro VERBOSE(msgs)
    esc(:(if $(@__MODULE__).VERBOSE() print($msgs...) end))
end

# macro to unlock debugging of RAM issue
DEBUG_RAM() = false




include("TuckerDecomposition.jl")
include("utils/utils.jl")
include("utils/cubicspline.jl")
include("broadening/broaden_1D.jl")
include("broadening/broaden_mp.jl")
include("broadening/broaden_logGauss.jl")
include("broadening/broaden_lin.jl")
include("broadening/broaden_Gauss.jl")
include("correlators/PartialCorrelator_reg.jl")
include("correlators/FullCorrelator_reg.jl")
include("HierarchicalTucker.jl")
include("utils/TCI_utils.jl")
include("utils/gen_dummyinput.jl")
include("improved_estimators/calc_SE.jl")
include("improved_estimators/symmetric_estimators_2D3D.jl")

# TCI
include("correlators/PartialCorrelator_TCI.jl")
include("correlators/ImaginarytimeCorrelator.jl")
include("correlators/PartialCorrelator_pointwiseTCI.jl")
include("correlators/KeldyshCorrelators_TCI.jl")
include("improved_estimators/improved_estimator_TCI.jl")

end # module TCI4Keldysh
