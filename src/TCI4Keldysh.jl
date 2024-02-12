module TCI4Keldysh

using LinearAlgebra
using LinearAlgebra.BLAS
using Match
using Printf
#using Tullio
using Dates
using QuanticsTCI
using QuanticsGrids
import QuanticsGrids: UnfoldingSchemes
import TensorCrossInterpolation as TCI
using ITensors

using FFTW
using StaticArrays
using Combinatorics
using MAT
using StridedViews
using Interpolations

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

macro VERBOSE(msgs)
    esc(:(if $(@__MODULE__).VERBOSE() print($msgs...) end))
end

include("types.jl")
include("utils/utils.jl")
include("utils/TCI_utils.jl")
include("broadening/broaden_1D.jl")
include("broadening/broaden_mp.jl")
include("broadening/broaden_logGauss.jl")
include("broadening/broaden_lin.jl")
include("broadening/broaden_Gauss.jl")
include("correlators/PartialCorrelator_reg.jl")
include("correlators/FullCorrelator_reg.jl")
include("improved_estimators/calc_SE.jl")
include("improved_estimators/symmetric_estimators_2D3D.jl")


end # module TCI4Keldysh
