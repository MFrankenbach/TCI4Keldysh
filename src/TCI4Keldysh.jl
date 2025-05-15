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

@compile_workload begin
    T = 1.
    N_MF = 100
    ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * π * T
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")

    G        = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=1, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    precompute_all_values(G)
    
    Σ_calc_aIE = 1 ./ ω_fer .+ 0*im
    ωconvMat_a = [
        0 -1  0;
        0  0  1;
        -1  0 -1;
        1  1  0;
    ]
    ωconvMat_K2a = [
        sum(view(ωconvMat_a, [2,3], [1,2]), dims=1);
        view(ωconvMat_a, [1,4], [1,2])
    ]
    N_K2_bos, N_K2_fer = 50, 50
    ω_bos = (collect(-N_K2_bos:N_K2_bos) * (2.)      ) * π * T
    ω_fer = (collect(-N_K2_fer:N_K2_fer-1) * (2.) .+ 1.) * π * T
    ωs_ext=(ω_bos, ω_fer)
    K2a_data = [ TCI4Keldysh.compute_K2r_symmetric_estimator("MF", PSFpath, ("Q23", "1", "3dag"), Σ_calc_aIE; T, ωs_ext, ωconvMat=ωconvMat_K2a, flavor_idx=i) for i in 1:2]

    
end

end # module TCI4Keldysh
