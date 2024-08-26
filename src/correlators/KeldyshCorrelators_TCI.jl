#=
Compute Keldysh correlators @ TCI by pointwise evaluation.
=#

"""
Compress Keldysh Correlator at given contour index.
* iK::Int linear index ranging from 1:2^D
"""
function compress_FullCorrelator_pointwise(GF::FullCorrelator_KF{D}, iK::Int; qtcikwargs...) where {D}
    R_f = log2(length(GF.ωs_ext[1]))
    @assert isinteger(R_f) "Need the same number of grid points of the form 2^R in each direction"
    R = round(Int, R_f)
    @assert all(2^R .== length.(GF.ωs_ext)) "Need the same number of grid points of the form 2^R in each direction"
    @assert 1<=iK<=2^D

    function f(idx::Vararg{Int,D})
        return evaluate(GF, idx...; iK=iK)        
    end

    kwargs_dict = Dict(qtcikwargs...)
    cutoff = haskey(kwargs_dict, :tolerance) ? kwargs_dict[:tolerance]*1.e-3 : 1.e-15
    KFev = FullCorrEvaluator_KF(GF, iK; cutoff=cutoff)

    qtt, _, _ = quanticscrossinterpolate(ComplexF64, KFev, ntuple(_ -> 2^R, D); qtcikwargs...)

    return qtt
end