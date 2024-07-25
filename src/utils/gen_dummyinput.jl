using Random
#=
Generate dummy PSFs/Correlators for trying things out and testing.
=#

"""
    function get_Adisc_multipeak_noano(D::Int; nωdisc::Int=5, ωdisc_min=1.e-6, ωdisc_max=1.e4)

Get PSF with multiple peaks.
"""
function get_Adisc_multipeak(D::Int; nωdisc::Int=5, ωdisc_min=1.e-3, ωdisc_max=1.e2, peakdens::Float64=1.0)
    ωdisc = exp.(range(log(ωdisc_min); stop=log(ωdisc_max), length=nωdisc))
    ωdisc = [reverse(-ωdisc); 0.; ωdisc]
    Adisc = zeros(Float64, ntuple(i -> (2*nωdisc + 1), D))
    Adisc[ntuple(i -> nωdisc+1, D)...] = 1.0
    rng = MersenneTwister(42)
    for i in eachindex(Adisc)
        if randn(rng) < peakdens
            Adisc[i] = randn(rng)
        end
    end
    return ωdisc, Adisc
end

"""
    function get_Adisc_multipeak_noano(D::Int; nωdisc::Int=5, ωdisc_min=1.e-6, ωdisc_max=1.e4)

Get PSF with multiple peaks, but none at zero, avoiding anomalous terms.
"""
function get_Adisc_multipeak_noano(D::Int; nωdisc::Int=5, ωdisc_min=1.e-3, ωdisc_max=1.e2, peakdens::Float64=1.0)
    ωdisc = exp.(range(log(ωdisc_min); stop=log(ωdisc_max), length=nωdisc))
    ωdisc = [reverse(-ωdisc); ωdisc]
    Adisc = zeros(Float64, ntuple(i -> (2*nωdisc + 1), D))
    # ensure that there is one nonzero value
    Adisc[ntuple(i -> nωdisc+1, D)...] = 1.0
    rng = MersenneTwister(42)
    for i in eachindex(Adisc)
        if randn(rng) < peakdens
            Adisc[i] = randn(rng)
        end
    end
    return ωdisc, Adisc
end

"""
    function multipeak_correlator_MF(npt::Int, R; beta::Float64=1.e3)

Get FullCorrelator_MF from multipeak Adisc
"""
function multipeak_correlator_MF(npt::Int, R; beta::Float64=1.e3, peakdens::Float64=1.0, nωdisc::Int=5)
    # grids
    T = 1.0/beta
    Rpos = R-1
    Nωcont_pos = 2^Rpos
    ωbos = π * T * collect(-Nωcont_pos:Nωcont_pos) * 2
    ωfer = π * T *(collect(-Nωcont_pos:Nωcont_pos-1) * 2 .+ 1)
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)

    D = npt-1
    Adiscs = Vector{Array{Float64, D}}(undef, factorial(npt))
    ωdisc = Vector{Float64}(undef, nωdisc)
    for i in 1:factorial(npt)
        ωdisc, Adiscs[i] = get_Adisc_multipeak(D; nωdisc=nωdisc, peakdens=peakdens)
    end
    ωs_ext = ntuple(i -> (i==1 ? ωbos : ωfer), D)
    Ops = TCI4Keldysh.dummy_operators(npt)
    isBos = (o -> length(o) == 3).(Ops)
    return TCI4Keldysh.FullCorrelator_MF(Adiscs, ωdisc; T=T, ωconvMat=ωconvMat, ωs_ext=ωs_ext, isBos=isBos, name=["$npt-multipeak correlator"])
end

"""
    function multipeak_tucker_decomp(npt::Int, R; beta::Float64=1.e3, peakdens::Float64=1.0, nωdisc::Int=5)

Get TuckerDecomposition from multipeak Adisc
"""
function multipeak_tucker_decomp(npt::Int, R; beta::Float64=1.e3, peakdens::Float64=1.0, nωdisc::Int=5)
    # grids
    T = 1.0/beta
    Rpos = R-1
    Nωcont_pos = 2^Rpos
    ωbos = π * T * collect(-Nωcont_pos:Nωcont_pos) * 2
    ωfer = π * T *(collect(-Nωcont_pos:Nωcont_pos-1) * 2 .+ 1)
    D = npt-1
    ωconvMat = diagm(ones(Int,D))

    # function PartialCorrelator_reg(T::Float64, formalism::String, Adisc::Array{Float64,D}, ωdisc::Vector{Float64}, ωs_ext::NTuple{D,Vector{Float64}}, ωconvMat::Matrix{Int}; is_compactAdisc::Bool=true) where {D}
    ωdisc, Adisc = get_Adisc_multipeak(D; nωdisc=nωdisc, peakdens=peakdens)
    ωs_ext = ntuple(i -> (i==1 ? ωbos : ωfer), D)
    Gp = TCI4Keldysh.PartialCorrelator_reg(T, "MF", Adisc, ωdisc, ωs_ext, ωconvMat; is_compactAdisc=false)
    return Gp.tucker
end
