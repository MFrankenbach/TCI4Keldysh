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

# ========== Keldysh

"""
Overload to create small correlator with 2^D peaks
"""
function multipeak_correlator_KF(
    ωs_ext::NTuple{D,Vector{Float64}}, 
    ωdisc::Float64=1.0
    ; 
    T::Float64=5.0,
    name::Vector{String}=["2^D-peak correlator"], 
    sigmak  ::Vector{Float64}=[1.0],
    γ       ::Float64=1.0,
    broadening_kwargs...
    ) where {D}
    @assert abs(ωdisc) > 1.e-8 "ωdisc should not be zero"
    Adiscs = fill(ones(Float64, fill(2, D)...), factorial(D+1))
    ωdisc_vec = [-abs(ωdisc), abs(ωdisc)]
    return multipeak_correlator_KF(ωs_ext, Adiscs, ωdisc_vec; T=T, name=name, sigmak=sigmak, γ=γ)
end

"""
Create Keldysh FullCorrelator with given (D+1)! PSFs (Adiscs) and ωdisc.
"""
function multipeak_correlator_KF(
    ωs_ext::NTuple{D,Vector{Float64}},
    Adiscs::Vector{Array{Float64, D}},
    ωdisc::Vector{Float64}
    ; 
    T::Float64=5.0,
    name::Vector{String}=["multipeak correlator"], 
    sigmak  ::Vector{Float64}=[1.0],
    γ       ::Float64=1.0,
    broadening_kwargs...
    ) where{D}

    ωconvMat = if D==1
            TCI4Keldysh.ωconvMat_K1()
        elseif D==2
            TCI4Keldysh.channel_trafo_K2("a", false)
        elseif D==3
            TCI4Keldysh.channel_trafo("a")
        end
    
    print("Loading stuff: ")
    @time begin
    perms = permutations(collect(1:D+1))
    isBos = BitVector(ntuple(i -> (i==1 && mod(D,2)==0), D+1))
    @assert length(Adiscs)==factorial(D+1) "Wrong number of PSFs"
    end
    print("Creating Broadened PSFs: ")
    function get_Acont_p(i, p)
        ωs_int, _, _ = _trafo_ω_args(ωs_ext, cumsum(ωconvMat[p[1:D],:], dims=1))
        return BroadenedPSF(ωdisc, Adiscs[i], sigmak, γ; ωconts=(ωs_int...,), broadening_kwargs...)
    end
    @time Aconts = [get_Acont_p(i, p) for (i,p) in enumerate(perms)]

    return FullCorrelator_KF(Aconts; T, isBos, ωs_ext, ωconvMat, name=name)
end