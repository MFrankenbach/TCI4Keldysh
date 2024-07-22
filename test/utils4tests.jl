using Random 

function get_Adisc_δpeak_mp(idx_ω′s, Nωs_pos, D; 
    ωdisc_min = 1.e-6,
    ωdisc_max = 1.e4
    )
    
    ωdisc = exp.(range(log(ωdisc_min); stop=log(ωdisc_max), length=Nωs_pos))
    ωdisc = [reverse(-ωdisc); 0.; ωdisc]
    Adisc = zeros((ones(Int, D).*(Nωs_pos*2+1))...)
    Adisc[(Nωs_pos + 1 .+ idx_ω′s)...] = 1.
    
    return ωdisc, Adisc
end;

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


function get_ωcont(ωmax, Nωcont_pos)
    ωcont = collect(range(-ωmax, ωmax; length=Nωcont_pos*2+1))
    return ωcont
end

function HA_exact_corr_F1_F1dag(ω; u::Float64)
    return  1 / (im*ω - u^2/(im*ω))
end


function HA_exact_corr_Q1_F1dag(ω; u::Float64)
    return  u / (im*ω - u)
end
function HA_exact_corr_Q1_Q1dag(ω; u::Float64)
    return 2 * u^2 / (im*ω - u)
end


function HA_exact_selfenergy(ω; u::Float64)
    return  u + u^2 / (im*ω)
end