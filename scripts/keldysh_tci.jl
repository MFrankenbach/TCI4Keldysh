using Plots
using BenchmarkTools
using Profile
using StatProfilerHTML
using QuanticsTCI
using Printf
using HDF5
using JSON
using LinearAlgebra
using Serialization
using Printf
import TensorCrossInterpolation as TCI
import QuanticsGrids as QG

using TCI4Keldysh
using Test

"""
β=2000.0 in Seung-Sup's data
"""
function default_T()
    return 1.0/(2000.0)
end

function plot_2pt_KeldyshCorrelator()
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    npt = 2
    D = npt-1
    Ops = ["F1", "F1dag"]
    T = default_T()

    R = 7
    ωmax = 1.0
    ωs_ext = TCI4Keldysh.KF_grid(ωmax, R, D)
    ωconvMat = reshape([1; -1], (2,1))
    γ, sigmak = TCI4Keldysh.default_broadening_γσ(T)
    KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops; T=T, ωs_ext=ωs_ext, flavor_idx=1, ωconvMat=ωconvMat, sigmak=sigmak, γ=γ, name="Kentucky fried chicken")

    data = TCI4Keldysh.precompute_all_values(KFC)
    @show size(data)

    # plot
    cont_idx = [1,2]
    p = TCI4Keldysh.default_plot()
    plot!(p, only(ωs_ext), real.(data[:, cont_idx...]); label="Re")
    plot!(p, only(ωs_ext), imag.(data[:, cont_idx...]); label="Im")
    savefig("KFC.png")
end

function plot_4pt_KeldyshCorrelator(γfac=30, sigmak_=0.1)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/4pt/")
    npt = 4
    D = npt-1
    Ops = ["F1", "F1dag", "F3", "F3dag"]
    T = default_T()

    R = 7
    ωmax = 1.0
    ωmin = -ωmax
    ωs_ext = ntuple(i -> collect(range(ωmin, ωmax; length=2^R)), D)
    ωconvMat = TCI4Keldysh.channel_trafo("a")
    # γ, sigmak = TCI4Keldysh.default_broadening_γσ(T)
    sigmak = [sigmak_]
    γ = γfac*T
    KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops; T=T, ωs_ext=ωs_ext, flavor_idx=1, ωconvMat=ωconvMat, sigmak=sigmak, γ=γ, estep=10000, name="Kentucky fried chicken")

    data = TCI4Keldysh.precompute_all_values(KFC)

    # plot
    keldysh_idx = [1,2,1,1]
    scfun = x -> log10(abs(x))
    heatmap(scfun.(data[:,:,div(length(ωs_ext[end]), 2), keldysh_idx...]))
    title!("4-pt Keldysh vertex k=$(Tuple(keldysh_idx)), γ/T=$γfac, σk=$sigmak_")
    savefig("KFC.png")

    plot(scfun.(data[:,div(length(ωs_ext[end]),2),div(length(ωs_ext[end]), 2), keldysh_idx...]))
    savefig("KFC1D.png")
end

"""
Compute error introduced in regular partial correlators when the kernels are truncated with via SVD.
"""
function svd_error_GF(R::Int, cutoff::Float64=1.e-15; ωmax::Float64=1.0)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/4pt/")
    npt = 4
    D = npt-1
    Ops = TCI4Keldysh.dummy_operators(npt)
    T = default_T()
    ωmin = -ωmax
    ωs_ext = ntuple(i -> collect(range(ωmin, ωmax; length=2^R)), D)
    ωconvMat = TCI4Keldysh.channel_trafo("a")
    g = 10.0 * T
    s = 0.2

    KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops; T=T, ωs_ext=ωs_ext, flavor_idx=1, ωconvMat=ωconvMat, sigmak=[s], γ=g, name="Kentucky fried chicken")
    # truncate
    for Gp in KFC.Gps[1:1]
        @time cutval = TCI4Keldysh.contract_KF_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center; cutoff=cutoff)
        @time ref = TCI4Keldysh.contract_KF_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)
        # compute error
        abs_err =  maximum(abs.(cutval .- ref))
        rel_err =  abs_err / maximum(abs.(ref))
        norm_err =  norm(cutval .- ref)
        printstyled("---- Errors: max(rel.)=$rel_err, max(abs.)=$abs_err, norm=$norm_err\n"; color=:green)
    end

end

"""
Check singular value distribution of kernels for different broadenings.
"""
function kernel_svd_ranks()
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/4pt/")
    npt = 4
    D = npt-1
    Ops = TCI4Keldysh.dummy_operators(npt)
    R = 9
    T = default_T()
    gammas = [5, 50, 500] .* T
    sigmas = [0.01, 0.1, 1.0]
    ωmin = -1.0
    ωmax = 1.0
    ωs_ext = ntuple(i -> collect(range(ωmin, ωmax; length=2^R)), D)
    ωconvMat = TCI4Keldysh.channel_trafo("a")
    perm_idx = 2
    for g in gammas
        for s in sigmas

            KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops; T=T, ωs_ext=ωs_ext, flavor_idx=1, ωconvMat=ωconvMat, sigmak=[s], γ=g, name="Kentucky fried chicken")
            Gp = KFC.Gps[perm_idx]
            legs = Gp.tucker.legs
            printstyled("---- For γ=$g, σ=$s:\n"; color=:blue)
            for l in legs
                _, S, _ = svd(l)
                S8 = S[S .>= 1.e-8]
                S10 = S[S .>= 1.e-10]
                S12 = S[S .>= 1.e-12]
                printstyled("  #singvals > 1.e-8: $(length(S8)) / $(length(S))\n"; color=:blue)
                printstyled("  #singvals > 1.e-10: $(length(S10)) / $(length(S))\n"; color=:blue)
                printstyled("  #singvals > 1.e-12: $(length(S12)) / $(length(S))\n\n"; color=:blue)
            end
            printstyled("--------\n"; color=:blue)
        end
    end
end


"""
To compare different pointwise evaluation methods.
"""
function time_pointwise_eval()
    # create correlator
    npt = 4
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/4pt")
    D = npt-1
    Ops = TCI4Keldysh.dummy_operators(npt)
    T = default_T()

    ωmax = 1.0
    ωmin = -ωmax
    R = 7
    ωs_ext = ntuple(i -> collect(range(ωmin, ωmax; length=2^R)), D)
    ωconvMat = if npt==4
            TCI4Keldysh.channel_trafo("t")
        elseif npt==3
            TCI4Keldysh.channel_trafo_K2("t", false)
        else
            TCI4Keldysh.ωconvMat_K1()
        end
    γ, sigmak = TCI4Keldysh.default_broadening_γσ(T)
    KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops; T=T, ωs_ext=ωs_ext, flavor_idx=1, ωconvMat=ωconvMat, sigmak=sigmak, γ=γ, name="Kentucky fried chicken")

    iK = 8
    KFev = TCI4Keldysh.FullCorrEvaluator_KF(KFC, iK; cutoff=1.e-10)
    function KFC_(idx::Vararg{Int,3})
        return TCI4Keldysh.evaluate(KFC, idx...; iK=iK)        
    end

    @btime $KFC_(rand(1:2^$R, 3)...)
    @btime $KFev(rand(1:2^$R, 3)...)

    v = rand(1:2^R, 3)
    @show abs(KFC_(v...) - KFev(v...)) / abs(KFev(v...))

    # profile
    # Profile.clear()
    # Profile.@profile begin
    #     dummy = zero(ComplexF64)
    #     for _ in 1:1000
    #         dummy += KFev(rand(1:2^R, 3)...)
    #     end
    #     println(dummy)
    # end
    # statprofilehtml()
end

function time_compress_FullCorrelator_KF(iK::Int; R=4, tolerance=1.e-3)
    npt = 4
    # PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/4pt")
    PSFpath = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg/4pt")
    D = npt-1
    Ops = TCI4Keldysh.dummy_operators(npt)
    T = TCI4Keldysh.dir_to_T(PSFpath)
    @show T

    ωmax = 1.0
    ωmin = -ωmax
    ωs_ext = ntuple(i -> collect(range(ωmin, ωmax; length=2^R)), D)
    ωconvMat = if npt==4
            TCI4Keldysh.channel_trafo("t")
        elseif npt==3
            TCI4Keldysh.channel_trafo_K2("t", false)
        else
            TCI4Keldysh.ωconvMat_K1()
        end
    # γ, sigmak = beta2000_broadening(T)
    γ, sigmak = TCI4Keldysh.default_broadening_γσ(T)
    KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops; T=T, ωs_ext=ωs_ext, flavor_idx=1, ωconvMat=ωconvMat, sigmak=sigmak, γ=γ, name="Kentucky fried chicken")

    t = @elapsed begin
        qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(KFC, iK; tolerance=tolerance, unfoldingscheme=:fused)
        end
    @show TCI4Keldysh.rank(qtt)
    printstyled("  Time for single component, tol=$(tolerance), R=$R: $t[s]\n"; color=:blue)
end

function test_compress_FullCorrelator_KF(npt::Int, iK; R=4, tolerance=1.e-6, channel::String="t")
    addpath = npt==4 ? "4pt/" : ""
    # PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg", addpath)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg", addpath)
    D = npt-1
    Ops = TCI4Keldysh.dummy_operators(npt)
    T = TCI4Keldysh.dir_to_T(PSFpath)

    ωmax = 1.0
    ωs_ext = TCI4Keldysh.KF_grid(ωmax, R, D)
    ωconvMat = if npt==4
            TCI4Keldysh.channel_trafo(channel)
        elseif npt==3
            TCI4Keldysh.channel_trafo_K2(channel, false)
        else
            TCI4Keldysh.ωconvMat_K1()
        end
    γ, sigmak = TCI4Keldysh.default_broadening_γσ(T)
    KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops; T=T, ωs_ext=ωs_ext, flavor_idx=1, ωconvMat=ωconvMat, sigmak=sigmak, γ=γ, name="Kentucky fried chicken")

    # reference
    data_ref = TCI4Keldysh.precompute_all_values(KFC)[fill(Colon(),D)..., TCI4Keldysh.KF_idx(iK,D)...]
    maxref = maximum(abs.(data_ref))
    @show maxref
    @show size(data_ref)

    # TCI
    qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(KFC, iK; tolerance=tolerance, unfoldingscheme=:fused)
    qttval = TCI4Keldysh.QTT_to_fatTensor(qtt, Base.OneTo.(fill(2^R, D)))
    @show TCI4Keldysh.rank(qtt)

    slice = fill(1:2^R, D)
    error = maximum(abs.(qttval .- data_ref[slice...]) ./ maxref)
    @assert error <= 3.0*tolerance
    printstyled("  Max error (tol=$tolerance): $error\n"; color=:blue)

    # plot
    plot_slice = (2^(R-1), 1:2^R, 1:2^R)
    heatmap(abs.(data_ref[plot_slice...]))
    savefig("foo.png")
end


function GFfilename(mode::String,
    xmin::Int,
    xmax::Int,
    tolerance::Float64,
    beta::Float64,
    ωmin::Float64,
    ωmax::Float64,
    iK::Int,
    γ::Float64,
    sigmak::Float64)

    ommin = @sprintf("%.2f", ωmin)
    ommax = @sprintf("%.2f", ωmax)
    sigmak_str = @sprintf("%.2f", sigmak)
    γ_str = @sprintf("%.2f", γ)

    str1 = "KF_timing_$(mode)_min=$(xmin)_max=$(xmax)_tol=$(TCI4Keldysh.tolstr(tolerance))_beta=$(beta)"
    str2 = "_omega" * ommin * "_to_" * ommax * "_iK=$iK" * "_broaden_γ=$(γ_str)_σ=$(sigmak_str)"

    return str1 * str2
end


function time_Γcore_KF(iK::Int, R=4, tolerance=1.e-6)
    channel = "a"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    T = TCI4Keldysh.dir_to_T(PSFpath)
    ωmax = 1.0
    D = 3
    γ, sigmak = beta2000_broadening(T)
    # γ, sigmak = TCI4Keldysh.default_broadening_γσ(T)
    flavor_idx = 1

    @time qtt = TCI4Keldysh.Γ_core_TCI_KF(
        PSFpath, R, iK, ωmax
        ; 
        sigmak=sigmak,
        γ=γ,
        T=T, ωconvMat=ωconvMat, flavor_idx=flavor_idx, tolerance=tolerance, unfoldingscheme=:interleaved,
        estep=20,
        emin=1.e-6,
        emax=1.e2
        )
end

function profile_Γcore_KF(iK::Int, R=3, tolerance=1.e-3)
    channel = "a"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    T = TCI4Keldysh.dir_to_T(PSFpath)
    ωmax = 1.0
    D = 3
    γ, sigmak = beta2000_broadening(T)
    # γ, sigmak = TCI4Keldysh.default_broadening_γσ(T)
    flavor_idx = 1
    
    Profile.clear()
    Profile.@profile begin
        @time qtt = TCI4Keldysh.Γ_core_TCI_KF(
            PSFpath, R, iK, ωmax
            ; 
            sigmak=sigmak,
            γ=γ,
            T=T, ωconvMat=ωconvMat, flavor_idx=flavor_idx, tolerance=tolerance, unfoldingscheme=:interleaved
            )
    end
    statprofilehtml()
end


function test_Γcore_KF(iK::Int, channel::String="a")
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    R = 6
    tolerance = 1.e-4
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    T = TCI4Keldysh.dir_to_T(PSFpath)
    ωmax = 1.0
    D = 3
    iK_tuple = TCI4Keldysh.KF_idx(iK, D)
    # γ, sigmak = beta2000_broadening(T)
    γ, sigmak = TCI4Keldysh.default_broadening_γσ(T)
    flavor_idx = 1

    # reference
    ωs_ext = TCI4Keldysh.KF_grid(ωmax, R, D)
    Σωgrid = TCI4Keldysh.KF_grid_fer(2*ωmax, R+1)
    Σ_ref = TCI4Keldysh.calc_Σ_KF_sIE_viaR(PSFpath, Σωgrid; T=T, flavor_idx=flavor_idx, sigmak, γ)
    Γcore_ref = TCI4Keldysh.compute_Γcore_symmetric_estimator(
        "KF",
        PSFpath*"4pt/",
        Σ_ref
        ;
        T,
        flavor_idx = flavor_idx,
        ωs_ext = ωs_ext,
        ωconvMat=ωconvMat,
        sigmak, γ
    )

    # tci
    qtt = TCI4Keldysh.Γ_core_TCI_KF(
        PSFpath, R, iK, ωmax
        ; 
        sigmak=sigmak,
        γ=γ,
        T=T, ωconvMat=ωconvMat, flavor_idx=flavor_idx, tolerance=tolerance
        )

    @show TCI.linkdims(qtt.tci)

    # compare
    Γcore_tci = TCI4Keldysh.QTT_to_fatTensor(qtt, Base.OneTo.(fill(2^R, D)))
    @show size(Γcore_tci)
    @show size(Γcore_ref)
    scfun = x -> log10(abs(x))
    diff = abs.(Γcore_tci .- Γcore_ref[1:2^R,:,:,iK_tuple...])
    maxref =  maximum(abs.(Γcore_ref))
    reldiff = diff ./ maxref
    @assert maximum(reldiff) < 3.0*tolerance
    amaxdiff = argmax(reldiff)[1]
    ref_slice = Γcore_ref[amaxdiff,:,:,iK_tuple...]
    tci_slice = Γcore_tci[amaxdiff,:,:]

    heatmap(scfun.(ref_slice))
    savefig("KFcore_ref.png")
    heatmap(scfun.(tci_slice))
    savefig("KFcore_tci.png")
    heatmap(scfun.(tci_slice .- ref_slice))
    savefig("diff.png")

    @show maximum(diff / maximum(abs.(Γcore_ref)))
    @show maximum(diff)
    @show maxref

end

function serialize_tt(qtt, outname::String, folder::String)
    R = qtt.grid.R
    fname_tt = joinpath(folder, outname*"_R=$(R)_qtt.serialized")
    serialize(fname_tt, qtt)
end

function deserialize_tt(R::Int, outname::String, folder::String)
    fname_tt = joinpath(folder, outname*"_R=$(R)_qtt.serialized")
    return deserialize(fname_tt)
end

"""
Store ranks and timings for computation of single keldysh components full 4-point correlators.
Can vary:
* iK : Keldysh component
* beta
* R
* tolerance
"""
function time_FullCorrelator_sweep(
        iK::Int, γ::Float64, sigmak::Float64, mode::String="R";
        tolerance=1.e-8, Rs=nothing, serialize_tts=false)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/4pt/")
    folder = "pwtcidata"
    beta = TCI4Keldysh.dir_to_beta(PSFpath)
    npt = 4
    times = []
    qttranks = []
    bonddims = []
    svd_kernel = true
    ωmax = 0.1
    ωmin = -ωmax
    flavor_idx = 1
    if mode=="R"
        Rs = isnothing(Rs) ? (5:8) : Rs
        # prepare output
        d = Dict()
        d["times"] = times
        d["ranks"] = qttranks
        d["bonddims"] = bonddims
        d["Rs"] = Rs
        d["tolerance"] = tolerance
        d["svd_kernel"] = svd_kernel
        d["gamma"] = γ
        d["sigmak"] = sigmak 
        d["beta"] = beta
        d["ommin"] = ωmin
        d["ommax"] = ωmax
        d["iK"] = iK
        d["PSFpath"] = PSFpath
        d["flavor"] = flavor_idx 
        outname = GFfilename(mode, first(Rs), last(Rs), tolerance, beta, ωmin, ωmax, iK, γ, sigmak)
        TCI4Keldysh.logJSON(d, outname, folder)

        for R in Rs
            
            # create correlator
            npt = 4
            D = npt-1
            Ops = ["F1", "F1dag", "F3", "F3dag"]
            T = 1.0/beta

            ωs_ext = TCI4Keldysh.KF_grid(ωmax, R, D)
            ωconvMat = TCI4Keldysh.channel_trafo("a")
            KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops; T=T, ωs_ext=ωs_ext, flavor_idx=flavor_idx, ωconvMat=ωconvMat, sigmak=[sigmak], γ=γ, name="Kentucky fried chicken")
            # create correlator END

            t = @elapsed begin
                dump_path = TCI4Keldysh.datadir()
                resume_path = TCI4Keldysh.datadir()
                qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(KFC, iK; dump_path=dump_path, resume_path=nothing, verbosity=2, tolerance=tolerance, unfoldingscheme=:interleaved)
            end
            push!(times, t)
            push!(qttranks, TCI4Keldysh.rank(qtt))
            push!(bonddims, TCI.linkdims(qtt.tci))
            TCI4Keldysh.updateJSON(outname, "times", times, folder)
            TCI4Keldysh.updateJSON(outname, "ranks", qttranks, folder)
            TCI4Keldysh.updateJSON(outname, "bonddims", bonddims, folder)

            if serialize_tts
                serialize_tt(qtt, outname, folder)
            end

            println(" ===== R=$R: time=$t, rankk(qtt)=$(TCI4Keldysh.rank(qtt))")
        end
    else
        error("Invalid mode $mode")
    end
end

function Γcore_filename(mode::String,
    xmin::Int,
    xmax::Int,
    tolerance::Float64,
    beta::Float64,
    ωmin::Float64,
    ωmax::Float64,
    iK::Int,
    γ::Float64,
    sigmak::Float64)

    ommin = @sprintf("%.2f", ωmin)
    ommax = @sprintf("%.2f", ωmax)
    sigmak_str = @sprintf("%.2f", sigmak)
    γ_str = @sprintf("%.2f", γ)

    str1 = "KF_gammacore_$(mode)_min=$(xmin)_max=$(xmax)_tol=$(TCI4Keldysh.tolstr(tolerance))_beta=$(beta)"
    str2 = "_omega" * ommin * "_to_" * ommax * "_iK=$iK" * "_broaden_γ=$(γ_str)_σ=$(sigmak_str)"

    return str1 * str2
end


function time_Γcore_KF_sweep(
    param_range, iK::Int, γ::Float64, sigmak::Float64, mode="R"; tolerance=1.e-8, serialize_tts=false
    )
    folder = "pwtcidata"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    beta = TCI4Keldysh.dir_to_beta(PSFpath)
    ωconvMat = TCI4Keldysh.channel_trafo("t")
    T = 1.0/beta
    times = []
    qttranks = []
    bonddims = []
    svd_kernel = true
    ωmax = 1.0
    ωmin = -1.0
    flavor_idx = 1
    if mode=="R"
        Rs = param_range
        # prepare output
        d = Dict()
        d["times"] = times
        d["ranks"] = qttranks
        d["bonddims"] = bonddims
        d["Rs"] = Rs
        d["tolerance"] = tolerance
        d["svd_kernel"] = svd_kernel
        d["numthreads"] = Threads.threadpoolsize()
        d["sigmak"] = sigmak 
        d["gamma"] = γ 
        d["beta"] = beta
        d["ommin"] = ωmin
        d["ommax"] = ωmax
        d["iK"] = iK
        d["PSFpath"] = PSFpath
        d["flavor"] = flavor_idx
        outname = Γcore_filename(mode, first(Rs), last(Rs), tolerance, beta, ωmin, ωmax, iK, γ, sigmak)
        TCI4Keldysh.logJSON(d, outname, folder)

        for R in Rs
            t = @elapsed begin
                dump_path = TCI4Keldysh.datadir()
                resume_path = nothing
                qtt = TCI4Keldysh.Γ_core_TCI_KF(
                    PSFpath, R, iK, ωmax
                    ; 
                    sigmak=[sigmak],
                    γ=γ,
                    T=T,
                    dump_path=dump_path,
                    resume_path=resume_path,
                    ωconvMat=ωconvMat,
                    flavor_idx=flavor_idx,
                    tolerance=tolerance,
                    unfoldingscheme=:interleaved
                    )
            end 
            push!(times, t)
            push!(qttranks, TCI4Keldysh.rank(qtt))
            push!(bonddims, TCI.linkdims(qtt.tci))
            TCI4Keldysh.updateJSON(outname, "times", times, folder)
            TCI4Keldysh.updateJSON(outname, "ranks", qttranks, folder)
            TCI4Keldysh.updateJSON(outname, "bonddims", bonddims, folder)

            if serialize_tts            
                serialize_tt(qtt, outname, folder)
            end

            println(" ===== R=$R: time=$t, rankk(qtt)=$(TCI4Keldysh.rank(qtt))")
        end
    else
        error("Invalid mode $mode")
    end
end


"""
Homebrewn broadening for beta=2000 🍺
"""
function beta2000_broadening(T)
    return (30*T, [0.1])
end

function store_KF_correlator(γfac, sigmak_;R=5)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/4pt/")
    npt = 4
    D = npt-1
    Ops = ["F1", "F1dag", "F3", "F3dag"]
    T = 5*1.e-4

    ωmin = -1.0
    ωmax = -ωmin
    ωs_ext = ntuple(i -> collect(range(ωmin, ωmax; length=2^R)), D)
    ωconvMat = TCI4Keldysh.channel_trafo("a")
    # γ, sigmak = TCI4Keldysh.default_broadening_γσ(T)
    γ = γfac * T
    sigmak = [sigmak_]
    KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops; T=T, ωs_ext=ωs_ext, flavor_idx=1, ωconvMat=ωconvMat, sigmak=sigmak, γ=γ, name="Kentucky fried chicken")

    data = TCI4Keldysh.precompute_all_values(KFC)
    fname = corrdata_fnameh5(R, γ, only(sigmak))
    if isfile(fname)
        # printstyled("=> $fname exists. Overwrite it? [y/n]\n"; color=:blue)
        # answer = readline()
        # if answer=="y"
            rm(fname)
        # elseif answer=="n"
        #     return nothing
        # end
    end
    h5write(fname, "KFCdata", data)
end

function compress_KFCdata_all(R; tcikwargs...)
    T = default_T()
    γ, sig_vec = beta2000_broadening(T)
    sigmak = only(sig_vec)
    return compress_KFCdata_all(R, γ, sigmak; tcikwargs...)
end

"""
For each pivot (iK, σ1, ... σj), look how many tuples (iK', σ1, ..., σj) are also pivots
"""
function iK_pivots(tt::TCI.TensorCI2)
    @assert tt.localdims[1]==16 "Is this really a Keldysh object with all Keldysh indices?"
    bondcount = 2
    for I in tt.Iset[2:end]
        d = Dict{Vector{Float64}, Int}()
        for i in I
            isub = i[2:end]
            if !haskey(d, isub)
                d[isub]=1
            else
                d[isub]+=1
            end
        end
        
        # analyze
        println("\n==== Bond $bondcount:")
        mean_iKs = 0.0
        for (_, val) in d
            mean_iKs += val/16
        end
        mean_iKs /= length(d)
        println("  Mean iKs per frequency point: $mean_iKs")
        bondcount += 1
    end
end

"""
QTCI-compress all Keldysh components, using precomputed data from disk.
"""
function compress_KFCdata_all(R, γ, sigmak; tcikwargs...)
    # has D frequency, D+1 Keldysh indices
    fname = corrdata_fnameh5(R,γ,sigmak)
    @assert isfile(fname) "File $fname does not exist"
    data = h5read(fname, "KFCdata")
    D = div(ndims(data)-1, 2)
    R = Int(log2(size(data, 1))) 

    grid = QG.InherentDiscreteGrid{D}(R; unfoldingscheme=:interleaved)

    # first KF idx, then frequencies
    function _eval_KFC(idx::Vector{Int})
        k = TCI4Keldysh.KF_idx(idx[1], D)
        om = QG.quantics_to_grididx(grid, idx[2:end])
        return data[om..., k...]
    end

    localdims = vcat([2^(D+1)], fill(2, D*R))
    initpivot = vcat([2], fill(1, length(localdims)-1))
    # c_eval = TCI.CachedFunction{eltype(data)}(_eval_KFC, localdims)
    @time tt, _, _ = TCI.crossinterpolate2(ComplexF64, _eval_KFC, localdims, [initpivot]; tcikwargs...)
    @show TCI.linkdims(tt)

    iK_pivots(tt)
end

function compress_KFCdata(R; qtcikwargs...)
    T = default_T()
    # γ, sig_vec = TCI4Keldysh.default_broadening_γσ(T)
    γ, sig_vec = beta2000_broadening(T)
    sigmak = only(sig_vec)
    return compress_KFCdata(R, γ, sigmak; qtcikwargs...)
end

"""
QTCI-compress Keldysh components separately, using precomputed data from disk.
"""
function compress_KFCdata(R, γ, sigmak; iK::Union{Int,Nothing}=nothing, qtcikwargs...)
    # has D frequency, D+1 Keldysh indices
    data = h5read(corrdata_fnameh5(R,γ,sigmak), "KFCdata")
    D = div(ndims(data)-1, 2)

    iK_it = Iterators.product(fill(1:2, D+1)...)
    iKrange = isnothing(iK) ? iK_it : [TCI4Keldysh.KF_idx(iK, D)]
    ranks = []
    for iK in iKrange
        if all(iK .== 1)
            # this component is zero
            continue
        end
        data_tmp = data[ntuple(_->Colon(),D)..., iK...]
        # plot
        if D==3
            heatmap(log.(abs.(data_tmp[:,:,div(size(data_tmp,3),2)])))
            savefig("KFC.png")
        end
        @time qtt, _, _ = quanticscrossinterpolate(data_tmp; qtcikwargs...)
        @show TCI4Keldysh.rank(qtt)
        data_tmp = nothing
        push!(ranks, TCI4Keldysh.rank(qtt))
    end
    return ranks
end

function corrdata_fnameh5(R::Int, γ::Float64, sigmak::Float64;folder="KFCdata")
    censtr = @sprintf("γ=%.2e_σ=%.2e", γ, sigmak)    
    return joinpath(folder, "KFC_"*censtr*"_R=$R.h5")
end


"""
Test whether KF Correlator with a single spectral peak is correct.
"""
function test_singlepeak_KF(npt::Int)
    D = npt-1
    R = 5
    # peak of strength 1.0 at (-2.0, ..., -2.0) in each PSF
    ωdisc = [-2.0, 1.0]    
    Adiscs = [zeros(Float64, ntuple(i->2, D)) for _ in 1:factorial(D+1)]
    ωs_ext = TCI4Keldysh.KF_grid(1.0, R, D)

    for A in Adiscs
        A[ones(Int, D)...] = 1.0
    end
    KFC = TCI4Keldysh.multipeak_correlator_KF(ωs_ext, Adiscs, ωdisc)
    data = TCI4Keldysh.precompute_all_values(KFC)
    error("Not yet implemented")
end

function qttdata_fname(R, iK, tolerance, gammarange, sigmarange)
    qttdata_fname(R, iK, tolerance, first(gammarange), last(gammarange), first(sigmarange), last(sigmarange))
end

function qttdata_fname(R, iK, tolerance, γmin, γmax, smin, smax)
    ss = @sprintf("γ%.2eto%.2e_σ%.2eto%.2e", γmin, γmax, smin, smax)
    return "KFC_ranks_R=$(R)_iK=$(iK)_tol=$(round(Int,log10(tolerance)))_$(ss)"
end

function KFC_ranks(iK::Int, gammarange, sigmarange; R=7, tolerance=1.e-6)
    d = Dict()
    ranks = []
    gamma = []
    sigma = []
    for g in gammarange    
        for s in sigmarange
            rank = compress_KFCdata(R, g, s; iK=iK, tolerance=tolerance, unfoldingscheme=:interleaved)
            push!(ranks, only(rank))
            push!(gamma, g)
            push!(sigma, s)
        end
    end
    d["ranks"] = ranks
    d["gamma"] = gamma
    d["sigma"] = sigma
    fname = qttdata_fname(R, iK, tolerance, gammarange, sigmarange)
    TCI4Keldysh.logJSON(d, fname, "KFCdata")
end

function plot_FullCorrelator_timing(param_range, mode="R"; beta=default_T(), tolerance=1.e-6, plot_mem=false)
    folder = "pwtcidata"    
    error("Provide all parameters to specify file")
    filename = GFfilename(mode, minimum(param_range), maximum(param_range), tolerance, beta)
    data = TCI4Keldysh.readJSON(filename, folder)

    if mode=="R"
        Rs = convert.(Int, data["Rs"])
        RAM_usage = RAM_usage_3D.(Rs)
        times = convert.(Float64, data["times"])
        p = TCI4Keldysh.default_plot()

        plot!(p, Rs, times; marker=:diamond, color=:blue, label="F1F1dagF3F3dag")
        xlabel!(p, "R")
        ylabel!(p, "Wall time [s]")
        title!(p, "Timings full correlator, β=$beta, tol=$tolerance")

        if plot_mem
            ptwin = twinx(p)
            plot!(ptwin, Rs, RAM_usage; marker=:circle, color=:black, linestyle=:dash, yscale=:log10, label=nothing)
            yticks!(ptwin, 10.0 .^ (round(Int, log10(minimum(RAM_usage))) : round(Int, log10(maximum(RAM_usage)))))
            ylabel!(ptwin, "Memory for dense corr. [GB]")
        end

        savefig(p, "corrtiming_beta=$(beta)_tol=$(round(Int,log10(tolerance))).png")
    end
end

"""
Scatter plot of qtt ranks against γ and sigmak (broadening params)
"""
function plot_ranks(gammarange, sigmarange, iK; R=7, tolerance=1.e-6)
    fname = qttdata_fname(R, iK, tolerance, gammarange, sigmarange)
    data = TCI4Keldysh.readJSON(fname, "KFCdata")
    ranks = data["ranks"]
    # tuples (γ, σ)
    gamma = data["gamma"]
    sigma = data["sigma"]
    Nsig = length(sigmarange)
    T = default_T()
    @assert allequal(gamma[1:Nsig]) "wrong ordering of γ, sigmak?"
    p = TCI4Keldysh.default_plot()
    for i in eachindex(sigmarange)
        plot!(p, gamma[1:Nsig:end] ./ T, ranks[i:Nsig:end]; xscale=:identity, label="σk=$(sigma[i])", marker=:circle, linewidth=2)
        @show gamma[1:Nsig:end]
    end
    k_idx = collect(Iterators.product(fill(1:2, 4)...))[iK]
    xlabel!(p, "γ/T")
    ylabel!(p, "Rank")
    title!(p, "4pt-corr ranks, R=$R, k=$(k_idx), tol=$(round(Int, log10(tolerance)))")
    # worst = 2^(div(3*R, 2))
    # hline!(p, div(worst,2); color=:black, linestyle=:dash, label=nothing)
    savefig(p, "ranks_KFC.png")
end

function update_datalist(gammarange, sigmarange, R::Int)
    open("KFCdata/available_data.txt", "a") do f
        towrite = "gamma=$(collect(gammarange)), sigma=$(collect(sigmarange)), R=$R\n"
        write(f, towrite)
    end
end

# gammarange = [30]
# sigmarange = [0.1]
# for R in 4:6
#     store_KF_correlator(gammarange[1], sigmarange[1]; R=R)
#     update_datalist(gammarange, sigmarange, R)
# end

# for R in 5:9
#     time_compress_FullCorrelator_KF(2; R=R, tolerance=1.e-3)
# end

function test_FullCorrEvaluator_KF(npt::Int, iK::Int)
    # create correlator
    addpath = npt==4 ? "4pt" : ""
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg", addpath)
    D = npt-1
    Ops = TCI4Keldysh.dummy_operators(npt)
    T = TCI4Keldysh.dir_to_T(PSFpath)

    ωmax = 1.0
    R = 4
    ωs_ext = TCI4Keldysh.KF_grid(ωmax, R, D)
    ωconvMat = if npt==4
            TCI4Keldysh.channel_trafo("t")
        elseif npt==3
            TCI4Keldysh.channel_trafo_K2("t", false)
        else
            TCI4Keldysh.ωconvMat_K1()
        end
    γ, sigmak = TCI4Keldysh.default_broadening_γσ(T)
    KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops; T=T, ωs_ext=ωs_ext, flavor_idx=1, ωconvMat=ωconvMat, sigmak=sigmak, γ=γ, name="Kentucky fried chicken")

    KFev = TCI4Keldysh.FullCorrEvaluator_KF_single(KFC, iK)
    function KFC_(idx::Vararg{Int,N}) where {N}
        return TCI4Keldysh.evaluate(KFC, idx...; iK=iK)        
    end

    for _ in 1:20
        idx = rand(1:2^R, D)
        @test isapprox(KFC_(idx...), KFev(idx...); atol=1.e-10)
    end

    KFev2 = TCI4Keldysh.FullCorrEvaluator_KF(KFC)
    function KFC2_(idx::Vararg{Int,N}) where {N}
        return TCI4Keldysh.evaluate_all_iK(KFC, idx...)
    end

    for _ in 1:20
        idx = rand(1:2^R, D)
        refval = vec(KFC2_(idx...))
        totest_nocut = vec(KFev2(Val{:nocut}(), idx...))
        totest_cut = vec(KFev2(idx...))
        @test isapprox(norm(refval .- totest_cut), 0.0; atol=1.e-11)
        @test isapprox(norm(refval .- totest_nocut), 0.0; atol=1.e-11)
    end
end


function benchmark_FullCorrEvaluator_KF_alliK(npt::Int, R::Int; profile=false)
    # create correlator
    addpath = npt==4 ? "4pt" : ""
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg", addpath)
    D = npt-1
    Ops = TCI4Keldysh.dummy_operators(npt)
    T = TCI4Keldysh.dir_to_T(PSFpath)

    ωmax = 1.0
    ωs_ext = TCI4Keldysh.KF_grid(ωmax, R, D)
    ωconvMat = if npt==4
            TCI4Keldysh.channel_trafo("t")
        elseif npt==3
            TCI4Keldysh.channel_trafo_K2("t", false)
        else
            TCI4Keldysh.ωconvMat_K1()
        end
    # γ, sigmak = TCI4Keldysh.default_broadening_γσ(T)
    γ, sigmak = beta2000_broadening(T)
    KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops; T=T, ωs_ext=ωs_ext, flavor_idx=1, ωconvMat=ωconvMat, sigmak=sigmak, γ=γ, name="Kentucky fried chicken")
    KFC2 = deepcopy(KFC)

    KFev2 = TCI4Keldysh.FullCorrEvaluator_KF(KFC2; cutoff=1.e-7)

    function KFC2_(idx::Vararg{Int,N}) where {N}
        return TCI4Keldysh.evaluate_all_iK(KFC, idx...)
    end
    printstyled("---- Benchmark\n"; color=:blue)
    println(" Naive\n")
    @btime $KFC2_(rand(1:2^$R, $D)...)
    println(" Optimized (nocut)\n")
    @btime $KFev2(rand(1:2^$R, $D)...)
    println(" Optimized\n")
    @btime $KFev2(Val{:cut}(), rand(1:2^$R, $D)...)


    if profile
        Profile.clear()
        function _toprofile()
            tot = zero(ComplexF64)
            for _ in 1:100
                KFev2(rand(1:2^R, D)...)
            end
            return tot
        end
        Profile.@profile _toprofile()
        statprofilehtml()
    end

    return nothing
end

# time_FullCorrelator_sweep(2, 1000.0, 0.5; Rs=3:4, tolerance=1.e-3, serialize_tts=true)
# time_Γcore_KF_sweep(10:10, 2, 2000.0, 0.4; tolerance=1.e-2, serialize_tts=false)
time_Γcore_KF(2,10,1.e-2)