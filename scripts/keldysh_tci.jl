using Plots
using BenchmarkTools
using Profile
using StatProfilerHTML
using QuanticsTCI
using Printf
using HDF5
using JSON
using LinearAlgebra
import TensorCrossInterpolation as TCI
import QuanticsGrids as QG

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
    ωmin = -ωmax
    ωs_ext = TCI4Keldysh.KF_grid(ωmin, ωmax, R, D)
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
    R = 4
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
    KFev = TCI4Keldysh.FullCorrEvaluator_KF(KFC, iK)
    function KFC_(idx::Vararg{Int,3})
        return TCI4Keldysh.evaluate(KFC, idx...; iK=iK)        
    end

    @btime $KFC_(rand(1:2^$R, 3)...)
    @btime $KFev(rand(1:2^$R, 3)...)

    # profile
    Profile.clear()
    Profile.@profile begin
        dummy = zero(ComplexF64)
        for _ in 1:1000
            dummy += KFev(rand(1:2^R, 3)...)
        end
        println(dummy)
    end
    statprofilehtml()
end

function time_compress_FullCorrelator_KF(iK::Int; R=4, tolerance=1.e-3)
    npt = 4
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/4pt")
    D = npt-1
    Ops = TCI4Keldysh.dummy_operators(npt)
    T = default_T()

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
    γ, sigmak = beta2000_broadening(T)
    KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops; T=T, ωs_ext=ωs_ext, flavor_idx=1, ωconvMat=ωconvMat, sigmak=sigmak, γ=γ, name="Kentucky fried chicken")

    t = @elapsed begin
        qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(KFC, iK; tolerance=tolerance, unfoldingscheme=:fused)
        end
    @show TCI4Keldysh.rank(qtt)
    printstyled("  Time for single component, tol=$(tolerance), R=$R: $t[s]\n"; color=:blue)
end

function test_compress_FullCorrelator_KF(npt::Int, iK; R=4, tolerance=1.e-3, channel::String="t")
    addpath = npt==4 ? "4pt/" : ""
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg", addpath)
    D = npt-1
    Ops = TCI4Keldysh.dummy_operators(npt)
    T = default_T()

    ωmax = 1.0
    ωmin = -ωmax
    ωs_ext = ntuple(i -> collect(range(ωmin, ωmax; length=2^R)), D)
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

    @assert maximum(abs.(qttval .- data_ref) ./ maxref) <= 3.0*tolerance
end


function GFfilename(mode::String, xmin::Int, xmax::Int, tolerance, beta)
    return "KF_timing_$(mode)_min=$(xmin)_max=$(xmax)_tol=$(TCI4Keldysh.tolstr(tolerance))_beta=$beta"
end

"""
Store ranks and timings for computation of single keldysh components full 4-point correlators.
Can vary:
* iK : Keldysh component
* beta
* R
* tolerance
"""
function time_FullCorrelator_sweep(iK::Int, γ::Float64, sigmak::Float64, mode::String="R"; tolerance=1.e-8, beta=1.0/default_T(), Rs=nothing)
    folder = "pwtcidata"
    npt = 4
    times = []
    qttranks = []
    svd_kernel = true
    if mode=="R"
        Rs = isnothing(Rs) ? (5:8) : Rs
        # prepare output
        d = Dict()
        d["times"] = times
        d["ranks"] = qttranks
        d["Rs"] = Rs
        d["tolerance"] = tolerance
        d["svd_kernel"] = svd_kernel
        d["gamma"] = γ
        d["sigmak"] = sigmak 
        d["beta"] = beta
        outname = GFfilename(mode, first(Rs), last(Rs), tolerance, beta)
        TCI4Keldysh.logJSON(d, outname, folder)

        for R in Rs
            
            # create correlator
            PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/4pt/")
            npt = 4
            D = npt-1
            Ops = ["F1", "F1dag", "F3", "F3dag"]
            T = 1.0/beta

            ωmax = 1.0
            ωmin = -ωmax
            ωs_ext = ntuple(i -> collect(range(ωmin, ωmax; length=2^R)), D)
            ωconvMat = TCI4Keldysh.channel_trafo("a")
            KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops; T=T, ωs_ext=ωs_ext, flavor_idx=1, ωconvMat=ωconvMat, sigmak=[sigmak], γ=γ, name="Kentucky fried chicken")
            # create correlator END

            t = @elapsed begin
                qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(KFC, iK; tolerance=tolerance, unfoldingscheme=:interleaved)
            end
            push!(times, t)
            push!(qttranks, TCI4Keldysh.rank(qtt))
            TCI4Keldysh.updateJSON(outname, "times", times, folder)
            TCI4Keldysh.updateJSON(outname, "ranks", qttranks, folder)
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
QTCI-compress all Keldysh components, using precomputed data from disk.
"""
function compress_KFCdata_all(R, γ, sigmak; tcikwargs...)
    # has D frequency, D+1 Keldysh indices
    data = h5read(corrdata_fnameh5(R,γ,sigmak), "KFCdata")
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
    ωs_ext = TCI4Keldysh.KF_grid(-1.0, 1.0, R, D)

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

function update_datalist(gammarange, sigmarange)
    open("KFCdata/available_data.txt", "a") do f
        towrite = "gamma=$(collect(gammarange)), sigma=$(collect(sigmarange))\n"
        write(f, towrite)
    end
end

# gammarange = (10:20:90)
# sigmarange = [0.01, 0.1, 1.0]
# for γfac in gammarange
#     for sigmak in sigmarange
#         store_KF_correlator(γfac, sigmak; R=7)
#     end
# end
# update_datalist(gammarange, sigmarange)

# for R in 5:9
#     time_compress_FullCorrelator_KF(2; R=R, tolerance=1.e-3)
# end

function main()
    time_compress_FullCorrelator_KF(2)
    beta = 200.0
    time_FullCorrelator_sweep(2, 30.0/beta, 0.1, "R"; tolerance=1.e-4, beta=beta, Rs=5:10)
end