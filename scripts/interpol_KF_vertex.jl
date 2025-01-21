using TCI4Keldysh
using QuanticsTCI
using LinearAlgebra
using Plots
using LaTeXStrings
using Serialization
using HDF5
using BenchmarkTools 
using Profile
using StatProfilerHTML
using FastChebInterp
import TensorCrossInterpolation as TCI
import QuanticsGrids as QG

default(right_margin=10Plots.mm)

#=
Different analyses of Keldysh core vertex interpolation.
=#

# UTILITIES
"""
Find info on qtt in json file
"""
function qttfile_to_json(qttfile::String)
    pattern = r"R=(\d+)"
    m = match(pattern, qttfile)
    if !isnothing(m)
        return qttfile[1:m.offset-2] * ".json"
    else
        error("Pattern not found")
    end
end
# UTILITIES END

function interpolate_kernel_nonlin()
    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/4pt")
    Ops = TCI4Keldysh.dummy_operators(4)
    ωdisc = TCI4Keldysh.load_ωdisc(PSFpath, Ops)

    (γ, sigmak) = TCI4Keldysh.read_broadening_params(basepath)
    kwargs = TCI4Keldysh.read_broadening_settings(basepath)
    kwargs[:estep] = 50

    ommax = 0.318
    R = 10
    # ωs_ext = TCI4Keldysh.KF_grid_fer(ommax, R)
    # for logarithmic grid: why the strange jumps around 0?
    ωs_ext = TCI4Keldysh.get_Acont_grid(;kwargs...)
    k = TCI4Keldysh.compute_broadened_kernel(ωdisc, sigmak, γ; ωs_ext=ωs_ext, kwargs...)

    # interpolate with Chebychev
    samples = 1:10:length(ωdisc)
    chebs = Dict{Int,Any}()
    wpos = findfirst(w -> w>0.0, ωs_ext)
    # ωids = collect(wpos:length(ωs_ext))
    ωids = 302:450
    for ie in samples
        chebs[ie] = chebregression(convert.(Float64, ωids), k[ωids,ie], 20)
    end

    p = TCI4Keldysh.default_plot()
    for ie in samples
        plot!(p, ωids, real.(k[ωids,ie]); alpha=0.8, label="")
        plot!(p, ωids, real.(chebs[ie].(ωids)); alpha=0.8, label="Cheb$ie", linestyle=:dash)
        # plot!(p, ωids, imag.(k[:,ie]); alpha=0.8, linestyle=:dot)
    end
    xlabel!(p, L"i_\omega")
    savefig(p, "foo.pdf")

end

function chebychev_broadened_kernrel()
    ommax = 0.3
    ommin = -0.3
    # large linear grid
    R = 12
    klin, oms, omdisc = get_broadened_kernel(;R=R, ommax=ommax)

    # interpolate kernel on subintervals for each epsilon
    deg = 20
    chebs = []
    for ie in axes(klin, 2)
        c = chebregression(oms, klin[:,ie], ommin, ommax, deg)
        push!(chebs, c)
    end

    # try out chebregression
    f(x::Float64) = 1/(1+x^2) + im * exp(-x^2)
    cf = chebregression(oms, f.(oms), ommin, ommax, 10)
    @show maximum(@. abs(cf(oms) - f(oms)))

    # print errors
    to_plot = [3, 60]
    p = TCI4Keldysh.default_plot()
    for ie in axes(klin, 2)
        c = chebs[ie]
        cvals = c.(oms)
        err = abs.(cvals - klin[:,ie])
        maxf = maximum(abs.(klin[:,ie]))
        println("Peak no. $ie: max(err)=$(maximum(err)) mean(err)=$(sum(err)/length(err))")
        println("Peak no. $ie: max(tcierr)=$(maximum(err)/maxf) mean(tcierr)=$(sum(err)/length(err)/maxf)")
        println("----")
        if ie in to_plot
            plot!(p, oms, real.(cvals); label="ReCheb$ie", alpha=0.8)
            plot!(p, oms, real.(klin[:,ie]); label="ReK$ie", linestyle=:dot, alpha=0.8)
        # plot!(oms, imag.(cvals); label="ImCheb$ie")
        end
    end
    savefig(p, "foo.pdf")

end

"""
Return: broadened kernel, frequency grid, discrete energy grid ωdisc
"""
function get_broadened_kernel(;R::Int=10, ommax::Float64=0.5)
    
    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/4pt")
    channel = "p"
    Ops = TCI4Keldysh.dummy_operators(4)
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(basepath; channel=channel)
    G = TCI4Keldysh.FullCorrelator_KF(
        PSFpath,
        Ops;
        T=TCI4Keldysh.dir_to_T(PSFpath),
        ωconvMat=TCI4Keldysh.channel_trafo(channel),
        ωs_ext=TCI4Keldysh.KF_grid(ommax, R, 3),
        flavor_idx=1,
        γ=γ,
        sigmak=sigmak,
        emax=max(20.0, 3*ommax),
        emin=2.5*1.e-5,
        estep=50
    )

    perm_idx = 1
    kernel_idx = 1
    k = G.Gps[perm_idx].tucker.legs[kernel_idx]
    oms = G.Gps[perm_idx].tucker.ωs_legs[kernel_idx]
    omdisc = TCI4Keldysh.load_ωdisc(PSFpath, Ops)
    # Adisc = TCI4Keldysh.load_Adisc(PSFpath, Ops, 1)
    # _, omdiscs, _ = TCI4Keldysh.compactAdisc(omdisc, Adisc)
    # omdisc = omdiscs[kernel_idx]
    
    return (k, oms, omdisc)
end


"""
Estimate memory requirements of kernel interpolation for a given partitioning
of frequency and energy grid, for a 3D spectral function.
"""
function memory_multipole_kernel(
    om_intervals::Vector{UnitRange{Int}},
    eps_intervals::Vector{UnitRange{Int}},
    k::Matrix{ComplexF64};
    cutoff::Float64=1.e-8
    )
    sidelengths = zeros(Int, length(om_intervals), length(eps_intervals))
    for (io, om_int) in enumerate(om_intervals)
        for (ie, eps_int) in enumerate(eps_intervals)
            k_act = k[om_int, eps_int]
            _,S,_ = svd(k_act)
            Smax = first(S)
            Scut = findfirst(s -> s/Smax<=cutoff, S)
            if isnothing(Scut)
                Scut=length(S)+1
            end
            sidelengths[io,ie] = Scut-1
        end
    end

    # 3 for retarded components, 16 bytes per complex double
    # 16 full and 24 partial correlators in core vertex
    prefac = 16*16*24*3
    return prefac * sum(sidelengths) ^ 3 
end

function memory_multipole_kernel()

    R = 12
    (k, oms, omdisc) = get_broadened_kernel(;R=R)
    @assert size(k) == (length(oms), length(omdisc))

    n_om = 5
    om_intervals = TCI4Keldysh.nested_intervals(1, length(oms), n_om)
    n_eps = 4
    eps_intervals = TCI4Keldysh.nested_intervals(1, length(omdisc), n_eps)

    @show memory_multipole_kernel(om_intervals[5], eps_intervals[1], k; cutoff=1.e-8) ./ 1.e9

end

"""
Investigate potential of Chebychev interpolation OR blockwise SVD,
i.e., 'generalized multipole expansion'.
Here, no plots, but actual interpolation of kernel.
"""
function multipole_kernel()

    R = 12
    (k, oms, omdisc) = get_broadened_kernel(;R=R)
    @assert size(k) == (length(oms), length(omdisc))

    n_om = 6
    om_intervals = TCI4Keldysh.nested_intervals(1, length(oms), n_om)
    n_eps = 4
    eps_intervals = TCI4Keldysh.nested_intervals(1, length(omdisc), n_eps)

    @show eps_intervals

    # look at lowest frequency level
    cut = 1.e-8
    # look at all frequency levels
    om_int_range = 1:length(om_intervals)
    eps_int_range = [1]
    for om_fine in om_intervals[om_int_range]
        printstyled("\nFREQUENCY LEVEL: $(length(om_fine))\n"; color=:magenta)
        for om_int in om_fine
            println("\n==  Frequency range: $(om_int)")
            for ie in eps_int_range
                eps_lvl = eps_intervals[ie]
                println("      At energy level $(ie):")
                for eps_int in eps_lvl
                    k_act = k[om_int, eps_int]
                    _,S,_ = svd(k_act)
                    Smax = first(S)
                    # Smax = 1.0
                    Scut = findfirst(s -> s/Smax<=cut, S)
                    if isnothing(Scut)
                        Scut=length(S)+1
                    end
                    cut_rel = (Scut-1)/length(eps_int)
                    npt = min(99, round(Int, 100 * cut_rel))
                    # println(S)
                    println(prod(*, vcat(fill("-",npt))), prod(*, vcat(fill(" ", 100-npt))), "||")
                end
            end
        end
    end
end


function hierarchical_Gp(;R=5, ommax=0.5, do_profile=false)
    
    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/4pt")
    channel = "p"
    Ops = TCI4Keldysh.dummy_operators(4)
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(basepath; channel=channel)
    G = TCI4Keldysh.FullCorrelator_KF(
        PSFpath,
        Ops;
        T=TCI4Keldysh.dir_to_T(PSFpath),
        ωconvMat=TCI4Keldysh.channel_trafo(channel),
        ωs_ext=TCI4Keldysh.KF_grid(ommax, R, 3),
        flavor_idx=1,
        γ=γ,
        sigmak=sigmak,
        emax=max(20.0, 3*ommax),
        emin=2.5*1.e-5,
        estep=50
    )

    center = G.Gps[1].tucker.center
    kernels = G.Gps[1].tucker.legs
    @show size(center)
    @show size.(kernels)
    # ht represents ONE partial correlator
    @time ht = TCI4Keldysh.HierarchicalTucker(center, Tuple(kernels), 4; cutoff=1.e-6)
    # Keldysh evaluator with one leg contracted
    # @time KFC = TCI4Keldysh.KFCEvaluator(G)
    printstyled("\n==  Memory\n"; color=:blue)
    @show Base.summarysize(G) * 16 / 1.e9
    @show Base.summarysize(ht) * 3*24*16 / 1.e9
    # @show Base.summarysize(KFC) / 1.e9
    printstyled("\n==  Benchmark\n"; color=:blue)
    # @btime $KFC(rand(1:2^$R,3)...)
    @btime $ht(rand(1:2^$R,3)...)

    # check accuracy
    if R<=7
        printstyled("\n==  Accuracy\n"; color=:blue)
        ref = TCI4Keldysh.contract_1D_Kernels_w_Adisc_mp(kernels, center)
        ht_dense = TCI4Keldysh.precompute_all_values(ht)
        maxref = maximum(abs.(ref))
        @show maxref
        diff = abs.(ht_dense .- ref) ./ maxref
        @show maximum(diff)
        @show norm(ref)
        @show norm(diff)
        amax = argmax(diff)
        scfun(x) = log10(abs(x))
        heatmap(scfun.(ref[:,amax[2],:]))
        savefig("ref.pdf")
        heatmap(scfun.(ht_dense[:,amax[2],:]))
        savefig("test.pdf")
        heatmap(scfun.(diff[:,amax[2],:]))
        savefig("diff.pdf")
    end

    # profile
    if do_profile
        printstyled("\n==  Profile\n"; color=:blue)
        Profile.clear()
        @profile begin
            x = 0.0
            for _ in 1:10^7
                x += ht(rand(1:2^R,3)...)
            end
            println(x)
        end
        statprofilehtml()
    end
end

"""
Investigate potential of Chebychev interpolation, 'generalized multipole expansion'.
"""
function plot_broadened_kernel()
    
    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/4pt")
    channel = "p"
    ommax = 0.3
    R = 10
    Ops = TCI4Keldysh.dummy_operators(4)
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(basepath; channel=channel)
    G = TCI4Keldysh.FullCorrelator_KF(
        PSFpath,
        Ops;
        T=TCI4Keldysh.dir_to_T(PSFpath),
        ωconvMat=TCI4Keldysh.channel_trafo(channel),
        ωs_ext=TCI4Keldysh.KF_grid(ommax, R, 3),
        flavor_idx=1,
        γ=γ,
        sigmak=sigmak,
        emax=20.0,
        emin=2.5*1.e-5,
        estep=50
    )

    @show Base.summarysize(G) / 1.e9

    perm_idx = 1
    kernel_idx = 1
    k = G.Gps[perm_idx].tucker.legs[kernel_idx]
    oms = G.Gps[perm_idx].tucker.ωs_legs[kernel_idx]
    omdisc = TCI4Keldysh.load_ωdisc(PSFpath, Ops)
    Adisc = TCI4Keldysh.load_Adisc(PSFpath, Ops, 1)
    _, omdiscs, _ = TCI4Keldysh.compactAdisc(omdisc, Adisc)
    omdisc = omdiscs[kernel_idx]
    omdisc_mid = div(length(omdisc),2)+1
    # omdisc_ids = omdisc_mid:2:length(omdisc)
    omdisc_ids = [1, 10, 20, length(omdisc), length(omdisc)-9, length(omdisc)-19]
    @show collect(omdisc_ids) .- omdisc_mid
    @show omdisc[omdisc_ids]
    # @show G.Gps[perm_idx].tucker.ωs_center[kernel_idx]

    # plot linear
    p = TCI4Keldysh.default_plot()
    for oi in omdisc_ids
        plot!(p, oms, real.(k[:,oi]); label="$oi", alpha=0.7)
        # plot!(p, oms, imag.(k[:,oi]); label="$oi", linestyle=:dot, legend=:bottomleft, alpha=0.7)
    end
    savefig("foo.pdf")
end


"""
Compare TCI-ed Keldysh vertex to exact result; ether precomputed and stored in `refdata` or
computed on the fly.
"""
function triptych_V_KF_data(qttfile::AbstractString, PSFpath::AbstractString;
    folder::AbstractString, refdata=nothing,
    do_plot=false,
    do_check_diff=true
    )
    (tci, grid) = deserialize(joinpath(folder, qttfile)) 
    qtt_data = TCI4Keldysh.readJSON(qttfile_to_json(qttfile), folder)
    R = grid.R
    Nhalf = 2^(R-1)
    beta = qtt_data["beta"]
    T = 1.0/beta
    broadening_kwargs = TCI4Keldysh.to_kwarg_dict(qtt_data["broadening_kwargs"])
    channel = qtt_data["channel"]
    iK = qtt_data["iK"]
    iKtuple = TCI4Keldysh.KF_idx(iK, 3)
    tolerance = qtt_data["tolerance"]
    # flavor_idx = qtt_data["flavor_idx"]
    flavor_idx = 1
    ωmax = qtt_data["ommax"]
    sigmak = [only(qtt_data["sigmak"])]
    γ = qtt_data["gamma"]

    # reference data
    Rplot = 5
    Nhplot = 2^(Rplot-1)
    offset = 0
    Nbos = max(2, 2*abs(offset))

    gamcore = nothing

    if isnothing(refdata)

        ωmax_plot = 2^(Rplot) * ωmax/2^R
        omfer = TCI4Keldysh.KF_grid_fer(ωmax_plot, Rplot)
        dω = omfer[2] - omfer[1]
        ombos = TCI4Keldysh.KF_grid_bos_(dω * div(Nbos,2), Nbos)
        omsig = TCI4Keldysh.KF_grid_fer_(ωmax_plot + ombos[end], 2^Rplot + Nbos)
        ωconvMat = TCI4Keldysh.channel_trafo(channel)
        @show dω
        @show length(omsig)
        @show length(ombos)
        @show length(omfer)

        # TODO: Somehow fails for Rplot=5?
        (ΣL,ΣR) = TCI4Keldysh.calc_Σ_KF_aIE_viaR(
            PSFpath,
            omsig; flavor_idx=flavor_idx,
            T=T,
            sigmak=sigmak,
            γ=γ,
            broadening_kwargs...
            )
        gamcore = TCI4Keldysh.compute_Γcore_symmetric_estimator(
            "KF",
            joinpath(PSFpath, "4pt"),
            ΣR;
            Σ_calcL=ΣL,
            T=T,
            flavor_idx=flavor_idx,
            ωs_ext=(ombos,omfer,omfer),
            ωconvMat=ωconvMat,
            sigmak=sigmak,
            γ=γ,
            broadening_kwargs...
        )
        # reference DONE

        @show size(gamcore)
    else
        gamcore = h5read(refdata, "V_KF")
        omgrid_ref = h5read(refdata, "omgrid1")
        if abs(ωmax - omgrid_ref[end])>1.e-8
            @warn "Frequency grids incompatible!"
        end
        @info "Read Γcore of size $(size(gamcore))"
    end

    tcival = TCI4Keldysh.qinterleaved_fattensor_to_regular(
        TCI4Keldysh.qtt_to_fattensor(tci.sitetensors),
        R
    )
    @info "Contracted tci to tensor of size $(size(tcival))"

    if do_check_diff
        println("  Checking error...")
        diff = abs.(tcival .- gamcore[1:size(tcival,1),:,:,iKtuple...]) ./ tci.maxsamplevalue
        @show tci.maxsamplevalue
        @show maximum(abs.(gamcore[:,:,:,iKtuple...]))
        fname = joinpath(TCI4Keldysh.pdatadir(), "_diff.h5")
        if isfile(fname)
            rm(fname)
        end
        h5write(fname, "diff", diff)
        h5write(fname, "qttdata", tcival)
        h5write(fname, "reference", gamcore)

        # print some accuracy measures
            # p-norms
        dom = 2 * ωmax / 2^R
        dV = dom^3
        gamcore_iK = gamcore[1:2^R,:,:,iKtuple...]
        for p in [1.0,2.0,3.0]
            println("$(round(Int,p))-norm")
            tcn = norm(tcival, p) * dV
            gn = norm(gamcore_iK, p) * dV
            @show (tcn, gn, abs(tcn-gn)/gn)
        end

            # integration along frequencies
        for dim in 1:3
            println("Integrate along dim: $dim")
            tcs = sum(tcival; dims=dim) * dom
            gs = sum(gamcore_iK; dims=dim) * dom
            tcs = dropdims(tcs; dims=dim)
            gs = dropdims(gs; dims=dim)
            heatmap(abs.(gs))
            savefig("integral_ref$dim.pdf")
            heatmap(abs.(tcs))
            savefig("   integral_tci$dim.pdf")
            heatmap(log10.(abs.(tcs.-gs) / maximum(abs.(gs))))
            savefig("integral_diff$dim.pdf")
        end
    end

    if do_plot
        # tci values
        println("-- Rank of tt : $(TCI.rank(tci))")
        # oneslice = Nhalf-Nhplot+1 : Nhalf+Nhplot
        # plot_slice = (Nhalf+1+offset:Nhalf+1+offset, oneslice, oneslice)
        # ids = Base.OneTo.(length.(plot_slice))
        # tcival = zeros(ComplexF64, length.(plot_slice))
        # Threads.@threads for id in collect(Iterators.product(ids...))
        #     w = ntuple(i -> plot_slice[i][id[i]], 3)
        #     tcival[id...] = tci(QG.origcoord_to_quantics(grid, w))
        # end
        # tci DONE

        # plot
        maxval = maximum(abs.(gamcore[:,:,:,iKtuple...]))
        scfun(x) = log10(abs(x))
        heatmap(
            scfun.(gamcore)[2^(R-1)+1+offset,:,:, iKtuple...];
            # scfun.(gamcore)[:,2^(R-1)+offset,:, iKtuple...];
            clim=(log10(maxval) + log10(tolerance), log10(maxval))
            )
        savefig("V_KF_ref.pdf")
        heatmap(
            scfun.(tcival)[2^(R-1)+1+offset,:,:];
            # scfun.(gamcore)[:,2^(R-1)+offset,:, iKtuple...];
            clim=(log10(maxval) + log10(tolerance), log10(maxval))
            )
        savefig("V_KF_tci.pdf")
    end
    return true
end

function compress_julia_precomputed(refdata::String, iK::Int, store::Bool=true; tcikwargs...)
    gamcore = h5read(refdata, "V_KF")
    iKtuple = TCI4Keldysh.KF_idx(iK,3)
    R = round(Int, trunc(log2(size(gamcore,1))))

    gc = gamcore[1:2^R,1:2^R,1:2^R,iKtuple...]
    unfsc = :fused
    qtt, _, _ = quanticscrossinterpolate(gc; unfoldingscheme=unfsc, tcikwargs...)
    @show TCI4Keldysh.rank(qtt)
    @show TCI.linkdims(qtt.tci)
    tcival = if unfsc==:interleaved
        tcival = TCI4Keldysh.qinterleaved_fattensor_to_regular(
            TCI4Keldysh.qtt_to_fattensor(qtt.tci.sitetensors),
            R
        )
    else
        qttfat = TCI4Keldysh.qtt_to_fattensor(qtt.tci.sitetensors)
        qttfat = reshape(qttfat, ntuple(_->2, 3*R))
        TCI4Keldysh.qinterleaved_fattensor_to_regular(
            qttfat, R
        )
    end

    if store
        maxval = maximum(abs.(gc))
        diff = abs.(tcival .- gc) ./ maxval
        @show maximum(diff)

        fname = joinpath(TCI4Keldysh.pdatadir(), "_diff_prec.h5")
        if isfile(fname)
            rm(fname)
        end
        h5write(fname, "diff", diff)
        h5write(fname, "qttdata", tcival)
        h5write(fname, "reference", gamcore)
        println("  File $fname written")
    else
        println("  DONE")
    end
end

function triptych_V_KF_plot(slice_dim::Int, slice_idx::Int)

    iK = 2
    tolerance = 1.e-2
    iKtuple = TCI4Keldysh.KF_idx(iK, 3)
    fname = "_diff_prec.h5"
    @show fname
    gamcore = h5read(fname, "reference")[:,:,:,iKtuple...]
    tcival = h5read(fname, "qttdata")
    dd = h5read(fname, "diff")
    maxval = maximum(abs.(gamcore))

    @show maximum(abs.(dd))
    @show argmax(abs.(dd))

    slice = ntuple(i -> ifelse(i==slice_dim, slice_idx, Colon()), 3)
    scfun(x) = log10(abs(x))
    heatmap(
        scfun.(gamcore[slice...]);
        clim=(log10(maxval) + log10(tolerance)-1, log10(maxval))
    )
    savefig("V_KF_ref_$(slice_dim)$(slice_idx)fix.pdf")
    heatmap(
        scfun.(tcival[slice...]);
        clim=(log10(maxval) + log10(tolerance)-1, log10(maxval))
    )
    savefig("V_KF_tci_$(slice_dim)$(slice_idx)fix.pdf")
    heatmap(
        log10.(abs.(dd[slice...]));
        clim=(log10(tolerance)-1, -1)
    )
    savefig("V_KF_diff_$(slice_dim)$(slice_idx)fix.pdf")
end

"""
Compare TCI-compression of:
- pointwisely evaluated vertex
- TCI4Keldysh precomputed vertex
- MuNRG precomputed vertex (differs slightly due to broadening implementation)
"""
function compare_pweval_Julia_MuNRG(iK::Int, munrg_compression::String, julia_data::String="_diff_prec.h5", pw_data="_diff.h5")
    
    iKtuple = TCI4Keldysh.KF_idx(iK,3)
    # pointwise evaluation, generated with triptych_V_KF_data
    println("  Loading data")
    gamcore_pw = h5read(pw_data, "reference")[:,:,:,iKtuple...]
    tcival_pw = h5read(pw_data, "qttdata")
    err_pw = h5read(pw_data, "diff")
    @show maximum(err_pw)

    # Julia
    gamcore_jl = h5read(julia_data, "reference")[:,:,:,iKtuple...]
    tcival_jl = h5read(julia_data, "qttdata")
    err_jl = h5read(julia_data, "diff")
    @show maximum(err_jl)

    # MuNRG
    gamcore_mat = h5read(munrg_compression, "reference")
    tcival_mat = h5read(munrg_compression, "qttdata")
    diff_mat = h5read(munrg_compression, "diff")
    err_mat = abs.(diff_mat) ./ maximum(abs.(gamcore_mat))
    @show maximum(err_mat)

    @show size(gamcore_pw)
    @show size(gamcore_jl)
    @show size(gamcore_mat)
    println("\n  Analyze data")

    # plot histograms of errors
end

function plot_slice(data::Array{T,3}, slice_dim::Int, slice_idx::Int; n_decades::Int=5, ommax::Float64) where {T<:Number}
    scfun(x::T) = log10(abs(x))    
    slice_tuple = ntuple(i -> ifelse(i==slice_dim, slice_idx, Colon()), 3)
    lmaxval = log10(maximum(abs.(data)))
    p = TCI4Keldysh.default_plot()
    heatmap!(p, scfun.(data[slice_tuple...]); clim=(lmaxval - n_decades, lmaxval))
    ntick = 6
    plot_dims = [i for i in 1:3 if i!=slice_dim]
    ticksteps = [div(size(data, i), ntick) for i in plot_dims]
    tickpos = [collect(1:ticksteps[i]:size(data,plot_dims[i])) for i in 1:2]
    ticklabels = [string.(collect(range(-ommax, ommax; length=length(tickpos[i])))) for i in 1:2]
    xticks!(p, tickpos[1], ticklabels[1])
    yticks!(p, tickpos[2], ticklabels[2])
    savefig(p, "foo.pdf")
end

# TODO
#=
- interpolate precomputed MuNRG and Julia with initial pivots
- so far: bond dimensions for MuNRG slightly lower than pointwise eval;
  see whether this changes with in initial pivots
- check whether interpolations fails similarly for all three compressions
=#

R = 8
qttfile = "keldyshcore_R_min=$(R)_max=$(R)_tol=-3_beta=2000.0_R=$(R)_qtt.serialized"
folder = "cluster_output_KCS/V_KF_tol3_R$(R)_9pivot_global"
# qttfile = "KF_gammacore_iK=2_R_min=5_max=9_tol=-2_beta=2000.0_omega-0.32_to_0.32_iK=2_broaden_γ=0.00_σ=0.40_R=8_qtt.serialized"
# folder = "cluster_output/V_KF_pch_tol2_iK2_R0509"
PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg")
refdata = "cluster_output/V_KF_conventional_u1.50_ommax1.5/V_KF_p_R=$(R).h5"

# triptych_V_KF_data(qttfile, PSFpath; folder=folder, refdata=refdata, do_plot=false, do_check_diff=true)
# compress_julia_precomputed(refdata, 2, true; tolerance=1.e-3)

# munrg_compression = "keldysh_seungsup_results/vertex_iK=2_tol=-2.h5"
# compare_pweval_Julia_MuNRG(2, munrg_compression)