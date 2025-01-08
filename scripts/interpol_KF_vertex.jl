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

function V_KF_eval_accuracy(
    refpath::String,
    R::Int,
    iK::Int,
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg"),
    KEV::Type=TCI4Keldysh.MultipoleKFCEvaluator,
    coreEvaluator_kwargs::Dict{Symbol,Any}=Dict{Symbol,Any}(:cutoff=>1.e-6, :nlevel=>4)
    )

    # load settings
    refjson = only(
        filter(f -> endswith(f, ".json"), readdir(refpath))
    )
    refsettings = TCI4Keldysh.readJSON(refjson, refpath)
    ωmax = refsettings["ommax"]
    broadening_kwargs_ = refsettings["broadening_kwargs"]
    broadening_kwargs = Dict{Symbol,Any}()
    for (key,val) in pairs(broadening_kwargs_)
        broadening_kwargs[Symbol(key)] = val
    end
    γ = refsettings["gamma"]
    sigmak = Vector{Float64}(refsettings["sigmak"])
    channel = refsettings["channel"]
    flavor_idx = refsettings["flavor_idx"]
    if !(R in refsettings["Rs"])
        error("Requested grid size is not available")
    end

    T = TCI4Keldysh.dir_to_T(PSFpath)

    # prepare core evaluator
    broadening_kwargs[:estep] = 20

    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    # make frequency grid
    D = size(ωconvMat, 2)
    @assert D==3
    ωs_ext = TCI4Keldysh.KF_grid(ωmax, R, D)

    # all 16 4-point correlators
    letters = ["F", "Q"]
    letter_combinations = kron(kron(letters, letters), kron(letters, letters))
    op_labels = ("1", "1dag", "3", "3dag")
    op_labels_symm = ("3", "3dag", "1", "1dag")
    is_incoming = (false, true, false, true)

    # create correlator objects
    Ncorrs = length(letter_combinations)
    GFs = Vector{TCI4Keldysh.FullCorrelator_KF{D}}(undef, Ncorrs)
    PSFpath_4pt = joinpath(PSFpath, "4pt")
    filelist = readdir(PSFpath_4pt)
    for l in 1:Ncorrs
        letts = letter_combinations[l]
        println("letts: ", letts)
        ops = [letts[i]*op_labels[i] for i in 1:4]
        if !any(TCI4Keldysh.parse_Ops_to_filename(ops) .== filelist)
            ops = [letts[i]*op_labels_symm[i] for i in 1:4]
        end
        GFs[l] = TCI4Keldysh.FullCorrelator_KF(PSFpath_4pt, ops; T, flavor_idx, ωs_ext, ωconvMat, sigmak=sigmak, γ=γ, broadening_kwargs...)
    end

    # evaluate self-energy
    incoming_trafo = diagm([inc ? -1 : 1 for inc in is_incoming])
    @assert all(sum(abs.(ωconvMat); dims=2) .<= 2) "Only two nonzero elements per row in frequency trafo allowed"
    ωstep = abs(ωs_ext[1][1] - ωs_ext[1][2])
    Σω_grid = TCI4Keldysh.KF_grid_fer(2*ωmax, R+1)
    (Σ_L,Σ_R) = TCI4Keldysh.calc_Σ_KF_aIE_viaR(PSFpath, Σω_grid; flavor_idx=flavor_idx, T=T, sigmak, γ, broadening_kwargs...)

    # frequency grid offset for self-energy
    ΣωconvMat = incoming_trafo * ωconvMat
    corner_low = [first(ωs_ext[i]) for i in 1:D]
    corner_idx = ones(Int, D)
    corner_image = ΣωconvMat * corner_low
    idx_image = ΣωconvMat * corner_idx
    desired_idx = [findfirst(w -> abs(w-corner_image[i])<ωstep*0.1, Σω_grid) for i in eachindex(corner_image)]
    ωconvOff = desired_idx .- idx_image

    sev = TCI4Keldysh.SigmaEvaluator_KF(Σ_R, Σ_L, ΣωconvMat, ωconvOff)

    gev = TCI4Keldysh.ΓcoreEvaluator_KF(GFs, iK, sev, KEV; coreEvaluator_kwargs...)

    # load reference
    refdatafile = joinpath(refpath, "V_KF_$(channel)_R=$(R).h5")
    refdata = h5read(refdatafile, "V_KF")[:,:,:,TCI4Keldysh.KF_idx(iK,3)...]
    # SETUP DONE

    # check
    N = 10^4
    errs = Vector{Float64}(undef, N)
    vals = Vector{ComplexF64}(undef, N)
    idx_range = Base.OneTo.(size(refdata))
    Threads.@threads for n in 1:N
        idx = ntuple(i -> rand(idx_range[i]), 3)
        val = gev(idx...)
        errs[n] = abs(val - refdata[idx...])
        vals[n] = val
    end

    h5write("errs.h5", "vals", vals)
    h5write("errs.h5", "errs", errs)
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


function triptych_V_KF(qttfile::AbstractString, PSFpath::AbstractString; folder::AbstractString)
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
    flavor_idx = qtt_data["flavor_idx"]
    ωmax = qtt_data["ommax"]
    sigmak = [only(qtt_data["sigmak"])]
    γ = qtt_data["gamma"]

    # reference data
    Rplot = 5
    Nhplot = 2^(Rplot-1)
    ωmax_plot = 2^(Rplot) * ωmax/2^R
    offset = Nhplot
    omfer = TCI4Keldysh.KF_grid_fer(ωmax_plot, Rplot)
    dω = omfer[2] - omfer[1]
    Nbos = max(2, 2*abs(offset))
    ombos = TCI4Keldysh.KF_grid_bos_(dω * div(Nbos,2), Nbos)
    omsig = TCI4Keldysh.KF_grid_fer_(ωmax_plot + ombos[end], 2^Rplot + Nbos)
    ωconvMat = TCI4Keldysh.channel_trafo(channel)

    @show dω
    @show length(omsig)
    @show length(ombos)
    @show length(omfer)

    # TODO: Load precomputed vertex
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

    # tci values
    println("-- Rank of tt : $(TCI.rank(tci))")
    oneslice = Nhalf-Nhplot+1 : Nhalf+Nhplot
    plot_slice = (Nhalf+1+offset:Nhalf+1+offset, oneslice, oneslice)
    ids = Base.OneTo.(length.(plot_slice))
    tcival = zeros(ComplexF64, length.(plot_slice))
    Threads.@threads for id in collect(Iterators.product(ids...))
        w = ntuple(i -> plot_slice[i][id[i]], 3)
        tcival[id...] = tci(QG.origcoord_to_quantics(grid, w))
    end
    # tci DONE

    @show size(gamcore)

    do_check_diff = true
    if do_check_diff
        println("  Checking error...")
        diffslice = (Nhalf+1-Nhplot : Nhalf+1+Nhplot, Nhalf-Nhplot+1 : Nhalf+Nhplot, Nhalf-Nhplot+1 : Nhalf+Nhplot)
        ids = Base.OneTo.(length.(diffslice))
        tcival = zeros(ComplexF64, length.(diffslice))
        Threads.@threads for id in collect(Iterators.product(ids...))
            w = ntuple(i -> diffslice[i][id[i]], 3)
            tcival[id...] = tci(QG.origcoord_to_quantics(grid, w))
        end
        diff = abs.(tcival .- gamcore[:,:,:,iKtuple...]) ./ tci.maxsamplevalue
        @show tci.maxsamplevalue
        @show maximum(abs.(gamcore[:,:,:,iKtuple...]))
        fname = joinpath(TCI4Keldysh.pdatadir(), "_diff.h5")
        if isfile(fname)
            rm(fname)
        end
        h5write(fname, "diff", diff)
    end


    # plot
    maxval = maximum(abs.(gamcore[:,:,:,iKtuple...]))
    scfun(x) = log10(abs(x))
    heatmap(
        scfun.(gamcore)[div(Nbos,2)+1+offset,:,:, iKtuple...];
        clim=(log10(maxval) + log10(tolerance), log10(maxval))
        )
    savefig("V_KF_ref.pdf")
    heatmap(
        scfun.(tcival)[1,:,:];
        clim=(log10(maxval) + log10(tolerance), log10(maxval))
        )
    savefig("V_KF_tci.pdf")
end

# qttfile = "keldyshcore_R_min=8_max=8_tol=-3_beta=2000.0_R=8_qtt.serialized"
# folder = "KF_KCS_rankdata/V_KF_tol3_R8_9pivot"
# PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg")
# triptych_V_KF(qttfile, PSFpath; folder=folder)