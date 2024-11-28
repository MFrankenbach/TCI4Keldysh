using Plots
using Profile
using StatProfilerHTML
using Serialization
using HDF5
using LinearAlgebra
using LaTeXStrings
using JSON
using QuanticsTCI
import QuanticsGrids as QG
import TensorCrossInterpolation as TCI

TCI4Keldysh.TIME() = false

# pythonplot()

const DEFAULT_β::Float64 = 2000.0
const DEFAULT_T::Float64 = 1.0/DEFAULT_β

"""
To merge two .json files containing results.
Use to add missing data to a json file where the calculation did not finish.
"""
function merge_jsondata(file1::String, file_add::String)
    # safety copy
    file1_old = file1[1:end-5] * "_original.json"
    cp(file1, file1_old; force=true)

    data1 = open(file1) do f
        JSON.parse(f)
    end

    data_add = open(file_add) do f
        JSON.parse(f)
    end

    # Rs is not included because that will also contain the missing values
    addkeys = ["times", "ranks", "bonddims"]

    for key in addkeys
        data1[key] = vcat(data1[key], data_add[key])
    end

    @show data1
    # write
    open(file1, "w") do f
        JSON.print(f, data1)
    end
end

"""
What is gained from caching central values?
"""
function time_cache_gain(;R=5, tolerance=1.e-6)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    ωconvMat = TCI4Keldysh.channel_trafo("t")
    beta = 2000.0
    T = 1.0/beta

    println("  No cache...")
    t1 = @elapsed begin
        foo = TCI4Keldysh.Γ_core_TCI_MF(
            PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved
            )
    end
    x = sum(foo)
    println("  With cache...")
    t2 = @elapsed begin
        Γcore = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved,
        cache_center=2^(R-2)
        )
    end 
    x = sum(Γcore)
    println(" TIME WITH CACHE: $t2")
    println(" TIME WITHOUT CACHE: $t1")
end

function qtt_filename(outname::String, R::Int, folder::String="pwtcidata")
    return joinpath(folder, outname*"_R=$(R)_qtt.serialized")
end

"""
To store qtts on disk (on cluster)
"""
function serialize_tt(qtt, outname::String, folder::String)
    R = qtt.grid.R
    fname_tt = qtt_filename(outname, R, folder)
    serialize(fname_tt, qtt)
end

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

# =========== K1

function plot_K1_ranks_MF(PSFpath;channel="t", flavor_idx=1)
    tols = reverse(collect(10.0 .^ (-6:2:-2)))
    Rs = 5:12
    T = TCI4Keldysh.dir_to_T(PSFpath)
    ranks = zeros(Int, length(Rs), length(tols))
    for (it,tol) in enumerate(tols)
        for (iR,R) in enumerate(Rs)
            qtt = TCI4Keldysh.K1_TCI(
                PSFpath, R;
                channel=channel, formalism="MF", flavor_idx=flavor_idx, T=T, tolerance=tol, unfoldingscheme=:interleaved
                )
            ranks[iR,it] = rank(qtt[1].tci)
        end
    end
    p = TCI4Keldysh.default_plot()
    for it in eachindex(tols)
        tolexp = round(Int, log10(tols[it]))
        plot!(p, Rs, ranks[:,it]; label=L"tol=$10^{%$tolexp}$", marker=:circle, seriescolor=:auto)
    end
    xlabel!("R")
    ylabel!("rank")
    title!(L"Rank of $K^1_{%$channel}$")
    savefig("K1_ranks_MF.pdf")
end

function plot_K1_ranks_KF(PSFpath;channel="t", flavor_idx=1)
    tols = reverse(collect(10.0 .^ (-6:1:-2)))
    Rs = 8:2:16
    T = TCI4Keldysh.dir_to_T(PSFpath)
    iK = (1,2)
    ranks = zeros(Int, length(Rs), length(tols))
    for (it,tol) in enumerate(tols)
        for (iR,R) in enumerate(Rs)
            qtt = TCI4Keldysh.K1_TCI(
                PSFpath, R;
                channel=channel,
                formalism="KF",
                flavor_idx=flavor_idx,
                T=T,
                tolerance=tol,
                unfoldingscheme=:interleaved
                )
            ranks[iR,it] = rank(qtt[iK...].tci)
        end
    end
    p = TCI4Keldysh.default_plot()
    for it in eachindex(tols)
        tolexp = round(Int, log10(tols[it]))
        plot!(p, Rs, ranks[:,it]; label=L"tol=$10^{%$tolexp}$", marker=:circle, yscale=:log10)
    end
    xlabel!("R")
    ylabel!("rank")
    title!(L"Rank of $K1_{%$channel}^{%$iK}$")
    savefig("K1_ranks_KF.pdf")
end

"""
Convolve with Gaussian
"""
function smoothen_gauss(xs::Vector{Float64}, ys::Vector{T}, sig::Float64) where {T<:Number}
    # sqrt(1/2πσ)exp(-x^2/2σ)
    # fourier: sqrt(2π*σ)exp(-σ*x^2/2)

    ny = length(ys)
    gauss(x::Float64) = 1/sqrt(2π*sig)*exp(-x^2/(2*sig))
    gaussval = gauss.(xs)
    dx = xs[2]-xs[1]
    ws = Int(ceil(5*sqrt(sig)/dx))
    midx = div(length(xs),2)+1
    res = zeros(T, ny)
    for i in eachindex(res)
        imin = max(1, i-ws)
        mincut = max(0, 1-(i-ws))
        imax = min(ny, i+ws)
        maxcut = max(0, (i+ws)-ny)
        # @show (midx-ws+mincut : midx+ws-maxcut, imin : imax)
        gvec = gaussval[midx-ws+mincut : midx+ws-maxcut]
        yvec = ys[imin : imax]
        @views res[i] = dot(gvec, yvec) * dx
    end
    return res

end

# here one can see the broadening: the function has edges for R=16
function plot_K1_zoomed()
    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/")
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
    channel = "t"
    flavor_idx = 1
    ωmax = 0.3183098861837907
    R = 12
    ωs_ext = TCI4Keldysh.KF_grid_bos(ωmax, R)
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(basepath)
    broadening_kwargs = TCI4Keldysh.read_broadening_settings(basepath)
    K1 = TCI4Keldysh.precompute_K1r(PSFpath, flavor_idx, "KF"; ωs_ext=ωs_ext, channel=channel, γ=γ, sigmak=sigmak, broadening_kwargs...)
    ik = (2,2)
    xs = collect(range(-ωmax, ωmax, size(K1,1)))
    K1_smooth = smoothen_gauss(xs, K1[:,ik...], 1.e-8)

    do_qtci = true
    if do_qtci
        printstyled("\n QTCI...\n"; color=:blue)
        tolerance = 1.e-8
        qtt,_,_ = quanticscrossinterpolate(K1[1:2^R,ik...]; tolerance=tolerance)
        qtt_smooth,_,_ = quanticscrossinterpolate(K1_smooth[1:2^R]; tolerance=tolerance)
        @show TCI.rank(qtt.tci)
        @show TCI.rank(qtt_smooth.tci)
    end
    
    K1ik = K1[:,ik...]
    K1max = maximum(abs.(K1ik))
    err_smooth = maximum(abs.(K1ik .- K1_smooth)) / K1max
    err_smooth_mid = maximum(abs.(K1ik[2^(R-2):2^(R-2)+2^(R-1)] .- K1_smooth[2^(R-2):2^(R-2)+2^(R-1)])) / K1max
    println("==== Error by smoothening: $err_smooth $err_smooth_mid")
    # plot small window
    # slice = 2^(R-1)+1-10:2^(R-1)+10
    slice = 1:2^R
    plot(ωs_ext[slice], abs.(K1ik)[slice]/K1max; label="original")
    plot!(ωs_ext[slice], abs.(K1_smooth[:])[slice]/K1max; label="smooth")
    savefig("K1zoomed.pdf")
end

# =========== K1 END

function plot_K12_ranks_MF(PSFpath;channel="t", flavor_idx=1, prime=false)
    tols = reverse(collect(10.0 .^ (-6:2:-2)))
    Rs = 5:12
    T = TCI4Keldysh.dir_to_T(PSFpath)
    ranks = zeros(Int, length(Rs), length(tols))
    ranks2 = zeros(Int, length(Rs), length(tols))
    for (it,tol) in enumerate(tols)
        for (iR,R) in enumerate(Rs)
            # K1
            qtt = TCI4Keldysh.K1_TCI(
                PSFpath, R;
                channel=channel, formalism="MF", flavor_idx=flavor_idx, T=T, tolerance=tol, unfoldingscheme=:interleaved
                )
            ranks[iR,it] = rank(qtt[1].tci)

            # K2
            qtt2 = TCI4Keldysh.K2_TCI_precomputed(
                PSFpath, R;
                channel=channel,
                prime=prime,
                formalism="MF",
                flavor_idx=flavor_idx,
                T=T,
                tolerance=tol,
                unfoldingscheme=:interleaved
                )
            ranks2[iR,it] = rank(qtt2[1].tci)
        end
    end
    p = TCI4Keldysh.default_plot()
    for it in eachindex(tols)
        tolexp = round(Int, log10(tols[it]))
        plot!(p, Rs, ranks[:,it]; label=L"K^1,tol=$10^{%$tolexp}$", marker=:circle, seriescolor=:auto, linestyle=:dot)
        plot!(p, Rs, ranks2[:,it]; label=L"K^2,tol=$10^{%$tolexp}$", marker=:diamond, seriescolor=:auto)
    end
    xlabel!("R")
    ylabel!("rank")
    title!(L"Rank of $K^1_{%$channel}$, $K^2_{%$channel}$")
    savefig("K12_ranks_MF.pdf")
end


# =========== K2
function plot_K2_ranks_MF(PSFpath;channel="t", flavor_idx=1, prime=false)
    tols = reverse(collect(10.0 .^ (-8:2:-2)))
    Rs = 5:12
    T = TCI4Keldysh.dir_to_T(PSFpath)
    ranks = zeros(Int, length(Rs), length(tols))
    for (it,tol) in enumerate(tols)
        for (iR,R) in enumerate(Rs)
            qtt = TCI4Keldysh.K2_TCI_precomputed(
                PSFpath, R;
                channel=channel,
                prime=prime,
                formalism="MF",
                flavor_idx=flavor_idx,
                T=T,
                tolerance=tol,
                unfoldingscheme=:interleaved
                )
            ranks[iR,it] = rank(qtt[1].tci)
        end
    end
    p = TCI4Keldysh.default_plot()
    for it in eachindex(tols)
        tolexp = round(Int, log10(tols[it]))
        plot!(p, Rs, ranks[:,it]; label=L"tol=$10^{%$tolexp}$", marker=:circle)
    end
    xlabel!("R")
    ylabel!("rank")
    title!(L"Rank of $K2_{%$channel}$")
    savefig("K2_ranks_MF.pdf")
end

# =========== K2 END

"""
Check whether vertex can be computed on a frequecy grid which has a small grid in one direction.
"""
function precompute_V_MF_slice(R::Int=5, compute_ref=false)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
    T = TCI4Keldysh.dir_to_T(PSFpath)
    flavor_idx = 1
    Nhalf = 2^(R-1)
    # bosonic direction is small.
    nh = 1
    omfer = TCI4Keldysh.MF_grid(T, Nhalf, true)
    ombos = TCI4Keldysh.MF_grid(T, nh, false)
    
    channel = "a"
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    omsig = TCI4Keldysh.MF_grid(T, Nhalf+nh, true)
    # slice computation
    (ΣL, ΣR) = TCI4Keldysh.calc_Σ_MF_aIE(PSFpath, omsig; T=T, flavor_idx=flavor_idx)
    @time gamcore = TCI4Keldysh.compute_Γcore_symmetric_estimator(
        "MF",
        PSFpath*"4pt/",
        ΣR;
        Σ_calcL=ΣL,
        T=T,
        flavor_idx=flavor_idx,
        ωs_ext=(ombos,omfer,omfer),
        ωconvMat=ωconvMat
    )

    # reference
    if compute_ref
        ombos_ref = TCI4Keldysh.MF_grid(T, Nhalf, false)
        omsig_ref = TCI4Keldysh.MF_grid(T, 2*Nhalf, true)
        (ΣL_ref, ΣR_ref) = TCI4Keldysh.calc_Σ_MF_aIE(PSFpath, omsig_ref; T=T, flavor_idx=flavor_idx)
        @time gamcore_ref = TCI4Keldysh.compute_Γcore_symmetric_estimator(
            "MF",
            PSFpath*"4pt/",
            ΣR_ref;
            Σ_calcL=ΣL_ref,
            T=T,
            flavor_idx=flavor_idx,
            ωs_ext=(ombos_ref,omfer,omfer),
            ωconvMat=ωconvMat
        )

        # compare
        refslice = gamcore_ref[Nhalf+1-nh:Nhalf+1+nh,:,:]
        @show maximum(abs.(refslice .- gamcore))
    end
end

"""
Generate data for triptych Reference - QTCI - Error
"""
function triptych_vertex_data(qttfile::String, Rplot::Int, PSFpath; folder="pwtcidata", store=true)
    (tci, grid) = deserialize(joinpath(folder, qttfile)) 
    qtt_data = TCI4Keldysh.readJSON(qttfile_to_json(qttfile), folder)
    tolerance = qtt_data["tolerance"]
    R = grid.R
    beta = TCI4Keldysh.dir_to_beta(PSFpath)
    T = 1. / beta
    @assert occursin("R=$R", qttfile) && occursin("$beta", qttfile) "Does the file $qttfile match parameters R=$R, beta=$beta?"

    if haskey(qtt_data, "flavor_idx")
        flavor_idx = qtt_data["flavor_idx"]
    else
        flavor_idx = 1
        @warn "Assuming flavor_idx=$flavor_idx"
        @warn "Assuming flavor_idx=$flavor_idx"
    end

    Nhalf = 2^(R-1)
    Nhplot = 2^(Rplot-1)
    oneslice = Nhalf-Nhplot+1 : Nhalf+Nhplot
    transfer_offset = 0
    plot_slice = (Nhalf+1+transfer_offset:Nhalf+1+transfer_offset, oneslice, oneslice)
    slice_id = findfirst(i -> length(plot_slice[i])==1, 1:3)
    ids = Base.OneTo.(length.(plot_slice))

    # ==== reference data

    ωconvMat = TCI4Keldysh.channel_trafo(qtt_data["channel"])
    # make frequency grid
    nh = max(1, transfer_offset)
    omfer = TCI4Keldysh.MF_grid(T, Nhalf, true)
    ombos = TCI4Keldysh.MF_grid(T, nh, false)

    omsig = TCI4Keldysh.MF_grid(T, Nhalf+nh, true)
    (ΣL, ΣR) = TCI4Keldysh.calc_Σ_MF_aIE(PSFpath, omsig; T=T, flavor_idx=flavor_idx)
    @time gamcore = TCI4Keldysh.compute_Γcore_symmetric_estimator(
        "MF",
        joinpath(PSFpath, "4pt"),
        ΣR;
        Σ_calcL=ΣL,
        T=T,
        flavor_idx=flavor_idx,
        ωs_ext=(ombos,omfer,omfer),
        ωconvMat=ωconvMat
    )

    # ==== SETUP DONE

    # bit_pos = collect(1:3:3*R)
    # bit_val = fill(0, length(bit_pos))
    # QG.index_to_quantics_fused!(bit_val, (Nhalf+1+transfer_offset,))
    # ttslice = TCI4Keldysh.saturate_bits(tci.sitetensors, bit_pos, bit_val)
    # @show size.(ttslice)
    # slice_fat_q = TCI4Keldysh.qtt_to_fattensor(ttslice)
    # slice_fat = TCI4Keldysh.qinterleaved_fattensor_to_regular(slice_fat_q, R)
    # @show size(slice)


    # tci values
    println("-- Rank of tt : $(TCI.rank(tci))")
    tcival = zeros(ComplexF64, length.(plot_slice))
    @show Base.summarysize(tcival)
    @show Base.summarysize(tci)
    Threads.@threads for id in collect(Iterators.product(ids...))
        w = ntuple(i -> plot_slice[i][id[i]], 3)
        tcival[id...] = tci(QG.origcoord_to_quantics(grid, w))
    end

    # ref values
    centre_id = div(size(gamcore,1),2)+1
    refval = reshape(gamcore[centre_id+transfer_offset, oneslice, oneslice], size(tcival)...)
    println("  Finished computation of vertex on slice")

    diff = refval .- tcival
    # diff = dropdims(diff; dims=slice_id)
    @show [p for p in collect(plot_slice)]
    slice_str = rstrip(reduce(*, [ifelse(length(p)==1, "$(p[1]-Nhalf-1),", "$(length(p)),") for p in collect(plot_slice)]), 'r')
    if store
        h5file = "vertex_MF_slice_beta=$(beta)_slices=($(slice_str))_tol=$(TCI4Keldysh.tolstr(tolerance)).h5"
        h5write(joinpath(folder, h5file), "reference", refval)
        h5write(joinpath(folder, h5file), "qttdata", tcival)
        h5write(joinpath(folder, h5file), "diff", diff)
        h5write(joinpath(folder, h5file), "maxref", abs(tci.maxsamplevalue))
    end
    return (refval, tcival, diff, abs(tci.maxsamplevalue))
end

function triptych_vertex_plot(h5file::String, qttfile::String; folder="pwtcidata")
    refval = h5read(joinpath(folder,h5file), "reference")
    tcival = h5read(joinpath(folder,h5file), "qttdata")
    diff = h5read(joinpath(folder,h5file), "diff")
    maxref = h5read(joinpath(folder,h5file), "maxref")
    triptych_vertex_plot(refval, tcival, diff, maxref, qttfile; folder=folder)
end

function triptych_vertex_plot(qttfile::String, Rplot::Int, PSFpath; folder="pwtcidata")
    (refval, tcival, diff, maxref) = triptych_vertex_data(qttfile, Rplot, PSFpath; folder=folder)

    triptych_vertex_plot(refval, tcival, diff, maxref, qttfile; folder=folder)
end

function triptych_vertex_plot(refval, tcival, diff, maxref, qttfile::String; folder="pwtcidata")
    if ndims(refval)==3
        sdims = findall(i -> size(refval, i)==1, 1:3)
        refval = dropdims(refval; dims=tuple(sdims...))
    end
    if ndims(tcival)==3
        sdims = findall(i -> size(tcival, i)==1, 1:3)
        tcival = dropdims(tcival; dims=tuple(sdims...))
    end
    if ndims(diff)==3
        sdims = findall(i -> size(diff, i)==1, 1:3)
        diff = dropdims(diff; dims=tuple(sdims...))
    end

    qtt_data = TCI4Keldysh.readJSON(qttfile_to_json(qttfile), folder)
    tolerance = qtt_data["tolerance"]
    beta = qtt_data["beta"]
    (_, om1g, om2g) = TCI4Keldysh.MF_npoint_grid(1.0/beta, div(size(refval,2),2), 3)
    p = TCI4Keldysh.default_plot()
    maxc = log10(abs(maxref))
    minc = maxc + log10(tolerance) - 1
    heatmap!(p, om1g, om2g, log10.(abs.(refval)); clim=(minc, maxc), right_margin=10Plots.mm)
    title!(L"\log_{10}|\Gamma_{\mathrm{core}}^{\mathrm{ref}}|")
    xlabel!(L"\omega'")
    ylabel!(L"\omega")
    savefig("V_MFref_tol=$(TCI4Keldysh.tolstr(tolerance))_beta=$(beta).pdf")

    heatmap!(p, om1g, om2g, log10.(abs.(tcival)); clim=(minc, maxc), right_margin=10Plots.mm)
    title!(L"\log_{10}|\Gamma_{\mathrm{core}}^{\mathrm{QTCI}}|")
    xlabel!(L"\omega'")
    ylabel!(L"\omega")
    savefig("V_MFtci_tol=$(TCI4Keldysh.tolstr(tolerance))_beta=$(beta).pdf")

    p = TCI4Keldysh.default_plot()
    scfun(x) = log10(abs(x)) 
    heatmap!(p, om1g, om2g, scfun.(diff ./ maxref); right_margin=10Plots.mm)
    title!(L"\log_{10}\left(|\Gamma_{\mathrm{core}}^{\mathrm{ref}}-\Gamma_{\mathrm{core}}^{\mathrm{QTCI}}|_\infty/|\Gamma_{\mathrm{core}}^{\mathrm{ref}}|\_infty\right)")
    xlabel!(L"\omega'")
    ylabel!(L"\omega")
    savefig("V_MFdiff_tol=$(TCI4Keldysh.tolstr(tolerance))_beta=$(beta).pdf")
end

"""
Check interpolation error of TCI-ed Γcore
"""
function check_interpolation(qttfile::String, R::Int, PSFpath; folder="pwtcidata_KCS")
    beta = TCI4Keldysh.dir_to_beta(PSFpath)
    T = 1.0/beta
    @assert occursin("R=$R", qttfile) && occursin("$beta", qttfile) "Does the file $qttfile match parameters R=$R, beta=$beta?"

    (tci, grid) = deserialize(joinpath(folder, qttfile))        
    qtt_data = TCI4Keldysh.readJSON(qttfile_to_json(qttfile), folder)
    tol = qtt_data["tolerance"]
    # conservative cutoff
    cutoff = 0.001*tol
    tucker_cut = 10.0*cutoff

    ωconvMat = TCI4Keldysh.channel_trafo(qtt_data["channel"])
    R = grid.R
    if haskey(qtt_data, "flavor_idx")
        flavor_idx = qtt_data["flavor_idx"]
    else
        @warn "Assuming flavor_idx=1"
        flavor_idx = 1
    end

    # ==== create ΓcoreEvaluator_MF
    gev_ref = TCI4Keldysh.ΓcoreEvaluator_MF(
        PSFpath,
        R;
        ωconvMat=ωconvMat,
        flavor_idx=flavor_idx,
        T=T,
        cutoff = cutoff
    )
    gbev_ref = TCI4Keldysh.ΓcoreBatchEvaluator_MF(gev_ref; use_ΣaIE=true)
    # ==== SETUP DONE

    # where to evaluate the stuff
    N_eval = 2^5
    # eval_step = max(div(2^R, N_eval), 1)
    # gridslice = 1:eval_step:2^R
    gridslice = 2^(R-1)-div(N_eval,2) : 2^(R-1)+div(N_eval,2)-1
    eval_size = ntuple(_->length(gridslice), 3)
    refval = zeros(ComplexF64, eval_size)
    tcival = zeros(ComplexF64, eval_size)
    ids = collect(Iterators.product(Base.OneTo.(eval_size)...))

    Threads.@threads for id in ids
        w = ntuple(i -> gridslice[id[i]], 3)
        wq = QG.origcoord_to_quantics(grid, tuple(w...))
        refval[id...] = gbev_ref(wq)
        tcival[id...] = tci(wq)
    end

    println("==== ERROR REPORT")
    diff = refval .- tcival
    maxref = abs(tci.maxsamplevalue)
    println("Accuracy for tolerance=$tol:")
    @show maxref
    @show norm(diff)
    amaxerr = argmax(abs.(diff))
    @show amaxerr
    @show maximum(abs.(diff)) ./ maxref

    h5file = "check_interpolation.h5"
    if isfile(h5file)
        rm(h5file)
    end
    println("==== Storing data to " * h5file)
    h5write(h5file, "diff", diff)
    h5write(h5file, "tcival", tcival)
    h5write(h5file, "refval", refval)
    h5write(h5file, "maxref", maxref)

    # plot
    slice = (amaxerr[1], Colon(), Colon())
    scfun = x -> log10(abs(x))
    cdepth = 1
    heatmap(scfun.(tcival[slice...] ./ maxref); clim=(log10(tol) - cdepth, cdepth))
    title!("Γ (TCI) / max|Γ|")
    savefig("gam_tci_check_interpolation.pdf")
    heatmap(scfun.(refval[slice...] ./ maxref); clim=(log10(tol) - cdepth, cdepth))
    title!("Γ (reference) / max|Γ|")
    savefig("gam_ref_check_interpolation.pdf")
    heatmap(scfun.(diff[slice...] ./ maxref))
    title!("Error (TCI-ref)/absmax(ref)")
    savefig("gam_diff_check_interpolation.pdf")
end

function time_Γcore()
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    R = 6
    tolerance = 1.e-4
    ωconvMat = TCI4Keldysh.channel_trafo("t")
    beta = TCI4Keldysh.dir_to_beta(PSFpath)
    T = 1.0/beta

    # compile
    println("  Compile run...")
    foo = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, 3; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=1.e-3, unfoldingscheme=:interleaved
        )
    x = sum(foo)
    println("  Time...")
    t = @elapsed begin
        Γcore = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved,
        cache_center=2^(R-2)
        # cache_center=0
        )
    end 
    x = sum(Γcore)
    println(" TIME: $t")
    @show TCI.linkdims(Γcore.tci)
end

"""
Check Γcore values at the fringes of the grid and compare to tolerance.
"""
function max_Γcore_tail(;R::Int=5, tolerance::Float64=1.e-5, beta::Float64=10.0)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    ωconvMat = TCI4Keldysh.channel_trafo("t")
    T = 1.0/beta
    @time qtt = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved
        )
    Nhalf = 2^(R-1)
    frval = abs(qtt(1,Nhalf,Nhalf))
    yvals = [qtt(i, Nhalf, Nhalf) for i in 1:2^R]
    cenval = maximum(abs.(yvals))
    printstyled("Fringe value: $frval\n"; color=:blue)
    printstyled("Center max value: $cenval\n"; color=:blue)
    printstyled("Ratio: $(frval/cenval) (tol=$tolerance)\n"; color=:blue)

    # plot line section
    plot(1:2^R, abs.(yvals) ./ cenval; yscale=:log10, label=nothing, linewidth=2)
    ylabel!("|Γcore(ω)| / max|Γcore(ω)|")
    xlabel!("ω1")
    hline!(tolerance; label="tol", color=:red, linestyle=:dashed)
    savefig("gammacore_tail.png")
    return nothing
end

function Γcore_filename(mode::String, xmin, xmax, tolerance::Float64, beta::Float64)
    return "gammacore_timing_$(mode)_min=$(xmin)_max=$(xmax)_tol=$(TCI4Keldysh.tolstr(tolerance))_beta=$beta"
end

"""
Find file for given beta and tolerance with maximum R-range
"""
function find_Γcore_file(tolerance::Float64, beta::Float64; folder="pwtcidata", subfolder_str=nothing)
    if !isnothing(subfolder_str)
        # files are distributed in sub-folders
        folder_content = readdir(folder)
        subdirs = [f for f in folder_content if isdir(joinpath(folder, f))]
        function _folder_relevant(f)
            return occursin(subfolder_str,f) && occursin("beta$(round(Int,beta))_",f) && occursin("gamcore",f) && occursin("tol$(-round(Int,log10(tolerance)))",f)
        end
        subdirs = filter(_folder_relevant, subdirs)
        files = [only(filter(f -> endswith(f,".json") && !occursin("original", f), readdir(joinpath(folder,sd)))) for sd in subdirs]
        @show files
        files = [joinpath(subdirs[i], files[i]) for i in eachindex(subdirs)]
    else
        function _file_relevant(f)
            return endswith(f, ".json") && !occursin("original", f) && occursin("beta=$beta", f) && occursin("tol=$(TCI4Keldysh.tolstr(tolerance))", f) && occursin("gammacore", f)
        end
        files = filter(
                _file_relevant,
                readdir(folder)
                )
    end

    if isempty(files)
        return nothing
    end

    function _Rrange(file)
        d = TCI4Keldysh.readJSON(file, folder)
        Rs = to_intvec(d["Rs"])
        Rran = maximum(Rs) - minimum(Rs)
        return Rran
    end

    return argmax(_Rrange, files)
end

function time_Γcore_sweep(param_range, mode="R"; beta=DEFAULT_β, tolerance=1.e-8)
    folder = "pwtcidata"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    ωconvMat = TCI4Keldysh.channel_trafo("t")
    T = 1.0/beta
    times = []
    qttranks = []
    svd_kernel = true
    if mode=="R"
        Rs = param_range
        # prepare output
        d = Dict()
        d["times"] = times
        d["ranks"] = qttranks
        d["Rs"] = Rs
        d["tolerance"] = tolerance
        d["svd_kernel"] = svd_kernel
        d["numthreads"] = Threads.threadpoolsize()
        outname = Γcore_filename(mode, first(Rs), last(Rs), tolerance, beta)
        TCI4Keldysh.logJSON(d, outname, folder)

        for R in Rs
            t = @elapsed begin
                qtt = TCI4Keldysh.Γ_core_TCI_MF(
                PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved, verbosity=2
                )
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

function RAM_usage_3D(R::Int)
    return 16 * 2^(3*R) / 1.e9
end

function plot_vertex_timing(param_range, mode="R"; beta=10.0, tolerance=1.e-6, plot_mem=true)
    folder = "pwtcidata"    
    filename = Γcore_filename(mode, minimum(param_range), maximum(param_range), tolerance, beta)
    data = TCI4Keldysh.readJSON(filename, folder)

    if mode=="R"
        Rs = convert.(Int, data["Rs"])
        RAM_usage = RAM_usage_3D.(Rs)
        times = convert.(Float64, data["times"])
        p = TCI4Keldysh.default_plot()

        plot!(p, Rs, times; marker=:diamond, color=:blue, label=nothing)
        xlabel!(p, "R")
        ylabel!(p, "Wall time [s]")
        title!(p, "Timings Γ core, β=$beta, tol=$tolerance")

        if plot_mem
            ptwin = twinx(p)
            plot!(ptwin, Rs, RAM_usage; marker=:circle, color=:black, linestyle=:dash, yscale=:log10, label=nothing)
            yticks!(ptwin, 10.0 .^ (round(Int, log10(minimum(RAM_usage))) : round(Int, log10(maximum(RAM_usage)))))
            ylabel!(ptwin, "Memory for dense corr. [GB]")
        end

        savefig(p, "vertextiming_beta=$(beta)_tol=$(round(Int,log10(tolerance))).png")
    end
end

function to_intvec(x) :: Vector{Int}
    return convert(Vector{Int}, x)
end

"""
Translate bonddims of a TT with complex entries and leg dimension d to RAM usage
in MB
"""
function bonddims_to_RAM(bonddims::Vector{Int}, d::Int=2)
    bonddims_ = vcat([1], bonddims, [1])
    CPX_BYTES = 16
    ram = 0
    for ib in eachindex(bonddims_)[2:end]
        ram += d*bonddims_[ib]*bonddims_[ib-1]
    end
    return ram*CPX_BYTES/10^6
end

function worstcase_bonddims(L::Int, d::Int=2)
    return [min(d^i, d^(L-i)) for i in 1:(L-1)]
end

function plot_vertex_ranks(tol_range::Vector{Int}, PSFpath::String; folder="pwtcidata_cluster", kwargs...)
    plot_vertex_ranks(10.0 .^ tol_range, PSFpath; folder=folder, kwargs...)
end

function tol_vs_rank_vertex(R::Int, tol_range::Vector{Int}, PSFpath::String; folder="pwtcidata")
    tol_vs_rank_vertex(R::Int, 10.0 .^ tol_range, PSFpath::String; folder=folder)
end

function tol_vs_rank_vertex(R::Int, tol_range, PSFpath::String; folder="pwtcidata")
    p = TCI4Keldysh.default_plot()    
    beta = TCI4Keldysh.dir_to_beta(PSFpath)
    ranks = Int[]
    for tol in tol_range
        file_act = find_Γcore_file(tol, beta; folder=folder)
        if isnothing(file_act)
            @warn "No file for tol=$tol, beta=$beta found!"
        else
            @info "Processing file:\n    $file_act"
        end

        d = TCI4Keldysh.readJSON(file_act, folder)
        Rs = to_intvec(d["Rs"])
        R_idx = findfirst(r -> r==R, Rs)
        if isnothing(R_idx)
            @warn "No data found for tolerance $tol (R=$R)"
            continue
        end
        rank = to_intvec(d["ranks"])[R_idx]
        push!(ranks, rank)
    end

    # plot
    plot!(p, tol_range, ranks; xflip=true, xscale=:log10, marker=:circle, label="")
    title!(p, "Matsubara core vertex, β=$beta")
    xlabel!("tolerance")
    ylabel!("rank")
    savefig("MFvertex_tol_vs_rank$(TCI4Keldysh.tolstr(minimum(tol_range)))to$(TCI4Keldysh.tolstr(maximum(tol_range)))_beta=$(beta)_R=$(R).pdf")
end

function plot_vertex_ranks(tol_range, PSFpath::String; folder="pwtcidata_cluster", subfolder_str=nothing, show_worstcase::Bool=true, ramplot::Bool=false)
    p = TCI4Keldysh.default_plot()    

    beta = TCI4Keldysh.dir_to_beta(PSFpath)

    Rs = []
    for tol in tol_range
        file_act = find_Γcore_file(tol, beta; folder=folder, subfolder_str=subfolder_str)
        if isnothing(file_act)
            @warn "No file for tol=$tol, beta=$beta found!"
        else
            @info "Processing file:\n    $file_act"
        end

        # plot
        d = TCI4Keldysh.readJSON(file_act, folder)
        Rs = to_intvec(d["Rs"])
        ranks = to_intvec(d["ranks"])
        if !ramplot
            @show Rs
            @show ranks
            plot!(p, Rs[1:length(ranks)], ranks; marker=:circle, label=L"tol=$10^{%$(TCI4Keldysh.tolstr(tol))}$")
            ylabel!("rank")
        else
            bonddims = to_intvec.(d["bonddims"])
            rams = [bonddims_to_RAM(b) for b in bonddims]
            plot!(p, Rs[1:length(ranks)], rams; marker=:circle, label=L"tol=$10^{%$(TCI4Keldysh.tolstr(tol))}$")
            ylabel!("RAM [MB]")
        end
    end

    if show_worstcase
        worstcase_ranks = [2^div(3*R,2) for R in Rs]
        # worstcase_rams = [bonddims_to_RAM(worstcase_bonddims(3*R,2)) for R in Rs]
        worstcase_rams = [16 * 2^(3*R) / 10^6 for R in Rs]
        yvals = if ramplot
                worstcase_rams
            else
                worstcase_ranks
            end
        yticks_exp = Int(floor(log10(yvals[1]))):Int(floor(log10(yvals[end])))
        yticks = 10.0 .^ yticks_exp
        yticks_labels = [L"10^{%$y}" for y in yticks_exp]
        label = ramplot ? "dense grid" : "worst case"
        plot!(p, Rs, yvals; label=label, yscale=:log10, color="black", linestyle=:dot, yticks=(yticks, yticks_labels), legend=:topleft)
    end

    title!(p, L"Matsubara core vertex, $\beta=%$beta$")
    xlabel!("R")
    ramstr = ramplot ? "_RAM" : ""
    savefig("MFvertex_ranks_tol=$(TCI4Keldysh.tolstr(minimum(tol_range)))to$(TCI4Keldysh.tolstr(maximum(tol_range)))_beta=$(beta)$(ramstr).pdf")
end

function profile_Γcore()
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    R = 6
    tolerance = 1.e-6
    ωconvMat = 
        [
            0 -1  0;
            1  1  0;
            -1  0 -1;
            0  0  1;
        ]
    beta = 2000.0
    T = 1.0/beta

    # compile
    println("  Compile run...")
    Γcore = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, 4; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=1.e-3, unfoldingscheme=:interleaved
        )
    # profile
    Profile.clear()
    println("  Profiling...")
    Profile.@profile Γcore = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved
        )
    statprofilehtml()
end

"""
test & plot
"""
function test_Gamma_core_TCI_MF(; freq_conv="a", R=4, beta=50.0, tolerance=1.e-5)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg/")
    # PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    T = 1.0 / beta
    logtol = round(Int, log10(tolerance))
    spin = 1
    # spin = 2 # FAILS because F1F1dag Adisc lacks spin 2 component...

    ωconvMat = if freq_conv == "a"
        [
            0 -1  0;
            0  0  1;
            -1  0 -1;
            1  1  0;
        ]
    elseif freq_conv == "p"
        [
            0 -1  0;
            1  0 -1;
            -1  1  0;
            0  0  1;
        ]
    elseif freq_conv == "t"
        # t convention
        [
            0 -1  0;
            1  1  0;
            -1  0 -1;
            0  0  1;
        ]
    else
        error("Invalid frequency convention")
    end

    # compute
    TCI4Keldysh.@TIME Γcore = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=spin, unfoldingscheme=:interleaved, tolerance=tolerance, verbosity=2
        ) "Γcore @ TCI"
    @show TCI4Keldysh.rank(Γcore)

    # reference

        # grids
    ω_bos = TCI4Keldysh.MF_grid(T, 2^(R-1), false)
    ω_fer = TCI4Keldysh.MF_grid(T, 2^(R-1), true)
    ω_fer_int = TCI4Keldysh.MF_grid(T, 2^R, true)
    ωs_ext=(ω_bos, ω_fer, ω_fer)

        # sIE self-energy
    U = 0.05
    G        = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_aux    = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_QQ_aux = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "Q1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_data      = TCI4Keldysh.precompute_all_values(G);
    G_aux_data  = TCI4Keldysh.precompute_all_values(G_aux)
    G_QQ_aux_data=TCI4Keldysh.precompute_all_values(G_QQ_aux)
    Σ_calc_sIE = TCI4Keldysh.calc_Σ_MF_sIE(G_QQ_aux_data, G_aux_data, G_aux_data, G_data, U/2)

        # Γ core
    TCI4Keldysh.@TIME refval = TCI4Keldysh.compute_Γcore_symmetric_estimator(
        "MF", PSFpath*"4pt/", Σ_calc_sIE; ωs_ext=ωs_ext, T=T, ωconvMat=ωconvMat, flavor_idx=spin
        ) "Γcore @ conventional"
    maxref = maximum(abs.(refval))
    logmaxref = round(Int, log10(maxref))

    # test
    testslice = fill(1:2^R, 3)
    test_qttval = reshape([Γcore(id...) for id in Iterators.product(testslice...)], length.(testslice)...)
    testdiff = abs.(refval[testslice...] .- test_qttval[testslice...]) ./ maxref
    printstyled("---- Γcore rank: $(TCI4Keldysh.rank(Γcore))\n"; color=:blue)
    printstyled("---- Maximum value Γcore: $(maxref)\n"; color=:green)
    printstyled("---- Maximum error: $(maximum(testdiff)) (tol=$tolerance)\n"; color=:green)

    # plot
    slice = [1:2^R, 1:2^R, 2^(R-1)]
    qttval = reshape([Γcore(id...) for id in Iterators.product(slice...)], length.(slice)...)

    scfun = x -> log10(abs(x))
    heatmap(scfun.(qttval[slice[1:2]...]); clim=(logmaxref + logtol - 1, logmaxref + 2))
    savefig("gammacore.pdf")

    heatmap(scfun.(refval)[slice...]; clim=(logmaxref + logtol - 1, logmaxref + 2))
    savefig("gammacore_ref.pdf")

    diff = abs.(refval[slice...] .- qttval[slice[1:2]...]) ./ maxref
    heatmap(log10.(diff))
    savefig("diff.pdf")
end

"""
test & plot
"""
function test_K2_TCI(; channel="t", R=4, tolerance=1.e-5, prime=false)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    # PSFpath = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg/")
    beta = TCI4Keldysh.dir_to_beta(PSFpath)
    T = 1.0 / beta
    logtol = round(Int, log10(tolerance))
    flavor = 1

    printstyled("---- REFERENCE\n"; color=:blue)
        # grids
    ω_fer_int = TCI4Keldysh.MF_grid(T, 2^R, true)
    ωs_ext=TCI4Keldysh.MF_npoint_grid(T, 2^(R-1), 2)
        # sIE self-energy
    U = 0.05
    G        = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=flavor, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_aux    = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=flavor, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_QQ_aux = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "Q1dag"]; T, flavor_idx=flavor, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_data      = TCI4Keldysh.precompute_all_values(G)
    G_aux_data  = TCI4Keldysh.precompute_all_values(G_aux)
    G_QQ_aux_data=TCI4Keldysh.precompute_all_values(G_QQ_aux)
    Σ_calc_sIE = TCI4Keldysh.calc_Σ_MF_sIE(G_QQ_aux_data, G_aux_data, G_aux_data, G_data, U/2)


    (i,j) = TCI4Keldysh.merged_legs_K2(channel, prime)
    nonij = sort(setdiff(1:4, (i,j)))
    leg_labels = ("1", "1dag", "3", "3dag")
    op_labels = ("Q$i$j", leg_labels[nonij[1]], leg_labels[nonij[2]])
    K2ref = TCI4Keldysh.compute_K2r_symmetric_estimator(
        "MF", PSFpath, op_labels, Σ_calc_sIE;
        ωs_ext=ωs_ext, T=T, flavor_idx=flavor, ωconvMat=TCI4Keldysh.channel_trafo_K2(channel, prime)
    )

    if channel=="p"
        @assert maximum(abs.(K2ref)) <= 1.e-12
        return
    end

    # TCI
    printstyled("---- TCI\n"; color=:blue)
    K2qtt = TCI4Keldysh.K2_TCI(
        PSFpath, R, channel, prime;
        T=T, flavor_idx=flavor, tolerance=tolerance, unfoldingscheme=:interleaved
        )

    @show TCI4Keldysh.rank(K2qtt)

    # test & plot
    printstyled("---- PLOT\n"; color=:blue)
    testslice = fill(1:2^R, 2)
    test_qttval = TCI4Keldysh.QTT_to_fatTensor(K2qtt, testslice)
    maxref = maximum(abs.(K2ref[testslice...]))
    diff = abs.(test_qttval .- K2ref[testslice...])
    maxdiff = maximum(diff) / maxref
    @assert maxdiff < 5.0 * tolerance

    clim_max = log10(maxref)
    clim = (clim_max + logtol - 1.0, clim_max)
    heatmap(log10.(abs.(test_qttval)); clim=clim)
    # heatmap(log10.(abs.(test_qttval)))
    savefig("K2.pdf")
    heatmap(log10.(abs.(K2ref)); clim=clim)
    savefig("K2_ref.pdf")
end

"""
This shows that K2==K2' in both flavors
"""
function compare_K2_K2prime(channel="t", formalism="MF")
    
    #precompute_K2r(PSFpath::String, flavor_idx::Int, formalism="MF"; ωs_ext::NTuple{2,Vector{Float64}}, channel="t", prime=false)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
    Nhalf = 32
    T = TCI4Keldysh.dir_to_T(PSFpath)
    ωs_ext = if formalism=="MF"
            TCI4Keldysh.MF_npoint_grid(T, Nhalf, 2)
        else
            ωmax = 0.3183098861837907
            TCI4Keldysh.KF_grid(ωmax, 5, 2)
        end
    flavor_idx = 2
    K2 = TCI4Keldysh.precompute_K2r(PSFpath, flavor_idx, formalism; ωs_ext=ωs_ext, channel=channel, prime=false)
    K2prime = TCI4Keldysh.precompute_K2r(PSFpath, flavor_idx, formalism; ωs_ext=ωs_ext, channel=channel, prime=true)

    @show maximum(abs.(K2 - K2prime))
    @show maximum(abs.(K2 + K2prime))
    @show maximum(abs.(K2 .- conj.(K2prime)))
    if formalism=="MF"
        heatmap(log10.(abs.(K2)))
        savefig("K2.pdf")
        heatmap(log10.(abs.(K2prime)))
        savefig("K2prime.pdf")
    else
        for ik in Iterators.product(fill([1,2],3)...)
            println("ik=$ik")
            @show norm([K2[:,:,ik...]])
            @show norm([K2prime[:,:,ik...]])
        end
    end
end

function check_serialized_files()
    failcount = 0
    successcount = 0
    files = filter(f -> occursin("gammacore", f) && occursin("serialized", f), readdir("pwtcidata"))
    for file in files
        try
            qtt = deserialize(joinpath("pwtcidata", file))
            successcount += 1
        catch
            printstyled("Failed with file $file\n"; color=:red)
            failcount += 1
        end
    end
    @show length(files)
    @show failcount
    @show successcount
end

folder = "pwtcidata_KCS"
# PSFpath = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg/")
PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
# plot_K12_ranks_MF(PSFpath)

# plot_vertex_ranks(collect(-7:-2), PSFpath; folder=folder, subfolder_str="shellpivot", show_worstcase=true, ramplot=true)

R = 5
beta = 2000
tol = 2
# thedirname = "gamcoreMF_tol$(tol)_beta$(beta)_nz4_aIE_shellpivot"
thedirname = "gamcoreMF_tol$(tol)_beta$(beta)_updown"
R = 8
beta = 2000
tol = 3
thedirname = "gamcoreMF_tol$(tol)_beta$(beta)_nz4_aIE_shellpivot"
# thedirname = "gamcoreMF_tol$(tol)_beta$(beta)_updown"
qttfile = "gammacore_timing_R_min=5_max=12_tol=-$(tol)_beta=$(beta).0_R=$(R)_qtt.serialized"
# # check_interpolation(joinpath(thedirname, qttfile), R, PSFpath; folder=folder)
# triptych_vertex_data(joinpath(thedirname, qttfile), R, PSFpath; folder=folder, store=true)

# VERTEX RANK PLOT
# plot_vertex_ranks(collect(-5:-2), PSFpath; folder=folder, subfolder_str="shellpivot", ramplot=true)

# tol_vs_rank_vertex(10, [-2, -3, -4, -5, -6], PSFpath; folder="pwtcidata")
# h5file = "vertex_MF_slice_beta=$(beta).0_slices=(1, 128, 128)_tol=-$tol.h5"
# triptych_vertex_plot(h5file, qttfile; folder=joinpath(folder,thedirname))

# MERGE JSON FILES
# file1 = joinpath(TCI4Keldysh.pdatadir(), folder, "gamcoreMF_tol5_beta200_nz4_aIE_shellpivot/gammacore_timing_R_min=5_max=12_tol=-5_beta=200.0.json")
# file2 = joinpath(TCI4Keldysh.pdatadir(), folder, "gamcoreMF_tol5_beta200_nz4_aIE_shellpivot_1012/gammacore_timing_R_min=10_max=12_tol=-5_beta=200.0.json")
# merge_jsondata(file1, file2)

# plot_K1_zoomed()
# plot_K1_ranks_KF(PSFpath)