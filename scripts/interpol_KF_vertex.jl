using QuanticsTCI
using LinearAlgebra
using Plots
using Serialization
using HDF5
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