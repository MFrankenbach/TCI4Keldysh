using PythonCall
using PythonPlot
using QuanticsTCI
using Serialization
using LaTeXStrings
using HDF5
using Printf
using LinearAlgebra
import TensorCrossInterpolation as TCI

INCH_TO_PT = 72
PT_TO_INCH = 1.0/INCH_TO_PT
COLUMN_PT = 432
COLUMN_INCH = COLUMN_PT*PT_TO_INCH
PLOT_COLUMN_PT = 420
PLOT_COLUMN_INCH = PLOT_COLUMN_PT*PT_TO_INCH
PLOT_PAGE_PT = PLOT_COLUMN_PT*2
PLOT_PAGE_INCH = PLOT_PAGE_PT*PT_TO_INCH

# ==== UTILS

"""
Find file for given beta and tolerance with maximum R-range
"""
function find_Γcore_file(tolerance::Float64, beta::Float64; folder="pwtcidata", subdir_str=nothing)
    if !isnothing(subdir_str)
        # files are distributed in sub-folders
        folder_content = readdir(folder)
        subdirs = [f for f in folder_content if isdir(joinpath(folder, f))]
        function _folder_relevant(f)
            return occursin(subdir_str,f) && occursin("beta$(round(Int,beta))",f) && occursin("gamcore",f) && occursin("tol$(-round(Int,log10(tolerance)))",f)
        end
        @show subdirs
        subdirs = filter(_folder_relevant, subdirs)
        @show subdirs
        files = [only(filter(f -> endswith(f,".json") && !occursin("original", f), readdir(joinpath(folder,sd)))) for sd in subdirs]
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

"""
Find file for given beta and tolerance with maximum R-range
"""
function find_GF_file(tolerance::Float64, beta::Float64; folder="pwtcidata", subdir_str=nothing)
    if isnothing(subdir_str)
        function _file_relevant(f)
            desired = endswith(f, ".json") && occursin("beta=$beta", f) && occursin("tol=$(TCI4Keldysh.tolstr(tolerance))", f) && startswith(f, "timing")
            allowed = !occursin("gammacore", f) && !occursin("KF", f)
            return desired && allowed
        end
        files = filter(
                _file_relevant,
                readdir(folder)
                )
            
        # @show files
    else
        # files are distributed in sub-folders
        folder_content = readdir(folder)
        subdirs = [f for f in folder_content if isdir(joinpath(folder, f))]
        # @show subdirs
        function _folder_relevant(f)
            return occursin(subdir_str,f) && occursin("beta$(round(Int,beta))_",f) && occursin("corrMF",f) && occursin("tol$(-round(Int,log10(tolerance)))",f)
        end
        subdirs = filter(_folder_relevant, subdirs)
        files = [only(filter(f -> endswith(f,".json"), readdir(joinpath(folder,sd)))) for sd in subdirs]
        files = [joinpath(subdirs[i], files[i]) for i in eachindex(subdirs)]
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

function to_intvec(x) :: Vector{Int}
    return convert(Vector{Int}, x)
end

# ==== UTILS END

# ==== PLOT UTILS
"""
Standard pyplot.rcParams
"""
function set_rcParams(fs::Int=12)
    pyplot.rcParams["font.size"] = fs        # Title font size
    # pyplot.rcParams["axes.labelsize"] = fs               # Axis label font size
    pyplot.rcParams["xtick.labelsize"] = fs             # X-axis tick label font size
    pyplot.rcParams["ytick.labelsize"] = fs             # Y-axis tick label font size
    pyplot.rcParams["legend.fontsize"] = fs             # Legend font size
end

function annotate_topleft(ax, text; color="black")
    subplotlabeloffset=3
    ax.annotate(
    text,
    xy=(0,1),
    xycoords="axes fraction",
    horizontalalignment="left",
    verticalalignment="top",
    xytext=(+subplotlabeloffset, -subplotlabeloffset),
    textcoords="offset points",
    color=color
    )
end

function abc_annotate(axs; color="black")
    subplotlabeloffset=3
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    naxs = length(axs.flatten())
    for i in 0:naxs-1
        axs.flatten()[i].annotate(
        "($(alphabet[i+1]))",
        xy=(1,1),
        xycoords="axes fraction",
        horizontalalignment="right",
        verticalalignment="top",
        xytext=(-subplotlabeloffset, -subplotlabeloffset),
        textcoords="offset points",
        color=color
        )
    end
end

"""
Save figure including a legend outside the figure (bbox_to_anchor with entries > 1.0)
"""
function save_bbox(name::String, fig, lgd)
    fig.savefig(
        name,
        bbox_extra_artists=(lgd,),
        bbox_inches="tight"
        )
end
# ==== PLOT UTILS END


"""
Plot kernel singular values of Keldysh kernel.
Compare with Matsubara kernel
"""
function plot_kernel_singvals_KF(R::Int; ωmax::Float64=1.0)
    # create correlator
    basepath = "SIAM_u=0.50"
    nz = 4
    PSFpath = joinpath(TCI4Keldysh.datadir(), basepath, "PSF_nz=$(nz)_conn_zavg/4pt/")
    beta = TCI4Keldysh.dir_to_beta(PSFpath)
    npt = 4
    D = npt-1
    Ops = TCI4Keldysh.dummy_operators(npt)
    T = TCI4Keldysh.dir_to_T(PSFpath)
    ωmin = -ωmax
    ωs_ext = ntuple(i -> collect(range(ωmin, ωmax; length=2^R)), D)
    channel = "t"
    ωconvMat = TCI4Keldysh.channel_trafo("t")
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(basepath; channel=channel)
    broadening_kwargs = TCI4Keldysh.read_broadening_settings(basepath; channel=channel)
    KFC = TCI4Keldysh.FullCorrelator_KF(
        PSFpath, Ops;
        T=T, ωs_ext=ωs_ext, flavor_idx=1, ωconvMat=ωconvMat, sigmak=sigmak, γ=γ, name="Kentucky fried chicken",
        broadening_kwargs...
        )

    # Matsubara correlator for comparison
    GF = TCI4Keldysh.FullCorrelator_MF(
        PSFpath, Ops;
        T=T, ωs_ext=ωs_ext, flavor_idx=1, ωconvMat=ωconvMat, is_compactAdisc=true
        )

    # SVD kernels
    Gp = KFC.Gps[1]
    k = Gp.tucker.legs[1]
    _,S,_ = svd(k)

    kMF = GF.Gps[1].tucker.legs[1]
    _,SMF,_ = svd(kMF)

    kMFfer = GF.Gps[1].tucker.legs[2]

    set_rcParams(12)

    fig, axs = subplots(figsize=(COLUMN_INCH, COLUMN_INCH*9/16))
    axs.grid(true)

    # should we normalize?
    normalize = false
    s0 = normalize ? maximum(S) : 1.0
    s0MF = normalize ? maximum(SMF) : 1.0
    yvals = S ./ s0
    yvalsMF = SMF ./ s0MF
    xvals = collect(1:length(S))
    xvalsMF = collect(1:length(SMF))
    h2, = axs.plot(xvalsMF, yvalsMF; linestyle="--", color="gray", label=L"\sigma_i\left((\mathrm{i}\omega-\omega')^{-1}\right)")
    h1, = axs.plot(xvals, yvals; color="blue", label=L"\sigma_i\left(k^{[0,0]}_b\right)")
    axs.set_yscale("log")
    axs.set_xlabel(L"i")
    ylabel = normalize ? L"\sigma_i/\sigma_0" : L"\sigma_i"
    axs.set_ylabel(ylabel)

    ommax_str = @sprintf "%.3f" ωmax
    # axs.set_title(L"Singular values of $k^{[0,0]}_b$, $\beta=%$beta$, $ω_{\mathrm{max}}=%$ommax_str$")
    axs.set_title("Kernel singular values: Keldysh vs. Matsubara")
    axs.legend(handles=[h1,h2])
    fig.tight_layout()
    savefig("keldyshsvd.pdf")
end


"""
Two betas in one plot
"""
function plot_vertex_ranks_both(tol_range; folder="pwtcidata", subdir_str=nothing, show_worstcase=true, ramplot=false)

    pyplot.rcParams["font.size"] = 12        # Title font size
    # pyplot.rcParams["axes.labelsize"] = 14               # Axis label font size
    pyplot.rcParams["xtick.labelsize"] = 12             # X-axis tick label font size
    pyplot.rcParams["ytick.labelsize"] = 12             # Y-axis tick label font size
    pyplot.rcParams["legend.fontsize"] = 12             # Legend font size

    fig, axs = subplots(1,2, figsize=(PLOT_PAGE_INCH, PLOT_PAGE_INCH*3/10))
    axs[0].grid(true)
    axs[1].grid(true)

    PSFpath1 = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
    PSFpath2 = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg/")
    beta1 = TCI4Keldysh.dir_to_beta(PSFpath1)
    beta2 = TCI4Keldysh.dir_to_beta(PSFpath2)

    Rs = []
    allplots = []
    colors = ["red", "blue", "green", "magenta", "brown"]
    markers = ["o", "d"]
    for (it, tol) in enumerate(tol_range)
        file_act1 = find_Γcore_file(tol, beta1; folder=folder, subdir_str=subdir_str)
        file_act2 = find_Γcore_file(tol, beta2; folder=folder, subdir_str=subdir_str)
        files = [file_act1,file_act2]
        for i in eachindex(files)
            file_act = files[i]
            if isnothing(file_act)
                @warn "No file for tol=$tol found!"
            else
                @info "Processing file:\n    $file_act"
            end

            # plot
            d = TCI4Keldysh.readJSON(file_act, folder)
            Rs = to_intvec(d["Rs"])
            bonddims = to_intvec.(d["bonddims"])
            rams = TCI4Keldysh.bonddims_to_RAM.(bonddims)
            rranks = to_intvec(d["ranks"])
            @show Rs
            @show rranks
            # rank plot
            pact = axs[0].plot(Rs[1:length(rranks)], rranks; color=colors[it], marker=markers[i], label="tol=$(TCI4Keldysh.tolstr(tol))")
            push!(allplots, pact) 
            # RAM plot
            pact = axs[1].plot(Rs[1:length(rranks)], rams; color=colors[it], marker=markers[i], label="tol=$(TCI4Keldysh.tolstr(tol))")
            push!(allplots, pact) 
        end
    end

    worst_line = nothing
    if show_worstcase
        worstcase_ranks = [2^div(3*R,2) for R in Rs]
        worstcase_rams = [16 * 2^(3*R) / 10^6 for R in Rs]
        label = ramplot ? "dense grid" : "worst case"
        worst_line, = axs[0].plot(Rs, worstcase_ranks, label=label, color="black", linestyle=":", marker="None")
        axs[1].plot(Rs, worstcase_rams, label=label, color="black", linestyle=":", marker="None")

        for i in [0,1]
            yticks_exp = if i==0
                    Int(floor(log10(worstcase_ranks[1]))):Int(floor(log10(worstcase_ranks[end])))
                else
                    Int(floor(log10(worstcase_rams[1]))):Int(floor(log10(worstcase_rams[end])))
                end
            yticks = 10.0 .^ yticks_exp
            yticks_labels = [L"10^{%$y}" for y in yticks_exp]

            axs[i].set_yscale("log")
            axs[i].set_yticks(yticks)
            axs[i].set_yticklabels(yticks_labels)
        end
    end

    axs[0].set_title(L"Matsubara $\Gamma_{\mathrm{core}}$ ranks")
    axs[0].set_xlabel(L"R")
    axs[0].set_ylabel("rank")
    (y1, y2) = axs[0].get_ylim()
    @show (y1,y2)
    axs[0].set_ylim(y1, 10^3)
    axs[1].set_title(L"Matsubara $\Gamma_{\mathrm{core}}$ RAM")
    axs[1].set_xlabel(L"R")
    (y1, y2) = axs[1].get_ylim()
    axs[1].set_ylim(y1, 10^4)
    axs[1].set_ylabel("RAM [MB]")

    # annotate
    abc_annotate(axs; color="black")

    handles = onecol_legend_colormarker(
        colors,
        markers,
        [L"tol=$10^{%$(TCI4Keldysh.tolstr(tol))}$" for tol in tol_range],
        [L"\beta=2000",L"\beta=200"]
    )

    lgd = fig.legend(handles=vcat(handles, [worst_line]), ncols=1, bbox_to_anchor=(1.04,0.90))
    # fig.tight_layout()
    save_bbox(
        "MFvertex_ranks_tol=$(TCI4Keldysh.tolstr(minimum(tol_range)))to$(TCI4Keldysh.tolstr(maximum(tol_range))).pdf",
        fig,
        lgd
    )
end

function plot_beta_vs_rank(betas_V_MF, ranks_V_MF)
    set_rcParams()
    fig, axs = subplots(figsize=(PLOT_COLUMN_INCH, PLOT_PAGE_INCH*9/16))
    # matsubara vertex
    axs.plot(betas_V_MF, ranks_V_MF)
    axs.set_xlabel(L"\beta")
    axs.set_ylabel(L"rank")
end


"""
legend with one column, labels for colors and markers
"""
function onecol_legend_colormarker(colors, markers, collabels, markerlabels)
    @assert length(colors)>=length(collabels)
    @assert length(markers)>=length(markerlabels)

    mlines = pyimport("matplotlib.lines")
    collines = [mlines.Line2D([], [], color=colors[j], label=collabels[j]) for j in eachindex(collabels)]
    markerlines = [mlines.Line2D([], [], color="black", marker=markers[j], label=markerlabels[j], linestyle="None") for j in eachindex(markerlabels)]

    handles = vcat(collines, markerlines)
    return handles
end

"""
tabular legend with same markers in columns and same colors in rows
"""
function table_legend_colormarker(colors, markers, coltitles, rowtitles, uppercorner="")
    @assert length(colors)>=length(rowtitles)
    @assert length(markers)>=length(coltitles)

    mlines = pyimport("matplotlib.lines")
    # does not work
    # mtext = pyimport("matplotlib.text")
    firstcol = [mlines.Line2D([], [], color="none", label=uppercorner)]
    firstcol = vcat(firstcol, [mlines.Line2D([], [], color="none", label=rt) for rt in rowtitles])
    lines = [mlines.Line2D([], [], color=colors[i], marker=markers[j]) for j in eachindex(coltitles) for i in eachindex(rowtitles)]

    clabels = [mlines.Line2D([], [], color="none", label=ct) for ct in coltitles]
    nrows = length(rowtitles)
    for i in eachindex(clabels)
        insert!(lines, (i-1)*nrows + i, clabels[i])
    end

    handles = vcat(firstcol, lines)
    return handles
end

"""
Two betas in one plot
"""
function plot_FullCorrelator_ranks_both(tol_range; folder="pwtcidata", subdir_str=nothing, show_worstcase=true, ramplot=false)

    pyplot.rcParams["font.size"] = 12        # Title font size
    # pyplot.rcParams["axes.labelsize"] = 14               # Axis label font size
    pyplot.rcParams["xtick.labelsize"] = 12             # X-axis tick label font size
    pyplot.rcParams["ytick.labelsize"] = 12             # Y-axis tick label font size
    pyplot.rcParams["legend.fontsize"] = 12             # Legend font size

    fig, axs = subplots(1,2, figsize=(PLOT_PAGE_INCH, PLOT_PAGE_INCH*3/10))
    axs[0].grid(true)
    axs[1].grid(true)

    PSFpath1 = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
    PSFpath2 = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg/")
    beta1 = TCI4Keldysh.dir_to_beta(PSFpath1)
    beta2 = TCI4Keldysh.dir_to_beta(PSFpath2)

    Rs = []
    allplots = []
    colors = ["red", "blue", "green", "magenta", "brown"]
    markers = ["o", "d"]
    for (it, tol) in enumerate(tol_range)
        file_act1 = find_GF_file(tol, beta1; folder=folder, subdir_str=subdir_str)
        file_act2 = find_GF_file(tol, beta2; folder=folder, subdir_str=subdir_str)
        files = [file_act1,file_act2]
        for i in eachindex(files)
            file_act = files[i]
            if isnothing(file_act)
                @warn "No file for tol=$tol found!"
            else
                @info "Processing file:\n    $file_act"
            end

            # plot
            d = TCI4Keldysh.readJSON(file_act, folder)
            Rs = to_intvec(d["Rs"])
            bonddims = to_intvec.(d["bonddims"])
            rams = TCI4Keldysh.bonddims_to_RAM.(bonddims)
            rranks = to_intvec(d["ranks"])
            @show Rs
            @show rranks
            # rank plot
            pact = axs[0].plot(Rs[1:length(rranks)], rranks; color=colors[it], marker=markers[i], label="tol=$(TCI4Keldysh.tolstr(tol))")
            push!(allplots, pact) 
            # RAM plot
            pact = axs[1].plot(Rs[1:length(rranks)], rams; color=colors[it], marker=markers[i], label="tol=$(TCI4Keldysh.tolstr(tol))")
            push!(allplots, pact) 
        end
    end

    worst_lines = []
    if show_worstcase
        worstcase_ranks = [2^div(3*R,2) for R in Rs]
        worstcase_rams = [16 * 2^(3*R) / 10^6 for R in Rs]
        label = ramplot ? "dense grid" : "worst case"
        wc, = axs[0].plot(Rs, worstcase_ranks, label=label, color="black", linestyle=":", marker="None")
        push!(worst_lines, wc)
        wc, = axs[1].plot(Rs, worstcase_rams, label=label, color="black", linestyle=":", marker="None")
        push!(worst_lines, wc)

        for i in [0,1]
            yticks_exp = if i==0
                    Int(floor(log10(worstcase_ranks[1]))):Int(floor(log10(worstcase_ranks[end])))
                else
                    Int(floor(log10(worstcase_rams[1]))):Int(floor(log10(worstcase_rams[end])))
                end
            yticks = 10.0 .^ yticks_exp
            yticks_labels = [L"10^{%$y}" for y in yticks_exp]

            axs[i].set_yscale("log")
            axs[i].set_yticks(yticks)
            axs[i].set_yticklabels(yticks_labels)
        end
    end

    axs[0].set_title("Matsubara correlator ranks")
    axs[0].set_xlabel("R")
    axs[0].set_ylabel("rank")
    (y1, y2) = axs[0].get_ylim()
    @show (y1,y2)
    axs[0].set_ylim(y1, 1000)
    axs[1].set_title("Matsubara correlator RAM")
    axs[1].set_xlabel("R")
    (y1, y2) = axs[1].get_ylim()
    axs[1].set_ylim(y1, 10^4)
    axs[1].set_ylabel("RAM [MB]")

    # annotate
    abc_annotate(axs)

    # handles = table_legend_colormarker(
    #     colors,
    #     markers, 
    #     [L"\beta=2000",L"\beta=200"], [L"$10^{%$(TCI4Keldysh.tolstr(tol))}$" for tol in tol_range],
    #     "tol"
    #     )

    handles = onecol_legend_colormarker(
        colors,
        markers,
        [L"tol=$10^{%$(TCI4Keldysh.tolstr(tol))}$" for tol in tol_range],
        [L"\beta=2000",L"\beta=200"]
    )

    lgd = fig.legend(handles=vcat(handles, worst_lines[1]), ncols=1, bbox_to_anchor=(1.04,0.9))
    # fig.legend(handles=vcat(handles, worst_lines[1]), ncols=1, loc="outside right")
    # fig.tight_layout()
    save_bbox(
        "MFcorr_ranks_tol=$(TCI4Keldysh.tolstr(minimum(tol_range)))to$(TCI4Keldysh.tolstr(maximum(tol_range))).pdf",
        fig,
        lgd
    )
end

"""
Tolerance vs. rank of different objects
"""
function tol_vs_rank(R::Int, tol_range; folder="pwtcidata", subdir_str=nothing)
    
    fig, axs = subplots(figsize=(PLOT_COLUMN_INCH, PLOT_COLUMN_INCH*3/5))
    set_rcParams(12)

    PSFpaths = [
        joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg/"),
        joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
        ]

    ranks_paths_gam = []
    ranks_paths_GF = []
    for PSFpath in PSFpaths
        beta = TCI4Keldysh.dir_to_beta(PSFpath)
        ranksGF = Int[]
        ranksgam = Int[]
        # collect gammacore data
        for tol in tol_range
            file_act_gam = find_Γcore_file(tol, beta; folder=folder, subdir_str=subdir_str)
            file_act_GF = find_GF_file(tol, beta; folder=folder, subdir_str=subdir_str)
            if isnothing(file_act_gam) || isnothing(file_act_GF)
                @warn "No file for tol=$tol, beta=$beta found!"
            else
                @info "Processing file:\n    $file_act_gam"
                @info "Processing file:\n    $file_act_GF"
            end

            dgam = TCI4Keldysh.readJSON(file_act_gam, folder)
            dGF = TCI4Keldysh.readJSON(file_act_GF, folder)
            for id in [1,2]
                d = id==1 ? dgam : dGF
                Rs = to_intvec(d["Rs"])
                R_idx = findfirst(r -> r==R, Rs)
                if isnothing(R_idx)
                    @warn "No data found for tolerance $tol (R=$R)"
                    continue
                end
                rank = to_intvec(d["ranks"])[R_idx]
                if id==1
                    push!(ranksgam, rank)
                else
                    push!(ranksGF, rank)
                end
            end
        end
        push!(ranks_paths_gam, ranksgam)
        push!(ranks_paths_GF, ranksGF)
    end

    # plot
    axs.grid(true)

    # beta 200
    axs.plot(tol_range, ranks_paths_gam[1]; color="blue", marker="d", label=L"\Gamma_{\mathrm{core}}, \beta=200")
    axs.plot(tol_range, ranks_paths_GF[1]; color="green", marker="d", label=L"G, \beta=200")
    # beta 2000
    axs.plot(tol_range, ranks_paths_gam[2]; color="blue", marker="o", label=L"\Gamma_{\mathrm{core}}, \beta=2000")
    axs.plot(tol_range, ranks_paths_GF[2]; color="green", marker="o", label=L"G, \beta=2000")
    axs.set_xscale("log")
    axs.set_ylabel("rank")
    axs.set_xlabel("tolerance")
    axs.set_title("Matsubara: tolerance vs rank")
    fig.legend(bbox_to_anchor=(0.9,0.9))
    tight_layout()
    savefig("MF_tol_vs_ranks=$(TCI4Keldysh.tolstr(minimum(tol_range)))to$(TCI4Keldysh.tolstr(maximum(tol_range))).pdf")
end

function plot_K12_ranks_MF(PSFpath;channel="t", flavor_idx=1)
    # prime=false since K2≡K2' in MF
    tols = collect(10.0 .^ (-5:1:-2))
    Rs = 5:10
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
            ranks[iR,it] = TCI.rank(qtt[1].tci)

            # K2
            qtt2 = TCI4Keldysh.K2_TCI_precomputed(
                PSFpath, R;
                channel=channel,
                prime=false,
                formalism="MF",
                flavor_idx=flavor_idx,
                T=T,
                tolerance=tol,
                unfoldingscheme=:interleaved
                )
            ranks2[iR,it] = TCI.rank(qtt2[1].tci)
        end
    end

    set_rcParams(12)
    fig, axs = subplots(figsize=(PLOT_COLUMN_INCH, PLOT_COLUMN_INCH*3/4), layout="constrained")
    axs.grid(true)

    # markers = ["o", "^"]
    colors = ["red", "blue", "green", "magenta", "brown"]
    # K1
    for it in eachindex(tols)
        tolexp = round(Int, log10(tols[it]))
        axs.plot(Rs, ranks[:,it]; label=L"$K_1^{%$channel}$, tol=$10^{%$tolexp}$", marker="o", color=colors[it], linestyle="dashed")
    end
    # K2
    for it in eachindex(tols)
        tolexp = round(Int, log10(tols[it]))
        axs.plot(Rs, ranks2[:,it]; label=L"$K_2^{%$channel}$, tol=$10^{%$tolexp}$", marker="^", color=colors[it])
    end

    axs.set_xlabel("R")
    axs.set_ylabel("rank")
    axs.set_title(L"Matsubara: $K_1^{%$channel}$, $K_2^{%$channel}$")
    lgd = fig.legend(bbox_to_anchor=(1.30,0.95))
    save_bbox("K12_ranks_MF.pdf", fig, lgd)
    # savefig("K12_ranks_MF.pdf")
end

function plot_K12_ranks_KF(PSFpath, ωmax::Float64;channel="t", flavor_idx=1)
    tols = reverse(collect(10.0 .^ (-4:1:-2)))
    Rs = 5:12
    T = TCI4Keldysh.dir_to_T(PSFpath)
    iK = (2,2)
    # K1
    ranks = zeros(Int, length(Rs), length(tols))
    # K2
    ranks2 = zeros(Int, length(Rs), length(tols))
    for (it,tol) in enumerate(tols)
        for (iR,R) in enumerate(Rs)
            # K1
            qtt = TCI4Keldysh.K1_TCI(
                PSFpath, R;
                channel=channel,
                formalism="KF",
                flavor_idx=flavor_idx,
                T=T,
                ωmax=ωmax,
                tolerance=tol,
                unfoldingscheme=:interleaved
                )
            ranks[iR,it] = TCI.rank(qtt[iK...].tci)

            # K2
            qtt2 = TCI4Keldysh.K2_TCI_precomputed(
                PSFpath, R;
                channel=channel,
                prime=false,
                formalism="KF",
                flavor_idx=flavor_idx,
                T=T,
                ωmax=ωmax,
                tolerance=tol,
                unfoldingscheme=:interleaved
                )
            ranks2[iR,it] = TCI.rank(qtt2[1].tci)
        end
    end
    set_rcParams(20)
    fig, axs = subplots(figsize=(PLOT_COLUMN_INCH, PLOT_COLUMN_INCH*3/4))
    for it in eachindex(tols)
        tolexp = round(Int, log10(tols[it]))
        axs.plot(Rs, ranks[:,it]; label=L"tol=$10^{%$tolexp}$", marker="o", linestyle="dashed")
        axs.plot(Rs, ranks2[:,it]; label=L"$K^2$,tol=$10^{%$tolexp}$", marker="^")
    end
    axs.set_xlabel("R")
    axs.set_ylabel("rank")
    axs.set_title(L"Keldysh: $K^1_{%$channel}$, $K^2_{%$channel}$")
    axs.grid(true)
    savefig("K12_ranks_KF.pdf")
end


function triptych_corr_plot(h5files; folder="")

    nrows = length(h5files)

    # read data
    refvals_ = [h5read(joinpath(folder, f), "reference") for f in h5files]
    qttvals_ = [h5read(joinpath(folder, f), "qttdata") for f in h5files]
    diffs_ = [h5read(joinpath(folder, f), "diff") for f in h5files]
    maxrefs = [h5read(joinpath(folder, f), "maxref") for f in h5files]
    # normalize errors
    diffs_ = [diffs_[i] ./ maxrefs[i] for i in eachindex(h5files)]

    # eliminate singleton dims
    dvecs_ = [refvals_, qttvals_, diffs_]
    refvals = Vector{Matrix{ComplexF64}}(undef, nrows)
    qttvals = Vector{Matrix{ComplexF64}}(undef, nrows)
    diffs = Vector{Matrix{ComplexF64}}(undef, nrows)
    dvecs = [refvals, qttvals, diffs]
    for (id, dvec_) in enumerate(dvecs_)
        for i in eachindex(dvec_)
            if ndims(dvec_[i])==3
                sdims = findall(j -> size(dvec_[i], j)==1, 1:3)
                dvecs[id][i] = dropdims(dvec_[i]; dims=tuple(sdims...))
            end
        end
    end

    # # get corresponding parameters
    # qtt_datas = [TCI4Keldysh.readJSON(qttfile_to_json(qttfile), folder) for qttfile in qttfiles]
    # tolerances = [qd["tolerance"] for qd in qtt_datas]
    # betas = [qd["beta"] for qd in qtt_datas]
    tolerances = fill(0.0001, nrows)


    # plot
    fig, axes = subplots(
        nrows,
        3,
        figsize=(PLOT_PAGE_INCH, PLOT_PAGE_INCH*nrows*1.3/5),
        # sharex=true,
        # sharey=true,
        layout="compressed"
        )
    # for special case nrow=1
    axs = reshape(pyconvert(Array, axes), (nrows,3))
    # assume the same sizes everywhere
    xsz, ysz = size(refvals[1])
    @assert xsz==ysz "Non-square data grid?"
    xvals = -div(xsz,2)-1 : xsz-div(xsz,2)
    # yvals = ysz-div(ysz,2)-1 : ysz+div(ysz,2)-1
    # steps of 5, including 0
    step = 50
    neg_0tick = -1 * reverse(collect(0:step:-xvals[1]))
    pos_tick = collect(step:step:xvals[end])
    xticks_label = vcat(neg_0tick, pos_tick)
    xticks = xticks_label .- xvals[1]
    yticks_label = copy(xticks_label)
    yticks = yticks_label .- xvals[1]

    xlabel = L"n_{\omega'}"
    ylabel = L"n_{\omega}"

    scfun(x) = log10(abs(x))
    # cols
    for ic in 1:3
        # rows
        for ir in 1:nrows

            logtol_act = log10(tolerances[ir])
            vmin, vmax = if ic==3
                # errorplot
                    (log10(tolerances[ir])-1, log10(tolerances[ir])-6)
                else
                # vertex plot
                    lmaxref = log10(abs(maxrefs[ir]))
                    (lmaxref + logtol_act - 1, lmaxref)
                end
            im = axs[ir,ic].imshow(scfun.(dvecs[ic][ir]), cmap="viridis", interpolation="nearest")
            if ic!=3
                im.set_clim(vmin, vmax)
            end
            if ic==2
                # common colorbar for axs[ir,0] and axs[ir,1]
                fig.colorbar(im, fraction=0.045, ax=axs[ir,1:2], location="right")
            elseif ic==3
                fig.colorbar(im, fraction=0.045, ax=axs[ir,3], location="right")
            end

            axs[ir,ic].invert_yaxis()
            axs[ir,ic].tick_params(axis="both", bottom=(ir==nrows), labelbottom=(ir==nrows), labelleft=(ic==1), left=(ic==1))
            if ic==1
                axs[ir,ic].set_yticks(xticks, labels=string.(xticks_label))
                axs[ir,ic].set_ylabel(ylabel)
            end
            if ir==nrows
                axs[ir,ic].set_xticks(yticks, labels=string.(yticks_label))
                axs[ir,ic].set_xlabel(xlabel)
            end

        end
    end

    abc_annotate(axes; color="white")
    axs[1,1].set_title(L"\log_{10}|G^{\mathrm{ref}}|")
    axs[1,2].set_title(L"\log_{10}|G^{\mathrm{QTCI}}|")
    axs[1,3].set_title(L"\log_{10}\left(|G^{\mathrm{ref}}-G^{\mathrm{QTCI}}|/|G^{\mathrm{ref}}|_\infty\right)")

    # fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, hspace=0.01, wspace=0.01)
    # fig.tight_layout()

    savefig("MFcorr_triptych.pdf")
end


function triptych_vertex_plot(h5files; folder="")

    nrows = length(h5files)

    # read data
    refvals_ = [h5read(joinpath(folder, f), "reference") for f in h5files]
    qttvals_ = [h5read(joinpath(folder, f), "qttdata") for f in h5files]
    diffs_ = [h5read(joinpath(folder, f), "diff") for f in h5files]
    maxrefs = [h5read(joinpath(folder, f), "maxref") for f in h5files]
    # normalize errors
    diffs_ = [diffs_[i] ./ maxrefs[i] for i in eachindex(h5files)]

    # eliminate singleton dims
    dvecs_ = [refvals_, qttvals_, diffs_]
    @show size.(refvals_)
    @show size.(qttvals_)
    @show size.(diffs_)
    refvals = Vector{Matrix{ComplexF64}}(undef, nrows)
    qttvals = Vector{Matrix{ComplexF64}}(undef, nrows)
    diffs = Vector{Matrix{ComplexF64}}(undef, nrows)
    dvecs = [refvals, qttvals, diffs]
    for (id, dvec_) in enumerate(dvecs_)
        for i in eachindex(dvec_)
            if ndims(dvec_[i])==3
                sdims = findall(j -> size(dvec_[i], j)==1, 1:3)
                dvecs[id][i] = dropdims(dvec_[i]; dims=tuple(sdims...))
            end
        end
    end

    # # get corresponding parameters
    # qtt_datas = [TCI4Keldysh.readJSON(qttfile_to_json(qttfile), folder) for qttfile in qttfiles]
    # tolerances = [qd["tolerance"] for qd in qtt_datas]
    # betas = [qd["beta"] for qd in qtt_datas]

    # plot
    fig, axes = subplots(
        nrows,
        3,
        figsize=(PLOT_PAGE_INCH, PLOT_PAGE_INCH*nrows*1/5),
        # sharex=true,
        # sharey=true,
        layout="compressed"
        )
    # for special case nrow=1
    axs = reshape(pyconvert(Array, axes), (nrows,3))
    @show axs
    @show size(axs)

    # CHANGE MANUALLY
    tolerances = fill(0.001, nrows)
    annotate_topleft(axs[1,2], L"\chi=96"; color="white")
    annotate_topleft(axs[2,2], L"\chi=96"; color="white")
    annotate_topleft(axs[3,2], L"\chi=111"; color="white")

    # assume the same sizes everywhere
    xsz, ysz = size(refvals[1])
    @assert xsz==ysz "Non-square data grid?"
    xvals = -div(xsz,2)-1 : xsz-div(xsz,2)
    @show xvals
    # yvals = ysz-div(ysz,2)-1 : ysz+div(ysz,2)-1
    # steps of 5, including 0
    step = 50
    neg_0tick = -1 * reverse(collect(0:step:-xvals[1]))
    pos_tick = collect(step:step:xvals[end])
    xticks_label = vcat(neg_0tick, pos_tick)
    xticks = xticks_label .- xvals[1]
    yticks_label = copy(xticks_label)
    yticks = yticks_label .- xvals[1]

    xlabel = L"n_{\omega'}"
    ylabel = L"n_{\omega}"

    scfun(x) = log10(abs(x))
    # cols
    for ic in 1:3
        # rows
        for ir in 1:nrows

            logtol_act = log10(tolerances[ir])
            vmin, vmax = if ic==3
                # errorplot
                    (log10(tolerances[ir])-1, log10(tolerances[ir])-6)
                else
                # vertex plot
                    lmaxref = log10(abs(maxrefs[ir]))
                    (lmaxref + logtol_act - 1, lmaxref)
                end
            im = axs[ir,ic].imshow(scfun.(dvecs[ic][ir]), cmap="viridis", interpolation="nearest")
            if ic!=3
                im.set_clim(vmin, vmax)
            end
            if ic==2
                # common colorbar for axs[ir,0] and axs[ir,1]
                fig.colorbar(im, fraction=0.045, ax=axs[ir,1:2], location="right")
            elseif ic==3
                fig.colorbar(im, fraction=0.045, ax=axs[ir,3], location="right")
            end

            axs[ir,ic].invert_yaxis()
            axs[ir,ic].tick_params(axis="both", bottom=(ir==nrows), labelbottom=(ir==nrows), labelleft=(ic==1), left=(ic==1))
            if ic==1
                axs[ir,ic].set_yticks(xticks, labels=string.(xticks_label))
                axs[ir,ic].set_ylabel(ylabel)
            end
            if ir==nrows
                axs[ir,ic].set_xticks(yticks, labels=string.(yticks_label))
                axs[ir,ic].set_xlabel(xlabel)
            end

        end
    end

    abc_annotate(axes; color="white")
    axs[1,1].set_title(L"\log_{10}|\Gamma_{\mathrm{core}}^{\mathrm{ref}}|")
    axs[1,2].set_title(L"\log_{10}|\Gamma_{\mathrm{core}}^{\mathrm{QTCI}}|")
    axs[1,3].set_title(L"\log_{10}\left(|\Gamma_{\mathrm{core}}^{\mathrm{ref}}-\Gamma_{\mathrm{core}}^{\mathrm{QTCI}}|/|\Gamma_{\mathrm{core}}^{\mathrm{ref}}|_\infty\right)")

    # fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, hspace=0.01, wspace=0.01)
    # fig.tight_layout()

    savefig("MFvertex_triptych.pdf")
end



# PSFpath = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg/")
PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")

folder="MF_pch_rankdata"

h5file1 = "vertex_MF_slice_beta=2000.0_slices=(0,256,256,)_tol=-2_upup.h5"
h5file2 = "vertex_MF_slice_beta=2000.0_slices=(5,256,256,)_tol=-3_upup.h5"
h5file3 = "vertex_MF_slice_beta=2000.0_slices=(5,256,256,)_tol=-3_updown.h5"
h5filecorr = "corrMF_slice_beta=2000.0_slices=(1, 256, 256)_tol=-4.h5"
# triptych_vertex_plot([h5file1, h5file2, h5file3]; folder=folder)
# triptych_corr_plot([h5filecorr]; folder=folder)

# plot_vertex_ranks_both(10.0 .^ collect(-5:-2); folder=folder, subdir_str="pch")
# plot_FullCorrelator_ranks_both(10.0 .^ collect(-5:-2); folder=folder, subdir_str="shellpivot")

# tol_vs_rank(10, 10.0 .^ collect(-6:-2); folder=folder, subdir_str="shellpivot")

# plot_K12_ranks_MF(PSFpath)

# plot_kernel_singvals_KF(10; ωmax=0.3183098861837907)