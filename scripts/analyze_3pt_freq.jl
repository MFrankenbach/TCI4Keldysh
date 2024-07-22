using TCI4Keldysh
using JSON
using ITensors
using Plots
using LaTeXStrings
using Measures

#=
Analyze bond dimension of 3pt functions in frequency domain, computed from 
PSFs in TCI fashion.
=#

# ========== Utilities
struct GFTTdata
    Ops::Vector{String}
    bonddims::Vector{Int}
    beta::Float64
    tolerance::Float64
    R::Int
    # bonddims of partial correlators
    Gp_bonddims::Vector{Vector{Int}}
end

function rank(gfdata::GFTTdata)
    return maximum(gfdata.bonddims)
end

function logJSON(gfdata, filename::String)
    fullname = filename*".json"
    open(joinpath("tci_data", fullname), "w") do file
        JSON.print(file, gfdata)
    end
    printstyled("File $filename.json written!\n", color=:green)
end

function readJSON(filename::String, folder::String="tci_data")
    path = joinpath(folder, filename*".json")
    data = open(path) do file
        JSON.parse(file)
    end 
    return data
end

"""
* singvalshift: whether singular values are shifted to center in TD_to_MPS_via_TTworld
"""
function gffilename(gfdata::GFTTdata; singvalshift::Bool=false)
    gffilename(gfdata.Ops, gfdata.R, gfdata.beta, gfdata.tolerance; singvalshift)
end

function gffilename(Ops, R::Int, beta::Float64, tolerance::Float64; singvalshift::Bool=false)
    svstr = singvalshift ? "_sv_" : "_"
    return "fullTCI3pt"*svstr*"$(reduce(*, Ops))_R=$(R)_beta=$(beta)_tol=$(round(Int, log10(tolerance)))"    
end

# ========== Utilities END

"""
Get all 3pt partial correlators in MPS format.
"""
function get_Gps(; R=7, beta=1.e3, tolerance=1e-8, Ops=["F1", "F1dag", "Q34"], verbose=false)
    
    spin = 1
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    npt=3
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    GFs = TCI4Keldysh.load_npoint(PSFpath, Ops, npt, R, ωconvMat; beta=beta, nested_ωdisc=false)

    Gps_out = MPS[]
    for perm_idx in 1:factorial(npt)
        if TCI4Keldysh.check_zero(GFs[spin].Gps[perm_idx], tolerance)
            printstyled("  Corr. $(reduce(*,Ops)), p=$perm_idx is zero!\n"; color=:red)
            continue
        end
        Gp_mps = TCI4Keldysh.TCI_precompute_reg_values_rotated(GFs[spin].Gps[perm_idx]; tolerance=tolerance, cutoff=1e-20)
        push!(Gps_out, Gp_mps)
        if verbose
            printstyled(" ---- Rank p=$perm_idx: $(TCI4Keldysh.rank(Gp_mps))\n"; color=:green)
        end
    end
    return Gps_out
end

"""
Compute all 3!=6 3pt PartialCorrelators as MPS, add them up and compare with reference.
"""
function test_full_correlator()

    ITensors.disable_warn_order()

    R = 7
    beta = 2000.0
    tolerance = 1e-6
    spin = 1
    
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    # PSFpath = "data/SIAM_u=1.00/PSF_nz=4_conn_zavg_new/"
    Ops = ["F1", "F1dag", "Q34"]
    npt=3
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    GFs = TCI4Keldysh.load_npoint(PSFpath, Ops, npt, R, ωconvMat; beta=beta, nested_ωdisc=false)

    Gps_out = Vector{MPS}(undef, factorial(npt))
    for perm_idx in 1:factorial(npt)
        Gp_mps = TCI4Keldysh.TCI_precompute_reg_values_rotated(GFs[spin].Gps[perm_idx]; tolerance=tolerance, cutoff=1e-20)
        Gps_out[perm_idx] = Gp_mps
        printstyled(" ---- Rank p=$perm_idx: $(TCI4Keldysh.rank(Gp_mps))\n"; color=:green)
    end

    @time Gfull = TCI4Keldysh.FullCorrelator_recompress(Gps_out; tolerance=tolerance)
    # @time Gfull = TCI4Keldysh.FullCorrelator_add(Gps_out; cutoff=1.e-8, use_absolute_cutoff=false)
    @show TCI4Keldysh.rank(Gfull)

    Gfull_fat = TCI4Keldysh.MPS_to_fatTensor(Gfull; tags=("ω1", "ω2"))
    heatmap(log10.(abs.(Gfull_fat)); clim=(-3.0,1.4))
    savefig("Gfull.png")

    # reference
    Gfull_ref = TCI4Keldysh.precompute_all_values(GFs[spin])
    heatmap(log10.(abs.(Gfull_ref)))
    savefig("Gfull_ref.png")
    diffslice = 2:2^R, 1:2^R # leave out first bosonic frequency
    diff = abs.(Gfull_fat[diffslice...] - Gfull_ref[diffslice...]) / maximum(abs.(Gfull_ref))
    heatmap(log10.(diff))
    @show maximum(diff)
    @show norm(diff)./prod(size(diff))
    savefig("Gfull_diff.png")
end

function plot_sample_3pt()
    
    R = 7
    beta = 2000.0
    spin = 1
    
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    # PSFpath = "data/SIAM_u=1.00/PSF_nz=4_conn_zavg_new/"
    Ops = ["F1", "F1dag", "Q34"]
    npt=3
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    GFs = TCI4Keldysh.load_npoint(PSFpath, Ops, npt, R, ωconvMat; beta=beta, nested_ωdisc=false)

    Gfull_ref = TCI4Keldysh.precompute_all_values(GFs[spin])
    # full correlator
    cbar = false
    p1 = blindschleichenplot()
    heatmap!(p1, log10.(abs.(Gfull_ref)); clim=(-3, 1.4), colorbar=cbar)
    plot!(p1; xformatter=:none)
    plot!(p1; yformatter=:none)
    xlabel!(p1, L"\omega_2")
    # ylabel!(p1, L"\omega_1")
    # title!(p1, L"\Sigma_p G_p(\mathrm{i}\omega_1,\mathrm{i}\omega_2)")
    savefig(p1, "Gfull_sample2.png")

    perm_idx = 2
    display(GFs[spin].Gps[perm_idx].ωconvMat)
    # not rotated
    Gp_unrot = TCI4Keldysh.precompute_all_values_MF_without_ωconv(GFs[spin].Gps[perm_idx])
    p2 = blindschleichenplot()
    plot!(p2; xformatter=:none)
    plot!(p2; yformatter=:none)
    heatmap!(p2, log10.(abs.(Gp_unrot)); colorbar=cbar)
    xlabel!(p2, L"\omega_2")
    ylabel!(p2, L"\omega_1")
    # title!(p2, L"G_p(\mathrm{i}\omega_1,\mathrm{i}\omega_2 - \mathrm{i}\omega_1)")
    savefig(p2, "Gp_sample_norot2.png")

    # rotated
    Gp_rot = TCI4Keldysh.precompute_all_values_MF(GFs[spin].Gps[perm_idx])
    p3 = blindschleichenplot()
    heatmap!(p3, log10.(abs.(Gp_rot)); colorbar=cbar)
    plot!(p3; xformatter=:none)
    plot!(p3; yformatter=:none)
    xlabel!(p3, L"\omega_2", margin=5mm)
    # ylabel!(p3, L"\omega_1")
    # title!(p3, L"G_p(\mathrm{i}\omega_1,\mathrm{i}\omega_2)")
    savefig(p3, "Gp_sample2.png")

end

"""
χ vs bond index for all partial correlators and the full correlator
"""
function chi_vs_bondidx(;beta::Float64=1.e3, tolerance::Float64=1e-6, R=9)
    
    spin = 1
    
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    # PSFpath = "data/SIAM_u=1.00/PSF_nz=4_conn_zavg_new/"
    Ops = ["F1", "F1dag", "Q34"]
    npt=3
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    GFs = TCI4Keldysh.load_npoint(PSFpath, Ops, npt, R, ωconvMat; beta=beta, nested_ωdisc=false)

    Gps_out = Vector{MPS}(undef, factorial(npt))
    for perm_idx in 1:factorial(npt)
        Gp_mps = TCI4Keldysh.TCI_precompute_reg_values_rotated(GFs[spin].Gps[perm_idx]; tolerance=tolerance, cutoff=1e-20)
        Gps_out[perm_idx] = Gp_mps
        printstyled(" ---- Rank p=$perm_idx: $(TCI4Keldysh.rank(Gp_mps))\n"; color=:green)
    end

    @time Gfull = TCI4Keldysh.FullCorrelator_recompress(Gps_out; tolerance=tolerance)
    # @time Gfull = TCI4Keldysh.FullCorrelator_add(Gps_out; cutoff=1.e-25, use_absolute_cutoff=false)
    # truncate!(Gfull; cutoff=1.e-25, use_absolute_cutoff=false)
    @show TCI4Keldysh.rank(Gfull)
    gfdata = GFTTdata(Ops, ITensors.linkdims(Gfull), beta, tolerance, R, linkdims.(Gps_out))
    logJSON(gfdata, gffilename(gfdata; singvalshift=true))

    # plot
    nbond = length(Gfull)-1
    chi_full = ITensors.linkdims(Gfull)
    chi_partial = [ITensors.linkdims(Gp) for Gp in Gps_out]

    tfont = 12
    titfont = 16
    gfont = 16
    lfont = 12
    chiplot = plot(;guidefontsize=gfont, titlefontsize=titfont, tickfontsize=tfont, legendfontsize=lfont)
    msize = 6
    yscale=:log10
    for i in eachindex(chi_partial)
        label = i==1 ? L"G_p(\mathrm{i}\omega)" : false
        plot!(chiplot, 1:nbond, chi_partial[i]; marker=:diamond, color=:black, linestyle=:dash, label=label, markersize=msize, yscale=yscale)
    end
    plot!(chiplot, 1:nbond, chi_full; marker=:circle, color=:green, label=L"G(\mathrm{i}\omega)", markersize=msize, yscale=yscale)
    # worst case
    plot!(chiplot, 1:nbond, [TCI4Keldysh.worstcase_bonddim(fill(2,length(Gfull)), i) for i in 1:nbond]; color=:gray, label="worst case", yscale=yscale)
    xlabel!(chiplot, "bond idx")
    ylabel!(chiplot, "χ")
    # title!(chiplot, "3-Point: Bonddim of Correlators, β=$beta")
    xticks!(chiplot, 1:nbond)
    savefig(chiplot, "3pt_bonddims_tcikernels_beta=$(beta)_tols=1e$(round(Int, log10(tolerance))).png")
end

function chimax_vs_beta(;tolerance=1e-6, R=7)
    
    # collect data
    betas = 10.0 .^ (-1.0:4.0)
    full_chimax = Int[]
    partial_chimax = fill(0, (factorial(3), length(betas)))
    for (b,beta) in enumerate(betas)
        Gps_out = get_Gps(;R=R, tolerance=tolerance, beta=beta, verbose=false)
        Gfull = TCI4Keldysh.FullCorrelator_recompress(Gps_out; tolerance=tolerance)
        push!(full_chimax, maximum(ITensors.linkdims(Gfull)))
        for i in axes(partial_chimax, 1)
            partial_chimax[i,b] = maximum(ITensors.linkdims(Gps_out[i]))
        end
    end

    # plot data
    betaplot = plot()
    msize = 6
    yscale=:identity
    xscale=:log10
    for i in axes(partial_chimax, 1)
        label = i==1 ? "G_partial" : false
        plot!(betaplot, betas, partial_chimax[i,:]; marker=:diamond, color=:black, linestyle=:dash, label=label, markersize=msize, yscale=yscale, xscale=xscale)
    end
    plot!(betaplot, betas, full_chimax; marker=:circle, color=:green, label="G_full", markersize=msize, yscale=yscale, xscale=xscale)
    xlabel!(betaplot, "β")
    ylabel!(betaplot, "χmax")
    title!(betaplot, "3-Point: max bonddim vs β, R=$R")
    xticks!(betaplot, betas)
    # savefig(betaplot, "3pt_maxbonddims_tols=1e$(round(Int, log10(tolerance))).png")
    savefig(betaplot, "3pt_maxbonddims_tcikernels_tols=1e$(round(Int, log10(tolerance))).png")
end

function chimax_vs_R(;tolerance::Float64=1e-6, beta::Float64=1000.0)
    
    # collect data
    Rs = 7:11
    full_chimax = Int[]
    partial_chimax = fill(0, (factorial(3), length(Rs)))
    worstcase = Int[]
    for (r,R) in enumerate(Rs)
        Gps_out = get_Gps(;R=R, tolerance=tolerance, beta=beta, verbose=false)
        Gfull = TCI4Keldysh.FullCorrelator_recompress(Gps_out; tolerance=tolerance)
        push!(full_chimax, maximum(ITensors.linkdims(Gfull)))
        push!(worstcase, 2^(div(length(Gfull),2)))
        for i in axes(partial_chimax, 1)
            partial_chimax[i,r] = maximum(ITensors.linkdims(Gps_out[i]))
        end
        printstyled(" ---- R=$R done in chimax_vs_R\n"; color=:blue)
    end

    # plot data
    Rplot = plot()
    msize = 6
    yscale=:identity
    xscale=:identity
    for i in axes(partial_chimax, 1)
        label = i==1 ? "G_partial" : false
        plot!(Rplot, Rs, partial_chimax[i,:]; marker=:diamond, color=:black, linestyle=:dash, label=label, markersize=msize, yscale=yscale, xscale=xscale)
    end
    plot!(Rplot, Rs, full_chimax; marker=:circle, color=:green, label="G_full", markersize=msize, yscale=yscale, xscale=xscale)
    plot!(Rplot, Rs, worstcase; color=:gray, label="worst case", yscale=yscale, xscale=xscale)
    xlabel!(Rplot, "R")
    ylabel!(Rplot, "χmax")
    title!(Rplot, "3-Point: max bonddim vs R, β=$beta")
    xticks!(Rplot, Rs)
    savefig(Rplot, "3pt_maxbonddims_vs_R_tols=1e$(round(Int, log10(tolerance))).png")
end

"""
List all operator combinations for 3pt-functions.
"""
function list_all_3pt()
    op_lists = []
    f_dict = Dict(1=>"F1",2=>"F1dag",3=>"F3",4=>"F3dag")
    for q in ["12","13","14","23","24","34"] 
        q1, q2 = divrem(parse(Int, q), 10)
        f1i, f2i = sort(setdiff([1,2,3,4], [q1,q2]))[1:2]
        # strings for operator names
        f1 = f_dict[f1i]
        f2 = f_dict[f2i]
        # add all permutations
        op_vec = ["Q"*q, f1, f2]
        for perm in [[1,2,3],[2,1,3],[1,3,2],[3,2,1],[3,1,2],[2,3,1]]
            push!(op_lists, op_vec[perm])
        end
    end
    @assert length(unique(op_lists)) == length(op_lists)

    return op_lists
end

"""
List all 3-point operator combinations by matching a pattern in the file name.
Only lists one permutation for each set of operators.
"""
function list_all_3pt_parse(dir="data/SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    matfiles = filter(x -> endswith(x, ".mat"), readdir(joinpath(pwd(),dir)))
    oplists = []
    opsets = []
    for f in matfiles
        opstr = f[findlast('(', f)+1 : findfirst(')', f)-1]
        oplist = String.(split(opstr, ','))
        opset = Set(oplist)
        if length(opset)==3
            if !(opset in opsets)
                push!(oplists, oplist)
                push!(opsets, opset)
            end
        end
    end
    return oplists
end

"""
Compress all 3pt functions via TCI.
"""
function compress_all_3pt(;R::Int=9, beta::Float64=1000.0, tolerance=1.e-6)
    op_lists = list_all_3pt_parse()

    # compute
    gfdatavec = GFTTdata[]
    for (i,Ops) in enumerate(op_lists)
        Gps_out = get_Gps(; R=R, beta=beta, tolerance=tolerance, Ops=Ops)
        if length(Gps_out)!=factorial(3)
            printstyled(" -- Skip corr. $(reduce(*,Ops))\n"; color=:red)
            continue
        end
        Gfull = TCI4Keldysh.FullCorrelator_recompress(Gps_out; tolerance=tolerance)
        printstyled(" ---- Compressed correlator $Ops (no. $i/$(length(op_lists)))\n"; color=:blue)

        # store data
        gfdata = GFTTdata(Ops, ITensors.linkdims(Gfull), beta, tolerance, R, linkdims.(Gps_out))
        push!(gfdatavec, gfdata)
        logJSON(gfdata, gffilename(gfdata))
    end
end

function plot_all_FullCorrelators(;R::Int=9, beta::Float64=1000.0, tolerance=1.e-6)
    op_lists = list_all_3pt_parse()
    files = readdir(joinpath(pwd(), "tci_data"))
    Gfull_bonddims = Dict()
    for Ops in op_lists
        fname = gffilename(Ops, R, beta, tolerance)
        if !(fname*".json" in files)
            continue
        end
        gfdata = readJSON(fname)
        Gfull_bonddims[reduce(*,gfdata["Ops"])] = gfdata["bonddims"]
    end

    # plot
    tfont = 12
    titfont = 16
    gfont = 16
    lfont = 6
    fullplot = plot(;guidefontsize=gfont, titlefontsize=titfont, tickfontsize=tfont, legendfontsize=lfont)
    msize = 6
    nbonds = length(first(Gfull_bonddims)[2])
    yscale = :log10
    for (name, bdims) in pairs(Gfull_bonddims)
        plot!(fullplot, 1:nbonds, bdims; label=name, marker=:diamond, markersize=msize, yscale=yscale)
    end
    # worst case
    plot!(fullplot, 1:nbonds, TCI4Keldysh.worstcase_bonddim(fill(2,nbonds+1)); linestyle=:dot, color=:gray, label="worst case", yscale=yscale)
    # title!(fullplot, "3-Point Correlators, β=$beta, R=$R")
    xticks!(collect(1:nbonds))
    xlabel!("bond idx")
    ylabel!("χ")
    savefig(fullplot, "all_3ptFullCorrelators_tol=$(round(Int,log10(tolerance))).png")
end

"""
Histogram of ranks of full correlators.
"""
function histogram_all_3pt()
    op_lists = list_all_3pt()
    beta = 2000.0
    tolerance = 1.e-6
    R = 9
    ranks = Int[]
    for Ops in op_lists
        gfdata = readJSON(gffilename(Ops, R, beta, tolerance))
        push!(ranks, maximum(gfdata["bonddims"]))
    end

    histogram(ranks; label="full 3-point corr.")
    title!("Rank distribution of 3pt-Correlators, β=$beta")
    savefig("histogram_all_3pt_R=$(R)_beta=$(beta)_tol=$(round(Int, log10(tolerance))).png")
end

function default_plot()
    tfont = 12
    titfont = 16
    gfont = 16
    lfont = 12
    p = plot(;guidefontsize=gfont, titlefontsize=titfont, tickfontsize=tfont, legendfontsize=lfont)
    return p
end

function blindschleichenplot()
    tfont = 18
    titfont = 30
    gfont = 30
    lfont = 18
    p = plot(;guidefontsize=gfont, titlefontsize=titfont, tickfontsize=tfont, legendfontsize=lfont)
    return p
end


# for beta in [1.0, 1.e3]
#     chi_vs_bondidx(;beta=beta, tolerance=1.e-6, R=9)
# end
