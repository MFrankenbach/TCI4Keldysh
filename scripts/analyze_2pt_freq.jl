using TCI4Keldysh
using JSON
using ITensors
using Plots
using LaTeXStrings

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

function gffilename(gfdata::GFTTdata)
    gffilename(gfdata.Ops, gfdata.R, gfdata.beta, gfdata.tolerance)
end

function gffilename(Ops, R::Int, beta::Float64, tolerance::Float64)
    return "fullTCI2pt_$(reduce(*, Ops))_R=$(R)_beta=$(beta)_tol=$(round(Int, log10(tolerance)))"    
end

# ========== Utilities END

"""
Get all 2pt partial correlators in MPS format.
"""
function get_Gps(; R=12, beta=1.e3, tolerance=1e-8, Ops=["F1", "F1dag"], verbose=false)
    
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    npt=2
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    # TODO: Automate process as to choose R large enough (2^R > length(ωs_center))
    GFs = TCI4Keldysh.load_npoint(PSFpath, Ops, npt, R, ωconvMat; beta=beta, nested_ωdisc=false)
    GF = only(GFs)

    Gps_out = MPS[]
    for perm_idx in 1:factorial(npt)
        if TCI4Keldysh.check_zero(GF.Gps[perm_idx], tolerance)
            printstyled("  Corr. $(reduce(*,Ops)), p=$perm_idx is zero!\n"; color=:red)
            continue
        end
        Gp_mps = TCI4Keldysh.TCI_precompute_reg_values_rotated(GF.Gps[perm_idx]; tolerance=tolerance, cutoff=1e-20)
        push!(Gps_out, Gp_mps)
        if verbose
            printstyled(" ---- Rank p=$perm_idx: $(TCI4Keldysh.rank(Gp_mps))\n"; color=:green)
        end
    end
    return Gps_out
end

function test_full_correlator()

    ITensors.disable_warn_order()

    R = 12
    beta = 1000.0
    tolerance = 1e-8
    spin = 1
    
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    # PSFpath = "data/SIAM_u=1.00/PSF_nz=4_conn_zavg_new/"
    Ops = ["F1", "F1dag"]
    npt=2
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

    Gfull_fat = TCI4Keldysh.MPS_to_fatTensor(Gfull; tags=("ω1",))
    6lot(1:length(Gfull_fat), abs.(Gfull_fat))
    savefig("Gfull.png")

    # reference
    Gfull_ref = TCI4Keldysh.precompute_all_values(GFs[spin])
    plot(1:length(Gfull_ref),log10.(abs.(Gfull_ref)))
    savefig("Gfull_ref.png")
    diffslice = 2:2^R # leave out first bosonic frequency
    diff = abs.(Gfull_fat[diffslice] - Gfull_ref[diffslice])
    plot(diffslice, log10.(diff))
    @show maximum(diff)
    @show norm(diff)./prod(size(diff))
    savefig("Gfull_diff.png")
end

"""
List all operator combinations for 2pt-functions.
"""
function list_all_2pt()
    op_lists = [["Q12","Q34"],["Q13","Q24"],["Q14","Q23"], ["F1","F1dag"], ["F1","Q1dag"], ["F1dag","Q1"]]
    op_lists = vcat(op_lists, reverse.(op_lists))

    return op_lists
end

"""
Compress all 2pt functions via TCI.
"""
function compress_all_2pt(;R::Int=12, beta::Float64=1000.0, tolerance=1.e-6)
    op_lists = list_all_2pt()

    # compute
    gfdatavec = GFTTdata[]
    for (i,Ops) in enumerate(op_lists)
        Gps_out = get_Gps(; R=R, beta=beta, tolerance=tolerance, Ops=Ops)
        # check whether all Gps have been computed (i.e. all are nonzero)
        if length(Gps_out)!=2
            printstyled(" -- Skip corr. $(reduce(*,Ops))\n"; color=:red)
            continue
        end
        Gfull = TCI4Keldysh.FullCorrelator_recompress(Gps_out; tolerance=tolerance)
        printstyled(" ---- Compressed correlator $Ops (no. $i)\n"; color=:blue)

        # store data
        gfdata = GFTTdata(Ops, ITensors.linkdims(Gfull), beta, tolerance, R, linkdims.(Gps_out))
        push!(gfdatavec, gfdata)
        logJSON(gfdata, gffilename(gfdata))
    end
end

function plot_all_FullCorrelators(;R::Int=12, beta::Float64=1000.0, tolerance=1.e-8)
    op_lists = list_all_2pt()
    files = readdir(joinpath(pwd(), "tci_data"))
    Gfull_bonddims = Dict()
    for Ops in op_lists
        fname = gffilename(Ops, R, beta, tolerance)
        @show fname
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
    lfont = 8
    fullplot = plot(;guidefontsize=gfont, titlefontsize=titfont, tickfontsize=tfont, legendfontsize=lfont,
                yscale=:log10)
    msize = 6
    nbonds = length(first(Gfull_bonddims)[2])
    for (name, bdims) in pairs(Gfull_bonddims)
        plot!(fullplot, 1:nbonds, bdims; label=name, marker=:diamond, markersize=msize)
    end
    # worst case
    plot!(fullplot, 1:nbonds, TCI4Keldysh.worstcase_bonddim(fill(2,nbonds+1)); linestyle=:dot, color=:gray, label="worst case")
    # title!(fullplot, "2-Point Correlators, β=$beta, R=$R")
    xticks!(collect(1:nbonds))
    xlabel!("bond idx")
    ylabel!("χ")
    savefig(fullplot, "all_2ptFullCorrelators_tol=$(round(Int,log10(tolerance))).png")
end

"""
χ vs bond index for all partial correlators and the full correlator
"""
function chi_vs_bondidx(;beta::Float64=1.e3, tolerance::Float64=1e-8, R=12)

    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    # PSFpath = "data/SIAM_u=1.00/PSF_nz=4_conn_zavg_new/"
    Ops = ["F1", "F1dag"]

    npt=2
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    GFs = TCI4Keldysh.load_npoint(PSFpath, Ops, npt, R, ωconvMat; beta=beta, nested_ωdisc=false)
    GF = only(GFs)

    Gps_out = Vector{MPS}(undef, factorial(npt))
    for perm_idx in 1:factorial(npt)
        display(GF.Gps[perm_idx].ωconvMat)
        Gp_mps = TCI4Keldysh.TCI_precompute_reg_values_rotated(GF.Gps[perm_idx]; tolerance=tolerance, cutoff=1e-20)
        Gps_out[perm_idx] = Gp_mps
        printstyled(" ---- Rank p=$perm_idx: $(TCI4Keldysh.rank(Gp_mps))\n"; color=:green)
    end

    @time Gfull = TCI4Keldysh.FullCorrelator_recompress(Gps_out; tolerance=tolerance)

    # plot
    nbond = length(Gfull)-1
    chi_full = ITensors.linkdims(Gfull)
    chi_partial = [ITensors.linkdims(Gp) for Gp in Gps_out]

    tfont = 12
    titfont = 16
    gfont = 16
    lfont = 12
    chiplot = plot(;guidefontsize=gfont, titlefontsize=titfont, tickfontsize=tfont, legendfontsize=lfont,
                yscale=:log10)
    msize = 6
    for i in eachindex(chi_partial)
        label = i==1 ? L"G_p(\mathrm{i}\omega)" : false
        plot!(chiplot, 1:nbond, chi_partial[i]; marker=:diamond, color=:black, linestyle=:dash, label=label, markersize=msize)
    end
    plot!(chiplot, 1:nbond, chi_full; marker=:circle, color=:green, label=L"G(\mathrm{i}\omega)", markersize=msize)
    # worst case
    plot!(chiplot, 1:nbond, [TCI4Keldysh.worstcase_bonddim(fill(2,length(Gfull)), i) for i in 1:nbond]; color=:gray, label="worst case")
    xlabel!(chiplot, "bond idx")
    ylabel!(chiplot, "χ")
    # title!(chiplot, "2-Point: Bonddim of Correlators, β=$beta")
    xticks!(chiplot, 1:nbond)
    yticks!(chiplot, [1.0, 10.0, 100.0])
    ylims!(chiplot, 1.0, 100.0)
    plot!(;yscale=:log10)
    savefig(chiplot, "2pt_bonddims_tcikernels_beta=$(beta)_tols=1e$(round(Int, log10(tolerance))).png")
end

function chimax_vs_beta(;tolerance=1e-8, R=12)
    
    # collect data
    betas = 10.0 .^ (-1:4)
    full_chimax = Int[]
    partial_chimax = fill(0, (2, length(betas)))
    for (b,beta) in enumerate(betas)
        Gps_out = get_Gps(;R=R, tolerance=tolerance, beta=beta, verbose=true)
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
    plot!(betaplot, betas, full_chimax; marker=:circle, color=:green, label="G_full", markersize=msize, yscale=yscale, xscale=xscale, legend=:topleft)
    xlabel!(betaplot, "β")
    ylabel!(betaplot, "χmax")
    title!(betaplot, "2-Point: max bonddim vs β, R=$R")
    xticks!(betaplot, betas)
    (_, ymax) = ylims(betaplot)
    ylims!(betaplot, 0, ymax)
    # savefig(betaplot, "2pt_maxbonddims_tols=1e$(round(Int, log10(tolerance))).png")
    savefig(betaplot, "2pt_maxbonddims_tcikernels_tols=1e$(round(Int, log10(tolerance))).png")
end

# for beta in [1.0, 1.e3]
#     chi_vs_bondidx(;beta=beta, tolerance=1.e-8, R=12)
# end
# chimax_vs_beta(;tolerance=1.e-10, R=12)