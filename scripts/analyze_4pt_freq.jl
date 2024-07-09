using TCI4Keldysh
using JSON
using ITensors
using Plots
using LaTeXStrings

# ========== Utilities
"""
Represents PartialCorrelator in MPS form
"""
struct GpTTdata
    Ops::Vector{String}
    perm_idx::Int
    bonddims::Vector{Int}
    beta::Float64
    tolerance::Float64
    cutoff::Float64
    R::Int
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

function gffilename(gpdata::GpTTdata)
    gffilename(gpdata.Ops, gpdata.perm_idx, gpdata.R, gpdata.beta, gpdata.tolerance, gpdata.cutoff)
end

function gffilename(Ops, p::Int, R::Int, beta::Float64, tolerance::Float64, cutoff::Float64)
    return "p=$(p)TCI4pt_$(reduce(*, Ops))_R=$(R)_beta=$(beta)_tol=$(round(Int, log10(tolerance)))_cut=$(round(Int, log10(cutoff)))"    
end

    # ========== FullCorrelator Utilities
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


function gffilename(gfdata::GFTTdata)
    gffilename(gfdata.Ops, gfdata.R, gfdata.beta, gfdata.tolerance)
end

function gffilename(Ops, R::Int, beta::Float64, tolerance::Float64)
    return "fullTCI4pt_$(reduce(*, Ops))_R=$(R)_beta=$(beta)_tol=$(round(Int, log10(tolerance)))"    
end
    # ========== FullCorrelator Utilities END

# ========== Utilities END

"""
Get all 4pt partial correlators in MPS format.
Store bonddims etc. on the fly.
"""
function get_Gps(; R=7, beta=1.e3, tolerance=1e-5, cutoff=1.e-5, Ops=["F1", "F1dag", "F3", "F3dag"], verbose=false)
    
    spin = 1
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    npt=4
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    GFs = TCI4Keldysh.load_npoint(PSFpath, Ops, npt, R, ωconvMat; beta=beta, nested_ωdisc=false)

    Gps_out = MPS[]
    for perm_idx in 1:factorial(npt)
        if TCI4Keldysh.check_zero(GFs[spin].Gps[perm_idx], tolerance)
            printstyled("  Corr. $(reduce(*,Ops)), p=$perm_idx is zero!\n"; color=:red)
            continue
        end
        Gp_mps = TCI4Keldysh.TCI_precompute_reg_values_rotated(GFs[spin].Gps[perm_idx]; tolerance=tolerance, cutoff=cutoff)
        Gpdata = GpTTdata(Ops, perm_idx, ITensors.linkdims(Gp_mps), beta, tolerance, cutoff, R)
        logJSON(Gpdata, gffilename(Gpdata))
        push!(Gps_out, Gp_mps)
        if verbose
            printstyled(" ---- Rank p=$perm_idx: $(TCI4Keldysh.rank(Gp_mps))\n"; color=:green)
        end
    end
    return Gps_out
end

"""
χ vs bond index for all partial correlators and the full correlator
"""
function chi_vs_bondidx(;beta::Float64=1.e1, tolerance::Float64=1e-6, cutoff::Float64=1.e-20, R=7, Ops=["F1", "F1dag", "F3", "F3dag"])

    # load partial correlators
    npt=4
    files = readdir(joinpath(pwd(), "tci_data"))
    Gpdata = Dict()
    for p in 1:factorial(npt)
        fname = gffilename(Ops, p, R, beta, tolerance, cutoff)
        if !(fname*".json" in files)
            continue
        end
        gpdata = readJSON(fname)
        Gpdata[p]=gpdata
    end

    # load full correlator
    GFdata = readJSON(gffilename(Ops, R, beta, tolerance))

    # plot
    nbond = length(GFdata["bonddims"])
    chi_full = GFdata["bonddims"]
    chi_partial = [Gpdata[p]["bonddims"] for p in eachindex(Gpdata)]

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
    plot!(chiplot, 1:nbond, [TCI4Keldysh.worstcase_bonddim(fill(2,nbond+1), i) for i in 1:nbond]; color=:gray, label="worst case", yscale=yscale)
    xlabel!(chiplot, "bond idx")
    ylabel!(chiplot, "χ")
    # title!(chiplot, "4-Point: Bonddim of Correlators, β=$beta")
    xticks!(chiplot, 1:nbond)
    savefig(chiplot, "4pt_bonddims_tcikernels_beta=$(beta)_tols=1e$(round(Int, log10(tolerance)))_cut=1e$(round(Int, log10(cutoff))).png")
end

"""
Compute all 4!=24 4pt PartialCorrelators as MPS, add them up and compare with reference.
"""
function test_full_correlator()

    ITensors.disable_warn_order()

    R = 7
    beta = 10.0
    tolerance = 1e-6
    cutoff = 1.e-20
    spin = 1
    
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    Ops = ["F1", "F1dag", "F3", "F3dag"]
    npt=4
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    GFs = TCI4Keldysh.load_npoint(PSFpath, Ops, npt, R, ωconvMat; beta=beta, nested_ωdisc=false)

    Gps_out = get_Gps(;R=R, beta=beta, tolerance=tolerance, cutoff=cutoff, Ops=Ops, verbose=true)
    @time Gfull = TCI4Keldysh.FullCorrelator_add(Gps_out; cutoff=1.e-20, use_absolute_cutoff=false)
    truncate!(Gfull; cutoff=cutoff, use_absolute_cutoff=true)
    @show TCI4Keldysh.rank(Gfull)


    # reference
    Gfull_ref = TCI4Keldysh.precompute_all_values_MF_noano(GFs[spin])
    gfdata = GFTTdata(Ops, linkdims(Gfull), beta, tolerance, R, [linkdims(Gp) for Gp in Gps_out])
    logJSON(gfdata, gffilename(gfdata))
    @show size(Gfull_ref)

    # plot
    Gfull_fat = TCI4Keldysh.MPS_to_fatTensor(Gfull; tags=("ω1", "ω2", "ω3"))
    slice_idx = 2^(R-1)
    refmax = maximum(abs.(Gfull_ref))
    heatmap(log10.(abs.(Gfull_fat[:,slice_idx,:])); clim=(log10(refmax)-10, log10(refmax))) 
    savefig("Gfull.png")
    heatmap(log10.(abs.(Gfull_ref[:,slice_idx,:])); clim=(log10(refmax)-10, log10(refmax))) 
    savefig("Gfull_ref.png")
    diffslice = (2:2^R, 1:2^R, 1:2^R) # leave out first bosonic frequency
    diff = abs.(Gfull_fat[diffslice...] - Gfull_ref[diffslice...]) ./ maximum(abs.(Gfull_ref[diffslice...]))
    heatmap(log10.(diff[:,slice_idx,:]))
    @show maximum(diff)
    @show norm(diff)./prod(size(diff))
    savefig("Gfull_diff.png")

end