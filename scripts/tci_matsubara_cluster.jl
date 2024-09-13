using TCI4Keldysh
using JSON
using ITensors

TCI4Keldysh.VERBOSE() = false
TCI4Keldysh.DEBUG() = false
TCI4Keldysh.TIME() = true

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

function test_full_correlator(;R=7, beta=1.e1, tolerance=1.e-8, cutoff=1.e-20)

    ITensors.disable_warn_order()

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
end

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

struct TCICorrelatorSettings
    R::Int
    beta::Float64
    tolerance::Float64
    cutoff::Float64
end

function print(t::TCICorrelatorSettings)
    println("R=$(t.R), beta=$(t.beta), tolerance=$(t.tolerance), cutoff=$(t.cutoff)")
end

function main(args)

    if length(args)<1
        println(ARGS)
        println(args)
        println("Need command line argument")
        exit(1)
    end

    println("\n-----COMPILE")
    t1 = TCICorrelatorSettings(7, 1.0, 1.e-3, 1.e-3)
    print(t1)
    test_full_correlator(R=t1.R, beta=t1.beta, tolerance=t1.tolerance, cutoff=t1.cutoff)

    println("Received cl arguments $(args)")
    run_nr = parse(Int, args[1])
    println("Run settings no. $run_nr")

    println("\n-----RUN")
    if run_nr==1
        t2 = TCICorrelatorSettings(7, 100.0, 1.e-6, 1.e-20)
        print(t2)
        test_full_correlator(R=t2.R, beta=t2.beta, tolerance=t2.tolerance, cutoff=t2.cutoff)
    elseif run_nr==2
        t3 = TCICorrelatorSettings(7, 1000.0, 1.e-6, 1.e-20)
        print(t3)
        test_full_correlator(R=t3.R, beta=t3.beta, tolerance=t3.tolerance, cutoff=t3.cutoff)
    elseif run_nr==3
        t4 = TCICorrelatorSettings(9, 10.0, 1.e-8, 1.e-20)
        print(t4)
        test_full_correlator(R=t4.R, beta=t4.beta, tolerance=t4.tolerance, cutoff=t4.cutoff)
    elseif run_nr==4
        t4 = TCICorrelatorSettings(9, 100.0, 1.e-6, 1.e-20)
        print(t4)
        test_full_correlator(R=t4.R, beta=t4.beta, tolerance=t4.tolerance, cutoff=t4.cutoff)
    else
        error("Invalid run number $run_nr")
    end

    println("\n-----DONE")
end

main(ARGS)
