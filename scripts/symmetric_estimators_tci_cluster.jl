using TCI4Keldysh
using Profile
using StatProfilerHTML
using QuanticsTCI
using Serialization
import TensorCrossInterpolation as TCI

TCI4Keldysh.TIME() = false

function report_mem(do_gc=false)
    println("---------- MEMORY REPORT ----------")
    if do_gc
        println("  Available system memory (before gc()): $(Sys.free_memory() / 1024^2) MB")
        Base.GC.gc()
        println("  Garbage collected")
    end
    println("  Total system memory: $(Sys.total_memory() / 1024^2) MB")
    println("  Available system memory: $(Sys.free_memory() / 1024^2) MB")
    println("-----------------------------------")
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

function time_Γcore()
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    R = 5
    tolerance = 1.e-6
    ωconvMat = TCI4Keldysh.channel_trafo("t")
    beta = 2000.0
    T = 1.0/beta

    # compile
    println("  Compile run...")
    foo = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=1.e-3, unfoldingscheme=:interleaved
        )
    x = sum(foo)
    println("  Time...")
    t = @elapsed begin
        Γcore = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved,
        cache_center=2^(R-2)
        )
    end 
    x = sum(Γcore)
    println(" TIME: $t")
end

function Γcore_filename(mode::String, xmin, xmax, tolerance::Float64, beta::Float64; folder="pwtcidata")
    return joinpath(TCI4Keldysh.pdatadir(),"$folder/gammacore_timing_$(mode)_min=$(xmin)_max=$(xmax)_tol=$(TCI4Keldysh.tolstr(tolerance))_beta=$beta")
end

function serialize_tt(qtt, outname::String, folder::String)
    R = qtt.grid.R
    fname_tt = joinpath(folder, outname*"_R=$(R)_qtt.serialized")
    serialize(fname_tt, qtt)
end

function serialize_tt(tci, grid, outname::String, folder::String)
    R = grid.R
    fname_tt = joinpath(folder, outname*"_R=$(R)_qtt.serialized")
    serialize(fname_tt, (tci, grid))
end

function time_Γcore_sweep(
    param_range, PSFpath, mode="R";
    tolerance=1.e-8,
    cache_center=0,
    folder="pwtcidata",
    serialize_tts=true,
    flavor_idx=1,
    # batched_eval=false,
    batched_eval=true,
    use_ΣaIE=true,
    tcikwargs...
    )
    channel = "t"
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    T = TCI4Keldysh.dir_to_T(PSFpath)
    beta = 1.0/T
    times = []
    qttranks = []
    qttbonddims = []
    svd_kernel = true
    @show svd_kernel
    @show batched_eval
    println("Additional TCI kwargs:")
    @show Dict(tcikwargs...)
    if mode=="R"
        Rs = param_range
        # prepare output
        d = Dict()
        d["times"] = times
        d["ranks"] = qttranks
        d["bonddims"] = qttbonddims
        d["Rs"] = Rs
        d["beta"] = beta
        d["flavor_idx"] = flavor_idx
        d["PSFpath"] = PSFpath
        d["channel"] = channel
        d["tolerance"] = tolerance
        d["svd_kernel"] = svd_kernel
        d["numthreads"] = Threads.threadpoolsize()
        d["job_id"] = ENV["SLURM_JOB_ID"]
        d["cache_center"] = cache_center
        d["use_Sigma_aIE"] = use_ΣaIE
        d["tcikwargs"] = Dict(tcikwargs)
        outname = Γcore_filename(mode, first(Rs), last(Rs), tolerance, beta; folder=folder)
        TCI4Keldysh.logJSON(d, outname, folder)

        for R in Rs
            t = @elapsed begin
                qtt = if batched_eval
                    TCI4Keldysh.Γ_core_TCI_MF_batched(
                    PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=flavor_idx, use_ΣaIE=use_ΣaIE, tolerance=tolerance, verbosity=2,
                    cache_center=cache_center, tcikwargs...
                    )
                else
                    TCI4Keldysh.Γ_core_TCI_MF(
                    PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=flavor_idx, tolerance=tolerance, unfoldingscheme=:interleaved, verbosity=2,
                    cache_center=cache_center, tcikwargs...
                    )
                end
            end 
            push!(times, t)
            push!(qttranks, TCI4Keldysh.rank(qtt))
            push!(qttbonddims, TCI.linkdims(qtt.tci))
            TCI4Keldysh.updateJSON(outname, "times", times, folder)
            TCI4Keldysh.updateJSON(outname, "ranks", qttranks, folder)
            TCI4Keldysh.updateJSON(outname, "bonddims", qttbonddims, folder)

            if serialize_tts
                # can't just serialize qtt because may contain anonymous functoins
                serialize_tt(qtt.tci, qtt.grid, outname, folder)
            end

            println(" ===== R=$R: time=$t, rankk(qtt)=$(TCI4Keldysh.rank(qtt))")
            report_mem(true)
            flush(stdout)
            flush(stderr)
        end
    else
        error("Invalid mode $mode")
    end
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

function parse_run_nr(run_nr::Int)
    # process last 4 digits
    run_nr = mod(run_nr, 10^4)
    psf_path_id = div(run_nr, 1000)
    PSFpath = if psf_path_id==1
        nz = 4
        joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=$(nz)_conn_zavg/")
    elseif psf_path_id==2
        joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg")
    else
        error("Invalid psf_path_id $(psf_path_id)")
    end
    tol_int = div(run_nr - 1000*psf_path_id, 100)
    tolerance = 10.0 ^ (-tol_int)
    R = rem(run_nr, 100) 
    println("Input parameters: R=$R, tol=$tolerance, PSFpath=$PSFpath")
    flush(stdout)
    return (PSFpath, R, tolerance)
end

function main(args)

    if length(args)<1
        println(ARGS)
        println(args)
        println("Need command line argument")
        exit(1)
    end

    #nthreads = Threads.threadpoolsize()

    job_id = ENV["SLURM_JOB_ID"]
    println("SLURM_JOB_ID: $job_id")


    println(" ==== COMPILE")
    nz = 4
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=$(nz)_conn_zavg/")
    R = 4
    ωconvMat = TCI4Keldysh.channel_trafo("t")
    Γcore = TCI4Keldysh.Γ_core_TCI_MF(
          PSFpath, 4; T=TCI4Keldysh.dir_to_T(PSFpath), ωconvMat=ωconvMat, flavor_idx=1, tolerance=1.e-3, unfoldingscheme=:interleaved, use_ΣaIE=false
        )

    flush(stdout)
    flush(stderr)
    report_mem(true)
    println(" ==== RUN")
    # first argument: different settings
    run_nr = parse(Int, args[1])
    # second argument: store everything locally
    folder = if length(args)>1 && args[2]=="local"
                ENV["PWTCIDIR"]
            else
                "pwtcidata"
            end
    # third argument: flavor_idx
    flavor_idx = if length(args)>2
                parse(Int, args[3])
            else
                # default
                1
            end
    @assert (flavor_idx in [1,2]) "Invalid flavor index"
    # fourth argument: range of Rs, default 5:12
    Rs = if length(args)>3
                Rs_code = parse(Int, args[4])
                dd = digits(Rs_code)
                if length(dd)<4
                    # assume 0 at the beginning
                    push!(dd,0)
                end
                @assert length(dd)==4 "Invalid specification of R range!"
                Rmin = dd[end]*10 + dd[end-1]
                Rmax = dd[2]*10 + dd[1]

                Rmin:Rmax
            else
                5:12
            end

    if run_nr==0
        println("TEST")
        time_Γcore_sweep(10:10, PSFpath, "R"; folder=folder, tolerance=1.e-2, flavor_idx=flavor_idx)
    elseif run_nr==1
        time_Γcore_sweep(Rs, PSFpath, "R"; folder=folder, tolerance=1.e-2, flavor_idx=flavor_idx)
    elseif run_nr==2
        time_Γcore_sweep(Rs, PSFpath, "R"; folder=folder, tolerance=1.e-4, flavor_idx=flavor_idx)
    elseif run_nr==3
        time_Γcore_sweep(Rs, PSFpath, "R"; folder=folder, tolerance=1.e-6, flavor_idx=flavor_idx)
    elseif run_nr==4
        time_Γcore_sweep(Rs, PSFpath, "R"; folder=folder, tolerance=1.e-8, flavor_idx=flavor_idx)
    elseif run_nr==5
        time_Γcore_sweep(Rs, PSFpath, "R"; folder=folder, tolerance=1.e-3, flavor_idx=flavor_idx)
    elseif run_nr==6
        time_Γcore_sweep(Rs, PSFpath, "R"; folder=folder, tolerance=1.e-5, flavor_idx=flavor_idx)
    elseif run_nr==7
        time_Γcore_sweep(Rs, PSFpath, "R"; folder=folder, tolerance=1.e-7, flavor_idx=flavor_idx)
    # beta=200.0 
    elseif run_nr==11
        PSFpath = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg")
        time_Γcore_sweep(Rs, PSFpath, "R"; folder=folder, tolerance=1.e-2, flavor_idx=flavor_idx)
    elseif run_nr==12
        PSFpath = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg")
        time_Γcore_sweep(Rs, PSFpath, "R"; folder=folder, tolerance=1.e-4, flavor_idx=flavor_idx)
    elseif run_nr==13
        PSFpath = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg")
        time_Γcore_sweep(Rs, PSFpath, "R"; folder=folder, tolerance=1.e-6, flavor_idx=flavor_idx)
    elseif run_nr==14
        PSFpath = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg")
        time_Γcore_sweep(Rs, PSFpath, "R"; folder=folder, tolerance=1.e-8, flavor_idx=flavor_idx)
    elseif run_nr==15
        PSFpath = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg")
        time_Γcore_sweep(Rs, PSFpath, "R"; folder=folder, tolerance=1.e-3, flavor_idx=flavor_idx)
    elseif run_nr==16
        PSFpath = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg")
        time_Γcore_sweep(Rs, PSFpath, "R"; folder=folder, tolerance=1.e-5, flavor_idx=flavor_idx)
    elseif run_nr==17
        PSFpath = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg")
        time_Γcore_sweep(Rs, PSFpath, "R"; folder=folder, tolerance=1.e-7, flavor_idx=flavor_idx)
    elseif run_nr>=10^4
        # for more global pivots
        nsearchglobalpivot = div(run_nr, 10^4)  
        (PSFpath, R, tolerance) = parse_run_nr(run_nr)
        time_Γcore_sweep(R:R, PSFpath, "R"; flavor_idx=flavor_idx, folder=folder, tolerance=tolerance, serialize_tts=true, batched_eval=true, nsearchglobalpivot=nsearchglobalpivot, maxnglobalpivot=nsearchglobalpivot)
    # single R/tol jobs: first digit: PSFpath, second digit: -log10(tolerance), last 2 digits: R
    elseif run_nr>=1000
        (PSFpath, R, tolerance) = parse_run_nr(run_nr)
        time_Γcore_sweep(R:R, PSFpath, "R"; flavor_idx=flavor_idx, folder=folder, tolerance=tolerance, serialize_tts=true, batched_eval=true)
    else
        error("invalid run number $run_nr")
    end

    println(" ==== DONE")
    flush(stdout)
end

main(ARGS)