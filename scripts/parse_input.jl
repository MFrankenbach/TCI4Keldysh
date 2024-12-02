using TCI4Keldysh
using Serialization
using QuanticsTCI
using QuanticsGrids
import TensorCrossInterpolation as TCI

"""
Determine whether a value to an input key should remain uppercase
"""
function is_uppercase_key(kw::AbstractString)
    list = ["psfpath"]
    return (lowercase(kw) in list)
end

# UTILITIES
function maybeparse(T::Type, val)
    return isa(val,T) ? val : parse(T, val)
end

function parse_input(input_file::AbstractString)
    d = Dict{String, Any}()
    open(input_file) do f    
        lines = readlines(f)
        if !occursin("TCI4Keldysh BEGIN", lines[1])
            @warn "Wrong input format: BEGIN missing"
        end
        if !occursin("TCI4Keldysh END", lines[end])
            @warn "Wrong input format: END missing"
        end
        for line in lines[2:end-1]
            if isempty(strip(line))
                continue
            end
            pts = convert.(String, split(strip(line)))
            if !is_uppercase_key(pts[1])
                pts = lowercase.(pts)
            else
                pts[1] = lowercase(pts[1])
            end
            kw = pts[1]
            d[kw] = if length(pts)==2
                    only(pts[2:end])
                else
                    pts[2:end]
                end
        end
    end
    return d
end

function get_default(key::String)
    default_values = Dict(
        "channel" => "p",
        "flavor_idx" => 1,
        "rrange" => "0512",
        "local" => "true"
    )
    if haskey(default_values, key) 
        return  default_values[key]
    else
        error("No default value for key $key")
    end
end

function read_arg(inp_args, key::String)
    if haskey(inp_args, key)    
        return inp_args[key]
    else
        return get_default(key)
    end
end

function PSFpath_id_to_path(id::Int)    
    if id==1
        nz = 4
        joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=$(nz)_conn_zavg/")
    elseif id==2
        joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg")
    else
        error("Invalid PSFpath id $id")
    end
end
# UTILITIES END

function json_filename(jobtype::String, xmin, xmax, tolerance::Float64, beta::Float64; folder="pwtcidata")
    return joinpath(TCI4Keldysh.pdatadir(), folder ,"$(jobtype)_R_min=$(xmin)_max=$(xmax)_tol=$(TCI4Keldysh.tolstr(tolerance))_beta=$beta")
end

function run_job(jobtype::String; Rs::AbstractRange{Int}, tolerance, PSFpath, folder, flavor_idx, channel, kwargs...)
    beta = TCI4Keldysh.dir_to_beta(PSFpath)
    outname = json_filename(jobtype, first(Rs), last(Rs), tolerance, beta; folder=folder)

    # prepare output with general info
    d = Dict()
    d["Rs"] = Rs
    d["beta"] = beta
    d["flavor_idx"] = flavor_idx
    d["PSFpath"] = PSFpath
    d["channel"] = channel
    d["tolerance"] = tolerance
    d["numthreads"] = Threads.threadpoolsize()
    d["job_id"] = if haskey(ENV, "SLURM_JOB_ID")
                    ENV["SLURM_JOB_ID"]
                else
                    000000
                end

    if jobtype=="matsubarafull"
        matsubarafull(outname, d; Rs=Rs, tolerance=tolerance, PSFpath=PSFpath, folder=folder, flavor_idx=flavor_idx, channel=channel, kwargs...)
    elseif jobtype=="matsubaracore"
        matsubaracore(outname, d; Rs=Rs, tolerance=tolerance, PSFpath=PSFpath, folder=folder, flavor_idx=flavor_idx, channel=channel, kwargs...)
    elseif jobtype=="keldyshcore"
        keldyshcore(outname, d; Rs=Rs, tolerance=tolerance, PSFpath=PSFpath, folder=folder, flavor_idx=flavor_idx, channel=channel, kwargs...)
    else
        error("Invalid jobtype $jobtype")
    end
end

function filter_tcikwargs(kwargs_dict::Dict)
    tcikwargs = Dict{Symbol, Any}()
    if haskey(kwargs_dict, :maxnglobalpivot)
        tcikwargs[:maxnglobalpivot]=maybeparse(Int, kwargs_dict[:maxnglobalpivot])
    end
    if haskey(kwargs_dict, :nsearchglobalpivot)
        tcikwargs[:nsearchglobalpivot]=maybeparse(Int, kwargs_dict[:nsearchglobalpivot])
    end
    if haskey(kwargs_dict, :tolmarginglobalsearch)
        tcikwargs[:tolmarginglobalsearch]=maybeparse(Float64, kwargs_dict[:tolmarginglobalsearch])
    end
    return tcikwargs
end

"""
Filter broadening_kwargs for those that are actually used by ΓcoreEvaluator_KF
"""
function filter_broadening_kwargs(;broadening_kwargs...)
    ret = Dict(broadening_kwargs)
    KEYLIST = [:estep, :emin, :emax]
    for key in keys(ret)
        if !(key in KEYLIST)
            delete!(key, ret)
        end
    end
    return ret
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

# ==== JOBTYPES
function matsubaracore(
    outname::AbstractString,
    d::Dict,
    ;
    Rs,
    PSFpath,
    tolerance,
    folder,
    flavor_idx,
    channel,
    cache_center=0,
    use_ΣaIE=true,
    batched_eval=true,
    serialize_tts=true,
    kwargs...
    )
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    T = TCI4Keldysh.dir_to_T(PSFpath)
    times = []
    qttranks = []
    qttbonddims = []
    svd_kernel = true
    @show svd_kernel
    @show batched_eval

    # prepare output
    tcikwargs = filter_tcikwargs(Dict(kwargs))
    d["tcikwargs"] = tcikwargs
    d["times"] = times
    d["ranks"] = qttranks
    d["bonddims"] = qttbonddims
    d["svd_kernel"] = svd_kernel
    d["cache_center"] = cache_center
    d["use_Sigma_aIE"] = use_ΣaIE
    d["tcikwargs"] = Dict(tcikwargs)
    TCI4Keldysh.logJSON(d, outname, folder)

    for R in Rs
        t = @elapsed begin
            qtt = if batched_eval
                TCI4Keldysh.Γ_core_TCI_MF_batched(
                PSFpath,
                R;
                T=T,
                ωconvMat=ωconvMat,
                flavor_idx=flavor_idx,
                use_ΣaIE=use_ΣaIE,
                tolerance=tolerance,
                verbosity=2,
                cache_center=cache_center,
                tcikwargs...
                )
            else
                TCI4Keldysh.Γ_core_TCI_MF(
                PSFpath,
                R;
                T=T,
                ωconvMat=ωconvMat,
                flavor_idx=flavor_idx,
                tolerance=tolerance,
                unfoldingscheme=:interleaved,
                verbosity=2,
                cache_center=cache_center,
                tcikwargs...
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
        TCI4Keldysh.report_mem(true)
        flush(stdout)
        flush(stderr)
    end
end
    
function matsubarafull(
    outname::String,
    d::Dict,
    ;
    Rs,
    PSFpath,
    tolerance,
    folder,
    flavor_idx,
    channel,
    serialize_tts=true,
    kwargs...
    )

    T = TCI4Keldysh.dir_to_T(PSFpath)
    tcikwargs = filter_tcikwargs(Dict(kwargs))
    d["tcikwargs"] = tcikwargs
    times = []
    qttranks = []
    qttbonddims = []
    d["times"] = times
    d["ranks"] = qttranks
    d["bonddims"] = qttbonddims

    TCI4Keldysh.logJSON(d, outname, folder)

    for R in Rs
        t = @elapsed begin
                foreign_channels = [ch for ch in ["a","p","t"] if ch!=channel]
                gbev = TCI4Keldysh.ΓBatchEvaluator_MF(
                    PSFpath,
                    R;
                    channel=channel,
                    T=T,
                    flavor_idx=flavor_idx,
                    foreign_channels=Tuple(foreign_channels)
                )

                # collect initial pivots
                initpivots_ω = TCI4Keldysh.initpivots_Γcore([gbev.gev.core.GFevs[i].GF for i in eachindex(gbev.gev.core.GFevs)])
                initpivots = [QuanticsGrids.origcoord_to_quantics(gbev.grid, tuple(iw...)) for iw in initpivots_ω]

                tt, _, _ = TCI.crossinterpolate2(ComplexF64, gbev, gbev.qf.localdims, initpivots; tolerance, tcikwargs...)
                qtt = QuanticsTCI.QuanticsTensorCI2{ComplexF64}(tt, gbev.grid, gbev.qf)

                do_check_interpolation = true
                if do_check_interpolation
                    Nhalf = 2^(R-1)
                    gridmin = max(1, Nhalf-2^5)
                    gridmax = min(2^R, Nhalf+2^5)
                    grid1D = gridmin:2:gridmax
                    grid = collect(Iterators.product(ntuple(_->grid1D,3)...))
                    qgrid = [QuanticsGrids.grididx_to_quantics(qtt.grid, g) for g in grid]
                    maxerr = TCI4Keldysh.check_interpolation(qtt.tci, gbev, qgrid)
                    println(" Maximum interpolation error: $maxerr (tol=$tolerance)")
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
        TCI4Keldysh.report_mem(true)
        flush(stdout)
        flush(stderr)
    end
end

function keldyshcore(
    outname::AbstractString,
    d::Dict;
    ik,
    Rs,
    PSFpath, 
    tolerance,
    folder,
    flavor_idx::Int,
    channel::String,
    ommax = 0.3183098861837907,
    # ωmin = -0.3183098861837907,
    npivot=5,
    pivot_steps=[div(2^R, maybeparse(Int,npivot) - 1) for R in Rs],
    serialize_tts=true,
    dump_path=nothing,
    resume_path=nothing,
    kwargs...
    )

    beta = TCI4Keldysh.dir_to_beta(PSFpath)
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    T = 1.0/beta
    times = []
    qttranks = []
    bonddims = []
    svd_kernel = true

    ik = maybeparse(Int, ik)
    npivot = maybeparse(Int, npivot)
    ommax = maybeparse(Float64, ommax)
    if !isa(pivot_steps, Vector{Int})
        pivot_steps = [maybeparse(Int, ps) for ps in pivot_steps]
    end

    # get broadening parameters
    base_path = dirname(rstrip(PSFpath, '/'))
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(base_path; channel=channel)
    broadening_kwargs = TCI4Keldysh.read_broadening_settings(base_path; channel=channel)
    if !haskey(broadening_kwargs, :estep)
        broadening_kwargs[:estep] = 50
    end

    # prepare output
    d["times"] = times
    d["ranks"] = qttranks
    d["bonddims"] = bonddims
    d["svd_kernel"] = svd_kernel
    d["numthreads"] = Threads.threadpoolsize()
    d["sigmak"] = sigmak 
    d["gamma"] = γ 
    # d["ommin"] = ωmin
    d["ommax"] = ommax
    d["iK"] = ik
    broadening_kwargs = filter_broadening_kwargs(;broadening_kwargs...)
    d["broadening_kwargs"] = broadening_kwargs
    tcikwargs = filter_tcikwargs(Dict(kwargs))
    d["tcikwargs"] = tcikwargs 
    TCI4Keldysh.logJSON(d, outname, folder)

    for (ir, R) in enumerate(Rs)
        t = @elapsed begin
            qtt = TCI4Keldysh.Γ_core_TCI_KF(
                PSFpath, R, ik, ommax
                ; 
                sigmak=sigmak,
                γ=γ,
                T=T,
                ωconvMat=ωconvMat,
                flavor_idx=flavor_idx,
                dump_path=dump_path,
                resume_path=resume_path,
                tolerance=tolerance,
                verbosity=2,
                unfoldingscheme=:interleaved,
                estep=broadening_kwargs[:estep],
                emin=broadening_kwargs[:emin],
                emax=broadening_kwargs[:emax],
                npivot=npivot,
                pivot_step=pivot_steps[ir],
                tcikwargs...
                )
        end 
        push!(times, t)
        push!(qttranks, TCI4Keldysh.rank(qtt))
        push!(bonddims, TCI.linkdims(qtt.tci))
        TCI4Keldysh.updateJSON(outname, "times", times, folder)
        TCI4Keldysh.updateJSON(outname, "ranks", qttranks, folder)
        TCI4Keldysh.updateJSON(outname, "bonddims", bonddims, folder)

        if serialize_tts            
            serialize_tt(qtt.tci, qtt.grid, outname, folder)
        end

        println(" ===== R=$R: time=$t, rankk(qtt)=$(TCI4Keldysh.rank(qtt))")
        flush(stdout)
        flush(stderr)
    end
end

# ==== JOBTYPES END

"""
Parse input file of the form:

```
TCI4Keldysh BEGIN
<keyword1> <value>
<keyword2> <value>
<etc...>
TCI4Keldysh END
```

Input arguments:
* jobtype
* flavor_idx::Int
* channel::String
* tolerance::Float64
* PSFpath::String
* PSFpath_id::Int
* Rrange, format RminRmax, 4 digits
* local::Bool

To WATCH OUT for:
- All input arguments will first be read in as strings. They must be parsed to the correct type.
- All keywords will be converted to lowercase. So will the corresponding arguments, unless listed in
`is_uppercase_key`
"""
function main(args)
    inp_file = args[1]    
    inp_args = parse_input(inp_file)
    
    println("==== INPUT BEGIN")
    open(inp_file) do f
        ftext = read(f, String)
        print(ftext)
    end
    println("==== INPUT END")

    @show inp_args

    # general arguments
    channel = read_arg(inp_args, "channel")
    flavor_idx = parse(Int, read_arg(inp_args, "flavor_idx"))
    tolerance = parse(Float64, read_arg(inp_args, "tolerance"))
    PSFpath = if haskey(inp_args, "psfpath")
        inp_args["psfpath"]
    else
        PSFpath_id_to_path(parse(Int, inp_args["psfpath_id"]))
    end

    Rrange_str = read_arg(inp_args, "rrange")
    Rmin = parse(Int, Rrange_str[1:2])
    Rmax = parse(Int, Rrange_str[3:4])

    do_local = parse(Bool, read_arg(inp_args, "local"))
    folder = if do_local
                ENV["PWTCIDIR"]
            else
                "pwtcidata"
            end

    # dispatch on jobtype
    jobtype = read_arg(inp_args, "jobtype")
    # avoid double keyword arguments...
    explicit_args = [:jobtype, :psfpath, :folder, :rrange, :flavor_idx, :channel, :tolerance]
    kwargs_dict = Dict{Symbol,Any}()
    for (k,v) in pairs(inp_args)
        if Symbol(k) in explicit_args
            continue
        else
            kwargs_dict[Symbol(k)] = v
        end
    end
    run_job(jobtype;
        PSFpath=PSFpath,
        Rs=Rmin:Rmax,
        folder=folder,
        flavor_idx=flavor_idx,
        channel=channel,
        tolerance=tolerance,
        kwargs_dict...
    )
    println("==== ELVIS HAS LEFT THE BUILDING")
end

main(ARGS)