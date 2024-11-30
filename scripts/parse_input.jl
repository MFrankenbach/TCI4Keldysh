using TCI4Keldysh
using Serialization
using QuanticsTCI
import TensorCrossInterpolation as TCI

# UTILITIES
function parse_input(input_file::String)
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
            pts = lowercase.(pts)
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
        "Rrange" => "0512",
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

function run_job(jobtype::String; Rs::AbstractRange{Int}, tolerance, PSFpath, folder, kwargs...)
    beta = TCI4Keldysh.dir_to_beta(PSFpath)
    outname = json_filename("matsubarafull", first(Rs), last(Rs), tolerance, beta; folder=folder)

    # prepare output with general info
    d = Dict()
    d["Rs"] = Rs
    d["beta"] = beta
    kwargs_dict = Dict(kwargs)
    d["flavor_idx"] = kwargs_dict[:flavor_idx]
    d["PSFpath"] = PSFpath
    d["channel"] = kwargs_dict[:channel]
    d["tolerance"] = tolerance
    d["numthreads"] = Threads.threadpoolsize()
    d["job_id"] = if haskey(ENV, "SLURM_JOB_ID")
                    ENV["SLURM_JOB_ID"]
                else
                    000000
                end

    if jobtype=="matsubarafull"
        matsubarafull(outname, d; Rs=Rs, tolerance=tolerance, PSFpath=PSFpath, folder=folder, kwargs...)
    else
        error("Invalid jobtype $jobtype")
    end
end

function filter_tcikwargs(kwargs_dict::Dict)
    tcikwargs = Dict{Symbol, Any}()
    if haskey(kwargs_dict, :maxnglobalpivot)
        tcikwargs[:maxnglobalpivot]=kwargs_dict[:maxnglobalpivot]
    end
    if haskey(kwargs_dict, :nsearchglobalpivot)
        tcikwargs[:nsearchglobalpivot]=kwargs_dict[:nsearchglobalpivot]
    end
    if haskey(kwargs_dict, :tolmarginglobalsearch)
        tcikwargs[:tolmarginglobalsearch]=kwargs_dict[:tolmarginglobalsearch]
    end
    return tcikwargs
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
                # TODO: initial pivots
                tt, _, _ = TCI.crossinterpolate2(ComplexF64, gbev, gbev.qf.localdims; tcikwargs...)
                qtt = QuanticsTCI.QuanticsTensorCI2{ComplexF64}(tt, gbev.grid, gbev.qf)
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

"""
Parse input file of the form:

```
TCI4Keldysh BEGIN
<keyword> <value>
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
    run_job(jobtype;
        PSFpath=PSFpath,
        Rs=Rmin:Rmax,
        folder=folder,
        flavor_idx=flavor_idx,
        channel=channel,
        tolerance=tolerance,
    )

end

main(ARGS)