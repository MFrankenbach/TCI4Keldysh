using TCI4Keldysh
using Serialization
using QuanticsTCI
using QuanticsGrids
using HDF5
import TensorCrossInterpolation as TCI

"""
Determine whether a value to an input key should remain uppercase
"""
function is_uppercase_key(kw::AbstractString)
    list = ["psfpath", "kev", "coreevaluator_kwargs"]
    return (lowercase(kw) in list)
end

# UTILITIES
function maybeparse(T::Type, val)
    return isa(val,T) ? val : parse(T, val)
end

function _to_type(s::Symbol)
    return TCI4Keldysh.eval(s)
end

_to_type(s::String)=_to_type(Symbol(s))

"""
parse dictionary:
key type value key type value
"""
function parse_to_dict(ls::AbstractVector{String})
    @assert mod(length(ls),3)==0 "Invalid dictionary specification"
    ret = Dict{Symbol,Any}()
    for i in 1:div(length(ls),3)
        key = Symbol(ls[3*(i-1)+1])
        T = _to_type(ls[3*(i-1)+2])
        val = maybeparse(T, ls[3*i])
        ret[key]=val
    end
    return ret
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
    elseif jobtype=="keldyshfull"
        keldyshfull(outname, d; Rs=Rs, tolerance=tolerance, PSFpath=PSFpath, folder=folder, flavor_idx=flavor_idx, channel=channel, kwargs...)
    elseif jobtype=="conv_keldyshcore"
        keldyshcore_conv(outname, d; Rs=Rs, PSFpath=PSFpath, folder=folder, flavor_idx=flavor_idx, channel=channel, kwargs...)
    elseif jobtype=="conv_keldyshfull"
        keldyshfull_conv(outname, d; Rs=Rs, PSFpath=PSFpath, folder=folder, flavor_idx=flavor_idx, channel=channel, kwargs...)
    # Correlator jobs
    elseif jobtype=="corrkeldysh"
        corrkeldysh(outname, d; Rs=Rs, tolerance=tolerance, PSFpath=PSFpath, folder=folder, flavor_idx=flavor_idx, channel=channel, kwargs...)
    elseif jobtype=="corrmatsubara"
        corrmatsubara(outname, d; Rs=Rs, tolerance=tolerance, PSFpath=PSFpath, folder=folder, flavor_idx=flavor_idx, channel=channel, kwargs...)
    # elseif jobtype=="keldyshaccuracy"
    #     keldyshaccuracy(outname, d)
    elseif jobtype=="keldyshslice_conv"
        keldyshslice_conv(outname, d; Rs=Rs, PSFpath=PSFpath, folder=folder, flavor_idx=flavor_idx, channel=channel, kwargs...)
    elseif jobtype=="nonlin_keldyshfull"
        nonlin_keldyshfull(outname, d; Rs=Rs, PSFpath=PSFpath, folder=folder, flavor_idx=flavor_idx, channel=channel, kwargs...)
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
    if haskey(kwargs_dict, :pivotsearch)
        tcikwargs[:pivotsearch]=Symbol(kwargs_dict[:pivotsearch])
    end
    return tcikwargs
end

"""
Filter broadening_kwargs for those that are actually used by ΓcoreEvaluator_KF.
If any keyword is explicitly given in override_kwargs, override it.
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

function filter_KFCEvaluator_kwargs(;kwargs...)
    ret = Dict{Symbol,Any}()
    if haskey(Dict(kwargs), :kev)
        val = Dict(kwargs)[:kev]
        # get the requested Type
        ret[:KEV] = _to_type(val)
    end
    if haskey(Dict(kwargs), :coreevaluator_kwargs)
        val = Dict(kwargs)[:coreevaluator_kwargs]
        ret[:coreEvaluator_kwargs] = parse_to_dict(val)
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

function keldyshfull(
    outname::String,
    d::Dict,
    ;
    ik,
    ommax=0.3183098861837907,
    Rs,
    npivot=5,
    pivot_steps=[div(2^R, maybeparse(Int,npivot) - 1) for R in Rs],
    unfoldingscheme=:fused,
    PSFpath,
    tolerance,
    folder,
    flavor_idx,
    channel,
    serialize_tts=true,
    kwargs...
    )

    T = TCI4Keldysh.dir_to_T(PSFpath)
    ik = maybeparse(Int, ik)
    npivot = maybeparse(Int, npivot)
    if !isa(pivot_steps, Vector{Int})
        pivot_steps = [maybeparse(Int, ps) for ps in pivot_steps]
    end
    ommax = maybeparse(Float64, ommax)
    tcikwargs = filter_tcikwargs(Dict(kwargs))
    d["tcikwargs"] = tcikwargs
    times = []
    qttranks = []
    qttbonddims = []
    d["times"] = times
    d["ranks"] = qttranks
    d["bonddims"] = qttbonddims
    d["iK"] = ik
    d["ommax"] = ommax
    grid_kwargs = Dict([(:unfoldingscheme, Symbol(unfoldingscheme))])
    d["unfoldingscheme"] = Symbol(unfoldingscheme)

    (γ, sigmak, broadening_kwargs) = all_broadening_settings(PSFpath, channel)
    broadening_kwargs = filter_broadening_kwargs(;broadening_kwargs...)
    TCI4Keldysh.override_dict!(Dict(kwargs), broadening_kwargs)
    d["broadening_kwargs"] = broadening_kwargs

    evaluator_kwargs = filter_KFCEvaluator_kwargs(;kwargs...)
    d["FullCorrEvaluator_kwargs"] = evaluator_kwargs

    TCI4Keldysh.logJSON(d, outname, folder)

    for (ir, R) in enumerate(Rs)
        t = @elapsed begin
                foreign_channels = [ch for ch in ["a","p","t"] if ch!=channel]
                gbev = TCI4Keldysh.ΓBatchEvaluator_KF(
                    PSFpath,
                    R,
                    ik
                    ;
                    ommax=ommax,
                    channel=channel,
                    T=T,
                    flavor_idx=flavor_idx,
                    foreign_channels=Tuple(foreign_channels),
                    grid_kwargs=grid_kwargs,
                    evaluator_kwargs...,
                    γ=γ, sigmak=sigmak, broadening_kwargs...
                )

                # collect initial pivots
                initpivots_ω = TCI4Keldysh.initpivots_general(ntuple(_->2^R, 3), npivot, pivot_steps[ir]; verbose=true)
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

function all_broadening_settings(PSFpath::AbstractString, channel::AbstractString)
    base_path = dirname(rstrip(PSFpath, '/'))
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(base_path; channel=channel)
    broadening_kwargs = TCI4Keldysh.read_broadening_settings(base_path; channel=channel)
    if !haskey(broadening_kwargs, :estep)
        broadening_kwargs[:estep] = 500
    end
    return (γ, sigmak, broadening_kwargs)
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
    unfoldingscheme=:fused,
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
    (γ, sigmak, broadening_kwargs) = all_broadening_settings(PSFpath, channel)

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
    TCI4Keldysh.override_dict!(Dict(kwargs), broadening_kwargs)
    d["broadening_kwargs"] = broadening_kwargs
    tcikwargs = filter_tcikwargs(Dict(kwargs))
    d["tcikwargs"] = tcikwargs 
    d["unfoldingscheme"] = Symbol(unfoldingscheme)
    TCI4Keldysh.logJSON(d, outname, folder)

    @show kwargs
    evaluator_kwargs = filter_KFCEvaluator_kwargs(;kwargs...)
    @show evaluator_kwargs
    d["FullCorrEvaluator_kwargs"] = evaluator_kwargs

    TCI4Keldysh.report_mem(true)

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
                unfoldingscheme=Symbol(unfoldingscheme),
                estep=broadening_kwargs[:estep],
                emin=broadening_kwargs[:emin],
                emax=broadening_kwargs[:emax],
                npivot=npivot,
                pivot_step=pivot_steps[ir],
                evaluator_kwargs...,
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

function nonlin_keldyshfull(
    outname, d::Dict;
    Rs,
    PSFpath,
    folder,
    flavor_idx,
    channel,
    kwargs...)
    
    (γ, sigmak, broadening_kwargs) = all_broadening_settings(PSFpath, channel)
    TCI4Keldysh.override_dict!(Dict(kwargs), broadening_kwargs)
    eval_kwargs = filter_KFCEvaluator_kwargs(;kwargs...)

    d["broadening_kwargs"] = broadening_kwargs
    d["numthreads"] = Threads.threadpoolsize()
    d["sigmak"] = sigmak 
    d["gamma"] = γ 
    d["FullCorrEvaluator_kwargs"] = eval_kwargs
    TCI4Keldysh.logJSON(d, outname, folder)

    println("==== Initialize frequency grids")
    gridfile = maybeparse(String, kwargs[:frequencygrid])
    om1 = h5read(joinpath(TCI4Keldysh.pdatadir(), gridfile), "om1")
    om2 = h5read(joinpath(TCI4Keldysh.pdatadir(), gridfile), "om2")
    om3 = h5read(joinpath(TCI4Keldysh.pdatadir(), gridfile), "om3")
    nonlin_grid = (om1,om2,om3)
    # ωs_ext = TCI4Keldysh.KF_grid(1.01* om1[end], Rs[begin], 3)
    ωs_ext = ntuple(_->TCI4Keldysh.KF_grid_bos(1.01*om1[end], Rs[begin]), 3)

    println("==== Creating ΓcoreEvaluator_KF...")
    iK = 1 # value does not matter
    gev = TCI4Keldysh.ΓEvaluator_KF(
        PSFpath,
        iK,
        eval_kwargs[:KEV];
        ωs_ext=ωs_ext,
        flavor_idx=flavor_idx,
        channel=channel,
        foreign_channels=TCI4Keldysh.foreign_channels(channel),
        KEV_kwargs=eval_kwargs[:coreEvaluator_kwargs],
        sigmak=sigmak,
        γ=γ,
        broadening_kwargs...
    )
    core = gev.core

    # correlator interpolation
    gpgrids = TCI4Keldysh.Gp_grids(ωs_ext, TCI4Keldysh.channel_trafo(channel))
    GFevs = [((w1,w2,w3) -> TCI4Keldysh.eval_interpol(G, gpgrids, w1,w2,w3)) for G in core.GFevs]
    # self-energy interpolation
    omsig = TCI4Keldysh.Σ_grid(ωs_ext[1:2])
    function sev_(il::Int, inc::Bool, w::Vararg{Float64,3})
        return TCI4Keldysh.eval_interpol(core.sev, il, inc, omsig, w...)
    end

    h5name = joinpath(TCI4Keldysh.pdatadir(), folder, "V_KF.h5")
    println("==== Evaluate K1 on nonlinear grid...")
    for ch in ["a","p","t"]
        res = zeros(ComplexF64, 2,2,2,2, length.(nonlin_grid)...)
        @time begin
            Threads.@threads for ic in collect(Iterators.product(Base.OneTo.(length.(nonlin_grid))...))
                w = ntuple(i -> nonlin_grid[i][ic[i]], 3)
                val = TCI4Keldysh.eval_K1(gev, ch, w...)
                res[:,:,:,:,Tuple(ic)...] .= TCI4Keldysh.unfold_K1(val, ch)
            end
        end
        h5write(h5name, "K1"*ch, res)
    end
    println("==== Evaluate K2 on nonlinear grid...")
    for ch in ["a","p","t"]
        for prime in [true, false]
            res = zeros(ComplexF64, 2,2,2,2, length.(nonlin_grid)...)
            @time begin
                Threads.@threads for ic in collect(Iterators.product(Base.OneTo.(length.(nonlin_grid))...))
                    w = ntuple(i -> nonlin_grid[i][ic[i]], 3)
                    val = TCI4Keldysh.eval_K2(gev, ch, prime, w...)
                    res[:,:,:,:,Tuple(ic)...] .= TCI4Keldysh.unfold_K2(val, ch, prime)
                end
            end
            h5write(h5name, "K2"*ch*"_$(ifelse(prime,"prime","noprime"))", res)
        end
    end
    println("==== Evaluate core vertex on nonlinear grid...")
    res = zeros(ComplexF64, 2,2,2,2, length.(nonlin_grid)...)
    @time begin
        Threads.@threads for ic in collect(Iterators.product(Base.OneTo.(length.(nonlin_grid))...))
            w = ntuple(i -> nonlin_grid[i][ic[i]], 3)
            val = TCI4Keldysh.eval_Γcore_general(
                GFevs,
                sev_,
                core.is_incoming,
                core.letter_combinations,
                w...
                )
            res[:,:,:,:,Tuple(ic)...] .= val
        end
    end
    h5write(h5name, "core", res)
end

function keldyshslice_conv(
    outname, d; 
    Rs,
    ik,
    sliceidx,
    slicedim,
    folder,
    PSFpath,
    flavor_idx,
    channel,
    ommax = 0.3183098861837907,
    kwargs...)

    iK = maybeparse(Int, ik)
    sliceidx = maybeparse(Int, sliceidx)
    slicedim = maybeparse(Int, slicedim)
    (γ, sigmak, broadening_kwargs) = all_broadening_settings(PSFpath, channel)
    TCI4Keldysh.override_dict!(Dict(kwargs), broadening_kwargs)
    eval_kwargs = filter_KFCEvaluator_kwargs(;kwargs...)

    ommax = maybeparse(Float64, ommax)
    d["ommax"] = ommax
    d["iK"] = iK
    d["broadening_kwargs"] = broadening_kwargs
    d["numthreads"] = Threads.threadpoolsize()
    d["sigmak"] = sigmak 
    d["gamma"] = γ 
    d["FullCorrEvaluator_kwargs"] = eval_kwargs
    TCI4Keldysh.logJSON(d, outname, folder)

    for R in Rs
        res = zeros(ComplexF64, 2^R,2^R)
        ωs_ext = TCI4Keldysh.KF_grid(ommax, R, 3)
        gev = TCI4Keldysh.ΓcoreEvaluator_KF(
            PSFpath,
            iK,
            ωs_ext,
            eval_kwargs[:KEV];
            channel=channel,
            flavor_idx=flavor_idx,
            KEV_kwargs=eval_kwargs[:coreEvaluator_kwargs],
            γ=γ,
            sigmak=sigmak,
            broadening_kwargs...
        )
        nonslice = [1,2,3]
        nonslice[slicedim:end] .-= 1
        # evaluate
        Threads.@threads for ic in collect(Iterators.product(1:2^R, 1:2^R))
            w = ntuple(n -> n==slicedim ? sliceidx : ic[nonslice[n]], 3)
            # println("evaluate at w=$(w)")
            res[Tuple(ic)...] = gev(ntuple(n -> n==slicedim ? sliceidx : ic[nonslice[n]], 3)...)
        end

        # store as h5
        h5name = "V_KF_dim$(slicedim)_slice$(sliceidx)_R$(R)_iK$(iK).h5"
        h5write(joinpath(TCI4Keldysh.pdatadir(), folder, h5name), "V_KF", res)
    end
end

"""
Conventional computation of full keldysh vertex
"""
function keldyshfull_conv(outname, d; 
    Rs,
    folder,
    PSFpath,
    flavor_idx,
    channel,
    ommax = 0.3183098861837907,
    kwargs...)

    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    ommax = maybeparse(Float64, ommax)
    (γ, sigmak, broadening_kwargs) = all_broadening_settings(PSFpath, channel)
    # can overwrite broadening params to test influence of broadening on vertex
    if haskey(Dict(kwargs), :sigmak)
        sigmak = [maybeparse(Float64, kwargs[:sigmak])]
    end
    if haskey(Dict(kwargs), :gamma)
        γ = maybeparse(Float64, kwargs[:gamma])
    end
    # overwrite other explicitly given broadening parameters
    @show (γ, sigmak)
    TCI4Keldysh.override_dict!(Dict(kwargs), broadening_kwargs)

    d["channel"] = channel
    d["gamma"] = γ 
    d["sigmak"] = sigmak
    d["broadening_kwargs"] = broadening_kwargs 
    d["ommax"] = ommax 
    TCI4Keldysh.logJSON(d, outname, folder)
    
    for R in Rs
        if R>9
            error("This calculation (R=$R) is doomed.")
        end
        ωs_ext = TCI4Keldysh.KF_grid(ommax, R, 3)
        T = TCI4Keldysh.dir_to_T(PSFpath)
        gamcore = TCI4Keldysh.compute_Γfull_symmetric_estimator(
            "KF",
            PSFpath;
            T,
            flavor_idx,
            ωs_ext,
            channel=channel,
            sigmak=sigmak,
            γ=γ,
            broadening_kwargs...
            )

        # store
        h5name = "V_KF_$(channel)_R=$(R).h5"
        h5write(joinpath(TCI4Keldysh.pdatadir(), folder, h5name), "V_KF", gamcore)
        for i in eachindex(ωs_ext)
            h5write(joinpath(TCI4Keldysh.pdatadir(), folder, h5name), "omgrid$i", ωs_ext[i])
        end
        println("==== R=$R DONE")
        flush(stdout)
        flush(stderr)
    end
end


"""
Conventional computation of keldysh core vertex
"""
function keldyshcore_conv(outname, d::Dict; 
    Rs,
    folder,
    PSFpath,
    flavor_idx,
    channel,
    ommax = 0.3183098861837907,
    kwargs...)

    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    ommax = maybeparse(Float64, ommax)
    (γ, sigmak, broadening_kwargs) = all_broadening_settings(PSFpath, channel)
    # can overwrite broadening params to test influence of broadening on vertex
    if haskey(Dict(kwargs), :sigmak)
        sigmak = [maybeparse(Float64, kwargs[:sigmak])]
    end
    if haskey(Dict(kwargs), :gamma)
        γ = maybeparse(Float64, kwargs[:gamma])
    end
    # overwrite other explicitly given broadening parameters
    @show (γ, sigmak)
    TCI4Keldysh.override_dict!(Dict(kwargs), broadening_kwargs)

    d["channel"] = channel
    d["gamma"] = γ 
    d["sigmak"] = sigmak
    d["broadening_kwargs"] = broadening_kwargs 
    d["ommax"] = ommax 
    TCI4Keldysh.logJSON(d, outname, folder)
    
    for R in Rs
        if R>9
            error("This calculation (R=$R) is doomed.")
        end
        ωs_ext = TCI4Keldysh.KF_grid(ommax, R, 3)
        T = TCI4Keldysh.dir_to_T(PSFpath)
        om_sig = TCI4Keldysh.Σ_grid(ntuple(i -> ωs_ext[i],2))
        (Σ_L, Σ_R) = TCI4Keldysh.calc_Σ_KF_aIE_viaR(PSFpath, om_sig; T=T, flavor_idx=flavor_idx, sigmak, γ, broadening_kwargs...)
        gamcore = TCI4Keldysh.compute_Γcore_symmetric_estimator(
            "KF",
            joinpath(PSFpath, "4pt"),
            Σ_R
            ;
            Σ_calcL=Σ_L,
            T,
            flavor_idx,
            ωs_ext,
            ωconvMat,
            sigmak=sigmak,
            γ=γ,
            broadening_kwargs...
            )

        # store
        h5name = "V_KF_$(channel)_R=$(R).h5"
        h5write(joinpath(TCI4Keldysh.pdatadir(), folder, h5name), "V_KF", gamcore)
        for i in eachindex(ωs_ext)
            h5write(joinpath(TCI4Keldysh.pdatadir(), folder, h5name), "omgrid$i", ωs_ext[i])
        end
        println("==== R=$R DONE")
        flush(stdout)
        flush(stderr)
    end
end


function corrmatsubara(
    outname::AbstractString, d::Dict;
    Rs,
    PSFpath,
    folder,
    flavor_idx,
    channel,
    tolerance,
    serialize_tts=true,
    unfoldingscheme=:interleaved,
    add_pivots=true,
    kwargs...)


    npt = 4
    times = []
    qttranks = []
    qttbonddims = []
    svd_kernel = true
    T = TCI4Keldysh.dir_to_T(PSFpath)
    beta = 1.0/T

    # prepare output
    d["times"] = times
    d["ranks"] = qttranks
    d["bonddims"] = qttbonddims
    d["svd_kernel"] = svd_kernel
    d["add_pivots"] = maybeparse(Bool, add_pivots)
    tcikwargs = filter_tcikwargs(Dict(kwargs))
    TCI4Keldysh.logJSON(d, outname, folder)

    for R in Rs
        GF = TCI4Keldysh.dummy_correlator(npt, R; PSFpath=PSFpath, beta=beta, channel=channel)[flavor_idx]
        t = @elapsed begin
            qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(
                GF, svd_kernel; 
                add_pivots=add_pivots, 
                tolerance=tolerance, 
                unfoldingscheme=Symbol(unfoldingscheme), 
                tcikwargs...
                )
        end
        push!(times, t)
        push!(qttranks, TCI4Keldysh.rank(qtt))
        push!(qttbonddims, TCI.linkdims(qtt.tci))
        TCI4Keldysh.updateJSON(outname, "times", times, folder)
        TCI4Keldysh.updateJSON(outname, "ranks", qttranks, folder)
        TCI4Keldysh.updateJSON(outname, "bonddims", qttbonddims, folder)

        if serialize_tts
            serialize_tt(qtt.tci, qtt.grid, outname, folder)
        end

        println(" ===== R=$R: time=$t, rankk(qtt)=$(TCI4Keldysh.rank(qtt))")
        flush(stdout)
    end
end



function corrkeldysh(
    outname::AbstractString, d::Dict;
    ik,
    Rs,
    PSFpath,
    folder,
    flavor_idx,
    channel,
    tolerance=tolerance,
    ommax = 0.3183098861837907,
    unfoldingscheme=:fused,
    serialize_tts=true,
    kwargs...)

    @show kwargs

    beta = TCI4Keldysh.dir_to_beta(PSFpath)
    Ops = ["F1", "F1dag", "F3", "F3dag"]
    T = 1.0/beta
    npt = 4
    times = []
    qttranks = []
    bonddims = []
    svd_kernel = true
    ik = maybeparse(Int, ik)
    ommax = maybeparse(Float64, ommax)
    # broadening
    (γ, sigmak, broadening_kwargs) = all_broadening_settings(PSFpath, channel)
    broadening_kwargs = filter_broadening_kwargs(; broadening_kwargs...)
    TCI4Keldysh.override_dict!(Dict(kwargs), broadening_kwargs)
    # prepare output
    d["times"] = times
    d["ranks"] = qttranks
    d["bonddims"] = bonddims
    d["svd_kernel"] = svd_kernel
    d["gamma"] = γ
    d["sigmak"] = sigmak 
    d["ommax"] = ommax
    d["iK"] = ik
    d["broadening_kwargs"] = broadening_kwargs
    tcikwargs = filter_tcikwargs(Dict(kwargs))
    d["tcikwargs"] = tcikwargs 
    d["Ops"] = Ops
    TCI4Keldysh.logJSON(d, outname, folder)

    for R in Rs
        
        # create correlator
        D = npt-1
        ωs_ext = TCI4Keldysh.KF_grid(ommax, R, D)
        ωconvMat = TCI4Keldysh.channel_trafo(channel)
        KFC = TCI4Keldysh.FullCorrelator_KF(
            joinpath(PSFpath, "4pt"), Ops;
            T=T,
            ωs_ext=ωs_ext,
            flavor_idx=flavor_idx,
            ωconvMat=ωconvMat,
            sigmak=sigmak,
            γ=γ,
            name="KFC",
            broadening_kwargs...
            )
        # create correlator END

        t = @elapsed begin
            qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(
                KFC, ik;
                tolerance=tolerance,
                dump_path=nothing,
                resume_path=nothing,
                verbosity=2,
                unfoldingscheme=Symbol(unfoldingscheme),
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

# function keldyshaccuracy(
#     outname::AbstractString, d::Dict;
#     refpath::String,
#     Rs,
#     ik::Int,
#     PSFpath,
#     KEV::Type=TCI4Keldysh.MultipoleKFCEvaluator,
#     coreEvaluator_kwargs::Dict{Symbol,Any}=Dict{Symbol,Any}(:cutoff=>1.e-6, :nlevel=>4)
#     )

#     iK = ik
#     R = only(Rs)
#     # load settings
#     refjson = only(
#         filter(f -> endswith(f, ".json"), readdir(refpath))
#     )
#     refsettings = TCI4Keldysh.readJSON(refjson, refpath)
#     ωmax = refsettings["ommax"]
#     broadening_kwargs_ = refsettings["broadening_kwargs"]
#     broadening_kwargs = Dict{Symbol,Any}()
#     for (key,val) in pairs(broadening_kwargs_)
#         broadening_kwargs[Symbol(key)] = val
#     end
#     γ = refsettings["gamma"]
#     sigmak = Vector{Float64}(refsettings["sigmak"])
#     channel = refsettings["channel"]
#     flavor_idx = refsettings["flavor_idx"]
#     if !(R in refsettings["Rs"])
#         error("Requested grid size is not available")
#     end

#     T = TCI4Keldysh.dir_to_T(PSFpath)

#     # prepare core evaluator
#     ωconvMat = TCI4Keldysh.channel_trafo(channel)
#     # make frequency grid
#     D = size(ωconvMat, 2)
#     @assert D==3
#     ωs_ext = TCI4Keldysh.KF_grid(ωmax, R, D)

#     # all 16 4-point correlators
#     letters = ["F", "Q"]
#     letter_combinations = kron(kron(letters, letters), kron(letters, letters))
#     op_labels = ("1", "1dag", "3", "3dag")
#     op_labels_symm = ("3", "3dag", "1", "1dag")
#     is_incoming = (false, true, false, true)

#     # create correlator objects
#     Ncorrs = length(letter_combinations)
#     GFs = Vector{TCI4Keldysh.FullCorrelator_KF{D}}(undef, Ncorrs)
#     PSFpath_4pt = joinpath(PSFpath, "4pt")
#     filelist = readdir(PSFpath_4pt)
#     for l in 1:Ncorrs
#         letts = letter_combinations[l]
#         println("letts: ", letts)
#         ops = [letts[i]*op_labels[i] for i in 1:4]
#         if !any(TCI4Keldysh.parse_Ops_to_filename(ops) .== filelist)
#             ops = [letts[i]*op_labels_symm[i] for i in 1:4]
#         end
#         GFs[l] = TCI4Keldysh.FullCorrelator_KF(PSFpath_4pt, ops; T, flavor_idx, ωs_ext, ωconvMat, sigmak=sigmak, γ=γ, broadening_kwargs...)
#     end

#     # evaluate self-energy
#     incoming_trafo = diagm([inc ? -1 : 1 for inc in is_incoming])
#     @assert all(sum(abs.(ωconvMat); dims=2) .<= 2) "Only two nonzero elements per row in frequency trafo allowed"
#     ωstep = abs(ωs_ext[1][1] - ωs_ext[1][2])
#     Σω_grid = TCI4Keldysh.KF_grid_fer(2*ωmax, R+1)
#     (Σ_L,Σ_R) = TCI4Keldysh.calc_Σ_KF_aIE_viaR(PSFpath, Σω_grid; flavor_idx=flavor_idx, T=T, sigmak, γ, broadening_kwargs...)

#     # frequency grid offset for self-energy
#     ΣωconvMat = incoming_trafo * ωconvMat
#     corner_low = [first(ωs_ext[i]) for i in 1:D]
#     corner_idx = ones(Int, D)
#     corner_image = ΣωconvMat * corner_low
#     idx_image = ΣωconvMat * corner_idx
#     desired_idx = [findfirst(w -> abs(w-corner_image[i])<ωstep*0.1, Σω_grid) for i in eachindex(corner_image)]
#     ωconvOff = desired_idx .- idx_image

#     sev = TCI4Keldysh.SigmaEvaluator_KF(Σ_R, Σ_L, ΣωconvMat, ωconvOff)

#     gev = TCI4Keldysh.ΓcoreEvaluator_KF(GFs, iK, sev, KEV; coreEvaluator_kwargs...)

#     # load reference
#     refdatafile = joinpath(refpath, "V_KF_$(channel)_R=$(R).h5")
#     refdata = h5read(refdatafile, "V_KF")[:,:,:,TCI4Keldysh.KF_idx(iK,3)...]
#     # SETUP DONE

#     # check
#     N = 10^4
#     errs = Vector{Float64}(undef, N)
#     vals = Vector{ComplexF64}(undef, N)
#     idx_range = Base.OneTo.(size(refdata))
#     Threads.@threads for n in 1:N
#         idx = ntuple(i -> rand(idx_range[i]), 3)
#         val = gev(idx...)
#         errs[n] = abs(val - refdata[idx...])
#         vals[n] = val
#     end

#     h5write(joinpath(refpath, "errs.h5"), "vals", vals)
#     h5write(joinpath(refpath, "errs.h5"), "errs", errs)
# end

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