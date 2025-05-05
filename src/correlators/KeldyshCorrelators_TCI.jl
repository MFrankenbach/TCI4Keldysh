#=
Compute Keldysh correlators using TCI by pointwise evaluation.
The computation of correlators in TCI comes with the caveat that, for the same tolerance, they tend
to have higher bond dimensions than the vertex. Using them as intermediates to obtain the vertex as a QTT
is therefore note efficient. Moreover, getting correlators in tensor train format is not the main objective of this code.
=#

function KFev_filename()
    return "KFev.jld2"    
end

function tciKFcorr_filename()
    return "tciKFcorr.jld2"    
end

function cacheKFcorr_filename()
    return "cacheKFcorr.jld2"
end

"""
Compress Keldysh Correlator at given contour index.
* iK::Int linear index ranging from 1:2^D
* dump_path: if given, store intermediate results in this path; no other calculation should rely on this path
* resume_path: if given, load previous results from this path and continue calculation; no other calculation should rely on this path
"""
function compress_FullCorrelator_pointwise(
    GF::FullCorrelator_KF{D}, iK::Int;
    dump_path=nothing,
    resume_path=nothing,
    do_check_interpolation=true,
    add_pivots=true,
    unfoldingscheme=:interleaved,
    tcikwargs...
    ) where {D}

    R_f = log2(length(GF.ωs_ext[1]))
    R = round(Int, R_f)
    @assert length(GF.ωs_ext[1])==2^R+1
    @assert all(length.(GF.ωs_ext[2:end]).==2^R)
    @assert 1<=iK<=2^(D+1)

    function f(idx::Vararg{Int,D})
        return evaluate(GF, idx...; iK=iK)        
    end

    kwargs_dict = Dict(tcikwargs...)
    tolerance = haskey(kwargs_dict, :tolerance) ? kwargs_dict[:tolerance] : 1.e-8
    if isnothing(resume_path)
        KFev::Union{FullCorrEvaluator_KF_single, KFCEvaluator} = if D==3
            KFCEvaluator(GF)
        else
            cutoff = haskey(kwargs_dict, :tolerance) ? kwargs_dict[:tolerance]*1.e-3 : 1.e-15
            FullCorrEvaluator_KF_single(GF, iK; cutoff=cutoff)
        end
    else
        # load
        error("Loading checkpoints for Keldysh correlator compression no longer supported")
        KFev = load(joinpath(resume_path, KFev_filename()))[jld2_to_dictkey(KFev_filename())]
        # check
        KFC_ = KFev.KFC
        @assert iK==KFev.iK
        @assert length(KFC_.ωs_ext)==length(GF.ωs_ext)
        @assert all([maximum(abs.(KFC_.ωs_ext[wi] .- GF.ωs_ext[wi]))<1.e-12 for wi in 1:D])
        @assert KFC_.ωconvMat==GF.ωconvMat
    end

    if !isnothing(dump_path) && resume_path!=dump_path
        # dump Keldysh correlator evaluator
        file = File(format"JLD2", joinpath(dump_path, KFev_filename()))
        save(file, jld2_to_dictkey(KFev_filename()), KFev)
    end

    pivots = if add_pivots
                initpivots_Γcore([GF])
            else
                # quanticscrossinterpolate default
                [ones(Int,D)]
            end

    # checkpoint results every couple of iterations

    # set up crossinterpolate in quantics representation
    grid = QuanticsGrids.InherentDiscreteGrid{D}(R; unfoldingscheme=unfoldingscheme)

    qlocaldimensions = if grid.unfoldingscheme === :interleaved
        fill(2, D * R)
    else
        fill(2^D, R)
    end

    qKFev_ = 
            if D==1
                q -> KFev(only(QuanticsGrids.quantics_to_origcoord(grid, q)))
            elseif D==2
                q -> KFev(QuanticsGrids.quantics_to_origcoord(grid, q)...)
            elseif D==3
                # KFCEvaluator evaluates all Keldysh components
                q -> KFev(QuanticsGrids.quantics_to_origcoord(grid, q)...)[iK]
            end
    if !isnothing(resume_path)
        # load saved cache and coefficients
        cache = load(joinpath(resume_path,cacheKFcorr_filename()))[jld2_to_dictkey(cacheKFcorr_filename())]
        println("Loaded cached function values")
        qKFev = TCI.CachedFunction{ComplexF64,UInt128}(qKFev_, qlocaldimensions, cache)
    else
        qKFev = TCI.CachedFunction{ComplexF64}(qKFev_, qlocaldimensions)
    end

    qpivots = [QuanticsGrids.origcoord_to_quantics(grid, Tuple(p)) for p in pivots]

    ncheckhistory = 3

    maxiter = haskey(kwargs_dict, :maxiter) ? kwargs_dict[:maxiter] : 20
    # if no dump path is given, just do a normal TCI with maxiter iterations
    ncheckpoint = isnothing(dump_path) ? maxiter : 3 
    if isnothing(resume_path)
        tci = TCI.TensorCI2{ComplexF64}(qKFev, qlocaldimensions, qpivots)
    else
        tci = load(joinpath(resume_path, tciKFcorr_filename()))[jld2_to_dictkey(tciKFcorr_filename())]
        println("Loaded TCI with rank $(TCI.rank(tci))")
    end
    # crossinterpolate2 with checkpoints every ncheckpoint sweeps
    converged = false
    for icheckpoint in 1:Int(ceil(maxiter/ncheckpoint))
        ranks, errors = TCI.optimize!(tci, qKFev; maxiter=ncheckpoint, tcikwargs...)
        println("  After <=$(icheckpoint*ncheckpoint) sweeps: ranks=$ranks, errors=$errors")
        if _tciconverged(ranks, errors, tolerance, ncheckhistory)
            converged = true
            println(" ==== CONVERGED")
            break
        elseif !isnothing(dump_path)
            # dump results
            filetci = File(format"JLD2", joinpath(dump_path, tciKFcorr_filename()))
            save(filetci, jld2_to_dictkey(tciKFcorr_filename()), tci)
            filecache = File(format"JLD2", joinpath(dump_path, cacheKFcorr_filename()))
            save(filecache, jld2_to_dictkey(cacheKFcorr_filename()), qKFev.cache)
        end
    end
    if !converged
        @warn "TCI did not converge within $maxiter sweeps!"
    end
    qtt = QuanticsTCI.QuanticsTensorCI2{ComplexF64}(tci, grid, qKFev)

    if do_check_interpolation
        Nhalf = 2^(R-1)
        gridmin = max(1, Nhalf-2^5)
        gridmax = min(2^R, Nhalf+2^5)
        grid1D = gridmin:2:gridmax
        grid = collect(Iterators.product(ntuple(_->grid1D,D)...))
        if isa(KFev, KFCEvaluator)
            maxerr = check_interpolation(qtt, (w1,w2,w3) -> KFev(w1,w2,w3)[iK], grid)
        else
            maxerr = check_interpolation(qtt, KFev, grid)
        end
        tol = haskey(kwargs_dict, :tolerance) ? kwargs_dict[:tolerance] : :default
        println(" Maximum interpolation error: $maxerr (tol=$tol)")
    end

    return qtt
end