#=
Compute PartialCorrelator in quantics representation by TCI-ing the tucker decomposition via pointwise evaluation.
=#

"""
QTCI-compress 3D tucker decomposition by pointwise evaluation.
Use TuckerEvaluator3D.
"""
function compress_tucker_pointwise_vectorized(tucker::TuckerDecomposition{T,D}, svd_kernel=true; qtcikwargs...) where {T,D}
    if svd_kernel
        shift_singular_values_to_center!(tucker)
    end
    R = grid_R(tucker)
    # make sure grid is large enough
    @assert all([size(tucker.legs[i], 1) >= 2^R for i in 1:D]) "Need tucker legs with 2^R or 2^R+1 frequencies (R=$R)"
    pivot = collect(ntuple(i -> 2^(R-1), D))
    tev = TuckerEvaluator3D(tucker)
    tqtt, _, _ = quanticscrossinterpolate(T, tev, ntuple(i -> 2^R, D), [pivot]; qtcikwargs...)

    return tqtt
end

"""
QTCI-compress tucker decomposition by pointwise evaluation
"""
function compress_tucker_pointwise(tucker::TuckerDecomposition{T,D}, svd_kernel=true; qtcikwargs...) where {T,D}
    if svd_kernel
        if any(size(tucker.center) .> 300) @warn "SVD-ing legs of sizes $(size.(tucker.legs))" end
        shift_singular_values_to_center!(tucker)
    end
    R = grid_R(tucker)
    # make sure grid is large enough
    @assert all([size(tucker.legs[i], 1) >= 2^R for i in 1:D]) "Need tucker legs with 2^R or 2^R+1 frequencies (R=$R)"
    pivot = collect(ntuple(i -> 2^(R-1), D))
    tqtt, _, _ = quanticscrossinterpolate(T, tucker, ntuple(i -> 2^R, D), [pivot]; qtcikwargs...)

    return tqtt
end

"""
QTCI-compress regular part of PartialCorrelator (including frequency rotation) by pointwise evaluation.
"""
function compress_reg_PartialCorrelator_pointwise(Gp::PartialCorrelator_reg{D}, svd_kernel::Bool=false; qtcikwargs...) where {D}
    R = grid_R(Gp)
    pivot = collect(ntuple(i -> 2^(R-1), D))
    if svd_kernel
        kwargs_dict = Dict(qtcikwargs)
        cutoff = haskey(Dict(kwargs_dict), :tolerance) ? kwargs_dict[:tolerance]*1.e-2 : 1.e-12
        svd_kernels!(Gp.tucker; cutoff=cutoff)
    end
    tqtt, _, _ = quanticscrossinterpolate(eltype(Gp.tucker.center), Gp, ntuple(i -> 2^R, D), [pivot]; qtcikwargs...)

    return tqtt
end

"""
QTCI-compress  PartialCorrelator (including frequency rotation and anomalous part) by pointwise evaluation.
"""
function compress_PartialCorrelator_pointwise(
    Gp::PartialCorrelator_reg{D}, svd_kernel::Bool=false; qtcikwargs...
    ) where {D}
    R = grid_R(Gp)
    pivot = collect(ntuple(i -> 2^(R-1), D))
    anev = AnomalousEvaluator(Gp)

    if svd_kernel
        if any(size(Gp.tucker.center) .> 300) @warn "SVD-ing legs of sizes $(size.(Gp.tucker.legs))" end
        kwargs_dict = Dict(qtcikwargs)
        # TODO: Rigorous scheme to choose cutoff
        cutoff = haskey(Dict(kwargs_dict), :tolerance) ? kwargs_dict[:tolerance]*1.e-2 : 1.e-12
        svd_kernels!(Gp.tucker; cutoff=cutoff)
    end

    function _qttfun(w::Vararg{Int,D})
        # return Gp(w...) + evaluate_ano_with_ωconversion(Gp, w...)
        return Gp(w...) + anev(w...)
    end
    tqtt, _, _ = quanticscrossinterpolate(eltype(Gp.tucker.center), _qttfun, ntuple(i -> 2^R, D), [pivot]; qtcikwargs...)

    return tqtt
end

"""
QTCI-compress anomalous part of PartialCorrelator (including frequency rotation) by pointwise evaluation.
Only for D>1 since 2-point anomalous terms are trivial.
\n     DEPRECATED: TCI often misses features because anomalous term is lower-dimensional.
"""
function compress_PartialCorrelator_ano_pointwise(Gp::PartialCorrelator_reg{D}; qtcikwargs...) where {D}
    @warn "DEPRECATED: TCI often misses features because anomalous term is lower-dimensional."
    @assert ano_term_required(Gp) "Trying to evaluate anomalous term for purely regular PartialCorrelator correlator"
    R = grid_R(Gp)

    # find pivot that hits anomalous part: w_int = Gp.ωconvMat*w_ext + Gp.ωconvOff
    bos_idx = get_bosonic_idx(Gp)
    zero_idx = findfirst(x -> abs(x)<=1.e-2*Gp.T, Gp.tucker.ωs_legs[bos_idx])
    # place ourselves in the middle of each grid
    w_int = collect(ntuple(i -> div(size(Gp.tucker.legs[i],1), 2), D))
    w_int[bos_idx] = zero_idx
    w_ext = Gp.ωconvMat \ (w_int - Gp.ωconvOff)
    pivot = round.(Int, w_ext)
    
    # qtci-compress
    function _qtcifun(w::Vararg{Int,D})
        return evaluate_ano_with_ωconversion(Gp, w...)
    end
    tqtt, _, _ = quanticscrossinterpolate(eltype(Gp.tucker.center), _qtcifun, ntuple(i -> 2^R, D), [pivot]; qtcikwargs...)

    return tqtt
end

"""
Obtain TT for partial correlator by pointwise evaluation as:
x--x--x
|  |  | 
ω1 ω2 ω3

TODO: test
"""
function compress_FullCorrelator_natural(GF::FullCorrelator_MF{D}, svd_kernel::Bool=false; tcikwargs...) where {D}

    kwargs_dict = Dict(tcikwargs)
    cutoff = haskey(Dict(kwargs_dict), :tolerance) ? kwargs_dict[:tolerance]*1.e-2 : 1.e-12
    fev = FullCorrEvaluator_MF(GF, svd_kernel; cutoff=cutoff)

    # collect anomalous term pivots
    pivots = [zeros(Int, D)]
    for i in eachindex(GF.Gps)
        if fev.ano_terms_required[i]
            Gp = GF.Gps[i]
            bos_idx = get_bosonic_idx(Gp)
            zero_idx = findfirst(x -> abs(x)<=1.e-2*Gp.T, Gp.tucker.ωs_legs[bos_idx])
            w_int = collect(ntuple(i -> div(size(Gp.tucker.legs[i],1), 2), D))
            w_int[bos_idx] = zero_idx
            w_ext = Gp.ωconvMat \ (w_int - Gp.ωconvOff)
            pivot = round.(Int, w_ext)
            push!(pivots, pivot)
        end
    end

    T = eltype(GF.Gps[1].tucker.center)
    f_(x::Vector{Int}) = fev(x...)

    R = grid_R(GF)
    localdims = fill(2^R, D)
    fc_ = TCI.CachedFunction{T}(f_, localdims)
    tt, _, _ = TCI.crossinterpolate2(T, fc_, localdims; tcikwargs...)

    return tt
end

"""
Obtain qtt for full correlator by pointwise evaluation.
Here, we count how many evaluations take place in a given region.
"""
function compress_FullCorrelator_pointwise_evalcount(GF::FullCorrelator_MF{D}, svd_kernel::Bool=false; qtcikwargs...) where {D}
    # check external frequency grids
    R = grid_R(GF)

    kwargs_dict = Dict(qtcikwargs)
    cutoff = haskey(Dict(kwargs_dict), :tolerance) ? kwargs_dict[:tolerance]*1.e-2 : 1.e-12
    fev = FullCorrEvaluator_MF(GF, svd_kernel; cutoff=cutoff)

    # collect anomalous term pivots
    pivots = [zeros(Int, D)]
    for i in eachindex(GF.Gps)
        if fev.ano_terms_required[i]
            Gp = GF.Gps[i]
            bos_idx = get_bosonic_idx(Gp)
            zero_idx = findfirst(x -> abs(x)<=1.e-2*Gp.T, Gp.tucker.ωs_legs[bos_idx])
            w_int = collect(ntuple(i -> div(size(Gp.tucker.legs[i],1), 2), D))
            w_int[bos_idx] = zero_idx
            w_ext = Gp.ωconvMat \ (w_int - Gp.ωconvOff)
            pivot = round.(Int, w_ext)
            push!(pivots, pivot)
        end
    end

    N_inner = 64
    N_stripe = 10
    name = "MF corr eval, tol=$(kwargs_dict[:tolerance]), R=$R, T=$(GF.Gps[1].T)"

    function innerpoint(w) :: Bool
        return all([abs(iw - 2^(R-1)) <= N_inner for iw in w])
    end

    function stripe_point(w) :: Bool
        if innerpoint(w)
            return false
        end
        return sum(ntuple(i -> abs(w[i] - 2^(R-1))<=N_stripe ? 0 : 1, D)) <= 1
    end

    midpoints = Vector{Vector{Int}}(undef, length(GF.Gps))
    for i in eachindex(GF.Gps)
        midpoints[i] = [div(length(GF.Gps[i].tucker.ωs_legs[i]), 2) for i in 1:D]
    end
    # are internal frequencies in any partial correlator close to some axis?
    function partial_stripe(w) :: Bool
        for (g,Gp) in enumerate(GF.Gps)
            w_int = Gp.ωconvMat * SA[w...] + Gp.ωconvOff
            if sum(ntuple(i -> abs(w_int[i] - midpoints[g][i])<=N_stripe ? 0 : 1, D)) <= 1
                return true
            end
        end
        return false
    end

    ecfev = EvaluationCounter(fev, [innerpoint, stripe_point, partial_stripe], name)

    qtt, _, _ = quanticscrossinterpolate(eltype(GF.Gps[1].tucker.center), ecfev, ntuple(i -> 2^R, D), pivots; qtcikwargs...)

    reportcount(ecfev; log="evalcounts.log")

    return qtt
end

"""
Obtain qtt for full correlator by combining block- and pointwise evaluation.
"""
function compress_FullCorrelator_batched(GF::FullCorrelator_MF{D}, svd_kernel::Bool=false; tcikwargs...) where {D}

    kwargs_dict = Dict(tcikwargs)
    cutoff = haskey(Dict(kwargs_dict), :tolerance) ? kwargs_dict[:tolerance]*1.e-2 : 1.e-12
    fevbatch = FullCorrBatchEvaluator_MF(GF, svd_kernel; cutoff=cutoff)

    # collect anomalous term pivots
    pivots = [ones(Int, length(fevbatch.localdims))]
    for i in eachindex(GF.Gps)
        if fevbatch.GFev.ano_terms_required[i]
            Gp = GF.Gps[i]
            bos_idx = get_bosonic_idx(Gp)
            zero_idx = findfirst(x -> abs(x)<=1.e-2*Gp.T, Gp.tucker.ωs_legs[bos_idx])
            w_int = collect(ntuple(i -> div(size(Gp.tucker.legs[i],1), 2), D))
            w_int[bos_idx] = zero_idx
            w_ext = Gp.ωconvMat \ (w_int - Gp.ωconvOff)
            pivot = round.(Int, w_ext)
            qpivot = QuanticsGrids.origcoord_to_quantics(fevbatch.grid, Tuple(pivot)) 
            push!(pivots, qpivot)
        end
    end

    tt, _, _ = TCI.crossinterpolate2(eltype(GF.Gps[1].tucker.center), fevbatch, fevbatch.localdims, pivots; tcikwargs...)

    return (tt, fevbatch)
end

function check_GF_evaluator(refval::Array{ComplexF64,D}, GFev, grid::NTuple{D, AbstractRange}; tolerance=1.e-12) where {D}
    Neval = length(grid[1])
    GFval = zeros(ComplexF64, size(refval))
    Threads.@threads for idx in collect(Iterators.product(ntuple(_->1:Neval,D)...) )
        w = ntuple(i -> grid[i][idx[i]], D)
        GFval[idx...] = GFev(w...)
    end

    maxref = maximum(abs.(refval))
    diff = (GFval .- refval) ./ maxref
    printstyled("Error of GF evaluator: \n"; color=:blue)
    maxerr = maximum(abs.(diff))
    @show maxref
    @show maxerr
    @show maxerr * maxref
    @assert maxerr <= tolerance

    if D==3
        heatmap(log10.(abs.(GFval))[:,:,div(Neval,2)])
        savefig("GF.png")
        heatmap(log10.(abs.(refval))[:,:,div(Neval,2)])
        savefig("ref.png")
        heatmap(log10.(abs.(diff))[:,:,div(Neval,2)])
        savefig("diff.png")
    end
end

"""
Check accuracy of FullCorrEvaluator with given SVD cutoff and Tucker cutoff.
"""
function test_FullCorrEvaluator_MF(npt::Int=4; R::Int=5, tolerance=1.e-20)
    D = npt-1
    GF = dummy_correlator(npt, R; beta=2000.0)[1]

    # reference
    N = 2^R
    Neval = 2^5
    eval_step = max(div(N, Neval), 1)
    gridslice = 1:eval_step:N
    @assert length(gridslice)==Neval

    @assert intact(GF)
    refval = zeros(ComplexF64, ntuple(_->Neval, D))
    Threads.@threads for idx in collect(Iterators.product(ntuple(_->1:Neval,D)...))
        w = ntuple(i -> gridslice[idx[i]], D)
        refval[idx...] = evaluate(GF, w...)
    end

    # numerically exact evaluators
    cutoff = 0.01 * tolerance
    tucker_cut = 0.1 * tolerance
    GFev = FullCorrEvaluator_MF(deepcopy(GF), true; cutoff=cutoff, tucker_cutoff=tucker_cut)
    check_GF_evaluator(refval, GFev, ntuple(_->gridslice, D); tolerance=tolerance)
    GFev = nothing

    GFbev = FullCorrBatchEvaluator_MF(deepcopy(GF), true; cutoff=cutoff, tucker_cutoff=tucker_cut)
    function _GFbev(w::Vararg{Int, D}) where {D}
        return evaluate(GFbev, w...)
    end
    check_GF_evaluator(refval, _GFbev, ntuple(_->gridslice, D); tolerance=tolerance)
end


"""
Obtain qtt for full correlator by pointwise evaluation.
"""
function compress_FullCorrelator_pointwise(GF::FullCorrelator_MF{D}, svd_kernel::Bool=false; cut_tucker=true, qtcikwargs...) where {D}
    # check external frequency grids
    R = grid_R(GF)

    kwargs_dict = Dict(qtcikwargs)
    cutoff = haskey(Dict(kwargs_dict), :tolerance) ? kwargs_dict[:tolerance]*1.e-2 : 1.e-12
    fev = FullCorrEvaluator_MF(GF, svd_kernel; cutoff=cutoff, tucker_cutoff=cutoff*10.0)

    # collect anomalous term pivots
    pivots = [zeros(Int, D)]
    for i in eachindex(GF.Gps)
        if fev.ano_terms_required[i]
            Gp = GF.Gps[i]
            bos_idx = get_bosonic_idx(Gp)
            zero_idx = findfirst(x -> abs(x)<=1.e-2*Gp.T, Gp.tucker.ωs_legs[bos_idx])
            w_int = collect(ntuple(i -> div(size(Gp.tucker.legs[i],1), 2), D))
            w_int[bos_idx] = zero_idx
            w_ext = Gp.ωconvMat \ (w_int - Gp.ωconvOff)
            pivot = round.(Int, w_ext)
            push!(pivots, pivot)
        end
    end

    if !cut_tucker
        function _eval(w::Vararg{Int,D}) where {D}
            return fev(Val{:nocut}(), w...)
        end
        qtt, _, _ = quanticscrossinterpolate(eltype(GF.Gps[1].tucker.center), _eval, ntuple(i -> 2^R, D), pivots; qtcikwargs...)

        return qtt
    else
        qtt, _, _ = quanticscrossinterpolate(eltype(GF.Gps[1].tucker.center), fev, ntuple(i -> 2^R, D), pivots; qtcikwargs...)
        return qtt
    end
end

# ========== TESTS & PLOTTING

function test_compress_PartialCorrelator_pointwise2D(;perm_idx::Int=1, ano=true)
    R = 7
    tolerance = 1.e-8
    GF = multipeak_correlator_MF(3, R; beta=1000.0, nωdisc=20)
    Gp = GF.Gps[perm_idx]
    @show get_bosonic_idx(Gp)
    @time qtt = if ano && ano_term_required(Gp)
                compress_PartialCorrelator_pointwise(Gp; tolerance=tolerance, unfoldingscheme=:interleaved)
            else
                compress_reg_PartialCorrelator_pointwise(Gp; tolerance=tolerance, unfoldingscheme=:interleaved)
            end
    @show rank(qtt)

    @time refval = if ano && ano_term_required(Gp)
                        precompute_all_values_MF(Gp)
                    else
                        precompute_all_values_MF_noano(Gp)
                    end
    slice = (1:2^R, 1:2^R)
    qttval = reshape([qtt(i,j) for j in slice[2] for i in slice[1]], 2^R, 2^R)
    diff = abs.(refval[slice...] - qttval) ./ maximum(abs.(refval))
    @show maximum(diff)

    scfun = x -> log10(abs(x))
    heatmap(scfun.(refval))
    savefig("Gpref.png")
    heatmap(scfun.(qttval))
    savefig("Gp.png")
    heatmap(scfun.(diff))
    savefig("diff.png")
end

function test_compress_PartialCorrelator_pointwise3D(
        ;R=7, beta::Float64=1.0, perm_idx=1, nomdisc=10, ano=true
        )
    tolerance = 1.e-8
    GF = multipeak_correlator_MF(4, R; beta=beta, nωdisc=nomdisc)
    Gp = GF.Gps[perm_idx]
    printstyled(" ---- Compress correlator...\n"; color=:blue)
    @time qtt = if ano
                    compress_PartialCorrelator_pointwise(Gp; tolerance=tolerance, unfoldingscheme=:interleaved)
                else
                    compress_reg_PartialCorrelator_pointwise(Gp; tolerance=tolerance, unfoldingscheme=:interleaved)
                end
    @show size(Gp.tucker.center)
    @show rank(qtt)

    printstyled(" ---- Compute reference...\n"; color=:blue)
    @time refval = if ano && ano_term_required(Gp)
                        precompute_all_values_MF(Gp)
                    else
                        precompute_all_values_MF_noano(Gp)
                    end
    maxref = maximum(abs.(refval))
    maxdiff = 0.0
    if ano_term_required(Gp)
        anovalues = precompute_ano_values_MF(Gp)
        printstyled("      Norm of anomalous contribution: $(norm(anovalues))\n"; color=:cyan)
    end
    printstyled(" ---- Compare reference vs. QTT ...\n"; color=:blue)
    for s in 1:100
        s = rand(1:2^R, 3)
        maxdiff = max(maxdiff, abs(qtt(s...) - refval[s...]) / maxref)
    end
    @show maxdiff
    printstyled(" ==== DONE\n\n"; color=:blue)
end

function test_compress_PartialCorrelator_ano_pointwise2D(;perm_idx=1)
    R = 9
    GF = multipeak_correlator_MF(3, R; beta=10.0, nωdisc=10)
    Gp = GF.Gps[perm_idx]
    @assert ano_term_required(Gp)
    tolerance = 1.e-8
    @time qtt = compress_PartialCorrelator_ano_pointwise(Gp; tolerance=tolerance, unfoldingscheme=:interleaved)
    @show rank(qtt)

    # reference
    refval = precompute_all_values_MF(Gp) .- precompute_all_values_MF_noano(Gp)

    slice = (1:2^R, 1:2^R)
    qttval = reshape([qtt(i,j) for j in slice[2] for i in slice[1]], 2^R, 2^R)
    diff = abs.(refval[slice...] - qttval) ./ maximum(abs.(refval))
    @show maximum(diff)

    scfun = x -> log10(abs(x))
    p = plot()
    plot!(p, scfun.(refval[2^(R-1)+1, :]))
    plot!(p, scfun.(qttval[2^(R-1)+1, :]))
    savefig("Gp.png")
end

"""
Test pointwise evaluation of anomalous term.
"""
function test_ano_pointwise(;npt=3, perm_idx=1)
    R = 4
    GF = multipeak_correlator_MF(npt, R; beta=1000.0, nωdisc=10)
    Gp = GF.Gps[perm_idx]
    if !ano_term_required(Gp)
        println("No anomalous term for p=$perm_idx")
        return
    end

    ref_noano = precompute_all_values_MF_noano(Gp)
    ref_all = precompute_all_values_MF(Gp)
    anoval = zeros(ComplexF64, size(ref_all)...)

    for i in CartesianIndices(ref_all)
        anoval[i] = evaluate_ano_with_ωconversion(Gp, Tuple(i)...)
    end

    diff = abs.(ref_all .- (ref_noano .+ anoval))
    @show maximum(diff)
    printstyled("    Norm of anomalous contribution: $(norm(anoval))\n"; color=:cyan)

    if npt==3
        scfun = x -> log10(abs(x))
        heatmap(scfun.(anoval))
        savefig("ano.png")
        heatmap(scfun.(ref_all .- ref_noano))
        savefig("ref.png")
    end
end

function test_compress_PartialCorrelator_ano_pointwise3D(;perm_idx=1)
    R = 5
    GF = multipeak_correlator_MF(4, R; beta=10.0, nωdisc=10)
    Gp = GF.Gps[perm_idx]
    @assert ano_term_required(Gp)
    tolerance = 1.e-8
    @time qtt = compress_PartialCorrelator_ano_pointwise(Gp; tolerance=tolerance, unfoldingscheme=:interleaved)
    @show rank(qtt)

    # reference
    refval = precompute_all_values_MF(Gp) .- precompute_all_values_MF_noano(Gp)

    slice = (1:2^R, 1:2^R, 1:2^R)
    qttval = reshape([qtt(i,j,k) for k in slice[3] for j in slice[2] for i in slice[1]], 2^R, 2^R, 2^R)
    diff = abs.(refval[slice...] - qttval) ./ maximum(abs.(refval))
    @show maximum(diff)

    bos_idx = 2^(R-1)+1
    heatmap(log.(abs.(refval[bos_idx,:,:])))
    savefig("Gpref.png")
    heatmap(log.(abs.(qttval[bos_idx,:,:])))
    savefig("Gp.png")
end

function test_compress_FullCorrelator_pw_plot(;npt=3)
    D = npt-1
    tolerance = 1.e-8
    R = 5
    GF = multipeak_correlator_MF(npt, R; beta=100.0, nωdisc=10)
    t = @elapsed begin
            qtt = compress_FullCorrelator_pointwise(GF; tolerance=tolerance, unfoldingscheme=:interleaved, verbosity=2)
        end
    @show rank(qtt)

    @time refval = precompute_all_values(GF)
    slice = Base.OneTo.(fill(2^R, D))
    maxref = maximum(abs.(refval[slice...]))

    if npt<4
        qttval = zeros(ComplexF64, fill(2^R, D)...)
        for i in Iterators.product(slice...)
            qttval[i...] = qtt(i...)
        end
        diff = abs.(refval[slice...] .- qttval) ./ maxref
        @show maximum(diff)
        printstyled("==== Max error: $(maximum(diff)) for tolerance=$tolerance, R=$R; TIME: $t sec\n"; color=:blue)
    elseif npt==4
        maxdiff = 0.0
        for _ in 1:500
            v = rand(1:2^R, D)
            qv = qtt(v...)
            maxdiff = max(maxdiff, abs(qv - refval[v...]) / maxref)
        end
        printstyled("==== Estimated max error: $maxdiff for tolerance=$tolerance, R=$R; TIME: $t sec\n"; color=:blue)
    end

    if npt==3
        heatmap(log.(abs.(refval)))
        savefig("GFref.png")
        heatmap(log.(abs.(qttval)))
        savefig("GF.png")
        heatmap(log.(abs.(diff)))
        savefig("diff.png")
    end
end
# ========== TESTS & PLOTTING END
