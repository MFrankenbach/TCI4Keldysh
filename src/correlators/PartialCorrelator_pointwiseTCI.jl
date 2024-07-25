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
function compress_reg_PartialCorrelator_pointwise(Gp::PartialCorrelator_reg{D}; qtcikwargs...) where {D}
    R = grid_R(Gp)
    pivot = collect(ntuple(i -> 2^(R-1), D))
    tqtt, _, _ = quanticscrossinterpolate(eltype(Gp.tucker.center), Gp, ntuple(i -> 2^R, D), [pivot]; qtcikwargs...)

    return tqtt
end

"""
QTCI-compress  PartialCorrelator (including frequency rotation and anomalous part) by pointwise evaluation.
"""
function compress_PartialCorrelator_pointwise(Gp::PartialCorrelator_reg{D}; qtcikwargs...) where {D}
    R = grid_R(Gp)
    pivot = collect(ntuple(i -> 2^(R-1), D))
    anev = AnomalousEvaluator(Gp)
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
Obtain qtt for full correlator by pointwise evaluation.
"""
function compress_FullCorrelator_pointwise(GF::FullCorrelator_MF{D}; qtcikwargs...) where {D}
    # check external frequency grids
    R = grid_R(GF)

    ano_terms_required = ano_term_required.(GF.Gps)
    ano_ids = [i for i in eachindex(GF.Gps) if ano_term_required(GF.Gps[i])]

    # collect anomalous term pivots
    pivots = [zeros(Int, D)]
    for i in eachindex(GF.Gps)
        if ano_terms_required[i]
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

    # create anomalous term evaluators
    T = eltype(GF.Gps[1].tucker.center)
    anevs = Vector{AnomalousEvaluator{T,D,D-1}}(undef, length(ano_ids))
    for i in eachindex(ano_ids)
        anevs[i] = AnomalousEvaluator(GF.Gps[ano_ids[i]])
    end
    anoid_to_Gpid = zeros(Int, length(GF.Gps))
    for i in eachindex(ano_ids)
        anoid_to_Gpid[ano_ids[i]] = i
    end

    function _qtcifun(w::Vararg{Int,D})
        ret = zero(ComplexF64)
        for i in eachindex(GF.Gps)
            if ano_terms_required[i]
                # ret += GF.Gps[i](w...) + evaluate_ano_with_ωconversion(GF.Gps[i], w...)
                anev_act = anevs[anoid_to_Gpid[i]]
                ret += GF.Gps[i](w...) + anev_act(w...)
            else
                ret += GF.Gps[i](w...)
            end
        end
        return ret
    end

    tqtt, _, _ = quanticscrossinterpolate(eltype(GF.Gps[1].tucker.center), _qtcifun, ntuple(i -> 2^R, D), pivots; qtcikwargs...)

    return tqtt
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
