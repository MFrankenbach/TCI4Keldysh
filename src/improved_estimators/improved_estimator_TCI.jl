using BenchmarkTools
#=
This file collects functions and structs used for pointwise evaluation of vertex functions.
These are needed to compute vertex functions in qunatics tensor train format in an efficient way.
=#

DEBUG_TCI_KF_RAM() = true

# ========== MATSUBARA

"""
Evaluate self-energy pointwisely by symmetric improved estimator.
(Eq. 108 Lihm et. al.)
"""
struct SigmaEvaluator_MF{D}
    G_QQ::FullCorrEvaluator_MF{ComplexF64, 1, 0}
    G_QF::FullCorrEvaluator_MF{ComplexF64, 1, 0}
    G_FQ::FullCorrEvaluator_MF{ComplexF64, 1, 0}
    G::FullCorrEvaluator_MF{ComplexF64, 1, 0}
    Σ_H::Float64
    ωconvMat::Matrix{Int}
    ωconvOff::Vector{Int}

    function SigmaEvaluator_MF(
        G_QQ_::FullCorrelator_MF{1},
        G_QF_::FullCorrelator_MF{1},
        G_FQ_::FullCorrelator_MF{1},
        G_::FullCorrelator_MF{1},
        Σ_H::Float64,
        ωconvMat::Matrix{Int};
        )

        G_QQ = FullCorrEvaluator_MF(G_QQ_, true; cutoff=1.e-20)
        G_QF = FullCorrEvaluator_MF(G_QF_, true; cutoff=1.e-20)
        G_FQ = FullCorrEvaluator_MF(G_FQ_, true; cutoff=1.e-20)
        G = FullCorrEvaluator_MF(G_, true; cutoff=1.e-20)

        @assert all(sum(abs.(ωconvMat); dims=2) .<= 2) "Only two nonzero elements per row in frequency trafo allowed"

        D = size(ωconvMat, 2)
        Nfer = length(only(G_.ωs_ext))
        return new{D}(G_QQ, G_QF, G_FQ, G, Σ_H, ωconvMat, freq_shift_rot(ωconvMat, div(Nfer,2)))
    end
end

function SigmaEvaluator_MF(PSFpath::String, R::Int, T::Float64, ωconvMat::Matrix{Int}; flavor_idx::Int=1)
    
    # need twice the grid size of the external grids
    ω_fer = MF_grid(T, 2^R, true)
    # TODO: Do required operators depend on flavor_idx here
    G = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_QF = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_FQ = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "Q1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_QQ = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "Q1dag"]; T, flavor_idx=flavor_idx, ωs_ext=(ω_fer,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");

    Adisc_Σ_H = load_Adisc_0pt(PSFpath, "Q12")
    Σ_H = only(Adisc_Σ_H)

    return SigmaEvaluator_MF(G_QQ, G_QF, G_FQ, G, Σ_H, ωconvMat)
end

"""
Frequency shift for frequency transform with one-based indices.
Assume that first external grid is bosonic, others are fermionic and that internal frequencies are all fermionic.
* Nfer : Number of external fermionic frequencies. Internal grids then have size 2*Nfer.
"""
function freq_shift_rot(ωconvMat::Matrix{Int}, Nfer::Int)
    D = size(ωconvMat, 2)
    @assert size(ωconvMat, 1)==D+1
    # trafo maps new -> old   
    corner_new = ntuple(i -> i==1 ? -Nfer : -Nfer + 1, D)
    corner_old = ωconvMat * collect(corner_new)
    @assert all(abs.(corner_old) .<= 2*Nfer - 1) "invalid frequency transformation"
    corner_old_idx = [div(c+2*Nfer, 2) + 1 for c in corner_old]

    idx_new = ones(Int, D)
    idx_old = ωconvMat * idx_new
    return corner_old_idx .- idx_old
end

function (sev::SigmaEvaluator_MF{D})(row::Int, w::Vararg{Int,D}) where {D}
    w_int = dot(sev.ωconvMat[row,:], SA[w...]) + sev.ωconvOff[row]
    # G_QQ .+ Σ_H .- G_QF ./ G .* G_FQ
    return sev.G_QQ(w_int) + sev.Σ_H - (sev.G_QF(w_int) / sev.G(w_int)) * sev.G_FQ(w_int)
end

"""
Evaluate left-sided asymmetric improved estimator of self-energy
"""
function eval_LaIE(sev::SigmaEvaluator_MF{D}, row::Int, w::Vararg{Int,D}) where {D}
    w_int = dot(sev.ωconvMat[row,:], SA[w...]) + sev.ωconvOff[row]
    return sev.G_QF(w_int) / sev.G(w_int)
end

"""
Evaluate right-sided asymmetric improved estimator of self-energy
"""
function eval_RaIE(sev::SigmaEvaluator_MF{D}, row::Int, w::Vararg{Int,D}) where {D}
    w_int = dot(sev.ωconvMat[row,:], SA[w...]) + sev.ωconvOff[row]
    return sev.G_FQ(w_int) / sev.G(w_int)
end

"Construct FullCorrelator_MF objects required for 4-point vertex from PSF data."
function read_GFs_Γcore!(
    GFs::Vector, PSFpath, letter_combinations;
    T::Float64, ωs_ext::NTuple{3, Vector{Float64}}, flavor_idx::Int, ωconvMat::Matrix{Int}
    )
    
    op_labels = ("1", "1dag", "3", "3dag")
    op_labels_symm = ("3", "3dag", "1", "1dag")
    # create correlator objects
    PSFpath_4pt = joinpath(PSFpath, "4pt")
    filelist = readdir(PSFpath_4pt)
    for l in 1:length(GFs)
        letts = letter_combinations[l]
        vprintln("letts: $letts", 2)
        ops = [letts[i]*op_labels[i] for i in 1:4]
        if !any(parse_Ops_to_filename(ops) .== filelist)
            ops = [letts[i]*op_labels_symm[i] for i in 1:4]
        end
        GFs[l] = TCI4Keldysh.FullCorrelator_MF(PSFpath_4pt, ops; T, flavor_idx, ωs_ext, ωconvMat);
        vprintstyled("== Memory usage [GB] of $(l)-th full correlator: $(Base.summarysize(GFs[l]) / 1024^3)\n", 2; color=:blue)
    end
end

function letter_combinations_Γcore()
    letters = ["F", "Q"]
    return kron(kron(letters, letters), kron(letters, letters))
end

function letter_combinations_K2()
    return ["FF", "FQ", "QF", "QQ"]
end

"""
Test accuracy of ΓcoreEvaluator_KF
"""
function test_ΓcoreEvaluator_KF(;R::Int, iK::Int=2, tolerance=1.e-8)

    basepath = "SIAM_u=0.50"
    nz = 4
    PSFpath = joinpath(TCI4Keldysh.datadir(), basepath, "PSF_nz=$(nz)_conn_zavg/")
    channel = "t"
    # read box size
    gamfile = joinpath(TCI4Keldysh.datadir(), basepath, "V_KF_$(channel_translate(channel))", "V_KF_U4.mat") 
    ωs_ext = nothing
    matopen(gamfile) do f
        CFdat = read(f, "CFdat")
        ωs_ext = ntuple(i -> real.(vec(vec(CFdat["ogrid"])[i])), 3)
    end
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    T = TCI4Keldysh.dir_to_T(PSFpath)
    ωmax = maximum(ωs_ext[1])
    (γ, sigmak) = read_broadening_params(basepath; channel=channel)
    broadening_kwargs = TCI4Keldysh.read_broadening_settings(basepath; channel=channel)
    ωconvMat = channel_trafo(channel)
    flavor_idx=1

    # make frequency grid
    D = 3
    ωs_ext = KF_grid(ωmax, R, D)

    # all 16 4-point correlators
    letters = ["F", "Q"]
    letter_combinations = kron(kron(letters, letters), kron(letters, letters))
    op_labels = ("1", "1dag", "3", "3dag")
    op_labels_symm = ("3", "3dag", "1", "1dag")
    is_incoming = (false, true, false, true)

    # create correlator objects
    Ncorrs = length(letter_combinations)
    GFs = Vector{FullCorrelator_KF{D}}(undef, Ncorrs)
    PSFpath_4pt = joinpath(PSFpath, "4pt")
    filelist = readdir(PSFpath_4pt)
    for l in 1:Ncorrs
        letts = letter_combinations[l]
        vprintln("letts: $letts", 2)
        ops = [letts[i]*op_labels[i] for i in 1:4]
        if !any(parse_Ops_to_filename(ops) .== filelist)
            ops = [letts[i]*op_labels_symm[i] for i in 1:4]
        end
        GFs[l] = TCI4Keldysh.FullCorrelator_KF(PSFpath_4pt, ops; T, flavor_idx, ωs_ext, ωconvMat, sigmak=sigmak, γ=γ, broadening_kwargs...);
    end

    # evaluate self-energy
    incoming_trafo = diagm([inc ? -1 : 1 for inc in is_incoming])
    @assert all(sum(abs.(ωconvMat); dims=2) .<= 2) "Only two nonzero elements per row in frequency trafo allowed"
    ωstep = abs(ωs_ext[1][1] - ωs_ext[1][2])
    Σω_grid = KF_grid_fer(2*ωmax, R+1)
    # Σ = calc_Σ_KF_sIE_viaR(PSFpath, Σω_grid; flavor_idx=flavor_idx, T=T, sigmak, γ)
    (Σ_L,Σ_R) = calc_Σ_KF_aIE_viaR(PSFpath, Σω_grid; flavor_idx=flavor_idx, T=T, sigmak, γ, broadening_kwargs...)

    # frequency grid offset for self-energy
    ΣωconvMat = incoming_trafo * ωconvMat
    corner_low = [first(ωs_ext[i]) for i in 1:D]
    corner_idx = ones(Int, D)
    corner_image = ΣωconvMat * corner_low
    idx_image = ΣωconvMat * corner_idx
    desired_idx = [findfirst(w -> abs(w-corner_image[i])<ωstep*0.1, Σω_grid) for i in eachindex(corner_image)]
    ωconvOff = desired_idx .- idx_image

    function sev(row::Int, is_inc::Bool, w::Vararg{Int,3})
        w_int = dot(ΣωconvMat[row,:], SA[w...]) + ωconvOff[row]
        if is_inc
            return Σ_R[w_int,:,:]
        else
            return Σ_L[w_int,:,:]
        end
    end

    cutoff = tolerance*1.e-2
    gev = ΓcoreEvaluator_KF(GFs, iK, sev; cutoff=cutoff)
    gev_ref = ΓcoreEvaluator_KF(GFs, iK, sev; cutoff=1.e-20)

    N = 2^R
    Neval = 2^4
    eval_step = max(div(N, Neval), 1)
    Ncenter_h = 2
    gridslice = vcat(collect(1:eval_step:N), collect(div(N,2)-Ncenter_h+1 : div(N,2)+Ncenter_h))
    Neval += 2*Ncenter_h
    @assert length(gridslice)==Neval

    println("Box extent: ommax=$ωmax, Neval=$Neval, Ncenter_half=$Ncenter_h")

    # reference values
    println("Computing reference...")
    flush(stdout)
    refval = zeros(ComplexF64, ntuple(_->Neval, D))
    Threads.@threads for idx in collect(Iterators.product(ntuple(_->1:Neval,D)...))
        w = ntuple(i -> gridslice[idx[i]], D)
        refval[idx...] = gev_ref(w...)
    end

    # approximate value4
    println("Computing test values...")
    flush(stdout)
    testval = zeros(ComplexF64, ntuple(_->Neval, D))
    Threads.@threads for idx in collect(Iterators.product(ntuple(_->1:Neval,D)...))
        w = ntuple(i -> gridslice[idx[i]], D)
        testval[idx...] = gev(w...)
    end

    maxref = maximum(abs.(refval))
    diffabs = abs.(testval - refval)
    diff = diffabs ./ maxref
    printstyled("Keldysh GammaCoreEvaluator error for tol=$tolerance\n"; color=:blue)
    @show maxref
    @show maximum(diff)
    @show maximum(diffabs)
    open("KFgamcoreEvaluator.txt", "a") do f
        write(f, "R=$R, iK=$iK, tol=$tolerance, maxref=$maxref, MAXERR=$(maximum(diff)) ($(length(testval)) evals)\n")
    end
end

function test_ΓcoreEvaluator(;R::Int, tolerance=1.e-8)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
    channel = "t"
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    T = TCI4Keldysh.dir_to_T(PSFpath)

    # numerically exact evaluator
    gev_ref = ΓcoreEvaluator_MF(PSFpath, R; ωconvMat=ωconvMat, T=T, flavor_idx=1, cutoff=1.e-20)

    cutoff = 0.01 * tolerance
    gev = ΓcoreEvaluator_MF(PSFpath, R; ωconvMat=ωconvMat, T=T, flavor_idx=1, cutoff=cutoff)

    D = 3
    N = 2^R
    Neval = 2^4
    eval_step = max(div(N, Neval), 1)
    Ncenter_h = 2
    gridslice = vcat(collect(1:eval_step:N), collect(div(N,2)-Ncenter_h+1 : div(N,2)+Ncenter_h))
    Neval += 2*Ncenter_h
    @assert length(gridslice)==Neval

    printstyled("Memory usage [GB] of ΓcoreEvaluator_MF (ref): $(Base.summarysize(gev_ref) / (1024^3))\n"; color=:blue)
    printstyled("Memory usage [GB] of ΓcoreEvaluator_MF (approx): $(Base.summarysize(gev) / (1024^3))\n"; color=:blue)

    # reference values
    println("Computing reference...")
    flush(stdout)
    refval = zeros(ComplexF64, ntuple(_->Neval, D))
    Threads.@threads for idx in collect(Iterators.product(ntuple(_->1:Neval,D)...))
        w = ntuple(i -> gridslice[idx[i]], D)
        if rand(Float64)<=1.e-2
            @time gev_ref(w...)
        end
        refval[idx...] = gev_ref(w...)
    end

    # approximate values
    println("Computing test values...")
    flush(stdout)
    testval = zeros(ComplexF64, ntuple(_->Neval, D))
    Threads.@threads for idx in collect(Iterators.product(ntuple(_->1:Neval,D)...))
        w = ntuple(i -> gridslice[idx[i]], D)
        testval[idx...] = gev(w...)
    end

    maxref = maximum(abs.(refval))
    diff = abs.(testval - refval) ./ maxref
    printstyled("Matsubara GammaCoreEvaluator error for tol=$tolerance\n"; color=:blue)
    @show maxref
    @show maximum(diff)
    open("MFgamcoreEvaluator.txt", "a") do f
        write(f, "R=$R, tol=$tolerance, maxref=$maxref, MAXERR=$(maximum(diff)) ($(length(testval)) evals)\n")
    end
end

"""
Determine initial pivots (of TCI) as a sub-grid of a frequency grid of given size
"""
function initpivots_general(gridsize::NTuple{D,Int}, npivot::Int, pivot_step::Int; verbose=false) where {D}
    # grid centre
    centre_ids = ntuple(i -> ifelse(isodd(gridsize[i]), div(gridsize[i],2)+1, div(gridsize[i],2)), D)
    pivot_block = [[centre_ids[i] + pivot_step * (s - div(npivot,2)-1) for s in 1:npivot] for i in 1:D]
    initpivots = Vector{Vector{Int}}()
    for p in Iterators.product(pivot_block...)
        pv = collect(Tuple(p))
        push!(initpivots, [clamp(pv[i], 1, gridsize[i]) for i in 1:D])
    end
    if verbose
        printstyled("==== Using $(length(initpivots)) initial pivots:\n"; color=:blue)
        display(initpivots)
        printstyled("====\n"; color=:blue)
    end
    return initpivots
end

"""
Initial pivots (of TCI) that one can use for vertex interpolation.
"""
function initpivots_Γcore(GFs::Union{Vector{FullCorrelator_MF{D}}, Vector{FullCorrelator_KF{D}}}; npivot::Int=2) where {D}

    pivots = Vector{Int}[]

    # central shell
    ωs_ext = first(GFs).ωs_ext
    centre_ids = ntuple(i -> ifelse(isodd(length(ωs_ext[i])), div(length(ωs_ext[i]),2)+1, div(length(ωs_ext[i]),2)), D)
    centre_block = ntuple(i -> centre_ids[i]-npivot:centre_ids[i]+npivot, D)
    for c in Iterators.product(centre_block...)
        if any([c[ic] in [first(centre_block[ic]), last(centre_block[ic])] for ic in 1:D])
            push!(pivots,  collect(c))
        end
    end

    # find all lines that are rotated onto some coordinate axis in and internal frequency grid of a partial correlator
    #=
    lines = [Vector{Float64}[] for _ in 1:D]
    for GF in GFs
        for Gp in GF.Gps

            w_cen = collect(ntuple(i -> div(length(Gp.tucker.ωs_legs[i]), 2), D))
            if isa(GFs[1], FullCorrelator_MF{D})
                bos_idx = get_bosonic_idx(Gp)
                if !isnothing(bos_idx)
                    zero_idx = findfirst(x -> abs(x)<=1.e-2*Gp.T, Gp.tucker.ωs_legs[bos_idx])
                    w_cen[bos_idx] = zero_idx
                end
            end

            w_ext = Gp.ωconvMat \ (w_cen .- Gp.ωconvOff)
            w_ext_max = length.(GF.ωs_ext)
            for l in 1:D

                w_int_actp = copy(w_cen)
                w_int_actm = copy(w_cen)
                # stay somewhat close to the centre, go in two opposite directions
                w_int_actp[l] += min(div(w_cen[l], 2), 10)
                w_int_actm[l] -= min(div(w_cen[l], 2), 10)
                w_ext_actp = Gp.ωconvMat \ (w_int_actp .- Gp.ωconvOff)
                w_ext_actm = Gp.ωconvMat \ (w_int_actm .- Gp.ωconvOff)
                line = w_ext_actp .- w_ext

                # eliminate invalid entries; only occurs for small ω_ext anyways
                w_ext_actp = [max(1, w) for w in w_ext_actp]
                w_ext_actm = [max(1, w) for w in w_ext_actm]
                w_ext_actp = [min(w_ext_max[l], w_ext_actp[l]) for l in 1:D]
                w_ext_actm = [min(w_ext_max[l], w_ext_actm[l]) for l in 1:D]

                # decide whether to add pivot
                pushline = true
                for line2 in lines[l]
                    if abs(dot(line2, line)) / (norm(line)*norm(line2)) >= 0.9
                        pushline = false
                        break
                    end
                end
                if pushline
                    push!(lines[l], line)
                    push!(pivots, round.(Int, w_ext_actp))
                    push!(pivots, round.(Int, w_ext_actm))
                end
            end
        end
    end
    =#

    if VERBOSITY[]>=2
        printstyled("==== Using $(length(pivots)) initial pivots:\n"; color=:blue)
        # display([p .- [div(length(GFs[1].ωs_ext[i]), 2) for i in 1:D] for p in pivots])
        display(pivots)
        printstyled("====\n"; color=:blue)
    end
    return pivots
end

"""
First row in Fig 13, Lihm et. al.
Return 3*R bit quantics tensor train.

Use BatchEvaluator and CachedFunction. Intended to run on multiple threads.
"""
function Γ_core_TCI_MF_batched(
    PSFpath::String,
    R::Int;
    cache_center::Int=0,
    ωconvMat::Matrix{Int},
    T::Float64,
    flavor_idx::Int=1,
    use_ΣaIE::Bool=true,
    do_check_interpolation::Bool=true,
    npivot::Int=2,
    unfoldingscheme=:interleaved,
    tcikwargs...
)

    kwargs_dict = Dict(tcikwargs)
    cutoff = haskey(Dict(kwargs_dict), :tolerance) ? kwargs_dict[:tolerance]*1.e-2 : 1.e-12
    gev = ΓcoreEvaluator_MF(
        PSFpath,
        R;
        cache_center=cache_center,
        ωconvMat=ωconvMat,
        flavor_idx=flavor_idx,
        T=T,
        cutoff=cutoff
    )

    # create batch evaluator
    gbev = ΓcoreBatchEvaluator_MF(gev; use_ΣaIE=use_ΣaIE, unfoldingscheme=unfoldingscheme)

    GC.gc(true)

    initpivots_ω = initpivots_Γcore([gev.GFevs[i].GF for i in eachindex(gev.GFevs)]; npivot=npivot)
    initpivots = [QuanticsGrids.origcoord_to_quantics(gbev.grid, tuple(iw...)) for iw in initpivots_ω]

    vprintln("Memory usage [GB] of ΓcoreBatchEvaluator_MF: $(Base.summarysize(gbev) / (1024^3))", 2)

    @info "BATCHED"
    t = @elapsed begin
        tt, _, _ = TCI.crossinterpolate2(ComplexF64, gbev, gbev.qf.localdims, initpivots; tcikwargs...)
    end
    qtt = QuanticsTCI.QuanticsTensorCI2{ComplexF64}(tt, gbev.grid, gbev.qf)
    @info "quanticscrossinterpolate time batched (nocache): $t"

    if do_check_interpolation
        Nhalf = 2^(R-1)
        gridmin = max(1, Nhalf-2^5)
        gridmax = min(2^R, Nhalf+2^5)
        grid1D = gridmin:2:gridmax
        grid = collect(Iterators.product(ntuple(_->grid1D,3)...))
        qgrid = [QuanticsGrids.grididx_to_quantics(qtt.grid, g) for g in grid]
        maxerr = check_interpolation(qtt.tci, gbev, qgrid)
        tol = haskey(kwargs_dict, :tolerance) ? kwargs_dict[:tolerance] : :default
        println(" Maximum interpolation error: $maxerr (tol=$tol)")
    end

    return qtt

end


"""
First row in Fig 13, Lihm et. al.
Return 3*R bit quantics tensor train.
* cache_center: if >0, precompute a block of size 2*cache_center along each dimension
and use it to save pointwise evaluations
"""
function Γ_core_TCI_MF(
    PSFpath::String,
    R::Int;
    cache_center::Int=0,
    ωconvMat::Matrix{Int},
    T::Float64,
    flavor_idx::Int=1,
    use_ΣaIE::Bool=false,
    npivot::Int=2,
    qtcikwargs...
)
    if use_ΣaIE
        error("Asymmetric self-energy estimators for non-batched MF vertex NYI!")
    end

    println(">>Starting Γcore calculation")
    flush(stdout)

    # make frequency grid
    D = size(ωconvMat, 2)
    Nhalf = 2^(R-1)
    ωs_ext = MF_npoint_grid(T, Nhalf, D)

    # all 16 4-point correlators
    letter_combinations = letter_combinations_Γcore()
    is_incoming = (false, true, false, true)

    Ncorrs = length(letter_combinations)
    GFs = Vector{FullCorrelator_MF{3}}(undef, Ncorrs)

    read_GFs_Γcore!(
        GFs, PSFpath, letter_combinations;
        T=T, ωs_ext=ωs_ext, ωconvMat=ωconvMat, flavor_idx=flavor_idx
        )

    println(">>Loaded correlators")
    flush(stdout)

    # create full correlator evaluators
    kwargs_dict = Dict(qtcikwargs)
    cutoff = haskey(Dict(kwargs_dict), :tolerance) ? kwargs_dict[:tolerance]*1.e-2 : 1.e-12
    GFevs = Vector{FullCorrEvaluator_MF{ComplexF64, 3, 2}}(undef, Ncorrs)
    for l in 1:Ncorrs
        GFevs[l] = FullCorrEvaluator_MF(GFs[l], true; cutoff=cutoff, tucker_cutoff=10.0*cutoff)
    end

    println(">>Created FullCorrEvaluators")
    flush(stdout)

    # create self-energy evaluator
    incoming_trafo = diagm([inc ? -1 : 1 for inc in is_incoming])
    sev = SigmaEvaluator_MF(PSFpath, R, T, incoming_trafo * ωconvMat; flavor_idx=flavor_idx)

    println(">>Created SelfEnergy evaluators")
    flush(stdout)

    # search initial pivots
    initpivots_ω = initpivots_Γcore([GFevs[i].GF for i in eachindex(GFevs)]; npivot=npivot)

    GC.gc(true)
    if cache_center > 0
        printstyled("-- Preparing cache for core vertex of size ($(2*cache_center))^$D...\n"; color=:cyan)
    # obtain cache values
        cache_center = min(cache_center, 2^(R-1))
        ω_cache_Σ = MF_grid(T, 2*cache_center, true)
        Σ_calc_sIE = calc_Σ_MF_sIE(PSFpath, ω_cache_Σ; flavor_idx=flavor_idx, T=T)
        ωs_ext_cache = MF_npoint_grid(T, cache_center, D)
        cacheval = TCI4Keldysh.compute_Γcore_symmetric_estimator(
            "MF", PSFpath*"4pt/", Σ_calc_sIE; ωs_ext=ωs_ext_cache, T=T, ωconvMat=ωconvMat, flavor_idx=flavor_idx
            )

        # locate cached frequency grids in larger grids
        cache_start = [findfirst(w -> abs(ωs_ext_cache[j][1] - w)<1.e-10, ωs_ext[j]) for j in 1:D]
        cache_end =   [findfirst(w -> abs(ωs_ext_cache[j][end] - w)<1.e-10, ωs_ext[j]) for j in 1:D]
        @assert !any(isnothing.(cache_start))
        @assert !any(isnothing.(cache_end))

        function is_cached(w::NTuple{3,Int})
            return all((w .>= cache_start) .&& (w .<= cache_end))
        end

        println(">>Starting quanticscrossinterpolate (cache)")
        flush(stdout)

        # evaluation with caching
        function eval_Γ_core_cache(w::Vararg{Int,3})
            if is_cached(w)
                w_c = w .- cache_start .+ 1
                return cacheval[w_c...]
            else
                addvals = Vector{ComplexF64}(undef, Ncorrs)
                Threads.@threads for i in 1:Ncorrs
                    addvals[i] = GFevs[i](w...)
                    for il in eachindex(letter_combinations[i])
                        if letter_combinations[i][il]==='F'
                            addvals[i] *= -sev(il, w...)
                        end
                    end
                end
            return sum(addvals)
            end
        end

        report_mem()

        t = @elapsed begin
            qtt, _, _ = quanticscrossinterpolate(ComplexF64, eval_Γ_core_cache, ntuple(i -> 2^R, D), initpivots_ω; qtcikwargs...)
        end
        @info "quanticscrossinterpolate time (cache): $t"
        return qtt
    else

        println(">>Starting quanticscrossinterpolate (nocache)")
        flush(stdout)

        # evaluation without caching
        function eval_Γ_core(w::Vararg{Int,3})
            addvals = Vector{ComplexF64}(undef, Ncorrs)
            Threads.@threads for i in 1:Ncorrs
                addvals[i] = GFevs[i](w...)
                for il in eachindex(letter_combinations[i])
                    if letter_combinations[i][il]==='F'
                        addvals[i] *= -sev(il, w...)
                    end
                end
            end
            return sum(addvals)
        end

        report_mem()

        t = @elapsed begin
            qtt, _, _ = quanticscrossinterpolate(ComplexF64, eval_Γ_core, ntuple(i -> 2^R, D), initpivots_ω; qtcikwargs...)
        end
        report_mem()
        @info "quanticscrossinterpolate time (nocache): $t"
        return qtt
    end

end

# ==== Lower-dim. asymptotic contributions K1r and K2r(')

"""
Second+third row in Fig 13, Lihm et. al., split in 6 terms in a, p, t channels
Performs QTCI on K2(prime) in given channel, in Keldysh. 
"""
function K2_TCI_KF(
    PSFpath,
    R::Int;
    channel::String,
    prime::Bool,
    flavor_idx::Int=1,
    ik::NTuple{3,Int},
    estep::Int=nothing,
    ommax::Float64,
    useFDR::Bool=USE_FDR_SE(),
    qtcikwargs...
)
    basepath = dirname(rstrip(PSFpath, '/'))
    broadening_kwargs = read_all_broadening_params(basepath; channel=channel)
    ωs_ext = KF_grid(ommax, R, 2)
    if !isnothing(estep)
        broadening_kwargs[:estep] = estep
    end
    K2ev = K2Evaluator_KF(
        PSFpath,
        ωs_ext,
        flavor_idx,
        channel,
        prime;
        useFDR=useFDR,
        broadening_kwargs...
        )
    function _f(i::Int,j::Int)
        return K2ev(i,j)[ik...]
    end
    qtt,_,_ = quanticscrossinterpolate(ComplexF64, _f, ntuple(_->2^R,2); qtcikwargs...)
    return qtt
end

"""
Second+third row in Fig 13, Lihm et. al., split in 6 terms in a, p, t channels
Performs QTCI on K2(prime) in given channel, in Matsubara. 
"""
function K2_TCI(
    PSFpath::String,
    R::Int,
    channel::String,
    prime::Bool;
    T::Float64,
    flavor_idx::Int=1,
    qtcikwargs...
)
    # grids
    Nhalf = 2^(R-1)
    ωs_ext = MF_npoint_grid(T, Nhalf, 2)    

    # for treating fat dots
    letter_combinations = letter_combinations_K2()
    op_labels = ["1", "1dag", "3", "3dag"]
    incoming_label = [false, true, false, true]

    # process channel specification and load correlators
    ωconvMat_3p = channel_trafo_K2(channel, prime)
    (i,j) = merged_legs_K2(channel, prime)
    nonij = sort(setdiff(1:4, (i,j)))
    Ncorrs = length(letter_combinations)
    GFs = Vector{FullCorrelator_MF}(undef, Ncorrs)
    is_incoming = (incoming_label[nonij[1]], incoming_label[nonij[2]])
    for (cc, letts) in enumerate(letter_combinations)
        Ops = ["Q$i$j", letts[1] * op_labels[nonij[1]], letts[2] * op_labels[nonij[2]]]
        GFs[cc] = FullCorrelator_MF(PSFpath, Ops; T=T, ωs_ext=ωs_ext, flavor_idx=flavor_idx, ωconvMat=ωconvMat_3p)
    end

    # create full correlator evaluators
    kwargs_dict = Dict(qtcikwargs)
    cutoff = haskey(Dict(kwargs_dict), :tolerance) ? kwargs_dict[:tolerance]*1.e-2 : 1.e-12
    GFevs = Vector{FullCorrEvaluator_MF{ComplexF64, 2, 1}}(undef, Ncorrs)
    for l in 1:Ncorrs
        GFevs[l] = FullCorrEvaluator_MF(GFs[l], true; cutoff=cutoff)
    end


    # self-energy
    incoming_trafo = diagm([1, is_incoming[1] ? -1 : 1, is_incoming[2] ? -1 : 1])
    sev = SigmaEvaluator_MF(PSFpath, R, T, incoming_trafo * ωconvMat_3p; flavor_idx=flavor_idx)

    # compress K2
    function eval_K2(w::Vararg{Int, 2})
        ret = zero(ComplexF64)
        for i in 1:Ncorrs        
            val = GFevs[i](w...)
            for il in eachindex(letter_combinations[i])
                if letter_combinations[i][il]==='F'
                    # il+1 because first index is composite operator and gets no self-energy
                    val *= -sev(il+1, w...)
                end
            end
            ret += val
        end
        return ret
    end

    pivots = [[2^(R-1) + 1, 2^(R-1)]]
    qtt, _, _ = quanticscrossinterpolate(ComplexF64, eval_K2, ntuple(i -> 2^R, 2), pivots; qtcikwargs...);

    return qtt
end

"""
Compute K2 on a given grid.
"""
function precompute_K2r(
        PSFpath::String, flavor_idx::Int, formalism="MF";
        ωs_ext::NTuple{2,Vector{Float64}},
        channel="t",
        prime=false,
        broadening_kwargs_...
        )
    T = dir_to_T(PSFpath)
    ωconvMat = channel_trafo_K2(channel,prime)
    op_labels = Tuple(oplabels_K2(channel,prime))
    basepath = dirname(rstrip(PSFpath, '/'))
    sign = channel_K2_sign(channel, prime)
    if formalism=="MF"

        ωs_Σ = Σ_grid(ωs_ext)
        (ΣL, ΣR) = calc_Σ_MF_aIE(PSFpath, ωs_Σ; flavor_idx=flavor_idx,T=T)
        printstyled("  Compute K2...\n"; color=:blue)
        K2 = compute_K2r_symmetric_estimator(
            "MF",
            PSFpath,
            op_labels,
            ΣR;
            Σ_calcL=ΣL,
            T=T,
            flavor_idx=flavor_idx,
            ωs_ext=ωs_ext,
            ωconvMat=ωconvMat
        )
        return sign*K2

    elseif formalism=="KF"

        (γ, sigmak) = read_broadening_params(basepath)
        broadening_kwargs = read_broadening_settings(basepath)
        broaden_dict = Dict(broadening_kwargs_)
        override_dict!(broaden_dict, broadening_kwargs)
        if !haskey(broadening_kwargs, :estep) && haskey(broaden_dict, :estep)
            broadening_kwargs[:estep] = broaden_dict[:estep]
        end
        γ = if haskey(broaden_dict,:γ)
                broaden_dict[:γ]
            else
                γ
            end
        sigmak = if haskey(broaden_dict,:sigmak)
                broaden_dict[:sigmak]
            else
                sigmak
            end

        # @assert isodd(length(ωs_ext[1]))
        # @assert iseven(length(ωs_ext[2]))
        # Nfer = length(ωs_ext[2])
        # ωmax = ωs_ext[1][end]
        # ωs_Σ = KF_grid_fer_(ωmax, 2*Nfer)
        ωs_Σ = Σ_grid(ωs_ext)
        (ΣL, ΣR) = calc_Σ_KF_aIE(PSFpath, ωs_Σ; flavor_idx=flavor_idx,T=T, γ=γ, sigmak=sigmak, broadening_kwargs...)
        printstyled("  Compute K2...\n"; color=:blue)
        K2 = compute_K2r_symmetric_estimator(
            "KF",
            PSFpath,
            op_labels,
            ΣR;
            Σ_calcL=ΣL,
            T=T,
            flavor_idx=flavor_idx,
            ωs_ext=ωs_ext,
            ωconvMat=ωconvMat,
            γ=γ,
            sigmak=sigmak,
            broadening_kwargs...
        )
        return sign*K2

    else
        error("Invalid formalism $formalism")
    end
end

"""
Compute K1 for given channel on 1D frequency grid, both MF and KF supported.
"""
function precompute_K1r(
    PSFpath::String, flavor_idx::Int, formalism="MF";
    mode=:normal,
    ωs_ext::Vector{Float64},
    channel="t", 
    increase_estep=true,
    broadening_kwargs...
    )

    T = dir_to_T(PSFpath)
    ops = channel_K1_Ops(channel)
    G = if formalism=="MF"
        FullCorrelator_MF(PSFpath, ops; T=T, flavor_idx=flavor_idx, ωs_ext=(ωs_ext,), ωconvMat=reshape([ 1; -1], (2,1)), name="K1$channel");
    elseif formalism=="KF"
        basepath = join(split(rstrip(PSFpath, '/'), "/")[1:end-1], "/")
        (γ, sigmak) = read_broadening_params(basepath; channel=channel)
        broaden_dict = Dict(broadening_kwargs)

        estep = if increase_estep && haskey(broaden_dict, :estep)
                min(500, 4 * broaden_dict[:estep])
            elseif haskey(broaden_dict, :estep)
                broaden_dict[:estep]
            else
                _ESTEP_DEFAULT()
            end
        println("ESTEP in K1: $(estep)")

        # are different broadening parameters requested?
        γ = if haskey(broaden_dict,:γ)
                broaden_dict[:γ]
            else
                γ
            end
        sigmak = if haskey(broaden_dict,:sigmak)
                broaden_dict[:sigmak]
            else
                sigmak
            end

        # actual computation
        FullCorrelator_KF(
            PSFpath, ops;
            T=T, flavor_idx=flavor_idx, ωs_ext=(ωs_ext,), ωconvMat=reshape([ 1; -1], (2,1)), γ=γ, sigmak=sigmak, name="K1$channel",
            broadening_kwargs...);
    end
    sign = channel_K1_sign(channel)
    if mode==:normal
        return sign * precompute_all_values(G)
    elseif mode==:fdt
        return sign * precompute_all_values_FDT(PSFpath, ops, ωs_ext; flavor_idx=flavor_idx, T=T, γ=γ, sigmak=sigmak, broadening_kwargs...)
    else
        error("Invalid mode $mode")
    end
end

"""
QTCI-compress K1 contributions to full vertex; fourth row in Fig 13, Lihm et. al.
Just compute correlator on dense grid and QTCI that.
Works for Matsubara AND Keldysh.
* formalism: For formalism="KF" (Keldysh), compress both Keldysh components in one go
* ωmax: Only relevant for Keldysh. Margins of frequency grid.
"""
function K1_TCI(
    PSFpath::String,
    R::Int;
    formalism::String,
    channel::String,
    T::Float64,
    flavor_idx::Int=1,
    ωmax::Float64=1.0,
    estep=nothing,
    # do_check_interpolation::Bool=true,
    qtcikwargs...
)::Array{<:Union{<:QuanticsTCI.QuanticsTensorCI2, Nothing}}
    ωconvMat = ωconvMat_K1()
    Ops = channel_K1_Ops(channel)
    Nhalf = 2^(R-1)
    initialpivots = [[Nhalf], [Nhalf + div(Nhalf,2)], [Nhalf - div(Nhalf,2)]]
    if formalism=="MF"
        # leave out final frequency
        ωs_ext = MF_grid(T, Nhalf, false)
        GF = FullCorrelator_MF(PSFpath, Ops; flavor_idx=flavor_idx, T=T, ωconvMat=ωconvMat, ωs_ext=(ωs_ext,), name="K1$channel")
        println("==== Computing $(GF.name)...")
        GFval = precompute_all_values(GF)
        GFval .*= channel_K1_sign(channel)
        # check whether component is identically 0
        qtts = Array{Union{QuanticsTCI.QuanticsTensorCI2{eltype(GFval)}, Nothing}}(undef, 1)
        if maximum(abs.(GFval))<1.e-10
            println("    K1 compent is zero!")
            qtts[1]=nothing
            return qtts
        end
        qtt, _, _ = quanticscrossinterpolate(GFval[1:2^R], initialpivots; qtcikwargs...)
        qtts[1]=qtt
        return qtts
    elseif formalism=="KF"
        ωs_ext = KF_grid_bos(ωmax, R)
        # broadening
        basepath = dirname(rstrip(PSFpath, '/'))
        (γ, sigmak) = read_broadening_params(basepath; channel=channel)
        broadening_kwargs = read_broadening_settings(basepath; channel=channel)
        if !isnothing(estep)
            broadening_kwargs[:estep] = estep
        end
        GF = FullCorrelator_KF(PSFpath, Ops; γ=γ, sigmak=sigmak, flavor_idx=flavor_idx, T=T, ωconvMat=ωconvMat, ωs_ext=(ωs_ext,), name="K1$channel", broadening_kwargs...)
        println("==== Computing $(GF.name)...")
        GFval = precompute_all_values(GF)
        GFval .*= channel_K1_sign(channel)
        # all Keldysh components
        qtts = Array{Union{QuanticsTCI.QuanticsTensorCI2{eltype(GFval)}, Nothing}}(undef, 2,2)
        for id in Iterators.product([1,2],[1,2]) 
            if maximum(abs.(GFval[1:2^R,id...]))<1.e-10
                qtts[id...] = nothing
            else
                qtts[id...], _, _ = quanticscrossinterpolate(GFval[1:2^R,id...], initialpivots; qtcikwargs...) 
            end
        end
        return qtts
    end
end

"""
Struct needed to serialize the self-energy evaluating function in Γcore_TCI_KF
TODO: Do this more cleanly / consistently?
"""
struct SigmaEvaluator_KF <: Function
    Σ_R::Array{ComplexF64,3}
    Σ_L::Array{ComplexF64,3}
    ΣωconvMat::Matrix{Int}
    ωconvOff::Vector{Int}
end

"""
Evaluate self-energy with memory buffer to avoid allocations.
"""
function eval_buff!(sev::SigmaEvaluator_KF, ret_buff::MMatrix{2,2,ComplexF64,4}, row::Int, is_inc::Bool, w::Vararg{Int,N}) where {N}
    @views w_int = dot(sev.ΣωconvMat[row,:], SA[w...]) + sev.ωconvOff[row]
    if is_inc
        ret_buff .= view(sev.Σ_R,w_int,:,:)
    else
        ret_buff .= view(sev.Σ_L,w_int,:,:)
    end
end

function (sev::SigmaEvaluator_KF)(row::Int, is_inc::Bool, w::Vararg{Int,N}) where {N}
    w_int = dot(sev.ΣωconvMat[row,:], SA[w...]) + sev.ωconvOff[row]
    if is_inc
        return sev.Σ_R[w_int,:,:]
    else
        return sev.Σ_L[w_int,:,:]
    end
end

"""
Interpolate self-energy onto requested frequency.
"""
function eval_interpol(sev::SigmaEvaluator_KF, row::Int, is_inc::Bool, ws::Vector{Float64}, w::Vararg{Float64,N}) where {N}
    w_int = dot(sev.ΣωconvMat[row,:], SA[w...])    
    idx_up = searchsortedfirst(ws, w_int)
    idx_low = idx_up - 1
    weight = (w_int - ws[idx_low]) / (ws[idx_up] - ws[idx_low])
    val_low = is_inc ? sev.Σ_R[idx_low,:,:] : sev.Σ_L[idx_low,:,:]
    val_up = is_inc ? sev.Σ_R[idx_up,:,:] : sev.Σ_L[idx_up,:,:]
    return weight * val_up + (1-weight) * val_low
end

"""
Struct to evaluate K2(prime) pointwise on 2D frequency grid for a fixed channel.
"""
struct K2Evaluator_KF
    GFevs::Vector{FullCorrEvaluator_KF{2,ComplexF64}}
    prime::Bool
    channel::AbstractString
    ωs_ext::NTuple{2,Vector{Float64}}
    is_incoming::NTuple{2,Bool}
    ωconvMat::Matrix{Int} # 2x3 Matrix
    sign::Int
    sevs::Vector{SigmaEvaluator_KF}

    function K2Evaluator_KF(
        PSFpath::String,
        ωs_ext::NTuple{2,Vector{Float64}},
        flavor_idx::Int,
        channel::String,
        prime::Bool;
        cutoff::Float64=1.e-20,
        useFDR::Bool=USE_FDR_SE(),
        γ::Float64,
        broadening_kwargs...
        )

        ωconvMat = channel_trafo_K2(channel, prime)
        (i,j) = merged_legs_K2(channel, prime)
        nonij = sort(setdiff(1:4, (i,j)))
        incoming_label = [false, true, false, true]
        is_incoming = (incoming_label[nonij[1]], incoming_label[nonij[2]])
        incoming_trafo = diagm([1, is_incoming[1] ? -1 : 1, is_incoming[2] ? -1 : 1])
        T = dir_to_T(PSFpath)

        # prepare self-energies
        sevs = Vector{SigmaEvaluator_KF}(undef, 2)
        ωconvMat_Σ = incoming_trafo * ωconvMat
        for i in 1:2
            trafo_act = reshape(ωconvMat_Σ[i+1,:], (1,2))
            ωs_Σ, ωconvOff_Σ = trafo_grids_offset(ωs_ext, trafo_act)
            @show ωconvMat_Σ
            @show ωconvOff_Σ
            @show trafo_act
            @show (first(only(ωs_Σ)), last(only(ωs_Σ)))
            @show (first.(ωs_ext), last.(ωs_ext))
            (ΣL, ΣR) = if useFDR
                calc_Σ_KF_aIE_viaR(
                    PSFpath, only(ωs_Σ);
                    flavor_idx=flavor_idx,
                    T=T,
                    γ=γ,
                    broadening_kwargs...
                    )
            else
                calc_Σ_KF_aIE(
                    PSFpath, only(ωs_Σ);
                    flavor_idx=flavor_idx,
                    T=T,
                    γ=γ,
                    broadening_kwargs...
                    )
            end
            sev = SigmaEvaluator_KF(
                ΣR,
                ΣL,
                trafo_act,
                ωconvOff_Σ
                )
            sevs[i] = sev
        end

        # create full correlators
        letter_combinations = letter_combinations_K2()
        op_labels = ["1", "1dag", "3", "3dag"]
        Ncorrs = length(letter_combinations)
        GFs = Vector{FullCorrelator_KF{2}}(undef, Ncorrs)
        for (cc, letts) in enumerate(letter_combinations)
            Ops = ["Q$i$j", letts[1] * op_labels[nonij[1]], letts[2] * op_labels[nonij[2]]]
            GFs[cc] = FullCorrelator_KF(
                PSFpath, Ops;
                T=T,
                ωs_ext=ωs_ext,
                flavor_idx=flavor_idx,
                ωconvMat=ωconvMat,
                # like in MuNRG: Qij gets double broadening
                γ=[2γ,γ,γ],
                broadening_kwargs...
                )
        end

        # create full correlator evaluators
        GFevs = Vector{FullCorrEvaluator_KF{2,ComplexF64}}(undef, Ncorrs)
        for l in 1:Ncorrs
            GFevs[l] = FullCorrEvaluator_KF(GFs[l]; cutoff=cutoff)
        end

        return new(
                GFevs,
                prime,
                channel,
                ωs_ext,
                is_incoming,
                ωconvMat,
                channel_K2_sign(channel, prime),
                sevs
                )
    end
end

"""
Evaluate all 2^3 Keldysh components of K2
"""
function (k2ev::K2Evaluator_KF)(w::Vararg{Int,2})
    ret = zeros(ComplexF64, 2,2,2)
    lc = letter_combinations_K2()
    for i in 1:4 # 4 correlators
        val = reshape(k2ev.GFevs[i](w...), 2,2,2)
        for il in 1:2
            if lc[i][il]==='F'
                Σ = -k2ev.sevs[il](1, k2ev.is_incoming[il], w...)
                if k2ev.is_incoming[il]
                    Σ = transpose(Σ)
                end
                val_new = zeros(ComplexF64, 2,2,2)
                for i in 1:2
                    for j in 1:2
                        id_i = ntuple(a -> ifelse(a==il+1, i, Colon()), 3)
                        id_j = ntuple(a -> ifelse(a==il+1, j, Colon()), 3)
                        # @show (i,j,il, id_i, id_j)
                        val_new[id_i...] .+= val[id_j...] * Σ[i,j]
                    end
                end
                val = val_new
            else
                # apply X
                reverse!(val; dims=1+il)
            end
        end
        ret += val        
    end
    return k2ev.sign * ret
end

function test_K2Evaluator_KF()
    base_path = "SIAM_u=0.50"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
    flavor_idx = 1
    ωs_ext = KF_grid(0.5, 4, 2)
    omsz = length.(ωs_ext)
    maxerr = 0.0
    maxerrs = []
    for channel in ["t", "a", "p"]
        broadening_kwargs = read_all_broadening_params(base_path; channel=channel)
        broadening_kwargs[:estep] = 10
        for prime in [true, false]
            K2ref = precompute_K2r(PSFpath, flavor_idx, "KF"; ωs_ext=ωs_ext, channel=channel, prime=prime, broadening_kwargs...)
            maxref = maximum(abs.(K2ref))
            K2ev = K2Evaluator_KF(PSFpath, ωs_ext, flavor_idx, channel, prime; broadening_kwargs...)
            for ic in Iterators.product(Base.OneTo.(omsz)...)
                val = K2ev(ic...)
                refval = K2ref[ic...,:,:,:] 
                maxerr = max(maxerr, norm(val .- refval)/maxref)
            end
            push!(maxerrs, maxerr)
        end
    end
    printstyled("Maxerrs: $maxerrs"; color=:red)
end

function test_ΓEvaluator_KF()
    base_path = "SIAM_u=0.50"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
    flavor_idx = 1
    ωs_ext = KF_grid(0.5, 3, 3)
    omsz = length.(ωs_ext)
    maxerr = 0.0
    maxerrs = []
    iK = 2
    iKtuple = KF_idx(iK,3)

    channel = "t"
    foreign_channels = ("a", "p")
    broadening_kwargs = read_all_broadening_params(base_path; channel=channel)
    broadening_kwargs[:estep] = 5
    gev = ΓEvaluator_KF(
        PSFpath, iK, MultipoleKFCEvaluator;
        channel=channel,
        foreign_channels=foreign_channels,
        flavor_idx=flavor_idx,
        KEV_kwargs = Dict([(:nlevel, 2), (:cutoff, 1.e-10)]),
        ωs_ext=ωs_ext,
        broadening_kwargs...
    )
    # reference
    gev_ref = compute_Γfull_symmetric_estimator(
        "KF", PSFpath;
        T=dir_to_T(PSFpath),
        flavor_idx=flavor_idx,
        ωs_ext=ωs_ext,
        channel=channel,
        broadening_kwargs...
    )
    maxref = maximum(abs.(gev_ref))
    for ic in Iterators.product(Base.OneTo.(omsz)...)
        val = gev(ic...)
        refval = gev_ref[ic..., iKtuple...]
        maxerr = max(maxerr, abs(val - refval)/maxref) 
    end
    push!(maxerrs, maxerr)

    printstyled("Maxerrs: $maxerrs\n"; color=:red)
end

# ==== Lower-dim. asymptotic contributions END

"""
QTCI-compress K2 contributions to full vertex; fourth row in Fig 13, Lihm et. al.
Just compute correlator on dense grid and QTCI that.
Works for Matsubara AND Keldysh.
* formalism: For formalism="KF" (Keldysh), compress both Keldysh components in one go
* ωmax: Only relevant for Keldysh. Margins of frequency grid.
"""
function K2_TCI_precomputed(
    PSFpath::String,
    R::Int;
    formalism::String,
    channel::String,
    prime::Bool,
    T::Float64,
    flavor_idx::Int=1,
    ωmax::Float64=1.0,
    estep=_ESTEP_DEFAULT(),
    # do_check_interpolation::Bool=true,
    qtcikwargs...
)::Array{<:Union{<:QuanticsTCI.QuanticsTensorCI2, Nothing}}
    Nhalf = 2^(R-1)
    # 5x5 block around centre
    initialpivots = vec([[i,j] for i in Nhalf-1:Nhalf+3, j in Nhalf-2:Nhalf+2])
    if formalism=="MF"
        # leave out final frequency
        ωs_ext = MF_npoint_grid(T,Nhalf,2)
        K2 = precompute_K2r(PSFpath, flavor_idx, formalism; ωs_ext=ωs_ext, channel=channel, prime=prime)
        # check whether component is identically 0
        qtts = Array{Union{QuanticsTCI.QuanticsTensorCI2{eltype(K2)}, Nothing}}(undef, 1)
        if maximum(abs.(K2))<1.e-10
            println("    K2 compent is zero!")
            qtts[1]=nothing
            return qtts
        end
        qtt, _, _ = quanticscrossinterpolate(K2[1:2^R,:], initialpivots; qtcikwargs...)
        qtts[1]=qtt
        return qtts
    elseif formalism=="KF"
        ωs_ext = KF_grid(ωmax, R, 2)
        # broadening
        # basepath = dirname(rstrip(PSFpath, '/'))
        # (γ, sigmak) = read_broadening_params(basepath; channel=channel)
        # broadening_kwargs = read_broadening_settings(basepath; channel=channel)
        K2 = precompute_K2r(PSFpath, flavor_idx, formalism; ωs_ext=ωs_ext, channel=channel, prime=prime, estep=estep)
        # all Keldysh components
        qtts = Array{Union{QuanticsTCI.QuanticsTensorCI2{eltype(K2)}, Nothing}}(undef, 2,2,2)
        for id in Iterators.product([1,2],[1,2],[1,2]) 
            if maximum(abs.(K2[:,:,id...]))<1.e-10
                qtts[id...] = nothing
            else
                qtts[id...], _, _ = quanticscrossinterpolate(K2[1:2^R,:,id...], initialpivots; qtcikwargs...) 
            end
        end
        return qtts
    end
end

# ========== MATSUBARA END

# ========== KELDYSH


"""
Name to serialize cache of ΓcoreBatchEvaluator_KF
"""
function gbevcache_filename()
    return "gammacorecacheKF.jld2"
end


"""
Name to serialize ΓcoreBatchEvaluator_KF
"""
function gev_filename()
    return "gammacoreEvKF.jld2"
end

"""
Name to serialize tci in Γcore Keldysh calculation
"""
function tcigammacore_filename()
    return "tcigammacoreKF.jld2"
end

"""
To evaluate Matsubara core vertex pointwise, wrapping the required setup and relevant data.
* sev: callable with signature sev(i::Int, w::Vararg{Int,D}) to evaluate self-energy
on i'th component of transformed frequency w
"""
struct ΓcoreEvaluator_MF{T}
    # GFevs::Vector{FullCorrEvaluator_MF{T,3,2}}
    GFevs::Vector{MFCEvaluator}
    Ncorrs::Int # number of full correlators
    is_incoming::NTuple{4,Bool}
    letter_combinations::Vector{String}
    sev::SigmaEvaluator_MF

    function ΓcoreEvaluator_MF(
        GFs::Vector{FullCorrelator_MF{3}},
        sev,
        is_incoming::NTuple{4,Bool},
        letter_combinations::Vector{String}
        ;
        cutoff::Float64=1.e-20)
        
        @warn "Keyword :cutoff is obsolete"

        # create correlator evaluators
        T = eltype(GFs[1].Gps[1].tucker.legs[1])
        # GFevs = [FullCorrEvaluator_MF(GFs[i], true; cutoff=cutoff) for i in eachindex(GFs)]
        GFevs = [MFCEvaluator(GFs[i]) for i in eachindex(GFs)]

        return new{T}(GFevs,length(GFs), is_incoming, letter_combinations, sev)
    end
end

function ΓcoreEvaluator_MF(
    GFs::Vector{FullCorrelator_MF{3}},
    sev,
    ;
    cutoff::Float64=1.e-20)

    is_incoming = (false, true, false, true)
    letters = ["F", "Q"]
    letter_combinations = kron(kron(letters, letters), kron(letters, letters))

    return ΓcoreEvaluator_MF(GFs, sev, is_incoming, letter_combinations; cutoff=cutoff)
end

function ΓcoreEvaluator_MF(
    PSFpath::String,
    R::Int;
    cache_center::Int=0,
    ωconvMat::Matrix{Int},
    T::Float64,
    flavor_idx::Int=1,
    cutoff::Float64=1.e-20
    )
    # make frequency grid
    D = size(ωconvMat, 2)
    Nhalf = 2^(R-1)
    ωs_ext = MF_npoint_grid(T, Nhalf, D)

    # all 16 4-point correlators
    letter_combinations = letter_combinations_Γcore()
    is_incoming = (false, true, false, true)

    Ncorrs = length(letter_combinations)
    GFs = Vector{FullCorrelator_MF{3}}(undef, Ncorrs)

    read_GFs_Γcore!(
        GFs, PSFpath, letter_combinations;
        T=T, ωs_ext=ωs_ext, ωconvMat=ωconvMat, flavor_idx=flavor_idx
        )

    # create self-energy evaluator
    incoming_trafo = diagm([inc ? -1 : 1 for inc in is_incoming])
    sev = SigmaEvaluator_MF(PSFpath, R, T, incoming_trafo * ωconvMat; flavor_idx=flavor_idx)

    # create core evaluator
    return ΓcoreEvaluator_MF(GFs, sev; cutoff=cutoff)
end

"""
Evaluate Γcore using sIE for self-energy on all legs
"""
function (gev::ΓcoreEvaluator_MF{T})(w::Vararg{Int,3}) where {T}
    addvals = Vector{ComplexF64}(undef, gev.Ncorrs)
    for i in 1:gev.Ncorrs
        addvals[i] = gev.GFevs[i](w...)
        for il in eachindex(gev.letter_combinations[i])
            if gev.letter_combinations[i][il]==='F'
                addvals[i] *= -gev.sev(il, w...)
            end
        end
    end
    return sum(addvals)
end

"""
Evaluate Γcore using aIE for self-energy, right on incoming, left on outgoing
"""
function eval_LR(gev::ΓcoreEvaluator_MF{T}, w::Vararg{Int,3}) where {T}
    addvals = Vector{ComplexF64}(undef, gev.Ncorrs)
    for i in 1:gev.Ncorrs
        addvals[i] = gev.GFevs[i](w...)
        for il in eachindex(gev.letter_combinations[i])
            if gev.letter_combinations[i][il]==='F'
                if gev.is_incoming[il]
                    addvals[i] *= -eval_RaIE(gev.sev, il, w...)
                else
                    addvals[i] *= -eval_LaIE(gev.sev, il, w...)
                end
            end
        end
    end
    return sum(addvals)
end

"""
Evaluate full vertex pointwise.
* K1s/K2s: Precomputed asymptotic classes, stored densely
* K1/2changeMats/offsets: transformation matrices and index offsets for reading
K1 and K2 from foreign channels
"""
struct ΓEvaluator_MF
    core::ΓcoreEvaluator_MF{ComplexF64}
    channel::String
    foreign_channels::Tuple{String,String}
    K2s::NTuple{3, Matrix{ComplexF64}}
    K2offsets::NTuple{2, Vector{Int}}
    K2changeMats::NTuple{2, Matrix{Int}}
    K2primeoffsets::NTuple{2, Vector{Int}}
    K2primechangeMats::NTuple{2, Matrix{Int}}
    K1s::NTuple{3, Vector{ComplexF64}}
    K1offsets::NTuple{2, Vector{Int}}
    K1changeMats::NTuple{2, Matrix{Int}}
    Γbare::Float64

    function ΓEvaluator_MF(
        PSFpath::String,
        R::Int
        ;
        T::Float64,
        flavor_idx::Int,
        channel::String,
        foreign_channels::Tuple{String,String}
    )
        ωconvMat = channel_trafo(channel)
        core = ΓcoreEvaluator_MF(PSFpath, R; T=T, flavor_idx=flavor_idx, ωconvMat=ωconvMat)
        ωs_ext = core.GFevs[1].GF.ωs_ext

        # precompute K2s
        K2s = Vector{Matrix{ComplexF64}}(undef,3)
        K2offsets = Vector{Int}[]
        K2changeMats = Matrix{Int}[]
        K2primeoffsets = Vector{Int}[]
        K2primechangeMats = Matrix{Int}[]
        for (ich, ch) in enumerate([channel, foreign_channels...])
            # compute offsets on common frequency grid for K2 and K2prime
            changeMat = channel_change(channel, ch)[[1,2],:]
            omK2, _ = trafo_grids_offset(ωs_ext, changeMat)
            changeMatprime = channel_change(channel, ch)[[1,3],:]
            omK2p, _ = trafo_grids_offset(ωs_ext, changeMatprime)
                # merge grids
            ωs_extK2 = ntuple(
                i -> ifelse(length(omK2[i])>=length(omK2p[i]), omK2[i], omK2p[i]),
                2
                )
            offset = idx_trafo_offset(ωs_ext, ωs_extK2, changeMat)
            offsetprime = idx_trafo_offset(ωs_ext, ωs_extK2, changeMatprime)
            
            K2 = precompute_K2r(PSFpath,
                flavor_idx,
                "MF"; 
                ωs_ext=ωs_extK2,
                channel=ch,
                prime=false
                )
            K2s[ich] = K2
            if ch!=channel
                push!(K2offsets, offset)
                push!(K2changeMats, changeMat)
                push!(K2primeoffsets, offsetprime)
                push!(K2primechangeMats, changeMatprime)
            end
        end

        # precompute K1s
        K1s = Vector{Vector{ComplexF64}}(undef,3)
        K1offsets = Vector{Int}[]
        K1changeMats = Matrix{Int}[]
        for (ich, ch) in enumerate([channel, foreign_channels...])
            changeMat = reshape(channel_change(channel, ch)[1,:], 1,3)
            ωs_extK1, offset = trafo_grids_offset(ωs_ext, changeMat)
            K1 = precompute_K1r(PSFpath,
            flavor_idx,
            "MF";
            ωs_ext=only(ωs_extK1),
            channel=ch
            )
            K1s[ich] = K1
            if ch!=channel
                push!(K1offsets, offset)
                push!(K1changeMats, changeMat)
            end
        end

        # bare interaction
        Γbare = flavor_idx==2 ? 2.0 * load_Adisc_0pt(PSFpath, "Q12") : 0.0

        return new(
            core,
            channel,
            foreign_channels,
            Tuple(K2s),
            Tuple(K2offsets),
            Tuple(K2changeMats),
            Tuple(K2primeoffsets),
            Tuple(K2primechangeMats),
            Tuple(K1s),
            Tuple(K1offsets),
            Tuple(K1changeMats),
            Γbare
            )
    end
end

"""
Evaluate full vertex on point
"""
function (gev::ΓEvaluator_MF)(w::Vararg{Int,3})
    # core vertex
    ret = eval_LR(gev.core, w...)

    # add K2
        # same channel
    ret += gev.K2s[1][w[1],w[2]]
        # foreign channel
    for i in [1,2]
        ret += gev.K2s[i+1][(gev.K2changeMats[i] * SA[w...] + gev.K2offsets[i])...]
    end

    # add K2prime
        # same channel
    ret += gev.K2s[1][w[1],w[3]]
        # foreign channel
    for i in [1,2]
        ret += gev.K2s[i+1][(gev.K2primechangeMats[i] * SA[w...] + gev.K2primeoffsets[i])...]
    end

    # add K1
    ret += gev.K1s[1][w[1]]
    for i in [1,2]
        ret += gev.K1s[i+1][only(gev.K1changeMats[i] * SA[w...] + gev.K1offsets[i])]
    end
    
    return ret + gev.Γbare
end

function create(::Type{MultipoleKFCEvaluator{3}}, GF::FullCorrelator_KF{3}; kwargs...)
    return MultipoleKFCEvaluator(GF; kwargs...)
end

function create(::Type{MultipoleKFCEvaluator}, GF::FullCorrelator_KF{3}; kwargs...)
    return MultipoleKFCEvaluator(GF; kwargs...)
end

function create(::Type{KFCEvaluator}, GF::FullCorrelator_KF{3}; kwargs...)
    return KFCEvaluator(GF; kwargs...)
end


"""
Structure to evaluate Keldysh core vertex, i.e., wrap the required setup and capture relevant data.
* sev: callable object with signature sev(i::Int, is_incoming::Bool, w::Vararg{Int,D}) to evaluate self-energy
on i'th component of transformed frequency w
"""
struct ΓcoreEvaluator_KF{T,KEV<:AbstractCorrEvaluator_KF{3,ComplexF64}}
# struct ΓcoreEvaluator_KF{T,KEV<:AbstractCorrEvaluator_KF}
    # GFevs::Vector{FullCorrEvaluator_KF{3,T}}
    GFevs::Vector{KEV}
    Ncorrs::Int # number of full correlators
    iK_tuple::NTuple{4,Int} # requested Keldysh idx
    X::Matrix{ComplexF64}
    is_incoming::NTuple{4,Bool}
    letter_combinations::Vector{String}
    sev::SigmaEvaluator_KF
    ωs_ext::NTuple{3,Vector{Float64}}

    function ΓcoreEvaluator_KF(
        GFs::Vector{FullCorrelator_KF{3}},
        iK::Int,
        sev,
        is_incoming::NTuple{4,Bool},
        letter_combinations::Vector{String},
        KEV_::Type=KFCEvaluator
        ; kwargs...)

        # create correlator evaluators
        T = eltype(GFs[1].Gps[1].tucker.legs[1])
        # GFevs = [FullCorrEvaluator_KF(GFs[i]; cutoff=cutoff) for i in eachindex(GFs)]
        GFevs = Vector{KEV_}(undef, length(GFs))
            if DEBUG_TCI_KF_RAM()
                for i in eachindex(GFs)
                    GFevs[i] = create(KEV_, GFs[i]; kwargs...)
                    report_mem(true)
                end
            else
                GFevs = [create(KEV_, GFs[i]; kwargs...) for i in eachindex(GFs)]
            end
        X = get_PauliX()
        iK_tuple = KF_idx(iK,3)

        return new{T,KEV_}(GFevs,length(GFs), iK_tuple, X, is_incoming, letter_combinations, sev, GFs[1].ωs_ext)
    end
end

function ΓcoreEvaluator_KF(
    GFs::Vector{FullCorrelator_KF{3}},
    iK::Int,
    sev,
    KEV_::Type=KFCEvaluator;
    kwargs...)

    is_incoming = (false, true, false, true)
    letters = ["F", "Q"]
    letter_combinations = kron(kron(letters, letters), kron(letters, letters))

    return ΓcoreEvaluator_KF(GFs, iK, sev, is_incoming, letter_combinations, KEV_; kwargs...)
end

function ΓcoreEvaluator_KF(
    PSFpath::String,
    iK::Int,
    ωs_ext::NTuple{3, Vector{Float64}},
    KEV_::Type=KFCEvaluator
    ;
    channel::String,
    flavor_idx::Int,
    KEV_kwargs::Dict=Dict(),
    useFDR::Bool=USE_FDR_SE(),
    broadening_kwargs...
)
    
    ωconvMat = channel_trafo(channel)
    T = dir_to_T(PSFpath)
    # all 16 4-point correlators
    letter_combinations = letter_combinations_Γcore()
    op_labels = ("1", "1dag", "3", "3dag")
    op_labels_symm = ("3", "3dag", "1", "1dag")
    is_incoming = (false, true, false, true)

    # create correlator objects
    Ncorrs = length(letter_combinations)
    GFs = Vector{FullCorrelator_KF{3}}(undef, Ncorrs)
    PSFpath_4pt = joinpath(PSFpath, "4pt")
    filelist = readdir(PSFpath_4pt)
    for l in 1:Ncorrs
        letts = letter_combinations[l]
        vprintln("letts: $letts", 2)
        ops = [letts[i]*op_labels[i] for i in 1:4]
        if !any(parse_Ops_to_filename(ops) .== filelist)
            ops = [letts[i]*op_labels_symm[i] for i in 1:4]
        end
        GFs[l] = TCI4Keldysh.FullCorrelator_KF(PSFpath_4pt, ops; T, flavor_idx, ωs_ext, ωconvMat, broadening_kwargs...)
        if DEBUG_TCI_KF_RAM()
            report_mem()
        end
    end

    # print out memory usage
    if DEBUG_TCI_KF_RAM()
        println("==== FULL CORRELATORS: MEMORY")
        report_mem(true)
        for i in eachindex(GFs)
            @show Base.summarysize(GFs[i]) / 1.e9
        end
        println("==== MEMORY END")
    end

    # self-energy
    incoming_trafo = diagm([inc ? -1 : 1 for inc in is_incoming])
    @assert all(sum(abs.(ωconvMat); dims=2) .<= 2) "Only two nonzero elements per row in frequency trafo allowed"
    Σω_grid = Σ_grid(ωs_ext[1:2])
    (Σ_L,Σ_R) = if useFDR
        calc_Σ_KF_aIE_viaR(PSFpath, Σω_grid; flavor_idx=flavor_idx, T=T, broadening_kwargs...)
    else
        calc_Σ_KF_aIE(PSFpath, Σω_grid; flavor_idx=flavor_idx, T=T, broadening_kwargs...)
    end

    if DEBUG_TCI_KF_RAM()
        println("==== SELF-ENERGIES: MEMORY")
        @show Base.summarysize((Σ_L, Σ_R)) / 1.e9
        println("==== MEMORY END")
    end

    ΣωconvMat = incoming_trafo * ωconvMat
    @show ΣωconvMat
    @show incoming_trafo
    @show ωconvMat
    @show typeof(Σω_grid)
    ωconvOff = idx_trafo_offset(ωs_ext, ntuple(_->Σω_grid, 4), ΣωconvMat)
    sev = SigmaEvaluator_KF(Σ_R, Σ_L, ΣωconvMat, ωconvOff)

    return ΓcoreEvaluator_KF(
        GFs,
        iK,
        sev,
        KEV_;
        KEV_kwargs...
    )
end

"""
* K1: evaluates K1 parametrized in channel
"""
function eval_K1_general(K1, trafo::Matrix{Int}, w::Vararg{Float64,3})
    @warn "WIP; not tested"
    w_int = dot(trafo[1,:], SA[w...])
    return 0.5 * K1(w_int)
end

"""
Generic method to evaluate the core vertex.
* sev: evaluates self-energy, returns 2x2 Complex Matrix
* GFevs: evaluate full correlators, return 16-entry complex vectors (all Keldysh components)
"""
function eval_Γcore_general(GFevs, sev, is_incoming::NTuple{4,Bool}, letter_combinations, w::Vararg{T,3}) where {T<:Number}
    addvals = Vector{Array{ComplexF64,4}}(undef, 16)
    for i in 1:16
        res = reshape(GFevs[i](w...), ntuple(_->2, 4))
        # println("==== GF val no. $i:")
        # display(res)
        val_legs = Vector{Matrix{ComplexF64}}(undef, 4)
        for il in 1:4
            mat = if letter_combinations[i][il]==='F'
                    -sev(il, is_incoming[il], w...)
                else
                    get_PauliX()
                end
            leg = if is_incoming[il]
                    transpose(mat)
                else
                    mat
                end
            val_legs[il] = leg
        end
        res = contract_1D_Kernels_w_Adisc_mp(val_legs, res)
        addvals[i] = res
    end
    return sum(addvals)
end

"""
Compute Keldysh core vertex using TCI for single Keldysh component

* iK: Keldysh component
* sigmak, γ: broadening parameters
* batched: Use batched evaluator
* do_check_interpolation: Check interpolation on small grid at the end, report error
* dump_path: set to save intermediate results every couple of sweeps
* resume_path: set to resume calculation based on intermediate results
* npivot: We have npivot^3 initial pivots
* pivot_step: Pivots are on a npivot^3 equidistant grid with this step size, centered around the frequency grid centre
"""
function Γ_core_TCI_KF(
    PSFpath::String,
    R::Int,
    iK::Int,
    ωmax::Float64;
    sigmak::Vector{Float64}, # broadening
    γ::Float64,
    emin::Float64=1.e-12,
    emax::Float64=1.e4,
    estep::Int=_ESTEP_DEFAULT(),
    # cache_center::Int=0, # maybe later: cache central values
    ωconvMat::Matrix{Int},
    T::Float64,
    flavor_idx::Int=1,
    dump_path=nothing,
    resume_path=nothing,
    batched=true,
    do_check_interpolation=true,
    useFDR::Bool=USE_FDR_SE(),
    npivot::Int=1,
    pivot_step::Int= npivot!=1 ? div(2^R, npivot-1) : 0,
    unfoldingscheme=:interleaved,
    KEV::Type=KFCEvaluator,
    coreEvaluator_kwargs::Dict{Symbol,Any}=Dict{Symbol,Any}(:cutoff=>1.e-6),
    tcikwargs...
    )

    @show KEV
    @show coreEvaluator_kwargs

    broadening_kwargs = Dict([(:emin, emin), (:emax, emax), (:estep, estep)])

    # make frequency grid
    D = size(ωconvMat, 2)
    @assert D==3
    ωs_ext = KF_grid(ωmax, R, D)

    # all 16 4-point correlators
    letters = ["F", "Q"]
    letter_combinations = kron(kron(letters, letters), kron(letters, letters))
    op_labels = ("1", "1dag", "3", "3dag")
    op_labels_symm = ("3", "3dag", "1", "1dag")
    is_incoming = (false, true, false, true)

    # create correlator objects
    Ncorrs = length(letter_combinations)
    GFs = Vector{FullCorrelator_KF{D}}(undef, Ncorrs)
    PSFpath_4pt = joinpath(PSFpath, "4pt")
    filelist = readdir(PSFpath_4pt)
    for l in 1:Ncorrs
        letts = letter_combinations[l]
        vprintln("letts: $letts", 2)
        ops = [letts[i]*op_labels[i] for i in 1:4]
        if !any(parse_Ops_to_filename(ops) .== filelist)
            ops = [letts[i]*op_labels_symm[i] for i in 1:4]
        end
        GFs[l] = TCI4Keldysh.FullCorrelator_KF(PSFpath_4pt, ops; T, flavor_idx, ωs_ext, ωconvMat, sigmak=sigmak, γ=γ, broadening_kwargs...)
        if DEBUG_TCI_KF_RAM()
            report_mem()
        end
    end

    # print out memory usage
    if DEBUG_TCI_KF_RAM()
        report_mem(true)
        println("==== FULL CORRELATORS: MEMORY")
        for i in eachindex(GFs)
            @show Base.summarysize(GFs[i]) / 1.e9
        end
        println("==== MEMORY END")
    end

    # evaluate self-energy
    incoming_trafo = diagm([inc ? -1 : 1 for inc in is_incoming])
    @assert all(sum(abs.(ωconvMat); dims=2) .<= 2) "Only two nonzero elements per row in frequency trafo allowed"
    ωstep = abs(ωs_ext[1][1] - ωs_ext[1][2])
    Σω_grid = KF_grid_fer(2*ωmax, R+1)
    # Σ = calc_Σ_KF_sIE_viaR(PSFpath, Σω_grid; flavor_idx=flavor_idx, T=T, sigmak, γ)
    (Σ_L,Σ_R) = if useFDR
        calc_Σ_KF_aIE_viaR(PSFpath, Σω_grid; flavor_idx=flavor_idx, T=T, sigmak, γ, broadening_kwargs...)
    else
        calc_Σ_KF_aIE(PSFpath, Σω_grid; flavor_idx=flavor_idx, T=T, sigmak, γ, broadening_kwargs...)
    end

    if DEBUG_TCI_KF_RAM()
        println("==== SELF-ENERGIES: MEMORY")
        @show Base.summarysize((Σ_L, Σ_R)) / 1.e9
        println("==== MEMORY END")
    end

    # frequency grid offset for self-energy
    ΣωconvMat = incoming_trafo * ωconvMat
    corner_low = [first(ωs_ext[i]) for i in 1:D]
    corner_idx = ones(Int, D)
    corner_image = ΣωconvMat * corner_low
    idx_image = ΣωconvMat * corner_idx
    desired_idx = [findfirst(w -> abs(w-corner_image[i])<ωstep*0.1, Σω_grid) for i in eachindex(corner_image)]
    ωconvOff = desired_idx .- idx_image

    sev = SigmaEvaluator_KF(Σ_R, Σ_L, ΣωconvMat, ωconvOff)

    gev = ΓcoreEvaluator_KF(GFs, iK, sev, KEV; coreEvaluator_kwargs...)
    if KEV==MultipoleKFCEvaluator{3}
        GFs = nothing
    end
    return Γ_core_TCI_KF(
        gev,
        ωs_ext,
        R;
        dump_path=dump_path,
        resume_path=resume_path,
        batched=batched,
        do_check_interpolation=do_check_interpolation,
        npivot=npivot,
        pivot_step=pivot_step,
        unfoldingscheme=unfoldingscheme,
        tcikwargs...
    )
end

"""
To introduce function barrier and allow for garbage collection
"""
function Γ_core_TCI_KF(
    gev::ΓcoreEvaluator_KF,
    ωs_ext::NTuple{3,Vector{Float64}},
    R::Int;
    dump_path=nothing,
    resume_path=nothing,
    batched=true,
    do_check_interpolation=true,
    npivot::Int=1,
    pivot_step::Int= npivot!=1 ? div(2^R, npivot-1) : 0,
    unfoldingscheme=:interleaved,
    tcikwargs...
    )

    kwargs_dict = Dict(tcikwargs)
    tolerance = haskey(kwargs_dict, :tolerance) ? kwargs_dict[:tolerance] : 1.e-8

    println("==== CORE EVALUATOR: MEMORY")
    report_mem(true)
    @show Base.summarysize(gev) / 1.e9
    println("==== MEMORY END")

    # determine initial pivots
    D = 3
    tcigridsize = ntuple(i -> 2^TCI4Keldysh.grid_R(length(ωs_ext[i])) + 1,D)
    initpivots_ω = initpivots_general(tcigridsize, npivot, pivot_step; verbose=true)

    if batched && isnothing(resume_path)
        gbev = ΓcoreBatchEvaluator_KF(gev; unfoldingscheme=unfoldingscheme)
        initpivots = [QuanticsGrids.origcoord_to_quantics(gbev.grid, tuple(iw...)) for iw in initpivots_ω]
        t = @elapsed begin
            tci, _, _ = TCI.crossinterpolate2(ComplexF64, gbev, gbev.qf.localdims, initpivots; tcikwargs...)
        end
        if DEBUG_TCI_KF_RAM()
            println("==== TT: MEMORY")
            report_mem()
            @show Base.summarysize(tci) / 1.e9
            println("==== MEMORY END")
        end
        qtt = QuanticsTCI.QuanticsTensorCI2{ComplexF64}(tci, gbev.grid, gbev.qf)

        if do_check_interpolation
            Nhalf = 2^(R-1)
            gridmin = max(1, Nhalf-2^5)
            gridmax = min(2^R, Nhalf+2^5)
            grid1D = gridmin:2:gridmax
            grid = collect(Iterators.product(ntuple(_->grid1D,3)...))
            qgrid = [QuanticsGrids.grididx_to_quantics(qtt.grid, g) for g in grid]
            maxerr = check_interpolation(qtt.tci, gbev, qgrid)
            tol = haskey(kwargs_dict, :tolerance) ? kwargs_dict[:tolerance] : :default
            println(" Maximum interpolation error: $maxerr (tol=$tol)")
        end


    elseif batched && !isnothing(resume_path)
        # load cached values
        cache = load(joinpath(resume_path, gbevcache_filename()))[jld2_to_dictkey(gbevcache_filename())]
        # load BatchEvaluator
        gev = load(joinpath(resume_path, gev_filename()))[jld2_to_dictkey(gev_filename())]
        gbev = ΓcoreBatchEvaluator_KF(gev; cache=cache, unfoldingscheme=unfoldingscheme)

        # it is enough to save ΓcoreEvaluator_KF
        if !isnothing(dump_path)
            file = File(format"JLD2", joinpath(dump_path, gev_filename()))
            save(file, jld2_to_dictkey(gev_filename()), gev)
        end

        initpivots = [QuanticsGrids.origcoord_to_quantics(gbev.grid, tuple(iw...)) for iw in initpivots_ω]

        # create/load tensor train
        tci = load(joinpath(resume_path, tcigammacore_filename()))[jld2_to_dictkey(tcigammacore_filename())]
        println("Loaded TCI with rank $(TCI.rank(tci))")

        # set tci kwargs related to global pivots, if not already specified
        if !haskey(kwargs_dict, :maxnglobalpivot)
            kwargs_dict[:maxnglobalpivot]=10
        end
        if !haskey(kwargs_dict, :nsearchglobalpivot)
            kwargs_dict[:nsearchglobalpivot]=100
        end
        if !haskey(kwargs_dict, :tolmarginglobalsearch)
            kwargs_dict[:tolmarginglobalsearch]=3.0
        end

        maxiter = haskey(kwargs_dict, :maxiter) ? kwargs_dict[:maxiter] : 80
        # save result every ncheckpoint sweeps if dump_path is given
        ncheckpoint = isnothing(dump_path) ? maxiter : 3  
        converged = false
        if !isnothing(dump_path)
            kwargs_dict[:maxiter] = ncheckpoint
        else
            kwargs_dict[:maxiter] = maxiter
        end

        if DEBUG_TCI_KF_RAM()
            println("==== TT: MEMORY")
            report_mem()
            @show Base.summarysize(tci) / 1.e9
            println("==== MEMORY END")
        end
        # run
        t = @elapsed begin
            for icheckpoint in 1:Int(ceil(maxiter/ncheckpoint))
                ranks, errors = TCI.optimize!(tci, gbev; kwargs_dict...)
                println("  After $((icheckpoint-1)*ncheckpoint + length(ranks)) sweeps: ranks=$ranks, errors=$errors")
                if _tciconverged(ranks, errors, tolerance, 3)
                    converged = true
                    println(" ==== CONVERGED")
                    break
                elseif !isnothing(dump_path)
                    # save checkpoint
                    filetci = File(format"JLD2", joinpath(dump_path, tcigammacore_filename()))
                    save(filetci, jld2_to_dictkey(tcigammacore_filename()), tci)
                    filecache = File(format"JLD2", joinpath(dump_path, gbevcache_filename()))
                    save(filecache, jld2_to_dictkey(gbevcache_filename()), gbev.qf.cache)
                end
            end
        end
        if !converged
            @warn "TCI did not converge within $maxiter sweeps!"
        end
        qtt = QuanticsTCI.QuanticsTensorCI2{ComplexF64}(tci, gbev.grid, gbev.qf)

        if do_check_interpolation
            Nhalf = 2^(R-1)
            gridmin = max(1, Nhalf-2^5)
            gridmax = min(2^R, Nhalf+2^5)
            grid1D = gridmin:2:gridmax
            grid = collect(Iterators.product(ntuple(_->grid1D,3)...))
            qgrid = [QuanticsGrids.grididx_to_quantics(qtt.grid, g) for g in grid]
            maxerr = check_interpolation(qtt.tci, gbev, qgrid)
            tol = haskey(kwargs_dict, :tolerance) ? kwargs_dict[:tolerance] : :default
            println(" Maximum interpolation error: $maxerr (tol=$tol)")
        end

    else
        if !isnothing(dump_path) || !isnothing(resume_path)
            error("Checkpoints only implemented for batched Γcore in Keldysh")
        end
        t = @elapsed begin
            # quanticscrossinterpolate automatically does caching
            qtt, _, _ = quanticscrossinterpolate(ComplexF64, gev, ntuple(i -> 2^R, D), initpivots_ω; unfoldingscheme=unfoldingscheme, tcikwargs...)
        end

        if do_check_interpolation
            Nhalf = 2^(R-1)
            gridmin = max(1, Nhalf-2^5)
            gridmax = min(2^R, Nhalf+2^5)
            grid1D = gridmin:2:gridmax
            grid = collect(Iterators.product(ntuple(_->grid1D,3)...))
            qgrid = [QuanticsGrids.grididx_to_quantics(qtt.grid, g) for g in grid]
            maxerr = check_interpolation(qtt, gev, grid)
            tol = haskey(kwargs_dict, :tolerance) ? kwargs_dict[:tolerance] : :default
            println(" Maximum interpolation error: $maxerr (tol=$tol)")
        end

    end

    @info "quanticscrossinterpolate time (nocache, batched): $t"
    return qtt
    
end


function test_eval_Γcore_general()
    base_path = "SIAM_u=0.50"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
    # base_path = "unittest_PSF"
    # PSFpath = joinpath(TCI4Keldysh.datadir(), "unittest_PSF/PSF")
    flavor_idx = 2
    # all Keldysh components will be tested
    iK = 2
    channel = "p"
    broadening_kwargs = read_all_broadening_params(base_path; channel=channel)
    broadening_kwargs[:estep]=100
    # reference
    ωs_ext_ref = KF_grid(0.25, 3, 3)
    Vref = compute_Γcore_symmetric_estimator(
        "KF",
        PSFpath,
        ωs_ext_ref;
        flavor_idx,
        channel,
        broadening_kwargs...
    )

    # evaluator
    ωs_ext = KF_grid(1.11 * 0.25, 9, 3)
    gev = ΓcoreEvaluator_KF(
        PSFpath,
        iK,
        ωs_ext,
        MultipoleKFCEvaluator;
        channel=channel,
        flavor_idx=flavor_idx,
        KEV_kwargs=Dict([(:nlevel, 2), (:cutoff, 1.e-6)]),
        broadening_kwargs...
    )

    gpgrids = Gp_grids(ωs_ext, channel_trafo(channel))

    GFevs = [((w1,w2,w3) -> eval_interpol(G, gpgrids, w1,w2,w3)) for G in gev.GFevs]

    omsig = Σ_grid(ωs_ext[1:2])

    function sev_(il::Int, inc::Bool, w::Vararg{Float64,3})
        return eval_interpol(gev.sev, il, inc, omsig, w...)
    end

    # test
    maxref = maximum(abs.(Vref))
    errs = Float64[]
    for iw in Iterators.product(Base.OneTo.(length.(ωs_ext_ref))...)
        ref = Vref[iw...,:,:,:,:]
        w = ntuple(i -> ωs_ext_ref[i][iw[i]], 3)
        val = eval_Γcore_general(
            GFevs,
            sev_,
            gev.is_incoming,
            gev.letter_combinations,
            w...
            )
        push!(errs, norm(ref .- val)/maxref)
    end
    @show maxref
    @show maximum(errs)
end


function compare_precomputed_eval_Γcore_general()
    base_path = "SIAM_u=0.50"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
    flavor_idx = 1
    # reference
    V_ref = h5read(joinpath("cluster_output/V_KF_conventional/V_KF_p_R=8.h5"), "V_KF")
    om1 = h5read(joinpath("cluster_output/V_KF_conventional/V_KF_p_R=8.h5"), "omgrid1")
    om2 = h5read(joinpath("cluster_output/V_KF_conventional/V_KF_p_R=8.h5"), "omgrid2")
    om3 = h5read(joinpath("cluster_output/V_KF_conventional/V_KF_p_R=8.h5"), "omgrid3")
    ωs_ext = KF_grid(1.11 * om1[end], 9, 3)
    iK = 2
    channel = "p"
    broadening_kwargs = read_all_broadening_params(base_path; channel=channel)
    broadening_kwargs[:estep]=200

    gev = ΓcoreEvaluator_KF(
        PSFpath,
        iK,
        ωs_ext,
        MultipoleKFCEvaluator;
        channel=channel,
        flavor_idx=flavor_idx,
        KEV_kwargs=Dict([(:nlevel, 2), (:cutoff, 1.e-6)]),
        broadening_kwargs...
    )

    gpgrids = Gp_grids(ωs_ext, channel_trafo(channel))

    GFevs = [((w1,w2,w3) -> eval_interpol(G, gpgrids, w1,w2,w3)) for G in gev.GFevs]

    omsig = Σ_grid(ωs_ext[1:2])

    function sev_(il::Int, inc::Bool, w::Vararg{Float64,3})
        return eval_interpol(gev.sev, il, inc, omsig, w...)
    end

    maxref = maximum(abs.(V_ref))
    omids = Base.OneTo.(length.((om1,om2,om3)))

    abserrs = Float64[]
    normederrs = Float64[]
    for ic in Iterators.product(ntuple(i->omids[i][2:end-1], 3)...)
        if rand()>1.e-4
            continue
        end
        w = (om1[ic[1]], om2[ic[2]], om3[ic[3]])
        val = eval_Γcore_general(
            GFevs,
            sev_,
            gev.is_incoming,
            gev.letter_combinations,
            w...
            )
        ref = V_ref[ic...,:,:,:,:]
        push!(abserrs, norm(ref.-val))
        push!(normederrs, norm(ref.-val)/maxref)
    end
end

"""
Evaluate `gev` on all Keldysh components
"""
function evaluate_all_iK(gev::ΓcoreEvaluator_KF{T}, w::Vararg{Int,3}) where {T}
    return eval_Γcore_general(gev.GFevs, gev.sev, gev.is_incoming, gev.letter_combinations, w...)
end

"""
Evaluate `gev` on all Keldysh components
"""
function evaluate_all_iK(gev::ΓcoreEvaluator_KF{T}, w::Vararg{Float64,3}) where {T}
    __f(ids::Vararg{Int,3}) = evaluate_all_iK(gev, ids...)
    return eval_interpol(zeros(ComplexF64,2,2,2,2), __f, gev.ωs_ext, w...)
end

function (gev::ΓcoreEvaluator_KF{T,MultipoleKFCEvaluator{3}})(w::Vararg{Int,3}) where {T}
    return eval_buff!(gev, w...)
end

"""
Evaluate Γcore (using sIE or aIE for self-energy, depending on sev function).
Memory buffers to reduce allocations.
"""
function eval_buff!(gev::ΓcoreEvaluator_KF{T,MultipoleKFCEvaluator{3}},w::Vararg{Int,3}) where {T}
    result = zero(T)
    val_legs = MArray{Tuple{4,2},ComplexF64,2,8}(undef)
    ret_buff = MVector{16,ComplexF64}(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    # restensor = MArray{Tuple{2,2,2,2},ComplexF64,4,16}(undef)
    retarded_buff = MVector{4,ComplexF64}(0,0,0,0)
    idx_int = MVector{3,Int}(0,0,0)
    mat = MMatrix{2,2,ComplexF64,4}(undef)
    for i in 1:gev.Ncorrs
        # first all Keldysh indices
        eval_buff!(gev.GFevs[i], ret_buff, retarded_buff, idx_int, w...)
        # contract 
        for il in eachindex(gev.is_incoming)
            if gev.letter_combinations[i][il]==='F'
                eval_buff!(gev.sev, mat, il, gev.is_incoming[il], w...)
                @. mat = -mat
            else
                # TODO: make gev.X an MMatrix
                mat .= gev.X
            end
            val_legs[il,:] .= if gev.is_incoming[il]
                    view(mat,:, gev.iK_tuple[il])
                else
                    view(mat,gev.iK_tuple[il], :)
                end
        end
        interm = zero(T)
        for k4 in 1:2
            it4 = zero(T)
            for k3 in 1:2
                it3 = zero(T)
                for k2 in 1:2
                    it2 = zero(T)
                    for k1 in 1:2
                        it2 += ret_buff[k1 + 2*(k2-1) + 4*(k3-1) + 8*(k4-1)] * val_legs[1,k1]
                    end
                    it3 += it2 * val_legs[2,k2]
                end
                it4 += it3 * val_legs[3,k3]
            end
            interm += it4 * val_legs[4,k4]
        end
        result += interm
        ret_buff .= zero(ComplexF64)
    end
    return result
end

function eval_nobuff(gev::ΓcoreEvaluator_KF{T,MultipoleKFCEvaluator{3}},w::Vararg{Int,3}) where {T}
    # TODO: can we avoid copy-paste from (gev::ΓcoreEvaluator_KF{T,KEV})(w::Vararg{Int,3}) where {T,KEV<:AbstractCorrEvaluator_KF{3,ComplexF64}}?
    addvals = Vector{T}(undef, gev.Ncorrs)
    val_legs = Vector{Vector{T}}(undef, length(gev.is_incoming))
    for i in 1:gev.Ncorrs
        # first all Keldysh indices
        res = reshape(gev.GFevs[i](w...), ntuple(_->2, 4))
        for il in eachindex(gev.is_incoming)
            mat = if gev.letter_combinations[i][il]==='F'
                    -gev.sev(il, gev.is_incoming[il], w...)
                else
                    gev.X
                end
            leg = if gev.is_incoming[il]
                    vec(mat[:, gev.iK_tuple[il]])
                else
                    vec(mat[gev.iK_tuple[il], :])
                end
            val_legs[il] = leg
        end
        for d in 1:4
            res = res[1,ntuple(_->Colon(),4-d)...].*val_legs[d][1] .+ res[2,ntuple(_->Colon(),4-d)...].*val_legs[d][2]
        end
        addvals[i] = res
    end
    return sum(addvals)
end

"""
Evaluate Γcore (using sIE or aIE for self-energy, depending on sev function)
"""
function (gev::ΓcoreEvaluator_KF{T,KEV})(w::Vararg{Int,3}) where {T,KEV<:AbstractCorrEvaluator_KF{3,ComplexF64}}
    addvals = Vector{T}(undef, gev.Ncorrs)
    val_legs = Vector{Vector{T}}(undef, length(gev.is_incoming))
    for i in 1:gev.Ncorrs
        # first all Keldysh indices
        res = reshape(gev.GFevs[i](w...), ntuple(_->2, 4))
        for il in eachindex(gev.is_incoming)
            mat = if gev.letter_combinations[i][il]==='F'
                    -gev.sev(il, gev.is_incoming[il], w...)
                else
                    gev.X
                end
            leg = if gev.is_incoming[il]
                    vec(mat[:, gev.iK_tuple[il]])
                else
                    vec(mat[gev.iK_tuple[il], :])
                end
            val_legs[il] = leg
        end
        for d in 1:4
            res = res[1,ntuple(_->Colon(),4-d)...].*val_legs[d][1] .+ res[2,ntuple(_->Colon(),4-d)...].*val_legs[d][2]
        end
        addvals[i] = res
    end
    return sum(addvals)
end

"""
Evaluate full Keldysh vertex in given channel pointwise. In contrast to the Matsubara version,
do not precompute K2 as this would take up significant memory.
"""
struct ΓEvaluator_KF
    core::ΓcoreEvaluator_KF{ComplexF64}
    iK::Int
    channel::String
    foreign_channels::Tuple{String,String}
    K2s::NTuple{6, K2Evaluator_KF}
    K2offsets::NTuple{2, Vector{Int}}
    K2changeMats::NTuple{2, Matrix{Int}}
    K2primeoffsets::NTuple{2, Vector{Int}}
    K2primechangeMats::NTuple{2, Matrix{Int}}
    iKK2::NTuple{6, NTuple{3,Int}}
    K1s::NTuple{3, Array{ComplexF64, 3}}
    K1offsets::NTuple{2, Vector{Int}}
    K1changeMats::NTuple{2, Matrix{Int}}
    K1grids::NTuple{3, Vector{Float64}}
    iKK1::NTuple{3, NTuple{2,Int}}
    Γbare::Float64

    function ΓEvaluator_KF(
        PSFpath::String,
        iK::Int,
        KEV_::Type=KFCEvaluator
        ;
        ωs_ext::NTuple{3, Vector{Float64}},
        flavor_idx::Int,
        channel::String,
        foreign_channels::Tuple{String,String},
        KEV_kwargs::Dict=Dict(),
        K2cutoff::Float64=1.e-20,
        useFDR::Bool=USE_FDR_SE(),
        broadening_kwargs...
    )
        iK_tuple = KF_idx(iK, 3)
        core = ΓcoreEvaluator_KF(
            PSFpath, iK, ωs_ext, KEV_; 
            KEV_kwargs=KEV_kwargs,
            flavor_idx=flavor_idx,
            channel=channel,
            useFDR=useFDR,
            broadening_kwargs...
            )

        if DEBUG_TCI_KF_RAM()
            println("---- core evaluator done")
            report_mem(true)
        end

        # K2
        K2s = Vector{K2Evaluator_KF}(undef, 6)
        K2offsets = Vector{Vector{Int}}(undef, 2)
        K2changemats = Vector{Matrix{Int}}(undef, 2)
        K2primeoffsets = Vector{Vector{Int}}(undef, 2)
        K2primechangemats = Vector{Matrix{Int}}(undef, 2)
        iKK2 = NTuple{3,Int}[]
        ic = 1
        for ch in [channel, foreign_channels...]
            # matrices/offsets
            changeMat = channel_change(channel, ch)[[1,2],:]
            omK2, K2off = trafo_grids_offset(ωs_ext, changeMat)
            changeMatprime = channel_change(channel, ch)[[1,3],:]
            omK2p, K2poff = trafo_grids_offset(ωs_ext, changeMatprime)
            if ic>1 # foreign channel
                K2offsets[ic-1] = K2off
                K2primeoffsets[ic-1] = K2poff
                K2changemats[ic-1] = changeMat
                K2primechangemats[ic-1] = changeMatprime
            end
            # K2r and K2r'
            K2 = K2Evaluator_KF(
                PSFpath, omK2, flavor_idx, ch, false;
                useFDR=useFDR,
                cutoff=K2cutoff,
                broadening_kwargs...
            )
            
            if DEBUG_TCI_KF_RAM()
                println("---- K2$(ch) (noprime) done")
                report_mem(true)
            end

            K2p = K2Evaluator_KF(
                PSFpath, omK2p, flavor_idx, ch, true;
                useFDR=useFDR,
                cutoff=K2cutoff,
                broadening_kwargs...
            )

            if DEBUG_TCI_KF_RAM()
                println("---- K2$(ch) (prime) done")
                report_mem(true)
            end

            K2s[2*ic-1] = K2
            K2s[2*ic] = K2p
            # iK
            push!(iKK2, merge_iK_K2(iK_tuple, ch, false))
            push!(iKK2, merge_iK_K2(iK_tuple, ch, true))
            ic += 1
        end

        if DEBUG_TCI_KF_RAM()
            println("---- K2 done")
            report_mem(true)
        end

        # precompute K1s
        K1s = Vector{Array{ComplexF64,3}}(undef,3)
        K1offsets = Vector{Int}[]
        K1changeMats = Matrix{Int}[]
        iKK1 = NTuple{2,Int}[]
        K1grids = Vector{Float64}[]
        ic = 1
        for ch in [channel, foreign_channels...]
            changeMat = reshape(channel_change(channel, ch)[1,:], 1,3)
            ωs_extK1, offset = trafo_grids_offset(ωs_ext, changeMat)
            K1 = precompute_K1r(
                PSFpath,
                flavor_idx,
                "KF";
                ωs_ext=only(ωs_extK1),
                channel=ch,
                broadening_kwargs...
                )
            K1s[ic] = K1
            push!(K1grids, only(ωs_extK1))
            if ic>1
                push!(K1offsets, offset)
                push!(K1changeMats, changeMat)
            end
            # iK
            push!(iKK1, merge_iK_K1(iK_tuple, ch))
            ic += 1
        end

        if DEBUG_TCI_KF_RAM()
            println("---- K1 done")
            report_mem(true)
        end

        # bare interaction
        Γbare = Γbare_KF(PSFpath, flavor_idx)[iK_tuple...]

        return new(
            core,
            iK,
            channel,
            foreign_channels,
            Tuple(K2s),
            Tuple(K2offsets),
            Tuple(K2changemats),
            Tuple(K2primeoffsets),
            Tuple(K2primechangemats),
            Tuple(iKK2),
            Tuple(K1s),
            Tuple(K1offsets),
            Tuple(K1changeMats),
            Tuple(K1grids),
            Tuple(iKK1),
            Γbare
            )
    end
end

function ΓEvaluator_KF(
    PSFpath::String,
    iK::Int,
    R::Int,
    KEV_::Type=KFCEvaluator;
    ommax::Float64,
    kwargs...
)
    ωs_ext = KF_grid(ommax, R, 3)
    return ΓEvaluator_KF(
        PSFpath,
        iK,
        KEV_;
        ωs_ext=ωs_ext,
        kwargs...
    )
end

"""
Evaluate full vertex on point.
"""
function (gev::ΓEvaluator_KF)(w::Vararg{Int,3})
    # core vertex
    ret = gev.core(w...)

    # K2
    K2fac = 1/sqrt(2)
    K2val = zero(ComplexF64)
        # same channel
    K2val += gev.K2s[1](w[1],w[2])[gev.iKK2[1]...]
        # foreign channel
    for i in 1:2
        K2val += gev.K2s[2*i+1]((gev.K2changeMats[i] * SA[w...] + gev.K2offsets[i])...)[gev.iKK2[2*i+1]...]
    end

    # K2prime
        # same channel
    K2val += gev.K2s[2](w[1],w[3])[gev.iKK2[2]...]
        # foreign channel
    for i in 1:2
        K2val += gev.K2s[2*i+2]((gev.K2primechangeMats[i] * SA[w...] + gev.K2primeoffsets[i])...)[gev.iKK2[2*i+2]...]
    end
    ret += K2fac * K2val

    # K1
    K1fac = 0.5
    K1val = zero(ComplexF64)
    K1val += gev.K1s[1][w[1], gev.iKK1[1]...]
    for i in 1:2
        K1val += gev.K1s[i+1][only(gev.K1changeMats[i] * SA[w...] + gev.K1offsets[i]), gev.iKK1[i+1]...]
    end
    ret += K1fac * K1val

    return ret + gev.Γbare
end

"""
Evaluate core vertex on point.
"""
function eval_core(gev::ΓEvaluator_KF, w::Vararg{Float64,3})
    return eval_interpol(gev.core, w...)
end

function eval_interpol(gev::ΓcoreEvaluator_KF, w::Vararg{Float64,3})
    return eval_interpol(ComplexF64, gev, gev.ωs_ext, w...)    
end

"""
Evaluate K2 by linear interpolation
"""
function eval_K2(gev::ΓEvaluator_KF, channel::String, prime::Bool, w::Vararg{Float64,3})
    g_id = 1
    w_ = [0.0,0.0]
    # transform frequency
    if channel==gev.channel
        g_id = prime ? 2 : 1
        w_[1] = w[1]
        w_[2] = w[ifelse(prime, 3, 2)]
    elseif channel==gev.foreign_channels[1]
        g_id = prime ? 4 : 3
        if prime
            w_ = gev.K2primechangeMats[1]*SA[w...]
        else
            w_ = gev.K2changeMats[1]*SA[w...]
        end
    elseif channel==gev.foreign_channels[2]
        g_id = prime ? 6 : 5
        if prime
            w_ = gev.K2primechangeMats[2]*SA[w...]
        else
            w_ = gev.K2changeMats[2]*SA[w...]
        end
    else
        error("INVALID CHANNEL $channel")
    end
    # evaluate/interpolate
    return eval_interpol(zeros(ComplexF64,2,2,2),gev.K2s[g_id], gev.K2s[g_id].ωs_ext, w_...)/sqrt(2)
end

"""
Evaluate K1 by linear interpolation
"""
function eval_K1(gev::ΓEvaluator_KF, channel::String, w::Vararg{Float64,3})
    g_id = 1
    w_ = 0.0
    if channel==gev.channel
        g_id = 1
        w_ = w[1]
    elseif channel==gev.foreign_channels[1]
        g_id = 2
        w_ = only(gev.K1changeMats[1] * SA[w...])
    elseif channel==gev.foreign_channels[2]
        g_id = 3
        w_ = only(gev.K1changeMats[2] * SA[w...])
    else
        error("INVALID CHANNEL $channel")
    end

    g = gev.K1grids[g_id]
    wup = searchsortedfirst(g, w_)
    wlow = wup - 1
    dw = g[wup]-g[wlow]
    wtup = (w_-g[wlow])/dw
    wtlow = (g[wup]-w_)/dw
    vallow = gev.K1s[g_id][wlow,:,:]
    valup = gev.K1s[g_id][wup,:,:]
    # 0.5 for K1 factor
    return 0.5*(wtup * valup .+ wtlow * vallow)
end

function test_nonlin_keldyshfull(flavor_idx=1)

    base_path = "SIAM_u5_U0.05_T0.0005_Delta0.0031831"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u5_U0.05_T0.0005_Delta0.0031831/PSF_nz=2_conn_zavg")
    ωs_ext = ntuple(_ -> KF_grid_bos(0.20,13), 3)
    channel="t"
    broadening_kwargs = read_all_broadening_params(base_path; channel=channel)
    broadening_kwargs[:estep]=100
    iK = 2
    # TODO: generalize for arbitrary channel (FOREIGN CHANNELS!)

    store_dir = joinpath(pdatadir(), "V_KF_JULIA_R13_$(flavor_idx)")

    gev = ΓEvaluator_KF(
        PSFpath,
        iK,
        MultipoleKFCEvaluator;
        ωs_ext=ωs_ext,
        flavor_idx=flavor_idx,
        channel=channel,
        foreign_channels=("a","pNRG"),
        KEV_kwargs=Dict(:nlevel => 2, :cutoff => 1.e-10),
        broadening_kwargs...
    )

    # interpolate onto this grid and store
    ωs_inter = ntuple(_ -> KF_grid_bos(0.17,3), 4)

    # INTERPOLATE gev

    # correlator interpolation
    core = gev.core
    gpgrids = TCI4Keldysh.Gp_grids(ωs_ext, TCI4Keldysh.channel_trafo(channel))
    GFevs = [((w1,w2,w3) -> TCI4Keldysh.eval_interpol(G, gpgrids, w1,w2,w3)) for G in core.GFevs]
    # self-energy interpolation
    omsig = TCI4Keldysh.Σ_grid(ωs_ext[1:2])
    function sev_(il::Int, inc::Bool, w::Vararg{Float64,3})
        return TCI4Keldysh.eval_interpol(core.sev, il, inc, omsig, w...)
    end

    h5name = joinpath(store_dir, "V_KF_test.h5")
    println("==== Evaluate K1 on nonlinear grid...")
    for ch in [gev.channel, gev.foreign_channels...]
        res = zeros(ComplexF64, 2,2,2,2, length.(ωs_inter)...)
        @time begin
            Threads.@threads for ic in collect(Iterators.product(Base.OneTo.(length.(ωs_inter))...))
                w = ntuple(i -> ωs_inter[i][ic[i]], 3)
                val = TCI4Keldysh.eval_K1(gev, ch, w...)
                res[:,:,:,:,Tuple(ic)...] .= TCI4Keldysh.unfold_K1(val, ch)
            end
        end
        h5write(h5name, "K1"*ch, res)
    end
    println("==== Evaluate K2 on nonlinear grid...")
    for ch in [gev.channel, gev.foreign_channels...]
        for prime in [true, false]
            res = zeros(ComplexF64, 2,2,2,2, length.(ωs_inter)...)
            @time begin
                Threads.@threads for ic in collect(Iterators.product(Base.OneTo.(length.(ωs_inter))...))
                    w = ntuple(i -> ωs_inter[i][ic[i]], 3)
                    val = TCI4Keldysh.eval_K2(gev, ch, prime, w...)
                    res[:,:,:,:,Tuple(ic)...] .= TCI4Keldysh.unfold_K2(val, ch, prime)
                end
            end
            h5write(h5name, "K2"*ch*"_$(ifelse(prime,"prime","noprime"))", res)
        end
    end
    println("==== Evaluate core vertex on nonlinear grid...")
    res = zeros(ComplexF64, 2,2,2,2, length.(ωs_inter)...)
    @time begin
        Threads.@threads for ic in collect(Iterators.product(Base.OneTo.(length.(ωs_inter))...))
            w = ntuple(i -> ωs_inter[i][ic[i]], 3)
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

    # REFERENCE on interpolated grid; is stored in asymptotic decomposition
    Vref = compute_Γfull_symmetric_estimator(
        "KF",
        PSFpath;
        T = dir_to_T(PSFpath),
        flavor_idx=flavor_idx,
        ωs_ext=ωs_inter,
        channel=channel,
        store_dir=store_dir,
        broadening_kwargs...
    )
end

function test_eval_K12_ΓEvaluator_KF()
    # base_path = "SIAM_u=0.50"
    # PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
    base_path = "SIAM_u5_U0.05_T0.0005_Delta0.0031831"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u5_U0.05_T0.0005_Delta0.0031831/PSF_nz=2_conn_zavg")
    # PSFpath = joinpath(TCI4Keldysh.datadir(), "unittest_PSF/PSF")
    flavor_idx = 2
    # all iK will be tested
    iK = 2
    channel = "t"
    broadening_kwargs = read_all_broadening_params(base_path; channel=channel)
    broadening_kwargs[:estep]=50

    # reference
    ωs_ext = ntuple(_ -> KF_grid_bos(0.26, 11), 3)
    gev = ΓEvaluator_KF(
        PSFpath,
        iK,
        MultipoleKFCEvaluator;
        ωs_ext=ωs_ext,
        flavor_idx=flavor_idx,
        channel=channel,
        foreign_channels=("a","pNRG"),
        KEV_kwargs=Dict(:nlevel => 2, :cutoff => 1.e-6),
        broadening_kwargs...
    )

    # reference grid
    ωs_ext_ref = ntuple(_ -> KF_grid_bos(0.25,3), 3)

    Vtest = zeros(ComplexF64, length.(ωs_ext_ref)..., 2,2,2,2)
    for iw in Iterators.product(Base.OneTo.(length.(ωs_ext_ref))...)
        w = ntuple(i -> ωs_ext_ref[i][iw[i]], 3)
        full = zeros(ComplexF64, 2,2,2,2)
        for ch in ["a", "pNRG", "t"]
            k1 = eval_K1(gev, ch, w...)
            full .+= unfold_K1(k1, ch)
            for prime in [true, false]
                k2 = eval_K2(gev, ch, prime, w...)
                full .+= unfold_K2(k2, ch, prime)
            end
        end
        full .+= evaluate_all_iK(gev.core, w...)
        full .+= Γbare_KF(PSFpath, flavor_idx)
        Vtest[iw...,:,:,:,:] .= full
    end

    # reference
    Vref = compute_Γfull_symmetric_estimator(
        "KF",
        PSFpath;
        T = dir_to_T(PSFpath),
        flavor_idx=flavor_idx,
        ωs_ext=ωs_ext_ref,
        channel=channel,
        broadening_kwargs...
    )
    h5write("foo_strong.h5", "V_KF_ref", Vref)
    h5write("foo_strong.h5", "V_KF_test", Vtest)
end


"""
BatchEvaluator that supports caching with multiple threads.
"""
abstract type CachedBatchEvaluator{T} <: TCI.BatchEvaluator{T} end

"""
Evaluates full Matsubara vertex. Batch evaluation will be used in TCI, which
is desirable when running on multiple threads.
"""
struct ΓBatchEvaluator_MF <: CachedBatchEvaluator{ComplexF64}
    grid::QuanticsGrids.InherentDiscreteGrid{3}
    qf::TCI.CachedFunction{ComplexF64}
    localdims::Vector{Int}
    gev::ΓEvaluator_MF

    function ΓBatchEvaluator_MF(
        gev::ΓEvaluator_MF;
        unfoldingscheme=:interleaved
    )
        # set up grid
        GFs = [gev.core.GFevs[i].GF for i in eachindex(gev.core.GFevs)]
        D = 3
        R = grid_R(GFs[1])
        @assert all(R .== grid_R.(GFs[2:end])) "Full correlator objects have different grid sizes"
        grid = QuanticsGrids.InherentDiscreteGrid{D}(R; unfoldingscheme=unfoldingscheme)
        localdims = grid.unfoldingscheme==:fused ? fill(grid.base^D, R) : fill(grid.base, D*R)

        # cached function
        qf_ = v -> gev(QuanticsGrids.quantics_to_origcoord(grid, v)...)
        qf = TCI.CachedFunction{ComplexF64}(qf_, localdims)

        return new(grid, qf, localdims, gev)
    end
end

function ΓBatchEvaluator_MF(
    PSFpath::String,
    R::Int
    ;
    T::Float64,
    flavor_idx::Int,
    channel::String,
    foreign_channels::Tuple{String,String},
    unfoldingscheme=:interleaved
)
    gev = ΓEvaluator_MF(
        PSFpath,
        R;
        T=T,
        flavor_idx=flavor_idx,
        channel=channel,
        foreign_channels=foreign_channels
    )
    return ΓBatchEvaluator_MF(gev; unfoldingscheme=unfoldingscheme)
end

"""
Evaluates full Keldysh vertex. Batch evaluation will be used in TCI, which
is desirable when running on multiple threads.
"""
struct ΓBatchEvaluator_KF <: CachedBatchEvaluator{ComplexF64}
    grid::QuanticsGrids.InherentDiscreteGrid{3}
    qf::TCI.CachedFunction{ComplexF64}
    localdims::Vector{Int}
    gev::ΓEvaluator_KF

    function ΓBatchEvaluator_KF(
        gev::ΓEvaluator_KF;
        grid_kwargs...
    )
        # set up grid
        D = 3
        R = grid_R(gev.core.GFevs[1])
        @assert all(R .== grid_R.(gev.core.GFevs[2:end])) "Full correlator objects have different grid sizes"
        grid = QuanticsGrids.InherentDiscreteGrid{D}(R; grid_kwargs...)
        localdims = grid.unfoldingscheme==:fused ? fill(grid.base^D, R) : fill(grid.base, D*R)

        # cached function
        qf_ = v -> gev(QuanticsGrids.quantics_to_origcoord(grid, v)...)
        qf = TCI.CachedFunction{ComplexF64}(qf_, localdims)

        return new(grid, qf, localdims, gev)
    end
end

function ΓBatchEvaluator_KF(
    PSFpath::String,
    R::Int,
    iK::Int;
    KEV::Type=KFCEvaluator,
    coreEvaluator_kwargs::Dict=Dict(),
    T::Float64,
    ommax::Float64,
    flavor_idx::Int,
    channel::String,
    foreign_channels::Tuple{String,String},
    grid_kwargs::Dict=Dict(),
    broadening_kwargs...
)
    gev = ΓEvaluator_KF(
        PSFpath,
        iK,
        R,
        KEV;
        ommax=ommax,
        T=T,
        flavor_idx=flavor_idx,
        channel=channel,
        foreign_channels=foreign_channels,
        KEV_kwargs=coreEvaluator_kwargs,
        broadening_kwargs...
    )
    return ΓBatchEvaluator_KF(gev; grid_kwargs...)
end

"""
Evaluates Matsubara core vertex. Batch evaluation will be used in TCI, which
is desirable when running on multiple threads.
"""
struct ΓcoreBatchEvaluator_MF{T} <: CachedBatchEvaluator{T}
    grid::QuanticsGrids.InherentDiscreteGrid{3}
    qf::TCI.CachedFunction{T}
    localdims::Vector{Int}
    gev::ΓcoreEvaluator_MF{T} # to access information

    function ΓcoreBatchEvaluator_MF(GFs::Vector{FullCorrelator_MF{3}}, sev; cutoff=1.e-20, use_ΣaIE::Bool=false, unfoldingscheme=:interleaved)
        # set up grid
        D = 3
        R = grid_R(GFs[1])
        @assert all(R .== grid_R.(GFs[2:end])) "Full correlator objects have different grid sizes"
        T = eltype(GF.Gps[1].tucker.center)
        grid = QuanticsGrids.InherentDiscreteGrid{D}(R; unfoldingscheme=unfoldingscheme)
        localdims = grid.unfoldingscheme==:fused ? fill(grid.base^D, R) : fill(grid.base, D*R)

        gev = ΓcoreEvaluator_MF(GFs, sev; cutoff=cutoff)

        # cached function
        qf_ = if use_ΣaIE
                v -> eval_LR(gev, QuanticsGrids.quantics_to_origcoord(grid, v)...)
            else # use sIE for Σ
                v -> gev(QuanticsGrids.quantics_to_origcoord(grid, v)...)
            end
        qf = TCI.CachedFunction{T}(qf_, localdims)

        return new{T}(grid, qf, localdims, gev)
    end

    function ΓcoreBatchEvaluator_MF(gev::ΓcoreEvaluator_MF{T}; use_ΣaIE::Bool=false, unfoldingscheme=:interleaved) where {T}
        # set up grid
        D = 3
        R = grid_R(gev.GFevs[1].GF)
        for i in eachindex(gev.GFevs)
            @assert R == grid_R(gev.GFevs[i].GF) "Full correlator objects have different grid sizes"
        end
        grid = QuanticsGrids.InherentDiscreteGrid{D}(R; unfoldingscheme=unfoldingscheme)
        localdims = grid.unfoldingscheme==:fused ? fill(grid.base^D, R) : fill(grid.base, D*R)

        # cached function
        qf_ = if use_ΣaIE
                v -> eval_LR(gev, QuanticsGrids.quantics_to_origcoord(grid, v)...)
            else # use sIE for Σ
                v -> gev(QuanticsGrids.quantics_to_origcoord(grid, v)...)
            end
        qf = TCI.CachedFunction{T}(qf_, localdims)

        return new{T}(grid, qf, localdims, gev)
    end
end

"""
Evaluates Keldysh core vertex. Batch evaluation will be used in TCI, which
is desirable when running on multiple threads.
"""
struct ΓcoreBatchEvaluator_KF{T} <: CachedBatchEvaluator{T}
    # discrete grid because we only need to address frequency indices, not actual frequencies
    grid::QuanticsGrids.InherentDiscreteGrid{3}
    qf::TCI.CachedFunction{T}
    localdims::Vector{Int}
    gev::ΓcoreEvaluator_KF{T} # to access information

    function ΓcoreBatchEvaluator_KF(GFs::Vector{FullCorrelator_KF{3}}, iK::Int, sev; cutoff=1.e-20, unfoldingscheme=:interleaved)
        # set up grid
        D = 3
        R = grid_R(GFs[1])
        @assert all(R .== grid_R.(GFs[2:end])) "Full correlator objects have different grid sizes"
        T = eltype(GF.Gps[1].tucker.center)
        grid = QuanticsGrids.InherentDiscreteGrid{D}(R; unfoldingscheme=unfoldingscheme)
        localdims = grid.unfoldingscheme==:fused ? fill(grid.base^D, R) : fill(grid.base, D*R)

        gev = ΓcoreEvaluator_KF(GFs, iK, sev; cutoff=cutoff)

        # cached function
        qf_ = v -> gev(QuanticsGrids.quantics_to_origcoord(grid, v)...)
        qf = TCI.CachedFunction{T}(qf_, localdims)

        return new{T}(grid, qf, localdims, gev)
    end

    function ΓcoreBatchEvaluator_KF(gev::ΓcoreEvaluator_KF{T}; cache::Dict{K,T}=Dict{UInt128,T}(), unfoldingscheme=:interleaved) where {T,K<:Union{UInt32,UInt64,UInt128,BigInt}}
        # set up grid
        D = 3
        R = grid_R(gev.GFevs[1])
        for i in eachindex(gev.GFevs)
            @assert R == grid_R(gev.GFevs[i]) "Full correlator objects have different grid sizes"
        end
        grid = QuanticsGrids.InherentDiscreteGrid{D}(R; unfoldingscheme=unfoldingscheme)
        localdims = grid.unfoldingscheme==:fused ? fill(grid.base^D, R) : fill(grid.base, D*R)

        # cached function
        qf_ = v -> gev(QuanticsGrids.quantics_to_origcoord(grid, v)...)
        qf = TCI.CachedFunction{T,K}(qf_, localdims, cache)

        return new{T}(grid, qf, localdims, gev)
    end
end



"""
Evaluation on single Quantics index.
"""
function (gbev::CachedBatchEvaluator{T})(v::Vector{Int}) where {T}
    return gbev.qf.f(v)
end

"""
If cached value exists, read it.
If qf(x) is not cached, cache it into d_write.
Intended to use CachedFunction with multiple threads.
"""
function (cf::TCI.CachedFunction{ValueType, K})(x::Vector{T}, d_write::Dict{K,ValueType}) where {ValueType, K, T<:Number}
    xkey = TCI._key(cf, x)
    return get(cf.cache, xkey) do
        # returns cf.f(x)
        d_write[xkey] = cf.f(x)
    end
end


"""
Batch evaluation of core vertex
"""
function (gbev::CachedBatchEvaluator{T})(
    leftindexsset::Vector{Vector{Int}}, rightindexsset::Vector{Vector{Int}}, ::Val{M}
    ) where {T,M}

    nleft = length(first(leftindexsset))
    cindexset = vec(collect(Iterators.product(ntuple(i -> 1:gbev.localdims[nleft+i], M)...)))
    out = Array{T,M+2}(undef, (length(leftindexsset), ntuple(i->gbev.localdims[nleft+i],M)..., length(rightindexsset)))

    # populate Pi-tensor
    Threads.@threads for il in eachindex(leftindexsset)
        left_act = leftindexsset[il]
        for ic in eachindex(cindexset)
            cen_act = cindexset[ic]
            for ir in eachindex(rightindexsset)
                idx = vcat(left_act, cen_act..., rightindexsset[ir])
                key = TCI._key(gbev.qf, idx)
                out[il, cindexset[ic]..., ir] = haskey(gbev.qf.cache, key) ? gbev.qf.cache[key] : gbev(idx)
            end
        end
    end

    # update cache, non-threaded!
    for il in eachindex(leftindexsset)
        left_act = leftindexsset[il]
        for ic in eachindex(cindexset)
            cen_act = cindexset[ic]
            for ir in eachindex(rightindexsset)
                idx = vcat(left_act, cen_act..., rightindexsset[ir])
                gbev.qf.cache[TCI._key(gbev.qf, idx)] = out[il, cindexset[ic]..., ir]
            end
        end
    end

    # chunklen = ceil(Int, length(leftindexsset) / Threads.nthreads())
    # chunks = Iterators.partition(eachindex(leftindexsset),  chunklen)
    # cache_dicts = Dict([tid => typeof(gbev.qf.cache)() for tid in Threads.threadpooltids(:default)])

    # @assert length(chunks) <= Threads.nthreads()
    # Threads.@threads :static for chunk in collect(chunks)
    #         tid_act = Threads.threadid()
    #         println("-- Thread $(tid_act)/$(Threads.nthreads()) is processing chunk of size $(length(chunk))x$(length(cindexset))x$(length(rightindexsset))")
    #         for il in chunk
    #             left_act = leftindexsset[il]
    #             for ic in eachindex(cindexset)
    #                 cen_act = cindexset[ic]
    #                 for ir in eachindex(rightindexsset)
    #                     idx = vcat(left_act, cen_act..., rightindexsset[ir])
    #                     out[il, cindexset[ic]..., ir] = gbev.qf(idx, cache_dicts[tid_act])
    #                 end
    #             end
    #         end
    #         println("   Cache dict $(tid_act)/$(Threads.nthreads()) has size: $(Base.summarysize(cache_dicts)/1.e6) MB")
    #         flush(stdout)
    #     end

    # # merge chached function values into cached function dict
    # for (_, d) in cache_dicts
    #     merge!(gbev.qf.cache, d)
    # end

    #= 
    if DEBUG_TCI_KF_RAM()
        println("\n----------------------------------------")
        println("     Size of gbev: $(Base.summarysize(gbev))")
        println("     Size of entire cache: $(Base.summarysize(gbev.qf.cache))")
        report_mem(true)
        println("---- Evaluations for 2-site update done")
        println("----------------------------------------\n")
    end
    =#

    return out
end