using BenchmarkTools
#=
Compute interection vertex with symmetric improved estimators and TCI
=#

# ========== MATSUBARA

"""
Evaluate self-energy pointwise by symmetric improved estimator.
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

    Adisc_Σ_H = load_Adisc_0pt(PSFpath, "Q12", flavor_idx)
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
    # Threads.@threads for l in 1:Ncorrs # can two threads try to read the same file here?
    for l in 1:length(GFs)
        letts = letter_combinations[l]
        println("letts: ", letts)
        ops = [letts[i]*op_labels[i] for i in 1:4]
        if !any(parse_Ops_to_filename(ops) .== filelist)
            ops = [letts[i]*op_labels_symm[i] for i in 1:4]
        end
        GFs[l] = TCI4Keldysh.FullCorrelator_MF(PSFpath_4pt, ops; T, flavor_idx, ωs_ext, ωconvMat);
        printstyled("== Memory usage [GB] of $(l)-th full correlator: $(Base.summarysize(GFs[l]) / 1024^3)\n"; color=:blue)
    end
end

function letter_combonations_Γcore()
    letters = ["F", "Q"]
    return kron(kron(letters, letters), kron(letters, letters))
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
        println("letts: ", letts)
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

    # approximate values
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

function initpivots_Γcore(GFs::Union{Vector{FullCorrelator_MF{D}}, Vector{FullCorrelator_KF{D}}}) where {D}

    pivots = Vector{Int}[]

    # # central 3^D block
    # ωs_ext = first(GFs).ωs_ext
    # centre_ids = ntuple(i -> ifelse(isodd(length(ωs_ext[i])), div(length(ωs_ext[i]),2)+1, div(length(ωs_ext[i]),2)), D)
    # centre_block = ntuple(i -> centre_ids[i]-1:centre_ids[i]+1, D)
    # for c in Iterators.product(centre_block...)
    #     push!(pivots,  collect(c))
    # end

    # find all lines that are rotated onto some coordinate axis in and internal frequency grid of a partial correlator
    lines = [Vector{Float64}[] for _ in 1:D]
    for GF in GFs
        for Gp in GF.Gps

            w_cen = collect(ntuple(i -> div(length(Gp.tucker.ωs_legs[i]), 2), D))
            bos_idx = get_bosonic_idx(Gp)
            if !isnothing(bos_idx)
                zero_idx = findfirst(x -> abs(x)<=1.e-2*Gp.T, Gp.tucker.ωs_legs[bos_idx])
                w_cen[bos_idx] = zero_idx
            end

            w_ext = Gp.ωconvMat \ (w_cen .- Gp.ωconvOff)
            for l in 1:D
                w_int_act = copy(w_cen)
                # stay somewhat close to the centre
                w_int_act[l] += min(div(w_cen[l], 2), 10)
                w_ext_act = Gp.ωconvMat \ (w_int_act .- Gp.ωconvOff)
                line = w_ext_act .- w_ext
                pushline = true
                # check whether we found a new direction
                for line2 in lines[l]
                    if abs(dot(line2, line)) / (norm(line)*norm(line2)) >= 0.9
                        pushline = false
                        break
                    end
                end
                if pushline
                    push!(lines[l], line)
                    push!(pivots, round.(Int, w_ext_act))
                end
            end
        end
    end

    printstyled("==== Using $(length(pivots)) initial pivots:\n"; color=:blue)
    # display([p .- [div(length(GFs[1].ωs_ext[i]), 2) for i in 1:D] for p in pivots])
    display(pivots)
    printstyled("====\n"; color=:blue)
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
    gbev = ΓcoreBatchEvaluator_MF(gev; use_ΣaIE=use_ΣaIE)

    GC.gc(true)

    initpivots_ω = initpivots_Γcore([gev.GFevs[i].GF for i in eachindex(gev.GFevs)])
    initpivots = [QuanticsGrids.origcoord_to_quantics(gbev.grid, tuple(iw...)) for iw in initpivots_ω]

    printstyled("Memory usage [GB] of ΓcoreBatchEvaluator_MF: $(Base.summarysize(gbev) / (1024^3))\n"; color=:blue)

    @info "BATCHED"
    t = @elapsed begin
        tt, _, _ = TCI.crossinterpolate2(ComplexF64, gbev, gbev.qf.localdims, initpivots; tcikwargs...)
    end
    qtt = QuanticsTCI.QuanticsTensorCI2{ComplexF64}(tt, gbev.grid, gbev.qf)
    @info "quanticscrossinterpolate time batched (nocache): $t"
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
    use_ΣaIE::Bool=true,
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
    letter_combinations = letter_combonations_Γcore()
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
    initpivots_ω = initpivots_Γcore([GFevs[i].GF for i in eachindex(GFevs)])

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

"""
Second+third row in Fig 13, Lihm et. al., split in 6 terms in a, p, t channels
Return 2*R bit quantics tensor train (2D function)
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
    letter_combinations = ["FF", "FQ", "QF", "QQ"]
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
    qtt, _, _ = quanticscrossinterpolate(ComplexF64, eval_K2, ntuple(i -> 2^R, 2), pivots; qtcikwargs...)

    return qtt
end

# ========== MATSUBARA END

# ========== KELDYSH

"""
Compute Keldysh core vertex for single Keldysh component

* iK: Keldysh component
* sigmak, γ: broadening parameters
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
    estep::Int=200,
    cache_center::Int=0, # later: cache central values
    ωconvMat::Matrix{Int},
    T::Float64,
    flavor_idx::Int=1,
    batched=true,
    unfoldingscheme=:interleaved,
    tcikwargs...
    )

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
        println("letts: ", letts)
        ops = [letts[i]*op_labels[i] for i in 1:4]
        if !any(parse_Ops_to_filename(ops) .== filelist)
            ops = [letts[i]*op_labels_symm[i] for i in 1:4]
        end
        GFs[l] = TCI4Keldysh.FullCorrelator_KF(PSFpath_4pt, ops; T, flavor_idx, ωs_ext, ωconvMat, sigmak=sigmak, γ=γ, broadening_kwargs...)
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

    # function eval_Γ_core(w::Vararg{Int,3})
    #     addvals = Vector{ComplexF64}(undef, Ncorrs)
    #     Threads.@threads for i in 1:Ncorrs
    #         # first all Keldysh indices
    #         # res = reshape(evaluate_all_iK(GFs[i], w...), ntuple(_->2, D+1))
    #         res = reshape(GFevs[i](w...), ntuple(_->2, D+1))
    #         val_legs = Vector{Vector{ComplexF64}}(undef, length(is_incoming))
    #         for il in eachindex(is_incoming)
    #             mat = if letter_combinations[i][il]==='F'
    #                     -sev(il, w...)
    #                 else
    #                     X
    #                 end
    #             leg = if is_incoming[il]
    #                     vec(mat[:, iK_tuple[il]])
    #                 else
    #                     vec(mat[iK_tuple[il], :])
    #                 end
    #             val_legs[il] = leg
    #         end
    #         for d in 1:D+1
    #             res = res[1,ntuple(_->Colon(),D+1-d)...].*val_legs[d][1] .+ res[2,ntuple(_->Colon(),D+1-d)...].*val_legs[d][2]
    #         end
    #         addvals[i] = res
    #     end
    #     return sum(addvals)
    # end

    kwargs_dict = Dict(tcikwargs)
    cutoff = haskey(Dict(kwargs_dict), :tolerance) ? kwargs_dict[:tolerance]*1.e-2 : 1.e-12
    gev = ΓcoreEvaluator_KF(GFs, iK, sev; cutoff=cutoff)

    initpivots_ω = initpivots_Γcore([gev.GFevs[i].KFC for i in eachindex(gev.GFevs)])
    GC.gc(true)

    if batched
        gbev = ΓcoreBatchEvaluator_KF(gev)
        initpivots = [QuanticsGrids.origcoord_to_quantics(gbev.grid, tuple(iw...)) for iw in initpivots_ω]
        t = @elapsed begin
            tt, _, _ = TCI.crossinterpolate2(ComplexF64, gbev, gbev.qf.localdims, initpivots; tcikwargs...)
        end
        qtt = QuanticsTCI.QuanticsTensorCI2{ComplexF64}(tt, gbev.grid, gbev.qf)
        @info "quanticscrossinterpolate time (nocache, batched): $t"
        return qtt
    else
        t = @elapsed begin
            qtt, _, _ = quanticscrossinterpolate(ComplexF64, gev, ntuple(i -> 2^R, D), initpivots_ω; unfoldingscheme=unfoldingscheme, tcikwargs...)
        end
        @info "quanticscrossinterpolate time (nocache, not batched): $t"
    return qtt
    end

end

"""
To evaluate Matsubara core vertex, wrapping the required setup and relevant data.
* sev: callable with signature sev(i::Int, w::Vararg{Int,D}) to evaluate self-energy
on i'th component of transformed frequency w
"""
struct ΓcoreEvaluator_MF{T}
    GFevs::Vector{FullCorrEvaluator_MF{T,3,2}}
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
        
        # create correlator evaluators
        T = eltype(GFs[1].Gps[1].tucker.legs[1])
        GFevs = [FullCorrEvaluator_MF(GFs[i], true; cutoff=cutoff) for i in eachindex(GFs)]

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
    cutoff::Float64
    )
    # make frequency grid
    D = size(ωconvMat, 2)
    Nhalf = 2^(R-1)
    ωs_ext = MF_npoint_grid(T, Nhalf, D)

    # all 16 4-point correlators
    letter_combinations = letter_combonations_Γcore()
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
Structure to evaluate Keldysh core vertex, i.e., wrap the required setup and capture relevant data.
* sev: function with signature sev(i::Int, is_incoming::Bool, w::Vararg{Int,D}) to evaluate self-energy
on i'th component of transformed frequency w
"""
struct ΓcoreEvaluator_KF{T}
    GFevs::Vector{FullCorrEvaluator_KF{3,T}}
    Ncorrs::Int # number of full correlators
    iK_tuple::NTuple{4,Int} # requested Keldysh idx
    X::Matrix{ComplexF64}
    is_incoming::NTuple{4,Bool}
    letter_combinations::Vector{String}
    sev::Function

    function ΓcoreEvaluator_KF(
        GFs::Vector{FullCorrelator_KF{3}},
        iK::Int,
        sev,
        is_incoming::NTuple{4,Bool},
        letter_combinations::Vector{String}
        ;
        cutoff::Float64=1.e-20)
        
        # create correlator evaluators
        T = eltype(GFs[1].Gps[1].tucker.legs[1])
        GFevs = [FullCorrEvaluator_KF(GFs[i]; cutoff=cutoff) for i in eachindex(GFs)]
        X = get_PauliX()
        iK_tuple = KF_idx(iK,3)

        return new{T}(GFevs,length(GFs), iK_tuple, X, is_incoming, letter_combinations, sev)
    end
end

function ΓcoreEvaluator_KF(
    GFs::Vector{FullCorrelator_KF{3}},
    iK::Int,
    sev,
    ;
    cutoff::Float64=1.e-20)

    is_incoming = (false, true, false, true)
    letters = ["F", "Q"]
    letter_combinations = kron(kron(letters, letters), kron(letters, letters))

    return ΓcoreEvaluator_KF(GFs, iK, sev, is_incoming, letter_combinations; cutoff=cutoff)
end

"""
Evaluate Γcore using sIE for self-energy on all legs
"""
function (gev::ΓcoreEvaluator_KF{T})(w::Vararg{Int,3}) where {T}
    addvals = Vector{T}(undef, gev.Ncorrs)
    for i in 1:gev.Ncorrs
        # first all Keldysh indices
        res = reshape(gev.GFevs[i](w...), ntuple(_->2, 4))
        val_legs = Vector{Vector{T}}(undef, length(gev.is_incoming))
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

abstract type ΓcoreBatchEvaluator{T} <: TCI.BatchEvaluator{T} end

struct ΓcoreBatchEvaluator_MF{T} <: ΓcoreBatchEvaluator{T}
    grid::QuanticsGrids.InherentDiscreteGrid{3}
    qf::TCI.CachedFunction{T}
    localdims::Vector{Int}
    gev::ΓcoreEvaluator_MF{T} # to access information

    function ΓcoreBatchEvaluator_MF(GFs::Vector{FullCorrelator_MF{3}}, sev; cutoff=1.e-20, use_ΣaIE::Bool=false)
        # set up grid
        D = 3
        R = grid_R(GFs[1])
        @assert all(R .== grid_R.(GFs[2:end])) "Full correlator objects have different grid sizes"
        T = eltype(GF.Gps[1].tucker.center)
        grid = QuanticsGrids.InherentDiscreteGrid{D}(R; unfoldingscheme=:interleaved)
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

    function ΓcoreBatchEvaluator_MF(gev::ΓcoreEvaluator_MF{T}; use_ΣaIE::Bool=false) where {T}
        # set up grid
        D = 3
        R = grid_R(gev.GFevs[1].GF)
        for i in eachindex(gev.GFevs)
            @assert R == grid_R(gev.GFevs[i].GF) "Full correlator objects have different grid sizes"
        end
        grid = QuanticsGrids.InherentDiscreteGrid{D}(R; unfoldingscheme=:interleaved)
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
To evaluate Keldysh core vertex in parallelized fashion (more than 16 threads).
"""
struct ΓcoreBatchEvaluator_KF{T} <: ΓcoreBatchEvaluator{T}
    # discrete grid because we only need to address frequency indices, not actual frequencies
    grid::QuanticsGrids.InherentDiscreteGrid{3}
    qf::TCI.CachedFunction{T}
    localdims::Vector{Int}
    gev::ΓcoreEvaluator_KF{T} # to access information

    function ΓcoreBatchEvaluator_KF(GFs::Vector{FullCorrelator_KF{3}}, iK::Int, sev; cutoff=1.e-20)
        # set up grid
        D = 3
        R = grid_R(GFs[1])
        @assert all(R .== grid_R.(GFs[2:end])) "Full correlator objects have different grid sizes"
        T = eltype(GF.Gps[1].tucker.center)
        grid = QuanticsGrids.InherentDiscreteGrid{D}(R; unfoldingscheme=:interleaved)
        localdims = grid.unfoldingscheme==:fused ? fill(grid.base^D, R) : fill(grid.base, D*R)

        gev = ΓcoreEvaluator_KF(GFs, iK, sev; cutoff=cutoff)

        # cached function
        qf_ = v -> gev(QuanticsGrids.quantics_to_origcoord(grid, v)...)
        qf = TCI.CachedFunction{T}(qf_, localdims)

        return new{T}(grid, qf, localdims, gev)
    end

    function ΓcoreBatchEvaluator_KF(gev::ΓcoreEvaluator_KF{T}) where {T}
        # set up grid
        D = 3
        R = grid_R(gev.GFevs[1].KFC)
        for i in eachindex(gev.GFevs)
            @assert R == grid_R(gev.GFevs[i].KFC) "Full correlator objects have different grid sizes"
        end
        grid = QuanticsGrids.InherentDiscreteGrid{D}(R; unfoldingscheme=:interleaved)
        localdims = grid.unfoldingscheme==:fused ? fill(grid.base^D, R) : fill(grid.base, D*R)

        # cached function
        qf_ = v -> gev(QuanticsGrids.quantics_to_origcoord(grid, v)...)
        qf = TCI.CachedFunction{T}(qf_, localdims)

        return new{T}(grid, qf, localdims, gev)
    end
end



"""
Evaluation on single Quantics index.
"""
function (gbev::ΓcoreBatchEvaluator{T})(v::Vector{Int}) where {T}
    # return gbev.qf(v)
    return gbev.qf.f(v)
end

"""
If cached value exists, read it.
If qf(x) is not cached, cache it into d_write.
Intended to use CachedFunction with multiple threads.
"""
function (cf::TCI.CachedFunction{ValueType, K})(x::Vector{T}, d_write::Dict{K,ValueType}) where {ValueType, K, T<:Number}
    # printstyled("  tid: $(Threads.threadid())\n"; color=:gray)
    xkey = TCI._key(cf, x)
    return get(cf.cache, xkey) do
        # returns cf.f(x)
        d_write[xkey] = cf.f(x)
    end
end


"""
Batch evaluation of core vertex
"""
function (gbev::ΓcoreBatchEvaluator{T})(
    leftindexsset::Vector{Vector{Int}}, rightindexsset::Vector{Vector{Int}}, ::Val{M}
    ) where {T,M}
    nleft = length(first(leftindexsset))

    cindexset = vec(collect(Iterators.product(ntuple(i -> 1:gbev.localdims[nleft+i], M)...)))
    out = Array{T,M+2}(undef, (length(leftindexsset), ntuple(i->gbev.localdims[nleft+i],M)..., length(rightindexsset)))

    # careful: manual treatment of caching required to avoid simultaneous writes to CachedFunction dict by different threads
    # Threads.@threads for il in eachindex(leftindexsset)
    #         left_act = leftindexsset[il]
    #     for ic in eachindex(cindexset)
    #         cen_act = cindexset[ic]
    #         for ir in eachindex(rightindexsset)
    #             idx = vcat(left_act, cen_act..., rightindexsset[ir])
    #             out[il, cindexset[ic]..., ir] = gbev.qf(idx)
    #         end
    #     end
    # end

    # chunklen = max(div(length(leftindexsset), Threads.nthreads()), 1)
    chunklen = ceil(Int, length(leftindexsset) / Threads.nthreads())
    chunks = Iterators.partition(eachindex(leftindexsset),  chunklen)
    printstyled("== CHUNKS: $(length.(chunks)) ($(Threads.nthreads()) threads)\n"; color=:blue)
    printstyled("  MEM[GB]: gbev $(Base.summarysize(gbev) / 1024^3), of which qf $(Base.summarysize(gbev.qf) / 1024^3); out $(Base.summarysize(out) / 1024^3)\n"; color=:blue)
    cache_dicts = Dict([tid => typeof(gbev.qf.cache)() for tid in Threads.threadpooltids(:default)])

    @assert length(chunks) <= Threads.nthreads()
    # TODO: Is there a better solution than using the :static schedule to make sure cache_dicts are not written
    # by two threads at the same time? -> Use Locks?
    Threads.@threads :static for chunk in collect(chunks)
            tid_act = Threads.threadid()
            # println("-- Thread $tid_act is processing chunk of size $(length(chunk))")
            for il in chunk
                    left_act = leftindexsset[il]
                for ic in eachindex(cindexset)
                    cen_act = cindexset[ic]
                    for ir in eachindex(rightindexsset)
                        idx = vcat(left_act, cen_act..., rightindexsset[ir])
                        out[il, cindexset[ic]..., ir] = gbev.qf(idx, cache_dicts[tid_act])
                    end
                end
            end
        end

    # merge chached function values into cached function dict
    for (_, d) in cache_dicts
        merge!(gbev.qf.cache, d)
    end

    return out
end