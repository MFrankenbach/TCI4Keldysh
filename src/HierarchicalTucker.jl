function nested_intervals(min::Int, max::Int, nlevel::Int)
    levels = [[min:max]]
    if max<=min error("Invalid boundaries") end
    if nlevel > log2(max-min) error("Too many levels requested") end
    for _ in 1:nlevel-1
        level_act = UnitRange{Int}[]
        level_last = levels[end]
        for l in level_last
            left = first(l)
            right = last(l)
            mid = div(right+left, 2)
            push!(level_act, left:mid)
            push!(level_act, mid+1:right)
        end
        push!(levels, level_act)
    end
    return levels
end

"""
Divide rows in groups and SVD corresponding horizontal slices
of matrix individually
"""
function multipole_matrix(mat::Matrix{T}, intervals::Vector{UnitRange{Int}}; cutoff::Float64=1.e-8) where {T<:Number}
    @assert sum(length.(intervals))==size(mat,1) "intervals do not partition the matrix"
    n_interval = length(intervals)
    Us = Vector{Matrix{T}}(undef, n_interval)
    SVs = Vector{Matrix{T}}(undef, n_interval)
    for (ii,int) in enumerate(intervals)
        U,S,V = svd(mat[int,:])
        Scut = findfirst(s -> s/S[1]<=cutoff, S)
        if isnothing(Scut)
            Us[ii] = U
            SVs[ii] = diagm(S)*V'
        else
            Us[ii] = U[:,1:Scut]
            SVs[ii] = diagm(S[1:Scut])*adjoint(V[:,1:Scut])
        end
    end
    return (Us, SVs)
end

"""
Divide rows in 2^nlevel groups and SVD corresponding horizontal slices
of matrix individually
"""
function multipole_matrix(mat::Matrix{T}, nlevel::Int; cutoff::Float64=1.e-8) where {T<:Number}
    intervals = nested_intervals(1, size(mat,1), nlevel)[end]
    return multipole_matrix(mat, intervals; cutoff=cutoff)
end

function contract_blocks(mats::NTuple{D,Vector{Matrix{T}}}, center::Array{S,D}) where {D,T<:Number,S<:Number}
    # array of blocks
    nmats = length.(mats)
    ret = Array{Array{promote_type(S,T), D}, D}(undef, Tuple(nmats))
    blockranges = Base.OneTo.(nmats)
    for d in 1:D
        @assert all([size(m,2)==size(center,d) for m in mats[d]])
    end
    for ic in Iterators.product(blockranges...)
        # relevant matrices
        ms = [mats[d][ic[d]] for d in 1:D]
        # contract with center
        ret[Tuple(ic)...] = contract_1D_Kernels_w_Adisc_mp(ms, center)
    end
    return ret
end

function prepare_hierarchical_tucker(
    kernels::NTuple{D,Matrix{T}},
    center::Array{S,D},
    nlevel::Int;
    cutoff=1.e-8
    ) where {D,T<:Number,S<:Number}
    
    USVs = [multipole_matrix(kernels[d], nlevel; cutoff=cutoff) for d in 1:D]
    center_new = contract_blocks(ntuple(i -> USVs[i][2],D), center)
    return (ntuple(i->USVs[i][1],D), center_new)
end

const Legs{D,T}=NTuple{D,Vector{Matrix{T}}} where {D,T}

"""
Implement a tucker decomposition
G(ω,ν,ν')=∑_ϵ k(ω,ϵ1)k(ν,ϵ2)k(ν',ϵ3) center(ϵ)
that performs blockwise SVD compression of kernel slices
k[slice,:] and contracts SV into the corresponding part of the
center. This can be used to achieve a balance between memory and
single point evaluation cost.
It can be viewed (kind of) as a multipole expansion of the kernels.

In principle, one could also partition the range of ϵ's, but this
did not look promising.

* ids_cumulated: at which indices the different kernel matrices
start in each direction (minus 1), i.e., [0, length(FirstMatrix), ...]
total length: number of kernels + 1
* kernels: have external frequency index as column index
because reading columns is faster
"""
struct HierarchicalTucker{D,T}
    center::Array{Array{T,D},D}
    kernels::Legs{D,T}
    ids_cumulated::NTuple{D,Vector{Int}}

    function HierarchicalTucker(center::Array{Array{T,D}}, kernels::Legs{D,T}) where {D,T}
        ids_cumulated = ntuple(
            i -> pushfirst!(cumsum(size.(kernels[i], 2)), 0),
            D
        )        

        return new{D,T}(center, kernels, ids_cumulated)
    end
end

function HierarchicalTucker(
    center::Array{T,D},
    kernels::NTuple{D,Matrix{S}},
    nlevel::Int
    ;
    cutoff::Float64=1.e-8
    ) where {D,T,S}
    
    Us, center_new = prepare_hierarchical_tucker(
                        kernels, center, nlevel; cutoff=cutoff
                        )
    return HierarchicalTucker(center_new, ntuple(i -> Matrix.(transpose.(Us[i])), D))
end

function findlast_typestable(f::Function, A::Vector{Int})
    ret = findlast(f, A)
    return isnothing(ret) ? 0 : ret 
end

function (ht::HierarchicalTucker{D,T})(idx::Vararg{Int,D}) where {D,T}
    matrix_ids = ntuple(d -> findlast_typestable(id -> id<idx[d], ht.ids_cumulated[d]), D)
    idx_new = ntuple(d -> idx[d] - ht.ids_cumulated[d][matrix_ids[d]], D)
    legs_idx = ntuple(d -> view(ht.kernels[d][matrix_ids[d]], :, idx_new[d]), D)
    return eval_tucker(ht.center[matrix_ids...], legs_idx) 
end

function locate_block(ht::HierarchicalTucker{D,T}, idx::NTuple{D,Int}) :: NTuple{D,Int} where {D,T}
    return ntuple(d -> findlast(id -> id<idx[d], ht.ids_cumulated[d]), D)
end

function find_legs(ht::HierarchicalTucker{D,T}, idx::NTuple{D,Int}) where {D,T}
    matrix_ids = locate_block(ht, idx)
    idx_new = ntuple(d -> idx[d] - ht.ids_cumulated[d][matrix_ids[d]], D)
    return (matrix_ids, ntuple(d -> view(ht.kernels[d][matrix_ids[d]], :, idx_new[d]), D))
end

function find_legs_blockbased(ht::HierarchicalTucker{D,T}, matrix_ids::NTuple{D,Int}, idx::NTuple{D,Int}) where {D,T}
    idx_new = ntuple(d -> idx[d] - ht.ids_cumulated[d][matrix_ids[d]], D)
    return ntuple(d -> view(ht.kernels[d][matrix_ids[d]], :, idx_new[d]), D)
end

"""
Interpolate function f(Vararg{Int,D})->::Array{ComplexF64,D} linearly onto point w
ws: 1D frequency grids
"""
function eval_interpol(::T, f, ws::NTuple{D,Vector{Float64}}, w::Vararg{Float64,D}) where {T,D}
    # surrounding indices
    idxup = ntuple(i -> min(searchsortedfirst(ws[i], w[i]), length(ws[i])), D)
    idxlow = ntuple(i -> max(idxup[i]-1, 1), D)
    # corners of enclosing cube
    pts = Array{SVector{D,Float64},D}(undef, ntuple(_->2,D)...)
    ids_pts = Array{SVector{D,Int},D}(undef, ntuple(_->2,D)...)
    values_pts = Array{T,D}(undef, ntuple(_->2,D)...)
    for t in Iterators.product(ntuple(_->1:2,D)...)
        ids_act = SA[ntuple(j -> t[j]==1 ? idxlow[j] : idxup[j], D)...]
        ids_pts[t...] = ids_act
        pts[t...] = SA[ntuple(j -> ws[j][ids_act[j]], D)...]
        values_pts[t...] = f(ids_act...)
    end
    # trilinear interpolation from cube corners
    return interpolate_trilinear(values_pts, pts, SA[w...] .- first(pts))
end

"""
Interpolate HierarchicalTucker linearly onto point w
ws: 1D frequency grids
"""
function eval_interpol(ht::HierarchicalTucker{D,T}, ws::NTuple{D,LinRange{Float64}}, w::Vararg{Float64,D}) where {D,T}
    # surrounding indices
    idxup = ntuple(i -> min(searchsortedfirst(ws[i], w[i]), length(ws[i])), D)
    idxlow = ntuple(i -> max(idxup[i]-1, 1), D)
    # corners of enclosing cube
    pts = Array{SVector{D,Float64},D}(undef, ntuple(_->2,D)...)
    ids_pts = Array{SVector{D,Int},D}(undef, ntuple(_->2,D)...)
    values_pts = Array{T,D}(undef, ntuple(_->2,D)...)
    for t in Iterators.product(ntuple(_->1:2,D)...)
        ids_act = SA[ntuple(j -> t[j]==1 ? idxlow[j] : idxup[j], D)...]
        ids_pts[t...] = ids_act
        pts[t...] = SA[ntuple(j -> ws[j][ids_act[j]], D)...]
        values_pts[t...] = ht(ids_act...)
    end
    # trilinear interpolation from cube corners
    return interpolate_trilinear(values_pts, pts, SA[w...] .- first(pts))
end

# function eval_interpol(ht::HierarchicalTucker{D,T}, ws::NTuple{D,LinRange{Float64}}, w::Vararg{Float64,D}) where {D,T}
#     # surrounding indices
#     idxup = ntuple(i -> min(searchsortedfirst(ws[i], w[i]), length(ws[i])), D)
#     idxlow = ntuple(i -> max(idxup[i]-1, 1), D)
#     # corners of enclosing cube
#     pts = Vector{SVector{D,Float64}}(undef, 2^D)
#     ids_pts = Vector{SVector{D,Int}}(undef, 2^D)
#     for (i,t) in enumerate(Iterators.product(ntuple(_->1:2,D)...))
#         ids_pts[i] = SA[ntuple(j -> t[j]==1 ? idxlow[j] : idxup[j], D)...]
#         pts[i] = SA[ntuple(j -> ws[j][ids_pts[i][j]], D)...]
#     end
#     # D+1 closest points
#     perm = sortperm(pts, by = p -> norm(p-SA[w...]))
#     simplex_pts = pts[perm[1:D+1]] 
#     # interpolate from simplex
#     a = interpolate_simplex_coeffs(Tuple(simplex_pts), SA[w...])
#     ret = a[1] * ht(ids_pts[perm[1]]...)
#     for i in 2:D+1
#         ret += a[i] * ht(ids_pts[perm[i]]...)
#     end
#     return ret
# end

# function eval_interpol(ht::HierarchicalTucker{D,T}, ws::NTuple{D,LinRange{Float64}}, w::Vararg{Float64,D}) where {D,T}
#     # surrounding indices
#     idxup = ntuple(i -> min(searchsortedfirst(ws[i], w[i]), length(ws[i])), D)
#     idxlow = ntuple(i -> max(idxup[i]-1, 1), D)

#     matrix_idslow, legs_idxlow = find_legs(ht, idxlow)

#     matrix_idsup, legs_idxup = find_legs(ht, idxup)

#     da = ntuple(i -> w[i]-ws[i][idxlow[i]], D)
#     db = ntuple(i -> ws[i][idxup[i]]-w[i], D)
#     weights = ntuple(i -> da[i]/(da[i]+db[i]), D)
#     println("---- eval_interpol")
#     println("     weights: $(da) $(db) $(weights)")
#     println("     ids: $(idxup) $(idxlow) $(matrix_idslow) $(matrix_idsup) $(legs_idxlow) $(legs_idxup)")
#     if matrix_idslow==matrix_idsup
#         # interpolate kernel onto requested frequency
#         legs = ntuple(i -> legs_idxup[i] * weights[i] + legs_idxlow[i] * (1.0-weights[i]), D)
#         ret = eval_tucker(ht.center[matrix_idsup...], legs) 
#         println("     general")
#         display(ret)
#         return ret
#     else
#         # special treatment if lower and upper frequencies correspond to different blocks
#         # LEAVE LINEAR INTERPOLATION FOR NOW
#             # interp_vals = zeros(T, ntuple(_->2,D))
#             # for ii in Iterators.product(ntuple(_->1:2,D))
#             #     matrix_ids = locate_block(ht, ntuple(j -> ii[j]==1 ? idxlow[j] : idxup[j], 3))
#             #     legs = find_legs_blockbased(ht, matrix_ids)
#             #     interp_vals = eval_tucker(ht.center[matrix_ids...], legs)
#             # end
#         idx_next = ntuple(i -> weights[i]<0.5 ? idxlow[i] : idxup[i], D)
#         _, legs = find_legs(ht, idx_next)
#         println("     special (idx_next: $(idx_next))")
#         ret = eval_tucker(ht.center[matrix_idsup...], legs)
#         display(ret)
#         return ret
#     end
# end

function precompute_all_values(ht::HierarchicalTucker{D,T}) where {D,T}
    ret = zeros(T, ntuple(d -> sum(size.(ht.kernels[d], 2)),D))
    for ic in CartesianIndices(ht.center)
        k_act = [transpose(ht.kernels[d][ic[d]]) for d in 1:D]
        window = [ht.ids_cumulated[d][ic[d]]+1:ht.ids_cumulated[d][ic[d]+1] for d in 1:D]
        ret[window...] .= contract_1D_Kernels_w_Adisc_mp(k_act, ht.center[ic])
    end
    return ret
end

"""
Evaluate full Keldysh correlator with partial correlators represented as HierarchicalTucker
decompositions
* Gps: Dx(D+1)! matrix for D+1 (= no. of fully retarded kernels) hierarchical tucker decompositions for each partial correlator
"""
struct MultipoleKFCEvaluator{D} <: AbstractCorrEvaluator_KF{D,ComplexF64}
    Gps::Matrix{HierarchicalTucker{D,ComplexF64}}
    ωconvOffs::Vector{SVector{D,Int}}
    # SMatrix needs four (!) types to be concrete: {S1,S2,T,L}
    ωconvMats::Vector{Matrix{Int}}
    ωs_ext::NTuple{D,Vector{Float64}}
    GR_to_GK::Array{Float64,3}

    function MultipoleKFCEvaluator(GF::FullCorrelator_KF{D}; nlevel::Int=4, cutoff::Float64=1.e-8) where {D}
        nGps = GF.NGps
        Gps_ = Matrix{HierarchicalTucker{D,ComplexF64}}(undef, D+1, nGps)
        ωconvOffs = Vector{SVector{D,Int}}(undef, nGps)
        ωconvMats = Vector{Matrix{Int}}(undef, nGps)
        Threads.@threads for ip in axes(GF.Gps,1)
            vprintln(" Processing partial correlator no. $ip/$nGps")
            for l in axes(GF.Gps,2)
                Gp = GF.Gps[ip,l]
                Gps_[l, ip] = HierarchicalTucker(
                    Gp.tucker.center, Tuple(Gp.tucker.legs), nlevel; cutoff=cutoff
                    )
            end
            # frequency rotations only change with permutation
            ωconvOffs[ip] = copy(GF.Gps[ip,1].ωconvOff)
            ωconvMats[ip] = copy(GF.Gps[ip,1].ωconvMat)
        end
        if VERBOSITY[]>=2
            println("==== COMBINED HIERARCHICAL TUCKER: MEMORY")
            @show Base.summarysize(Gps_) / 1.e9
            println("==== MEMORY END")
        end
        return new{D}(Gps_, ωconvOffs, ωconvMats, deepcopy(GF.ωs_ext), copy(GF.GR_to_GK))
    end
end

"""
Call with buffers
ret has length D1 =2^(D+1)
retarded has length D2 =D+1
"""
function eval_buff!(ev::MultipoleKFCEvaluator{D}, ret::MVector{D1,ComplexF64}, retarded::MVector{D2,ComplexF64}, idx_int::MVector{D,Int}, idx::Vararg{Int,D}) where {D,D1,D2}
    @inbounds for ip in axes(ev.Gps,2)    
        idx_int .= ev.ωconvMats[ip] * SA{Int}[idx...] + ev.ωconvOffs[ip]
        for id in 1:D+1
            retarded[id] = ev.Gps[id,ip](idx_int...)
        # transform to Keldysh, no matmul to avoid allocation
            for ik in 1:2^(D+1)
                ret[ik] += retarded[id] * ev.GR_to_GK[id,ik,ip]
            end
        end
    end
end


function (ev::MultipoleKFCEvaluator{D})(idx::Vararg{Int,D}) :: Vector{ComplexF64} where {D}
    ret = zeros(ComplexF64, 1, 2^(D+1))
    retarded = zeros(ComplexF64, D+1)
    @inbounds for ip in axes(ev.Gps,2)    
        idx_int = ev.ωconvMats[ip] * SA{Int}[idx...] + ev.ωconvOffs[ip]
        for id in 1:D+1
            retarded[id] = ev.Gps[id,ip](idx_int...)
        # transform to Keldysh, no matmul to avoid allocation
            for ik in 1:2^(D+1)
                ret[ik] += retarded[id] * ev.GR_to_GK[id,ik,ip]
            end
        end
    end
    return vec(ret)
end

function eval_interpol(ev::MultipoleKFCEvaluator{D}, ws::Vector{NTuple{D,LinRange{Float64}}}, w::Vararg{Float64,D}) where {D}
    ret = zeros(ComplexF64, 1, 2^(D+1))
    for ip in axes(ev.Gps,2)    
        retarded = zeros(ComplexF64, D+1)
        w_int = ev.ωconvMats[ip] * SA[w...]
        for id in 1:D+1
            retarded[id] = eval_interpol(ev.Gps[id,ip], ws[ip], w_int...)
        end
        # transform to Keldysh
        ret += transpose(retarded) * ev.GR_to_GK[:,:,ip]
    end
    return vec(ret)
end

function truncatable_matrix(sz::Tuple{Int,Int})
    URV = randn(Float64,sz)
    U,_,V = svd(URV)
    S = 2.0 .^ (0:-1:-minimum(sz)+1)
    return U*diagm(S)*V'
end

using BenchmarkTools
"""
Test against KFCEvaluator.
"""
function test_MultipoleKFCEvaluator_largeR()

    npt = 4
    D = npt-1
    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/4pt")
    channel = "p"
    Ops = TCI4Keldysh.dummy_operators(npt)
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(basepath; channel=channel)
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    ommax = 0.5
    R = 8
    G = TCI4Keldysh.FullCorrelator_KF(
        PSFpath,
        Ops;
        T=TCI4Keldysh.dir_to_T(PSFpath),
        ωconvMat=ωconvMat,
        ωs_ext=TCI4Keldysh.KF_grid(ommax, R, D),
        flavor_idx=1,
        γ=γ,
        sigmak=sigmak,
        emax=max(20.0, 3*ommax),
        emin=2.5*1.e-5,
        estep=50
    )

    Gev = MultipoleKFCEvaluator(G; nlevel=4, cutoff=1.e-6)
    Gref = KFCEvaluator(G)
    printstyled("== Test\n"; color=:blue)
    for _ in 1:1000
        idx = rand(1:2^R, 3)
        gval = Gev(idx...)
        refval = Gref(idx...)
        if maximum(abs.(gval .- refval)) / norm(gval) > 1.e-4
            @show idx
            @show norm(gval .- refval) / norm(gval)
            @warn "Large error from SVD truncations"
        end
    end
    printstyled("== Memory\n"; color=:blue)
    @show Base.summarysize(Gref)/1.e9
    @show Base.summarysize(Gev)/1.e9
    printstyled("== Benchmark\n"; color=:blue)
    @btime $Gev(rand(1:2^$R,3)...)
    @btime $Gref(rand(1:2^$R,3)...)
end

function speedup_MultipoleKFCEvaluator()

    npt = 4
    D = npt-1
    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/4pt")
    channel = "p"
    Ops = TCI4Keldysh.dummy_operators(npt)
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(basepath; channel=channel)
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    ommax = 0.65
    R = 14
    ωs_ext = TCI4Keldysh.KF_grid(ommax, R, D)
    G = TCI4Keldysh.FullCorrelator_KF(
        PSFpath,
        Ops;
        T=TCI4Keldysh.dir_to_T(PSFpath),
        ωconvMat=ωconvMat,
        ωs_ext=ωs_ext,
        flavor_idx=1,
        γ=γ,
        sigmak=sigmak,
        emax=max(20.0, 3*ommax),
        emin=2.5*1.e-5,
        estep=20
    )

    # time full correlator
    w = ntuple(i -> rand(1:length(ωs_ext[i])), 3)
    res1 = @benchmark evaluate_all_iK($G, $w...)
    display(res1)

    # for R=12, nlevel=4->5 yields a threefold speedup in evaluations
    Gev = MultipoleKFCEvaluator(G; nlevel=4, cutoff=1.e-6)
    # time hierarchical tuckers
    function __f()
        w = ntuple(i -> rand(1:length(ωs_ext[i])), 3)
        return Gev(w...)
    end

    res2 = @benchmark $__f()
    display(res2)
end


function test_MultipoleKFCEvaluator()

    npt = 4
    D = npt-1
    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/4pt")
    channel = "p"
    Ops = TCI4Keldysh.dummy_operators(npt)
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(basepath; channel=channel)
    ωconvMat = TCI4Keldysh.channel_trafo(channel)
    ommax = 0.5
    R = 4
    G = TCI4Keldysh.FullCorrelator_KF(
        PSFpath,
        Ops;
        T=TCI4Keldysh.dir_to_T(PSFpath),
        ωconvMat=ωconvMat,
        ωs_ext=TCI4Keldysh.KF_grid(ommax, R, D),
        flavor_idx=1,
        γ=γ,
        sigmak=sigmak,
        emax=max(20.0, 3*ommax),
        emin=2.5e-5,
        estep=50
    )

    Gref = TCI4Keldysh.precompute_all_values(G)
    cut = 1.e-3
    Gev = MultipoleKFCEvaluator(G; nlevel=2, cutoff=cut)
    maxref = maximum(abs.(Gref))
    for idx in Iterators.product(fill(1:2^R,D)...)
        gval = reshape(Gev(idx...), ntuple(_->2,D+1))
        refval = Gref[idx...,:,:,:,:]
        @assert maximum(abs.(gval .- refval)) / maxref < cut
    end
end

function test_hierarchical_tucker()
    center = randn(20,25,21)
    N_oms = (50,30,40)
    kernels = ntuple(i -> truncatable_matrix((N_oms[i],size(center,i))), ndims(center))

    cutoff = 1.e-5
    ht = TCI4Keldysh.HierarchicalTucker(center, kernels, 4; cutoff=cutoff)

    ref = TCI4Keldysh.contract_1D_Kernels_w_Adisc_mp(kernels, center)

    for _ in 1:100
        test_id = ntuple(i -> rand(1:N_oms[i]), 3)
        @assert abs(ht(test_id...) - ref[test_id...]) / ref[test_id...] <= cutoff * 1.e3
    end

    # test precompute_all_values
    ht_dense = TCI4Keldysh.precompute_all_values(ht)
    @assert maximum(abs.(ht_dense .- ref)) / maximum(abs.(ref)) < cutoff * 1.e3
end

function test_multipole_matrix()
    A = truncatable_matrix((50,30))
    A *= 1.e5
    Us, SVs = TCI4Keldysh.multipole_matrix(A, 1; cutoff=1.e-8)
    Anew = vcat([Us[i] * SVs[i] for i in eachindex(Us)]...) 
    @assert norm(A - Anew)/norm(A) < 1.e-8
end