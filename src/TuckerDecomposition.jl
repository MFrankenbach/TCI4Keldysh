abstract type AbstractTuckerDecomp{D} end

DO_TUCKER_FASTMATH() = true
macro TUCKER_FASTMATH(expr)
    if DO_TUCKER_FASTMATH()
        esc(:(@fastmath $expr))
    else
        esc(expr)
    end
end

"""
    TuckerDecomposition{T,D} <: AbstractTuckerDecomp{D}

Stores a D-dimensional tensor in form of a Tucker decomposition (TD). \n
It consists out of a center C and D legs L, such that for D=2 we get TDᵢⱼ = ∑ᵦᵧ Lᵢᵦ Lⱼᵧ Cᵦᵧ. 

# Pointwise evaluation
```julia-repl
julia> TD(1, 2)
2.0
```

# Slicing
```julia-repl
julia> TD[1, :]
4-element Vector{Float64}:
 1.0
 2.0
 3.0
 4.0
```
"""
mutable struct TuckerDecomposition{T,D} <: AbstractTuckerDecomp{D}                 ### D = number of frequency dimensions
    center      ::Array{T,D}          ### discrete spectral data; best: compactified with compactAdisc(...)
    legs        ::Vector{Matrix{T}}   ### broadening kernels
    sz          ::NTuple{D,Int}       ### size of center

    # informational:
    name        ::String
    ωs_center   ::Vector{Vector{Float64}}   ### internal frequencies corresponding to the arguments of the center
    ωs_legs     ::Vector{Vector{Float64}}   ### external frequencies of the legs
    idxs_center ::Vector{Vector{Int}}       ### internal indices corresponding to the arguments of the center
    idxs_legs   ::Vector{Vector{Int}}       ### external indices of the legs
    modified :: Bool # whether center of legs have been changed

    function TuckerDecomposition(
        center  ::Array{T1,D}        ,
        legs    ::Vector{Matrix{T2}} ;
        name       ::String                  = "",
        ωs_center  ::Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, D),
        ωs_legs    ::Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, D),
        idxs_center::Vector{Vector{Int}}     = Vector{Vector{Int}}(undef, D),
        idxs_legs  ::Vector{Vector{Int}}     = Vector{Vector{Int}}(undef, D)

    ) where{T1, T2, D}
    @DEBUG D == length(legs)        "number of legs inconsistent with dims(center)=$D."
    @DEBUG D == length(ωs_center)   "number of ωs_center inconsistent with dims(center)=$D."
    @DEBUG D == length(ωs_legs)     "number of ωs_legs inconsistent with dims(center)=$D."
    @DEBUG D == length(idxs_center) "number of idxs_center inconsistent with dims(center)=$D."
    @DEBUG D == length(idxs_legs)   "number of idxs_legs inconsistent with dims(center)=$D."
        return new{promote_type(T1,T2),D}(center, legs, size(center), name, ωs_center, ωs_legs, idxs_center, idxs_legs, false)
    end

end


function Base.:getindex(
    td  :: TuckerDecomposition{T,D},
    w   :: Vararg{Union{Int, Colon, Vector{Int}, UnitRange}, D} # , UnitRange
    )   :: Union{T, AbstractArray{T}} where {T,D}
    return _getindex(td, w...)
end





function _getindex(
    td :: TuckerDecomposition{T,D},
    w  :: Vararg{Union{Int, Colon, Vector{Int}, UnitRange}, D} # , UnitRange
    )  :: Union{T, AbstractArray{T}} where {T,D}

    function check_bounds_int(d)
        if !(1<=w[d]<=size(td.legs[d], 1))
            throw(BoundsError(td, w))
        end
        return nothing
    end
    function check_bounds_ur(d)
        if !(1<=firstindex(w[d])<lastindex(w[d])<=size(td.legs[d], 1))
            throw(BoundsError(td, w))
        end
        return nothing
    end

    legs_new = [td.legs[i] for i in 1:D]# [qtt.tci.T[i][:,qw[i],:] for i in 1:DR]
    # bounds check
    #@assert all(1 .<= w .<= 2^R)

    #qw = Array{Union{UnitRange,Colon}}(undef, DR)
    for d in 1:D
        if typeof(w[d]) == Int
            @boundscheck check_bounds_int(d)
                
            legs_new[d] = legs_new[d][w[d]:w[d],:]    # maybe replace this by a view => no copying needed
            
        elseif typeof(w[d]) != Colon
            @boundscheck check_bounds_ur(d)
            
            legs_new[d] = legs_new[d][w[d],:]
            
        end
     end
    
    result = contract_1D_Kernels_w_Adisc_mp(legs_new, td.center)

    return result
end


function _getindex(td::TuckerDecomposition{T,D}, idx::Vararg{Int,D})  ::T where {T,D}
    res = td.center
    sz_center = size(res)
    for i in 1:D
        res = view(td.legs[i], idx[i]:idx[i], :) * reshape(res, (sz_center[i], prod(sz_center[i+1:D])))
    end
    return res[1]
end


function (td::TuckerDecomposition{T,D})(idx::Vararg{Int,D})  ::T where {T,D}
    res = td.center
    for i in 1:D
        #println("i: ", i)
        #res = view(td.legs[i], idx[i]:idx[i], :) * reshape(res, (td.sz[i], prod(td.sz[i+1:D])))
        #res = td.legs[i][idx[i], :]' * reshape(res, (td.sz[i], prod(td.sz[i+1:D])))
        j = D - i + 1
        #res = reshape(res, (prod(td.sz[1:j-1]), td.sz[j])) * view(td.legs[j], idx[j], :)
        res = reshape(res, (prod(td.sz[1:j-1]), td.sz[j])) * td.legs[j][idx[j], :]
    end
    #println("length(res): ", length(res))
    return res[1]
end


"""
TODO: prune tucker center for D<3
"""
function (td::TuckerDecomposition{T,1})(::Int, idx::Vararg{Int,1})  ::T where {T}
    return conj(td.legs[1][idx[1],:]') * td.center
end

function (td::TuckerDecomposition{T,1})(idx::Vararg{Int,1})  ::T where {T}
    #return mapreduce(*,+,view(td.legs[1], idx[1], :),td.center)
    #return dot(view(td.legs[1], idx[1], :),td.center)
    # return LinearAlgebra.BLAS.dot(view(td.legs[1], idx[1], :),td.center)
    return conj(td.legs[1][idx[1],:]') * td.center
end

"""
TODO: prune tucker center for D<3
"""
function (td::TuckerDecomposition{T,2})(::Int, idx::Vararg{Int,2})  ::T where {T}
    return conj(td.legs[1][idx[1],:]') * ( td.center * td.legs[2][idx[2],:])
end

function (td::TuckerDecomposition{T,2})(idx::Vararg{Int,2})  ::T where {T}
    # return LinearAlgebra.BLAS.dot(view(td.legs[1], idx[1], :), LinearAlgebra.BLAS.gemv('N',td.center,view(td.legs[2], idx[2], :)))
    #return dot(view(td.legs[1], idx[1], :),td.center,view(td.legs[2], idx[2], :))
    return conj(td.legs[1][idx[1],:]') * ( td.center * td.legs[2][idx[2],:])
end

#=
"""
Pointwise tucker decomposition evaluation with BLAS.
A bit slower than direct summation.
"""
function (td::TuckerDecomposition{T,3})(idx::Vararg{Int,3})  ::T where {T}
    @inbounds N = size(td.legs[3])[2]
    # cenleg2 = [LinearAlgebra.BLAS.gemv('N',view(td.center, :,:,i),view(td.legs[2], idx[2], :)) for i in 1:N]
    # @show typeof.(cenleg2)
    @inbounds @views temp = [LinearAlgebra.BLAS.dotu(td.legs[1][idx[1],:], LinearAlgebra.BLAS.gemv('N',td.center[:,:,i],td.legs[2][idx[2],:])) for i in 1:N]
    # temp = [LinearAlgebra.BLAS.dotu(view(td.legs[1], idx[1], :), LinearAlgebra.BLAS.gemv('N',view(td.center, :,:,i),view(td.legs[2], idx[2], :))) for i in 1:N]
    @inbounds return LinearAlgebra.BLAS.dotu(temp, td.legs[3][idx[3],:])
end
=#

"""
Pointwise eval. by direct summation.
Tried linear indexing for tucker center, yields no improvement.
* tucker_cut: In the Tucker center, elements (i,j,k) with i+j+k > prune_idx are neglected
"""
function (td::TuckerDecomposition{T,3})(tucker_cut::Int, idx::Vararg{Int,3}) :: T where {T}
    ret = zero(T)    

    n1, n2, n3 = size(td.center)
    @TUCKER_FASTMATH @inbounds for k in 1:n3
        ret3 = zero(T)
        cutk = tucker_cut - k
        for j in 1:n2
            ret2 = zero(T)
            for i in 1:min(cutk - j, n1)
                ret2 += td.legs[1][idx[1], i] * td.center[i, j, k]
            end
            ret3 += ret2 * td.legs[2][idx[2], j] 
        end
        ret += ret3 * td.legs[3][idx[3], k]
    end

    return ret
end

"""
Pointwise eval. by direct summation.
Tried linear indexing for tucker center, yields no improvement.
"""
function (td::TuckerDecomposition{T,3})(idx::Vararg{Int,3}) :: T where {T}
    ret = zero(T)    

    n1, n2, n3 = size(td.center)
    @TUCKER_FASTMATH @inbounds for k in 1:n3
        ret3 = zero(T)
        for j in 1:n2
            ret2 = zero(T)
            for i in 1:n1
                ret2 += td.legs[1][idx[1], i] * td.center[i, j, k]
            end
            ret3 += ret2 * td.legs[2][idx[2], j] 
        end
        ret += ret3 * td.legs[3][idx[3], k] 
    end

    return ret
end

"""
CARFUL: Legs are TRANSPOSED, i.e. columns are contracted with center.
"""
function eval_tucker(center::Array{T,D}, legs::Vector{Matrix{T}}, idx::Vararg{Int,D})  ::T where {T,D}
    res = center
    sz = size(res)
    for i in 1:D
        j = D - i + 1
        res = reshape(res, (prod(sz[1:j-1]), sz[j])) * legs[j][:,idx[j]]
    end
    return res[1]
end

function eval_tucker(center::Array{T,D}, legs_idx::Vector{Vector{T}})  ::T where {T,D}
    res = center
    sz = size(res)
    for i in 1:D
        j = D - i + 1
        res = reshape(res, (prod(sz[1:j-1]), sz[j])) * legs_idx[j]
    end
    return res[1]
end

function eval_tucker(center::Array{T,D}, legs_idx::NTuple{D,AbstractVector{T}})  ::T where {T,D}
    res = center
    sz = size(res)
    for i in 1:D
        j = D - i + 1
        res = reshape(res, (prod(sz[1:j-1]), sz[j])) * legs_idx[j]
    end
    return res[1]
end

"""
Pointwise eval. of a tucker decomposition by direct summation.
Tried linear indexing for tucker center, yields no improvement.
CARFUL: Legs are TRANSPOSED, i.e. columns are contracted with center.
"""
function eval_tucker(center::Array{T,3}, legs::Vector{Matrix{T}}, idx::Vararg{Int,3}) :: T where {T}
    ret = zero(T)    

    n1, n2, n3 = size(center)
    @TUCKER_FASTMATH @inbounds for k in 1:n3
        ret3 = zero(T)
        for j in 1:n2
            ret2 = zero(T)
            for i in 1:n1
                ret2 += legs[1][i, idx[1]] * center[i, j, k]
            end
            ret3 += ret2 * legs[2][j, idx[2]] 
        end
        ret += ret3 * legs[3][k, idx[3]] 
    end

    return ret
end

function eval_tucker(tucker_cut::Int, center::Array{T,3}, legs::Vector{Matrix{T}}, idx::Vararg{Int,3}) :: T where {T}
    ret = zero(T)    

    n1, n2, n3 = size(center)
    @TUCKER_FASTMATH @inbounds for k in 1:n3
        ret3 = zero(T)
        cutk = tucker_cut - k
        for j in 1:n2
            ret2 = zero(T)
            for i in 1:min(cutk - j, n1)
                ret2 += legs[1][i, idx[1]] * center[i, j, k]
            end
            ret3 += ret2 * legs[2][j, idx[2]] 
        end
        ret += ret3 * legs[3][k, idx[3]]
    end

    return ret
end


# TODO: Unnecessary because of generic method
"""
* legs: corresponds to legs[i][idx[i],:] in other eval_tucker function
"""
function eval_tucker(center::Array{T,3}, legs::Vector{Vector{T}}) :: T where {T}
    ret = zero(T)    

    n1, n2, n3 = size(center)
    @TUCKER_FASTMATH @inbounds for k in 1:n3
        ret3 = zero(T)
        for j in 1:n2
            ret2 = zero(T)
            for i in 1:n1
                ret2 += legs[1][i] * center[i, j, k]
            end
            ret3 += ret2 * legs[2][j]
        end
        ret += ret3 * legs[3][k]
    end

    return ret
end

function eval_tucker_mat(center::Array{T,3}, legs::Vector{Vector{T}}) :: T where {T}
    n1, n2, n3 = size(center)
    l1c = reshape(transpose(legs[1]) * reshape(center, n1, n2*n3), n2, n3)
    return transpose(legs[2]) * l1c * legs[3]
end

function eval_tucker_mat(center::Array{T,3}, legs::NTuple{3,AbstractVector{T}}) :: T where {T}
    n1, n2, n3 = size(center)
    l1c = reshape(transpose(legs[1]) * reshape(center, n1, n2*n3), n2, n3)
    return transpose(legs[2]) * l1c * legs[3]
end

function eval_tucker_mat(center::Array{T,D}, legs::NTuple{D,AbstractVector{T}}) :: T where {T,D}
    return eval_tucker(center, legs)
end

"""
* legs: corresponds to legs[i][idx[i],:] in other eval_tucker function
"""
function eval_tucker(center::Array{T,3}, legs::NTuple{3,AbstractVector{T}}) :: T where {T}
    ret = zero(T)
    n1, n2, n3 = size(center)
    v1, v2, v3 = legs
    @TUCKER_FASTMATH @inbounds for k in 1:n3
        ret3 = zero(T)
        for j in 1:n2
            ret2 = zero(T)
            for i in 1:n1
                ret2 += v1[i] * center[i, j, k]
            end
            ret3 += ret2 * v2[j]
        end
        ret += ret3 * v3[k]
    end

    return ret
end

"""
Struct to accelerate pointwise evaluation of 3D tucker decompositions.
Does NOT yield improvement.
"""
mutable struct TuckerEvaluator3D{T}
    td::TuckerDecomposition{T,3}
    inter2D::Array{T,2}
    inter1D::Array{T,1}

    function TuckerEvaluator3D(td::TuckerDecomposition{T,3}) where {T}
        _, n2, n3 = size(td.center)
        inter2D = zeros(T, n2, n3)
        inter1D = zeros(T, n3)
        return new{T}(td, inter2D, inter1D)
    end
end


function (te::TuckerEvaluator3D{T})(idx::Vararg{Int,3}) :: T where {T}
    v1 = te.td.legs[1][idx[1], :]
    v2 = te.td.legs[2][idx[2], :]
    v3 = te.td.legs[3][idx[3], :]

    _, n2, n3 = size(te.td.center)

    # Contract over the first dimension
    @inbounds @views for j in 1:n2
        for k in 1:n3
            te.inter2D[j, k] = sum(v1 .* te.td.center[:, j, k])
        end
    end

    # Contract over the second dimension
    @inbounds @views for k in 1:n3
        te.inter1D[k] = sum(v2 .* te.inter2D[:, k])
    end

    # Contract over the third dimension
    ret = sum(v3 .* te.inter1D)

    return ret
end

"""
Return number of elements el in tucker center with abs(el) < zero_thresh * maximum(abs.(tucker.center))
"""
function sparsity(td::TuckerDecomposition, zero_thresh::Float64=1.e-14)
    maxcen = maximum(abs.(td.center))
    sparsecount = 0
    relthresh = maxcen * zero_thresh
    for i in eachindex(td.center)
        if abs(td.center[i]) < relthresh
            sparsecount += 1
        end
    end
    return sparsecount
end

"""
Upper bound on maximum value of the tucker decomposition (Hoelder):
|td(ω)| ≤ |K(ω - ⋅)|_p * |S|_q
where |K(ω - ⋅)|_p factorizes
"""
function upperbound(td::TuckerDecomposition, p::Float64=1.0)
    q = 1.0/(1.0 - 1.0/p)    
    if abs(p-1.0)<1.e-6
        q = Inf
    end

    Sq = norm(td.center, q)
    
    return Sq * legnorm(td, p)
end

function lowerbound(td::TuckerDecomposition{T,D}) where {T,D}
    idx = ntuple(i -> div(size(td.legs[i],1),2), D)
    return abs(td(idx...))
end

function legnorm(td::TuckerDecomposition, p=1.0)
    return legnorm(td.legs)
end

function legnorm(legs::Vector{Matrix{T}}, p=1.0; transpose_legs=false) where {T}
    lps = Float64[]
    slice_fun  = i -> ifelse(transpose_legs, (Colon(), i), (i, Colon()))
    for leg in legs
        lp = maximum([norm(leg[slice_fun(i)...], p) for i in axes(leg,1)])
        push!(lps, lp)
    end
    return prod(lps)
end

function compute_tucker_cut(center::Array{T,D}, legs::Vector{Matrix{T}}, GFmin::Float64, tucker_cutoff::Float64, p::Float64=2.0; transpose_legs=false) where {T,D}
    q = 1.0/(1.0 - 1.0/p)
    # check whether legs are transposed
    Knorm = legnorm(legs, p; transpose_legs=transpose_legs)
    # see how far tucker center can be pruned
    cen_  = center
    cen_q = abs.(center) .^ q
    sz_cen = size(cen_)
    prune_idx = sum(sz_cen)
    lower_thresh = GFmin * tucker_cutoff

    idx_sums = zeros(Float64, sum(sz_cen))
    for ic in CartesianIndices(cen_)
        idx_sums[sum(Tuple(ic))] += cen_q[ic]
    end

    sums_acc = accumulate(+, reverse(idx_sums))
    cuts = reverse(Knorm .* (sums_acc .^ (1.0/q)))
    prune_idx = findfirst(s -> s<=lower_thresh, cuts)
    if isnothing(prune_idx)
        @warn "Tucker cutoff is pessimal"
        prune_idx = sum(sz_cen)
    elseif prune_idx==0
        prune_idx = 1
    end

    if DEBUG_RAM()
        cen_q = nothing
        sums_acc = nothing
        cuts = nothing
        idx_sums = nothing
    end
    return prune_idx
end

function compute_tucker_cut(tucker::TuckerDecomposition{T,D}, GFmin::Float64, tucker_cutoff::Float64, p::Float64=2.0) where {T,D}
    return compute_tucker_cut(tucker.center, tucker.legs, GFmin, tucker_cutoff, p)
end