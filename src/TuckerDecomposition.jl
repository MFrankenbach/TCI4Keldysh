abstract type AbstractTuckerDecomp{D} end

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
        return new{promote_type(T1,T2),D}(center, legs, size(center), name, ωs_center, ωs_legs, idxs_center, idxs_legs)
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


function (td::TuckerDecomposition{T,1})(idx::Vararg{Int,1})  ::T where {T}
    #return mapreduce(*,+,view(td.legs[1], idx[1], :),td.center)
    #return dot(view(td.legs[1], idx[1], :),td.center)
    # return LinearAlgebra.BLAS.dot(view(td.legs[1], idx[1], :),td.center)
    return conj(td.legs[1][idx[1],:]') * td.center
end


function (td::TuckerDecomposition{T,2})(idx::Vararg{Int,2})  ::T where {T}
    # return LinearAlgebra.BLAS.dot(view(td.legs[1], idx[1], :), LinearAlgebra.BLAS.gemv('N',td.center,view(td.legs[2], idx[2], :)))
    #return dot(view(td.legs[1], idx[1], :),td.center,view(td.legs[2], idx[2], :))
    return conj(td.legs[1][idx[1],:]') * ( td.center * td.legs[2][idx[2],:])
end

#=
function (td::TuckerDecomposition{T,3})(idx::Vararg{Int,3})  ::T where {T}
    N = size(td.legs[3])[2]
    #temp = [dot(view(td.legs[1], idx[1], :),view(td.center, :,:,i),view(td.legs[2], idx[2], :)) for i in 1:N]
    #return dot(temp,view(td.legs[3], idx[3], :))
    temp = [LinearAlgebra.BLAS.dot(view(td.legs[1], idx[1], :), LinearAlgebra.BLAS.gemv('N',view(td.center, :,:,i),view(td.legs[2], idx[2], :))) for i in 1:N]
    return LinearAlgebra.BLAS.dot(temp,view(td.legs[3], idx[3], :))
end
=#

"""
Pointwise eval. by direct summation
"""
# function (td::TuckerDecomposition{T,3})(idx::Vararg{Int,3}) :: T where {T}
#     ret = zero(T)    

#     n1, n2, n3 = size(td.center)
#     @inbounds for k in 1:n3
#         for j in 1:n2
#             for i in 1:n1
#                 ret += td.legs[1][idx[1], i] * td.legs[2][idx[2], j] * td.legs[3][idx[3], k] * td.center[i, j, k]
#             end
#         end
#     end

#     return ret
# end

"""
Pointwise eval. by direct summation, optimized.
Tried linear indexing for tucker center, yields no improvement.
"""
function (td::TuckerDecomposition{T,3})(idx::Vararg{Int,3}) :: T where {T}
    ret = zero(T)    

    n1, n2, n3 = size(td.center)
    @inbounds for k in 1:n3
        k3 = td.legs[3][idx[3], k]
        ret3 = zero(T)
        for j in 1:n2
            ret2 = zero(T)
            k2 = td.legs[2][idx[2], j]
            for i in 1:n1
                ret2 += td.legs[1][idx[1], i] * td.center[i, j, k]
            end
            ret3 += ret2 * k2
        end
        ret += ret3 * k3
    end

    return ret
end

"""
Struct to accelerate pointwise evaluation of 3D tucker decompositions
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