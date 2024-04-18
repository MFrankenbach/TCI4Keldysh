

"""    
    getAcont_mp(ωdisc::Vector{Float64}, Adisc::Array{Float64}, sigmak::Vector{Float64}, γ::Float64; kwargs...)

Broaden multipoint partial spectral function (PSF) data with log-Gaussian-type and linear broadening. 

# Arguments
1. ωdisc   ::Vector{Float64} :      Logarithimic frequency bins. 
                                    Here the original frequency values from the differences
                                    b/w energy eigenvalues are shifted to the closest bins.
2. Adisc   ::Array{Float64}  :      Spectral function in layout |ωdisc|x|sigmak|.
3. sigmak  ::Vector{Float64} :      Sensitivity of logarithmic position of spectral
                                    weight to z-shift, i.e. |d[log(|ω|)]/dz|. These values will
                                    be used to broaden discrete data. (σ _{ij} or σ_k in
                                    Lee2016.)
4. γ       ::Float64         :      Parameter for secondary linear broadening kernel. (γ in Lee2016.)

# Keyword arguments 
This function accepts the same options as for "getAcont". A difference
from the standard usage of "getAcont" is that the default values of
'emin', 'emax', and 'estep' are set to 1e-6, 10, and 16, respectively.
This is necessary since this function broadens two- or higher dimensional
data; with the default values of "getAcont", your memory might blow up.

# Returns
1. ωcont   ::Vector{Float64}:    continous frequencies
2. Acont   ::Vector{Float64}:    continous spectral data


# Output
1. ocont ::Vector{Float64} :    One-dimensional frequency grid for "Acont", used to
                                describe the frequency space for "Acont" along all dimensions.
2. Acont ::Vector{Float64} :    Multi-dimensional array that represent the broadened spectral data.
"""
function getAcont_mp(
    ωdisc   ::Vector{Float64},  
    Adisc   ::Array{Float64},   
    sigmak  ::Vector{Float64},
    γ       ::Float64           
    ; 
    kwargs...                   # keyword arguments to be passed to 1D broadening function
    )

    D, _, Adisc, Kernels, ωcont = _prepare_broadening_mp(ωdisc, Adisc, sigmak, γ; kwargs...)
    Acont = contract_1D_Kernels_w_Adisc_mp(Kernels, Adisc)

    return ωcont, Acont
end


function _prepare_broadening_mp(
    ωdisc   ::Vector{Float64},  # Logarithimic frequency bins. 
                                # Here the original frequency values from the differences
                                # b/w energy eigenvalues are shifted to the closest bins.
                                # 
    Adisc   ::Array{Float64},   # Spectral function in layout |ωdisc|x|sigmak|.
    sigmak  ::Vector{Float64},  # Sensitivity of logarithmic position of spectral
                                # weight to z-shift, i.e. |d[log(|omega|)]/dz|. These values will
                                # be used to broaden discrete data. (\sigma_{ij} or \sigma_k in
                                # Lee2016.)
    γ       ::Float64           # Parameter for secondary linear broadening kernel. (\gamma
                                # in Lee2016.)
    ; 
    ωconts  ::NTuple{D,Vector{Float64}},
    kwargs...
) where{D}
    # Check if sigmak is a single number
    if length(sigmak) != 1
        error("ERR: Logarithmic broadening width parameter 'sigmak' should be a single number.")
    end
    # remove singleton dimensions
    Adisc = dropdims(Adisc,dims=tuple(findall(size(Adisc).==1)...))

    # dimensions that need to be broadened
    if !(1 <= D <=3)
        throw(ArgumentError("Only 1-/2-/3-dimensional Adisc supported."))
    end
    ωdiscs = ntuple(x -> ωdisc, D)
    #sz = [prod(size(x)) for x in ωdiscs]
    ωcont_lengths = length.(ωconts)
    ωcont_largest = ωconts[argmax(ωcont_lengths)]
    # Broadening kernels
    ωcont, Kernel = getAcont(ωdisc, Matrix{Float64}(LinearAlgebra.I, (ones(Int, 2).*length(ωdisc))...), sigmak .+ zeros(length(ωdisc)), γ; ωcont=ωcont_largest, kwargs...)
    
    # Delete rows/columns that contain only zeros
    AdiscIsZero_oks, ωdiscs, Adisc = compactAdisc(ωdisc, Adisc)
    ishifts = ntuple(i -> div(length(ωcont_largest) - length(ωconts[i]), 2), D)
    Kernels = [Kernel[1+ishifts[i]:end-ishifts[i], .!AdiscIsZero_oks[i]] for i in 1:D]
    return D, ωdiscs, Adisc, Kernels, ωcont
end


"""
    BroadenedPSF{D} <: AbstractTuckerDecomp{D}

Stores a broadened D-dimensional PSF in form of a Tucker decomposition. \n
Kernels are precomputed, such that e.g. for D=2 we get Acontᵢⱼ = ∑ᵦᵧ Kᵢᵦ Kⱼᵧ Adiscᵦᵧ. \n
A BroadenedPSF can be evaluated at indices 1:length(ωconts[d]) with 1≤d≤D.

# Pointwise evaluation
```julia-repl
julia> psf(1, 2)
2.0
```

# Slicing
```julia-repl
julia> psf[1, :]
4-element Vector{Float64}:
 1.0
 2.0
 3.0
 4.0
```
"""
mutable struct BroadenedPSF{D} <: AbstractTuckerDecomp{D}                 ### D = number of frequency dimensions
    Adisc   ::Array{Float64,D}          ### discrete spectral data; best: compactified with compactAdisc(...)
    ωdiscs  ::Vector{Vector{Float64}}   ### discrete frequencies for all D dimensions
    Kernels ::Vector{Matrix{Float64}}   ### broadening kernels
    ωconts  ::NTuple{D,Vector{Float64}} ### continous frequencies for all D dimensions
    sz      ::NTuple{D,Int}             ### size of Adisc


    function BroadenedPSF(
        Adisc   ::Array{Float64,D}          ,
        ωdiscs  ::Vector{Vector{Float64}}   ,
        Kernels ::Vector{Matrix{Float64}} ,
        ωconts  ::NTuple{D,Vector{Float64}} ,
        sz      ::NTuple{D,Int}             
    ) where{D}
        return new{D}(Adisc, ωdiscs, Kernels, ωconts, sz)
    end

    function BroadenedPSF(
        ωdisc   ::Vector{Float64},  # Logarithimic frequency bins. 
                                    # Here the original frequency values from the differences
                                    # b/w energy eigenvalues are shifted to the closest bins.
                                    # 
        Adisc   ::Array{Float64,D}, # Spectral function in layout |ωdisc|x|sigmak|.
        sigmak  ::Vector{Float64},  # Sensitivity of logarithmic position of spectral
                                    # weight to z-shift, i.e. |d[log(|omega|)]/dz|. These values will
                                    # be used to broaden discrete data. (\sigma_{ij} or \sigma_k in
                                    # Lee2016.)
        γ       ::Float64           # Parameter for secondary linear broadening kernel. (\gamma
                                    # in Lee2016.)
        ; 
        ωconts  ::NTuple{D,Vector{Float64}},
        kwargs...
    ) where{D}
        @TIME _, ωdiscs, Adisc, Kernels, _ = _prepare_broadening_mp(ωdisc, Adisc, sigmak, γ; ωconts, kwargs...) "Prepare broadening."
        sz = size(Adisc)
        return new{D}(Adisc, ωdiscs, Kernels, ωconts, sz)
    end
end

function (psf::BroadenedPSF{D})(idx::Vararg{Int,D})  ::Float64 where {D}
    res = psf.Adisc
    for i in 1:D
        #println("i: ", i)
        #res = view(psf.Kernels[i], idx[i]:idx[i], :) * reshape(res, (psf.sz[i], prod(psf.sz[i+1:D])))
        #res = psf.Kernels[i][idx[i], :]' * reshape(res, (psf.sz[i], prod(psf.sz[i+1:D])))
        j = D - i + 1
        #res = reshape(res, (prod(psf.sz[1:j-1]), psf.sz[j])) * view(psf.Kernels[j], idx[j], :)
        res = reshape(res, (prod(psf.sz[1:j-1]), psf.sz[j])) * psf.Kernels[j][idx[j], :]
    end
    #println("length(res): ", length(res))
    return res[1]
end

function (psf::BroadenedPSF{1})(idx::Vararg{Int,1})  ::Float64# where {D}
    #return mapreduce(*,+,view(psf.Kernels[1], idx[1], :),psf.Adisc)
    #return dot(view(psf.Kernels[1], idx[1], :),psf.Adisc)
    return LinearAlgebra.BLAS.dot(view(psf.Kernels[1], idx[1], :),psf.Adisc)
end

function (psf::BroadenedPSF{2})(idx::Vararg{Int,2})  ::Float64# where {D}
    return LinearAlgebra.BLAS.dot(view(psf.Kernels[1], idx[1], :), LinearAlgebra.BLAS.gemv('N',psf.Adisc,view(psf.Kernels[2], idx[2], :)))
    #return dot(view(psf.Kernels[1], idx[1], :),psf.Adisc,view(psf.Kernels[2], idx[2], :))
end

function (psf::BroadenedPSF{3})(idx::Vararg{Int,3})  ::Float64 #where {D}
    N = size(psf.Kernels[3])[2]
    #temp = [dot(view(psf.Kernels[1], idx[1], :),view(psf.Adisc, :,:,i),view(psf.Kernels[2], idx[2], :)) for i in 1:N]
    #return dot(temp,view(psf.Kernels[3], idx[3], :))
    temp = [LinearAlgebra.BLAS.dot(view(psf.Kernels[1], idx[1], :), LinearAlgebra.BLAS.gemv('N',view(psf.Adisc, :,:,i),view(psf.Kernels[2], idx[2], :))) for i in 1:N]
    return LinearAlgebra.BLAS.dot(temp,view(psf.Kernels[3], idx[3], :))
end


function Base.:getindex(
    psf :: BroadenedPSF{D},
    w   :: Vararg{Union{Int, Colon}, D} # , UnitRange
    )   :: Union{Float64, AbstractArray{Float64}} where {D}

    function check_bounds_int(d)
        if !(1<=w[d]<=size(psf.Kernels[d], 1))
            throw(BoundsError(psf, w))
        end
        return nothing
    end
    function check_bounds_ur(d)
        if !(1<=firstindex(w[d])<lastindex(w[d])<=size(psf.Kernels[d], 1))
            throw(BoundsError(psf, w))
        end
        return nothing
    end

    kernels_new = [psf.Kernels[i] for i in 1:D]# [qtt.tci.T[i][:,qw[i],:] for i in 1:DR]
    # bounds check
    #@assert all(1 .<= w .<= 2^R)

    #qw = Array{Union{UnitRange,Colon}}(undef, DR)
    for d in 1:D
        if typeof(w[d]) == Int
            @boundscheck check_bounds_int(d)
                
            kernels_new[d] = kernels_new[d][w[d]:w[d],:]
            
        elseif typeof(w[d]) == UnitRange
            @boundscheck check_bounds_ur(d)

            kernels_new[d] = kernels_new[d][w[d],:]
            
        end
     end
    
    result = contract_1D_Kernels_w_Adisc_mp(kernels_new, psf.Adisc)

    return result
end