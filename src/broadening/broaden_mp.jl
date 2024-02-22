

"""
< Description >
[ocont,Acont] = getAcont_mp (ωdisc,Adisc,sigmak,γ [,option])

Broaden multipoint partial spectral function (PSF) data with log-Gaussian
-type broadening used by the "getAcont" function. 

< Keyword arguments >
This function accepts the same options as for "getAcont". A difference
from the standard usage of "getAcont" is that the default values of
'emin', 'emax', and 'estep' are set to 1e-6, 10, and 16, respectively.
This is necessary since this function broadens two- or higher dimensional
data; with the default values of "getAcont", your memory might blow up.

< Output >
ocont : [numeric] One-dimensional frequency grid for "Acont", used to
      describe the frequency space for "Acont" along all dimensions.
Acont : [numeric] Multi-dimensional array that represent the broadened
      spectral data.
"""
function getAcont_mp(
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
    kwargs...
    )
    # Broaden multipoint partial spectral function (PSF) data with log-Gaussian
    # -type broadening used by the "getAcont" function.




    D, _, Adisc, Kernels, ωcont = prepare_broadening_mp(ωdisc, Adisc, sigmak, γ; kwargs...)
    Acont = contract_Kernels_w_Adisc_mp(Kernels, Adisc)

    return ωcont, Acont
end


function prepare_broadening_mp(
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
Contracts kernels with Adisc

For 3p correlators we e.g. get K[i,a] K[j,b] Adisc[a,b]
"""
function contract_Kernels_w_Adisc_mp(Kernels, Adisc)
    sz = [size(Adisc)...]
    D = ndims(Adisc)

    ##########################################################
    ### EFFICIENCY IN TERMS OF   CPU TIME: 🙈     RAM: 😄  ###
    ##########################################################
    #if D == 1
    #    M1 = Kernels[1]
    #    @tullio Acont[i] := M1[i,a] * Adisc[a]
    #elseif D == 2
    #    M1, M2 = Kernels[1], Kernels[2]
    #    @tullio Acont[i,j] := M1[i,a] * M2[j,b] * Adisc[a,b]
    #else     # D == 3
    #    M1, M2, M3 = Kernels[1], Kernels[2], Kernels[3]
    #    @tullio Acont[i,j,k] := M1[i,a] * M2[j,b] * M3[k,c] * Adisc[a,b,c]
    #end

    
    ##########################################################
    ### EFFICIENCY IN TERMS OF   CPU TIME: 😄     RAM: 🙈  ###
    ##########################################################
    Acont = copy(Adisc)  # Initialize
    for it1 in 1:D
        #println(Dates.format(Dates.now(), "HH:MM:SS"), ":\tit1 = ", it1)
        Acont = reshape(Acont, (sz[it1], prod(sz) ÷ sz[it1]))
        Acont = Kernels[it1] * Acont

        #println(Dates.format(Dates.now(), "HH:MM:SS"), ":\tconvolution [done]")
        sz[it1] = size(Kernels[it1])[1]
        Acont = reshape(Acont, (sz[it1], sz[[it1+1:end; 1:it1-1]]...))
        if D>1
            Acont = permutedims(Acont, (collect(2:D)..., 1))
        end
        #println(Dates.format(Dates.now(), "HH:MM:SS"), ":\tpermutation [done]")
        #GC.gc()
    end

    return Acont
end


"""
Contracts retarded kernels with Adisc and deduces all fully-retarded kernels

For 3p correlators we e.g. get K^{R/A}[i,a] K^{R/A}[j,b] Adisc[a,b]

We need
for 2p:     K^[1](ω₁,ω₂)        =       K^R(ω₁)
            K^[2](ω₁,ω₂)        =       K^A(ω₁)                 = c.c.of first line
for 3p:     K^[1](ω₁,ω₂,ω₃)     =       K^R(ω₁)K^R(ω₂)         \\=2p result x K^R(ω₂)
            K^[2](ω₁,ω₂,ω₃)     =       K^A(ω₁)K^R(ω₂)         /
            K^[3](ω₁,ω₂,ω₃)     =       K^A(ω₁)K^A(ω₂)          = c.c.of first line
for 4p:     K^[1](ω₁,ω₂,ω₃,ω₄)  =       K^R(ω₁)K^R(ω₂)K^R(ω₃)  \\
            K^[2](ω₁,ω₂,ω₃,ω₄)  =       K^A(ω₁)K^R(ω₂)K^R(ω₃)  |=3p result x K^R(ω₃)
            K^[3](ω₁,ω₂,ω₃,ω₄)  =       K^A(ω₁)K^A(ω₂)K^R(ω₃)  /
            K^[3](ω₁,ω₂,ω₃,ω₄)  =       K^A(ω₁)K^A(ω₂)K^A(ω₃)   = c.c.of first line
"""
function contract_KF_Kernels_w_Adisc_mp(Kernels, Adisc)
    sz = [size(Adisc)...]
    D = ndims(Adisc)
    
    ##########################################################
    ### EFFICIENCY IN TERMS OF   CPU TIME: 😄     RAM: 🙈  ###
    ##########################################################
    Acont = copy(Adisc)  # Initialize
    for it1 in 1:D
        #println(Dates.format(Dates.now(), "HH:MM:SS"), ":\tit1 = ", it1)
        Acont = reshape(Acont, (sz[it1], prod(sz) ÷ sz[it1] * it1))
        Acont = Kernels[it1] * Acont
        sz[it1] = size(Kernels[it1])[1]
        Acont = reshape(Acont, ((sz[it1], prod(sz) ÷ sz[it1], it1)))
        Acont = cat(Acont, conj.(Acont[:,:,1]), dims=3)

        #println(Dates.format(Dates.now(), "HH:MM:SS"), ":\tconvolution [done]")
        #Acont = reshape(Acont, (sz[it1], sz[[it1+1:end; 1:it1-1]]...))
        if D>1
            Acont = permutedims(Acont, (collect(2:D)..., 1, D+1))
        end
        #println(Dates.format(Dates.now(), "HH:MM:SS"), ":\tpermutation [done]")
        #GC.gc()
    end

    return Acont
end

struct BroadenedPSF{D} <: AbstractTuckerDecomp{D}                 ### D = number of frequency dimensions
    Adisc   ::Array{Float64,D}          ### discrete spectral data; best: compactified with compactAdisc(...)
    ωdiscs  ::Vector{Vector{Float64}}   ### discrete frequencies for all D dimensions
    Kernels ::Vector{Matrix{Float64}}   ### broadening kernels
    ωconts  ::NTuple{D,Vector{Float64}} ### continous frequencies for all D dimensions
    sz      ::NTuple{D,Int}             ### size of Adisc


    function BroadenedPSF(
        Adisc   ::Array{Float64,D}          ,
        ωdiscs  ::Vector{Vector{Float64}}   ,
        Kernels ::NTuple{D,Matrix{Float64}} ,
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
        @TIME _, ωdiscs, Adisc, Kernels, _ = prepare_broadening_mp(ωdisc, Adisc, sigmak, γ; ωconts, kwargs...) "Prepare broadening."
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

    kernels_new = [psf.Kernels[i] for i in 1:D]# [qtt.tt.T[i][:,qw[i],:] for i in 1:DR]
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
    
    result = contract_Kernels_w_Adisc_mp(kernels_new, psf.Adisc)

    return result
end