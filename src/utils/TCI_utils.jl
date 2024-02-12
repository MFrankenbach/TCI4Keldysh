function qtt_to_fattensor(Ts::Vector{Array{T, 3}} ) where{T}
    # fatten: turn tensor train into fat tensor:
    DR = length(Ts)
    shapes = [size(Ts[i]) for i in 1:DR]
    result = Ts[1]
    for i in 2:DR
        left = reshape(result, (div(length(result), shapes[i-1][3]), shapes[i-1][3]))
        right = reshape(Ts[i], (shapes[i][1], prod(shapes[i][2:3])))
        result = left * right
    end
    result = reshape(result, ntuple(i -> shapes[i][2], DR));
    return result
end

function qinterleaved_fattensor_to_regular(input, R)
    
    if length(input) == 1
        result = input[1]
    else
        # remove singleton dimensions
        input = dropdims(input,dims=tuple(findall(size(input).==1)...))

        # permute quantics suitably
        DR = ndims(input)
        D = div(DR, R)
        
        permtupl = append!([reverse(collect(i:D:DR)) for i in 1:D]...)
        
        result = permutedims(input, permtupl);
        result = reshape(result, ntuple(i->2^R, D))
    end
    return result
end


function Base.:getindex(
    qtt :: QuanticsTCI.QuanticsTensorCI2{Q},
    w   :: Vararg{Union{Int, Colon}, D} # , UnitRange
    )   :: Union{Q, AbstractArray{Q}} where {D, Q <: Number}

    function check_bounds(d)
        if !(1<=w[d]<=2^R)
            throw(BoundsError(qtt, w))
        end
        return nothing
    end

    DR = length(qtt.tt.T)
    R = div(DR, D)

    Ts_new = [qtt.tt.T[i] for i in 1:DR]# [qtt.tt.T[i][:,qw[i],:] for i in 1:DR]
    # bounds check
    #@assert all(1 .<= w .<= 2^R)

    #qw = Array{Union{UnitRange,Colon}}(undef, DR)
    for d in 1:D
        if typeof(w[d]) == Int
            @boundscheck check_bounds(d)
                
            
            qw = QuanticsGrids.index_to_quantics((w[d]...); numdigits=R, base=2)
            for i in 1:R
                Ts_new[d + (i-1)*D] = Ts_new[d + (i-1)*D][:,qw[i]:qw[i],:]
            end
        end
     end
    
    result = qinterleaved_fattensor_to_regular(qtt_to_fattensor(Ts_new), R)

    return result
end




"""
    Convert TensorCI2 object to MPS (from ITensor library)
"""
function TCItoMPS(tci::TCI.TensorCI2{T}, sites = nothing)::MPS where {T}
    tensors = TCI.tensortrain(tci)
    ranks = TCI.rank(tci)
    N = length(tensors)
    localdims = [size(t, 2) for t in tensors]

    if sites === nothing
        sites = [Index(localdims[n], "n=$n") for n = 1:N]
    else
        all(localdims .== dim.(sites)) ||
            error("ranks are not consistent with dimension of sites")
    end

    linkdims = [[size(t, 1) for t in tensors]..., 1]
    links = [Index(linkdims[l+1], "link,l=$l") for l = 0:N]

    tensors_ = [ITensor(deepcopy(tensors[n]), links[n], sites[n], links[n+1]) for n = 1:N]
    tensors_[1] *= onehot(links[1] => 1)
    tensors_[end] *= onehot(links[end] => 1)

    return MPS(tensors_)
end

"""
Convert Tucker decomposition to fat tensor
"""
function TDtoFatTensor(tc::TCI4Keldysh.AbstractTuckerDecomp{D}) where{D}
    fattensor = tc[ntuple(i -> Colon(), D)...]
    return fattensor
end

"""
Convert tucker decomposition to QuanticsTensorCI2
"""
function TDtoQTCI(tc::TCI4Keldysh.AbstractTuckerDecomp{D}; method="svd", tolerance=1e-8,
    unfoldingscheme::UnfoldingSchemes.UnfoldingScheme=UnfoldingSchemes.interleaved
    ) where{D}

    dims_ext = size.(tc.Kernels, 1)
    Rs = trunc.(Int, log2.(dims_ext))
    R = Rs[1]
    @assert all(R .== Rs)
    qttRanges = ntuple(i -> 1:2^R, D)


    fattensor = TDtoFatTensor(tc)
    # truncate to powers of 2
    fattensor = fattensor[qttRanges...]

    qtt = fatTensortoQTCI(fattensor; method, tolerance, unfoldingscheme)

    return qtt

end

function fatTensortoQTCI(fattensor::Array{T,D} ; method="svd", tolerance=1e-8,
    unfoldingscheme::UnfoldingSchemes.UnfoldingScheme=UnfoldingSchemes.interleaved,
    kwargs...
    ) where{T, D}

    dims_ext = size(fattensor)
    Rs = trunc.(Int, log2.(dims_ext))
    R = Rs[1]
    @assert all(R .== Rs)       # all R must be identical
    @assert dims_ext[1] == 2^R  # size of fattensor must be power of 2

    if method=="svd"
        localdimensions = 2*ones(Int, D*R)
        fattensor = reshape(fattensor, ((localdimensions...)))

        #p = reverse(append!([reverse(collect(i:R:D*R)) for i in 1:R]...))
        p = invperm(append!([reverse(collect(i:D:D*R)) for i in 1:D]...))

        fattensor = permutedims(fattensor, p)
        qtt_dat = Vector{Array{T}}(undef, R*D)

        # convert fat tensor into mps via SVD:
        sz_left = 1
        for i in 1:(R*D)
            fattensor = reshape(fattensor, (sz_left*2, div(length(fattensor), sz_left*2)))
            U,S,V = svd(fattensor)
            in_tol = S .>= tolerance
            U = U[:,in_tol]
            qtt_dat[i] = reshape(U, (div(size(U,1),2),2,size(U,2)))
            fattensor = Diagonal(S[in_tol]) * V[:,in_tol]'

            sz_left = size(fattensor,1)
        end
        qtt_dat[end] *= fattensor[1]
        @assert length(fattensor) == 1

        tt = TCI.TensorCI2{T}(localdimensions)
        for d in 1:(D*R)
            tt.T[d] = qtt_dat[d]
        end

        grid = QuanticsGrids.InherentDiscreteGrid{D}(R; unfoldingscheme=unfoldingscheme)
        qtt = QuanticsTCI.QuanticsTensorCI2{T}(tt, grid)


    elseif method == "qtci"

        qtt, ranks, errors = quanticscrossinterpolate(
            fattensor;
            tolerance,
            unfoldingscheme,
            kwargs...
            #; maxiter=400
        )  

    else
        throw(RuntimeError("method unknown."))
    end

    return qtt
end