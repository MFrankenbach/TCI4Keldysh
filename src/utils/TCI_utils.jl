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

    DR = length(qtt.tt.sitetensors)
    R = div(DR, D)

    Ts_new = [qtt.tt.sitetensors[i] for i in 1:DR]# [qtt.tt.T[i][:,qw[i],:] for i in 1:DR]
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


function getsitesforqtt(qtt::QuanticsTCI.QuanticsTensorCI2; tags=("bla", "blu"))
    D = length(qtt.grid.origin)
    R = qtt.grid.R
    @assert length(tags) == D "number of tags inconsitent with QTCI dimensions."
    
    localdims = [size(t, 2) for t in TCI.tensortrain(qtt.tt)]
    sites = [Index(localdims[R*(d-1)+r], "Qubit, $(tags[d])=$r") for r in 1:R for d in 1:D]
    return sites

end


"""
    Convert TensorCI2 object to MPS (from ITensor library)
"""
function TCItoMPS(tci::TCI.TensorCI2{T}; sites = nothing)::MPS where {T}
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

# DOESN'T SEEM TO WORK
#function MPStoQTCI(mps::ITensors.MPS)
#    DR = length(mps)
#    R = parse(Int, split(string(siteind(mps, 1).tags)[2:end-1], "=")[2])
#    D = div(DR, R)
#    T = eltype(mps[1])
#    localdimensions = dim.(siteinds(mps))
#    lnkdims = [1; linkdims(mps); 1]
#    tt = TCI.TensorCI2{T}(localdimensions)
#    for d in 1:(DR)
#        tt.T[d] = reshape(mps[d].tensor, (lnkdims[d], localdimensions[d], lnkdims[d+1]))
#    end
#
#    grid = QuanticsGrids.InherentDiscreteGrid{D}(R; unfoldingscheme=QuanticsGrids.UnfoldingSchemes.interleaved)
#    qtt = QuanticsTCI.QuanticsTensorCI2{T}(tt, grid)
#
#    return qtt
#end



"""
Convert tucker decomposition to QuanticsTensorCI2
"""
function TDtoQTCI(tc_in::TCI4Keldysh.AbstractTuckerDecomp{D}; method="svd", tolerance=1e-8,
    unfoldingscheme::UnfoldingSchemes.UnfoldingScheme=UnfoldingSchemes.interleaved
    ) where{D}

    function truncateTD!(tc)
        dims_ext = size.(tc.Kernels, 1)
        
        if !allequal(dims_ext)

            @VERBOSE "Truncating dimensions of Tucker decomposition to common size.\n"

            Rs = trunc.(Int, log2.(dims_ext))
            R = min(Rs...)
            N = 2^R
            println("N = ", N)
            
            for d in eachindex(tc.Kernels)
                idx_zero = div.(dims_ext[d], 2) + 1
                idx_min = idx_zero - 2^(R-1)
                idx_max = idx_min + N -1
                tc.Kernels[d] = tc.Kernels[d][idx_min:idx_max,:]
                @VERBOSE "Truncating dim $d of length=$(dims_ext[d]) to range=$idx_min:$idx_max\n"
            end
        end    
        return nothing
    end

    tc = deepcopy(tc_in)
    truncateTD!(tc)

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


function MPS_to_fatTensor(mps::MPS; tags)
    D = length(tags)
    sites = siteinds(mps)
    nbit = div(length(sites), D)
    sites_reshuffled = [reverse([sites[findfirst(x -> hastags(x, "$(tags[d])=$n"), sites)] for n in 1:nbit]) for d in 1:D]
    arr = Array(reduce(*, mps), vcat(sites_reshuffled...))
    arr = reshape(arr, ntuple(i -> 2^nbit, D))
    return arr
end


function affine_freq_transform(mps::MPS; tags, ωconvMat::Matrix{Int}, isferm_ωnew::Vector{Int})

    D = length(isferm_ωnew) # number of frequencies
    R = div(length(mps), D)
    tags = collect(tags)
    N = 2^R
    halfN = 2^(R - 1)

    # consistency checks
    @assert all(size(ωconvMat) .== (D, D))
    @assert all(sum(abs.(ωconvMat), dims=2) .<= 2)  # check that the matrix represents a supported 'affinetransform'

    

    isferm_ωold = mod.(ωconvMat * isferm_ωnew, 2)
    begin # parse matrix into dictionary with coefficients for affinetransform
        iωs = [partialsortperm(abs.(ωconvMat[i,:]), 1:2, rev=true)  for i in 1:D] # 
        coeffs_dic = [Dict([tags[iωs[i][j]] => ωconvMat[i,iωs[i][j]] for j in 1:2]...) for i in 1:D]
    end
    #println("coeffs_dic = ", coeffs_dic)
    shift = halfN * (sum(abs.(ωconvMat), dims=2)[:] .- 1) + div.(ωconvMat * isferm_ωnew - isferm_ωold, 2)
    #println("shift = ", shift)
    
    bc = ones(Int, D) # boundary condition periodic
    mps_rot = Quantics.affinetransform(
        mps,
        tags,
        coeffs_dic,
        shift,
        bc
    )
    return mps_rot
end



function freq_transform(mps::MPS; tags, ωconvMat::Matrix{Int}, isferm_ωnew::Vector{Int})

    function convert_to_affineTrafos(m::Matrix{Int}) ::Vector{Matrix{Int}}
        D = size(m, 1)
        is_leq2 = sum(abs.(m), dims=2)[:] .<= 2
        isaffine = all(is_leq2)
    
        if isaffine
            return [m]
        else
            irow3 = argmin(is_leq2) # row with 3 non-zero entries
            other = mod(irow3+1, D) + 1
    
            # eliminate problematic row 
            jother = argmax(abs.(m[other,:]))
            T = collect(Diagonal(ones(Int, D)))
            T[irow3,other] = -m[irow3,jother]*m[other,jother]
            msnew = [
            round.(Int, inv(T)),
            T*m
            ]
    
            @assert all(msnew[1]*msnew[2] .== m)
    
            return msnew
        end
    end
    
    

    ωconvMat_list = convert_to_affineTrafos(ωconvMat)
    
    if length(ωconvMat_list) > 1
        isferm_ωnew_list = [mod.(ωconvMat_list[2] * isferm_ωnew, 2), isferm_ωnew]
    else
        isferm_ωnew_list = [isferm_ωnew]
    end

    for i in eachindex(ωconvMat_list)
        mps = affine_freq_transform(mps; tags, ωconvMat=ωconvMat_list[i], isferm_ωnew=isferm_ωnew_list[i])
    end
    return mps
end




function zeropad_QTCI2(qtt_in::QuanticsTCI.QuanticsTensorCI2; N::Int, nonzeroinds_left=nothing)
    #qtt_in = qtt_SVDed2
    #nonzeroinds_left = nothing
    #N = 2
    R_old = qtt_in.grid.R
    D = div(length(qtt_in.tt), R_old)
    if nonzeroinds_left === nothing
        nonzeroinds_left = ones(Int, N*D)
    else
        println("Ping! N=$N, D=$D")
        println(length(nonzeroinds_left))
        if length(nonzeroinds_left) != N*D
            throw(ArgumentError("keyword argument 'nonzeroinds_left' needs to be a $(N*D) length Vector{Int64}."))
        end
    end
    R_new = R_old + N * D
    
    #tensors_new = qtt_in.tt.sitetensors
    #tensors_new = [[deepcopy(trivialtensor) for _ in 1:N*D]; tensors]
    
    T = eltype(qtt_in.tt.sitetensors[1])
    localdims_new = [2*ones(Int, N*D); qtt_in.tt.localdims]
    tt_new = TCI.TensorCI2{T}(localdims_new)
    for i in eachindex(tt_new.sitetensors)
        if i <= N*D
        trivialtensor = zeros(Int, 2)
        trivialtensor[nonzeroinds_left[i]] = 1
        trivialtensor = reshape(trivialtensor, (1,2,1))
        tt_new.sitetensors[i] = trivialtensor
        else
            tt_new.sitetensors[i] = qtt_in.tt.sitetensors[i-N*D]

        end
    end
    

    function modify_IJsets!(isets, jsets, isets_old, jsets_old)
        for i in eachindex(isets)
            if i <= N*D
                isets[i] = [nonzeroinds_left[1:i-1]] 
                jsets[i] = [[nonzeroinds_left[1:i-1]; ones(Int, R_old)]] # here the Jsets can be chosen arbitrarily, right?
            else
                isets[i] = [[nonzeroinds_left; iset] for iset in isets_old[i-N*D]]

                jsets[i] = jsets_old[i-N*D]
            end
        end
        return nothing
    end
    modify_IJsets!(tt_new.Iset, tt_new.Jset, qtt_in.tt.Iset, qtt_in.tt.Jset)
    for i in eachindex(qtt_in.tt.Iset_history)
        push!(tt_new.Iset_history, qtt_in.tt.Iset_history[i])
        push!(tt_new.Jset_history, qtt_in.tt.Jset_history[i])
        modify_IJsets!(tt_new.Iset_history[i], tt_new.Jset_history[i], qtt_in.tt.Iset_history[i], qtt_in.tt.Jset_history[i])
    end
    #tt_new.Iset
    #tt_new.Jset
    grid_new = QuanticsGrids.InherentDiscreteGrid{D}(R_new; unfoldingscheme=QuanticsGrids.UnfoldingSchemes.interleaved)
    @assert all(TCI.linkdims(tt_new) .== length.(tt_new.Iset)[2:end])
    @assert all(TCI.linkdims(tt_new) .== length.(tt_new.Jset)[1:end-1])

    return QuanticsTCI.QuanticsTensorCI2{T}(tt_new, grid_new)
 
end