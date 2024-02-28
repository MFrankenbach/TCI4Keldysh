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
function QTCItoMPS(qtci::QuanticsTCI.QuanticsTensorCI2{T}, tags::NTuple{D,String})::MPS where {T, D}
    D_ = length(qtci.grid.origin)
    D_ == D || throw(ArgumentError("tags must be $D_-Tuple of Strings."))
    sites = getsitesforqtt(qtci; tags)
    return TCItoMPS(qtci.tt; sites)
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

    #println("length=$(length(fattensor))")

    dims_ext = size(fattensor)
    Rs = trunc.(Int, log2.(dims_ext))
    R = Rs[1]
    @assert all(R .== Rs)       # all R must be identical
    @assert dims_ext[1] == 2^R  # size of fattensor must be power of 2

    if method=="svd"
        localdimensions = 2*ones(Int, D*R)
        fattensor = reshape(fattensor, ((localdimensions...)))

        #println("length=$(length(fattensor))")

        #p = reverse(append!([reverse(collect(i:R:D*R)) for i in 1:R]...))
        p = invperm(append!([reverse(collect(i:D:D*R)) for i in 1:D]...))

        fattensor = permutedims(fattensor, p)

        #println("size=$(size(fattensor))")
        qtt_dat = Vector{Array{T}}(undef, R*D)

        # convert fat tensor into mps via SVD:
        sz_left = 1
        for i in 1:(R*D)
            #println("i=$i, sz_left=$sz_left, size=$(size(fattensor))")
            fattensor = reshape(fattensor, (sz_left*2, div(length(fattensor), sz_left*2)))
            U,S,V = svd(fattensor)
            in_tol = S .>= tolerance
            U = U[:,in_tol]
            qtt_dat[i] = reshape(U, (div(size(U,1),2),2,size(U,2)))
            fattensor = Diagonal(S[in_tol]) * V[:,in_tol]'
            #println("S=", S)

            sz_left = size(fattensor,1)
        end
        qtt_dat[end] *= fattensor[1]
        @assert length(fattensor) == 1

        tt = TCI.TensorCI2{T}(localdimensions)
        for d in 1:(D*R)
            tt.sitetensors[d] = qtt_dat[d]
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
        #println("Ping! N=$N, D=$D")
        #println(length(nonzeroinds_left))
        if length(nonzeroinds_left) != N*D
            throw(ArgumentError("keyword argument 'nonzeroinds_left' needs to be a $(N*D) length Vector{Int64}."))
        end
    end
    R_new = R_old + N
    
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


"""
Replace sitesold with sitesnew in M
"""
function _replaceinds!(M::MPS, sitesold::Vector{<:Index}, sitesnew::Vector{<:Index})
    length(sitesold) == length(sitesnew) ||
        error("sitesold and sitesnew must have the same length")

    for n = 1:length(sitesold)
        for i in findsites(M, sitesold[n])
            replaceinds!(M[i], sitesold[n] => sitesnew[n])
        end
    end
end


function _adoptinds_by_tags!(M::MPS, M_ref::MPS, tag::String, tag_ref::String, R::Int)
    sites_1 = siteinds(M)
    sitesx_1 = [sites_1[findfirst(x -> hastags(x, "$tag=$n"), sites_1)] for n in 1:R]
    sites_2 = siteinds(M_ref)
    sitesx_2 = [sites_2[findfirst(x -> hastags(x, "$tag_ref=$n"), sites_2)] for n in 1:R]
    _replaceinds!(M, sitesx_1, sitesx_2)
    return nothing
end

function add_dummy_dim(mps::MPS; pos::Int, tag::String="Qubit, dummy", delta_dim=nothing, D_old::Int)
    #mps = mps_Kernel1
    #pos = 3
    #tag = "dummy"
    #delta_dim = nothing
    #delta_dim = 2
    #D_old = 2 # should be deducable from tags; but for now just give it as argument
    mps_in = deepcopy(mps)
    R = div(length(mps_in), D_old)#6

    #mps_in.data[1]
    #mps_in.data[2]
    #tensors = mps_in.data[1] * mps_in.data[2]
    if !(delta_dim === nothing)
        qdelta_dim = Quantics.tobin(delta_dim, R) .+ 1
    end
    T = Float64
    linkdims_old = [1; linkdims(mps_in); 1]
    dummy_sitetensors = [ begin
                            L = linkdims_old[pos+(i-1)*D_old]; # linkdim
                            ident = reshape(collect(Diagonal(ones(T, L, L))), (L, 1, L));
                                if delta_dim === nothing
                                    localstructure = ones(T, 2)
                                else
                                    localstructure = zeros(T, 2)
                                    localstructure[qdelta_dim[i]] = 1
                                end
                            dummy_sitetensor = ident .* reshape(localstructure, (1,2))
                            if pos == 1 && i == 1
                                dummy_sitetensor = dropdims(dummy_sitetensor; dims=1)
                                indices = (Index(2, tag*"=$i"), Index(L, "dummylink=$i"))
                            elseif pos == D_old+1 && i == R
                                dummy_sitetensor = dropdims(dummy_sitetensor; dims=3)
                                indices = (Index(L, "dummylink=$(i-1)"), Index(2, tag*"=$i"))
                            else
                                indices = (Index(L, "dummylink=$(i-1)"), Index(2, tag*"=$i"), Index(L, "dummylink=$i"))

                            end
                            ITensor(dummy_sitetensor, indices...)
                            end
                        for i in 1:R]

    sitetensors_new = deepcopy(mps_in.data[1:pos-1]) # Vector{ITensor}(undef, 0)
    linkdims_new = linkdims_old[2:pos]
    for i in 1:R
        #println(i)
        #println("push mps range $(pos+D_old*(i-1)):$(min(R*D_old, pos+D_old*(i)-1))")
        push!(sitetensors_new, dummy_sitetensors[i], mps_in.data[pos+D_old*(i-1):min(R*D_old, pos+D_old*(i)-1)]...)
        push!(linkdims_new, linkdims_old[pos+D_old*(i-1)], linkdims_old[1+pos+D_old*(i-1):1+min(R*D_old-1, pos+D_old*(i)-1)]...)
        #insert!(sitetensors_new, dummy_sitetensors[i], pos + D_old*(i-1))
        #insert!(linkdims_new, linkdims_old[pos + D_old*(i-1)], pos + D_old*(i-1))
    end
    #linkdims_new
    #linkdims_old
    # correct "dummy" legs
    if pos == 1
        t = sitetensors_new[pos+1]
        dat = Array(t, inds(t)...)
        sitetensors_new[pos+1] = ITensor(dat, Index(1, "dummylink=1"), inds(t)...)
    elseif pos == D_old+1
        t = sitetensors_new[end-1]
        dat = Array(t, inds(t)...)
        sitetensors_new[end-1] = ITensor(dat, inds(t)..., Index(1, "dummylink=$(R-1)"))
    end

    linkids_new = [Index(l, "link, l=$i") for (i,l) in enumerate(linkdims_new)]
    for i in eachindex(linkdims_new)
        #tagr = tags(inds(sitetensors_new[i])[end])
        #settags!(sitetensors_new[i], "link, l=$i"; tags=tagr)
        #tagl = tags(inds(sitetensors_new[i+1])[1])
        #settags!(sitetensors_new[i+1], "link, l=$i"; tags=tagl)
        #println("i=$i")
        idxl = inds(sitetensors_new[i])[end]
        replaceind!(sitetensors_new[i], idxl, linkids_new[i])
        idxr = inds(sitetensors_new[i+1])[1]
        replaceind!(sitetensors_new[i+1], idxr, linkids_new[i])
    end
    #tags(inds(sitetensors_new[1])[end])
    inds.(sitetensors_new)
    #inds(sitetensors_new[1])
    mps_new = MPS(sitetensors_new)
    #linkdims(mps_new)
    #siteinds(mps_new)
    #linkinds(mps_new)
    return mps_new

end





"""
pad array with zeros. larger array has size 2^R in every dimension (for some R)
"""
function zeropad_array(arr::Array{T,D}) where{T,D}
    R_ = maximum(ceil.(Int, log2.(size(arr)))    )
    res = zeros(T, (2^R_.*ones(Int, D))...)
    view(res, Base.OneTo.(size(arr))...) .= arr
    return res
end

function TD_to_MPS_via_TTworld(broadenedPsf::TCI4Keldysh.AbstractTuckerDecomp{2}; tolerance::Float64=1e-14, alg="tci2")
    D = 2
    #tolerance = 1e-14
    #kwargs = Dict(:alg=>"densitymatrix")
    kwargs = Dict(:alg=>alg, :tolerance=>tolerance)

    

    R = trunc(Int, log2(size(broadenedPsf.Kernels[1],1)))
    residue = size(broadenedPsf.Kernels[1],1) - 2^R
    @assert residue == 1 || residue == 0
    R_Adisc = maximum(ceil.(Int, log2.(size(broadenedPsf.Adisc)))    )

    TCI4Keldysh.@TIME TCI4Keldysh.shift_singular_values_to_center!(broadenedPsf) "Shifting singular values."

    qtt_Adisc, _, _ = quanticscrossinterpolate(
            zeropad_array(broadenedPsf.Adisc),#[end-2^R_Adisc+1:end,end-2^R_Adisc+1:end],
            tolerance=tolerance
        )  
    
    #qtt_Adisc = TCI4Keldysh.fatTensortoQTCI(broadenedPsf.Adisc[end-2^6+1:end,end-2^6+1:end,end-2^6+1:end]; tolerance)
    

    # pad qtt:
    nonzeroinds_left=ones(Int, (R-R_Adisc)*D)
    qtt_Adisc_padded = TCI4Keldysh.zeropad_QTCI2(qtt_Adisc; N=R-R_Adisc, nonzeroinds_left)
    all(qtt_Adisc_padded.tt.sitetensors[(R-R_Adisc)*D+1:end] .== qtt_Adisc.tt.sitetensors)
    all(getindex.(argmax.(qtt_Adisc_padded.tt.sitetensors[1:(R-R_Adisc)*D]), 2) .== nonzeroinds_left)



    # convert qtt for Tucker center
    mps_Adisc_padded = TCI4Keldysh.QTCItoMPS(qtt_Adisc_padded, ntuple(i->"eps$i", D))

    #Kernels = [zeros(eltype(broadenedPsf.Kernels[1]), 2^R, 2^R) for i in 1:D]
    #for i in 1:D
    #    Kernels[i][:,1:2^6] .= broadenedPsf.Kernels[i][1:end-residue,end-2^6+1:end]
    #end
    Kernels = [zeropad_array(broadenedPsf.Kernels[i][1:end-residue,:]) for i in 1:D]
    qtt_Kernels = [TCI4Keldysh.fatTensortoQTCI(Kernels[i]; tolerance=-1.) for i in 1:D]
    mps_Kernels = [TCI4Keldysh.QTCItoMPS(qtt_Kernels[i], ("ω$i", "eps$i")) for i in 1:D]

    ### contract Kernel with Tucker center

    # add dummy dimension
    #TCI4Keldysh.@TIME begin
    #    mps_Kernel1_exp = TCI4Keldysh.add_dummy_dim(mps_Kernels[1]; pos=3, D_old=2)
    #    mps_Kernel2_exp = TCI4Keldysh.add_dummy_dim(mps_Kernels[2]; pos=1, D_old=2)
    #    #mps_Kernel3_exp = TCI4Keldysh.add_dummy_dim(mps_Kernels[3]; pos=1, D_old=2)
    #end "Adding dummy dimensions."
    mps_Kernel1_exp = mps_Kernels[1]
    mps_Kernel2_exp = mps_Kernels[2]

    # adopt shared site indices from Tucker center
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel1_exp, mps_Adisc_padded, "eps1",  "eps1", R)
    #TCI4Keldysh._adoptinds_by_tags!(mps_Kernel1_exp, mps_Adisc_padded, "dummy", "eps3", R)

    TCI4Keldysh.@TIME ab = Quantics.automul(
            mps_Kernel1_exp,
            mps_Adisc_padded;
            tag_row = "ω1",
            tag_shared = "eps1",
            tag_col = "eps2",
            kwargs...#,
            #tolerance=tolerance
        ) "Contraction 1"
    #ITensors.siteinds(ab)
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel2_exp, ab, "eps2",  "eps2", R)
    #TCI4Keldysh._adoptinds_by_tags!(mps_Kernel2_exp, ab, "dummy", "eps3", R)

    #findallsiteinds_by_tag(siteinds(); tag="eps2")
    TCI4Keldysh.@TIME res = Quantics.automul(
        ab,
        mps_Kernel2_exp;
            tag_row = "ω1",
            tag_shared = "eps2",
            tag_col = "ω2",
            kwargs...
            #alg=alg,
            #tolerance=tolerance
        ) "Contraction 2"

    #siteinds(res)
    
    # siteinds(res)
    return res
end




function TD_to_MPS_via_TTworld(broadenedPsf::TCI4Keldysh.AbstractTuckerDecomp{3}; tolerance::Float64=1e-14, alg="tci2")
    D = 3
    #tolerance = 1e-14
    kwargs = Dict(:alg=>alg, :tolerance=>tolerance)


    R = trunc(Int, log2(size(broadenedPsf.Kernels[1],1)))
    residue = size(broadenedPsf.Kernels[1],1) - 2^R
    @assert residue == 1 || residue == 0
    R_Adisc = maximum(ceil.(Int, log2.(size(broadenedPsf.Adisc)))    )

    @TIME TCI4Keldysh.shift_singular_values_to_center!(broadenedPsf) "Shifting singular values."

    qtt_Adisc, _, _ = quanticscrossinterpolate(
        zeropad_array(broadenedPsf.Adisc),#broadenedPsf.Adisc[end-2^6+1:end,end-2^6+1:end,end-2^6+1:end],
            tolerance=tolerance
        )  
    
    #qtt_Adisc = TCI4Keldysh.fatTensortoQTCI(broadenedPsf.Adisc[end-2^6+1:end,end-2^6+1:end,end-2^6+1:end]; tolerance)
    

    # pad qtt:
    nonzeroinds_left=ones(Int, (R-R_Adisc)*D)
    qtt_Adisc_padded = TCI4Keldysh.zeropad_QTCI2(qtt_Adisc; N=R-R_Adisc, nonzeroinds_left)
    all(qtt_Adisc_padded.tt.sitetensors[(R-R_Adisc)*D+1:end] .== qtt_Adisc.tt.sitetensors)
    all(getindex.(argmax.(qtt_Adisc_padded.tt.sitetensors[1:(R-R_Adisc)*D]), 2) .== nonzeroinds_left)



    # convert qtt for Tucker center
    mps_Adisc_padded = TCI4Keldysh.QTCItoMPS(qtt_Adisc_padded, ntuple(i->"eps$i", D))

    #Kernels = [zeros(eltype(broadenedPsf.Kernels[1]), 2^R, 2^R) for i in 1:D]
    #for i in 1:3
    #    Kernels[i][:,1:2^R_Adisc] .= broadenedPsf.Kernels[i][1:end-1,end-2^6+1:end]
    #end
    Kernels = [zeropad_array(broadenedPsf.Kernels[i][1:end-residue,:]) for i in 1:D]
    qtt_Kernels = [TCI4Keldysh.fatTensortoQTCI(Kernels[i]; tolerance=-1.) for i in 1:D]
    mps_Kernels = [TCI4Keldysh.QTCItoMPS(qtt_Kernels[i], ("ω$i", "eps$i")) for i in 1:D]

    ### contract Kernel with Tucker center

    # add dummy dimension
    @TIME begin
        mps_Kernel1_exp = TCI4Keldysh.add_dummy_dim(mps_Kernels[1]; pos=3, D_old=2)
        mps_Kernel2_exp = TCI4Keldysh.add_dummy_dim(mps_Kernels[2]; pos=3, D_old=2)
        mps_Kernel3_exp = TCI4Keldysh.add_dummy_dim(mps_Kernels[3]; pos=1, D_old=2)
    end "Adding dummy dimensions."

    # adopt shared site indices from Tucker center
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel1_exp, mps_Adisc_padded, "eps1",  "eps1", R)
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel1_exp, mps_Adisc_padded, "dummy", "eps3", R)

    @TIME ab = Quantics.automul(
            mps_Kernel1_exp,
            mps_Adisc_padded;
            tag_row = "ω1",
            tag_shared = "eps1",
            tag_col = "eps2",
            alg="tci2",
            tolerance=1e-8
        ) "Contraction 1"
    #siteinds(ab)
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel2_exp, ab, "eps2",  "eps2", R)
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel2_exp, ab, "dummy", "eps3", R)

    #findallsiteinds_by_tag(siteinds(); tag="eps2")
    @TIME ab2 = Quantics.automul(
        ab,
        mps_Kernel2_exp;
            tag_row = "ω1",
            tag_shared = "eps2",
            tag_col = "ω2",
            alg="tci2",
            tolerance=1e-8
        ) "Contraction 2"

    #siteinds(ab2)
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel3_exp, ab2, "eps3",  "eps3", R)
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel3_exp, ab2, "dummy", "ω1"  , R)

    @TIME res = Quantics.automul(
        ab2,
        mps_Kernel3_exp;
            tag_row = "ω2",
            tag_shared = "eps3",
            tag_col = "ω3",
            alg="tci2",
            tolerance=1e-8
        ) "Contraction 3"

    # siteinds(res)
    return res
end