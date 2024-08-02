"""
qtt_to_fattensor(Ts::Vector{Array{T, 3}})

Convert QTT to fat tensor.
"""
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

"""
qinterleaved_fattensor_to_regular(input::Array, R)

Permute dimensions to convert a fattensor in (interleaved) Quantics format to a fat tensor in regular format.
"""
function qinterleaved_fattensor_to_regular(input::Array, R)
    
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

    DR = length(qtt.tci.sitetensors)
    R = div(DR, D)

    Ts_new = [qtt.tci.sitetensors[i] for i in 1:DR]# [qtt.tci.T[i][:,qw[i],:] for i in 1:DR]
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
    getsitesforqtt(qtt::QuanticsTCI.QuanticsTensorCI2; tags)

Construct strings to be used as site tags of an MPS (from iTensor) that represents a QTT.
"""
function getsitesforqtt(qtt::QuanticsTCI.QuanticsTensorCI2; tags=("bla", "blu"))
    D = length(qtt.grid.origin)
    R = qtt.grid.R
    @assert length(tags) == D "number of tags inconsitent with QTCI dimensions."
    
    localdims = [size(t, 2) for t in TCI.tensortrain(qtt.tci)]
    sites = [Index(localdims[R*(d-1)+r], "Qubit, $(tags[d])=$r") for r in 1:R for d in 1:D]
    return sites

end

"""
    function quanticssites(R::Int, tag="bla")

Construct binary indices [tag=1, tag=2, ...]
"""
function quanticssites(R::Int, tag::String)
    sites = [Index(2, "Qubit, $(tag)=$r") for r in 1:R]
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
    return TCItoMPS(qtci.tci; sites)
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
#    grid = QuanticsGrids.InherentDiscreteGrid{D}(R; unfoldingscheme=:interleaved)
#    qtt = QuanticsTCI.QuanticsTensorCI2{T}(tt, grid)
#
#    return qtt
#end



"""
Convert tucker decomposition to QuanticsTensorCI2
"""
function TDtoQTCI(tc_in::TCI4Keldysh.AbstractTuckerDecomp{D}; method="svd", tolerance=1e-8,
    unfoldingscheme=:interleaved
    ) where{D}

    function truncateTD!(tc)
        dims_ext = size.(tc.legs, 1)
        
        if !allequal(dims_ext)

            @VERBOSE "Truncating dimensions of Tucker decomposition to common size.\n"

            Rs = trunc.(Int, log2.(dims_ext))
            R = min(Rs...)
            N = 2^R
            println("N = ", N)
            
            for d in eachindex(tc.legs)
                idx_zero = div.(dims_ext[d], 2) + 1
                idx_min = idx_zero - 2^(R-1)
                idx_max = idx_min + N -1
                tc.legs[d] = tc.legs[d][idx_min:idx_max,:]
                @VERBOSE "Truncating dim $d of length=$(dims_ext[d]) to range=$idx_min:$idx_max\n"
            end
        end    
        return nothing
    end

    tc = deepcopy(tc_in)
    truncateTD!(tc)

    dims_ext = size.(tc.legs, 1)
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
    unfoldingscheme=:interleaved,
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
        
        qf_ =  q -> f(QuanticsGrids.quantics_to_origcoord(grid, q)...)
        qf = TCI.CachedFunction{T}(qf_, localdimensions)

        qtt = QuanticsTCI.QuanticsTensorCI2{T}(tt, grid, qf)


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

"""
Turn mps to array on indices given by slices.
TODO: What about reverse_idx?
"""
function MPS_to_fatTensor(mps::MPS, slices::Vector{T}; reverse_idx=false, tags) where {T<:AbstractRange}
    @assert all(dim.(siteinds(mps)) .== 2) "Need all sitedims to be 2"
    @assert length(slices)==length(tags)
    D = length(tags)
    sites = siteinds(mps)
    nbit = div(length(sites), D)
    out_arr = Array{eltype(MPS), D}(undef, length.(slices)...)
    for s in Iterators.product(slices...)
        s_quantics = [QuanticsGrids.index_to_quantics(ss; numdigits=nbit, base=2) for ss in s]
        s_idx = QuanticsGrids.interleave_dimensions(s_quantics...)
        s_out = ntuple(i -> findfirst(ss -> ss==s[i], slices[i]), D)
        eval_idx = reverse_idx ? reverse(s_idx) : s_idx
        out_arr[s_out...] = eval(mps, eval_idx)
    end
    return out_arr
end

"""
Does not seem to be faster than normal eval...
"""
struct MPSEvaluator
    mps::MPS
    onehotpool::Vector{ITensor}
    hotids::Vector{Int}
    sids::Vector{Index}

    function MPSEvaluator(mps::MPS)
        onehotpool = [onehot(s=>1) for s in siteinds(mps)]
        hotids = fill(1,length(mps))
        return new(mps, onehotpool, hotids, siteinds(mps))
    end
end

function (obj::MPSEvaluator)(idx::Vector{Int})
    # update onehots    
    for (i,s) in enumerate(obj.sids)
        obj.onehotpool[i][s=>obj.hotids[i]]=0.0
        obj.onehotpool[i][s=>idx[i]]=1.0
    end
    res = ITensor(1.0)
    for i in eachindex(idx)
        res *= obj.mps[i] * obj.onehotpool[i]
    end
    return scalar(res)
end

function eval(mps::MPS, idx::Vector{Int})
    TCI4Keldysh.@DEBUG length(idx)==length(mps) "idx and mps lengths do not match"

    res = ITensor(one(eltype(mps[1])))
    sidx = siteinds(mps)
    for i in eachindex(idx)
        res *= mps[i] * onehot(sidx[i] => idx[i])
    end
    return scalar(res)
end

"""
Eval MPS by contracting towards the given bond. Should be the one with largest bond dim.
SLOWER than normal eval even for χ_max ≈ 500
"""
function eval_lr(mps::MPS, idx::Vector{Int}, fatbond::Int)
    TCI4Keldysh.@DEBUG length(idx)==length(mps) "idx and mps lengths do not match"

    res_l = ITensor(one(eltype(mps[1])))
    res_r = ITensor(one(eltype(mps[1])))
    sidx = siteinds(mps)
    for i in 1:fatbond
        res_l *= mps[i] * onehot(sidx[i] => idx[i])
    end
    for i in fatbond+1:length(mps)
        res_r *= mps[i] * onehot(sidx[i] => idx[i])
    end
    return scalar(res_l*res_r)
end

"""
freq_shift(isferm_ωnew::Vector{Int}, isferm_ωold::Vector{Int}, ωconvMat::Matrix{Int})

Determine shift required for correct frequency rotation on tensor train.
old_gridsizes: sizes of the internal frequency grids, where a size 2^R+1 (bosonic) has been mapped to 2^R
"""
function freq_shift(isferm_ωnew::Vector{Int}, ωconvMat::Matrix{Int}, old_gridsizes::Vector{Int}, new_gridsizes::Vector{Int})
    # @assert all(old_gridsizes .>= new_gridsizes)
    # @assert all(mod.(new_gridsizes, 2) .== 0)

    # rotation maps new -> old
    corner_new = [isferm_ωnew[i]==1 ? (-2*div(new_gridsizes[i],2) + 1) : (-2*div(new_gridsizes[i],2)) for i in eachindex(isferm_ωnew)]
    # we want the corner at the lower left of the new TT, so we have:
    # corner_new_idx = zeros(Int, length(isferm_ωnew))

    # compute desired image of corner under frequency rotation
    corner_old = ωconvMat * corner_new

    # find index of corner_old in old grids
    isferm_ωold = gridtypes_from_freq_rotation(ωconvMat, isferm_ωnew)
    corner_old_idx = zeros(Int, length(isferm_ωnew))
    for i in eachindex(isferm_ωold)
        gridmin = -2*div(old_gridsizes[i], 2)
        if isferm_ωold[i]==1
            gridmin += 1
        end
        corner_old_idx[i] = div(corner_old[i] - gridmin, 2) # should not have a remainder
    end

    return corner_old_idx
end

function gridtypes_from_freq_rotation(ωconvMat::Matrix{Int}, isferm_ωnew::Vector{Int})
    return mod.(ωconvMat * isferm_ωnew, 2)
end

"""
Rotation maps new ↦ old
Expects uneven gridsizes for bosonic, even gridsizes for fermionic grids
"""
function gridsizes_after_freq_rotation(ωconvMat::Matrix{Int}, gridsizes_new::Vector{Int})
    isbos = mod.(gridsizes_new, 2)
    ω_max_new = [isbos[i]==1 ? 2*div(gridsizes_new[i], 2) : 2*div(gridsizes_new[i], 2) - 1 for i in eachindex(isbos)]

    ω_max_old = abs.(ωconvMat) * ω_max_new
    return [iseven(w) ? (2*div(w,2) + 1) : 2*div(w+1, 2) for w in ω_max_old]
end

"""
affine_freq_transform(mps::MPS; tags, ωconvMat::Matrix{Int}, isferm_ωnew::Vector{Int})

Perform an affine frequency transform (i.e. one that combines at most 2 frequencies into a new frequency)

Arguments:
 * mps          ::MPS
 * tags         ::Vector{String}
 * ωconvMat     ::Matrix{Int}    matrix M encoding the frequency conversion ω_old = M * ω_new
 * isferm_ωnew  ::Vector{Int}    0 for bosonic, 1 for fermionic, by convention isferm_ωnew[1]=0
 * expl_shift   ::Vector{Int}    (optionally) explicitly given shift in Quantics.affinetransform
"""
function affine_freq_transform(mps::MPS; tags, ωconvMat::Matrix{Int}, isferm_ωnew::Vector{Int}, 
        old_gridsizes::Vector{Int}, new_gridsizes::Vector{Int}, expl_shift::Union{Nothing,Vector{Int}}=nothing, kwargs...)

    D = length(isferm_ωnew) # number of frequencies
    R = div(length(mps), D)
    tags = collect(tags)
    halfN = 2^(R - 1)

    # consistency checks
    @assert all(size(ωconvMat) .== (D, D))
    @assert all(sum(abs.(ωconvMat), dims=2) .<= 2)  # check that the matrix represents a supported 'affinetransform'

    

    begin # parse matrix into dictionary with coefficients for affinetransform
        iωs = [partialsortperm(abs.(ωconvMat[i,:]), 1:2, rev=true)  for i in 1:D] # 
        coeffs_dic = [Dict([tags[iωs[i][j]] => ωconvMat[i,iωs[i][j]] for j in 1:2]...) for i in 1:D]
    end
    if TCI4Keldysh.DEBUG()
        # original shift
        isferm_ωold = mod.(ωconvMat * isferm_ωnew, 2) # abs.(ωconvMat) -> no, should be fine
        shift = halfN * (sum(abs.(ωconvMat), dims=2)[:] .- 1) + div.(ωconvMat * isferm_ωnew - isferm_ωold, 2)
        printstyled("  \nOriginal shift: $shift\n"; color=:green)
    end

    shift = if isnothing(expl_shift)
                freq_shift(isferm_ωnew, ωconvMat, old_gridsizes, new_gridsizes)
            else
                ref_shift = freq_shift(isferm_ωnew, ωconvMat, old_gridsizes, new_gridsizes)
                println("Shift in affine transform: $ref_shift")
                expl_shift
            end
    if TCI4Keldysh.DEBUG()
        printstyled("  New shift: $shift\n"; color=:green)
    end

    bc = ones(Int, D) # boundary condition periodic
    mps_rot = Quantics.affinetransform(
        mps,
        tags,
        coeffs_dic,
        shift,
        bc;
        kwargs...
    )
    return mps_rot
end


"""
freq_transform(mps::MPS; tags, ωconvMat::Matrix{Int}, isferm_ωnew::Vector{Int})

Perform a general frequency transform ωold ↦ ωnew = ωconvMat * ωold.

Arguments:
 * mps          ::MPS
 * tags         ::Vector{String}
 * ωconvMat     ::Matrix{Int}    matrix M encoding the frequency conversion ωold = M * ωnew
 * isferm_ωnew  ::Vector{Int}    0 for bosonic, 1 for fermionic
 * ext_gridsizes::Vector{Int}    sizes of external frequency grids (the frequencies fed into the rotation)
"""
function freq_transform(mps::MPS; tags, ωconvMat::Matrix{Int}, isferm_ωnew::Vector{Int},
    ext_gridsizes::Vector{Int}, kwargs...)

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
    @assert length(ωconvMat_list)<=2
    
    if length(ωconvMat_list) > 1
        isferm_ωnew_list = [gridtypes_from_freq_rotation(ωconvMat_list[2], isferm_ωnew), isferm_ωnew]
    else
        isferm_ωnew_list = [isferm_ωnew]
    end

    # @show ext_gridsizes
    gridsizes_list = [ext_gridsizes]
    for i in eachindex(ωconvMat_list)
        gr = gridsizes_after_freq_rotation(ωconvMat_list[i], gridsizes_list[begin]) 
        pushfirst!(gridsizes_list, gr)
    end

    gr_old = gridsizes_after_freq_rotation(ωconvMat, ext_gridsizes) 
    println("\ngridsizes_old: $gr_old")
    total_shift = freq_shift(isferm_ωnew, ωconvMat, gr_old, ext_gridsizes)

    for i in eachindex(ωconvMat_list)
        s = if length(ωconvMat_list)<2
                total_shift
            else
                i==2 ? fill(0, length(ext_gridsizes)) : total_shift
            end
        println("\nRotation no. $i/$(length(ωconvMat_list)), shift: $s")
        mps = affine_freq_transform(mps; tags, ωconvMat=ωconvMat_list[i], isferm_ωnew=isferm_ωnew_list[i],
                old_gridsizes=gridsizes_list[i], new_gridsizes=gridsizes_list[i+1], expl_shift=s, kwargs...)
    end
    return mps
end

"""
Maps G(iν_n) to  G'(n)=G(iν_{-n}) where ν_n = 2π*n/β with the maximum frequency ν_N cut off.
CAREFUL: G(iν_{-N}) is mapped to itself, i.e. only G'(1)=G(iν_{N-1}) to G'(2^R)=G(iν_{-N+1}) are valid.
"""
function flip_bos(Gmps::MPS, sites; kwargs...)
    error("Need to test this...")
    flipmpsprime = Quantics.flipop(sites; rev_carrydirec=false, bc=1)
    flipmps = Quantics.matchsiteinds(flipmpsprime, siteinds(Gmps))
    G_out = apply(flipmps, Gmps; kwargs... )
    return noprime(G_out)
end


function zeropad_QTCI2(qtt_in::QuanticsTCI.QuanticsTensorCI2; N::Int, nonzeroinds_left=nothing)
    #qtt_in = qtt_SVDed2
    #nonzeroinds_left = nothing
    #N = 2
    R_old = qtt_in.grid.R
    D = div(length(qtt_in.tci), R_old)
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
    
    #tensors_new = qtt_in.tci.sitetensors
    #tensors_new = [[deepcopy(trivialtensor) for _ in 1:N*D]; tensors]
    
    T = eltype(qtt_in.tci.sitetensors[1])
    localdims_new = [2*ones(Int, N*D); qtt_in.tci.localdims]
    tt_new = TCI.TensorCI2{T}(localdims_new)
    for i in eachindex(tt_new.sitetensors)
        if i <= N*D
        trivialtensor = zeros(Int, 2)
        trivialtensor[nonzeroinds_left[i]] = 1
        trivialtensor = reshape(trivialtensor, (1,2,1))
        tt_new.sitetensors[i] = trivialtensor
        else
            tt_new.sitetensors[i] = qtt_in.tci.sitetensors[i-N*D]

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
    modify_IJsets!(tt_new.Iset, tt_new.Jset, qtt_in.tci.Iset, qtt_in.tci.Jset)
    for i in eachindex(qtt_in.tci.Iset_history)
        push!(tt_new.Iset_history, qtt_in.tci.Iset_history[i])
        push!(tt_new.Jset_history, qtt_in.tci.Jset_history[i])
        modify_IJsets!(tt_new.Iset_history[i], tt_new.Jset_history[i], qtt_in.tci.Iset_history[i], qtt_in.tci.Jset_history[i])
    end
    #tt_new.Iset
    #tt_new.Jset
    grid_new = QuanticsGrids.InherentDiscreteGrid{D}(R_new; unfoldingscheme=:interleaved)
    @assert all(TCI.linkdims(tt_new) .== length.(tt_new.Iset)[2:end])
    @assert all(TCI.linkdims(tt_new) .== length.(tt_new.Jset)[1:end-1])

    return QuanticsTCI.QuanticsTensorCI2{T}(tt_new, grid_new, qtt_in.quanticsfunction)
 
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
pad array with zeros. larger array has size 2^R in every dimension (for given R)
"""
function zeropad_array(arr::Array{T,D}, R::Int) where{T,D}
    R_ = padding_R(size(arr))
    @assert R_ <= R "Required R_=$R_ larger than given R=$R"
    res = zeros(T, (2^R.*ones(Int, D))...)
    view(res, Base.OneTo.(size(arr))...) .= arr
    return res
end

"""
pad array with zeros. larger array has size 2^R in every dimension (for some R)
"""
function zeropad_array(arr::Array{T,D}) where{T,D}
    R_ = padding_R(size(arr))
    res = zeros(T, (2^R_.*ones(Int, D))...)
    view(res, Base.OneTo.(size(arr))...) .= arr
    return res
end

function padding_R(dims) :: Int
    return maximum(ceil.(Int, log2.(dims)))
end

function grid_R(gridsize::Int)
    R = trunc(Int, log2(gridsize))
    # fermionic or bosonic grid
    # need not be satisfied because three grids may be summed as well, yielding a size of e.g. 3*2^R + 1
    if (gridsize - 2^R) == 0 || (gridsize - 2^R) == 1 
        return R
    else
        @DEBUG begin
            @warn "Gridsize $gridsize is not of the form 2^R + 0/1\n"
            true;
        end "Gridsize is $gridsize"
        return R+1
    end
end

function grid_R(tucker::TCI4Keldysh.AbstractTuckerDecomp{D}) :: Int where {D}
    return maximum(grid_R.([size(leg, 1) for leg in tucker.legs]))
end

function grid_R(sizetuple::NTuple{N, Int}) where {N}
    return maximum(grid_R.(sizetuple))
end

"""
Determine number of bits needed for quantics representation of Gp.
Gp is required to have 2^R fermionic or 2^R+1 bosonic frequencies in each direction.
"""
function grid_R(Gp::PartialCorrelator_reg{D}) :: Int where {D}
    extωsizes = ntuple(i -> length(Gp.ωs_ext[i]), D)
    R = grid_R(extωsizes)
    @assert all(extωsizes .>= 2^R) "Need external frequency grids of sizes 2^R or 2^R+1 (R=$R)"
    return R
end

"""
Determine number of bits needed for quantics representation of GF.
GF is required to have 2^R fermionic or 2^R+1 bosonic frequencies in each direction for each partial correlator.
"""
function grid_R(GF::FullCorrelator_MF{D}) :: Int where {D}
    R = grid_R(GF.Gps[1])
    for i in 2:length(GF.Gps)
        R_ = grid_R(GF.Gps[i])
        @assert R_==R "All partial correlators must have the same grid size"
    end
    return R
end

"""
Compress frequency kernels (legs) of partial correlator to MPS of given length.
CAREFUL: Compression is done via SVD!
Excessive bits are zeropadded.
"""
function compress_frequencykernels(R::Int, psf::AbstractTuckerDecomp{D}; tags=("ω","eps"), tolerance=1e-12) :: Vector{MPS} where {D}
    # TODO: Avoid zeropadding, write function instead...
    om_upper_ids = [min(2^grid_R(size(leg, 1)), size(leg,1)) for leg in psf.legs]
    legs = [zeropad_array(psf.legs[i][1:om_upper_ids[i], :], R) for i in eachindex(psf.legs)]
    # heatmap(log10.(abs.(legs[2])))
    # savefig("foo.png")
    # error("plotting")
    qtt_Kernels = [TCI4Keldysh.fatTensortoQTCI(legs[i]; tolerance=tolerance) for i in 1:D]
    mps_Kernels = [TCI4Keldysh.QTCItoMPS(qtt_Kernels[i], ("$(tags[1])$i", "$(tags[2])$i")) for i in 1:D]
    return mps_Kernels
end

"""
Compress 3-dimensional frequency kernel
"""
function compress_full_frequencykernel(R::Int, psf::AbstractTuckerDecomp{3}; tags=("ω", "eps"), tolerance=1.e-8) :: MPS
    om_upper_ids = [min(2^grid_R(size(leg, 1)), size(leg,1)) for leg in psf.legs]
    eps_sizes = [size(psf.legs[i], 2) for i in 1:3]

    function oneD_kernelfun(a::Int, b::Int, k_id::Int)
        if (a <= om_upper_ids[k_id]) && (b <= eps_sizes[k_id]) 
            return  psf.legs[k_id][a,b]
        else
            return zero(ComplexF64)
        end
    end

    function kernelfun(om1::Int, eps1::Int, om2::Int, eps2::Int, om3::Int, eps3::Int)
        return oneD_kernelfun(om1, eps1, 1) * oneD_kernelfun(om2, eps2, 2) * oneD_kernelfun(om3, eps3, 3) 
    end

    qtt, _, _ = quanticscrossinterpolate(ComplexF64, kernelfun, (ntuple(i -> 2^R, 2*3)); tolerance=tolerance)

    mps_kernel = TCI4Keldysh.QTCItoMPS(qtt, ntuple(i -> (rem(i,2)==1 ? "$(tags[1])$(div(i+1,2))" : "$(tags[2])$(div(i,2))"), 2*3))
    return mps_kernel
end

function test_compress_full_frequencykernel()
    # get PSF
    R = 7
    npt = 4
    perm_idx = 1
    beta = 10.0
    perms = [p for p in permutations(collect(1:npt))]
    perm = perms[perm_idx]
    path = npt<=3 ? "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/" : joinpath("data/SIAM_u=0.50/PSF_nz=2_conn_zavg/", "4pt")
    Ops = if npt==2
            ["F1", "F1dag"][perm]
        elseif npt==3
            ["F1", "F1dag", "Q34"][perm]
        else
            ["F1", "F1dag", "F3", "F3dag"][perm]
        end
    spin = 1
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    ωconvMat_sum = cumsum(ωconvMat[perm[1:(npt-1)],:]; dims=1)
    Gp = TCI4Keldysh.PartialCorrelator_reg(npt-1, 1/beta, R, path, Ops, ωconvMat_sum; flavor_idx=spin, nested_ωdisc=false)

    tolerance = 1.e-3
    printstyled("Compress 1D...\n"; color=:blue)
    onedkernels = compress_frequencykernels_light(R, Gp.tucker; tolerance=tolerance)
    printstyled("Compress 3D...\n"; color=:blue)
    fullkernel = compress_full_frequencykernel(R, Gp.tucker; tolerance=tolerance)

    epsvec = fill(1,R)
    for i in 1:10
        samplevecs = [rand([1,2], R)  for _ in 1:3]
        # ugly...
        refval = prod([eval(onedkernels[i], QuanticsGrids.interleave_dimensions(samplevecs[i], epsvec)) for i in eachindex(samplevecs)])
        fullval = eval(fullkernel, QuanticsGrids.interleave_dimensions(samplevecs[1], epsvec, samplevecs[2], epsvec, samplevecs[3], epsvec))

        @show refval
        @show fullval
        @show abs(refval - fullval)
    end
end

"""
Compress frequency kernels (legs) of partial correlator to MPS of given length.
Excessive bits are zeropadded via a function.
Compression is done via TCI.
"""
function compress_frequencykernels_light(R::Int, psf::AbstractTuckerDecomp{D}; tags=("ω","eps"), tolerance=1e-12) :: Vector{MPS} where {D}
    om_upper_ids = [min(2^grid_R(size(leg, 1)), size(leg,1)) for leg in psf.legs]
    eps_sizes = [size(psf.legs[i], 2) for i in 1:D]
    qtt_Kernels = Vector{QuanticsTCI.QuanticsTensorCI2{ComplexF64}}(undef, D)
    for i in eachindex(psf.legs)

        function kernelfun(a::Int, b::Int)
            if (a <= om_upper_ids[i]) && (b <= eps_sizes[i]) 
                return  psf.legs[i][a,b]
            else
                return zero(ComplexF64)
            end
        end

        qtt, _, _ = quanticscrossinterpolate(ComplexF64, kernelfun, (2^R, 2^R); tolerance=tolerance)
        qtt_Kernels[i] = qtt
    end
    

    mps_Kernels = [TCI4Keldysh.QTCItoMPS(qtt_Kernels[i], ("$(tags[1])$i", "$(tags[2])$i")) for i in 1:D]
    return mps_Kernels
end

"""
Creates local copy of tucker decomposition, thus the input remains untouched.
"""
function TD_to_MPS_via_TTworld(broadenedPsf_::TCI4Keldysh.AbstractTuckerDecomp{1}; tolerance::Float64=1e-12, cutoff::Float64=1e-12, tcikernels=true)
    D = 1

    broadenedPsf = deepcopy(broadenedPsf_)
    R = grid_R(broadenedPsf)
    @show size(broadenedPsf.center)
    @show size.(broadenedPsf.legs)
    @show size.(broadenedPsf.ωs_legs)
    @show size.(broadenedPsf.ωs_center)

    @warn "Singular values are not shifted to center!"
    # TCI4Keldysh.@TIME TCI4Keldysh.shift_singular_values_to_center!(broadenedPsf) "Shifting singular values."

    Adisc_size = size(broadenedPsf.center)
    function evalAdisc(i...)
        if all(i .<= Adisc_size)        
            return broadenedPsf.center[i...]
        else
            return zero(Float64)
        end
    end
    padded_size = ntuple(i -> 2^R, ndims(broadenedPsf.center))
    @show maximum(abs.(broadenedPsf.center))
    println("  Interpolating...")
    qtt_Adisc, _, _ = quanticscrossinterpolate(
        eltype(broadenedPsf.center),
        evalAdisc,
        padded_size;
        tolerance=tolerance)
    
    # qtt_Adisc_padded = TCI4Keldysh.zeropad_QTCI2(qtt_Adisc; N=0)
    qtt_Adisc_padded = qtt_Adisc

    # convert qtt for Tucker center
    mps_Adisc_padded = TCI4Keldysh.QTCItoMPS(qtt_Adisc_padded, ntuple(i->"eps$i", D))

    mps_Kernels = if tcikernels
                    compress_frequencykernels_light(R, broadenedPsf; tolerance=tolerance)
                else
                    legs = [zeropad_array(leg[1:2^grid_R(size(leg, 1)), :], R) for leg in broadenedPsf.legs]
                    qtt_Kernels = [TCI4Keldysh.fatTensortoQTCI(legs[i]; tolerance=tolerance) for i in 1:D]
                    [TCI4Keldysh.QTCItoMPS(qtt_Kernels[i], ("ω$i", "eps$i")) for i in 1:D]
                end

    # contract kernel with tucker center
    k1 = mps_Kernels[1]
    TCI4Keldysh._adoptinds_by_tags!(k1, mps_Adisc_padded, "eps1",  "eps1", R)
    k1MPO = MPO([k1[2*r-1]*k1[2*r] for r in 1:R])

    TCI4Keldysh.@TIME res = apply(k1MPO, mps_Adisc_padded; alg="densitymatrix", cutoff=cutoff, use_absolute_cutoff=true) "Contraction 1"

    return res
end


"""
cutoff: cutoff for densitymatrix algorithm in MPO-MPO contraction; small cutoff affordable in 2D
Creates local copy of tucker decomposition, thus the input remains untouched.
"""
function TD_to_MPS_via_TTworld(broadenedPsf_::TCI4Keldysh.AbstractTuckerDecomp{2}; tolerance::Float64=1e-12, cutoff::Float64=1e-12, tcikernels=true)
    D = 2

    # fit algorithm works for D=2 (in contrast to D=3...)
    # kwargs = Dict(:alg=>"fit")

    broadenedPsf = deepcopy(broadenedPsf_)
    kwargs = Dict(:alg=>"densitymatrix", :cutoff=>cutoff, :use_absolute_cutoff=>true)

    R = grid_R(broadenedPsf)

    # @warn "Singular values are not shifted to center!"
    # TCI4Keldysh.@TIME TCI4Keldysh.shift_singular_values_to_center!(broadenedPsf) "Shifting singular values."
    TCI4Keldysh.@TIME svd_kernels!(broadenedPsf; cutoff=tolerance*1.e-2) "Shifting singular values."

    # find nonzero value
    pivot = findfirst(x -> abs(x)>tolerance, broadenedPsf.center)
    # qpivot = Vector{Int}[]
    # for i in 1:D
    #     push!(qpivot, QuanticsGrids.index_to_quantics(pivot[i]; numdigits=R, base=2))
    # end
    # initpivot = QuanticsGrids.interleave_dimensions(qpivot...)
    # @show initpivot

    #compress PSF
    qtt_Adisc, _, _ = quanticscrossinterpolate(
            zeropad_array(broadenedPsf.center, R),
            [collect(Tuple(pivot))];
            tolerance=tolerance
        )  
    
    # pad qtt:
    # nonzeroinds_left=ones(Int, (R-R_Adisc)*D)
    # qtt_Adisc_padded = TCI4Keldysh.zeropad_QTCI2(qtt_Adisc; N=R-R_Adisc, nonzeroinds_left)
    qtt_Adisc_padded = TCI4Keldysh.zeropad_QTCI2(qtt_Adisc; N=0)
    # all(qtt_Adisc_padded.tci.sitetensors[(R-R_Adisc)*D+1:end] .== qtt_Adisc.tci.sitetensors)
    # Adisc is complex-valued...
    # all(getindex.(argmax.(qtt_Adisc_padded.tci.sitetensors[1:(R-R_Adisc)*D]), 2) .== nonzeroinds_left)

    # convert qtt for Tucker center
    mps_Adisc_padded = TCI4Keldysh.QTCItoMPS(qtt_Adisc_padded, ntuple(i->"eps$i", D))

    mps_Kernels = if tcikernels
                    compress_frequencykernels_light(R, broadenedPsf; tolerance=tolerance)
                else
                    legs = [zeropad_array(leg[1:2^grid_R(size(leg, 1)), :], R) for leg in broadenedPsf.legs]
                    qtt_Kernels = [TCI4Keldysh.fatTensortoQTCI(legs[i]; tolerance=tolerance) for i in 1:D]
                    [TCI4Keldysh.QTCItoMPS(qtt_Kernels[i], ("ω$i", "eps$i")) for i in 1:D]
                end

    ### contract Kernel with Tucker center

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

    # compare to partial contraction
    @DEBUG begin
        printstyled("  -- Test contraction\n"; color=:blue)
        @show size(broadenedPsf.center)
        Acontr1 = contract_1D_Kernels_w_Adisc_mp_partial(broadenedPsf.legs, broadenedPsf.center, 1)
        @show size(Acontr1)
        tags = ("ω1", "eps2")
        fatty = TCI4Keldysh.MPS_to_fatTensor(ab; tags=tags)
        @show size(fatty)
        diffslice = [1:2^grid_R(size(Acontr1,1)), 1:size(Acontr1,2)]
        diff = abs.(fatty[diffslice...] - Acontr1[diffslice...])
        meandiff = sum(diff)/prod(size(diff))
        @show meandiff
        @show maximum(diff)
        meandiff<=1e-8;
    end "\n ---- Testing contraction 1 ----\n"

    ITensors.truncate!(ab; cutoff=cutoff, use_absolute_cutoff=true)

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

    # compare to partial contraction
    @DEBUG begin
        printstyled("  -- Test contraction 2\n"; color=:blue)
        @show size(broadenedPsf.center)
        Acontr1 = contract_1D_Kernels_w_Adisc_mp_partial(broadenedPsf.legs, broadenedPsf.center, 2)
        tags = ("ω1", "ω2")
        fatty = TCI4Keldysh.MPS_to_fatTensor(res; tags=tags)
        diffslice = [1:2^grid_R(size(Acontr1,1)), 1:2^grid_R(size(Acontr1,2))]
        diff = abs.(fatty[diffslice...] - Acontr1[diffslice...])
        @show sum(diff)/prod(size(diff))
        @show maximum(diff)
        # heatmap(abs.(fatty))
        # savefig("contr2_tci.png")
        # heatmap(abs.(Acontr1))
        # savefig("contr2_ref.png")
        # heatmap(log10.(abs.(diff)))
        # savefig("contr2_diff.png")
        maximum(diff)<=1e-6;
    end "\n ---- Testing contraction 2 ----\n"

    # truncate res, not ab!
    ITensors.truncate!(res; cutoff=cutoff, use_absolute_cutoff=true)
    
    # siteinds(res)
    return res
end




"""
cutoff: cutoff for densitymatrix algorithm in MPO-MPO contraction
Creates local copy of tucker decomposition, thus the input remains untouched.
"""
function TD_to_MPS_via_TTworld(broadenedPsf_::TCI4Keldysh.AbstractTuckerDecomp{3}; tolerance::Float64=1e-12, cutoff::Float64=1e-5, tcikernels::Bool=true)

    D = 3
    # scaling for maxbondim(MPS)=m, maxbonddim(MPO)=k: m^3k + m^2k^2
    # kwargs = Dict(:alg=>"fit", :nsweeps=>10, :cutoff=>1e-15)

    # scaling for maxbondim(MPS)=m, maxbonddim(MPO)=k: m^3k^2 + m^2k^3
    # works and improves with increasing cutoff; finite cutoff makes contractions feasible
    kwargs = Dict(:alg=>"densitymatrix", :cutoff=>cutoff, :use_absolute_cutoff=>true)


    broadenedPsf = deepcopy(broadenedPsf_)
    @show size.(broadenedPsf.legs,1)
    R = grid_R(broadenedPsf)

    # @warn "Singular values are not shifted to center!"
    begin
        psf_copy = deepcopy(broadenedPsf)
        printstyled("  Before old shift: $(size(psf_copy.center))\n"; color=:cyan)
        shift_singular_values_to_center!(psf_copy)
        printstyled("  After old shift: $(size(psf_copy.center))\n"; color=:cyan)
    end
    printstyled("  Before new shift: $(size(broadenedPsf.center))\n"; color=:cyan)
    @TIME svd_kernels!(broadenedPsf; cutoff=tolerance*1.e-2) "Shifting singular values."
    printstyled("  After new shift: $(size(broadenedPsf.center))\n"; color=:cyan)

    qtt_Adisc, _, _ = quanticscrossinterpolate(
        zeropad_array(broadenedPsf.center, R); tolerance=tolerance
        )  

    # pad qtt:
    # nonzeroinds_left=ones(Int, (R-R_Adisc)*D)
    # qtt_Adisc_padded = TCI4Keldysh.zeropad_QTCI2(qtt_Adisc; N=R-R_Adisc, nonzeroinds_left)
    qtt_Adisc_padded = TCI4Keldysh.zeropad_QTCI2(qtt_Adisc; N=0)
    # all(qtt_Adisc_padded.tci.sitetensors[(R-R_Adisc)*D+1:end] .== qtt_Adisc.tci.sitetensors)
    # all(getindex.(argmax.(qtt_Adisc_padded.tci.sitetensors[1:(R-R_Adisc)*D]), 2) .== nonzeroinds_left)



    # convert qtt for Tucker center
    mps_Adisc_padded = TCI4Keldysh.QTCItoMPS(qtt_Adisc_padded, ntuple(i->"eps$i", D))

    #legs = [zeros(eltype(broadenedPsf.legs[1]), 2^R, 2^R) for i in 1:D]
    #for i in 1:3
    #    legs[i][:,1:2^R_Adisc] .= broadenedPsf.legs[i][1:end-1,end-2^6+1:end]
    #end
    # legs = [zeropad_array(broadenedPsf.legs[i][1:end-residue,:]) for i in 1:D]

    mps_Kernels = if tcikernels
                    printstyled(" --- Compressing kernels with TCI\n"; color=:blue)
                    compress_frequencykernels_light(R, broadenedPsf; tolerance=tolerance)
                else
                    om_upper_ids = [min(2^grid_R(size(leg, 1)), size(leg,1)) for leg in broadenedPsf.legs]
                    legs = [zeropad_array(broadenedPsf.legs[i][1:om_upper_ids[i], :], R) for i in eachindex(broadenedPsf.legs)]
                    qtt_Kernels = [TCI4Keldysh.fatTensortoQTCI(legs[i]; tolerance=1e-15) for i in 1:D]
                    [TCI4Keldysh.QTCItoMPS(qtt_Kernels[i], ("ω$i", "eps$i")) for i in 1:D]
                end

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

    @DEBUG begin
        printstyled("  -- Test Tucker\n"; color=:blue)
        # check center
        Adisc = broadenedPsf.center
        @show size(Adisc)
        tags = ("eps1", "eps2", "eps3")
        fatty = TCI4Keldysh.MPS_to_fatTensor(mps_Adisc_padded; tags=tags)
        @show size(fatty)
        diffslice = size(Adisc)
        diff = abs.(fatty[diffslice...] - Adisc[diffslice...])
        @assert maximum(abs.(diff))<=1e-12

        # check kernels
        fatKernel1 = TCI4Keldysh.MPS_to_fatTensor(mps_Kernel1_exp; tags=("ω1", "eps1", "eps3"))
        # fatKernel2 = TCI4Keldysh.MPS_to_fatTensor(mps_Kernel2_exp; tags=("ω2", "eps2", "eps3"))
        # fatKernel3 = TCI4Keldysh.MPS_to_fatTensor(mps_Kernel3_exp; tags=("eps1", "ω3", "eps3"))
        # fatKernels = [fatKernel1, fatKernel2, fatKernel3]
        fatKernels = [fatKernel1]

        # invariance across dummy dim
        for ik in eachindex(fatKernels)
            dd = ik==3 ? 1 : 3
            k = fatKernels[ik]
            for i in 1:size(k, dd)
                slice = k==3 ? [i, Colon(), Colon()] : [Colon(), Colon(), i]
                refslice = k==3 ? [1, Colon(), Colon()] : [Colon(), Colon(), 1]
                @assert norm(k[slice...] - k[refslice...]) <= 1e-10
            end
        end

        # comparison to tucker legs
        refkernel1 = broadenedPsf.legs[1]
        C = norm(fatKernel1[1:size(refkernel1, 1), 1:size(refkernel1, 2), 1] - refkernel1)
        @assert C <= 1e-12 "error is $C"

        true;
    end "\n ---- Check Tucker ----\n"

    printstyled("\n  Ranks before contraction 1 MPS/Kernel: $(rank(mps_Adisc_padded)) / $(rank(mps_Kernel1_exp))\n"; color=:magenta)
    printstyled("        Cutoff: $cutoff\n"; color=:magenta)

    @TIME ab = Quantics.automul(
            mps_Kernel1_exp,
            mps_Adisc_padded;
            tag_row = "ω1",
            tag_shared = "eps1",
            tag_col = "eps2",
            kwargs...
        ) "Contraction 1"

    # compare to partial contraction
    # WORKS for alg="densitymatrix"
    @DEBUG begin
        printstyled("  -- Test contraction\n"; color=:blue)
        @show dim.(linkinds(mps_Kernel1_exp))
        @show dim.(linkinds(mps_Adisc_padded))
        @show size(broadenedPsf.center)
        Acontr1 = contract_1D_Kernels_w_Adisc_mp_partial(broadenedPsf.legs, broadenedPsf.center, 1)
        @show size(Acontr1)
        tags = ("ω1", "eps2", "eps3")
        fatty = TCI4Keldysh.MPS_to_fatTensor(ab; tags=tags)
        @show size(fatty)
        diffslice = [1:2^grid_R(size(Acontr1,1)), 1:size(Acontr1,2), 1:size(Acontr1,3)]
        diff = abs.(fatty[diffslice...] - Acontr1[diffslice...])
        @show sum(diff)/prod(size(diff))
        @show argmax(diff)

        # plot
        complexfun = abs
        slice_idx = 5
        # diffslice[1] = 70:128
        heatmap(complexfun.(Acontr1[diffslice[1], slice_idx, diffslice[3]]))
        savefig("contr1_ref.png")
        # heatmap(complexfun.(fatty[diffslice[1], slice_idx, diffslice[3]]))
        heatmap(complexfun.(fatty[:, slice_idx, :]))
        savefig("contr1_tci.png")
        heatmap(log10.(abs.(complexfun.(diff[diffslice[1], slice_idx, diffslice[3]]))))
        savefig("contr1_diff.png")
        true;
    end "\n ---- Testing contraction 1 ----\n"

    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel2_exp, ab, "eps2",  "eps2", R)
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel2_exp, ab, "dummy", "eps3", R)

    ITensors.truncate!(ab; cutoff=1e-14, use_absolute_cutoff=true)
    ITensors.truncate!(mps_Kernel2_exp; cutoff=1e-14, use_absolute_cutoff=true)

    printstyled("\n  Ranks before contraction 2 MPS/Kernel: $(rank(ab)) / $(rank(mps_Kernel2_exp))\n"; color=:magenta)

    #findallsiteinds_by_tag(siteinds(); tag="eps2")
    @TIME ab2 = Quantics.automul(
        ab,
        mps_Kernel2_exp;
            tag_row = "ω1",
            tag_shared = "eps2",
            tag_col = "ω2",
            kwargs...
        ) "Contraction 2"

    #siteinds(ab2)
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel3_exp, ab2, "eps3",  "eps3", R)
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel3_exp, ab2, "dummy", "ω1"  , R)

    ITensors.truncate!(ab2; cutoff=1e-14, use_absolute_cutoff=true)
    ITensors.truncate!(mps_Kernel3_exp; cutoff=1e-14, use_absolute_cutoff=true)
    # much less accurate, also with use_absolute_cutoff=false
    # ITensors.truncate!(ab2; cutoff=cutoff*1e-2, use_absolute_cutoff=true)
    # ITensors.truncate!(mps_Kernel3_exp; cutoff=cutoff*1e-2, use_absolute_cutoff=true)

    printstyled("\n  Ranks before contraction 3 MPS/Kernel: $(rank(ab2)) / $(rank(mps_Kernel3_exp))\n"; color=:magenta)

    @TIME res = Quantics.automul(
        ab2,
        mps_Kernel3_exp;
            tag_row = "ω2",
            tag_shared = "eps3",
            tag_col = "ω3",
            kwargs...
        ) "Contraction 3"

    ITensors.truncate!(res; cutoff=1e-14, use_absolute_cutoff=true)
    printstyled("\n  MPS rank after contraction 3 + truncation: $(rank(res))\n"; color=:magenta)

    return res
end


"""
Kernel contractions without elementwise multiplication / dummy indices.
See whether this makes alg=densitymatrix faster and fixes problem with alg=fit
"""
function noewmul_TD_to_MPS_via_TTworld(broadenedPsf::TCI4Keldysh.AbstractTuckerDecomp{3}; tolerance::Float64=1e-14, alg="tci2")

    error("Function noewmul_TD_to_MPS_via_TTworld not yet implemented!")

    D = 3
    #tolerance = 1e-14
    # scaling for maxbondim(MPS)=m, maxbonddim(MPO)=k: m^3k + m^2k^2

    # scaling for maxbondim(MPS)=m, maxbonddim(MPO)=k: m^3k^2 + m^2k^3
    alg = "fit"


    R = maximum(grid_R.([size(leg, 1) for leg in broadenedPsf.legs]))
    @show R
    # R_Adisc = padding_R(size(broadenedPsf.center))
    # @show R_Adisc

    @TIME TCI4Keldysh.shift_singular_values_to_center!(broadenedPsf) "Shifting singular values."

    qtt_Adisc, _, _ = quanticscrossinterpolate(
        zeropad_array(broadenedPsf.center, R); tolerance=tolerance
        )  

    # pad qtt:
    # nonzeroinds_left=ones(Int, (R-R_Adisc)*D)
    # qtt_Adisc_padded = TCI4Keldysh.zeropad_QTCI2(qtt_Adisc; N=R-R_Adisc, nonzeroinds_left)
    qtt_Adisc_padded = TCI4Keldysh.zeropad_QTCI2(qtt_Adisc; N=0)
    # all(qtt_Adisc_padded.tci.sitetensors[(R-R_Adisc)*D+1:end] .== qtt_Adisc.tci.sitetensors)
    # all(getindex.(argmax.(qtt_Adisc_padded.tci.sitetensors[1:(R-R_Adisc)*D]), 2) .== nonzeroinds_left)



    # convert qtt for Tucker center
    mps_Adisc_padded = TCI4Keldysh.QTCItoMPS(qtt_Adisc_padded, ntuple(i->"eps$i", D))

    #legs = [zeros(eltype(broadenedPsf.legs[1]), 2^R, 2^R) for i in 1:D]
    #for i in 1:3
    #    legs[i][:,1:2^R_Adisc] .= broadenedPsf.legs[i][1:end-1,end-2^6+1:end]
    #end
    # legs = [zeropad_array(broadenedPsf.legs[i][1:end-residue,:]) for i in 1:D]
    legs = [zeropad_array(leg[1:2^grid_R(size(leg, 1)), :], R) for leg in broadenedPsf.legs]
    mps_Kernels = Array{MPS}(undef, D)
    for i in 1:D
        function kernel(om::Int, eps::Int, _::Int, _::Int)
            return legs[i][om, eps]             
        end
        qtt, _, _ = quanticscrossinterpolate(R, fill(2^R, 4); tolerance=1e-14, unfoldingscheme=:interleaved)
        tags = i<3 ? ("ω$i", "eps$i", "eps3", "eps3prime") : ("eps1", "eps1prime", "ω$i", "eps$i")
        mps_Kernels[i] = TCI4Keldysh.QTCItoMPS(qtt, tags)
    end

    mps_Kernel1_exp = mps_Kernels[1]
    mps_Kernel2_exp = mps_Kernels[2]
    mps_Kernel3_exp = mps_Kernels[3]

    # adopt shared site indices from Tucker center
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel1_exp, mps_Adisc_padded, "eps1",  "eps1", R)
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel1_exp, mps_Adisc_padded, "eps3", "eps3", R)

    @DEBUG begin
        printstyled("  -- Test Tucker\n"; color=:blue)

        # check center
        Adisc = broadenedPsf.center
        @show size(Adisc)
        tags = ("eps1", "eps2", "eps3")
        fatty = TCI4Keldysh.MPS_to_fatTensor(mps_Adisc_padded; tags=tags)
        @show size(fatty)
        diffslice = size(Adisc)
        diff = abs.(fatty[diffslice...] - Adisc[diffslice...])
        @assert maximum(abs.(diff))<=1e-12

        # # check kernels
        # fatKernel1 = TCI4Keldysh.MPS_to_fatTensor(mps_Kernel1_exp; tags=("ω1", "eps1", "eps3"))
        # fatKernels = [fatKernel1]

        # # invariance across dummy dim
        # for ik in eachindex(fatKernels)
        #     dd = ik==3 ? 1 : 3
        #     k = fatKernels[ik]
        #     for i in 1:size(k, dd)
        #         slice = k==3 ? [i, Colon(), Colon()] : [Colon(), Colon(), i]
        #         refslice = k==3 ? [1, Colon(), Colon()] : [Colon(), Colon(), 1]
        #         @assert norm(k[slice...] - k[refslice...]) <= 1e-10
        #     end
        # end

        # # comparison to tucker legs
        # refkernel1 = broadenedPsf.legs[1]
        # @assert norm(fatKernel1[1:size(refkernel1, 1), 1:size(refkernel1, 2), 1] - refkernel1) <= 1e-12

        true;
    end "\n ---- Check Tucker ----\n"

    if TCI4Keldysh.VERBOSE()
        mps_idx_info(mps_Adisc_padded)
        mps_idx_info(mps_Kernel1_exp)
    end

    error("NYI")

    # MPO-MPO contraction
    @TIME begin

        # get sites
        tagr1 = "ω1"
        tagr2 = "eps3prime"
        sites_row1 = findallsiteinds_by_tag(siteinds(M1); tag=tagr1)
        sites_row2 = findallsiteinds_by_tag(siteinds(M1); tag=tagr2)
        tags1 = "eps1"
        tags2 = "eps3"
        sites_shared1 = findallsiteinds_by_tag(siteinds(M1); tag=tag_shared)
        tagc1 = "eps2"
        tagc2 = "eps3"
        sites_col1 = findallsiteinds_by_tag(siteinds(M2); tag=tagc1)
        sites_col2 = findallsiteinds_by_tag(siteinds(M2); tag=tagc2)
        sites_col = vcat(sites_col1, sites_col2)
        sites_row = vcat(sites_row1, sites_row2)

        matmul = MatrixMultiplier(sites_row, sites_shared, sites_col)
    
        M1_ = Quantics.asMPO(mps_Kernel1_exp)
        M2_ = Quantics.asMPO(mps_Adisc_padded)
        M1_, M2_ = preprocess(matmul, M1_, M2_)
    
        # M = FastMPOContractions.contract_mpo_mpo(M1_, M2_; alg=alg, kwargs...)
        M = Quantics.contract_mpo_mpo(M1_, M2_; alg=alg, kwargs...)
    
        printstyled("  Postprocess\n"; color=:blue)
        M = Quantics.postprocess(matmul, M)
        M = Quantics.postprocess(ewmul, M)
        printstyled("  ----------\n"; color=:blue)
    
        if in(:maxdim, keys(kwargs))
            truncate!(M; maxdim=kwargs[:maxdim])
        end 
    end "Contraction 1"

    # @TIME ab = Quantics.automul(
    #         mps_Kernel1_exp,
    #         mps_Adisc_padded;
    #         tag_row = "ω1",
    #         tag_shared = "eps1",
    #         tag_col = "eps2",
    #         kwargs...
    #     ) "Contraction 1"

    # compare to partial contraction
    # WORKS for alg="densitymatrix"
    @DEBUG begin
        printstyled("  -- Test contraction\n"; color=:blue)
        @show size(broadenedPsf.center)
        Acontr1 = contract_1D_Kernels_w_Adisc_mp_partial(broadenedPsf.legs, broadenedPsf.center, 1)
        @show size(Acontr1)
        tags = ("ω1", "eps2", "eps3")
        fatty = TCI4Keldysh.MPS_to_fatTensor(ab; tags=tags)
        @show size(fatty)
        diffslice = [1:2^grid_R(size(Acontr1,1)), 1:size(Acontr1,2), 1:size(Acontr1,3)]
        diff = abs.(fatty[diffslice...] - Acontr1[diffslice...])
        @show sum(diff)/prod(size(diff))
        @show argmax(diff)

        # plot
        complexfun = abs
        slice_idx = 5
        # diffslice[1] = 70:128
        heatmap(complexfun.(Acontr1[diffslice[1], slice_idx, diffslice[3]]))
        savefig("contr1_ref.png")
        # heatmap(complexfun.(fatty[diffslice[1], slice_idx, diffslice[3]]))
        heatmap(complexfun.(fatty[:, slice_idx, :]))
        savefig("contr1_tci.png")
        heatmap(log10.(abs.(complexfun.(diff[diffslice[1], slice_idx, diffslice[3]]))))
        savefig("contr1_diff.png")
        false;
    end "\n ---- Testing contraction 1 ----\n"

    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel2_exp, ab, "eps2",  "eps2", R)
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel2_exp, ab, "dummy", "eps3", R)

    if TCI4Keldysh.VERBOSE()
        mps_idx_info(ab)
        mps_idx_info(mps_Kernel2_exp)
    end
    ITensors.truncate!(ab; cutoff=1e-14, use_absolute_cutoff=true)
    ITensors.truncate!(mps_Kernel2_exp; cutoff=1e-14, use_absolute_cutoff=true)

    #findallsiteinds_by_tag(siteinds(); tag="eps2")
    @TIME ab2 = Quantics.automul(
        ab,
        mps_Kernel2_exp;
            tag_row = "ω1",
            tag_shared = "eps2",
            tag_col = "ω2",
            kwargs...
        ) "Contraction 2"

    #siteinds(ab2)
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel3_exp, ab2, "eps3",  "eps3", R)
    TCI4Keldysh._adoptinds_by_tags!(mps_Kernel3_exp, ab2, "dummy", "ω1"  , R)

    if TCI4Keldysh.VERBOSE()
        mps_idx_info(ab2)
        mps_idx_info(mps_Kernel3_exp)
    end
    ITensors.truncate!(ab2; cutoff=1e-14, use_absolute_cutoff=true)
    ITensors.truncate!(mps_Kernel3_exp; cutoff=1e-14, use_absolute_cutoff=true)

    @TIME res = Quantics.automul(
        ab2,
        mps_Kernel3_exp;
            tag_row = "ω2",
            tag_shared = "eps3",
            tag_col = "ω3",
            kwargs...
        ) "Contraction 3"

    if TCI4Keldysh.VERBOSE()
        mps_idx_info(res)
    end
    ITensors.truncate!(res; cutoff=1e-14, use_absolute_cutoff=true)

    return res
end

"""
    function renew_link_tags!(mps::MPS)

Replace tags in common indices of adjacent tensors i, i+1 by "Link, l=i".
Adjacent ITensors should have exactly one Index in common.
"""
function renew_link_tags!(mps::MPS)
    return renew_link_tags!(mps.data)
end

"""
    function renew_link_tags!(mpsdata::Vector{ITensor})
"""
function renew_link_tags!(mpsdata::Vector{ITensor})
    for i in 1:length(mpsdata)-1    
        lidx = only(commoninds(mpsdata[i], mpsdata[i+1]))
        newtag = "Link, l=$i"
        newidx = settags(lidx, newtag)
        oldidxl = inds(mpsdata[i])
        oldidxr = inds(mpsdata[i+1])
        newidxl = replace(oldidxl, lidx=>newidx)
        newidxr = replace(oldidxr, lidx=>newidx)
        replaceinds!(mpsdata[i], oldidxl, newidxl)
        replaceinds!(mpsdata[i+1], oldidxr, newidxr)
    end
end

"""
    function insert_tensor(mpsdata::Vector{ITensor}, t::ITensor, iafter::Int)

Insert tensor into vector of tensor trains, adapting the link indices.
* iafter: Insert after this index. Insert at the start if iafter==0
* t: should have site index in the middle.
"""
function insert_tensor!(mpsdata::Vector{ITensor}, t::ITensor, iafter::Int)
    # adapt indices
    if iafter==0
        tinds = inds(t)
        @assert length(tinds)==2
        lidx = tinds[2]
        ldim = dim(lidx)
        # add leg to first tensor
        addleg = ITensor(ones(eltype(t), ldim), lidx)
        mpsdata[1] *= addleg
    elseif iafter==length(mpsdata)
        tinds = inds(t)
        @assert length(inds(t))==2
        lidx = tinds[1]
        ldim = dim(lidx)
        # add leg to last tensor
        addleg = ITensor(ones(eltype(t), ldim), lidx)
        mpsdata[end] *= addleg
    else
        tleft = mpsdata[iafter]
        tright = mpsdata[iafter+1]
        old_link = only(commoninds(tleft, tright))
        tinds = inds(t)
        @assert length(tinds)==3
        llink = tinds[1]
        rlink = tinds[3]
        @assert dim(old_link)==dim(llink) "Dimension mismatch: $(dim(old_link)) vs $(dim(llink))"
        @assert dim(old_link)==dim(rlink) "Dimension mismatch: $(dim(old_link)) vs $(dim(rlink))"
        replaceind!(tleft, old_link, llink)
        replaceind!(tright, old_link, rlink)
    end
    # insert
    insert!(mpsdata, iafter+1, t)
    renew_link_tags!(mpsdata)
end

"""
Construct 2-leg tensor that contains only ones for a given site index value and zero else.
    * left=true/false gives
        x-  OR  -x
        |        |
"""
function delta_tensor_end(::Type{T}, d::Int, sitedim::Int, linkdim::Int, left::Bool; sitetag::String="Site")::ITensor where {T}
    revfun = left ? identity : reverse
    arr = zeros(T, revfun((sitedim, linkdim)))
    arr[revfun((d, Colon()))...] = ones(linkdim)
    sidx = Index(sitedim, sitetag)
    linktag = left ? "Link, left" : "Link, right"
    lidx = Index(linkdim, linktag)
    return ITensor(arr, revfun([sidx, lidx]))
end

"""
    delta_tensor_end with linkdim=1
"""
function delta_tensor_end(::Type{T}, d::Int, sitedim::Int, left::Bool; sitetag::String="Site")::ITensor where {T}
    return delta_tensor_end(T, d, sitedim, 1, left; sitetag=sitetag)
end

"""
Construct 3-leg tensor that is unity for a given site index value and zero else.
"""
function delta_tensor(::Type{T}, d::Int, sitedim::Int, linkdim::Int; sitetag::String="Site")::ITensor where {T}
    arr = zeros(T, (linkdim, sitedim, linkdim))
    arr[:,d,:] = diagm(ones(T, linkdim))
    sidx = Index(sitedim, sitetag)
    llidx = Index(linkdim, "Link, left")
    rlidx = Index(linkdim, "Link, right")
    return ITensor(arr, [llidx, sidx, rlidx])
end


"""
    function anomalous_TD_to_MPS(Gp::TCI4Keldysh.PartialCorrelator_reg{D}; tolerance=1e-6, cutoff=1.e-20)::MPS where {D}

Compute anomalous contribution to correlator for 3/4-point.
Return MPS that can be added to regular part.
"""
function anomalous_TD_to_MPS_full(Gp::TCI4Keldysh.PartialCorrelator_reg{D}; tolerance, cutoff)::MPS where {D}
    # has (D-1)*R legs
    bos_idx = get_bosonic_idx(Gp)
    zero_idx = findfirst(x -> abs(x)<=1.e-15, Gp.tucker.ωs_legs[bos_idx])
    delta_tag = "ω$bos_idx"

    if D==1
        # only one anomalous value
        ano_val = only(Gp.Adisc_anoβ)
        R = maximum(grid_R.([size(leg, 1) for leg in Gp.tucker.legs]))
        localdims = fill(2, R)
        sites = [Index(localdims[i], "Qubit,$delta_tag=$i") for i in eachindex(localdims)]
        state = QuanticsGrids.index_to_quantics(zero_idx; numdigits=R, base=2)
        return MPS(ComplexF64, sites, state) * ano_val
    else
        Gano_mps = anomalous_TD_to_MPS(Gp; tolerance=tolerance, cutoff=cutoff)
        R = div(length(Gano_mps), (D-1))
        zero_state = QuanticsGrids.index_to_quantics(zero_idx; numdigits=R, base=2)
        Gano_data = Gano_mps.data
        # insert R tensors
        for i in 1:R
            if i==1 && bos_idx==1
                printstyled("\n ---- Before insertion $i\n"; color=:blue)
                for g in Gano_data
                    @show inds(g)
                end
                printstyled(" ----------\n"; color=:blue)
                # left end
                t = delta_tensor_end(ComplexF64, zero_state[i], 2, true; sitetag="$delta_tag=$i")
                iafter = 0
                insert_tensor!(Gano_data, t, iafter)
            elseif i==R && bos_idx==D
                # right end
                t = delta_tensor_end(ComplexF64, zero_state[i], 2, false; sitetag="$delta_tag=$i")
                iafter = R*(D-1) + i-1
                insert_tensor!(Gano_data, t, iafter)
            else
                # d; i-1 tensors have already been inserted
                iafter = (i-1)*(D-1) + bos_idx-1 + i-1
                ltensor = Gano_data[iafter]
                rtensor = Gano_data[iafter+1]
                linkdim = dim(only(commoninds(ltensor, rtensor)))
                t = delta_tensor(ComplexF64, zero_state[i], 2, linkdim; sitetag="$delta_tag=$i")
                insert_tensor!(Gano_data, t, iafter)
            end
        end
        return MPS(Gano_data)
    end
end

"""
    function anomalous_TD_to_MPS(Gp::TCI4Keldysh.PartialCorrelator_reg{D}; tolerance=1e-6, cutoff=1.e-20)::MPS where {D}

Compute anomalous contribution to correlator for 3/4-point.
Return MPS with (D-1)*R legs, i.e., the legs for the bosonic direction have to be added afterwards.
2-point is separate because there the anomalous term consists of a single nonzero value.
"""
function anomalous_TD_to_MPS(Gp::TCI4Keldysh.PartialCorrelator_reg{D}; tolerance=1e-6, cutoff=1.e-20)::MPS where {D}
    @assert !all(Gp.isFermi) "Gp must have bosonic frequency to compute anomalous term"
    @assert D>1 "2-point case is best treated separately"

    bos_idx = get_bosonic_idx(Gp)
    @assert !isnothing(bos_idx)

    # how many bits are required due to frequency grid?
    R = maximum(grid_R.([size(leg, 1) for leg in Gp.tucker.legs]))

    Adisc_ano = dropdims(Gp.Adisc_anoβ; dims=(bos_idx,)) # D-1 dimensional

    # compress Adisc_anoβ
    pivot = findfirst(x -> abs(x)>tolerance, Adisc_ano)
    @TIME qtt_Adisc_ano, _, _ = quanticscrossinterpolate(
        zeropad_array(Adisc_ano, R), [collect(Tuple(pivot))]; tolerance=tolerance
        ) "Compress anomalous part of PSF"
    @show rank(qtt_Adisc_ano)
    tags_Adisc = tuple(["eps$i" for i in 1:D if i!=bos_idx]...)
    mps_Adisc_ano = TCI4Keldysh.QTCItoMPS(qtt_Adisc_ano, tags_Adisc)

    # get kernel
    kernel_mps = compress_anomalous_kernel(Gp, R; tolerance=tolerance)

    for i in 1:D-1
        _adoptinds_by_tags!(kernel_mps, mps_Adisc_ano, tags_Adisc[i], tags_Adisc[i], R)
    end

    # kernel: ω1, eps1, ω2, eps2...
    sites_row = siteinds(kernel_mps)[1:2:end]
    sites_shared = siteinds(kernel_mps)[2:2:end]

    # kernel to MPO
    kernel_mpo = Quantics.asMPO(kernel_mps)
    for (s_row, s_shared) in zip(sites_row, sites_shared)
        kernel_mpo = Quantics.combinesites(kernel_mpo, s_row, s_shared)
    end

    # contract
    Gp_ano = contract(mps_Adisc_ano, kernel_mpo; cutoff=cutoff, use_absolute_cutoff=true)

    return Gp_ano
end

"""
Compress anomalous part of Matsubara frequency kernel to MPS.
"""
function compress_anomalous_kernel(Gp::TCI4Keldysh.PartialCorrelator_reg{D}, R::Int; tolerance=1e-10)::MPS where {D}
    # for now, compress in D-1 dimensions to make subsequent addition to regular part trivial...
    bos_idx = get_bosonic_idx(Gp)

    grid_ids = [i for i in 1:D if i!=bos_idx]
    leg_sizes = ntuple(i -> length(Gp.tucker.ωs_legs[grid_ids[i]]), D-1)
    omdisc_sizes = ntuple(i -> length(Gp.tucker.ωs_center[grid_ids[i]]), D-1)

    kernelfun = if D==3
                    function anomalous_kernel3(i1::Int, i1p::Int, i2::Int, i2p::Int)
                        om_ids = (i1, i2)
                        omprime_ids = (i1p, i2p)

                        # index out of bounds
                        if any(om_ids .> leg_sizes) || any(omprime_ids .> omdisc_sizes)
                            return zero(ComplexF64)
                        end

                        # TODO: This is atrocious...
                        oms = [Gp.tucker.ωs_legs[grid_ids[i]][om_ids[i]] for i in eachindex(grid_ids)]
                        omprimes = [Gp.tucker.ωs_center[grid_ids[i]][omprime_ids[i]] for i in eachindex(grid_ids)]

                        return eval_ano_matsubara_kernel(oms, omprimes, 1/Gp.T)
                    end
                elseif  D==2
                    function anomalous_kernel2(i1::Int, i1p::Int)
                        om_ids = (i1,)
                        omprime_ids = (i1p,)

                        # if index out of bounds
                        if any(om_ids .> leg_sizes) || any(omprime_ids .> omdisc_sizes)
                            return zero(ComplexF64)
                        end

                        oms = [Gp.tucker.ωs_legs[grid_ids[i]][om_ids[i]] for i in eachindex(grid_ids)]
                        omprimes = [Gp.tucker.ωs_center[grid_ids[i]][omprime_ids[i]] for i in eachindex(grid_ids)]

                        return eval_ano_matsubara_kernel(oms, omprimes, 1/Gp.T)
                    end
                end    

    kernel_qtt, _, _ = quanticscrossinterpolate(ComplexF64, kernelfun, ntuple(i -> 2^R, 2*(D-1)); tolerance=tolerance)

    println("  Anomalous kernel before truncation: $(rank(kernel_qtt))")
    println("  Worst case for $R bits: $(maximum(worstcase_bonddim(fill(2, 2*(D-1)*R))))")

    tags = tuple(vcat([["ω$i", "eps$i"] for i in grid_ids]...)...)
    kernel_mps = TCI4Keldysh.QTCItoMPS(kernel_qtt, tags)

    truncate!(kernel_mps; cutoff=1e-14, use_absolute_cutoff=true)
    println("  Anomalous kernel after truncation: $(rank(kernel_mps))")

    return kernel_mps
end

function compress_anomalous_kernel_patched(Gp::TCI4Keldysh.PartialCorrelator_reg{D}, R::Int; tolerance=1e-10) where {D}

    bos_idx = get_bosonic_idx(Gp)

    grid_ids = [i for i in 1:D if i!=bos_idx]
    leg_sizes = ntuple(i -> length(Gp.tucker.ωs_legs[grid_ids[i]]), D-1)
    omdisc_sizes = ntuple(i -> length(Gp.tucker.ωs_center[grid_ids[i]]), D-1)

    kernelfun = if D==3
        function anomalous_kernel(i1::Int, i1p::Int, i2::Int, i2p::Int)
            om_ids = (i1, i2)
            omprime_ids = (i1p, i2p)

            # index out of bounds
            if any(om_ids .> leg_sizes) || any(omprime_ids .> omdisc_sizes)
                return zero(ComplexF64)
            end

            # TODO: This is atrocious...
            oms = [Gp.tucker.ωs_legs[grid_ids[i]][om_ids[i]] for i in eachindex(grid_ids)]
            omprimes = [Gp.tucker.ωs_center[grid_ids[i]][omprime_ids[i]] for i in eachindex(grid_ids)]

            return eval_ano_matsubara_kernel(oms, omprimes, 1/Gp.T)
        end
    elseif  D==2
        function anomalous_kernel(i1::Int, i1p::Int)
            om_ids = (i1,)
            omprime_ids = (i1p,)

            # if index out of bounds
            if any(om_ids .> leg_sizes) || any(omprime_ids .> omdisc_sizes)
                return zero(ComplexF64)
            end

            oms = [Gp.tucker.ωs_legs[grid_ids[i]][om_ids[i]] for i in eachindex(grid_ids)]
            omprimes = [Gp.tucker.ωs_center[grid_ids[i]][omprime_ids[i]] for i in eachindex(grid_ids)]

            return eval_ano_matsubara_kernel(oms, omprimes, 1/Gp.T)
        end
    end    

    # prepare patching TCI
    # grid origin is zero since kernel function takes grid indices anyways
    grid = InherentDiscreteGrid{2*(D-1)}(R; unfoldingscheme=:fused)
    proj_tt = patchedTCI(kernelfun, grid; tolerance=tolerance, maxbonddim=500)

    # proj_tt contains vector 'data' of ProjTensorTrain objects, which in turn contain TCI.TensorTrain objects, also called 'data'
    proj_tt_info(proj_tt)

    @show eltype(proj_tt.data)
    @show typeof(proj_tt.projector)

    error("NYI")
    println("Before truncation: $(rank(kernel_qtt))")

    tags = tuple(vcat([["ω$i", "eps$i"] for i in grid_ids]...)...)
    kernel_mps = TCI4Keldysh.QTCItoMPS(kernel_qtt, tags)

    truncate!(kernel_mps; cutoff=1e-14, use_absolute_cutoff=true)
    println("After truncation: $(rank(kernel_mps))")

    return kernel_mps
end

"""
try patching compression of Adisc_anoβ
"""
function compress_Adisc_ano_patched(Gp::TCI4Keldysh.PartialCorrelator_reg{3}; rtol=1e-5, maxbonddim=50)
    bos_idx = get_bosonic_idx(Gp)
    R = maximum(grid_R.([size(leg, 1) for leg in Gp.tucker.legs]))
    Adisc_ano = dropdims(Gp.Adisc_anoβ; dims=(bos_idx,)) # D-1 dimensional

    # patched
    maxsample = maximum(abs.(Adisc_ano))
    patch_tol = rtol*maxsample
    @TIME ptt_Adisc_ano = patchedTCI(zeropad_array(Adisc_ano, R); tolerance=patch_tol, maxbonddim=maxbonddim) "Compress anomalous part of PSF (patched)"
    proj_tt_info(ptt_Adisc_ano)

    # compare to normal tci
    @TIME qtt_Adisc_ano, _, _ = quanticscrossinterpolate(
        zeropad_array(Adisc_ano, R); tolerance=rtol
        ) "Compress anomalous part of PSF"
    @show rank(qtt_Adisc_ano)

    # compare errors to Adisc_ano
end

"""
See rank of compressed frequency kernels
"""
function frequency_kernel_ranks(;npt=2, R=12, perm_idx=1, beta=2000.0, tolerance=1e-12, fermikernel=true, singvalshift=false)
    # get PSF
    perms = [p for p in permutations(collect(1:npt))]
    perm = perms[perm_idx]
    path = npt<=3 ? "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/" : joinpath("data/SIAM_u=0.50/PSF_nz=2_conn_zavg/", "4pt")
    Ops = if npt==2
            ["F1", "F1dag"][perm]
        elseif npt==3
            ["F1", "F1dag", "Q34"][perm]
        else
            ["F1", "F1dag", "F3", "F3dag"][perm]
        end
    spin = 1

    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    ωconvMat_sum = cumsum(ωconvMat[perm[1:(npt-1)],:]; dims=1)
    psf = PartialCorrelator_reg(npt-1, 1/beta, R, path, Ops, ωconvMat_sum; flavor_idx=spin, nested_ωdisc=false)
    @show psf.isFermi

    if singvalshift
        TCI4Keldysh.shift_singular_values_to_center!(psf.tucker)
    end
    kernels = compress_frequencykernels_light(R, psf.tucker; tolerance=tolerance)
    printstyled("\n --- Frequency kernel ranks: $(rank.(kernels))  R=$R  β=$beta ---\n"; color=:green)
    if fermikernel
        return maximum([rank(kernels[i]) for i in 1:npt-1 if psf.isFermi[i]])
    else
        return rank(kernels[findfirst(.!psf.isFermi)])
    end
end

function compress_padded_array(A::Array{T,D}; tags=nothing, kwargs...)::MPS where {T,D} 
    tags = isnothing(tags) ? ntuple(i -> "ω$i", D) : tags
    R = grid_R(size(A))
    return compress_padded_array(A, R; tags, kwargs...)
end

"""
TODO: TEST
"""
function compress_padded_array(A::Array{T,D}, R::Int; tags=nothing, kwargs...)::MPS where {T,D} 
    tags = isnothing(tags) ? ntuple(i -> "ω$i", D) : tags
    asize = size(A)
    function evalA(i...)
        if all(i .<= asize)
            return A[i...]
        else
            return zero(T)
        end
    end
    qtt, _, _ = quanticscrossinterpolate(T, evalA, ntuple(i -> 2^R, D); kwargs...)
    mps = TCI4Keldysh.QTCItoMPS(qtt, tags)
    return mps
end


function mps_idx_info(mps::Union{MPS,MPO})
    println("\n-- MPS:")
    for i in 1:length(mps)-1
        println(siteinds(mps)[i])
        println("Linkdim $i:  $(dim(linkind(mps, i)))")
    end
    println(siteinds(mps)[end])
    println("----\n")
end

function worstcase_bonddim(dims::Vector{Int}, i::Int)
    @assert 1<= i <=length(dims)-1 "Invalid bond index $i"
    return min(prod(dims[1:i]; init=1), prod(dims[i+1:end]; init=1))
end

function worstcase_bonddim(dims::Vector{Int})
    return [worstcase_bonddim(dims, i) for i in 1:length(dims)-1]
end

"""
Check whether tucker center (PSF) contains any values larger than tolerance.
"""
function check_zero(Gp::TCI4Keldysh.PartialCorrelator_reg{D}, tolerance::Float64) :: Bool where {D}
    return all(abs.(Gp.tucker.center) .<= tolerance)
end