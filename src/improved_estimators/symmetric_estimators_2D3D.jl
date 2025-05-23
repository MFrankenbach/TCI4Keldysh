#=
Methods for the computing MF/KF four-point vertex and K2 contributions.
For the method, see "Symmetric improved estimators for multipoint vertex functions", Lihm et.al. (2024).
=#

# whether to use fluctuation-dissipation theorem in asymmetric estimators of self-energy (relevant for core vertex, K2)
USE_FDR_SE() = false
# 2d
"""
Symmetric estimator for K2 class.
"""
function compute_K2r_symmetric_estimator(
    formalism ::String,
    PSFpath::String,
    op_labels::NTuple{3,String},
    Σ_calcR  ::Array{ComplexF64,N}
    ;
    Σ_calcL  ::Union{Nothing, Array{ComplexF64,N}}=nothing,
    T :: Float64,
    flavor_idx::Int,
    ωs_ext  ::NTuple{2,Vector{Float64}},
    ωconvMat::Matrix{Int},
    γ=nothing, # need to assign for Matsubara
    broadening_kwargs...
    ) where{N}
    if formalism == "MF"
        @assert N == 1
    else
        @assert N == 3
    end

    letters = ["F", "Q"]
    letter_combinations = kron(letters, letters)


    #K2a      = TCI4Keldysh.FullCorrelator_MF(PSFpath, [op_labels[1], "F"*op_labels[2], "F"*op_labels[3]]; flavor_idx, ωs_ext, ωconvMat);
    K2a_data = zeros(ComplexF64, length.(ωs_ext)..., (formalism == "MF" ? [] : 2 .*ones(Int, 3))...)
    #println("max dat 0: ", maxabs(K2a_data))

    # precompute Σs:
    is_incoming = 1 .< length.(op_labels[2:3])
    #println("is_incoming: ", is_incoming)
    Σs_R = precompute_Σs(Σ_calcR, length.(ωs_ext), ωconvMat[2:3,:], is_incoming)

    Σs_L = if !isnothing(Σ_calcL)
        precompute_Σs(Σ_calcL, length.(ωs_ext), ωconvMat[2:3,:], is_incoming)
    else
        Σs_R
    end

    for letts in letter_combinations#[2:end]

        if formalism == "MF"
            K2a_tmp      = TCI4Keldysh.FullCorrelator_MF(PSFpath, [op_labels[1], letts[1]*op_labels[2], letts[2]*op_labels[3]]; T, flavor_idx, ωs_ext, ωconvMat);
            K2a_data_tmp = TCI4Keldysh.precompute_all_values(K2a_tmp)

            for il in eachindex(letts)
                if letts[il] == 'F'
                    if is_incoming[il]
                        K2a_data_tmp = -K2a_data_tmp .* Σs_R[il]
                    else
                        K2a_data_tmp = -K2a_data_tmp .* Σs_L[il]
                    end
                end
            end
        else
            K2a_tmp      = TCI4Keldysh.FullCorrelator_KF(
                            PSFpath, [op_labels[1], letts[1]*op_labels[2], letts[2]*op_labels[3]];
                            T, flavor_idx, ωs_ext, ωconvMat, γ=[2γ,γ,γ], broadening_kwargs...
                            );

            K2a_data_tmp = TCI4Keldysh.precompute_all_values(K2a_tmp)

            for il in eachindex(letts)
                if letts[il] == 'F'
                    #K2a_data_tmp[:,:,] = -K2a_data_tmp .* Σs[il]
                    if is_incoming[il]
                        K2a_data_tmp = _mult_Σ_KF(-K2a_data_tmp, Σs_R[il]; idim=3+il, is_incoming=is_incoming[il])
                    else
                        K2a_data_tmp = _mult_Σ_KF(-K2a_data_tmp, Σs_L[il]; idim=3+il, is_incoming=is_incoming[il])
                    end
                else
                    reverse!(K2a_data_tmp, dims=3+il)
                end
            end
        end
        #println("max dat 1 ", letts," : ", maxabs(K2a_data_tmp))
        # multiply Σ if necessary:
        

        #println("max dat 2 ", letts," : ", maxabs(K2a_data_tmp))
        #println("size of K2a_data: ", size(K2a_data), "\t size of K2a_data_tmp: ", size(K2a_data_tmp))
        K2a_data += K2a_data_tmp
    end

    return K2a_data
end

# 3d
"""
cf. eq. (132) Lihm et. al.

* Σ_calcL: If provided, Σ_calcR will be used for incoming legs and Σ_calcL for outgoing legs.
They should then be the right (incoming) resp. left-sided(outgoing) asymmetric estimators for the self-energies.
"""
function compute_Γcore_symmetric_estimator(
    formalism ::String,
    PSFpath::String,
    Σ_calcR  ::Array{ComplexF64,N}
    ;
    Σ_calcL :: Union{Nothing, Array{ComplexF64,N}}=nothing,
    T::Float64,
    flavor_idx::Int,
    ωs_ext  ::NTuple{3,Vector{Float64}},
    ωconvMat::Matrix{Int},
    broadening_kwargs...
    ) where {N}
    if formalism == "MF"
        @assert N == 1
    else
        @assert N == 3
    end

    letters = ["F", "Q"]
    letter_combinations = kron(kron(letters, letters), kron(letters, letters))
    op_labels = ("1", "1dag", "3", "3dag")

    Γcore_data = zeros(ComplexF64, length.(ωs_ext)..., (formalism == "MF" ? [] : 2 .*ones(Int, 4))...)

    #println("max dat 0: ", maxabs(Γcore))

    # precompute Σs:
    is_incoming = 1 .< length.(op_labels)
    #println("is_incoming: ", is_incoming)
    Σs_R = precompute_Σs(Σ_calcR, length.(ωs_ext), ωconvMat, is_incoming)

    Σs_L = if !isnothing(Σ_calcL)
        precompute_Σs(Σ_calcL, length.(ωs_ext), ωconvMat, is_incoming)
    else
        Σs_R
    end

    filelist = readdir(PSFpath)

    # avoid race condition
    gamcore_lock = ReentrantLock()

    Threads.@threads for letts in letter_combinations#[2:end]
        println("letts: ", letts)
        ops = [letts[i]*op_labels[i] for i in 1:4]
        if !any(parse_Ops_to_filename(ops) .== filelist)
            op_labels_symm = ("3", "3dag", "1", "1dag")   # if discrete PSFs where not computed explicitly, exchange labels 1 <-> 3 (the underlying operators in QSpace are identical)
            ops = [letts[i]*op_labels_symm[i] for i in 1:4]
        end

        if formalism == "MF"
            Γcore_tmp      = TCI4Keldysh.FullCorrelator_MF(PSFpath, ops; T, flavor_idx, ωs_ext, ωconvMat);
            Γcore_data_tmp = TCI4Keldysh.precompute_all_values(Γcore_tmp)
            println("max dat 1 ", letts," : ", maxabs(Γcore_data_tmp))
            # multiply Σ if necessary:
            for il in eachindex(letts)
                if letts[il] == 'F'
                    if is_incoming[il]
                        Γcore_data_tmp = -Γcore_data_tmp .* Σs_R[il]
                    else
                        Γcore_data_tmp = -Γcore_data_tmp .* Σs_L[il]
                    end
                end
            end

        else
            Γcore_tmp      = TCI4Keldysh.FullCorrelator_KF(PSFpath, ops; T, flavor_idx, ωs_ext, ωconvMat, broadening_kwargs...);
            Γcore_data_tmp = TCI4Keldysh.precompute_all_values(Γcore_tmp)

            for il in eachindex(letts)
                if letts[il] == 'F'
                    #K2a_data_tmp[:,:,] = -K2a_data_tmp .* Σs[il]
                    if is_incoming[il]
                        Γcore_data_tmp = _mult_Σ_KF(-Γcore_data_tmp, Σs_R[il]; idim=3+il, is_incoming=is_incoming[il])
                    else
                        Γcore_data_tmp = _mult_Σ_KF(-Γcore_data_tmp, Σs_L[il]; idim=3+il, is_incoming=is_incoming[il])
                    end
                else
                    reverse!(Γcore_data_tmp, dims=3+il)
                end
            end

        end
        println("max dat 2 ", letts," : ", maxabs(Γcore_data_tmp))

        lock(gamcore_lock) do
            Γcore_data += Γcore_data_tmp
        end
    end

    return Γcore_data
end

function compute_Γcore_symmetric_estimator(
    formalism::String,
    PSFpath::String,
    ωs_ext::NTuple{3,Vector{Float64}};
    flavor_idx::Int,
    channel::String,
    broadening_kwargs...
)
    PSFpath_Σ = PSFpath
    if !contains(PSFpath, "4pt")
        PSFpath = joinpath(PSFpath, "4pt")
    end
    ωconvMat = channel_trafo(channel)
    T = dir_to_T(PSFpath)

    omsig = Σ_grid(ωs_ext[1:2])
    (ΣL,ΣR) = if formalism=="MF"
            calc_Σ_MF_aIE(PSFpath_Σ, omsig; flavor_idx=flavor_idx, T=T)
        else
            calc_Σ_KF_aIE_viaR(PSFpath_Σ, omsig; flavor_idx=flavor_idx, T=T, broadening_kwargs...)
        end

    return compute_Γcore_symmetric_estimator(
        formalism,
        PSFpath,
        ΣR;
        Σ_calcL=ΣL,
        flavor_idx=flavor_idx,
        T=T,
        ωs_ext=ωs_ext,
        ωconvMat=ωconvMat,
        broadening_kwargs...
    )
end


"""
Contract self-energy with precomputed correlator data.
"""
function _mult_Σ_KF(G_data::Array{ComplexF64,N}, Σ::Array{ComplexF64,NΣ}; idim::Int, is_incoming::Bool) where{N,NΣ}
    G_out = zeros(ComplexF64, size(G_data))
    Ndims_freqs = ndims(Σ) - 2

    for i in 1:2, j in 1:2
        if is_incoming
            G_out[[Colon() for _ in 1:(idim-1)]..., j, [Colon() for _ in (idim+1):N]...] += G_data[[Colon() for _ in 1:(idim-1)]..., i, [Colon() for _ in (idim+1):N]...] .* Σ[[Colon() for _ in 1:Ndims_freqs]..., i, j]
        else
            G_out[[Colon() for _ in 1:(idim-1)]..., j, [Colon() for _ in (idim+1):N]...] += G_data[[Colon() for _ in 1:(idim-1)]..., i, [Colon() for _ in (idim+1):N]...] .* Σ[[Colon() for _ in 1:Ndims_freqs]..., j, i]
        end
    end
    return G_out
end

"""
Bare Keldysh vertex
"""
function Γbare_KF(PSFpath::String, flavor_idx::Int)
    if flavor_idx==2
        gam0 = load_Adisc_0pt(PSFpath, "Q12")
        ret = zeros(ComplexF64, 2,2,2,2)
        for iK in ids_KF(4)
            if isodd(sum(iK))
                ret[iK...] = gam0
            end
        end
        return ret
    else
        return zeros(ComplexF64, 2,2,2,2)
    end
end

"""
Compute full vertex in given formalism. Can store individual contributions (K1,K2 in different channels and core)
by providing `store_dir`.
"""
function compute_Γfull_symmetric_estimator(
    formalism ::String,
    PSFpath::String,
    ;
    T::Float64,
    flavor_idx::Int,
    ωs_ext  ::NTuple{3,Vector{Float64}},
    channel::String,
    store_dir=nothing,
    useFDR::Bool=USE_FDR_SE(),
    broadening_kwargs...
)

    ωconvMat4pt = channel_trafo(channel)

    # self-energies
    omsig = only(trafo_grids(ωs_ext[1:2], reshape([1,1],(1,2))))
    (ΣL, ΣR) = if formalism=="MF"
            calc_Σ_MF_aIE(PSFpath, omsig; flavor_idx=flavor_idx, T=T)
        else
            if useFDR
                calc_Σ_KF_aIE_viaR(PSFpath, omsig; flavor_idx=flavor_idx, T=T, broadening_kwargs...)
            else
                calc_Σ_KF_aIE(PSFpath, omsig; flavor_idx=flavor_idx, T=T, broadening_kwargs...)
            end
        end

    # Γcore
    Γfull = compute_Γcore_symmetric_estimator(
        formalism,
        PSFpath * "/4pt",
        ΣR;
        Σ_calcL=ΣL,
        ωs_ext=ωs_ext,
        T=T,
        flavor_idx=flavor_idx,
        ωconvMat=ωconvMat4pt,
        broadening_kwargs...
    )

    if !isnothing(store_dir)
        h5write(joinpath(store_dir, "V_KF_U4.h5"), "core", Γfull)
    end

    channels = ["a","t","pNRG"]
    if !(channel in channels)
        error("Channel $channel not supported")
    end

    @show maximum(abs.(Γfull))

    # add K2r, K2'r, r=a,t,p
    for ch in channels
        for prime in [true, false]
            fidx = prime ? 3 : 2
            nonfidx = prime ? 2 : 3
            if ch==channel # compute K2, no frequency trafo needed
                ωs_extK2 = (ωs_ext[1],ωs_ext[fidx])
                K2 = precompute_K2r(PSFpath,
                    flavor_idx,
                    formalism; 
                    ωs_ext=ωs_extK2,
                    channel=ch,
                    prime=prime,
                    broadening_kwargs...)
                # add K2
                if formalism=="MF"
                    for i in axes(Γfull,nonfidx)
                        slice = ntuple(j -> j==nonfidx ? i : Colon(), 3)
                        Γfull[slice...] .+= K2
                    end
                else
                    K2 /= sqrt(2)
                    before = copy(Γfull)
                    for ik2 in ids_KF(3)
                        for iK in equivalent_iK_K2(ik2, ch, prime)
                            for i in axes(Γfull,nonfidx)
                                slice = ntuple(j -> j==nonfidx ? i : Colon(), 3)
                                Γfull[slice..., iK...] .+= K2[:,:,ik2...]
                            end
                        end
                    end
                    if !isnothing(store_dir)
                        outdata = Γfull .- before
                        h5write(joinpath(store_dir, "V_KF_U3_$(channel_translate(ch))_$(prime).h5"), "K2", collect(outdata))
                    end
                end
            else
                # compute K2
                changeMat = channel_change(channel, ch)[[1,fidx],:]
                ωs_extK2, offset = trafo_grids_offset(ωs_ext, changeMat)
                K2 = precompute_K2r(PSFpath,
                    flavor_idx,
                    formalism;
                    ωs_ext=ωs_extK2,
                    channel=ch,
                    prime=prime,
                    broadening_kwargs...)
                # add K2
                K2strides = transpose(collect(strides(K2))[1:2])
                if formalism=="MF"
                    off_stride = K2strides*offset + sum(K2strides*changeMat) - sum(K2strides)
                    sv = StridedView(K2, size(Γfull), Tuple(K2strides * changeMat), off_stride)
                    # testsv = collect(sv)
                    Γfull .+= collect(sv)
                else
                    K2 /= sqrt(2)
                    # take care of Keldysh components
                    before = copy(Γfull)
                    for ik in Iterators.product(fill([1,2], 3)...)
                        off_stride = K2strides*offset + sum(K2strides*changeMat) - sum(K2strides)
                        sv = StridedView(K2[:,:,ik...], size(Γfull)[1:3], Tuple(K2strides * changeMat), off_stride)
                        for iK in equivalent_iK_K2(ik, ch, prime)
                            Γfull[:,:,:,iK...] .+= collect(sv)
                        end
                    end
                    if !isnothing(store_dir)
                        outdata = Γfull .- before
                        h5write(joinpath(store_dir, "V_KF_U3_$(channel_translate(ch))_$(prime).h5"), "K2", collect(outdata))
                    end
                end
            end
        end
    end

    @show maximum(abs.(Γfull))

    # add K1r, r=a,t,p
    for ch in channels
        if ch==channel
            # no trafo needed
            K1 = precompute_K1r(PSFpath, flavor_idx, formalism; ωs_ext=ωs_ext[1], channel=ch, broadening_kwargs...)
            if formalism=="MF"
                Γfull .+= reshape(K1, (length(K1),1,1))
            else
                K1 *= 0.5 
                before = copy(Γfull)
                for ik1 in ids_KF(2)
                    for iK in equivalent_iK_K1(ik1, ch)
                        Γfull[:,:,:,iK...] .+= reshape(K1[:,ik1...], (size(K1,1),1,1)) 
                    end
                end
                if !isnothing(store_dir)
                    outdata = Γfull .- before
                    h5write(joinpath(store_dir, "V_KF_U2_$(channel_translate(ch)).h5"), "K1", collect(outdata))
                end
            end
        else
            changeMat = reshape(channel_change(channel, ch)[1,:], 1,3)
            ωs_extK1, offset = trafo_grids_offset(ωs_ext, changeMat)
            K1 = precompute_K1r(PSFpath, flavor_idx, formalism; ωs_ext=only(ωs_extK1), channel=ch, broadening_kwargs...)
            printstyled("==== NORM K1 (channel=$(ch)): $(norm(K1))\n"; color=:magenta)
            if formalism=="MF"
                off_stride = only(offset) + sum(changeMat) - 1
                sv = StridedView(K1, size(Γfull), Tuple([1] * changeMat), off_stride)
                Γfull .+= collect(sv)
            else
                K1 *= 0.5
                before = copy(Γfull)
                printstyled("==== NORM BEFORE: $(norm(before))\n"; color=:magenta)
                for ik1 in ids_KF(2)
                    for iK in equivalent_iK_K1(ik1, ch)
                        off_stride = only(offset) + sum(changeMat) - 1
                        sv = StridedView(K1[:,ik1...], size(Γfull)[1:3], Tuple([1] * changeMat), off_stride)
                        printstyled("  == NORM SV: $(norm(collect(sv)))\n"; color=:magenta)
                        Γfull[:,:,:,iK...] .+= collect(sv)
                    end
                end
                printstyled("==== NORM AFTER: $(norm(Γfull))\n"; color=:magenta)
                if !isnothing(store_dir)
                    outdata = Γfull .- before
                    h5write(joinpath(store_dir, "V_KF_U2_$(channel_translate(ch)).h5"), "K1", collect(outdata))
                end
            end
        end
    end

    @show maximum(abs.(Γfull))

    # add bare vertex Γ_0 for updown flavor
    gam0 = 2.0 * load_Adisc_0pt(PSFpath, "Q12")
    if flavor_idx==2
        if formalism=="MF"
            @show gam0
            Γfull .+= gam0
        else
            for iK in ids_KF(4)
                if isodd(sum(iK))
                    Γfull[:,:,:,iK...] .+= 0.5 * gam0
                end
            end
        end
    end

    if !isnothing(store_dir)
        h5write(joinpath(store_dir, "V_KF_symm.h5"), "V_KF", Γfull)
    end
    
    return Γfull
end

"""
NOT YET IMPLEMENTED
"""
function compute_Γcore_pointwise(
    formalism ::String,
    PSFpath::String,
    R::Int=2^12
    ;
    T::Float64=dir_to_T(PSFpath),
    flavor_idx::Int,
    ωs_nonlin::Vector{Float64},
    channel::String,
    broadening_kwargs...
)
    if formalism=="MF"    
        error("NYI")
    else
        # formalism=="KF"
        # determine linear grid
        ommax = maximum(abs.(ωs_nonlin))
        ωs_ext = KF_grid(ommax, R, 3)
        # TODO need to evaluate vertex on arbitrary frequencies
        # by interpolation
        error("NYI")
    end
end