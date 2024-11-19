# 2d
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
            K2a_tmp      = TCI4Keldysh.FullCorrelator_KF(PSFpath, [op_labels[1], letts[1]*op_labels[2], letts[2]*op_labels[3]]; T, flavor_idx, ωs_ext, ωconvMat, broadening_kwargs...);

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

    for letts in letter_combinations#[2:end]
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

        Γcore_data += Γcore_data_tmp
    end

    return Γcore_data
end



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

function compute_Γfull_symmetric_estimator(
    formalism ::String,
    PSFpath::String,
    ;
    T::Float64,
    flavor_idx::Int,
    omsig::Vector{Float64},
    ωs_ext  ::NTuple{3,Vector{Float64}},
    channel::String,
    broadening_kwargs...
)

    ωconvMat4pt = channel_trafo(channel)

    # self-energies
    (ΣL, ΣR) = if formalism=="MF"
            calc_Σ_MF_aIE(PSFpath, omsig; flavor_idx=flavor, T=T)
        else
            calc_Σ_KF_aIE(PSFpath, omsig; flavor_idx=flavor, T=T, broadening_kwargs...)
        end

    # Γcore
    Γcore = compute_Γcore_symmetric_estimator(
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

    channels = ["a","t","p"]
    if !(channel in channels)
        error("Channel $channels not supported")
    end

    # add K2r, K2'r, r=a,t,p
    K2s = []
    for ch in channels
        # TODO compute ωs_ext_K2
        changeMat = channel_change(ch, channel)[1:2,:]
        # does NOT work for non-square matrices
        ωs_extK2, ωconvOff, _ = _trafo_ω_args(ωs_ext, changeMat)
        K2 = precompute_K2r(PSFpath, flavor_idx, formalism; ωs_ext=ωs_extK2, channel=ch, prime=false, broadening_kwargs...)
        push!(K2s, K2)
    end

    K2primes = []
    for ch in channels
        # TODO compute ωs_ext_K2prime
        changeMat = channel_change(ch, channel)[[1,3],:]
        K2prime = precompute_K2r(PSFpath, flavor_idx, formalism; ωs_ext=ωs_extK2prime, channel=ch, prime=true, broadening_kwargs...)
        push!(K2primes, K2prime)
    end

    # add K1r, r=a,t,p
    K1s = []
    for ch in channels
        # TODO compute ωs_ext_K1
        changeMat = reshape(channel_change(ch, channel)[1,:], 1,3)
        K1 = precompute_K1r(PSFpath, flavor_idx, formalism; ωs_ext=ωs_ext_K1, channel=ch, broadening_kwargs...)
        push!(K1s, K1)
    end

    # add bare vertex Γ_0
end