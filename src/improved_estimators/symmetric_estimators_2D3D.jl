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

    if formalism=="KF" && !isnothing(Σ_calcL)
        error("Still need to make KF part use two different self-energies")
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
                    K2a_data_tmp = _mult_Σ_KF(-K2a_data_tmp, Σs_R[il]; idim=3+il, is_incoming=is_incoming[il])
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