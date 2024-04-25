function compute_K2r_symmetric_estimator(
    PSFpath::String,
    op_labels::NTuple{3,String},
    Σ_calc_aIE  ::Vector{ComplexF64}
    ;
    flavor_idx::Int,
    ωs_ext  ::NTuple{2,Vector{Float64}},
    ωconvMat::Matrix{Int}
    )

    letters = ["F", "Q"]
    letter_combinations = kron(letters, letters)


    #K2a      = TCI4Keldysh.FullCorrelator_MF(PSFpath, [op_labels[1], "F"*op_labels[2], "F"*op_labels[3]]; flavor_idx, ωs_ext, ωconvMat);
    K2a_data = zeros(ComplexF64, length.(ωs_ext))
    #println("max dat 0: ", maxabs(K2a_data))

    # precompute Σs:
    is_incoming = 1 .< length.(op_labels[2:3])
    #println("is_incoming: ", is_incoming)
    Σs = precompute_Σs(Σ_calc_aIE, length.(ωs_ext), ωconvMat[2:3,:], is_incoming)

    #size(Σs[1])
    #size(Σs[2])

    for letts in letter_combinations#[2:end]

        K2a_tmp      = TCI4Keldysh.FullCorrelator_MF(PSFpath, [op_labels[1], letts[1]*op_labels[2], letts[2]*op_labels[3]]; flavor_idx, ωs_ext, ωconvMat);
        K2a_data_tmp = TCI4Keldysh.precompute_all_values(K2a_tmp)
        #println("max dat 1 ", letts," : ", maxabs(K2a_data_tmp))
        # multiply Σ if necessary:
        for il in eachindex(letts)
            if letts[il] == 'F'
                K2a_data_tmp = -K2a_data_tmp .* Σs[il]
            end
        end

        #println("max dat 2 ", letts," : ", maxabs(K2a_data_tmp))

        K2a_data += K2a_data_tmp
    end

    return K2a_data
end

function compute_Γcore_symmetric_estimator(
    PSFpath::String,
    Σ_calc_aIE  ::Vector{ComplexF64}
    ;
    flavor_idx::Int,
    ωs_ext  ::NTuple{3,Vector{Float64}},
    ωconvMat::Matrix{Int}
    )

    letters = ["F", "Q"]
    letter_combinations = kron(kron(letters, letters), kron(letters, letters))
    op_labels = ("1", "1dag", "3", "3dag")

    Γcore_data = zeros(ComplexF64, length.(ωs_ext))
    #println("max dat 0: ", maxabs(Γcore))

    # precompute Σs:
    is_incoming = 1 .< length.(op_labels)
    #println("is_incoming: ", is_incoming)
    Σs = precompute_Σs(Σ_calc_aIE, length.(ωs_ext), ωconvMat, is_incoming)

    #size(Σs[1])
    #size(Σs[2])

    filelist = readdir(PSFpath)

    for letts in letter_combinations#[2:end]
        println("letts: ", letts)
        ops = [letts[i]*op_labels[i] for i in 1:4]
        if !any(parse_Ops_to_filename(ops) .== filelist)
            op_labels_symm = ("3", "3dag", "1", "1dag")   # if discrete PSFs where not computed explicitly, exchange labels 1 <-> 3 (the underlying operators in QSpace are identical)
            ops = [letts[i]*op_labels_symm[i] for i in 1:4]
        end

        Γcore_tmp      = TCI4Keldysh.FullCorrelator_MF(PSFpath, ops; flavor_idx, ωs_ext, ωconvMat);
        Γcore_data_tmp = TCI4Keldysh.precompute_all_values(Γcore_tmp)
        println("max dat 1 ", letts," : ", maxabs(Γcore_data_tmp))
        # multiply Σ if necessary:
        for il in eachindex(letts)
            if letts[il] == 'F'
                Γcore_data_tmp = -Γcore_data_tmp .* Σs[il]
            end
        end

        println("max dat 2 ", letts," : ", maxabs(Γcore_data_tmp))

        Γcore_data += Γcore_data_tmp
    end

    return Γcore_data
end