using LinearAlgebra

m = [1 0 0;
     1 1 1;
     1 0 1]

invm = inv(m)
trunc.(Int, invm)

m * inv([1 1 0; 1 1 0; ])


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




convert_to_affineTrafos(m)
convert_to_affineTrafos([1 0 0; 1 1 0; 0 0 1])


function freq_transform(mps::MPS; tags, ωconvMat::Matrix{Int}, isferm_ωnew::Vector{Int})

    ωconvMat_list = convert_to_affineTrafos(ωconvMat)
    for m in ωconvMat_list
        mps = affine_freq_transform(mps; tags, ωconvMat=m, isferm_ωnew)
        isferm_ωnew = mod.(m * isferm_ωnew, 2)
    end
    return mps
end


