using JSON

"""
Counts number of evaluations of a function.
Can also count number of evaluations where arguments satisfy a given condition
"""
mutable struct EvaluationCounter
    f # something callable
    evalcount::Int
    specialcond::Vector{Function} # conditions on function input when to increase specialcount
    specialcount::Vector{Int}
    name::String

    function EvaluationCounter(f, specialcond::Vector{Function}, name="")
        return new(f, 0, specialcond, fill(0, length(specialcond)), name)
    end
end

function EvaluationCounter(f, specialcond::Function, name="")
    return EvaluationCounter(f, [specialcond], name)
end

function (ec::EvaluationCounter)(x)
    ec.evalcount += 1
    for i in eachindex(ec.specialcond)
        if ec.specialcond[i](x)
            ec.specialcount[i] += 1
        end
    end
    return ec.f(x)
end

function (ec::EvaluationCounter)(x::Vararg{Any, D}) where {D}
    ec.evalcount += 1
    for i in eachindex(ec.specialcond)
        if ec.specialcond[i](x)
            ec.specialcount[i] += 1
        end
    end
    return ec.f(x...)
end


function reportcount(ec::EvaluationCounter; log::Union{String,Nothing}=nothing)
    printstyled("  No. of evaluations: $(ec.evalcount)\n"; color=:cyan)
    # print out
    for i in eachindex(ec.specialcond)
        fname = nameof(ec.specialcond[i])
        printstyled("  No. of special evaluations: $(ec.specialcount[i]) (function: $(String(fname)))\n"; color=:cyan)
        printstyled("  Ratio: $(ec.specialcount[i] / ec.evalcount)\n"; color=:cyan)
        println("\n")
    end
    # log to file
    if !isnothing(log)
        open(log, "a") do f
            write(f, "==== $(ec.name)\n")
            write(f, "  No. of evaluations: $(ec.evalcount)\n")
            for i in eachindex(ec.specialcond)
                fname = nameof(ec.specialcond[i])
                write(f, "  No. of special evaluations: $(ec.specialcount[i]) (function: $(String(fname)))\n")
                write(f, "  Ratio: $(ec.specialcount[i] / ec.evalcount)\n")
                write(f, "\n")
            end
        end
    end
end

function allconcrete(x::T) where T 
    return all(isconcretetype, fieldtypes(T))
end

function maxabs(arr::AbstractArray)
    return maximum(abs.(arr))
end

function quadtrapz(xs::Vector, ys::Vector)
    return sum((ys[2:end]+ys[1:end-1]).*(xs[2:end]-xs[1:end-1]))/2
end

"""
Insert vals after elements with indices ids into vec
"""
function insert_vec!(vec::Vector{T}, ids::Vector{Int}, vals::Vector{T}) where {T}
    @assert length(ids)==length(vals)
    ishift = 1
    for i in eachindex(ids)
        insert!(vec, ids[i]+1, vals[i])
        ishift += 1
    end
end

function interpolate_simplex_coeffs(simplex::NTuple{D1, SVector{D,Float64}}, point::SVector{D,Float64}) where {D1,D}
    y = SA[point..., 1.0]
    M = hcat([SA[s..., 1.0] for s in simplex]...)
    return pinv(M)*y
    # return dot(a, SA[vals...])
end

function interpolate_simplex(simplex::NTuple{D1, SVector{D,Float64}}, vals::SVector{D1,T}, point::SVector{D,Float64}) where {D1,D,T}
    a = interpolate_simplex_coeffs(simplex, point)
    return dot(a, vals)
end

function test_interpolate_simplex()
    simplex = (SA[2.0,0.0], SA[0.0,2.0], SA[0.0,0.0])
    point = SA[1.0,1.0]
    vals = SA[-1.0, 1.5, 1.0]
    res = 0.25
    res_test = interpolate_simplex(simplex, vals, point)
    @assert isapprox(res, res_test, atol=1.e-10)
end


"""Currently just test whether it runs"""
function test_interpolate_trilinear()
    block = reshape(1.1im .* collect(1:(5*2*2*5)), (5,2,2,5))
    interpol_dims = [1,4]
    b1 = 0.3 .* collect(-2:2)
    i1 = 0.2 .* collect(-2:2)
    block_grid = (b1,b1)
    interpol_grid = (i1,i1)

    _ = interpolate_trilinear(block, block_grid, interpol_grid, interpol_dims)
end

function interpolate_trilinear(
    block::Array{T,D2},
    block_grid::NTuple{D,Vector{Float64}},
    interpol_grid::NTuple{D,Vector{Float64}},
    interpol_dims::Vector{Int}
    ) where {T,D2,D}

    @assert length(interpol_dims)==D
    outsize = collect(size(block))    
    gridsize = length.(interpol_grid)
    outsize[interpol_dims] .= gridsize
    outdata = zeros(T, Tuple(outsize)...)
    where_interpol = fill(0, D2)
    where_interpol[interpol_dims] .= collect(1:length(interpol_dims))
    val_sz = size(block)[where_interpol.==0]
    @assert all(val_sz .== 2) "Only possible for 2x2x...x2 blocks atm"
    val_zero = zeros(T, val_sz...)

    # iterate over interpolation grid points
    function _f(w::Vararg{Int,D})
        idx = ntuple(i -> where_interpol[i]==0 ? Colon() : w[where_interpol[i]] , D2)
        return block[idx...]
    end
    for ic in Iterators.product(Base.OneTo.(gridsize)...)
        point = ntuple(i -> interpol_grid[i][ic[i]], D)
        out_idx = ntuple(i -> where_interpol[i]==0 ? Colon() : ic[where_interpol[i]], D2)
        outdata[out_idx...] .= eval_interpol(val_zero, _f, block_grid, point...)
    end
    return outdata
end

function interpolate_trilinear(vals::Vector{T}, corners::Vector{SVector{D,Float64}}, point::SVector{D,Float64}) where {T,D}
    return interpolate_trilinear(reshape(vals, ntuple(_->2,D)), reshape(corners, ntuple(_->2,D)), point)
end

function zero2n(::Type{Array{T,D}}) where {T<:Number, D}
    return zeros(T,ntuple(_->2,D)...)
end

function zero2n(::Type{T}) where {T<:Number}
    zero(T)    
end

"""
point: coordinates of the point when the cuboid has been shifted to the origin.
T::Array of number
"""
function interpolate_trilinear(vals::Array{T,D}, corners::Array{SVector{D,Float64},D}, point::SVector{D,Float64}) where {T,D}
    ret = zero2n(T)
    wt_tot = 0.0
    for ic in CartesianIndices(vals)
        opposed = ntuple(i -> ifelse(ic[i]==1, 2,1), D)
        wt = abs(prod(corners[opposed...] .- point))
        ret += vals[ic] * wt
        wt_tot += wt
    end
    return ret / wt_tot
end

"""
for a function `i::Int -> f(i)::Array{T,D}` with a given range of indices,
find interpolation points such that the f(i) is linearly interpolated by neighbouring
points to a given accuracy
* p: norm exponent for the error measure
"""
function lin_interp_array(f, idomain::Vector{Int}, ipos=1:length(idomain); p::Float64=Inf, abstol::Float64=1.e-8)
    (interp_ids, interp_val) = lin_interp_array_init(f, idomain, ipos; p=p, abstol=abstol)
    println("No. of points in initial guess: $(length(interp_ids))")
    return lin_interp_array!(interp_ids, interp_val, f, idomain, ipos; p=p, abstol=abstol)
end

"""
Check whether `interp_val` interpolates function `f` on domain `idomain`.
If not, insert all values needed to get that.
"""
function lin_interp_array!(interp_ids::Vector{Int}, interp_val, f, idomain::Vector{Int}, ipos=1:length(idomain); p::Float64=Inf, abstol::Float64=1.e-8)

    # make sure end points are in interpolation points
    npt = length(idomain)
    if !(1 in interp_ids)
        insert!(interp_ids, 1, 1)
        insert!(interp_val, 1, f(idomain[1]))
    end
    if !(npt in interp_ids)
        insert!(interp_ids, length(interp_ids)+1, npt)
        insert!(interp_val, length(interp_ids)+1, f(idomain[npt]))
    end

    to_insert = Int[]
    to_insert_mat = []
    for il in 1:length(interp_ids)-1
        for j in interp_ids[il]+1:interp_ids[il+1]-1
            mat = f(idomain[j])
            dd = ipos[interp_ids[il+1]] - ipos[interp_ids[il]] 
            a1 = (ipos[interp_ids[il+1]] - ipos[j]) / dd
            a0 = (ipos[j] - ipos[interp_ids[il]]) / dd
            interp_mat = a0*interp_val[il+1] + a1*interp_val[il]
            if norm(mat .- interp_mat, p) > abstol
                push!(to_insert, j)
                push!(to_insert_mat, mat)
            end
        end
    end

    interp_ids_ret = vcat(interp_ids, to_insert)
    interp_val_ret = vcat(interp_val, to_insert_mat)
    perm = sortperm(interp_ids_ret)
    permute!(interp_ids_ret, perm)
    permute!(interp_val_ret, perm)

    return (interp_ids_ret, interp_val_ret)
end

"""
for a function `i::Int -> f(i)::Array{T,D}` with a given range of indices,
find initial guess for interpolation points such that the f(i) is linearly interpolated by neighbouring
points to a given accuracy
* p: norm exponent for the error measure
"""
function lin_interp_array_init(f, idomain::Vector{Int}, ipos=1:length(idomain); p::Float64=Inf, abstol::Float64=1.e-8)
    npt = length(idomain)
    @assert length(ipos)==npt

    perm = sortperm(idomain)
    idomain = idomain[perm]
    ipos = ipos[perm]

    interp_ids = [1, npt]    
    interp_val = f.(idomain[interp_ids])
    while true
        ids = [div(interp_ids[i]+interp_ids[i+1],2) for i in 1:length(interp_ids)-1]
        new_ids = []
        new_ids_left = []
        new_ids_right = []
        new_ids_idx = []
        for i in eachindex(ids)
            if !(ids[i] in interp_ids)
                push!(new_ids_idx, i)
                push!(new_ids, ids[i])
                push!(new_ids_left, interp_ids[i])
                push!(new_ids_right, interp_ids[i+1])
            end
        end

        if isempty(new_ids_idx)
            break
        end

        # check interpolation on new points
        lenbefore = length(interp_ids)
        to_insert = Int[]
        to_insert_mat = []
        for i in eachindex(new_ids)
            mat = f(idomain[new_ids[i]])
            dd = ipos[new_ids_right[i]] - ipos[new_ids_left[i]] 
            a1 = (ipos[new_ids[i]] - ipos[new_ids_left[i]]) / dd
            a0 = (ipos[new_ids_right[i]] - ipos[new_ids[i]]) / dd
            interp_mat = a0 * interp_val[new_ids_idx[i]] + a1 * interp_val[new_ids_idx[i]+1]
            err = norm(mat .- interp_mat, p)
            if err > abstol
                push!(to_insert, i)
                push!(to_insert_mat, mat)
            # else
            #     printstyled("err: $(norm(mat .- interp_mat, p)), interp=$(a0)*$(interp_val[new_ids_idx[i]]) + $(a1)*$(interp_val[new_ids_idx[i]+1])\n"; color=:red)
            end
        end
        ishift = 1
        for i in eachindex(to_insert)
            insert!(interp_ids, new_ids_idx[to_insert[i]]+ishift, new_ids[to_insert[i]])
            insert!(interp_val, new_ids_idx[to_insert[i]]+ishift, to_insert_mat[i])
            ishift += 1
        end
        if lenbefore == length(interp_ids)
            break
        end
    end
    return (interp_ids, interp_val)
end

"""
Make the discrete spectral data ("Adisc") of a partial spectral function
compact, by removing the rows, columns, or slices of "Adisc" that contain
zeros only.
* keep_symmetry: whether to discard only ϵ's where -ϵ also corresponds to a slice of 0s;
can be set true in combination with reduce_Gps!
"""
function compactAdisc(
    ωdisc   ::Vector{Float64}, 
    Adisc   ::Array{Float64};
    keep_symmetry::Bool=false
    )
    # Make the discrete spectral data ("Adisc") of a partial spectral function
    # compact by removing the rows, columns, or slices of "Adisc" that contain
    # zeros only.

    # remove singleton dimensions
    Adisc = dropdims(Adisc,dims=tuple(findall(size(Adisc).==1)...))
    nO = ndims(Adisc)

    # Sanity check
    sz = size(Adisc)
    if !all(sz .== (length(ωdisc) * ones(Int, nO)))
        @show sz
        @show nO
        @show length(ωdisc)
        error("ERR: size of Adisc is inconsistent with the size of odisc.")
    end

    oks = [BitVector(undef, length(ωdisc)) for i in 1:nO]
    AdiscIsZero = if keep_symmetry
                    abs.(Adisc .+ reverse(Adisc)) .< (maxabs(Adisc) * 1.e-14)
                else
                    abs.(Adisc) .< (maxabs(Adisc) * 1.e-14)
                end
    for i in 1:nO
        # find slices that contain only zeros:
        all!(reshape(oks[i], (ones(Int, i-1)..., length(ωdisc))), AdiscIsZero)
    end
    Adisc = Adisc[[.!oks[i] for i in 1:nO]...]
    ωdiscs = [deepcopy(ωdisc[.!oks[i]]) for i in 1:nO]

    return oks, ωdiscs, Adisc
end

function get_ω_binwidths(ωs::Vector{Float64}) ::Vector{Float64}
    Δωs = [ωs[2] - ωs[1]; (ωs[3:end] - ωs[1:end-2]) / 2; ωs[end] - ωs[end-1]]  # width of the frequency bin
    return Δωs 
end



"""
contract_1D_Kernels_w_Adisc_mp(Kernels, Adisc)

Contracts kernels with Adisc

For 3p correlators we e.g. get K[i,a] K[j,b] Adisc[a,b]
"""
function contract_1D_Kernels_w_Adisc_mp(Kernels, Adisc)
    sz = [size(Adisc)...]
    D = ndims(Adisc)

    @DEBUG all(sz .== size.(Kernels, 2)) "Incompatible sizes for Adisc ($sz) and Kernels ($(size.(Kernels, 2)))"

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

    # @show size(Acont)
    # @show typeof(Acont)
    # @show size(Kernels)
    # @show size.(Kernels,1)
    # @show typeof(Kernels)
    # @assert sum(size(Acont) .- size.(Kernels, 1)) == 0

    return Acont
end

# """
# see `contract_KF_Kernels_w_Adisc_mp`, but here kernels are singular-value decomposed with a given cutoff.
# """
# function contract_KF_Kernels_w_Adisc_mp_svd(Kernels, Adisc; cutoff::Float64=1.e-15)
    
#     sz = [size(Adisc)...]
#     D = ndims(Adisc)
    
#     Acont = copy(Adisc)  # Initialize
#     for it1 in 1:D
#         Acont = reshape(Acont, (sz[it1], prod(sz) ÷ sz[it1] * it1))
#         Acont = Kernels[it1] * Acont
#         sz[it1] = size(Kernels[it1])[1]
#         Acont = reshape(Acont, ((sz[it1], sz[[it1+1:end; 1:it1-1]]..., it1)))
#         Acont = cat(Acont, conj.(Acont[[Colon() for _ in 1:D]...,1]), dims=D+1)

#         if D>1
#             Acont = permutedims(Acont, (collect(2:D)..., 1, D+1))
#         end
#     end

#     return Acont
# end


"""
    contract_KF_Kernels_w_Adisc_mp(Kernels, Adisc)

Contracts retarded kernels with Adisc and deduces all fully-retarded kernels.

For 3p correlators we e.g. get K^{R/A}[i,a] K^{R/A}[j,b] Adisc[a,b].\\

# Returns
Array with contracted data with all available external frequencies in the first D dimensions.
The trailing dimension D+1 is of size D and enumerates the fully-retarded kernels.

We need
for 2p:     K^[1](ω₁,ω₂)        =       K^R(ω₁)
            K^[2](ω₁,ω₂)        =       K^A(ω₁)                 = c.c.of first line
for 3p:     K^[1](ω₁,ω₂,ω₃)     =       K^R(ω₁)K^R(ω₂)         \\=2p result x K^R(ω₂)
            K^[2](ω₁,ω₂,ω₃)     =       K^A(ω₁)K^R(ω₂)         /
            K^[3](ω₁,ω₂,ω₃)     =       K^A(ω₁)K^A(ω₂)          = c.c.of first line
for 4p:     K^[1](ω₁,ω₂,ω₃,ω₄)  =       K^R(ω₁)K^R(ω₂)K^R(ω₃)  \\
            K^[2](ω₁,ω₂,ω₃,ω₄)  =       K^A(ω₁)K^R(ω₂)K^R(ω₃)  |=3p result x K^R(ω₃)
            K^[3](ω₁,ω₂,ω₃,ω₄)  =       K^A(ω₁)K^A(ω₂)K^R(ω₃)  /
            K^[4](ω₁,ω₂,ω₃,ω₄)  =       K^A(ω₁)K^A(ω₂)K^A(ω₃)   = c.c.of first line
"""
function contract_KF_Kernels_w_Adisc_mp(Kernels, Adisc; cutoff::Float64=-1.0)
    sz = [size(Adisc)...]
    D = ndims(Adisc)

    # leaves kernel sizes invariant -> no speedup; just for checking truncation errors
    if cutoff > 0.0
        Kernels = deepcopy(Kernels)
        for i in eachindex(Kernels)
            U, S, V = svd(Kernels[i])
            Snew_mask = S .>= cutoff
            knew = U[:, Snew_mask] * Diagonal(S[Snew_mask]) * adjoint(V[:, Snew_mask])
            Kernels[i] = knew
        end
    end
    
    ##########################################################
    ### EFFICIENCY IN TERMS OF   CPU TIME: 😄     RAM: 🙈  ###
    ##########################################################
    Acont = copy(Adisc)  # Initialize
    for it1 in 1:D
        #println(Dates.format(Dates.now(), "HH:MM:SS"), ":\tit1 = ", it1)
        Acont = reshape(Acont, (sz[it1], prod(sz) ÷ sz[it1] * it1))
        Acont = Kernels[it1] * Acont
        sz[it1] = size(Kernels[it1])[1]
        Acont = reshape(Acont, ((sz[it1], sz[[it1+1:end; 1:it1-1]]..., it1)))
        Acont = cat(Acont, conj.(Acont[[Colon() for _ in 1:D]...,1]), dims=D+1)

        #println(Dates.format(Dates.now(), "HH:MM:SS"), ":\tconvolution [done]")
        #Acont = reshape(Acont, (sz[it1], sz[[it1+1:end; 1:it1-1]]...))
        #println("size of Acont: ", size(Acont))
        if D>1
            Acont = permutedims(Acont, (collect(2:D)..., 1, D+1))
        end
        #println(Dates.format(Dates.now(), "HH:MM:SS"), ":\tpermutation [done]")
        #GC.gc()
    end

    return Acont
end


"""
contract_1D_Kernels_w_Adisc_mp(Kernels, Adisc)

Contract given kernels with Adisc. For testing purposes.

For 3p correlators we e.g. get K[i,a] K[j,b] Adisc[a,b]
"""
function contract_1D_Kernels_w_Adisc_mp_partial(Kernels, Adisc, n::Int)
    sz = [size(Adisc)...]
    D = ndims(Adisc)

    @DEBUG all(sz[1:n] .== size.(Kernels, 2)[1:n]) "Incompatible sizes for Adisc ($sz) and Kernels ($(size.(Kernels, 2)))"
    @assert n<=D && 1<=n

    ##########################################################
    ### EFFICIENCY IN TERMS OF   CPU TIME: �😄     RAM: 🙈  ###
    ##########################################################
    Acont = copy(Adisc)  # Initialize
    for it1 in 1:n
        Acont = reshape(Acont, (sz[it1], prod(sz) ÷ sz[it1]))
        Acont = Kernels[it1] * Acont

        sz[it1] = size(Kernels[it1])[1]
        Acont = reshape(Acont, (sz[it1], sz[[it1+1:end; 1:it1-1]]...))
        if D>1
            Acont = permutedims(Acont, (collect(2:D)..., 1))
        end
        #GC.gc()
    end

    # restore order of dimensions
    for _ in n+1:D
        Acont = permutedims(Acont, (collect(2:D)..., 1))
    end

    return Acont
end


"""
    hilbert(x)
    hilbert(x, n)
Analytic signal, computed using the Hilbert transform.
`y = hilbert(x)` gives the analytic signal `y = x + i*xi` where `xi` is the
Hilbert transform of vector `x`. 
If `x` is a matrix, then `hilbert` operates along columns.

**Example**
```julia
x = randn(10,10)
y = hilbert(x)
```

NOTE ON CONVERGENCE:
 - convergence: for slowly decaying Fourier transform of the signal ys ∝ 1/x one one is limited by the Nyquist theorem
                idea 1:     for long-term tails in Fourier transform: try out linear least squares fit to obtain coeffs for 1, 1/ω and 1/ω²
                            then insert/add the analytically known result

"""
function hilbert_fft(x::AbstractArray{Float64,d}; dims::Int=1) where {d}
    @warn "This function is wrong! Uncle Hilbert is not amused."
    if d > 2
        throw(ArgumentError("Currently only Vector and Matrix are supported for signal array ys."))
    end

    x_ = copy(x)

    # work along columns
    #size(x_,1)==1 ? x_ = permutedims(x_,[2 1]) : nothing

    #if(n>0 && n<size(x_,1))
    #    x_ = x_[1:n,:]
    #elseif(n>0 && n>size(x_,1))
    #    x_ = cat(1,x_,zeros(n-size(x_,1),size(x_,2)))
    #else
        n = size(x_,dims)
    #end

    xf = fft(x_,dims)
    h = reshape(zeros(Int64,n), (ones(Int, dims-1)..., n))      # represents the step function in time --> product in time corresponds to convolution with retarded kernel in frequency space
    if n>0 && n % 2 == 0
        #even, nonempty
        h[1:div(n,2)+1] .= 1
        h[2:div(n,2)] .= 2
    elseif n>0
        #odd, nonempty
        h[1] = 1
        h[2:div(n + 1,2)] .= 2
    end
    x_ = ifft(xf .* h, dims)

    # restore to original shape if necessary
    #size(x,1)==1 ? x_ = permutedims(x_,[2 1]) : nothing

    return x_

end

#=
# try fixing hilbert_fft by zeropadding; not tested
function hilbert_fft_fixed(x::AbstractArray{Float64,d}; dims::Int=1, zeropad=false) where {d}
    @warn "This function may well be wrong!"
    if d > 2
        throw(ArgumentError("Currently only Vector and Matrix are supported for signal array ys."))
    end

    x_ = copy(x)

    # zeropad x_ along dims
    slice = nothing
    if zeropad
        n_orig = size(x_, dims)
        x__ = zeros(Float64, ntuple(i -> i==dims ? 3*n_orig : size(x_, i), d))
        slice = if dims==1
            (n_orig+1 : 2*n_orig, Colon())
        elseif dims==2
            (Colon(), n_orig+1 : 2*n_orig)
        else
            error("dims=$dims not supported")
        end
        x__[slice...] .= x_
        x_ = x__
    end

    heatmap(abs.(x_))
    display(x_)
    savefig("x_")

    n = size(x_,dims)

    xf = fft(x_,dims)
    h = reshape(zeros(Int64,n), (ones(Int, dims-1)..., n))      # represents the step function in time --> product in time corresponds to convolution with retarded kernel in frequency space
    if n>0 && n % 2 == 0
        #even, nonempty
        h[1:div(n,2)+1] .= 1
        h[2:div(n,2)] .= 2
    elseif n>0
        #odd, nonempty
        h[1] = 1
        h[2:div(n + 1,2)] .= 2
    end
    x_ = ifft(xf .* h, dims)

    heatmap(abs.(xf))
    savefig("xf")

    heatmap(abs.(x_))
    savefig("xif")

    error("done")

    if zeropad
        return x_[slice...]
    else
        return x_
    end
end
=#




"""
my_hilbert_trafo

Computes the Hilbert transform of signal ys (sampled on a grid xs_in) for output frequencies xs_out.
We assume that the signal ys is piecewise linear.
If ys is multidimensional, the transform is computed along the direction dims=1.

The Hilbert transform is:
h(x_out) = 1/π P.V. ∫ dx y(x) / (x_out - x)
where P.V. stands for a principal value integral.

NOTE ON CONVENTIONS:    Analogously to the hilbert_fft(...) we return the original signal 'ys' 
                        with the Hilbert transform in the imaginary part.

NOTE ON PERFORMANCE / CONVERGENCE:
 - bottlenecks: construction of L (uncomment @time to see details)
 - convergence: for slowly decaying ys ∝ 1/x one needs a huge frequency range for xs_in
                analyze convergence in frequency range and mesh density with maxdev_in_hilbertTrafo_sinc(...) and maxdev_in_hilbertTrafo_rat(...)
                for high-frequency tails in spectrum: try out linear least squares fit to obtain coeffs for 1/ω and 1/ω²
"""
function my_hilbert_trafo(
    xs_out::Vector{Float64},    # real output frequencies xs_out[o]
    xs_in::Vector{Float64},     # real  input frequencies xs_in[i]
    ys::Array{Float64,d};       # piecewisely linear signal ys[i,j] where dim=1 is sampled on frequencies xs_in
    interpolation=:linear                           # and j represents the trailing dimensions in ys
                                # assume 1/x-decay outside of the box
    ) where {d}
    # As y(x) is piecewise linear we compute h(xs_out[o]) piecewisely, giving h(xs_out[o]) = ∑_i H[o,i,j] with
    #       H[o,i,j] = \int_{x[i]}^{x[i+1]} dx y⃗(x) / (xs_out[o] - x) 
    # On the i-th segment we have y⃗(x) = a[i,j] * (x - x[i]) + y[i,j]
    #       with a[i,j] = (y[i+1,j] - y[i,j]) / (x[i+1] - x[i])
    # such that the integral contribution from the i-th segment gives 
    # H[o,i,j] = - (x[i+1] - x[i]) * a[i,j] - ln( |x_out[o] - x[i+1]| / |x_out[o] - x[i]| ) * (y[i,j] + (xs_out[o] - x[i]) a[i,j])
    #          = - Δys[i,j]                .- ΔL[o,i] (y[i,j] + (xs_out[o] - x[i]) a[i,j])
    # with  Δxs_in[i] = xs_in[i+1] - xs_in[i]
    #       L[o,i]  = ln|xs_out[o] - xs_in[i]|
    #       ΔL[o,i] = L[o,i+1] - L[o,i]
    #
    # EDGE CASES:
    #   x_out == x[i]   -->     Then lim_{ϵ→0} ϵ ln(ϵ) = 0  and the divergent terms form segment i and i-1 cancel exactly!

    if d > 2
        throw(ArgumentError("Currently only Vector and Matrix are supported for signal array ys."))
    end

    #print("Time for construction of M: \t")
    #@time M = xs_out .- xs_in'
    M = xs_out .- xs_in'
    #print("Time for construction of L: \t")
    #@time L = log.(abs.(M))
    L = log.(abs.(M))
    ## clean L from terms where xs_out[i] == xs_in[j]
    is_divergentL = .!isfinite.(L)
    (@view L[is_divergentL]) .= 0.

    #print("Time for construction of ΔL: \t")
    #@time ΔL= diff(L, dims=2)                 
    ΔL= diff(L, dims=2)                 


    #print("Time for construction of Δy: \t")
    #@time Δys = diff(ys, dims=1)              
    #Δxs_in = diff(xs_in)                
    Δys = diff(ys, dims=1)              
    Δxs_in = diff(xs_in)                

    #print("Time for construction of a: \t")
    #@time a = Δys ./ Δxs_in
    a = Δys ./ Δxs_in

    #print("Time for construction of term1: \t")
    #@time term1 = - (ys[1,:] - ys[end,:])
    #print("Time for construction of term2: \t")
    #@time term2 = ΔL * ys[1:end-1,:]
    #print("Time for construction of term3: \t")
    #@time term3 = (ΔL .* (@view M[:,1:end-1])) * a
    term1 = - (ys[1,:] - ys[end,:])
    term2 = ΔL * ys[1:end-1,:]
    term3 = (ΔL .* (@view M[:,1:end-1])) * a

    #println("shapes of terms:\t", size(term1), "\t", size(term2), "\t", size(term3))

    result = term1' .- term2 .- term3
    result ./= π
    #result = - (ys[1,:] - ys[end,:]) .- ΔL * ys[1:end-1,:] - (ΔL .* (xs_out .- xs_in[1:end-1]')) * a

    # mid_id = div(length(xs_in),2) + 1
    # printstyled("\nCentral grid points: $(xs_in[mid_id-2:mid_id+2])\n"; color=:red)
    if d==1
        interp = if interpolation==:linear
            linear_interpolation(xs_in, ys)
        elseif interpolation==:spline
            CubicSpline(xs_in, ys)
        else
            error("Interpolation $(interoplation) invalid!")
        end
        ys_interp = interp.(xs_out)
        # result will be type matrix
        return ys_interp .+ im .* vec(result)
    elseif d==2
        # interpolate for each ϵ individually
        ys_interp = zeros(size(result))
        for i in axes(ys, 2)
            interp = if interpolation==:linear
                linear_interpolation(xs_in, ys[:,i])
            elseif interpolation==:spline
                CubicSpline(xs_in, ys[:,i])
            else
                error("Interpolation $(interoplation) invalid!")
            end
            ys_interp[:,i] .= interp.(xs_out)
        end

        # check out ys_interp vs ys
            # zero_id_in = findfirst(x -> abs(x)<=1.e-14, xs_in)
            # zero_id_out = findfirst(x -> abs(x)<=1.e-14, xs_out)
            # @show norm(ys[zero_id_in,:] .- ys_interp[zero_id_out,:])
        #--> at the problematic point they are the same
        # p = default_plot()
        # xop = xs_out .> 0.0
        # xip = xs_in .> 0.0
        # for i in 1:10:size(ys,2)
        #     plot!(p, xs_in[xip], ys[xip,i]; color=:blue, xscale=:log10)
        #     plot!(p, xs_out[xop], ys_interp[xop,i]; color=:red, linestyle=:dot, xscale=:log10)
        # end
        # savefig("ys_interp.pdf")
        # nonsense return FOR TESTING
        # return ys_interp .+ im * ones(size(result)) * 1.e-2

        return ys_interp .+ im .* result
    end
end

"""
Load 0-point PSF, i.e., a single value.
"""
function load_Adisc_0pt(path::String, Op::String) :: Float64
    f = matopen(joinpath(path, parse_Ops_to_filename([Op])), "r")
    try 
        keys(f)
    catch
        keys(f)
    end
    Adisc = only(read(f, "Adisc"))
    close(f)
    return only(Adisc)
end


"""
Read PSF.
Some 2-point PSFs only have one flavor.
"""
function load_Adisc(path::String, Ops::Vector{String}, flavor_idx::Int)
    fname = "PSF_(("*mapreduce(*,*,Ops, ["," for i in 1:length(Ops)])[1:end-1]*")).mat"
    f = matopen(joinpath(path, fname), "r")
    try 
        keys(f)
    catch
        keys(f)
    end
    Adisc_full = read(f, "Adisc")
    Adisc = if length(Ops)==2 && length(Adisc_full)==1
            only(Adisc_full) 
        elseif length(Adisc_full)>1
            Adisc_full[flavor_idx]
        else
            error("Flavor component may be missing in $(length(Ops))-point function")
        end
    Adisc = dropdims(Adisc,dims=tuple(findall(size(Adisc).==1)...))

    close(f)
    return Adisc 
end


function load_ωdisc(path::String, Ops::Vector{String}; nested_ωdisc::Bool=false)
    f = matopen(joinpath(path, "PSF_(("*mapreduce(*,*,Ops, ["," for i in 1:length(Ops)])[1:end-1]*")).mat"), "r")
    try 
        keys(f)
    catch
        keys(f)
    end
    ωdisc  = if !nested_ωdisc
        read(f, "odisc")[:]
    else
        read(f, "PSF")["odisc_info"]["odisc"][:]
    end
    close(f)
    return ωdisc 
end


function parse_filename_to_Ops(fn)
    return split(fn[7:end-6], ",")
end

function parse_Ops_to_filename(ops)
    temp = prod([o*"," for o in ops])
    return "PSF_(("*temp[1:end-1]*")).mat"
end

function isBosonic(Op::AbstractString)
    return Op[1]=='Q' && length(Op)==3
end

"""
Return symmetric 1D Matsubara grid.
"""
function MF_grid(T::Float64, Nhalf::Int, fermi::Bool)
    if fermi
        return π * T *(collect(-Nhalf:Nhalf-1) * 2 .+ 1)
    else
        return π * T * collect(-Nhalf:Nhalf) * 2
    end
end

"""
Bosonic grid ranges from ωmin to ωmax
"""
function KF_grid(ωmax::Float64, R::Int, D::Int)
    ωbos = KF_grid_bos(ωmax, R)
    ωfer = KF_grid_fer(ωmax, R)
    return ntuple(i -> ifelse(i==1, ωbos, ωfer), D)
end

function KF_grid_fer(ωmax::Float64, R::Int)
    KF_grid_fer_(ωmax, 2^R)
end

"""
1D grid
"""
function KF_grid_fer_(ωmax::Float64, N::Int)
    @assert ωmax > 0.0
    @assert iseven(N)
    ωmin = -ωmax
    ωfer_off = 0.5*(ωmax - ωmin)/N
    return collect(range(-ωmax + ωfer_off, ωmax - ωfer_off; length=N))
end

function KF_grid_bos(ωmax::Float64, R::Int)
    KF_grid_bos_(ωmax,2^R)
end

"""
1D grid with N+1 points
"""
function KF_grid_bos_(ωmax::Float64, N::Int)
    @assert ωmax > 0.0
    @assert iseven(N)
    return collect(range(-ωmax, ωmax; length=N+1))
end

"""
Linear iK ∈ {1,...,2^D} to tuple (k1, ..., kD)
"""
function KF_idx(iK::Int, D::Int)
    TCI4Keldysh.@DEBUG 1<=iK<=2^(D+1) "Invalid Keldysh index"
    iK_it = Iterators.product(fill(1:2, D+1)...)
    return collect(iK_it)[iK]
end


"""
Tuple (k1, ..., kD+1) to linear idx iK ∈ {1,...,2^D+1}
"""
function KF_idx(K::NTuple{N, Int}, D::Int) :: Int where {N}
    TCI4Keldysh.@DEBUG all(1 .<=K .<=2) "Invalid Keldysh index"
    K_it = collect(Iterators.product(fill(1:2, D+1)...))
    c_idx = findfirst(k -> k==K, K_it)
    return LinearIndices(K_it)[c_idx]
end

"""
Frequency transformation from external to internal frequencies of a partial correlator.
"""
function Gp_trafo(mat::Matrix{T}, perm) where {T}
    D = size(mat,2)
    return cumsum(mat[perm[1:D],:], dims=1)
end

function _to_LinRange(v::AbstractVector{Float64})
    return LinRange(first(v), last(v), length(v))     
end

"""
Construct the frequency grids of all (D+1)! partial correlators based on the
external frequency grid.
"""
function Gp_grids(ωs_ext::NTuple{D, Vector{Float64}}, ωconvMat::Matrix{Int}) where {D}
    pp = permutations(1:D+1)
    grids = Vector{NTuple{D,LinRange{Float64}}}(undef, length(pp))
    for (ip,p) in enumerate(pp)
        # DxD matrix
        mat_p = Gp_trafo(ωconvMat, p)
        ωs_int,_,_ = _trafo_ω_args(ωs_ext, mat_p)
        grids[ip] = ntuple(i -> _to_LinRange(ωs_int[i]), D)
    end
    return grids
end

function ids_KF(npt::Int)
    return Iterators.product(fill([1,2],npt)...)
end

function get_PauliX()
    return [0.0 1.0; 1.0 0.0] .+ 0.0*im
end

"""
Return D symmetric Matsubara grids, starting with one bosonic grid.
"""
function MF_npoint_grid(T::Float64, Nhalf::Int, D::Int)
    ωbos = MF_grid(T, Nhalf, false)
    ωfer = MF_grid(T, Nhalf, true)
    return ntuple(i -> (i==1) ? ωbos : ωfer, D)
end

"""
Determine equidistant grid such that trafo*ω for ω ∈ ωs_ext is always contained in grid.
ωs_ext must be equidistant.
"""
function trafo_grids(ωs_ext::NTuple{D,Vector{Float64}}, trafo::Matrix{Int}) where {D}
    @assert size(trafo,2)==D

    # check grid spacing
    spacings = Vector{Float64}(undef,D)
    for (i,om) in enumerate(ωs_ext)
        dom = diff(om)
        @assert all(isapprox.(dom, dom[1]; atol=1.e-10)) "Grid must be equidistant"
        spacings[i]=dom[1]
    end
    @assert all(isapprox.(spacings, spacings[1]; atol=1.e-12))

    # determine grid bounds
    vertices_ext = collect(Iterators.product(ntuple(i -> [ωs_ext[i][1], ωs_ext[i][end]], D)...)) 
    vertices_new = [trafo * collect(v) for v in vertices_ext]
    D2 = size(trafo,1)
    gridmins = [minimum([v[d] for v in vertices_new]) for d in 1:D2]
    gridmaxs = [maximum([v[d] for v in vertices_new]) for d in 1:D2]

    # return new grids with same spacing as old ones
    nsteps = [round(Int, (gridmaxs[i] - gridmins[i])/spacings[1]) + 1 for i in 1:D2]
    ωs_new = ntuple(i -> collect(range(gridmins[i], gridmaxs[i]; length=nsteps[i])), D2)
    return ωs_new
end

"""
Determine index shift s such that for all idx:
ωs_int[trafo * idx + s] = trafo * ωs_ext[idx] 
"""
function idx_trafo_offset(
    ωs_ext::NTuple{D1,Vector{Float64}},
    ωs_int::NTuple{D2,Vector{Float64}},
    trafo::Matrix{Int}
    )::Vector{Int} where {D1,D2}

    @assert size(trafo)==(D2,D1)

    idx = fill(1, D1)
    trafoidx = trafo*idx
    trafo_om = trafo*[ωs_ext[i][idx[i]] for i in 1:D1]
    # find transformed index in internal grid
    idx_int = [findfirst(om -> isapprox(trafo_om[i],om; atol=1.e-10), ωs_int[i]) for i in 1:D2]
    @assert all(.!isnothing.(idx_int)) "External and internal grids do not match"
    # idx_int = trafo*idx + s
    s = idx_int - trafoidx
    return s
end

function trafo_grids_offset(ωs_ext::NTuple{D,Vector{Float64}}, trafo::Matrix{Int}) where {D}
    ωs_int = trafo_grids(ωs_ext, trafo)
    s = idx_trafo_offset(ωs_ext, ωs_int, trafo)
    return (ωs_int, s)
end

"""
Convenience wrapper to compute self-energy grid that contains
all values Σ(ν+ω) for frequencies ν,ω
"""
function Σ_grid(ωs_ext::NTuple{2,Vector{Float64}})
    return only(trafo_grids(ωs_ext, reshape([1,1],(1,2))))
end

"""

Deduce missing S[O₁,O₂,O₃,O₄] by use of symmetries => relate to exchange of the 2 creation (annihilation) operators
"""
function symmetry_expand(path::String, Ops::Vector{String}; nested_ωdisc=false)
    if !(length(Ops)==4)
        throw(ArgumentError("Ops must contain 4 strings."))
    end
    filelist = readdir(path)
    
    function deduce_Adisc(path, Ops::Vector{String}, perm::Vector{Int}; combination::Matrix{Int})

        Adiscs = [TCI4Keldysh.load_Adisc(path, Ops, i) for i in 1:2]
        ωdisc = TCI4Keldysh.load_ωdisc(path, Ops; nested_ωdisc=nested_ωdisc)
        
        filename = parse_Ops_to_filename(Ops[perm])
        fullfilename = joinpath(path, filename)
        if any(filename .==filelist) # don't overwrite existing files
            @VERBOSE filename*" exists!\n"
        else
            @VERBOSE "Creating "* filename* ".\n"


            Adiscs_new = [combination[i,1] * Adiscs[1] + combination[i,2] * Adiscs[2] for i in 1:2]
        
            try
                f = matopen(fullfilename)

                write(f, "Adisc", Adiscs_new)
                write(f, "odisc", ωdisc)
                close(f)
            catch
                f = matopen(fullfilename, "w")

                write(f, "Adisc", Adiscs_new)
                write(f, "odisc", ωdisc)
                close(f)
            end
        end
    
    end
    
    ops_dagged = length.(Ops) .>= 5 # BitVector for dag-ed Strings in Ops
    
    # operators can be exchanged if both are F(Q) operators
    exchange_annih_ops = Ops[.!ops_dagged][1][1] == Ops[.!ops_dagged][2][1]
    exchange_creat_ops = Ops[  ops_dagged][1][1] == Ops[  ops_dagged][2][1]
    
    if exchange_annih_ops
        perm = collect(1:4)
        to_perm = perm[.!ops_dagged]
        perm[to_perm[1]], perm[to_perm[2]] = perm[to_perm[2]], perm[to_perm[1]]
        deduce_Adisc(path, Ops, perm, combination=[1 0; 1 -1])
    end
    
    if exchange_creat_ops
        perm = collect(1:4)
        to_perm = perm[ops_dagged]
        perm[to_perm[1]], perm[to_perm[2]] = perm[to_perm[2]], perm[to_perm[1]]
        deduce_Adisc(path, Ops, perm, combination=[1 0; 1 -1])
    end
    
    if exchange_annih_ops && exchange_creat_ops
        perm = collect(1:4)
        to_perm = perm[ops_dagged]
        perm[to_perm[1]], perm[to_perm[2]] = perm[to_perm[2]], perm[to_perm[1]]
        to_perm = perm[.!ops_dagged]
        perm[to_perm[1]], perm[to_perm[2]] = perm[to_perm[2]], perm[to_perm[1]]
        deduce_Adisc(path, Ops, perm, combination=[1 0; 0 1])
    end
    return nothing
end

function svd_kernels!(td::AbstractTuckerDecomp{D}; cutoff::Float64=1.e-15) where {D}
    tmp_legs = Vector{eltype(td.legs)}(undef, D)
    for d in 1:D
        U, S, V = svd(td.legs[d])
        notcut = S .> cutoff
        tmp_legs[d] = Diagonal(S[notcut]) * V'[notcut, :]
        td.legs[d] = U[:,notcut]
        if TCI4Keldysh.DEBUG_RAM()
            U=nothing
            S=nothing
            V=nothing
        end
    end
    center_new = TCI4Keldysh.contract_1D_Kernels_w_Adisc_mp(tmp_legs, td.center)
    td.center = center_new
    td.modified = true
    if TCI4Keldysh.DEBUG_RAM()
        tmp_legs=nothing
    end
    return nothing
end

"""
Automatic svd cutoff to guarantee a pointwise error below the tolerance (regular part of correlator):
|G(ω) - G'(ω)| ≤ (∑_ϵ (K(ω-ϵ)-K'(ω-ϵ))^2) ^ (1/2) * || A ||_2 (A≡PSF)
≤ ( ||S1 - S'1||_∞ * ||S2||_∞ * ||S3||_∞ + 
    ||S1||_∞ * ||S2 - S'2||_∞ * ||S3||_∞ + 
    ||S1||_∞ * ||S2||_∞ * ||S3 - S'3||_∞ ) * ||A||_2 !≤ tolerance

where S is the singular value vector of the i-th tucker leg.
Analogously:
|G(ω)| ≤ ||S1||_∞ * ||S2||_∞ * ||S3||_∞ * ||A||_2

For estimator=1, we estimate ∑_ϵ |K(ω-ϵ)|^2 ≤ ||S||_∞^2
For estimator=2, we estimate ∑_ϵ |K(ω-ϵ)|^2 ≤ max_ω ∑_ϵ |K(ω-ϵ)|^2 (only slightly better estimate...)

If requested, the tolerance can be divided by an estimate of max_ω G(ω) to get a relative error bound.
CAREFUL: Estimate seems to be very conservative, at least for Matsubara
"""
function auto_svd_cutoff(td::AbstractTuckerDecomp{D}, tolerance::Float64, relative=false; estimator::Int=1) where {D}
    AL2 = norm(td.center)    
    # obtain singular values info
    Ss = Vector{Vector{Float64}}(undef, D)
    Sprod = ones(D)
    if estimator==1
        for d in 1:D
            _, S, _ = svd(td.legs[d])
            Ss[d] = S
        end
        Smax = [S[1] for S in Ss]
        Sprod = [prod(i -> i==j ? 1.0 : Smax[i], 1:D) for j in 1:D]
    elseif estimator==2
        Kws = []
        for d in 1:D
            _, S, _ = svd(td.legs[d])
            Ss[d] = S
            Kw = sqrt(maximum(sum(abs.(td.legs[d]) .^ 2; dims=2)))
            push!(Kws, Kw)
        end
        Sprod = [prod(i -> i==j ? 1.0 : Kws[i], 1:D) for j in 1:D]
    else
        error("Invalid estimator $estimator")
    end
    @show Sprod
    Ss = nothing
    # ensures |G(ω) - G'(ω)| ≤ tolerance
    cutoff = tolerance / (sum(Sprod) * AL2)
    if relative
        # lower bound on max_ω G(ω), assuming maximum is in the middle
        N = 2^4
        windowsz = [div(min(N, size(leg, 1)), 2) - 1 for leg in td.legs]
        halfsz = [div(size(leg,1),2) for leg in td.legs]
        window_Kernels = [td.legs[i][halfsz[i]-windowsz[i] : halfsz[i]+windowsz[i], :] for i in eachindex(td.legs)]
        data_middle = contract_1D_Kernels_w_Adisc_mp(window_Kernels, td.center)
        cutoff *= maximum(abs.(data_middle))
    end
    return cutoff
end


function shift_singular_values_to_center!(td::AbstractTuckerDecomp{D}) where{D}
    tmp_legs = Vector{typeof(td.legs[1])}(undef, 0)
    for d in 1:D
        # shift all singular values to central tensor
        U, S, V = svd(td.legs[d])
        push!(tmp_legs, Diagonal(S)*V')
        td.legs[d][:] = U[:]
    end
    center_new = TCI4Keldysh.contract_1D_Kernels_w_Adisc_mp(tmp_legs, td.center)
    td.center[:] = center_new[:]
    td.modified = true
    
    return nothing
end


function shift_singular_values_to_center_DIRTY!(broadenedPsf::AbstractTuckerDecomp{D}) where{D}
    tiny = 1.e-13

    # modify Adisc and Kernels by multiplying / dividing by |ωdisc|
    for d in 1:D
        ωdisc = broadenedPsf.ωs_center[d]
        modifier_Kernel = [.-ωdisc[ωdisc.<-tiny]; ones(sum(abs.(ωdisc).< tiny)); ωdisc[ωdisc.> tiny]]
        broadenedPsf.legs[d] .= broadenedPsf.legs[d] .* modifier_Kernel'
        
        modifier_Adisc = 1 ./ modifier_Kernel
        broadenedPsf.center .= broadenedPsf.center .* reshape(modifier_Adisc, (ones(Int, d-1)..., length(modifier_Adisc)))
    end
    broadenedPsf.modified = true

    return nothing
end

function svd_trunc_Adisc!(td::AbstractTuckerDecomp{D}; atol::Float64) where{D}
    Adisc_tmp = td.center
    Kernels_new = [td.legs...]

    for it1 in 1:D

        sizetmp = [size(Adisc_tmp)...]
        U, S, V = svd(reshape(Adisc_tmp, (sizetmp[1], prod(sizetmp[2:end]))))
        oktmp = S.>atol
        sizetmp[1] = sum(oktmp)
        Kernels_new[it1] = (td.legs[it1] * U)[:,oktmp]
        Adisc_tmp = reshape((Diagonal(S[oktmp]) * V[:,oktmp]'), (sizetmp...))

        if D>1
            Adisc_tmp = permutedims(Adisc_tmp, (collect(2:D)..., 1))
        end

    end

    td.center = Adisc_tmp
    td.legs = Kernels_new
    td.ωs_center = typeof(td.ωs_center)(undef, D) # after shifting the singular values there is no concept of ωdiscs anymore
    return nothing# broadenedPsf_new
end


"""
Performs interpolative decomposition of matrix A_in.
For a given tolerance it returns the column indices that should be kept.

# Arguments
1. A_in     ::Matrix    matrix that is to be decomposed

# Keyword arguments
1. atol     ::Float64   absolute tolerance (approximately in terms of truncated singular values)
2. ncols_min::Int       determines the minimal number of columns

# Returns
1. p        ::Vector{Int}   column indices that should be kept.
"""
function interp_decomp(A_in::AbstractMatrix; atol::Float64=1e-8, rtol::Float64=1e-8, ncols_min::Int=-1)
    # check input
    @assert atol > 0.
    @assert ncols_min ≤ size(A_in, 2)

    _, s, _ = svd(A_in)
    _, r, p = qr(A_in, Val(true))
    #s
    #i_s = argmax(s .< tol)
    #s[i_s]
    
    imax = -1
    #println("argmax(s .< atol): \t", argmax(s .< atol))
    #println("argmax(s ./ s[1] .< rtol): \t", argmax(s ./ s[1] .< rtol))
    #println("length(s): ", length(s))
    #for i_s in max(argmax(s .< atol),argmax(s ./ s[1] .< rtol)):length(s)
    for i_s in min(argmax(s .< atol),argmax(s ./ s[1] .< rtol)):length(s)
        #println(i_s)
        R22 = r[i_s:end, i_s:end]
        _, s2, _ = svd(R22)
            if (s2[1] < atol && s2[1] < s[1]*rtol) || i_s == length(s)
                #println("Found an imax!")
                imax = i_s
                break
            end
    end
    imax = max(imax, ncols_min)
    #println("imax: ", imax)
    return sort(p[1:imax])
    
end

function linear_least_squares(A, b; metric=nothing)
    if !(metric === nothing)
        @assert ndims(metric) == 1
        A = metric .* A
        b = metric .* b
    end
    @DEBUG begin A_bckup = deepcopy(A); true end ""    #Check linear least squares fit:
    qrA = qr(A);                    # QR decomposition
    x = qrA\b;
    #println(qrA.R)
    @DEBUG maximum(abs.(A - A_bckup)) < 1e-13 "Matrix A got changed!"
    @DEBUG begin dev = maximum(abs.(b - A*x)); dev < 1e-12 end "Linear least squares fit unstable. Deviation in to: \t$dev"
    return x
end


"""
    interpolDecomp4TD(Gp::AbstractTuckerDecomp{D}; atol::Float64=1e-0, rtol::Float64=1e-5)

Perform interpolative decomposition of tucker legs
"""
function interpolDecomp4TD(Gp::AbstractTuckerDecomp{D}; atol::Float64=1e-0, rtol::Float64=1e-5) where{D}
    Kernels = Gp.legs
    #ωdiscs = Gp.ωdiscs
    #iωs = Gp.ωs_int
    p_ωdiscs = [ones(Int, 1) for _ in 1:D]
    p_iωs = [ones(Int, 1) for _ in 1:D]

    Gp_data = Gp[[Colon() for _ in 1:D]...]
    Gp_data_tmp = deepcopy(Gp_data)
    Kernels_new = deepcopy(Kernels)
    for i in 1:D
        K_in = Kernels[i]
        p_ωdisc = interp_decomp(K_in; atol, rtol)
        K_interm = K_in[:,p_ωdisc]
        println("length(p_ωdisc): ", length(p_ωdisc))
        p_iω = interp_decomp(transpose(K_interm); atol, rtol, ncols_min=length(p_ωdisc))
        K_new = K_interm[p_iω,:]
        Kernels_new[i] = K_new

        _, s, _ = svd(K_in)
        #sum(s .> atol)

        
        #Gp_data = Gp.Kernels[1] * Gp.Adisc

        sz_Gp_data = size(Gp_data_tmp)
        Gp_data_tmp = reshape(Gp_data_tmp, (sz_Gp_data[1], prod(sz_Gp_data[2:end])))
        Gp_data_DLR = Gp_data_tmp[p_iω,:]
        adisc_DLR = linear_least_squares(K_new, Gp_data_DLR)
        Gp_data_DLRapprox = K_new * adisc_DLR

        #println("dimenstion i/D = $i/D")
        #println("\t abs error of approximant: ", maximum(abs.(Gp_data_DLR - Gp_data_DLRapprox)))
        #println("\t deviation in coefficients", maximum(abs.(Gp_data_tmp[p_ωdisc,:] - adisc_DLR)))

        p_ωdiscs[i] = p_ωdisc
        p_iωs[i] = p_iω
        Gp_data_tmp = permutedims(reshape(adisc_DLR, (size(adisc_DLR, 1), prod(sz_Gp_data[2:end]))), [collect(2:D)..., 1])
    end

    Adisc_new = Gp_data_tmp
    size(Adisc_new)
    size.(Kernels)
    
    @DEBUG begin
        Kernels_new_large = [Gp.legs[i][:,p_ωdiscs[i]] for i in 1:D]
        Gp_data_new = contract_1D_Kernels_w_Adisc_mp(Kernels_new_large, Adisc_new)
        devabs = maximum(abs.(Gp_data - Gp_data_new))
        println("abs. deviation of compressed MF correlator on original domain: \t", devabs)
        devabs < atol && devabs/maximum(abs.(Gp_data)) < rtol
    end "DLR compression did not work within the required tolerance."

    td_new = TuckerDecomposition(Adisc_new, Kernels_new; ωs_center=Gp.ωs_center, ωs_legs=Gp.ωs_legs, idxs_center=p_ωdiscs, idxs_legs=p_iωs)
    return td_new#(kernels_new=Kernels_new, adisc_new=Adisc_new, p_iωs=p_iωs, p_ωdiscs=p_ωdiscs)

end

"""
    function eval_ano_matsubara_kernel(oms::Vector{Float64}, omprimes::Vector{Float64}, beta::Float64)

Evaluate anomalous part of matsubara kernel
-0.5(β + ∑_{j≠i} 1/(i⋅ωj - ωj')) * ∏_{i≠j} 1/(i⋅ωj - ωj')
where i is the bosonic frequency index.

oms, omprimes should NOT contain the zero frequencies
"""
function eval_ano_matsubara_kernel(oms::Vector{Float64}, omprimes::Vector{Float64}, beta::Float64)
    product = 1.0
    sum = 0.0
    for i in eachindex(oms)
        Ominv = 1/(im*oms[i] - omprimes[i])
        sum += Ominv
        product *= Ominv
    end
    return -0.5*(beta + sum)*product
end

"""
Fit y = α*x + β. \\
NOTE: method linear_least_squares already exists...
"""
function linear_fit(x::Vector{T}, y::Vector{T}) where {T<:Number}
    A = hcat(x, ones(T, length(x)))
    out = A \ y # y = αx + β
    println("  LSQ err: $(norm(A*out - y))")
    return out
end

function rank(qtt::QuanticsTCI.QuanticsTensorCI2)
    return maximum(TCI.linkdims(qtt.tci))
end

function rank(mps::MPS)
    return maximum(dim.(linkinds(mps)))
end

function absmax(v)
    return maximum(abs.(v))
end

"""
Where to find PSF data
"""
function datadir()
    return joinpath(dirname(Base.current_project()), "data")
    # return joinpath("/scratch/m/M.Frankenbach/tci4keldysh", "data")
end

"""
Parent dir of datadir
"""
function pdatadir()
    return dirname(datadir())
end

function jld2_to_dictkey(jldfile::AbstractString)
    return reduce(*, split(jldfile, '.')[1:end-1])
end

function to_kwarg_dict(d::Dict{String,Any})::Dict{Symbol,Any}
    dret = Dict{Symbol,Any}()
    for (k,v) in pairs(d)
        dret[Symbol(k)] = v
    end
    return dret
end

"""
For given PSFpath, get corresponding temperature.
"""
function dir_to_T(PSFpath::String) :: Float64
    d = Dict(
        "SIAM_u=1.00"=>1.0/2000.0,
        "SIAM_u=0.50"=>1.0/2000.0,
        "SIAM_u=1.50"=>1.0/2000.0,
        "SIAM_u3_U0.05_T0.0005_Delta0.0053052" => 1.0/2000.0,
        "SIAM_u5_U0.05_T0.0005_Delta0.0031831" => 1.0/2000.0,
        "siam05_U0.05_T0.005_Delta0.0318"=>1.0/200.0,
        "SIAM_strong2_U0.2_T0.0001"=>1.0/10000.0,
        "siam05_U0.05_T0.05_Delta0.0318"=>1.0/20.0,
        "PRX_jae-mo_PSF"=>1.e-4,
        "unittest_PSF"=>1.0/2000.0
        )
    for (key, val) in d
        if contains(PSFpath, key)
            return val
        end
    end
    error("No temperature found for path: $PSFpath")
    return 0.0
end

function dir_to_beta(PSFpath::String) :: Float64
    return 1.0 / dir_to_T(PSFpath)
end


function iterate_binvec(R::Int)
    return Iterators.product(fill([1,2], R)...)
end

"""
Where to store broadened spectral function data
"""
function Acont_h5fname(perm_idx::Int, D::Int; Acont_folder="Acontdata")
    return joinpath(Acont_folder, "Acont$(D)D_$perm_idx.h5")
end

function report_mem(do_gc=false)
    println("---------- MEMORY REPORT ----------")
    if do_gc
        println("  Available system memory (before gc()): $(Sys.free_memory() / 1024^2) MB")
        println("  GC live: $(Base.gc_live_bytes() / 1024^2) MB")
        Base.GC.gc(true)
        println("  Garbage collected")
    end
    println("  GC live: $(Base.gc_live_bytes() / 1024^2) MB")
    println("  JIT: $(Base.jit_total_bytes() / 1024^2) MB")
    println("  Max RSS: $(Sys.maxrss() / 1024^2) MB")
    println("  Total system memory: $(Sys.total_memory() / 1024^2) MB")
    println("  Available system memory: $(Sys.free_memory() / 1024^2) MB")
    println("-----------------------------------")
    flush(stdout)
    flush(stderr)
end

function channel_K1_Ops(channel::String)
    if channel=="t"
        return ["Q12", "Q34"]
    elseif channel=="a"
        return ["Q14", "Q23"]
    elseif channel in p_conventions()
        return ["Q13", "Q24"]
    else
        error("Invalid channel $channel")
    end
end

function channel_K1_sign(channel::String)
    return if channel=="t"
        1
    elseif channel in p_conventions()
        # ζ=-1 in symmetric estimators
        1
    elseif channel=="a"
        -1
    else
        error("Invalid channel")
    end
end

function channel_translate(channel::String)
    if channel=="t"
        return "ph"
    elseif channel=="a"
        return "pht"
    elseif channel in p_conventions()
        return "pp"
    else
        error("Invalid channel $channel")
    end
end

function foreign_channels(channel::String)
    if channel=="a"
        return ("p","t")
    elseif channel in p_conventions()
        return ("a","t")
    elseif channel=="t"
        return ("a","p")
    else
        error("Invalid channel $channel")
    end
end


"""
* channel: a, p or t
"""
function channel_trafo(channel::String)
    #= cf. Seung-Sup's README.txt:
-            pht/ph bar (a) : (
            nu_r,
            -nu'_r          ,
            nu'_r + omega_r,
            -nu_r  - omega_r
            ) 
-            pp(p) : (
            nu_r,
            -nu'_r          ,
            -nu_r - omega_r,
            nu'_r  + omega_r
            ) 
-            ph(t) : (
            nu_r,
            -nu_r  - omega_r,
            nu'_r + omega_r,
            -nu'_r
            ) 
    =#
    ωconvMat = if channel == "a"
          # -ω -ν -ν' (cf. eq. 138 Lihm et al 2021)
        [
            0 -1  0; # ν
            0  0  1; # -ν'
            -1  0 -1; # ω+ν'
            1  1  0; # -ω-ν
        ]
    # MBE solver convention
    elseif channel == "p"
        [
            0 -1  0; # ν
            1  0 -1; # -ω+ν'
            -1  1  0; # ω-ν
            0  0  1; # -ν'
        ]
    elseif channel == "pQFT"
        [
            0 -1  0; # ν
            -1  0 -1; # -ω+ν'
            1  1  0; # ω-ν
            0  0  1; # -ν'
        ]
    # NRG convention (cf. eq. 138 Lihm et al symmetric estimators)
    elseif channel == "pNRG"
        [
        0 -1  0;
        0  0  1;
        1  1  0;
        -1  0 -1;
        ]
    elseif channel == "t"
        [
            0 -1  0; # ν
            1  1  0; # -ω-ν
            -1  0 -1; # ω+ν'
            0  0  1; # -ν'
        ]
    else
        error("Invalid frequency convention $channel")
    end
    return ωconvMat
end

function p_conventions()
    return ["p", "pNRG", "pQFT"]
end

function channel_change(from::String, to::String)
    trafo_from = channel_trafo(from)    
    trafo_to = channel_trafo(to)
    return round.(Int, pinv(trafo_to) * trafo_from)
end

function merged_legs_K2(channel::String, prime::Bool)
    noprime_dict = Dict("a" => (2,3), "p" => (2,4), "t" => (3,4))
    noprime_dict["pNRG"] = noprime_dict["p"]
    noprime_dict["pQFT"] = noprime_dict["p"]
    prime_dict = Dict("a" => (1,4), "p" => (1,3), "t" => (1,2))
    prime_dict["pNRG"] = prime_dict["p"]
    prime_dict["pQFT"] = prime_dict["p"]
    if !prime
        return noprime_dict[channel]
    else
        return prime_dict[channel]
    end
end

function merged_legs_K1(channel::String)
    if channel=="t"
        return [(1,2), (3,4)]
    elseif channel in p_conventions()
        return [(1,3), (2,4)]
    elseif channel=="a"
        return [(1,4), (2,3)]
    else
        error("Invalid channel $channel")
    end
end


function oplabels_K2(channel::String, prime::Bool)
    (i,j) = merged_legs_K2(channel, prime)
    nonij = sort(setdiff(1:4, (i,j)))
    leg_labels = ("1", "1dag", "3", "3dag")
    return ["Q$i$j", leg_labels[nonij[1]], leg_labels[nonij[2]]]
end

"""
Which sign belongs to which K2 contribution (cf. Lihm et. al. Fig. 13)
"""
function channel_K2_sign(channel::String, prime::Bool)
    if channel=="a"
        return ifelse(prime, 1, -1)
    elseif channel in p_conventions()
        return ifelse(prime, 1, -1)
    elseif channel=="t"
        return 1
    else
        error("Invalid channel $channel")
    end
end

function channel_trafo_K2(channel::String, prime::Bool)
    ωconvMat = channel_trafo(channel)
    if !prime
        if channel=="a"
            return [
                sum(view(ωconvMat, [2,3], [1,2]), dims=1);
                view(ωconvMat, [1,4], [1,2])
            ]
        elseif channel in p_conventions()
            return [
                sum(view(ωconvMat, [2,4], [1,2]), dims=1);
                view(ωconvMat, [1,3], [1,2])
            ]
        elseif channel=="t"
            return [
                sum(view(ωconvMat, [3,4], [1,2]), dims=1);
                view(ωconvMat, [1,2], [1,2])
            ]
        end
    else
        if channel=="a"
            return [
                sum(view(ωconvMat, [1,4], [1,3]), dims=1);
                view(ωconvMat, [2,3], [1,3])
            ]
        elseif channel in p_conventions()
            return [
                sum(view(ωconvMat, [1,3], [1,3]), dims=1);
                view(ωconvMat, [2,4], [1,3])
            ]
        elseif channel=="t"
            return [
                sum(view(ωconvMat, [1,2], [1,3]), dims=1);
                view(ωconvMat, [3,4], [1,3])
            ]
        end
    end
end

"""
External -> internal frequency conversion for 2pt functions
"""
function ωconvMat_K1()
    return reshape([1; -1], (2,1))
end

"""
cf. eqs. (73, 111) Lihm et. al.
"""
function merge_iK(ik1::Int, ik2::Int)
    return iseven(ik1+ik2) ? 1 : 2
end

function unfold_iK(ik::Int)
    if ik==1
        return [[1,1],[2,2]]
    else
        return [[1,2],[2,1]]
    end
end

"""
Get Keldysh 4-indices that coincide in ik2 component of K2
* ik2: Keldysh idx of K2
* merged_idx: which idx in ik2 should be unfolded into two indices
"""
function equivalent_iK_K2(ik2::NTuple{3,Int}, channel::String, prime::Bool; merged_idx::Int=1)
    ikunfold = unfold_iK(ik2[merged_idx])
    l1,l2 = merged_legs_K2(channel, prime)    
    ret = []
    for k in ikunfold
        ret_act = zeros(Int, 4)
        ret_act[l1] = k[1]
        ret_act[l2] = k[2]
        ret_act[ret_act.==0] .= ik2[[i!=merged_idx for i in 1:3]]
        push!(ret, ret_act)
    end
    return ret
end

"""
Assign 4-component Keldysh index to 3-component Keldysh index of K2r(')
"""
function merge_iK_K2(ik::NTuple{4,Int}, channel::String, prime::Bool; merged_idx::Int=1)::NTuple{3,Int}
    l1,l2 = merged_legs_K2(channel, prime)    
    ik1 = merge_iK(ik[l1], ik[l2])
    ret = [ik[l] for l in 1:4 if !(l in [l1,l2])]
    insert!(ret, merged_idx, ik1)
    return Tuple(ret)
end

"""
Get tensor that implements Keldysh index merging for K1
    ret[ik4,ik2]=1 means that ik4 merges to ik2
"""
function iK_K2_trafo(channel::String, prime::Bool)
    ret = zeros(Int, ntuple(_->2,7))
    for ik in ids_KF(4)
        ret[ik..., merge_iK_K2(ik, channel, prime)] = 1
    end
end

function unfold_K2(K2val::Array{ComplexF64,3}, channel::String, prime::Bool)
    ret = zeros(eltype(K2val), 2,2,2,2)    
    for ik in ids_KF(4)
        ret[ik...] = K2val[merge_iK_K2(ik,channel,prime)...]
    end
    return ret
end

"""
Assign 4-component Keldysh index to 2-component Keldysh index of K1r
"""
function merge_iK_K1(ik::NTuple{4,Int}, channel::String)::NTuple{2,Int}
    l12, l34 = merged_legs_K1(channel)    
    ik1 = merge_iK(ik[l12[1]], ik[l12[2]])
    ik2 = merge_iK(ik[l34[1]], ik[l34[2]])
    return (ik1, ik2)
end

"""
Get tensor that implements Keldysh index merging for K1
    ret[ik4,ik2]=1 means that ik4 merges to ik2
"""
function iK_K1_trafo(channel::String)
    ret = zeros(Int, ntuple(_->2,6))
    for ik in ids_KF(4)
        ret[ik..., merge_iK_K1(ik, channel)] = 1
    end
end

function unfold_K1(K1val::Array{ComplexF64,2}, channel::String)
    ret = zeros(eltype(K1val), 2,2,2,2)    
    for ik in ids_KF(4)
        ret[ik...] = K1val[merge_iK_K1(ik,channel)...]
    end
    return ret
end

"""
Get Keldysh 4-indices that coincide in ik1 component of K1
* ik1: Keldysh idx of K1
"""
function equivalent_iK_K1(ik1::NTuple{2,Int}, channel::String)
    unfold1 = unfold_iK(ik1[1])
    unfold2 = unfold_iK(ik1[2])
    l1,l2 = merged_legs_K1(channel)    
    ret = []
    for (u1,u2) in Iterators.product(unfold1, unfold2)
        ret_act = zeros(Int, 4)
        ret_act[collect(l1)] .= u1
        ret_act[collect(l2)] .= u2
        push!(ret, ret_act)
    end
    return ret
end


# ==== JSON
function logJSON(data::Any, filename::String, folder::String="tci_data"; verbose=true)
    fullname = filename*".json"
    open(joinpath(folder, fullname), "w") do file
        JSON.print(file, data)
    end
    if verbose
        printstyled(" ---- File $filename.json written!\n", color=:green)
    end 
end

function readJSON(filename::String, folder::String="tci_data")
    path = if endswith(filename, ".json")
        joinpath(folder, filename)
    else
        joinpath(folder, filename*".json")
    end
    data = open(path) do file
        JSON.parse(file)
    end 
    return data
end

function updateJSON(filename::String, key::String, val::Any, folder::String="tci_data")
    data = readJSON(filename, folder)
    data[key] = val
    logJSON(data, filename, folder; verbose=false)
end
# ==== JSON END

function log_message(io, message::String; color=:cyan)
    printstyled(message * "\n"; color=color)
    println(io, message)
end

function tolstr(tolerance::Float64)
    return "$(round(Int, log10(tolerance)))"
end

"""
Translate bonddims of a TT with complex entries and leg dimension d to RAM usage
in MB
"""
function bonddims_to_RAM(bonddims::Vector{Int}, d::Int=2)
    bonddims_ = vcat([1], bonddims, [1])
    CPX_BYTES = 16
    ram = 0
    for ib in eachindex(bonddims_)[2:end]
        ram += d*bonddims_[ib]*bonddims_[ib-1]
    end
    return ram*CPX_BYTES/10^6
end


"""
Get plotting object with reasonable font sizes.
"""
function default_plot(kwargs...)
    tfont = 12
    titfont = 16
    gfont = 16
    lfont = 12
    p = plot(;guidefontsize=gfont, titlefontsize=titfont, tickfontsize=tfont, legendfontsize=lfont, kwargs...)
    return p
end

#=
"""
Convenience wrapper to do patched TCI with TCIAlgorithms

Returns ProjTTContainer
"""
function patchedTCI(f::Function, grid::QuanticsGrids.Grid{d}; tolerance=1e-8, maxbonddim=50) where {d}
    R = grid.R
    localdims = grid.unfoldingscheme==:fused ? fill(2^d, R) : fill(2, d*R)
    qf = x -> f(quantics_to_origcoord(grid, x)...)
    pordering = TCIA.PatchOrdering(collect(1:R))

    creator = TCIA.TCI2PatchCreator(
        ComplexF64, qf, localdims; maxbonddim=maxbonddim, tolerance=tolerance, verbosity=0, ntry=10
    )

    return TCIA.adaptiveinterpolate(creator, pordering; verbosity=0)
end

function patchedTCI(A::Array{T,D}; kwargs...) where {T,D}
    
    @assert all(isinteger.(log2.(size(A)))) "A must have length 2^R in each direction"
    R = round(Int, log2.(size(A,1)))
    grid = InherentDiscreteGrid{D}(R; unfoldingscheme=:fused) # interleaved crashes with patching?!
    localdims = grid.unfoldingscheme==:fused ? fill(2^D, R) : fill(2, D*R)
    qf = x -> A[quantics_to_origcoord(grid, x)...]
    pordering = TCIA.PatchOrdering(collect(1:R))

    creator = TCIA.TCI2PatchCreator(
        T, qf, localdims; verbosity=0, ntry=10, kwargs...
    )

    return TCIA.adaptiveinterpolate(creator, pordering; verbosity=0)
end
=#

"""
Read broadening settings from file in the corresponding directory
TODO: What about estep
"""
function read_broadening_settings(path=joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50"); channel="t")

    if !isdir(path)
        path = joinpath(datadir(), path)
    end

    d = Dict{Symbol, Any}()
    matopen(joinpath(path, "mpNRG_$(channel_translate(channel)).mat")) do f
        d[:emin] = read(f, "emin")
        d[:emax] = read(f, "emax")
    end
    return d
end

"""
Obtain broadening parameters from the corresponding mpNRG files
"""
function read_broadening_params(path::String; channel="t")

    if !isdir(path)
        path = joinpath(datadir(), path)
    end

    files = filter(f -> occursin("mpNRG", f), readdir(path; join=true))
    if length(files)==1
        file = only(files)
    else
        file_id = findfirst(f -> occursin(channel_translate(channel), f), files)
        if isnothing(file_id)
            error("No file for channel $channel among files: $files")
        end
        file = files[file_id]
    end

    (γ, sigmak) = (0.0, [0.0])
    matopen(file, "r") do f
        γ = read(f, "Lwidth")
        sigmak = read(f, "Hwidth")
    end
    return (γ, [sigmak])
end

function read_all_broadening_params(base_path; channel)
    (γ, sigmak) = read_broadening_params(base_path; channel=channel)
    broadening_kwargs = read_broadening_settings(joinpath(datadir(), base_path); channel=channel)
    broadening_kwargs[:γ]=γ
    broadening_kwargs[:sigmak]=sigmak
    return broadening_kwargs
end

"""
Group set into orbits w.r.t. operations in ops. Each op∈ops should map op:set->set
"""
function group_orbits(set::Set{T}, ops::Vector{<:Function}) where {T}
    error("NOT TESTED")
    orbits = Vector{Set{T}}()
    not_assigned = copy(set)
    while !isempty(not_assigned)
        s = first(not_assigned)
        orbit = Set(vcat([op(s) for op in ops], [s]))
        not_assigned = setdiff(not_assigned, orbit)
        push!(orbits, orbit)
    end
    return orbits
end

function maybeparse(T::Type, val)
    return isa(val,T) ? val : parse(T, val)
end

"""
Override values in dst with values from src on matching keys.
"""
function override_dict!(src::Dict, dst::Dict)
    for (key,val) in pairs(dst)
        if haskey(src, key)
            T_ = typeof(val)
            dst[key] = maybeparse(T_, src[key])
        end
    end
end

function vprintln(msg::AbstractString, minlevel::Int=1)
    if VERBOSITY[]>=minlevel
        println(msg)
    end
end

function vprint(msg::AbstractString, minlevel::Int=1)
    if VERBOSITY[]>=minlevel
        print(msg)
    end
end

function vprintstyled(msg::AbstractString, minlevel::Int=1; kwargs...)
    if VERBOSITY[]>=minlevel
        printstyled(msg; kwargs...)
    end
end