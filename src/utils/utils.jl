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
Make the discrete spectral data ("Adisc") of a partial spectral function
compact, by removing the rows, columns, or slices of "Adisc" that contain
zeros only.
"""
function compactAdisc(
    ωdisc   ::Vector{Float64}, 
    Adisc   ::Array{Float64}
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
        error("ERR: size of Adisc is inconsistent with the size of odisc.")
    end

    oks = [BitVector(undef, length(ωdisc)) for i in 1:nO]
    AdiscIsZero = abs.(Adisc) .< (maxabs(Adisc) * 1.e-14)
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

    @assert sum(size(Acont) .- size.(Kernels, 1)) == 0

    return Acont
end


"""
    contract_KF_Kernels_w_Adisc_mp(Kernels, Adisc)

Contracts retarded kernels with Adisc and deduces all fully-retarded kernels.

For 3p correlators we e.g. get K^{R/A}[i,a] K^{R/A}[j,b] Adisc[a,b]

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
function contract_KF_Kernels_w_Adisc_mp(Kernels, Adisc)
    sz = [size(Adisc)...]
    D = ndims(Adisc)
    
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
    ys::Array{Float64,d}        # piecewisely linear signal ys[i,j] where dim=1 is sampled on frequencies xs_in
                                # and j represents the trailing dimensions in ys
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

    interp_linear = linear_interpolation(xs_in, ys)
    ys_interp = interp_linear.(xs_out)

    return ys_interp .+ im .* result
end


function load_Adisc(path::String, Ops::Vector{String}, flavor_idx::Int)
    f = matopen(joinpath(path, "PSF_(("*mapreduce(*,*,Ops, ["," for i in 1:length(Ops)])[1:end-1]*")).mat"), "r")
    try 
        keys(f)
    catch
        keys(f)
    end
    Adisc = read(f, "Adisc")[flavor_idx]
    Adisc = dropdims(Adisc,dims=tuple(findall(size(Adisc).==1)...))

    return Adisc 
end


function load_ωdisc(path::String, Ops::Vector{String})
    f = matopen(joinpath(path, "PSF_(("*mapreduce(*,*,Ops, ["," for i in 1:length(Ops)])[1:end-1]*")).mat"), "r")
    try 
        keys(f)
    catch
        keys(f)
    end
    ωdisc  = read(f, "odisc")[:]
    return ωdisc 
end


function parse_filename_to_Ops(fn)
    return split(fn[7:end-6], ",")
end

function parse_Ops_to_filename(ops)
    temp = prod([o*"," for o in ops])
    return "PSF_(("*temp[1:end-1]*")).mat"
end



"""

Deduce missing S[O₁,O₂,O₃,O₄] by use of symmetries => relate to exchange of the 2 creation (annihilation) operators
"""
function symmetry_expand(path::String, Ops::Vector{String})
    if !(length(Ops)==4)
        throw(ArgumentError("Ops must contain 4 strings."))
    end
    filelist = readdir(path)
    
    function deduce_Adisc(path, Ops::Vector{String}, perm::Vector{Int}; combination::Matrix{Int})

        Adiscs = [TCI4Keldysh.load_Adisc(path, Ops, i) for i in 1:2]
        ωdisc = TCI4Keldysh.load_ωdisc(path, Ops)
        
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