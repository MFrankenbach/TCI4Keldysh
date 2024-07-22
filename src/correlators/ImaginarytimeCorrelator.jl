#=
Compute partial and full correlators in imaginary time domain.
EXPERIMENTAL/WIP; NOT TESTED
=#

struct CompressedPSF
    Adisc_mps :: MPS
    beta :: Float64
    ωdisc :: Vector{Vector{Float64}}
    dim :: Int
end

struct ImagtimePartialCorrelator
    Gtau::MPS
    beta::Float64
    isFermi::Vector{Bool}
    dim::Int

    function ImagtimePartialCorrelator(Gtau::MPS, beta::Float64, isFermi::Vector{Bool})
        dim = length(isFermi)
        return new(Gtau, beta, isFermi, dim)
    end
end

function nbits(psf::CompressedPSF)
    return div(length(psf.Adisc_mps), psf.dim)
end

function nbits(G::ImagtimePartialCorrelator)
    return div(length(G.Gtau), G.dim)
end

"""
Get MPS of a PSF (no broadening)
"""
function CompressedPSF(beta::Float64, path::String, Ops::Vector{String}, flavor_idx::Int; tagname="eps", pad_R=nothing, nested_ωdisc=false, kwargs...)
    Adisc = load_Adisc(path, Ops, flavor_idx)
    # load frequencies as well for compactification
    ωdisc = load_ωdisc(path, Ops; nested_ωdisc=nested_ωdisc)
    _, ωdisc, Adisc = compactAdisc(ωdisc, Adisc)
    R_ = if isnothing(pad_R)
            # maximum(trunc.(Int, log2.(collect(size(Adisc)))) .+ 1)
            padding_R(size(Adisc))
        else
            pad_R
        end
    
    function evalAdisc(i...)
        if all(i .<= size(Adisc))        
            return Adisc[i...]
        else
            return zero(Float64)
        end
    end
    # A_qtt, _, _ = quanticscrossinterpolate(TCI4Keldysh.zeropad_array(Adisc, R); kwargs...)  
    padded_size = ntuple(i -> 2^R_, ndims(Adisc))
    A_qtt, _, _ = quanticscrossinterpolate(
        eltype(Adisc),
        evalAdisc,
        padded_size;
        kwargs...)
    Adisc_mps = TCI4Keldysh.QTCItoMPS(A_qtt, ntuple(i->"$(tagname)$i", ndims(Adisc)))

    println("  -- Compressed Adisc with maxbonddim $(rank(Adisc_mps))")
    return CompressedPSF(Adisc_mps, beta, ωdisc, ndims(Adisc))
end

"""
Get TT for 1D spectral function containing peaks at given values.
    * A0: (uniform) peak strength
"""
function get_dummy_CompressedPSF(R::Int, beta::Float64, peaks::Vector{Float64}, A0::Float64=1.0; tagname="eps", kwargs...)

    N_eps = length(peaks)
    Adisc = fill(A0, N_eps)
    ωdisc = [sort(peaks)]
    
    function evalAdisc(i...)
        if only(i) <= N_eps
            return Adisc[i...]
        else
            return zero(Float64)
        end
    end
    padded_size = ntuple(i -> 2^R, ndims(Adisc))
    initpivot =  [1]
    A_qtt, _, _ = quanticscrossinterpolate(
        eltype(Adisc),
        evalAdisc,
        padded_size,
        # make sure eps0 gets sampled as initial pivot
        [initpivot];
        kwargs...)
    Adisc_mps = TCI4Keldysh.QTCItoMPS(A_qtt, ntuple(i->"$(tagname)$i", ndims(Adisc)))

    println("  -- Compressed dummy Adisc(ϵ)=A0*δ(ϵ-ϵ0) with maxbonddim $(rank(Adisc_mps))")
    return CompressedPSF(Adisc_mps, beta, ωdisc, ndims(Adisc))
end

"""
Θ(τ,ϵ), the Matsubara summed 1D regular kernel, bosonic case
* thresh: to tell when ϵ≈0
"""
function imagtimeKernel1D_bos(tau::Float64, eps::Float64, beta::Float64; thresh::Float64=1e-10)
    if abs(eps)<thresh
        # limit ϵ->0 with singularity removed
        return tau >= 0.0 ? tau/beta - 1 : tau/beta
    end
    # seems to be proof against large positive arguments in exp
    if tau >=0.0
        # return -exp(eps*(beta - tau)) / (exp(beta*eps) - 1)
        return -1 / (exp(eps*tau) - exp(eps*(tau - beta)))
    else
        # return -exp(-eps*tau) / (exp(beta*eps) - 1)
        return - 1 / (exp(eps*(tau+beta)) - exp(eps*tau))
    end
end

"""
Θ(τ,ϵ), the Matsubara summed 1D regular kernel, fermionic case
"""
function imagtimeKernel1D_fer(tau::Float64, eps::Float64, beta::Float64)
    # seems to be proof against large positive arguments in exp
    if tau>= 0.0
        # return -exp(eps*(beta - tau)) / (1 + exp(beta*eps))
        return - 1 / (exp(eps*(tau - beta)) + exp(eps*tau))
    else
        # return exp(-eps*tau) / (1 + exp(beta*eps))
        return 1 / (exp(eps*tau) + exp(eps*(beta+tau)))
    end
end

"""
Compress 1D imagtime kernel with bit order τ,ϵ
* ωdisc::Vector{Float64} : Frequency grid of PSF
* taushift τ_j = β(j+taushift)/2^R
"""
function compress_imagtimeKernel(R::Int, ωdisc::Vector{Float64}, beta::Float64, fermion=true, taushift::Float64=0.5;
        tagnames::Tuple{String,String}=("tau","eps"), kwargs...)::MPS
    grid = InherentDiscreteGrid{2}(R; unfoldingscheme=:interleaved)

    # let τ run from 0 to β(1-1/2^R)
    # having τ ∈ [-β,β] does not seem to increase bond dimensions
    step = beta/2^R
    # offset = -beta
    n_omdisc = length(ωdisc)
    _kernelfun = if fermion
                    function _kernelfun_fer(ntau::Int, neps::Int)
                        if neps<=n_omdisc
                            return imagtimeKernel1D_fer(step*(ntau-1 + taushift), ωdisc[neps], beta)
                        else
                            return 0.0
                        end
                    end
                else
                    function _kernelfun_bos(ntau::Int, neps::Int)
                        if neps<=n_omdisc
                            return imagtimeKernel1D_bos(step*(ntau-1 + taushift), ωdisc[neps], beta)
                        else
                            return 0.0
                        end
                    end
                end
    
    # use ComplexF64 here because PSF is also complex?
    kernel_qtt, _, _ = quanticscrossinterpolate(Float64, _kernelfun, grid; kwargs...)
    kernel_mps = TCI4Keldysh.QTCItoMPS(kernel_qtt, tagnames)

    return kernel_mps
end

# ========== TEST FUNCTIONS 

"""
Test kernels by summation
"""
function test_imagtime_kernels()
    beta = 2.1
    epsis = [-3.3, 0.2]

    tol = 1e-6
    for tau in [0.4, -0.3, beta/2]
        @show tau
        for eps in epsis
            @show eps
            fer_num = sum_imagtime_kernel_fer(tau, eps, beta; N=1e6)
            bos_num = sum_imagtime_kernel_bos(tau, eps, beta; N=1e6)
            fer_ana = imagtimeKernel1D_fer(tau, eps, beta)
            bos_ana = imagtimeKernel1D_bos(tau, eps, beta)
            @show fer_ana
            @show bos_ana
            @show abs(fer_num - fer_ana)
            @show abs(bos_num - bos_ana)
            @assert abs(fer_num - fer_ana) < tol
            @assert abs(bos_num - bos_ana) < tol
        end
    end
end

"""
Fermionic imagtime kernel by summation
"""
function sum_imagtime_kernel_fer(tau::Float64, eps::Float64, beta::Float64; N=500)
    res_up = zero(Float64)
    for i in -(N+1):0
        om = π*(2*i+1)/beta
        res_up += real(exp(-im*om*tau) / (im*om - eps))
    end
    res_low = zero(Float64)
    for i in N:-1:1
        om = π*(2*i+1)/beta
        res_low += real(exp(-im*om*tau) / (im*om - eps))
    end
    return (res_up + res_low)/beta
end

"""
Bosonic imagtime kernel by summation
"""
function sum_imagtime_kernel_bos(tau::Float64, eps::Float64, beta::Float64; N=500)
    res_up = zero(Float64)
    for i in -N:0
        om = π*(2*i)/beta
        res_up += real(exp(-im*om*tau) / (im*om - eps))
    end
    res_low = zero(Float64)
    for i in N:-1:1
        om = π*(2*i)/beta
        res_low += real(exp(-im*om*tau) / (im*om - eps))
    end
    return (res_up + res_low)/beta
end

"""
"""
function imagtime_kernel_ranks(;sweep=false, npt=2, R=12, beta=2000.0, perm_idx=1, tolerance=1e-12)
    
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
    
    Rs = if sweep
            12:4:28
        else
            [R]
        end
    betas = if sweep
                [1.e2, 1.e3, 1.e4]
            else
                [beta]
            end
    ranks = Int[]
    for beta_loc in betas
        for R in Rs
            psf = CompressedPSF(beta_loc, path, Ops, spin; nested_ωdisc=false, pad_R=R, tolerance=tolerance)
            ranks = test_kernel_compression(psf; tolerance=tolerance)
            printstyled("\n --- Imagtime kernel ranks: $(ranks)  R=$R  β=$beta_loc --- \n"; color=:green)
        end
    end
    # return last ranks for convenience
    return ranks
end

"""
Investigate kernel compression (in part. ranks) of imaginary time kernels for given psf
"""
function test_kernel_compression(psf::CompressedPSF; plot_kernel=false, kwargs...)
    R = nbits(psf)
    @assert all(log2.(size(psf.ωdisc)) .<= R)

    kernels_fer = Vector{MPS}(undef, psf.dim)
    kernels_bos = Vector{MPS}(undef, psf.dim)
    ranks = Int[]
    for i in 1:psf.dim
        kernels_fer[i] = compress_imagtimeKernel(R, psf.ωdisc[i], psf.beta, true; kwargs...)
        println(" -- Fermionic kernel $i:")
        @show rank(kernels_fer[i])
        push!(ranks, rank(kernels_fer[i]))
    end
    for i in 1:psf.dim
        kernels_bos[i] = compress_imagtimeKernel(R, psf.ωdisc[i], psf.beta, false; kwargs...)
        println(" -- Bosonic kernel:")
        @show rank(kernels_bos[i])
        push!(ranks, rank(kernels_bos[i]))
    end

    if plot_kernel
        fatFermi = TCI4Keldysh.MPS_to_fatTensor(kernels_fer[1]; tags=("tau","eps"))
        fatBose = TCI4Keldysh.MPS_to_fatTensor(kernels_bos[1]; tags=("tau","eps"))
        heatmap(fatFermi)
        savefig("fermi_kernel.png")
        heatmap(fatBose)
        savefig("bose_kernel.png")
    end
    return ranks
end

# ========== TEST FUNCTIONS END

function imagtime_PartialCorrelator_1D_explicit(R::Int, beta::Float64,
        path::String, Ops::Vector{String}, flavour_idx::Int; nested_ωdisc=false)
    Adisc = load_Adisc(path, Ops, flavour_idx)    
    @assert ndims(Adisc)==1 "Only 1D PSF allowed here"
    ωdisc = load_ωdisc(path, Ops; nested_ωdisc=nested_ωdisc)
    epsis = ωdisc

    taus = [(i-1)*beta/2^R for i in 1:2^R]
    kernel = Matrix{Float64}(undef, (2^R, length(epsis)))
    for i in eachindex(taus)
        for j in eachindex(epsis)
            @inbounds kernel[i,j] = imagtimeKernel1D_bos(taus[i], epsis[j], beta)
        end
    end

    heatmap(kernel)
    savefig("kernel.png")

    return kernel * Adisc
end


"""
Compute Partial correlator in imaginary time.
* isFermi: which of the D frequency grids where fermionic/bosonic
* taushift τ_j = β(j+taushift)/2^R
"""
function compute_imagtime_PartialCorrelator_1D(psf::CompressedPSF, isFermi::Vector{Bool}, taushift::Float64=0.5; cutoff=1e-15, kwargs_tci...)
    D = psf.dim
    R = nbits(psf)
    @assert length(isFermi)==D

    println(" -- Compressing kernels")
    kernels = Vector{MPS}(undef, D) 
    for i in eachindex(kernels)
        kernels[i] = compress_imagtimeKernel(R, psf.ωdisc[i], psf.beta, isFermi[i], taushift; tagnames=("tau$i","eps$i"), kwargs_tci...)
        @show rank(kernels[i])
    end

    k1 = kernels[1]
    TCI4Keldysh._adoptinds_by_tags!(k1, psf.Adisc_mps, "eps1",  "eps1", R)

    # view kernel as MPO
    k1MPO = MPO([k1[2*r-1]*k1[2*r] for r in 1:R])
    res = apply(k1MPO, psf.Adisc_mps; alg="densitymatrix", cutoff=cutoff, use_absolute_cutoff=true)

    return res
end

"""
Compute Partial correlator in imaginary time.
* isFermi: which of the D frequency grids where fermionic/bosonic
"""
function compute_imagtime_PartialCorrelator_2D(psf::CompressedPSF, isFermi::Vector{Bool}, taushift::Float64=0.5; cutoff=1e-10, kwargs_tci...)
    D = psf.dim
    R = nbits(psf)
    @assert length(isFermi)==D
    kwargs_automul = Dict(:alg=>"densitymatrix", :cutoff=>cutoff, :use_absolute_cutoff=>true)

    println(" -- Compressing kernels")
    kernels = Vector{MPS}(undef, D) 
    for i in eachindex(kernels)
        kernels[i] = compress_imagtimeKernel(R, psf.ωdisc[i], psf.beta, isFermi[i], taushift; tagnames=("tau$i","eps$i"), kwargs_tci...)
        @show rank(kernels[i])
    end

    k1 = kernels[1]
    TCI4Keldysh._adoptinds_by_tags!(k1, psf.Adisc_mps, "eps1",  "eps1", R)
    TCI4Keldysh.@TIME ak1 = Quantics.automul(
            k1,
            psf.Adisc_mps;
            tag_row = "tau1",
            tag_shared = "eps1",
            tag_col = "eps2",
            kwargs_automul...,
        ) "Contraction 1"
    mps_idx_info(ak1)

    k2 = kernels[2]
    TCI4Keldysh._adoptinds_by_tags!(k2, ak1, "eps2",  "eps2", R)
    TCI4Keldysh.@TIME res = Quantics.automul(
            ak1,
            k2;
            tag_row = "tau1",
            tag_shared = "eps2",
            tag_col = "tau2",
            kwargs_automul...,
        ) "Contraction 2"
    mps_idx_info(res)

    return res
end

"""
Compute Partial correlator in imaginary time.
* isFermi: which of the D frequency grids where fermionic/bosonic
"""
function compute_imagtime_PartialCorrelator_3D(psf::CompressedPSF, isFermi::Vector{Bool}; cutoff=1e-10, kwargs_tci...)
    D = psf.dim
    R = nbits(psf)
    @assert length(isFermi)==D
    kwargs_automul = Dict(:alg=>"densitymatrix", :cutoff=>cutoff, :use_absolute_cutoff=>true)

    println(" -- Compressing kernels")
    kernels = Vector{MPS}(undef, D) 
    for i in eachindex(kernels)
        kernels[i] = compress_imagtimeKernel(R, psf.ωdisc[i], psf.beta, isFermi[i]; tagnames=("tau$i","eps$i"), kwargs_tci...)
        @show rank(kernels[i])
    end

    @TIME begin
        k1 = TCI4Keldysh.add_dummy_dim(kernels[1]; pos=3, D_old=2)
        k2 = TCI4Keldysh.add_dummy_dim(kernels[2]; pos=3, D_old=2)
        k3 = TCI4Keldysh.add_dummy_dim(kernels[3]; pos=1, D_old=2)
    end "Adding dummy dimensions."

    TCI4Keldysh._adoptinds_by_tags!(k1, psf.Adisc_mps, "eps1",  "eps1", R)
    TCI4Keldysh._adoptinds_by_tags!(k1, psf.Adisc_mps, "dummy",  "eps3", R)
    TCI4Keldysh.@TIME ak1 = Quantics.automul(
            k1,
            psf.Adisc_mps;
            tag_row = "tau1",
            tag_shared = "eps1",
            tag_col = "eps2",
            kwargs_automul...,
        ) "Contraction 1"
    printstyled("\n  Rank after contraction 1 MPS/Kernel: $(rank(ak1))/$(rank(k1))\n"; color=:magenta)

    TCI4Keldysh._adoptinds_by_tags!(k2, ak1, "eps2",  "eps2", R)
    TCI4Keldysh._adoptinds_by_tags!(k2, ak1, "dummy",  "eps3", R)
    TCI4Keldysh.@TIME ak1k2 = Quantics.automul(
            ak1,
            k2;
            tag_row = "tau1",
            tag_shared = "eps2",
            tag_col = "tau2",
            kwargs_automul...,
        ) "Contraction 2"
    printstyled("\n  Rank after contraction 2 MPS/Kernel: $(rank(ak1k2))/$(rank(k2))\n"; color=:magenta)

    TCI4Keldysh._adoptinds_by_tags!(k3, ak1k2, "eps3",  "eps3", R)
    TCI4Keldysh._adoptinds_by_tags!(k3, ak1k2, "dummy",  "tau1", R)
    TCI4Keldysh.@TIME res = Quantics.automul(
            ak1k2,
            k3;
            tag_row = "tau2",
            tag_shared = "eps3",
            tag_col = "tau3",
            kwargs_automul...,
        ) "Contraction 3"
    printstyled("\n  Rank after contraction 3 MPS/Kernel: $(rank(res))/$(rank(k3))\n"; color=:magenta)

    return res
end

"""
Gp(iω_n)=∫_0^β dτ exp(iω_n*τ)Gp(τ)
"""
function imagtime_to_freq_explicit_1D(Gtau::Vector{T}, beta::Float64, fermi::Bool=false) where {T<:Number}
    N = length(Gtau)
    ftmat = Matrix{ComplexF64}(undef, (N,N))

    for j in 0:N-1
        phase = fermi ? exp(-im*π*j*(N+1)/N) : exp(-im*π*j)
        for i in 0:N-1
            # grid origin at -N/2 * π/β and 
            @inbounds ftmat[i+1,j+1] = exp(im*2*π*i*j/N) * phase
        end
    end

    return ftmat * Gtau * beta/N
end

"""
See how many points in numerical quadrature of a sine yield which accuracy
"""
function sine_quadrature()
    # integrate from 0 to 1
    up = 100.0
    f(x::Float64) = sin(π*x/up)
    exact_int = 2*up/π
    rel_errors = []
    Rs = 2:20
    for npoints in 2 .^ Rs
        int = 0.0
        step = up/npoints
        for i in 0:npoints-1
            int += f(step*i)
        end
        int *= step
        println("Riemann integral for $npoints points: $int")
        rel_err = abs(1.0 - int/exact_int)
        push!(rel_errors, rel_err)
        println("1-int/exact_int: $(rel_err)")
    end
    p = plot(Rs, rel_errors; yscale=:log10)
    display(p)
end

"""
Transform MPS from imaginary times to frequencies: Gp(iω_n)=∫_0^β dτ exp(iω_n*τ)Gp(τ)
    * taushift τ_j = β(j+taushift)/2^R
CAREFUL: Fouriertrafo reverses bits!
"""
function imagtime_to_freq(Gtau::ImagtimePartialCorrelator, taushift::Float64=0.5; ftkwargs...)
    R = nbits(Gtau)
    N = 2^R
    D = Gtau.dim
    Gp = deepcopy(Gtau.Gtau)
    # Quantics.fouriertransform already has prefactor 1/√N
    dtau = Gtau.beta / sqrt(N)

    # determine sites corresponding to each variable
    tausites = Vector{Vector{Index}}(undef, D)
    sinds = siteinds(Gp)
    tag = "tau"
    for i in 1:D
        tausites[i] = [sinds[findfirst(x -> hastags(x, "$(tag)$i=$n"), sinds)] for n in 1:R]
    end

    omsites = Vector{Vector{Index}}(undef, D)
    for i in 1:D
        # TODO: reverse these
        # omsites[i] = reverse(quanticssites(R, "ω$i"))
        omsites[i] = quanticssites(R, "ω$i")
    end
    
    for i in eachindex(Gtau.isFermi)
        # perform phase shift to get frequency grids origins right
        # copies MPS...
        # theta_tau = Gtau.isFermi[i] ? π*(N+1)/N : -Float64(π)
        theta_tau = Gtau.isFermi[i] ? π*(-N+1)/N : -Float64(π)
        Gp = Quantics.phase_rotation(Gp, theta_tau; targetsites=tausites[i])
        # Fouriertrafo
        Gp = dtau * Quantics.fouriertransform(Gp; sign=1, sitessrc=tausites[i], sitesdst=omsites[i], ftkwargs...)
        # ω-phase rotation due to taushift
        theta_ω = 2π*taushift/N
        Gp = Quantics.phase_rotation(Gp, theta_ω ; targetsites=omsites[i])
        # global phase rotation due to taushift
        theta_global = theta_tau * taushift
        Gp *= exp(im*theta_global)
    end

    return Gp
end

"""
Test computation of a partial correlator for a PSF with a single peak A(ϵ) = A_0 * δ(ϵ-ϵ_0)
"""
function test_1peak_imagtime_PartialCorrelator(R_tau::Int, beta::Float64)

    ITensors.disable_warn_order()

    R_om = 7
    Nom = 2^R_om
    Ntau = 2^R_tau
    # NOTE: maximum error in frequency domain is pretty much linear in beta
    eps0 = [0.1]
    # eps0 = collect(range(0.1, 1.0, 10))
    A0 = 1.0

    isFermi = [false]
    # reference data
    bosgrid = MF_grid(1/beta, div(Nom, 2), isFermi[1])
    taushift = 0.9
    # taurange = 0:(Ntau-1)
    taurange = 1:div(Ntau,2^12):Ntau
    taugrid = [((i-1)+taushift)*beta/Ntau for i in taurange]
    # exact result (tau)
    Gtau_exact = [sum([A0*imagtimeKernel1D_bos(tau, e, beta) for e in eps0]) for tau in taugrid]
    # exact result (frequency)
    Gom_exact = [sum([A0/(im*ω - e) for e in eps0]) for ω in bosgrid]

    # TCI computation
    tol = 1e-12
    psf = get_dummy_CompressedPSF(R_tau, beta, eps0, A0; tolerance=tol)

    Gtau = compute_imagtime_PartialCorrelator_1D(psf, isFermi, taushift; tolerance=tol)
    Gtau_fat = TCI4Keldysh.MPS_to_fatTensor(Gtau, [taurange]; tags=("tau1",), reverse_idx=false)

    plot(taugrid, Gtau_fat)
    savefig("Gtau.png")

    @show norm(Gtau_fat - Gtau_exact)
    @show maximum(Gtau_fat - Gtau_exact)

    Gtau = ImagtimePartialCorrelator(Gtau, psf.beta, isFermi)
    Gom = imagtime_to_freq(Gtau, taushift; cutoff_MPO=1e-20, cutoff=1e-20, use_absolute_cutoff=true)
    omslice = 2^(R_tau-1) - 2^(R_om-1)+1 : 2^(R_tau-1) + 2^(R_om-1)+1
    Gom_fat = TCI4Keldysh.MPS_to_fatTensor(Gom, [omslice]; tags=("ω1",), reverse_idx=true)

    omplot = plot()
    plot!(omplot, bosgrid, real.(Gom_exact - Gom_fat); label="real")
    plot!(omplot, bosgrid, imag.(Gom_exact - Gom_fat); label="imag")
    savefig(omplot, "Gom_vs_exact.png")
    diff = abs.(Gom_exact - Gom_fat)
    @show argmax(diff)
    maxdiff = maximum(diff)
    # NOTE: std is practically 0 !!!
    std = sqrt(sum(diff .- sum(diff)/length(diff)) .^ 2)/length(diff)
    printstyled("  -- Max error: $maxdiff\n"; color=:magenta)
    printstyled("  -- log inverse max error: $(log10(1/maxdiff))\n"; color=:magenta)
    printstyled("  -- Std of error: $std\n"; color=:magenta)
    plot(bosgrid, diff; label="error", yscale=:log10)
    savefig("diff_peak.png")

    # return errors
    return (maxdiff, std)
end

function test_imagtime_PartialCorrelator(;npt=2, tolerance=1.e-12, perm_idx=1, beta=1000.0, R=18, cutoff=1.e-20)

    ITensors.disable_warn_order()

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

    if npt==2

        diffplot = plot()
        tfont = 12
        titfont = 16
        gfont = 16
        lfont = 12
        maxerrplot = plot(;guidefontsize=gfont, titlefontsize=titfont, tickfontsize=tfont, legendfontsize=lfont)
        Rs = 12:3:42
        # Rs = [12]
        betas = [1.e2, 1.e3, 1.e4]
        # betas = [1.e3]
        for beta_loc in betas
            max_errors = Float64[]
            for R in Rs
                TCI4Keldysh.@TIME psf = CompressedPSF(beta_loc, path, Ops, spin; nested_ωdisc=false, pad_R = R, tolerance=tolerance) "Compressing Adisc"

                isFermi = [false]
                Gp = compute_imagtime_PartialCorrelator_1D(psf, isFermi; tolerance=tolerance)

                # test_kernel_compression(psf; plot_kernel=true, tolerance=tolerance)
                # Gp_exp = imagtime_PartialCorrelator_1D_explicit(length(Gp), beta_loc, path, Ops, spin)
                # Gp_small = TCI4Keldysh.MPS_to_fatTensor(Gp; tags=("tau1",))

                # # plot
                # slice = 1:length(Gp_exp)
                # @show size(Gp_small)
                # @show size(Gp_exp)
                # plot(slice, real.(Gp_small)[slice]; label="TCI")
                # plot!(slice, real.(Gp_exp); label="conventional")
                # @show maximum(abs.(Gp_small - Gp_exp))
                # savefig("imagtime_correlator1D.png")

                # back to frequency
                printstyled(" -- Fouriertrafo\n"; color=:blue)
                Gtau = ImagtimePartialCorrelator(Gp, psf.beta, isFermi)
                tauslice = (1:div(2^R, 2^12):2^R)
                Gtau_small = TCI4Keldysh.MPS_to_fatTensor(Gtau.Gtau, [tauslice]; reverse_idx=true, tags=("tau1",))
                tauplot = plot()
                plot!(tauplot, collect(tauslice) ./ 2^R, real.(Gtau_small); label="real")
                plot!(tauplot, collect(tauslice) ./ 2^R, imag.(Gtau_small); label="imag")
                title!(tauplot, "Gτ")
                savefig("Gtau.png")
                Gom = imagtime_to_freq(Gtau; cutoff_MPO=1e-20, cutoff=cutoff, use_absolute_cutoff=true)

                # @time Gom_exp = imagtime_to_freq_explicit_1D(Gp_exp, beta_loc, isFermi[begin])
                # Gom_fat = TCI4Keldysh.MPS_to_fatTensor(Gom; tags=("ω1",))
                # @show absmax(real.(Gom_fat - Gom_exp))
                # @show absmax(imag.(Gom_fat - Gom_exp))
                # @show absmax(Gom_fat - Gom_exp)
                # @show argmax(abs.(Gom_fat - Gom_exp))
                # exp_plot = plot()
                # exp_slice1 = 2^(R-1)-64:2^(R-1)+64
                # exp_slice2 = 2^(R-1)-64:2^(R-1)+64
                # # plot!(exp_plot, exp_slice1, real.(Gom_fat[exp_slice1]))
                # # plot!(exp_plot, exp_slice2, real.(Gom_exp[exp_slice2]))
                # plot!(exp_plot, exp_slice1, abs.(Gom_fat[exp_slice1] - Gom_exp[exp_slice2]))
                # savefig("exp_plot.png")

                # compare to conventional computation
                ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
                R_ext = 7
                ωconvMat_sum = cumsum(ωconvMat[perm[1:(npt-1)],:]; dims=1)
                Gp = PartialCorrelator_reg(npt-1, 1/beta_loc, R_ext, path, Ops, ωconvMat_sum)
                data_unrotated = contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)
                # no anomalous part present
                plot(1:length(data_unrotated), real.(data_unrotated))
                savefig("freq_correlatorREF1D.png")

                diffslice = 2^(nbits(psf)-1) - 2^(R_ext-1) + 1:2^(nbits(psf)-1) + 2^(R_ext-1) + 1
                Gom_small = TCI4Keldysh.MPS_to_fatTensor(Gom, [diffslice]; reverse_idx=true, tags=("ω1",))
                @show size(Gom_small)
                diff = abs.(data_unrotated - Gom_small)
                @show argmax(abs.(data_unrotated))
                @show argmax(abs.(Gom_small))

                plot!(diffplot, 1:length(diff), abs.(real.(data_unrotated .- Gom_small)); label="Rτ=$R, real", yscale=:log10)
                plot!(diffplot, 1:length(diff), abs.(imag.(data_unrotated .- Gom_small)); label="Rτ=$R, imag", yscale=:log10)
                xlabel!(diffplot, "ω")
                ylabel!(diffplot, "error")
                savefig(diffplot, "diff1D.png")

                foo = plot()
                plot!(foo, 1:length(data_unrotated), real.(data_unrotated); label="Rτ=$R, real ref", yscale=:identity)
                plot!(foo, 1:length(data_unrotated), imag.(data_unrotated); label="Rτ=$R, imag ref", yscale=:identity)
                plot!(foo, 1:length(data_unrotated), real.(Gom_small); label="Rτ=$R, real Gom", yscale=:identity)
                plot!(foo, 1:length(data_unrotated), imag.(Gom_small); label="Rτ=$R, imag Gom", yscale=:identity)
                savefig("foo.png")

                @show maximum(diff)
                @show abs(data_unrotated[argmax(diff)]) 
                @show argmax(diff)
                # push!(max_errors, maximum(diff) / abs(data_unrotated[argmax(diff)]))
                # push!(max_errors, maximum(diff) / abs(data_unrotated[argmax(diff)]))
                push!(max_errors, maximum(abs.(diff ./ data_unrotated)))
            end
            plot!(maxerrplot, Rs, max_errors; label="β=$beta_loc", marker=:diamond, yscale=:log10)
        end
        xlabel!(maxerrplot, "Rτ")
        yticks!(maxerrplot, 10.0 .^ (round(Int,log10(tolerance))-1:-2))
        ylabel!(maxerrplot, "Max. rel. error")
        # title!(maxerrplot, "2pt G(ω) via imaginary time TCI")
        savefig(maxerrplot, "maxdiff_vs_R_beta=$(beta)_tol=$(round(Int,log10(tolerance))).png")

    elseif npt==3
        taushift = 0.5
        TCI4Keldysh.@TIME psf = CompressedPSF(beta, path, Ops, spin; nested_ωdisc=false, pad_R=R, tolerance=tolerance) "Compressing Adisc"

        isFermi = [false, true]
        Gp = compute_imagtime_PartialCorrelator_2D(psf, isFermi, taushift; tolerance=tolerance)
        mps_idx_info(Gp)
        @show rank(Gp)

        # back to frequency
        printstyled(" -- Fouriertrafo\n"; color=:blue)
        Gtau = ImagtimePartialCorrelator(Gp, psf.beta, isFermi)
        Gom = imagtime_to_freq(Gtau, taushift; cutoff_MPO=1e-14, cutoff=cutoff, use_absolute_cutoff=true)
        mps_idx_info(Gom)
        @show rank(Gom)

        # # plot
        # Gom_fat = TCI4Keldysh.MPS_to_fatTensor(Gom; tags=("ω1","ω2"))
        # heatmap(log10.(abs.(Gom_fat)))
        # savefig("freq_correlator2D.png")

        # compare to conventional computation
        ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
        R_ext = 7
        ωconvMat_sum = cumsum(ωconvMat[perm[1:(npt-1)],:]; dims=1)
        display(ωconvMat_sum) 
        Gp = PartialCorrelator_reg(npt-1, 1/beta, R_ext, path, Ops, ωconvMat_sum)
        data_unrotated = contract_1D_Kernels_w_Adisc_mp(Gp.tucker.legs, Gp.tucker.center)
        heatmap(log10.(abs.(data_unrotated)))
        savefig("freq_correlatorREF2D.png")

        @show argmax(abs.(data_unrotated))
        diffslice = [2^(R-1)+1 - 2^(R_ext-1):2^(R-1)+1 + 2^(R_ext-1), 2^(R-1)+1 - 2^(R_ext-1):2^(R-1) + 2^(R_ext-1)]
        # TODO: WHY NEED TO REVERSE
        Gom_small = transpose(TCI4Keldysh.MPS_to_fatTensor(Gom, reverse(diffslice); reverse_idx=true, tags=("ω1","ω2")))
        heatmap(log10.(abs.(Gom_small)))
        savefig("GomSmall2D.png")
        @show argmax(abs.(Gom_small))
        @show size(Gom_small)
        @show size(data_unrotated)

        # diff = data_unrotated - Gom_small[diffslice...]
        diff = data_unrotated - Gom_small
        heatmap(log10.(abs.(diff)))
        savefig("diff2D.png")
        @show maximum(abs.(diff))

    elseif npt==4
        TCI4Keldysh.@TIME psf = CompressedPSF(beta, path, Ops, spin; nested_ωdisc=false, pad_R=R, tolerance=tolerance) "Compressing Adisc"
        isFermi = [false, true, true]
        TCI4Keldysh.@TIME Gp = compute_imagtime_PartialCorrelator_3D(psf, isFermi; cutoff=cutoff, tolerance=tolerance) "Imag. Time Correlator"
        # mps_idx_info(Gp)
        @show rank(Gp)

        # # plot
        # Gp_fat = TCI4Keldysh.MPS_to_fatTensor(Gp; tags=("tau1","tau2","tau3"))
        # @show size(Gp_fat)
        # heatmap(log10.(abs.(Gp_fat[2^(R-1),:,:])))
        # savefig("imagtime_correlator3D.png")

        # back to frequency
        printstyled(" -- Fouriertrafo\n"; color=:blue)
        Gtau = ImagtimePartialCorrelator(Gp, psf.beta, isFermi)
        TCI4Keldysh.@TIME Gom = imagtime_to_freq(Gtau; cutoff_MPO=1e-15, cutoff=cutoff, use_absolute_cutoff=true) "Fourier Transform"
        ITensors.truncate!(Gom; cutoff=1.e-15, use_absolute_cutoff=true)
        @show rank(Gom)

        # # plot
        # Gom_fat = TCI4Keldysh.MPS_to_fatTensor(Gom; tags=("ω1","ω2","ω3"))
        # heatmap(log10.(abs.(Gom_fat[2^(R-1),:,:])))
        # savefig("freq_correlator3D.png")

    end
end