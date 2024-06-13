using TCI4Keldysh
using QuanticsTCI
using Plots
using ITensors

import TensorCrossInterpolation as TCI
import QuanticsGrids as QG

function rank(qtt::QuanticsTCI.QuanticsTensorCI2)
    return maximum(TCI.linkdims(qtt.tci))
end

function rank(mps::MPS)
    return maximum(dim.(linkinds(mps)))
end

"""
Eval full matsubara kernel on 2 x D frequencies
"""
function eval_matsubara_kernel(oms::Vector{Float64}, omprimes::Vector{Float64}, beta::Float64)
    if TCI4Keldysh.DEBUG()
        @assert length(oms)==length(omprimes)
    end
    bos_zero = findfirst(i -> abs(i)<=1e-10, oms)
    # anomalous term
    if !isnothing(bos_zero) && abs(omprimes[bos_zero])<=1e-10
        printstyled("anomalous\n", color=:red)
        return eval_ano_matsubara_kernel(oms, omprimes, bos_zero, beta)
    else
        return reduce(*, 1 ./ (im*oms .- omprimes); init=1.0)
    end
end

"""
Eval regular matsubara kernel on 2 x D frequencies. Return 0 in anomalous case.
"""
function eval_reg_matsubara_kernel(oms::Vector{Float64}, omprimes::Vector{Float64}, ::Float64)
    if TCI4Keldysh.DEBUG()
        @assert length(oms)==length(omprimes)
    end
    bos_zero = findfirst(i -> abs(i)<=1e-10, oms)
    # anomalous term
    if !isnothing(bos_zero) && abs(omprimes[bos_zero])<=1e-10
        return 0.0
    else
        return reduce(*, 1 ./ (im*oms .- omprimes); init=1.0)
    end
end


"""
Eval anomalous part of full matsubara kernel on 2 x D frequencies
"""
function eval_ano_matsubara_kernel(oms::Vector{Float64}, omprimes::Vector{Float64}, bos_zero::Int, beta::Float64)
    # gives a factor one in product instead of divergence
    product = 1.0
    sum = 0.0
    for i in eachindex(oms)
        if i==bos_zero
            continue
        end
        Ominv = 1/(im*oms[i] - omprimes[i])
        sum += Ominv
        product *= Ominv
    end
    return -0.5*(beta + sum)*product
end

"""
Eval prefactor of anomalous part of full matsubara kernel on 2 x D frequencies.
Used to see how compressible this is.
"""
function eval_ano_prefac(oms::Vector{Float64}, omprimes::Vector{Float64}, bos_zero::Int, beta::Float64)
    # gives a factor one in product instead of divergence
    sum = 0.0
    for i in eachindex(oms)
        if i==bos_zero
            continue
        end
        Ominv = 1/(im*oms[i] - omprimes[i])
        sum += Ominv
    end
    return -0.5*(beta + sum)
end

function ff(n::Int, beta::Float64)
    return (2*n+1)*π/beta
end

function bf(n::Int, beta::Float64)
    return 2*n*π/beta
end

function test_eval_matsubara_kernel()
    oms = [0.0, 2.0]
    omprimes = [0.0, 3.0]
    beta = 0.2
    @assert isapprox(eval_matsubara_kernel(oms, omprimes, beta), eval_ano_matsubara_kernel(oms, omprimes, 1, beta); atol=1e-10)
end

function test_eval_ano_matsubara_kernel()
    oms = [0.0, 2.0]
    omprimes = [0.0, 1.0]
    beta = 2.0
    comp = -0.5*(2.0 + 1/(2.0*im-1.0)) * 1/(2.0*im-1.0)
    @assert isapprox(eval_ano_matsubara_kernel(oms, omprimes, 1, beta), comp; atol=1e-10)
end

function plot_kernel(xfactor=1.0)
    R = 7    
    beta = 1.0
    ferval = [ff(i, xfactor*beta) for i in -2^(R-1):2^(R-1)-1]
    bosval = [bf(i, beta) for i in -2^(R-1):2^(R-1)-1]
    kernelfunc = eval_matsubara_kernel
    kernelval = [kernelfunc([b], [f], beta) for f in ferval, b in bosval]
    heatmap(ferval, bosval, log.(abs.(kernelval)))

    savefig("kernel_heatmap.png")
end

function plot_anomalous(prefac_only=false)
    R = 7    
    beta = 1.0
    b1 = bf(0, beta)
    b2 = bf(0, beta)
    ferval1 = [ff(i, beta) for i in -2^(R-1):2^(R-1)-1]
    ferval2 = copy(ferval1)
    kernelfun = prefac_only ? eval_ano_prefac : eval_ano_matsubara_kernel
    kernelval = [kernelfun([b1, ff1], [b2, ff1p], 1, beta) for ff1 in ferval1, ff1p in ferval2]
    heatmap(ferval2, ferval1, log.(abs.(kernelval)))

    savefig("ano_kernel_heatmap.png")
end


"""
kerneltype=:full or :ano
"""
function compress_matsubara_kernel(dim, beta=1.0; kerneltype=:full, tolerance=1e-10, R=7)
    
    gmin_bos = -2^(R-1)
    gmin_fer = -2^(R-1)

    # 1D
    if dim==1
        ombos = QG.InherentDiscreteGrid{2}(R, ntuple(i -> gmin_bos, 2); unfoldingscheme=:interleaved)
        omfer = QG.InherentDiscreteGrid{2}(R, ntuple(i -> gmin_fer, 2); unfoldingscheme=:interleaved)
        
        xfactor = 1.0
        function kernel_boson(om1::Int, om1p::Int)
            return eval_matsubara_kernel([bf(om1,xfactor*beta)], [bf(om1p,xfactor*beta)], beta)
        end
       
        function kernel_fermion(om1::Int, om1p::Int)
            return eval_matsubara_kernel([ff(om1,beta)], [ff(om1p,beta)], beta)
        end

        @time qttbos, _, _ = quanticscrossinterpolate(ComplexF64, kernel_boson, ombos; tolerance=tolerance)
        @time qttfer, _, _ = quanticscrossinterpolate(ComplexF64, kernel_fermion, omfer; tolerance=tolerance)
        @show length(qttbos.tci)
        @show length(qttfer.tci)
        @show TCI.linkdims(qttbos.tci)
        @show TCI.linkdims(qttfer.tci)
        return (qttbos, qttfer)
    end

    # 2D
    if dim==2
        omgrid = QG.InherentDiscreteGrid{4}(R, (-2^(R-1), -2^(R-1)-1, -2^(R-1), -2^(R-1)-1); unfoldingscheme=:interleaved)
        
        kernelfunc = eval_matsubara_kernel
        if kerneltype == :ano
            kernelfunc = (om, omp, beta) -> eval_ano_matsubara_kernel(om, omp,  1, beta)
        elseif kerneltype == :anoprefac
            kernelfunc = (om, omp, beta) -> eval_ano_prefac(om, omp,  1, beta)
        end

        function kernel(om1::Int, om1p::Int, om2::Int, om2p::Int)
            return kernelfunc([bf(om1,beta), ff(om2,beta)], [bf(om1p,beta), ff(om2p,beta)], beta)
        end

        @time qtt, _, _ = quanticscrossinterpolate(ComplexF64, kernel, omgrid; tolerance=tolerance)
        @show length(qtt.tci)
        @show TCI.linkdims(qtt.tci)
        return qtt
    end

    # 3D
    if dim==3
        omgrid3d = QG.InherentDiscreteGrid{6}(R, ntuple(i -> (i%2==0 ? -2^(R-1) : -2^(R-1)-1), 6); unfoldingscheme=:interleaved)

        kernelfunc = eval_matsubara_kernel
        if kerneltype == :ano
            kernelfunc = (om, omp, beta) -> eval_ano_matsubara_kernel(om, omp,  1, beta)
        elseif kerneltype == :anoprefac
            kernelfunc = (om, omp, beta) -> eval_ano_prefac(om, omp,  1, beta)
        end

        function kernel3d(om1::Int, om1p::Int, om2::Int, om2p::Int, om3::Int, om3p::Int)
            return kernelfunc([bf(om1,beta), ff(om2,beta), ff(om3,beta)], [bf(om1p,beta), ff(om2p,beta), ff(om3p,beta)], beta)
        end

        qtt3d, _, _ = quanticscrossinterpolate(ComplexF64, kernel3d, omgrid3d; tolerance=tolerance)
        @show length(qtt3d.tci)
        @show TCI.linkdims(qtt3d.tci)
        return qtt3d
    end

end

"""
Perform svd-cutoff on kernel; yields significantly lower rank for anomalous part, even for cutoff=1e-14
"""
function truncate_matsubara_kernel(dimension, beta=1.0; kwargs...)

    @assert dimension>1 "dim must be >1"
    qtt = compress_matsubara_kernel(dimension, beta; kwargs...)

    # to MPS
    qtt_mps = TCI4Keldysh.QTCItoMPS(qtt, ("ω1", "eps1", "ω2", "eps2", "ω3", "eps3"))

    println("Rank before truncate: $(rank(qtt))")
    ITensors.truncate!(qtt_mps; cutoff=kwargs[:tolerance]*1e-2, use_absolute_cutoff=true)
    @show typeof(qtt_mps)
    println("Rank after truncate: $(rank(qtt_mps))")

    return qtt_mps
end

# compile
compress_matsubara_kernel(1)

# # run
# printstyled("===== Run\n"; color=:blue)
# for beta in 10.0 .^ (-3:2)
#     printstyled("-- D=1\n"; color=:blue)
#     compress_matsubara_kernel(1, beta)
#     # printstyled("-- D=2\n"; color=:blue)
#     # compress_matsubara_kernel(2, beta)
# end

# println("========== 2D ano")
# @time compress_matsubara_kernel(2, 1.e0; kerneltype=:ano)
# println("========== 3D anoprefac")
# @time compress_matsubara_kernel(3, 1.e0; kerneltype=:anoprefac)
# println("========== 3D ano")
# @time compress_matsubara_kernel(3, 1.e0; kerneltype=:ano)

# # see rank scaling with tolerance
# ranks = Int[]
# tols = 10.0 .^ (-3:-1:-11)
# for tol in tols
#     push!(ranks, rank(compress_matsubara_kernel(3, 1e0; kerneltype=:ano, tolerance=tol, R=7)))
# end
# plot(tols, ranks; xscale=:log10)
# savefig("ano_rank_vs_tol.png")

# see rank scaling with tolerance
ranks = Int[]
Rs = 5:10
for R in Rs
    push!(ranks, rank(truncate_matsubara_kernel(3, 1e0; kerneltype=:ano, tolerance=1e-10, R=R)))
end
plot(Rs, ranks)
savefig("ano_rank_vs_R.png")