using HDF5
using QuanticsTCI
using TCI4Keldysh
using Plots
import TensorCrossInterpolation as TCI

function grid_R(N::Int)
    return trunc(Int, log2(N))
end

function grid_R(arr::AbstractArray)
    Rs =  ntuple(i -> grid_R(size(arr,i)), ndims(arr))
    return minimum(collect(Rs))
end

function quanticsslice(arr::AbstractArray)
    R = grid_R(arr)
    return ntuple(_->1:2^R, ndims(arr))
end

function gridweights(grid::Vector{Float64})
    dg = diff(grid)
    return vcat([dg[1]/2], dg[1:end-1]./2 + dg[2:end]./2, [dg[end]/2])
end

"""
Compute
∫dν1 Γ(ω,ν1,ν)*Γ(ω,ν',ν1) = B(ω,ν,ν')
* ws: integration weights
"""
function dummy_bubble(gamcore::Array{ComplexF64,3}, ws::Vector{Float64})
    @assert length(ws)==size(gamcore,2)
    @assert length(ws)==size(gamcore,3)

    gamt = permutedims(gamcore, (2,3,1))
    B = Array{ComplexF64}(undef, size(gamt))
    @views for iw in axes(B,3)
        @show iw
        MR = gamt[:,:,iw] .* ws
        ML = gamt[:,:,iw]
        B[:,:,iw] .= ML * transpose(MR)
    end
    # (ν',ν,ω) -> (ω,ν,ν')
    return permutedims(B, (3,2,1))
end

"""
Produce multidimensional array of gridweights
"""
function gridweights(grid::Vector{Vector{Float64}})
    ng = length(grid)
    ngs = length.(grid)
    gws = [reshape(gridweights(grid[i]), ntuple(j -> ifelse(j==i,ngs[i],1), ng)) for i in eachindex(grid)]
    res = ones(Float64, ngs...)
    for i in 1:ng
        res .*= gws[i]
    end
    return res
end

"""
To check whether vertex on two different grids gives comparable
accuracy upon integration.
"""
function assess_grid_quality(fname::String, reffile::String)
    gamcore = h5read(fname, "V_KF")
    # last grid corresponds to first dimension due to different layout in Python vs Julia!
    grid3 = h5read(fname, "grid1")
    grid2 = h5read(fname, "grid2")
    grid1 = h5read(fname, "grid3")

    dg1 = grid1[end]-grid1[1]
    dg2 = grid2[end]-grid2[1]
    dg3 = grid3[end]-grid3[1]
    @show size(gamcore)

    iKtuple = TCI4Keldysh.KF_idx(2,3)
    ref = h5read(reffile, "V_KF")[:,:,:,iKtuple...]
    rg1 = h5read(reffile, "omgrid1")
    rg2 = h5read(reffile, "omgrid2")
    rg3 = h5read(reffile, "omgrid3")

    dwg1 = rg1[end]-rg1[1]
    dwg2 = rg2[end]-rg2[1]
    dwg3 = rg3[end]-rg3[1]
    @show size(ref)

    rescale = dwg1*dwg2*dwg3 / (dg1*dg2*dg3)

    wg = gridweights([grid1,grid2,grid3])
    rwg = gridweights([rg1, rg2, rg3])

    for p in [0.5, 1.0, 2.0]
        n = sum(abs.(gamcore) .^ p .* wg) ^ (1/p) * rescale^(1/p)
        rn = sum(abs.(ref) .^ p .* rwg) ^ (1/p)
        println("==== $p-norms:")
        println("    ref $rn")
        println("    gamcore $n")
        println("    Δ=$(abs(rn-n))")
        println("    Δ/ref=$(abs(rn-n)/rn)")
    end
end

function compress_vertex(fname::String, weightmode="BSE"; tcikwargs...)
    # load data
    gamcore = h5read(fname, "V_KF")
    grid1 = h5read(fname, "grid1")
    grid2 = h5read(fname, "grid2")
    grid3 = h5read(fname, "grid3")

    if weightmode=="BSE"
        wg1 = gridweights(grid2)
        wg2 = gridweights(grid3)
        wgmat = transpose(sqrt.(wg1)) * sqrt.(wg2)
        for i in axes(gamcore,1)
            gamcore[i,:,:] .*= wgmat
        end
    elseif weightmode=="L1"
        wg = gridweights([grid1,grid2,grid3])
        gamcore .*= wg
    elseif weightmode=="L2"
        wg = gridweights([grid1,grid2,grid3])
        gamcore .*= sqrt.(wg)
    end

    @show size(gamcore)
    R = grid_R(gamcore)
    @show R
    display(grid1[2^(R-1)-10:2^(R-1)+10])

    npivot = 9
    pivot_step = div(2^R, npivot-1)
    initpivots_ω = TCI4Keldysh.initpivots_general(Tuple(length.(quanticsslice(gamcore))), npivot, pivot_step)
    # initpivots_ω = [[1,1,1]]
    @show length(initpivots_ω)
    qtt, _, _ = quanticscrossinterpolate(gamcore[quanticsslice(gamcore)...], initpivots_ω; tcikwargs...)
    println("  Bonddims: $(TCI.linkdims(qtt.tci))")
    println("==== Rank: $(TCI.rank(qtt.tci))")

    # recover fat tensor
    qttfat = if tcikwargs[:unfoldingscheme]==:interleaved
        TCI4Keldysh.qinterleaved_fattensor_to_regular(
            TCI4Keldysh.qtt_to_fattensor(qtt.tci.sitetensors), grid_R(gamcore)
            )
    else
        # fused
        qttfat_ = TCI4Keldysh.qtt_to_fattensor(qtt.tci.sitetensors)
        R = ndims(qttfat_)
        @assert R==grid_R(gamcore)
        qtt_interl = reshape(qttfat_, ntuple(_->2, 3*R))
        TCI4Keldysh.qinterleaved_fattensor_to_regular(
            qtt_interl, R
        )
    end
    refmax = maximum(abs.(gamcore))
    diff = abs.(qttfat .- gamcore[quanticsslice(gamcore)...]) / refmax
    lmaxref = log10(maximum(abs.(gamcore)))
    scfun(x) = log10(abs(x))
    tol = tcikwargs[:tolerance]
    logtol = log10(tol)
    plotslice = (2^(R-1)+1, :, :)
    heatmap(scfun.(qttfat[plotslice...]); clim=(lmaxref+logtol-1, lmaxref))
    pref = split(fname, '.')[begin]
    savefig("$(pref)tci.pdf")
    heatmap(scfun.(gamcore[quanticsslice(gamcore)...][plotslice...]); clim=(lmaxref+logtol-1, lmaxref))
    savefig("$(pref)ref.pdf")
    heatmap(scfun.(diff[plotslice...]); clim=(logtol-1, -1))
    savefig("$(pref)diff.pdf")
    @show argmax(diff)
    @show maximum(diff)

end

compress_vertex("V_KF_lorentzR8.h5", "BSE"; tolerance=1.e-2, unfoldingscheme=:fused)
# assess_grid_quality("V_KF_logR7.h5", "V_KF_conventional/V_KF_p_R=8.h5")