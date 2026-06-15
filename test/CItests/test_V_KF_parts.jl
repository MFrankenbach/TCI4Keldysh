using TCI4Keldysh
using PythonCall
using PythonPlot
using HDF5
using MAT

include(joinpath(dirname(Base.current_project()), "scripts", "plot_utils.jl"))

# load data
juliadatapath = joinpath(dirname(Base.current_project()), "test", "CItests", "reference_data", "keldyshfull_MuNRGcompare_updown_NEW")
# MuNRG data
channel = "t"
refdatapath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50", "V_KF_$(TCI4Keldysh.channel_translate(channel))")


function plot_alliK(data::Array{T,7}, slice_dim::Int, slice_idx::Int, scfun::Function=abs) where {T<:Number}
    fig, axs = subplots(4,4; figsize=(12,12))
    slice_tuple = ntuple(i -> ifelse(i==slice_dim, slice_idx, Colon()), 3)
    for ir in 1:4
        for ic in 1:4
            iK = 4*(ir-1) + ic
            axs[ir-1,ic-1].imshow(scfun.(data[slice_tuple..., TCI4Keldysh.KF_idx(iK,3)...]), cmap="viridis", interpolation="nearest")
            annotate_topleft(axs[ir-1,ic-1], "$(TCI4Keldysh.KF_idx(iK,3))"; color="white")  
        end
    end
    return fig, axs
end

function compare_alliK1D(
        data::Array{T,7}, dataref::Array{T,7},
        grid::Vector{Float64}, grid_ref::Vector{Float64},
        slice_tuple, slice_tupleref
        ) where {T}
    fig, axs = subplots(4,4; figsize=(12,12))
    for ir in 1:4
        for ic in 1:4
            iK = 4*(ir-1) + ic
            # reference
            axs[ir-1,ic-1].plot(grid_ref, real.(dataref[slice_tupleref..., TCI4Keldysh.KF_idx(iK,3)...]); color="cyan")
            axs[ir-1,ic-1].plot(grid_ref, imag.(dataref[slice_tupleref..., TCI4Keldysh.KF_idx(iK,3)...]); color="red")
            # data
            axs[ir-1,ic-1].plot(grid, real.(data[slice_tuple..., TCI4Keldysh.KF_idx(iK,3)...]); color="black", linestyle="dotted")
            axs[ir-1,ic-1].plot(grid, imag.(data[slice_tuple..., TCI4Keldysh.KF_idx(iK,3)...]); color="black", linestyle="dotted")
            annotate_topleft(axs[ir-1,ic-1], "$(TCI4Keldysh.KF_idx(iK,3))"; color="black")  
        end
    end
    return fig, axs
end


function compare_alliK1D(
    # data(ref): 4 Keldysh + 1 frequency index
        data::Array{T,5}, dataref::Array{T,5},
        grid::Vector{Float64}, grid_ref::Vector{Float64},
        slice_tuple=Colon(), slice_tupleref=Colon()
        ) where {T}
    fig, axs = subplots(4,4; figsize=(12,12))
    for ir in 1:4
        for ic in 1:4
            iK = 4*(ir-1) + ic
            # reference
            axs[ir-1,ic-1].plot(grid_ref, real.(dataref[slice_tupleref, TCI4Keldysh.KF_idx(iK,3)...]); color="cyan")
            axs[ir-1,ic-1].plot(grid_ref, imag.(dataref[slice_tupleref, TCI4Keldysh.KF_idx(iK,3)...]); color="red")
            # data
            axs[ir-1,ic-1].plot(grid, real.(data[slice_tuple, TCI4Keldysh.KF_idx(iK,3)...]); color="black", linestyle="dotted")
            axs[ir-1,ic-1].plot(grid, imag.(data[slice_tuple, TCI4Keldysh.KF_idx(iK,3)...]); color="black", linestyle="dotted")
            annotate_topleft(axs[ir-1,ic-1], "$(TCI4Keldysh.KF_idx(iK,3))"; color="black")  
        end
    end
    return fig, axs
end

function adjust_matdata(data::Array)
    out = permutedims(data, (3,1,2,4,5,6,7))
    out = reverse(out; dims=(1,2,3))
    return out
end

function _readMAT!(filename; flavor_idx::Int=1)
    out = nothing
    matopen(joinpath(refdatapath, filename)) do f
        CFdat = read(f, "CFdat")
        out = CFdat["Ggrid"][flavor_idx]
        out = adjust_matdata(out)
    end
    return out
end

function main(;)

    flavor_idx = 2

    # CORE VERTEX
    core_ref = _readMAT!(joinpath(refdatapath, "V_KF_U4.mat"); flavor_idx=flavor_idx);
    core_julia = h5read(joinpath(juliadatapath, "V_KF_U4.h5"), "core")
    @show size(core_ref)
    @show size(core_julia)

    file_prefix = "parts_V_KF_core_$(channel)_spincomponent$(flavor_idx)_"

    # plot core
    figref, _ = plot_alliK(core_ref, 2, 101, imag)
    fig, _ = plot_alliK(core_julia, 2, 65, imag)
    PythonPlot.savefig(figref, joinpath(dirname(@__FILE__), "output", "plots", file_prefix * "core_ref.pdf"))
    PythonPlot.savefig(fig, joinpath(dirname(@__FILE__), "output", "plots", file_prefix * "core.pdf"))

    # inspect difference
    window = 37:165
    diff = core_julia .- core_ref[window,window,window,:,:,:,:]
    for ik in Iterators.product(ntuple(_->1:2,4)...)
        println("-- ik=$ik")
        @show argmax(abs.(diff[:,:,:,ik...]))
        @show maximum(abs.(diff[:,:,:,ik...]))
        maxref = maximum(abs.(core_ref[:,:,:,ik...]))
        @show maximum(abs.(diff[:,:,:,ik...]) / maxref)
    end

    # 1D comparison
    ommax = 0.3183098861837907
    ommax_j = 0.20371832715762606
    # ommax_j = 0.050929581789406514/2
    juliagrid = TCI4Keldysh.KF_grid(ommax_j, 7, 3)[1]
    @show juliagrid[10]
    @show refgrid[103]
    refgrid = collect(range(-ommax, ommax, 201))
    fig, _ = compare_alliK1D(core_julia, core_ref, juliagrid, refgrid, (:,85,105), (:,121,141))
    PythonPlot.savefig(fig, joinpath(dirname(@__FILE__), "output", "plots", file_prefix * "core_1D_comparison.pdf"))


    K2t = h5read(joinpath(juliadatapath, "V_KF_U3_ph_false.h5"), "K2")
    K2p = h5read(joinpath(juliadatapath, "V_KF_U3_pp_false.h5"), "K2")
    K2a = h5read(joinpath(juliadatapath, "V_KF_U3_pht_false.h5"), "K2")
    K2t_prime = h5read(joinpath(juliadatapath, "V_KF_U3_ph_true.h5"), "K2")
    K2p_prime = h5read(joinpath(juliadatapath, "V_KF_U3_pp_true.h5"), "K2")
    K2a_prime = h5read(joinpath(juliadatapath, "V_KF_U3_pht_true.h5"), "K2")

    # K2 CONTRIBUTIONS
    # t
    K2tref_prime = _readMAT!("V_KF_U3_1.mat"; flavor_idx=2)
    K2tref = _readMAT!("V_KF_U3_6.mat"; flavor_idx=2)
    # p
    K2pref_prime = _readMAT!("V_KF_U3_2.mat"; flavor_idx=2)
    K2pref = _readMAT!("V_KF_U3_5.mat"; flavor_idx=2)
    # a
    K2aref_prime = _readMAT!("V_KF_U3_3.mat"; flavor_idx=2)
    K2aref = _readMAT!("V_KF_U3_4.mat"; flavor_idx=2)

    # plot
    fig, _ = plot_alliK(K2p_prime, 2, 65)
    PythonPlot.savefig(fig, joinpath(dirname(@__FILE__), "output", "plots", file_prefix * "K2p_prime.pdf"))
    fig, _ = plot_alliK(K2pref_prime, 2, 101)
    PythonPlot.savefig(fig, joinpath(dirname(@__FILE__), "output", "plots", file_prefix * "K2pref_prime.pdf"))

    # 1D comparison
    ommax = 0.3183098861837907
    ommax_j = 0.20371832715762606
    juliagrid = TCI4Keldysh.KF_grid(ommax_j, 7, 3)[1]
    refgrid = collect(range(-ommax, ommax, 201))
    fig, _ = compare_alliK1D(K2p_prime, K2pref_prime, juliagrid, refgrid, (65,:,65), (101,:,101))
    PythonPlot.savefig(fig, joinpath(dirname(@__FILE__), "output", "plots", file_prefix * "K2p_prime_1D_comparison.pdf"))


    # inspect difference
    window = 37:165
    slice = (window,window,window,Colon(),Colon(),Colon(),Colon())

    tmax = maximum(abs.(K2tref))
    dt = abs.(K2tref[slice...] .- K2t)
    @show maximum(dt)
    @show maximum(dt/tmax)

    tmax = maximum(abs.(K2tref_prime))
    dt = abs.(K2tref_prime[slice...] .- K2t_prime)
    @show maximum(dt)
    @show maximum(dt/tmax)

    amax = maximum(abs.(K2aref))
    da = abs.(K2aref[slice...] .- K2a)
    @show maximum(da)
    @show maximum(da/amax)

    amax = maximum(abs.(K2aref_prime))
    da = abs.(K2aref_prime[slice...] .- K2a_prime)
    @show maximum(da)
    @show maximum(da/amax)

    pmax = maximum(abs.(K2pref))
    dp = abs.(K2pref[slice...] .- K2p)
    @show maximum(dp)
    @show maximum(dp/pmax)

    dp = abs.(K2pref_prime[slice...] .- K2p_prime)
    pmax = maximum(abs.(K2pref))
    pmax = maximum(abs.(K2pref_prime))
    @show maximum(dp)
    @show maximum(dp/pmax)

    # K1 CONTRIBUTIONS
    K1t = h5read(joinpath(juliadatapath, "V_KF_U2_ph.h5"), "K1")
    K1p = h5read(joinpath(juliadatapath, "V_KF_U2_pp.h5"), "K1")
    K1a = h5read(joinpath(juliadatapath, "V_KF_U2_pht.h5"), "K1")

    # MuNRG data
    # t
    K1tref = _readMAT!("V_KF_U2_1.mat"; flavor_idx=2)
    # p
    K1pref = _readMAT!("V_KF_U2_2.mat"; flavor_idx=2)
    # a
    K1aref = _readMAT!("V_KF_U2_3.mat"; flavor_idx=2)

    # inspect difference
    window = 37:165
    slice = (window,window,window,Colon(),Colon(),Colon(),Colon())
    tmax = maximum(abs.(K1tref))
    amax = maximum(abs.(K1aref))
    pmax = maximum(abs.(K1pref))
    dt = abs.(K1tref[slice...] .- K1t)
    da = abs.(K1aref[slice...] .- K1a)
    dp = abs.(K1pref[slice...] .- K1p)
    @show maximum(dt)
    @show maximum(dt/tmax)
    @show maximum(da)
    @show maximum(da/amax)
    @show maximum(dp)
    @show maximum(dp/pmax)

    ommax = 0.3183098861837907
    ommax_j = 0.20371832715762606
    juliagrid = TCI4Keldysh.KF_grid(ommax_j, 7, 3)[1]
    refgrid = collect(range(-ommax, ommax, 201))
    fig, _ = compare_alliK1D(K1t, K1tref, juliagrid, refgrid, (:,65,85), (:,101,121))
    PythonPlot.savefig(fig, joinpath(dirname(@__FILE__), "output", "plots", file_prefix * "K1t_1D_comparison.pdf"))
end
