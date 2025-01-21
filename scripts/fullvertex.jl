using Plots
using MAT
using LinearAlgebra
using BenchmarkTools
using Serialization
using QuanticsTCI
import TensorCrossInterpolation as TCI
import QuanticsGrids as QG 

function Γfull_TCI_MF()
    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
    PSFpath = joinpath(basepath, "PSF_nz=4_conn_zavg/")
    T = TCI4Keldysh.dir_to_T(PSFpath)
    flavor_idx = 2
    channel = "p"
    foreign_channels = ("a","t")
    R = 5

    gev = TCI4Keldysh.ΓEvaluator_MF(PSFpath, R;
        T=T,
        flavor_idx=flavor_idx,
        channel=channel,
        foreign_channels=foreign_channels
        )

    qtt, _, _ = quanticscrossinterpolate(ComplexF64, gev, ntuple(_->2^R, 3); tolerance=1.e-2)
    @show TCI4Keldysh.rank(qtt)

    gbev = TCI4Keldysh.ΓBatchEvaluator_MF(gev)
    tt, _, _ = TCI.crossinterpolate2(ComplexF64, gbev, gbev.qf.localdims; tolerance=1.e-2)
    @show TCI.rank(tt)

end

function check_fullvertex_qtt(qttpath::String)
    (tci, grid) = deserialize(qttpath)
end

function test_ΓEvaluator_MF(;do_benchmark=false, do_test=true, test_batcheval=false)
    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
    PSFpath = joinpath(basepath, "PSF_nz=4_conn_zavg/")
    T = TCI4Keldysh.dir_to_T(PSFpath)
    flavor_idx = 1
    channel = "a"
    foreign_channels = ("t","p")
    R = 3

    gev = TCI4Keldysh.ΓEvaluator_MF(PSFpath, R;
        T=T,
        flavor_idx=flavor_idx,
        channel=channel,
        foreign_channels=foreign_channels
        )

    if do_benchmark
        @btime $gev(1,1,1)
        @btime $gev.core(1,1,1)
    end

    # test
    if do_test
        Γfull = TCI4Keldysh.compute_Γfull_symmetric_estimator(
            "MF",
            PSFpath;
            T=T,
            ωs_ext=gev.core.GFevs[1].GF.ωs_ext,
            flavor_idx=flavor_idx,
            channel=channel
        )

        Γtest = zeros(ComplexF64, size(Γfull))
        Γbatchtest = zeros(ComplexF64, size(Γfull))
        Threads.@threads for id in CartesianIndices(Γfull)
            Γtest[id] = gev(Tuple(id)...)
        end

        @assert maximum(abs.(Γtest .- Γfull)) < 1.e-14

        if test_batcheval

            gbev = TCI4Keldysh.ΓBatchEvaluator_MF(gev)

            @show gev(1,1,1)
            @show QG.origcoord_to_quantics(gbev.grid, (1,1,1))
            @show gbev(QG.origcoord_to_quantics(gbev.grid, (1,1,1)))

            batchslice = (1:2^R,:,:)
            Γbatchtest = zeros(ComplexF64, size(Γfull[batchslice...]))
            grid = gbev.grid
            Threads.@threads for id in CartesianIndices(Γfull[batchslice...])
                Γbatchtest[id] = gbev(QG.origcoord_to_quantics(grid, Tuple(id)))
            end

            @assert maximum(abs.(Γbatchtest .- Γfull[batchslice...])) < 1.e-14
        end
    end
end

function test_compute_Γfull_symmetric_estimator(
    formalism = "MF"
)
    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
    PSFpath = joinpath(basepath, "PSF_nz=4_conn_zavg/")
    T = TCI4Keldysh.dir_to_T(PSFpath)
    flavor_idx = 1
    channel = "t"
    R = 4
    Nhalf = 2^(R-1)
    ωs_ext = TCI4Keldysh.MF_npoint_grid(T, Nhalf, 3)

    Γfull = TCI4Keldysh.compute_Γfull_symmetric_estimator(
        formalism,
        PSFpath;
        T=T,
        ωs_ext=ωs_ext,
        flavor_idx=flavor_idx,
        channel=channel
    )

    @show any(isnan.(Γfull))
    @show maximum(abs.(Γfull))

    slice = (5,:,:)
    heatmap(log10.(abs.(Γfull[slice...])))
    savefig("foo.pdf")
end

function check_V_MFfull(channel="t", flavor_idx=1)
    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
    PSFpath = joinpath(basepath, "PSF_nz=4_conn_zavg/")
    Vpath = joinpath(basepath, "V_MF_$(TCI4Keldysh.channel_translate(channel))")

    T = TCI4Keldysh.dir_to_T(PSFpath)

    # load MuNRG
    Vref = nothing
    ωs_ref = nothing
    matopen(joinpath(Vpath, "V_MF_sym.mat")) do f
        CFdat = read(f, "CFdat")
        Vref = CFdat["Ggrid"][flavor_idx]
        ωs_ref = ntuple(i -> imag.(vec(vec(CFdat["ogrid"])[i])), 3)
        ωs_ref = ωs_ref[[3,1,2]]
    end
    Vref = reverse(permutedims(Vref, (3,1,2)))

    # Julia
    
    # ωs_ext = ωs_ref
    R = 4
    Nhalf = 2^(R-1)
    ωs_ext = TCI4Keldysh.MF_npoint_grid(T, Nhalf, 3)
    Vfull = TCI4Keldysh.compute_Γfull_symmetric_estimator(
        "MF",
        PSFpath;
        T=T,
        flavor_idx=flavor_idx,
        ωs_ext=ωs_ext,
        channel=channel
    )

    # compare
    offset = TCI4Keldysh.idx_trafo_offset(ωs_ext, ωs_ref, diagm([1,1,1]))
    @show (1+offset[1]:offset[1]+2^R+1, 1+offset[2]:offset[2]+2^R, 1+offset[3]:offset[3]+2^R)
    Vref_window = Vref[1+offset[1]:offset[1]+2^R+1, 1+offset[2]:offset[2]+2^R, 1+offset[3]:offset[3]+2^R]
    @show size(Vref)
    @show length.(ωs_ref)
    @show size(Vfull)
    @show size(Vref_window)

    diff = abs.(Vref_window .- Vfull)
    @show maximum(abs.(Vref_window))
    @show maximum(abs.(Vfull))
    printstyled("Maximum deviation Julia vs. MuNRG: $(maximum(diff))\n"; color=:blue)

    # plot
    heatmap(abs.(Vfull[Nhalf+1,:,:]))
    savefig("V_MF_full.pdf")
    heatmap(abs.(Vref_window[Nhalf+1,:,:]))
    savefig("V_MF_full_ref.pdf")
    heatmap(abs.(diff[Nhalf+1,:,:]))
    savefig("V_MF_full_diff.pdf")
end


function check_V_KFfull(iK::Int=2, channel="t")

    flavor_idx=1

    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
    PSFpath = joinpath(basepath, "PSF_nz=4_conn_zavg/")
    Vpath = joinpath(basepath, "V_KF_$(TCI4Keldysh.channel_translate(channel))")

    T = TCI4Keldysh.dir_to_T(PSFpath)

    # load MuNRG
    Vref = nothing
    ωs_ref = nothing
    matopen(joinpath(Vpath, "V_KF_sym.mat")) do f
        CFdat = read(f, "CFdat")
        Vref = CFdat["Ggrid"][flavor_idx]
        ωs_ref = ntuple(i -> real.(vec(vec(CFdat["ogrid"])[i])), 3)
        ωs_ref = ωs_ref[[3,1,2]]
    end
    Vref = permutedims(Vref, (3,1,2,4,5,6,7))
    Vref = reverse(Vref; dims=(1,2,3))
    @show size(Vref)

    # Julia
    (γ, sigmak) = TCI4Keldysh.read_broadening_params(basepath; channel=channel)
    broadening_kwargs = TCI4Keldysh.read_broadening_settings(joinpath(TCI4Keldysh.datadir(), basepath); channel=channel)
    if !haskey(broadening_kwargs, :estep)
        broadening_kwargs[:estep] = 10
    end

    # ωs_ext = ωs_ref
    R = 4
    Nhalf = 2^(R-1)
    ωs_cen = [div(length(om), 2)+1 for om in ωs_ref]
    @show [ωs_ref[i][ωs_cen[i]] for i in eachindex(ωs_ref)]
    om_small = ntuple(i -> ωs_ref[i][ωs_cen[i] - Nhalf : ωs_cen[i] + Nhalf], 3)
    Vfull = TCI4Keldysh.compute_Γfull_symmetric_estimator(
        "KF",
        PSFpath;
        T=T,
        flavor_idx=flavor_idx,
        ωs_ext=om_small,
        channel=channel,
        γ=γ,
        sigmak=sigmak,
        broadening_kwargs...
    )
    @show size(Vfull)
    
    # compare
    # offset = TCI4Keldysh.idx_trafo_offset(ωs_ext, ωs_ref, diagm([1,1,1]))
    # @show (1+offset[1]:offset[1]+2^R+1, 1+offset[2]:offset[2]+2^R, 1+offset[3]:offset[3]+2^R)
    # Vref_window = Vref[1+offset[1]:offset[1]+2^R+1, 1+offset[2]:offset[2]+2^R, 1+offset[3]:offset[3]+2^R]
    # @show size(Vref)
    # @show length.(ωs_ref)
    # @show size(Vfull)
    # @show size(Vref_window)

    # diff = abs.(Vref_window .- Vfull)
    # @show maximum(abs.(Vref_window))
    # @show maximum(abs.(Vfull))
    # printstyled("Maximum deviation Julia vs. MuNRG: $(maximum(diff))\n"; color=:blue)
end

# check_V_MFfull("a", 1)
