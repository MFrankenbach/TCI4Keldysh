using TCI4Keldysh
using LinearAlgebra
using Test

function test_ΓEvaluator_KF(;flavor_idx::Int=1, iK::Int=2, channel, foreign_channels)
    base_path = "SIAM_u=0.50"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=4_conn_zavg/")
    ωs_ext = TCI4Keldysh.KF_grid(0.5, 3, 3)
    omsz = length.(ωs_ext)
    maxerr = 0.0
    maxerrs = []
    iKtuple = TCI4Keldysh.KF_idx(iK,3)

    broadening_kwargs = TCI4Keldysh.read_all_broadening_params(base_path; channel=channel)
    broadening_kwargs[:estep] = 5
    gev = TCI4Keldysh.ΓEvaluator_KF(
        PSFpath, iK, TCI4Keldysh.MultipoleKFCEvaluator;
        channel=channel,
        foreign_channels=foreign_channels,
        flavor_idx=flavor_idx,
        KEV_kwargs = Dict([(:nlevel, 2), (:cutoff, 1.e-10)]),
        ωs_ext=ωs_ext,
        broadening_kwargs...
    )
    # reference
    gev_ref = TCI4Keldysh.compute_Γfull_symmetric_estimator(
        "KF", PSFpath;
        T=TCI4Keldysh.dir_to_T(PSFpath),
        flavor_idx=flavor_idx,
        ωs_ext=ωs_ext,
        channel=channel,
        broadening_kwargs...
    )
    maxref = maximum(abs.(gev_ref))
    for ic in Iterators.product(Base.OneTo.(omsz)...)
        val = gev(ic...)
        refval = gev_ref[ic..., iKtuple...]
        maxerr = max(maxerr, abs(val - refval)/maxref) 
    end
    push!(maxerrs, maxerr)

    @test maxerr <= 1.e-9
end

@testset begin
    for iK in 1:15
        println("==== iK: $iK")
        for flavor_idx in 1:2
            test_ΓEvaluator_KF(;flavor_idx=flavor_idx, iK=iK, channel="t", foreign_channels=("a", "pNRG"))
            test_ΓEvaluator_KF(;flavor_idx=flavor_idx, iK=iK, channel="pNRG", foreign_channels=("a", "t"))
        end
    end
end