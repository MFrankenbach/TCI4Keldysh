using Plots

function test_compute_Γfull_symmetric_estimator(
    formalism = "MF"
)
    basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
    PSFpath = joinpath(basepath, "PSF_nz=4_conn_zavg/")
    T = TCI4Keldysh.dir_to_T(PSFpath)
    flavor_idx = 1
    channel = "t"
    R = 7
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

    slice = (5,:,:)
    heatmap(log10.(abs.(Γfull[slice...])))
    savefig("foo.pdf")
end