using Plots
using Profile
using StatProfilerHTML

TCI4Keldysh.TIME() = true

function time_Γcore()
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    R = 4
    tolerance = 1.e-5
    ωconvMat = 
        [
            0 -1  0;
            1  1  0;
            -1  0 -1;
            0  0  1;
        ]
    beta = 50.0
    T = 1.0/beta

    # compile
    println("  Compile run...")
    foo = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved
        )
    println("  Time...")
    t = @elapsed begin
        Γcore = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved
        )
    end 
    println(" TIME: $t")
end

function profile_Γcore()
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    R = 4
    tolerance = 1.e-5
    ωconvMat = 
        [
            0 -1  0;
            1  1  0;
            -1  0 -1;
            0  0  1;
        ]
    beta = 50.0
    T = 1.0/beta

    # compile
    println("  Compile run...")
    Γcore = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved
        )
    # profile
    Profile.clear()
    println("  Profiling...")
    Profile.@profile Γcore = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved
        )
    statprofilehtml()
end

"""
test & plot
"""
function test_Gamma_core_TCI_MF(; freq_conv="a", R=4, beta=50.0, tolerance=1.e-5)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    T = 1.0 / beta
    logtol = round(Int, log10(tolerance))
    spin = 1
    # spin = 2 # FAILS because F1F1dag Adisc lacks spin 2 component...

    ωconvMat = if freq_conv == "a"
        [
            0 -1  0;
            0  0  1;
            -1  0 -1;
            1  1  0;
        ]
    elseif freq_conv == "p"
        [
            0 -1  0;
            1  0 -1;
            -1  1  0;
            0  0  1;
        ]
    elseif freq_conv == "t"
        # t convention
        [
            0 -1  0;
            1  1  0;
            -1  0 -1;
            0  0  1;
        ]
    else
        error("Invalid frequency convention")
    end

    # compute
    TCI4Keldysh.@TIME Γcore = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=spin, tolerance=tolerance, unfoldingscheme=:interleaved, verbosity=2
        ) "Γcore @ TCI"
    @show TCI4Keldysh.rank(Γcore)

    # reference

        # grids
    ω_bos = TCI4Keldysh.MF_grid(T, 2^(R-1), false)
    ω_fer = TCI4Keldysh.MF_grid(T, 2^(R-1), true)
    ω_fer_int = TCI4Keldysh.MF_grid(T, 2^R, true)
    ωs_ext=(ω_bos, ω_fer, ω_fer)

        # sIE self-energy
    U = 0.05
    G        = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_aux    = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_QQ_aux = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "Q1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_data      = TCI4Keldysh.precompute_all_values(G)
    G_aux_data  = TCI4Keldysh.precompute_all_values(G_aux)
    G_QQ_aux_data=TCI4Keldysh.precompute_all_values(G_QQ_aux)
    Σ_calc_sIE = TCI4Keldysh.calc_Σ_MF_sIE(G_QQ_aux_data, G_aux_data, G_aux_data, G_data, U/2)

        # Γ core
    TCI4Keldysh.@TIME refval = TCI4Keldysh.compute_Γcore_symmetric_estimator(
        "MF", PSFpath*"4pt/", Σ_calc_sIE; ωs_ext=ωs_ext, T=T, ωconvMat=ωconvMat, flavor_idx=spin
        ) "Γcore @ conventional"
    maxref = maximum(abs.(refval))
    logmaxref = round(Int, log10(maxref))

    # test
    testslice = fill(1:2^R, 3)
    test_qttval = reshape([Γcore(id...) for id in Iterators.product(testslice...)], length.(testslice)...)
    testdiff = abs.(refval[testslice...] .- test_qttval[testslice...]) ./ maxref
    printstyled("---- Γcore rank: $(TCI4Keldysh.rank(Γcore))\n"; color=:blue)
    printstyled("---- Maximum value Γcore: $(maxref)\n"; color=:green)
    printstyled("---- Maximum error: $(maximum(testdiff))\n"; color=:green)

    # plot
    slice = [1:2^R, 1:2^R, 2^(R-1)]
    qttval = reshape([Γcore(id...) for id in Iterators.product(slice...)], length.(slice)...)

    scfun = x -> log10(abs(x))
    heatmap(scfun.(qttval[slice[1:2]...]); clim=(logmaxref + logtol - 1, logmaxref + 2))
    savefig("gammacore.png")

    heatmap(scfun.(refval)[slice...]; clim=(logmaxref + logtol - 1, logmaxref + 2))
    savefig("gammacore_ref.png")

    diff = abs.(refval[slice...] .- qttval[slice[1:2]...]) ./ maxref
    heatmap(log10.(diff))
    savefig("diff.png")
end