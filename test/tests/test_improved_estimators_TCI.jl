@testset "SIE: Self-energy" begin
        
    function test_freq_shift_rot()
        M = [
            0 -1  0;
            1  1  0;
            -1  0 -1;
            0  0  1;
        ]
        shift = TCI4Keldysh.freq_shift_rot(M, 4)
        @test shift==[7, -1, 10, 2]
    end

    # no TCI involved here
    function test_SigmaEvaluator()
        R = 4
        PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
        beta = 60.0
        T = 1.0 / beta
        channel = "a"
        ωconvMat = TCI4Keldysh.channel_trafo(channel)
        spin = 1
        sev = TCI4Keldysh.SigmaEvaluator_MF(PSFpath, R, T, ωconvMat; flavor_idx=spin)

        # reference
        ω_fer_int = TCI4Keldysh.MF_grid(T, 2^R, true)
        U = 0.05
        G        = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
        G_aux    = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
        G_QQ_aux = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "Q1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
        G_data      = TCI4Keldysh.precompute_all_values(G)
        G_aux_data  = TCI4Keldysh.precompute_all_values(G_aux)
        G_QQ_aux_data=TCI4Keldysh.precompute_all_values(G_QQ_aux)
        Σ_calc_sIE = TCI4Keldysh.calc_Σ_MF_sIE(G_QQ_aux_data, G_aux_data, G_aux_data, G_data, U/2)

        for _ in 1:10
            w = rand(1:2^R, 3)
            w_int = sev.ωconvMat*w + sev.ωconvOff
            for il in eachindex(w_int)
                @test isapprox(Σ_calc_sIE[w_int[il]], sev(il, w...); atol=1.e-12)
            end
        end
    end

    test_freq_shift_rot()
    test_SigmaEvaluator()
end

@testset "SIE: Vertex" begin
    
    function test_Gamma_core_TCI_MF(; freq_conv="a", R=4, beta=15.0, tolerance=1.e-5)
        PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
        T = 1.0 / beta
        spin = 1

        ωconvMat = TCI4Keldysh.channel_trafo(freq_conv)

        # compute
        Γcore = TCI4Keldysh.Γ_core_TCI_MF(
            PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=spin, tolerance=tolerance, unfoldingscheme=:interleaved, verbosity=2
            )

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
        refval = TCI4Keldysh.compute_Γcore_symmetric_estimator(
            "MF", PSFpath*"4pt/", Σ_calc_sIE; ωs_ext=ωs_ext, T=T, ωconvMat=ωconvMat, flavor_idx=spin
            )
        maxref = maximum(abs.(refval))

        # test
        testslice = fill(1:2^R, 3)
        test_qttval = reshape([Γcore(id...) for id in Iterators.product(testslice...)], length.(testslice)...)
        testdiff = abs.(refval[testslice...] .- test_qttval[testslice...]) ./ maxref
        maxerr = maximum(testdiff)

        @test maxerr < 5.0 * tolerance
    end

    function test_K2_TCI(; channel="a", R=5, beta=100.0, tolerance=1.e-7, prime=false)
        PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
        T = 1.0 / beta
        flavor = 1

            # grids
        ω_fer_int = TCI4Keldysh.MF_grid(T, 2^R, true)
        ωs_ext=TCI4Keldysh.MF_npoint_grid(T, 2^(R-1), 2)
            # sIE self-energy
        U = 0.05
        G        = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=flavor, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
        G_aux    = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=flavor, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
        G_QQ_aux = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "Q1dag"]; T, flavor_idx=flavor, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
        G_data      = TCI4Keldysh.precompute_all_values(G)
        G_aux_data  = TCI4Keldysh.precompute_all_values(G_aux)
        G_QQ_aux_data=TCI4Keldysh.precompute_all_values(G_QQ_aux)
        Σ_calc_sIE = TCI4Keldysh.calc_Σ_MF_sIE(G_QQ_aux_data, G_aux_data, G_aux_data, G_data, U/2)


        (i,j) = TCI4Keldysh.merged_legs_K2(channel, prime)
        nonij = sort(setdiff(1:4, (i,j)))
        leg_labels = ("1", "1dag", "3", "3dag")
        op_labels = ("Q$i$j", leg_labels[nonij[1]], leg_labels[nonij[2]])
        K2ref = TCI4Keldysh.compute_K2r_symmetric_estimator(
            "MF", PSFpath, op_labels, Σ_calc_sIE;
            ωs_ext=ωs_ext, T=T, flavor_idx=flavor, ωconvMat=TCI4Keldysh.channel_trafo_K2(channel, prime)
        )

        if channel=="p"
            @test maximum(abs.(K2ref)) <= 1.e-12
            return
        end

        # TCI
        K2qtt = TCI4Keldysh.K2_TCI(
            PSFpath, R, channel, prime;
            T=T, flavor_idx=flavor, tolerance=tolerance, unfoldingscheme=:interleaved
            )

        # test & plot
        testslice = fill(1:2^R, 2)
        test_qttval = TCI4Keldysh.QTT_to_fatTensor(K2qtt, testslice)
        maxref = maximum(abs.(K2ref[testslice...]))
        maxdiff = maximum(abs.(test_qttval .- K2ref[testslice...])) / maxref
        @test maxdiff < 5.0 * tolerance
    end

    for channel in ["a", "p", "t"]
        test_Gamma_core_TCI_MF(;freq_conv=channel)
        for prime in [true, false]
            test_K2_TCI(;channel=channel, prime=prime)
        end
    end
end