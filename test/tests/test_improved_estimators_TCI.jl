# data sets to check
const PSFpath_list = [
    joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/"),
    joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg/")
]
for path in PSFpath_list
    @assert isdir(PSFpath_list) "directory $path does not exist"
end

# temporary change
const _ESTEP_OLD_DEFAULT = TCI4Keldysh._ESTEP_DEFAULT()
TCI4Keldysh._ESTEP_DEFAULT() = 50

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
    function test_SigmaEvaluator(PSFpath)
        R = 4
        beta = 60.0
        T = 1.0 / beta
        channel = "a"
        ωconvMat = TCI4Keldysh.channel_trafo(channel)
        spin = 1
        sev = TCI4Keldysh.SigmaEvaluator_MF(PSFpath, R, T, ωconvMat; flavor_idx=spin)

        # reference
        ω_fer_int = TCI4Keldysh.MF_grid(T, 2^R, true)
        U = 0.05
        Σ_calc_sIE = TCI4Keldysh.calc_Σ_MF_sIE(PSFpath, ω_fer_int; flavor_idx=spin, T=T)

        for _ in 1:10
            w = rand(1:2^R, 3)
            w_int = sev.ωconvMat*w + sev.ωconvOff
            for il in eachindex(w_int)
                @test isapprox(Σ_calc_sIE[w_int[il]], sev(il, w...); atol=1.e-12)
            end
        end
    end

    test_freq_shift_rot()
    for PSFpath in PSFpath_list
        test_SigmaEvaluator(PSFpath)
    end
end

@testset "SIE: K1" begin
    
    function test_K1_TCI(formalism="MF"; ωmax::Float64=1.0, channel="t", flavor=1)
        basepath = "SIAM_u=0.50"
        PSFpath = joinpath(TCI4Keldysh.datadir(), basepath, "PSF_nz=2_conn_zavg/")
        R = 8
        T = TCI4Keldysh.dir_to_T(PSFpath)

        # TCI4Keldysh
        ωs_ext = if formalism=="MF"
                TCI4Keldysh.MF_grid(T, 2^(R-1), false)
            else
                TCI4Keldysh.KF_grid_bos(ωmax, R)
            end
        K1_slice = formalism=="MF" ? (1:2^R,) : (1:2^R,:,:)
        broadening_kwargs = TCI4Keldysh.read_broadening_settings(basepath;channel=channel)
        K1_test = TCI4Keldysh.precompute_K1r(PSFpath, flavor, formalism; channel=channel, ωs_ext=ωs_ext, broadening_kwargs...)[K1_slice...]

        ncomponents = formalism=="MF" ? 1 : 4
        # TCI4Keldysh: TCI
        tolerance=1.e-4
        K1_tci = TCI4Keldysh.K1_TCI(
            PSFpath,
            R;
            ωmax=ωmax,
            formalism=formalism,
            channel=channel,
            T=T,
            flavor_idx=flavor,
            unfoldingscheme=:interleaved,
            tolerance=tolerance
            )

        K1_tcivals_ = fill(zeros(ComplexF64, ntuple(_->2,R)), Int(sqrt(ncomponents)), Int(sqrt(ncomponents)))
        for i in eachindex(K1_tcivals_)
            if !isnothing(K1_tci[i])
                K1_tcivals_[i] = TCI4Keldysh.qtt_to_fattensor(K1_tci[i].tci.sitetensors)
            end
        end
        K1_tcivals = [TCI4Keldysh.qinterleaved_fattensor_to_regular(k1, R) for k1 in K1_tcivals_]
        K1_tcivals_block = reshape(vcat(K1_tcivals...), (2^R, ncomponents))

        K1_test = reshape(K1_test, 2^R, ncomponents)
        diff = abs.(K1_tcivals_block .- K1_test)
        if maximum(diff)<1.e-10
            @test true
            return
        end
        @test  maximum(diff) / maximum(abs.(K1_test)) < 2.0*tolerance
    end

    for channel in ["a","p","t"]
        for flavor in [1,2]
            for formalism in ["MF","KF"]
                test_K1_TCI(formalism; channel=channel, flavor=flavor)
            end
        end
    end

end

@testset "SIE: K2" begin

    function test_K2_TCI_precomputed(;formalism="MF", channel="t", prime=false, flavor_idx=1)
        basepath = "SIAM_u=0.50"
        PSFpath = joinpath(TCI4Keldysh.datadir(), basepath, "PSF_nz=4_conn_zavg/")

        R = 5
        Nhalf = 2^(R-1)
        ωmax = 1.0

        # TCI4Keldysh: Reference
        T = TCI4Keldysh.dir_to_T(PSFpath)
        ωs_ext = if formalism=="MF"
                TCI4Keldysh.MF_npoint_grid(T, Nhalf, 2)
            else
                TCI4Keldysh.KF_grid(ωmax, R, 2)
            end
        K2 = TCI4Keldysh.precompute_K2r(PSFpath, flavor_idx, formalism; ωs_ext=ωs_ext, channel=channel, prime=prime)
        K2slice = if formalism=="MF"
                (1:2^R,Colon())
            else
                (1:2^R,Colon(),:,:,:)
            end
        K2 = K2[K2slice...]

        # TCI4Keldysh: TCI way
        tolerance = 1.e-6
        K2tci = TCI4Keldysh.K2_TCI_precomputed(
            PSFpath,
            R;
            formalism=formalism,
            flavor_idx=flavor_idx,
            channel=channel,
            prime=prime,
            T=T,
            ωmax=ωmax,
            tolerance=tolerance
        )

        ncomponents = formalism=="MF" ? 1 : 8
        nkeldysh = formalism=="MF" ? 1 : 2
        K2tcivals = fill(zeros(ComplexF64, ntuple(_->2, 2*R)), nkeldysh,nkeldysh,nkeldysh)
        for i in eachindex(K2tci)
            if !isnothing(K2tci[i])
                K2tcivals[i] = TCI4Keldysh.qtt_to_fattensor(K2tci[i].tci.sitetensors)
            end
        end
        K2tcivals = [TCI4Keldysh.qinterleaved_fattensor_to_regular(k2, R) for k2 in K2tcivals]
        @show size.(K2tcivals)

        K2_test = reshape(K2, 2^R, 2^R, ncomponents)
        @show size(K2_test)
        for i in 1:ncomponents
            diff = abs.(K2_test[:,:,i] .- K2tcivals[i]) ./ maximum(abs.(K2_test[:,:,i]))
            if !isnan(maximum(diff))
                @test maximum(diff) < 2.0*tolerance
            else
                @test maximum(abs.(K2_test[:,:,i]))<1.e-10
            end
        end
    end

    # for channel in ["a","p","t"]
    #     for flavor in [1,2]
    #         for formalism in ["MF","KF"]
    #             for prime in [true,false]
    #                 test_K2_TCI_precomputed(;formalism=formalism, prime=prime, channel=channel, flavor_idx=flavor)
    #             end
    #         end
    #     end
    # end
    test_K2_TCI_precomputed(;formalism="MF", prime=true, channel="t", flavor_idx=1)
    test_K2_TCI_precomputed(;formalism="MF", prime=false, channel="pNRG", flavor_idx=2)
    test_K2_TCI_precomputed(;formalism="MF", prime=true, channel="a", flavor_idx=2)
    test_K2_TCI_precomputed(;formalism="KF", prime=true, channel="a", flavor_idx=1)
    test_K2_TCI_precomputed(;formalism="KF", prime=false, channel="pNRG", flavor_idx=2)

end

@testset "SIE: Vertex@Matsubara" begin
    
    function test_Gamma_core_TCI_MF(PSFpath; freq_conv="a", R=4, beta=15.0, tolerance=1.e-5, batched=false, use_ΣaIE=false)
        T = 1.0 / beta
        spin = 1

        ωconvMat = TCI4Keldysh.channel_trafo(freq_conv)

        # tci
        Γcore = if batched
                TCI4Keldysh.Γ_core_TCI_MF_batched(
                PSFpath, R; use_ΣaIE=use_ΣaIE, T=T, ωconvMat=ωconvMat, flavor_idx=spin, tolerance=tolerance, verbosity=2
                )
            else
                TCI4Keldysh.Γ_core_TCI_MF(
                PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=spin, tolerance=tolerance, unfoldingscheme=:interleaved, verbosity=2
                )
            end

            # grids
        ω_bos = TCI4Keldysh.MF_grid(T, 2^(R-1), false)
        ω_fer = TCI4Keldysh.MF_grid(T, 2^(R-1), true)
        ω_fer_int = TCI4Keldysh.MF_grid(T, 2^R, true)
        ωs_ext=(ω_bos, ω_fer, ω_fer)

            # sIE self-energy
        U = 0.05
        G        = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
        G_auxL    = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
        G_auxR   = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "Q1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
        G_QQ_aux = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "Q1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
        G_data      = TCI4Keldysh.precompute_all_values(G)
        G_auxL_data  = TCI4Keldysh.precompute_all_values(G_auxL)
        G_auxR_data  = TCI4Keldysh.precompute_all_values(G_auxR)
        G_QQ_aux_data=TCI4Keldysh.precompute_all_values(G_QQ_aux)
        
        refval = nothing

        if use_ΣaIE
            Σ_calcL = TCI4Keldysh.calc_Σ_MF_aIE(G_auxL_data, G_data)
            Σ_calcR = TCI4Keldysh.calc_Σ_MF_aIE(G_auxL_data, G_data)
            # Γ core
            refval = TCI4Keldysh.compute_Γcore_symmetric_estimator(
                "MF", PSFpath*"4pt/", Σ_calcR; Σ_calcL=Σ_calcL, ωs_ext=ωs_ext, T=T, ωconvMat=ωconvMat, flavor_idx=spin
                )
        else
            Σ_calc_sIE = TCI4Keldysh.calc_Σ_MF_sIE(G_QQ_aux_data, G_auxL_data, G_auxR_data, G_data, U/2)

            # Γ core
            refval = TCI4Keldysh.compute_Γcore_symmetric_estimator(
                "MF", PSFpath*"4pt/", Σ_calc_sIE; ωs_ext=ωs_ext, T=T, ωconvMat=ωconvMat, flavor_idx=spin
                )
        end
        maxref = maximum(abs.(refval))

        # test
        testslice = fill(1:2^R, 3)
        test_qttval = reshape([Γcore(id...) for id in Iterators.product(testslice...)], length.(testslice)...)
        testdiff = abs.(refval[testslice...] .- test_qttval[testslice...]) ./ maxref
        maxerr = maximum(testdiff)

        @test maxerr < 5.0 * tolerance
    end

    function test_K2_TCI(PSFpath; channel="a", R=5, beta=100.0, tolerance=1.e-7, prime=false)
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


        op_labels = Tuple(TCI4Keldysh.oplabels_K2(channel, prime))
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

    for PSFpath in PSFpath_list
        for channel in ["a", "p", "t"]
            test_Gamma_core_TCI_MF(PSFpath; freq_conv=channel)
            for prime in [true, false]
                test_K2_TCI(PSFpath; channel=channel, prime=prime)
            end
        end
    end

    # test BatchEvaluator
    for PSFpath in PSFpath_list
        test_Gamma_core_TCI_MF(PSFpath; R=4, freq_conv="a", beta=TCI4Keldysh.dir_to_beta(PSFpath), tolerance=1.e-6, batched=true)
        test_Gamma_core_TCI_MF(PSFpath; R=4, freq_conv="a", beta=TCI4Keldysh.dir_to_beta(PSFpath), tolerance=1.e-6, batched=true, use_ΣaIE=true)
    end

end

@testset "SIE: Vertex@Keldysh" begin
    
    function test_Γcore_KF(
        iK::Int, flavor_idx, channel::String="a";
        R=3, tolerance=1.e-5, batched=false,
        kwargs...
        )
        basepath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50")
        PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
        ωconvMat = TCI4Keldysh.channel_trafo(channel)
        T = TCI4Keldysh.dir_to_T(PSFpath)
        # broadening_kwargs = TCI4Keldysh.read_broadening_settings(basepath; channel=channel)
        # broadening_kwargs[:estep] = 50
        ωmax = 0.1
        D = 3
        iK_tuple = TCI4Keldysh.KF_idx(iK, D)
        γ, sigmak = TCI4Keldysh.default_broadening_γσ(T)

        # reference
        ωs_ext = TCI4Keldysh.KF_grid(ωmax, R, D)
        Σωgrid = TCI4Keldysh.KF_grid_fer(2*ωmax, R+1)
        # Σ_ref = TCI4Keldysh.calc_Σ_KF_sIE_viaR(PSFpath, Σωgrid; T=T, flavor_idx=flavor_idx, sigmak, γ)
        (Σ_L,Σ_R) = TCI4Keldysh.calc_Σ_KF_aIE_viaR(PSFpath, Σωgrid; T=T, flavor_idx=flavor_idx, sigmak, γ)
        Γcore_ref = TCI4Keldysh.compute_Γcore_symmetric_estimator(
            "KF",
            PSFpath*"4pt/",
            # Σ_ref
            Σ_R
            ;
            Σ_calcL=Σ_L,
            T,
            flavor_idx = flavor_idx,
            ωs_ext = ωs_ext,
            ωconvMat=ωconvMat,
            sigmak,
            γ
            # broadening_kwargs...
        )

        # tci
        qtt = TCI4Keldysh.Γ_core_TCI_KF(
            PSFpath, R, iK, ωmax
            ; 
            sigmak=sigmak,
            γ=γ,
            T=T, ωconvMat=ωconvMat, flavor_idx=flavor_idx,
            tolerance=tolerance,
            unfoldingscheme=:interleaved,
            batched=batched,
            # KEV=KEV,
            # coreEvaluator_kwargs=coreEvaluator_kwargs
            kwargs...
            )

        # compare
        Γcore_tci = TCI4Keldysh.QTT_to_fatTensor(qtt, Base.OneTo.(fill(2^R, D)))
        diff = abs.(Γcore_tci .- Γcore_ref[1:2^R,:,:,iK_tuple...])
        maxref =  maximum(abs.(Γcore_ref))
        reldiff = diff ./ maxref
        @test maximum(reldiff) < 3.0*tolerance
    end

    test_Γcore_KF(
        6, 1, "p";
        R=4,
        tolerance=1.e-3,
        batched=true,
        KEV=TCI4Keldysh.MultipoleKFCEvaluator,
        coreEvaluator_kwargs=Dict{Symbol,Any}(:cutoff=>1.e-6, :nlevel=>2),
        )
    test_Γcore_KF(2, 1, "p")
    test_Γcore_KF(8, 1, "a"; R=4, tolerance=1.e-4, batched=true)
    # test_Γcore_KF(14, 1, "t"; R=3, tolerance=1.e-8, batched=true)
    test_Γcore_KF(9, 1, "p"; R=3, tolerance=1.e-8, batched=false)
end

# revert temporary change
TCI4Keldysh._ESTEP_DEFAULT() = _ESTEP_OLD_DEFAULT