using ITensors
using QuanticsTCI
using LinearAlgebra

@testset "PSF -> Correlator @ pointwise TCI" begin

    function maxdiff_qtt_ref(R::Int, qtt, refval::Array{T,D}) where {T,D}
        maxref = maximum(abs.(refval))
        slice_end = fill(2^R, D)
        slice = Base.OneTo.(slice_end)
        qttval = reshape([qtt(id...) for id in Iterators.product(slice...)], slice_end...)
        return maximum(abs.(refval[slice...] - qttval) ./ maxref)
    end

    function test_compress_tucker_pointwise(npt::Int, svd_kernel=false)
        R = 5
        tucker = TCI4Keldysh.multipeak_tucker_decomp(npt, R; beta=100.0, nωdisc=10)
        tolerance = 1.e-8
        refval = TCI4Keldysh.contract_1D_Kernels_w_Adisc_mp(tucker.legs, tucker.center)
        qtt = TCI4Keldysh.compress_tucker_pointwise(tucker, svd_kernel; tolerance=tolerance, unfoldingscheme=:interleaved)
        @test maxdiff_qtt_ref(R, qtt, refval) < 5.0 * tolerance
    end

    function test_compress_PartialCorrelator_pointwise(svd_kernel=false; npt=2, perm_idx=1, ano=true)
        R = 5
        beta = 1.0
        nomdisc = 9
        tolerance = 1.e-8
        GF = TCI4Keldysh.multipeak_correlator_MF(npt, R; beta=beta, nωdisc=nomdisc)

        Gp = GF.Gps[perm_idx]
        # reference
        refval = if ano && TCI4Keldysh.ano_term_required(Gp)
                    TCI4Keldysh.precompute_all_values_MF(Gp)
                else
                    TCI4Keldysh.precompute_all_values_MF_noano(Gp)
                end

        qtt = if ano && TCI4Keldysh.ano_term_required(Gp)
                        TCI4Keldysh.compress_PartialCorrelator_pointwise(Gp, svd_kernel; tolerance=tolerance, unfoldingscheme=:interleaved)
                    else
                        TCI4Keldysh.compress_reg_PartialCorrelator_pointwise(Gp, svd_kernel; tolerance=tolerance, unfoldingscheme=:interleaved)
                    end

        maxdiff = maxdiff_qtt_ref(R, qtt, refval)       
        @test maxdiff < 5.0 * tolerance
    end

    # FullCorrelator with dummy data
    function test_compress_FullCorrelator_pointwise1(npt::Int, svd_kernel=false)
        tolerance = 1.e-8
        R = 5
        GF = TCI4Keldysh.multipeak_correlator_MF(npt, R; beta=1000.0, nωdisc=8)
        refval = TCI4Keldysh.precompute_all_values(GF)
        qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(GF, svd_kernel; tolerance=tolerance, unfoldingscheme=:interleaved)

        maxdiff = maxdiff_qtt_ref(R, qtt, refval)
        @test maxdiff < 5.0 * tolerance
    end

    function test_FullCorrelator_batch(;R::Int=5, tolerance::Float64=1.e-8, beta::Float64=10.0)
        GF = TCI4Keldysh.dummy_correlator(4, R; beta=beta)[1]
        data_ref = TCI4Keldysh.precompute_all_values(GF)
        tt, fevbatch = TCI4Keldysh.compress_FullCorrelator_batched(GF, true; tolerance=tolerance) 

        qtt = QuanticsTCI.QuanticsTensorCI2{ComplexF64}(tt, fevbatch.grid, fevbatch.qf) 

        slice = fill(1:2^R, 3)
        data = TCI4Keldysh.QTT_to_fatTensor(qtt, slice)

        maxref = maximum(abs.(data_ref))
        @test maximum(abs.(data .- data_ref[slice...])) / maxref <= 5 * tolerance
    end


    # FullCorrelator with actual data
    function test_compress_FullCorrelator_pointwise2(npt::Int, svd_kernel=false; tolerance=1.e-8, channel="t")
        R = 5
        GF = TCI4Keldysh.dummy_correlator(npt, R; beta=10.0, channel=channel)[1]
        refval = TCI4Keldysh.precompute_all_values(GF)
        qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(GF, svd_kernel; tolerance=tolerance, unfoldingscheme=:interleaved)

        maxdiff = maxdiff_qtt_ref(R, qtt, refval)
        @test maxdiff < 5.0 * tolerance
    end


    test_compress_tucker_pointwise(2)
    test_compress_tucker_pointwise(3)
    test_compress_tucker_pointwise(3, true)
    test_compress_tucker_pointwise(4)

    test_compress_PartialCorrelator_pointwise(;npt=2, perm_idx=1, ano=false)
    test_compress_PartialCorrelator_pointwise(;npt=2, perm_idx=2, ano=true)
    test_compress_PartialCorrelator_pointwise(true; npt=2, perm_idx=2, ano=true)
    test_compress_PartialCorrelator_pointwise(;npt=3, perm_idx=2, ano=false)
    test_compress_PartialCorrelator_pointwise(;npt=3, perm_idx=6, ano=true)
    test_compress_PartialCorrelator_pointwise(true; npt=3, perm_idx=6, ano=true)
    test_compress_PartialCorrelator_pointwise(;npt=4, perm_idx=3, ano=false)
    test_compress_PartialCorrelator_pointwise(;npt=4, perm_idx=13, ano=true)
    test_compress_PartialCorrelator_pointwise(true; npt=4, perm_idx=13, ano=true)

    for svd_kernel in [true,false]
        test_compress_FullCorrelator_pointwise1(2, svd_kernel)
        test_compress_FullCorrelator_pointwise1(3, svd_kernel)
        test_compress_FullCorrelator_pointwise1(4, svd_kernel)
    end
    printstyled("CHANNEL t\n"; color=:green)
    test_compress_FullCorrelator_pointwise2(4, true; tolerance=1.e-5, channel="t")
    test_compress_FullCorrelator_pointwise2(3, true; tolerance=1.e-5, channel="t")
    test_compress_FullCorrelator_pointwise2(2, true; tolerance=1.e-5, channel="t")
    printstyled("CHANNEL a\n"; color=:green)
    test_compress_FullCorrelator_pointwise2(4, true; tolerance=1.e-5, channel="a")
    test_compress_FullCorrelator_pointwise2(3, true; tolerance=1.e-5, channel="a")
    # test_compress_FullCorrelator_pointwise2(2, true; tolerance=1.e-5, channel="a")
    printstyled("CHANNEL p\n"; color=:green)
    test_compress_FullCorrelator_pointwise2(4, true; tolerance=1.e-8, channel="p")
    test_compress_FullCorrelator_pointwise2(3, true; tolerance=1.e-8, channel="p")
    test_compress_FullCorrelator_pointwise2(2, true; tolerance=1.e-8, channel="p")
end

@testset "Keldysh: PSF -> Correlator @ pointwise TCI" begin
    
    function test_FullCorrEvaluator_KF(npt::Int, iK::Int)
        # create correlator
        addpath = npt==4 ? "4pt" : ""
        PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg", addpath)
        D = npt-1
        Ops = TCI4Keldysh.dummy_operators(npt)
        T = TCI4Keldysh.dir_to_T(PSFpath)

        ωmax = 1.0
        R = 4
        ωs_ext = TCI4Keldysh.KF_grid(ωmax, R, D)
        ωconvMat = if npt==4
                TCI4Keldysh.channel_trafo("t")
            elseif npt==3
                TCI4Keldysh.channel_trafo_K2("t", false)
            else
                TCI4Keldysh.ωconvMat_K1()
            end
        γ, sigmak = TCI4Keldysh.default_broadening_γσ(T)
        KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops;
            T=T,
            ωs_ext=ωs_ext,
            flavor_idx=1,
            ωconvMat=ωconvMat,
            sigmak=sigmak,
            γ=γ,
            name="Kentucky fried chicken",
            estep=50
            )

        KFev = TCI4Keldysh.FullCorrEvaluator_KF_single(KFC, iK)
        function KFC_(idx::Vararg{Int,N}) where {N}
            return TCI4Keldysh.evaluate(KFC, idx...; iK=iK)        
        end

        for _ in 1:30
            idx = rand(1:2^R, D)
            @test isapprox(KFC_(idx...), KFev(idx...); atol=1.e-11)
        end


        KFev2 = TCI4Keldysh.FullCorrEvaluator_KF(KFC)
        function KFC2_(idx::Vararg{Int,N}) where {N}
            return TCI4Keldysh.evaluate_all_iK(KFC, idx...)
        end

        for _ in 1:30
            idx = rand(1:2^R, D)
            refval = vec(KFC2_(idx...))
            totest_nocut = vec(KFev2(Val{:nocut}(), idx...))
            totest_cut = vec(KFev2(idx...))
            @test isapprox(norm(refval .- totest_cut), 0.0; atol=1.e-11)
            @test isapprox(norm(refval .- totest_nocut), 0.0; atol=1.e-11)
        end
    end


    function test_compress_FullCorrelator_KF(npt::Int, iK; R=4, tolerance=1.e-5, channel::String="t")
        addpath = npt==4 ? "4pt/" : ""
        PSFpath = joinpath(TCI4Keldysh.datadir(), "siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg", addpath)
        D = npt-1
        Ops = TCI4Keldysh.dummy_operators(npt)
        T = TCI4Keldysh.dir_to_T(PSFpath)

        ωmax = 1.0
        ωs_ext = TCI4Keldysh.KF_grid(ωmax, R, D)
        ωconvMat = if npt==4
                TCI4Keldysh.channel_trafo(channel)
            elseif npt==3
                TCI4Keldysh.channel_trafo_K2(channel, false)
            else
                TCI4Keldysh.ωconvMat_K1()
            end
        γ, sigmak = TCI4Keldysh.default_broadening_γσ(T)
        KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops;
            T=T,
            ωs_ext=ωs_ext,
            flavor_idx=1,
            ωconvMat=ωconvMat,
            sigmak=sigmak,
            γ=γ,
            name="Kentucky fried chicken",
            estep=50
            )

        # reference
        data_ref = TCI4Keldysh.precompute_all_values(KFC)[fill(Colon(),D)..., TCI4Keldysh.KF_idx(iK,D)...]
        maxref = maximum(abs.(data_ref))

        # TCI
        qtt = TCI4Keldysh.compress_FullCorrelator_pointwise(KFC, iK; tolerance=tolerance, unfoldingscheme=:fused)
        qttval = TCI4Keldysh.QTT_to_fatTensor(qtt, Base.OneTo.(fill(2^R, D)))

        slice = fill(1:2^R, D)
        error = maximum(abs.(qttval .- data_ref[slice...]) ./ maxref)
        @test error <= 3.0*tolerance
    end

    test_FullCorrEvaluator_KF(2, 2)
    test_FullCorrEvaluator_KF(3, 3)
    test_FullCorrEvaluator_KF(4, 6)

    test_compress_FullCorrelator_KF(4, 2)
    test_compress_FullCorrelator_KF(4, 13)
    test_compress_FullCorrelator_KF(3, 3)
    test_compress_FullCorrelator_KF(3, 6)
    test_compress_FullCorrelator_KF(2, 2)
    test_compress_FullCorrelator_KF(2, 4)
end

@testset "Miscellaneous: PSF -> Correlator @ pointwise TCI" begin
    
    function test_tucker_cut()
        R = 5 
        npt = 4
        GF = TCI4Keldysh.multipeak_correlator_MF(4, R; beta=2000.0, nωdisc=20)
        exact_data = TCI4Keldysh.precompute_all_values(GF)

        cutoff = 1.e-8
        tucker_cutoff = 1.e-7
        fev = TCI4Keldysh.FullCorrEvaluator_MF(GF, true; cutoff=cutoff, tucker_cutoff=tucker_cutoff)

        GFmax = maximum(abs.(exact_data))
        errors = zeros(Float64, 2^(R*(npt-1)))
        cc = 1
        for w in Iterators.product(fill(1:2^R, npt-1)...)
            error = abs(fev(w...) - fev(Val{:nocut}(),w...)) / GFmax
            errors[cc] = error
            cc += 1
        end
        @test maximum(errors) < tucker_cutoff
    end

    test_tucker_cut()
end