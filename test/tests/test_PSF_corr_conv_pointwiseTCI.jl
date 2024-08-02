using ITensors

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