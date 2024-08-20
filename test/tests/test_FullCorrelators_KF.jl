using Printf

@testset "formula G^R[S](ω)" begin
    # test formula for computing a retarded correlator from a spectral function
    # formula used in KF constructor for PartialCorrelator_reg

    N = 1000
    ωdisc = collect( -N:N) / 10

    Adisc = 1 ./ (ωdisc.^2 .+ 1) ./ π       #   Lorentzian
    GR = -im * π * (TCI4Keldysh.hilbert_fft(Adisc))



    @test TCI4Keldysh.maxabs(real.(GR) - ωdisc ./ (ωdisc.^2 .+ 1)) < 1e-2
    @test TCI4Keldysh.maxabs(imag.(GR) +     1 ./ (ωdisc.^2 .+ 1)) < 1e-14
    #plot(ωdisc, [real.(GR), imag.(GR), ωdisc ./ (ωdisc.^2 .+ 1), -1. ./ (ωdisc.^2 .+ 1)], labels=["Re(GR)" "Im(GR)" "ω/(ω^2+1)" "-1/(ω^2+1)"], xlims=[-10, 10])

end

@testset "FullCorrelator Keldysh" begin
    
    """
    Make sure pointwise and block evaluation yield the same result
    """
    function test_pointwise_vs_block_eval_KF()
        PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/4pt/")
        npt = 4
        D = npt-1
        Ops = ["F1", "F1dag", "F3", "F3dag"]
        T = 10.0

        R = 3
        ωmin = -5.0
        ωmax = 5.0
        ωs_ext = ntuple(i -> collect(range(ωmin, ωmax; length=2^R)), D)
        ωconvMat = TCI4Keldysh.channel_trafo("a")
        γ, sigmak = TCI4Keldysh.default_broadening_γσ(T)
        KFC = TCI4Keldysh.FullCorrelator_KF(PSFpath, Ops; T=T, ωs_ext=ωs_ext, flavor_idx=1, ωconvMat=ωconvMat, sigmak=sigmak, γ=γ, name="Kentucky fried chicken")

        # block
        data_block = TCI4Keldysh.precompute_all_values(KFC)
        @show size(data_block)

        # pointwise
        data_pw = zeros(ComplexF64, vcat(collect(length.(ωs_ext)), [2^npt])...)
        num_eval = prod(size(data_pw))
        count = 0
        for id in Iterators.product(Base.OneTo.(size(data_pw))...)
            data_pw[id...] = TCI4Keldysh.evaluate(KFC, id[1:D]...; iK=id[end])
            count += 1
            if mod(count, 1000)==0
                @printf("%.2f percent of evaluations\n", count/num_eval * 100)
            end
        end

        data_pw = reshape(data_pw, size(data_block))
        @test maximum(abs.(data_pw .- data_block)) <= 1.e-12
    end

    """
    Test whether Keldysh correlator is linear in Adisc
    """
    function test_linearAdisc_KF(npt::Int)
        D = npt-1
        R = 4
        ωs_ext = TCI4Keldysh.KF_grid(-2.0, 2.0, R, D)
        γ=0.01
        sigmak=[0.2]

        KFC1 = TCI4Keldysh.multipeak_correlator_KF(ωs_ext, 1.0; sigmak=sigmak, γ=γ)
        KFC2 = TCI4Keldysh.multipeak_correlator_KF(ωs_ext, 3.0; sigmak=sigmak, γ=γ)

        data1 = TCI4Keldysh.precompute_all_values(KFC1)
        data2 = TCI4Keldysh.precompute_all_values(KFC2)

        # merge Adiscs from previous correlators
        Adiscs = [zeros(Float64, fill(4, D)...) for _ in 1:factorial(D+1)]
        ωdisc = [-3.0, -1.0, 1.0, 3.0]
        for A in Adiscs
            for ic in CartesianIndices(A)
                if all([i in [2,3] for i in Tuple(ic)])
                    A[ic] = 1.0
                elseif all([i in [1,4] for i in Tuple(ic)])
                    A[ic] = 1.0
                end
            end
        end

        KFC_sum = TCI4Keldysh.multipeak_correlator_KF(ωs_ext, Adiscs, ωdisc; sigmak=sigmak, γ=γ)
        data_sum = TCI4Keldysh.precompute_all_values(KFC_sum)

        @test size(data_sum)==size(data1)
        @test size(data_sum)==size(data2)
        @test maximum(abs.(data_sum .- (data1 .+ data2))) <= 1.e-11 # 1.e-12 can get violated
    end

    """
    Test index conversion retarded kernel -> keldysh index (eq. 63 Kugler et al)
    """
    function test_GR_to_GK()
        D = 3
        R = 4
        ωs_ext = TCI4Keldysh.KF_grid(-1.0, 1.0, R, D)

        KFC = TCI4Keldysh.multipeak_correlator_KF(ωs_ext, 1.0)
        perm_idx = 2
        GR_to_GK = KFC.GR_to_GK[:,:,perm_idx]
        @test size(GR_to_GK)==(D+1, 2^(D+1))

        # perm=[1,2,4,3]
        # k=9≡[1,1,1,2]
        @test GR_to_GK[:,1]==zeros(Int, D+1)
        k = 9
        expl = [0,0,1,0] * 2^(-D/2+0.5)
        @test GR_to_GK[:,k]==expl
    end

    test_linearAdisc_KF(2)
    test_linearAdisc_KF(3)
    test_linearAdisc_KF(4)
    test_GR_to_GK()
    test_pointwise_vs_block_eval_KF()
end