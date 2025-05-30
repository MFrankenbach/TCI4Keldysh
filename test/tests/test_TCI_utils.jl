using ITensors

@testset "TCI utils" begin


    begin
        filename = joinpath(dirname(@__FILE__), "data_PSF_2D.h5")
        f = h5open(filename, "r")
        Adisc = read(f, "Adisc")
        ωdisc = read(f, "ωdisc")
        close(f)

        Adisc = dropdims(Adisc,dims=tuple(findall(size(Adisc).==1)...))

        ### System parameters of SIAM ### 
        D = 1.
        # Keldysh paper:    u=0.5 OR u=1.0
        U = 1. / 20.
        T = 0.01 * U
        Δ = (U/pi)/0.5
        # EOM paper:        U=5*Δ
        #Δ = 0.1
        #U = 0.5*Δ
        #T = 0.01*Δ

        ### Broadening ######################
        #   parameters      σ       γ       #
        #       by JaeMo:   0.3     T/2     #
        #       by SSL:     0.6     T*3     #
        #   my choice:                      #
        #       u = 0.5:    0.6     T*3     #
        #       u = 1.0:    0.6     T*2     #
        #       u = 5*Δ:    0.6     T*0.5   #
        #####################################
        σ = 0.6
        sigmab = [σ]
        g = T * 1.
        tol = 1.e-14
        estep = 2048
        emin = 1e-6; emax = 1e4;
        Lfun = "FD" 
        is2sum = false
        verbose = false



        R = 6
        Nωcont_pos = 2^R # 512#
        ωcont = get_ωcont(D*0.5, Nωcont_pos)
        ωconts=ntuple(i->ωcont, ndims(Adisc))

        # Directly obtained broadened data
        ωcont, Acont = TCI4Keldysh.getAcont_mp(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
        Acont = Acont[1:end-1,1:end-1]
        # get functor which can evaluate broadened data pointwisely
        broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
    end


    qtt_orig, ranks_orig, errors_orig = quanticscrossinterpolate(
            broadenedPsf.center[end-2^6+1:end,end-2^6+1:end],
            tolerance=1e-8
        )  

    # test zero padding of qtt
    nonzeroinds_left=[1,2,2,1]
    N = 2
    D = 2
    qtt_padded = TCI4Keldysh.zeropad_QTCI2(qtt_orig; N, nonzeroinds_left)
    @test all(qtt_padded.tci.sitetensors[N*D+1:end] .== qtt_orig.tci.sitetensors)
    @test all(getindex.(argmax.(qtt_padded.tci.sitetensors[1:N*D]), 2) .== nonzeroinds_left)


    function test_eval_mps()
        N = 10
        s = siteinds(2, N)
        ld = TCI4Keldysh.worstcase_bonddim(dim.(s))
        mps = random_mps(Float64, s; linkdims=ld)

        fat_mps = ITensor(one(eltype(mps[1])))
        for m in mps
            fat_mps *= m
        end

        for _ in 1:5
            v = rand([1,2], N)
            e1 = TCI4Keldysh.eval(mps, v)
            e2 = TCI4Keldysh.eval_lr(mps, v, argmax(ld))
            ref = fat_mps[v...]

            @test isapprox(e1, ref; atol=1.e-12)
            @test isapprox(e2, ref; atol=1.e-12)

            mev = TCI4Keldysh.MPSEvaluator(mps)
            me = mev(v)
            @test isapprox(me, ref; atol=1.e-12)
        end
    end

    function test_delta_tensor()
        sitedim = 3
        linkdim = 4
        t1 = TCI4Keldysh.delta_tensor(Float64, 1, sitedim, linkdim)
        t2 = TCI4Keldysh.delta_tensor(Float64, 1, sitedim, linkdim)
        replaceinds!(t2, inds(t2), inds(t1))
        t3 = TCI4Keldysh.delta_tensor(Float64, 2, sitedim, linkdim)
        replaceinds!(t3, inds(t3), inds(t1))

        @test isapprox(scalar(t1*t2), linkdim*1.0; atol=1.e-10)
        @test isapprox(scalar(t1*t3), linkdim*0.0; atol=1.e-10)
    end

    function test_delta_tensor_end()
        t1 = TCI4Keldysh.delta_tensor_end(Float64, 2, 4, 3, true)
        t2 = TCI4Keldysh.delta_tensor_end(Float64, 2, 4, 3, true)
        replaceinds!(t2, inds(t2), inds(t1))
        t3 = TCI4Keldysh.delta_tensor_end(Float64, 1, 4, 3, true)
        replaceinds!(t3, inds(t3), inds(t1))

        @test isapprox(scalar(t1*t2), 3.0; atol=1.e-10) 
        @test isapprox(scalar(t1*t3), 0.0; atol=1.e-10) 

        tr = TCI4Keldysh.delta_tensor_end(Float64, 2, 4, 3, false)
        replaceinds!(tr, reverse(inds(tr)), inds(t1))
        @test isapprox(scalar(tr*t2), 3.0; atol=1.e-10) 
    end

    function test_saturate_bits()
        localdims = [2,3,2,2,3]
        fat = zeros(Float64, Tuple(localdims))
        for id in Iterators.product(Base.OneTo.(localdims)...)
            fat[id...] = sum(Tuple(id))
        end
        maxfat = maximum(abs.(fat))
        fat_sat12 = fat[2,1,:,:,:]
        fat_sat25 = fat[:,1,2,:,2]
        fat_sat245 = fat[:,1,:,2,1]

        tci, _, _ = TCI.crossinterpolate2(eltype(fat), i -> fat[i...], localdims; tolerance=1.e-8)

        tt12 = TCI4Keldysh.saturate_bits(tci.sitetensors, [1,2], [2,1])
        tt12fat = TCI4Keldysh.qtt_to_fattensor(tt12)
        tt25 = TCI4Keldysh.saturate_bits(tci.sitetensors, [2,3,5], [1,2,2])
        tt25fat = TCI4Keldysh.qtt_to_fattensor(tt25)
        tt245 = TCI4Keldysh.saturate_bits(tci.sitetensors, [2,4,5], [1,2,1])
        tt245fat = TCI4Keldysh.qtt_to_fattensor(tt245)

        @test maximum(abs.(fat_sat12 .- tt12fat))/maxfat < 2 * 1.e-8
        @test maximum(abs.(fat_sat25 .- tt25fat))/maxfat < 2 * 1.e-8
        @test maximum(abs.(fat_sat245 .- tt245fat))/maxfat < 2 * 1.e-8
    end

    function test_project_legs()
        localdims = [6,3,4]
        fat = zeros(Float64, Tuple(localdims))
        for id in Iterators.product(Base.OneTo.(localdims)...)
            fat[id...] = sum(Tuple(id))^2
        end
        tt, _, _ = TCI.crossinterpolate2(eltype(fat), i -> fat[i...], localdims; tolerance=1.e-8)

        rm_pos = [1,3]
        rm_val = [[1,4,5], [1,2]]
        tt_proj = TCI4Keldysh.project_legs(tt.sitetensors, rm_pos, rm_val)
        fat_proj = fat[[2,3,6], :, [3,4]]
        ttfat_proj = TCI4Keldysh.qtt_to_fattensor(tt_proj)
        @test norm(fat_proj .- ttfat_proj) < 1.e-8
    end

    function test_unfuse()
        block = reshape(collect(1:24), 3,4,2)
        unfused = TCI4Keldysh.unfuse(block, 2)
        @test size(unfused)==(3,2,2,2)
        @test block[1,1,2]==unfused[1,1,1,2]
        @test block[2,2,1]==unfused[2,2,1,1]
        @test block[3,3,1]==unfused[3,1,2,1]
        @test block[2,4,2]==unfused[2,2,2,2]
    end

    test_eval_mps()
    test_delta_tensor()
    test_delta_tensor_end()
    test_saturate_bits()
    test_project_legs()
    test_unfuse()
end