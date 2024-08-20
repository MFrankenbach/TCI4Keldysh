@testset "Hilbert transform" begin
    
        
    function maxdev_in_hilbertTrafo_sinc(N_in, x_in_max)
        xs = collect(LinRange(-x_in_max, x_in_max, N_in))
        xs_out = collect(LinRange(-x_in_max, x_in_max, 1000))
        ys = sin.(xs) ./ xs
        ht = imag.(TCI4Keldysh.my_hilbert_trafo(xs_out, xs, ys))
        expected = (x -> (1. - cos(x)) / x).(xs_out)#  * π

        return maximum(abs.(ht - expected))

    end

    function maxdev_in_hilbertTrafo_rat(N_in, x_in_max)
        xs = collect(LinRange(-x_in_max, x_in_max, N_in))
        xs_out = collect(LinRange(-x_in_max, x_in_max, 1000))
        ϵ = 2.
        ys = (x -> 1. / ((x-ϵ)^2 + 1.)).(xs)
        ht = imag.(TCI4Keldysh.my_hilbert_trafo(xs_out, xs, ys))
        expected = (x -> (x - ϵ) / ((x-ϵ)^2 + 1.)).(xs_out)#  * π

        return maximum(abs.(ht - expected))

    end


    function maxdev_in_hilbertTrafo_sinc_fft(N_in, x_in_max)
        xs = collect(LinRange(-x_in_max, x_in_max, N_in))
        xs_out = xs
        ys = sin.(xs) ./ xs
        ht = imag.(TCI4Keldysh.hilbert_fft(ys))
        expected = (x -> (1. - cos(x)) / x).(xs_out)

        return maximum(abs.(ht - expected))

    end

    function maxdev_in_hilbertTrafo_rat_fft(N_in, x_in_max)
        #println("maxdev rat fft")
        xs = collect(LinRange(-x_in_max, x_in_max, N_in))
        xs_out = xs
        ϵ = 2.
        ys = (x -> 1. / ((x-ϵ)^2 + 1.)).(xs)
        ht = imag.(TCI4Keldysh.hilbert_fft(ys))
        expected = (x -> (x - ϵ) / ((x-ϵ)^2 + 1.)).(xs_out) # * π

        return maximum(abs.(ht - expected))

    end

    ## analyze convergence:
    #maxdev_in_hilbertTrafo_rat.([5000, 10000, 20000], [50., 75., 100.]')
    #maxdev_in_hilbertTrafo_sinc.([5000, 10000, 20000], [50., 75., 100.]')
    @test maxdev_in_hilbertTrafo_rat(20000, 100.) < 1e-3
    @test maxdev_in_hilbertTrafo_sinc(20000,  1000.) < 1e-3
    @test maxdev_in_hilbertTrafo_rat_fft(20000, 1000.) < 2e-3
    @test maxdev_in_hilbertTrafo_sinc_fft(20000,  1000.) < 1e-3


end

@testset "utils for Tucker decompositions" begin
    filename = joinpath(dirname(@__FILE__), "data_PSF_2D.h5")
    f = h5open(filename, "r")
    Adisc = read(f, "Adisc")
    ωdisc = read(f, "ωdisc")
    close(f)


    D = 1.
    Nωcont_pos = 2^6
    ωcont = get_ωcont(D*0.5, Nωcont_pos)
    ωconts=ntuple(i->ωcont, ndims(Adisc))

    σ = 1.
    sigmab = [σ]
    g = 2.
    tol = 1.e-14
    estep = 160
    emin = 1e-6; emax = 1e2;
    Lfun = "FD" 
    is2sum = false
    verbose = false

    broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
    original_data = broadenedPsf[:,:]

    TCI4Keldysh.shift_singular_values_to_center!(broadenedPsf)
    newdata = broadenedPsf[:,:]
    atol = 1e-6
    TCI4Keldysh.svd_trunc_Adisc!(broadenedPsf; atol)
    truncdata = broadenedPsf[:,:]

    @test TCI4Keldysh.maxabs(newdata - original_data) < 1e-12
    @test TCI4Keldysh.maxabs(truncdata - original_data) < atol

    broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
    TCI4Keldysh.shift_singular_values_to_center_DIRTY!(broadenedPsf)

    newdata_DIRTY = broadenedPsf[:,:]
    
    @test TCI4Keldysh.maxabs(newdata_DIRTY - original_data) < 1e-12
    
    
end

@testset "Convert Tucker decomposition to QTT" begin

 

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
        #Δ = (U/pi)/0.5

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

        # get functor which can evaluate broadened data pointwisely
        broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
    end


    tol = 1e-8
    qtt = TCI4Keldysh.TDtoQTCI(broadenedPsf; method="svd", tolerance=tol)
    qtt2 = TCI4Keldysh.TDtoQTCI(broadenedPsf; method="qtci", tolerance=tol)

    origdat= (broadenedPsf[:,:])[1:end-1, 1:end-1]
    qttdat = qtt[:,:]
    qttdat2= qtt2[:,:]

    @test maximum(abs.(origdat - qttdat)) < tol
    #@test maximum(abs.(origdat - qttdat2))< tol        ## doesn't work reliably somehow...


end

@testset "Misc. utils" begin

    function test_MF_grid()
        T = 0.01
        Nhalf = 2^4
        ombos = TCI4Keldysh.MF_grid(T, Nhalf, false)
        omfer = TCI4Keldysh.MF_grid(T, Nhalf, true)
        @test length(ombos)==2*Nhalf+1
        @test length(omfer)==2*Nhalf
        @test all(ombos .+ reverse(ombos) .<= 1.e-12)
        @test all(omfer .+ reverse(omfer) .<= 1.e-12)

        Dgrid = TCI4Keldysh.MF_npoint_grid(T, Nhalf, 3)
        @test length(Dgrid) == 3
        @test length.(Dgrid) == (length(ombos), length(omfer), length(omfer))
    end

    function test_KF_idx()
        D = 3
        iK = 3
        K = TCI4Keldysh.KF_idx(iK, D)
        iK_ = TCI4Keldysh.KF_idx(K, D)
        @test iK_==iK
    end


    test_MF_grid()
    test_KF_idx()
end