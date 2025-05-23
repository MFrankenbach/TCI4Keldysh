@testset "QTCI PSF" begin

 

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

    # QTCI broadenedPsf:
    qmesh = collect(1:size(Acont)[1])
    qtt , ranks , errors  = quanticscrossinterpolate(Float64, broadenedPsf, ntuple(i -> length(qmesh), 2); tolerance=1e-8, maxiter=400)
    qtt2, ranks2, errors2 = quanticscrossinterpolate(Acont; tolerance=1e-8, maxiter=400)
    #[any(isnan.(qtt.tci.T[i])) for i in 1:2*R]      # check for NaN's

    qttdata = qtt[:,:]
    qtt2data= qtt2[:,:]
    inputdata = (broadenedPsf[:,:])[1:end-1, 1:end-1]

    @test maximum(abs.((qttdata .- inputdata))) < 1.e-4     # WHY DOESN'T THIS WORK PROPERLY?
    #@test maximum(abs.((qttdata .- qtt2data)))  < 1.e-13
    #@test maximum(abs.((qtt2data .- inputdata)))  < 1.e-6

    # test slicing for qtts:
    @test qttdata ≈ qtt[:,:]
    @test qttdata[:,100] ≈ qtt[:,100]
    @test qttdata[100,:] ≈ qtt[100,:]
    @test qttdata[100,:] ≈ qtt[100,:]

    # test slicing for BroadenedPSF:
    @test (broadenedPsf[:,:])[1:end-1,1:end-1] ≈ Acont
end