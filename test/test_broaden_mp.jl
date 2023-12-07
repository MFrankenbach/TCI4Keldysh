
@testset "broadening of multipoint functions" begin
    
    σ = 1.
    sigmab = [σ]
    g = 2.
    tol = 1.e-14
    estep = 160
    emin = 1e-6; emax = 1e2;
    Lfun = "FD" 
    is2sum = false
    verbose = false

    Nωs_pos = 100
    Nωcont_pos = 500
    ωcont = get_ωcont(1.5, Nωcont_pos)
    Δωcont = TCI4Keldysh.get_ω_binwidths(ωcont)



    ### compare with broadened Dirac-δ peak:        2p
    idx_ω′s1 = [40]
    ωdisc, Adisc1 = get_Adisc_δpeak_mp(idx_ω′s1, Nωs_pos, 1)
    Δωdisc = TCI4Keldysh.get_ω_binwidths(ωdisc)
    idx_ω′2 = [-40]
    _, Adisc2 = get_Adisc_δpeak_mp(idx_ω′2, Nωs_pos, 1)
    α = 0.3
    Adisc3 = α * Adisc1 + (1. - α) * Adisc2

    ωcont_new, Acont_new = TCI4Keldysh.getAcont_mp(ωdisc, Adisc3, sigmab, g; ωconts=ntuple(i->ωcont, ndims(Adisc3)), emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum)
    @test sum(Adisc3) - sum(Acont_new  .* Δωcont) ≈ 0 atol=1.e-3
    
    
    ### compare with broadened Dirac-δ peak:        3p
    Nωcont_pos = 50
    ωcont = get_ωcont(1.5, Nωcont_pos)
    Δωcont = TCI4Keldysh.get_ω_binwidths(ωcont)

    Nωs_pos = 30
    idx_ω′s1 = [10, 12]
    ωdisc, Adisc1 = get_Adisc_δpeak_mp(idx_ω′s1, Nωs_pos, 2)
    Δωdisc = TCI4Keldysh.get_ω_binwidths(ωdisc)
    idx_ω′2 = [-10, 13]
    _, Adisc2 = get_Adisc_δpeak_mp(idx_ω′2, Nωs_pos, 2)
    α = 0.3
    Adisc3 = α * Adisc1 + (1. - α) * Adisc2
    
    ωcont_new, Acont_new = TCI4Keldysh.getAcont_mp(ωdisc, Adisc3, sigmab, g; ωconts=ntuple(i->ωcont, ndims(Adisc3)), emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum)
    @test sum(Adisc3) - sum(Acont_new  .* Δωcont .* Δωcont') ≈ 0 atol=1.e-3

    Nωcont_pos = 10
    ωcont = get_ωcont(1.5, Nωcont_pos)
    Δωcont = TCI4Keldysh.get_ω_binwidths(ωcont)


    Nωs_pos = 10
    idx_ω′s1 = [2, 3, 4]
    ωdisc, Adisc1 = get_Adisc_δpeak_mp(idx_ω′s1, Nωs_pos, 3)
    Δωdisc = TCI4Keldysh.get_ω_binwidths(ωdisc)
    idx_ω′2 = [-2, -3, 4]
    _, Adisc2 = get_Adisc_δpeak_mp(idx_ω′2, Nωs_pos, 3)
    α = 0.3
    Adisc3 = α * Adisc1 + (1. - α) * Adisc2
    
    ωcont_new, Acont_new = TCI4Keldysh.getAcont_mp(ωdisc, Adisc3, sigmab, g; ωconts=ntuple(i->ωcont, ndims(Adisc3)), emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum)
    @test sum(Adisc3) - sum(Acont_new  .* Δωcont .* Δωcont' .* reshape(Δωcont, (1,1,Nωcont_pos*2+1))) ≈ 0 atol=1.e-3

end;




@testset "on-the-fly broadened Acont" begin
    f = h5open(joinpath(dirname(@__FILE__), "../data/3pPSF_example.h5"), "r")
    Adisc = read(f, "Adisc")
    ωdisc = read(f, "ωdisc")
    D = read(f, "D")
    U = read(f, "U")
    T = read(f, "T")
    Δ = read(f, "Δ")
    close(f)
    
    ### Broadening ######################
    #   parameters      σ       γ       #
    #       by JaeMo:   0.3     T/2     #
    #       by SSL:     0.6     T*3     #
    #####################################
    σ = 0.6
    sigmab = [σ]
    g = T * 5.
    tol = 1.e-14
    estep = 160
    emin = 1e-6; emax = 1e2;
    Lfun = "FD" 
    is2sum = false
    verbose = false

    Nωcont_pos = 256
    ωcont = get_ωcont(D*0.5, Nωcont_pos)
    ωconts=ntuple(i->ωcont, ndims(Adisc))

    #    println("parameters: \n\tT = ", T, "\n\tU = ", U, "\n\tΔ = ", Δ)

    @time ωcont, Acont = TCI4Keldysh.getAcont_mp(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
    broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);

    D = ndims(Adisc)
    idxs = collect(1:length(ωcont));
    
    for i in 1:1000
        idx = rand(idxs, D)
        @test broadenedPsf(idx...) ≈ Acont[idx...] rtol=1.e-10
    end
end;

