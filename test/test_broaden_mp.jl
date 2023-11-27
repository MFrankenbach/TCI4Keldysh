function get_Adisc_δpeak_mp(idx_ω′s, Nωs_pos, D)
    
        
    ωdisc_min = 1.e-6
    ωdisc_max = 1.e4
    ωdisc = exp.(range(log(ωdisc_min); stop=log(ωdisc_max), length=Nωs_pos))
    ωdisc = [reverse(-ωdisc); 0.; ωdisc]
    Adisc = zeros((ones(Int, D).*(Nωs_pos*2+1))...)
    Adisc[(Nωs_pos + 1 .+ idx_ω′s)...] = 1.
    
    return ωdisc, Adisc
end;

function get_ωcont(ωmax, Nωcont_pos)
    ωcont = collect(range(-ωmax, ωmax; length=Nωcont_pos*2+1))
    return ωcont
end

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
    ωcont, Δωcont, _ = get_ωcont(1.5, Nωcont_pos, 1)


    ### compare with broadened Dirac-δ peak:        2p
    idx_ω′s1 = [40]
    ωdisc, Adisc1 = get_Adisc_δpeak_mp(idx_ω′s1, Nωs_pos, 1)
    Δωdisc = TCI4Keldysh.get_ω_binwidths(ωdisc)
    idx_ω′2 = [-40]
    _, Adisc2 = get_Adisc_δpeak_mp(idx_ω′2, Nωs_pos, 1)
    α = 0.3
    Adisc3 = α * Adisc1 + (1. - α) * Adisc2

    ωcont_new, Acont_new = TCI4Keldysh.getAcont_mp(ωdisc, Adisc3, sigmab, g; ωcont, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum)
    @test sum(Adisc3) - sum(Acont_new  .* Δωcont) ≈ 0 atol=1.e-3
    
    
    ### compare with broadened Dirac-δ peak:        3p
    Nωcont_pos = 50
    ωcont, Δωcont, _ = get_ωcont(1.5, Nωcont_pos, 1)

    Nωs_pos = 30
    idx_ω′s1 = [10, 12]
    ωdisc, Adisc1 = get_Adisc_δpeak_mp(idx_ω′s1, Nωs_pos, 2)
    Δωdisc = TCI4Keldysh.get_ω_binwidths(ωdisc)
    idx_ω′2 = [-10, 13]
    _, Adisc2 = get_Adisc_δpeak_mp(idx_ω′2, Nωs_pos, 2)
    α = 0.3
    Adisc3 = α * Adisc1 + (1. - α) * Adisc2
    
    ωcont_new, Acont_new = TCI4Keldysh.getAcont_mp(ωdisc, Adisc3, sigmab, g; ωcont, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum)
    @test sum(Adisc3) - sum(Acont_new  .* Δωcont .* Δωcont') ≈ 0 atol=1.e-3

    Nωcont_pos = 10
    ωcont, Δωcont, _ = get_ωcont(1.5, Nωcont_pos, 1)

    Nωs_pos = 10
    idx_ω′s1 = [2, 3, 4]
    ωdisc, Adisc1 = get_Adisc_δpeak_mp(idx_ω′s1, Nωs_pos, 3)
    Δωdisc = TCI4Keldysh.get_ω_binwidths(ωdisc)
    idx_ω′2 = [-2, -3, 4]
    _, Adisc2 = get_Adisc_δpeak_mp(idx_ω′2, Nωs_pos, 3)
    α = 0.3
    Adisc3 = α * Adisc1 + (1. - α) * Adisc2
    
    ωcont_new, Acont_new = TCI4Keldysh.getAcont_mp(ωdisc, Adisc3, sigmab, g; ωcont, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum)
    @test sum(Adisc3) - sum(Acont_new  .* Δωcont .* Δωcont' .* reshape(Δωcont, (1,1,Nωcont_pos*2+1))) ≈ 0 atol=1.e-3

end;




@testset "on-the-fly broadened Acont" begin
    f = h5open("../data/3pPSF_example.h5", "r")
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

    #    println("parameters: \n\tT = ", T, "\n\tU = ", U, "\n\tΔ = ", Δ)

    @time ωcont, Acont = TCI4Keldysh.getAcont_mp(ωdisc, Adisc, sigmab, g; ωcont, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
    broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωcont, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);

    D = ndims(Adisc)
    idxs = collect(1:length(ωcont));
    
    for i in 1:1000
        idx = rand(idxs, D)
        @test broadenedPsf(idx...) ≈ Acont[idx...] rtol=1.e-10
    end
end;

