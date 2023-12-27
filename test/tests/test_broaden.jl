@testset "broaden with log then lin: " begin

    function get_Adisc_δpeak(idx_ω′, Nωs_pos)

        
        ωdisc_min = 1.e-6
        ωdisc_max = 1.e4
        ωdisc = exp.(range(log(ωdisc_min); stop=log(ωdisc_max), length=Nωs_pos))
        ωdisc = [reverse(-ωdisc); 0.; ωdisc]
        Adisc = zeros(Nωs_pos*2+1,1)
        Adisc[Nωs_pos + 1 + idx_ω′] = 1.
        
        return ωdisc, Adisc
    end;

    begin
        Nωs_pos = 1000
        σ = 2.
        sigmab = [σ]
        g = 1.
        tol = 1.e-14
        estep = 100
        emin = 1e-5; emax = 1e5;
        Lfun = "FD" 
        is2sum = true
        verbose = false

        ### compare with broadened Dirac-δ peak:
        idx_ω′1 = 700
        ωdisc, Adisc1 = get_Adisc_δpeak(idx_ω′1, Nωs_pos)
        Δωdisc = TCI4Keldysh.get_ω_binwidths(ωdisc)
        idx_ω′2 = -600
        _, Adisc2 = get_Adisc_δpeak(idx_ω′2, Nωs_pos)
        α = 0.3
        Adisc3 = α * Adisc1 + (1. - α) * Adisc2
        Nωcont_pos = 10000
        ωcont = get_ωcont(1000., Nωcont_pos)
        Δωcont = TCI4Keldysh.get_ω_binwidths(ωcont)
        Acont = zeros(Nωcont_pos*2+1,1)


        #println("ωdisc1 = ", ωdisc[Nωs_pos+1 + idx_ω′1])
        #println("ωdisc2 = ", ωdisc[Nωs_pos+1 + idx_ω′2])
    end

    ωcont_new, Acont_new = TCI4Keldysh.getAcont(ωdisc, Adisc1, sigmab, g; ωcont=ωcont, tol=tol, Lfun=Lfun, verbose=verbose)
    @test TCI4Keldysh.quadtrapz(ωcont_new, Acont_new[:,1]) ≈ 1 rtol=1.e-2
    
    ωcont3, Acont3 = TCI4Keldysh.getAcont(ωdisc, Adisc3, sigmab, g; ωcont=ωcont, tol=tol, Lfun=Lfun, verbose=verbose)
    @test TCI4Keldysh.quadtrapz(ωcont3, Acont3[:,1]) ≈ 1 rtol=1.e-2
    
end