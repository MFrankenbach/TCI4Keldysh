# convolution kernels for broadening via δ*Adisc
function logGaussian_SLG_fct(ω, ω′; σ=1.)
    return exp(-((log(ω) - log(ω′) + σ^2/4.)/σ)^2) / (sqrt(pi)*σ*ω′)
end

function logGaussian_CLG_fct(ω, ω′; σ=1.)
   #return exp(-((log(ω) - logm)/σ)^2 - σ^2/4.) / (sqrt(pi)*σ*ω)
    return exp(-((log(ω) - log(ω′)         )/σ)^2) / (sqrt(pi)*σ*ω′)
end


@testset "logGauss broadening of Dirac-δ peaks: " begin

    function get_Adisc_δpeak_posω(idx_ω′, Nωs_pos, D)

    
        ωdisc_min = 1.e-6
        ωdisc_max = 1.e4
        ωdisc = exp.(range(log(ωdisc_min); stop=log(ωdisc_max), length=Nωs_pos))
        Adisc = zeros(Nωs_pos,D)
        view(Adisc, idx_ω′,:) .= 1.
        
        return ωdisc, Adisc
    end;

    begin
        Nωs_pos = 1000
        σ = 2.
        D = 2
        sigmab = ones(D).*σ
        tol = 1.e-14
        estep = 100
        emin = 1e-5; emax = 1e5;
        Hfun = "SLG" 
        is2sum=false

        ### compare with broadened Dirac-δ peak:
        idx_ω′1 = 100
        ωdisc, Adisc1 = get_Adisc_δpeak_posω(idx_ω′1, Nωs_pos, D)
        idx_ω′2 = 300
        _, Adisc2 = get_Adisc_δpeak_posω(idx_ω′2, Nωs_pos, D)
        α = 0.3
        Adisc3 = α * Adisc1 + (1. - α) * Adisc2
    end
    
    ωcont1, Δωcont1, Acont1 = TCI4Keldysh.getAcont_logBroaden(ωdisc, Adisc1, sigmab, tol, Hfun, emin, emax, estep, is2sum)
    @test (x -> logGaussian_SLG_fct.(x, ωdisc[idx_ω′1]; σ=σ)).(ωcont1) ≈ Acont1[:,1] rtol=1e-14
    
    ### compare with sum of broadened Dirac-δ peaks:

    ωcont2, Δωcont2, Acont2 = TCI4Keldysh.getAcont_logBroaden(ωdisc, Adisc3, sigmab, tol, Hfun, emin, emax, estep, is2sum)
    @test (x -> α * logGaussian_SLG_fct(x, ωdisc[idx_ω′1]; σ=σ) + (1. - α) * logGaussian_SLG_fct(x, ωdisc[idx_ω′2]; σ=σ)).(ωcont2) ≈ Acont2[:,1] rtol=1e-14

    # same with CLG: 
    Hfun = "CLG" 
    ωcont3, Δωcont3, Acont3 = TCI4Keldysh.getAcont_logBroaden(ωdisc, Adisc1, sigmab, tol, Hfun, emin, emax, estep, is2sum)
    @test (x -> logGaussian_CLG_fct.(x, ωdisc[idx_ω′1]; σ=σ)).(ωcont3) ≈ Acont3[:,1] rtol=1e-14
    ωcont4, Δωcont4, Acont4 = TCI4Keldysh.getAcont_logBroaden(ωdisc, Adisc3, sigmab, tol, Hfun, emin, emax, estep, is2sum)
    @test (x -> α * logGaussian_CLG_fct(x, ωdisc[idx_ω′1]; σ=σ) + (1. - α) * logGaussian_CLG_fct(x, ωdisc[idx_ω′2]; σ=σ)).(ωcont4) ≈ Acont4[:,1] rtol=1e-14

end


