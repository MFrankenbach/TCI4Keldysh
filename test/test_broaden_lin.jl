# convolution kernels for broadening via δ*Adisc
function gaussian(ω, ω′; σ=1.)
    return exp(-((ω - ω′)/σ)^2) / (sqrt(pi)*σ)
end

function diff_FD_distr(ω, ω′; σ=1.)
    return 1 / (1. + cosh((ω - ω′)/σ)) / 2. / σ
end
function lorentzian(ω, ω′; σ=1.)
    return 1. / pi / σ / (1 + ((ω - ω′)/σ)^2)
end

function get_Δω(ωs)
    Δωs = [ωs[2] - ωs[1]; (ωs[3:end] - ωs[1:end-2]) / 2; ωs[end] - ωs[end-1]]  # width of the frequency bin
    return Δωs 
end


function get_ωcont(ωmax, Nωcont_pos, D)
    ωcont = collect(range(-ωmax, ωmax; length=Nωcont_pos*2+1))
    Δωcont = get_Δω(ωcont)
    Acont = zeros(Nωcont_pos*2+1,D)
    return ωcont, Δωcont, Acont
end

@testset "linear broadening of Dirac-δ peaks: " begin

    function get_Adisc_δpeak(idx_ω′, Nωs_pos, D)
    
        
        ωdisc_min = 1.e-6
        ωdisc_max = 1.e4
        ωdisc = exp.(range(log(ωdisc_min); stop=log(ωdisc_max), length=Nωs_pos))
        ωdisc = [reverse(-ωdisc); 0.; ωdisc]
        Adisc = zeros(Nωs_pos*2+1,D)
        view(Adisc, Nωs_pos + 1 + idx_ω′, :) .= 1.
        
        return ωdisc, Adisc
    end;

    begin
        Nωs_pos = 1000
        D = 2
        σ = 2.
        tol = 1.e-14
        estep = 100
        emin = 1e-5; emax = 1e5;
        Lfun = "G" 
        is2sum=true

        ### compare with broadened Dirac-δ peak:
        idx_ω′1 = 700
        ωdisc, Adisc1 = get_Adisc_δpeak(idx_ω′1, Nωs_pos, D)
        Δωdisc = get_Δω(ωdisc)
        idx_ω′2 = -400
        _, Adisc2 = get_Adisc_δpeak(idx_ω′2, Nωs_pos, D)
        α = 0.3
        Adisc3 = α * Adisc1 + (1. - α) * Adisc2
        ωcont, Δωcont, Acont = get_ωcont(100., 1000, D)
    end
    
    TCI4Keldysh.getAcont_linBroaden(ωdisc, Δωdisc, Adisc1, σ, ωcont, Δωcont, Acont, tol, Lfun)
    @test TCI4Keldysh.maxabs((x -> gaussian.(x, ωdisc[Nωs_pos+1+idx_ω′1]; σ=σ)).(ωcont) - Acont[:,1]) < 1e-13

    Acont .= 0.
    TCI4Keldysh.getAcont_linBroaden(ωdisc, Δωdisc, Adisc3, σ, ωcont, Δωcont, Acont, tol, Lfun)
    @test (x -> α * gaussian(x, ωdisc[Nωs_pos+1+idx_ω′1]; σ=σ) + (1. - α) * gaussian(x, ωdisc[Nωs_pos+1+idx_ω′2]; σ=σ)).(ωcont) ≈ Acont[:,1] atol=1e-13

    # same with "FD":
    Lfun = "FD" 
    Acont .= 0.
    TCI4Keldysh.getAcont_linBroaden(ωdisc, Δωdisc, Adisc1, σ, ωcont, Δωcont, Acont, tol, Lfun)
    @test (x -> diff_FD_distr.(x, ωdisc[Nωs_pos+1+idx_ω′1]; σ=σ)).(ωcont) ≈ Acont[:,1] atol=1e-12

    Acont .= 0.
    TCI4Keldysh.getAcont_linBroaden(ωdisc, Δωdisc, Adisc3, σ, ωcont, Δωcont, Acont, tol, Lfun)
    @test (x -> α * diff_FD_distr(x, ωdisc[Nωs_pos+1+idx_ω′1]; σ=σ) + (1. - α) * diff_FD_distr(x, ωdisc[Nωs_pos+1+idx_ω′2]; σ=σ)).(ωcont) ≈ Acont[:,1] atol=1e-13

    # same with "L"
    ### linear grid did not work all too well... so here we use a logarithmic one which is slightly better
    Acont = copy(Adisc1)
    ωcont, Δωcont = ωdisc, Δωdisc

    Lfun = "L"
    Acont .= 0.
    TCI4Keldysh.getAcont_linBroaden(ωdisc, Δωdisc, Adisc1, σ, ωcont, Δωcont, Acont, tol, Lfun)
    @test TCI4Keldysh.maxabs((x -> lorentzian.(x, ωdisc[Nωs_pos+1+idx_ω′1]; σ=σ)).(ωcont) - Acont[:,1]) < 1.e-5

    Acont .= 0.
    TCI4Keldysh.getAcont_linBroaden(ωdisc, Δωdisc, Adisc3, σ, ωcont, Δωcont, Acont, tol, Lfun)
    @test TCI4Keldysh.maxabs((x -> α * lorentzian(x, ωdisc[Nωs_pos+1+idx_ω′1]; σ=σ) + (1. - α) * lorentzian(x, ωdisc[Nωs_pos+1+idx_ω′2]; σ=σ)).(ωcont) - Acont[:,1]) < 1e-5
end