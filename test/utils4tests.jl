function get_Adisc_δpeak_mp(idx_ω′s, Nωs_pos, D; 
    ωdisc_min = 1.e-6,
    ωdisc_max = 1.e4
    )
    
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