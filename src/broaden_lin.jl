"""
linear broadening of Adisc(w') with
 a) Gaussian
   f(w,w') = exp(-((w - w')/σ).^2)/(σ*sqrt(pi))

 b) Derivative of Fermi-Dirac function (Default)
   f(w,w') = 1/(1+cosh((w - w')/σ))*(1/2/σ)
           = - d(Fermi-Dirac function of temperature σ)/d(energy)

 c) Lorentzian
   f(w,w') = (1/pi/σ)/(1 + ((w - w')/σ)^2)

<Returns>
ωcont   ::Vector{Float64}   centers of frequency bins
Acont   ::Vector{Float64}   broadened spectral data

< Issues >
    a)  Sensitivity to choice of grid (linear / logarithmic)
        It seems that Gaussian ("G") and Fermi-Dirac ("FD") broadening prefer a linear grid for ωcont
        while lorentzian broadening prefers a logarithmic one.
"""
function getAcont_linBroaden(
    ωdisc   ::Vector{Float64},      
    Δωdisc  ::Vector{Float64}, 
    Adisc   ::Matrix{Float64}, 
    σ       ::Float64, 
    ωcont   ::Vector{Float64},  # continuous frequencies on which we want to evaluate A
    δωcont   ::Vector{Float64}, 
    Acont   ::Matrix{Float64}, 
    tol     ::Float64, 
    Lfun    ::String
    )
    # secondary linear broadening
    if !isempty(ωdisc) && !isempty(Adisc)
        isG = Lfun == "G"
        isFD = Lfun == "FD"

        if isG
            Atmp = (tol * sqrt(pi) * σ) ./ abs.(Adisc)
            okA = Atmp .> 1
            Atmp[okA] .= 1
            widtho = abs.(sqrt.(-log.(Atmp[.!okA])) .* σ)
        elseif isFD
            Atmp = (tol * 2 * σ) ./ abs.(Adisc)
            okA = Atmp .> 0.5
            Atmp[okA] .= 0.5
            widtho = abs.(acosh.(1 ./ Atmp[.!okA] .- 1) .* σ)
        else
            Atmp = (tol * pi * σ) ./ abs.(Adisc)
            okA = Atmp .> 1
            Atmp[okA] .= 1
            widtho = abs.(sqrt.(1 ./ Atmp[.!okA] .- 1) .* σ)
        end

        ωdisc = ωdisc .+ zeros(1, size(Adisc, 2))
        Δωdisc = Δωdisc .+ zeros(1, size(Adisc, 2))
        ids = reshape(zeros(Int, size(Adisc, 1)) .+ collect(1:size(Adisc, 2))', size(Adisc))

        Adisc = Adisc[.!okA]
        ωdisc = ωdisc[.!okA]
        Δωdisc = Δωdisc[.!okA]
        ids = ids[.!okA]

        if !isempty(Adisc)
            widtho = max.(widtho, Δωdisc)  # in case of widtho == 0
            #Acont .= 0. # reset to zero

            if isG
                g2 = max.(σ, Δωdisc ./ 5)
            elseif isFD
                g2 = max.(σ, Δωdisc ./ 25)
            else
                g2 = max.(σ, Δωdisc ./ 25)
            end

            for (ito, ωdisc_val) in enumerate(ωdisc)#1:length(ωdisc)
                okt = abs.(ωcont .- ωdisc[ito]) .<= widtho[ito]
                if any(okt)
                    if isG
                        ytmp = exp.(-((ωcont[okt] .- ωdisc_val) ./ g2[ito]).^2)
                    elseif isFD
                        ytmp = 1.0 ./ (1.0 .+ cosh.((ωcont[okt] .- ωdisc_val) ./ g2[ito]))
                    else    # Lorentzian
                        ytmp = (1 / (pi * g2[ito])) ./ (1.0 .+ ((ωcont[okt] .- ωdisc_val) ./ g2[ito]).^2)
                    end
                    ytmp = ytmp .* δωcont[okt]  # height -> weight
                    ytmp = ytmp ./ sum(ytmp)  # normalize
                    Acont[okt, ids[ito]] .+= (ytmp ./ δωcont[okt]) .* Adisc[ito]  # weight -> height
                end
            end
        end
    end

    return nothing
end