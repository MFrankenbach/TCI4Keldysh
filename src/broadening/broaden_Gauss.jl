


"""
currently unused
"""
function getAcont_GBroaden(odisc, Adisc, sigmab, tol, ocont, docs, is2sum)

    error("This function is completely unreliable. -> Needs to go through testing.")

    # primary broadening: Gaussian broadening
    #   L(w,w') = exp(-((w-w')/sigmab/w')^2) /(sqrt(pi)*sigmab*|w'|)

    sigmab = sigmab'  # column vector -> row vector

    Atmp = (tol * sqrt(pi)) .* (odisc .* sigmab ./ abs(Adisc))
    okA = Atmp .> 1
    Atmp[okA] .= 1

    widtho = abs.(sqrt.(abs.(-log.(Atmp))) .* sigmab)

    omat = odisc .+ zeros(1, length(sigmab))
    smat = zeros(length(odisc), 1) .+ sigmab
    ids = zeros(length(odisc), 1) .+ (1:length(sigmab))

    omat[okA] .= []
    smat[okA] .= []
    widtho[okA] .= []
    Adisc[okA] .= []
    ids[okA] .= []

    yts = zeros(length(ocont), length(sigmab))

    for ito in 1:length(omat)
        okt = abs.(ocont .- omat[ito]) .<= widtho[ito]
        if any(okt)
            ytmp = -(ocont[okt] .- omat[ito]) ./ (omat[ito] * smat[ito]).^2
            ytmp = exp.(ytmp) / (sqrt(pi) * smat[ito] * abs(omat[ito]))
            yts[okt, ids[ito]] .= yts[okt, ids[ito]] .+ (ytmp .* docs[okt]) * Adisc[ito]  # height -> weight
        end
    end

    if is2sum
        yts = sum(yts, dims=2)
    end
end
