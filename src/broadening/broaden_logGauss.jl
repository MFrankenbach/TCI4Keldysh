"""
    getAcont_logBroaden(ωdisc::Vector{Float64}, Adisc::Matrix{Float64}, sigmab::Vector{Float64}; kwargs...)

Performs log-Gaussian broadening of discrete spectral data Adisc(w') --> Acont(w) := ∫dw' L(w,w')Adisc(w') with the broadening kernels
1.  Symmetric logarithmic Gaussian ("SLG") (sign(w) == sign(w'))  (*Default*)   \n
    L(w,w') = exp(-(log(w'/w)/sigmab - sigmab/4)^2)         /(sqrt(pi) * sigmab * abs(w')) \n
    CONSERVES: MEAN
2.  Centered logarithmic Gaussian ("CLG") (sign(w) == sign(w')) \n
    L(w,w') = exp(-(log(w'/w)/sigmab)^2) /(sqrt(pi) * sigmab * abs(w')) \n
    CONSERVES: MODE

# Arguments
1. ωdisc   ::Vector{Float64}:      centers of frequency bins
2. Adisc   ::Matrix{Float64}:      spectral data in format |#ωdisc|x|#sigmab|
3. sigmab  ::Vector{Float64}:      broadening parameter(s) - see explanation

# Keyword arguments
* Hfun    ::String (="SLG"): determines type of broadening kernel - see explanation
* emin    ::Float64 (=1e-2): suggestion for smallest positive energy scale (for continous frequency)
* emax    ::Float64 (=1e+2): suggestion for largest  positive energy scale (for continous frequency)
* estep   ::Int (=256):      determines number of bins (for continous frequency)
* tol     ::Float64 (=1e-14):Minimum value of (spectral weight)*(value of
                            broadening kernel) to consider. The spectral weights whose
                            contribution to the curve Acont are smaller than tol are not
                            considered. Also the far-away tails of the broadening kernel, whose
                            resulting contribution to the curve are smaller than tol, are not
                            considered also.
* is2sum  ::Bool (=true):    sum over all sigmab-variations?

# Returns
1. ωcont   ::Vector{Float64}   centers of frequency bins (pos. ωs on logarithmic grid)
2. Δωcont  ::Vector{Float64}   widths of frequency bins; mimics trapezoidal rule, i.e., ∫dω A(ω) ~ Δωcont ⋅ Acont
3. Acont   ::Vector{Float64}   broadened spectral data

# Comments
For comparison with Matlab code: sigmab = sigmak * alphaz
"""
function getAcont_logBroaden(
    ωdisc   ::Vector{Float64},
    Adisc   ::Matrix{Float64},
    sigmab  ::Vector{Float64}
    ;
    Hfun    ::String    ="SLG",
    emin    ::Float64   =1e-2, 
    emax    ::Float64   =1e+2, 
    estep   ::Int       =256,  
    tol     ::Float64   =1e-14,
    is2sum  ::Bool      =true  
    )

    if !all([length(ωdisc), length(sigmab)] .== size(Adisc))
        throw(DimensionMismatch("|ωdisc|x|sigmab| = "*(@sprintf "%s" length(ωdisc)) *" x "* (@sprintf "%s" length(sigmab))*", but Adisc has size "*string(size(Adisc))))
    end
    isCLG = Hfun == "CLG"

    function get_log10ωs_cont()
        # Determine the additional range of frequencies smaller than emin.
        # It is necessary since the log-Gaussian broadening of the spectral weights at
        # low-frequency bins results in the tail towards small frequency,
        # and the tail should be fully included in the frequency range.
        # Otherwise, the spectral weight of the truncated tail is missing in the
        # secondary linear broadening, which can lead to wrong results.
        
        Atmp = (tol * sqrt(pi)) * (ωdisc .* sigmab' ./ abs.(Adisc))
        if isCLG
            # Atmp = bsxfun(@times, Atmp, exp((sigmab.^2) / 4))
            Atmp = Atmp .* exp.((sigmab'.^2) / 4)
        end
        okA = Atmp .> 1 # mask to get higher frequencies
        Atmp[okA] .= 1
        
        # widthx: the maximum half-range of the kernel in the log-frequency
        # widthx = bsxfun(@times, abs(sqrt(-log(Atmp)), sigmab)
        widthx = abs.(sqrt.(-log.(Atmp)) .* sigmab')

        # xtmp = bsxfun(@minus, log(ωdisc), widthx)
        xtmp = log.(ωdisc) .- widthx
        if Hfun == "SLG"
            # xtmp = bsxfun(@minus, xtmp, (sigmab.^2) / 4)
            xtmp = xtmp .- ((sigmab.^2) / 4)'
        end

        # xmin: the log of the minimum frequency of the additional range
        xmin = minimum(xtmp) / log(10)
        # modify xmin so that the grid of logωcont matches with the grid of the result
        xdiff = (xmin - log10(emin)) * estep
        xmin = xmin - (xdiff - floor(xdiff)) / estep

        # temporary frequency grid
        logωcont = collect((xmin):1/estep:(log10(emax)))   # exponents of frequencies (base 10); increasing, column vector
        return okA, logωcont, widthx
    end
    okA, log10ωcont, widthx = get_log10ωs_cont()

    # printstyled("  (emax=$(emax))\n"; color=:blue)
    # printstyled("  (log10ωcont[end]=$(log10ωcont[end]))\n"; color=:blue)

    if length(log10ωcont) > 1
        logωcont = log10ωcont * log(10)  # exponents of frequencies (base e)
        ωcont = 10 .^ log10ωcont  # center of the frequency bin
        Δωcont = get_ω_binwidths(ωcont)  # width of the frequency bin
        Acont = zeros(length(ωcont), length(sigmab))  # temporary result; weights are to be binned

        odiscmat = ωdisc .+ zeros(1, length(sigmab))
        smat = zeros(length(ωdisc), 1) .+ sigmab'
        ids = zeros(Int, length(ωdisc), 1) .+ (1:length(sigmab))'

        logωdiscmat = log.(ωdisc)      ## matrix of (shifted) discrete logarithmic frequencies == centers of frequency bins
        if Hfun == "SLG"
            logωdiscmat = logωdiscmat .- ((sigmab.^2) / 4)'
        else     # "CLG"
            logωdiscmat = logωdiscmat .+ zeros(1, length(sigmab))
        end

        
        odiscmat = odiscmat[.!okA]
        smat = smat[.!okA]
        logωdiscmat = logωdiscmat[.!okA] 
        widthx = widthx[.!okA]
        Adisc = Adisc[.!okA]
        ids = ids[.!okA]
        
        # core part of broadening code:
        #     1. Centered logarithmic Gaussian (sign(w) == sign(w'))
        #   L(w,w') = exp(-(log(w'/w)/sigmab)^2) * exp(-sigmab^2/4) /(sqrt(pi)*sigmab*abs(w'))
        #     2. Symmetric logarithmic Gaussian (sign(w) == sign(w'))  (* Default)
        #   L(w,w') = exp(-(log(w'/w)/sigmab - sigmab/4)^2) /(sqrt(pi)*sigmab*abs(w'))
        for (ito, odisc_val) in enumerate(odiscmat) # 1:length(odiscmat)
            okt = abs.(logωcont .- logωdiscmat[ito]) .<= widthx[ito] # mask to get frequencies ωcont within bin
            if any(okt)
                Acont_tmp = -((logωdiscmat[ito] .- logωcont[okt]) ./ smat[ito]) .^ 2
                if isCLG
                    Acont_tmp .-= (smat[ito] .^ 2 / 4)
                end
                Acont_tmp = exp.(Acont_tmp) * (Adisc[ito] / (sqrt(pi) * smat[ito] * odisc_val))

                Acont[okt, ids[ito]] .+= Acont_tmp #.* Δωcont[okt]  # height -> weight
            end
        end

        if is2sum
            Acont = sum(Acont, dims=2)
        end
    else
        ωcont = []
        Acont = []
    end

    return ωcont, Δωcont, Acont
end