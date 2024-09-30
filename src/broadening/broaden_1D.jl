"""
    getAcont(ωdisc::Vector{Float64}, Adisc::Matrix{Float64}, sigmak::Vector{Float64}, γ::Float64; kwargs...)

Broadens 1-dimensional discrete spectrum.

# Arguments:
1. ωdisc   ::Vector{Float64}:   Logarithimic frequency bins. 
                                Here the original frequency values from the differences
                                b/w energy eigenvalues are shifted to the closest bins. 
2. Adisc   ::Matrix{Float64},   Spectral function in layout |ωdisc|x|sigmak|.
3. sigmak  ::Vector{Float64},   Sensitivity of logarithmic position of spectral
                                weight to z-shift, i.e. |d[log(|omega|)]/dz|. These values will
                                be used to broaden discrete data. (sigma_{ij} or sigma_k in
                                Lee2016.)
4. γ       ::Float64            Parameter for secondary linear broadening kernel. (γ in Lee2016.)
                                    
# Keyword arguments
* emin    ::Float64 ()= 1e-12): Minimum absolute value of frequency grid. Set this
                                be equal to or smaller than the minimum of finite elements of
                                ωdisc, to properly broaden the spectral weights at frequencies
                                lower than 'emin'. The spectral weights binned at frequencies
                                smaller that emin are *not* broadened by the primary logarithmic
                                broadening; they are broadened only by the secondary linear
                                broadening.
* emax    ::Float64 (= 1e4):    Maximum absoulte value of frequency grid.
* estep   ::Int (= 200) :       Number of frequency grid points per decade, i.e.,
                                between a frequency and the frequency times 10 (e.γ., between 1 and
                                10).
* isw0    ::Bool (= false) :    If true, the result frequency grid 'ωcont' contains
                                zero frequency. If false, the grid has only finite frequencies.
                                (Default: false)
* ωcont   ::Vector{Float64} (= zeros(0)) :   
                                Frequency grid to be used for the result Acont.
                                First, the logarithmic and linear broadenings are applied by using
                                the frequency grid defined by emin, emax, and estep. Then the
                                result Acont is obtained by inter-/extra-polating for the broadened
                                curve with respect to this optional input of frequency grid.
                                (Default: not used)
* alphaz  ::Float64 (= 1.) :    Overall factor of broadening. (\alpha_z in Lee2016.)
* smin    ::Float64 (= 0.) :    Minimum broadening width (= sigmak*alphaz, not bare sigmak). 
                                (Default : 3/estep) 
* smax    ::Float64 (= Inf) :   Maximum boradening width (= sigmak*alphaz, not bare sigmak).
* Hfun    ::String (= "SLG") :  Name of the functional form of the primary
                                logarithmic broadening kernel. There are three possibilities:
    * "CLG" : centered log. Gaussian.
    * "SLG" : symmetric log. Gaussian.
    * "G"   : regular Gaussian.
* Lfun    ::String (= "FD") :   Name of the functional form of the secondary
                                linear broadening kernel. There are two possibilities:
    * "FD" : derivative of Fermi-Dirac distribution function.
    * "G"  : regular Gaussian.
    * "L"  : Lorentzian.
* A0      ::Vector{Float64} (= zeros(0)): 
                                Discrete weight as delta function at w = 0. This
                                weight is broadened by the secondary linear broadening kernel. If
                                the option 'sum' is set as false, 'A0' should be a row vector whose
                                n-th element corresponds to the n-th column of Adisc and Acont (see
                                'sum' below).
                                (Default: 0 if 'sum',true (default); zeros(1,length(sigmak)) if
                                'sum',false)
* tol     ::Float64 (= 1e-14) : Minimum value of (spectral weight)*(value of
                                broadening kernel) to consider. The spectral weights whose
                                contribution to the curve Acont are smaller than tol are not
                                considered. Also the far-away tails of the broadening kernel, whose
                                resulting contribution to the curve are smaller than tol, are not
                                considered also.
* is2sum  ::Bool (= true) :     If false, Acont is a matrix whose columns are the
                                broadening of the corresponding columns of Adisc. If true, Acont is
                                the sum of such columns.
* verbose ::Bool (= false) :    Show details.

# Returns
1. ωcont ::Vector{Float64} :    Logarithimic frequency grid.
2. Acont ::Vector{Float64} :    Smoothened spectral function.

# Issues
1. rediscretization might lead to discretization artifacts in final Acont
2. Sensitivity to choice of grid (linear / logarithmic)
3. Currently no support for zshifts

"""
function getAcont(
    ωdisc   ::Vector{Float64},  
    Adisc   ::Matrix{Float64},  
    sigmak  ::Vector{Float64},  
    γ       ::Float64           
    ; 
    emin    ::Float64   = 1e-12,
    emax    ::Float64   = 1e4,  
    estep   ::Int       = 200,  
    isw0    ::Bool      = false,
    ωcont   ::Vector{Float64}   = zeros(0),   
    alphaz  ::Float64   = 1.,   
    smin    ::Float64   = 0.,   
    smax    ::Float64   = Inf,  
    Hfun    ::String    = "SLG",
    Lfun    ::String    = "FD", 
    A0      ::Vector{Float64}   = zeros(0),   
    tol     ::Float64   = 1e-14,
    is2sum  ::Bool      = false, 
    verbose ::Bool      = false 
    )

    #######################################
    ### parse/check function arguments: ###
    #######################################

    isCLG = Hfun == "CLG"
    isSLG = Hfun == "SLG"
    ocin = copy(ωcont)
    #if !(ωcont==[])
    #    throw(ArgumentError("ωcont currently not supported."))
    #end

    emin = abs(emin)
    emax = abs(emax)
    estep = abs(estep)
    tol = abs(tol)
    if !(emin > 0)
        throw(ArgumentError("emin should be positive and finite."))
    elseif !(emax > emin)
        throw(ArgumentError("emax should be larger than emin."))
    elseif tol <= 0
        throw(ArgumentError("tol should be positive and finite."))
    elseif estep < 1
        throw(ArgumentError("estep should be equal to or larger than 1."))
    end

    if isempty(Adisc) || isempty(ωdisc) || isempty(sigmak)
        throw(ArgumentError("Adisc, ωdisc, and sigmak must not be empty."))
    elseif estep != round(estep)
        throw(ArgumentError("estep must be an integer."))
    elseif length(ωdisc) != size(Adisc, 1)
        throw(ArgumentError("Input dimensions do not match; length(ωdisc) != size(Adisc, 1)."))
    elseif length(sigmak) != size(Adisc, 2)
        throw(ArgumentError("Input dimensions do not match; length(sigmak) != size(Adisc, 2)."))
    elseif !(Hfun in ["CLG", "SLG", "G"])
        throw(ArgumentError("Hfun must be 'CLG', 'SLG', or 'G'."))
    elseif !(Lfun in ["FD", "G", "L"])
        throw(ArgumentError("Lfun must be 'FD', 'G', or 'L'."))
    elseif length(alphaz) > 1
        throw(ArgumentError("length(alphaz) > 1."))
    elseif length(γ) > 1
        throw(ArgumentError("length(γ) > 1."))
    end

    emin = abs(emin)
    emax = abs(emax)
    estep = abs(estep)
    tol = abs(tol)
    if !(emin > 0)
        throw(ArgumentError("''emin'' should be positive and finite."))
    elseif !(emax > emin)
        throw(ArgumentError("''emax'' should be larger than''emin''."));
    elseif tol <= 0
        throw(ArgumentError("''tol'' should be positive and finite."));
    elseif estep < 1
        throw(ArgumentError("''estep'' should be equal to or larger than 1."));
    end

    # sort input sigmak and Adisc w/ the increasing order of sigmak
    if any(sigmak .< 0)
        println("WRN: some of ''sigmak'' is negative; take absolute values.");
    end

    sigmab = alphaz .* sigmak # multiply input sigmak with factor alphaz

    if smin <= 0.
        smin = 3 / estep
    end

    sigmab[sigmab .< smin] .= smin
    sigmab[sigmab .> smax] .= smax

    if A0 == []
        A0 = zeros((length(sigmab) - 1) * (1 - is2sum) + 1)
    elseif length(A0) != ((length(sigmab) - 1) * (1 - is2sum) + 1)
        throw(ArgumentError("length(A0) is inconsistent with input sigmak and option is2sum."))
    end

    function display_info()
        strs = Vector{String}(undef,5);
        strs[1] = "Broadening (";
        strs[1] *= @match Hfun begin
            "CLG" => "Cent. log. Gauss.";
            "SLG" => "Symm. log. Gauss.";
            "G"   => "Regular Gauss.";
        end
        
        strs[1] *= " + ";
        strs[1] *= @match Lfun begin
            "FD" => "Diff. Fermi-Dirac dist.";
            "G"  => "Regular Gauss.";
            "L"  => "Lorentzian.";
        end
        strs[1] *= ")  (v.2020.04.20)";

        
        if isempty(ocin)
            strs[2] = "+-10.^(" * (@sprintf "%.3g" xs[1]) *":1/"* (@sprintf "%.3g" estep) *":"*(@sprintf "%.4g" xs[end]),")";
            if isw0
                strs[2] = "   omega   = [0, "*strs[2]*"]";
            else
                strs[2] = "   omega   = "*strs[2];
            end
        else
            strs[2] = "   omega   = "*(@sprintf "%.4g" minimum(ocin))*", ..., "*(@sprintf "%.4g" maximum(ocin))*"  (input)";
        end
        
        sigmab2 = sort(sigmab[:]); # for showing message

        if (length(sigmab2) > 2) && sum(abs(diff(sigmab2))) < 1e-10 # uniform sigmab
            strs[3] = "  sigmabar = ("*(@sprintf "%.4g" sigmab2[1])*":1/"*(@sprintf "%.4g" 1/(sigmab2[2]-sigmab2[1]))*":"*(@sprintf "%.4g" sigmab2[end])*")";
        elseif length(sigmab2) > 2
            strs[3] = "  sigmabar = ["*(@sprintf "%.4g" sigmab2[1])*", ..., "*(@sprintf "%.4g" sigmab2[end])*"]";
        elseif length(sigmab2) == 2  # 2 entries
            strs[3] = "  sigmabar = ["*(@sprintf "%.4g" sigmab2[1])*", "*(@sprintf "%.4g" sigmab2[end])*"]";
        else    # 1 entry
            strs[3] = "  sigmabar = "*(@sprintf "%.4g" sigmab2[1]);
        end

        strs[4] = "   alphaz  = "*(@sprintf "%.4g" alphaz)*", γ = "*(@sprintf "%.4g" γ);
        strs[5] = "   sum(A)  = "*(@sprintf "%.4g" real(Adiscsum));
        if imag(Adiscsum) != 0
            strs[5] *= (@sprintf "%+.4g" imag(Adiscsum))*"i";
        end

        println.(strs);
        return nothing
    end


    #####################################
    ### Core part of broadening code: ###
    #####################################

    Adiscsum = sum(Adisc) # total weight

    # create standard (log) frequency grid of no ωcont is supplied
    if ωcont == []
        # temporary frequency grid; includes 0 in the middle:
        xs = collect(log10(emin):1 / estep:log10(emax)) ./ estep
        ωcont_pos = 10 .^ xs
        ωcont = [reverse(-ωcont_pos); 0; ωcont_pos]
    else
        ωcont_pos = ωcont[ωcont.>0.]
    end
    # width of frequency bins
    Δωcont = [ωcont[2] - ωcont[1]; (ωcont[3:end] - ωcont[1:end-2]) ./ 2; ωcont[end] - ωcont[end-1]]
    # buffer for result
    Acont = zeros(length(ωcont), (length(sigmab) - 1) * (1 - is2sum) + 1)
    
    if verbose
        display_info()
    end

    DEBUG_BROADEN=false
    if isCLG || isSLG
        oks1 = ωdisc .>= ωcont_pos[1] # filter for positive frequencies
        if any(oks1)
            odtmp = ωdisc[oks1]
            Adtmp = Adisc[oks1, :]
            ots, dots, yts = getAcont_logBroaden(odtmp, Adtmp, sigmab; tol, Hfun, emin, emax, estep, is2sum)

            # eps_id = 2956
            # plot(ots, abs.(yts[:,eps_id]); xscale=:log10)
            # savefig("foo.png")
            yts_disc = yts .* dots # rediscretization

            if DEBUG_BROADEN
                broadened_weights = vec(sum(yts .* dots; dims=1))[oks1]
                weights_err = sum(abs.(broadened_weights .- 1.0)) / length(broadened_weights) 
                weights_maxerr = maximum(abs.(broadened_weights .- 1.0))

                # evaluate explicitly
                eps_id = size(yts, 2)
                eps = ωdisc[eps_id]
                σ=sigmab[1]
                Acont1peak = [1/(sqrt(pi)*σ*eps) * exp(-(log(abs(eps/om))/σ - σ/4)^2) for om in ots]
                @show maximum(abs.(Acont1peak .- yts[:,eps_id]))
                @show eps
                @show sum(Acont1peak .* dots .* ots)
                open("ots$(sigmab[1]).txt", "w") do f
                    write(f, "  w                     dw\n")
                    for i in eachindex(ots)
                        write(f, "$(ots[i])   $(dots[i])\n")
                    end
                end

                broadened_means = vec(sum(yts .* dots .* ots; dims=1))[oks1]
                plot(odtmp, broadened_means; xscale=:log10, yscale=:log10, label="broadened_means")
                savefig("foo.pdf")
                means_err = sum(abs.(broadened_means .- odtmp)) / length(broadened_means) 
                means_maxerr = maximum(abs.(broadened_means .- odtmp))
                display((broadened_means .- odtmp) ./ odtmp)

                printstyled("\n∑Ainter(+) = $(sum(yts .* dots)), $(size(Adtmp))\n"; color=:cyan)
                printstyled("  peakwise err(+) = $(weights_err), $(weights_maxerr)\n"; color=:cyan)
                printstyled("∑Ainter * ω(+) = $(sum(yts .* dots .* ots)), $(sum(odtmp))\n"; color=:cyan)
                printstyled("  peakwise err(+) = $(means_err), $(means_maxerr)\n\n"; color=:cyan)
                @show size(Acont)
                @show size(yts_disc)
            end

            getAcont_linBroaden(ots, dots, yts_disc, γ; ωcont, Δωcont, Acont, tol, Lfun)

            if DEBUG_BROADEN
                printstyled("\n∑Aend(+) = $(sum(Acont .* Δωcont)), $(size(Adtmp))\n"; color=:cyan)
                printstyled("∑Aend * ω(+) = $(sum(Acont .* Δωcont .* ωcont)), $(sum(odtmp))\n"; color=:cyan)
            end
        end

        oks2 = ωdisc .<= -ωcont_pos[1] # filter for negative frequencies
        if any(oks2)
            odtmp = -ωdisc[oks2] # negative -> positive
            Adtmp = Adisc[oks2, :]
            ots, dots, yts = getAcont_logBroaden(odtmp, Adtmp, sigmab; tol, Hfun, emin, emax, estep, is2sum)

            if DEBUG_BROADEN
                broadened_weights = vec(sum(yts .* dots; dims=1))[oks2]
                weights_err = sum(abs.(broadened_weights .- 1.0)) / length(broadened_weights) 
                weights_maxerr = maximum(abs.(broadened_weights .- 1.0))

                broadened_means = vec(sum(yts .* dots .* ots; dims=1))[oks2]
                means_err = sum(abs.(broadened_means .- odtmp)) / length(broadened_means) 
                means_maxerr = maximum(abs.(broadened_means .- odtmp))
                display((broadened_means .- odtmp) ./ odtmp)

                printstyled("\n∑Ainter(-) = $(sum(yts .* dots)), $(size(Adtmp))\n"; color=:cyan)
                printstyled("  peakwise err(-) = $(weights_err), $(weights_maxerr)\n"; color=:cyan)
                printstyled("∑Ainter * ω(-) = $(sum(yts .* dots .* ots)), $(sum(odtmp))\n"; color=:cyan)
                printstyled("  peakwise err(-) = $(means_err), $(means_maxerr)\n\n"; color=:cyan)
            end

            yts_disc = yts .* dots # rediscretization
            getAcont_linBroaden(-ots, dots, yts_disc, γ; ωcont, Δωcont, Acont, tol, Lfun) # -ots: return to negative frequency

            if DEBUG_BROADEN
                @show size(Acont), size(ωcont)
                printstyled("∑Aend(-) = $(sum(Acont .* Δωcont)), $(size(Adtmp))\n"; color=:cyan)
                printstyled("∑Aend * ω(-) = $(sum(Acont .* Δωcont .* ωcont)), $(sum(odtmp))\n\n"; color=:cyan)
            end
        end

        oks3 = (oks1.+oks2).==0
        
    ## should Gaussian broadening even be supported?
    #else # "G"
    #    oks = abs(ωdisc) .>= emin;
    #    if any(oks)
    #        yts = getAcont_GBroaden(ωdisc[oks],Adisc[oks,:],sigmab,tol,ωcont,docs,is2sum);
    #    else
    #        yts = zeros(size(Acont,2));
    #    end
    #    Acont = getAcont_linBroaden(ωcont,docs,yts,γ,ωcont,docs,Acont,tol,Lfun);
    #    
    #    oks3 = !oks;
    end

    if any(oks3) # for frequencies below emin: only broaden linearly, contains at least 0
        Adtmp = Adisc[oks3,:];
        if is2sum
            Adtmp = sum(Adtmp,dims=2);
        end
        getAcont_linBroaden([ωdisc[oks3];0],Δωcont[div(end+1,2)].+zeros(sum(oks3)+1),[Adtmp;A0'],γ;ωcont,Δωcont,Acont,tol,Lfun);
    end
    if DEBUG_BROADEN
        printstyled("---- getAcont done\n")
    end

    # Interpolate data on ocin
    #if !isempty(ocin)
    #    Acont = interp1(ωcont,Acont,ocin,'linear','extrap');
    #    ωcont = ocin;
    #elseif ~isw0
    #    % remove element at w = 0
    #    ωcont((end+1)/2) = [];
    #    Acont((end+1)/2,:) = [];
    #end    
    
    if verbose
        Acontsum = sum(Acont, dims=2)[:];
        # trapezoidal quadrature rule:
        #Acontsum = sum((Acontsum[2:end]+Acontsum[1:end-1]).*(ωcont[2:end]-ωcont[1:end-1]))/2;
        Acontsum = quadtrapz(ωcont, Acontsum)
        println("END : ∫ Acont(x) dx - ∑Adisc = "*(@sprintf "%+.3e" Acontsum-Adiscsum));
        #toc2(tobj,'-v'); ∫
    end
    return ωcont, Acont
end


