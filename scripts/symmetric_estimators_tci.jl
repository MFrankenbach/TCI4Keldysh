using Plots
using Profile
using StatProfilerHTML

TCI4Keldysh.TIME() = false

function time_Γcore()
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    R = 5
    tolerance = 1.e-5
    ωconvMat = TCI4Keldysh.channel_trafo("t")
    beta = 100.0
    T = 1.0/beta

    # compile
    println("  Compile run...")
    foo = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved
        )
    println("  Time...")
    t = @elapsed begin
        Γcore = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved
        )
    end 
    x = sum(foo)
    x = sum(Γcore)
    println(" TIME: $t")
end

"""
Check Γcore values at the fringes of the grid and compare to tolerance.
"""
function max_Γcore_tail(;R::Int=5, tolerance::Float64=1.e-5, beta::Float64=10.0)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    ωconvMat = TCI4Keldysh.channel_trafo("t")
    T = 1.0/beta
    @time qtt = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved
        )
    Nhalf = 2^(R-1)
    frval = abs(qtt(1,Nhalf,Nhalf))
    yvals = [qtt(i, Nhalf, Nhalf) for i in 1:2^R]
    cenval = maximum(abs.(yvals))
    printstyled("Fringe value: $frval\n"; color=:blue)
    printstyled("Center max value: $cenval\n"; color=:blue)
    printstyled("Ratio: $(frval/cenval) (tol=$tolerance)\n"; color=:blue)

    # plot line section
    plot(1:2^R, abs.(yvals) ./ cenval; yscale=:log10, label=nothing, linewidth=2)
    ylabel!("|Γcore(ω)| / max|Γcore(ω)|")
    xlabel!("ω1")
    hline!(tolerance; label="tol", color=:red, linestyle=:dashed)
    savefig("gammacore_tail.png")
    return nothing
end

function Γcore_filename(mode::String, xmin, xmax, tolerance::Float64, beta::Float64)
    return "gammacore_timing_$(mode)_min=$(xmin)_max=$(xmax)_tol=$(TCI4Keldysh.tolstr(tolerance))_beta=$beta"
end

function time_Γcore_sweep(param_range, mode="R"; beta=10.0, tolerance=1.e-8)
    folder = "pwtcidata"
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    ωconvMat = TCI4Keldysh.channel_trafo("t")
    T = 1.0/beta
    times = []
    qttranks = []
    svd_kernel = true
    if mode=="R"
        Rs = param_range
        # prepare output
        d = Dict()
        d["times"] = times
        d["ranks"] = qttranks
        d["Rs"] = Rs
        d["tolerance"] = tolerance
        d["svd_kernel"] = svd_kernel
        d["numthreads"] = Threads.threadpoolsize()
        outname = Γcore_filename(mode, first(Rs), last(Rs), tolerance, beta)
        TCI4Keldysh.logJSON(d, outname, folder)

        for R in Rs
            t = @elapsed begin
                qtt = TCI4Keldysh.Γ_core_TCI_MF(
                PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved, verbosity=2
                )
            end 
            push!(times, t)
            push!(qttranks, TCI4Keldysh.rank(qtt))
            TCI4Keldysh.updateJSON(outname, "times", times, folder)
            TCI4Keldysh.updateJSON(outname, "ranks", qttranks, folder)
            println(" ===== R=$R: time=$t, rankk(qtt)=$(TCI4Keldysh.rank(qtt))")
        end
    else
        error("Invalid mode $mode")
    end
end

function RAM_usage_3D(R::Int)
    return 16 * 2^(3*R) / 1.e9
end

function plot_vertex_timing(param_range, mode="R"; beta=10.0, tolerance=1.e-6, plot_mem=true)
    folder = "pwtcidata"    
    filename = Γcore_filename(mode, minimum(param_range), maximum(param_range), tolerance, beta)
    data = TCI4Keldysh.readJSON(filename, folder)

    if mode=="R"
        Rs = convert.(Int, data["Rs"])
        RAM_usage = RAM_usage_3D.(Rs)
        times = convert.(Float64, data["times"])
        p = TCI4Keldysh.default_plot()

        plot!(p, Rs, times; marker=:diamond, color=:blue, label=nothing)
        xlabel!(p, "R")
        ylabel!(p, "Wall time [s]")
        title!(p, "Timings Γ core, β=$beta, tol=$tolerance")

        if plot_mem
            ptwin = twinx(p)
            plot!(ptwin, Rs, RAM_usage; marker=:circle, color=:black, linestyle=:dash, yscale=:log10, label=nothing)
            yticks!(ptwin, 10.0 .^ (round(Int, log10(minimum(RAM_usage))) : round(Int, log10(maximum(RAM_usage)))))
            ylabel!(ptwin, "Memory for dense corr. [GB]")
        end

        savefig(p, "vertextiming_beta=$(beta)_tol=$(round(Int,log10(tolerance))).png")
    end
end

function profile_Γcore()
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    R = 4
    tolerance = 1.e-5
    ωconvMat = 
        [
            0 -1  0;
            1  1  0;
            -1  0 -1;
            0  0  1;
        ]
    beta = 50.0
    T = 1.0/beta

    # compile
    println("  Compile run...")
    Γcore = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved
        )
    # profile
    Profile.clear()
    println("  Profiling...")
    Profile.@profile Γcore = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=1, tolerance=tolerance, unfoldingscheme=:interleaved
        )
    statprofilehtml()
end

"""
test & plot
"""
function test_Gamma_core_TCI_MF(; freq_conv="a", R=4, beta=50.0, tolerance=1.e-5)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    T = 1.0 / beta
    logtol = round(Int, log10(tolerance))
    spin = 1
    # spin = 2 # FAILS because F1F1dag Adisc lacks spin 2 component...

    ωconvMat = if freq_conv == "a"
        [
            0 -1  0;
            0  0  1;
            -1  0 -1;
            1  1  0;
        ]
    elseif freq_conv == "p"
        [
            0 -1  0;
            1  0 -1;
            -1  1  0;
            0  0  1;
        ]
    elseif freq_conv == "t"
        # t convention
        [
            0 -1  0;
            1  1  0;
            -1  0 -1;
            0  0  1;
        ]
    else
        error("Invalid frequency convention")
    end

    # compute
    TCI4Keldysh.@TIME Γcore = TCI4Keldysh.Γ_core_TCI_MF(
        PSFpath, R; T=T, ωconvMat=ωconvMat, flavor_idx=spin, tolerance=tolerance, unfoldingscheme=:interleaved, verbosity=2
        ) "Γcore @ TCI"
    @show TCI4Keldysh.rank(Γcore)

    # reference

        # grids
    ω_bos = TCI4Keldysh.MF_grid(T, 2^(R-1), false)
    ω_fer = TCI4Keldysh.MF_grid(T, 2^(R-1), true)
    ω_fer_int = TCI4Keldysh.MF_grid(T, 2^R, true)
    ωs_ext=(ω_bos, ω_fer, ω_fer)

        # sIE self-energy
    U = 0.05
    G        = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_aux    = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_QQ_aux = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "Q1dag"]; T, flavor_idx=spin, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_data      = TCI4Keldysh.precompute_all_values(G)
    G_aux_data  = TCI4Keldysh.precompute_all_values(G_aux)
    G_QQ_aux_data=TCI4Keldysh.precompute_all_values(G_QQ_aux)
    Σ_calc_sIE = TCI4Keldysh.calc_Σ_MF_sIE(G_QQ_aux_data, G_aux_data, G_aux_data, G_data, U/2)

        # Γ core
    TCI4Keldysh.@TIME refval = TCI4Keldysh.compute_Γcore_symmetric_estimator(
        "MF", PSFpath*"4pt/", Σ_calc_sIE; ωs_ext=ωs_ext, T=T, ωconvMat=ωconvMat, flavor_idx=spin
        ) "Γcore @ conventional"
    maxref = maximum(abs.(refval))
    logmaxref = round(Int, log10(maxref))

    # test
    testslice = fill(1:2^R, 3)
    test_qttval = reshape([Γcore(id...) for id in Iterators.product(testslice...)], length.(testslice)...)
    testdiff = abs.(refval[testslice...] .- test_qttval[testslice...]) ./ maxref
    printstyled("---- Γcore rank: $(TCI4Keldysh.rank(Γcore))\n"; color=:blue)
    printstyled("---- Maximum value Γcore: $(maxref)\n"; color=:green)
    printstyled("---- Maximum error: $(maximum(testdiff))\n"; color=:green)

    # plot
    slice = [1:2^R, 1:2^R, 2^(R-1)]
    qttval = reshape([Γcore(id...) for id in Iterators.product(slice...)], length.(slice)...)

    scfun = x -> log10(abs(x))
    heatmap(scfun.(qttval[slice[1:2]...]); clim=(logmaxref + logtol - 1, logmaxref + 2))
    savefig("gammacore.png")

    heatmap(scfun.(refval)[slice...]; clim=(logmaxref + logtol - 1, logmaxref + 2))
    savefig("gammacore_ref.png")

    diff = abs.(refval[slice...] .- qttval[slice[1:2]...]) ./ maxref
    heatmap(log10.(diff))
    savefig("diff.png")
end

"""
test & plot
"""
function test_K2_TCI(; channel="a", R=4, beta=50.0, tolerance=1.e-5, prime=false)
    PSFpath = joinpath(TCI4Keldysh.datadir(), "SIAM_u=0.50/PSF_nz=2_conn_zavg/")
    T = 1.0 / beta
    logtol = round(Int, log10(tolerance))
    flavor = 1

    printstyled("---- REFERENCE\n"; color=:blue)
        # grids
    ω_fer_int = TCI4Keldysh.MF_grid(T, 2^R, true)
    ωs_ext=TCI4Keldysh.MF_npoint_grid(T, 2^(R-1), 2)
        # sIE self-energy
    U = 0.05
    G        = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["F1", "F1dag"]; T, flavor_idx=flavor, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_aux    = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "F1dag"]; T, flavor_idx=flavor, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_QQ_aux = TCI4Keldysh.FullCorrelator_MF(PSFpath, ["Q1", "Q1dag"]; T, flavor_idx=flavor, ωs_ext=(ω_fer_int,), ωconvMat=reshape([ 1; -1], (2,1)), name="SIAM 2pG");
    G_data      = TCI4Keldysh.precompute_all_values(G)
    G_aux_data  = TCI4Keldysh.precompute_all_values(G_aux)
    G_QQ_aux_data=TCI4Keldysh.precompute_all_values(G_QQ_aux)
    Σ_calc_sIE = TCI4Keldysh.calc_Σ_MF_sIE(G_QQ_aux_data, G_aux_data, G_aux_data, G_data, U/2)


    (i,j) = TCI4Keldysh.merged_legs_K2(channel, prime)
    nonij = sort(setdiff(1:4, (i,j)))
    leg_labels = ("1", "1dag", "3", "3dag")
    op_labels = ("Q$i$j", leg_labels[nonij[1]], leg_labels[nonij[2]])
    K2ref = TCI4Keldysh.compute_K2r_symmetric_estimator(
        "MF", PSFpath, op_labels, Σ_calc_sIE;
        ωs_ext=ωs_ext, T=T, flavor_idx=flavor, ωconvMat=TCI4Keldysh.channel_trafo_K2(channel, prime)
    )

    if channel=="p"
        @assert maximum(abs.(K2ref)) <= 1.e-12
        return
    end

    # TCI
    printstyled("---- TCI\n"; color=:blue)
    K2qtt = TCI4Keldysh.K2_TCI(
        PSFpath, R, channel, prime;
        T=T, flavor_idx=flavor, tolerance=tolerance, unfoldingscheme=:interleaved
        )

    @show TCI4Keldysh.rank(K2qtt)

    # test & plot
    printstyled("---- PLOT\n"; color=:blue)
    testslice = fill(1:2^R, 2)
    test_qttval = TCI4Keldysh.QTT_to_fatTensor(K2qtt, testslice)
    maxref = maximum(abs.(K2ref[testslice...]))
    diff = abs.(test_qttval .- K2ref[testslice...])
    maxdiff = maximum(diff) / maxref
    @assert maxdiff < 5.0 * tolerance

    clim_max = ceil(Int, log10(maxref))
    clim = (clim_max + logtol - 1, clim_max)
    # heatmap(log10.(abs.(test_qttval)); clim=clim)
    heatmap(log10.(abs.(test_qttval)))
    savefig("K2.png")
    heatmap(log10.(abs.(K2ref)); clim=clim)
    savefig("K2_ref.png")
end

time_Γcore()
time_Γcore_sweep(5:8; beta=2000.0, tolerance=1.e-6)