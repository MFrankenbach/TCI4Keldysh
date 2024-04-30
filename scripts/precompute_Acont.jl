using Revise
using MAT
using TCI4Keldysh
TCI4Keldysh.TIME() = true
TCI4Keldysh.DEBUG() = true
TCI4Keldysh.VERBOSE() = true

using CairoMakie
using BenchmarkTools
#using Plots
using HDF5
using Plots
using LinearAlgebra


function get_ωcont(ωmax, Nωcont_pos)
    ωcont = collect(range(-ωmax, ωmax; length=Nωcont_pos*2+1))
    return ωcont
end


function save_Acont(filename, ωdisc, Adisc, ωcont, Acont)
    f_plot = h5open(filename*".h5", "w")
    f_plot["Adisc"] = Adisc
    f_plot["ωdisc"] = ωdisc
    f_plot["Acont"] = Acont
    f_plot["ωcont"] = ωcont
    close(f_plot)
    return nothing
end



# 2p function
PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
Ops = ["Q12", "Q34"  ]

# 3p function
#PSFpath = "data/PSF_nz=2_conn_zavg/"
Ops = ["F1", "F1dag", "Q34"]
#Ops = ["F1dag", "F1", "Q34"]

# 4p function
PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/4pt/"
Ops = ["F1", "F1dag", "F3", "F3dag"]


begin
   
    Adisc = TCI4Keldysh.load_Adisc(PSFpath, Ops, 1)
    Adisc = dropdims(Adisc,dims=tuple(findall(size(Adisc).==1)...))
    ωdisc = TCI4Keldysh.load_ωdisc(PSFpath, Ops)
    
    ### System parameters of SIAM ### 
    D = 1.
    # Keldysh paper:    u=0.5 OR u=1.0
    U = 1. / 20.
    T = 0.01 * U
    Δ = (U/pi)/0.5
    # EOM paper:        U=5*Δ
    Δ = 0.1
    U = 0.5*Δ
    T = 0.01*Δ

    ### Broadening ######################
    #   parameters      σ       γ       #
    #       by JaeMo:   0.3     T/2     #
    #       by SSL:     0.6     T*3     #
    #   my choice:                      #
    #       u = 0.5:    0.6     T*3     #
    #       u = 1.0:    0.6     T*2     #
    #       u = 5*Δ:    0.6     T*0.5   #
    #####################################
    σ = 0.3
    sigmab = [σ]
    g = T * 0.5
    tol = 1.e-14
    estep = 512
    emin = 1e-6; emax = 1e4;
    Lfun = "FD" 
    is2sum = false
    verbose = false

    Rpos = 6
    R = Rpos+1
    Nωcont_pos = 2^Rpos
    #ωcont = get_ωcont(D*0.5, Nωcont_pos)
    ωcont = get_ωcont(D*2., Nωcont_pos)
    ωconts=ntuple(i->ωcont, ndims(Adisc))

    println("parameters: \n\tT = ", T, "\n\tU = ", U, "\n\tΔ = ", Δ)
end;


#@time ωcont, Acont = TCI4Keldysh.getAcont_mp(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
broadenedPsf_BCK = deepcopy(broadenedPsf)
broadenedPsf_shift_DIRTY = deepcopy(broadenedPsf)
broadenedPsf_shift_SVDed = deepcopy(broadenedPsf)

### Q: How can we shift singular values to the center? (Important for truncating data.)
TCI4Keldysh.shift_singular_values_to_center_DIRTY!(broadenedPsf_shift_DIRTY)
TCI4Keldysh.shift_singular_values_to_center!(      broadenedPsf_shift_SVDed)

function save_tucker(filename, dset_name, tucker; overwrite_dset=false)
    f = h5open(filename, "cw")

    if overwrite_dset
        g = create_group(f, dset_name)
    else
        try 
            g = open_group(f, dset_name)
        catch
            g = create_group(f, dset_name)
        end
    end

    g["Adisc"] = tucker.Adisc

    for ik in eachindex(tucker.Kernels)
        g["Kernel_"*string(ik)] = tucker.Kernels[ik]
    end

    close(f)

    return nothing
end

#save_tucker("data/tucker_comparison.h5", "original", broadenedPsf_BCK; overwrite_dset=false)
#save_tucker("data/tucker_comparison.h5", "dirty", broadenedPsf_shift_DIRTY; overwrite_dset=false)
#save_tucker("data/tucker_comparison.h5", "SVDed", broadenedPsf_shift_SVDed; overwrite_dset=false)
#plot(broadenedPsf_shift_SVDed.Kernels[1][259:end,50], xscale=:log10)



# look at singular values of original and modified kernel:
U, S, V = svd(broadenedPsf.Kernels[1])
Un, Sn, Vn = svd(broadenedPsf_shift_DIRTY.Kernels[1])
# compare singular values of original kernel δ(ω,ωdisc) and of δ(ω,ωdisc)|ωdisc|:

#plot([S, Sn], labels=["δ(ω,ωdisc)" "δ(ω,ωdisc)×|ωdisc|"],yaxis=:log10, title="singular values of broadening kernels", xlabel="i", ylabel="s_i", ylim=[1e-2,1e3]) # , xlim=[0,50]
fig = Figure(size = (400, 300));
ax1 = Axis(fig[1, 1],
    title="singular values of broadening kernels", 
    xlabel="i", 
    ylabel="s_i",
    yscale=log10,
    limits=(0, 50, 1e-5, 1e4))

l1 = lines!(ax1, S , yscale=log, label="δ(ω,ω′)")
l2 = lines!(ax1, Sn, yscale=log, label="δ(ω,ω′)*|ω′|")

axislegend(ax1)#, merge = merge, unique = unique)

fig

save("scripts/plots/singularvalues_QTCI4Adisc.pdf", fig)


# QTCI 3D Adisc: (Q: Are the Adisc compressible?)
using QuanticsTCI
import TensorCrossInterpolation as TCI
#R = 6
qmesh = collect(1:2^6);
tolerance = 1e-6;

qtt_orig, ranks_orig, errors_orig = quanticscrossinterpolate(
        broadenedPsf.Adisc[end-2^6+1:end,end-2^6+1:end,end-2^6+1:end],
        tolerance=tolerance
    )  
qttdat = qtt_orig[:,:,:]
TCI4Keldysh.maxabs(qttdat - broadenedPsf.Adisc[end-2^6+1:end,end-2^6+1:end,end-2^6+1:end])

qtt_DIRTY, ranks_DIRTY, errors_DIRTY = quanticscrossinterpolate(
        broadenedPsf_shift_DIRTY.Adisc[end-2^6+1:end,end-2^6+1:end,end-2^6+1:end],
        tolerance=tolerance
    )  
qttdat_DIRTY = qtt_DIRTY[:,:,:]
TCI4Keldysh.maxabs(qttdat_DIRTY - broadenedPsf_shift_DIRTY.Adisc[end-2^6+1:end,end-2^6+1:end,end-2^6+1:end])


qtt_SVDed, ranks_SVDed, errors_SVDed = quanticscrossinterpolate(
        broadenedPsf_shift_SVDed.Adisc[end-2^6+1:end,end-2^6+1:end,end-2^6+1:end],
        tolerance=tolerance
    )  
qttdat_SVDed = qtt_SVDed[:,:,:]
TCI4Keldysh.maxabs(qttdat_SVDed - broadenedPsf_shift_SVDed.Adisc[end-2^6+1:end,end-2^6+1:end,end-2^6+1:end])
worst_case = 2 .^ min.(1:3*6-1, 3*6-1:-1:1)




#plot([worst_case, TCI.linkdims.([qtt_orig.tci, qtt_DIRTY.tci, qtt_SVDed.tci])...], 
# labels=["worst case" "orig" "dirty" "SVD"], 
# title="link dimensions for tol="*string(tolerance), 
# xlabel="link")

fig = Figure(size = (600, 400));
ax1 = Axis(fig[1, 1],
    title="link dimensions for tol="*string(tolerance), 
    xlabel="link", 
    #yscale=log10,
    #limits=(0, 50, 1e-5, 1e4)
    )

l1 = lines!(ax1, worst_case, color="gray")
l2 = lines!(ax1, TCI.linkdims(qtt_orig.tci), label="original Adisc")
l3 = lines!(ax1, TCI.linkdims(qtt_DIRTY.tci), label="Adisc weighted by |ω′|")
l4 = lines!(ax1, TCI.linkdims(qtt_SVDed.tci), label="Adisc weighted by sᵢ from B")

axislegend(ax1)#, merge = merge, unique = unique)

fig
save("scripts/plots/linksdims_QTCI4Adisc.pdf", fig)



plot([ranks_orig, ranks_DIRTY, ranks_SVDed], labels=["orig" "dirty" "SVD"], yscale=:log10)
worst_case = 2 .^ min.(1:3R-1, 3R-1:-1:1)
plot([worst_case, TCI.linkdims.([qtt_orig.tci, qtt_DIRTY.tci, qtt_SVDed.tci])...], labels=["worst case" "orig" "dirty" "SVD"], title="link dimensions for tol="*string(tolerance), xlabel="link")

savefig("data/linksdims_QTCI4Adisc.pdf")

plot([qtt_orig.tci.pivoterrors, qtt_DIRTY.tci.pivoterrors,  qtt_SVDed.tci.pivoterrors], labels=["orig" "dirty" "SVD"], title="pivoterrors", yscale=:log10, ylabel="abs. error", xlabel="D_max")


f_plotdata = h5open("data/plotdata_QTCI4Adisc.h5", "w")
f_plotdata["linkdims_worstcase"] = worst_case
f_plotdata["linkdims_origAdisc"] = TCI.linkdims(qtt_orig.tci)
f_plotdata["linkdims_dirtyAdisc"] = TCI.linkdims(qtt_DIRTY.tci)
f_plotdata["linkdims_SVDedAdisc"] = TCI.linkdims(qtt_SVDed.tci)

f_plotdata["singularvalues_origKernel"] = S
f_plotdata["singularvalues_dirtyKernel"] = Sn

close(f_plotdata)




TCI4Keldysh.shift_singular_values_to_center!(broadenedPsf)

newdata = broadenedPsf[:,:,:]


# truncate data via higher-order SVD:
atol = 1e-6
broadenedPsf_new = TCI4Keldysh.svd_trunc_Adisc(broadenedPsf; atol)
truncdata = broadenedPsf_new[:,:,:]
broadenedPsf_new.sz

# check consistency:
TCI4Keldysh.maxabs(newdata - original_data)
TCI4Keldysh.maxabs(truncdata - original_data)


################## profiling runs:   ######################

## 1D: 
@benchmark for idxs in Iterators.product(1:10)#, 1:10)#, 1:10)
    broadenedPsf(idxs...)
end

# implementation with dot:
#BenchmarkTools.Trial: 10000 samples with 200 evaluations.
# Range (min … max):  409.715 ns …  16.428 μs  ┊ GC (min … max): 0.00% … 94.98%
# Time  (median):     431.000 ns               ┊ GC (median):    0.00%
# Time  (mean ± σ):   459.392 ns ± 316.784 ns  ┊ GC (mean ± σ):  1.73% ±  2.47%
#
#     ██                                                          
#  ▂▃████▅▃▃▁▁▂▄▅▄▅▅▃▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
#  410 ns           Histogram: frequency by time          641 ns <
#
# Memory estimate: 160 bytes, allocs estimate: 10.

# implementation with mapreduce:
#BenchmarkTools.Trial: 10000 samples with 5 evaluations.
# Range (min … max):  5.990 μs … 862.450 μs  ┊ GC (min … max): 0.00% … 97.32%
# Time  (median):     6.117 μs               ┊ GC (median):    0.00%
# Time  (mean ± σ):   7.200 μs ±  21.506 μs  ┊ GC (mean ± σ):  9.21% ±  3.09%
#
#  ▇█▅▂▁   ▃▃      ▁▁▂▁▁▁                                      ▁
#  ██████▇█████▇▆▇█████████▇▇▇▇██▇▇▇▇▆▆▇▇▇▇▆▆▆▆▆▅▅▅▅▅▅▄▄▄▅▅▅▅▅ █
#  5.99 μs      Histogram: log(frequency) by time      11.1 μs <
#
# Memory estimate: 9.22 KiB, allocs estimate: 80.
#

## 2D: 
@benchmark for idxs in Iterators.product(1:10, 1:10)#, 1:10)
    #idx = rand(axes(ωconts[1],1),2)
    broadenedPsf_BCK(Nωcont_pos .+idxs...)
end
@benchmark for idxs in Iterators.product(1:10, 1:10)#, 1:10)
    #idx = rand(axes(ωconts[1],1),2)
    broadenedPsf(Nωcont_pos .+idxs...)
end

@benchmark for idxs in Iterators.product(1:10, 1:10)#, 1:10)
    broadenedPsf(idxs...)
end

broadenedPsf(1,1)
@code_warntype broadenedPsf_BCK(1,1)

broadenedPsf.Kernels[1]
TCI4Keldysh.maxabs(broadenedPsf_BCK.Kernels[1][:,2])
TCI4Keldysh.maxabs(broadenedPsf_BCK.Kernels[1][end-3,:])

TCI4Keldysh.allconcrete(broadenedPsf)
TCI4Keldysh.allconcrete(broadenedPsf_BCK)

broadenedPsf.Kernels[1][:] .= 1.
#broadenedPsf.Kernels[1][:] .= 1.

broadenedPsf.Kernels[1] = ones(2,3)
#broadenedPsf.Kernels = ntuple(i -> ones(2,3), 2)

# old implementation:
#BenchmarkTools.Trial: 10000 samples with 1 evaluation.
# Range (min … max):  115.018 μs …   3.228 ms  ┊ GC (min … max): 0.00% … 94.09%
# Time  (median):     117.939 μs               ┊ GC (median):    0.00%
# Time  (mean ± σ):   134.699 μs ± 147.333 μs  ┊ GC (mean ± σ):  7.08% ±  6.18%
#
#  ▇█▆▄▂▁▅▃▂▂▂▁▂▁▁▂▁▁▁▁                                          ▁
#  ██████████████████████▇▇▆▆▇█▇▆█▇▇▆▆▅▅▆▆▅▅▆▅▅▅▅▆▅▆▆▆▆▄▅▅▅▅▄▄▅▅ █
#  115 μs        Histogram: log(frequency) by time        216 μs <
#
# Memory estimate: 209.38 KiB, allocs estimate: 1100.

# implementation with dot:
#BenchmarkTools.Trial: 10000 samples with 1 evaluation.
# Range (min … max):   9.461 μs … 35.978 μs  ┊ GC (min … max): 0.00% … 0.00%
# Time  (median):      9.838 μs              ┊ GC (median):    0.00%
# Time  (mean ± σ):   10.263 μs ±  1.049 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%
#
#   ▄▇██▇▅▂    ▁▅▆▆▆▅▅▄▃▂▁▁ ▁▁▁▁▁▁▁▁▁                          ▂
#  ▇███████▇▅▄▅███████████████████████▇▆▇▇▆▆▆▆▇▇▆▇▆▇▆▆▅▄▄▄▅▂▃▅ █
#  9.46 μs      Histogram: log(frequency) by time      13.6 μs <
#
# Memory estimate: 1.56 KiB, allocs estimate: 100.


@benchmark for idxs in Iterators.product(1:10, 1:10, 1:10)
    broadenedPsf(Nωcont_pos .+ idxs...)
end
@benchmark for idxs in Iterators.product(1:10, 1:10, 1:10)
    broadenedPsf_new(Nωcont_pos .+ idxs...)
end
@benchmark for idxs in Iterators.product(1:10, 1:10, 1:10)
    broadenedPsf_BCK(Nωcont_pos .+ idxs...)
end

@code_warntype broadenedPsf(1,1,1)
typeof(broadenedPsf)

# general implementation
#BenchmarkTools.Trial: 169 samples with 1 evaluation.
# Range (min … max):  21.369 ms … 50.246 ms  ┊ GC (min … max): 0.00% … 5.23%
# Time  (median):     24.913 ms              ┊ GC (median):    7.93%
# Time  (mean ± σ):   29.785 ms ±  8.566 ms  ┊ GC (mean ± σ):  6.32% ± 2.96%
#
#     ▁█▇▃██                                                    
#  ▄▅▇███████▄▁▁▄▃▁▁▁▃▃▃▄▁▁▁▁▄▁▃▃▁▁▁▄▄▁▁▃▁▃▄▄▄▁▁▄▄▄▇▇▄▁▁▃▅▄▄▄▄ ▃
#  21.4 ms         Histogram: frequency by time        47.2 ms <
#
# Memory estimate: 38.22 MiB, allocs estimate: 21000.

# implementation with dot
#BenchmarkTools.Trial: 881 samples with 1 evaluation.
# Range (min … max):  5.123 ms …   9.636 ms  ┊ GC (min … max): 0.00% … 29.06%
# Time  (median):     5.489 ms               ┊ GC (median):    0.00%
# Time  (mean ± σ):   5.665 ms ± 551.493 μs  ┊ GC (mean ± σ):  0.50% ±  2.99%
#
#    ▄█▄▁                                                       
#  ▃▇█████▇█▇▇▇▇▄▆▆▃▄▃▃▄▃▃▃▂▃▃▃▃▃▂▃▃▂▃▂▃▃▃▃▂▃▂▃▁▂▁▂▂▂▂▁▂▁▂▂▁▁▂ ▃
#  5.12 ms         Histogram: frequency by time        7.75 ms <
#
# Memory estimate: 625.00 KiB, allocs estimate: 2000.
## ==> time spent per point: 5.489 ms / 1000 = 5.489 μs

@benchmark broadenedPsf_BCK[:,:,:]
@benchmark broadenedPsf[:,:,:]
@benchmark broadenedPsf_new[:,:,:]
#BenchmarkTools.Trial: 22 samples with 1 evaluation.
# Range (min … max):  224.051 ms … 231.063 ms  ┊ GC (min … max): 4.99% … 6.02%
# Time  (median):     227.568 ms               ┊ GC (median):    6.21%
# Time  (mean ± σ):   227.618 ms ±   1.604 ms  ┊ GC (mean ± σ):  6.12% ± 0.63%
#
#  ▁          ▁     ▁  █ █  █ ▁ ▁ ▁▁ ▁ ▁█ ▁    ▁    ▁  ▁       ▁  
#  █▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁█▁▁█▁█▁▁█▁█▁█▁██▁█▁██▁█▁▁▁▁█▁▁▁▁█▁▁█▁▁▁▁▁▁▁█ ▁
#  224 ms           Histogram: frequency by time          231 ms <
#
# Memory estimate: 347.03 MiB, allocs estimate: 49.
## ==> time spent per point: 227.568 ms / prod(length.(ωconts)) = 0.0134 μs
2081 / prod(length.(ωconts))

plaindata = broadenedPsf[:,:,:]
@benchmark for idxs in Iterators.product(1:10, 1:10, 1:10)
    plaindata[idxs...]
end

#BenchmarkTools.Trial: 10000 samples with 1 evaluation.
# Range (min … max):  16.119 μs …  4.763 ms  ┊ GC (min … max): 0.00% … 98.92%
# Time  (median):     19.432 μs              ┊ GC (median):    0.00%
# Time  (mean ± σ):   20.304 μs ± 47.574 μs  ┊ GC (mean ± σ):  2.32% ±  0.99%
#
#  ▇▆▁▁▂▁▅▃▃▅█▅▆▅▂▁▄▄▃▃▂▁▁                                     ▂
#  ███████████████████████████▇██▇▇▇█▇█▇▇▆▇▆▆▇▆▆▆▇▇▇▅▆▆▆▅▇▅▅▆▆ █
#  16.1 μs      Histogram: log(frequency) by time      34.7 μs <
#
# Memory estimate: 15.62 KiB, allocs estimate: 1000.
## ==> time spent per point: 19.432 μs / 1000 = 0.0194 μs







# evaluation of a single point for a 1D PSF:
#BenchmarkTools.Trial: 10000 samples with 732 evaluations.
# Range (min … max):  166.949 ns …   5.607 μs  ┊ GC (min … max):  0.00% … 94.35%
# Time  (median):     175.589 ns               ┊ GC (median):     0.00%
# Time  (mean ± σ):   245.915 ns ± 451.711 ns  ┊ GC (mean ± σ):  20.25% ± 10.51%
#
#  █▄▃▂                                                          ▁
#  █████▅▄▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▄▅▆▅ █
#  167 ns        Histogram: log(frequency) by time        3.8 μs <
#
# Memory estimate: 800 bytes, allocs estimate: 5.

# evaluation of a single point for a K2 PSF:

# Nωcont_pos = 2^6:
#BenchmarkTools.Trial: 10000 samples with 10 evaluations.
# Range (min … max):  1.122 μs … 354.289 μs  ┊ GC (min … max): 0.00% … 97.90%
# Time  (median):     1.291 μs               ┊ GC (median):    0.00%
# Time  (mean ± σ):   1.608 μs ±   6.351 μs  ┊ GC (mean ± σ):  7.68% ±  1.95%
#
#  ▂▅▃▆█▇▅▃▃▂▂▂▂▁▃▃▂▃▄▃▃▃▂▂▂▂▂▂▁▁▁                             ▂
#  ██████████████████████████████████▇██▇▇█▇▇▇▆▇▇▆▅▇▆▅▆▅▅▄▅▆▅▄ █
#  1.12 μs      Histogram: log(frequency) by time      2.98 μs <
#
# Memory estimate: 2.09 KiB, allocs estimate: 11.

# Nωcont_pos = 2^7:
#BenchmarkTools.Trial: 10000 samples with 10 evaluations.
# Range (min … max):  1.105 μs … 309.568 μs  ┊ GC (min … max): 0.00% … 98.30%
# Time  (median):     1.203 μs               ┊ GC (median):    0.00%
# Time  (mean ± σ):   1.462 μs ±   5.378 μs  ┊ GC (mean ± σ):  7.17% ±  1.96%
#
#  ▄▇█▆▅▆▅▃▁    ▂▂▂▄▄▃▂▃▃▃▁▁▁▁                                 ▂
#  ██████████████████████████████▇█▇▆▇▆▇▇▆▇▆▇▆▆▅▅▅▅▄▆▅▅▅▆▆▆▆▅▆ █
#  1.1 μs       Histogram: log(frequency) by time      2.77 μs <
#
# Memory estimate: 2.09 KiB, allocs estimate: 11.

# evaluation of a single point for a K3 PSF:
#BenchmarkTools.Trial: 10000 samples with 1 evaluation.
# Range (min … max):  17.667 μs …  2.419 ms  ┊ GC (min … max): 0.00% … 95.21%
# Time  (median):     19.727 μs              ┊ GC (median):    0.00%
# Time  (mean ± σ):   24.513 μs ± 60.216 μs  ┊ GC (mean ± σ):  5.70% ±  2.52%
#
#  ▃█                                                           
#  ███▇▆▄▄▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▂ ▂
#  17.7 μs         Histogram: frequency by time        80.1 μs <
#
# Memory estimate: 39.14 KiB, allocs estimate: 21.



using BenchmarkTools

GKeldysh = TCI4Keldysh.FullCorrelator_KF(PSFpath, ["F1", "F1dag"  ]; flavor_idx=1, ωs_ext=(ωcont,), ωconvMat=reshape([1 ; -1], (2, 1)), name="SIAM G", sigmak=sigmab, γ=g, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
GKeldysh = TCI4Keldysh.FullCorrelator_KF(PSFpath, ["F1", "F1dag", "Q34"]; flavor_idx=1, ωs_ext=(ωcont,ωcont,), ωconvMat=[1 1; 0 -1; -1 0], name="SIAM 3pG", sigmak=sigmab, γ=g, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);

using BenchmarkTools

TCI4Keldysh.evaluate(GKeldysh, 1,1; iK=1)
TCI4Keldysh.evaluate_all_iK(GKeldysh, 1,1)

length(ωcont)

## evaluate Keldysh correlator:
gRtmp(idx) = TCI4Keldysh.evaluate(GKeldysh, 2, idx)
GRtmp_data = gRtmp.(collect(axes(ωcont)[1]))



## evaluate MF correlator:
N_MF = 1000
T = 3.
ω_bos = (collect(-N_MF:N_MF  ) * (2.)      ) * π * T
ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * π * T


GM = TCI4Keldysh.FullCorrelator_MF("data/PSF_nz=2_conn_zavg/3pt/", ["F1", "F1dag", "Q34"]; flavor_idx=1, ωs_ext=(ω_bos,ω_fer), 
ωconvMat=[ 0  1 ;
           1  0 ; 
          -1 -1], name="SIAM 3pG");
@benchmark GM_dat = TCI4Keldysh.precompute_all_values(GM)


#######################
### 3p functions:   ###
#######################

# N_MF = 10
#BenchmarkTools.Trial: 10000 samples with 1 evaluation.
# Range (min … max):  211.104 μs …   9.076 ms  ┊ GC (min … max):  0.00% … 59.47%
# Time  (median):     225.833 μs               ┊ GC (median):     0.00%
# Time  (mean ± σ):   282.134 μs ± 311.054 μs  ┊ GC (mean ± σ):  10.70% ± 10.19%
#
#  █▅▃▃▁▁▁                                                       ▁
#  █████████▇▅▄▅▃▃▁▁▁▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▃▁▁▁▁▁▁▃▁▁▃▁▁▁▃▃▁▄▃▅▄▆▅▆▆▇ █
#  211 μs        Histogram: log(frequency) by time       2.18 ms <
#
# Memory estimate: 768.56 KiB, allocs estimate: 303.

# N_MF = 100
#BenchmarkTools.Trial: 850 samples with 1 evaluation.
# Range (min … max):  4.181 ms … 22.531 ms  ┊ GC (min … max):  0.00% …  7.34%
# Time  (median):     5.771 ms              ┊ GC (median):    20.56%
# Time  (mean ± σ):   5.871 ms ±  1.798 ms  ┊ GC (mean ± σ):  12.08% ± 10.66%
#
#  █▂                                                          
#  ██▇▇▆▅▆▅▄▅▅▄▅▅▇▇▆▆▇▅▇▇▆▅▅▄▄▄▃▄▄▃▃▃▂▃▂▂▃▂▂▂▃▂▂▂▂▂▂▁▂▁▂▁▁▁▁▂ ▃
#  4.18 ms        Histogram: frequency by time        10.6 ms <
#
# Memory estimate: 22.52 MiB, allocs estimate: 336.

# N_MF = 1000
#BenchmarkTools.Trial: 4 samples with 1 evaluation.
# Range (min … max):  1.097 s …    1.353 s  ┊ GC (min … max): 10.36% … 7.80%
# Time  (median):     1.292 s               ┊ GC (median):     8.44%
# Time  (mean ± σ):   1.258 s ± 118.575 ms  ┊ GC (mean ± σ):   8.69% ± 1.11%
#
#  █                                █                     █ █  
#  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁█ ▁
#  1.1 s          Histogram: frequency by time         1.35 s <
#
# Memory estimate: 1.88 GiB, allocs estimate: 388.

N_MF = 10
T = 3.
ω_bos = (collect(-N_MF:N_MF  ) * (2.)      ) * π * T
ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * π * T
ωs_ext=(ω_bos,ω_fer,ω_fer)

GM = TCI4Keldysh.FullCorrelator_MF("data/PSF_nz=2_conn_zavg/4pt/", ["F1", "F1dag", "F3", "F3dag"]; flavor_idx=1, ωs_ext, 
ωconvMat=[ 0  1  0;
          -1 -1  0; 
           0  0 -1;
           1  0  1], name="SIAM 4pG");



@benchmark GM_dat = TCI4Keldysh.precompute_all_values(GM)
# N_MF = 10
#BenchmarkTools.Trial: 68 samples with 1 evaluation.
# Range (min … max):  62.650 ms … 104.985 ms  ┊ GC (min … max): 11.54% … 10.04%
# Time  (median):     67.621 ms               ┊ GC (median):    13.59%
# Time  (mean ± σ):   74.306 ms ±  13.282 ms  ┊ GC (mean ± σ):  12.63% ±  1.21%
#
#     ▆ █                                                        
#  ▅█▇███▇▇▇█▁▁▅▅▁▁▄▁▄▁▁▄▁▁▁▅▁▁▁▄▅▁▁▁▁▅▁▁▁▁▁▁▁▁▁▄▁▁▁▁▁▁▄▁▄▅▁▄▅▅ ▁
#  62.7 ms         Histogram: frequency by time          104 ms <
#
# Memory estimate: 258.74 MiB, allocs estimate: 4828.

# N_MF = 50
#BenchmarkTools.Trial: 2 samples with 1 evaluation.
# Range (min … max):  4.862 s …    5.449 s  ┊ GC (min … max): 9.18% … 8.01%
# Time  (median):     5.156 s               ┊ GC (median):    8.56%
# Time  (mean ± σ):   5.156 s ± 415.236 ms  ┊ GC (mean ± σ):  8.56% ± 0.83%
#
#  █                                                        █  
#  █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
#  4.86 s         Histogram: frequency by time         5.45 s <
#
# Memory estimate: 5.87 GiB, allocs estimate: 4996.

# N_MF = 100
#BenchmarkTools.Trial: 1 sample with 1 evaluation.
# Single result which took 32.174 s (6.25% GC) to evaluate,
# with a memory estimate of 37.47 GiB, over 5047 allocations.

total_time_3D = [67.621, 5156., 32174.]
N_MFs = [10, 50, 100]
pts_3D = (N_MFs .* 2 .+ 1) .* (N_MFs .* 2).^2
time_per_pt_3D = total_time_3D ./ pts_3D


@benchmark GM.([reshape((collect ∘ first ∘ axes)(ωs_ext[i]), (ones(Int, i-1)..., length(ωs_ext[i]))) for i in 1:3]...)
# N_MF = 10
#BenchmarkTools.Trial: 1 sample with 1 evaluation.
# Single result which took 33.138 s (1.08% GC) to evaluate,
# with a memory estimate of 14.32 GiB, over 7249245 allocations.

@benchmark for idxs in Iterators.product(1:2*N_MF+1, 1:2*N_MF, 1:2*N_MF)
    GM(idxs...)
end
# N_MF = 10:
#BenchmarkTools.Trial: 1 sample with 1 evaluation.
# Single result which took 31.136 s (1.22% GC) to evaluate,
# with a memory estimate of 14.32 GiB, over 7274405 allocations.


rand(1:2*N_MF)

@benchmark GM(1,1,1)
#BenchmarkTools.Trial: 1454 samples with 1 evaluation.
# Range (min … max):  3.024 ms …   6.049 ms  ┊ GC (min … max): 0.00% … 23.41%
# Time  (median):     3.190 ms               ┊ GC (median):    0.00%
# Time  (mean ± σ):   3.429 ms ± 478.038 μs  ┊ GC (mean ± σ):  1.16% ±  4.77%
#
#   ▅██▆▁                                                       
#  ▅█████▆▆▄▄▃▃▃▃▃▂▃▃▃▂▃▃▃▂▃▃▄▄▄▃▄▃▃▃▃▃▃▂▃▃▃▂▃▂▂▁▂▂▂▂▂▂▂▂▂▂▂▂▂ ▃
#  3.02 ms         Histogram: frequency by time        4.98 ms <
#
# Memory estimate: 1.75 MiB, allocs estimate: 863.

filename_broadened = "data/precomputedAcont_estep"*string(estep)*"_Nw"*string(Nωcont_pos)
save_Acont(filename_broadened, ωdisc, Adisc, ωcont, Acont)

