using Revise
using TCI4Keldysh
using HDF5
using BenchmarkTools

D = 2
Ops = ["F1", "F1dag", "Q34"]

T = 0.01;
N_MF = 4000;
ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * im * pi * T;
ω_bos = collect(-N_MF:N_MF) * (2.) * im * pi * T;
ωs_ext = (ω_bos, ω_fer);
ωsconvMat = [1 0 ; 0 1; -1 -1];

using Combinatorics
perms = permutations(4)
perms = multiset_permutations([1,2,1,2], 4)
for p in perms


println(p)
end



perm_olabels = multiset_permutations(olabels, 4)
perms = permutations(4)

for p in perms

end


Ops[3][1]


@time G = TCI4Keldysh.FullCorrelator_MF(data_dir, Ops; flavor_idx=1, ωs_ext=ωs_ext, ωconvMat=ωsconvMat, name="Hubbard atom 3p correlator");
@time data = TCI4Keldysh.precompute_all_values(G);
#data_axes = ntuple(i -> reshape(collect(axes(ωs_ext[i])[1]), (ones(Int,i-1)..., length(ωs_ext[i]))), D)
#@time data = G.(data_axes...)

# results from @time precompute_all_values(...)
N_MFs = [1000, 2000, 4000]
runtimes_1thread = [0.886964, 3.932792,  19.740861] # in seconds
allocations_num = [409, 409, 409]
allocations_sto = [1.642, 6.503, 25.880] # in GiB

plot(N_MFs, [runtimes_1thread, allocations_sto, N_MFs.^2 ./N_MFs[1]^2 .* allocations_sto[1], N_MFs.^3 ./N_MFs[1]^3 .* runtimes_1thread[1]], labels=["CPU time [s]" "allocations [GiB]" "N^2" "N^3"], xscale=:ln, yscale=:ln)


cg_mat = [1 0; 0.5 0.5]
cg = kron(cg_mat, cg_mat)
ζ_mat = [1 0; 0 -1]
ζ = kron(ζ_mat, ζ_mat)

cg_exc = cg * ζ / cg
cg_excs= cg_mat * ζ_mat / cg_mat

D = 3
Ops = ["F1", "F1dag", "F3", "F3dag"]

T = 0.1;
N_MF = 50;
ω_fer = (collect(-N_MF:N_MF-1) * (2.) .+ 1.) * im * pi * T;
ω_bos = collect(-N_MF:N_MF) * (2.) * im * pi * T;
ωs_ext = (ω_bos, ω_fer, ω_fer);
ωsconvMat = [0 1 0 ; -1 -1 0; 1 0 1; 0 0 -1];

@time G = TCI4Keldysh.FullCorrelator_MF(data_dir, Ops; flavor_idx=1, ωs_ext=ωs_ext, ωsconvMat=ωconvMat, name="Hubbard atom vertex");
@time data = TCI4Keldysh.precompute_all_values(G);

a = collect(1:2)
b = collect(1:3)
kron(a,b)

using Plots
using BenchmarkTools
@benchmark G(1,1)
@benchmark G.Gps[1].Adisc[1,1]

heatmap(real.(data)[])

TCI4Keldysh.evaluate(G, 1, 1)
G(1,1)

TCI4Keldysh.precompute_all_values(G.Gps[4])

using StaticArrays
v=SA[1,2,3]
typeof(v')

using StridedViews
sv = StridedView()


begin
    function get_ωcont(ωmax, Nωcont_pos)
        ωcont = collect(range(-ωmax, ωmax; length=Nωcont_pos*2+1))
        return ωcont
    end


    filename = "test/test_PSF_2D.h5"
    f = h5open(filename, "r")
    Adisc = read(f, "Adisc")
    ωdisc = read(f, "ωdisc")
    close(f)

    Adisc = dropdims(Adisc,dims=tuple(findall(size(Adisc).==1)...))

    ### System parameters of SIAM ### 
    D = 1.
    # Keldysh paper:    u=0.5 OR u=1.0
    U = 1. / 20.
    T = 0.01 * U
    Δ = (U/pi)/0.5
    # EOM paper:        U=5*Δ
    #Δ = 0.1
    #U = 0.5*Δ
    #T = 0.01*Δ

    ### Broadening ######################
    #   parameters      σ       γ       #
    #       by JaeMo:   0.3     T/2     #
    #       by SSL:     0.6     T*3     #
    #   my choice:                      #
    #       u = 0.5:    0.6     T*3     #
    #       u = 1.0:    0.6     T*2     #
    #       u = 5*Δ:    0.6     T*0.5   #
    #####################################
    σ = 0.6
    sigmab = [σ]
    g = T * 1.
    tol = 1.e-14
    estep = 2048
    emin = 1e-6; emax = 1e4;
    Lfun = "FD" 
    is2sum = false
    verbose = false



    R = 6
    Nωcont_pos = 2^R # 512#
    ωcont = get_ωcont(D*0.5, Nωcont_pos)

    # Directly obtained broadened data
    #ωcont, Acont = TCI4Keldysh.getAcont_mp(ωdisc, Adisc, sigmab, g; ωcont, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
    # get functor which can evaluate broadened data pointwisely
    broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωcont, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
end

using StaticArrays

using FileIO
using MAT
f = matopen("./data/PSF_nz=2_conn_zavg/withQ/PSF_((F1,F1dag,Q34)).mat", "r")
f4 = matopen("./data/PSF_nz=2_conn_zavg/PSF_((F1,F1dag,F3,F3dag)).mat", "r")
try 
    keys(f)
catch
    keys(f)
end

Adiscs = read(f4, "Adisc")
Adiscs[1]
Adiscs[2]

read(f, "odisc")

maximum(abs.(Adiscs[1] - Adiscs[2]))

v =  [1, 2, 3]
@benchmark sm * v
v1, v2, v3 = 1, 2, 3;
@benchmark sm * SA[v1, v2, v3]
vt= (1, 2, 3)
@benchmark sm * SA[vt...]

mapreduce(*,*,Ops, ["," for i in 1:length(Ops)])

@benchmark G_p_disc[1,2]
@benchmark G_p_disc(1,2)

G_p_disc(1,2)

kernel = TCI4Keldysh.get_regular_1DKernel(ω_bos, broadenedPsf.ωcont)
sum(isnan.(kernel))

size(Adisc)
ωdisc[50]
maximum(abs.(Adisc[50,:]))
maximum(abs.(Adisc[:,50]))
Adisc[50:80,50]


TCI4Keldysh.trafo_ω_args((ω_bos, ω_fer, ω_fer), [0 1 0; -1 -1 0; 1 0 1])


trunc(Int, 1.1)
fill(100)

N_in = 1000; x_in_max = 100.;
xs = collect(LinRange(-x_in_max, x_in_max, N_in))
xs_out = collect(LinRange(-50., 50., 1000))
dims2 = collect(1:2)'
ys = sin.(xs) ./ xs# .* dims2
@time ht = TCI4Keldysh.my_hilbert_trafo(xs_out, xs, ys);
expected = (x -> (1. - cos(x)) / x).(xs_out) .* π * dims2
maximum(abs.(ht - expected))

@time ht = hilbert_fft(ys);
expected = (x -> (1. - cos(x)) / x).(xs) #* dims2
maximum(abs.(imag.(ht) - expected))

using Plots

plot([xs, xs, xs/π], [real.(ht)[:,1], imag.(ht)[:,1], expected[:,1]], labels=["Re" "Im" "expected"])
plot([imag.(ht)[:,1], expected[:,1]], labels=["Im" "expected"])
maximum(abs.(expected - imag.(ht))) / maximum(abs.(expected))
using Plots

N = 1000
ωdisc = collect( -N:N) / sqrt(7)

Adisc = 1 ./ (ωdisc.^2 .+ 1) ./ π       #   Lorentzian
GR = -im * π * (TCI4Keldysh.hilbert_fft(Adisc))

GR[N+1-5:N+1+5]

plot(ωdisc, [real.(GR), imag.(GR), ωdisc ./ (ωdisc.^2 .+ 1), -1. ./ (ωdisc.^2 .+ 1)], labels=["Re(GR)" "Im(GR)" "ω/(ω^2+1)" "-1/(ω^2+1)"], xlims=[-10, 10])


Adisc = zeros(2*N+1); Adisc[N+1] = 1.; Adisc[N] = 0.5; Adisc[N+2] = 0.5;  #   Dirac-delta peak
Adisc = zeros(2*N+1); Adisc[N+2] = 1.; #Adisc[N] = 0.5; Adisc[N+2] = 0.5;  #   Dirac-delta peak
GR = -im * π * (TCI4Keldysh.hilbert_fft(Adisc))

GR[N+1-5:N+1+5]

plot(ωdisc, [real.(GR), 1 ./ ωdisc * 2 * (ωdisc[2]-ωdisc[1])], labels=["Re(GR)"  "1/ω" ], xlims=[-10, 10])
plot(ωdisc[1:2:end], [real.(GR)[1:2:end], 1 ./ ωdisc[1:2:end] * 4 * (ωdisc[2]-ωdisc[1])], labels=["Re(GR)"  "1/ω" ], xlims=[-10, 10])
plot(ωdisc[2:2:end], [real.(GR)[2:2:end], 1 ./ ωdisc[2:2:end] * 2 * (ωdisc[2]-ωdisc[1])], labels=["Re(GR)"  "1/ω" ], xlims=[-10, 10])
(1 ./ ωdisc[2:2:end] * 2 * (ωdisc[2]-ωdisc[1]))[500:500+10]
TCI4Keldysh.maxabs(real.(GR)[2:2:end] - 1 ./ ωdisc[2:2:end] / 5.) / TCI4Keldysh.maxabs(real.(GR)[2:2:end])
(1 ./ ωdisc[1:2:end] * 2 * (ωdisc[2]-ωdisc[1]))[500:510]

x = Adisc
dims = 1
x_ = copy(x)
n = size(x_,dims)

xf = fft(x_,dims)
h = reshape(zeros(Int64,n), (ones(Int, dims-1)..., n))      # represents the step function in time --> product in time corresponds to convolution with retarded kernel in frequency space
if n>0 && n % 2 == 0
    #even, nonempty
    h[1:div(n,2)+1] .= 1
    h[2:div(n,2)] .= 2
elseif n>0
    #odd, nonempty
    h[1] = 1
    h[2:div(n + 1,2)] .= 2
end
x_ = ifft(xf .* h, dims)
x_

plot(ωdisc, [real.(x_), imag.(x_)], xlims=[-101, -90])

1. ./ imag.(x_[2:2:end]) * 2. / π