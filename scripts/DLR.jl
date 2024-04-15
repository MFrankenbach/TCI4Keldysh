##
# Try out compression via Discrete Lehmann Representation
## 

using Revise
using TCI4Keldysh
using Quantics
using QuanticsTCI
import TensorCrossInterpolation as TCI
using QuanticsGrids
import QuanticsGrids: UnfoldingSchemes
using LinearAlgebra
using ITensors
using HDF5

TCI4Keldysh.VERBOSE() = true


begin

    function get_ωcont(ωmax, Nωcont_pos)
        ωcont = collect(range(-ωmax, ωmax; length=Nωcont_pos*2+1))
        return ωcont
    end
    
    

    filename = "test/tests/data_PSF_2D.h5"
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
    #Δ = (U/pi)/0.5

    σ = 0.6
    sigmab = [σ]
    g = T * 1.
    tol = 1.e-14
    estep = 2048
    emin = 1e-6; emax = 1e4;
    Lfun = "FD" 
    is2sum = false
    verbose = false

    Rpos = 10
    R = Rpos + 1
    Nωcont_pos = 2^Rpos # 512#
    ωcont = get_ωcont(D*0.5, Nωcont_pos)
    ωconts=ntuple(i->ωcont, ndims(Adisc))
    
    # get functor which can evaluate broadened data pointwisely
    #broadenedPsf = TCI4Keldysh.BroadenedPSF(ωdisc, Adisc, sigmab, g; ωconts, emin=emin, emax=emax, estep=estep, tol=tol, Lfun=Lfun, verbose=verbose, is2sum);
    ωbos = im * π * T * collect(-Nωcont_pos+1:Nωcont_pos-1) * 2
    ωbos = im * π * T * collect(-Nωcont_pos:Nωcont_pos) * 2
    ωfer = im * π * T *(collect(-Nωcont_pos:Nωcont_pos-1) * 2 .+ 1)
    ωs_ext = (ωbos, ωfer)
    ωconv = [
         1  0;
         -1  -1
    ]
    Gp = TCI4Keldysh.PartialCorrelator_reg("MF", Adisc, ωdisc, ωs_ext, ωconv)
end

function interp_decomp(A_in; atol, i_s_min=-1)
    _, s, _ = svd(A_in)
    _, r, p = qr(A_in, Val(true))
    #s
    #i_s = argmax(s .< tol)
    #s[i_s]
    
    imax = -1
    for i_s in argmax(s .< atol)-1:length(s)
        #println(i_s)
        R22 = r[i_s:end, i_s:end]
        _, s2, _ = svd(R22)
            if s2[1] < atol
                #println("Found an imax!")
                imax = i_s
                break
            end
    end
    imax = max(imax, i_s_min)
    return p[1:imax]
    
end

function least_squares(A, b; metric=nothing)
    if !(metric === nothing)
        @assert ndims(metric) == 1
        A = metric .* A
        b = metric .* b
    end
    qrA = qr(A);                    # QR decomposition
    x = qrA\b;
    #println(qrA.R)
    return x
end

atol = 1e-4
Gp.ωdiscs[1]
Gp.Kernels[1]

Kernels = Gp.Kernels
Adisc = Gp.Adisc
ωdiscs = Gp.ωdiscs
iωs = Gp.ωs_int
D = length(Kernels)
p_ωdiscs = [ones(Int, 1) for _ in 1:D]
p_iωs = [ones(Int, 1) for _ in 1:D]

Gp_data = Gp[[Colon() for _ in 1:D]...]
Gp_data_tmp = deepcopy(Gp_data)

for i in 1:D
    K_in = Gp.Kernels[i]
    p_ωdisc = interp_decomp(K_in; atol)
    K_interm = K_in[:,p_ωdisc]
    p_iω = interp_decomp(transpose(K_interm); atol, i_s_min=length(p_ωdisc))
    K_new = K_interm[p_iω,:]
    Kernels[i] = K_new

    _, s, _ = svd(K_in)
    #sum(s .> atol)

    
    #Gp_data = Gp.Kernels[1] * Gp.Adisc

    sz_Gp_data = size(Gp_data_tmp)
    Gp_data_tmp = reshape(Gp_data_tmp, (sz_Gp_data[1], prod(sz_Gp_data[2:end])))
    Gp_data_DLR = Gp_data_tmp[p_iω,:]
    adisc_DLR = least_squares(K_new, Gp_data_DLR)
    Gp_data_DLRapprox = K_new * adisc_DLR

    println("dimenstion i/D = $i/D")
    println("\t abs error of approximant: ", maximum(abs.(Gp_data_DLR - Gp_data_DLRapprox)))
    #println("\t deviation in coefficients", maximum(abs.(Gp_data_tmp[p_ωdisc,:] - adisc_DLR)))

    p_ωdiscs[i] = p_ωdisc
    p_iωs[i] = p_iω
    Gp_data_tmp = permutedims(reshape(adisc_DLR, (size(adisc_DLR, 1), prod(sz_Gp_data[2:end]))), [collect(2:D)..., 1])
end

Adisc_new = Gp_data_tmp
size(Adisc_new)
size.(Kernels)