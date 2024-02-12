using Revise
using QuanticsTCI
import TensorCrossInterpolation as TCI
using MAT
using TCI4Keldysh

using CairoMakie


function load_NRG_data(filename; i_Keldysh, ispin)
    
    file = matopen(filename, "r")
    try 
        keys(file)
    catch
        keys(file)
    end
    data= read(file, "CFdat")
    Ggrid = data["Ggrid"][ispin][:,:,:,i_Keldysh...]
    ogrid = data["ogrid"]
    return Ggrid, ogrid
end

ispin = 1
i_Keldysh = (1,1,1,1)
filename = "data/SIAM_u=1.50/V_KF_ph/V_KF_sym.mat"
Ggrid, ogrid = load_NRG_data(filename; i_Keldysh, ispin)

GC.gc()

ceil(Int64, log2(201)/2)
# Quantics-interpolate:
function my_qinterpolate(Ggrid::Array{T,N}; tolerance=1e-8) where{T,N}
    @assert all(size(Ggrid) .== size(Ggrid,1)) # assert that all are equal (otherwise change this implementation)
    R = ceil(Int64, log2(size(Ggrid, 1)))
    println("R = ", R)
    
    Ggrid_padded = zeros(eltype(Ggrid), (2^R .* ones(Int, ndims(Ggrid)))...)
    Nωs_pos = div(size(Ggrid, 1), 2)
    ran = 2^(R-1)-Nωs_pos:2^(R-1)+Nωs_pos
    println("size of Ggrid_padded[[ran for _ in ndims(Ggrid)]...]", size(Ggrid_padded[ntuple(i -> ran, ndims(Ggrid))...]))
    Ggrid_padded[ntuple( i-> ran, ndims(Ggrid))...] .= Ggrid
    #tolerance = 1e-4
    
     
    qtt, ranks, errors = quanticscrossinterpolate(
        Ggrid_padded,
        tolerance=tolerance
        ; maxiter=400
    )  
    return qtt, ranks, errors
end





U = 0.05
Δ = U / (π*1.5)

Nωs_pos = div(length.(ogrid)[1], 2)



fig = Figure(size = (900, 300));
ax1 = Axis(fig[1, 1],
    title = L"\Gamma_{\mathrm{full}}(\omega=0,\nu,\nu')-\Gamma_0",
    xlabel = "ν/Δ",
    ylabel = "ν′/Δ")

#Legend(fig[1:2, 2], [l1, l2], ["sin", "cos"])

plotdata = imag.(Ggrid[:,:,Nωs_pos+1])
maxmax=maximum(abs.(plotdata))
hm = heatmap!(ax1, [ogrid[i][:]/Δ for i in 1:2]..., plotdata, colormap=:balance, colorrange=[-maxmax, maxmax])
Colorbar(fig[1, 2], hm)


qtt, ranks, errors = my_qinterpolate(Ggrid; tolerance=1e-8)
qtt2d, ranks2d, errors2d = my_qinterpolate(plotdata; tolerance=1e-6)
qttdata = qtt[:,:, 128]
qttdata = qtt2d[:,:]
shift = div(size(qttdata, 1) - size(plotdata, 1), 2)
qttdata = qttdata[1+shift:end-shift-1, 1+shift:end-shift-1]

ax2 = Axis(fig[1, 3],
title = L"\log(\Gamma_{\mathrm{QTT}}-\Gamma_{\mathrm{orig}})",
xlabel = "ν/Δ",
ylabel = "ν′/Δ")
dif = log10.(abs.(qttdata - plotdata))
#infs = isinf.(diff)
#diff[infs] = 
maxmax=maximum(dif)
#any(isinf.(diff))

hm = heatmap!(ax2, [ogrid[i][:]/Δ for i in 1:2]..., dif, colormap=Reverse(:ice), colorscale=Makie.pseudolog10)
Colorbar(fig[1, 4], hm)


ax3 = Axis(fig[1, 5],
    xlabel = L"\ell",
    ylabel = L"D_\ell")
ldims = TCI.linkdims(qtt.tt)
D = length(qtt.grid.origin)
R = 8
2^R
worst_case = 2 .^ min.(1:D*R-1, D*R-1:-1:1)
l1 = lines!(ax3, ldims, yscale=log)
l2 = lines!(ax3, worst_case, yscale=log, color="grey")

#Legend(f[1, 5],
#    [l1, l2,
#    ["a line", "some dots", "both together", "rect markers"])


fig 

save("scripts/plots/QTCI_3D_SIAM_fullvertex_upup_u=1.50_tol=1e-8.pdf", fig )