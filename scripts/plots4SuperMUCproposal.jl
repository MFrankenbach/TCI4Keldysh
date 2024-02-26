using Revise
using QuanticsTCI
import TensorCrossInterpolation as TCI
using MAT
using TCI4Keldysh
using FileIO
using CairoMakie



function load_NRG_data(filename; i_Keldysh, ispin)
    
    file = matopen(filename, "r")
    try 
        keys(file)
    catch
        keys(file)
    end
    #data= read(file, "CFdat")
    Ggrid = read(file, "CFdat/Ggrid")[ispin][:,:,:,i_Keldysh...]
    ogrid = read(file, "CFdat/ogrid")# data["ogrid"]
    return Ggrid, ogrid
    #return ogrid
end


ispin = 1
i_Keldysh = (1,1,1,1)
filename = "data/SIAM_u=1.50/V_KF_ph/V_KF_sym.mat"
Ggrid, ogrid = load_NRG_data(filename; i_Keldysh, ispin)
#ogrid = load_NRG_data(filename; i_Keldysh, ispin)
#Ggrid = qtt[:,:,:]


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



fig = Figure(size = (1000, 300));
ax1 = Axis(fig[1, 1],
    title = L"\Gamma_{\mathrm{full}}(\omega=0,\nu,\nu')-\Gamma_0",
    xlabel = "ν/Δ",
    ylabel = "ν′/Δ")

#Legend(fig[1:2, 2], [l1, l2], ["sin", "cos"])

plotdata = imag.(Ggrid[:,:,Nωs_pos+1])
maxmax=maximum(abs.(plotdata))
hm = CairoMakie.heatmap!(ax1, [ogrid[i][:]/Δ for i in 1:2]..., plotdata, colormap=:balance, colorrange=[-maxmax, maxmax])
Colorbar(fig[1, 2], hm)


# Load data
qtt = load("data/qtt_SIAM_u=0.50_PSF_tol=1e-4.jld2", "qtt_tol=1e-4")
qtt5 = load("data/qtt_SIAM_u=0.50_PSF_tol=1e-5.jld2", "qtt_tol=1e-5")
qtt6 = load("data/qtt_SIAM_u=0.50_PSF_tol=1e-6.jld2", "qtt_tol=1e-6")
# Gen data
#qtt , ranks, errors = my_qinterpolate(Ggrid; tolerance=1e-4)
#qtt5, ranks, errors = my_qinterpolate(Ggrid; tolerance=1e-5)
#qtt6, ranks, errors = my_qinterpolate(Ggrid; tolerance=1e-6)
#save("data/qtt_SIAM_u=0.50_PSF_tol=1e-4.jld2", "qtt_tol=1e-4", qtt)
#save("data/qtt_SIAM_u=0.50_PSF_tol=1e-5.jld2", "qtt_tol=1e-5", qtt)
#save("data/qtt_SIAM_u=0.50_PSF_tol=1e-6.jld2", "qtt_tol=1e-6", qtt)

#qtt2d, ranks2d, errors2d = my_qinterpolate(plotdata; tolerance=1e-5)
qttdata = (qtt[:,:, 128])
#qttdata = qtt2d[:,:]
shift = div(size(qttdata, 1) - size(plotdata, 1), 2)
#qttdata = qttdata[1+shift:end-shift-1, 1+shift:end-shift-1]
qttdata = qttdata[1+shift:end-shift-1, 1+shift:end-shift-1]

ax2 = Axis(fig[1, 3],
title = L"\log(\Gamma_{\mathrm{QTT}}-\Gamma_{\mathrm{orig}})",
xlabel = "ν/Δ",
ylabel = "ν′/Δ")
dif = log10.(abs.(qttdata - Ggrid[:,:,Nωs_pos+1]))
#infs = isinf.(diff)
#diff[infs] = 
maxmax=maximum((dif))
maximum(abs.(qttdata))
#any(isinf.(diff))

hm = CairoMakie.heatmap!(ax2, [ogrid[i][:]/Δ for i in 1:2]..., dif, colormap=Reverse(:ice), colorscale=Makie.pseudolog10)
#maxmax = maximum(abs.(qttdata))
#hm = CairoMakie.heatmap!(ax2, [ogrid[i][:]/Δ for i in 1:2]..., qttdata, colormap=:balance, colorrange=[-maxmax,maxmax])
Colorbar(fig[1, 4], hm)#, ticks = [-1,-5,-15])


ax3 = Axis(fig[1, 5],
    xlabel = L"\ell",
    ylabel = L"D_\ell")
ldims = TCI.linkdims(qtt.tt)
ldims5= TCI.linkdims(qtt5.tt)
ldims6= TCI.linkdims(qtt6.tt)
D = length(qtt.grid.origin)
R = 8
2^R
worst_case = 2 .^ min.(1:D*R-1, D*R-1:-1:1)
l1 = lines!(ax3, ldims , yscale=log, label="tol=10^-4")
l5 = lines!(ax3, ldims5, yscale=log, label="tol=10^-5")
l6 = lines!(ax3, ldims6, yscale=log, label="tol=10^-6")
l2 = lines!(ax3, worst_case, yscale=log, color="grey")

axislegend(ax3)
#Legend(f[1, 5],
#    [l1, l2,
#    ["a line", "some dots", "both together", "rect markers"])


fig 


save("scripts/plots/QTCI_3D_SIAM_fullvertex_upup_u=1.50_tol=1e-4.pdf", fig )
