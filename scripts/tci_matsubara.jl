using TCI4Keldysh
using JSON
using ITensors

TCI4Keldysh.VERBOSE() = false
TCI4Keldysh.DEBUG() = false
TCI4Keldysh.TIME() = true

for i in 1:6
    TCI4Keldysh.test_TCI_precompute_anomalous_values(;npt=3, perm_idx=i)
end


"""
Monitor mean and max error of all permutations for 3/4-point functions before frequency rotation
"""
function check_all_permutations_reg()
    # compile
    TCI4Keldysh.test_TCI_precompute_reg_values_MF_without_ωconv(npt=3, perm_idx=1)
    npt = 4
    file_out = "tci_$(npt)-pt_errors.txt"
    open(file_out, "a") do f 
        write(f, "perm_idx  mean_err     max_err\n")
    end
    for i in 1:factorial(npt)
        printstyled(" ---- $i-th permutation\n"; color=:blue)
        mean_err, max_err = TCI4Keldysh.test_TCI_precompute_reg_values_MF_without_ωconv(npt=npt, perm_idx=i)
        open(file_out, "a") do f 
            write(f, "$i        $mean_err, $max_err\n")
        end
    end
end

function time_TCI_precompute_reg_values(;npt=3)
    
    ITensors.disable_warn_order()

    # load data
    PSFpath = "data/SIAM_u=0.50/PSF_nz=2_conn_zavg/"
    Ops = npt==3 ? ["F1", "F1dag", "Q34"] : ["F1", "F1dag", "F3", "F3dag"]
    ωconvMat = TCI4Keldysh.dummy_frequency_convention(npt)
    R = 7
    GFs = TCI4Keldysh.load_npoint(PSFpath, Ops, npt, R, ωconvMat; nested_ωdisc=false)

    # pick PSF
    spin = 1
    perm_idx = 1
    Gp = GFs[spin].Gps[perm_idx]

    # TCI computation
    cutoffs = 10.0 .^ (-2:-1:-5)

    times = []
    dummy = Vector{MPS}(undef, 1)
    # for some reason, the second loop with npt=4 takes ages here, also for large cutoffs
    for c in cutoffs
        t = @elapsed begin 
                dummy[1] = TCI4Keldysh.TD_to_MPS_via_TTworld(Gp.tucker; tolerance=1e-12, cutoff=c)
            end
        @show t
        push!(times, t)
        GC.gc()
    end

    # write to file
    d = Dict("cutoffs" => cutoffs, "times" => times, "npt" => npt, "perm_idx" => perm_idx, "ops" => Ops)
    logJSON(d, "reg_val_times_$(npt)pt")
end

# utilities ========== 
function logJSON(data, filename::String)
    fullname = filename*".json"
    open(fullname, "w") do file
        JSON.print(file, data)
    end
    printstyled("File $filename.json written!\n", color=:green)
end

function readJSON(filename::String)
    data = open(filename) do file
        JSON.parse(file)
    end 
    return data
end
# utilities END ========== 

# time_TCI_precompute_reg_values(;npt=3)
# TCI4Keldysh.test_TCI_precompute_reg_values_MF_without_ωconv(;npt=4, perm_idx=2, cutoff=1e-3)
# TCI4Keldysh.test_TCI_frequency_rotation_reg_values()