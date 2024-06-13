using TCI4Keldysh

TCI4Keldysh.VERBOSE() = false
TCI4Keldysh.DEBUG() = false
TCI4Keldysh.TIME() = false
for i in 1:1
    TCI4Keldysh.test_TCI_precompute_anomalous_values(;npt=4, perm_idx=i)
end


# # compile
# TCI4Keldysh.test_TCI_precompute_reg_values_MF_without_ωconv(npt=3, perm_idx=1)
# TCI4Keldysh.DEBUG() = false
# npt = 4
# file_out = "tci_$(npt)-pt_errors.txt"
# open(file_out, "a") do f 
#     write(f, "perm_idx  mean_err     max_err\n")
# end
# for i in 15:factorial(npt)
#     printstyled(" ---- $i-th permutation\n"; color=:blue)
#     mean_err, max_err = TCI4Keldysh.test_TCI_precompute_reg_values_MF_without_ωconv(npt=npt, perm_idx=i)
#     open(file_out, "a") do f 
#         write(f, "$i        $mean_err, $max_err\n")
#     end
# end