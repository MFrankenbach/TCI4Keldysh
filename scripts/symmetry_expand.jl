using Revise
using TCI4Keldysh

data_dir = "/scratch/m/M.Frankenbach/tci4keldysh/data/SIAM_u5_U0.05_T0.0005_Delta0.0031831/PSF_nz=2_conn_zavg/4pt";

symmred_correlators = [
    ["F1", "F1dag", "F3", "F3dag"], # 1 FFFF
    ["F1", "Q1dag", "F3", "F3dag"], # 2 FQFF 3
    ["F1", "Q1dag", "F3", "Q3dag"], # 3 FQFQ 2
    ["F1", "Q1dag", "Q3", "F3dag"], # 4 FQQF 2
    ["Q1", "F1dag", "F3", "F3dag"], # 5 QFFF 3
    ["Q1", "F1dag", "Q3", "F3dag"], # 6 QFQF 2
    ["Q1", "Q1dag", "F3", "F3dag"], # 7 QQFF 2
    ["Q1", "Q1dag", "F3", "Q3dag"], # 8 QQFQ 1
    ["Q1", "Q1dag", "Q3", "F3dag"], # 9 QQQF 1
    ["Q1", "Q1dag", "Q3", "Q3dag"], # 10 QQQQ
]
i_symmred_Cs = [1, 4, 5, 6, 9, 11, 14, 15, 16]
needed_i_symmred = [
    1,  # FFFF
    2,  # FFFQ
    5,  # FFQF
    7,  # FFQQ
    2,  # FQFF
    3,   # FQFQ
    4,   # FQQF
    8,   # FQQQ
    5,  # QFFF
    4,  # QFFQ
    6,  # QFQF
    9,  # QFQQ
    7,  # QQFF
    8,   # QQFQ
    9,   # QQQF
    10,   # QQQQ
]


filelist = readdir(data_dir)
for fn in filelist
    println(string.(TCI4Keldysh.parse_filename_to_Ops(fn)))
    TCI4Keldysh.symmetry_expand(data_dir, string.(TCI4Keldysh.parse_filename_to_Ops(fn)); nested_ωdisc=false)
end

# using MAT
# f = matopen("data/SIAM_u=1.00/PSF_nz=4_conn_zavg/PSF_((F1dag,F1)).mat")
# keys(f)
# read(f, "Adisc")[1]
# read(f, "PSF")
# close(f)
