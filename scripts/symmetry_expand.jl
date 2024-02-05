using Revise
using TCI4Keldysh

data_dir = "./data/PSF_nz=2_conn_zavg/4pt/";

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
    println(string.(parse_filename_to_Ops(fn)))
    TCI4Keldysh.symmetry_expand(data_dir, string.(parse_filename_to_Ops(fn)))
end

