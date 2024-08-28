using TCI4Keldysh
#=
Read in PSFs stored as .mat files in which ωdisc is stored as:
read(f, "PSF")["odisc_info"]["odisc"]

Afterwards, we have fields:
read(f, "Adisc")
read(f, "odisc") (same as read(f, "PSF")["odisc_info"]["odisc"])
=#

data_dir = "data/siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg/4pt";
backup_dir = "data/siam05_U0.05_T0.005_Delta0.0318/PSF_nz=2_conn_zavg/original_files/4pt"

using MAT

function change_PSF_layout()
    for file in readdir(data_dir)
        if !isfile(joinpath(data_dir, file))
            continue
        end
        # backup
        fullname = joinpath(data_dir, file)
        f_in = matopen(fullname)
        try
            keys(f_in)
        catch
            keys(f_in)
        end
        try
            println("Altering file: $fullname")
            odisc = read(f_in, "PSF")["odisc_info"]["odisc"]
            Adisc = read(f_in, "Adisc")
            mv(joinpath(data_dir, file), joinpath(backup_dir, file); force=true)
            close(f_in)
            f_out = matopen(fullname, "w")
            write(f_out, "Adisc", Adisc)
            write(f_out, "odisc", odisc)
            close(f_out)
        catch
            # check that format is fine
            try
                odisc = read(f_in, "odisc")
                Adisc = read(f_in, "Adisc")
            catch
                error("In file $file: Cannot read odisc")
            end
            close(f_in)
            continue
        end
    end
end

function check_layout(dir::String; layout=:old)
    for file in readdir(dir)
        if !isfile(joinpath(dir, file))
            continue
        end
        f = matopen(joinpath(dir, file))
        try
            keys(f)
        catch
            keys(f)
        end
        try 
            if layout==:new
                odisc = read(f, "odisc")
                Adisc = read(f, "Adisc")
            elseif layout==:old
                odisc = read(f, "PSF")["odisc_info"]["odisc"]
                Adisc = read(f, "Adisc")
            else
                throw(ArgumentError("Invalid argument layout=$layout"))
            end
        catch
            error("In file: $file: key not found")
        end
        close(f)
    end
end

# check_layout(data_dir; layout=:new)