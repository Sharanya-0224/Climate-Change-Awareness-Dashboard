import imdlib as imd

START, END = 1951, 2025

for var in ["tmax", "tmin"]:
    imd.get_data(var, START, END, fn_format="yearwise", file_dir="data/raw/imd")
    