import imdlib as imd

for var in ["tmax", "tmin", "rain"]:
    imd.get_data(var, 2020, 2020, fn_format="yearwise", file_dir="data/raw/imd")

data = imd.open_data("tmax", 2020, 2020, "yearwise", "data/raw/imd")
print(data)