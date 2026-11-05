import imdlib as imd

MISSING = {"tmax": 99.9, "tmin": 99.9, "rain": -999.0}

for var in ["tmax", "tmin", "rain"]:
    data = imd.open_data(var, 2020, 2020, "yearwise", "data/raw/imd")
    da = data.get_xarray()[var]
    valid = da.where(abs(da - MISSING[var]) > 0.01).notnull()
    cells = int(valid.any(dim="time").sum())
    vals = da.where(valid)
    print(f"{var}: grid cells with data = {cells}, "
          f"min = {float(vals.min()):.1f}, max = {float(vals.max()):.1f}")