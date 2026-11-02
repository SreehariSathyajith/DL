"""Step 3: download PSL's ERSST grid yourself, check its internal version, cut the D3 subset, compare with the issued one.
Usage:  python verify_grid_download.py TrackC_ERSSTv5_subset.npz [--file sst.mnmean.nc]    (needs: pip install netCDF4)
Without --file it downloads https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc (~159 MB).
"""
import sys, re, hashlib, urllib.request, numpy as np, netCDF4

URL = "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc"
ISSUED_FILE_BYTES, ISSUED_FILE_SHA_PREFIX = 159_094_577, "651c3ced"      # Annex D3 (live file is revised monthly)
ISSUED_ARRAY_SHA = "8e566e66ee212a2441709072bac6fe6d8ba380370efb6b11c50470c4c7963570"

issued = sys.argv[1]
path = sys.argv[sys.argv.index("--file") + 1] if "--file" in sys.argv else "sst.mnmean.nc"
def download(url, dest, tries=5):
    import time, urllib.error
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (course data check; contact your instructor)"})
    for k in range(1, tries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                got = 0
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b); got += len(b); print(f"\r  {got/2**20:7.1f} MB", end="", flush=True)
            print(); return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            print(f"\nattempt {k}/{tries} failed: {e}")
            if k < tries: time.sleep(15 * k)
    sys.exit("All attempts failed. Retry later, or download the URL in a browser and run again with: --file sst.mnmean.nc")

if "--file" not in sys.argv:
    print("downloading", URL); download(URL, path)
raw = open(path, "rb").read()
print(f"bytes transferred/on disk: {len(raw):,}  (issued: {ISSUED_FILE_BYTES:,})   sha256: {hashlib.sha256(raw).hexdigest()}  (issued prefix {ISSUED_FILE_SHA_PREFIX})")

ds = netCDF4.Dataset(path)
print("\n--- global attributes (the version lives here, not on the web page) ---")
blob = ""
for k in ds.ncattrs():
    v = str(ds.getncattr(k)); blob += f" {k} {v}"; print(f"{k}: {v[:300]}")
v6 = re.search(r"(version\s*6|\bv\s*6\b)", blob, re.I); v5 = re.search(r"(version\s*5|\bv\s*5\b)", blob, re.I)
print("\nVERSION CHECK:", "FAIL - mentions v6" if v6 else ("PASS - v5 found, no v6" if v5 else "INCONCLUSIVE - read the attributes above"))

t = ds["time"]; dates = netCDF4.num2date(t[:], t.units)
yr = np.array([x.year for x in dates]); mo = np.array([x.month for x in dates])
print("time axis:", yr[0], mo[0], "->", yr[-1], mo[-1], "| months:", len(yr))
lat = ds["lat"][:].astype(np.float32); lon = ds["lon"][:].astype(np.float32)
ti = np.where((yr >= 1950) & (yr <= 2024))[0]
li = np.where((lat <= 20) & (lat >= -20))[0]; li = li[np.argsort(-lat[li])]     # 20N -> 20S
lj = np.where((lon >= 120) & (lon <= 280))[0]
sst = np.ma.filled(ds["sst"][ti[0]:ti[-1] + 1][:, li][:, :, lj].astype(np.float32), np.nan)
print("subset shape", sst.shape, "(expect 900, 21, 81); land cells:", int(np.isnan(sst).all(0).sum()), "(expect 75)")
mine = np.ascontiguousarray(sst, dtype=np.float32)
print("float32 array sha256:", hashlib.sha256(mine.tobytes()).hexdigest(), "\nissued              :", ISSUED_ARRAY_SHA)

z = np.load(issued)["sst"]
same = mine.shape == z.shape and np.array_equal(np.isnan(mine), np.isnan(z))
diff = np.nanmax(np.abs(mine - z)) if same else float("nan")
print("\nSUBSET MATCH:", "IDENTICAL" if same and diff == 0 else f"DIFFERS (same shape/mask: {same}, max|diff| = {diff})")
if same and diff != 0:
    bad = np.where(np.nanmax(np.abs(mine - z), axis=(1, 2)) > 0)[0]
    print("months that differ:", [f"{yr[ti[0]+i]}-{mo[ti[0]+i]:02d}" for i in bad[:12]], "..." if len(bad) > 12 else "", f"({len(bad)} months)")
    print("Report this difference in the note and use the ISSUED copies (Annex D3).")