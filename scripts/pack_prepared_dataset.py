"""Package prepared CNRPark crop subset for *permitted* internal sharing only.

Not included in the distributed upgrade ZIP. Verify dataset conditions first.
"""
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'dataset'/'real_cnrpark'
target=ROOT/'SmartParking_CNRPark_Prepared_Local.zip'
if not (source/'split_manifest.csv').is_file():
    raise SystemExit('Prepared dataset missing; run cnrpark_real.py prepare first')
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(source.rglob('*')):
        if p.is_file():
            z.write(p,arcname=str(Path('real_cnrpark')/p.relative_to(source)))
print('Created (for permitted internal use only):',target)
print('Source & attribution: https://cnrpark.it/ . Check the original dataset terms before distributing.')
