"""Fetch pinned author data. Dataset is intentionally absent from submission ZIP."""
import io, urllib.request, zipfile
import dreci as d

dest=d.ROOT/'local_data/intra_sent_causality.json'
expected=d.read(d.ROOT/'data_info/audit.json')['data_sha256']
if dest.exists() and d.digest(dest.read_bytes())==expected:
    print('Pinned data already present')
else:
    try:content=urllib.request.urlopen(d.DATA_URL,timeout=30).read()
    except Exception:
        url=f'https://codeload.github.com/DMIRLAB-Group/Dr.ECI/zip/{d.SHA}'
        with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url,timeout=120).read())) as z:
            name=next(n for n in z.namelist() if n.endswith('/data/ESC/intra_sent_causality.json'))
            content=z.read(name)
    if d.digest(content)!=expected:raise ValueError('Downloaded file checksum differs from frozen benchmark')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content);print('Downloaded and SHA256 verified')
