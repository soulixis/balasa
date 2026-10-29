"""Runs on GitHub Actions: fetch public-domain / CC0 images, cut them out (or keep whole), save + credits.
Sources: Cleveland Museum of Art (CC0), Wikimedia Commons (public domain / CC0 only), Artvee (public domain)."""
import io
import json
import os
import re
import sys
import time
from urllib.parse import quote_plus

import requests
from PIL import Image
from rembg import new_session, remove

qfile, out = sys.argv[1], sys.argv[2]
for sub in ("cut", "raw", "debug"):
    os.makedirs(f"{out}/{sub}", exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "balasa-asset-fetch/1.0 (https://github.com/soulixis/balasa; personal art project)"
sess = new_session("isnet-general-use")
mpath = f"{out}/manifest.json"
manifest = json.load(open(mpath)) if os.path.exists(mpath) else []
seen = {m["id"] for m in manifest}


def save(key, url, mode, meta):
    key = re.sub(r"[^A-Za-z0-9_]+", "_", key)[:80]
    if key in seen:
        return False
    seen.add(key)
    try:
        r = S.get(url, timeout=90)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        im.thumbnail((1200, 1200))
        if mode == "raw":
            im.save(f"{out}/raw/{key}.jpg", quality=88)
        else:
            cut = remove(im, session=sess)
            bb = cut.getbbox()
            if not bb:
                return False
            cut.crop(bb).save(f"{out}/cut/{key}.png", optimize=True)
        manifest.append(dict(id=key, image=url, mode=mode, **meta))
        print("ok", key)
        return True
    except Exception as e:
        print("fail", key, e)
        return False


def cma(q, n, mode):
    d = S.get("https://openaccess-api.clevelandart.org/api/artworks/", params=dict(q=q, cc0=1, has_image=1, limit=n), timeout=60).json()
    for a in d.get("data", []):
        img = ((a.get("images") or {}).get("web") or {}).get("url")
        if img:
            save(f"cma_{a['id']}", img, mode, dict(title=a.get("title"), source="Cleveland Museum of Art Open Access (CC0)",
                 date=a.get("creation_date"), query=q, page=a.get("url")))


def commons(q, n, mode):
    d = S.get("https://commons.wikimedia.org/w/api.php", params=dict(
        action="query", format="json", generator="search", gsrsearch=f"{q} filetype:bitmap", gsrnamespace=6, gsrlimit=30,
        prop="imageinfo", iiprop="url|extmetadata", iiurlwidth=1400), timeout=60).json()
    got = 0
    for p in sorted((d.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 0)):
        ii = (p.get("imageinfo") or [{}])[0]
        lic = ((ii.get("extmetadata") or {}).get("LicenseShortName") or {}).get("value", "")
        if not re.search(r"public domain|pd|cc0", lic, re.I):
            continue
        if save(f"wc_{p['pageid']}", ii.get("thumburl") or ii.get("url"), mode,
                dict(title=p.get("title"), source=f"Wikimedia Commons ({lic})", query=q, page=ii.get("descriptionurl"))):
            got += 1
        if got >= n:
            break


def artvee(q, n, mode):
    r = S.get(f"https://artvee.com/main/?s={quote_plus(q)}", timeout=60)
    open(f"{out}/debug/artvee_{re.sub(r'[^a-z]+', '_', q)}.html", "w").write(r.text[:400000])
    urls = []
    for u in re.findall(r'https://mdl\.artvee\.com/[^"\'\s>]+?\.(?:jpg|jpeg|png)', r.text):
        if u not in urls:
            urls.append(u)
    for i, u in enumerate(urls[:n]):
        save(f"artvee_{re.sub(r'[^a-z]+', '_', q)}_{i}", u, mode, dict(title=u.rsplit('/', 1)[-1], source="Artvee (public domain)", query=q))


for line in open(qfile):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    src, q, n, mode = line.split("|")
    try:
        {"cma": cma, "commons": commons, "artvee": artvee}[src](q, int(n), mode)
    except Exception as e:
        print("source fail", line, e)
    time.sleep(0.5)

json.dump(manifest, open(mpath, "w"), indent=1)
print("total", len(manifest))
