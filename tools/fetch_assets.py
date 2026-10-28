"""Runs on GitHub Actions: find public-domain (CC0) museum images, cut them out, save PNGs + credits."""
import io
import json
import os
import sys
import time

import requests
from PIL import Image
from rembg import new_session, remove

qfile, out = sys.argv[1], sys.argv[2]
N_PER = 4
os.makedirs(f"{out}/cut", exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "balasa-asset-fetch/1.0 (personal art project)"
sess = new_session("isnet-general-use")
manifest, seen = [], set()


def save(src, oid, title, meta, url):
    key = f"{src}_{oid}"
    if key in seen:
        return False
    seen.add(key)
    try:
        r = S.get(url, timeout=60)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        im.thumbnail((900, 900))
        cut = remove(im, session=sess)
        bb = cut.getbbox()
        if not bb:
            return False
        cut.crop(bb).save(f"{out}/cut/{key}.png", optimize=True)
        manifest.append(dict(id=key, title=title, image=url, **meta))
        print("ok", key, title)
        return True
    except Exception as e:  # keep going on any single failure
        print("fail", key, e)
        return False


for line in open(qfile):
    q = line.strip()
    if not q or q.startswith("#"):
        continue
    try:
        ids = S.get("https://collectionapi.metmuseum.org/public/collection/v1/search",
                    params=dict(q=q, hasImages="true"), timeout=60).json().get("objectIDs") or []
        got = 0
        for oid in ids[:40]:
            o = S.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}", timeout=60).json()
            if o.get("isPublicDomain") and o.get("primaryImageSmall"):
                if save("met", oid, o.get("title"), dict(source="The Met Open Access (CC0)", date=o.get("objectDate"),
                         culture=o.get("culture"), query=q, page=o.get("objectURL")), o["primaryImageSmall"]):
                    got += 1
                if got >= N_PER:
                    break
            time.sleep(0.1)
    except Exception as e:
        print("met fail", q, e)
    try:
        d = S.get("https://openaccess-api.clevelandart.org/api/artworks/",
                  params=dict(q=q, cc0=1, has_image=1, limit=N_PER), timeout=60).json()
        for a in d.get("data", []):
            img = ((a.get("images") or {}).get("web") or {}).get("url")
            if img:
                save("cma", a["id"], a.get("title"), dict(source="Cleveland Museum of Art Open Access (CC0)",
                     date=a.get("creation_date"), culture=(a.get("culture") or [None])[0], query=q, page=a.get("url")), img)
    except Exception as e:
        print("cma fail", q, e)

json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print("total", len(manifest))
