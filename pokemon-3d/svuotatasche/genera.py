"""Svuota tasche a tema Pikachu - generatore parametrico di varianti (manifold3d).
Uso: python3 genera.py [cartella_uscita]
Per ogni variante crea: 3MF multicolore, STL per colore, STL unico, JSON di anteprima."""
import json, math, sys, zipfile, base64, io, os
import numpy as np
import manifold3d
from manifold3d import Manifold, Mesh, CrossSection

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
COL = {"giallo": "#F7C600", "nero": "#1C1C1C", "rosso": "#E3262B", "marrone": "#7A4A22",
       "bianco": "#F4F4F0", "verde_acqua": "#2E6E78", "crema": "#EBD7A8", "nocciola": "#B5793F",
       "cioccolato": "#4E321A", "rosa": "#F6A9C4", "turchese": "#1F8A9A",
       "verde": "#1F7A3F", "oro": "#D9A82E", "argento": "#C3C9D2", "arancio": "#F28A22", "ghiaccio": "#9FD3EC"}
ADD = manifold3d.OpType.Add

VARIANTI = {
    # 1. la piu' semplice: tonda, faccia in rilievo, si stampa anche in un solo colore
    "mini": dict(nome="Mini tonda", L=120, WD=120, H=18, W=3.2, F=2.4, RF=3,
                 ear=dict(s=0.72, phi=50, theta=58, inset=5), face=dict(s=0.62, cx=0, cy=-4),
                 face_mode="rilievo", divider=None, tail=None),
    # 2. classica ovale con faccia in rilievo (monocolore o cambio filamento)
    "classica_rilievo": dict(nome="Classica in rilievo", L=180, WD=88, H=22, W=4, F=3, RF=4,
                 ear=dict(s=1, phi=58, theta=48, inset=7), face=dict(s=1, cx=0, cy=0),
                 face_mode="rilievo", divider=None, tail=None),
    # 3. classica ovale con faccia intarsiata a filo (multicolore AMS/MMU)
    "classica": dict(nome="Classica intarsiata", L=180, WD=88, H=22, W=4, F=3, RF=4,
                 ear=dict(s=1, phi=58, theta=48, inset=7), face=dict(s=1, cx=0, cy=0),
                 face_mode="intarsio", divider=None, tail=None),
    # 4. la piu' complessa: due scomparti + coda a fulmine con base marrone, 4 colori
    "deluxe": dict(nome="Deluxe con coda", L=160, WD=92, H=24, W=4, F=3, RF=4,
                 ear=dict(s=1, phi=60, theta=50, inset=7), face=dict(s=0.72, cx=-24, cy=-1),
                 face_mode="intarsio", divider=dict(x=33, t=3.2, drop=5),
                 tail=dict(s=0.85, x_in=4, y=-6, theta=22, h=7)),
    # --- altri Pokemon ---
    "pokeball": dict(nome="Poké Ball", L=130, WD=130, H=20, W=3.6, F=3, RF=4, body="split_pokeball",
                 ear=None, face_fn="pokeball", face_mode="intarsio", divider=None, tail=None),
    "eevee": dict(nome="Eevee", L=150, WD=100, H=22, W=4, F=3, RF=4, body="nocciola",
                 ear=dict(s=1.0, ln=1.35, phi=52, theta=62, inset=7, col="nocciola", tip="cioccolato", tipf=0.78),
                 face_fn="eevee", face_mode="intarsio", divider=None, tail=None),
    "snorlax": dict(nome="Snorlax", L=170, WD=110, H=22, W=4, F=3, RF=4, body="verde_acqua",
                 ear=dict(s=0.78, ln=0.9, phi=60, theta=70, inset=6, col="verde_acqua", tip=None),
                 face_fn="snorlax", face_mode="intarsio", divider=None, tail=None),
}

INLAY, RELIEF = 0.8, 1.0
M = 72


def build(p):
    L, WD, H, W, F, RF = p["L"], p["WD"], p["H"], p["W"], p["F"], p["RF"]
    R0 = WD / 2
    A = L / 2 - R0

    def stadium(d):
        r = R0 + d
        pts = []
        for c, t0 in ((A, -math.pi / 2), (-A, math.pi / 2)):
            for i in range(M + 1):
                t = t0 + math.pi * i / M
                pts.append((c + r * math.cos(t), r * math.sin(t)))
        return pts

    def loft(profile):
        n = 2 * M + 2
        verts, tris = [], []
        for d, z in profile:
            verts += [(x, y, z) for x, y in stadium(d)]
        k = len(profile)
        for j in range(k - 1):
            for i in range(n):
                a, b = j * n + i, j * n + (i + 1) % n
                tris += [(a, b, b + n), (a, b + n, a + n)]
        cb = len(verts); verts.append((0, 0, profile[0][1]))
        ct = len(verts); verts.append((0, 0, profile[-1][1]))
        top = (k - 1) * n
        for i in range(n):
            tris.append((cb, (i + 1) % n, i))
            tris.append((ct, top + i, top + (i + 1) % n))
        m = Manifold(Mesh(vert_properties=np.array(verts, np.float32),
                          tri_verts=np.array(tris, np.uint32)))
        assert m.status().name == "NoError" and m.volume() > 0
        return m

    def arc(cd, cz, r, a0, a1, n=12):
        return [(cd + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
                 cz + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]

    def dedup(pts):
        out = [pts[0]]
        for q in pts[1:]:
            if q[1] - out[-1][1] > 1e-6:
                out.append(q)
        return out

    outer = [(-1.2 + 1.2 * math.sin(math.radians(t)), 1.2 - 1.2 * math.cos(math.radians(t))) for t in range(0, 91, 10)]
    outer = dedup(outer + arc(-W / 2, H - W / 2, W / 2, 0, 90, 10))
    cav = arc(-W - RF, F + RF, RF, 270, 360, 10) + arc(-W / 2, H - W / 2, W / 2, 180, 100, 8)
    cav = dedup(cav + [(cav[-1][0], H + 3)])
    shell, cavity = loft(outer), loft(cav)
    floor_cut = Manifold.cube((600, 600, 200), True).translate((0, 0, 100))

    parts = {}
    def add(col, m):
        if m is None or m.is_empty():
            return
        parts[col] = parts[col] + m if col in parts else m

    body_col = p.get("body", "giallo")
    body = shell

    # --- orecchie (cupola a goccia, punta di colore diverso)
    e = p.get("ear")
    extra_body = []
    if e:
        s, ln = e["s"], e.get("ln", 1.0)
        ear0 = Manifold.batch_hull([Manifold.sphere(1, 64).scale((16 * s, 15 * s, 18 * s)),
                                    Manifold.sphere(1, 48).scale((8 * s, 7 * s, 9 * s)).translate((31 * s * ln, 0, 5 * s))])
        tipx = (e.get("tipf", 0.87) * 31 * ln) * s
        cut = Manifold.cube((300, 300, 300), True).translate((150 + tipx, 0, 0))
        phi = math.radians(e["phi"])
        bx, by = A + (R0 - e["inset"]) * math.cos(phi), (R0 - e["inset"]) * math.sin(phi)
        for side in (1, -1):
            def place(m):
                m = m.rotate((0, 0, e["theta"])).translate((bx, by, 0))
                return m.mirror((1, 0, 0)) if side < 0 else m
            tipc = e.get("tip", "nero") if "tip" in e else "nero"
            if tipc:
                extra_body.append((e.get("col", body_col), place(ear0 - cut) ^ floor_cut))
                add(tipc, (place(ear0 ^ cut) ^ floor_cut) - cavity)
            else:
                extra_body.append((e.get("col", body_col), place(ear0) ^ floor_cut))
    for c, m in extra_body:
        if c == body_col:
            body = body + m
        else:
            add(c, m - shell - cavity)

    # --- coda a fulmine (piatta, appoggiata al piatto)
    t = p.get("tail")
    if t:
        ts = t["s"]
        bolt = [(0, -4), (16, -4), (12, 4), (28, 4), (24, 12), (46, 12), (46, 26), (18, 26),
                (22, 18), (6, 18), (10, 8), (0, 8)]
        cs = CrossSection([[(x * ts, y * ts) for x, y in bolt]]).offset(-1.2, manifold3d.JoinType.Round).offset(1.2, manifold3d.JoinType.Round)
        tail = Manifold.extrude(cs, t["h"])
        base_box = Manifold.cube((14 * ts + 20, 100, 100), True).translate((14 * ts - (14 * ts + 20) / 2, 0, 0))
        def placet(m):
            return m.translate((-t["x_in"], 0, 0)).rotate((0, 0, t["theta"])).translate((A + R0, t["y"], 0))
        add("marrone", placet(tail ^ base_box) - shell - cavity)
        body = body + placet(tail - base_box)

    # --- divisorio (affonda nel fondo e nelle pareti per saldarsi)
    d = p.get("divider")
    if d:
        prof = CrossSection.square((d["t"], H - d["drop"] - d["t"] / 2)).translate((-d["t"] / 2, 0)) + \
               CrossSection.circle(d["t"] / 2, 32).translate((0, H - d["drop"] - d["t"] / 2))
        wall = Manifold.extrude(prof, 300).translate((0, 0, -150)).rotate((90, 0, 0)).translate((d["x"], 0, 0))
    body = body - cavity
    if d:
        body = body + (wall ^ shell)

    # --- faccia: lista (colore, forma 2D); le forme successive "bucano" le precedenti
    feats = FACES[p.get("face_fn", "pikachu")](p)
    final = []
    for i, (c, cs) in enumerate(feats):
        later = [f for _, f in feats[i + 1:]]
        if later:
            cs = cs - CrossSection.batch_boolean(later, ADD)
        final.append((c, cs))
    if p["face_mode"] == "intarsio":
        z0, hh = F - INLAY, INLAY
    else:
        z0, hh = F, RELIEF
    feat_m = [(c, Manifold.extrude(cs, hh).translate((0, 0, z0))) for c, cs in final if not cs.is_empty()]
    if p["face_mode"] == "intarsio":
        for _, m in feat_m:
            body = body - m

    # --- colore del corpo (o suddivisione Poke Ball)
    if body_col == "split_pokeball":
        band = 3.5
        big = 400
        add("rosso", body ^ Manifold.cube((big, big, big), True).translate((0, big / 2 + band, 0)))
        add("bianco", body ^ Manifold.cube((big, big, big), True).translate((0, -big / 2 - band, 0)))
        add("nero", body ^ Manifold.cube((big, 2 * band, big), True))
    else:
        add(body_col, body)
    for c, m in feat_m:
        add(c, m)
    return parts


# ---------------- facce ----------------
def _tools(cx=0.0, cy=0.0, fs=1.0):
    def circ(x, y, r):
        return CrossSection.circle(r * fs, 64).translate((cx + x * fs, cy + y * fs))
    def ell(x, y, rx, ry, rot=0):
        return CrossSection.circle(1, 64).scale((rx * fs, ry * fs)).rotate(rot).translate((cx + x * fs, cy + y * fs))
    def stroke(points, w):
        segs = [CrossSection.batch_hull([circ(*a, w / 2), circ(*b, w / 2)]) for a, b in zip(points, points[1:])]
        return CrossSection.batch_boolean(segs, ADD)
    def arcpts(x, y, r, a0, a1, step=10):
        n = max(2, int(abs(a1 - a0) / step))
        return [(x + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)), y + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]
    return circ, ell, stroke, arcpts

def omega(arcpts, x, y, r):
    return arcpts(x - r, y, r, 180, 360) + arcpts(x + r, y, r, 180, 360)[1:]

def face_pikachu(p):
    f = p["face"]; fs = f["s"]
    circ, ell, stroke, arcpts = _tools(f["cx"], f["cy"], fs)
    w = 2.4 if fs >= 0.8 else 2.4 / fs * 0.8 * fs / fs
    return [("rosso", circ(-53, -11, 10) + circ(53, -11, 10)),
            ("nero", circ(-25, 5, 4.6) + circ(25, 5, 4.6) + stroke(omega(arcpts, 0, -1, 5.5), max(2.4, 2.0 / fs) if fs < 1 else 2.4))]

def face_pokeball(p):
    circ, ell, stroke, arcpts = _tools()
    return [("nero", circ(0, 0, 19)), ("bianco", circ(0, 0, 13)), ("nero", circ(0, 0, 13) - circ(0, 0, 10.5))]

def face_eevee(p):
    circ, ell, stroke, arcpts = _tools(0, -2, 1)
    return [("crema", ell(0, -16, 34, 13)),
            ("nero", ell(-24, 8, 7.5, 9) + ell(24, 8, 7.5, 9) + ell(0, -4, 3.2, 2.2)
                     + stroke(omega(arcpts, 0, -8, 4), 2.2)),
            ("bianco", circ(-21.5, 11.5, 2.8) + circ(26.5, 11.5, 2.8))]

def face_snorlax(p):
    circ, ell, stroke, arcpts = _tools(0, -4, 1)
    return [("crema", ell(0, 0, 66, 38)),
            ("nero", stroke(arcpts(-26, 22, 14, 245, 295), 2.6) + stroke(arcpts(26, 22, 14, 245, 295), 2.6)
                     + stroke(arcpts(0, 18, 34, 235, 305), 2.8)),
            ("bianco", CrossSection([[(-17, -10.9), (-11, -11.1), (-14, -5.5)]]).translate((0, -4))
                       + CrossSection([[(11, -11.1), (17, -10.9), (14, -5.5)]]).translate((0, -4)))]

FACES = {"pikachu": face_pikachu, "pokeball": face_pokeball, "eevee": face_eevee, "snorlax": face_snorlax}


# ---------------- export ----------------
def arrays(m):
    mesh = m.to_mesh()
    return np.asarray(mesh.vert_properties)[:, :3].astype(np.float64), np.asarray(mesh.tri_verts).astype(np.int64)

def stl_bytes(m, title):
    v, f = arrays(m)
    t = v[f]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, t
    return title.encode()[:80].ljust(80, b" ") + np.uint32(len(t)).tobytes() + rec.tobytes()

def threemf_bytes(parts, title):
    objs, comps, mats = [], [], []
    for i, (k, m) in enumerate(parts.items()):
        v, f = arrays(m)
        mats.append(f'<base name="{k}" displaycolor="{COL[k]}FF"/>')
        vs = "".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in v)
        ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in f)
        objs.append(f'<object id="{i+2}" name="{k}" type="model" pid="1" pindex="{i}"><mesh><vertices>{vs}</vertices><triangles>{ts}</triangles></mesh></object>')
        comps.append(f'<component objectid="{i+2}"/>')
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="it-IT" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             f'<metadata name="Title">{title}</metadata><resources>'
             f'<basematerials id="1">{"".join(mats)}</basematerials>{"".join(objs)}'
             f'<object id="20" name="{title}" type="model"><components>{"".join(comps)}</components></object>'
             '</resources><build><item objectid="20" transform="1 0 0 0 1 0 0 0 1 128 128 0"/></build></model>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr("3D/3dmodel.model", model)
    return buf.getvalue()


if __name__ == "__main__":
    only = sys.argv[2:] or list(VARIANTI)
    summary = {}
    for key in only:
        p = VARIANTI[key]
        parts = build(p)
        d = os.path.join(OUT, key); os.makedirs(d, exist_ok=True)
        title = f"Svuota tasche Pikachu - {p['nome']}"
        for k, m in parts.items():
            open(f"{d}/svuotatasche_pikachu_{key}_{k}.stl", "wb").write(stl_bytes(m, title))
        unico = Manifold.batch_boolean(list(parts.values()), ADD)
        open(f"{d}/svuotatasche_pikachu_{key}_unico.stl", "wb").write(stl_bytes(unico, title))
        open(f"{d}/svuotatasche_pikachu_{key}.3mf", "wb").write(threemf_bytes(parts, title))
        prev = {}
        for k, m in parts.items():
            v, f = arrays(m)
            prev[k] = {"color": COL[k], "v": base64.b64encode(np.round(v * 100).astype("<i2").tobytes()).decode(),
                       "i": base64.b64encode(f.astype("<u4").tobytes()).decode()}
        json.dump(prev, open(f"{d}/preview.json", "w"))
        bb = unico.bounding_box()
        size = [round(bb[i + 3] - bb[i]) for i in range(3)]
        grams = round(sum(m.volume() for m in parts.values()) / 1000 * 1.24 * 0.62)  # PLA, ~15% infill + 3 perimetri
        summary[key] = dict(nome=p["nome"], ingombro=size, grammi=grams, colori=list(parts),
                            status={k: (m.status().name, m.genus()) for k, m in parts.items()})
        print(key, summary[key])
    json.dump(summary, open(os.path.join(OUT, "varianti.json"), "w"), indent=1)
