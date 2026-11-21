"""Svuota tasche a tema Pikachu - generatore parametrico (manifold3d).
Uscite: STL per colore, STL unico, 3MF multicolore, JSON per l'anteprima."""
import json, math, sys, zipfile, base64, io
import numpy as np
from manifold3d import Manifold, Mesh, CrossSection

OUT = sys.argv[1] if len(sys.argv) > 1 else "."

# ---------------- parametri (mm) ----------------
L, WD = 180.0, 88.0      # lunghezza e larghezza esterne
H = 22.0                 # altezza bordo
W = 4.0                  # spessore parete
F = 3.0                  # spessore fondo
RF = 4.0                 # raggio raccordo fondo/parete
INLAY = 0.8              # profondita' intarsi colorati (faccia)
M = 72                   # segmenti per semicerchio
R0 = WD / 2
A = L / 2 - R0           # semi-distanza dei centri

def stadium(d):
    r = R0 + d
    pts = []
    for i in range(M + 1):
        t = -math.pi / 2 + math.pi * i / M
        pts.append((A + r * math.cos(t), r * math.sin(t)))
    for i in range(M + 1):
        t = math.pi / 2 + math.pi * i / M
        pts.append((-A + r * math.cos(t), r * math.sin(t)))
    return pts

def loft(profile):
    """profile: lista (offset d, z) dal basso verso l'alto -> solido chiuso."""
    n = 2 * M + 2
    verts, tris = [], []
    for d, z in profile:
        verts += [(x, y, z) for x, y in stadium(d)]
    k = len(profile)
    for j in range(k - 1):
        for i in range(n):
            a, b = j * n + i, j * n + (i + 1) % n
            c, e = a + n, b + n
            tris += [(a, b, e), (a, e, c)]
    cb = len(verts); verts.append((0, 0, profile[0][1]))
    ct = len(verts); verts.append((0, 0, profile[-1][1]))
    top = (k - 1) * n
    for i in range(n):
        tris.append((cb, (i + 1) % n, i))
        tris.append((ct, top + i, top + (i + 1) % n))
    m = Manifold(Mesh(vert_properties=np.array(verts, np.float32),
                      tri_verts=np.array(tris, np.uint32)))
    if m.volume() < 0 or m.status().name != "NoError":
        raise RuntimeError(m.status())
    return m

def arc(cd, cz, r, a0, a1, n=12):
    return [(cd + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cz + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]

# --- guscio esterno: base raccordata + bordo superiore arrotondato
outer = [(-1.2 + 1.2 * math.sin(math.radians(t)), 1.2 - 1.2 * math.cos(math.radians(t)))
         for t in range(0, 91, 10)]
outer += arc(-W / 2, H - W / 2, W / 2, 0, 90, 10)[0:]
outer = [outer[0]] + [p for i, p in enumerate(outer[1:]) if abs(p[1] - outer[i][1]) > 1e-6]
# --- cavita': raccordo sul fondo, parete, labbro arrotondato
cav = arc(-W - RF, F + RF, RF, 270, 360, 10)
cav += arc(-W / 2, H - W / 2, W / 2, 180, 100, 8)
cav.append((cav[-1][0], H + 3))
cav = [cav[0]] + [p for i, p in enumerate(cav[1:]) if p[1] - cav[i][1] > 1e-6]

shell = loft(outer)
cavity = loft(cav)

# ---------------- orecchie ----------------
def ear(side):
    base = Manifold.sphere(1, 64).scale((16, 15, 18))
    tip = Manifold.sphere(1, 48).scale((8, 7, 9)).translate((31, 0, 5))
    e = Manifold.batch_hull([base, tip])
    cut = Manifold.cube((200, 200, 200), True).translate((100 + 27, 0, 0))   # x_locale > 27 = punta nera
    phi, theta = math.radians(58), 48
    bx = A + (R0 - 7) * math.cos(phi)
    by = (R0 - 7) * math.sin(phi)
    def place(m):
        m = m.rotate((0, 0, theta)).translate((bx, by, 0))
        return m.mirror((1, 0, 0)) if side < 0 else m
    floor_cut = Manifold.cube((400, 400, 100), True).translate((0, 0, 50))
    yel = place(e - cut) ^ floor_cut
    blk = place(e ^ cut) ^ floor_cut
    return yel, blk

eyl, ebl = ear(1)
eyr, ebr = ear(-1)

# ---------------- faccia (intarsi a filo) ----------------
def circ(x, y, r):
    return CrossSection.circle(r, 64).translate((x, y))

def stroke(points, w):
    parts = [CrossSection.batch_hull([circ(*p, w / 2), circ(*q, w / 2)])
             for p, q in zip(points, points[1:])]
    return CrossSection.batch_boolean(parts, __import__("manifold3d").OpType.Add)

eyes = circ(-25, 5, 4.6) + circ(25, 5, 4.6)
mr = 5.5
mouth_pts = [(-mr + mr * math.cos(math.radians(t)), -1 + mr * math.sin(math.radians(t)))
             for t in range(180, 361, 10)]
mouth_pts += [(mr + mr * math.cos(math.radians(t)), -1 + mr * math.sin(math.radians(t)))
              for t in range(190, 361, 10)]
mouth = stroke(mouth_pts, 2.4)
cheeks = circ(-53, -11, 10) + circ(53, -11, 10)

def inlay(cs):
    return Manifold.extrude(cs, INLAY).translate((0, 0, F - INLAY))

black_face = inlay(eyes + mouth)
red_face = inlay(cheeks)

yellow = (shell + eyl + eyr) - cavity - black_face - red_face
black = ((ebl + ebr) - cavity) + black_face
red = red_face

parts = {"giallo": (yellow, "#F7C600"), "nero": (black, "#1C1C1C"), "rosso": (red, "#E3262B")}

# ---------------- export ----------------
def arrays(m):
    mesh = m.to_mesh()
    return np.asarray(mesh.vert_properties)[:, :3].astype(np.float64), np.asarray(mesh.tri_verts).astype(np.int64)

def stl_bytes(meshes):
    tris = []
    for v, f in meshes:
        tris.append(v[f])
    t = np.concatenate(tris)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, t
    return b"Svuota tasche Pikachu".ljust(80, b" ") + np.uint32(len(t)).tobytes() + rec.tobytes()

data = {k: arrays(m) for k, (m, _) in parts.items()}
files = {}
for k, (v, f) in data.items():
    files[f"svuotatasche_pikachu_{k}.stl"] = stl_bytes([(v, f)])
files["svuotatasche_pikachu_unico.stl"] = stl_bytes(list(data.values()))

# 3MF: un oggetto con 3 parti (componenti), colori in basematerials
objs, comps, mats = [], [], []
for i, (k, (m, col)) in enumerate(parts.items()):
    v, f = data[k]
    mats.append(f'<base name="{k}" displaycolor="{col}FF"/>')
    vs = "".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in v)
    ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in f)
    objs.append(f'<object id="{i+2}" name="{k}" type="model" pid="1" pindex="{i}"><mesh><vertices>{vs}</vertices><triangles>{ts}</triangles></mesh></object>')
    comps.append(f'<component objectid="{i+2}"/>')
model = ('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="it-IT" '
         'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
         '<metadata name="Title">Svuota tasche Pikachu</metadata><resources>'
         f'<basematerials id="1">{"".join(mats)}</basematerials>{"".join(objs)}'
         f'<object id="10" name="Svuota tasche Pikachu" type="model"><components>{"".join(comps)}</components></object>'
         '</resources><build><item objectid="10" transform="1 0 0 0 1 0 0 0 1 120 120 0"/></build></model>')
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
    z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    z.writestr("3D/3dmodel.model", model)
files["svuotatasche_pikachu.3mf"] = buf.getvalue()

for name, b in files.items():
    open(f"{OUT}/{name}", "wb").write(b)

# JSON compatto per l'anteprima web (int16, 0.01 mm)
prev = {}
for k, (v, f) in data.items():
    q = np.round(v * 100).astype("<i2")
    prev[k] = {"color": parts[k][1], "v": base64.b64encode(q.tobytes()).decode(),
               "i": base64.b64encode(f.astype("<u4").tobytes()).decode()}
json.dump(prev, open(f"{OUT}/preview.json", "w"))

allv = np.concatenate([v for v, _ in data.values()])
print("ingombro mm:", np.round(allv.min(0), 1), np.round(allv.max(0), 1))
for k, (m, _) in parts.items():
    print(k, "vol cm3=%.1f" % (m.volume() / 1000), "tri", m.num_tri(), "genus", m.genus(), m.status())
