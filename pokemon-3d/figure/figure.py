"""Figure Pokemon stile chibi - generatore parametrico (manifold3d).
Uso: python3 figure.py [cartella_uscita] [nomi...]
Le decorazioni (occhi, guance, pancia...) sono intarsi a filo di 0,8 mm sulla superficie."""
import json, math, os, sys
import numpy as np
import manifold3d
from manifold3d import Manifold, CrossSection

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "svuotatasche"))
from genera import stl_bytes, threemf_bytes, arrays, COL  # noqa: E402

ADD = manifold3d.OpType.Add
DEPTH = 0.8
N = 96
BIG = 600
GROUND = Manifold.cube((BIG, BIG, BIG), True).translate((0, 0, BIG / 2))


def ell(r, c, n=N):
    return Manifold.sphere(1, n).scale(r).translate(c)

def union(ms):
    ms = [m for m in ms if m is not None and not m.is_empty()]
    return Manifold.batch_boolean(ms, ADD) if len(ms) > 1 else ms[0]

def C(r, x, z, n=64):
    return CrossSection.circle(r, n).translate((x, z))

def E2(rx, rz, x, z, rot=0):
    return CrossSection.circle(1, 64).scale((rx, rz)).rotate(rot).translate((x, z))

def stroke(points, w):
    segs = [CrossSection.batch_hull([C(w / 2, *a, 24), C(w / 2, *b, 24)]) for a, b in zip(points, points[1:])]
    return CrossSection.batch_boolean(segs, ADD)

def arcpts(x, z, r, a0, a1, step=8):
    n = max(2, int(abs(a1 - a0) / step))
    return [(x + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)), z + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]

def omega(x, z, r):
    return arcpts(x - r, z, r, 180, 360) + arcpts(x + r, z, r, 180, 360)[1:]

def prism_y(cs):
    """estrude una forma 2D (x, z) lungo tutto l'asse Y"""
    return Manifold.extrude(cs, BIG).translate((0, 0, -BIG / 2)).rotate((90, 0, 0))

def decals(r, c, feats, side=-1):
    """intarsi sulla superficie di un ellissoide (r, c), visti dal davanti (side=-1) o da dietro (+1)."""
    # +0,05 mm: evita superfici coincidenti con il corpo (invisibile in stampa)
    shell = ell(tuple(v + 0.05 for v in r), c) - ell(tuple(v - DEPTH for v in r), c)
    half = Manifold.cube((BIG, BIG, BIG), True).translate((0, c[1] + side * BIG / 2, 0))
    shell = shell ^ half
    out = []
    for i, (col, cs) in enumerate(feats):
        later = [f for _, f in feats[i + 1:]]
        if later:
            cs = cs - CrossSection.batch_boolean(later, ADD)
        if not cs.is_empty():
            out.append((col, shell ^ prism_y(cs)))
    return out

def ear(base, direction, length, rb, rt=1.5):
    d = np.array(direction, float); d /= np.linalg.norm(d)
    tip = np.array(base) + d * length
    return Manifold.batch_hull([ell(rb, base, 48), ell((rt, rt, rt), tuple(tip), 24)]), tuple(tip)


def assemble(body_parts, decal_list, extra=None):
    """body_parts: {colore: [solidi]}; decal_list: [(colore, solido)] da scavare nel corpo."""
    parts = {}
    for col, ms in body_parts.items():
        parts[col] = union(ms) ^ GROUND
    order = list(parts)
    # i colori dichiarati prima "vincono" sugli altri dove si sovrappongono
    for i, a in enumerate(order):
        for b in order[:i]:
            parts[a] = parts[a] - parts[b]
    dec = {}
    for col, m in decal_list:
        dec[col] = dec[col] + m if col in dec else m
    alldec = union(list(dec.values())) if dec else None
    if alldec is not None:
        for k in parts:
            parts[k] = parts[k] - alldec
    for col, m in dec.items():
        parts[col] = parts[col] + m if col in parts else m
    clean = {}
    for k, v in parts.items():
        comps = [c for c in v.decompose() if c.volume() > 1.0]   # via i frammenti di spessore nullo
        if comps:
            clean[k] = union(comps)
    return clean


# ---------------- Pikachu ----------------
def pikachu():
    body_r, body_c = (21, 18, 25), (0, 0, 23)
    head_r, head_c = (25, 21, 20), (0, -3, 60)
    yellow = [ell(body_r, body_c), ell(head_r, head_c),
              ell((7.5, 11, 5), (-11, -9, 2.5)), ell((7.5, 11, 5), (11, -9, 2.5)),
              ell((5.5, 6, 9), (-14, -15, 35)).rotate((0, 0, 0)), ell((5.5, 6, 9), (14, -15, 35))]
    black = []
    for s in (1, -1):
        e, tip = ear((s * 13, -2, 74), (s * 0.55, 0.05, 1), 34, (7.5, 4.5, 7.5))
        tipball = ell((13, 13, 13), tip, 48)
        yellow.append(e - tipball)
        black.append(e ^ tipball)
    bolt = [(y - 5, z) for y, z in [(12, 0), (24, 0), (30, 14), (26, 16), (36, 30), (32, 32), (46, 58), (34, 60),
            (24, 38), (28, 36), (18, 22), (22, 20), (14, 6)]]
    tail = Manifold.extrude(CrossSection([[(y, z) for y, z in bolt]]), 6).translate((0, 0, -3))
    # piano 2D (u,v) -> (y,z): (x,y,z) -> (z, x, y)
    m = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]], float)
    tail = tail.transform(m)
    brown_tail = tail ^ Manifold.cube((BIG, BIG, 9), True).translate((0, 0, 4.5))
    yellow.append(tail - brown_tail)
    feats_head = [("rosso", C(6.2, -16.5, 55) + C(6.2, 16.5, 55)),
                  ("nero", E2(4.3, 5.2, -10, 64) + E2(4.3, 5.2, 10, 64) + E2(1.4, 1.0, 0, 59.5)
                   + stroke(omega(0, 57.5, 2.6), 1.4)),
                  ("bianco", C(1.6, -11.2, 66) + C(1.6, 8.8, 66))]
    feats_back = [("marrone", E2(11, 2.6, 0, 34, 0) + E2(10, 2.4, 0, 27, 0))]
    dl = decals(head_r, head_c, feats_head) + decals(body_r, body_c, feats_back, side=1)
    return assemble({"giallo": yellow, "nero": black, "marrone": [brown_tail]}, dl)


# ---------------- Jigglypuff ----------------
def jigglypuff():
    r, c = (30, 28, 29), (0, 0, 24)
    pink = [ell(r, c), ell((7, 9, 5), (-12, -16, 2)), ell((7, 9, 5), (12, -16, 2)),
            ell((5, 5, 8), (-27, -8, 26)), ell((5, 5, 8), (27, -8, 26))]
    black = []
    for s in (1, -1):
        e, tip = ear((s * 17, 0, 44), (s * 0.75, 0, 1), 20, (9, 4.5, 9), 2)
        inner, _ = ear((s * 17.5, -2.6, 45), (s * 0.75, 0, 1), 16, (6, 2.4, 6), 1.2)
        pink.append(e)
        black.append(inner ^ Manifold.cube((BIG, BIG, BIG), True).translate((0, -2.6 - BIG / 2 + 0.01, 0)))
    # ciuffo arricciato sulla fronte
    sp = [(9 * (1 - t / 760) * math.cos(math.radians(t + 90)), 10 * (1 - t / 760) * math.sin(math.radians(t + 90)))
          for t in range(0, 560, 15)]
    curl = Manifold.extrude(stroke([(x, 44 + z) for x, z in sp], 4.2), 7).translate((0, 0, -3.5)) \
        .transform(np.array([[1, 0, 0, 0], [0, 0, 1, -19], [0, 1, 0, 0]], float))
    pink.append(curl)
    feats = [("turchese", C(9, -11.5, 30) + C(9, 11.5, 30)),
             ("bianco", C(3.6, -14.5, 33) + C(3.6, 8.5, 33) + C(1.6, -8.5, 26.5) + C(1.6, 14.5, 26.5)),
             ("nero", stroke(arcpts(0, 21, 4, 210, 330), 1.6))]
    parts = assemble({"rosa": pink, "nero": black}, decals(r, c, feats))
    return parts


# ---------------- Snorlax ----------------
def snorlax():
    body_r, body_c = (34, 27, 31), (0, 0, 27)
    head_r, head_c = (24, 20, 16), (0, -4, 60)
    teal = [ell(body_r, body_c), ell(head_r, head_c),
            ell((8, 9, 15), (-31, -6, 34)), ell((8, 9, 15), (31, -6, 34))]
    for s in (1, -1):
        e, _ = ear((s * 13, -3, 70), (s * 0.45, 0, 1), 10, (5, 3.5, 5), 1.5)
        teal.append(e)
    cream = [ell((12, 8, 11), (-18, -22, 9)), ell((12, 8, 11), (18, -22, 9))]
    feats_body = [("crema", E2(25, 22, 0, 26))]
    feats_head = [("crema", E2(18, 11, 0, 57)),
                  ("nero", stroke(arcpts(-8, 64, 6, 235, 305, 10), 1.6) + stroke(arcpts(8, 64, 6, 235, 305, 10), 1.6)
                   + stroke(arcpts(0, 62, 11, 240, 300, 8), 1.6)),
                  ("bianco", CrossSection([[(-7, 51.9), (-3.6, 51.3), (-5.5, 55)]]) + CrossSection([[(3.6, 51.3), (7, 51.9), (5.5, 55)]]))]
    dl = decals(body_r, body_c, feats_body) + decals(head_r, head_c, feats_head)
    claws = []
    for s in (1, -1):
        for k in (-1, 0, 1):
            claws.append(ell((1.6, 1.6, 2.4), (s * 18 + k * 5, -29.5, 13.5), 24))
    return assemble({"bianco": claws, "crema": cream, "verde_acqua": teal}, dl)


# ---------------- Poke Ball ----------------
def pokeball():
    R, cz = 32, 27
    ball = ell((R, R, R), (0, 0, cz), 128)
    band = 3.2
    # pulsante: anello nero che sporge 1,5 mm, disco bianco che sporge 2,5 mm
    button = Manifold.cylinder(7.5, 10, 10, 96).rotate((90, 0, 0)).translate((0, -R + 6, cz))
    btn_w = Manifold.cylinder(8.5, 6.5, 6.5, 96).rotate((90, 0, 0)).translate((0, -R + 6, cz))
    zc = lambda lo, hi: Manifold.cube((BIG, BIG, hi - lo), True).translate((0, 0, (lo + hi) / 2))
    return assemble({"bianco": [btn_w], "nero": [(ball ^ zc(cz - band, cz + band)) + button],
                     "rosso": [ball ^ zc(cz + band, 200)], "bianco_basso": [ball ^ zc(-200, cz - band)]}, [])


FIGURE = {
    "pikachu_figura": ("Pikachu", pikachu),
    "jigglypuff_figura": ("Jigglypuff", jigglypuff),
    "snorlax_figura": ("Snorlax", snorlax),
    "pokeball_figura": ("Poké Ball", pokeball),
}

if __name__ == "__main__":
    OUT = sys.argv[1] if len(sys.argv) > 1 else "."
    only = sys.argv[2:] or list(FIGURE)
    summary = {}
    for key in only:
        nome, fn = FIGURE[key]
        parts = fn()
        if "bianco_basso" in parts:   # stessa plastica bianca: un'unica parte
            parts["bianco"] = parts.get("bianco", parts["bianco_basso"]) + parts.pop("bianco_basso") if "bianco" in parts else parts.pop("bianco_basso")
        d = os.path.join(OUT, key); os.makedirs(d, exist_ok=True)
        title = f"Figura {nome}"
        for k, m in parts.items():
            open(f"{d}/{key}_{k}.stl", "wb").write(stl_bytes(m, title))
        unico = union([c for c in union(list(parts.values())).decompose() if c.volume() > 1.0])
        open(f"{d}/{key}_unico.stl", "wb").write(stl_bytes(unico, title))
        open(f"{d}/{key}.3mf", "wb").write(threemf_bytes(parts, title))
        import base64
        prev = {}
        for k, m in parts.items():
            v, f = arrays(m)
            prev[k] = {"color": COL[k], "v": base64.b64encode(np.round(v * 100).astype("<i2").tobytes()).decode(),
                       "i": base64.b64encode(f.astype("<u4").tobytes()).decode()}
        json.dump(prev, open(f"{d}/preview.json", "w"))
        bb = unico.bounding_box()
        size = [round(bb[i + 3] - bb[i]) for i in range(3)]
        grams = round(unico.volume() / 1000 * 1.24 * 0.45)
        summary[key] = dict(nome=nome, ingombro=size, grammi=grams, colori=list(parts),
                            status={k: (m.status().name, m.genus(), round(m.volume())) for k, m in parts.items()},
                            unico_genus=unico.genus())
        print(key, summary[key])
    json.dump(summary, open(os.path.join(OUT, "figure.json"), "w"), indent=1)
