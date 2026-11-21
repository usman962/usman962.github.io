"""Decorazioni di Natale - generatore parametrico (manifold3d).
Uso: python3 natale.py [cartella_uscita] [nomi...]"""
import base64, json, math, os, sys
import numpy as np
import manifold3d
from manifold3d import Manifold, CrossSection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "svuotatasche"))
sys.path.insert(0, os.path.join(HERE, "..", "figure"))
from genera import stl_bytes, threemf_bytes, arrays, COL  # noqa: E402
from figure import (ell, union, C, E2, stroke, arcpts, omega, decals, ear, assemble,  # noqa: E402
                    GROUND, BIG, ADD)


def star2d(n, ro, ri, rot=90):
    pts = []
    for i in range(2 * n):
        r = ro if i % 2 == 0 else ri
        a = math.radians(rot + 180 * i / n)
        pts.append((r * math.cos(a), r * math.sin(a)))
    return pts

def bevel_star(n, ro, ri, t, base=0.0, both=False):
    """stella sfaccettata: punte a quota 0 (o base), apice centrale a +t (e -t se both)."""
    pts = star2d(n, ro, ri)
    arms = []
    for i in range(0, 2 * n, 2):
        tip, a, b = pts[i], pts[i - 1], pts[(i + 1) % (2 * n)]
        P = [(tip[0], tip[1], 0), (a[0], a[1], 0), (b[0], b[1], 0), (0, 0, t)]
        P += [(0, 0, -t)] if both else [(tip[0], tip[1], -base), (a[0], a[1], -base), (b[0], b[1], -base), (0, 0, -base)]
        arms.append(Manifold.hull_points(P))
    return union(arms)

def ring(ro, ri, h):
    return Manifold.extrude(CrossSection.circle(ro, 64) - CrossSection.circle(ri, 64), h)

def torus(R, r, n=96):
    return Manifold.revolve(CrossSection.circle(r, 32).translate((R, 0)), n)


# ---------------- 1. Fiocco di neve ----------------
def fiocco():
    L, w, H = 42, 3.4, 3.0
    def arm(width):
        segs = [[(0, 0), (0, L)]]
        for t, ln in ((0.38, 13), (0.62, 10), (0.82, 6)):
            y = L * t
            for s in (1, -1):
                a = math.radians(90 - s * 55)
                segs.append([(0, y), (ln * math.cos(a) * -s * -1 if False else s * ln * math.sin(math.radians(55)), y + ln * math.cos(math.radians(55)))])
        cs = CrossSection.batch_boolean([stroke(sg, width) for sg in segs], ADD)
        return cs
    base_arm = arm(w)
    # rombo vicino alla punta
    diamond = CrossSection([[(0, L - 13), (4.5, L - 7.5), (0, L - 2), (-4.5, L - 7.5)]])
    flake = CrossSection.batch_boolean([base_arm.rotate(60 * k) + diamond.rotate(60 * k) for k in range(6)], ADD)
    hexa = CrossSection([[(9 * math.cos(math.radians(60 * k + 30)), 9 * math.sin(math.radians(60 * k + 30))) for k in range(6)]])
    flake = flake + hexa
    loop = (CrossSection.circle(5, 48) - CrossSection.circle(2.6, 48)).translate((0, L + 4.2))
    flake = flake + loop
    body = Manifold.extrude(flake, H)
    # dettaglio ghiaccio a filo sulla faccia superiore: linee centrali + esagono interno
    det = CrossSection.batch_boolean([stroke([(0, 11), (0, L - 15)], 1.2).rotate(60 * k) for k in range(6)], ADD) + \
        (CrossSection([[(6 * math.cos(math.radians(60 * k + 30)), 6 * math.sin(math.radians(60 * k + 30))) for k in range(6)]])
         - CrossSection([[(3.4 * math.cos(math.radians(60 * k + 30)), 3.4 * math.sin(math.radians(60 * k + 30))) for k in range(6)]]))
    inl = Manifold.extrude(det, 0.6).translate((0, 0, H - 0.6))
    return {"bianco": body - inl, "ghiaccio": inl}


# ---------------- 2. Pallina a spirale ----------------
def pallina():
    R = 32
    cz = R - 3            # base piatta di appoggio
    sphere = ell((R, R, R), (0, 0, cz), 160)
    # solchi a spirale da polo a polo: spicchi sottili ritorti, solo nel guscio esterno di 1,3 mm
    wedge = CrossSection([[(0, 0), (R + 8, -2.4), (R + 8, 2.4)]])
    wedges = CrossSection.batch_boolean([wedge.rotate(360 * k / 14) for k in range(14)], ADD)
    tw = Manifold.extrude(wedges, 2 * R + 4, n_divisions=120, twist_degrees=150).translate((0, 0, cz - R - 2))
    grooves = (sphere - ell((R - 1.3,) * 3, (0, 0, cz), 160)) ^ tw
    ball = (sphere - grooves) ^ GROUND
    # fascia centrale dorata, leggermente in rilievo
    band = (ell((R + 0.7,) * 3, (0, 0, cz), 160) ^ Manifold.cube((BIG, BIG, 6), True).translate((0, 0, cz))) - grooves
    cap = Manifold.cylinder(9, 7.5, 7.5, 64).translate((0, 0, cz + R - 3.5))
    cap_ribs = union([Manifold.cube((1.2, 16, 9), True).rotate((0, 0, a)).translate((0, 0, cz + R + 1)) for a in range(0, 180, 30)]) ^ Manifold.cylinder(9, 8.2, 8.2, 64).translate((0, 0, cz + R - 3.5))
    hook = torus(4.2, 1.5).rotate((90, 0, 0)).translate((0, 0, cz + R + 9.5))
    gold = union([cap, cap_ribs, hook, band])
    return {"oro": gold, "rosso": ball - gold}


# ---------------- 3. Albero a spirale ----------------
TW_SIGN = 1
def albero():
    Ht, trunk_h = 112, 9
    star = CrossSection([star2d(8, 40, 27, 0)]).offset(1.5, manifold3d.JoinType.Round)
    tw = 260
    tree = Manifold.extrude(star, Ht, n_divisions=140, twist_degrees=tw, scale_top=(0.03, 0.03)).translate((0, 0, trunk_h))
    trunk = Manifold.cylinder(trunk_h + 2, 11, 9, 64)
    # palline sulle punte, lungo la spirale
    balls_r, balls_g = [], []
    k = 0
    for zf in (0.12, 0.3, 0.48, 0.66, 0.82):
        for j in range(8):
            if (j + int(zf * 10)) % 2:
                continue
            z = Ht * zf
            sc = 1 - zf * 0.97
            ang = math.radians(TW_SIGN * tw * zf + j * 45)
            r = (40 + 1.5) * sc - 1.2
            b = ell((3.6, 3.6, 3.6), (r * math.cos(ang), r * math.sin(ang), trunk_h + z), 32)
            (balls_r if k % 2 else balls_g).append(b)
            k += 1
    topper = bevel_star(5, 15, 6.5, 4.5, both=True).rotate((90, 0, 0)).translate((0, 0, trunk_h + Ht + 9))
    stem = Manifold.cylinder(11, 2.2, 2.2, 24).translate((0, 0, trunk_h + Ht - 4))
    gold = union([topper, stem] + balls_g)
    red = union(balls_r) - gold
    green = tree - gold - red
    return {"verde": green, "oro": gold, "rosso": red, "marrone": trunk - green}


# ---------------- 4. Stella cometa ----------------
def stella():
    ro, ri = 38, 16
    plate = Manifold.extrude(CrossSection([star2d(5, ro, ri)]), 1.6)
    top = bevel_star(5, ro, ri, 8.5).translate((0, 0, 1.6))
    loop = ring(4.8, 2.4, 3).translate((0, ro + 2.6, 0))
    star = union([plate, top, loop])
    # stellina centrale argento a filo sulla punta
    inner = (bevel_star(5, 11, 4.6, 8.5 * 11 / ro + 0.05).translate((0, 0, 1.6 + 8.5 * (1 - 11 / ro)))) ^ star
    return {"oro": star - inner, "argento": inner}


# ---------------- 5. Pupazzo di neve ----------------
def pupazzo():
    b1, b2, b3 = ((23, 23, 22), (0, 0, 19)), ((17, 17, 16), (0, 0, 49)), ((13, 13, 13), (0, 0, 72))
    snow = [ell(*b1), ell(*b2), ell(*b3)]
    hat = union([Manifold.cylinder(2.4, 14, 14, 64).translate((0, 0, 81.5)),
                 Manifold.cylinder(15, 9, 9.6, 64).translate((0, 0, 83))])
    band = Manifold.cylinder(3.2, 9.9, 10, 64).translate((0, 0, 85)) - Manifold.cylinder(3.2, 9.0, 9.4, 64).translate((0, 0, 85))
    nose = Manifold.cylinder(12, 2.6, 0.5, 32).rotate((90, 0, 0)).translate((0, -11, 71))
    scarf = torus(13.2, 3.6).translate((0, 0, 61.5))
    tail = Manifold.batch_hull([ell((3.6, 2, 2.5), (7, -12, 60), 24), ell((4.5, 2, 2.5), (10, -15, 44), 24)])
    arms = []
    for s in (1, -1):
        a = Manifold.batch_hull([ell((1.8, 1.8, 1.8), (s * 13, 0, 52), 16), ell((1.4, 1.4, 1.4), (s * 30, -2, 64), 16)])
        f = Manifold.batch_hull([ell((1.3, 1.3, 1.3), (s * 25, -1.6, 60), 16), ell((1.1, 1.1, 1.1), (s * 27, -2, 69), 16)])
        arms += [a, f]
    feats_head = [("nero", C(1.9, -4.6, 75) + C(1.9, 4.6, 75)
                   + union_cs([C(1.0, 6.5 * math.cos(math.radians(a)), 70 + 6.5 * math.sin(math.radians(a))) for a in range(225, 316, 18)]))]
    feats_mid = [("nero", C(2.4, 0, 52) + C(2.4, 0, 44))]
    feats_low = [("nero", C(2.6, 0, 27))]
    dl = decals(*b3, feats_head) + decals(*b2, feats_mid) + decals(*b1, feats_low)
    return assemble({"arancio": [nose], "rosso": [scarf, tail, band], "nero": [hat], "marrone": arms, "bianco": snow}, dl)

def union_cs(cs_list):
    return CrossSection.batch_boolean(cs_list, ADD)


# ---------------- 6. Pikachu Babbo Natale (busto) ----------------
def pikachu_natale():
    head_r, head_c = (27, 23, 22), (0, 0, 20)
    yellow = [ell(head_r, head_c)]
    black = []
    for s in (1, -1):
        e, tip = ear((s * 18, 2, 30), (s * 0.95, 0.15, 0.55), 30, (7.5, 4.5, 7.5))
        tipball = ell((12, 12, 12), tip, 48)
        yellow.append(e - tipball); black.append(e ^ tipball)
    # cappello: cono piegato all'indietro, bordo di pelo e pompon
    hat_base_z = 36
    cone = Manifold.batch_hull([Manifold.cylinder(1, 17, 17, 64).translate((0, 2, hat_base_z)),
                                ell((3, 3, 3), (6, 18, hat_base_z + 26), 24)])
    hat = cone ^ Manifold.cube((BIG, BIG, BIG), True).translate((0, 0, hat_base_z + BIG / 2))
    fur = torus(16.5, 4.2).translate((0, 2, hat_base_z + 1.5))
    pom = ell((6, 6, 6), (6, 18, hat_base_z + 27), 48)
    feats = [("rosso", C(6.8, -18, 13) + C(6.8, 18, 13)),
             ("nero", E2(4.6, 5.6, -10.5, 23) + E2(4.6, 5.6, 10.5, 23) + E2(1.5, 1.0, 0, 17)
              + stroke(omega(0, 14.5, 2.8), 1.5)),
             ("bianco", C(1.7, -11.7, 25) + C(1.7, 9.3, 25))]
    dl = decals(head_r, head_c, feats)
    return assemble({"bianco": [fur, pom], "rosso": [hat], "nero": black, "giallo": yellow}, dl)


NATALE = {
    "fiocco_neve": ("Fiocco di neve", fiocco),
    "stella_cometa": ("Stella", stella),
    "pallina_spirale": ("Pallina a spirale", pallina),
    "albero_spirale": ("Albero a spirale", albero),
    "pupazzo_neve": ("Pupazzo di neve", pupazzo),
    "pikachu_natale": ("Pikachu Babbo Natale", pikachu_natale),
}

if __name__ == "__main__":
    OUT = sys.argv[1] if len(sys.argv) > 1 else "."
    only = sys.argv[2:] or list(NATALE)
    summary = json.load(open(os.path.join(OUT, "natale.json"))) if os.path.exists(os.path.join(OUT, "natale.json")) else {}
    for key in only:
        nome, fn = NATALE[key]
        raw = fn()
        parts = {}
        for k, m in raw.items():
            comps = [c for c in m.decompose() if c.volume() > 1.0]
            if comps:
                parts[k] = union(comps)
        d = os.path.join(OUT, key); os.makedirs(d, exist_ok=True)
        title = f"Natale - {nome}"
        for k, m in parts.items():
            open(f"{d}/{key}_{k}.stl", "wb").write(stl_bytes(m, title))
        unico = union([c for c in union(list(parts.values())).decompose() if c.volume() > 1.0])
        open(f"{d}/{key}_unico.stl", "wb").write(stl_bytes(unico, title))
        open(f"{d}/{key}.3mf", "wb").write(threemf_bytes(parts, title))
        prev = {}
        for k, m in parts.items():
            v, f = arrays(m)
            prev[k] = {"color": COL[k], "v": base64.b64encode(np.round(v * 100).astype("<i2").tobytes()).decode(),
                       "i": base64.b64encode(f.astype("<u4").tobytes()).decode()}
        json.dump(prev, open(f"{d}/preview.json", "w"))
        bb = unico.bounding_box()
        size = [round(bb[i + 3] - bb[i]) for i in range(3)]
        summary[key] = dict(nome=nome, ingombro=size, grammi=max(3, round(unico.volume() / 1000 * 1.24 * 0.5)),
                            colori=list(parts), pezzi_unico=len(unico.decompose()),
                            status={k: (m.status().name, len(m.decompose())) for k, m in parts.items()})
        print(key, summary[key])
    json.dump(summary, open(os.path.join(OUT, "natale.json"), "w"), indent=1)
