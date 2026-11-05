#!/usr/bin/env python3
"""
gridart.py — contribution-grid art generator for the mehedyk profile README.

Makes three kinds of animated SVG, all drawn as GitHub-style contribution cells
that "unlock" into place (same idea as jasineri/gitartwork, extended):

  assets/hero.svg          portrait (scan + unlock) + ASSALAMU ALAIKUM
  assets/div-<label>.svg   section dividers: label in cells + a travelling pulse
  assets/div-wassalam.svg  footer divider

Usage:
  pip install numpy opencv-python-headless
  python tools/gridart.py --photo path/to/portrait.jpg --out assets

The portrait is only read at build time; it is not stored in the repo.
Everything animates with plain CSS inside the SVG (no JS), so it works inside
a GitHub README <img>.  With prefers-reduced-motion, or in a viewer that
ignores CSS animation, the SVG shows its final "unlocked" state.
"""
import argparse, pathlib, random

PAL = ["#161b22", "#0e4429", "#006d32", "#26a641", "#39d353"]   # GitHub dark scale
MONO = 'ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace'

# 5x7 pixel font — only the letters this profile needs
FONT = {
 "A": ["01110","10001","10001","11111","10001","10001","10001"],
 "B": ["11110","10001","10001","11110","10001","10001","11110"],
 "C": ["01110","10001","10000","10000","10000","10001","01110"],
 "E": ["11111","10000","10000","11110","10000","10000","11111"],
 "I": ["11111","00100","00100","00100","00100","00100","11111"],
 "K": ["10001","10010","10100","11000","10100","10010","10001"],
 "L": ["10000","10000","10000","10000","10000","10000","11111"],
 "M": ["10001","11011","10101","10101","10001","10001","10001"],
 "N": ["10001","11001","10101","10011","10001","10001","10001"],
 "O": ["01110","10001","10001","10001","10001","10001","01110"],
 "R": ["11110","10001","10001","11110","10100","10010","10001"],
 "S": ["01111","10000","10000","01110","00001","00001","11110"],
 "T": ["11111","00100","00100","00100","00100","00100","00100"],
 "U": ["10001","10001","10001","10001","10001","10001","01110"],
 "V": ["10001","10001","10001","10001","10001","01010","00100"],
 "W": ["10001","10001","10001","10101","10101","11011","10001"],
 "Y": ["10001","10001","01010","00100","00100","00100","00100"],
}

def text_cells(word):
    """-> (list of (col,row), width_in_cols) for a word in the 5x7 font."""
    cells, x = [], 0
    for ch in word:
        for r, line in enumerate(FONT[ch]):
            for c, bit in enumerate(line):
                if bit == "1":
                    cells.append((x + c, r))
        x += 6
    return cells, x - 1

# --------------------------------------------------------------------------- face
def face_levels(photo):
    import cv2, numpy as np
    src = cv2.imread(photo)
    COLS, ROWS = 76, 100
    cx, top, ch = 292.0, 322.0, 345.0            # tuned crop around the face
    cw = ch * COLS / ROWS
    x0, y0 = int(cx - cw / 2), int(top)
    crop = src[y0:y0 + int(ch), x0:x0 + int(cw)]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.fastNlMeansDenoising(g, None, 4, 7, 21)
    f = ((g / 255.0) ** 0.58 * 255).astype("uint8")           # lift the shadows
    cl = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(6, 6)).apply(f)
    H, W = cl.shape
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - .56 * W) / (.40 * W)) ** 2 + ((yy - .47 * H) / (.50 * H)) ** 2)
    m = cl.astype(float) * np.clip(1.25 - d, 0, 1) ** 1.4      # soft vignette on the head
    small = cv2.resize(m, (COLS, ROWS), interpolation=cv2.INTER_AREA)
    th = [14, 42, 88, 132]
    lv = np.zeros(small.shape, dtype=int)
    for t in th:
        lv += (small >= t)
    # drop stray specks (lens-flare bits, hoodie crumbs)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((lv > 0).astype("uint8"), connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 60:
            lv[lab == i] = 0
    ys, xs = np.where(lv > 0)
    lv[ys.max() - 7:, :] = 0                       # trim collar clump
    ys, xs = np.where(lv > 0)
    pad = 3
    r0, r1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, ROWS)
    c0, c1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, COLS)
    return lv[r0:r1, c0:c1]


def face_levels_bright(photo):
    """Evenly lit photo on a light background: cut the person out (GrabCut), then map
    luminance inside the cut-out. A solid black rectangle (e.g. an eye bar) becomes empty cells."""
    import cv2, numpy as np
    src = cv2.imread(photo)
    H0, W0 = src.shape[:2]
    gm = np.zeros((H0, W0), np.uint8)
    cv2.grabCut(src, gm, (500, 215, 290, 420), np.zeros((1, 65)), np.zeros((1, 65)), 10, cv2.GC_INIT_WITH_RECT)
    fg = ((gm == 1) | (gm == 3)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    fg = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg = np.zeros_like(fg); cv2.drawContours(fg, cnts, -1, 1, -1)
    fg = fg.astype(np.float32)
    COLS, ROWS = 76, 100
    ch, cx, y0 = 370, 635, 205
    cw = ch * COLS / ROWS; x0 = int(cx - cw / 2)
    crop = src[y0:y0 + ch, x0:x0 + int(cw)]; m = fg[y0:y0 + ch, x0:x0 + int(cw)]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.fastNlMeansDenoising(g, None, 4, 7, 21)
    g = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8)).apply(g)
    sg = cv2.resize(g, (COLS, ROWS), interpolation=cv2.INTER_AREA).astype(float)
    sm = cv2.resize(m, (COLS, ROWS), interpolation=cv2.INTER_AREA)
    inside = sm > 0.5
    # the black bar: biggest very-dark blob near the eyes
    G = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    reg = np.zeros_like(G); reg[320:400, 500:650] = 1
    bm = cv2.morphologyEx(((G < 12) & (reg > 0)).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(bm, connectivity=8)
    bar = np.zeros((ROWS, COLS), bool)
    if n2 > 1:
        k = 1 + np.argmax(st2[1:, cv2.CC_STAT_AREA]); bx, by, bw, bh = st2[k, :4]
        bf = np.zeros(G.shape, np.float32); bf[by - 2:by + bh + 2, bx - 2:bx + bw + 2] = 1
        bar = inside & (cv2.resize(bf[y0:y0 + ch, x0:x0 + int(cw)], (COLS, ROWS), interpolation=cv2.INTER_AREA) > 0.45)
    valid = inside & ~bar
    t = np.percentile(sg[valid], [30, 50, 72])
    skin = valid & (sg >= t[1])
    ts = np.percentile(sg[skin], [35, 70])
    lv = np.zeros((ROWS, COLS), int)
    lv[inside] = 1 + sum((sg[inside] >= x).astype(int) for x in t[:2])
    lv[skin & (sg < ts[0])] = 2
    lv[skin & (sg >= ts[0])] = 3
    lv[skin & (sg >= ts[1])] = 4
    lv[bar] = 0
    lv[90:, :] = 0                                   # dissolve the shoulders out
    for r in range(76, 90):
        keep = (90 - r) / 14
        lv[r][np.random.default_rng(r).random(COLS) > keep] = 0
    ys, xs = np.where(lv > 0)
    pad = 3
    r0, r1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, ROWS)
    c0, c1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, COLS)
    return lv[r0:r1, c0:c1]

# --------------------------------------------------------------------------- hero
def build_hero(photo, out, mode='bright'):
    rnd = random.Random(7)
    L = (face_levels_bright if mode == 'bright' else face_levels)(photo)
    R, C = L.shape
    W = 888
    TP, TC = 10, 8                    # text lattice: pitch / cell
    FP, FC = TP / 3, 2.7              # face lattice = 3x finer, same grid lines
    fw, fh = C * FP, R * FP
    l1, w1 = text_cells("ASSALAMU")
    l2, w2 = text_cells("ALAIKUM")
    GAP = 44
    total = fw + GAP + (w1 + 3) * TP
    fx0 = round((W - total) / 2, 2)
    tx0 = round(fx0 + fw + GAP, 2)
    fy0 = 34
    th_px = 16 * TP - 2
    ty0 = round(fy0 + (fh - th_px) / 2 - 16, 2)
    H = round(fy0 + fh + 62)
    Tu = 3.7                           # moment the lock opens
    scan_t0, scan_dur = 0.5, 3.1

    css = f"""
:root{{--g0:{PAL[0]};--g1:{PAL[1]};--g2:{PAL[2]};--g3:{PAL[3]};--g4:{PAL[4]}}}
.l1{{fill:{PAL[1]}}}.l2{{fill:{PAL[2]}}}.l3{{fill:{PAL[3]}}}.l4{{fill:{PAL[4]}}}
.m{{font-family:{MONO};letter-spacing:2px;font-size:11px}}
.fr{{animation:fr .55s ease-out both}}
@keyframes fr{{0%{{opacity:0;transform:translateX(-7px)}}100%{{opacity:1;transform:none}}}}
.tc{{transform-box:fill-box;transform-origin:center;
     animation:pop .6s cubic-bezier(.2,.9,.3,1.25) both,glint 8s linear infinite}}
@keyframes pop{{0%{{opacity:0;transform:scale(.15) rotate(-90deg)}}
 65%{{opacity:1;transform:scale(1.14) rotate(7deg)}}100%{{opacity:1;transform:none}}}}
@keyframes glint{{0%{{opacity:1}}3%{{opacity:.3}}8%{{opacity:1}}100%{{opacity:1}}}}
.scan{{opacity:0;animation:scan {scan_dur}s linear {scan_t0}s both}}
@keyframes scan{{0%{{transform:translateY(0);opacity:0}}5%{{opacity:1}}
 95%{{opacity:1}}100%{{transform:translateY({fh:.1f}px);opacity:0}}}}
.brk{{fill:none;stroke:{PAL[4]};stroke-width:2;stroke-linecap:round;animation:tog .45s {Tu}s both}}
@keyframes tog{{from{{stroke:#484f58}}to{{stroke:{PAL[4]}}}}}
.lk{{stroke:{PAL[4]};fill:none;stroke-width:1.6;animation:lkc .4s {Tu}s both}}
.lkb{{fill:{PAL[4]};animation:lkf .4s {Tu}s both}}
@keyframes lkc{{from{{stroke:#484f58}}to{{stroke:{PAL[4]}}}}}
@keyframes lkf{{from{{fill:#484f58}}to{{fill:{PAL[4]}}}}}
.shk{{transform-box:fill-box;transform-origin:0% 100%;transform:translateY(-2px) rotate(-26deg);
      animation:shk .4s {Tu}s both}}
@keyframes shk{{from{{transform:none}}to{{transform:translateY(-2px) rotate(-26deg)}}}}
.cs{{fill:#484f58;opacity:0;animation:cs {Tu+0.25}s linear both}}
@keyframes cs{{0%{{opacity:1}}92%{{opacity:1}}100%{{opacity:0}}}}
.cu{{fill:{PAL[4]};animation:cu {Tu+0.25}s linear both}}
@keyframes cu{{0%,92%{{opacity:0}}100%{{opacity:1}}}}
.cur{{animation:bl 1.1s steps(1) infinite}}
@keyframes bl{{0%{{opacity:1}}50%{{opacity:0}}100%{{opacity:1}}}}
.cap{{fill:#8b949e;animation:capin .8s 5.2s both}}
@keyframes capin{{from{{opacity:0}}to{{opacity:1}}}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important}}}}
"""
    o = []
    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Assalamu Alaikum — contribution-grid portrait that unlocks into place">')
    o.append(f"<title>Assalamu Alaikum</title><style>{css}</style>")
    o.append("<defs>")
    o.append(f'<pattern id="gf" x="{fx0}" y="{fy0}" width="{FP:.4f}" height="{FP:.4f}" patternUnits="userSpaceOnUse">'
             f'<rect width="{FC}" height="{FC}" rx=".7" fill="{PAL[0]}" opacity=".7"/></pattern>')
    o.append(f'<pattern id="gt" x="{tx0}" y="{ty0}" width="{TP}" height="{TP}" patternUnits="userSpaceOnUse">'
             f'<rect width="{TC}" height="{TC}" rx="2" fill="{PAL[0]}"/></pattern>')
    o.append('<linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">'
             f'<stop offset="0" stop-color="{PAL[4]}" stop-opacity="0"/>'
             f'<stop offset="1" stop-color="{PAL[4]}" stop-opacity=".38"/></linearGradient>')
    o.append("</defs>")
    o.append(f'<rect x=".5" y=".5" width="{W-1}" height="{H-1}" rx="14" fill="#0d1117" stroke="#21262d"/>')

    # faint lattices
    o.append(f'<rect x="{fx0}" y="{fy0}" width="{fw:.2f}" height="{fh:.2f}" fill="url(#gf)"/>')
    gx1 = tx0 + (w1 + 3) * TP - 2
    up = int((ty0 - fy0) // TP)
    gy0 = ty0 - up * TP
    o.append(f'<rect x="{tx0}" y="{gy0}" width="{gx1 - tx0:.2f}" height="{fy0 + fh - gy0:.2f}" fill="url(#gt)"/>')

    # face rows (one animated group per row, synced with the scan line)
    for r in range(R):
        cells = [c for c in range(C) if L[r, c] > 0]
        if not cells:
            continue
        d = scan_t0 + (r + 0.5) / R * scan_dur
        o.append(f'<g class="fr" style="animation-delay:{d:.3f}s">')
        for c in cells:
            o.append(f'<rect class="l{L[r,c]}" x="{fx0 + c*FP:.2f}" y="{fy0 + r*FP:.2f}" '
                     f'width="{FC}" height="{FC}" rx=".7"/>')
        o.append("</g>")

    # viewfinder brackets
    p, a = 12, 22
    x0, y0, x1, y1 = fx0 - p, fy0 - p, fx0 + fw + p, fy0 + fh + p
    o.append('<path class="brk" d="'
             f'M{x0} {y0+a}V{y0}H{x0+a} M{x1-a} {y0}H{x1}V{y0+a} '
             f'M{x1} {y1-a}V{y1}H{x1-a} M{x0+a} {y1}H{x0}V{y1-a}"/>')
    # scan line
    o.append(f'<g class="scan"><rect x="{x0}" y="{fy0-16}" width="{x1-x0}" height="16" fill="url(#sg)"/>'
             f'<rect x="{x0}" y="{fy0}" width="{x1-x0}" height="1.5" fill="#7ee787"/></g>')
    # lock + status caption
    lx, ly = fx0, fy0 + fh + 30
    o.append(f'<g transform="translate({lx} {ly})">'
             '<g class="shk"><path class="lk" d="M2.5 6V4a2.5 2.5 0 0 1 5 0v2"/></g>'
             '<rect class="lkb" x="0" y="6" width="10" height="8" rx="1.8"/></g>')
    o.append(f'<text class="m cs" x="{lx+18}" y="{ly+11}">SCANNING</text>')
    o.append(f'<text class="m cu" x="{lx+18}" y="{ly+11}">UNLOCKED</text>')

    # text cells
    def place(cells, row_off, extra, add_cursor=False):
        for (c, r) in cells:
            lvl = rnd.choices([2, 3, 4], weights=[1, 3, 3])[0]
            dp = 3.0 + c * 0.028 + (r + row_off) * 0.012 + extra + rnd.random() * 0.06
            dg = dp + 1.1 + c * 0.05
            o.append(f'<rect class="tc l{lvl}" x="{tx0 + c*TP}" y="{ty0 + (r+row_off)*TP}" '
                     f'width="{TC}" height="{TC}" rx="2" style="animation-delay:{dp:.3f}s,{dg:.3f}s"/>')
    place(l1, 0, 0.0)
    place(l2, 9, 0.35)
    # blinking block cursor after line 2 (3x7 cells, same lattice as the letters)
    o.append('<g class="cur">')
    for cc in range(3):
        for rr in range(7):
            o.append(f'<rect class="l4" x="{tx0 + (w2 + 2 + cc)*TP}" y="{ty0 + (9+rr)*TP}" '
                     f'width="{TC}" height="{TC}" rx="2"/>')
    o.append("</g>")
    o.append(f'<text class="m cap" x="{tx0}" y="{ty0 + 16*TP + 24}">// peace be upon you</text>')
    o.append("</svg>")
    pathlib.Path(out).write_text("".join(o), encoding="utf-8")

# --------------------------------------------------------------------------- dividers
def build_divider(label, out, seed=1):
    rnd = random.Random(seed)
    W, P, CELL = 888, 6, 5
    cols = W // P
    cells, lw = text_cells(label.upper())
    start = (cols - lw) // 2
    H = 7 * P + 6
    y0 = 3
    css = f"""
.l2{{fill:{PAL[2]}}}.l3{{fill:{PAL[3]}}}.l4{{fill:{PAL[4]}}}
.f{{fill:{PAL[0]}}}.e{{fill:{PAL[0]};opacity:.45}}
.pl{{animation:pl 6s linear infinite}}
@keyframes pl{{0%{{fill:{PAL[0]}}}3%{{fill:{PAL[4]}}}8%{{fill:{PAL[3]}}}14%{{fill:{PAL[1]}}}20%,100%{{fill:{PAL[0]}}}}}
.t{{transform-box:fill-box;transform-origin:center;animation:pop .55s cubic-bezier(.2,.9,.3,1.25) both}}
@keyframes pop{{0%{{opacity:0;transform:scale(.15) rotate(-90deg)}}65%{{opacity:1;transform:scale(1.14) rotate(7deg)}}100%{{opacity:1;transform:none}}}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important}}}}
"""
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{label.lower()}">',
         f"<title>{label.lower()}</title><style>{css}</style>"]
    mid = 3
    for c in range(1, cols - 1):
        in_label = start - 3 <= c <= start + lw + 2
        if in_label:
            continue
        x = c * P + 1
        for r in (mid - 1, mid + 1):
            o.append(f'<rect class="e" x="{x}" y="{y0 + r*P}" width="{CELL}" height="{CELL}" rx="1.2"/>')
        d = c * 0.032
        o.append(f'<rect class="f pl" x="{x}" y="{y0 + mid*P}" width="{CELL}" height="{CELL}" rx="1.2" '
                 f'style="animation-delay:{d:.3f}s"/>')
    for (c, r) in cells:
        lvl = rnd.choices([2, 3, 4], weights=[1, 3, 3])[0]
        d = 0.15 + c * 0.03 + rnd.random() * 0.05
        o.append(f'<rect class="t l{lvl}" x="{(start + c)*P + 1}" y="{y0 + r*P}" width="{CELL}" height="{CELL}" '
                 f'rx="1.2" style="animation-delay:{d:.3f}s"/>')
    o.append("</svg>")
    pathlib.Path(out).write_text("".join(o), encoding="utf-8")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--out", default="assets")
    ap.add_argument("--mode", choices=["bright", "lit"], default="bright",
                    help="bright: evenly lit photo (cut-out); lit: dark low-key photo (shadow lift)")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    build_hero(a.photo, out / "hero.svg", a.mode)
    for i, name in enumerate(["snake", "stack", "work", "stats", "activity", "connect", "wassalam"]):
        build_divider(name, out / f"div-{name}.svg", seed=i + 3)
    print("done ->", out)
