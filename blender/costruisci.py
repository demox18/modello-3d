# Costruisce il personaggio in Blender a partire dalle misure dell'immagine di riferimento.
# Uso: Blender -b -P costruisci.py -- <cartella_output>
# Unità di lavoro: 1 = 100 px dell'immagine (alta 1220 px). Alla fine tutto viene scalato a 1,75 m.
# Assi: X = destra dell'immagine, Z = su, -Y = verso chi guarda.

import bpy, bmesh, math, os, random, sys
from mathutils import Vector, Matrix, Euler

OUT = sys.argv[sys.argv.index('--') + 1] if '--' in sys.argv else os.path.dirname(__file__)
HERE = os.path.dirname(os.path.abspath(__file__))
V = Vector

# ---------------------------------------------------------------- scena pulita
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
root = bpy.data.objects.new('personaggio', None)
scene.collection.objects.link(root)

# ---------------------------------------------------------------- materiali
def hexcol(h):
    h = h.lstrip('#')
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c] + [1]

def material(name, color, rough=0.8, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = hexcol(color)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    m.diffuse_color = hexcol(color)
    return m

M = {
    'pelle': material('pelle', '#f0c39e', 0.6),
    'capelli': material('capelli', '#33241b', 0.55),
    'giallo': material('maglione_giallo', '#e6aa2a', 0.95),
    'nero': material('maglione_nero', '#1f1d1c', 0.95),
    'bianco': material('maglione_bianco', '#ece5d4', 0.95),
    'tshirt': material('tshirt', '#f6f3ec', 0.9),
    'jeans': material('pantaloni', '#2f3035', 0.9),
    'jeans_scuro': material('pantaloni_scuro', '#26272b', 0.9),
    'cucitura': material('cuciture', '#6a6b72', 0.9),
    'scarpa': material('scarpe', '#f3f1ec', 0.7),
    'suola': material('suola', '#d6d1c7', 0.8),
    'argento': material('argento', '#d9dde2', 0.25, 1.0),
}

# texture della faccia (proiezione frontale)
face_mat = bpy.data.materials.new('faccia')
face_mat.use_nodes = True
nt = face_mat.node_tree
img = bpy.data.images.load(os.path.join(HERE, 'faccia.png'))
img.pack()
tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = img
bsdf = nt.nodes['Principled BSDF']
bsdf.inputs['Roughness'].default_value = 0.6
nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
face_mat.diffuse_color = hexcol('#f0c39e')
M['faccia'] = face_mat

# ---------------------------------------------------------------- utilità
def new_obj(name, bm, mats, parent=None, smooth=False, loc=None, rot=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    for m in mats:
        me.materials.append(M[m] if isinstance(m, str) else m)
    for p in me.polygons:
        p.use_smooth = smooth
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    ob.parent = parent or root
    if loc: ob.location = loc
    if rot: ob.rotation_euler = rot
    return ob

def catmull(pts, t):
    """Punto su una spline Catmull-Rom che passa per pts, t in [0, len-1]."""
    n = len(pts)
    i = min(int(t), n - 2); u = t - i
    p0 = pts[max(i - 1, 0)]; p1 = pts[i]; p2 = pts[i + 1]; p3 = pts[min(i + 2, n - 1)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)

def lerp_list(vals, t):
    i = min(int(t), len(vals) - 2); u = t - i
    a, b = vals[i], vals[i + 1]
    return tuple(x + (y - x) * u for x, y in zip(a, b))

def tube(name, pts, radii, bands, seg=10, front=V((0, -1, 0)), cap=(True, True),
         rings_per_band=2, smooth=False, parent=None, twist=0.0, jitter=0.0, seed=1):
    """Tubo lungo una spline con sezione ellittica (rx, ry).
    bands: lista (materiale, lunghezza relativa), dall'inizio alla fine del tubo:
    gli anelli cadono esattamente sui confini, così le righe sono nette."""
    pts = [V(p) for p in pts]
    # tabella lunghezza d'arco
    N = 400
    ts = [i / N * (len(pts) - 1) for i in range(N + 1)]
    ps = [catmull(pts, t) for t in ts]
    acc = [0.0]
    for a, b in zip(ps, ps[1:]):
        acc.append(acc[-1] + (b - a).length)
    L = acc[-1]

    def at_len(s):
        s = max(0, min(L, s))
        for k in range(1, len(acc)):
            if acc[k] >= s:
                f = (s - acc[k - 1]) / max(acc[k] - acc[k - 1], 1e-9)
                return ts[k - 1] + (ts[k] - ts[k - 1]) * f
        return ts[-1]

    tot = sum(b[1] for b in bands)
    mats = []
    for m, _ in bands:
        if m not in mats: mats.append(m)
    # parametri degli anelli e materiale di ogni tratto
    params, seg_mat = [0.0], []
    s = 0.0
    for m, w in bands:
        n = max(1, round(rings_per_band * w / tot * len(bands)))
        for k in range(1, n + 1):
            params.append((s + w * k / n) / tot)
            seg_mat.append(mats.index(m))
        s += w

    bm = bmesh.new()
    rings = []
    jr = random.Random(seed)
    for i, f in enumerate(params):
        t = at_len(f * L)
        p = catmull(pts, t)
        tan = (catmull(pts, min(t + 0.01, len(pts) - 1)) - catmull(pts, max(t - 0.01, 0))).normalized()
        side = tan.cross(front)
        if side.length < 1e-4: side = tan.cross(V((0, 0, 1)))
        side.normalize()
        up = side.cross(tan).normalized()
        rx, ry = lerp_list(radii, t)
        ring = []
        for k in range(seg):
            a = 2 * math.pi * k / seg + twist
            j = 1 + (jr.uniform(-jitter, jitter) if 0 < i < len(params) - 1 else 0)
            ring.append(bm.verts.new(p + side * (math.cos(a) * rx * j) + up * (math.sin(a) * ry * j)))
        rings.append(ring)
    for i in range(len(rings) - 1):
        for k in range(seg):
            f = bm.faces.new((rings[i][k], rings[i][(k + 1) % seg], rings[i + 1][(k + 1) % seg], rings[i + 1][k]))
            f.material_index = seg_mat[i]
    if cap[0]:
        f = bm.faces.new(list(reversed(rings[0]))); f.material_index = seg_mat[0]
    if cap[1]:
        f = bm.faces.new(rings[-1]); f.material_index = seg_mat[-1]
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return new_obj(name, bm, mats, parent=parent, smooth=smooth)

def box(name, size, loc, mat, rot=(0, 0, 0), bevel=0.0, parent=None, smooth=False):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=V(size), verts=bm.verts)
    if bevel:
        bmesh.ops.bevel(bm, geom=bm.edges[:] + bm.verts[:], offset=bevel, segments=2, affect='EDGES')
    return new_obj(name, bm, [mat], parent=parent, loc=V(loc), rot=Euler(rot), smooth=smooth)

def add_subsurf(ob, levels=1):
    m = ob.modifiers.new('Subsurf', 'SUBSURF'); m.levels = levels; m.render_levels = levels
    return ob

# ---------------------------------------------------------------- PANTALONI CARGO
# gamba destra del personaggio = sinistra dell'immagine (X negativa)
jeans = [('jeans', 1)]
leg_r = tube('gamba_destra',
             [(-0.5, 0.0, 5.85), (-0.68, 0.0, 4.9), (-0.72, -0.04, 3.2), (-0.62, -0.04, 1.7), (-0.56, -0.06, 0.9)],
             [(0.88, 0.78), (0.8, 0.77), (0.74, 0.72), (0.74, 0.72), (0.84, 0.8)],
             jeans, seg=12, rings_per_band=12, jitter=0.07, seed=3, smooth=True)
add_subsurf(leg_r, 1)
leg_l = tube('gamba_sinistra',
             [(0.5, 0.0, 5.85), (0.78, 0.0, 4.9), (1.08, -0.2, 3.25), (1.12, 0.12, 1.75), (1.06, 0.2, 1.05)],
             [(0.88, 0.78), (0.8, 0.77), (0.74, 0.72), (0.74, 0.72), (0.84, 0.8)],
             jeans, seg=12, rings_per_band=12, jitter=0.07, seed=5, smooth=True)
add_subsurf(leg_l, 1)

def cargo_pocket(name, center, side, h, w, yaw):
    """Tasca laterale girata verso il davanti: corpo, patella e cuciture chiare."""
    g = bpy.data.objects.new(name, None); scene.collection.objects.link(g)
    g.parent = root; g.location = V(center); g.rotation_euler = (0, side * 0.04, yaw)
    box(name + '_corpo', (0.14, w, h), (0, 0, 0), 'jeans_scuro', bevel=0.03, parent=g)
    box(name + '_patella', (0.16, w + 0.05, h * 0.3), (side * 0.02, 0, h * 0.38), 'jeans', bevel=0.03, parent=g)
    x = side * 0.1
    for z0, hh in ((-h * 0.08, h * 0.74), (h * 0.38, h * 0.26)):
        for dz in (-hh / 2, hh / 2):
            box(name + '_cuc', (0.02, w - 0.1, 0.025), (x, 0, z0 + dz), 'cucitura', parent=g)
        for dy in (-(w - 0.1) / 2, (w - 0.1) / 2):
            box(name + '_cuc', (0.02, 0.025, hh), (x, dy, z0), 'cucitura', parent=g)

cargo_pocket('tasca_destra', (-1.2, -0.4, 3.55), -1, 1.7, 0.85, 0.75)
cargo_pocket('tasca_sinistra', (1.62, -0.5, 3.75), 1, 1.55, 0.8, -0.85)

# patta e passanti
for x in (0.12, 0.36):
    box('patta', (0.03, 0.03, 0.75), (x, -0.79, 5.15), 'cucitura')
box('patta', (0.27, 0.03, 0.03), (0.24, -0.79, 4.78), 'cucitura')
for x in (-0.95, -0.35, 0.45, 1.05):
    box('passante', (0.1, 0.06, 0.32), (x, -0.78 if abs(x) < 0.6 else -0.6, 5.62), 'jeans_scuro')
# taglio delle tasche davanti
for s in (-1, 1):
    tube('taglio_tasca', [(s * 0.75, -0.72, 5.58), (s * 1.0, -0.62, 5.1), (s * 1.25, -0.4, 4.85)],
         [(0.025, 0.025)] * 3, [('cucitura', 1)], seg=4)
# ---------------------------------------------------------------- SCARPE
def sneaker(name, loc, rot_z):
    """Sneaker bianca grossa: tomaia morbida lungo il piede (punta verso -Y) sopra una suola piatta."""
    g = bpy.data.objects.new(name, None); scene.collection.objects.link(g)
    g.parent = root; g.location = V(loc); g.rotation_euler = (0, 0, rot_z)
    up = V((0, 0, 1))
    tube(name + '_suola', [(0, -1.05, 0.14), (0, -0.6, 0.14), (0, 0.4, 0.14), (0, 0.85, 0.14)],
         [(0.36, 0.12), (0.54, 0.14), (0.55, 0.14), (0.45, 0.14)], [('suola', 1)], seg=12, front=up,
         rings_per_band=6, parent=g)
    upper = tube(name + '_tomaia', [(0, -1.0, 0.3), (0, -0.62, 0.38), (0, -0.05, 0.5), (0, 0.45, 0.58), (0, 0.78, 0.6)],
                 [(0.3, 0.16), (0.47, 0.28), (0.5, 0.42), (0.48, 0.5), (0.36, 0.42)], [('scarpa', 1)], seg=12,
                 front=up, rings_per_band=7, smooth=True, parent=g)
    add_subsurf(upper, 1)
    box(name + '_linguetta', (0.4, 0.45, 0.08), (0, -0.2, 0.9), 'scarpa', rot=(0.45, 0, 0), parent=g)
    for i in range(3):  # strisce sul fianco esterno e interno
        for sd in (-1, 1):
            box(name + '_riga', (0.03, 0.5, 0.06), (sd * 0.5, -0.05 + i * 0.13, 0.4 + i * 0.1),
                'suola', rot=(0.35, 0, 0), parent=g)
    for i in range(3):  # lacci
        box(name + '_laccio', (0.38, 0.05, 0.05), (0, -0.52 + i * 0.16, 0.78 + i * 0.07), 'suola', rot=(0.45, 0, 0), parent=g)

sneaker('scarpa_destra', (-0.55, -0.3, 0.02), 0.18)
sneaker('scarpa_sinistra', (1.05, 0.05, 0.12), -0.12)

# ---------------------------------------------------------------- MAGLIONE (busto)
# righe dall'orlo verso il collo
torso_bands = [('bianco', 0.38), ('giallo', 0.6), ('nero', 0.5), ('bianco', 0.38), ('giallo', 0.72),
               ('nero', 0.52), ('giallo', 0.42)]
torso = tube('maglione',
             [(0.0, 0.0, 5.62), (0.0, -0.02, 6.4), (0.0, -0.04, 7.4), (0.0, -0.02, 8.4), (0.0, 0.0, 9.0), (-0.05, 0.02, 9.32)],
             [(1.32, 0.8), (1.25, 0.76), (1.2, 0.74), (1.2, 0.72), (1.05, 0.64), (0.46, 0.36)],
             torso_bands, seg=14, front=V((0, -1, 0)), cap=(True, True), rings_per_band=1, smooth=True, jitter=0.03, seed=9)
add_subsurf(torso, 1)
# colletto della t-shirt e collo
tube('colletto', [(-0.12, -0.02, 9.2), (-0.12, -0.02, 9.38)], [(0.38, 0.32), (0.34, 0.29)], [('tshirt', 1)], seg=12)
tube('collo', [(-0.12, 0.02, 9.2), (-0.14, 0.04, 9.6), (-0.15, 0.06, 9.9)], [(0.27, 0.25), (0.25, 0.24), (0.27, 0.25)],
     [('pelle', 1)], seg=10, smooth=True)

# ---------------------------------------------------------------- BRACCIA
# braccio destro (sinistra dell'immagine): gomito in basso, mano davanti al petto col segno della pace
arm_r_pts = [(-0.62, 0.0, 8.75), (-1.25, 0.02, 8.4), (-1.55, -0.1, 7.5), (-1.58, -0.35, 6.75), (-1.25, -0.8, 7.35), (-0.88, -1.0, 7.95)]
arm_r = tube('manica_destra', arm_r_pts,
             [(0.48, 0.48), (0.52, 0.5), (0.5, 0.48), (0.47, 0.46), (0.42, 0.4), (0.38, 0.37)],
             [('giallo', 0.45), ('nero', 0.55), ('giallo', 0.55), ('bianco', 0.45), ('nero', 0.5),
              ('bianco', 0.45), ('giallo', 0.5)],
             seg=12, front=V((0, 0, 1)), cap=(True, False), rings_per_band=2, smooth=True)
add_subsurf(arm_r, 1)
# braccio sinistro (destra dell'immagine): lungo il fianco, mano in tasca
arm_l = tube('manica_sinistra',
             [(0.55, 0.0, 8.75), (1.15, 0.02, 8.4), (1.42, 0.05, 7.4), (1.48, 0.0, 6.5), (1.32, -0.12, 5.75), (1.12, -0.2, 5.3)],
             [(0.48, 0.48), (0.52, 0.5), (0.5, 0.48), (0.48, 0.47), (0.44, 0.43), (0.38, 0.37)],
             [('giallo', 0.6), ('nero', 0.5), ('giallo', 0.75), ('bianco', 0.4), ('nero', 0.55), ('giallo', 0.55)],
             seg=12, front=V((0, -1, 0)), cap=(True, True), rings_per_band=2, smooth=True)
add_subsurf(arm_l, 1)

# ---------------------------------------------------------------- MANO (segno della pace)
hand = bpy.data.objects.new('mano_pace', None); scene.collection.objects.link(hand)
hand.parent = root
hand.location = V((-0.78, -1.02, 7.85))
hand.scale = (1.45,) * 3
hand.rotation_euler = Euler((0.12, 0.25, -0.3))
palm = box('palmo', (0.38, 0.18, 0.4), (0, 0, 0.26), 'pelle', bevel=0.07, parent=hand, smooth=True)
add_subsurf(palm, 1)
# indice e medio aperti a V
for s, ang in ((-1, 0.2), (1, -0.18)):
    base = V((s * 0.09, -0.01, 0.44))
    d = V((math.sin(-ang), 0, math.cos(ang)))
    tube('dito', [base, base + d * 0.2, base + d * 0.4], [(0.055, 0.05), (0.05, 0.046), (0.044, 0.04)],
         [('pelle', 1)], seg=6, smooth=True, parent=hand)
# anulare e mignolo piegati
for x in (0.0, 0.12):
    tube('dito_piegato', [(x - 0.06, -0.02, 0.44), (x - 0.06, -0.14, 0.5), (x - 0.06, -0.2, 0.4)],
         [(0.055, 0.05)] * 3, [('pelle', 1)], seg=6, smooth=True, parent=hand)
# pollice che tiene giù le dita
tube('pollice', [(0.2, -0.04, 0.12), (0.14, -0.16, 0.28), (0.0, -0.2, 0.36)],
     [(0.075, 0.07), (0.07, 0.065), (0.06, 0.055)], [('pelle', 1)], seg=6, smooth=True, parent=hand)
# polsino della manica attorno al polso
tube('polsino', [(0, 0, -0.08), (0, 0, 0.1)], [(0.3, 0.27), (0.27, 0.24)], [('giallo', 1)], seg=10, parent=hand)

# ---------------------------------------------------------------- COLLANA
chain_pts = [(-0.48, 0.2, 9.25), (-0.45, -0.32, 9.05), (-0.25, -0.68, 8.6), (0.05, -0.8, 8.38),
             (0.3, -0.72, 8.62), (0.42, -0.36, 9.08), (0.3, 0.2, 9.28)]
cu = bpy.data.curves.new('catenina', 'CURVE'); cu.dimensions = '3D'
sp = cu.splines.new('POLY'); sp.points.add(len(chain_pts) - 1)
for pt, c in zip(sp.points, chain_pts): pt.co = (*c, 1)
cu.splines[0].type = 'NURBS'; cu.splines[0].order_u = 3; cu.splines[0].use_endpoint_u = True
cu.bevel_depth = 0.025; cu.bevel_resolution = 1
chain = bpy.data.objects.new('catenina', cu); scene.collection.objects.link(chain); chain.parent = root
chain.data.materials.append(M['argento'])
bm = bmesh.new()
bmesh.ops.create_icosphere(bm, subdivisions=0, radius=1)
bm2 = bmesh.new()
for co in [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]:
    bm2.verts.new(co)
bm.free()
bmesh.ops.convex_hull(bm2, input=bm2.verts)
bmesh.ops.scale(bm2, vec=V((0.12, 0.06, 0.22)), verts=bm2.verts)
new_obj('ciondolo', bm2, ['argento'], loc=V((0.05, -0.86, 8.12)), rot=Euler((0.2, 0, 0)))

# ---------------------------------------------------------------- TESTA
HEAD_C = V((-0.18, -0.02, 10.25))
head = bpy.data.objects.new('testa', None); scene.collection.objects.link(head)
head.parent = root; head.location = HEAD_C; head.rotation_euler = Euler((0.08, 0.06, -0.12))

bm = bmesh.new()
bmesh.ops.create_uvsphere(bm, u_segments=28, v_segments=20, radius=1.0)
for v in bm.verts:
    x, y, z = v.co
    x *= 0.76; y *= 0.76; z *= 0.84
    if z < 0:  # mento a punta stile anime
        t = min(1.0, -z / 0.88)
        x *= 1 - 0.48 * t ** 1.7
        y *= 1 - 0.3 * t ** 1.4
        if y < 0: y -= 0.06 * t
        z -= 0.04 * t ** 2
    if y < 0 and z > -0.3 and z < 0.3:  # guance leggermente piene
        x *= 1.02
    v.co = V((x, y, z))
uv = bm.loops.layers.uv.new('UVMap')
for f in bm.faces:
    for l in f.loops:
        x, y, z = l.vert.co
        if y > 0.12:
            l[uv].uv = (0.02, 0.02)
        else:
            l[uv].uv = ((x * 0.92 + 0.8) / 1.6, (z * 1.05 + 1.1) / 1.6)
skull = new_obj('cranio', bm, ['faccia'], parent=head, smooth=True)
for s in (-1, 1):
    bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=1)
    bmesh.ops.scale(bm, vec=V((0.06, 0.12, 0.17)), verts=bm.verts)
    new_obj('orecchio', bm, ['pelle'], parent=head, loc=V((s * 0.74, 0.06, -0.1)), smooth=True)

# ---------------------------------------------------------------- CAPELLI
hair_bm = bmesh.new()
HC = V((0.0, 0.12, 0.1))  # centro da cui "escono" le ciocche

def strand(root_p, ctrl, tip, w, th=None, rings=7):
    """Ciocca larga a foglia: sezione esagonale schiacciata lungo una Bézier quadratica, punta finale."""
    root_p, ctrl, tip = V(root_p), V(ctrl), V(tip)
    th = th if th is not None else w * 0.42
    ring_verts = []
    for i in range(rings):
        t = i / (rings - 1)
        p = (1 - t) ** 2 * root_p + 2 * (1 - t) * t * ctrl + t * t * tip
        tan = (2 * (1 - t) * (ctrl - root_p) + 2 * t * (tip - ctrl)).normalized()
        out = (p - HC).normalized()
        side = tan.cross(out)
        if side.length < 1e-4: side = tan.cross(V((0, 0, 1)))
        side.normalize()
        nrm = side.cross(tan).normalized()
        if i == rings - 1:
            ring_verts.append([hair_bm.verts.new(p)])
            continue
        k = (1 - t ** 1.4) * (0.85 + 0.3 * math.sin(math.pi * min(1, t * 1.6)))  # si allarga e poi si stringe
        ring = []
        for j in range(6):
            a = 2 * math.pi * j / 6
            ca, sa = math.cos(a), math.sin(a)
            ring.append(hair_bm.verts.new(p + side * (ca * w * k) + nrm * (sa * th * k * (1 if sa > 0 else 0.5))))
        ring_verts.append(ring)
    for i in range(rings - 2):
        a, b = ring_verts[i], ring_verts[i + 1]
        for k in range(6):
            hair_bm.faces.new((a[k], a[(k + 1) % 6], b[(k + 1) % 6], b[k]))
    a, tipv = ring_verts[-2], ring_verts[-1][0]
    for k in range(6):
        hair_bm.faces.new((a[k], a[(k + 1) % 6], tipv))
    hair_bm.faces.new(list(reversed(ring_verts[0])))

# calotta: volume di base, più grande del cranio e aperta sulla faccia
cap = bmesh.new()
bmesh.ops.create_uvsphere(cap, u_segments=24, v_segments=16, radius=1.0)
for v in cap.verts:
    v.co = V((v.co.x * 0.86, v.co.y * 0.9 + 0.08, v.co.z * 0.9 + 0.12))
bmesh.ops.delete(cap, geom=[f for f in cap.faces
                            if (f.calc_center_median().y < -0.3 and f.calc_center_median().z < 0.45)
                            or f.calc_center_median().z < -0.6], context='FACES')
cap_ob = new_obj('calotta', cap, ['capelli'], parent=head, smooth=True)

# ciocche disposte a file attorno alla testa: in alto puntano su, più in basso cadono
# (tutto spostato un po' verso sinistra dell'immagine, come nel riferimento)
rnd = random.Random(11)
LEAN = V((-0.18, 0, 0))
rows = [  # elevazione, quante, lunghezza, gravità, larghezza
    (1.3, 5, 0.6, 0.1, 0.42),
    (0.85, 10, 0.8, 0.45, 0.46),
    (0.4, 13, 0.85, 0.8, 0.46),
    (0.0, 13, 0.8, 1.1, 0.44),
    (-0.35, 10, 0.7, 1.4, 0.4),
]
for el, n, ln, grav, w in rows:
    for i in range(n):
        az = 2 * math.pi * (i + rnd.uniform(-0.25, 0.25) + (0.5 if el < 1 and int(el * 10) % 2 else 0)) / n
        d = V((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el))).normalized()
        if d.y < -0.42 and el < 1.0:  # davanti c'è la frangia
            continue
        r = HC + V((d.x * 0.78, d.y * 0.82, d.z * 0.82))
        L = ln * rnd.uniform(0.85, 1.15)
        tipdir = (d + LEAN + V((0, 0, -grav))).normalized()
        ctrl = r + d * L * 0.4 + tipdir * L * 0.15
        tip = r + d * L * 0.4 + tipdir * L * 0.7
        strand(r, ctrl, tip, w * rnd.uniform(0.85, 1.1))

# frangia: ciocche larghe che cadono sulla fronte e sulla faccia
bangs = [
    ((-0.55, -0.45, 0.6), (-0.78, -0.86, 0.25), (-0.74, -0.66, -0.45), 0.24),
    ((-0.32, -0.55, 0.68), (-0.42, -0.98, 0.3), (-0.36, -0.86, -0.22), 0.24),
    ((-0.08, -0.58, 0.7), (-0.14, -1.0, 0.32), (-0.08, -0.92, -0.42), 0.2),  # ciocca tra gli occhi
    ((0.16, -0.58, 0.68), (0.22, -1.0, 0.28), (0.3, -0.88, -0.14), 0.22),
    ((0.4, -0.52, 0.64), (0.52, -0.94, 0.22), (0.6, -0.72, -0.34), 0.22),
    ((0.62, -0.38, 0.52), (0.8, -0.72, 0.1), (0.74, -0.46, -0.52), 0.2),
    ((-0.25, -0.5, 0.8), (-0.6, -0.9, 0.62), (-0.98, -0.62, 0.3), 0.24),
    ((0.25, -0.5, 0.8), (0.58, -0.9, 0.62), (0.92, -0.58, 0.36), 0.22),
    ((0.0, -0.45, 0.9), (0.0, -0.95, 0.85), (-0.1, -1.0, 0.45), 0.26),
]
for sb in bangs:
    strand(*sb)

hair_ob = new_obj('capelli', hair_bm, ['capelli'], parent=head, smooth=True)
add_subsurf(hair_ob, 1)

# ---------------------------------------------------------------- rifinitura
for ob in scene.objects:
    if ob.type == 'MESH':
        ob.data.update()

# scala finale a 1,75 m (12,2 unità = altezza dell'immagine)
root.scale = (1.75 / 12.2,) * 3
root.rotation_euler = (0, 0, 0)
bpy.context.view_layer.update()

# ---------------------------------------------------------------- luci e camere per i render di controllo
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.film_transparent = True
scene.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('mondo'); scene.world = world
world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.85, 0.85, 0.85, 1)
world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.7

def light(name, kind, energy, rot, loc=(0, 0, 0)):
    l = bpy.data.lights.new(name, kind); l.energy = energy
    o = bpy.data.objects.new(name, l); scene.collection.objects.link(o)
    o.rotation_euler = Euler([math.radians(a) for a in rot]); o.location = loc
    return o

key = light('sole', 'SUN', 3.6, (48, 0, -35))  # da in alto a sinistra, come nel riferimento
key.data.angle = math.radians(8)
light('riempimento', 'SUN', 0.8, (70, 0, 60)).data.use_shadow = False
light('controluce', 'SUN', 1.6, (55, 0, 170))
for attr, val in (('use_gtao', True), ('gtao_distance', 0.4), ('use_shadows', True), ('shadow_ray_count', 2)):
    try: setattr(scene.eevee, attr, val)
    except Exception: pass

S = 1.75 / 12.2
cam_d = bpy.data.cameras.new('camera_fronte'); cam_d.type = 'ORTHO'; cam_d.ortho_scale = 12.2 * S
cam = bpy.data.objects.new('camera_fronte', cam_d); scene.collection.objects.link(cam)
cam.location = (0, -10, 5.95 * S); cam.rotation_euler = (math.radians(90), 0, 0)
cam3_d = bpy.data.cameras.new('camera_tre_quarti'); cam3_d.lens = 60
cam3 = bpy.data.objects.new('camera_tre_quarti', cam3_d); scene.collection.objects.link(cam3)
cam3.location = (1.75, -2.95, 1.2)
direction = V((0, 0, 0.86)) - cam3.location
cam3.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
camh_d = bpy.data.cameras.new('camera_testa'); camh_d.lens = 85
camh = bpy.data.objects.new('camera_testa', camh_d); scene.collection.objects.link(camh)
camh.location = (0.3, -2.4, 1.48)
direction = V((-0.02, 0, 1.45)) - camh.location
camh.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

def render(cam_ob, path, w, h):
    scene.camera = cam_ob
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)

os.makedirs(OUT, exist_ok=True)
render(cam, os.path.join(OUT, 'render_fronte.png'), 379, 1220)
render(cam3, os.path.join(OUT, 'render_tre_quarti.png'), 800, 1000)
render(camh, os.path.join(OUT, 'render_testa.png'), 700, 700)
if '--dettagli' in sys.argv:
    for nome, pos, mira in (('piedi', (0.9, -1.6, 0.35), (0.05, 0, 0.08)), ('retro', (-1.2, 3.2, 1.3), (0, 0, 0.95)),
                            ('mano', (-0.3, -1.6, 1.25), (-0.1, -0.1, 1.15))):
        cd = bpy.data.cameras.new(nome); cd.lens = 50
        co = bpy.data.objects.new(nome, cd); scene.collection.objects.link(co)
        co.location = pos; co.rotation_euler = (V(mira) - V(pos)).to_track_quat('-Z', 'Y').to_euler()
        render(co, os.path.join(OUT, 'dettaglio_' + nome + '.png'), 600, 600)
        bpy.data.objects.remove(co)
scene.camera = cam3
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, 'personaggio.blend'))
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, 'personaggio.glb'), export_format='GLB',
                          export_apply=True, use_selection=False, export_cameras=False, export_lights=False)
print('FATTO')
