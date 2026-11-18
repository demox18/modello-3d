# Colora la forma usando 4 viste (davanti, dietro, fianco destro, fianco sinistro) e cuoce tutto in un'unica texture.
# Uso: Blender -b -P colora_multivista.py -- <forma.glb> <cartella_viste> <cartella_output> [nome=...] [facce=120000] [atlante=2048]
# La cartella delle viste contiene fronte.png, retro.png, destra.png, sinistra.png con lo sfondo trasparente.
#
# Per ogni vista: 1) disegno la sagoma del modello da quel lato e cerco scala/spostamento che la fanno combaciare
# con la sagoma dell'immagine; 2) per ogni vertice controllo con un raggio se da quella vista è coperto;
# 3) il peso di ogni vista è quanto il vertice la guarda di fronte; 4) cuocio la miscela in una texture sola.

import bpy, bmesh, math, os, sys
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

args = sys.argv[sys.argv.index('--') + 1:]
FORMA, VISTE, OUT = args[0], args[1], args[2]
opt = dict(a.split('=', 1) for a in args[3:])
NOME = opt.get('nome', 'personaggio_4viste')
FACCE = int(opt.get('facce', 120000))
ATL = int(opt.get('atlante', 2048))
ALTEZZA = 1.75
R = 512  # risoluzione delle sagome per l'allineamento
os.makedirs(OUT, exist_ok=True)

# dir = dal personaggio verso la camera; right = cosa va a destra nell'immagine.
# Il personaggio guarda verso -Y; il suo fianco destro è verso -X.
VIEWS = {
    'fronte':   dict(dir=Vector((0, -1, 0)), right=Vector((1, 0, 0))),
    'retro':    dict(dir=Vector((0, 1, 0)),  right=Vector((-1, 0, 0))),
    'destra':   dict(dir=Vector((-1, 0, 0)), right=Vector((0, -1, 0))),
    'sinistra': dict(dir=Vector((1, 0, 0)),  right=Vector((0, 1, 0))),
}

# ---------------------------------------------------------------- forma
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=FORMA)
meshes = [o for o in scene.objects if o.type == 'MESH']
for o in scene.objects: o.select_set(o in meshes)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1: bpy.ops.object.join()
ob = bpy.context.view_layer.objects.active
ob.name = 'personaggio'
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
for o in list(scene.objects):
    if o.type == 'EMPTY': bpy.data.objects.remove(o)

co = np.array([v.co[:] for v in ob.data.vertices])
mn, mx = co.min(0), co.max(0)
s = ALTEZZA / (mx[2] - mn[2])
for v in ob.data.vertices:
    v.co = Vector(((v.co.x - (mn[0] + mx[0]) / 2) * s, (v.co.y - (mn[1] + mx[1]) / 2) * s, (v.co.z - mn[2]) * s))
ob.data.update()
nf = len(ob.data.polygons)
if nf > FACCE:
    m = ob.modifiers.new('decima', 'DECIMATE'); m.ratio = FACCE / nf
    bpy.ops.object.modifier_apply(modifier=m.name)
for p in ob.data.polygons: p.use_smooth = True
ob.data.materials.clear()

# ---------------------------------------------------------------- immagini delle viste (riga 0 = in alto)
def load_view(name):
    img = bpy.data.images.load(os.path.join(VISTE, name + '.png'))
    W, H = img.size
    a = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, 4)[::-1].copy()
    bpy.data.images.remove(img)
    return a

def erode(m, n):
    for _ in range(n):
        m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    return m

def bleed(arr, a, iters=60):
    """Allunga i colori fuori dalla sagoma, così ai bordi non finisce mai il bianco dello sfondo."""
    col = arr[..., :3].copy(); a = a.copy()
    for _ in range(iters):
        acc = np.zeros_like(col); cnt = np.zeros(a.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sa = np.roll(a, (dy, dx), (0, 1)); acc += np.roll(col, (dy, dx), (0, 1)) * sa[..., None]; cnt += sa
        new = (~a) & (cnt > 0)
        col[new] = acc[new] / cnt[new][:, None]; a |= new
    out = np.ones(arr.shape, np.float32); out[..., :3] = col
    return out

for name, V in VIEWS.items():
    px = load_view(name)
    V['alpha'] = px[..., 3] > 0.5
    V['inside'] = erode(V['alpha'], 4)              # zona sicura per prendere il colore
    clean = erode(V['alpha'], 3)                    # via l'alone chiaro lasciato dallo sfondo bianco
    V['pixels'] = bleed(px, clean)
    V['size'] = px.shape[1], px.shape[0]

# ---------------------------------------------------------------- allineamento delle sagome
scene.render.engine = 'BLENDER_WORKBENCH'
scene.render.film_transparent = True
scene.display.shading.light = 'FLAT'
scene.display.shading.color_type = 'SINGLE'
scene.render.resolution_x = scene.render.resolution_y = R
OS = ALTEZZA * 1.03  # circa la stessa grandezza della figura nelle immagini (alta ~98% del quadrato)
ZC = ALTEZZA / 2

def ortho_cam(name, V, center_s, center_z, scale):
    cd = bpy.data.cameras.new(name); cd.type = 'ORTHO'; cd.ortho_scale = scale
    c = bpy.data.objects.new(name, cd); scene.collection.objects.link(c)
    c.location = V['right'] * center_s + Vector((0, 0, center_z)) + V['dir'] * 10
    c.rotation_euler = (-V['dir']).to_track_quat('-Z', 'Y').to_euler()
    return c

def render_mask(V):
    cam = ortho_cam('sagoma', V, 0.0, ZC, OS)
    scene.camera = cam
    path = os.path.join(bpy.app.tempdir, 'sagoma.png')
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    m = np.array(img.pixels[:], dtype=np.float32).reshape(R, R, 4)[::-1, :, 3] > 0.5
    bpy.data.images.remove(img); bpy.data.objects.remove(cam)
    return m

GY, GX = np.mgrid[0:R, 0:R]

def iou_search(S, A, sy, ty, ks, txs, tys):
    """Per ogni pixel dell'immagine guarda il pixel corrispondente della sagoma del modello
    (mappa inversa: niente buchi) e tiene la combinazione con l'IoU più alto."""
    best = (-1, None)
    for k in ks:
        sx = sy * k
        for dty in tys:
            ys = np.round((GY - ty - dty) / sy).astype(int)
            okY = (ys >= 0) & (ys < R)
            for tx in txs:
                xs = np.round((GX - tx) / sx).astype(int)
                ok = okY & (xs >= 0) & (xs < R)
                T = np.zeros((R, R), bool)
                T[ok] = S[ys[ok], xs[ok]]
                iou = (T & A).sum() / max(1, (T | A).sum())
                if iou > best[0]:
                    best = (iou, (sx, tx, sy, ty + dty))
    return best

def fit_view(name, V, kmin=0.7, kmax=1.4):
    S = render_mask(V)
    W, H = V['size']
    A = V['alpha'].reshape(R, H // R, R, W // R).mean((1, 3)) > 0.5
    sr = np.nonzero(S.any(1))[0]; ar = np.nonzero(A.any(1))[0]
    sc = np.nonzero(S.any(0))[0]; ac = np.nonzero(A.any(0))[0]
    sy = (ar[-1] - ar[0]) / (sr[-1] - sr[0]); ty = ar[0] - sr[0] * sy
    tx0 = (ac[0] + ac[-1]) / 2 - (sc[0] + sc[-1]) / 2 * sy
    ks = np.linspace(0.75, 1.3, 23) if name in ('destra', 'sinistra') else np.linspace(0.9, 1.1, 21)
    iou, (sx, tx, sy2, ty2) = iou_search(S, A, sy, ty, ks, tx0 + np.arange(-16, 17, 2), [0])
    iou, (sx, tx, sy2, ty2) = iou_search(S, A, sy, ty, sx / sy + np.linspace(-0.012, 0.012, 7), tx + np.arange(-2, 3), [0])
    iou, (sx, tx, sy2, ty2) = iou_search(S, A, sy, ty, [sx / sy], tx + np.arange(-3, 4), np.arange(-4, 5))
    # pixel della sagoma -> coordinate nel mondo: col_render = (s/OS)*R + R/2 ; row_render = (ZC + OS/2 - z)/OS*R
    f = W / R
    V['ax'] = sx * R / OS * f;  V['bx'] = (sx * R / 2 + tx) * f
    V['ay'] = -sy2 * R / OS * f; V['by'] = (sy2 * (ZC + OS / 2) * R / OS + ty2) * f
    V['iou'] = float(iou)
    if name in ('destra', 'sinistra'):
        # di lato la profondità del modello non coincide col disegno: allineo riga per riga
        # il bordo davanti e quello dietro della sagoma del modello con quelli dell'immagine
        fi = int(f)
        Sf = np.repeat(np.repeat(S, fi, 0), fi, 1)
        rows = np.arange(H)
        rr = np.round(((rows / f) - ty2) / sy2 * f).astype(int)
        mL, mR, aL, aR = (np.full(H, np.nan) for _ in range(4))
        for ri in rows:
            if 0 <= rr[ri] < Sf.shape[0]:
                cols = np.nonzero(Sf[rr[ri]])[0]
                if len(cols):
                    mL[ri] = cols[0] * sx + tx * f; mR[ri] = (cols[-1] + 1) * sx + tx * f
            ca_ = np.nonzero(V['alpha'][ri])[0]
            if len(ca_):
                aL[ri] = ca_[0]; aR[ri] = ca_[-1] + 1
        def smooth(x, w=21):
            pad = np.pad(x, w // 2, mode='edge')
            return np.nanmedian(np.lib.stride_tricks.sliding_window_view(pad, w), axis=1)
        mL, mR, aL, aR = (smooth(x) for x in (mL, mR, aL, aR))
        ok = ~np.isnan(mL + mR + aL + aR) & ((mR - mL) > 8) & ((aR - aL) > 8)
        k = np.clip((aR - aL) / (mR - mL), kmin, kmax); o = aL - k * mL
        V['rowk'] = np.interp(rows, rows[ok], k[ok]); V['rowo'] = np.interp(rows, rows[ok], o[ok])
        print(f'  {name}: correzione per riga, scala da {np.nanmin(k[ok]):.2f} a {np.nanmax(k[ok]):.2f}')
    if opt.get('debug'):  # rosso = solo immagine, verde = solo modello, giallo = tutti e due
        ys = np.round((GY - ty2) / sy2).astype(int); xs = np.round((GX - tx) / sx).astype(int)
        ok = (ys >= 0) & (ys < R) & (xs >= 0) & (xs < R); T = np.zeros((R, R), bool); T[ok] = S[ys[ok], xs[ok]]
        ov = np.zeros((R, R, 4), np.float32); ov[..., 3] = 1
        ov[..., 0] = A; ov[..., 1] = T
        dbg = bpy.data.images.new('dbg', R, R); dbg.pixels[:] = ov[::-1].ravel()
        dbg.filepath_raw = os.path.join(OUT, f'allineamento_{name}.png'); dbg.file_format = 'PNG'; dbg.save()
        rawS = np.zeros((R, R, 4), np.float32); rawS[..., 3] = 1; rawS[..., 1] = S
        dbg.pixels[:] = rawS[::-1].ravel(); dbg.filepath_raw = os.path.join(OUT, f'sagoma_{name}.png'); dbg.save()
    print(f'ALLINEAMENTO {name}: IoU {iou:.3f} scala x/y {sx / sy2:.3f}')


def deform_depth():
    """Sposta la profondità (Y) dei vertici, altezza per altezza, così la sagoma di profilo del modello
    coincide con le viste di lato. X e Z non cambiano, quindi davanti e dietro restano allineati."""
    co = np.array([v.co[:] for v in ob.data.vertices])
    est = []
    for name in ('destra', 'sinistra'):
        V = VIEWS[name]
        W, H = V['size']
        right = np.array(V['right'][:])
        sv = co @ right
        r = np.clip(np.round(V['ay'] * co[:, 2] + V['by']).astype(int), 0, H - 1)
        c = V['ax'] * sv + V['bx']
        s2 = (V['rowk'][r] * c + V['rowo'][r] - V['bx']) / V['ax']
        est.append(s2 * right[1])  # right è ±Y: torno alla coordinata Y
    y_new = np.mean(est, 0)
    print(f'DEFORMAZIONE profondità: spostamento medio {np.abs(y_new - co[:, 1]).mean() * 100:.1f} cm, massimo {np.abs(y_new - co[:, 1]).max() * 100:.1f} cm')
    co[:, 1] = y_new
    ob.data.vertices.foreach_set('co', co.ravel())
    ob.data.update()


for name, V in VIEWS.items():
    fit_view(name, V)
if opt.get('deforma', '1') == '1':
    deform_depth()
    for name, V in VIEWS.items():
        V.pop('rowk', None); V.pop('rowo', None)
        fit_view(name, V, 0.85, 1.15)

if opt.get('debug') == 'stop':
    sys.exit(0)

def to_img(V, p):
    """Punto nel mondo -> (colonna, riga) in pixel dell'immagine della vista."""
    c = V['ax'] * p.dot(V['right']) + V['bx']; r = V['ay'] * p.z + V['by']
    if 'rowk' in V:
        ri = min(max(int(round(r)), 0), len(V['rowk']) - 1)
        c = V['rowk'][ri] * c + V['rowo'][ri]
    return c, r

# ---------------------------------------------------------------- pesi per vertice (con controllo di visibilità)
bm = bmesh.new(); bm.from_mesh(ob.data)
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
bm.normal_update()
bvh = BVHTree.FromBMesh(bm)
names = list(VIEWS)
Wt = np.zeros((len(bm.verts), 4), np.float32)
Wfree = np.zeros_like(Wt)
C = np.zeros((len(bm.verts), 4, 3), np.float32)  # colore che ogni vista dà al vertice
for i, v in enumerate(bm.verts):
    p, n = v.co, v.normal
    for j, name in enumerate(names):
        V = VIEWS[name]
        d = n.dot(V['dir'])
        if d <= 0:
            continue
        w = d ** 6
        Wfree[i, j] = w
        hit = bvh.ray_cast(p + V['dir'] * 0.002, V['dir'])
        if hit[0] is not None:
            continue  # coperto da un'altra parte del corpo
        c, r = to_img(V, p)
        ci, ri = int(round(c)), int(round(r))
        Wv, Hv = V['size']
        if not (0 <= ci < Wv and 0 <= ri < Hv) or not V['inside'][ri, ci]:
            w *= 0.15  # cade sul bordo della sagoma: colore poco affidabile
        C[i, j] = V['pixels'][min(max(ri, 0), Hv - 1), min(max(ci, 0), Wv - 1), :3]
        Wt[i, j] = w
# coerenza: davanti e dietro sono allineati meglio dei fianchi; se un fianco dà un colore
# molto diverso da quello che vedono davanti/dietro, quel fianco conta meno
wfb = Wt[:, 0] + Wt[:, 1]
has_ref = wfb > 0.25 ** 6
ref = (C[:, 0] * Wt[:, :1] + C[:, 1] * Wt[:, 1:2]) / np.maximum(wfb, 1e-9)[:, None]
for j in (2, 3):
    diff = np.linalg.norm(C[:, j] - ref, axis=1)
    fac = np.exp(-(diff / 0.2) ** 2)
    Wt[has_ref, j] *= fac[has_ref]
print('vertici dei fianchi corretti per coerenza:', int((has_ref & ((Wt[:, 2] + Wt[:, 3]) > 0)).sum()))
# pesi ammorbiditi tra vertici vicini: niente "patchwork" dove le faccette cambiano orientamento di continuo
ed = np.array([[e.verts[0].index, e.verts[1].index] for e in bm.edges])
deg = np.bincount(ed.ravel(), minlength=len(bm.verts)).astype(np.float32)[:, None]
for _ in range(int(opt.get('liscia', 6))):
    acc = np.zeros_like(Wt)
    np.add.at(acc, ed[:, 0], Wt[ed[:, 1]]); np.add.at(acc, ed[:, 1], Wt[ed[:, 0]])
    Wt = 0.4 * Wt + 0.6 * acc / np.maximum(deg, 1)
none = Wt.sum(1) == 0
Wt[none] = Wfree[none]
none = Wt.sum(1) == 0
Wt[none] = 1.0  # rivolto verso il basso (suole): media di tutte le viste
Wt /= Wt.sum(1, keepdims=True)
print('vertici senza vista libera:', int((Wfree.sum(1) > 0).sum() - (Wt.max(1) > 0).sum()))

# UV di proiezione per ogni vista
for name in names:
    V = VIEWS[name]
    uv = bm.loops.layers.uv.new('uv_' + name)
    Wv, Hv = V['size']
    for f in bm.faces:
        for l in f.loops:
            c, r = to_img(V, l.vert.co)
            l[uv].uv = ((c + 0.5) / Wv, 1 - (r + 0.5) / Hv)
bm.to_mesh(ob.data); bm.free()
ob.data.update()
ca = ob.data.color_attributes.new('pesi', 'FLOAT_COLOR', 'POINT')
ca.data.foreach_set('color', Wt.ravel())

# ---------------------------------------------------------------- materiale che miscela le 4 viste
mat = bpy.data.materials.new('miscela'); mat.use_nodes = True
nt = mat.node_tree; nt.nodes.remove(nt.nodes['Principled BSDF'])
attr = nt.nodes.new('ShaderNodeVertexColor'); attr.layer_name = 'pesi'
sep = nt.nodes.new('ShaderNodeSeparateColor')
nt.links.new(attr.outputs['Color'], sep.inputs['Color'])
weights = [sep.outputs[0], sep.outputs[1], sep.outputs[2], attr.outputs['Alpha']]
acc = None
for name, wsock in zip(names, weights):
    V = VIEWS[name]
    W, H = V['size']
    im = bpy.data.images.new('vista_' + name, W, H, alpha=True)
    im.pixels[:] = V['pixels'][::-1].ravel()
    uvn = nt.nodes.new('ShaderNodeUVMap'); uvn.uv_map = 'uv_' + name
    tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = im; tex.extension = 'EXTEND'
    nt.links.new(uvn.outputs['UV'], tex.inputs['Vector'])
    sc = nt.nodes.new('ShaderNodeVectorMath'); sc.operation = 'SCALE'
    nt.links.new(tex.outputs['Color'], sc.inputs[0]); nt.links.new(wsock, sc.inputs['Scale'])
    if acc is None:
        acc = sc.outputs['Vector']
    else:
        ad = nt.nodes.new('ShaderNodeVectorMath'); ad.operation = 'ADD'
        nt.links.new(acc, ad.inputs[0]); nt.links.new(sc.outputs['Vector'], ad.inputs[1])
        acc = ad.outputs['Vector']
em = nt.nodes.new('ShaderNodeEmission')
nt.links.new(acc, em.inputs['Color'])
nt.links.new(em.outputs['Emission'], nt.nodes['Material Output'].inputs['Surface'])
ob.data.materials.append(mat)

# ---------------------------------------------------------------- cottura in un'unica texture
atl_uv = ob.data.uv_layers.new(name='atlante')
ob.data.uv_layers.active = atl_uv
bpy.context.view_layer.objects.active = ob; ob.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.002)
bpy.ops.object.mode_set(mode='OBJECT')
atlas = bpy.data.images.new('texture_' + NOME, ATL, ATL, alpha=False)
target = nt.nodes.new('ShaderNodeTexImage'); target.image = atlas
nt.nodes.active = target
scene.render.engine = 'CYCLES'
scene.cycles.samples = 1
scene.cycles.device = 'CPU'
scene.render.bake.margin = 12
bpy.ops.object.bake(type='EMIT', margin=12, use_clear=True)
atlas.filepath_raw = os.path.join(OUT, 'texture_' + NOME + '.png'); atlas.file_format = 'PNG'
atlas.save()
atlas.pack()

# materiale finale: solo la texture cotta, "unlit" (il .glb lo esporta come KHR_materials_unlit)
final = bpy.data.materials.new(NOME); final.use_nodes = True
fn = final.node_tree; fn.nodes.remove(fn.nodes['Principled BSDF'])
ft = fn.nodes.new('ShaderNodeTexImage'); ft.image = atlas
fu = fn.nodes.new('ShaderNodeUVMap'); fu.uv_map = 'atlante'
fn.links.new(fu.outputs['UV'], ft.inputs['Vector'])
fn.links.new(ft.outputs['Color'], fn.nodes['Material Output'].inputs['Surface'])
ob.data.materials.clear(); ob.data.materials.append(final)
for name in names:
    ob.data.uv_layers.remove(ob.data.uv_layers['uv_' + name])
    bpy.data.images.remove(bpy.data.images['vista_' + name])
ob.data.color_attributes.remove(ob.data.color_attributes['pesi'])
bpy.data.materials.remove(mat)

# ---------------------------------------------------------------- render di controllo, alla stessa inquadratura delle immagini
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('mondo'); scene.world = world; world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)

def render(cam, fname, w, h):
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.filepath = os.path.join(OUT, fname)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam)

for name, V in VIEWS.items():
    W, H = V['size']
    s0 = (W / 2 - V['bx']) / V['ax']; z0 = (H / 2 - V['by']) / V['ay']
    render(ortho_cam('cmp', V, s0, z0, W / V['ax']), f'{NOME}_{name}.png', W, H)

def persp(name, pos, aim, lens):
    cd = bpy.data.cameras.new(name); cd.lens = lens
    c = bpy.data.objects.new(name, cd); scene.collection.objects.link(c)
    c.location = pos; c.rotation_euler = (Vector(aim) - Vector(pos)).to_track_quat('-Z', 'Y').to_euler()
    return c

render(persp('tq', (2.0, -3.3, 1.2), (0, 0, 0.86), 55), f'{NOME}_tre_quarti.png', 800, 1000)
render(persp('tq2', (-2.4, 2.9, 1.3), (0, 0, 0.86), 55), f'{NOME}_tre_quarti_retro.png', 800, 1000)
render(persp('testa', (0.25, -1.6, 1.5), (0, 0, 1.5), 70), f'{NOME}_testa.png', 700, 700)
render(persp('testa_lato', (-1.5, -0.6, 1.5), (0, 0, 1.5), 70), f'{NOME}_testa_lato.png', 700, 700)

# ---------------------------------------------------------------- salvataggi
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, NOME + '.blend'))
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, NOME + '.glb'), export_format='GLB',
                          use_selection=False, export_cameras=False, export_lights=False)
# .usdz per Quick Look: base nera + emissione con la texture (Quick Look non conosce l'unlit)
b = fn.nodes.new('ShaderNodeBsdfPrincipled')
b.inputs['Base Color'].default_value = (0, 0, 0, 1); b.inputs['Roughness'].default_value = 1.0
b.inputs['Specular IOR Level'].default_value = 0.0; b.inputs['Emission Strength'].default_value = 1.0
fn.links.new(ft.outputs['Color'], b.inputs['Emission Color'])
fn.links.new(b.outputs['BSDF'], fn.nodes['Material Output'].inputs['Surface'])
try:
    bpy.ops.wm.usd_export(filepath=os.path.join(OUT, NOME + '.usdz'), selected_objects_only=False, export_materials=True)
except Exception as e:
    print('usdz non esportato:', e)
print('FATTO', len(ob.data.polygons), 'facce', {k: round(v['iou'], 3) for k, v in VIEWS.items()})
