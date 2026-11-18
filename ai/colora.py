# Colora la forma generata da Hunyuan3D proiettando l'immagine di riferimento dal davanti.
# Uso: Blender -b -P colora.py -- <forma.glb> <riferimento.png> <cartella_output> [chiave=valore ...]
#   profilo=posa (prima immagine, mano davanti al petto) | apose (immagine a braccia aperte)
#   nome=personaggio_ai   rot=0 (rotazione in gradi attorno a Z)
# Il davanti usa l'immagine così com'è; il retro usa una copia ritoccata (capelli al posto della faccia,
# righe del maglione al posto della mano e della collana), altrimenti la faccia comparirebbe sulla nuca.

import bpy, bmesh, math, os, sys
import numpy as np
from mathutils import Vector, Euler

args = sys.argv[sys.argv.index('--') + 1:]
FORMA, RIF, OUT = args[0], args[1], args[2]
opt = dict(a.split('=', 1) for a in args[3:])
ROT_Z = math.radians(float(opt.get('rot', 0)))
NOME = opt.get('nome', 'personaggio_ai')
# zone dell'immagine (in pixel dall'alto) da ritoccare per il retro
PROFILI = {
    'posa': dict(mento=265, capelli=(20, 110), mano=(285, 445, 60, 240), collo=(268, 385, 130, 245)),
    'apose': dict(mento=203, capelli=(20, 110), mano=None, collo=(282, 314, 478, 528), scollo=(203, 283, 452, 550)),
}
P = PROFILI[opt.get('profilo', 'posa')]
ALTEZZA = 1.75
os.makedirs(OUT, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=FORMA)
meshes = [o for o in scene.objects if o.type == 'MESH']
for o in scene.objects: o.select_set(o in meshes)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1: bpy.ops.object.join()
ob = bpy.context.view_layer.objects.active
ob.name = 'personaggio'
ob.rotation_euler.z += ROT_Z
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
for o in list(scene.objects):
    if o.type == 'EMPTY': bpy.data.objects.remove(o)

# --- normalizza: alto 1,75 m, piedi a terra, centrato
co = np.array([v.co[:] for v in ob.data.vertices])
mn, mx = co.min(0), co.max(0)
s = ALTEZZA / (mx[2] - mn[2])
for v in ob.data.vertices:
    v.co = Vector(((v.co.x - (mn[0] + mx[0]) / 2) * s, (v.co.y - (mn[1] + mx[1]) / 2) * s, (v.co.z - mn[2]) * s))
ob.data.update()

# --- meno poligoni per un .glb leggero
nf = len(ob.data.polygons)
if nf > 90000:
    m = ob.modifiers.new('decima', 'DECIMATE'); m.ratio = 90000 / nf
    bpy.ops.object.modifier_apply(modifier=m.name)
for p in ob.data.polygons: p.use_smooth = True

# --- immagini davanti / dietro
img = bpy.data.images.load(RIF)
W, H = img.size
px = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, 4)  # riga 0 = in basso
alpha = px[..., 3] > 0.5
rows = np.where(alpha.any(1))[0]; cols = np.where(alpha.any(0))[0]
r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()

def top(y):  # coordinata "dall'alto" (come nell'immagine) -> riga numpy
    return H - 1 - y

back = px.copy()
# testa: tutto ciò che sta sopra il mento diventa capelli (prende il colore medio dei capelli in cima)
hair_rows = slice(top(P['capelli'][1]), top(P['capelli'][0]))
hair = px[hair_rows][alpha[hair_rows]][:, :3]
hair_col = np.median(hair[hair.sum(1) < 1.0], axis=0) if len(hair) else np.array([0.2, 0.14, 0.11])
rng = np.random.default_rng(3)
head = np.zeros_like(alpha); head[top(P['mento']):, :] = True
mask = head & alpha
back[mask, :3] = hair_col * (0.85 + 0.3 * rng.random((mask.sum(), 1)))
# mano davanti al petto: solo i pixel color pelle diventano il maglione della stessa riga, preso più a destra
c = px[..., :3]
skin = (c[..., 0] > 0.55) & (c[..., 0] > c[..., 1] + 0.08) & (c[..., 1] > c[..., 2] + 0.04) & alpha
zone = np.zeros_like(alpha)
if P['mano']:
    y0, y1, x0, x1 = P['mano']; zone[top(y1):top(y0), x0:x1] = True
skin &= zone
for _ in range(3):  # allarga un po' la maschera per prendere anche i contorni della mano
    skin |= np.roll(skin, 1, 0) | np.roll(skin, -1, 0) | np.roll(skin, 1, 1) | np.roll(skin, -1, 1)
skin &= zone & alpha
# collana e colletto della t-shirt (grigi/bianchi neutri sul petto): sulla schiena non ci sono
mx_, mn_ = c.max(-1), c.min(-1)
neck = np.zeros_like(alpha); y0, y1, x0, x1 = P['collo']; neck[top(y1):top(y0), x0:x1] = True
necklace = neck & alpha & (mx_ - mn_ < 0.1) & (mx_ > 0.3)
hole = skin | necklace
if P.get('scollo'):  # davanti si vedono colletto e catenina: sulla schiena è maglione giallo, preso dai lati
    y0, y1, x0, x1 = P['scollo']
    r0_, r1_ = top(y1), top(y0)
    side = np.concatenate([px[r0_:r1_, x0 - 25:x0 - 5, :3].reshape(-1, 3), px[r0_:r1_, x1 + 5:x1 + 25, :3].reshape(-1, 3)])
    yellow = np.median(side[(side[:, 0] > side[:, 2] + 0.2)], axis=0)
    sc = np.zeros_like(alpha); sc[r0_:r1_, x0:x1] = True; sc &= alpha
    back[sc, :3] = yellow * (0.96 + 0.08 * rng.random((sc.sum(), 1)))
    hole &= ~sc
for _ in range(2):
    hole |= np.roll(hole, 1, 0) | np.roll(hole, -1, 0) | np.roll(hole, 1, 1) | np.roll(hole, -1, 1)
hole &= alpha & ~head

def erode(a, n):
    for _ in range(n):
        a = a & np.roll(a, 1, 0) & np.roll(a, -1, 0) & np.roll(a, 1, 1) & np.roll(a, -1, 1)
    return a

# alone chiaro azzurro/verde attorno alla figura nell'immagine originale: lo tratto come vuoto
lum = c.mean(-1)
halo = alpha & ~erode(alpha, 10) & (lum > 0.28) & ((c[..., 2] > c[..., 0] + 0.03) | (c[..., 1] > c[..., 0] + 0.02))
halo |= alpha & (lum > 0.25) & (c[..., 1] > c[..., 0] + 0.03)  # il verde non c'è da nessuna parte sul personaggio
# bordo dei capelli schiarito dallo sfondo bianco: i capelli sono scuri, quindi lì il chiaro vicino al bordo è alone
hair_zone = np.zeros_like(alpha); hair_zone[top(P['mento']):, :] = True
halo |= hair_zone & alpha & ~erode(alpha, 8) & (lum > 0.42)
clean = alpha & ~halo

def bleed(arr, a, erode=3, iters=140):
    """Toglie l'alone del bordo e allunga i colori fuori dalla sagoma (niente nero sui fianchi)."""
    a = a.copy(); col = arr[..., :3].copy()
    for _ in range(erode):
        a = a & np.roll(a, 1, 0) & np.roll(a, -1, 0) & np.roll(a, 1, 1) & np.roll(a, -1, 1)
    for _ in range(iters):
        acc = np.zeros_like(col); cnt = np.zeros(a.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sa = np.roll(a, (dy, dx), (0, 1)); acc += np.roll(col, (dy, dx), (0, 1)) * sa[..., None]; cnt += sa
        new = (~a) & (cnt > 0)
        col[new] = acc[new] / cnt[new][:, None]; a |= new
    out = np.ones_like(arr); out[..., :3] = col
    return out

front = bleed(px, clean)
back = bleed(back, clean & ~hole)  # i buchi (mano, collana) si riempiono sfumando i colori vicini
img_front = bpy.data.images.new('davanti', W, H, alpha=True)
img_front.pixels[:] = front.ravel()
img_front.pack()
img_back = bpy.data.images.new('retro', W, H, alpha=True)
img_back.pixels[:] = back.ravel()
img_back.pack()

def mat(name, image):
    """Materiale "unlit": l'immagine ha già le sue ombre, quindi il colore va dritto all'uscita
    (il .glb lo esporta come KHR_materials_unlit e i colori restano uguali all'immagine)."""
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.remove(nt.nodes['Principled BSDF'])
    t = nt.nodes.new('ShaderNodeTexImage'); t.image = image; t.extension = 'EXTEND'
    va = nt.nodes.new('ShaderNodeVertexColor'); va.layer_name = 'ombra'
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'
    mix.inputs['Factor'].default_value = 1.0
    nt.links.new(t.outputs['Color'], mix.inputs['A']); nt.links.new(va.outputs['Color'], mix.inputs['B'])
    nt.links.new(mix.outputs['Result'], nt.nodes['Material Output'].inputs['Surface'])
    return m

def mat_emissivo(m):
    """Per il .usdz (Quick Look non conosce l'unlit): base nera + emissione con l'immagine."""
    nt = m.node_tree; mix = next(n for n in nt.nodes if n.type == 'MIX')
    b = nt.nodes.new('ShaderNodeBsdfPrincipled')
    b.inputs['Base Color'].default_value = (0, 0, 0, 1)
    b.inputs['Roughness'].default_value = 1.0
    b.inputs['Specular IOR Level'].default_value = 0.0
    nt.links.new(mix.outputs['Result'], b.inputs['Emission Color'])
    b.inputs['Emission Strength'].default_value = 1.0
    nt.links.new(b.outputs['BSDF'], nt.nodes['Material Output'].inputs['Surface'])

ob.data.materials.clear()
ob.data.materials.append(mat('davanti', img_front))
ob.data.materials.append(mat('dietro', img_back))

# --- UV: proiezione frontale, il riquadro della forma combacia con il riquadro della figura nell'immagine
co = np.array([v.co[:] for v in ob.data.vertices])
mn, mx = co.min(0), co.max(0)
bm = bmesh.new(); bm.from_mesh(ob.data)
uv = bm.loops.layers.uv.verify()
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
for f in bm.faces:
    f.material_index = 1 if f.normal.y > 0.1 else 0  # in Blender il davanti guarda verso -Y
    for l in f.loops:
        x, y, z = l.vert.co
        u = (c0 + (x - mn[0]) / (mx[0] - mn[0]) * (c1 - c0) + 0.5) / W
        v = (r0 + (z - mn[2]) / (mx[2] - mn[2]) * (r1 - r0) + 0.5) / H
        l[uv].uv = (u, v)
bm.to_mesh(ob.data); bm.free()
ob.data.update()
ca = ob.data.color_attributes.new('ombra', 'BYTE_COLOR', 'POINT')
for v in ob.data.vertices:
    n = v.normal
    k = 1.0 - 0.28 * min(1.0, abs(n.x) ** 1.5) - 0.22 * max(0.0, -n.z) + 0.04 * max(0.0, n.z)
    k = max(0.55, min(1.0, k))
    ca.data[v.index].color = (k, k, k, 1.0)

# --- luci, camere, render di controllo
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.film_transparent = True
scene.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('mondo'); scene.world = world; world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.9
def light(name, energy, rot):
    l = bpy.data.lights.new(name, 'SUN'); l.energy = energy
    o = bpy.data.objects.new(name, l); scene.collection.objects.link(o)
    o.rotation_euler = Euler([math.radians(a) for a in rot]); return o
light('sole', 1.6, (48, 0, -35))
light('controluce', 0.8, (55, 0, 170))

def camera(name, pos, aim, lens=50, ortho=None):
    cd = bpy.data.cameras.new(name)
    if ortho: cd.type = 'ORTHO'; cd.ortho_scale = ortho
    else: cd.lens = lens
    co_ = bpy.data.objects.new(name, cd); scene.collection.objects.link(co_)
    co_.location = pos; co_.rotation_euler = (Vector(aim) - Vector(pos)).to_track_quat('-Z', 'Y').to_euler()
    return co_

def render(cam, name, w, h):
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.filepath = os.path.join(OUT, name)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(cam)

asp = W / H
render(camera('fronte', (0, -6, 0.875), (0, 0, 0.875), ortho=ALTEZZA * 1.02), NOME + '_fronte.png', W, H)
render(camera('tre_quarti', (2.0, -3.3, 1.2), (0, 0, 0.86), lens=55), NOME + '_tre_quarti.png', 800, 1000)
render(camera('lato', (4.0, 0, 0.9), (0, 0, 0.86), lens=55), NOME + '_lato.png', 600, 1000)
render(camera('retro', (-1.4, 3.6, 1.2), (0, 0, 0.86), lens=55), NOME + '_retro.png', 800, 1000)
render(camera('testa', (0.25, -1.6, 1.5), (0, 0, 1.5), lens=70), NOME + '_testa.png', 700, 700)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, NOME + '.blend'))
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, NOME + '.glb'), export_format='GLB',
                          use_selection=False, export_cameras=False, export_lights=False)
for m in ob.data.materials:
    mat_emissivo(m)
try:
    bpy.ops.wm.usd_export(filepath=os.path.join(OUT, NOME + '.usdz'), selected_objects_only=False, export_materials=True)
except Exception as e:
    print('usdz non esportato:', e)
print('FATTO', len(ob.data.polygons), 'facce')
