# Genera la forma 3D del personaggio dall'immagine di riferimento con Hunyuan3D-2mini (sul Mac, GPU "mps").
# Uso: venv/bin/python genera_forma.py <immagine.png> <uscita.glb> [seed] [risoluzione]
import os, sys, time

QUI = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('HY3DGEN_MODELS', os.path.join(QUI, 'modelli'))
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
sys.path.insert(0, os.path.join(QUI, 'Hunyuan3D-2'))

import torch
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover

immagine, uscita = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1234

t0 = time.time()
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mini', subfolder='hunyuan3d-dit-v2-mini', variant='fp16',
    device='mps', dtype=torch.float16)
print(f'modello caricato in {time.time() - t0:.0f}s', flush=True)

img = Image.open(immagine).convert('RGBA')
t0 = time.time()
RIS = int(sys.argv[4]) if len(sys.argv) > 4 else 320  # risoluzione della griglia: più alta = più dettagli, più lento
mesh = pipe(image=img, num_inference_steps=50, octree_resolution=RIS, num_chunks=6000,
            generator=torch.manual_seed(seed), output_type='trimesh')[0]
print(f'forma generata in {time.time() - t0:.0f}s', flush=True)
mesh = FloaterRemover()(mesh)
mesh = DegenerateFaceRemover()(mesh)
mesh.export(uscita)
print('salvato', uscita, len(mesh.vertices), 'vertici', len(mesh.faces), 'facce')
