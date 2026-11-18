# Toglie lo sfondo bianco: riempie dal bordo solo il bianco "puro" collegato all'esterno,
# così le parti bianche del personaggio (scarpe, righe) restano.
# Uso: venv/bin/python scontorna.py <ingresso> <uscita.png>
import sys
import numpy as np
from PIL import Image
from scipy import ndimage

img = np.array(Image.open(sys.argv[1]).convert('RGB')).astype(np.int16)
quasi_bianco = img.min(-1) >= 246
lab, n = ndimage.label(quasi_bianco)
bordo = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
sfondo = np.isin(lab, bordo[bordo > 0])
figura = ~sfondo
# tieni solo il pezzo più grande (niente puntini sparsi) e chiudi eventuali buchi
lab2, n2 = ndimage.label(figura)
if n2 > 1:
    sizes = ndimage.sum(figura, lab2, range(1, n2 + 1))
    figura = lab2 == (np.argmax(sizes) + 1)
figura = ndimage.binary_fill_holes(figura)
alpha = (figura * 255).astype(np.uint8)
out = np.dstack([img.astype(np.uint8), alpha])
Image.fromarray(out, 'RGBA').save(sys.argv[2])
print('pixel figura:', int(figura.sum()), 'riquadro:', np.argwhere(figura).min(0), np.argwhere(figura).max(0))
