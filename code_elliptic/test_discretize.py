from pathlib import Path
import sys
# ensure workspace package imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from discretize_elliptic import discretize

print('Calling discretize...')
try:
    fom = discretize(dx=8)
    print('state_dim', fom.pde.state_dim)
    print('input_dim', fom.pde.input_dim)
    print('A shape', fom.pde.A.shape)
    print('B shape', getattr(fom.pde.B).shape)
    print('M shape', fom.pde.M.shape)
except Exception as e:
    print('ERROR during discretize():', e)
    raise

