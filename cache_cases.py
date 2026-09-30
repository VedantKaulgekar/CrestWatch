"""Build and cache the synthetic training/test cases once, so repeated short training calls don't
re-pay the ~5s/case data-generation cost.  python cache_cases.py [n_train] [n_test]"""
import sys, pickle, pathlib
from train_gnn import build_cases
n_tr, n_te = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (30, 15)
pathlib.Path("data/cache").mkdir(parents=True, exist_ok=True)
pickle.dump(build_cases(n_tr, seed0=0), open("data/cache/train_cases.pkl", "wb"))
pickle.dump(build_cases(n_te, seed0=1_000_000), open("data/cache/test_cases.pkl", "wb"))
print(f"cached {n_tr} train + {n_te} test cases")
