"""Compatibility shims to replace small utilities from the reference project.
Provides a minimal `collection()` factory and a lightweight `model` class alias.
Put here so other modules can import from `compat` instead of depending on
`github_Kartmann` or its `methods`/`model` modules.
"""

class Collection:
    """Minimal container for arbitrary attributes."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def as_dict(self):
        return self.__dict__.copy()
    def __repr__(self):
        return f"Collection({self.__dict__})"


def collection(**kwargs):
    return Collection(**kwargs)


class Model:
    """Lightweight placeholder model wrapper.
    Stores pde/cost/time/space/options as attributes. Replace with the full
    project `model` implementation when available.
    """
    def __init__(self, pde, cost, time, space, options=None):
        self.pde = pde
        self.cost = cost
        self.time = time
        self.space = space
        self.options = options
    def __repr__(self):
        return f"Model(pde={len(getattr(self.pde,'__dict__',{}))} attrs, cost={len(getattr(self.cost,'__dict__',{}))})"

# expose compatibility names used in the reference code
model = Model


# Small convenience: allow `from compat import Collection, Model` as well
Collection = Collection
Model = Model

