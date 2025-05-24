from config import BConfig

class Domain:
    def __init__(self, spatial, temporal = None):
        self.spatial = spatial
        self.temporal = temporal

    def to_dict(self):
        return {
            "spatial": self.spatial,
            "temporal": self.temporal
        }
    def from_dict(data):
        spatial = data["spatial"]
        temporal = data["temporal"]
        return Domain(spatial=spatial, temporal=temporal)
    def __repr__(self):
        return f"Domain(spatial={self.spatial}, temporal={self.temporal})"

class METADATA:
    def __init__(self, domain:Domain, config:BConfig, lr:float, loss_label:list):
        self.config = config
        self.lr = lr
        self.domain = domain
        self.loss_label = loss_label

    def to_dict(self):
        return {
            "config": self.config,
            "lr": self.lr,
            "domain": self.domain
        }
    def from_dict(self, data):
        self.config = data["config"]
        self.lr = data["lr"]
        self.domain = data["domain"]