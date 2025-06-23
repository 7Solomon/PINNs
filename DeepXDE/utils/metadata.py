from dataclasses import dataclass
import json
import inspect

class BSaver:
    def __init__(self, **attributes):
        """
        Initializes a BSaver instance with given attributes.
        Primarily used by load_from_json.
        """
        self.__dict__.update(attributes)

    def save_to_json(self, filepath):
        """Saves all instance attributes and properties to a JSON file."""
        data_to_save = self.__dict__.copy()
        
        for name, _ in inspect.getmembers(self.__class__, lambda o: isinstance(o, property)):
            data_to_save[name] = getattr(self, name)

        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)

    @classmethod
    def load_from_json(cls, filepath):
        """
        Loads attributes from a JSON file and creates a new BSaver instance.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

class BConfig(BSaver):
    def get(self, attr_name, default=None):
        return getattr(self, attr_name, default)
    def save_to_json(self, filepath):
        """
        Saves the Pydantic model to a JSON file using its built-in serialization.
        """
        if hasattr(self, 'model_dump'):
            data_to_save = self.model_dump()
        else:
            data_to_save = self.dict()
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)
    
    
    def dict(self):
        """
        Returns the model as a dictionary.
        """
        if hasattr(self, 'model_dump'):
            return self.model_dump()
        else:
            return self.__dict__

        
class Domain:
    def __init__(self, spatial, temporal = None):
        self.spatial = spatial
        self.temporal = temporal
        #self.scale = BScale(self)

    def set_scale(self, scale):
        self.scale = scale

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

