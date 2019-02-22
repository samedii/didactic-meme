import collections


class AttributeDictionary:
    attributes = {}
    def __init__(self, **attributes):
        self.attributes = attributes
        for key, value in attributes.items():
            if isinstance(value, dict):
                attributes[key] = adict(**value)

    def __getattr__(self, name):
        if name in self.attributes:
            return self.attributes[name]
        return getattr(self.attributes, name)

    def __setattr__(self, name, value):
        if name in self.attributes:
            self.attributes[name] = value
        else:
            super().__setattr__(name, value)

    def __getitem__(self, name):
        if name in self.attributes:
            return self.attributes[name]
        return None

    def __setitem__(self, name, value):
        self.attributes[name] = value

    def to_dict(self):
        attributes = self.attributes.copy()
        for key, value in attributes.items():
            if isinstance(value, AttributeDictionary):
                attributes[key] = value.to_dict()
        return attributes

    def deep_update(self, overrides):
        for key, value in overrides.items():
            if isinstance(value, dict) or isinstance(value, adict):
                self[key] = self.get(key, adict()).deep_update(value)
            else:
                self[key] = overrides[key]
        return self


adict = AttributeDictionary
