

class AttributeDictionary(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        return super().__getattribute__(key)

    def __setattr__(self, key, value):
        if key in self:
            self.__setitem__(key, value)
        else:
            super().__setattr__(key, value)

    def deep_update(self, overrides):
        for key, value in overrides.items():
            if isinstance(value, dict) or isinstance(value, adict):
                self[key] = self.get(key, adict()).deep_update(value)
            else:
                self[key] = overrides[key]
        return self


adict = AttributeDictionary
