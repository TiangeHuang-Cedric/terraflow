class Operator:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Children class must implement __call__")
