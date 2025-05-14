class Scale:
    def __init__(self, domain_scale=[(0.0,1.0)], time_scale=(0.0,1.0), load_scale=(0.0,1.0)):
        self.domain_scale = domain_scale
        self.time_scale = time_scale
        self.load_scale = load_scale

    def scale(self, input):
        pass
    def rescale(self, input):
        pass
