class Domain:
    def __init__(self, spatial_dim, temporal_dim):
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.input_dim = spatial_dim + temporal_dim
        
        self.output_dim = 1

        