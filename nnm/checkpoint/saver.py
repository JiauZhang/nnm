class Saver():
    def __init__(self, model, max_to_keep=5):
        self.model = model
        self.max_to_keep = max_to_keep
