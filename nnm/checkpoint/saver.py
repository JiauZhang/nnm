import os, torch
from conippets import json

class Saver():
    def __init__(self, save_dir, model, max_to_keep=5):
        self.save_dir = save_dir
        self.model = model
        self.max_to_keep = max_to_keep
        os.makedirs(self.save_dir, exist_ok=True)
        self.file_lock_path = os.path.join(self.save_dir, 'file-lock.json')
        self.file_lock = []
        self.save_count = -1
        if os.path.exists(self.file_lock_path):
            self.file_lock = json.read(self.file_lock_path)
            if len(self.file_lock):
                self.save_count = int(self.file_lock[-1][-10:-4])

    def save(self, prefix=None):
        self.save_count += 1
        save_path = os.path.join(self.save_dir, 'checkpoint')
        if prefix: save_path += f'-{prefix}'
        save_path += f'-{self.save_count:06d}.pth'
        torch.save(self.model.state_dict(), save_path)
        return save_path
