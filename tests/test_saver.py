import torch, random
from glob import glob
from nnm.checkpoint.saver import Saver, BestSaver

class TestSaver:
    conv = torch.nn.Conv2d(3, 3, 3)
    prefix = 'saver-test'

    def test_saver(self):
        cache = '.pytest_cache/saver'
        saver = Saver(cache, self.conv)
        for i in range(saver.max_to_keep + 3):
            saver.save(prefix=self.prefix)
        pth = glob(f'{cache}/*{self.prefix}*.pth')
        assert len(pth) == saver.max_to_keep

    def test_best_saver(self):
        cache = '.pytest_cache/bestsaver'
        saver = BestSaver(cache, self.conv)
        for i in range(saver.max_to_keep + 3):
            saver.save(random.random(), prefix=self.prefix)
        pth = glob(f'{cache}/*{self.prefix}*.pth')
        assert len(pth) == saver.max_to_keep
