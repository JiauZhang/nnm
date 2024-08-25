from torchvision import (
    datasets as _D,
    transformers as _T,
)
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

class MNISTDataset(_D.MNIST):
    def __init__(
        self, root: Union[str, Path], train: bool = True,
        transform: Optional[Callable] = _T.Compose([_T.ToTensor()]),
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ) -> None:
        super().__init__(
            root, train=train, transform=transform,
            target_transform=target_transform, download=download,
        )
