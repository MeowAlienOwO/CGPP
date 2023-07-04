from typing import Dict


class Solution:
    """Stores the solution of 1d bpp
    """

    @property
    def capacity(self) -> int:
        raise NotImplementedError

    @property
    def waste(self) -> int:
        raise NotImplementedError
    
    @property
    def metrics(self) -> Dict:
        raise NotImplementedError

