# src/metagen_ai/datasets/toy_arith.py
from __future__ import annotations
from typing import Dict, Iterator, Tuple
import random

class ToyArithDataset:
    """
    Tiny arithmetic task generator for self-evolving loop.
    Each sample is: question = "What is a + b?" with answer = str(a+b).
    """
    def __init__(self, seed: int = 42, low: int = 1, high: int = 50):
        self.rnd = random.Random(seed)
        self.low = int(low)
        self.high = int(high)

    def sample(self) -> Dict[str, str]:
        a = self.rnd.randint(self.low, self.high)
        b = self.rnd.randint(self.low, self.high)
        return {"question": f"What is {a} + {b}?", "answer": str(a + b)}

    def stream(self) -> Iterator[Dict[str, str]]:
        while True:
            yield self.sample()
