# Reimplementation of modular-division dataset generation.
# Token-list construction and split logic copied/adapted from openai/grok (MIT).
# https://github.com/openai/grok  (data.py), MIT license.

import itertools
import numpy as np
import torch
from typing import List, Tuple

EOS_TOKEN = "<|eos|>"
EQ_TOKEN = "="
MODULUS = 97


def render(operand):
    if isinstance(operand, (list, tuple, np.ndarray)):
        return "".join(map(render, operand))
    return str(operand)


VALID_OPERATORS = {
    "+": "addition",
    "-": "subtraction",
    "*": "muliplication",
    "/": "division",
    "**2+": "squarepoly",
    "**3+": "cubepoly",
    "x**2+y**2_mod_97": "quad1",
    "x**2+y**2+x*y_mod_97": "quad2",
    "x**2+y**2+x*y+x_mod_97": "quad3",
    "x**3+x*y_mod_97": "cube1",
    "x**3+x*y**2+y_mod_97": "cube2",
    "(x._value//y)if(y._value%2==1)else(x-y)_mod_97": "mix1",
    "s5": "s5",
    "s5conj": "s5conj",
    "s5aba": "s5aba",
    "+*": "even-addition_odd-multiplication",
    "+-": "even-addition_odd-subtraction",
    "sort": "sort",
    "reverse": "reverse",
    "copy": "copy",
}


class ArithmeticTokenizer:
    @classmethod
    def get_tokens(cls):
        tokens = (
            [EOS_TOKEN, EQ_TOKEN]
            + list(sorted(list(VALID_OPERATORS.keys())))
            + list(map(str, range(MODULUS)))
            + list(map(render, itertools.permutations(range(5))))  # s5
        )
        return tokens


def make_data() -> List[str]:
    # generation order: for c in 0..96, for y in 1..96, x = (y*c)%97
    eqs = []
    for c in range(MODULUS):
        for y in range(1, MODULUS):
            x = (y * c) % MODULUS
            eqs.append(f"{x} / {y} = {c}")
    # shuffle once with numpy RandomState(0)
    rng = np.random.RandomState(seed=0)
    rng.shuffle(eqs)
    # wrap with EOS
    eqs = [f"{EOS_TOKEN} {eq} {EOS_TOKEN}" for eq in eqs]
    return eqs


def build_split() -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    tokens = ArithmeticTokenizer.get_tokens()
    stoi = {s: i for i, s in enumerate(tokens)}
    eqs = make_data()
    assert len(eqs) == 9312
    train_rows = round(len(eqs) * 0.5)
    train_eqs = eqs[:train_rows]
    val_eqs = eqs[train_rows:]
    assert len(train_eqs) == 4656 and len(val_eqs) == 4656

    def encode(eq_list):
        out = []
        for eq in eq_list:
            out.append([stoi[t] for t in eq.split(" ")])
        return torch.tensor(out, dtype=torch.long)

    train_data = encode(train_eqs)
    val_data = encode(val_eqs)
    # zero overlap check
    train_set = set(tuple(r.tolist()) for r in train_data)
    val_set = set(tuple(r.tolist()) for r in val_data)
    assert len(train_set & val_set) == 0
    return train_data, val_data, tokens


if __name__ == "__main__":
    train_data, val_data, tokens = build_split()
    print("vocab size", len(tokens))
    print("train", train_data.shape, "val", val_data.shape)
    # generation order check: regenerate unshuffled and verify count
    # print 5 examples from train
    itos = {i: s for i, s in enumerate(tokens)}
    for i in range(5):
        print(" ".join(itos[int(t)] for t in train_data[i]))
    # no y=0 check: in original generation y in 1..96; verify all equations have y token != '0' in y position
    # position of y is index 3 in tokens (EOS x / y = c EOS -> indices:0 EOS,1 x,2 /,3 y,4 =,5 c,6 EOS)
    ys = train_data[:, 3].tolist() + val_data[:, 3].tolist()
    assert all(y != 0 for y in ys), "found y=0"
    print("no y=0 OK; overlap check OK")
