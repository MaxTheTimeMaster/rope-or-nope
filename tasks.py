import random

DIGITS = [str(d) for d in range(10)]
PAD, BOS, SEP, EOS = "<pad>", "<bos>", "<sep>", "<eos>"
VOCAB = [PAD, BOS, SEP, EOS, *DIGITS]
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}

TRAIN_LENGTHS = range(4, 17)
EVAL_LENGTHS = (16, 24, 32, 48, 64)
MAX_LENGTH = max(EVAL_LENGTHS)

TASKS = {
    "copy": lambda xs: list(xs),
    "reverse": lambda xs: xs[::-1],
    "sort": sorted,
    "love57" : lambda xs: list(map(lambda x: str(5) if int(x) % 2 == 0 else str(7), list(xs))),
}

def make_example(task: str, length: int, rng: random.Random) -> tuple[list[str], list[str]]:
    """Random input digits and the expected answer for one task instance."""
    xs = [rng.choice(DIGITS) for _ in range(length)]
    return xs, TASKS[task](xs)


def to_sequence(xs: list[str], ys: list[str]) -> list[str]:
    return [BOS, *xs, SEP, *ys, EOS]
