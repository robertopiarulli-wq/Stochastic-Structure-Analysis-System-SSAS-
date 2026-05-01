import random

def generate_sestina():
    return sorted(random.sample(range(1, 91), 6))


def generate_ensemble(n=100000):
    for _ in range(n):
        yield generate_sestina()
