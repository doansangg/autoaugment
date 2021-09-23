import contextlib

def f(x):
    yield x**2

with contextlib.nested(f(3), f(4)) as (A, B):
    print ("squares:", A, B)
