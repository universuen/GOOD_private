array = [i for i in range(10)]


def split(list_a: list, chunk_size: int):
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]


for i in split(array, 10):
    print(i)
