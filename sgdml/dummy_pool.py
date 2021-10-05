class Pool():
    def __init__(self, i):
        print(f'Dummpy pool of {i} threads')

    def imap_unordered(self, func, iterable, processes=None):
        self._processes = 1
        for iter in iterable:
            yield func(iter)

    def imap(self, func, iterable):
        for iter in iterable:
            yield func(iter)

    def close(self):
        pass

    def join(self):
        pass