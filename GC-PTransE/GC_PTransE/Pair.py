class Pair:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        if self == other:
            return True
        if other is None or type(other) != type(self):
            return False
        pair = other
        if self.a != pair.a:
            return False
        return self.b == pair.b

    def __hash__(self):
        result = hash(self.a) if self.a is not None else 0
        result = 31 * result + (hash(self.b) if self.b is not None else 0)
        return result
