import math


class Metrics():
    def euclidean_distance(self, X, Y):
        total = 0

        for x, y in zip(X, Y):
            total += (x - y)**2

        return math.sqrt(total)

    def manhattan_distance(self, X, Y):
        total = 0

        for x, y in zip(X, Y):
            total += abs(x - y)

        return total

    def cosine_similarity(self, X, Y):
        num = 0
        den_a = sum(map(lambda x : x**2, X))
        den_b = sum(map(lambda x : x**2, Y))

        for x, y in zip(X, Y):
            print(f"{x=}, {y=}")
            num += x*y

        return num / (math.sqrt(den_a) * math.sqrt(den_b))

    def jaccard_similarity(self, X, Y):
        return len(set(X).intersection(set(Y))) / len(set(X).union(set(Y)))


def distances_and_similarities(X, Y):
    metrics = Metrics()
    return [
        metrics.euclidean_distance(X, Y),
        metrics.manhattan_distance(X, Y),
        metrics.cosine_similarity(X, Y),
        metrics.jaccard_similarity(X, Y)
    ]


if __name__ == "__main__":
    X = list(map(int, input().split(" ")))
    Y = list(map(int, input().split(" ")))

    print(distances_and_similarities(X, Y))


