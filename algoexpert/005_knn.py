import math


def euclidean_distance(X, Y):
    total = 0

    for x, y in zip(X, Y):
        total += (x - y)**2

    return math.sqrt(total)


# Should use the `find_k_nearest_neighbors` function below.
def predict_label(examples, features, k, label_key="is_intrusive"):
    k_nearests_neighbors = find_k_nearest_neighbors(examples, features, k)

    label_freq_map = dict()

    for nearests_neighbor in k_nearests_neighbors:
        label = examples[nearests_neighbor][label_key]

        if label in label_freq_map:
            label_freq_map[label] += 1
        else:
            label_freq_map[label] = 1

    max_freq = 0
    target_label = None

    for label, freq in label_freq_map.items():
        if freq > max_freq:
            max_freq = freq
            target_label = label

    return target_label



def find_k_nearest_neighbors(examples: dict, features, k):
    distances = list() # list(tuple(distance, pid))
    k_nearests_neighbors = list()

    for pid, example in examples.items():
        distance = euclidean_distance(example["features"], features)
        distances.append((distance, pid))

    distances.sort()

    for idx in range(k):
        k_nearests_neighbors.append(distances[idx][1])

    return k_nearests_neighbors