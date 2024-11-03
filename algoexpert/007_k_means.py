import random
from typing import List


class Centroid:
    def __init__(self, location):
        self.location = location
        self.closest_users = set()

    def update_location(self, user_feature_map: dict, num_features_per_user: int):
        user_features = [user_feature_map[user_id] for user_id in self.closest_users]

        for jdx in range(num_features_per_user):
            acc = 0
            for idx in range(len(user_features)):
                acc += user_features[idx][jdx]
            self.location[jdx] = acc / len(user_features)

    def clear_users(self):
        self.closest_users = set()


def manhattan_distance(X, Y):
    total = 0

    for x, y in zip(X, Y):
        total += abs(x - y)

    return total


def get_k_means(user_feature_map: dict, num_features_per_user: int, k):
    # Don't change the following two lines of code.
    random.seed(42)
    # Gets the inital users, to be used as centroids.
    inital_centroid_users = random.sample(sorted(list(user_feature_map.keys())), k)

    centroids: List[Centroid] = list()

    for inital_centroid_user in inital_centroid_users:
        centroid = Centroid(location=user_feature_map[inital_centroid_user].copy())

        centroids.append(centroid)

    NUM_ITERS = 10

    for _ in range(NUM_ITERS):
        for user_id, features in user_feature_map.items():
            min_distance = float("inf")
            centroid_ref = None

            for centroid in centroids:
                distance = manhattan_distance(X=features, Y=centroid.location)

                if min_distance > distance:
                    min_distance = distance
                    centroid_ref = centroid

            centroid_ref.closest_users.add(user_id)

        for centroid in centroids:
            centroid.update_location(user_feature_map, num_features_per_user)
            centroid.clear_users()

    return [centroid.location for centroid in centroids]


if __name__ == "__main__":
    num_features_per_user = 2
    # user_feature_map = {
    #     "uid_0": [-1.479359467505669, -1.895497044385029, -2.0461402601759096, -1.7109256402185178],
    #     "uid_1": [-1.8284426855307128, -1.714098142408679, -0.98936286696649455, -1.5766569391907947],
    #     "uid_2": [-1.839893321836004, -1.7896757009107565, -1.1370171775666063, -1.0218512556938231],
    #     "uid_3": [-1.23224975874512, -1.8447858273094768, -1.8496517744301924, -2.4720755654344186],
    #     "uid_4": [-1.7714377791268318, -1.2725603446513774, -1.5512094954034525, -1.2589442628984848],
    # }
    user_feature_map = {
        "uid_0": [2, 3],
        "uid_1": [5, 8],
    }

    k = 1

    print(get_k_means(user_feature_map, num_features_per_user, k))
