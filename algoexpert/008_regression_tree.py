FEATURE_NAMES = (
    "porosity",
    "gamma",
    "sonic",
    "density",
)
N = len(FEATURE_NAMES)
TARGET_NAME = "bpd"



class TreeNode:
    def __init__(self, examples):
        self.examples: list = examples
        self.left = None
        self.right = None
        self.split_point = None
        self.feature = None
        self.avg = None # only computed at leaves

    def split(self, feature_idx: int = 0):
        if len(self.examples) == 1:
            return
        
        # look for the split point which minimizes the MSE
        best_split_point = {
            "feature": None,
            "value": None,
            "mse": float("inf"),
            "split_index": None
        }

        for feature in FEATURE_NAMES:
            # order examples using the values of the current feature to get the split points
            self.examples.sort(key=lambda example : example[feature])

            # each split point will be the avg between two adjacent values
            for i in range(len(self.examples) - 1):
                split_point = (self.examples[i][feature] + self.examples[i+1][feature]) / 2

                # compute mse to determine if it's the best split point
                mse, split_idx = self.__get_split_point_mse(feature, split_point)

                if best_split_point["mse"] > mse:
                    best_split_point = {
                        "feature": feature,
                        "value": split_point,
                        "mse": mse,
                        "split_index": split_idx
                    }
        
        self.split_point = best_split_point
        self.examples.sort(key=lambda example: example[self.split_point["feature"]])

        self.left = TreeNode(examples=self.examples[:self.split_point["split_index"]])
        self.left.split()

        self.right = TreeNode(examples=self.examples[self.split_point["split_index"]:])
        self.right.split()


    def __get_split_point_mse(self, feature: str, split_point: float):
        left_labels = [example[TARGET_NAME] for example in self.examples if example[feature] <= split_point]
        right_labels = [example[TARGET_NAME] for example in self.examples if example[feature] > split_point]

        # we want to know if it's really has splitted the examples
        if not len(left_labels) or not len(right_labels):
            return None, None
        
        left_mse = get_mse(left_labels)
        right_mse = get_mse(right_labels)
        num_samples = len(left_labels) + len(right_labels)

        mse = ((len(left_labels) * left_mse) + (len(right_labels) * right_mse)) / num_samples

        # marks the idx where the values before belong to the left child
        split_index = len(left_labels)

        return mse, split_index


def get_mse(values: list) -> float:
    n = len(values)
    avg = sum(values) / n

    mse = 0

    for value in values:
        mse += (avg - value)**2

    return mse / n


class RegressionTree:
    def __init__(self, examples):
        # Don't change the following two lines of code.
        self.root = TreeNode(examples)
        self.train()

    def train(self):
        # Don't edit this line.
        self.root.split()

    def predict(self, example):
        node = self.root

        while node.left and node.right:
            if example[node.split_point["feature"]] <= node.split_point["value"]:
                node = node.left
            else:
                node = node.right
            
        
        leaf_labels = [leaf[TARGET_NAME] for leaf in node.examples]

        return sum(leaf_labels) / len(node.examples)

    def print(self):
        queue = list()
        queue.append(self.root)
        queue.append(None)

        while len(queue) > 0:
            node: TreeNode = queue.pop(0)

            if not node:
                if len(queue) > 0:
                    queue.append(None)
                print("")
            else:
                print(node.split_point, end=" ")

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)


if __name__ == "__main__":
    examples = [
        {
            "porosity": 0.24230820638070603,
            "gamma": 1.5000463819136288,
            "sonic": 2568.8231147730116,
            "density": -0.353639698833012,
            "bpd": 164.7544334411493,  # The label for this example.
        },
        {
            "porosity": 0.4821959432320581,
            "gamma": 1.4953123610344377,
            "sonic": 2768.8665660695128,
            "density": 1.1231264377284371,
            "bpd": 157.33821193599536,  # The label for this example.
        },
        {
            "porosity": 0.058672948847231135,
            "gamma": 1.5384704880812365,
            "sonic": 3236.794545516582,
            "density": 1.269807135982118,
            "bpd": 159.49129568528647,  # The label for this example.
        },
    ]

    tree = RegressionTree(examples)

    tree.train()

    print("PREDICTION\n")
    print(tree.predict(examples[2]))
    
    print("-"*24)
    tree.print()