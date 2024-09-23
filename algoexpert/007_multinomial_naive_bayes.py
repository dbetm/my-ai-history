import math
from collections import defaultdict
from pprint import pprint



class MultinomialNB:
    """it's wrong by a reason I've found :(
    """
    def __init__(self, articles_per_tag):
        # Don't change the following two lines of code.
        self.articles_per_tag: dict = articles_per_tag  # See question prompt for details.
        self.num_articles = self.count_articles()
        self.priors = dict()
        self.legit_likelihoods = dict()
        self.alpha = 1
        self.train()

    def train(self):
        # compute the priors for each tag (class)
        """To compute priors, given a C class, we need to do this
        count the number of articles belonging to C class and divide by the total number of articles
        """
        self.priors = self.compute_priors()

        # compute the legit likelihood based on the vocabulary for each tag, use Laplacian smoothing +1/+2
        """For each word in each tag, we want to know P(W=word|tag), so we will divide the number that the word
        appears in the articles between the total of words of all the articles for the choosen tag.
        """
        self.legit_likelihoods = self.compute_legit_likelihoods()

    def predict(self, article) -> dict:
        """Prediction in this case, instead of Bayes formula we will use:
        P(tag|article) ~ log(P(tag)) + log(P(article|tag)) which are the posteriors
        """
        DEFAULT_LEGIT_LIKELIHOOD = 0.5

        posteriors_per_tag = {tag: math.log(self.priors[tag]) for tag in self.articles_per_tag.keys()}

        for word in article:
            for tag in self.articles_per_tag.keys():
                posteriors_per_tag[tag] += math.log(
                    self.legit_likelihoods[tag].get(word, DEFAULT_LEGIT_LIKELIHOOD)
                )

        return posteriors_per_tag

        # for tag in self.articles_per_tag:
        #     numerator = 0

        #     for word in article:
        #         numerator += math.log(self.legit_likelihoods[tag].get(word, DEFAULT_LEGIT_LIKELIHOOD))

        #     posteriors[tag] = math.log(self.priors[tag]) + numerator

        # return posteriors

    def predict_bayes(self, article) -> dict:
        # use the Bayes formula, when computing the legits likelihoods, when a word is not found, assign a 
        # 50% of prob (0.5)
        predictions = dict()
        DEFAULT_LEGIT_LIKELIHOOD = 0.5

        common_denominator = 0
        for tag in self.articles_per_tag:
            tmp = self.priors[tag]

            for word in article:
                tmp *= self.legit_likelihoods[tag].get(word, DEFAULT_LEGIT_LIKELIHOOD)

            common_denominator += tmp

        for tag in self.articles_per_tag:
            numerator = self.priors[tag]

            for word in article:
                numerator *= self.legit_likelihoods[tag].get(word, DEFAULT_LEGIT_LIKELIHOOD)

            predictions[tag] = (numerator / common_denominator) * 100

        return predictions

    def count_articles(self):
        cont = 0

        for tag, articles in self.articles_per_tag.items():
            cont += len(articles)

        return cont

    def compute_priors(self):
        priors = dict()

        for tag, articles in self.articles_per_tag.items():
            priors[tag] = len(articles) / self.num_articles

        return priors

    def compute_legit_likelihoods(self):
        legit_likelihoods = dict()

        for tag, articles in self.articles_per_tag.items():
            legit_likelihoods[tag] = self.compute_legit_likelihood(articles)

        return legit_likelihoods

    def compute_legit_likelihood(self, articles) -> dict:
        legit_likelihood = dict()
        vocab = set()
        num_words = 0

        for article in articles:
            for word in article:
                vocab.add(word) # is it case sensitive?
                num_words += 1

        for word in vocab:
            cont = 0
            for article in articles:
                for word_ in article:
                    if word_ == word:
                        cont += 1
            legit_likelihood[word] = (cont + 1*self.alpha) / (num_words + 2*self.alpha)

        return legit_likelihood



class Solution:
    def __init__(self, articles_per_tag):
        # Don't change the following two lines of code.
        self.articles_per_tag: dict = articles_per_tag  # See question prompt for details.
        self.priors_per_tag = dict()
        self.likelihood_per_word_per_tag = dict()
        self.tags = self.articles_per_tag.keys()
        self.alpha = 1
        self.train()

    def train(self):
        tags_count_map = {tag: len(self.articles_per_tag[tag]) for tag in self.tags}
        total_articles = sum(tags_count_map.values())
        self.priors_per_tag = {tag: tags_count_map[tag] / total_articles for tag in self.tags}
        self.likelihood_per_word_per_tag = self.__get_word_likelihoods_per_tag()

    def predict(self, article) -> dict:
        posteriors_per_tag = {tag: math.log(self.priors_per_tag[tag]) for tag in self.tags}

        for word in article:
            for tag in self.tags:
                posteriors_per_tag[tag] += math.log(self.likelihood_per_word_per_tag[word][tag])

        return posteriors_per_tag

    def __get_word_likelihoods_per_tag(self):
        # Ocurrences of each word for each tag, by default for a single word, no appears in any tag
        word_frequencies_per_tag = defaultdict(lambda: {tag: 0 for tag in self.tags})
        # Total of words for each tag, by default will be 0
        total_word_count_per_tag = defaultdict(int)

        for tag in self.tags:
            for article in self.articles_per_tag[tag]:
                for word in article:
                    word_frequencies_per_tag[word][tag] += 1
                    total_word_count_per_tag[tag] += 1

        word_likelihoods_per_tag = defaultdict(lambda: {tag: 0.5 for tag in self.tags})

        for word, tags_map in word_frequencies_per_tag.items():
            for tag in tags_map.keys():
                word_likelihoods_per_tag[word][tag] = (
                    (word_frequencies_per_tag[word][tag] + 1*self.alpha)
                    / (total_word_count_per_tag[tag] + 2*self.alpha)
                )

        return word_likelihoods_per_tag


if __name__ == "__main__":
    articles_per_tag = {
        "politics": [
            ["article", "writes", "Joel", "Furr", "writes"], # ... more words],
            ["Distribution", "world", "following", "posted"], # ... more words],
            # ... More articles.
        ],
        "sports": [
            ["article", "writes", "just", "wanted"], # ... more words],
            ["Phillies", "salvaged", "their", "weekend"], # ... more words],
            # ... More articles.
        ],
        "tech": [
            ["Thanks", "Steve", "your", "helpful"], # ... more words],
            ["Please", "unsubscribe", "This", "user"], # ... more words],
            # ... More articles.
        ],
        # ... More categories.
    }

    classifier1 = MultinomialNB(articles_per_tag)
    classifier2 = Solution(articles_per_tag)

    article = [
        "article", "writes", "while", "when", "owned", "Plus",
        "wanted", "upgrade", "memory", "just", "ordered", "toolkit",
        "from", "Macwarehouse", "something", "like", "included", "antistatic"
    ]

    print(classifier1.predict(article))
    print("-"*23)
    print(classifier2.predict(article))