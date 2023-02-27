## Apriori

The fundamental idea is that some rules that we can notice in data...Apriori can be seen as "people who bought also bought"...It's similar to Naive Bayes classifier. But this can be used in a recommendation system.


**Support** - fractions of cases to support our hyphotesis first part.

Examples

1) Movie recommendation:

    `suport(M) = # users watchlists containing M / # user watchlists`

2) Market Basket Optimisation:

    `support(I) = # transactions containing I / # transactions`


**Confidence** - how well is our hyphotesis considering the second part.

Examples

1) Movie recommendation:
    
    `confidence(M1 -> M2) = (
        # user watchlist containing M_1 and M_2 
        / # use watchlists containing M_1`
    )


**Lift** - give us a general idea of the association rule. Measure the relevance of the pretended associative rule.

Examples

1) Movie recommendation:
    
    `lift(M1 -> M2) = confidence(M_1->M_2) / support(M2)`



It's a slow algorithm, because explores all the combinations between each possible pair (or maybe more items) of the data universe.

**Steps of the algorithm**

1) Set a minumum support and confidence.
2) Take all the subsets in the transactions having higher support than the support threshold.
3) Take all the rules of these subsets having higher confidence than the confidence threshold.
4) Sort the rules by decreasing lift.


## Ectlat intuition

This approach only has support (different from the apriori algo.).

**Support** - fractions of cases to support our hyphotesis first part.

Example

1) Market Basket Optimisation:

    `support(I) = # transactions containing I / # transactions`


**Steps of the algorithm**

1) Set a minumum support.
2) Take all the subsets in transactions having higher than threshold support.
3) Sort these subsets by decreasing support.