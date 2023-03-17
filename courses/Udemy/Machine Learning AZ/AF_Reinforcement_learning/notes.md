## The Multi-armed Bandit Problem

The problem to tackle in this section of reinforcement learning will be the multi-armed bandit problem which consist in figure out the distributions of outputs of several machine, which have an "arm" to be operated. This in order to maximize the profit (or something like that). While using a machine you are spending money.

![](./assets/slot_machines.png)

For example you can find that the third machine has the best distribution, because its output are skewed to the left which means higher outcomes.


## Upper Confidence Bound (algorithm) Intuition (UCB)

Example:
- We have d arms. For example, arms are ads that we display to users each time they connect to a web page.
- Each time a user connects to this web page, that makes a round.
- At each round n, we choose one ad to display to the user.
- At each round n, ad i gives reward r_i(n) E {0, 1}; r_i(n) = 1 if the user clicked on the ad i, 0 if the user didn't.
- Our goal is to maximize the total reward we get over many rounds.

**Algorithm**

Step 1: At each round n, we consider two numbers for each ad i:

    - N_i(n) - the number of times the ad i was selected up to round n.
    - R_i(n) - the sum of rewards of the ad i up to round n.

Step 2: From these 2 numbers, we compute:

![](./assets/ucb_algo_formulas.png)

Step 3: We select the ad i that has the maximum UCb r_i(n) + delta_i(n).