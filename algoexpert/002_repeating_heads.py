def repeating_heads(n, x):

    prob_win = 1 / 2**n
    prob_win_at_least_one = 1 - (1-prob_win)**x

    # I want to make sure the win at least 100 to don't lose anything, so the bet payout must be
    # prob_win_at_least_one * [bet_payout] = 100
    bet_payout = 100 / prob_win_at_least_one

    return [prob_win_at_least_one * 100, bet_payout]


if __name__ == "__main__":
    n, x = tuple(map(int, input().split(" ")))

    print(repeating_heads(n, x))