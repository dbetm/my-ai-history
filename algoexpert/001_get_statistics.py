import math



def get_statistics(input_list):
    sorted_arr = sorted(input_list)
    n = len(input_list)

    total = 0
    freqs = dict()

    # MEAN
    for x in sorted_arr:
        total += x

        if x in freqs:
            freqs[x] += 1
        else:
            freqs[x] = 1

    mean = total / n

    # MEDIAN
    idx = n // 2
    if n & 1: # odd
        median = sorted_arr[idx]
    else:
        median = (sorted_arr[idx] + sorted_arr[idx-1]) / 2

    # MODE
    max_freq = 0
    mode = -1

    for key, val in freqs.items():
        if val > max_freq:
            max_freq = val
            mode = key

    # VARIANCE
    variance = 0
    for x in sorted_arr:
        variance += (x - mean)**2

    variance /= (n - 1)

    # STANDARD DEVIATION
    standard_deviation = math.sqrt(variance)

    # MEAN CONFIDENCE INTERVAL
    """A 95% confidence interval for the mean means that if you were to take many samples and construct a
    confidence interval from each of them, approximately 95% of those intervals would contain the true 
    population mean. It does not mean there is a 95% chance that the true mean is within the calculated 
    interval for your single sample. Instead, it reflects the reliability of the estimation method.
    """
    sdt_error_mean = standard_deviation / math.sqrt(n)
    """ For a 95% confidence interval, the critical value from the Z-distribution is 1.96 
    (for large samples). For smaller samples (n < 30), use the t-distribution and the appropriate 
    t-value for your degrees of freedom (df = n − 1).
    """
    z = 1.96
    margin_error = z * sdt_error_mean

    return {
        "mean": mean,
        "median": median,
        "mode": mode,
        "sample_variance": variance,
        "sample_standard_deviation": standard_deviation,
        "mean_confidence_interval": [mean - margin_error, mean + margin_error],
    }



if __name__ == "__main__":
    input_list = list(map(int, input().split(" ")))

    print(get_statistics(input_list))