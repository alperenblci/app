import numpy as np


# It's taking a list of numbers and returning the number that appears the most.
def trim_with_iqr(lst, max_rec: int = 10):
    """
    Trims the list with Inter-quartile Elimination method
    :param lst = List of data that we'll be trimmed
    :param max_rec =  Maximum recursion amount
    """

    # Get the first and third quartile
    q1_percentile = np.quantile(lst, 0.25)
    q3_percentile = np.quantile(lst, 0.75)

    # Calculate the Inter-quartile diff
    iqr = q3_percentile - q1_percentile

    # Calculating both Upper and Lower limit
    upper_limit = q3_percentile + (1.5 * iqr)
    lower_limit = q1_percentile - (1.5 * iqr)

    trimmed_list = []

    # For each element in list
    for element in lst:
        # Element is between the lists...
        if upper_limit >= element >= lower_limit:
            # ...put the element into list
            trimmed_list.append(element)

    # Recursive callback stop condition
    if upper_limit - lower_limit < 10 or max_rec == 0:
        return trimmed_list
    # Recursive call
    else:
        return trim_with_iqr(trimmed_list, max_rec - 1)


# It's taking a list of numbers and returning the number that appears the most.
def trim_with_interval(lst, max_range: float = .5):
    try:
        amount_map = {}
        for element in lst:
            key = (element / max_range) * max_range
            if key not in amount_map:
                amount_map[key] = 1
            else:
                amount_map[key] += 1

        # It's taking the `amount_map` dictionary and returning only the values that are greater than 1.
        amount_map = {key: value for (key, value) in amount_map.items() if value > 1}

        print(amount_map)

        return max(amount_map.items(), key=lambda item: item[1])[0]
    except ValueError:
        return 0.0
