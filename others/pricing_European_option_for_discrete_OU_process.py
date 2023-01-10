from pricing_discrete_ou_process import discrete_OU_model_generator

p = discrete_OU_model_generator.up_probability
Delta = discrete_OU_model_generator.Delta

d = {}
val1 = None
val2 = None


def european_option_pricing(num_time_partition=100, time=0.01, expiration=1, alpha=1.1, sigma=1.2, initial_value=0.8,
                            strike_price=0.2, interest_rate=0.03):
    delta = Delta(num_time_partition, expiration, sigma)
    dt = expiration / num_time_partition
    x_plus = initial_value + delta
    x_minus = initial_value - delta

    if abs(time - expiration) < 1e-5:
        return max(0, initial_value - strike_price)

    if time + dt in d.keys():
        if x_plus in d[time + dt].keys():
            pass
        else:
            upper_value = european_option_pricing(num_time_partition, time + dt, expiration, alpha, sigma, x_plus,
                                                  strike_price, interest_rate)
            d[time + dt][x_plus] = upper_value
        if x_minus in d[time + dt].keys():
            pass
        else:
            lower_value = european_option_pricing(num_time_partition, time + dt, expiration, alpha, sigma, x_minus,
                                                  strike_price, interest_rate)
            d[time + dt][x_minus] = lower_value
    else:
        upper_value = european_option_pricing(num_time_partition, time + dt, expiration, alpha, sigma, x_plus,
                                              strike_price, interest_rate)
        d[time + dt] = {x_plus: upper_value}
        lower_value = european_option_pricing(num_time_partition, time + dt, expiration, alpha, sigma, x_minus,
                                              strike_price, interest_rate)
        d[time + dt][x_minus] = lower_value

    if time in d.keys():
        if initial_value in d[time].keys():
            pass
        else:
            val = max(max(0, initial_value - strike_price),
                      (p(alpha, initial_value) * d[time + dt][x_plus] + (1 - p(alpha, initial_value)) * d[time + dt][x_minus]) / (1 + interest_rate))
            d[time][initial_value] = val
    else:
        val = max(max(0, initial_value - strike_price),
                  (p(alpha, initial_value) * d[time + dt][x_plus] + (1 - p(alpha, initial_value)) * d[time + dt][x_minus]) / (1 + interest_rate))
        d[time] = {initial_value: val}

    return d[time][initial_value]


if __name__ == '__main__':
    import time

    n = 100
    t = 0.01
    T = 1
    alpha = 0.008
    sigma = 1.2
    x = 0.09
    K = 0.1
    r = 0.03

    start = time.time()

    print(european_option_pricing(n, t, T, alpha, sigma, x, K, r))
    print(f'total : {(time.time() - start)} second.')
