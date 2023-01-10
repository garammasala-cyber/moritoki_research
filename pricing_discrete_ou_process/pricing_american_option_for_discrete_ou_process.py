import numpy as np
import pricing_discrete_ou_process.discrete_OU_model_generator
from pathlib import Path
import json
import matplotlib.pyplot as plt

parent = Path(__file__).resolve().parent

with open(parent.joinpath('../parameter.json')) as parameter:
    parameters = json.load(parameter)

alpha = parameters["OU_Process"]["alpha"]
sigma = parameters["OU_Process"]["sigma"]
strike_price = parameters["Option"]["strike_price"]
num_partition = parameters["Deep_Learning"]["num_partition"]
expiration = parameters["OU_Process"]["T"]


def payoffs_function(model_value):
    return max(0, np.exp(model_value) - strike_price)


up_probability = pricing_discrete_ou_process.discrete_OU_model_generator.up_probability

delta = pricing_discrete_ou_process.discrete_OU_model_generator.Delta(num_partition, expiration, sigma)


def weight_deltas(k):
    return [i * delta for i in range(-k, k + 1, 2)]


sum_probability_dic = {-1: {0: 1 - up_probability(num_partition, expiration, alpha, sigma, 0.01),
                            1: up_probability(num_partition, expiration, alpha, sigma, 0.01)}}


def dic_init(initial_value=0.01):
    global sum_probability_dic
    sum_probability_dic = {1: {0: 1 - up_probability(num_partition, expiration, alpha, sigma, initial_value),
                               1: up_probability(num_partition, expiration, alpha, sigma, initial_value)}}


def sum_probability_for_weight(k=3, position=3, initial_value=0.01):
    global sum_probability_dic
    if k in sum_probability_dic.keys():
        if position in sum_probability_dic[k].keys():
            pass
        else:
            if position == k:
                sum_probability = up_probability(num_partition, expiration, alpha, sigma, initial_value
                                                 + weight_deltas(k - 1)[k - 1]) \
                                  * sum_probability_for_weight(k - 1, k - 1, initial_value)
            elif position == 0:
                sum_probability = (1 - up_probability(num_partition, expiration, alpha, sigma,
                                                      initial_value + weight_deltas(k - 1)[
                                                          0])) * sum_probability_for_weight(k - 1, 0, initial_value)
            else:
                if k - 1 in sum_probability_dic.keys():
                    if position in sum_probability_dic[k - 1].keys():
                        pass
                    else:
                        sum_probability1 = sum_probability_for_weight(k - 1, position, initial_value)
                        sum_probability_dic[k - 1][position] = sum_probability1
                    if position - 1 in sum_probability_dic[k - 1].keys():
                        pass
                    else:
                        sum_probability2 = sum_probability_for_weight(k - 1, position - 1, initial_value)
                        sum_probability_dic[k - 1][position - 1] = sum_probability2
                else:
                    sum_probability1 = sum_probability_for_weight(k - 1, position, initial_value)
                    sum_probability_dic[k - 1] = {position: sum_probability1}
                    sum_probability2 = sum_probability_for_weight(k - 1, position - 1, initial_value)
                    sum_probability_dic[k - 1][position - 1] = sum_probability2
                sum_probability \
                    = up_probability(num_partition, expiration, alpha, sigma, initial_value +
                                     weight_deltas(k - 1)[position - 1]) * sum_probability_dic[k - 1][position - 1] + \
                    (1 - up_probability(num_partition, expiration, alpha, sigma, initial_value +
                                        weight_deltas(k - 1)[position])) * sum_probability_dic[k - 1][position]
            sum_probability_dic[k][position] = sum_probability

    else:
        if position == k:
            sum_probability = up_probability(num_partition, expiration, alpha, sigma,
                                             initial_value + weight_deltas(k - 1)[
                                                 position - 1]) * sum_probability_for_weight(k - 1, position - 1)
        elif position == 0:
            sum_probability = (1 - up_probability(num_partition, expiration, alpha, sigma,
                                                  initial_value + weight_deltas(k - 1)[
                                                      0])) * sum_probability_for_weight(k - 1, 0)
        else:
            sum_probability1 = sum_probability_for_weight(k - 1, position)
            sum_probability_dic[k - 1] = {position: sum_probability1}
            sum_probability2 = sum_probability_for_weight(k - 1, position - 1)
            sum_probability_dic[k - 1][position - 1] = sum_probability2
            sum_probability \
                = up_probability(num_partition, expiration, alpha, sigma,
                                 initial_value + weight_deltas(k - 1)[position - 1]) * \
                sum_probability_dic[k - 1][position - 1] + \
                (1 - up_probability(num_partition, expiration, alpha, sigma, initial_value +
                                    weight_deltas(k - 1)[position])) * sum_probability_dic[k - 1][position]

        sum_probability_dic[k] = {position: sum_probability}

    return sum_probability_dic[k][position]


def european_option_pricing_discrete_ou(initial_value=0.01, time=0.01):
    k = int(num_partition * time)
    weights = [initial_value + deltas for deltas in weight_deltas(k)]
    probabilities = [sum_probability_for_weight(k, i) for i in range(k + 1)]
    return sum([payoffs_function(weight) * p for weight, p in zip(weights, probabilities)])


def american_option_pricing_discrete_ou(initial_value=0.01):
    dic_init(initial_value)
    value = max([european_option_pricing_discrete_ou(initial_value, i / num_partition)
                 for i in range(1, int(expiration * num_partition) + 1)])
    # print(initial_value, sum_probability_dic)
    return value


def main():
    test_x = [j / 10 for j in range(-50, 50 + 1)]
    test_american_values = [american_option_pricing_discrete_ou(variable) for variable in test_x]
    return test_x, test_american_values, 'american'


if __name__ == '__main__':
    x = [j / 10 for j in range(-50, 50 + 1)]
    american_values = [american_option_pricing_discrete_ou(variable) for variable in x]
    european_values = [european_option_pricing_discrete_ou(variable, expiration) for variable in x]
    plt.plot(x, european_values, label='european')
    plt.plot(x, american_values, label='american')
    plt.legend()
    plt.show()
