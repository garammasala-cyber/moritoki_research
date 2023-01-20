import matplotlib.pyplot as plt
import deep_morse_semiflow.sta
import pricing_discrete_ou_process.pricing_american_option_for_discrete_ou_process
import numpy as np

def main():
    penalties = [10 ** (-i/2) for i in range(50, 53 + 1)]
    for penalty in penalties:
        test_x, values, label = deep_morse_semiflow.sta.main(penalty)
        plt.plot(test_x, values, label=label)

    test_x, values, label = pricing_discrete_ou_process.pricing_american_option_for_discrete_ou_process.main()
    plt.plot(test_x, values, label=label)
    """x = np.array([j / 10 for j in range(-50, 50 + 1)])
    y = np.exp(- x**2) - 0.5
    plt.plot(x, y, label='phi')"""
    plt.legend()
    plt.savefig("sta_experiment.png")


if __name__ == '__main__':
    main()
