import matplotlib.pyplot as plt
import deep_morse_semiflow.DMS_for_ou_process
import pricing_discrete_ou_process.pricing_american_option_for_discrete_ou_process


def main():
    penalties = [10 ** (-i) for i in range(3, 6)]
    for penalty in penalties:
        test_x, values, label = deep_morse_semiflow.DMS_for_ou_process.main(penalty)
        plt.plot(test_x, values, label=label)
    test_x, values, label = pricing_discrete_ou_process.pricing_american_option_for_discrete_ou_process.main()
    plt.plot(test_x, values, label=label)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
