""" Lab 6 """
from exercise2 import exercise2
from exercise3 import exercise3
import cmc_pylog as pylog


def main():
    """Main function that runs all the exercises."""
    pylog.info('Implementing Lab 6 : Exercise 2')
    exercise2()
    pylog.info('Implementing Lab 6 : Exercise 3')
    exercise3()
    return


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    main()

