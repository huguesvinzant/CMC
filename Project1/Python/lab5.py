""" Lab 5 """
from exercise1 import exercise1
import cmc_pylog as pylog


def main():
    """Main function that runs all the exercises."""
    pylog.info('Implementing Lab 5 : Exercise 1')
    exercise1()
    return


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    main()

