""" Parse command arguments """

import argparse
from .default import DEFAULT


def parse_args(**kwargs):
    """ Parse arguments """
    course = kwargs.pop("course", "Computational Motor Control")
    lab = kwargs.pop("lab", "Lab")
    comp = kwargs.pop("compatibility", "Python2 and Python3 compatible")
    parser = argparse.ArgumentParser(
        description="{} - {} ({})".format(course, lab, comp),
        usage="python {}".format(__file__)
    )
    parser.add_argument(
        "--save_figures", "-s",
        help="Save all figures",
        dest="save_figures",
        action="store_true"
    )
    extension_support = "png/pdf/ps/eps/svg/..."
    extension_usage = "-e png -e pdf ..."
    parser.add_argument(
        "--extension", "-e",
        help="Output extension (Formats: {}) (Usage: {})".format(
            extension_support,
            extension_usage
        ),
        dest="extension",
        action="append"
    )
    args = parser.parse_args()
    DEFAULT["save_figures"] = args.save_figures
    if args.extension:
        DEFAULT["save_extensions"] = args.extension
    return args

