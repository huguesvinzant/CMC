"""Run all simulations in parallel"""

import os
import subprocess
from multiprocessing import Pool

import cmc_pylog as pylog


def run_world(world):
    """Run world"""
    path = os.path.dirname(os.path.realpath(__file__))
    webots_cmd = "webots {} --mode=fast --minimize".format(world)
    pylog.info(webots_cmd)
    subprocess.check_call(webots_cmd.format(world), shell=True, cwd=path)


def main():
    """Main"""
    path = os.path.dirname(os.path.realpath(__file__))
    worlds = [
        filename
        for filename in os.listdir(path)
        if filename.endswith('.wbt')
    ]
    with Pool(4) as pool:
        print(pool.map(run_world, worlds))


if __name__ == '__main__':
    main()

