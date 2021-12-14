import argparse
import time

from src.checker.tester import Tester


def register_launch_arguments():
    parser = argparse.ArgumentParser(description='Serve the connector application')
    parser.add_argument('-o', '--objects', help='path to the directory with objects images', required=True)
    parser.add_argument('-c', '--config', help='path to the configuration file', required=True)

    return parser.parse_args()


def check_placer():
    args = register_launch_arguments()
    start = time.time()
    tester = Tester(args.objects, args.config)
    tester.run()
    end = time.time()
    print(f'Time of placer check: {end - start} seconds')


if __name__ == '__main__':
    check_placer()
