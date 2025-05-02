
import argparse
import sys
from manager import manage_args
from vars import device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HEAT PINN')
    subparsers = parser.add_subparsers(dest='command', help='Verfügnare Commands', required=True)

    parser_create = subparsers.add_parser('add', help='Erstelle ein neues model.')
    parser_load = subparsers.add_parser('load', help='Lade ein model.')

    parser_load.add_argument('--vis', choices=['loss', 'field', 'all'], help='Wie visualisieren? loss oder field', nargs='?')
    parser_load.add_argument('--type', choices=['steady_heat', 'transient_heat', 'moisture'], help='Joe')
    parser_create.add_argument('--vis', choices=['loss', 'field', 'all'], help='Wie visualisieren? loss oder field', nargs='?')
    parser_create.add_argument('--type', choices=['steady_heat', 'transient_heat', 'moisture'], help='Joe')
    
    manage_args(parser.parse_args())


