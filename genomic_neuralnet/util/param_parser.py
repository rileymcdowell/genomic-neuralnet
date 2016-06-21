from __future__ import print_function
from argparse import ArgumentParser
from genomic_neuralnet.config import data

_parser = ArgumentParser()
_parser.add_argument('-s', '--species', default='arabidopsis', choices=data.keys())
_parser.add_argument('-t', '--trait', default='flowering')
_parser.add_argument('-l', '--list', action='store_true', help='Print a list of traits for the chosen species and exit.')
_parser.add_argument('-v', '--verbose', action='store_true')

_arguments = None
def get_arguments():
    global _arguments
    if _arguments is None:
        _arguments = _parser.parse_args()
        handle_list_option(_arguments)
        allowed_traits = data[_arguments.species].keys()
        if not _arguments.trait in allowed_traits:
            msg = 'Trait not found. Expected one of {}.'
            print(msg.format(', '.format(allowed_traits)))
    return _arguments

def handle_list_option(args):
    if args.list:
        species, _ = get_species_and_trait()
        print(' '.join(data[species].keys()))
        exit()
    else:
        return

def get_markers_and_pheno():
    species, trait = get_species_and_trait()
    markers = data[species][trait].markers
    pheno = data[species][trait].pheno
    return markers, pheno

def get_species_and_trait():
    args = get_arguments()
    return args.species, args.trait

def get_verbose():
    args = get_arguments()
    return args.verbose
