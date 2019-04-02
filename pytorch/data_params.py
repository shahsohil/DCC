from easydict import EasyDict as edict
from extractSDAE import extractSDAE
from extractconvSDAE import extractconvSDAE
from SDAE import SDAE
from convSDAE import convSDAE

easy = edict()
easy.name = 'easy'
easy.dim = [2]


# TODO port other dataset's parameters to here
def extract_convsdae_mnist(slope=0.0):
    return extractconvSDAE(dim=[1, 50, 50, 50, 10], output_padding=[0, 1, 0], numpen=4, slope=slope)


def extract_convsdae_coil100(slope=0.0):
    return extractconvSDAE(dim=[3, 50, 50, 50, 50, 50, 10], output_padding=[0, 1, 1, 1, 1], numpen=4, slope=slope)


def extract_convsdae_ytf(slope=0.0):
    return extractconvSDAE(dim=[3, 50, 50, 50, 50, 10], output_padding=[1, 0, 1, 0], numpen=4, slope=slope)


def extract_convsdae_yale(slope=0.0):
    return extractconvSDAE(dim=[1, 50, 50, 50, 50, 50, 10], output_padding=[(0, 0), (1, 1), (1, 1), (0, 1), (0, 1)],
                           numpen=6,
                           slope=slope)


def extract_sdae_mnist(slope=0.0, dim=10):
    return extractSDAE(dim=[784, 500, 500, 2000, dim], slope=slope)


def extract_sdae_reuters(slope=0.0, dim=10):
    return extractSDAE(dim=[2000, 500, 500, 2000, dim], slope=slope)


def extract_sdae_ytf(slope=0.0, dim=10):
    return extractSDAE(dim=[9075, 500, 500, 2000, dim], slope=slope)


def extract_sdae_coil100(slope=0.0, dim=10):
    return extractSDAE(dim=[49152, 500, 500, 2000, dim], slope=slope)


def extract_sdae_yale(slope=0.0, dim=10):
    return extractSDAE(dim=[32256, 500, 500, 2000, dim], slope=slope)


def extract_sdae_easy(slope=0.0, dim=1):
    return extractSDAE(dim=easy.dim + [dim], slope=slope)


def sdae_mnist(dropout=0.2, slope=0.0, dim=10):
    return SDAE(dim=[784, 500, 500, 2000, dim], dropout=dropout, slope=slope)


def sdae_reuters(dropout=0.2, slope=0.0, dim=10):
    return SDAE(dim=[2000, 500, 500, 2000, dim], dropout=dropout, slope=slope)


def sdae_ytf(dropout=0.2, slope=0.0, dim=10):
    return SDAE(dim=[9075, 500, 500, 2000, dim], dropout=dropout, slope=slope)


def sdae_coil100(dropout=0.2, slope=0.0, dim=10):
    return SDAE(dim=[49152, 500, 500, 2000, dim], dropout=dropout, slope=slope)


def sdae_yale(dropout=0.2, slope=0.0, dim=10):
    return SDAE(dim=[32256, 500, 500, 2000, dim], dropout=dropout, slope=slope)


def sdae_easy(dropout=0.2, slope=0.0, dim=1):
    return SDAE(dim=easy.dim + [dim], dropout=dropout, slope=slope)


def convsdae_mnist(dropout=0.2, slope=0.0):
    return convSDAE(dim=[1, 50, 50, 50, 10], output_padding=[0, 1, 0], numpen=4, dropout=dropout, slope=slope)


def convsdae_coil100(dropout=0.2, slope=0.0):
    return convSDAE(dim=[3, 50, 50, 50, 50, 50, 10], output_padding=[0, 1, 1, 1, 1], numpen=4, dropout=dropout,
                    slope=slope)


def convsdae_ytf(dropout=0.2, slope=0.0):
    return convSDAE(dim=[3, 50, 50, 50, 50, 10], output_padding=[1, 0, 1, 0], numpen=4, dropout=dropout, slope=slope)


def convsdae_yale(dropout=0.2, slope=0.0):
    return convSDAE(dim=[1, 50, 50, 50, 50, 50, 10], output_padding=[(0, 0), (1, 1), (1, 1), (0, 1), (0, 1)], numpen=6,
                    dropout=dropout, slope=slope)


def load_predefined_net(args, params):
    if args.db == 'mnist':
        net = sdae_mnist(dropout=params['dropout'], slope=params['reluslope'], dim=args.dim)
    elif args.db == 'reuters' or args.db == 'rcv1':
        net = sdae_reuters(dropout=params['dropout'], slope=params['reluslope'], dim=args.dim)
    elif args.db == 'ytf':
        net = sdae_ytf(dropout=params['dropout'], slope=params['reluslope'], dim=args.dim)
    elif args.db == 'coil100':
        net = sdae_coil100(dropout=params['dropout'], slope=params['reluslope'], dim=args.dim)
    elif args.db == 'yale':
        net = sdae_yale(dropout=params['dropout'], slope=params['reluslope'], dim=args.dim)
    elif args.db == 'cmnist':
        net = convsdae_mnist(dropout=params['dropout'], slope=params['reluslope'])
    elif args.db == 'ccoil100':
        net = convsdae_coil100(dropout=params['dropout'], slope=params['reluslope'])
    elif args.db == 'cytf':
        net = convsdae_ytf(dropout=params['dropout'], slope=params['reluslope'])
    elif args.db == 'cyale':
        net = convsdae_yale(dropout=params['dropout'], slope=params['reluslope'])
    elif args.db == 'easy':
        net = sdae_easy(dropout=params['dropout'], slope=params['reluslope'], dim=args.dim)
    else:
        raise ValueError("Unexpected database %s" % args.db)

    return net


def load_predefined_extract_net(args):
    reluslope = 0.0

    if args.db == 'mnist':
        net = extract_sdae_mnist(slope=reluslope, dim=args.dim)
    elif args.db == 'reuters' or args.db == 'rcv1':
        net = extract_sdae_reuters(slope=reluslope, dim=args.dim)
    elif args.db == 'ytf':
        net = extract_sdae_ytf(slope=reluslope, dim=args.dim)
    elif args.db == 'coil100':
        net = extract_sdae_coil100(slope=reluslope, dim=args.dim)
    elif args.db == 'yale':
        net = extract_sdae_yale(slope=reluslope, dim=args.dim)
    elif args.db == 'cmnist':
        net = extract_convsdae_mnist(slope=reluslope)
    elif args.db == 'ccoil100':
        net = extract_convsdae_coil100(slope=reluslope)
    elif args.db == 'cytf':
        net = extract_convsdae_ytf(slope=reluslope)
    elif args.db == 'cyale':
        net = extract_convsdae_yale(slope=reluslope)
    elif args.db == easy.name:
        net = extract_sdae_easy(slope=reluslope, dim=args.dim)
    else:
        raise ValueError("Unexpected database %s" % args.db)

    return net
