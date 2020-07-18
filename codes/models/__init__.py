import logging

import models.arch.RRDBNet_arch as RRDBNet_arch


logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    # image restoration
    if model == 'RRDB':  # RRDB
        from .RRDB_model import RRDBM as M

    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)


    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m





def define_SR(opt):
    opt_net = opt['network_G_SR']
    which_model = opt_net['which_model_G']

    if which_model == 'RRDBONet':
        netG = RRDBNet_arch.RRDBONet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])

    else:
        raise NotImplementedError('Generator models [{:s}] not recognized'.format(which_model))

    return netG



