import torch.optim as Opt

def get_optimizer(model, args):

    # classes, not instances: the dict-of-instances form built all three
    # optimizers on every call and, on an unknown name, returned the string
    # "Invalid Optimizer" that then blew up at optimizer.param_groups.
    opt_fns = {
        'adam': Opt.Adam,
        'sgd': Opt.SGD,
        'adagrad': Opt.Adagrad,
    }
    if args.optimizer not in opt_fns:
        raise ValueError(
            "Unknown optimizer '{}'; choose from {}".format(
                args.optimizer, sorted(opt_fns)))

    return opt_fns[args.optimizer](model.parameters(), lr=args.lr_start)
