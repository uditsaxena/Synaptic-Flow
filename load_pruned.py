import torch

from Utils import load

'''
'--dataset', type=str, default='mnist',
                               choices=['mnist', 'cifar10', 'cifar100', 'tiny-imagenet', 'imagenet'],
                               help='dataset (default: mnist)')
'--model-class', type=str, default='default',
                               choices=['default', 'lottery', 'tinyimagenet', 'imagenet'],
                               help='model class (default: default)'
'--model', type=str, default='fc', choices=['fc', 'conv', 'strconv', ... ] 
'--dense-classifier', type=bool, default=False,
                               help='ensure last layer of model is dense (default: False)'                              
'''


def get_model(dataset, model_class, model_name, pruner="synflow", epoch=0, custom_path="", dense_classifier=False):
    ## Data ##
    print('Loading {} dataset.'.format(dataset))
    input_shape, num_classes = load.dimension(dataset)

    ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(model_class, model_name))
    model = load.model(model_name, model_class)(input_shape, num_classes, dense_classifier, pretrained=False)

    pruned_path = "../Results/pruned/%s/%s/%s/%s_prune.pth" % (model_class, model_name, pruner, epoch,)
    if len(custom_path) != 0:
        pruned_path = custom_path
    print("Loading model from: %s" % (pruned_path))

    model.load_state_dict(torch.load(pruned_path))
    return model

