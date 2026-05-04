import argparse
import yaml
import torch


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value
            

config_parser = argparse.ArgumentParser(description='Training Config', add_help=False)
config_parser.add_argument('-c', '--config', default='', type=str,
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='ImageNet Training')

parser.add_argument('--mode', type=str, default='multimodal', choices=['text', 'image', 'multimodal'],
                    help='Training mode: text, image, or multimodal')
parser.add_argument('--datapath', type=str, default='data/', 
                    help='Path to dataset')
parser.add_argument('--text_model', type=str, default='bert-base-uncased', 
                    help='Pretrained text model name, e.g. bert-base-uncased or roberta-base')
parser.add_argument('--image_model', type=str, default='resnet18', 
                    help='Pretrained image model name, e.g. resnet18, mobilenetv1, efficientnet_b0')
parser.add_argument('--num_classes', type=int, default=3, 
                    help='Number of output classes')
parser.add_argument('--batch_size', type=int, default=32, 
                    help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=3, 
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-5, 
                    help='Learning rate for optimizer')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                    help='Device to use for training (cuda or cpu)')
parser.add_argument('--kwargs', nargs='*', action=ParseKwargs,
                    help='Additional keyword arguments as key=value pairs')
parser.add_argument('--opt', type=str, default='sgd', 
                    help='Optimizer to use (sgd, adam, etc.)')
parser.add_argument('--momentum', type=float, default=0.9, 
                    help='Momentum for SGD optimizer')


def parse_args():
    # First parse the config file if provided
    args, remaining_argv = config_parser.parse_known_args()
    if args.config:
        with open(args.config, 'r') as f:
            config_args = yaml.safe_load(f)
        parser.set_defaults(**config_args)

    # Now parse the command line arguments, which will override config file values
    final_args = parser.parse_args(remaining_argv)
    return final_args