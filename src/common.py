import argparse

DATA_PATH = './dataset'
CKPT_PATH = './checkpoint'

def parse_args(mode):
    assert mode in ['train', 'eval']

    # ========== Common ========== #
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help='dataset (news|review|imdb|etc.)',
                        required=True, type=str)
    parser.add_argument("--backbone", help='backbone network',
                        choices=['bert', 'roberta', 'roberta_large', 'roberta_mc', 'roberta_mc_large', 'albert'],
                        default='bert', type=str)
    parser.add_argument("--seed", help='random seed',
                        default=0, type=int)

    parser = _parse_args_train(parser)

    return parser.parse_args()


def _parse_args_train(parser):
    # ========== Training ========== #
    parser.add_argument("--train_type", help='train type',
                        default='base', type=str)
    parser.add_argument("--epochs", help='training epochs',
                        default=7, type=int)
    parser.add_argument("--batch_size", help='training bacth size',
                        default=16, type=int)
    parser.add_argument("--model_lr", help='learning rate for model update',
                        default=1e-5, type=float)
    parser.add_argument("--save_ckpt", help='save the best model checkpoint',
                        action='store_true')
    parser.add_argument("--grad_accumulation", help='inverval for model update',
                        default=1, type=int)

    parser.add_argument("--pre_ckpt", help='path for the pre-trained model',
                        default=None, type=str)
    parser.add_argument("--selected", help='path to selected subset of augmented samples',
                        default=None, type=str)
    
    # ========== InfoVerse ========== #
    parser.add_argument('--seed_list', help='delimited list input', type=str)
    parser.add_argument("--info_path", help='path to pre-generated infoverse file',
                        default=None, type=str) 
    
    # ========== Data Pruning ========== #
    parser.add_argument("--data_ratio", help='data ratio',
                        default=1.0, type=float)
    
    # ========== Data Annotation ========== #
    parser.add_argument("--annotation", help='annotation type',
                        choices=['random', 'uncertain', 'infoverse', None],
                        default=None, type=str)

    return parser

