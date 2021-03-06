import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Classify connect-4 states.')
    # data
    parser.add_argument('--input', type=str, default="data/database.txt",
                        help='Input text file path.')
    parser.add_argument('--train_input', type=str, default="data/train.src.txt",
                        help='Input train text file path.')
    parser.add_argument('--train_output', type=str, default="data/train.tgt.txt",
                        help='Output train text file path.')
    parser.add_argument('--valid_input', type=str, default="data/valid.src.txt",
                        help='Input valid text file path.')
    parser.add_argument('--valid_output', type=str, default="data/valid.tgt.txt",
                        help='Output valid text file path.')
    parser.add_argument('--test_input', type=str, default="data/test.src.txt",
                        help='Input test text file path.')
    parser.add_argument('--test_output', type=str, default="data/test.out.txt",
                        help='Output test text file path.')
    # model
    parser.add_argument('--model', type=str, default="linear",
                        help='Model type, currently support mlp, attention, linear.')
    parser.add_argument('--input_size', type=int, default=42,
                        help='Size of input.')
    parser.add_argument('--output_size', type=int, default=3,
                        help='Size of output.')
    parser.add_argument('--hidden_size', type=list, default=[256, 64],
                        help='Size of hidden layers.')
    parser.add_argument('--layers', type=int, default=5,
                        help='Number of hidden layers.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout ratio.')
    # train
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--gpu', type=bool, default=True,
                        help='Use gpu.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Learning rate.')
    parser.add_argument('--optim', type=str, default="rmsprop",
                        help='Optimizer type, currently support adam, sgd, rmsprop.')
    parser.add_argument('--model_path', type=str, default="result/params.pt",
                        help='Path to save or load model.')
    parser.add_argument('--grad_path', type=str, default="result/grad.pt",
                        help='Path to save or load model.')

    args = parser.parse_args()
    return args