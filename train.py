from data.data import Connect4Dataset
from opt import parse_args
from models.MLP import MLP

def get_model(args):
    if args.model == "mlp":
        return MLP(args.input_size, args.hidden_size, args.dropout, args.output_size)
    else:
        assert False

if __name__ == "__main__":
    args = parse_args()
    train_dataset = Connect4Dataset(args.train_input, args.train_output)
    val_dataset = Connect4Dataset(args.valid_input, args.valid_output)
    model = get_model(args)

    for epoch in args.epochs:
        pass
