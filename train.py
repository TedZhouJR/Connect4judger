import torch
from torch.utils.data import DataLoader
from data.data import Connect4Dataset, Connect4Collect
from opt import parse_args
from models.MLP import MLP

def get_model(args):
    if args.model == "mlp":
        return MLP(args.input_size, args.hidden_size, args.dropout, args.output_size)
    else:
        assert False

def get_optim(args, model):
    if args.optim == "adam":
        return torch.optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=args.lr)

def get_loss_func():
    return torch.nn.CrossEntropyLoss()

if __name__ == "__main__":
    args = parse_args()
    train_dataset = Connect4Dataset(args.train_input, args.train_output)
    val_dataset = Connect4Dataset(args.valid_input, args.valid_output)
    train_dataloader = DataLoader(train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=Connect4Collect())
    val_dataloader = DataLoader(val_dataset,
                        batch_size=args.batch_size,
                        collate_fn=Connect4Collect())
    model = get_model(args)
    model_optim = get_optim(args, model)
    loss_func = get_loss_func()

    for epoch in range(args.epochs):
        train_loss = []
        val_loss = []
        val_acc = []

        # train
        model.train()
        for batch_id, batch in enumerate(train_dataloader):
            src, tgt = batch
            y = model(src)
            loss = loss_func(y, tgt)
            model_optim.zero_grad()
            loss.backward()
            model_optim.step()
            train_loss.append(float(loss.data))
        train_loss = sum(train_loss) / len(train_loss)
        print("Epoch %d Training loss %.2f" % (epoch+1, train_loss))

        # validate
        model.eval()
        correct = 0
        total = 0
        for batch_id, batch in enumerate(val_dataloader):
            src, tgt = batch
            y = model(src)
            prediction = y.argmax(dim=-1)
            loss = loss_func(y, tgt)
            val_loss.append(float(loss.data))
            correct += int(sum(prediction==tgt))
            total += len(tgt)
        val_acc = correct / total
        val_loss = sum(val_loss) / len(val_loss)
        print("Validation loss %.2f, accuracy %.2f" % (val_loss, val_acc))

    torch.save(model, args.model_path)
