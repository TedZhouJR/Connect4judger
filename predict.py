from opt import parse_args
from data.data import Connect4Dataset, Connect4Collect
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    args = parse_args()
    model = torch.load(args.model_path)
    if args.gpu:
        model = model.cuda()
    test_dataset = Connect4Dataset(args.test_input, args.test_output, gpu=args.gpu)
    test_dataloader = DataLoader(test_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=Connect4Collect(fl=args.model!="attention"))
    correct = 0
    total = 0
    model.eval()
    for batch in test_dataloader:
        src, tgt = batch
        y = model(src)
        prediction = y.argmax(dim=-1)
        correct += int(sum(prediction==tgt))
        total += len(tgt)
    acc = 100 * correct / total
    print("Test accuracy %.2f" % acc)
