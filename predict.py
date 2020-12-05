from opt import parse_args
from data.data import Connect4Dataset, Connect4Collect
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    args = parse_args()
    model = torch.load(args.model_path)
    test_dataset = Connect4Dataset(args.test_input, args.test_output)
    test_dataloader = DataLoader(test_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        collate_fn=Connect4Collect())
    correct = 0
    total = 0
    for batch in test_dataloader:
        src, tgt = batch
        y = model(src)
        prediction = y.argmax(dim=-1)
        correct += int(sum(prediction==tgt))
        total += len(tgt)
    acc = correct/total
    print("Test accuracy %.2f" % acc)
