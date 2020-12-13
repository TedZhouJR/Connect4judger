import torch, random
from opt import parse_args
lookup_dict = {"x":0, "o":2, "b":1, "-":0, "+":1, "=":2}

class Connect4Dataset(torch.utils.data.Dataset):
    def __init__(self, ipf, opf, gpu=False):
        with open(ipf) as input_f:
            inputs = input_f.readlines()
            inputs = [line.strip() for line in inputs]
        with open(opf) as output_f:
            outputs = output_f.readlines()
            outputs = [line.strip() for line in outputs]
        assert len(inputs) == len(outputs)
        self.data_size = len(inputs)
        src, tgt = self.lookup(inputs, outputs)
        if gpu:
            src = src.cuda()
            tgt = tgt.cuda()
        self.data = {"Source": src, "Target": tgt}

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        src = self.data['Source'][idx]
        tgt = self.data['Target'][idx]
        sample = {'Source': src, 'Target': tgt}
        return sample

    def lookup(self, inputs, outputs):
        src = []
        tgt = []
        for inp in inputs:
            src_tmp = [lookup_dict[i] == 0 for i in inp]
            src_tmp2 = [lookup_dict[i] == 2 for i in inp]
            src_tmp.extend(src_tmp2)
            src_tmp = torch.LongTensor(src_tmp)
            src.append(src_tmp)
        for out in outputs:
            tgt_tmp = [lookup_dict[out]]
            tgt_tmp = torch.LongTensor(tgt_tmp)
            tgt.append(tgt_tmp)
        src = torch.stack(src)
        tgt = torch.cat(tgt)
        return src, tgt

class Connect4Collect:
    def __init__(self, fl=True):
        self.fl = fl
        pass

    def __call__(self, batch):
        src = torch.stack([x['Source'] for x in batch])
        if self.fl:
            src = src.float()
        tgt = torch.stack([x['Target'] for x in batch])
        return src, tgt

def split_data(args, shuffle=True, train_ratio=0.8, val_ratio=0.1):
    with open(args.input) as f:
        lines = f.readlines()
        lines = [(line[:-2], line[-2]) for line in lines]
    if shuffle:
        random.shuffle(lines)
    d_size = len(lines)
    train_size = int(train_ratio*d_size)
    val_size = int(val_ratio*d_size)
    train_d = lines[:train_size]
    val_d = lines[train_size:train_size+val_size]
    test_d = lines[train_size+val_size:]
    for ipf, opf, data in [(args.train_input, args.train_output, train_d),
                     (args.valid_input, args.valid_output, val_d),
                     (args.test_input, args.test_output, test_d)]:
        with open(ipf, "w") as f:
            for d in data:
                f.write(d[0]+"\n")
        with open(opf, "w") as f:
            for d in data:
                f.write(d[1]+"\n")

if __name__ == "__main__":
    args = parse_args()

    # split raw data to train/val/test set
    split_data(args)
