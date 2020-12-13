import torch
import torch.nn.functional as F
from opt import parse_args
from data.data import Connect4Dataset

def integrated_gradients(inputs, model, baseline, steps=50, cuda=False):
    gradients = []
    src, tgt = inputs
    for i, ip in enumerate(zip(src, tgt)):
        if i % 500 == 0:
            print("Processing", i, "of total", len(inputs[0]))
        observation, label = ip
        if cuda:
            baseline = baseline.cuda()
        scaled_inputs = [baseline + (float(i) / steps) * (observation - baseline) for i in range(0, steps + 1)]
        scaled_inputs = torch.stack(scaled_inputs)
        scale = observation - baseline
        scaled_inputs.requires_grad_(True)
        output = model(scaled_inputs)
        output = F.softmax(output, dim=-1)
        output = output[:, 1]
        ob_output = output[-1]
        if ob_output < 0.5:
            gradients.append(torch.zeros(observation.shape))
            continue
        output = output.mean()
        output.retain_grad()
        output.backward()
        gradient = scaled_inputs.grad.detach()
        gradient = gradient.sum(dim=0)
        gradient *= scale
        gradients.append(gradient.cpu())
    # gradients_all = []
    gradients = torch.stack(gradients)
    return gradients

def random_baseline_integrated_gradients(inputs, model, steps, cuda):
    baseline = torch.zeros(inputs[0].shape[-1]).long()
    integrated_grad = integrated_gradients(inputs, model, baseline, steps=steps, cuda=cuda)
    torch.save(integrated_grad.cpu(), args.grad_path)

if __name__ == "__main__":
    args = parse_args()
    testdataset = Connect4Dataset(args.test_input, args.test_output, args.gpu).data
    model = torch.load(args.model_path)
    src = testdataset["Source"]
    tgt = testdataset["Target"]
    random_baseline_integrated_gradients((src, tgt), model,
                                         steps=200,
                                         cuda=args.gpu)
