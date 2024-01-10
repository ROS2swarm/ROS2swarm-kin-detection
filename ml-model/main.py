from tqdm import tqdm
import yaml
from easydict import EasyDict
from RosSwLidDataset import RosSwLidDataset
import torch
from torch.nn import functional as F
import numpy as np
from torch.utils.data import DataLoader
from model import build_model

def train_epoch(model, datasetloader, optimizer):
    model.train()
    train_tqdm = tqdm(enumerate(datasetloader), desc="Train", total=len(datasetloader))
    losses = []; accs = []
    for i, batch in train_tqdm:
        pts, y = batch
        pts.cuda(); y.cuda()

        optimizer.zero_grad()

        out = model(pts.unsqueeze(1)).transpose(2, 1)
        # one_hot = torch.zeros_like(out).scatter(-1, y.unsqueeze(-1), 1)
        loss = F.cross_entropy(out.transpose(2, 1), y)
        # print(one_hot, out)
        # print(one_hot.shape, out.shape)
        loss.backward()
        optimizer.step()

        pred = F.log_softmax(out, dim=-1).argmax(-1)
        acc = (pred == y).to(torch.float32).mean()
        losses.append(loss.item()); accs.append(acc.item())
        train_tqdm.set_postfix_str(f"accuracy={accs[-1]:.4f}, loss={losses[-1]:.5f}")

    return np.mean(accs), np.mean(losses)

def val(model, val_loadeer):
    model.eval()
    val_tqdm = tqdm(enumerate(val_loadeer), desc='val', total=len(val_loadeer))
    accs = []

    for i, batch in val_tqdm:
        pts, y = batch
        pts.cuda(); y.cuda()
        out = model(pts.unsqueeze(1)).transpose(2, 1)

        pred = F.log_softmax(out, dim=-1).argmax(-1)
        acc = (pred == y).to(torch.float32).mean()
        accs.append(acc.item())

    return np.mean(accs)


def main():
    with open('config.yaml', 'r') as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    model = build_model(config.model)
    ds = RosSwLidDataset(config.dataset)
    ds_val = RosSwLidDataset(config.dataset, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.general.LR)
    dataloader = DataLoader(ds, batch_size=config.general.BATCH_SZ, shuffle=True, num_workers=config.general.NUM_DSWORK, persistent_workers=True)
    val_loader = DataLoader(ds_val, batch_size=config.general.BATCH_SZ, shuffle=True, num_workers=config.general.NUM_DSWORK, persistent_workers=True)
    train_accs = []
    val_accs = []

    for epoch in tqdm(range(config.general.EPOCHS)):
        acc, loss = train_epoch(model, dataloader, optimizer)
        train_accs.append(acc)
        print(epoch, acc, loss)
        acc = val(model, val_loader)
        val_accs.append(acc)
        print(acc)
        if epoch % 10 == 0:
            torch.save({
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, f"saved_models/{config.model.NAME}_{epoch}.pth")

    print(train_accs)
    print(val_accs)

if __name__ == "__main__":
    main()





