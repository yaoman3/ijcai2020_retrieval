import os
import argparse
import numpy as np
import json
import random
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def parser():
    parser = argparse.ArgumentParser(description="2D to 3D evalutaion model.")
    # parser.add_argument("--image_dir", type=str, default='/home/public/IJCAI_2020_retrieval/train/image')
    parser.add_argument("--image_dir", type=str, default='./extract_workshop_baseline_notexture_2d_v1')
    parser.add_argument("--model_dir", type=str, default='./extract_workshop_baseline_notexture_3d_v1')
    parser.add_argument("--train_set", type=str, default='/home/public/IJCAI_2020_retrieval/train/data_info/train_set.json')
    parser.add_argument("--predict_out", type=str, default='retrieval_results.txt')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--nfeat", type=int, default=256)
    parser.add_argument("--ncate", type=int, default=7)
    parser.add_argument("--nviews", type=int, default=12)
    return parser

def load_shapes(model_dir, shapes):
    print("Load 3d features")
    shape_list = list()
    feat_bank = list()
    cate_bank = list()
    for i in range(len(shapes)):
        shape_id = shapes[i].split('/')[-1].split('_')[0]
        feat = np.load(shapes[i], allow_pickle=True).item()
        if len(shape_list)==0 or shape_id != shape_list[-1]:
            shape_list.append(shape_id)
        feat_bank.append(feat['feat_query'])
        cate_bank.append(feat['pred_cate'])
    return np.reshape(feat_bank, (len(shape_list), -1)), shape_list


def load_images(image_dir, images):
    print("Load images")
    image_feat = list()
    for i in range(len(images)):
        image = np.load(os.path.join(image_dir, images[i]), allow_pickle=True).item()
        image_feat.append(image['feat_query'])
    return np.reshape(image_feat, (len(images), -1))


class Mapper(nn.Module):
    def __init__(self, nfeat, nviews):
        super(Mapper, self).__init__()
        self.input_size = (nviews+1)*nfeat
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = torch.sigmoid(self.fc3(x))
        return x


class RetrievalDataset(Dataset):
    def __init__(self, image_feats, shape_feats, shape_list, train_set=None, negative_size=2, offset=0):
        self.image_feats = image_feats
        self.shape_feats = shape_feats
        self.train_set = train_set
        self.sample_size = negative_size + 1
        self.shape_index = {}
        for i in range(len(shape_list)):
            self.shape_index[shape_list[i]] = i
        if train_set is not None:
            with open(train_set, 'r') as f:
                train_info = json.load(f)
                self.train_set = sorted(train_info, key=lambda x: x['image'])
        else:
            self.train_set = None
        self.offset = offset


    def __len__(self):
        if self.train_set is None:
            return len(self.image_feats)*len(self.shape_feats)
        else:
            return len(self.image_feats)*self.sample_size


    def __getitem__(self, idx):
        if self.train_set is None:
            k = idx // len(self.shape_feats)
        else:
            k = idx // self.sample_size
        x = self.image_feats[k]
        k += self.offset
        if self.train_set is not None and self.offset==0:
            if idx % self.sample_size == 0:
                return (np.concatenate((x, self.shape_feats[self.shape_index[self.train_set[k]['model']]]), axis=0), 1.0)
            else:
                while True:
                    i = random.choice(range(len(self.shape_feats)))
                    if self.train_set is None or i != self.shape_index[self.train_set[k]['model']]:
                        return (np.concatenate((x, self.shape_feats[i]), axis=0), 0.0)
        elif self.train_set is not None:
            i = idx % len(self.shape_feats)
            if i == self.shape_index[self.train_set[k]['model']]:
                return (np.concatenate((x, self.shape_feats[i]), axis=0), 1.0)
            else:
                return (np.concatenate((x, self.shape_feats[i]), axis=0), 0.0)
        else:
            i = idx % len(self.shape_feats)
            return np.concatenate((x, self.shape_feats[i]), axis=0)


if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    model = Mapper(args.nfeat, args.nviews)
    if args.train:
        train_set = args.train_set
    else:
        train_set = None
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = model.to(device)
    if args.train:
        images = os.listdir(args.image_dir)
        images.sort()
        shapes = sorted(glob.glob(args.model_dir + '/*.npy'))
        shape_feats, shape_list = load_shapes(args.model_dir, shapes)
        image_feats = load_images(args.image_dir, images)
        val_offset = int(len(images)*0.8)
        train_feats = image_feats[:val_offset]
        val_feats = image_feats[val_offset:]
        train_data = RetrievalDataset(train_feats, shape_feats, shape_list, train_set)
        trainloader = DataLoader(train_data, batch_size=1000, shuffle=args.train)
        val_data = RetrievalDataset(val_feats, shape_feats, shape_list, train_set, offset=val_offset)
        valloader = DataLoader(val_data, batch_size=len(shape_feats))
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        model.train()
        min_loss = float('inf')
        for epoch in range(30):
            running_loss = 0.0
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.float().to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 10 == 9:
                    model.eval()
                    val_loss = 0.0
                    cnt = 0
                    for tmp in valloader:
                        inputs, labels = tmp
                        inputs = inputs.to(device)
                        labels = labels.float().to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels.view(-1, 1))
                        val_loss += loss.item()
                        cnt += 1
                    print('[%d, %d] train loss: %.5f, val loss: %.5f' % (epoch+1, i+1, running_loss/100, val_loss/cnt))
                    if val_loss<min_loss:
                        min_loss = val_loss
                        torch.save(model.state_dict(), 'checkpoints/eval_models.pth')
                    running_loss = 0.0
                    model.train()
    else:
        images = os.listdir(args.image_dir)
        images.sort()
        shapes = sorted(glob.glob(args.model_dir + '/*.npy'))
        shape_feats, shape_list = load_shapes(args.model_dir, shapes)
        image_feats = load_images(args.image_dir, images)
        dataset = RetrievalDataset(image_feats, shape_feats, shape_list, train_set)
        dataloader = DataLoader(dataset, batch_size=1000, shuffle=args.train)
        model.load_state_dict(torch.load('checkpoints/eval_models.pth'))
        model.eval()
        images = os.listdir(args.image_dir)
        images.sort()
        results = []
        for data in tqdm(dataloader):
            inputs = data.to(device)
            outputs = model(inputs)
            results.append(np.reshape(outputs.cpu().detach().numpy(), -1))
        results = np.concatenate(results, axis=0)
        results = 1 - np.reshape(results, (len(images), -1))
        with open(args.predict_out, 'w') as f:
            for i in tqdm(range(len(images))):
                line = images[i].split('.')[-2] + ': '
                for j in range(results.shape[1]):
                    if j>0:
                        line += ', '
                    line += "%0.3f" % results[i,j]
                line += '\n'
                f.write(line)