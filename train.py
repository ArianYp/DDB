import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torchvision
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from langsegmentanything.lang_sam import LangSAM
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from captum.attr import IntegratedGradients
import torch.nn.functional as F
from data import Waterbirds, CelebA, Metashift
import random
import os
import torch.optim as optim
import copy

dataset_map = {
    'celeba': (CelebA.get_celeba_loaders, CelebA.get_transform_celeba, CelebA.get_celeba_dataset),
    'waterbirds': (Waterbirds.get_waterbird_dataloaders, Waterbirds.get_transform_waterbirds, Waterbirds.get_waterbird_dataset),
    'metashift': (Metashift.get_metashift_loaders, Metashift.get_transform_metashift, Metashift.get_metashift_dataset)
}


np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=False)
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, 2)
    
    def forward (self, X):
        X = self.model(X)
        return X

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def train_cnn(dataset, model, opt, step,args):
    """
    Training invariant model on MNIST.
    """
    criterion = nn.CrossEntropyLoss()

    opt = opt
    avg_inv_acc = 0
    avg_inv_loss = 0
    avg_var_loss = 0
    count = 0

    num_sample = 0

    model.train()

    for (batch, (inputs, labels, envs)) in enumerate(tqdm(dataset)):
        count +=1

        inputs = inputs.to(device)
        labels = labels.to(device)
        envs = envs.to(device)

        opt.zero_grad()

        out = model(inputs)


        total_loss = criterion (out, labels)

        total_loss.backward()

        opt.step()

        avg_inv_loss += total_loss

        avg_inv_acc += torch.sum(torch.argmax(out, dim=1)==torch.argmax(labels, dim=1))

    # results
    avg_inv_acc = avg_inv_acc/(count*args['batch_size'])
    avg_inv_loss = avg_inv_loss/(count*args['batch_size'])

    print("{:s}{:d}: {:s}{:.4f}, {:s}{:.4f}.".format(
        "----> [Train] Total iteration #", step, "inv acc: ",
        avg_inv_acc, "inv loss: ", avg_inv_loss),
          flush=True)

    # scheduler.step()

    return step+1

def weight_init(m):
    """
    Initialize the weights of a given module.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def test_cnn(dataset, model, number_of_envs=4):
    """
    Conventional testing of a classifier.
    """
    avg_inv_acc = 0
    count = 0


    corrects_envs = [0]*number_of_envs
    totals_envs = [0] *number_of_envs
    avg_acc_envs = [0] *number_of_envs

    model.eval()
    for (batch, (inputs, labels, envs)) in enumerate(tqdm(dataset)):
        count+=1
        inputs = inputs.to(device)
        labels = labels.to(device)
        envs = envs.to(device)

        logits = model(inputs)

        for env_num in range(number_of_envs):
            logits_env = logits[envs[:,env_num]==1]
            labels_env = labels[envs[:,env_num]==1]
            corrects_envs[env_num] += torch.sum(torch.argmax(logits_env, dim=1) == torch.argmax(labels_env, dim=1)).item()
            totals_envs[env_num] += len(logits_env)

    all_correct = 0
    all_totals = 0
    for env_num in range(number_of_envs):
        avg_acc_envs[env_num] = round(corrects_envs[env_num] / totals_envs[env_num], 4)
        print(f"env {env_num}, acc: {avg_acc_envs[env_num]}")
        all_correct += corrects_envs[env_num]
        all_totals += totals_envs[env_num]
    avg_inv_acc = round(all_correct / all_totals, 6)

    print(f"all envs mean acc: {avg_inv_acc}")

    return avg_inv_acc, avg_acc_envs



class RetrainerDataset(Dataset):
    def __init__(self, transform, path ):

        self.dataset_dir= "./"
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f'{self.dataset_dir} does not exist yet. Please generate the dataset first.')
        self.metadata_df = pd.read_csv(
            os.path.join(self.dataset_dir, path))
        
        self.env_dict = {
            (0, 0): torch.Tensor(np.array([2,0,0,0])),
            (0, 1): torch.Tensor(np.array([0,2,0,0])),
            (1, 0): torch.Tensor(np.array([0,0,2,0])),
            (1, 1): torch.Tensor(np.array([0,0,0,2])),
        }
        y_array = torch.Tensor(np.array(self.metadata_df['label'].values)).type(torch.LongTensor)
        
        self.y_array = self.metadata_df['label'].values
        self.filename_array = self.metadata_df['filename'].values
        self.env_array = self.metadata_df['place'].values
        self.transform = transform

        self.y_one_hot = nn.functional.one_hot(y_array, num_classes=2).type(torch.FloatTensor)
        
    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
       
        img_filename = os.path.join(
            self.dataset_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)
        place = self.env_array[idx]
        y = self.y_array[idx]

        label = self.y_one_hot[idx]

        return img, label, self.env_dict[(y, place)]
    


def retrain_cnn(dataloader, model, optimizer, step,args, device):
    criterion = nn.CrossEntropyLoss()
    model.train()
    corrects_envs = [0]*5
    totals_envs = [0] *5
    avg_acc_envs = [0] *5

    for inputs, labels, envs in tqdm(dataloader):
        inputs, labels, envs = inputs.to(device), labels.to(device), envs.to(device)
        optimizer.zero_grad()
        mask_env2_col2 = (envs[:, 3] == 2) 
        mask_env2_col1 = (envs[:, 1] == 2)  
        out = model(inputs)
        out_col2 = out[mask_env2_col2]
        out_col1 = out[mask_env2_col1]
        labels_col2 = labels[mask_env2_col2]
        labels_col1 = labels[mask_env2_col1]
        mask_other = ~(mask_env2_col2 | mask_env2_col1)
        out_other = out[mask_other]
        labels_other = labels[mask_other]
        loss_col2 = args.gamma2 * criterion(out_col2, labels_col2)
        loss_col1 = args.gamma1 * criterion(out_col1, labels_col1)  
        loss_other = criterion(out_other, labels_other)
        total_loss = loss_col2 + loss_col1 + loss_other
        total_loss.backward()
        optimizer.step()
        for env_num in range(4):
            logits_env = out[envs[:,env_num]==1]
            logits_env2 = out[envs[:,env_num]==2]
            labels_env = labels[envs[:,env_num]==1 ]
            labels_env2 = labels[envs[:,env_num]==2 ]
            corrects_envs[env_num] += torch.sum(torch.argmax(logits_env, dim=1) == torch.argmax(labels_env, dim=1)).item() 
            corrects_envs[4] += torch.sum(torch.argmax(logits_env2, dim=1) == torch.argmax(labels_env2, dim=1)).item()
            totals_envs[env_num] += len(logits_env)
            totals_envs[4] += len(logits_env2)
    all_correct = 0
    all_totals = 0
    for env_num in range(5):
        avg_acc_envs[env_num] = round(corrects_envs[env_num] / totals_envs[env_num], 4)
        print(f"env {env_num}, acc: {avg_acc_envs[env_num]}")
        all_correct += corrects_envs[env_num]
        all_totals += totals_envs[env_num]
    avg_inv_acc_n = round(all_correct / all_totals, 6)


def run_retraining(args):
    get_loaders, get_transform, get_dataset = dataset_map[args.dataset]
    trainloader, valloader, testloader = get_loaders(f'data/{args.dataset}',args.batch_size) 
    args.transform=  get_transform(True)
    traindset = get_dataset('train', f'data/{args.dataset}', transform=args.transform)
    cnn_image_encoder = ResNet50().to(device)
    cnn_image_encoder.load_state_dict(torch.load(args.model_path))
    number_of_envs = 2 if args.dataset == 'metashift' else 4
    
    test_cnn(testloader, cnn_image_encoder, number_of_envs)
    for n,p in cnn_image_encoder.named_parameters():
        p.requires_grad = True
    weight_init(cnn_image_encoder.model.fc)
    for p in cnn_image_encoder.model.fc.parameters():
        p.requires_grad = True
    retrainerdset = RetrainerDataset(args.transform, f'{args.new_data}/metadata_cls0.csv')
    retrainerdset2 = RetrainerDataset(args.transform, f'{args.new_data}/metadata_cls1.csv')
    merged_dataset = ConcatDataset([retrainerdset,  traindset, retrainerdset2])
    merged_dataloader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(cnn_image_encoder.parameters(), lr= args.lr )
    best_acc = 0
    print('learning rate:', args.lr)
    for i in range(args.epochs):
        retrain_cnn(merged_dataloader, cnn_image_encoder, opt,i,args,'cuda')
        test_cnn(testloader, cnn_image_encoder, number_of_envs)
        _, acc = test_cnn(valloader, cnn_image_encoder, number_of_envs)
        
        acc = min(acc)
        if acc > best_acc:
            best_acc = acc
            print("New best acc:", acc)
            torch.save(cnn_image_encoder.state_dict(), os.path.join(args.output_dir, f"best_model.pth"))

    cnn_image_encoder.load_state_dict(torch.load(os.path.join(args.output_dir, f"best_model.pth")))
    cnn_image_encoder.device = device
    cnn_image_encoder.eval()
    test_cnn(testloader, cnn_image_encoder, number_of_envs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retraining with synthetic and original data")
    parser.add_argument("--new_data", type=str, default='new_data', help="Path generated data")
    parser.add_argument('--dataset', type=str, default='waterbirds', choices=['celeba', 'waterbirds', 'metashift'])
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for retraining")
    parser.add_argument("--gamma1", type=int, default=6)
    parser.add_argument("--gamma2", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to retrain")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument('--model_path', type=str, default='../DaC/Copy of 100resnet50_erm_ll.model')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_retraining(args)
    