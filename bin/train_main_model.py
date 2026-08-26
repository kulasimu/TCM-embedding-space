import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import pickle
import numpy as np
from torchmetrics.classification import BinaryJaccardIndex
from model import TransformerContra, ContrastiveLoss
import random as Random
import sys
import matplotlib.pyplot as plt
import os


try:
    data_dir = sys.argv[1]  # dataset directory
    save_dir = sys.argv[2]  # model saving directory
    show_loss = sys.argv[3]  # Whether to show the loss curve during training
    BETA = float(sys.argv[4])   # Weight trade-off between 2 parts of loss
    TRAIN_MODE = sys.argv[5]
except:
    data_dir = 'data/training'
    save_dir = 'core/trained_model'
    show_loss = 'Y'
    BETA = 1.5
    TRAIN_MODE = "full"

print(data_dir, save_dir)
if TRAIN_MODE not in ["full", "independent"]:
    raise ValueError("TRAIN_MODE must be 'full' or 'independent'.")
print("Contrastive loss weight BETA:", BETA)
print("Training mode:", TRAIN_MODE)

######## Dataset for training and validation
with open(data_dir + '/train_dataloader.pkl','rb') as fp:
    train_loader = pickle.load(fp)
with open(data_dir + '/val_dataloader.pkl','rb') as fp:
    val_loader = pickle.load(fp)
with open('core/standard_TCM_entities/symptom_semantic_encodings.pkl','rb') as fp:
    symptom_semantics = pickle.load(fp)


######## Configurations
num_herbs = 780  # Number of herbs in the TCM data
num_sym = 1436  # Number of symptoms in the TCM data
batch_size = 256
n_herb_seq = 30   # Max number of herbs in a formula
n_sym_seq = 40  # Max number of symptoms in a symptom pattern
MAX_HERB_LEN = n_herb_seq + 1
MAX_SYM_LEN = n_sym_seq + 1
VOCAB_SIZE_HERB = num_herbs + 3  # Herbs and <pad>, <S>, <E>
VOCAB_SIZE_SYM = num_sym + 3  # Symptoms and <pad>, <S>, <E>
SYM_EMB_D = 512  # Symptom semantic vector dimension
NUM_EPOCHS = 300  # Max training epochs
HIDDEN_SIZE = 128  # Hidden dimension of the attention block
EMBEDDING_DIM = 256  # Dimension of the embeddings
NUM_HEADS = 8  # Number of attention heads
NUM_LAYERS = 4  # Number of attention blocks
DROPOUT = 0.6  # Drop rate
LEARNING_RATE = 0.0001  # Learning rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'
# BETA = 1.5
print('Run on:', DEVICE)


######## Set up
herb_pad_idx = int(num_herbs)
sym_pad = torch.FloatTensor(symptom_semantics[num_sym]).to(DEVICE)
cos_simi = nn.CosineSimilarity()
mse = torch.nn.MSELoss(reduction='none')
c_loss = ContrastiveLoss()
criterion = nn.CrossEntropyLoss()

######## Model
transformer_contrastive = TransformerContra(SYM_EMB_D,VOCAB_SIZE_HERB, EMBEDDING_DIM, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE_SYM,VOCAB_SIZE_HERB, DROPOUT,sym_pad=sym_pad,herb_pad_idx=herb_pad_idx,device=DEVICE).to(DEVICE)
print(transformer_contrastive)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer_contrastive.parameters(), lr=LEARNING_RATE)


######## Loss calculation
def batch_jaccard(a,b):
    jaccard = BinaryJaccardIndex().to(DEVICE)
    return torch.FloatTensor([jaccard(a[i],b[i]) for i in range(a.shape[0])]).to(DEVICE)


def loss_function(y1,y11_,y12_,y2, y21_,y22_,z1,z2, w):
    # Reconstruction/prediction loss
    l_re = (criterion(y11_, y1) + criterion(y12_, y1) + criterion(y21_, y2) + criterion(y22_, y2))/4

    # Shuffle the data in the batch as negative pairs
    rand_idx = list(np.arange(z1.size(0)))
    Random.shuffle(rand_idx)
    z3 = z1[rand_idx].view(z1.size(0),z1.size(1))  # Randomized symptom pattern embeddings
    z4 = z2[rand_idx].view(z2.size(0),z1.size(1))  # Randomized formula embeddings
    l_weight = 1-batch_jaccard(w,w[rand_idx])

    # Contrastive loss
    l_c12 = c_loss(z1, z2, 0) + l_weight * (c_loss(z1, z3, 1) + c_loss(z1, z4, 1))*0.5
    l_c21 = c_loss(z2, z1, 0) + l_weight * (c_loss(z2, z3, 1) + c_loss(z2, z4, 1))*0.5
    l_c34 = c_loss(z3, z4, 0) + l_weight * (c_loss(z3, z1, 1) + c_loss(z3, z2, 1))*0.5
    l_c43 = c_loss(z4, z3, 0) + l_weight * (c_loss(z4, z1, 1) + c_loss(z4, z2, 1))*0.5

    l_contrastive = torch.mean((l_c12+l_c21+l_c34+l_c43)/4)

    return l_re, l_contrastive


def independent_loss_function(y1, y1_, y2, y2_):
    """
    Reconstruction loss for independent null model.

    No contrastive loss.
    No cross-modal reconstruction.
    """
    l_re = (criterion(y1_, y1) + criterion(y2_, y2)) / 2
    l_contrastive = torch.tensor(0.0).to(DEVICE)

    return l_re, l_contrastive


#### Training function
def train(epoch):
    transformer_contrastive.train()
    train_loss = 0  # Total loss
    train_loss_R = 0  # Reconstruction/prediction loss
    train_loss_C = 0  # Contrastive loss
    for batch_idx, (batch_x1, batch_x2, batch_y1, batch_y2, batch_label) in enumerate(train_loader):
        # x1, y1: symptom input and output;
        # x2, y2: herb input and output;
        # w: to calculate the similarity between symptom patterns or formulas
        x1, x2, y1, y2, w = batch_x1.to(DEVICE), batch_x2.to(DEVICE), batch_y1.to(DEVICE), batch_y2.to(DEVICE), batch_label.to(DEVICE)

        # z1, z2: symptom pattern and formula embeddings
        # y11, y21: output when formula input was masked
        # y12, y22: output when symptom pattern input was masked
        if TRAIN_MODE == "independent":
            # Independent reconstruction:
            # symptoms reconstruct symptoms; herbs reconstruct herbs.
            y1_, y2_, z1, z2, _ = transformer_contrastive.forward_independent(x1, x2)
            y1_flat = y1.reshape(y1.shape[0] * y1.shape[1])
            y2_flat = y2.reshape(y2.shape[0] * y2.shape[1])
            y1_pred = y1_.reshape(y1_.shape[0] * y1_.shape[1], -1)
            y2_pred = y2_.reshape(y2_.shape[0] * y2_.shape[1], -1)

            loss_r, loss_c = independent_loss_function(y1_flat, y1_pred, y2_flat, y2_pred,)
            loss = loss_r

        else:
            # Full model:
            # reconstruction + contrastive alignment + cross-attention reconstruction.
            y11_, y21_, z1, _, _ = transformer_contrastive(x1, x2, mask_input='herb')
            y12_, y22_, _, z2, _ = transformer_contrastive(x1, x2, mask_input='symptom')

            y1_flat = y1.reshape(y1.shape[0] * y1.shape[1])
            y2_flat = y2.reshape(y2.shape[0] * y2.shape[1])

            y11_ = y11_.reshape(y11_.shape[0] * y11_.shape[1], -1)
            y12_ = y12_.reshape(y12_.shape[0] * y12_.shape[1], -1)
            y21_ = y21_.reshape(y21_.shape[0] * y21_.shape[1], -1)
            y22_ = y22_.reshape(y22_.shape[0] * y22_.shape[1], -1)

            loss_r, loss_c = loss_function(y1_flat, y11_, y12_, y2_flat, y21_, y22_, z1, z2, w,)
            loss = loss_r + BETA * loss_c


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_loss_R += loss_r.item()
        train_loss_C += loss_c.item()

        if batch_idx % 10 == 0:
            print('Train Epoch: {}_{} [{}/{} ({:.0f}%)]\tLoss: {:.4f} ,{:.4f} ,{:.4f}'.format(
                epoch,batch_idx, batch_idx * batch_size, len(train_loader.dataset),
                       batch_idx * batch_size / len(train_loader.dataset) * 100, loss.item() / len(batch_x1),loss_r.item() / len(batch_x1),loss_c.item() / len(batch_x1)))
    print('====> Epoch: {e} Average loss: {t:.4f},{r:.4f}, {c:.4f}'.format(e=epoch,t=train_loss / len(train_loader.dataset),r=train_loss_R / len(train_loader.dataset),c=train_loss_C / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset),train_loss_R / len(train_loader.dataset), train_loss_C / len(train_loader.dataset)


#### Validation function
def validation():
    transformer_contrastive.eval()
    val_loss = 0
    val_loss_R = 0
    val_loss_C = 0
    with torch.no_grad():
        for batch_idx, (batch_x1, batch_x2, batch_y1, batch_y2,batch_label) in enumerate(val_loader):
            x1, x2, y1, y2, w = batch_x1.to(DEVICE), batch_x2.to(DEVICE), batch_y1.to(DEVICE), batch_y2.to(DEVICE), batch_label.to(DEVICE)

            if TRAIN_MODE == "independent":
                y1_, y2_, z1, z2, _ = transformer_contrastive.forward_independent(x1, x2)
                y1_flat = y1.reshape(y1.shape[0] * y1.shape[1])
                y2_flat = y2.reshape(y2.shape[0] * y2.shape[1])
                y1_pred = y1_.reshape(y1_.shape[0] * y1_.shape[1], -1)
                y2_pred = y2_.reshape(y2_.shape[0] * y2_.shape[1], -1)

                loss_r, loss_c = independent_loss_function(y1_flat, y1_pred, y2_flat, y2_pred,)
                loss = loss_r

            else:
                y11_, y21_, z1, _, _ = transformer_contrastive(x1, x2, mask_input='herb')
                y12_, y22_, _, z2, _ = transformer_contrastive(x1, x2, mask_input='symptom')
                y1_flat = y1.reshape(y1.shape[0] * y1.shape[1])
                y2_flat = y2.reshape(y2.shape[0] * y2.shape[1])
                y11_ = y11_.reshape(y11_.shape[0] * y11_.shape[1], -1)
                y12_ = y12_.reshape(y12_.shape[0] * y12_.shape[1], -1)
                y21_ = y21_.reshape(y21_.shape[0] * y21_.shape[1], -1)
                y22_ = y22_.reshape(y22_.shape[0] * y22_.shape[1], -1)

                loss_r, loss_c = loss_function(y1_flat, y11_, y12_, y2_flat, y21_, y22_, z1, z2, w,)
                loss = loss_r + BETA * loss_c


            val_loss += loss.item()
            val_loss_R += loss_r.item()
            val_loss_C += loss_c.item()

    val_loss /= len(val_loader.dataset)
    val_loss_R /= len(val_loader.dataset)
    val_loss_C /= len(val_loader.dataset)
    print('====> Validation set loss: {:.4f}, {:.4f}, {:.4f}'.format(val_loss, val_loss_R, val_loss_C))
    return val_loss, val_loss_R, val_loss_C


######## Training steps

train_loss = []
train_loss_R = []
train_loss_C = []

val_loss = []
val_loss_R = []
val_loss_C = []

epoch_num = []

loss_curve = []  # Save the loss

model_save_path = save_dir
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

if show_loss == 'Y':
    f, axes = plt.subplots(3, 1, sharex=True)
else:
    f, axes = None, None
for epoch in range(1, NUM_EPOCHS+1):
    train_l, train_l_r, train_l_c = train(epoch)
    val_l, val_l_r, val_l_c = validation()
    train_loss.append(train_l)
    train_loss_R.append(train_l_r)
    train_loss_C.append(train_l_c)
    val_loss.append(val_l)
    val_loss_R.append(val_l_r)
    val_loss_C.append(val_l_c)
    epoch_num.append(epoch)
    loss_curve.append([int(epoch),train_l,train_l_r,train_l_c,val_l,val_l_r,val_l_c])
    if show_loss == 'Y':
        axes[0].clear()
        axes[0].plot(epoch_num, val_loss, 'b-', lw=1.5, label='Validation')
        axes[0].plot(epoch_num, train_loss, 'r-', lw=1.5, label='Training')
        axes[0].set_ylabel('Loss')
        axes[0].set_yscale('log')
        axes[0].set_xscale('log')
        axes[0].legend()
        axes[0].set_title('Total Loss')

        axes[1].clear()
        axes[1].plot(epoch_num, val_loss_R, color='b', lw=1.5, label='Validation')
        axes[1].plot(epoch_num, train_loss_R, color='r', lw=1.5, label='Training')
        axes[1].set_ylabel('Loss')
        axes[1].set_yscale('log')
        axes[1].set_xscale('log')
        axes[1].set_title('Reconstruction Loss')
        axes[1].legend()

        axes[2].clear()
        axes[2].plot(epoch_num, val_loss_C, color='b', lw=1.5, label='Validation')
        axes[2].plot(epoch_num, train_loss_C, color='r', lw=1.5, label='Training')
        axes[2].set_ylabel('Loss')
        axes[2].set_yscale('log')
        axes[2].set_xscale('log')
        axes[2].set_title('Contrastive Loss')
        axes[2].legend()

        plt.pause(0.1)
        if epoch == NUM_EPOCHS:
            plt.show()

    torch.save(transformer_contrastive, model_save_path+'/model_epoch_' + str(epoch) + '.pkl')
    np.savetxt(model_save_path+'/epoch_loss.txt',loss_curve)
