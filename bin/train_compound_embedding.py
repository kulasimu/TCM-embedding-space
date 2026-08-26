from cmath import isnan
import json
import torch
import torch.nn as nn
import pickle
import numpy as np
from model import SimpleEncoder, SetTransformerEncoder
import sys
import matplotlib.pyplot as plt
import os
import torch.utils.data as Data
from sklearn.model_selection import KFold, train_test_split
from scipy.spatial import distance
import deepchem as dc
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm
# molfeat：直接调预训练 ChemBERTa
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer, HFModel
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Subset, Dataset


try:
    model_save_dir = sys.argv[1]  # model saving directory
    gen_data = sys.argv[2]  # Generate new data
    train_compound_model = sys.argv[3]  # Train embedding model for single compound
    train_combo_model = sys.argv[4]
    emb_dir = sys.argv[5] # Embedding saving directory
except:
    model_save_dir = 'core/trained_compound_model/'
    gen_data = 'N'
    train_compound_model = 'N'
    train_combo_model = 'Y'
    emb_dir = 'results/embeddings/'


train_data_save_dir = 'data/training/compound_training'
if not os.path.exists(train_data_save_dir):
    os.makedirs(train_data_save_dir)


######## Load TCM herb embeddings for correspondence alignment
with open(emb_dir + 'individual_herb_embeddings.pkl','rb') as fp:
    herb_embeddings = pickle.load(fp)

######## Herb ingredient compound data
# with open('data/herb_compounds/compound_list.pkl', 'rb') as fp:  # All herbal compounds included in the TCM-ES
#     compound_list = pickle.load(fp)
# num_compound = len(compound_list)


with open('data/herb_compounds/herb_compounds.pkl', 'rb') as fp:  # Herb-compound associations
    herb_compounds = pickle.load(fp)

# Modern medicine drugs and SMILES representations.
# Here we use the drugs from "Network-based in silico drug efficacy screening" (NC 2015) for illustration
with open('data/modern_drugs/FDA_drug_smiles_NC2015.txt','rb') as fp:
    drug_smiles = pickle.load(fp)



########################################################################################################################
##### Preprocessing (Represent compound using SMILES, and transfer SMILES to vectors using trained ChemBERTa)
def clean_smiles(smi: str) -> str | None:
    """
    Preprocessing (commonly used for herbal data):
        - Parse SMILES
        - Desalt/Keep only the largest fragment (LargestFragmentChooser)
        - Uncharger
    - Return canonical SMILES
    """
    # print(smi)
    if smi is None:
        print('invalid:', smi)
        return None
    # smi = str(smi).strip()
    if smi == "" or smi.lower() in {"nan", "none"}:
        print('invalid:', smi)
        return None

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print('invalid:', smi)
        return None

    # Take the largest fragment (desalted/solventized/counterionized).
    lfc = rdMolStandardize.LargestFragmentChooser()
    mol = lfc.choose(mol)

    # De-charge (commonly: salt form, charged form)
    uncharger = rdMolStandardize.Uncharger()
    mol = uncharger.uncharge(mol)

    # canonical smiles (preserving stereoscopic information)
    cleaned = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return cleaned

def chunked(iterable, chunk_size: int):
    for i in range(0, len(iterable), chunk_size):
        yield i, iterable[i : i + chunk_size]


def SMILE_vec_generation(
    df: str,
    smiles_col: str,
    output_dir: str = "results/smiles2vectors/",
    save_file_name: str = "compound_smile_vectors.pkl",
    # model_kind: str = "ChemBERTa-77M-MLM",
    batch_size: int = 64,
):
    os.makedirs(output_dir, exist_ok=True)


    # 2) 预处理
    tqdm.pandas(desc="Cleaning SMILES")
    df["smiles_raw"] = df[smiles_col]
    # print(df["smiles_raw"])

    df["smiles_clean"] = df["smiles_raw"].progress_apply(clean_smiles)
    df["is_valid_smiles"] = df["smiles_clean"].notna()

    valid_df = df[df["is_valid_smiles"]].copy()
    invalid_df = df[~df["is_valid_smiles"]].copy()

    print(f"Total rows: {len(df)}")
    print(f"Valid SMILES: {len(valid_df)}")
    print(f"Invalid SMILES: {len(invalid_df)}")

    # 3) 用 molfeat + ChemBERTa 生成向量
    # 1) 直接从 Hugging Face Hub 加载（model/tokenizer 都用同一个 repo id）
    hf_model = HFModel.from_pretrained(
        model="DeepChem/ChemBERTa-77M-MLM",
        tokenizer="DeepChem/ChemBERTa-77M-MLM",
        model_name="ChemBERTa-77M-MLM"
    )  # from_pretrained 支持 hub 名称或本地路径 :contentReference[oaicite:1]{index=1}

    # 2) 把已加载的 HFModel 传给 molfeat 的 PretrainedHFTransformer（此时 kind 是 HFModel，不会再去 store 下载）
    featurizer = PretrainedHFTransformer(
        kind=hf_model,  # 关键点：传 HFModel 对象而不是字符串 :contentReference[oaicite:2]{index=2}
        notation="smiles",
        pooling="mean",  # 建议用 mean，确定性更强
        preload=True,
        max_length=512,  # 你数据若更长可再调大（但会更慢/更吃显存）
        device="cpu",  # 有 GPU 可改 "cuda"
    )

    smiles_list = valid_df["smiles_clean"].tolist()

    # 分批生成，避免一次性太大
    vecs = []
    for start_idx, batch in tqdm(list(chunked(smiles_list, batch_size)), desc="SMILES to vectors"):
        X = featurizer(batch)  # numpy array: (B, D)
        # print(X.shape)
        vecs.append(X)

    X_all = np.concatenate(vecs, axis=0) if vecs else np.zeros((0, 0), dtype=float)
    print("SMILES vector matrix shape:", X_all.shape)

    valid_df["chemberta_emb"] = [row.tolist() for row in X_all]
    # 写 pickle
    valid_df.to_pickle(output_dir+save_file_name)
    # print(valid_df.columns)
    # print(valid_df[['Common Name',  "chemberta_emb", "smiles_clean",]])
    print("Saved:", output_dir+save_file_name)
    print('Original data size:', len(df))
    print('Valid data size:', len(valid_df))
    return valid_df


########################################################################################################################
######## Prepare training data

def generate_label(compound, herb_compounds, herbs_order=None, unknown_value=-1):
    """
    compound: 目标 compound（注意类型要和 herb_compounds 里一致：都用字符串id或都用同一种对象）
    herb_compounds: dict {herb: [c1, c2, ...]}，空 list 表示 unknown
    herbs_order: 可选，固定输出顺序；不传就用 dict 的顺序（不推荐用于可复现实验）
    """
    if herbs_order is None:
        herbs_order = list(herb_compounds.keys())

    labels = np.full(len(herbs_order), unknown_value, dtype=np.int8)

    for i, herb in enumerate(herbs_order):
        comps = herb_compounds.get(herb, [])
        if comps:  # 非空：可判定 1/0
            # 如果 comps 是 list，membership O(n)；可以提前把 comps 转成 set
            labels[i] = 1 if compound in comps else 0

    return labels


def prepare_compound_data(compound_list, smiles_vector_list, herb_compounds, herbs_order=None):
    compound_inputs = []
    herb_labels = []
    for i, compound in enumerate(compound_list):
        herb_labels.append(generate_label(compound,herb_compounds=herb_compounds, herbs_order=herbs_order))
        compound_inputs.append(smiles_vector_list[i])
        # print(compound)
    # print('Dataset size:', len(compound_inputs))
    return torch.FloatTensor(compound_inputs), torch.IntTensor(herb_labels)

if gen_data == 'Y':
    compound_smiles_df = pd.read_excel('data/herb_compounds/compound_smiles.xlsx', sheet_name="filtered")
    smiles_vec_df = SMILE_vec_generation(compound_smiles_df, smiles_col='Smiles')
    smiles_vec_df.to_pickle('data/herb_compounds/compound_smiles_vectors_preprocessed.pkl')
    print(smiles_vec_df[['Common Name', "chemberta_emb", "smiles_clean", ]])

    compound_inputs, herb_labels = prepare_compound_data(compound_list=list(smiles_vec_df['Common Name']),smiles_vector_list=list(smiles_vec_df['chemberta_emb']), herb_compounds=herb_compounds)
    dataset = Data.TensorDataset(compound_inputs, herb_labels)
    print('Dataset size:',len(dataset))
    with open(train_data_save_dir+'/dataset.pkl', 'wb') as fp:
        pickle.dump(dataset,fp)

with open(train_data_save_dir+'/dataset.pkl', 'rb') as fp:
    dataset = pickle.load(fp)


########################################################################################################################
######## Model configurations
data_x, _ = dataset[0]

batch_size = 32
NUM_EPOCHS = 500
LEARNING_RATE = 0.0001

INPUT_D = data_x.numel()
print('input dimension:', INPUT_D)
EMBEDDING_DIM = herb_embeddings.shape[-1]
print('Embedding dimension:', EMBEDDING_DIM)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Run on:', DEVICE)
BETA = 1
TAU = 1.2  # temperature，可调
negative_margin = 1.2
DROPOUT = 0.3

model = SimpleEncoder(input_dim=INPUT_D, out_dim=EMBEDDING_DIM, hidden_dims=[256], dropout=DROPOUT).to(DEVICE)
print(model)

herb_emb_device = torch.from_numpy(herb_embeddings).to(DEVICE)          # [H, d]
# herb_emb = F.normalize(herb_emb, dim=1)                      # L2 normalize (cosine-friendly)
target_norm = herb_emb_device.norm(dim=1).mean().detach()
print("Target herb norm:", target_norm.item())
print("Herb embedding tensor:", herb_emb_device.shape, herb_emb_device.device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,  weight_decay=5e-5)
mse_loss = nn.MSELoss()


########################################################################################################################
######## Loss functions
# '''
def norm_regularization(z, target_norm):
    z_norm = z.norm(dim=1)
    return ((z_norm - target_norm) ** 2).mean()


def positive_center_alignment(z, herb_labels, herb_emb):
    """
    z: [B, d]
    herb_labels: [B, H] in {1,0,-1}
    herb_emb: [H, d]
    """
    pos = (herb_labels == 1)
    keep = pos.sum(dim=1) > 0
    if keep.sum() == 0:
        return torch.tensor(0.0, device=z.device)

    z = z[keep]
    pos = pos[keep]

    # 每个样本的正 herb 中心
    pos_w = pos.float()
    pos_center = pos_w @ herb_emb / pos_w.sum(dim=1, keepdim=True).clamp(min=1.0)

    return ((z - pos_center) ** 2).mean()


def contrastive_all_herbs_euclidean(z, herb_labels, herb_emb, margin=2.0,
                                   pos_weight=1.0, neg_weight=1.0, eps=1e-9):
    """
    z: [B, d] compound embeddings
    herb_labels: [B, H] with values {1,0,-1} (1=contains, 0=not contains, -1=unknown)
    herb_emb: [H, d] fixed herb embeddings on DEVICE
    margin: push negatives to be at least margin away (in Euclidean distance)
    """

    labels = herb_labels
    valid = labels != -1
    pos = labels == 1
    neg = labels == 0

    # 避免某些样本没有正例导致 loss 失真：没有正例的行直接跳过
    pos_count = pos.sum(dim=1)
    keep = pos_count > 0
    if keep.sum() == 0:
        return torch.tensor(0.0, device=z.device)

    z = z[keep]                 # [Bk, d]
    pos = pos[keep]
    neg = neg[keep]
    valid = valid[keep]

    # 计算平方欧式距离矩阵 dist2: [Bk, H]
    # dist2 = ||z||^2 + ||h||^2 - 2 z·h
    z2 = (z * z).sum(dim=1, keepdim=True)                      # [Bk, 1]
    h2 = (herb_emb * herb_emb).sum(dim=1).unsqueeze(0)         # [1, H]
    dist2 = z2 + h2 - 2.0 * (z @ herb_emb.t())                 # [Bk, H]
    dist2 = dist2.clamp(min=0.0)

    # 用距离（带 sqrt）做 margin hinge；也可以用 dist2 和 margin^2（更快）
    dist = torch.sqrt(dist2 + eps)                              # [Bk, H]

    # 正样本：拉近 -> 最小化 dist^2（或 dist）
    pos_loss = (dist2 * pos).sum() / pos.sum().clamp(min=1)

    # 负样本：推远 -> 只惩罚落在 margin 内的负样本（hard negatives）
    neg_hinge = torch.clamp(margin - dist, min=0.0)            # [Bk, H]
    neg_loss = ((neg_hinge ** 2) * neg).sum() / neg.sum().clamp(min=1)

    return pos_weight * pos_loss + neg_weight * neg_loss


def loss_function(x, x_true, z, herb_labels, herb_emb):
    l_reconstruction = mse_loss(x, x_true)
    l_contrastive = contrastive_all_herbs_euclidean(
        z, herb_labels, herb_emb, margin=negative_margin, pos_weight=1.0, neg_weight=1.0
    )
    return l_reconstruction, l_contrastive



def multi_pos_infonce_euclidean(z, herb_labels, herb_emb, tau=1.0, eps=1e-9):
    """
    z: [B, d]
    herb_labels: [B, H] values in {1,0,-1}  (1=pos, 0=neg, -1=unknown)
    herb_emb: [H, d]
    """
    labels = herb_labels
    valid = labels != -1
    pos = labels == 1

    # 过滤掉没有任何正例的样本行，避免 numer = -inf
    pos_count = pos.sum(dim=1)
    keep = pos_count > 0
    if keep.sum() == 0:
        return torch.tensor(0.0, device=z.device)

    z = z[keep]           # [Bk, d]
    valid = valid[keep]   # [Bk, H]
    pos = pos[keep]       # [Bk, H]

    # dist2: [Bk, H]  squared Euclidean distance
    z2 = (z * z).sum(dim=1, keepdim=True)                 # [Bk, 1]
    h2 = (herb_emb * herb_emb).sum(dim=1).unsqueeze(0)    # [1, H]
    dist2 = z2 + h2 - 2.0 * (z @ herb_emb.t())            # [Bk, H]
    dist2 = dist2.clamp(min=0.0)

    # logits 越大越相似，所以用 -dist2
    logits = -dist2 / tau                                  # [Bk, H]

    denom = torch.logsumexp(logits.masked_fill(~valid, -1e9), dim=1)  # [Bk]
    numer = torch.logsumexp(logits.masked_fill(~pos,   -1e9), dim=1)  # [Bk]
    loss = -(numer - denom).mean()
    return loss


########################################################################################################################
#### Training and validation
def train_val_split(n, val_ratio=0.2, seed=42):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = int(n * val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx

train_idx, val_idx = train_val_split(len(dataset), val_ratio=0.2, seed=42)
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True, drop_last=False)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False, drop_last=False)



def run_one_epoch(model, loader, herb_emb, train=True,
                  lambda_align=0.5, lambda_norm=0.1):
    model.train(train) if train else model.eval()
    total_loss = 0.0
    total_rank = 0.0
    total_align = 0.0
    total_norm = 0.0
    total_n = 0

    with torch.set_grad_enabled(train):
        for x, labels in loader:
            x = x.to(DEVICE).float()
            if train:
                x = F.dropout(x, p=0.1, training=True)

            labels = labels.to(DEVICE)

            z = model(x)

            loss_rank = multi_pos_infonce_euclidean(z, labels, herb_emb, tau=TAU)
            loss_align = positive_center_alignment(z, labels, herb_emb)
            loss_norm = norm_regularization(z, target_norm)

            loss = loss_rank + lambda_align * loss_align + lambda_norm * loss_norm

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_rank += loss_rank.item() * bs
            total_align += loss_align.item() * bs
            total_norm += loss_norm.item() * bs
            total_n += bs

    return {
        "loss": total_loss / max(1, total_n),
        "rank": total_rank / max(1, total_n),
        "align": total_align / max(1, total_n),
        "norm": total_norm / max(1, total_n),
    }


def recall_at_k(model, dataset_subset, herb_emb, k=10, tau=1.0):
    model.eval()
    hits = 0
    total = 0

    for x, labels in DataLoader(dataset_subset, batch_size=128, shuffle=False):
        x = x.to(DEVICE).float()
        labels = labels.to(DEVICE)

        z = model(x)  # [B, d]

        # Euclidean logits = -dist2/tau
        z2 = (z * z).sum(dim=1, keepdim=True)                 # [B,1]
        h2 = (herb_emb * herb_emb).sum(dim=1).unsqueeze(0)    # [1,H]
        dist2 = (z2 + h2 - 2.0 * (z @ herb_emb.t())).clamp(min=0.0)
        logits = -dist2 / tau                                 # [B,H]

        topk = logits.topk(k, dim=1).indices                  # [B,k]
        pos = labels == 1                                     # [B,H]

        for i in range(x.size(0)):
            if pos[i, topk[i]].any().item():
                hits += 1
            total += 1

    return hits / max(1, total)


def mean_pos_in_topk_truek(model, dataset_subset, herb_emb, tau=1.0, cap_k=None, batch_size=128, device=None):
    """
    对每个 compound i:
      k_i = #positives (labels==1)
      取 top k_i 个 herb（按 logits 排序），统计其中 positives 数量 count_i
    返回：
      mean_count = mean(count_i)
      std_count  = std(count_i)
      mean_frac  = mean(count_i / k_i)   # 更推荐看这个（0~1）
    """
    model.eval()
    if device is None:
        device = herb_emb.device

    counts = []
    fracs = []

    for x, labels in DataLoader(dataset_subset, batch_size=batch_size, shuffle=False):
        x = x.to(device).float()
        labels = labels.to(device)  # [B,H] in {1,0,-1}

        z = model(x)  # [B,d]

        # Euclidean logits = -dist2/tau
        z2 = (z * z).sum(dim=1, keepdim=True)                  # [B,1]
        h2 = (herb_emb * herb_emb).sum(dim=1).unsqueeze(0)     # [1,H]
        dist2 = (z2 + h2 - 2.0 * (z @ herb_emb.t())).clamp(min=0.0)  # [B,H]
        logits = -dist2 / tau                                  # [B,H]

        pos = (labels == 1)                                    # [B,H]
        valid = (labels != -1)                                 # [B,H]
        logits = logits.masked_fill(~valid, -1e9)              # unknown 不参与排序

        p = pos.sum(dim=1)                                     # [B]

        for i in range(x.size(0)):
            k_i = int(p[i].item())
            if k_i <= 0:
                continue
            if cap_k is not None:
                k_i = min(k_i, cap_k)

            idx = logits[i].topk(k_i).indices                  # [k_i]
            c = int(pos[i, idx].sum().item())
            counts.append(c)
            fracs.append(c / k_i)

    if len(counts) == 0:
        return 0.0, 0.0, 0.0

    return float(np.mean(counts)), float(np.mean(fracs)), float(np.std(fracs)),



#### Running epochs
if train_compound_model == 'Y':
    best_score = -1.0
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    best_metrics = None

    baseline_recall = recall_at_k(model, dataset, herb_emb_device, k=10, tau=TAU)
    _, baseline_mean_frac, _ = mean_pos_in_topk_truek(
        model, dataset, herb_emb_device, tau=TAU, cap_k=50, batch_size=128, device=DEVICE
    )
    print("Baseline recall:", baseline_recall, ", fraction:", baseline_mean_frac)

    for epoch in range(1, NUM_EPOCHS + 1):

        train_out = run_one_epoch(model, train_loader, herb_emb_device, train=True)
        val_out   = run_one_epoch(model, val_loader, herb_emb_device, train=False)

        val_recall10 = recall_at_k(model, Subset(dataset, val_idx), herb_emb_device, k=10, tau=TAU)
        val_mean_count, val_mean_frac, val_std_frac = mean_pos_in_topk_truek(
            model,
            Subset(dataset, val_idx),
            herb_emb_device,
            tau=TAU,
            cap_k=50,
            batch_size=128,
            device=DEVICE
        )

        # 主指标：检索表现
        val_score = 0.5 * val_recall10 + 0.5 * val_mean_frac

        # 更新 best
        if  (val_score > best_score) or (val_score == best_score and val_out["loss"] < best_val):
            best_score = float(val_score)
            best_epoch = epoch
            best_val = float(val_out["loss"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "val_loss": float(val_out["loss"]),
                "val_rank": float(val_out["rank"]),
                "val_align": float(val_out["align"]),
                "val_norm": float(val_out["norm"]),
                "val_r10": float(val_recall10),
                "val_mean_frac": float(val_mean_frac),
            }

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"train {train_out['loss']:.4f} "
                f"(rank {train_out['rank']:.4f}, align {train_out['align']:.4f}, norm {train_out['norm']:.4f}) | "
                f"val {val_out['loss']:.4f} "
                f"(rank {val_out['rank']:.4f}, align {val_out['align']:.4f}, norm {val_out['norm']:.4f}) | "
                f"R@10 {val_recall10:.4f} | meanFrac@k {val_mean_frac:.3f} | "
                f"best {best_score:.4f}@{best_epoch}"
            )

    print(f"Best epoch: {best_epoch}, loss: {best_val:.4f}, score: {best_score:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

        save_path = model_save_dir + 'single_compound_embedding_model.pt'
        torch.save({
            "epoch": best_epoch,
            "best_score": float(best_score),
            "best_val_loss": float(best_val),
            "state_dict": best_state,
            "config": {
                "input_dim": INPUT_D,
                "out_dim": EMBEDDING_DIM,
                "tau": TAU,
            },
            "best_metrics": best_metrics,
        }, save_path)

        print(f"[Saved] best checkpoint to: {save_path}")

'''
load_path = model_save_dir + 'single_compound_embedding_model.pt'
ckpt = torch.load(load_path, map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
val_recall10 = recall_at_k(model, Subset(dataset, val_idx), herb_emb_device, k=10, tau=TAU)
val_mean_count, val_mean_frac, val_std_frac = mean_pos_in_topk_truek(
            model, Subset(dataset, val_idx), herb_emb_device, tau=TAU, cap_k=50, batch_size=128, device=DEVICE)
print(f"Best epoch {ckpt['epoch']:03d}"
    f"| R@10 {val_recall10:.4f} | meanFrac@k {val_mean_frac:.3f} | stdFrac@k {val_std_frac:.3f}")
# '''


########################################################################################################################
#### Encoder for multiple compounds (a Transformer, input compound embeddings, output a combo embedding)


## 0) Load trained model (single compound)
save_path = model_save_dir+'single_compound_embedding_model.pt'
ckpt = torch.load(save_path, map_location="cpu")
model.load_state_dict(ckpt["state_dict"])

processed_smiles_df = pd.read_pickle('data/herb_compounds/compound_smiles_vectors_preprocessed.pkl')
# print(processed_smiles_df[['Common Name', "chemberta_emb", "smiles_clean" ]])
compounds = list(processed_smiles_df['Common Name'])
compound_smiles_vecs = torch.FloatTensor(list(processed_smiles_df['chemberta_emb'].values)).to(DEVICE)
with torch.no_grad():
    Z_comp = model(compound_smiles_vecs).detach().cpu()
print("Z_comp:", Z_comp.shape)
# print(compound_embeddings.shape)

compound_herb_labels = []
for i, compound in enumerate(compounds):
    compound_herb_labels.append(generate_label(compound,herb_compounds=herb_compounds))
compound_herb_labels = torch.IntTensor(np.array(compound_herb_labels)).to(DEVICE)
# print(compound_herb_labels.shape)



## 1) 预处理：有效 herb 过滤 + 预计算 compound→z（256）
# 只保留有效 herb（不是全 -1 的列）
herb_keep = (compound_herb_labels != -1).any(dim=0)                 # [H_total]
herb_labels_eff = compound_herb_labels[:, herb_keep].to(torch.int8) # [N,H_eff]
H_eff = herb_labels_eff.size(1)

herb_emb_np = herb_embeddings[herb_keep.cpu().numpy()]  # [H_eff,256]
herb_emb_device = torch.from_numpy(herb_emb_np).to(DEVICE)      # [H_eff,256]
target_norm_combo = herb_emb_device.norm(dim=1).mean().detach()

print("H_eff:", H_eff, "herb_emb:", herb_emb_device.shape)
print("Target combo herb norm:", target_norm_combo.item())

compound2idx = {c:i for i,c in enumerate(compounds)}

# 为组合 AND 快速准备 pos/valid 矩阵（bool）
pos_mat   = (herb_labels_eff == 1)   # [N,H_eff] bool
valid_mat = (herb_labels_eff != -1)  # [N,H_eff] bool


## 2) 数据集：采样“组合”（长度可变），输出 token 序列 + compound 索引
class ComboListDataset(Dataset):
    def __init__(self, combo_list, Z_comp, shuffle_tokens=True):
        self.combo_list = combo_list          # list[torch.LongTensor], each [k]
        self.Z_comp = Z_comp                  # [N,256] tensor (CPU ok)
        self.shuffle_tokens = shuffle_tokens

    def __len__(self):
        return len(self.combo_list)

    def __getitem__(self, idx):
        cids = self.combo_list[idx].clone()    # [k]
        X = self.Z_comp[cids]                  # [k,256]
        if self.shuffle_tokens and cids.numel() > 1:
            perm = torch.randperm(cids.numel())
            X = X[perm]
            cids = cids[perm]
        return X, cids


## collate：pad 到同一长度，并输出 pad_mask：
def collate_combos(batch):
    X_list, cids_list = zip(*batch)
    lens = [x.size(0) for x in X_list]
    Kmax = max(lens)
    d = X_list[0].size(1)
    B = len(X_list)

    X = torch.zeros(B, Kmax, d, dtype=X_list[0].dtype)
    pad_mask = torch.ones(B, Kmax, dtype=torch.bool)      # True=pad
    cids_pad = torch.full((B, Kmax), -1, dtype=torch.long)

    for i, (x, cids) in enumerate(zip(X_list, cids_list)):
        k = x.size(0)
        X[i, :k] = x
        pad_mask[i, :k] = False
        cids_pad[i, :k] = cids

    return X, pad_mask, cids_pad


## 4) 组合版“单体同款”label & loss：全 herb 多正样本 InfoNCE（用 AND 得到 pos/valid）
def combo_positive_center_alignment(zS, cids_pad, herb_emb, pos_mat, valid_mat=None):
    """
    zS: [B,d]
    cids_pad: [B,K]
    herb_emb: [H,d]
    pos_mat: [N,H] bool
    """
    losses = []
    B = zS.size(0)

    for i in range(B):
        cids = cids_pad[i]
        cids = cids[cids != -1]
        if cids.numel() == 0:
            continue

        pos = pos_mat[cids].all(dim=0)   # [H]
        if pos.sum() == 0:
            continue

        pos_center = herb_emb[pos].mean(dim=0)   # [d]
        losses.append(((zS[i] - pos_center) ** 2).mean())

    if len(losses) == 0:
        return torch.tensor(0.0, device=zS.device)
    return torch.stack(losses).mean()


def combo_norm_regularization(zS, target_norm):
    z_norm = zS.norm(dim=1)
    return ((z_norm - target_norm) ** 2).mean()


def combo_multi_pos_infonce(zS, cids_pad, herb_emb, pos_mat, valid_mat, tau=0.1):
    """
    zS: [B,256] 组合 embedding
    cids_pad: [B,K] compound indices, pad=-1
    herb_emb: [H,256]
    pos_mat/valid_mat: [N,H] bool (在 CPU 也行，但建议搬到 DEVICE)
    """

    # Cosine similarity
    # zS = F.normalize(zS, dim=1)
    # h  = F.normalize(herb_emb, dim=1)
    # logits = (zS @ h.t()) / tau  # [B,H]

    # square Euclidean distance
    # dist2: [B,H]  (用恒等式避免广播太慢)
    z2 = (zS * zS).sum(dim=1, keepdim=True)  # [B,1]
    h2 = (herb_emb * herb_emb).sum(dim=1).unsqueeze(0)  # [1,H]
    dist2 = (z2 + h2 - 2.0 * (zS @ herb_emb.t())).clamp(min=0.0)  # [B,H]
    logits = -dist2 / tau  # [B,H]

    losses = []
    B = zS.size(0)
    for i in range(B):
        cids = cids_pad[i]
        cids = cids[cids != -1]
        if cids.numel() == 0:
            continue

        # AND across compounds -> combo label
        pos   = pos_mat[cids].all(dim=0)      # [H]
        valid = valid_mat[cids].all(dim=0)    # [H]

        # 只在 valid herb 上算；且必须有正例
        if pos.sum() == 0:
            continue

        li = logits[i]
        denom = torch.logsumexp(li.masked_fill(~valid, -1e9), dim=0)
        numer = torch.logsumexp(li.masked_fill(~pos,   -1e9), dim=0)
        losses.append(-(numer - denom))

    if len(losses) == 0:
        return torch.tensor(0.0, device=zS.device)
    return torch.stack(losses).mean()



## 5) 训练 + 验证
## 5.1 验证指标：Recall@10（top 10 对“任意正 herb”命中） + meanFrac@k（top K 正样本覆盖率）, 组合有多个正 herb，所以 Recall@10 定义为：top10 里是否命中任意一个正 herb。
@torch.no_grad()
def combo_recall_at_k(model, loader, herb_emb, pos_mat, valid_mat, k=10, tau=0.1):
    model.eval()
    # h = F.normalize(herb_emb, dim=1)
    h = herb_emb
    hits = 0
    total = 0

    for X, pad_mask, cids_pad in loader:
        X = X.to(DEVICE).float()
        pad_mask = pad_mask.to(DEVICE)
        cids_pad = cids_pad.to(DEVICE)

        # zS = F.normalize(model(X, pad_mask=pad_mask), dim=1)  # [B,256]
        zS = model(X, pad_mask=pad_mask)
        # logits = (zS @ h.t()) / tau                           # [B,H]
        z2 = (zS * zS).sum(dim=1, keepdim=True)  # [B,1]
        h2 = (h * h).sum(dim=1).unsqueeze(0)  # [1,H]
        dist2 = (z2 + h2 - 2.0 * (zS @ herb_emb.t())).clamp(min=0.0)  # [B,H]
        logits = -dist2 / tau  # [B,H]

        topk = logits.topk(k, dim=1).indices                  # [B,k]

        for i in range(X.size(0)):
            cids = cids_pad[i][cids_pad[i] != -1]
            pos = pos_mat[cids].all(dim=0)
            if pos.sum() == 0:
                continue
            if pos[topk[i]].any().item():
                hits += 1
            total += 1

    return hits / max(1, total)

@torch.no_grad()
def combo_mean_frac_at_truek(model, loader, herb_emb, pos_mat, valid_mat, tau=0.1, cap_k=50):
    model.eval()
    # h = F.normalize(herb_emb, dim=1)
    h = herb_emb
    fracs = []

    for X, pad_mask, cids_pad in loader:
        X = X.to(DEVICE).float()
        pad_mask = pad_mask.to(DEVICE)
        cids_pad = cids_pad.to(DEVICE)

        # zS = F.normalize(model(X, pad_mask=pad_mask), dim=1)
        zS = model(X, pad_mask=pad_mask)
        # logits = (zS @ h.t()) / tau
        z2 = (zS * zS).sum(dim=1, keepdim=True)  # [B,1]
        h2 = (h * h).sum(dim=1).unsqueeze(0)  # [1,H]
        dist2 = (z2 + h2 - 2.0 * (zS @ herb_emb.t())).clamp(min=0.0)  # [B,H]
        logits = -dist2 / tau  # [B,H]

        for i in range(X.size(0)):
            cids = cids_pad[i][cids_pad[i] != -1]
            pos = pos_mat[cids].all(dim=0)
            valid = valid_mat[cids].all(dim=0)

            p = int(pos.sum().item())
            if p <= 0:
                continue
            k = min(p, cap_k)

            li = logits[i].masked_fill(~valid, -1e9)
            idx = li.topk(k, dim=0).indices
            hit = int(pos[idx].sum().item())
            fracs.append(hit / k)

    return float(np.mean(fracs)) if fracs else 0.0




## 6.0) Functions for data preparing

def sample_combo_pool_restricted(
    herb_names, herb_compounds, compound2idx, allowed_cids,
    k_min=2, k_max=3, combos_per_herb=80, seed=42
):
    rng = random.Random(seed)
    combo_set = set()

    for herb in herb_names:
        comps = herb_compounds.get(herb, [])
        cids = [compound2idx[c] for c in comps if c in compound2idx and (compound2idx[c] in allowed_cids)]
        if len(cids) < k_min:
            continue

        for _ in range(combos_per_herb):
            k = rng.randint(k_min, k_max)
            if len(cids) < k:
                continue
            chosen = tuple(sorted(rng.sample(cids, k)))
            combo_set.add(chosen)

    combo_list = [torch.tensor(t, dtype=torch.long) for t in combo_set]
    rng.shuffle(combo_list)
    return combo_list


def sample_combo_pool_with_heldout(
    herb_names, herb_compounds, compound2idx, heldout_cids,
    k_min=2, k_max=3, combos_per_herb=80, seed=123
):
    rng = random.Random(seed)
    combo_set = set()

    for herb in herb_names:
        comps = herb_compounds.get(herb, [])
        all_cids = [compound2idx[c] for c in comps if c in compound2idx]
        test_c = [cid for cid in all_cids if cid in heldout_cids]
        if len(test_c) == 0:
            continue

        for _ in range(combos_per_herb):
            k = rng.randint(k_min, k_max)
            # 强制至少 1 个 heldout
            chosen = [rng.choice(test_c)]
            remain = k - 1
            pool = list(set(all_cids) - set(chosen))
            if len(pool) < remain:
                continue
            chosen += rng.sample(pool, remain)
            combo_set.add(tuple(sorted(chosen)))

    combo_list = [torch.tensor(t, dtype=torch.long) for t in combo_set]
    rng.shuffle(combo_list)
    return combo_list


# =========================================================
# A) 只用有效 herb
effective_herb_names = [h for j,h in enumerate(list(herb_compounds.keys())) if herb_keep[j].item()]

# B) leave-compound-out：划分 compound
N = Z_comp.size(0)  # N_compounds
all_cids = np.arange(N)
rng = np.random.RandomState(42)
rng.shuffle(all_cids)

heldout_ratio = 0.2
n_heldout = int(N * heldout_ratio)
heldout_cids = set(all_cids[:n_heldout].tolist())
train_cids   = set(all_cids[n_heldout:].tolist())

print(f"Compounds: train =", len(train_cids), "heldout =", len(heldout_cids))

# C) 生成训练组合池（只允许 train compounds）
combo_pool = sample_combo_pool_restricted(
    herb_names=effective_herb_names,
    herb_compounds=herb_compounds,
    compound2idx=compound2idx,
    allowed_cids=train_cids,
    k_min=2, k_max=3,
    combos_per_herb=80,
    seed=42
)
print("Train combo pool size:", len(combo_pool))

# D) 训练/验证：按组合划分（不是按 herb）
train_combos, val_combos = train_test_split(combo_pool, test_size=0.2, random_state=42)

train_set = ComboListDataset(train_combos, Z_comp, shuffle_tokens=True)
val_set   = ComboListDataset(val_combos,   Z_comp, shuffle_tokens=False)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  collate_fn=collate_combos)
val_loader   = DataLoader(val_set,   batch_size=128, shuffle=False, collate_fn=collate_combos)

# （可选）最终测试：组合里必须含 heldout compound
test_combos = sample_combo_pool_with_heldout(
    herb_names=effective_herb_names,
    herb_compounds=herb_compounds,
    compound2idx=compound2idx,
    heldout_cids=heldout_cids,
    k_min=2, k_max=3,
    combos_per_herb=80,
    seed=42
)
test_loader = DataLoader(ComboListDataset(test_combos, Z_comp, shuffle_tokens=False),
                         batch_size=128, shuffle=False, collate_fn=collate_combos)
print("Heldout test combos:", len(test_combos))

# =========================================================
# E) 搬 pos/valid 到 GPU
pos_mat_dev   = pos_mat.to(DEVICE)
valid_mat_dev = valid_mat.to(DEVICE)

# F) 模型与优化器
set_model = SetTransformerEncoder(d_model=EMBEDDING_DIM, nhead=4, num_layers=2, dropout=0.2).to(DEVICE)
opt = torch.optim.Adam(set_model.parameters(), lr=1e-4, weight_decay=1e-4)

TAU2 = 0.1
NUM_EPOCHS2 = 200

best_score = -1
best_state = None
best_epoch = -1

# =========================================================
# G) 训练
def run_combo_epoch(model, loader, herb_emb, pos_mat, valid_mat, train=True,
                    lambda_align=0.5, lambda_norm=0.1):
    model.train(train) if train else model.eval()

    total_loss = 0.0
    total_rank = 0.0
    total_align = 0.0
    total_norm = 0.0
    total_n = 0

    with torch.set_grad_enabled(train):
        for X, pad_mask, cids_pad in loader:
            X = X.to(DEVICE).float()
            pad_mask = pad_mask.to(DEVICE)
            cids_pad = cids_pad.to(DEVICE)

            zS = model(X, pad_mask=pad_mask)

            loss_rank = combo_multi_pos_infonce(
                zS, cids_pad, herb_emb, pos_mat, valid_mat, tau=TAU2
            )
            loss_align = combo_positive_center_alignment(
                zS, cids_pad, herb_emb, pos_mat, valid_mat
            )
            loss_norm = combo_norm_regularization(zS, target_norm_combo)

            loss = loss_rank + lambda_align * loss_align + lambda_norm * loss_norm

            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            bs = X.size(0)
            total_loss += loss.item() * bs
            total_rank += loss_rank.item() * bs
            total_align += loss_align.item() * bs
            total_norm += loss_norm.item() * bs
            total_n += bs

    return {
        "loss": total_loss / max(1, total_n),
        "rank": total_rank / max(1, total_n),
        "align": total_align / max(1, total_n),
        "norm": total_norm / max(1, total_n),
    }



best_score = -1.0
best_state = None
best_epoch = -1
best_metrics = None

if train_combo_model == 'Y':
    print(f"Training on {DEVICE}...")

    for epoch in range(1, NUM_EPOCHS2 + 1):
        train_out = run_combo_epoch(
            set_model, train_loader, herb_emb_device, pos_mat_dev, valid_mat_dev, train=True
        )
        val_out = run_combo_epoch(
            set_model, val_loader, herb_emb_device, pos_mat_dev, valid_mat_dev, train=False
        )

        r10 = combo_recall_at_k(
            set_model, val_loader, herb_emb_device, pos_mat_dev, valid_mat_dev, k=10, tau=TAU2
        )
        frac = combo_mean_frac_at_truek(
            set_model, val_loader, herb_emb_device, pos_mat_dev, valid_mat_dev, tau=TAU2, cap_k=50
        )

        score = 0.5 * r10 + 0.5 * frac

        if score > best_score:
            best_score = float(score)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in set_model.state_dict().items()}
            best_metrics = {
                "val_loss": float(val_out["loss"]),
                "val_rank": float(val_out["rank"]),
                "val_align": float(val_out["align"]),
                "val_norm": float(val_out["norm"]),
                "val_r10": float(r10),
                "val_mean_frac": float(frac),
            }

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[Combo-TrainCIDs] Ep {epoch:03d} | "
                f"train {train_out['loss']:.4f} "
                f"(rank {train_out['rank']:.4f}, align {train_out['align']:.4f}, norm {train_out['norm']:.4f}) | "
                f"val {val_out['loss']:.4f} "
                f"(rank {val_out['rank']:.4f}, align {val_out['align']:.4f}, norm {val_out['norm']:.4f}) | "
                f"Val R@10 {r10:.4f} | Val meanFrac {frac:.4f} | "
                f"best {best_score:.4f}@{best_epoch}"
            )

            if best_state is not None:
                save_path = model_save_dir + f'compound_combo_embedding_model_epoch{best_epoch}.pt'
                torch.save({
                    "epoch": best_epoch,
                    "best_score": float(best_score),
                    "state_dict": best_state,
                    "config": {
                        "input_dim": EMBEDDING_DIM,
                        "out_dim": EMBEDDING_DIM,
                        "tau": TAU2,
                    },
                    "best_metrics": best_metrics,
                }, save_path)
                print(f"[Saved] best checkpoint to: {save_path}")

    set_model.load_state_dict(best_state)
    set_model.to(DEVICE).eval()
    print("Best epoch:", best_epoch, "best score:", best_score)



# =========================================================
# H) 在 heldout compound 的组合上做最终评估：真正的“未见 compound”泛化
'''
test_epoch = 191
# test_loader = train_loader
if len(test_combos) > 0:
    load_path = model_save_dir + f'compound_combo_embedding_model_epoch{test_epoch}.pt'
    ckpt = torch.load(load_path, map_location="cpu")
    set_model.load_state_dict(ckpt["state_dict"])
    test_r10  = combo_recall_at_k(set_model, test_loader, herb_emb_device, pos_mat_dev, valid_mat_dev, k=5, tau=TAU2)
    test_frac = combo_mean_frac_at_truek(set_model, test_loader, herb_emb_device, pos_mat_dev, valid_mat_dev, tau=TAU2, cap_k=10)
    print(f"[Heldout-Combo] R@10 {test_r10:.4f} | meanFrac {test_frac:.4f}")


# '''

