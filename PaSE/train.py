
import time
import datetime
import random
import argparse
import numpy as np
from itertools import combinations

import torch.nn.functional as F
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, recall_score
import sys
from utils import Logger, get_loaders, build_model, generate_mask, generate_inputs
from loss import MaskedCELoss, MaskedMSELoss
import os
import warnings
sys.path.append('./')
warnings.filterwarnings("ignore")
import config
from model import PrototypeBank, EntropicOTAligner, ShapleySGM, PaSE


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_samples_for_prototypes(x_out_m, labels, umask):

    B, S, D = x_out_m.shape
    # flatten
    feats = x_out_m.reshape(-1, D)  # [B*S, D]
    labs = labels.reshape(-1)       # [B*S]
    mask = umask.reshape(-1).bool()
    # select masked valid frames
    if mask.sum() == 0:
        # fallback: take first frame of each batch (rare)
        mask = torch.zeros_like(mask)
        mask[0:B] = 1
    feats_sel = feats[mask]
    labs_sel = labs[mask]
    return feats_sel.detach(), labs_sel.detach()


def train_or_eval_model(args, model, reg_loss, cls_loss, dataloader, optimizer=None, train=False,
                        first_stage=True, mark='train', prototype_banks=None, aligner=None,
                        use_sgm=False, phi=None, modalities=['audio','text','vision']):
    
    assert not train or optimizer != None
    if train:
        model.train()
    else:
        model.eval()

    preds, preds_a, preds_t, preds_v, masks, labels = [], [], [], [], [], []
    losses = []

    device = next(model.parameters()).device
    for data_idx, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()

        # unpack data (same as before)
        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
        qmask, umask, label = data[6], data[7], data[8]
        vidnames = [data[-1][i] for i in range(len(data[-1]))]

        seqlen = audio_host.size(0)
        batch = audio_host.size(1)

        # generate masks as original
        matrix = generate_mask(seqlen, batch, args.test_condition, first_stage)
        audio_host_mask = np.reshape(matrix[0], (batch, seqlen, 1))
        text_host_mask = np.reshape(matrix[1], (batch, seqlen, 1))
        visual_host_mask = np.reshape(matrix[2], (batch, seqlen, 1))
        audio_host_mask = torch.LongTensor(audio_host_mask.transpose(1, 0, 2))
        text_host_mask = torch.LongTensor(text_host_mask.transpose(1, 0, 2))
        visual_host_mask = torch.LongTensor(visual_host_mask.transpose(1, 0, 2))
        # guest mask
        matrix = generate_mask(seqlen, batch, args.test_condition, first_stage)
        audio_guest_mask = np.reshape(matrix[0], (batch, seqlen, 1))
        text_guest_mask = np.reshape(matrix[1], (batch, seqlen, 1))
        visual_guest_mask = np.reshape(matrix[2], (batch, seqlen, 1))
        audio_guest_mask = torch.LongTensor(audio_guest_mask.transpose(1, 0, 2))
        text_guest_mask = torch.LongTensor(text_guest_mask.transpose(1, 0, 2))
        visual_guest_mask = torch.LongTensor(visual_guest_mask.transpose(1, 0, 2))

        masked_audio_host = audio_host * audio_host_mask
        masked_audio_guest = audio_guest * audio_guest_mask
        masked_text_host = text_host * text_host_mask
        masked_text_guest = text_guest * text_guest_mask
        masked_visual_host = visual_host * visual_host_mask
        masked_visual_guest = visual_guest * visual_guest_mask

        cuda = torch.cuda.is_available() and not args.no_cuda
        if cuda:
            masked_audio_host, audio_host_mask = masked_audio_host.to(device), audio_host_mask.to(device)
            masked_text_host, text_host_mask = masked_text_host.to(device), text_host_mask.to(device)
            masked_visual_host, visual_host_mask = masked_visual_host.to(device), visual_host_mask.to(device)
            masked_audio_guest, audio_guest_mask = masked_audio_guest.to(device), audio_guest_mask.to(device)
            masked_text_guest, text_guest_mask = masked_text_guest.to(device), text_guest_mask.to(device)
            masked_visual_guest, visual_guest_mask = masked_visual_guest.to(device), visual_guest_mask.to(device)

            qmask = qmask.to(device)
            umask = umask.to(device)
            label = label.to(device)

        masked_input_features = generate_inputs(masked_audio_host, masked_text_host, masked_visual_host,
                                                masked_audio_guest, masked_text_guest, masked_visual_guest, qmask)
        input_features_mask = generate_inputs(audio_host_mask, text_host_mask, visual_host_mask,
                                              audio_guest_mask, text_guest_mask, visual_guest_mask, qmask)

        # forward
        hidden, out, out_a, out_t, out_v, weight_save = model(masked_input_features[0], input_features_mask[0], umask, first_stage)

        # losses as before (task loss)
        lp_ = out.view(-1, out.size(2))
        lp_a, lp_t, lp_v = out_a.view(-1, out_a.size(2)), out_t.view(-1, out_t.size(2)), out_v.view(-1, out_v.size(2))
        labels_ = label.view(-1)

        if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            if first_stage:
                loss_a = cls_loss(lp_a, labels_, umask)
                loss_t = cls_loss(lp_t, labels_, umask)
                loss_v = cls_loss(lp_v, labels_, umask)
                task_loss = (loss_a + loss_t + loss_v) / 3.0
            else:
                task_loss = cls_loss(lp_, labels_, umask)
        else:
            if first_stage:
                loss_a = reg_loss(lp_a, labels_, umask)
                loss_t = reg_loss(lp_t, labels_, umask)
                loss_v = reg_loss(lp_v, labels_, umask)
                task_loss = (loss_a + loss_t + loss_v) / 3.0
            else:
                task_loss = reg_loss(lp_, labels_, umask)


        L_intra = torch.tensor(0.0, device=device)
        if not first_stage and prototype_banks is not None:
            # get per-modality flattened features and labels for prototype update
            feats_a, labs_a = get_samples_for_prototypes(out_a, label, umask)
            feats_t, labs_t = get_samples_for_prototypes(out_t, label, umask)
            feats_v, labs_v = get_samples_for_prototypes(out_v, label, umask)

            # update prototype banks (EMA)
            if feats_a.size(0) > 0:
                prototype_banks['audio'].momentum_update(feats_a.to(prototype_banks['audio'].device), labs_a.to(prototype_banks['audio'].device))
            if feats_t.size(0) > 0:
                prototype_banks['text'].momentum_update(feats_t.to(prototype_banks['text'].device), labs_t.to(prototype_banks['text'].device))
            if feats_v.size(0) > 0:
                prototype_banks['vision'].momentum_update(feats_v.to(prototype_banks['vision'].device), labs_v.to(prototype_banks['vision'].device))

            # compute L_intra as MSE between sample feats and their class prototype
            if feats_a.size(0) > 0:
                proto_a = prototype_banks['audio'].get()[labs_a.to(prototype_banks['audio'].device)]
                L_intra = L_intra + F.mse_loss(feats_a.to(device), proto_a.to(device))
            if feats_t.size(0) > 0:
                proto_t = prototype_banks['text'].get()[labs_t.to(prototype_banks['text'].device)]
                L_intra = L_intra + F.mse_loss(feats_t.to(device), proto_t.to(device))
            if feats_v.size(0) > 0:
                proto_v = prototype_banks['vision'].get()[labs_v.to(prototype_banks['vision'].device)]
                L_intra = L_intra + F.mse_loss(feats_v.to(device), proto_v.to(device))


        L_ot = torch.tensor(0.0, device=device)
        if not first_stage and aligner is not None and prototype_banks is not None:
            # sum pairwise L_inter between modality prototypes
            mods = ['audio','text','vision']
            for a, b in combinations(mods, 2):
                Pa = prototype_banks[a].get()
                Pb = prototype_banks[b].get()
                res = aligner.align(Pa.to(aligner.device), Pb.to(aligner.device), alpha=0.1, beta=0.05)
                L_ot = L_ot + res['L_inter']

        lambda_intra = 0.1
        lambda_ot = 0.2
        loss = task_loss + lambda_intra * L_intra + lambda_ot * L_ot

        # backward & SGM gradient modulation
        if train:
            loss.backward()

            # apply SGM gradient modulation if requested
            if use_sgm and phi is not None:
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    # Heuristic mapping: adjust based on your model naming scheme
                    lname = name.lower()
                    if 'audio' in lname or 'a_' in lname or '.a' in lname:
                        param.grad.mul_(phi.get('audio', 1.0))
                    elif 'text' in lname or 't_' in lname or 'bert' in lname:
                        param.grad.mul_(phi.get('text', 1.0))
                    elif 'vision' in lname or 'v_' in lname:
                        param.grad.mul_(phi.get('vision', 1.0))
                    else:
                        pass

            optimizer.step()


        preds.append(lp_.data.cpu().numpy())
        preds_a.append(lp_a.data.cpu().numpy())
        preds_t.append(lp_t.data.cpu().numpy())
        preds_v.append(lp_v.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

    preds  = np.concatenate(preds)
    preds_a = np.concatenate(preds_a)
    preds_t = np.concatenate(preds_t)
    preds_v = np.concatenate(preds_v)
    labels = np.concatenate(labels)
    masks  = np.concatenate(masks)

    if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        preds = np.argmax(preds, 1)
        preds_a = np.argmax(preds_a, 1)
        preds_t = np.argmax(preds_t, 1)
        preds_v = np.argmax(preds_v, 1)
        avg_loss = 0
        avg_accuracy = accuracy_score(labels, preds, sample_weight=masks)
        avg_fscore = f1_score(labels, preds, sample_weight=masks, average='weighted')
        mae = 0
        ua = recall_score(labels, preds, sample_weight=masks, average='macro')
        avg_acc_a = accuracy_score(labels, preds_a, sample_weight=masks)
        avg_acc_t = accuracy_score(labels, preds_t, sample_weight=masks)
        avg_acc_v = accuracy_score(labels, preds_v, sample_weight=masks)
        return mae, ua, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], vidnames, avg_loss, None

    else:
        non_zeros = np.array([i for i, e in enumerate(labels) if e != 0])
        avg_loss = 0
        avg_accuracy = accuracy_score((labels[non_zeros] > 0), (preds[non_zeros] > 0))
        avg_fscore = f1_score((labels[non_zeros] > 0), (preds[non_zeros] > 0), average='weighted')
        mae = np.mean(np.absolute(labels[non_zeros] - preds[non_zeros].squeeze()))
        corr = np.corrcoef(labels[non_zeros], preds[non_zeros].squeeze())[0][1]
        avg_acc_a = accuracy_score((labels[non_zeros] > 0), (preds_a[non_zeros] > 0))
        avg_acc_t = accuracy_score((labels[non_zeros] > 0), (preds_t[non_zeros] > 0))
        avg_acc_v = accuracy_score((labels[non_zeros] > 0), (preds_v[non_zeros] > 0))
        return mae, corr, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], vidnames, avg_loss, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio-feature', type=str, default=None)
    parser.add_argument('--text-feature', type=str, default=None)
    parser.add_argument('--video-feature', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='IEMOCAPFour')
    parser.add_argument('--time-attn', action='store_true', default=False)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--attn_drop_rate', type=float, default=0.0)
    parser.add_argument('--hidden', type=int, default=100)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--n_speakers', type=int, default=2)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--l2', type=float, default=0.00001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num-folder', type=int, default=5)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--test_condition', type=str, default='atv', help='test condition: atv, av, tv, a, t, v')
    parser.add_argument('--stage_epoch', type=int, default=100)
    parser.add_argument('--sgm_freq', type=int, default=5, help='epochs between SGM updates')
    parser.add_argument('--sgm_perm', type=int, default=200, help='permutations for Shapley approx')

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    args.device = device

    save_folder_name = f'{args.dataset}'
    save_log_dir = os.path.join(config.LOG_DIR, 'training_result', save_folder_name)
    os.makedirs(save_log_dir, exist_ok=True)

    time_dataset = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + f"_{args.dataset}"

    sys.stdout = Logger(filename=os.path.join(
        save_log_dir,
        f"{time_dataset}_bs-{args.batch_size}_lr-{args.lr}_seed-{args.seed}_tc-{args.test_condition}.txt"
    ), stream=sys.stdout)

    seed_torch(args.seed)

    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        args.num_folder = 1
        args.n_classes = 1
        args.n_speakers = 1
    elif args.dataset == 'IEMOCAPFour':
        args.num_folder = 5
        args.n_classes = 4
        args.n_speakers = 2
    elif args.dataset == 'IEMOCAPSix':
        args.num_folder = 5
        args.n_classes = 6
        args.n_speakers = 2

    audio_feature, text_feature, video_feature = args.audio_feature, args.text_feature, args.video_feature
    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], audio_feature)
    text_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], video_feature)

    train_loaders, test_loaders, adim, tdim, vdim = get_loaders(
        audio_root=audio_root,
        text_root=text_root,
        video_root=video_root,
        num_folder=args.num_folder,
        batch_size=args.batch_size,
        dataset=args.dataset,
        num_workers=0
    )

    print(f"Dataset {args.dataset} with {args.num_folder} folds, batch size {args.batch_size}, "
          f"features: {audio_feature}, {text_feature}, {video_feature}")
    print(f'===*4 Training and Testing ===*4')

    folder_mae, folder_corr, folder_acc, folder_f1, folder_model = [], [], [], [], []

    for fold_idx in range(args.num_folder):
        print(f"=== Fold {fold_idx + 1}/{args.num_folder} ===")
        train_loader = train_loaders[fold_idx]
        test_loader = test_loaders[fold_idx]
        start_time = time.time()

        model = build_model(args, adim, tdim, vdim)
        model.to(device)

        reg_loss = MaskedMSELoss()
        cls_loss = MaskedCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        try:
            D_e = model.block.dim
        except Exception:
            D_e = args.hidden

        modalities = ['audio', 'text', 'vision']
        prototype_banks = {
            m: PrototypeBank(num_classes=args.n_classes, dim=D_e, momentum=0.98, device=device) for m in modalities
        }
        aligner = EntropicOTAligner(reg=0.05, sinkhorn_iters=200, device=device)

        def utility_fn(S: set):
            if not S:
                return 0.0
            L_inter_S = 0.0
            L_intra_S = 0.0
            for a, b in combinations(S, 2):
                res = aligner.align(prototype_banks[a].get().to(device), prototype_banks[b].get().to(device))
                L_inter_S += float(res['L_inter'].item())
            for m in S:
                L_intra_S += float(prototype_banks[m].get().var().item())
            rho = 0.6
            return rho / (1.0 + L_inter_S) + (1.0 - rho) / (1.0 + L_intra_S)

        sgm = ShapleySGM(modalities, utility_fn, n_permutations=args.sgm_perm, device=device)
        phi = None

        train_acc_atv_list = []
        test_fscores, test_accs, test_maes, test_corrs = [], [], [], []
        models = []

        for epoch in range(args.epochs):
            first_stage = epoch < args.stage_epoch
            use_sgm = False

            if not first_stage and (epoch - args.stage_epoch) % args.sgm_freq == 0:
                print(f"[PaSE] Estimating Shapley values at epoch {epoch} ...")
                shap_vals = sgm.estimate_shapley()
                phi = sgm.modulation_factors(shap_vals)
                print(f"[PaSE] Shapley approx: {shap_vals}")
                print(f"[PaSE] modulation phi: {phi}")
                use_sgm = True

            train_mae, train_corr, train_acc, train_fscore, train_acc_atv, _, _, _ = train_or_eval_model(
                args, model, reg_loss, cls_loss, train_loader, optimizer=optimizer, train=True,
                first_stage=first_stage, mark='train', prototype_banks=prototype_banks,
                aligner=aligner, use_sgm=use_sgm, phi=phi, modalities=modalities)

            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, _, _, _ = train_or_eval_model(
                args, model, reg_loss, cls_loss, test_loader, optimizer=None, train=False,
                first_stage=first_stage, mark='test', prototype_banks=prototype_banks,
                aligner=aligner, use_sgm=False, phi=None, modalities=modalities)

            print(f"Epoch {epoch} - Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")


            test_accs.append(test_acc)
            test_fscores.append(test_fscore)
            test_maes.append(test_mae)
            test_corrs.append(test_corr)
            models.append(model)
            train_acc_atv_list.append(train_acc_atv)


        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            best_idx = np.argmax(test_fscores)
        else:
            best_idx = np.argmax(test_accs)

        bestmae = test_maes[best_idx]
        bestcorr = test_corrs[best_idx]
        bestf1 = test_fscores[best_idx]
        bestacc = test_accs[best_idx]
        bestmodel = models[best_idx]

        folder_mae.append(bestmae)
        folder_corr.append(bestcorr)
        folder_f1.append(bestf1)
        folder_acc.append(bestacc)
        folder_model.append(bestmodel)

        elapsed_time = time.time() - start_time

        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            print(f"Best epoch for fold {fold_idx+1}: {best_idx} | Test MAE: {bestmae:.4f}, Corr: {bestcorr:.4f}, F1: {bestf1:.4f}, Acc: {bestacc:.4f}")
        else:
            print(f"Best epoch for fold {fold_idx+1}: {best_idx} | Test Acc: {bestacc:.4f}, UA (macro recall): {bestcorr:.4f}")

        print(f"Fold {fold_idx+1} finished in {elapsed_time:.2f} seconds.")
        print('-' * 80)


    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        print(f"Average across folds - MAE: {np.mean(folder_mae):.4f}, Corr: {np.mean(folder_corr):.4f}, "
              f"F1: {np.mean(folder_f1):.4f}, Acc: {np.mean(folder_acc):.4f}")
    else:
        print(f"Average across folds - Acc: {np.mean(folder_acc):.4f}, UA (macro recall): {np.mean(folder_corr):.4f}")


    save_model_dir = os.path.join(config.MODEL_DIR, 'main_result', save_folder_name)
    os.makedirs(save_model_dir, exist_ok=True)

    suffix_name = f"{time_dataset}_hidden-{args.hidden}_bs-{args.batch_size}"
    feature_name = f'{audio_feature};{text_feature};{video_feature}'

    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        res_name = f'mae-{np.mean(folder_mae):.3f}_corr-{np.mean(folder_corr):.3f}_f1-{np.mean(folder_f1):.4f}_acc-{np.mean(folder_acc):.4f}'
    else:
        res_name = f'acc-{np.mean(folder_acc):.4f}_ua-{np.mean(folder_corr):.4f}'

    save_path = os.path.join(save_model_dir,
                             f'{suffix_name}_features-{feature_name}_{res_name}_tc-{args.test_condition}.pth')
    torch.save({'model': bestmodel.state_dict()}, save_path)
    print(f"Model saved to: {save_path}")

    print("All folds finished.")
