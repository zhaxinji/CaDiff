import torch
import numpy as np
import os
from loguru import logger
from tqdm import tqdm
from utils import ndcg, hit


class Trainer:
    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.args = args
        self.topk = args.topk
        self.max_k = max(self.topk)
        self.main_k = self.topk[0]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.best_hr = 0.0
        self.best_ndcg = 0.0
        self.best_res = {}
        self.best_epoch = 0
        self.early_stop_patience = args.early_stop
        self.early_stop_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        train_loader = self.data['train_loader']
        with tqdm(total=len(train_loader), desc='Training', unit='batch', leave=False) as pbar:
            for user_indices, pos_indices, neg_indices in train_loader:
                user_indices, pos_indices, neg_indices = user_indices.to(self.args.device), pos_indices.to(self.args.device), neg_indices.to(self.args.device)
                loss = self.model.loss(user_indices, pos_indices, neg_indices)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description(f'Batch Loss: {loss.item():.4f}')
                pbar.update()
                total_loss += loss.item()
        return total_loss / len(pbar)
    
    def train_model(self):
        num_epochs = self.args.num_epochs
        pbar = tqdm(range(num_epochs), desc='Epoch', unit='epoch', leave=False)
        for epoch in pbar:
            loss = self.train_epoch(epoch)
            pbar.set_description(f'Epoch {epoch+1} total loss: {loss:.4f}')
            pbar.update()
            res = self.evaluate()

            improved = self.save_best_model(epoch, res)
            if improved:
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.early_stop_patience:
                    logger.info(f"Early stopping triggered after {self.early_stop_patience} epochs without improvement.")
                    break
            logger.info(f"Test {self.format_result(res)}")

        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.args.dataset)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(self.model.state_dict(), f'{checkpoint_dir}/model.pt')

        self.print_best_test_result()

    def format_result(self, res):
        return ', '.join(f'HR@{k}: {res[k][0]:.4f}, NDCG@{k}: {res[k][1]:.4f}' for k in self.topk)

    def save_best_model(self, epoch, res):
        hr, ndcg = res[self.main_k]
        if hr > self.best_hr:
            self.best_hr = hr
            self.best_ndcg = ndcg
            self.best_res = res
            self.best_epoch = epoch + 1

            checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.args.dataset, 'best_model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save(self.model.state_dict(), f'{checkpoint_dir}/best_model.pt')
            logger.info(f"Best model saved at epoch {self.best_epoch} with {self.format_result(res)}")
            return True
        return False

    def print_best_test_result(self):
        logger.info(f"Best Test Result - Epoch: {self.best_epoch}, {self.format_result(self.best_res)}")

    def evaluate(self):
        device = self.args.device
        
        self.model.eval()
        topk_list = []
        with torch.no_grad():
            for user_indices, pos_indices, neg_indices in tqdm(self.data['test_loader'], desc='Test', leave=False):
                user_indices, pos_indices, neg_indices = user_indices.to(device), pos_indices.to(device), neg_indices.to(device)
                scores = self.model.predict(user_indices)
                
                for user_idx in range(user_indices.size(0)):
                    user = user_indices[user_idx].item()
                    train_items = self.data['train_gt'].get(str(user), [])
                    scores[user_idx, train_items] = -np.inf
                
                _, topk_indices = torch.topk(scores, self.max_k, dim=1)
                
                for idx, user in enumerate(user_indices):
                    gt_items = np.array(self.data['test_gt'][str(user.item())])
                    topk_items = topk_indices[idx].cpu().numpy()
                    mask = np.isin(topk_items, gt_items)
                    topk_list.append(mask)
        
        topk_list = np.vstack(topk_list)
        hr_res = hit(topk_list, self.data['test_gt_length']).mean(axis=0)
        ndcg_res = ndcg(topk_list, self.data['test_gt_length']).mean(axis=0)
        return {k: (hr_res[k-1], ndcg_res[k-1]) for k in self.topk}