# TRAINER IMPORTS
import os
import json
from tqdm.notebook import tqdm, trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import numpy as np
import warnings

# TRAINER
class LSTMTrainer:
    def __init__(self, model, train_loaders, test_loaders, optimizer, loss_fn, device, num_epochs=100, pbar_relative_position=0):
        # Objects
        self.model = model
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # Settings
        self.device = device
        self.num_epochs = num_epochs

        # Saved logs
        self.train_losses = []
        self.val_losses = []
        self.aurocs = []
        self.auprcs = []

        # Trainer config
        self.pbar_relative_position = pbar_relative_position

    def train(self, verbose=True):
        """
        Train the model using the provided train_loaders and the specified num_epochs.

        :returns: None
        """
        self.model.to(self.device)

        pbar_epochs = trange(
            self.num_epochs,
            desc=f'Training',
            unit='epoch',
            leave=False  # Remove old bar once each epoch finishes
        )
        for epoch in pbar_epochs:
            # --- TRAINING LOOP ---
            self.model.train()

            # TODO: Fix loss

            pbar_loaders = tqdm(
                    self.train_loaders,
                    desc=f'Epoch {epoch}',
                    unit='period',
                    position=self.pbar_relative_position + 1,
                    leave=False  # Remove old batch bars once each epoch finishes
                )

            for period_idx, train_loader in enumerate(pbar_loaders):
                # TODO: Hidden initialization for each epoch
                train_loss = 0.0

                h0, c0 = self.model.init_hidden(train_loader.batch_size)
                hidden = (h0.to(self.device), c0.to(self.device))

                # Training loop
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()

                    output, hidden = self.model(data, hidden)
                    hidden = (hidden[0].detach(), hidden[1].detach())
                    current_loss = self.loss_fn(output, target)
                    current_loss.backward()
                    self.optimizer.step()

                    train_loss += current_loss.item()

                    pbar_loaders.set_postfix({'Period Loss': train_loss})

                # Compute average training loss
                train_loss /= len(train_loader.dataset)
                self.train_losses.append(train_loss)

                pbar_loaders.set_postfix({f'Period {period_idx} Loss': f'{train_loss:.4f}'})

            # --- VALIDATION / EVALUATION ---
            if self.test_loaders is not None:
                val_loss, period_aurocs = self.evaluate()
                self.val_losses.append(val_loss)
                self.aurocs.append(period_aurocs)

                # Update the epoch bar description
                pbar_epochs.set_postfix({'Validation Loss': val_loss, 'AUROC': f'{sum(period_aurocs)/len(period_aurocs):.3f}'})

        # Close the epoch bar
        pbar_epochs.close()

    def evaluate(self, test_loaders=None):
        test_loaders = self.test_loaders if test_loaders is None else test_loaders

        """Evaluate loop."""
        self.model.eval()

        pbar_loaders = tqdm(
            test_loaders,
            desc=f'Evaluating',
            unit='period',
            position=self.pbar_relative_position + 1,
            leave=False  # Remove old batch bars once each epoch finishes
        )
        period_aurocs = []

        for period_idx, test_loader in enumerate(pbar_loaders):
            test_loss = 0.0
            y_true = []
            y_scores = []

            h0, c0 = self.model.init_hidden(test_loader.batch_size)
            hidden = (h0.to(self.device), c0.to(self.device))

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output, _ = self.model(data, hidden)
                    test_loss += self.loss_fn(output, target).item()  # sum up batch loss

                    y_true.extend(target.cpu().numpy())
                    y_scores.extend(output.softmax(dim=1).cpu().numpy())

            test_loss /= len(test_loader.dataset)

            y_true = np.array(y_true)
            y_scores = np.array(y_scores)

            if y_scores.shape[1] != len(np.unique(y_true)):
                # Log a warning about skipping evaluation
                warnings.warn(
                    f"Skipping AUROC calculation for period {period_idx}. "
                    f"Shape mismatch: y_true has {len(np.unique(y_true))} unique classes, "
                    f"but y_scores has {y_scores.shape[1]} columns."
                )
            else:
                # Compute AUROC as usual
                auroc = roc_auc_score(y_true, y_scores, multi_class='ovr', average='macro')
                period_aurocs.append(auroc)

        return test_loss, period_aurocs

    def predict(self, dataloaders, with_logits=False):
        self.model.to(self.device)
        self.model.eval()
        y_true = []
        y_scores = []
        logits = []

        pbar_loader = tqdm(dataloaders, desc=f'Predict', unit='Batch', position=0+self.pbar_relative_position)
        with torch.no_grad():
            for data, target in pbar_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                if with_logits:
                    logits.extend(output.cpu().numpy())

                y_true.extend(target.cpu().numpy())
                y_scores.extend(output.softmax(dim=1).cpu().numpy())

        pbar_loader.close()
        if with_logits:
            return y_true, y_scores, logits
        else:
            return y_true, y_scores

    def save_model(self, path, model_name=None):
        os.makedirs(f'{path}/{model_name}', exist_ok = True)
        full_pth = f'{path}/{model_name}'

        ## Save weights
        self.model.to('cpu') # Move to CPU first
        torch.save(self.model.state_dict(), f'{full_pth}/state_dict.pt')

        ## Save model configuration
        try:
            config = {
                'optimizer': type(self.optimizer),
                'lr': self.optimizer.param_groups[0]['lr'],
                'loss': type(self.loss_fn)
            }

            with open(f'{full_pth}/config.json', 'w') as fp:
                json.dump(config, fp)
        except:
            print('An error occurred while saving the model configuration.')
            pass

        ## Save training statistics
        with open(f'{full_pth}/train_losses.json', 'w') as fp: # Train losses
            json.dump(self.train_losses, fp)
        if len(self.val_losses) > 0: # Check if validation was ran
            with open(f'{full_pth}/val_losses.json', 'w') as fp: # Val losses
                json.dump(self.val_losses, fp)
            with open(f'{full_pth}/aurocs.json', 'w') as fp: # AUROCs
                json.dump(self.aurocs, fp)
            with open(f'{full_pth}/auprcs.json', 'w') as fp: # AUPRCs
                json.dump(self.auprcs, fp)

        print(f'Successfully saved model to {full_pth}')