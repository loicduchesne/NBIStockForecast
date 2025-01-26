# TRAINER IMPORTS
import os
import json
from tqdm.notebook import tqdm, trange
import torch

# TRAINER
class LSTMTrainer:
    def __init__(self, model, train_loaders, test_loader, optimizer, loss_fn, device, num_epochs=100, pbar_relative_position=0):
        # Objects
        self.model = model
        self.train_loaders = train_loaders
        self.test_loader = test_loader
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
            train_loss = 0.0
            # TODO: Fix loss

            for train_loader in self.train_loaders:
                # TODO: Hidden initialization for each epoch
                hidden = self.model.init_hidden(train_loader.batch_size)

                # Training loop
                for batch_idx, (data, target) in train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()

                    output, hidden = self.model(data, hidden)
                    current_loss = self.loss_fn(output, target.squeeze())
                    current_loss.backward()
                    self.optimizer.step()

                    train_loss += current_loss.item()

                # Compute average training loss
                train_loss /= len(train_loader.dataset)
                self.train_losses.append(train_loss)

            # --- VALIDATION / EVALUATION ---
            if self.test_loader is not None:
                val_loss, auroc, auprc = self.evaluate()
                self.val_losses.append(val_loss)
                self.aurocs.append(auroc)
                self.auprcs.append(auprc)

                # Update the epoch bar description
                pbar_epochs.set_description(
                    f'Training | Epoch [{epoch+2}/{self.num_epochs}] - (Val.) Loss={val_loss:.4f}, AUROC={auroc:.3f}, AUPRC={auprc:.3f}'
                )
            else:
                # Update the epoch bar description for training loss only
                pbar_epochs.set_description(
                    f'Training | Epoch [{epoch+2}/{self.num_epochs}] - (Train.) Loss={train_loss:.4f}'
                )

        # Close the epoch bar
        pbar_epochs.close()

    def evaluate(self, test_loader=None):
        test_loader = self.test_loader if test_loader is None else test_loader

        """Evaluate loop."""
        self.model.eval()
        test_loss = 0
        y_true = []
        y_scores = []

        pbar_loader = tqdm(test_loader, desc=f'Evaluating', unit='Batch', position=1+self.pbar_relative_position)
        with torch.no_grad():
            for data, target in pbar_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_fn(output, target.squeeze()).item()  # sum up batch loss

                y_true.extend(target.cpu().numpy())
                y_scores.extend(output.softmax(dim=1)[:, 1].cpu().numpy())

        test_loss /= len(test_loader.dataset)
        auroc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)

        pbar_loader.close()
        return test_loss, auroc, auprc

    def predict(self, dataloader, with_logits=False):
        self.model.to(self.device)
        self.model.eval()
        y_true = []
        y_scores = []
        logits = []

        pbar_loader = tqdm(dataloader, desc=f'Predict', unit='Batch', position=0+self.pbar_relative_position)
        with torch.no_grad():
            for data, target in pbar_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                if with_logits:
                    logits.extend(output.cpu().numpy())

                y_true.extend(target.cpu().numpy())
                y_scores.extend(output.softmax(dim=1)[:, 1].cpu().numpy())

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