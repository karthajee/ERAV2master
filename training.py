from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch_lr_finder import LRFinder
import copy

class Trainer:

    def __init__(self, model, device, train_loader, test_loader, opt = 'adam', epochs=24, finder_start_lr = 1e-5, 
                 finder_end_lr=10, finder_num_iter=200, scheduler_div_factor = 10, scheduler_pct_start=0.2):
        """
        Initializes the Trainer instance, setting up containers for tracking training and testing metrics.
        """
        
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs        
        if opt == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=finder_start_lr, weight_decay=1e-4)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=finder_start_lr, momentum=0.9, weight_decay=1e-4)
        self.lr_max = self.get_lr_max(finder_start_lr, finder_end_lr, finder_num_iter)        
        self.lr_min = self.lr_max/scheduler_div_factor        
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                       max_lr=self.lr_max,
                                                       div_factor=scheduler_div_factor, 
                                                       final_div_factor=scheduler_div_factor,
                                                       epochs=self.epochs, 
                                                       steps_per_epoch=len(self.train_loader), 
                                                       pct_start=scheduler_pct_start,
                                                       anneal_strategy='linear',
                                                       three_phase=False)
        
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
        self.lrs = []

    def get_lr_max(self, start_lr, end_lr, num_iter):
        model_copy = copy.deepcopy(self.model)
        criterion = nn.CrossEntropyLoss()        
        optimizer_copy = copy.deepcopy(self.optimizer)
        lr_finder = LRFinder(model_copy, optimizer_copy, criterion, device=self.device)
        print('[INFO] Executing LR Finder range test to obtain LR max:')
        lr_finder.range_test(self.train_loader, end_lr=end_lr, num_iter=num_iter)
        _, lr_max = lr_finder.plot()
        lr_finder.reset()
        return lr_max
    
    def train(self):
        
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0
        processed = 0
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(data)
            loss = F.cross_entropy(y_pred, target)                        
            self.train_losses.append(loss.cpu().detach().numpy())
            self.lrs.append(self.get_lr())
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}% LR={self.lrs[-1]}')
            self.train_acc.append(100*correct/processed)

    def test(self):
        
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))
        self.test_acc.append(100. * correct / len(self.test_loader.dataset))

    def exec_train_test_loop(self, save_flag=True):

        for epoch in range(self.epochs):
            print(f'[INFO] Epoch #{epoch + 1}')
            self.train()
            self.test()            
        if save_flag:
            self.model.save()

    def get_lr(self):
        """"
        for tracking how your learning rate is changing throughout training
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    
    def visualize_loss_acc(self):

        """
        Visualizes training and testing losses and accuracies from the training instance.

        Function plots five charts: training loss, testing loss, training accuracy and testing accuracy, 
        and the absolute difference between training and testing accuracies (normalized by 100) across epochs.
        """

        train_test_diff = [np.abs(tr-te)/100 for tr, te in zip(self.train_acc, self.test_acc)]
        fig, axs = plt.subplots(ncols=6, figsize=(30,5))
        axs[0].plot(self.train_losses)
        axs[0].set(title="Train loss", xlabel="Steps")

        axs[1].plot(self.test_losses)
        axs[1].set(title="Test loss", xlabel="Epochs")

        axs[2].plot(self.train_acc)
        axs[2].set(title="Train accuracy", xlabel="Steps")

        axs[3].plot(self.test_acc)
        axs[3].set(title="Test accuracy", xlabel="Epochs")

        axs[4].plot(train_test_diff)
        axs[4].set(title="Train-test acc difference", xlabel="Epochs")

        axs[5].plot(self.lrs)
        axs[5].set(title="Learning Rate", xlabel="Steps")

        plt.show()