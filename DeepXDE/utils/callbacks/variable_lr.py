import numpy as np
import deepxde as dde

class VariableLearningRateCallback(dde.callbacks.Callback):
    """
    Variable learning rate callback with multiple strategies
    """
    def __init__(self, mode='adaptive', patience=500, factor=0.5, min_lr=1e-7, 
                 monitor='loss_train', verbose=True):
        super().__init__()
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.monitor = monitor
        self.verbose = verbose
        
        # Internal state
        self.wait = 0
        self.best_loss = np.inf
        self.initial_lr = None
        
    def on_train_begin(self):
        """Store initial learning rate"""
        if hasattr(self.model.opt, 'param_groups'):
            self.initial_lr = self.model.opt.param_groups[0]['lr']
        else:
            self.initial_lr = self.model.opt.learning_rate
        
        if self.verbose:
            print(f"Variable LR Callback - Initial LR: {self.initial_lr:.2e}")
    
    def on_epoch_end(self):
        """Called at the end of each epoch"""
        current_loss = self.get_current_loss()
        
        if self.mode == 'adaptive':
            self._adaptive_lr(current_loss)
        elif self.mode == 'plateau':
            self._plateau_lr(current_loss)
        elif self.mode == 'exponential':
            self._exponential_lr()
        elif self.mode == 'cosine':
            self._cosine_lr()
        elif self.mode == 'loss_based':
            self._loss_based_lr(current_loss)
    
    def get_current_loss(self):
        """Get current training loss"""
        if hasattr(self.model, 'train_state') and self.model.train_state.loss_train is not None:
            return self.model.train_state.loss_train[-1] if len(self.model.train_state.loss_train) > 0 else np.inf
        return np.inf
    
    def _adaptive_lr(self, current_loss):
        """Adaptive learning rate based on loss improvement"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            old_lr = self.get_lr()
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if new_lr != old_lr:
                self.set_lr(new_lr)
                if self.verbose:
                    print(f"Epoch {self.model.train_state.epoch}: "
                          f"Reducing LR from {old_lr:.2e} to {new_lr:.2e}")
            
            self.wait = 0
    
    def _plateau_lr(self, current_loss):
        """Reduce LR on plateau"""
        if current_loss < self.best_loss - 1e-6:  # Small improvement threshold
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            old_lr = self.get_lr()
            new_lr = max(old_lr * self.factor, self.min_lr)
            self.set_lr(new_lr)
            
            if self.verbose and old_lr != new_lr:
                print(f"Plateau detected. LR: {old_lr:.2e} → {new_lr:.2e}")
            
            self.wait = 0
    
    def _exponential_lr(self):
        """Exponential decay"""
        current_lr = self.get_lr()
        new_lr = max(current_lr * 0.999, self.min_lr)  # Decay by 0.1% each epoch
        self.set_lr(new_lr)
    
    def _cosine_lr(self):
        """Cosine annealing"""
        epoch = self.model.train_state.epoch
        total_epochs = 10000  # You can make this configurable
        
        cosine_factor = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
        new_lr = max(self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor, self.min_lr)
        self.set_lr(new_lr)
    
    def _loss_based_lr(self, current_loss):
        """Adjust LR based on loss magnitudes"""
        if current_loss > 1e6:
            target_lr = 1e-6
        elif current_loss > 1e3:
            target_lr = 5e-5
        elif current_loss > 1e0:
            target_lr = 1e-4
        else:
            target_lr = 5e-4
            
        target_lr = max(target_lr, self.min_lr)
        self.set_lr(target_lr)
    
    def get_lr(self):
        """Get current learning rate"""
        if hasattr(self.model.opt, 'param_groups'):
            return self.model.opt.param_groups[0]['lr']
        else:
            return self.model.opt.learning_rate
    
    def set_lr(self, new_lr):
        """Set new learning rate"""
        if hasattr(self.model.opt, 'param_groups'):
            for param_group in self.model.opt.param_groups:
                param_group['lr'] = new_lr
        else:
            self.model.opt.learning_rate = new_lr