import deepxde as dde

from utils.MHNN import MultiHeadNN

class SeperateResidualCallback(dde.callbacks.Callback):
    def __init__(self, pde_indices, epoch_etappen):

        super().__init__()
        if not isinstance(pde_indices, list):
            pde_indices = [pde_indices]
        if len(pde_indices) != len(epoch_etappen):
            raise ValueError("pde_indices and epoch_etappen must have the same length.") 

        self.pde_indices = pde_indices
        self.epoch_etappen = epoch_etappen


    def on_train_begin(self):
        """Called at the beginning of training to store initial weights."""
        self.initial_weights = list(self.model.loss_weights)

def on_epoch_end(self):
    epoch = self.model.train_state.epoch
    current_weights = list(self.model.loss_weights)

    # Set all PDE weights to 0.0 by default
    for idx in self.pde_indices:
        current_weights[idx] = 0.0

    # Activate PDEs according to epoch_etappen
    for i, etappe_epoch in enumerate(self.epoch_etappen):
        if epoch >= etappe_epoch:
            print(f"Activating PDE index {self.pde_indices[i]} at epoch {epoch}")
            current_weights[self.pde_indices[i]] = 1.0

    self.model.loss_weights = current_weights

    if epoch > 0 and epoch % 100 == 0:
        print(f"\nEpoch {epoch}: Active PDE indices {[self.pde_indices[i] for i, e in enumerate(self.epoch_etappen) if epoch >= e]}")



class MultiHeadSchedulerCallback(dde.callbacks.Callback):
    """
    A callback to manage training of a MultiHeadNN by scheduling which heads
    are active (trainable) and their corresponding PDE loss weights over epochs.
    This version supports one-to-many mapping from heads to PDE losses.
    """
    def __init__(self, schedule, head_to_pde_map):
        """
        Args:
            schedule (list of dict): A list defining the training stages.
                Each dict must contain:
                - 'epoch' (int): The epoch at which this stage begins.
                - 'active_heads' (list of int): Indices of the network heads to train.
            head_to_pde_map (list of lists): Maps head index to a list of PDE loss
                indices. E.g., [[0, 1], [2]] means head 0 is linked to PDE losses
                0 and 1, and head 1 is linked to PDE loss 2.
        """
        super().__init__()
        self.schedule = sorted(schedule, key=lambda s: s['epoch'])
        self.head_to_pde_map = head_to_pde_map
        self.current_stage_idx = -1
        self.net = None
        self.num_heads = None

    def on_train_begin(self):
        """Called at the beginning of training to initialize and validate."""
        self.net = self.model.net
        if not isinstance(self.net, MultiHeadNN):
            raise TypeError("The model's network must be an instance of MultiHeadNN.")
        
        self.num_heads = len(self.net.heads)
        if self.num_heads != len(self.head_to_pde_map):
            raise ValueError("The number of heads in the net must match the length of head_to_pde_map.")

        print("--- MultiHeadScheduler Initialized ---")
        self._update_training_stage(epoch=0)

    def on_epoch_end(self):
        """Called at the end of each epoch to check if a new stage should begin."""
        epoch = self.model.train_state.epoch
        self._update_training_stage(epoch)

    def _update_training_stage(self, epoch):
        """
        Checks the current epoch against the schedule and updates the training
        state (frozen heads and loss weights) if necessary.
        """
        new_stage_idx = -1
        for i, stage in enumerate(self.schedule):
            if epoch >= stage['epoch']:
                new_stage_idx = i
            else:
                break
        
        if new_stage_idx != self.current_stage_idx:
            self.current_stage_idx = new_stage_idx
            stage_info = self.schedule[self.current_stage_idx]
            active_heads = stage_info['active_heads']
            
            print(f"\n--- Epoch {epoch}: Entering Training Stage {self.current_stage_idx} ---")
            print(f"    - Active Heads: {active_heads}")
            
            self._update_head_freezing(active_heads)
            self._update_loss_weights(active_heads)

    def _update_head_freezing(self, active_heads):
        """Freezes or unfreezes network heads based on the schedule."""
        for i in range(self.num_heads):
            head_is_active = i in active_heads
            for param in self.net.heads[i].parameters():
                param.requires_grad = head_is_active
            status = "UNFROZEN" if head_is_active else "FROZEN"
            print(f"    - Head {i}: {status}")

    def _update_loss_weights(self, active_heads):
        """Updates the model's loss weights based on the active heads."""
        current_weights = self.model.loss_weights
        if not isinstance(current_weights, list):
            current_weights = current_weights.tolist()

        # Deactivate all managed PDE losses first
        all_managed_pdes = [pde_idx for pde_list in self.head_to_pde_map for pde_idx in pde_list]
        for pde_index in all_managed_pdes:
            if pde_index < len(current_weights):
                current_weights[pde_index] = 0.0
        
        # Activate PDE losses corresponding to active heads
        active_pde_indices = []
        for head_index in active_heads:
            pde_indices_to_activate = self.head_to_pde_map[head_index]
            for pde_index in pde_indices_to_activate:
                 if pde_index < len(current_weights):
                    current_weights[pde_index] = 1.0
                    active_pde_indices.append(pde_index)
            
        print(f"    - Activating PDE loss indices: {sorted(list(set(active_pde_indices)))}")
        self.model.loss_weights = current_weights
