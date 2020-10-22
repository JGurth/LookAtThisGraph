from torch_geometric.data import DataLoader
from lookatthisgraph.utils.trainer import Trainer




class LTrainer(Trainer):
    'Needs to be used with LDataset'    
    
    
    def _get_loaders(self):
        train_loader = DataLoader(self.dataset, self._batch_size, drop_last=True, shuffle=True)
        val_loader = DataLoader(self.dataset.val_list, self._batch_size, drop_last=True)
        test_loader = DataLoader(self.dataset.test_list, self._batch_size, drop_last=True)
        
        return train_loader, val_loader, test_loader