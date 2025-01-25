import wandb
import torch.cuda as cuda
from parametric_umap import ParametricUMAP
from functools import wraps
from tqdm.auto import tqdm

class UMAPMonitor:
    def __init__(self, use_wandb=False, wandb_project=None, wandb_run_name=None):
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=wandb_project, name=wandb_run_name)
        self.epoch_losses = []
        self.batch_losses = []
        self.gpu_memory = []
        self.current_epoch = 0
        
    def on_batch_end(self, batch_idx, postfix_dict):
        # Extract loss from tqdm postfix
        if 'loss' in postfix_dict:
            loss = float(postfix_dict['loss'])
            memory = cuda.memory_allocated()/1e9 if cuda.is_available() else 0
            
            self.batch_losses.append({
                'epoch': self.current_epoch,
                'batch': batch_idx,
                'loss': loss,
                'gpu_memory': memory
            })
            
            if self.use_wandb:
                wandb.log({
                    'batch_loss': loss,
                    'batch_gpu_memory': memory,
                    'batch': batch_idx,
                    'epoch': self.current_epoch
                })
        
    def on_epoch_end(self, epoch, loss):
        self.current_epoch = epoch
        self.epoch_losses.append(loss)
        if cuda.is_available():
            memory = cuda.memory_allocated()/1e9
            peak_memory = cuda.max_memory_allocated()/1e9
            self.gpu_memory.append({'current': memory, 'peak': peak_memory})
            print(f"GPU Memory: {memory:.2f} GB (Peak: {peak_memory:.2f} GB)")
        
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
        
        if self.use_wandb:
            metrics = {
                'epoch': epoch,
                'epoch_loss': loss,
            }
            if cuda.is_available():
                metrics.update({
                    'gpu_memory': memory,
                    'peak_gpu_memory': peak_memory
                })
            wandb.log(metrics)

    def __del__(self):
        if self.use_wandb:
            wandb.finish()

class MonitoredParametricUMAP(ParametricUMAP):
    def fit(self, X, monitor=None, *args, **kwargs):
        original_fit = super().fit
        
        @wraps(original_fit)
        def monitored_fit(X, *args, **kwargs):
            # Store the original verbose setting and force it to True to capture loss
            original_verbose = kwargs.get('verbose', True)
            kwargs['verbose'] = True
            
            # Patch tqdm to capture batch-level progress
            original_tqdm = tqdm.__init__
            
            def custom_tqdm_init(self, *args, **kwargs):
                original_tqdm(self, *args, **kwargs)
                self._original_update = self.update
                
                def update_and_monitor(*args, **kwargs):
                    self._original_update(*args, **kwargs)
                    if monitor is not None and hasattr(self, 'postfix'):
                        if isinstance(self.postfix, dict):
                            monitor.on_batch_end(self.n, self.postfix)
                
                self.update = update_and_monitor
            
            tqdm.__init__ = custom_tqdm_init
            
            # Patch print to capture epoch-level metrics
            original_print = print
            def custom_print(*args, **kwargs):
                message = ' '.join(str(arg) for arg in args)
                if 'Loss:' in message and monitor is not None:
                    try:
                        epoch = int(message.split('/')[0].split()[-1]) - 1
                        loss = float(message.split('Loss:')[-1].strip())
                        monitor.on_epoch_end(epoch, loss)
                    except (ValueError, IndexError):
                        pass
                if original_verbose:
                    original_print(*args, **kwargs)
            
            import builtins
            builtins.print = custom_print
            
            try:
                result = original_fit(X, *args, **kwargs)
            finally:
                # Restore original functions
                builtins.print = original_print
                tqdm.__init__ = original_tqdm
            
            return result
            
        return monitored_fit(X, *args, **kwargs)