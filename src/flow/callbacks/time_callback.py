import time
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info

__all__ = ['TimeCallback']

class TimeCallback(Callback):
  def __init__(
    self,
    timelimit_in_min = 1320, ##22h 1320
    verbose = True,
    max_epoch_count = -1
  ):
    super().__init__()
    
    self.verbose = verbose
    self.timelimit_in_min = timelimit_in_min
    self.epoch = -1
    self.time_buffer = time.time()
    self.training_start_epoch = 0
    self.max_epoch_count = max_epoch_count
    
    if self.verbose:
      rank_zero_info(f'TimeLimitCallback is set to {self.timelimit_in_min}min')

  def on_validation_end(self, trainer, pl_module):
    if trainer.running_sanity_check:
      return
    self._run_early_stopping_check(trainer, pl_module)

  def on_validation_epoch_end(self, trainer, pl_module):
    # trainer.callback_metrics['task_count/dataloader_idx_0']
    if trainer.running_sanity_check:
      return

  def on_train_start(self, trainer, pl_module):
    """Called when the train begins."""
    # set task start time
    self.time_buffer = time.time()
    
    self.training_start_epoch = pl_module.current_epoch
      
  def _run_early_stopping_check(self, trainer, pl_module):
    should_stop = False
    
    if  self.epoch != trainer.current_epoch:
      self.epoch = trainer.current_epoch
      # check time 
      if ((time.time() - self.time_buffer)/60 > self.timelimit_in_min or 
        ( self.max_epoch_count != -1 and self.epoch - self.training_start_epoch > self.max_epoch_count )):
        # time limit reached
        should_stop = True
        rank_zero_info('STOPPED due to timelimit reached!')

      if bool(should_stop):
          self.stopped_epoch = trainer.current_epoch
          trainer.should_stop = True
          # stop every ddp process if any world process decides to stop
          trainer.should_stop = trainer.training_type_plugin.reduce_boolean_decision(trainer.should_stop)
          
      if self.verbose:
        string = 'Callback State\n'
        string += f'Trainger should stop: {should_stop} ' + str(int( (time.time() - self.time_buffer)/60 ))
        rank_zero_info(string)
    else:
      if self.verbose:
        rank_zero_info('Visited twice at same epoch')