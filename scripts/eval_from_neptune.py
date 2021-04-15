from pytorch_lightning.loggers.neptune import NeptuneLogger
import os
logger = NeptuneLogger(
          api_key=os.environ["NEPTUNE_API_TOKEN"],
          project_name="jonasfrey96/rpose",
          experiment_id='RPOS-137',
          close_after_fit = False,
        )
print(logger.experiment.id)

logger.experiment



add_s = logger.experiment.get_numeric_channels_values("test_add_s_h_pred_step/epoch_0","test_add_s_h_init_step/epoch_0" )
adds = logger.experiment.get_numeric_channels_values("test_adds_h_pred_step/epoch_0","test_adds_h_init_step/epoch_0" )


res = logger.experiment.get_channels()


# task_numbers = logger.experiment.get_numeric_channels_values('task_count/dataloader_idx_0')['task_count/dataloader_idx_0']
    

import time
time.sleep(1)



