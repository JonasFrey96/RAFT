import torch
import torch.nn as nn
import torch.nn.functional as F

from models_asl import FastSCNN
from raft import RAFT
from src_utils import DotDict
from torchvision import transforms

__all__ = ['Inferencer']

class Inferencer(nn.Module):
  def __init__(self, exp, env):
    super(Inferencer, self).__init__()
    self._exp = exp
    self._env = env
    self.model = RAFT(args = DotDict(self._exp['model']['args']) )
    self.seg = FastSCNN(**self._exp['seg']['cfg'])
    self.output_transform_seg = transforms.Compose([
          transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    self.seg.eval()
    self.model.eval()

  @torch.no_grad()
  def forward(self, batch):
    # FLOW
    flow_predictions = self.model(batch[0], batch[1], iters=self._exp['model']['iters'])
    
    inp = torch.cat ( [self.output_transform_seg(batch[0]/255.0),
      self.output_transform_seg(batch[1]/255.0 ) ],dim=1)
    # SEG
    outputs = self.seg(inp)
    probs = torch.nn.functional.softmax(outputs[0], dim=1)
    pred_valid = torch.argmax( probs, dim = 1)
    
    return flow_predictions, pred_valid