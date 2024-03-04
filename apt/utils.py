import os
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple, TypeVar
from torch import Tensor

from clip import clip
from trainers.apt import PromptLearner, TextEncoder


mu = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)

class ImageNormalizer(nn.Module):

    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std

    def __repr__(self):
        return f'ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})'  # type: ignore


class CustomCLIP(nn.Module):
    def __init__(self,
                 model,
                 classnames,
                 cls_prompt='a photo of a {}',
                 atk_prompt=None,
                 cfg=None):
        super().__init__()

        self.cfg = cfg
        self.logit_scale = model.logit_scale
        self.classnames = classnames
        self.model = model
        self.mode = 'classification'
        
        self.normalize = ImageNormalizer(mu, std).cuda()
        
        self.set_prompts(cls_prompt, atk_prompt)
        
    def _prompt_text_features(self, prompt):
        if '{}' in prompt:
            # manual prompt template
            prompts = torch.cat([clip.tokenize(prompt.format(c))
                                 for c in self.classnames])
            self.model = self.model
            text_features = self.model.encode_text(prompts)
        else:
            # optimized prompt vector
            prompter_ckp = prompt
            assert os.path.isfile(prompter_ckp)
            prompter = PromptLearner(self.cfg, self.classnames, self.model)
            
            state_dict = torch.load(prompter_ckp)["state_dict"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            prompter.load_state_dict(state_dict, strict=False)
            text_encoder = TextEncoder(self.model)
            prompts = prompter()
            text_features = text_encoder(prompts, prompter.tokenized_prompts)
            
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach()
        
    def set_prompts(self, cls_prompt, atk_prompt=None):
        print(f'classification prompt: {cls_prompt}')
        self.cls_tfeatures = self._prompt_text_features(cls_prompt).cuda()
        
        if atk_prompt is None or cls_prompt == atk_prompt:
            print(f'attack prompt: {cls_prompt}')
            self.atk_tfeatures = self.cls_tfeatures
        else:
            print(f'attack prompt: {atk_prompt}')
            self.atk_tfeatures = self._prompt_text_features(atk_prompt).cuda()
            
    def forward(self, image):
        image_features = self.model.encode_image(self.normalize(image))        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        text_features = self.cls_tfeatures if self.mode == 'classification' else self.atk_tfeatures
        logits = logit_scale * image_features @ text_features.t()
        
        return logits


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
