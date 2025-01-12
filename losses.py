# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float, gamma: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'soft_fd']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        student_hidden_list = None
        if self.distillation_type == 'soft_fd' and not isinstance(outputs, torch.Tensor):
            outputs, student_hidden_list = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if self.distillation_type == 'soft_fd' and student_hidden_list is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, [Tensor, Tensor....]] with the output of the "
                             "class_token and the list of intermediate outputs")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs, return_intermediate = (self.distillation_type == 'soft_fd'))
            
        teacher_hidden_list = None
        if self.distillation_type == 'soft_fd' and not isinstance(teacher_outputs , torch.Tensor):
            teacher_outputs, teacher_hidden_list  = teacher_outputs


        T = self.tau
        # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
        # with slight modifications
        distillation_loss = F.kl_div(
            F.log_softmax(outputs / T, dim=1),
            #We provide the teacher's targets in log probability because we use log_target=True 
            #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
            #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
            F.log_softmax(teacher_outputs / T, dim=1),
            reduction='sum',
            log_target=True
        ) * (T * T) / outputs.numel()

        if self.distillation_type == 'soft':
            return base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        
        # calculate hidden loss
        layer_num = len(student_hidden_list)
        hidden_loss = 0.
        for student_hidden, teacher_hidden in zip(student_hidden_list, teacher_hidden_list):
            hidden_loss +=  torch.nn.MSELoss()(student_hidden, teacher_hidden)
        hidden_loss /= layer_num
        
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha + self.gamma * hidden_loss
        return loss
