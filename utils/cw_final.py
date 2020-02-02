"""
Carlini-Wagner attack (http://arxiv.org/abs/1608.04644).

Referential implementation:

- https://github.com/carlini/nn_robust_attacks.git (the original implementation)
- https://github.com/rwightman/pytorch-nips2017-attack-example.git
"""
import operator as op

from typing import Union, Tuple
from numba import jit
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from skimage import feature
from skimage import color
from skimage import io

import utils.runutils as runutils


def _var2numpy(var):
    """
    Make Variable to numpy array. No transposition will be made.

    :param var: Variable instance on whatever device
    :type var: Variable
    :return: the corresponding numpy array
    :rtype: np.ndarray
    """
    return var.data.cpu().numpy()


def atanh(x, eps=1e-6):
    """
    The inverse hyperbolic tangent function, missing in pytorch.

    :param x: a tensor or a Variable
    :param eps: used to enhance numeric stability
    :return: :math:`\\tanh^{-1}{x}`, of the same type as ``x``
    """
    x = x * (1 - eps)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))

def to_tanh_space(x, box):
    # type: (Union[Variable, torch.FloatTensor], Tuple[float, float]) -> Union[Variable, torch.FloatTensor]
    """
    Convert a batch of tensors to tanh-space. This method complements the
    implementation of the change-of-variable trick in terms of tanh.

    :param x: the batch of tensors, of dimension [B x C x H x W]
    :param box: a tuple of lower bound and upper bound of the box constraint
    :return: the batch of tensors in tanh-space, of the same dimension;
             the returned tensor is on the same device as ``x``
    """
    _box_mul = (box[1] - box[0]) * 0.5
    _box_plus = (box[1] + box[0]) * 0.5
    return atanh((x - _box_plus) / _box_mul)

def from_tanh_space(x, box):
    # type: (Union[Variable, torch.FloatTensor], Tuple[float, float]) -> Union[Variable, torch.FloatTensor]
    """
    Convert a batch of tensors from tanh-space to oridinary image space.
    This method complements the implementation of the change-of-variable trick
    in terms of tanh.

    :param x: the batch of tensors, of dimension [B x C x H x W]
    :param box: a tuple of lower bound and upper bound of the box constraint
    :return: the batch of tensors in ordinary image space, of the same
             dimension; the returned tensor is on the same device as ``x``
    """
    _box_mul = (box[1] - box[0]) * 0.5
    _box_plus = (box[1] + box[0]) * 0.5
    return torch.tanh(x) * _box_mul + _box_plus


class L2Adversary(object):
    """
    The L2 attack adversary. To enforce the box constraint, the
    change-of-variable trick using tanh-space is adopted.

    The loss function to optimize:

    .. math::
        \\|\\delta\\|_2^2 + c \\cdot f(x + \\delta)

    where :math:`f` is defined as

    .. math::
        f(x') = \\max\\{0, (\\max_{i \\ne t}{Z(x')_i} - Z(x')_t) \\cdot \\tau + \\kappa\\}

    where :math:`\\tau` is :math:`+1` if the adversary performs targeted attack;
    otherwise it's :math:`-1`.

    Usage::

        attacker = L2Adversary()
        # inputs: a batch of input tensors
        # targets: a batch of attack targets
        # model: the model to attack
        advx = attacker(model, inputs, targets)


    The change-of-variable trick
    ++++++++++++++++++++++++++++

    Let :math:`a` be a proper affine transformation.

    1. Given input :math:`x` in image space, map :math:`x` to "tanh-space" by

    .. math:: \\hat{x} = \\tanh^{-1}(a^{-1}(x))

    2. Optimize an adversarial perturbation :math:`m` without constraint in the
    "tanh-space", yielding an adversarial example :math:`w = \\hat{x} + m`; and

    3. Map :math:`w` back to the same image space as the one where :math:`x`
    resides:

    .. math::
        x' = a(\\tanh(w))

    where :math:`x'` is the adversarial example, and :math:`\\delta = x' - x`
    is the adversarial perturbation.

    Since the composition of affine transformation and hyperbolic tangent is
    strictly monotonic, $\\delta = 0$ if and only if $m = 0$.

    Symbols used in docstring
    +++++++++++++++++++++++++

    - ``B``: the batch size
    - ``C``: the number of channels
    - ``H``: the height
    - ``W``: the width
    - ``M``: the number of classes
    """

    def __init__(self, targeted=True, confidence=0.0, c_range=(1e-3, 1e10),
                 search_steps=5, max_steps=10000, abort_early=True,
                 box=(-1., 1.), optimizer_lr=1e-2, init_rand=False):
        """
        :param targeted: ``True`` to perform targeted attack in ``self.run``
               method
        :type targeted: bool
        :param confidence: the confidence constant, i.e. the $\\kappa$ in paper
        :type confidence: float
        :param c_range: the search range of the constant :math:`c`; should be a
               tuple of form (lower_bound, upper_bound)
        :type c_range: Tuple[float, float]
        :param search_steps: the number of steps to perform binary search of
               the constant :math:`c` over ``c_range``
        :type search_steps: int
        :param max_steps: the maximum number of optimization steps for each
               constant :math:`c`
        :type max_steps: int
        :param abort_early: ``True`` to abort early in process of searching for
               :math:`c` when the loss virtually stops increasing
        :type abort_early: bool
        :param box: a tuple of lower bound and upper bound of the box
        :type box: Tuple[float, float]
        :param optimizer_lr: the base learning rate of the Adam optimizer used
               over the adversarial perturbation in clipped space
        :type optimizer_lr: float
        :param init_rand: ``True`` to initialize perturbation to small Gaussian;
               False is consistent with the original paper, where the
               perturbation is initialized to zero
        :type init_rand: bool
        :rtype: None

        Why to make ``box`` default to (-1., 1.) rather than (0., 1.)? TL;DR the
        domain of the problem in pytorch is [-1, 1] instead of [0, 1].
        According to Xiang Xu (samxucmu@gmail.com)::

        > The reason is that in pytorch a transformation is applied first
        > before getting the input from the data loader. So image in range [0,1]
        > will subtract some mean and divide by std. The normalized input image
        > will now be in range [-1,1]. For this implementation, clipping is
        > actually performed on the image after normalization, not on the
        > original image.

        Why to ``optimizer_lr`` default to 1e-2? The optimizer used in Carlini's
        code adopts 1e-2. In another pytorch implementation
        (https://github.com/rwightman/pytorch-nips2017-attack-example.git),
        though, the learning rate is set to 5e-4.
        """
        if len(c_range) != 2:
            raise TypeError('c_range ({}) should be of form '
                            'tuple([lower_bound, upper_bound])'
                            .format(c_range))
        if c_range[0] >= c_range[1]:
            raise ValueError('c_range lower bound ({}) is expected to be less '
                             'than c_range upper bound ({})'.format(*c_range))
        if len(box) != 2:
            raise TypeError('box ({}) should be of form '
                            'tuple([lower_bound, upper_bound])'
                            .format(box))
        if box[0] >= box[1]:
            raise ValueError('box lower bound ({}) is expected to be less than '
                             'box upper bound ({})'.format(*box))
        self.targeted = targeted
        self.confidence = float(confidence)
        self.c_range = (float(c_range[0]), float(c_range[1]))
        self.binary_search_steps = search_steps
        self.max_steps = max_steps
        self.abort_early = abort_early
        self.ae_tol = 1e-4  # tolerance of early abort
        self.box = tuple(map(float, box))  # type: Tuple[float, float]
        self.optimizer_lr = optimizer_lr

        # `self.init_rand` is not in Carlini's code, it's an attempt in the
        # referencing pytorch implementation to improve the quality of attacks.
        self.init_rand = init_rand

        # Since the larger the `scale_const` is, the more likely a successful
        # attack can be found, `self.repeat` guarantees at least attempt the
        # largest scale_const once. Moreover, since the optimal criterion is the
        # L2 norm of the attack, and the larger `scale_const` is, the larger
        # the L2 norm is, thus less optimal, the last attempt at the largest
        # `scale_const` won't ruin the optimum ever found.
        self.repeat = (self.binary_search_steps >= 10)

    def __call__(self, model, detectors, inputs, targets, to_numpy=True):
        """
        Produce adversarial examples for ``inputs``.

        :param model: the model to attack
        :type model: nn.Module
        :param inputs: the original images tensor, of dimension [B x C x H x W].
               ``inputs`` can be on either CPU or GPU, but it will eventually be
               moved to the same device as the one the parameters of ``model``
               reside
        :type inputs: torch.FloatTensor
        :param targets: the original image labels, or the attack targets, of
               dimension [B]. If ``self.targeted`` is ``True``, then ``targets``
               is treated as the attack targets, otherwise the labels.
               ``targets`` can be on either CPU or GPU, but it will eventually
               be moved to the same device as the one the parameters of
               ``model`` reside
        :type targets: torch.LongTensor
        :param to_numpy: True to return an `np.ndarray`, otherwise,
               `torch.FloatTensor`
        :type to_numpy: bool
        :return: the adversarial examples on CPU, of dimension [B x C x H x W]
        """
        device = torch.device("cuda")

	# sanity check
        assert isinstance(model, nn.Module)
        assert len(inputs.size()) == 4
        assert len(targets.size()) == 1

        # get a copy of targets in numpy before moving to GPU, used when doing
        # the binary search on `scale_const`
        targets_np = targets.clone().cpu().numpy()  # type: np.ndarray

        # the type annotations here are used only for type hinting and do
        # not indicate the actual type (cuda or cpu); same applies to all codes
        # below
        inputs = runutils.make_cuda_consistent(model, inputs)[0]  # type: torch.FloatTensor
        targets = runutils.make_cuda_consistent(model, targets)[0]  # type: torch.FloatTensor

        # run the model a little bit to get the `num_classes`
        num_classes = model(Variable(inputs[0][None, :], requires_grad=False)).size(1)  # type: int
        batch_size = inputs.size(0)  # type: int

        # `lower_bounds_np`, `upper_bounds_np` and `scale_consts_np` are used
        # for binary search of each `scale_const` in the batch. The element-wise
        # inquality holds: lower_bounds_np < scale_consts_np <= upper_bounds_np
        lower_bounds_np = np.zeros(batch_size)
        upper_bounds_np = np.ones(batch_size) * self.c_range[1]
        scale_consts_np = np.ones(batch_size) * self.c_range[0]

        # Optimal attack to be found.
        # The three "placeholders" are defined as:
        # - `o_best_l2`: the least L2 norms
        # - `o_best_l2_ppred`: the perturbed predictions made by the adversarial
        #    perturbations with the least L2 norms
        # - `o_best_advx`: the underlying adversarial example of
        #   `o_best_l2_ppred`
        o_best_l2 = np.ones(batch_size) * np.inf
        o_best_l2_ppred = -np.ones(batch_size)
        o_best_advx = inputs.clone().cpu().numpy()  # type: np.ndarray

        # convert `inputs` to tanh-space
        inputs_tanh = self._to_tanh_space(inputs)  # type: torch.FloatTensor
        inputs_tanh_var = Variable(inputs_tanh, requires_grad=False)

        # the one-hot encoding of `targets`
        targets_oh = torch.zeros(targets.size() + (num_classes,)).to(device)  # type: torch.FloatTensor
        targets_oh = runutils.make_cuda_consistent(model, targets_oh)[0]
        targets_oh.scatter_(1, targets.unsqueeze(1), 1.0)
        targets_oh_var = Variable(targets_oh, requires_grad=False)

        # the perturbation variable to optimize.
        # `pert_tanh` is essentially the adversarial perturbation in tanh-space.
        # In Carlini's code it's denoted as `modifier`
        pert_tanh = torch.zeros(inputs.size()).to(device)  # type: torch.FloatTensor
        if self.init_rand:
            nn.init.normal(pert_tanh, mean=0, std=1e-3)
        pert_tanh = runutils.make_cuda_consistent(model, pert_tanh)[0]
        pert_tanh_var = Variable(pert_tanh, requires_grad=True)

        optimizer = optim.Adam([pert_tanh_var], lr=self.optimizer_lr)
        for sstep in range(self.binary_search_steps):
            if self.repeat and sstep == self.binary_search_steps - 1:
                scale_consts_np = upper_bounds_np
            scale_consts = torch.from_numpy(np.copy(scale_consts_np)).float().to(device)  # type: torch.FloatTensor
            scale_consts = runutils.make_cuda_consistent(model, scale_consts)[0]
            scale_consts_var = Variable(scale_consts, requires_grad=False)
            #print('Using scale consts:', list(scale_consts_np))  # FIXME

            # the minimum L2 norms of perturbations found during optimization
            best_l2 = np.ones(batch_size) * np.inf
            # the perturbed predictions corresponding to `best_l2`, to be used
            # in binary search of `scale_const`
            best_l2_ppred = -np.ones(batch_size)
            # previous (summed) batch loss, to be used in early stopping policy
            prev_batch_loss = np.inf  # type: float
            for optim_step in range(self.max_steps):
                #start = torch.cuda.Event(enable_timing=True)
                #end = torch.cuda.Event(enable_timing=True)
                
                #start.record()
                batch_loss, pert_norms_np, pert_outputs_np, advxs_np, out_detector = \
                    self._optimize(model, detectors, optimizer, inputs_tanh_var,
                                   pert_tanh_var, targets_oh_var,
                                   scale_consts_var)
                #if optim_step % 10 == 0: print('batch [{}] loss: {}'.format(optim_step, batch_loss))  # FIXME
                #end.record()
                #torch.cuda.synchronize()
                #print(start.elapsed_time(end))

                if self.abort_early and not optim_step % (self.max_steps // 10):
                    if batch_loss > prev_batch_loss * (1 - self.ae_tol):
                        break
                    prev_batch_loss = batch_loss

                # update best attack found during optimization
                pert_predictions_np = np.argmax(pert_outputs_np, axis=1)
                comp_pert_predictions_np = np.argmax(
                        self._compensate_confidence(pert_outputs_np,
                                                    targets_np),
                        axis=1)
                for i in range(batch_size):
                    l2 = pert_norms_np[i]
                    cppred = comp_pert_predictions_np[i]
                    ppred = pert_predictions_np[i]
                    tlabel = targets_np[i]
                    ax = advxs_np[i]
                    o_detect = out_detector[i]
                    if self._attack_successful(cppred, tlabel, o_detect):
                        #print("Attack Success")
                        
                        assert cppred == ppred
                        if l2 < best_l2[i]:
                            best_l2[i] = l2
                            best_l2_ppred[i] = ppred
                        if l2 < o_best_l2[i]:
                            #print(l2)
                            o_best_l2[i] = l2
                            o_best_l2_ppred[i] = ppred
                            o_best_advx[i] = ax.cpu().numpy()
                            #print(out_detector)
                            #print(detector(torch.Tensor(feature.canny(ax.reshape(28, 28).numpy(), 
                             #                      sigma=1.8).astype(float)).reshape(1,784)))

            # binary search of `scale_const`
            for i in range(batch_size):
                tlabel = targets_np[i]
                assert best_l2_ppred[i] == -1 or \
                       self._attack_successful(best_l2_ppred[i], tlabel)
                assert o_best_l2_ppred[i] == -1 or \
                       self._attack_successful(o_best_l2_ppred[i], tlabel)
                if best_l2_ppred[i] != -1:
                    # successful; attempt to lower `scale_const` by halving it
                    if scale_consts_np[i] < upper_bounds_np[i]:
                        upper_bounds_np[i] = scale_consts_np[i]
                    # `upper_bounds_np[i] == c_range[1]` implies no solution
                    # found, i.e. upper_bounds_np[i] has never been updated by
                    # scale_consts_np[i] until
                    # `scale_consts_np[i] > 0.1 * c_range[1]`
                    if upper_bounds_np[i] < self.c_range[1] * 0.1:
                        scale_consts_np[i] = (lower_bounds_np[i] + upper_bounds_np[i]) / 2
                else:
                    # failure; multiply `scale_const` by ten if no solution
                    # found; otherwise do binary search
                    if scale_consts_np[i] > lower_bounds_np[i]:
                        lower_bounds_np[i] = scale_consts_np[i]
                    if upper_bounds_np[i] < self.c_range[1] * 0.1:
                        scale_consts_np[i] = (lower_bounds_np[i] + upper_bounds_np[i]) / 2
                    else:
                        scale_consts_np[i] *= 10

        if not to_numpy:
            o_best_advx = torch.from_numpy(o_best_advx).float()
        #if o_best_l2 == np.inf:
         #   print("Attack Failed!")
        #else:
            #print("before ", detector(torch.Tensor(o_best_advx).reshape(1, 784)))
        return o_best_advx, o_best_l2

    def _optimize(self, model, detectors, optimizer, inputs_tanh_var, pert_tanh_var,
                  targets_oh_var, c_var):
        """
        Optimize for one step.

        :param model: the model to attack
        :type model: nn.Module
        :param optimizer: the Adam optimizer to optimize ``modifier_var``
        :type optimizer: optim.Adam
        :param inputs_tanh_var: the input images in tanh-space
        :type inputs_tanh_var: Variable
        :param pert_tanh_var: the perturbation to optimize in tanh-space,
               ``pert_tanh_var.requires_grad`` flag must be set to True
        :type pert_tanh_var: Variable
        :param targets_oh_var: the one-hot encoded target tensor (the attack
               targets if self.targeted else image labels)
        :type targets_oh_var: Variable
        :param c_var: the constant :math:`c` for each perturbation of a batch,
               a Variable of FloatTensor of dimension [B]
        :type c_var: Variable
        :return: the batch loss, squared L2-norm of adversarial perturbations
                 (of dimension [B]), the perturbed activations (of dimension
                 [B]), the adversarial examples (of dimension [B x C x H x W])
        """
        batch_size = inputs_tanh_var.size(0)
        device = torch.device("cuda")

        # the adversarial examples in the image space
        # of dimension [B x C x H x W]
        advxs_var = self._from_tanh_space(inputs_tanh_var + pert_tanh_var)  # type: Variable  
        
        #start = time.time()
        # input to detector
        if inputs_tanh_var.size(1) == 3:
            inp_detector = torch.zeros((batch_size, 1024)).to(device)
            for i in range(batch_size):
                img = color.rgb2gray(data[i].permute(1,2,0).numpy())
                inp_detector[i] = torch.Tensor(feature.canny(adv_im, 
                                                       sigma=1.8).astype(float)).reshape(1024)
        else:    
            inp_detector = torch.zeros((batch_size, 784)).to(device)
            for i in range(batch_size):
                inp_detector[i] = torch.Tensor(feature.canny(advxs_var[i].reshape(28, 28).detach().cpu().numpy(), 
                                                       sigma=1.8).astype(float)).reshape(784).to(device)
        #end = time.time()
        #print(end-start)
        # the perturbed activation before softmax
        pert_outputs_var = model(advxs_var)  # type: Variable
        # the original inputs
        inputs_var = self._from_tanh_space(inputs_tanh_var)  # type: Variable

        perts_norm_var = torch.pow(advxs_var - inputs_var, 2)
        perts_norm_var = torch.sum(perts_norm_var.view(
                perts_norm_var.size(0), -1), 1)

        # In Carlini's code, `target_activ_var` is called `real`.
        # It should be a Variable of tensor of dimension [B], such that the
        # `target_activ_var[i]` is the final activation (right before softmax)
        # of the $t$th class, where $t$ is the attack target or the image label
        #
        # noinspection PyArgumentList
        target_activ_var = torch.sum(targets_oh_var * pert_outputs_var, 1)
        inf = 1e4  # sadly pytorch does not work with np.inf;
                   # 1e4 is also used in Carlini's code
        # In Carlini's code, `maxother_activ_var` is called `other`.
        # It should be a Variable of tensor of dimension [B], such that the
        # `maxother_activ_var[i]` is the maximum final activation of all classes
        # other than class $t$, where $t$ is the attack target or the image
        # label.
        #
        # The assertion here ensures (sufficiently yet not necessarily) the
        # assumption behind the trick to get `maxother_activ_var` holds, that
        # $\max_{i \ne t}{o_i} \ge -\text{_inf}$, where $t$ is the target and
        # $o_i$ the $i$th element along axis=1 of `pert_outputs_var`.
        #
        # noinspection PyArgumentList
        assert (pert_outputs_var.max(1)[0] >= -inf).all(), 'assumption failed'
        # noinspection PyArgumentList
        maxother_activ_var = torch.max(((1 - targets_oh_var) * pert_outputs_var
                                        - targets_oh_var * inf), 1)[0]
        maxother_activ_ind = torch.argmax(((1 - targets_oh_var) * pert_outputs_var
                                        - targets_oh_var * inf), 1)
        
        # Compute $f(x')$, where $x'$ is the adversarial example in image space.
        # The result `f_var` should be of dimension [B]
        if self.targeted:
            # if targeted, optimize to make `target_activ_var` larger than
            # `maxother_activ_var` by `self.confidence`
            #
            # noinspection PyArgumentList
            f_var = torch.clamp(maxother_activ_var - target_activ_var
                                + self.confidence, min=0.0)
            out_detector = torch.zeros(batch_size, 2).to(device)
            for i in range(batch_size):
                #print(targets_oh_var[i])
                ind = torch.argmax(targets_oh_var[i])
                #print(ind)
                #print(type(inp_detector[i]))
                #print(type(detectors[ind]))
                out_detector[i] = detectors[ind](inp_detector[i])
        else:
            # if not targeted, optimize to make `maxother_activ_var` larger than
            # `target_activ_var` (the ground truth image labels) by
            # `self.confidence`
            #
            # noinspection PyArgumentList
            f_var = torch.clamp(target_activ_var - maxother_activ_var
                                + self.confidence, min=0.0)
            out_detector = torch.zeros(batch_size, 2)
            for i in range(batch_size):
                max_other_target = maxother_activ_ind[i]
                #print(detectors[max_other_target](inp_detector[i]).size())
                out_detector[i] = detectors[max_other_target](inp_detector[i].reshape(1, 784))
          
        #print(out_detector)
        # high loss for detector if input belongs to true class
        detector_loss = torch.zeros(batch_size).to(device)
        for i in range(batch_size):
            #detector_loss[i] = (out_detector[i][0][1] - out_detector[i][0][0])
            detector_loss[i] = (out_detector[i][0] - out_detector[i][1])
            
        # the total loss of current batch, should be of dimension [1]
        batch_loss_var = torch.sum(perts_norm_var + c_var * (f_var + detector_loss))  # type: Variable

        # Do optimization for one step
        optimizer.zero_grad()
        batch_loss_var.backward()#retain_graph = True)
        # summing gradient for detector loss
        #detector_loss.backward()
        
        optimizer.step()

        # Make some records in python/numpy on CPU
        batch_loss = batch_loss_var.item()#.data[0]  # type: float
        pert_norms_np = _var2numpy(perts_norm_var)
        pert_outputs_np = _var2numpy(pert_outputs_var)
        advxs_np = advxs_var.detach()#_var2numpy(advxs_var)
        #print(advxs_np.dtype)
        #print(torch.Tensor(advxs_np).dtype)
        return batch_loss, pert_norms_np, pert_outputs_np, advxs_np, out_detector

    def _attack_successful(self, prediction, target, out_detector=torch.Tensor([1,0])):
        """
        See whether the underlying attack is successful.

        :param prediction: the prediction of the model on an input
        :type prediction: int
        :param target: either the attack target or the ground-truth image label
        :type target: int
        :return: ``True`` if the attack is successful
        :rtype: bool
        """
        #print(out_detector)
        if self.targeted:
            #return prediction == target and (out_detector[0][0] >= out_detector[0][1]).numpy().astype(bool)
            return prediction == target and (out_detector[0] >= out_detector[1])#.numpy().astype(bool)
        else:
            #return prediction != target and (out_detector[0][0] >= out_detector[0][1]).numpy().astype(bool)
            return prediction != target and (out_detector[0] >= out_detector[1]).numpy().astype(bool)

    # noinspection PyUnresolvedReferences
    def _compensate_confidence(self, outputs, targets):
        """
        Compensate for ``self.confidence`` and returns a new weighted sum
        vector.

        :param outputs: the weighted sum right before the last layer softmax
               normalization, of dimension [B x M]
        :type outputs: np.ndarray
        :param targets: either the attack targets or the real image labels,
               depending on whether or not ``self.targeted``, of dimension [B]
        :type targets: np.ndarray
        :return: the compensated weighted sum of dimension [B x M]
        :rtype: np.ndarray
        """
        outputs_comp = np.copy(outputs)
        rng = np.arange(targets.shape[0])
        if self.targeted:
            # for each image $i$:
            # if targeted, `outputs[i, target_onehot]` should be larger than
            # `max(outputs[i, ~target_onehot])` by `self.confidence`
            outputs_comp[rng, targets] -= self.confidence
        else:
            # for each image $i$:
            # if not targeted, `max(outputs[i, ~target_onehot]` should be larger
            # than `outputs[i, target_onehot]` (the ground truth image labels)
            # by `self.confidence`
            outputs_comp[rng, targets] += self.confidence
        return outputs_comp

    def _to_tanh_space(self, x):
        """
        Convert a batch of tensors to tanh-space.

        :param x: the batch of tensors, of dimension [B x C x H x W]
        :return: the batch of tensors in tanh-space, of the same dimension
        """
        return to_tanh_space(x, self.box)

    def _from_tanh_space(self, x):
        """
        Convert a batch of tensors from tanh-space to input space.

        :param x: the batch of tensors, of dimension [B x C x H x W]
        :return: the batch of tensors in tanh-space, of the same dimension;
                 the returned tensor is on the same device as ``x``
        """
        return from_tanh_space(x, self.box)
