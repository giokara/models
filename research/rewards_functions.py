# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Reward shaping functions used by Contexts.

  Each reward function should take the following inputs and return new rewards,
    and discounts.

  new_rewards, discounts = reward_fn(states, actions, rewards,
    next_states, contexts)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gin.tf


def summarize_stats(stats):
  """Summarize a dictionary of variables.

  Args:
    stats: a dictionary of {name: tensor} to compute stats over.
  """
  for name, stat in stats.items():
    mean = tf.reduce_mean(input_tensor=stat)
    tf.compat.v1.summary.scalar('mean_%s' % name, mean)
    tf.compat.v1.summary.scalar('max_%s' % name, tf.reduce_max(input_tensor=stat))
    tf.compat.v1.summary.scalar('min_%s' % name, tf.reduce_min(input_tensor=stat))
    std = tf.sqrt(tf.reduce_mean(input_tensor=tf.square(stat)) - tf.square(mean) + 1e-10)
    tf.compat.v1.summary.scalar('std_%s' % name, std)
    tf.compat.v1.summary.histogram(name, stat)


def index_states(states, indices):
  """Return indexed states.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    indices: (a list of Numpy integer array) Indices of states dimensions
      to be mapped.
  Returns:
    A [batch_size, num_indices] Tensor representing the batch of indexed states.
  """
  if indices is None:
    return states
  indices = tf.constant(indices, dtype=tf.int32)
  return tf.gather(states, indices=indices, axis=1)


def record_tensor(tensor, indices, stats, name='states'):
  """Record specified tensor dimensions into stats.

  Args:
    tensor: A [batch_size, num_dims] Tensor.
    indices: (a list of integers) Indices of dimensions to record.
    stats: A dictionary holding stats.
    name: (string) Name of tensor.
  """
  if indices is None:
    indices = range(tensor.shape.as_list()[1])
  for index in indices:
    stats['%s_%02d' % (name, index)] = tensor[:, index]


@gin.configurable
def potential_rewards(states,
                      actions,
                      rewards,
                      next_states,
                      contexts,
                      gamma=1.0,
                      reward_fn=None):
  """Return the potential-based rewards.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    gamma: Reward discount.
    reward_fn: A reward function.
  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
  del actions  # unused args
  gamma = tf.cast(gamma, dtype=tf.float32)
  rewards_tp1, discounts = reward_fn(None, None, rewards, next_states, contexts)
  rewards, _ = reward_fn(None, None, rewards, states, contexts)
  return -rewards + gamma * rewards_tp1, discounts


@gin.configurable
def timed_rewards(states,
                  actions,
                  rewards,
                  next_states,
                  contexts,
                  reward_fn=None,
                  dense=False,
                  timer_index=-1):
  """Return the timed rewards.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    reward_fn: A reward function.
    dense: (boolean) Provide dense rewards or sparse rewards at time = 0.
    timer_index: (integer) The context list index that specifies timer.
  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
  assert contexts[timer_index].get_shape().as_list()[1] == 1
  timers = contexts[timer_index][:, 0]
  rewards, discounts = reward_fn(states, actions, rewards, next_states,
                                 contexts)
  terminates = tf.cast(timers <= 0, dtype=tf.float32)  # if terminate set 1, else set 0
  for _ in range(rewards.shape.ndims - 1):
    terminates = tf.expand_dims(terminates, axis=-1)
  if not dense:
    rewards *= terminates  # if terminate, return rewards, else return 0
  discounts *= (tf.cast(1.0, dtype=tf.float32) - terminates)
  return rewards, discounts


@gin.configurable
def reset_rewards(states,
                  actions,
                  rewards,
                  next_states,
                  contexts,
                  reset_index=0,
                  reset_state=None,
                  reset_reward_function=None,
                  include_forward_rewards=True,
                  include_reset_rewards=True):
  """Returns the rewards for a forward/reset agent.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    reset_index: (integer) The context list index that specifies reset.
    reset_state: Reset state.
    reset_reward_function: Reward function for reset step.
    include_forward_rewards: Include the rewards from the forward pass.
    include_reset_rewards: Include the rewards from the reset pass.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
  reset_state = tf.constant(
      reset_state, dtype=next_states.dtype, shape=next_states.shape)
  reset_states = tf.expand_dims(reset_state, 0)

  def true_fn():
    if include_reset_rewards:
      return reset_reward_function(states, actions, rewards, next_states,
                                   [reset_states] + contexts[1:])
    else:
      return tf.zeros_like(rewards), tf.ones_like(rewards)

  def false_fn():
    if include_forward_rewards:
      return plain_rewards(states, actions, rewards, next_states, contexts)
    else:
      return tf.zeros_like(rewards), tf.ones_like(rewards)

  rewards, discounts = tf.cond(
      pred=tf.cast(contexts[reset_index][0, 0], dtype=tf.bool), true_fn=true_fn, false_fn=false_fn)
  return rewards, discounts


@gin.configurable
def tanh_similarity(states,
                    actions,
                    rewards,
                    next_states,
                    contexts,
                    mse_scale=1.0,
                    state_scales=1.0,
                    goal_scales=1.0,
                    summarize=False):
  """Returns the similarity between next_states and contexts using tanh and mse.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    mse_scale: A float, to scale mse before tanh.
    state_scales: multiplicative scale for (next) states. A scalar or 1D tensor,
      must be broadcastable to number of state dimensions.
    goal_scales: multiplicative scale for contexts. A scalar or 1D tensor,
      must be broadcastable to number of goal dimensions.
    summarize: (boolean) enable summary ops.


  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
  del states, actions, rewards  # Unused
  mse = tf.reduce_mean(input_tensor=tf.math.squared_difference(next_states * state_scales,
                                             contexts[0] * goal_scales), axis=-1)
  tanh = tf.tanh(mse_scale * mse)
  if summarize:
    with tf.compat.v1.name_scope('RewardFn/'):
      tf.compat.v1.summary.scalar('mean_mse', tf.reduce_mean(input_tensor=mse))
      tf.compat.v1.summary.histogram('mse', mse)
      tf.compat.v1.summary.scalar('mean_tanh', tf.reduce_mean(input_tensor=tanh))
      tf.compat.v1.summary.histogram('tanh', tanh)
  rewards = tf.cast(1 - tanh, dtype=tf.float32)
  return rewards, tf.ones_like(rewards)


@gin.configurable
def negative_mse(states,
                 actions,
                 rewards,
                 next_states,
                 contexts,
                 state_scales=1.0,
                 goal_scales=1.0,
                 summarize=False):
  """Returns the negative mean square error between next_states and contexts.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    state_scales: multiplicative scale for (next) states. A scalar or 1D tensor,
      must be broadcastable to number of state dimensions.
    goal_scales: multiplicative scale for contexts. A scalar or 1D tensor,
      must be broadcastable to number of goal dimensions.
    summarize: (boolean) enable summary ops.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
  del states, actions, rewards  # Unused
  mse = tf.reduce_mean(input_tensor=tf.math.squared_difference(next_states * state_scales,
                                             contexts[0] * goal_scales), axis=-1)
  if summarize:
    with tf.compat.v1.name_scope('RewardFn/'):
      tf.compat.v1.summary.scalar('mean_mse', tf.reduce_mean(input_tensor=mse))
      tf.compat.v1.summary.histogram('mse', mse)
  rewards = tf.cast(-mse, dtype=tf.float32)
  return rewards, tf.ones_like(rewards)


@gin.configurable
def negative_distance(states,
                      actions,
                      rewards,
                      next_states,
                      contexts,
                      state_scales=1.0,
                      goal_scales=1.0,
                      reward_scales=1.0,
                      weight_index=None,
                      weight_vector=None,
                      summarize=False,
                      termination_epsilon=1e-4,
                      state_indices=None,
                      goal_indices=None,
                      vectorize=False,
                      relative_context=False,
                      diff=False,
                      norm='L2',
                      epsilon=1e-10,
                      bonus_epsilon=0., #5.,
                      offset=0.0):
  """Returns the negative euclidean distance between next_states and contexts.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    state_scales: multiplicative scale for (next) states. A scalar or 1D tensor,
      must be broadcastable to number of state dimensions.
    goal_scales: multiplicative scale for goals. A scalar or 1D tensor,
      must be broadcastable to number of goal dimensions.
    reward_scales: multiplicative scale for rewards. A scalar or 1D tensor,
      must be broadcastable to number of reward dimensions.
    weight_index: (integer) The context list index that specifies weight.
    weight_vector: (a number or a list or Numpy array) The weighting vector,
      broadcastable to `next_states`.
    summarize: (boolean) enable summary ops.
    termination_epsilon: terminate if dist is less than this quantity.
    state_indices: (a list of integers) list of state indices to select.
    goal_indices: (a list of integers) list of goal indices to select.
    vectorize: Return a vectorized form.
    norm: L1 or L2.
    epsilon: small offset to ensure non-negative/zero distance.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
  del actions, rewards  # Unused
  stats = {}
  record_tensor(next_states, state_indices, stats, 'next_states')
  states = index_states(states, state_indices)
  next_states = index_states(next_states, state_indices)
  goals = index_states(contexts[0], goal_indices)
  if relative_context:
    goals = states + goals
  sq_dists = tf.math.squared_difference(next_states * state_scales,
                                   goals * goal_scales)
  old_sq_dists = tf.math.squared_difference(states * state_scales,
                                       goals * goal_scales)
  record_tensor(sq_dists, None, stats, 'sq_dists')
  if weight_vector is not None:
    sq_dists *= tf.convert_to_tensor(value=weight_vector, dtype=next_states.dtype)
    old_sq_dists *= tf.convert_to_tensor(value=weight_vector, dtype=next_states.dtype)
  if weight_index is not None:
    #sq_dists *= contexts[weight_index]
    weights = tf.abs(index_states(contexts[0], weight_index))
    #weights /= tf.reduce_sum(weights, -1, keepdims=True)
    sq_dists *= weights
    old_sq_dists *= weights
  if norm == 'L1':
    dist = tf.sqrt(sq_dists + epsilon)
    old_dist = tf.sqrt(old_sq_dists + epsilon)
    if not vectorize:
      dist = tf.reduce_sum(input_tensor=dist, axis=-1)
      old_dist = tf.reduce_sum(input_tensor=old_dist, axis=-1)
  elif norm == 'L2':
    if vectorize:
      dist = sq_dists
      old_dist = old_sq_dists
    else:
      dist = tf.reduce_sum(input_tensor=sq_dists, axis=-1)
      old_dist = tf.reduce_sum(input_tensor=old_sq_dists, axis=-1)
    dist = tf.sqrt(dist + epsilon)  # tf.gradients fails when tf.sqrt(-0.0)
    old_dist = tf.sqrt(old_dist + epsilon)  # tf.gradients fails when tf.sqrt(-0.0)
  else:
    raise NotImplementedError(norm)
  discounts = dist > termination_epsilon
  if summarize:
    with tf.compat.v1.name_scope('RewardFn/'):
      tf.compat.v1.summary.scalar('mean_dist', tf.reduce_mean(input_tensor=dist))
      tf.compat.v1.summary.histogram('dist', dist)
      summarize_stats(stats)
  bonus = tf.cast(dist < bonus_epsilon, dtype=tf.float32)
  dist *= reward_scales
  old_dist *= reward_scales
  if diff:
    return bonus + offset + tf.cast(old_dist - dist, dtype=tf.float32), tf.cast(discounts, dtype=tf.float32)
  return bonus + offset + tf.cast(-dist, dtype=tf.float32), tf.cast(discounts, dtype=tf.float32)


@gin.configurable
def cosine_similarity(states,
                      actions,
                      rewards,
                      next_states,
                      contexts,
                      state_scales=1.0,
                      goal_scales=1.0,
                      reward_scales=1.0,
                      normalize_states=True,
                      normalize_goals=True,
                      weight_index=None,
                      weight_vector=None,
                      summarize=False,
                      state_indices=None,
                      goal_indices=None,
                      offset=0.0):
  """Returns the cosine similarity between next_states - states and contexts.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    state_scales: multiplicative scale for (next) states. A scalar or 1D tensor,
      must be broadcastable to number of state dimensions.
    goal_scales: multiplicative scale for goals. A scalar or 1D tensor,
      must be broadcastable to number of goal dimensions.
    reward_scales: multiplicative scale for rewards. A scalar or 1D tensor,
      must be broadcastable to number of reward dimensions.
    weight_index: (integer) The context list index that specifies weight.
    weight_vector: (a number or a list or Numpy array) The weighting vector,
      broadcastable to `next_states`.
    summarize: (boolean) enable summary ops.
    termination_epsilon: terminate if dist is less than this quantity.
    state_indices: (a list of integers) list of state indices to select.
    goal_indices: (a list of integers) list of goal indices to select.
    vectorize: Return a vectorized form.
    norm: L1 or L2.
    epsilon: small offset to ensure non-negative/zero distance.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
  del actions, rewards  # Unused
  stats = {}
  record_tensor(next_states, state_indices, stats, 'next_states')
  states = index_states(states, state_indices)
  next_states = index_states(next_states, state_indices)
  goals = index_states(contexts[0], goal_indices)

  if weight_vector is not None:
    goals *= tf.convert_to_tensor(value=weight_vector, dtype=next_states.dtype)
  if weight_index is not None:
    weights = tf.abs(index_states(contexts[0], weight_index))
    goals *= weights

  direction_vec = next_states - states
  if normalize_states:
    direction_vec = tf.nn.l2_normalize(direction_vec, -1)
  goal_vec = goals
  if normalize_goals:
    goal_vec = tf.nn.l2_normalize(goal_vec, -1)

  similarity = tf.reduce_sum(input_tensor=goal_vec * direction_vec, axis=-1)
  discounts = tf.ones_like(similarity)
  return offset + tf.cast(similarity, dtype=tf.float32), tf.cast(discounts, dtype=tf.float32)


@gin.configurable
def diff_distance(states,
                  actions,
                  rewards,
                  next_states,
                  contexts,
                  state_scales=1.0,
                  goal_scales=1.0,
                  reward_scales=1.0,
                  weight_index=None,
                  weight_vector=None,
                  summarize=False,
                  termination_epsilon=1e-4,
                  state_indices=None,
                  goal_indices=None,
                  norm='L2',
                  epsilon=1e-10):
  """Returns the difference in euclidean distance between states/next_states and contexts.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    state_scales: multiplicative scale for (next) states. A scalar or 1D tensor,
      must be broadcastable to number of state dimensions.
    goal_scales: multiplicative scale for goals. A scalar or 1D tensor,
      must be broadcastable to number of goal dimensions.
    reward_scales: multiplicative scale for rewards. A scalar or 1D tensor,
      must be broadcastable to number of reward dimensions.
    weight_index: (integer) The context list index that specifies weight.
    weight_vector: (a number or a list or Numpy array) The weighting vector,
      broadcastable to `next_states`.
    summarize: (boolean) enable summary ops.
    termination_epsilon: terminate if dist is less than this quantity.
    state_indices: (a list of integers) list of state indices to select.
    goal_indices: (a list of integers) list of goal indices to select.
    vectorize: Return a vectorized form.
    norm: L1 or L2.
    epsilon: small offset to ensure non-negative/zero distance.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
  del actions, rewards  # Unused
  stats = {}
  record_tensor(next_states, state_indices, stats, 'next_states')
  next_states = index_states(next_states, state_indices)
  states = index_states(states, state_indices)
  goals = index_states(contexts[0], goal_indices)
  next_sq_dists = tf.math.squared_difference(next_states * state_scales,
                                        goals * goal_scales)
  sq_dists = tf.math.squared_difference(states * state_scales,
                                   goals * goal_scales)
  record_tensor(sq_dists, None, stats, 'sq_dists')
  if weight_vector is not None:
    next_sq_dists *= tf.convert_to_tensor(value=weight_vector, dtype=next_states.dtype)
    sq_dists *= tf.convert_to_tensor(value=weight_vector, dtype=next_states.dtype)
  if weight_index is not None:
    next_sq_dists *= contexts[weight_index]
    sq_dists *= contexts[weight_index]
  if norm == 'L1':
    next_dist = tf.sqrt(next_sq_dists + epsilon)
    dist = tf.sqrt(sq_dists + epsilon)
    next_dist = tf.reduce_sum(input_tensor=next_dist, axis=-1)
    dist = tf.reduce_sum(input_tensor=dist, axis=-1)
  elif norm == 'L2':
    next_dist = tf.reduce_sum(input_tensor=next_sq_dists, axis=-1)
    next_dist = tf.sqrt(next_dist + epsilon)  # tf.gradients fails when tf.sqrt(-0.0)
    dist = tf.reduce_sum(input_tensor=sq_dists, axis=-1)
    dist = tf.sqrt(dist + epsilon)  # tf.gradients fails when tf.sqrt(-0.0)
  else:
    raise NotImplementedError(norm)
  discounts = next_dist > termination_epsilon
  if summarize:
    with tf.compat.v1.name_scope('RewardFn/'):
      tf.compat.v1.summary.scalar('mean_dist', tf.reduce_mean(input_tensor=dist))
      tf.compat.v1.summary.histogram('dist', dist)
      summarize_stats(stats)
  diff = dist - next_dist
  diff *= reward_scales
  return tf.cast(diff, dtype=tf.float32), tf.cast(discounts, dtype=tf.float32)


@gin.configurable
def binary_indicator(states,
                     actions,
                     rewards,
                     next_states,
                     contexts,
                     termination_epsilon=1e-4,
                     offset=0,
                     epsilon=1e-10,
                     state_indices=None,
                     summarize=False):
  """Returns 0/1 by checking if next_states and contexts overlap.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    termination_epsilon: terminate if dist is less than this quantity.
    offset: Offset the rewards.
    epsilon: small offset to ensure non-negative/zero distance.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
  del states, actions  # unused args
  next_states = index_states(next_states, state_indices)
  dist = tf.reduce_sum(input_tensor=tf.math.squared_difference(next_states, contexts[0]), axis=-1)
  dist = tf.sqrt(dist + epsilon)
  discounts = dist > termination_epsilon
  rewards = tf.logical_not(discounts)
  rewards = tf.cast(rewards, dtype=tf.float32) + offset
  return tf.cast(rewards, dtype=tf.float32), tf.ones_like(tf.cast(discounts, dtype=tf.float32)) #tf.to_float(discounts)


@gin.configurable
def plain_rewards(states, actions, rewards, next_states, contexts):
  """Returns the given rewards.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
  del states, actions, next_states, contexts  # Unused
  return rewards, tf.ones_like(rewards)


@gin.configurable
def ctrl_rewards(states,
                 actions,
                 rewards,
                 next_states,
                 contexts,
                 reward_scales=1.0):
  """Returns the negative control cost.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    reward_scales: multiplicative scale for rewards. A scalar or 1D tensor,
      must be broadcastable to number of reward dimensions.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
  del states, rewards, contexts  # Unused
  if actions is None:
    rewards = tf.cast(tf.zeros(shape=next_states.shape[:1]), dtype=tf.float32)
  else:
    rewards = -tf.reduce_sum(input_tensor=tf.square(actions), axis=1)
    rewards *= reward_scales
    rewards = tf.cast(rewards, dtype=tf.float32)
  return rewards, tf.ones_like(rewards)


@gin.configurable
def diff_rewards(
    states,
    actions,
    rewards,
    next_states,
    contexts,
    state_indices=None,
    goal_index=0,):
  """Returns (next_states - goals) as a batched vector reward."""
  del states, rewards, actions  # Unused
  if state_indices is not None:
    next_states = index_states(next_states, state_indices)
  rewards = tf.cast(next_states - contexts[goal_index], dtype=tf.float32)
  return rewards, tf.ones_like(rewards)


@gin.configurable
def state_rewards(states,
                  actions,
                  rewards,
                  next_states,
                  contexts,
                  weight_index=None,
                  state_indices=None,
                  weight_vector=1.0,
                  offset_vector=0.0,
                  summarize=False):
  """Returns the rewards that are linear mapping of next_states.

  Args:
    states: A [batch_size, num_state_dims] Tensor representing a batch
        of states.
    actions: A [batch_size, num_action_dims] Tensor representing a batch
      of actions.
    rewards: A [batch_size] Tensor representing a batch of rewards.
    next_states: A [batch_size, num_state_dims] Tensor representing a batch
      of next states.
    contexts: A list of [batch_size, num_context_dims] Tensor representing
      a batch of contexts.
    weight_index: (integer) Index of contexts lists that specify weighting.
    state_indices: (a list of Numpy integer array) Indices of states dimensions
      to be mapped.
    weight_vector: (a number or a list or Numpy array) The weighting vector,
      broadcastable to `next_states`.
    offset_vector: (a number or a list of Numpy array) The off vector.
    summarize: (boolean) enable summary ops.

  Returns:
    A new tf.float32 [batch_size] rewards Tensor, and
      tf.float32 [batch_size] discounts tensor.
  """
  del states, actions, rewards  # unused args
  stats = {}
  record_tensor(next_states, state_indices, stats)
  next_states = index_states(next_states, state_indices)
  weight = tf.constant(
      weight_vector, dtype=next_states.dtype, shape=next_states[0].shape)
  weights = tf.expand_dims(weight, 0)
  offset = tf.constant(
      offset_vector, dtype=next_states.dtype, shape=next_states[0].shape)
  offsets = tf.expand_dims(offset, 0)
  if weight_index is not None:
    weights *= contexts[weight_index]
  rewards = tf.cast(tf.reduce_sum(input_tensor=weights * (next_states+offsets), axis=1), dtype=tf.float32)
  if summarize:
    with tf.compat.v1.name_scope('RewardFn/'):
      summarize_stats(stats)
  return rewards, tf.ones_like(rewards)
