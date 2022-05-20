# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training script for Nerf."""

import functools
from typing import Any, Dict, Union, Optional, Sequence
from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import optim
from flax.core import freeze
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax import traverse_util
import gin
import jax
from jax import numpy as jnp
from jax import random
import numpy as np
import tensorflow as tf
import pdb
from jax.config import config as jax_config
import shutil


from hypernerf import configs
from hypernerf import datasets
from hypernerf import gpath
from hypernerf import model_utils
from hypernerf import image_utils
from hypernerf import models
from hypernerf import schedules
from hypernerf import training
from hypernerf import utils
from hypernerf import types
from hypernerf import evaluation

flags.DEFINE_enum('mode', None, ['jax_cpu', 'jax_gpu', 'jax_tpu'],
                  'Distributed strategy approach.')

flags.DEFINE_string('base_folder', None, 'where to store ckpts and logs')
flags.mark_flag_as_required('base_folder')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
flags.DEFINE_multi_string('gin_configs', (), 'Gin config files.')
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.DEFINE_boolean('remote_debug', False, 'Debugging with remote HPC service (not using debugger)')
FLAGS = flags.FLAGS


def _log_to_tensorboard(writer: tensorboard.SummaryWriter,
                        state: model_utils.TrainState,
                        scalar_params: training.ScalarParams,
                        stats: Dict[str, Union[Dict[str, jnp.ndarray],
                                               jnp.ndarray]],
                        time_dict: Dict[str, jnp.ndarray]):
  """Log statistics to Tensorboard."""
  step = int(state.optimizer.state.step)

  def _log_scalar(tag, value):
    if value is not None:
      writer.scalar(tag, value, step)

  _log_scalar('params/learning_rate', scalar_params.learning_rate)
  _log_scalar('params/nerf_alpha', state.nerf_alpha)
  _log_scalar('params/warp_alpha', state.warp_alpha)
  _log_scalar('params/hyper_sheet_alpha', state.hyper_sheet_alpha)
  _log_scalar('params/elastic_loss/weight', scalar_params.elastic_loss_weight)

  # pmean is applied in train_step so just take the item.
  for branch in {'coarse', 'fine'}:
    if branch not in stats:
      continue
    for stat_key, stat_value in stats[branch].items():
      writer.scalar(f'{stat_key}/{branch}', stat_value, step)

  _log_scalar('loss/background', stats.get('background_loss'))
  _log_scalar('loss/bg_decompose', stats.get('bg_decompose_loss'))
  _log_scalar('loss/blendw_loss', stats.get('blendw_loss'))
  _log_scalar('loss/blendw_pixel_loss', stats.get('blendw_pixel_loss'))
  _log_scalar('loss/coase_blendw_mean', stats.get('coarse_blendw'))
  _log_scalar('loss/fine_blendw_mean', stats.get('fine_blendw'))
  _log_scalar('loss/force_blendw_loss', stats.get('force_blendw_loss'))
  _log_scalar('loss/blendw_ray_loss', stats.get('blendw_ray_loss'))
  _log_scalar('loss/sigma_s_ray_loss', stats.get('sigma_s_ray_loss'))
  _log_scalar('loss/sigma_d_ray_loss', stats.get('sigma_d_ray_loss'))
  _log_scalar('loss/blendw_area_loss', stats.get('blendw_area_loss'))
  _log_scalar('loss/shadow_loss', stats.get('shadow_loss'))
  _log_scalar('loss/blendw_sample_loss', stats.get('blendw_sample_loss'))
  _log_scalar('loss/shadow_r_loss', stats.get('shadow_r_loss'))
  _log_scalar('loss/shadow_r_consistency_loss', stats.get('shadow_r_consistency_loss'))
  _log_scalar('loss/shadow_r_l2_loss', stats.get('shadow_r_l2_loss'))
  _log_scalar('loss/blendw_spatial_loss', stats.get('blendw_spatial_loss'))
  _log_scalar('loss/ex_blendw_ray_loss', stats.get('ex_blendw_ray_loss'))
  _log_scalar('loss/ex_density_ray_loss', stats.get('ex_density_ray_loss'))

  for k, v in time_dict.items():
    writer.scalar(f'time/{k}', v, step)


def _log_histograms(writer: tensorboard.SummaryWriter,
                    state: model_utils.TrainState,
                    model_out):
  """Log histograms to Tensorboard."""
  step = int(state.optimizer.state.step)
  params = state.optimizer.target['model']
  if 'nerf_embed' in params:
    embeddings = params['nerf_embed']['embed']['embedding']
    writer.histogram('nerf_embedding', embeddings, step)
  if 'hyper_embed' in params:
    embeddings = params['hyper_embed']['embed']['embedding']
    writer.histogram('hyper_embedding', embeddings, step)
  if 'warp_embed' in params:
    embeddings = params['warp_embed']['embed']['embedding']
    writer.histogram('warp_embedding', embeddings, step)

  for branch in {'coarse', 'fine'}:
    if 'warped_points' in model_out[branch]:
      points = model_out[branch]['points']
      warped_points = model_out[branch]['warped_points']
      writer.histogram(f'{branch}/spatial_points',
                       warped_points[..., :3], step)
      writer.histogram(f'{branch}/spatial_points_delta',
                       warped_points[..., :3] - points, step)
      if warped_points.shape[-1] > 3:
        writer.histogram(f'{branch}/hyper_points',
                         warped_points[..., 3:], step)


def _log_grads(writer: tensorboard.SummaryWriter, model: models.NerfModel,
               state: model_utils.TrainState):
  """Log histograms to Tensorboard."""
  step = int(state.optimizer.state.step)
  params = state.optimizer.target['model']
  if 'nerf_metadata_encoder' in params:
    embeddings = params['nerf_metadata_encoder']['embed']['embedding']
    writer.histogram('nerf_embedding', embeddings, step)
  if 'hyper_metadata_encoder' in params:
    embeddings = params['hyper_metadata_encoder']['embed']['embedding']
    writer.histogram('hyper_embedding', embeddings, step)
  if 'warp_field' in params and model.warp_metadata_config['type'] == 'glo':
    embeddings = params['warp_metadata_encoder']['embed']['embedding']
    writer.histogram('warp_embedding', embeddings, step)


def main(argv):
  jax.config.parse_flags_with_absl()
  tf.config.experimental.set_visible_devices([], 'GPU')
  del argv
  logging.info('*** Starting experiment')
  # Assume G3 path for config files when running locally.
  gin_configs = FLAGS.gin_configs

  logging.info('*** Loading Gin configs from: %s', str(gin_configs))

  if FLAGS.debug:
    print('Debug mode on! Jitting is disabled')
    jax_config.update('jax_disable_jit', True)
    jax_config.update("jax_debug_nans", True)
  if FLAGS.remote_debug:
    print('Remote debug mode on! Jitting is enabled but will check for nan and produce additional log')
    jax_config.update("jax_debug_nans", True)


  # add simple fix for VS debugger:
  if FLAGS.debug and FLAGS.gin_bindings[0][0]=='"':
    FLAGS.gin_bindings = FLAGS.gin_bindings[0][1:-1]

  gin.parse_config_files_and_bindings(
      config_files=gin_configs,
      bindings=FLAGS.gin_bindings, # additional configs
      skip_unknown=True)

  # Load configurations.
  exp_config = configs.ExperimentConfig()
  train_config = configs.TrainConfig()
  eval_config = configs.EvalConfig()

  # assert exp_config.render_mode in types.RENDER_MODE.keys(), f"render mode {exp_config.render_mode} not recognized!"

  # add a few more steps to run for debug
  if FLAGS.debug:
    train_config.max_steps += 100

  if train_config.use_decompose_nerf:
    dummy_model = models.DecomposeNerfModel({}, 0, 0)
  else:
    dummy_model = models.NerfModel({}, 0, 0) # (embeddings_dict, near, far). Dummpy model is used to get configurations from gin


  # Get directory information.
  exp_dir = gpath.GPath(FLAGS.base_folder)
  if exp_config.subname:
    exp_dir = exp_dir / exp_config.subname
  summary_dir = exp_dir / 'summaries' / 'train'
  checkpoint_dir = exp_dir / 'checkpoints'

  renders_dir = exp_dir / f'renders-runtime'
  logging.info('\trenders_dir = %s', renders_dir)
  if not renders_dir.exists():
    renders_dir.mkdir(parents=True, exist_ok=True)

  # Log and create directories if this is the main process.
  if jax.process_index() == 0:
    logging.info('exp_dir = %s', exp_dir)
    if not exp_dir.exists():
      exp_dir.mkdir(parents=True, exist_ok=True)

    logging.info('summary_dir = %s', summary_dir)
    if not summary_dir.exists():
      summary_dir.mkdir(parents=True, exist_ok=True)

    logging.info('checkpoint_dir = %s', checkpoint_dir)
    if not checkpoint_dir.exists():
      checkpoint_dir.mkdir(parents=True, exist_ok=True)

  logging.info('Starting process %d. There are %d processes.',
               jax.process_index(), jax.process_count())
  logging.info('Found %d accelerator devices: %s.', jax.local_device_count(),
               str(jax.local_devices()))
  logging.info('Found %d total devices: %s.', jax.device_count(),
               str(jax.devices()))

  rng = random.PRNGKey(exp_config.random_seed)
  # Shift the numpy random seed by process_index() to shuffle data loaded by
  # different processes.
  np.random.seed(exp_config.random_seed + jax.process_index())

  if train_config.batch_size % jax.device_count() != 0:
    raise ValueError('Batch size must be divisible by the number of devices.')

  devices = jax.local_devices()
  logging.info('Creating datasource')
  datasource = exp_config.datasource_cls(
      image_scale=exp_config.image_scale,
      random_seed=exp_config.random_seed,
      # Enable metadata based on model needs.
      use_warp_id=True, # dummy_model.use_warp,
      use_appearance_id=(
          dummy_model.nerf_embed_key == 'appearance'
          or dummy_model.hyper_embed_key == 'appearance'),
      use_camera_id=dummy_model.nerf_embed_key == 'camera',
      use_time=dummy_model.warp_embed_key == 'time',
      use_mask=train_config.use_mask_sep_train,
      mask_interest_region=exp_config.mask_interest_region)

  # Create Model.
  logging.info('Initializing models.')
  rng, key = random.split(rng)
  params = {}

  construct_nerf_func = models.construct_nerf if not train_config.use_decompose_nerf else models.construct_decompose_nerf
  model, params['model'] = construct_nerf_func(
      key,
      batch_size=train_config.batch_size,
      embeddings_dict=datasource.embeddings_dict,
      near=datasource.near,
      far=datasource.far)

  # Create Jax iterator.
  logging.info('Creating dataset iterator.')
  train_iter = datasource.create_iterator(
      datasource.train_ids,
      flatten=True,
      shuffle=True,
      batch_size=train_config.batch_size,
      prefetch_size=3,
      shuffle_buffer_size=train_config.shuffle_buffer_size,
      devices=devices,
  )

  points_iter = None
  if train_config.use_background_loss:
    points = datasource.load_points(shuffle=True)
    points_batch_size = min(
        len(points),
        len(devices) * train_config.background_points_batch_size)
    points_batch_size -= points_batch_size % len(devices)
    points_dataset = tf.data.Dataset.from_tensor_slices(points)
    points_iter = datasets.iterator_from_dataset(
        points_dataset,
        batch_size=points_batch_size,
        prefetch_size=3,
        devices=devices)

  learning_rate_sched = schedules.from_config(train_config.lr_schedule)
  nerf_alpha_sched = schedules.from_config(train_config.nerf_alpha_schedule)
  warp_alpha_sched = schedules.from_config(train_config.warp_alpha_schedule)
  hyper_alpha_sched = schedules.from_config(train_config.hyper_alpha_schedule)
  hyper_sheet_alpha_sched = schedules.from_config(
      train_config.hyper_sheet_alpha_schedule)
  elastic_loss_weight_sched = schedules.from_config(
      train_config.elastic_loss_weight_schedule)
  blendw_loss_weight_sched = schedules.from_config(train_config.blendw_loss_weight_schedule)
  blendw_pixel_loss_weight_sched = schedules.from_config(train_config.blendw_pixel_loss_weight_schedule)
  shadow_r_loss_weight_sched = schedules.from_config(train_config.shadow_r_loss_weight)
  cubic_shadow_r_loss_weight_sched = schedules.from_config(train_config.cubic_shadow_r_loss_weight_schedule)
  shadow_r_consistency_loss_weight_sched = schedules.from_config(train_config.shadow_r_consistency_loss_weight_schedule)


  if train_config.freeze_dynamic_steps > 0:
    multi_optimizer = True
  else:
    multi_optimizer = False

  # multi_optimizer = True

  if multi_optimizer:
    if not train_config.use_decompose_nerf:
      raise NotImplementedError('multi_optimizer can only be set when using decompose nerf!')
    # seperate the optimizer for static and dynamic components 
    static_traversal = traverse_util.ModelParamTraversal(lambda path, _: 'static_nerf' in path)
    dynamic_traversal = traverse_util.ModelParamTraversal(lambda path, _: 'static_nerf' not in path)

    static_opt = optim.Adam(learning_rate_sched(0))
    dynamic_opt = optim.Adam(0.)
    if train_config.use_weight_norm:
      static_opt = optim.WeightNorm(static_opt)
      dynamic_opt = optim.WeightNorm(dynamic_opt)

    optimizer_def = optim.MultiOptimizer((static_traversal, static_opt),((dynamic_traversal, dynamic_opt)))
  else:
    optimizer_def = optim.Adam(learning_rate_sched(0))
    if train_config.use_weight_norm:
      optimizer_def = optim.WeightNorm(optimizer_def)

  optimizer = optimizer_def.create(params)

  state = model_utils.TrainState(
      optimizer=optimizer,
      nerf_alpha=nerf_alpha_sched(0),
      warp_alpha=warp_alpha_sched(0),
      hyper_alpha=hyper_alpha_sched(0),
      hyper_sheet_alpha=hyper_sheet_alpha_sched(0),
      freeze_static=False,
      freeze_dynamic=False,
      freeze_blendw=False,
      freeze_blendw_value=train_config.fix_blendw_value
      )
  scalar_params = training.ScalarParams(
      learning_rate=learning_rate_sched(0),
      elastic_loss_weight=elastic_loss_weight_sched(0),
      warp_reg_loss_weight=train_config.warp_reg_loss_weight,
      warp_reg_loss_alpha=train_config.warp_reg_loss_alpha,
      warp_reg_loss_scale=train_config.warp_reg_loss_scale,
      background_loss_weight=train_config.background_loss_weight,
      bg_decompose_loss_weight=train_config.bg_decompose_loss_weight,
      blendw_loss_weight=blendw_loss_weight_sched(0),
      blendw_pixel_loss_weight=blendw_pixel_loss_weight_sched(0),
      blendw_loss_skewness=train_config.blendw_loss_skewness,
      blendw_pixel_loss_skewness=train_config.blendw_pixel_loss_skewness,
      force_blendw_loss_weight=train_config.force_blendw_loss_weight,
      blendw_ray_loss_weight=train_config.blendw_ray_loss_weight,
      sigma_s_ray_loss_weight=train_config.sigma_s_ray_loss_weight,
      sigma_d_ray_loss_weight=train_config.sigma_d_ray_loss_weight,
      blendw_ray_loss_threshold=train_config.blendw_ray_loss_threshold,
      blendw_area_loss_weight=train_config.blendw_area_loss_weight,
      shadow_loss_threshold=train_config.shadow_loss_threshold,
      shadow_loss_weight=train_config.shadow_loss_weight,
      blendw_sample_loss_weight=train_config.blendw_sample_loss_weight,
      shadow_r_loss_weight=shadow_r_loss_weight_sched(0),
      cubic_shadow_r_loss_weight=cubic_shadow_r_loss_weight_sched(0),
      shadow_r_consistency_loss_weight=shadow_r_consistency_loss_weight_sched(0),
      shadow_r_l2_loss_weight=train_config.shadow_r_l2_loss_weight,
      blendw_spatial_loss_weight=train_config.blendw_spatial_loss_weight,
      hyper_reg_loss_weight=train_config.hyper_reg_loss_weight)
  new_state = state
  state = checkpoints.restore_checkpoint(checkpoint_dir, state)

  # # to restore only static model:
  # params = state.optimizer.target['model'].unfreeze()
  # params['hyper_sheet_mlp'] = new_state.optimizer.target['model']['hyper_sheet_mlp']
  # params['nerf_mlps_coarse'] = new_state.optimizer.target['model']['nerf_mlps_coarse']
  # params['nerf_mlps_fine'] = new_state.optimizer.target['model']['nerf_mlps_fine']
  # params['warp_embed'] = new_state.optimizer.target['model']['warp_embed']
  # params['warp_field'] = new_state.optimizer.target['model']['warp_field']
  # params = freeze(params)
  # state.optimizer.replace(target=params)

  print(f'Loaded step {state.optimizer.state.step}')
  init_step = state.optimizer.state.step + 1
  state = jax_utils.replicate(state, devices=devices)
  del params

  summary_writer = None
  if jax.process_index() == 0:
    config_str = gin.operative_config_str()
    logging.info('Configuration: \n%s', config_str)
    with (exp_dir / 'config.gin').open('w') as f:
      f.write(config_str)
    summary_writer = tensorboard.SummaryWriter(str(summary_dir))
    summary_writer.text('gin/train', textdata=gin.markdown(config_str), step=0)

    # copy source gin config for better readability
    shutil.copy(gin_configs[0], exp_dir / 'source.gin')

  train_step = functools.partial(
      training.train_step, # rng_key, state, batch, scalar_params
      model,
      elastic_reduce_method=train_config.elastic_reduce_method,
      elastic_loss_type=train_config.elastic_loss_type,
      use_elastic_loss=train_config.use_elastic_loss,
      use_background_loss=train_config.use_background_loss,
      use_bg_decompose_loss=train_config.use_bg_decompose_loss,
      use_warp_reg_loss=train_config.use_warp_reg_loss,
      use_hyper_reg_loss=train_config.use_hyper_reg_loss,
      multi_optimizer=multi_optimizer,
      use_ex_ray_entropy_loss=train_config.use_ex_ray_entropy_loss,)

  if FLAGS.debug:
    # vmap version for debugging
    ptrain_step = jax.vmap(
        train_step,
        axis_name='batch',
        # rng_key, state, batch, scalar_params.
        in_axes=(0, 0, 0, None) 
    )
  else:
    ptrain_step = jax.pmap( # jax parallel map
        train_step,
        axis_name='batch', # assigns a hashable name to the axis, which can be later referred to by other functions
        devices=devices,
        # rng_key, state, batch, scalar_params.
        in_axes=(0, 0, 0, None), # in_axes is used to align and pad the inputs to match dimensions
        # Treat use_elastic_loss as compile-time static.
        donate_argnums=(2,),  # Donate the 'batch' argument -- arguments that are no longer needed after computation can be donated to reduce memory requirement
    )


  if devices:
    n_local_devices = len(devices)
  else:
    n_local_devices = jax.local_device_count()

  logging.info('Starting training')
  # Make random seed separate across processes.
  rng = rng + jax.process_index()
  keys = random.split(rng, n_local_devices)
  time_tracker = utils.TimeTracker()
  time_tracker.tic('data', 'total')
  for step, batch in zip(range(init_step, train_config.max_steps + 1),
                         train_iter):
    if points_iter is not None:
      batch['background_points'] = next(points_iter)
    time_tracker.toc('data')
    # See: b/162398046.
    # pytype: disable=attribute-error
    scalar_params = scalar_params.replace( # update the per-step params: lr and elastic loss
        learning_rate=learning_rate_sched(step),
        elastic_loss_weight=elastic_loss_weight_sched(step),
        blendw_loss_weight=blendw_loss_weight_sched(step),
        blendw_pixel_loss_weight=blendw_pixel_loss_weight_sched(step),
        shadow_r_loss_weight=shadow_r_loss_weight_sched(step),
        cubic_shadow_r_loss_weight=cubic_shadow_r_loss_weight_sched(step),
        shadow_r_consistency_loss_weight=shadow_r_consistency_loss_weight_sched(step),
        )
    # pytype: enable=attribute-error
    nerf_alpha = jax_utils.replicate(nerf_alpha_sched(step), devices)
    warp_alpha = jax_utils.replicate(warp_alpha_sched(step), devices)
    hyper_alpha = jax_utils.replicate(hyper_alpha_sched(step), devices)
    hyper_sheet_alpha = jax_utils.replicate(
        hyper_sheet_alpha_sched(step), devices)
    # render_mode = jax_utils.replicate(types.RENDER_MODE[exp_config.render_mode])
    freeze_static = jax_utils.replicate(False)
    freeze_dynamic = jax_utils.replicate(step<train_config.freeze_dynamic_steps)
    freeze_blendw = jax_utils.replicate(step<train_config.fix_blendw_steps)
    force_blendw = jax_utils.replicate(step<train_config.force_blendw_steps)
    state = state.replace(nerf_alpha=nerf_alpha,
                          warp_alpha=warp_alpha,
                          hyper_alpha=hyper_alpha,
                          hyper_sheet_alpha=hyper_sheet_alpha,
                          # render_mode=render_mode,
                          freeze_static=freeze_static,
                          freeze_dynamic=freeze_dynamic,
                          freeze_blendw=freeze_blendw,
                          force_blendw=force_blendw)

    if train_config.use_mask_sep_train:
      # check the mask in the batch to disable training of the opposite component
      # Note that this is disabled at the moment
      raise NotImplementedError('mask training bug not fixed')
      pred = (batch['mask'] > 0)[0,:,0]
      if all(pred) !=  any(pred):
        raise ValueError('Batch separation is incorrect, one batch contains rays for both static and dynamic components')
      if all(pred):
        # dynamic batch
        freeze_static = False
      else:
        # static batch
        freeze_dynamic = False

    # Sample additional ray batch,
    # which contains unseen combination of time + view
    # Used for regularization
    if train_config.use_ex_ray_entropy_loss:
      test_rng = random.PRNGKey(step)
      shape = batch['origins'][..., :1].shape
      metadata = {}
      if datasource.use_warp_id:
        warp_id = random.choice(test_rng, jnp.asarray(datasource.warp_ids))
        metadata['warp'] = jnp.full(shape, fill_value=warp_id, dtype=jnp.uint32)

      # following two are usually not used
      if datasource.use_appearance_id:
        appearance_id = random.choice(
            test_rng, jnp.asarray(datasource.appearance_ids))
        metadata['appearance'] = jnp.full(shape, fill_value=appearance_id,
                                          dtype=jnp.uint32)
      if datasource.use_camera_id:
        camera_id = random.choice(test_rng, jnp.asarray(datasource.camera_ids))
        metadata['camera'] = jnp.full(shape, fill_value=camera_id,
                                      dtype=jnp.uint32)
      if datasource.use_time:
        timestamp = random.uniform(test_rng, minval=0.0, maxval=1.0)
        metadata['time'] = jnp.full(
            shape, fill_value=timestamp, dtype=jnp.uint32)

      batch['ex_metadata'] = metadata
    else:
      batch['ex_metadata'] = None

    with time_tracker.record_time('train_step'):
      state, stats, keys, model_out = ptrain_step(
          keys, state, batch, scalar_params)
      time_tracker.toc('total')

    if step % train_config.print_every == 0 and jax.process_index() == 0:
      logging.info('step=%d, nerf_alpha=%.04f, warp_alpha=%.04f, %s', step,
                   nerf_alpha_sched(step),
                   warp_alpha_sched(step),
                   time_tracker.summary_str('last'))
      coarse_metrics_str = ', '.join(
          [f'{k}={v.mean():.04f}' for k, v in stats['coarse'].items()])
      fine_metrics_str = ', '.join(
          [f'{k}={v.mean():.04f}' for k, v in stats['fine'].items()])
      logging.info('\tcoarse metrics: %s', coarse_metrics_str)
      if 'fine' in stats:
        logging.info('\tfine metrics: %s', fine_metrics_str)

      if FLAGS.debug:
        logging.info('loss/background: %s', stats.get('background_loss'))
        logging.info('loss/bg_decompose: %s', stats.get('bg_decompose_loss'))
        logging.info('loss/blendw_loss: %s', stats.get('blendw_loss'))
        logging.info('loss/blendw_pixel_loss: %s', stats.get('blendw_pixel_loss'))
        logging.info('loss/coase_blendw_mean: %s', stats.get('coarse_blendw'))
        logging.info('loss/fine_blendw_mean: %s', stats.get('fine_blendw'))
        logging.info('loss/force_blendw_loss: %s', stats.get('force_blendw_loss'))
        logging.info('loss/blendw_ray_loss: %s', stats.get('blendw_ray_loss'))
        logging.info('loss/blendw_area_loss: %s', stats.get('blendw_area_loss'))
        logging.info('loss/shadow_loss: %s', stats.get('shadow_loss'))
        logging.info('loss/blendw_sample_loss: %s', stats.get('blendw_sample_loss'))
        logging.info('loss/shadow_r_loss: %s', stats.get('shadow_r_loss'))
        logging.info('loss/shadow_r_l2_loss: %s', stats.get('shadow_r_l2_loss'))
        logging.info('loss/blendw_spatial_loss: %s', stats.get('blendw_spatial_loss'))
        logging.info('loss/ex_blendw_ray_loss: %s', stats.get('ex_blendw_ray_loss'))
        logging.info('loss/ex_density_ray_loss: %s', stats.get('ex_density_ray_loss'))

    if step % train_config.save_every == 0 and jax.process_index() == 0:
      training.save_checkpoint(checkpoint_dir, state, keep=2)

    if step % train_config.log_every == 0 and jax.process_index() == 0:
      # Only log via process 0.
      _log_to_tensorboard(
          summary_writer,
          jax_utils.unreplicate(state),
          scalar_params,
          jax_utils.unreplicate(stats),
          time_dict=time_tracker.summary('mean'))
      time_tracker.reset()

    if step % train_config.histogram_every == 0 and jax.process_index() == 0:
      _log_histograms(summary_writer, jax_utils.unreplicate(state), model_out)

    time_tracker.tic('data', 'total')

    # Run time evaluation
    if step % eval_config.niter_runtime_eval == 0 and jax.process_index() == 0:
      train_eval_ids = utils.strided_subset(
          datasource.train_ids, eval_config.nimg_runtime_eval) 
          
      train_eval_ids += list(eval_config.ex_runtime_eval_targets)
      train_eval_iter = datasource.create_iterator(train_eval_ids, batch_size=0)

      def _model_fn(key_0, key_1, params, rays_dict, extra_params):
        out = model.apply({'params': params},
                          rays_dict,
                          extra_params=extra_params,
                          metadata_encoded=False,
                          rngs={
                              'coarse': key_0,
                              'fine': key_1
                          },
                          mutable=False)
        return jax.lax.all_gather(out, axis_name='batch')

      if FLAGS.debug:
        # vmap version for debugging
        pmodel_fn = jax.vmap(
            _model_fn,
            in_axes=(0, 0, 0, 0, 0),
            axis_name='batch',
        )
      else:
        pmodel_fn = jax.pmap(
            _model_fn,
            in_axes=(0, 0, 0, 0, 0), 
            devices=devices,
            axis_name='batch',
        )

      render_fn = functools.partial(evaluation.render_image,
                                    model_fn=pmodel_fn,
                                    device_count=n_local_devices,
                                    chunk=eval_config.chunk,
                                    normalise_rendering=False,
                                    use_tsne=False)

      save_dir = renders_dir
      
      extra_render_tags = model.extra_renders
      process_iterator(tag='runtime_eval',
              item_ids=train_eval_ids,
              iterator=train_eval_iter,
              state=state,
              rng=rng,
              step=step,
              render_fn=render_fn,
              save_dir=save_dir,
              model=model,
              extra_render_tags=extra_render_tags,
              save_out=step==train_config.max_steps)

  if train_config.max_steps % train_config.save_every != 0:
    training.save_checkpoint(checkpoint_dir, state, keep=2)

  
def process_iterator(tag: str,
                     item_ids: Sequence[str],
                     iterator,
                     rng: types.PRNGKey,
                     state: model_utils.TrainState,
                     step: int,
                     render_fn: Any,
                     save_dir: gpath.GPath,
                     model: models.NerfModel,
                     extra_render_tags: Optional[tuple],
                     save_out: bool = False):
  """Process a dataset iterator and compute metrics."""
  params = state.optimizer.target['model']
  save_dir = save_dir / f'{step:08d}' / tag
  for i, (item_id, batch) in enumerate(zip(item_ids, iterator)):
    logging.info('[%s:%d/%d] Processing %s ', tag, i+1, len(item_ids), item_id)

    model_out = render_fn(state, batch, rng=rng)
    plot_images(
        tag=tag,
        item_id=item_id,
        model_out=model_out,
        save_dir=save_dir,
        extra_render_tags=extra_render_tags)

    if save_out:
      # save all returned arrays for debugging purpose
      dict_path = save_dir / 'model_out' 
      dict_path.mkdir(exist_ok=True, parents=True)
      if 'rgb_d' in model_out:
        del model_out['rgb_d']
      if 'rgb_s' in model_out:
        del model_out['rgb_s']
      if isinstance(model,models.DecomposeNerfModel) and not model.use_shadow_model:
        del model_out['shadow_r']
      for k in model_out.keys():
        model_out[k] = model_out[k].astype(np.half)

      np.save(str(dict_path / f"{item_id.replace('/', '_')}.npy"), model_out)


def plot_images(tag: str,
                item_id: str,
                model_out: Any,
                save_dir: gpath.GPath,
                extra_render_tags=None):
  """Process and plot a single batch."""
  item_id = item_id.replace('/', '_')
  rgb = model_out['rgb'][..., :3]

  save_dir = save_dir / tag
  save_dir.mkdir(parents=True, exist_ok=True)
  image_utils.save_image(save_dir / f'regular_rgb_{item_id}.png',
                          image_utils.image_to_uint8(rgb))

  if extra_render_tags is not None:
    for extra_tag in extra_render_tags:
      image_utils.save_image(save_dir / f'{extra_tag}_rgb_{item_id}.png',
                            image_utils.image_to_uint8(model_out[f'extra_rgb_{extra_tag}'][..., :3]))



if __name__ == '__main__':
  app.run(main)
