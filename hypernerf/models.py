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

"""Different model implementation plus a general port for all the models."""
import functools
from typing import Any, Callable, Dict, Optional, Tuple, Sequence, Mapping

from flax import linen as nn
import gin
import immutabledict
import jax
from jax import random
import jax.numpy as jnp

from hypernerf import model_utils
from hypernerf import utils
from hypernerf import modules
from hypernerf import types
# pylint: disable=unused-import
from hypernerf import warping


def filter_sigma(points, sigma, render_opts):
  """Filters the density based on various rendering arguments.

   - `dust_threshold` suppresses any sigma values below a threshold.
   - `bounding_box` suppresses any sigma values outside of a 3D bounding box.

  Args:
    points: the input points for each sample.
    sigma: the array of sigma values.
    render_opts: a dictionary containing any of the options listed above.

  Returns:
    A filtered sigma density field.
  """
  if render_opts is None:
    return sigma

  # Clamp densities below the set threshold.
  if 'dust_threshold' in render_opts:
    dust_thres = render_opts.get('dust_threshold', 0.0)
    sigma = (sigma >= dust_thres).astype(jnp.float32) * sigma
  if 'bounding_box' in render_opts:
    xmin, xmax, ymin, ymax, zmin, zmax = render_opts['bounding_box']
    render_mask = ((points[..., 0] >= xmin) & (points[..., 0] <= xmax)
                   & (points[..., 1] >= ymin) & (points[..., 1] <= ymax)
                   & (points[..., 2] >= zmin) & (points[..., 2] <= zmax))
    sigma = render_mask.astype(jnp.float32) * sigma

  return sigma


@gin.configurable(denylist=['name'])
class NerfModel(nn.Module):
  """Nerf NN Model with both coarse and fine MLPs.

  Attributes:
    embeddings_dict: a dictionary containing the embeddings of each metadata
      key.
    use_viewdirs: bool, use viewdirs as a condition.
    noise_std: float, std dev of noise added to regularize sigma output.
    nerf_trunk_depth: int, the depth of the first part of MLP.
    nerf_trunk_width: int, the width of the first part of MLP.
    nerf_rgb_branch_depth: int, the depth of the second part of MLP.
    nerf_rgb_branch_width: int, the width of the second part of MLP.
    nerf_skips: which layers to add skip layers in the NeRF model.
    spatial_point_min_deg: min degree of positional encoding for positions.
    spatial_point_max_deg: max degree of positional encoding for positions.
    hyper_point_min_deg: min degree of positional encoding for hyper points. Hyper points are the ambient space coordinates!
    hyper_point_max_deg: max degree of positional encoding for hyper points.
    viewdir_min_deg: min degree of positional encoding for viewdirs.
    viewdir_max_deg: max degree of positional encoding for viewdirs.

    alpha_channels: int, the number of alpha_channelss.
    rgb_channels: int, the number of rgb_channelss.
    activation: the activation function used in the MLP.
    sigma_activation: the activation function applied to the sigma density.

    near: float, near clip.
    far: float, far clip.
    num_coarse_samples: int, the number of samples for coarse nerf.
    num_fine_samples: int, the number of samples for fine nerf.
    use_stratified_sampling: use stratified sampling.
    use_white_background: composite rendering on to a white background.
    use_linear_disparity: sample linearly in disparity rather than depth.

    use_nerf_embed: whether to use the template metadata.
    use_alpha_condition: whether to feed the appearance metadata to the alpha
      branch.
    use_rgb_condition: whether to feed the appearance metadata to the rgb
      branch.

    use_warp: whether to use the warp field or not.
    warp_metadata_config: the config for the warp metadata encoder.
    warp_min_deg: min degree of positional encoding for warps.
    warp_max_deg: max degree of positional encoding for warps.
  """
  embeddings_dict: Mapping[str, Sequence[int]] = gin.REQUIRED
  near: float = gin.REQUIRED
  far: float = gin.REQUIRED

  # NeRF architecture.
  use_viewdirs: bool = True
  noise_std: Optional[float] = None
  nerf_trunk_depth: int = 8
  nerf_trunk_width: int = 256
  nerf_rgb_branch_depth: int = 1
  nerf_rgb_branch_width: int = 128
  nerf_skips: Tuple[int] = (4,)

  # NeRF rendering.
  num_coarse_samples: int = 196
  num_fine_samples: int = 196
  use_stratified_sampling: bool = True
  use_white_background: bool = False
  use_linear_disparity: bool = False
  use_sample_at_infinity: bool = True

  spatial_point_min_deg: int = 0
  spatial_point_max_deg: int = 10
  hyper_point_min_deg: int = 0
  hyper_point_max_deg: int = 4
  viewdir_min_deg: int = 0
  viewdir_max_deg: int = 4
  use_posenc_identity: bool = True

  alpha_channels: int = 1
  rgb_channels: int = 3
  activation: types.Activation = nn.relu
  norm_type: Optional[str] = None
  sigma_activation: types.Activation = nn.softplus

  # NeRF metadata configs.
  use_nerf_embed: bool = False
  nerf_embed_cls: Callable[..., nn.Module] = (
      functools.partial(modules.GLOEmbed, num_dims=8))
  nerf_embed_key: str = 'appearance'
  use_alpha_condition: bool = False
  use_rgb_condition: bool = False
  hyper_slice_method: str = 'none'
  hyper_embed_cls: Callable[..., nn.Module] = (
      functools.partial(modules.GLOEmbed, num_dims=8))
  hyper_embed_key: str = 'appearance'
  hyper_use_warp_embed: bool = True
  hyper_sheet_mlp_cls: Callable[..., nn.Module] = modules.HyperSheetMLP
  hyper_sheet_use_input_points: bool = True

  # Warp configs.
  use_warp: bool = False
  warp_field_cls: Callable[..., nn.Module] = warping.SE3Field
  warp_embed_cls: Callable[..., nn.Module] = (
      functools.partial(modules.GLOEmbed, num_dims=8))
  warp_embed_key: str = 'warp'

  # render configs, decrepitated
  render_mode: str = None 

  # Addition render modes options that will be included in the output for evaluation purpose
  # Only support 'deformation', 'deformation_norm' and 'time'
  extra_renders: tuple = () 

  # Scale applied to deformation rendering
  deformation_render_scale: float = 1.0

  @property
  def num_nerf_embeds(self):
    return max(self.embeddings_dict[self.nerf_embed_key]) + 1

  @property
  def num_warp_embeds(self):
    return max(self.embeddings_dict[self.warp_embed_key]) + 1

  @property
  def num_hyper_embeds(self):
    return max(self.embeddings_dict[self.hyper_embed_key]) + 1

  @property
  def nerf_embeds(self):
    return jnp.array(self.embeddings_dict[self.nerf_embed_key], jnp.uint32)

  @property
  def warp_embeds(self):
    return jnp.array(self.embeddings_dict[self.warp_embed_key], jnp.uint32)

  @property
  def hyper_embeds(self):
    return jnp.array(self.embeddings_dict[self.hyper_embed_key], jnp.uint32)

  @property
  def has_hyper(self):
    """Whether the model uses a separate hyper embedding."""
    return self.hyper_slice_method != 'none'

  @property
  def has_hyper_embed(self):
    """Whether the model uses a separate hyper embedding."""
    # If the warp field outputs the hyper coordinates then there is no separate
    # hyper embedding.
    return self.has_hyper

  @property
  def has_embeds(self):
    return self.has_hyper_embed or self.use_warp or self.use_nerf_embed

  @staticmethod
  def _encode_embed(embed, embed_fn):
    """Encodes embeddings.

    If the channel size 1, it is just a single metadata ID.
    If the channel size is 3:
      the first channel is the left metadata ID,
      the second channel is the right metadata ID,
      the last channel is the progression from left to right (between 0 and 1).

    Args:
      embed: a (*, 1) or (*, 3) array containing metadata.
      embed_fn: the embedding function.

    Returns:
      A (*, C) array containing encoded embeddings.

    WW: Not exactly sure what it does!
        Not positional embedding, but embedding similiar to appearance/warp latent?
    """
    if embed.shape[-1] == 3:
      left, right, progression = jnp.split(embed, 3, axis=-1)
      left = embed_fn(left.astype(jnp.uint32))
      right = embed_fn(right.astype(jnp.uint32))
      return (1.0 - progression) * left + progression * right
    else:
      return embed_fn(embed)

  def encode_hyper_embed(self, metadata):
    if self.hyper_slice_method == 'axis_aligned_plane':
      # return self._encode_embed(metadata[self.hyper_embed_key],
      #                           self.hyper_embed)
      if self.hyper_use_warp_embed:
        return self._encode_embed(metadata[self.warp_embed_key],
                                  self.warp_embed)
      else:
        return self._encode_embed(metadata[self.hyper_embed_key],
                                  self.hyper_embed)
    elif self.hyper_slice_method == 'bendy_sheet':
      # The bendy sheet shares the metadata of the warp.
      if self.hyper_use_warp_embed:
        return self._encode_embed(metadata[self.warp_embed_key],
                                  self.warp_embed)
      else:
        return self._encode_embed(metadata[self.hyper_embed_key],
                                  self.hyper_embed)
    else:
      raise RuntimeError(
          f'Unknown hyper slice method {self.hyper_slice_method}.')

  def encode_nerf_embed(self, metadata):
    return self._encode_embed(metadata[self.nerf_embed_key], self.nerf_embed)

  def encode_warp_embed(self, metadata):
    return self._encode_embed(metadata[self.warp_embed_key], self.warp_embed)

  def setup(self):
    # setup is called when model.init() is called
    if (self.use_nerf_embed
        and not (self.use_rgb_condition
                 or self.use_alpha_condition)):
      raise ValueError('Template metadata is enabled but none of the condition'
                       'branches are.')

    if self.use_nerf_embed:
      self.nerf_embed = self.nerf_embed_cls(num_embeddings=self.num_nerf_embeds)
    # if self.use_warp:
    self.warp_embed = self.warp_embed_cls(num_embeddings=self.num_warp_embeds)

    if self.hyper_slice_method == 'axis_aligned_plane':
      self.hyper_embed = self.hyper_embed_cls(
          num_embeddings=self.num_hyper_embeds)
    elif self.hyper_slice_method == 'bendy_sheet':
      if not self.hyper_use_warp_embed:
        self.hyper_embed = self.hyper_embed_cls(
            num_embeddings=self.num_hyper_embeds)
      self.hyper_sheet_mlp = self.hyper_sheet_mlp_cls()

    if self.use_warp:
      self.warp_field = self.warp_field_cls() # warp_field is a MLP

    norm_layer = modules.get_norm_layer(self.norm_type)
    nerf_mlps = {
        'coarse': modules.NerfMLP(
            trunk_depth=self.nerf_trunk_depth,
            trunk_width=self.nerf_trunk_width,
            rgb_branch_depth=self.nerf_rgb_branch_depth,
            rgb_branch_width=self.nerf_rgb_branch_width,
            activation=self.activation,
            norm=norm_layer,
            skips=self.nerf_skips,
            alpha_channels=self.alpha_channels,
            rgb_channels=self.rgb_channels)
    }
    if self.num_fine_samples > 0:
      nerf_mlps['fine'] = modules.NerfMLP(
          trunk_depth=self.nerf_trunk_depth,
          trunk_width=self.nerf_trunk_width,
          rgb_branch_depth=self.nerf_rgb_branch_depth,
          rgb_branch_width=self.nerf_rgb_branch_width,
          activation=self.activation,
          norm=norm_layer,
          skips=self.nerf_skips,
          alpha_channels=self.alpha_channels,
          rgb_channels=self.rgb_channels)
    self.nerf_mlps = nerf_mlps

  def get_condition_inputs(self, viewdirs, metadata, metadata_encoded=False):
    """Create the condition inputs for the NeRF template. -- basically additional inputs, for example view directions for RGB query"""
    alpha_conditions = []
    rgb_conditions = []

    # Point attribute predictions
    if self.use_viewdirs:
      viewdirs_feat = model_utils.posenc(
          viewdirs,
          min_deg=self.viewdir_min_deg,
          max_deg=self.viewdir_max_deg,
          use_identity=self.use_posenc_identity)
      rgb_conditions.append(viewdirs_feat)

    if self.use_nerf_embed:
      if metadata_encoded:
        nerf_embed = metadata['encoded_nerf']
      else:
        nerf_embed = metadata[self.nerf_embed_key]
        nerf_embed = self.nerf_embed(nerf_embed)
      if self.use_alpha_condition:
        alpha_conditions.append(nerf_embed)
      if self.use_rgb_condition:
        rgb_conditions.append(nerf_embed)

    # The condition inputs have a shape of (B, C) now rather than (B, S, C)
    # since we assume all samples have the same condition input. We might want
    # to change this later.
    alpha_conditions = (
        jnp.concatenate(alpha_conditions, axis=-1)
        if alpha_conditions else None)
    rgb_conditions = (
        jnp.concatenate(rgb_conditions, axis=-1)
        if rgb_conditions else None)
    return alpha_conditions, rgb_conditions

  def query_template(self,
                     level, # coarse or fine
                     points,
                     viewdirs,
                     metadata,
                     extra_params,
                     metadata_encoded=False):
    """Queries the NeRF template."""
    alpha_condition, rgb_condition = (
        self.get_condition_inputs(viewdirs, metadata, metadata_encoded))

    points_feat = model_utils.posenc(
        points[..., :3],
        min_deg=self.spatial_point_min_deg,
        max_deg=self.spatial_point_max_deg,
        use_identity=self.use_posenc_identity,
        alpha=extra_params['nerf_alpha']) # alpha for windowed positional encoding
    # Encode hyper-points if present.
    if points.shape[-1] > 3:
      hyper_feats = model_utils.posenc(
          points[..., 3:],
          min_deg=self.hyper_point_min_deg,
          max_deg=self.hyper_point_max_deg,
          use_identity=False,
          alpha=extra_params['hyper_alpha'])
      points_feat = jnp.concatenate([points_feat, hyper_feats], axis=-1)

    raw = self.nerf_mlps[level](points_feat, alpha_condition, rgb_condition)
    raw = model_utils.noise_regularize(
        self.make_rng(level), raw, self.noise_std, self.use_stratified_sampling)

    rgb = nn.sigmoid(raw['rgb'])
    sigma = self.sigma_activation(jnp.squeeze(raw['alpha'], axis=-1))

    return rgb, sigma

  def map_spatial_points(self, points, warp_embed, extra_params, use_warp=True,
                         return_warp_jacobian=False):
    warp_jacobian = None
    if self.use_warp and use_warp:
      warp_fn = jax.vmap(jax.vmap(self.warp_field, in_axes=(0, 0, None, None)),
                         in_axes=(0, 0, None, None)) # double vmap needed, because the points and warp_embed are in shape (#batch, #point, value)
      warp_out = warp_fn(points,
                         warp_embed,
                         extra_params,
                         return_warp_jacobian)
      if return_warp_jacobian:
        warp_jacobian = warp_out['jacobian']
      warped_points = warp_out['warped_points']
    else:
      warped_points = points

    return warped_points, warp_jacobian

  def map_hyper_points(self, points, hyper_embed, extra_params,
                       hyper_point_override=None):
    """Maps input points to hyper points.

    Args:
      points: the input points.
      hyper_embed: the hyper embeddings.
      extra_params: extra params to pass to the slicing MLP if applicable.
      hyper_point_override: this may contain an override for the hyper points.
        Useful for rendering at specific hyper dimensions.

    Returns:
      An array of hyper points.
    """
    if hyper_point_override is not None:
      hyper_points = jnp.broadcast_to(
          hyper_point_override[:, None, :],
          (*points.shape[:-1], hyper_point_override.shape[-1]))
    elif self.hyper_slice_method == 'axis_aligned_plane': # no non-linear slice
      hyper_points = hyper_embed
    elif self.hyper_slice_method == 'bendy_sheet':
      hyper_points = self.hyper_sheet_mlp(
          points,
          hyper_embed,
          alpha=extra_params['hyper_sheet_alpha'])
    else:
      return None

    return hyper_points

  def map_points(self, points, warp_embed, hyper_embed, extra_params,
                 use_warp=True, return_warp_jacobian=False,
                 hyper_point_override=None):
    """Map input points to warped spatial and hyper points.

    Args:
      points: the input points to warp.
      warp_embed: the warp embeddings.
      hyper_embed: the hyper embeddings.
      extra_params: extra parameters to pass to the warp field/hyper field.
      use_warp: whether to use the warp or not.
      return_warp_jacobian: whether to return the warp jacobian or not.
      hyper_point_override: this may contain an override for the hyper points.
        Useful for rendering at specific hyper dimensions.

    Returns:
      A tuple containing `(warped_points, warp_jacobian)`.
    """
    # Map input points to warped spatial and hyper points.
    spatial_points, warp_jacobian = self.map_spatial_points(
        points, warp_embed, extra_params, use_warp=use_warp,
        return_warp_jacobian=return_warp_jacobian)
    hyper_points = self.map_hyper_points(
        points, hyper_embed, extra_params,
        # Override hyper points if present in metadata dict.
        hyper_point_override=hyper_point_override)

    if hyper_points is not None:
      warped_points = jnp.concatenate([spatial_points, hyper_points], axis=-1)
    else:
      warped_points = spatial_points

    return warped_points, warp_jacobian

  def apply_warp(self, points, warp_embed, extra_params):
    warp_embed = self.warp_embed(warp_embed)
    return self.warp_field(points, warp_embed, extra_params)

  def render_samples(self,
                     level,
                     points,
                     z_vals,
                     directions,
                     viewdirs,
                     metadata,
                     extra_params,
                     use_warp=True,
                     metadata_encoded=False,
                     return_warp_jacobian=False,
                     use_sample_at_infinity=False,
                     render_opts=None):
    out = {'points': points}

    batch_shape = points.shape[:-1]
    # Create the warp embedding.
    if use_warp:
      if metadata_encoded:
        warp_embed = metadata['encoded_warp']
      else:
        warp_embed = metadata[self.warp_embed_key] # self.hyper_embed_key: warp
        warp_embed = self.warp_embed(warp_embed) # embed each key (integer) to 8 digit vector
    else:
      warp_embed = None

    # Create the hyper embedding.
    if self.has_hyper_embed:
      if metadata_encoded:
        hyper_embed = metadata['encoded_hyper']
      elif self.hyper_use_warp_embed:
        hyper_embed = warp_embed # hyper embed is just the warp embed
      else:
        hyper_embed = metadata[self.hyper_embed_key] # self.hyper_embed_key: appearance
        hyper_embed = self.hyper_embed(hyper_embed)
    else:
      hyper_embed = None

    # Broadcast embeddings.
    if warp_embed is not None:
      warp_embed = jnp.broadcast_to( # boardcast the embeddings to each point
          warp_embed[:, jnp.newaxis, :],
          shape=(*batch_shape, warp_embed.shape[-1]))
    if hyper_embed is not None:
      hyper_embed = jnp.broadcast_to(
          hyper_embed[:, jnp.newaxis, :],
          shape=(*batch_shape, hyper_embed.shape[-1]))

    # Map input points to warped spatial and hyper points.
    warped_points, warp_jacobian = self.map_points(
        points, warp_embed, hyper_embed, extra_params, use_warp=use_warp,
        return_warp_jacobian=return_warp_jacobian,
        # Override hyper points if present in metadata dict.
        hyper_point_override=metadata.get('hyper_point'))

    rgb, sigma = self.query_template(
        level,
        warped_points,
        viewdirs,
        metadata,
        extra_params=extra_params,
        metadata_encoded=metadata_encoded)

    if self.render_mode is not None:
      if self.render_mode == 'deformation':
        # render the amount of deformation in dynamic component
        # need to ensure range [0,1]. TODO: better normalization
        rgb = jnp.clip((warped_points[...,:3] - points), 0, 1) * self.deformation_render_scale
      elif self.render_mode == 'time':
        # render the warped time coordinate
        # Trying volume rendering with 4D data
        rgb = jnp.clip((warped_points[...,3:]), 0, 1)
      else:
        raise NotImplementedError(f'Rendering model {self.render_mode} is not recognized')



    for render_mode in self.extra_renders:
      if render_mode == 'deformation':
        ex_rgb = jnp.clip((warped_points[...,:3] - points), 0, 1) * self.deformation_render_scale
      elif render_mode == 'deformation_norm':
        ex_rgb = jnp.clip((warped_points[...,:3] - points), 0, 1) 
        ex_rgb = jnp.ones_like(ex_rgb) * jnp.sqrt(jnp.sum(ex_rgb ** 2, axis=-1, keepdims=True)) * self.deformation_render_scale
      elif render_mode == 'time':
        ex_rgb = jnp.clip((warped_points[...,3:]), 0, 1)
      else:
        raise NotImplementedError(f'Rendering model {render_mode} is not recognized')
      extra_render = model_utils.volumetric_rendering(
                          ex_rgb,
                          sigma,
                          z_vals,
                          directions,
                          use_white_background=self.use_white_background,
                          sample_at_infinity=use_sample_at_infinity)
      out[f'extra_rgb_{render_mode}'] = extra_render['rgb']

    # Filter densities based on rendering options.
    sigma = filter_sigma(points, sigma, render_opts)

    if warp_jacobian is not None:
      out['warp_jacobian'] = warp_jacobian
    out['warped_points'] = warped_points
    out.update(model_utils.volumetric_rendering(
        rgb,
        sigma,
        z_vals,
        directions,
        use_white_background=self.use_white_background,
        sample_at_infinity=use_sample_at_infinity))

    # Add a map containing the returned points at the median depth.
    depth_indices = model_utils.compute_depth_index(out['weights'])
    med_points = jnp.take_along_axis(
        # Unsqueeze axes: sample axis, coords.
        warped_points, depth_indices[..., None, None], axis=-2)
    out['med_points'] = med_points

    return out

  def __call__(
      self,
      rays_dict: Dict[str, Any],
      extra_params: Dict[str, Any],
      metadata_encoded=False,
      use_warp=True,
      return_points=False,
      return_weights=False,
      return_warp_jacobian=False,
      near=None,
      far=None,
      use_sample_at_infinity=None,
      render_opts=None,
      deterministic=False,
  ):
    """Nerf Model.

    Args:
      rays_dict: a dictionary containing the ray information. Contains:
        'origins': the ray origins.
        'directions': unit vectors which are the ray directions.
        'viewdirs': (optional) unit vectors which are viewing directions.
        'metadata': a dictionary of metadata indices e.g., for warping.
      extra_params: parameters for the warp e.g., alpha.
      metadata_encoded: if True, assume the metadata is already encoded.
      use_warp: if True use the warp field (if also enabled in the model).
      return_points: if True return the points (and warped points if
        applicable).
      return_weights: if True return the density weights.
      return_warp_jacobian: if True computes and returns the warp Jacobians.
      near: if not None override the default near value.
      far: if not None override the default far value.
      use_sample_at_infinity: override for `self.use_sample_at_infinity`.
      render_opts: an optional dictionary of render options.
      deterministic: whether evaluation should be deterministic.

    Returns:
      ret: list, [(rgb, disp, acc), (rgb_coarse, disp_coarse, acc_coarse)]
    """
    use_warp = self.use_warp and use_warp
    # Extract viewdirs from the ray array
    origins = rays_dict['origins']
    directions = rays_dict['directions']
    metadata = rays_dict['metadata']
    if 'viewdirs' in rays_dict:
      viewdirs = rays_dict['viewdirs']
    else:  # viewdirs are normalized rays_d
      viewdirs = directions

    if near is None:
      near = self.near
    if far is None:
      far = self.far
    if use_sample_at_infinity is None:
      use_sample_at_infinity = self.use_sample_at_infinity

    # Evaluate coarse samples.
    z_vals, points = model_utils.sample_along_rays(
        self.make_rng('coarse'), origins, directions, self.num_coarse_samples,
        near, far, self.use_stratified_sampling,
        self.use_linear_disparity)
    coarse_ret = self.render_samples(
        'coarse',
        points,
        z_vals,
        directions,
        viewdirs,
        metadata,
        extra_params,
        use_warp=use_warp,
        metadata_encoded=metadata_encoded,
        return_warp_jacobian=return_warp_jacobian,
        use_sample_at_infinity=self.use_sample_at_infinity)
    out = {'coarse': coarse_ret}

    # Evaluate fine samples.
    if self.num_fine_samples > 0:
      z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
      z_vals, points = model_utils.sample_pdf(
          self.make_rng('fine'), z_vals_mid, coarse_ret['weights'][..., 1:-1],
          origins, directions, z_vals, self.num_fine_samples,
          self.use_stratified_sampling)
      out['fine'] = self.render_samples(
          'fine',
          points,
          z_vals,
          directions,
          viewdirs,
          metadata,
          extra_params,
          use_warp=use_warp,
          metadata_encoded=metadata_encoded,
          return_warp_jacobian=return_warp_jacobian,
          use_sample_at_infinity=use_sample_at_infinity,
          render_opts=render_opts)

    if not return_weights:
      del out['coarse']['weights']
      del out['fine']['weights']

    if not return_points:
      del out['coarse']['points']
      del out['coarse']['warped_points']
      del out['fine']['points']
      del out['fine']['warped_points']

    return out


@gin.configurable(denylist=['name'])
class StaticNerfModel(nn.Module):
  """Static Nerf NN Model with both coarse and fine MLPs.

  Attributes:
    embeddings_dict: a dictionary containing the embeddings of each metadata
      key.
    use_viewdirs: bool, use viewdirs as a condition.
    noise_std: float, std dev of noise added to regularize sigma output.
    nerf_trunk_depth: int, the depth of the first part of MLP.
    nerf_trunk_width: int, the width of the first part of MLP.
    nerf_rgb_branch_depth: int, the depth of the second part of MLP.
    nerf_rgb_branch_width: int, the width of the second part of MLP.
    nerf_skips: which layers to add skip layers in the NeRF model.
    spatial_point_min_deg: min degree of positional encoding for positions.
    spatial_point_max_deg: max degree of positional encoding for positions.
    viewdir_min_deg: min degree of positional encoding for viewdirs.
    viewdir_max_deg: max degree of positional encoding for viewdirs.

    alpha_channels: int, the number of alpha_channelss.
    rgb_channels: int, the number of rgb_channelss.
    activation: the activation function used in the MLP.
    sigma_activation: the activation function applied to the sigma density.

    near: float, near clip.
    far: float, far clip.
    num_coarse_samples: int, the number of samples for coarse nerf.
    num_fine_samples: int, the number of samples for fine nerf.
    use_stratified_sampling: use stratified sampling.
    use_white_background: composite rendering on to a white background.
    use_linear_disparity: sample linearly in disparity rather than depth.

    use_nerf_embed: whether to use the template metadata.
    use_alpha_condition: whether to feed the appearance metadata to the alpha
      branch.
    use_rgb_condition: whether to feed the appearance metadata to the rgb
      branch.

  """
  embeddings_dict: Mapping[str, Sequence[int]] = gin.REQUIRED
  near: float = gin.REQUIRED
  far: float = gin.REQUIRED

  # NeRF architecture.
  use_viewdirs: bool = True
  noise_std: Optional[float] = None
  nerf_trunk_depth: int = 8
  nerf_trunk_width: int = 256
  nerf_rgb_branch_depth: int = 1
  nerf_rgb_branch_width: int = 128
  nerf_skips: Tuple[int] = (4,)

  # NeRF rendering.
  num_coarse_samples: int = 196
  num_fine_samples: int = 196
  use_stratified_sampling: bool = True
  use_white_background: bool = False
  use_linear_disparity: bool = False
  use_sample_at_infinity: bool = True

  spatial_point_min_deg: int = 0
  spatial_point_max_deg: int = 10
  viewdir_min_deg: int = 0
  viewdir_max_deg: int = 4
  use_posenc_identity: bool = True

  alpha_channels: int = 1
  rgb_channels: int = 3
  activation: types.Activation = nn.relu
  norm_type: Optional[str] = None
  sigma_activation: types.Activation = nn.softplus

  # NeRF metadata configs.
  use_nerf_embed: bool = False
  nerf_embed_cls: Callable[..., nn.Module] = (
      functools.partial(modules.GLOEmbed, num_dims=8))
  nerf_embed_key: str = 'appearance'
  use_alpha_condition: bool = False
  use_rgb_condition: bool = False

  @property
  def num_nerf_embeds(self):
    return max(self.embeddings_dict[self.nerf_embed_key]) + 1


  @property
  def nerf_embeds(self):
    return jnp.array(self.embeddings_dict[self.nerf_embed_key], jnp.uint32)


  @property
  def has_embeds(self):
    return self.use_nerf_embed

  @staticmethod
  def _encode_embed(embed, embed_fn):
    """Encodes embeddings.

    If the channel size 1, it is just a single metadata ID.
    If the channel size is 3:
      the first channel is the left metadata ID,
      the second channel is the right metadata ID,
      the last channel is the progression from left to right (between 0 and 1).

    Args:
      embed: a (*, 1) or (*, 3) array containing metadata.
      embed_fn: the embedding function.

    Returns:
      A (*, C) array containing encoded embeddings.
    """
    if embed.shape[-1] == 3:
      left, right, progression = jnp.split(embed, 3, axis=-1)
      left = embed_fn(left.astype(jnp.uint32))
      right = embed_fn(right.astype(jnp.uint32))
      return (1.0 - progression) * left + progression * right
    else:
      return embed_fn(embed)

  def encode_nerf_embed(self, metadata):
    return self._encode_embed(metadata[self.nerf_embed_key], self.nerf_embed)

  def setup(self):
    if (self.use_nerf_embed
        and not (self.use_rgb_condition
                 or self.use_alpha_condition)):
      raise ValueError('Template metadata is enabled but none of the condition'
                       'branches are.')

    if self.use_nerf_embed:
      self.nerf_embed = self.nerf_embed_cls(num_embeddings=self.num_nerf_embeds)

    norm_layer = modules.get_norm_layer(self.norm_type)
    nerf_mlps = {
        'coarse': modules.NerfMLP(
            trunk_depth=self.nerf_trunk_depth,
            trunk_width=self.nerf_trunk_width,
            rgb_branch_depth=self.nerf_rgb_branch_depth,
            rgb_branch_width=self.nerf_rgb_branch_width,
            activation=self.activation,
            norm=norm_layer,
            skips=self.nerf_skips,
            alpha_channels=self.alpha_channels,
            rgb_channels=self.rgb_channels)
    }
    if self.num_fine_samples > 0:
      nerf_mlps['fine'] = modules.NerfMLP(
          trunk_depth=self.nerf_trunk_depth,
          trunk_width=self.nerf_trunk_width,
          rgb_branch_depth=self.nerf_rgb_branch_depth,
          rgb_branch_width=self.nerf_rgb_branch_width,
          activation=self.activation,
          norm=norm_layer,
          skips=self.nerf_skips,
          alpha_channels=self.alpha_channels,
          rgb_channels=self.rgb_channels)
    self.nerf_mlps = nerf_mlps

  def get_condition_inputs(self, viewdirs, metadata, metadata_encoded=False):
    """Create the condition inputs for the NeRF template."""
    alpha_conditions = []
    rgb_conditions = []

    # Point attribute predictions
    if self.use_viewdirs:
      viewdirs_feat = model_utils.posenc(
          viewdirs,
          min_deg=self.viewdir_min_deg,
          max_deg=self.viewdir_max_deg,
          use_identity=self.use_posenc_identity)
      rgb_conditions.append(viewdirs_feat)

    if self.use_nerf_embed:
      if metadata_encoded:
        nerf_embed = metadata['encoded_nerf']
      else:
        nerf_embed = metadata[self.nerf_embed_key]
        nerf_embed = self.nerf_embed(nerf_embed)
      if self.use_alpha_condition:
        alpha_conditions.append(nerf_embed)
      if self.use_rgb_condition:
        rgb_conditions.append(nerf_embed)

    # The condition inputs have a shape of (B, C) now rather than (B, S, C)
    # since we assume all samples have the same condition input. We might want
    # to change this later.
    alpha_conditions = (
        jnp.concatenate(alpha_conditions, axis=-1)
        if alpha_conditions else None)
    rgb_conditions = (
        jnp.concatenate(rgb_conditions, axis=-1)
        if rgb_conditions else None)
    return alpha_conditions, rgb_conditions

  def query_template(self,
                     level,
                     points,
                     viewdirs,
                     metadata,
                     extra_params,
                     metadata_encoded=False):
    """Queries the NeRF template."""
    alpha_condition, rgb_condition = (
        self.get_condition_inputs(viewdirs, metadata, metadata_encoded))

    points_feat = model_utils.posenc(
        points[..., :3],
        min_deg=self.spatial_point_min_deg,
        max_deg=self.spatial_point_max_deg,
        use_identity=self.use_posenc_identity,
        alpha=extra_params['nerf_alpha'])

    raw = self.nerf_mlps[level](points_feat, alpha_condition, rgb_condition)
    raw = model_utils.noise_regularize(
        self.make_rng(level), raw, self.noise_std, self.use_stratified_sampling)

    rgb = nn.sigmoid(raw['rgb'])
    sigma = self.sigma_activation(jnp.squeeze(raw['alpha'], axis=-1))

    return rgb, sigma

  def render_samples(self,
                     level,
                     points,
                     z_vals,
                     directions,
                     viewdirs,
                     metadata,
                     extra_params,
                     metadata_encoded=False,
                     use_sample_at_infinity=False,
                     render_opts=None):
    out = {'points': points}

    rgb, sigma = self.query_template(
        level,
        points,
        viewdirs,
        metadata,
        extra_params=extra_params,
        metadata_encoded=metadata_encoded)

    # Filter densities based on rendering options.
    sigma = filter_sigma(points, sigma, render_opts)

    out.update(model_utils.volumetric_rendering(
        rgb,
        sigma,
        z_vals,
        directions,
        use_white_background=self.use_white_background,
        sample_at_infinity=use_sample_at_infinity))

    # Add a map containing the returned points at the median depth.
    depth_indices = model_utils.compute_depth_index(out['weights'])
    med_points = jnp.take_along_axis(
        # Unsqueeze axes: sample axis, coords.
        points, depth_indices[..., None, None], axis=-2)
    out['med_points'] = med_points

    return out

  def __call__(
      self,
      rays_dict: Dict[str, Any],
      extra_params: Dict[str, Any],
      metadata_encoded=False,
      return_points=False,
      return_weights=False,
      near=None,
      far=None,
      use_sample_at_infinity=None,
      render_opts=None,
      deterministic=False,
  ):
    """Nerf Model.

    Args:
      rays_dict: a dictionary containing the ray information. Contains:
        'origins': the ray origins.
        'directions': unit vectors which are the ray directions.
        'viewdirs': (optional) unit vectors which are viewing directions.
        'metadata': a dictionary of metadata indices e.g., for warping.
      extra_params: parameters for the warp e.g., alpha.
      metadata_encoded: if True, assume the metadata is already encoded.
      use_warp: if True use the warp field (if also enabled in the model).
      return_points: if True return the points (and warped points if
        applicable).
      return_weights: if True return the density weights.
      return_warp_jacobian: if True computes and returns the warp Jacobians.
      near: if not None override the default near value.
      far: if not None override the default far value.
      use_sample_at_infinity: override for `self.use_sample_at_infinity`.
      render_opts: an optional dictionary of render options.
      deterministic: whether evaluation should be deterministic.

    Returns:
      ret: list, [(rgb, disp, acc), (rgb_coarse, disp_coarse, acc_coarse)]
    """
    # Extract viewdirs from the ray array
    origins = rays_dict['origins']
    directions = rays_dict['directions']
    metadata = rays_dict['metadata']
    if 'viewdirs' in rays_dict:
      viewdirs = rays_dict['viewdirs']
    else:  # viewdirs are normalized rays_d
      viewdirs = directions

    if near is None:
      near = self.near
    if far is None:
      far = self.far
    if use_sample_at_infinity is None:
      use_sample_at_infinity = self.use_sample_at_infinity

    # Evaluate coarse samples.
    z_vals, points = model_utils.sample_along_rays(
        self.make_rng('coarse'), origins, directions, self.num_coarse_samples,
        near, far, self.use_stratified_sampling,
        self.use_linear_disparity)
    coarse_ret = self.render_samples(
        'coarse',
        points,
        z_vals,
        directions,
        viewdirs,
        metadata,
        extra_params,
        metadata_encoded=metadata_encoded,
        use_sample_at_infinity=self.use_sample_at_infinity)
    out = {'coarse': coarse_ret}

    # Evaluate fine samples.
    if self.num_fine_samples > 0:
      z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
      z_vals, points = model_utils.sample_pdf(
          self.make_rng('fine'), z_vals_mid, coarse_ret['weights'][..., 1:-1],
          origins, directions, z_vals, self.num_fine_samples,
          self.use_stratified_sampling)
      out['fine'] = self.render_samples(
          'fine',
          points,
          z_vals,
          directions,
          viewdirs,
          metadata,
          extra_params,
          metadata_encoded=metadata_encoded,
          use_sample_at_infinity=use_sample_at_infinity,
          render_opts=render_opts)

    if not return_weights:
      del out['coarse']['weights']
      del out['fine']['weights']

    if not return_points:
      del out['coarse']['points']
      del out['fine']['points']

    return out


@gin.configurable(denylist=['name'])
class DecomposeNerfModel(NerfModel):
  """Nerf NN Model with both coarse and fine MLPs.
     A decomposed version which contains two NeRFs, a static one for modelling background and a dynamic one for the rest
     Dynamic NeRF additionally predicts a blending weight for the 3D radiance field

  Attributes:
    embeddings_dict: a dictionary containing the embeddings of each metadata
      key.
    use_viewdirs: bool, use viewdirs as a condition.
    noise_std: float, std dev of noise added to regularize sigma output.
    nerf_trunk_depth: int, the depth of the first part of MLP.
    nerf_trunk_width: int, the width of the first part of MLP.
    nerf_rgb_branch_depth: int, the depth of the second part of MLP.
    nerf_rgb_branch_width: int, the width of the second part of MLP.
    nerf_skips: which layers to add skip layers in the NeRF model.
    spatial_point_min_deg: min degree of positional encoding for positions.
    spatial_point_max_deg: max degree of positional encoding for positions.
    hyper_point_min_deg: min degree of positional encoding for hyper points. Hyper points are the ambient space coordinates!
    hyper_point_max_deg: max degree of positional encoding for hyper points.
    viewdir_min_deg: min degree of positional encoding for viewdirs.
    viewdir_max_deg: max degree of positional encoding for viewdirs.

    alpha_channels: int, the number of alpha_channelss.
    rgb_channels: int, the number of rgb_channelss.
    activation: the activation function used in the MLP.
    sigma_activation: the activation function applied to the sigma density.

    near: float, near clip.
    far: float, far clip.
    num_coarse_samples: int, the number of samples for coarse nerf.
    num_fine_samples: int, the number of samples for fine nerf.
    use_stratified_sampling: use stratified sampling.
    use_white_background: composite rendering on to a white background.
    use_linear_disparity: sample linearly in disparity rather than depth.

    use_nerf_embed: whether to use the template metadata.
    use_alpha_condition: whether to feed the appearance metadata to the alpha
      branch.
    use_rgb_condition: whether to feed the appearance metadata to the rgb
      branch.

    use_warp: whether to use the warp field or not.
    warp_metadata_config: the config for the warp metadata encoder.
    warp_min_deg: min degree of positional encoding for warps.
    warp_max_deg: max degree of positional encoding for warps.
  """
  embeddings_dict: Mapping[str, Sequence[int]] = gin.REQUIRED
  near: float = gin.REQUIRED
  far: float = gin.REQUIRED
  static_nerf_cls: Callable = StaticNerfModel

  # NeRF architecture.
  use_viewdirs: bool = True
  noise_std: Optional[float] = None
  nerf_trunk_depth: int = 8
  nerf_trunk_width: int = 256
  nerf_rgb_branch_depth: int = 1
  nerf_rgb_branch_width: int = 128
  nerf_skips: Tuple[int] = (4,)

  # NeRF rendering.
  num_coarse_samples: int = 196
  num_fine_samples: int = 196
  use_stratified_sampling: bool = True
  use_white_background: bool = False
  use_linear_disparity: bool = False
  use_sample_at_infinity: bool = True

  spatial_point_min_deg: int = 0
  spatial_point_max_deg: int = 10
  hyper_point_min_deg: int = 0
  hyper_point_max_deg: int = 4
  viewdir_min_deg: int = 0
  viewdir_max_deg: int = 4
  use_posenc_identity: bool = True

  alpha_channels: int = 1
  rgb_channels: int = 3
  activation: types.Activation = nn.relu
  norm_type: Optional[str] = None
  sigma_activation: types.Activation = nn.softplus

  # NeRF metadata configs.
  use_nerf_embed: bool = False # whether to use extra nerf embed (e.g., appearance code) or not
  nerf_embed_cls: Callable[..., nn.Module] = (
      functools.partial(modules.GLOEmbed, num_dims=8))
  nerf_embed_key: str = 'appearance'
  use_alpha_condition: bool = False # use extra nerf embed as density condition or not (input with coordinate)
  use_rgb_condition: bool = False # use extra nerf embed as rgb condition or not (input with view direction)
  hyper_slice_method: str = 'none'
  hyper_embed_cls: Callable[..., nn.Module] = (
      functools.partial(modules.GLOEmbed, num_dims=8))
  hyper_embed_key: str = 'appearance'
  hyper_use_warp_embed: bool = True
  hyper_sheet_mlp_cls: Callable[..., nn.Module] = modules.HyperSheetMLP
  hyper_sheet_use_input_points: bool = True

  # Warp configs.
  use_warp: bool = False
  warp_field_cls: Callable[..., nn.Module] = warping.SE3Field
  warp_embed_cls: Callable[..., nn.Module] = (
      functools.partial(modules.GLOEmbed, num_dims=8))
  warp_embed_key: str = 'warp'

  # Blending mode, 'old' or 'nsff'
  blend_mode: str = 'old'

  # Evaluation render configs
  render_mode: str = 'both'

  # Addition render modes options that will be included in the output for evaluation purpose
  extra_renders: tuple = ()
  # Scale applied to deformation rendering
  deformation_render_scale: float = 1.0

  # where to output blendw. -1 means outputting together with density (last layer)
  blendw_out_depth: int = -1

  # Whether to use an additional shadow model to deal with dynamic shadow effects
  use_shadow_model: bool = False
  # The threshold on sigma to determine whether shadow blending should be applied or not
  sigma_threshold: float = 0.1
  separate_shadow_model: bool = False
  shadow_r_shift: float = 0.

  # threshold for blendw mask
  blendw_mask_threshold: float = 0.5

  # Whether to deal with motion blur in the dynamic component or not
  handle_motion_blur: bool = False

  # Whether to use spatial smoothness loss
  # If true, would return an extra set of results that come from neighbouring points
  use_spatial_loss: bool = False 
  jitter_std: float = 0.01


  @property
  def num_nerf_embeds(self):
    return max(self.embeddings_dict[self.nerf_embed_key]) + 1

  @property
  def num_warp_embeds(self):
    return max(self.embeddings_dict[self.warp_embed_key]) + 1

  @property
  def num_hyper_embeds(self):
    return max(self.embeddings_dict[self.hyper_embed_key]) + 1

  @property
  def nerf_embeds(self):
    return jnp.array(self.embeddings_dict[self.nerf_embed_key], jnp.uint32)

  @property
  def warp_embeds(self):
    return jnp.array(self.embeddings_dict[self.warp_embed_key], jnp.uint32)

  @property
  def hyper_embeds(self):
    return jnp.array(self.embeddings_dict[self.hyper_embed_key], jnp.uint32)

  @property
  def has_hyper(self):
    """Whether the model uses a separate hyper embedding."""
    return self.hyper_slice_method != 'none'

  @property
  def has_hyper_embed(self):
    """Whether the model uses a separate hyper embedding."""
    # If the warp field outputs the hyper coordinates then there is no separate
    # hyper embedding.
    return self.has_hyper

  @property
  def has_embeds(self):
    return self.has_hyper_embed or self.use_warp or self.use_nerf_embed

  @staticmethod
  def _encode_embed(embed, embed_fn):
    """Encodes embeddings.

    If the channel size 1, it is just a single metadata ID.
    If the channel size is 3:
      the first channel is the left metadata ID,
      the second channel is the right metadata ID,
      the last channel is the progression from left to right (between 0 and 1).

    Args:
      embed: a (*, 1) or (*, 3) array containing metadata.
      embed_fn: the embedding function.

    Returns:
      A (*, C) array containing encoded embeddings.

    WW: Not exactly sure what it does!
        Not positional embedding, but embedding similiar to appearance/warp latent?
    """
    if embed.shape[-1] == 3:
      left, right, progression = jnp.split(embed, 3, axis=-1)
      left = embed_fn(left.astype(jnp.uint32))
      right = embed_fn(right.astype(jnp.uint32))
      return (1.0 - progression) * left + progression * right
    else:
      return embed_fn(embed)

  def encode_hyper_embed(self, metadata):
    if self.hyper_slice_method == 'axis_aligned_plane':
      # return self._encode_embed(metadata[self.hyper_embed_key],
      #                           self.hyper_embed)
      if self.hyper_use_warp_embed:
        return self._encode_embed(metadata[self.warp_embed_key],
                                  self.warp_embed)
      else:
        return self._encode_embed(metadata[self.hyper_embed_key],
                                  self.hyper_embed)
    elif self.hyper_slice_method == 'bendy_sheet':
      # The bendy sheet shares the metadata of the warp.
      if self.hyper_use_warp_embed:
        return self._encode_embed(metadata[self.warp_embed_key],
                                  self.warp_embed)
      else:
        return self._encode_embed(metadata[self.hyper_embed_key],
                                  self.hyper_embed)
    else:
      raise RuntimeError(
          f'Unknown hyper slice method {self.hyper_slice_method}.')

  def encode_nerf_embed(self, metadata):
    return self._encode_embed(metadata[self.nerf_embed_key], self.nerf_embed)

  def encode_warp_embed(self, metadata):
    return self._encode_embed(metadata[self.warp_embed_key], self.warp_embed)

  def setup(self):
    if (self.use_nerf_embed
        and not (self.use_rgb_condition
                 or self.use_alpha_condition)):
      raise ValueError('Template metadata is enabled but none of the condition'
                       'branches are.')

    if self.use_nerf_embed:
      self.nerf_embed = self.nerf_embed_cls(num_embeddings=self.num_nerf_embeds)
    if self.use_warp:
      self.warp_embed = self.warp_embed_cls(num_embeddings=self.num_warp_embeds)

    if self.hyper_slice_method == 'axis_aligned_plane':
      self.hyper_embed = self.hyper_embed_cls(
          num_embeddings=self.num_hyper_embeds)
    elif self.hyper_slice_method == 'bendy_sheet':
      if not self.hyper_use_warp_embed:
        self.hyper_embed = self.hyper_embed_cls(
            num_embeddings=self.num_hyper_embeds)
      self.hyper_sheet_mlp = self.hyper_sheet_mlp_cls()

    if self.use_warp:
      self.warp_field = self.warp_field_cls() # warp_field is a MLP

    self.static_nerf = self.static_nerf_cls( # TODO: remove cls as it's not needed 
            embeddings_dict=immutabledict.immutabledict(self.embeddings_dict),
            near=self.near,
            far=self.far
        )

    norm_layer = modules.get_norm_layer(self.norm_type)
    nerf_mlps = {
        'coarse': modules.BlendwNerfMLP(
            trunk_depth=self.nerf_trunk_depth,
            trunk_width=self.nerf_trunk_width,
            rgb_branch_depth=self.nerf_rgb_branch_depth,
            rgb_branch_width=self.nerf_rgb_branch_width,
            activation=self.activation,
            norm=norm_layer,
            skips=self.nerf_skips,
            alpha_channels=self.alpha_channels,
            rgb_channels=self.rgb_channels,
            blendw_output_depth=self.blendw_out_depth if self.blend_mode != 'add' else -2,
            output_shadow_r = self.use_shadow_model
            # if use add style blending, no need to output blendw
            )
    }
    if self.num_fine_samples > 0:
      nerf_mlps['fine'] = modules.BlendwNerfMLP(
          trunk_depth=self.nerf_trunk_depth,
          trunk_width=self.nerf_trunk_width,
          rgb_branch_depth=self.nerf_rgb_branch_depth,
          rgb_branch_width=self.nerf_rgb_branch_width,
          activation=self.activation,
          norm=norm_layer,
          skips=self.nerf_skips,
          alpha_channels=self.alpha_channels,
          rgb_channels=self.rgb_channels,
          blendw_output_depth=self.blendw_out_depth if self.blend_mode != 'add' else -2,
          output_shadow_r = self.use_shadow_model
          )
    self.nerf_mlps = nerf_mlps

    if self.use_shadow_model:
      if not self.use_warp:
        raise NotImplementedError('Shadow model can only be used when use_warp is true')
      self.shadow_warp_field = self.warp_field_cls()
      # self.shadow_r_mlp = modules.ShadowMLP()
      if self.separate_shadow_model:
        self.shadow_model =  modules.ShadowMLP(
          trunk_depth=self.nerf_trunk_depth,
          trunk_width=self.nerf_trunk_width,
          activation=self.activation,
          norm=norm_layer,
          skips=self.nerf_skips,
          )


    if self.handle_motion_blur:
      self.blur_mlp = modules.BlurMLP()
      

  def get_condition_inputs(self, viewdirs, metadata, metadata_encoded=False):
    """Create the condition inputs for the NeRF template. -- basically additional inputs, for example view directions for RGB query"""
    alpha_conditions = []
    rgb_conditions = []

    # Point attribute predictions
    if self.use_viewdirs:
      viewdirs_feat = model_utils.posenc(
          viewdirs,
          min_deg=self.viewdir_min_deg,
          max_deg=self.viewdir_max_deg,
          use_identity=self.use_posenc_identity)
      rgb_conditions.append(viewdirs_feat)

    if self.use_nerf_embed: # usually false, enabled to allow radiance to condition on appearance code
      if metadata_encoded:
        nerf_embed = metadata['encoded_nerf']
      else:
        nerf_embed = metadata[self.nerf_embed_key]
        nerf_embed = self.nerf_embed(nerf_embed)
      if self.use_alpha_condition:
        alpha_conditions.append(nerf_embed)
      if self.use_rgb_condition:
        rgb_conditions.append(nerf_embed)

    # The condition inputs have a shape of (B, C) now rather than (B, S, C)
    # since we assume all samples have the same condition input. We might want
    # to change this later.
    alpha_conditions = (
        jnp.concatenate(alpha_conditions, axis=-1)
        if alpha_conditions else None)
    rgb_conditions = (
        jnp.concatenate(rgb_conditions, axis=-1)
        if rgb_conditions else None)
    return alpha_conditions, rgb_conditions

  def query_template(self,
                     level, # coarse or fine
                     points, 
                     viewdirs, 
                     metadata, 
                     extra_params,
                     metadata_encoded=False):
    """Queries the NeRF template."""
    alpha_condition, rgb_condition = (
        self.get_condition_inputs(viewdirs, metadata, metadata_encoded))

    points_feat = model_utils.posenc(
        points[..., :3],
        min_deg=self.spatial_point_min_deg,
        max_deg=self.spatial_point_max_deg,
        use_identity=self.use_posenc_identity,
        alpha=extra_params['nerf_alpha']) # alpha for windowed positional encoding
    # Encode hyper-points if present.
    if points.shape[-1] > 3:
      hyper_feats = model_utils.posenc(
          points[..., 3:],
          min_deg=self.hyper_point_min_deg,
          max_deg=self.hyper_point_max_deg,
          use_identity=False,
          alpha=extra_params['hyper_alpha'])
      points_feat = jnp.concatenate([points_feat, hyper_feats], axis=-1)

    raw = self.nerf_mlps[level](points_feat, alpha_condition, rgb_condition)
    raw = model_utils.noise_regularize(
        self.make_rng(level), raw, self.noise_std, self.use_stratified_sampling) 

    rgb = nn.sigmoid(raw['rgb'])
    sigma = self.sigma_activation(jnp.squeeze(raw['alpha'], axis=-1))
    blendw = nn.sigmoid(jnp.squeeze(raw['blendw'], axis=-1))
    shadow_r = None
    if self.use_shadow_model:
      if self.separate_shadow_model:
        shadow_r = self.shadow_model(points_feat, alpha_condition)['shadow_r']
        # shiftted sigmoid to start with low shadow_r
        shadow_r = nn.sigmoid(jnp.squeeze(shadow_r - self.shadow_r_shift, axis=-1))
      else:
        shadow_r = nn.sigmoid(jnp.squeeze(raw['shadow_r'] - self.shadow_r_shift, axis=-1))
      
      # # shadow_r disabled!
      # shadow_r = jnp.zeros_like(shadow_r)

    return rgb, sigma, blendw, shadow_r

  def map_spatial_points(self, points, warp_embed, extra_params, use_warp=True,
                         return_warp_jacobian=False, shadow_warp=False):
    warp_jacobian = None
    if self.use_warp and use_warp:
      warp_fn = jax.vmap(jax.vmap(self.warp_field if not shadow_warp else self.shadow_warp_field, in_axes=(0, 0, None, None)),
                         in_axes=(0, 0, None, None)) # double vmap needed, because the points and warp_embed are in shape (#batch, #point, value)
      warp_out = warp_fn(points,
                         warp_embed,
                         extra_params,
                         return_warp_jacobian)
      if return_warp_jacobian:
        warp_jacobian = warp_out['jacobian']
      warped_points = warp_out['warped_points']
    else:
      warped_points = points

    return warped_points, warp_jacobian

  def map_hyper_points(self, points, hyper_embed, extra_params,
                       hyper_point_override=None):
    """Maps input points to hyper points.

    Args:
      points: the input points.
      hyper_embed: the hyper embeddings.
      extra_params: extra params to pass to the slicing MLP if applicable.
      hyper_point_override: this may contain an override for the hyper points.
        Useful for rendering at specific hyper dimensions.

    Returns:
      An array of hyper points.
    """
    if hyper_point_override is not None:
      hyper_points = jnp.broadcast_to(
          hyper_point_override[:, None, :],
          (*points.shape[:-1], hyper_point_override.shape[-1]))
    elif self.hyper_slice_method == 'axis_aligned_plane': # no non-linear slice
      hyper_points = hyper_embed
    elif self.hyper_slice_method == 'bendy_sheet':
      hyper_points = self.hyper_sheet_mlp(
          points,
          hyper_embed,
          alpha=extra_params['hyper_sheet_alpha'])
    else:
      return None

    return hyper_points

  def map_points(self, points, warp_embed, hyper_embed, extra_params,
                 use_warp=True, return_warp_jacobian=False,
                 hyper_point_override=None, shadow_warp=False):
    """Map input points to warped spatial and hyper points.

    Args:
      points: the input points to warp.
      warp_embed: the warp embeddings.
      hyper_embed: the hyper embeddings.
      extra_params: extra parameters to pass to the warp field/hyper field.
      use_warp: whether to use the warp or not.
      return_warp_jacobian: whether to return the warp jacobian or not.
      hyper_point_override: this may contain an override for the hyper points.
        Useful for rendering at specific hyper dimensions.

    Returns:
      A tuple containing `(warped_points, warp_jacobian)`.
    """
    # Map input points to warped spatial and hyper points.
    spatial_points, warp_jacobian = self.map_spatial_points(
        points, warp_embed, extra_params, use_warp=use_warp,
        return_warp_jacobian=return_warp_jacobian, shadow_warp=shadow_warp)
    hyper_points = self.map_hyper_points(
        points, hyper_embed, extra_params,
        # Override hyper points if present in metadata dict.
        hyper_point_override=hyper_point_override)

    if hyper_points is not None:
      warped_points = jnp.concatenate([spatial_points, hyper_points], axis=-1)
    else:
      warped_points = spatial_points

    return warped_points, warp_jacobian

  def apply_warp(self, points, warp_embed, extra_params):
    # used only for background nomarlizaion
    warp_embed = self.warp_embed(warp_embed)
    return self.warp_field(points, warp_embed, extra_params)

  # TODO: remove this function and refactory background decompose loss
  def get_blendw(self,
                level,
                points,
                viewdirs,
                metadata,
                extra_params,
                use_warp=True,
                metadata_encoded=False,
                return_warp_jacobian=False):

    batch_shape = points.shape[:-1]
    # Create the warp embedding.
    if use_warp:
      if metadata_encoded:
        warp_embed = metadata['encoded_warp']
      else:
        warp_embed = metadata
        warp_embed = self.warp_embed(warp_embed) # embed each key (integer) to 8 digit vector
    else:
      warp_embed = None

    # Create the hyper embedding.
    if self.has_hyper_embed:
      if metadata_encoded:
        hyper_embed = metadata['encoded_hyper']
      elif self.hyper_use_warp_embed:
        hyper_embed = warp_embed # hyper embed is just the warp embed
      else:
        hyper_embed = metadata[self.hyper_embed_key]
        hyper_embed = self.hyper_embed(hyper_embed)
    else:
      hyper_embed = None

    # Broadcast embeddings.
    if warp_embed is not None:
      warp_embed = jnp.broadcast_to( # boardcast the embeddings to each point
          warp_embed[:, jnp.newaxis, :],
          shape=(*batch_shape, warp_embed.shape[-1]))
    if hyper_embed is not None:
      hyper_embed = jnp.broadcast_to(
          hyper_embed[:, jnp.newaxis, :],
          shape=(*batch_shape, hyper_embed.shape[-1]))

    # Map input points to warped spatial and hyper points.
    warped_points, _ = self.map_points(
        points, warp_embed, hyper_embed, extra_params, use_warp=use_warp,
        return_warp_jacobian=return_warp_jacobian) # check if removing overwrite is ok

    _, _, blendw = self.query_template(
        level,
        warped_points,
        viewdirs,
        metadata,
        extra_params=extra_params,
        metadata_encoded=metadata_encoded)

    return blendw

  def blend_results(self, render_mode, rgb_d, sigma_d, rgb_s, sigma_s, blendw, points, warped_points):
    # function for blending dynamic and static outputs together, based on the render mode
    
    blendw_rev = jnp.ones_like(blendw) - blendw
    if render_mode == 'both':
      # combine static and dynamic nerf outputs
      rgb = rgb_d * blendw[...,None] + rgb_s * blendw_rev[...,None]
      sigma = sigma_d * blendw + sigma_s * blendw_rev
    elif render_mode == 'dynamic':
      # render dynamic component only for evaluation
      rgb = rgb_d * blendw[...,None] + jnp.zeros_like(rgb_s) * blendw_rev[...,None]
      sigma = sigma_d * blendw + jnp.zeros_like(sigma_s) * blendw_rev
    elif render_mode == 'dynamic_full':
      # render dynamic component fully, ignoring the value of blendw
      rgb = rgb_d 
      sigma = sigma_d
    elif render_mode == 'dynamic_valid':
      # render valid dynamic component
      # background would just be green
      # blendw = blendw.at[blendw > 0.01].set(1.)
      # blendw = blendw.at[blendw <= 0.01].set(0.)
      blendw = jnp.clip(blendw - 0.01, 0., 0.01) * 100
      blendw_rev = jnp.ones_like(blendw) - blendw
      rgb_s = jnp.ones_like(rgb_s) * jnp.array([[[0.,.5,0.]]])
      rgb = rgb_d * blendw[...,None] + rgb_s * blendw_rev[...,None]
      sigma = sigma_d * blendw + jnp.zeros_like(sigma_s) * blendw_rev
    elif render_mode == 'static':
      # render static component only for evaluation
      rgb = jnp.zeros_like(rgb_d) * blendw[...,None] + rgb_s * blendw_rev[...,None]
      sigma = jnp.zeros_like(sigma_d) * blendw + sigma_s * blendw_rev
    elif render_mode == 'static_full':
      # render static component fully, ignoring the value of blendw
      rgb = rgb_s 
      sigma = sigma_s
    elif render_mode == 'blendw':
      # render blending weights
      rgb = jnp.ones_like(rgb_d) * blendw[...,None]
      sigma = sigma_d * blendw + sigma_s * blendw_rev
    elif render_mode == 'deformation':
      # render the amount of deformation in dynamic component
      # need to ensure range [0,1]. TODO: better normalization
      rgb = jnp.clip((warped_points[...,:3] - points), 0, 1) 
      sigma = sigma_d * blendw + jnp.zeros_like(sigma_s) * blendw_rev
    elif render_mode == 'deformation_norm':
      # render the amount of deformation in dynamic component in norm
      # need to ensure range [0,1]. TODO: better normalization
      rgb = jnp.clip((warped_points[...,:3] - points), 0, 1)
      rgb = jnp.ones_like(rgb) * jnp.sqrt(jnp.sum(rgb ** 2, axis=-1, keepdims=True)) * self.deformation_render_scale
      sigma = sigma_d * blendw + jnp.zeros_like(sigma_s) * blendw_rev
    elif render_mode == 'time':
      # render the warped time coordinate
      rgb = jnp.clip((warped_points[...,3:]), 0, 1)
      sigma = sigma_d * blendw + jnp.zeros_like(sigma_s) * blendw_rev
    else:
      raise NotImplementedError(f'Rendering model {self.render_mode} is not recognized')
    return rgb, sigma

  def render_samples_init_both(self,
                     level,
                     points, 
                     z_vals,
                     directions,
                     viewdirs, 
                     metadata, 
                     extra_params,
                     use_warp=True,
                     metadata_encoded=False,
                     return_warp_jacobian=False,
                     use_sample_at_infinity=False,
                     render_opts=None):
    '''
    render_sample method that is used when wishing to initialize both static and dynamic components separately
    Two components are queried and rendered individually, and both results are returned to obtain photometric loss for training 
    '''
    out = {'points': points}

    batch_shape = points.shape[:-1]
    # Create the warp embedding.
    if use_warp:
      if metadata_encoded:
        warp_embed = metadata['encoded_warp']
      else:
        warp_embed = metadata[self.warp_embed_key]
        warp_embed = self.warp_embed(warp_embed) # embed each key (integer) to 8 digit vector
    else:
      warp_embed = None

    # Create the hyper embedding.
    if self.has_hyper_embed:
      if metadata_encoded:
        hyper_embed = metadata['encoded_hyper']
      elif self.hyper_use_warp_embed:
        hyper_embed = warp_embed # hyper embed is just the warp embed
      else:
        hyper_embed = metadata[self.hyper_embed_key]
        hyper_embed = self.hyper_embed(hyper_embed)
    else:
      hyper_embed = None

    # Broadcast embeddings.
    if warp_embed is not None:
      warp_embed = jnp.broadcast_to( # boardcast the embeddings to each point
          warp_embed[:, jnp.newaxis, :],
          shape=(*batch_shape, warp_embed.shape[-1]))
    if hyper_embed is not None:
      hyper_embed = jnp.broadcast_to(
          hyper_embed[:, jnp.newaxis, :],
          shape=(*batch_shape, hyper_embed.shape[-1]))

    # Map input points to warped spatial and hyper points.
    warped_points, warp_jacobian = self.map_points(
        points, warp_embed, hyper_embed, extra_params, use_warp=use_warp,
        return_warp_jacobian=return_warp_jacobian,
        # Override hyper points if present in metadata dict.
        hyper_point_override=metadata.get('hyper_point'))

    rgb_d, sigma_d, blendw = self.query_template(
        level,
        warped_points,
        viewdirs,
        metadata,
        extra_params=extra_params,
        metadata_encoded=metadata_encoded)

    # Filter densities based on rendering options.
    sigma_d = filter_sigma(points, sigma_d, render_opts)

    if warp_jacobian is not None:
      out['warp_jacobian'] = warp_jacobian
    out['warped_points'] = warped_points

    # query static nerf
    rgb_s, sigma_s = self.static_nerf.query_template(
        level,
        points,
        viewdirs,
        metadata,
        extra_params=extra_params,
        metadata_encoded=metadata_encoded)


    def freeze_blendw_blending():
      # when blendw is frozen, we want both static and dynamic component to be equally trained
      # so run volume rendering on both the outputs, and get simple average
      out_d = model_utils.volumetric_rendering(
        rgb_d,
        sigma_d,
        z_vals,
        directions,
        use_white_background=self.use_white_background,
        sample_at_infinity=use_sample_at_infinity)
      out_s = model_utils.volumetric_rendering(
        rgb_s,
        sigma_s,
        z_vals,
        directions,
        use_white_background=self.use_white_background,
        sample_at_infinity=use_sample_at_infinity)

      out = out_d
      out['rgb_s'] = out_s['rgb']
      for render_mode in self.extra_renders:
        rgb, sigma = self.blend_results(render_mode, rgb_d, sigma_d, rgb_s, sigma_s, jnp.ones_like(blendw) * 0.5, points, warped_points)
        extra_render = model_utils.volumetric_rendering(
                            rgb,
                            sigma,
                            z_vals,
                            directions,
                            use_white_background=self.use_white_background,
                            sample_at_infinity=use_sample_at_infinity)
        out[f'extra_rgb_{render_mode}'] = extra_render['rgb']

      if self.blend_mode == 'nsff':
        out['weights_dynamic'] = out['weights']
        out['weights_static'] = out['weights']
      return out

    def unfreeze_blendw_blending():
      if self.blend_mode == 'old':
            rgb, sigma = self.blend_results(self.render_mode, rgb_d, sigma_d, rgb_s, sigma_s, blendw, points, warped_points)
            out = model_utils.volumetric_rendering(
                rgb,
                sigma,
                z_vals,
                directions,
                use_white_background=self.use_white_background,
                sample_at_infinity=use_sample_at_infinity)

            for render_mode in self.extra_renders:
              rgb, sigma = self.blend_results(render_mode, rgb_d, sigma_d, rgb_s, sigma_s, blendw, points, warped_points)
              extra_render = model_utils.volumetric_rendering(
                                  rgb,
                                  sigma,
                                  z_vals,
                                  directions,
                                  use_white_background=self.use_white_background,
                                  sample_at_infinity=use_sample_at_infinity)
              out[f'extra_rgb_{render_mode}'] = extra_render['rgb']
      elif self.blend_mode == 'nsff':
        out = model_utils.volumetric_rendering_blending(
            rgb_d,
            sigma_d,
            rgb_s,
            sigma_s,
            blendw,
            z_vals,
            directions,
            use_white_background=self.use_white_background,
            sample_at_infinity=use_sample_at_infinity)

        for render_mode in self.extra_renders:
          ex_rgb_d, ex_sigma_d = rgb_d, sigma_d 
          ex_rgb_s, ex_sigma_s = rgb_s, sigma_s
          ex_blendw = blendw
          if render_mode == 'static':
            ex_rgb_d = jnp.zeros_like(ex_rgb_d)
            ex_sigma_d = jnp.zeros_like(ex_sigma_d)
          elif render_mode == 'static_full':
            ex_blendw = jnp.zeros_like(ex_blendw)
          elif render_mode == 'dynamic':
            ex_rgb_s = jnp.zeros_like(ex_rgb_s)
            ex_sigma_s = jnp.zeros_like(ex_sigma_s)
          elif render_mode == 'dynamic_full':
            ex_blendw = jnp.ones_like(ex_blendw)
          elif render_mode == 'blendw':
            ex_rgb_d = jnp.ones_like(ex_rgb_d) # * blendw[...,None]
            ex_rgb_s = jnp.zeros_like(ex_rgb_d)
            # ex_sigma_d = blendw
            # ex_sigma_s = 1. - blendw
          else:
            raise NotImplementedError(f'Rendering model {render_mode} is not recognized')
          
          extra_render = model_utils.volumetric_rendering_blending(
            ex_rgb_d,
            ex_sigma_d,
            ex_rgb_s,
            ex_sigma_s,
            ex_blendw,
            z_vals,
            directions,
            use_white_background=self.use_white_background,
            sample_at_infinity=use_sample_at_infinity)
          out[f'extra_rgb_{render_mode}'] = extra_render['rgb']

      else:
        raise NotImplementedError(f'Blending mode {self.blend_mode} not recognised')
      
      # extra assignment to keep type consistent
      out['rgb_s'] = out['rgb']
      return out

    cond = extra_params['freeze_blendw']
    # this assigned of blendw here is to prevent gradient flow on blendw loss when it's frozen
    blendw = jax.lax.cond(cond, lambda: jnp.ones_like(blendw) * 0.5, lambda: blendw)
    out.update(jax.lax.cond(cond, freeze_blendw_blending, unfreeze_blendw_blending))

    out['blendw'] = blendw

    # Add a map containing the returned points at the median depth.
    depth_indices = model_utils.compute_depth_index(out['weights'])
    med_points = jnp.take_along_axis(
        # Unsqueeze axes: sample axis, coords.
        warped_points, depth_indices[..., None, None], axis=-2)
    out['med_points'] = med_points

    return out

  def render_samples(self,
                     level,
                     points, 
                     z_vals,
                     directions,
                     viewdirs, 
                     metadata, 
                     extra_params,
                     use_warp=True,
                     metadata_encoded=False,
                     return_warp_jacobian=False,
                     use_sample_at_infinity=False,
                     render_opts=None):
    out = {'points': points}

    batch_shape = points.shape[:-1]
    # Create the warp embedding.
    if use_warp:
      if metadata_encoded:
        warp_embed = metadata['encoded_warp']
      else:
        warp_embed = metadata[self.warp_embed_key] # warp
        warp_embed = self.warp_embed(warp_embed) # embed each key (integer) to 8 digit vector
    else:
      warp_embed = None

    # Create the hyper embedding.
    if self.has_hyper_embed:
      if metadata_encoded:
        hyper_embed = metadata['encoded_hyper']
      elif self.hyper_use_warp_embed: # True: hyper does not use appearance but only warp
        hyper_embed = warp_embed # hyper embed is just the warp embed
      else:
        hyper_embed = metadata[self.hyper_embed_key]
        hyper_embed = self.hyper_embed(hyper_embed)
    else:
      hyper_embed = None

    # Broadcast embeddings.
    if warp_embed is not None:
      warp_embed = jnp.broadcast_to( # boardcast the embeddings to each point
          warp_embed[:, jnp.newaxis, :],
          shape=(*batch_shape, warp_embed.shape[-1]))
    if hyper_embed is not None:
      hyper_embed = jnp.broadcast_to(
          hyper_embed[:, jnp.newaxis, :],
          shape=(*batch_shape, hyper_embed.shape[-1]))

    # Map input points to warped spatial and hyper points.
    warped_points, warp_jacobian = self.map_points(
        points, warp_embed, hyper_embed, extra_params, use_warp=use_warp,
        return_warp_jacobian=return_warp_jacobian,
        # Override hyper points if present in metadata dict.
        hyper_point_override=metadata.get('hyper_point'))

    rgb_d, sigma_d, blendw, shadow_r = self.query_template(
        level,
        warped_points,
        viewdirs,
        metadata,
        extra_params=extra_params,
        metadata_encoded=metadata_encoded)

    if self.use_shadow_model:
      # # generate predictions for shadow using shadow warp network
      # warped_points_shadow, _ = self.map_points(
      #     points, warp_embed, hyper_embed, extra_params, use_warp=use_warp,
      #     return_warp_jacobian=return_warp_jacobian,
      #     hyper_point_override=metadata.get('hyper_point'), shadow_warp=True)

      # rgb_d_s, sigma_d_s, blendw_s, shadow_r = self.query_template(
      #     level,
      #     warped_points_shadow,
      #     viewdirs,
      #     metadata,
      #     extra_params=extra_params,
      #     metadata_encoded=metadata_encoded)

      # get shadow_r from shadow mlp
      # shadow_r = self.shadow_r_mlp(warp_embed[:, 0, :]) * jnp.ones_like(sigma_d_s)
      # shadow_r = nn.sigmoid(shadow_r)
      
      # # sigma version: only apply shadow effects where sigma_d is above a threshold
      # # sigma_threshold = 0.0001
      # mask = jnp.where(sigma_d_s > self.sigma_threshold, 1., 0.) 
      mask = jnp.ones_like(shadow_r)
      shadow_r = shadow_r * mask
      # shadow_r = jnp.zeros_like(shadow_r)
    else:
      shadow_r = jnp.zeros_like(sigma_d)

    out['shadow_r'] = shadow_r

    # Filter densities based on rendering options.
    sigma_d = filter_sigma(points, sigma_d, render_opts)

    if warp_jacobian is not None:
      out['warp_jacobian'] = warp_jacobian
    out['warped_points'] = warped_points

    # query static nerf
    rgb_s, sigma_s = self.static_nerf.query_template(
        level,
        points,
        viewdirs,
        metadata,
        extra_params=extra_params,
        metadata_encoded=metadata_encoded)

    blendw = jax.lax.cond(
      extra_params['freeze_blendw'], 
      lambda: jnp.ones_like(blendw) * extra_params['freeze_blendw_value'], 
      lambda: blendw
      )

    if self.use_spatial_loss:
      # query addition results with jittered point locations
      # only their density values will be kept for computing regularizing terms
      _, sub_key = jax.random.split(self.make_rng(level))
      jitter = jax.random.normal(sub_key, shape=points.shape) * self.jitter_std
      jittered_points = points + jitter

      warped_jpoints, _ = self.map_points(
          jittered_points, warp_embed, hyper_embed, extra_params, use_warp=use_warp,
          return_warp_jacobian=return_warp_jacobian,
          # Override hyper points if present in metadata dict.
          hyper_point_override=metadata.get('hyper_point'))

      _, exs_sigma_d, exs_blendw, _ = self.query_template(
          level,
          warped_jpoints,
          viewdirs,
          metadata,
          extra_params=extra_params,
          metadata_encoded=metadata_encoded)

      _, exs_sigma_s = self.static_nerf.query_template(
          level,
          jittered_points,
          viewdirs,
          metadata,
          extra_params=extra_params,
          metadata_encoded=metadata_encoded)
      
      out['exs_blendw'] = exs_sigma_d / jnp.clip(exs_sigma_d + exs_sigma_s, 1e-19)
    

    if self.handle_motion_blur:
      # query additoinal rays to simulate motion blur effect
      # time ratio controls the merging of current time stamp with nearby ones
      if (not use_warp) or metadata_encoded or (not self.has_hyper_embed) or (not self.hyper_use_warp_embed):
        raise NotImplementedError("Current DecomposeNeRF configuration doesn't support motion blur handlling ")

      now_warp_embed = warp_embed[:,0,:]
      out_raw = self.blur_mlp(now_warp_embed)
      blur_w, time_ratio = out_raw[...,0,None], out_raw[...,1,None]
      # because blur_w is used for both pre and post blur, the maximum value should be 0.5
      blur_w = nn.sigmoid(blur_w) / 2.
      time_ratio = nn.sigmoid(time_ratio)

      # compute for pre frame
      pre_warp_embed = jnp.clip(metadata[self.warp_embed_key] - 1, 0, self.num_warp_embeds-1)
      pre_warp_embed = self.warp_embed(pre_warp_embed)
      mb_embeds = [now_warp_embed * time_ratio + pre_warp_embed * (1-time_ratio)]
      # compute for post frame
      post_warp_embed = jnp.clip(metadata[self.warp_embed_key] + 1, 0, self.num_warp_embeds-1)
      post_warp_embed = self.warp_embed(post_warp_embed)
      mb_embeds.append(now_warp_embed * time_ratio + post_warp_embed * (1-time_ratio))

      rgb_ray_blur = 0.
      rgb_adjacent = []

      for warp_embed in mb_embeds:
        # Broadcast embeddings.
        if warp_embed is not None:
          warp_embed = jnp.broadcast_to( # boardcast the embeddings to each point
              warp_embed[:, jnp.newaxis, :],
              shape=(*batch_shape, warp_embed.shape[-1]))
        hyper_embed = warp_embed

        # Map input points to warped spatial and hyper points.
        warped_points, warp_jacobian = self.map_points(
            points, warp_embed, hyper_embed, extra_params, use_warp=use_warp,
            return_warp_jacobian=return_warp_jacobian,
            # Override hyper points if present in metadata dict.
            hyper_point_override=metadata.get('hyper_point'))

        rgb_b, sigma_b, blendw_b = self.query_template(
          level,
          warped_points,
          viewdirs,
          {}, # metadata removed, not used and not supported
          extra_params=extra_params,
          metadata_encoded=metadata_encoded)

        # Filter densities based on rendering options.
        sigma_b = filter_sigma(points, sigma_b, render_opts)
        out_rgb = model_utils.volumetric_rendering_blending(
            rgb_b,
            sigma_b,
            rgb_s,
            sigma_s,
            blendw_b,
            z_vals,
            directions,
            use_white_background=self.use_white_background,
            sample_at_infinity=use_sample_at_infinity
          )['rgb'] 
        rgb_adjacent.append(out_rgb)
        rgb_ray_blur += out_rgb * blur_w
        
        
    if self.blend_mode == 'old':
      rgb, sigma = self.blend_results(self.render_mode, rgb_d, sigma_d, rgb_s, sigma_s, blendw, points, warped_points)
      out.update(model_utils.volumetric_rendering(
          rgb,
          sigma,
          z_vals,
          directions,
          use_white_background=self.use_white_background,
          sample_at_infinity=use_sample_at_infinity))

      for render_mode in self.extra_renders:
        rgb, sigma = self.blend_results(render_mode, rgb_d, sigma_d, rgb_s, sigma_s, blendw, points, warped_points)
        extra_render = model_utils.volumetric_rendering(
                            rgb,
                            sigma,
                            z_vals,
                            directions,
                            use_white_background=self.use_white_background,
                            sample_at_infinity=use_sample_at_infinity)
        out[f'extra_rgb_{render_mode}'] = extra_render['rgb']
    elif self.blend_mode == 'nsff':
      # blendw = jnp.zeros_like(blendw)
      out.update(model_utils.volumetric_rendering_blending(
          rgb_d,
          sigma_d,
          rgb_s,
          sigma_s,
          blendw,
          z_vals,
          directions,
          use_white_background=self.use_white_background,
          sample_at_infinity=use_sample_at_infinity))
      
      if self.handle_motion_blur:
        rgb_no_blur = out['rgb']
        out['rgb'] = (1 - 2*blur_w) * out['rgb'] + rgb_ray_blur

      for render_mode in self.extra_renders:
        ex_rgb_d, ex_sigma_d = rgb_d, sigma_d 
        ex_rgb_s, ex_sigma_s = rgb_s, sigma_s
        ex_blendw = blendw
        if render_mode == 'static':
          ex_rgb_d = jnp.zeros_like(ex_rgb_d) # [3,144], [2, 690], [2,1010]
          ex_sigma_d = jnp.zeros_like(ex_sigma_d)
        elif render_mode == 'static_full':
          # ex_rgb_d = jnp.zeros_like(ex_rgb_d)
          # ex_sigma_d = jnp.zeros_like(ex_sigma_d)
          ex_blendw = jnp.zeros_like(ex_blendw)
        elif render_mode == 'dynamic':
          ex_rgb_s = jnp.zeros_like(ex_rgb_s)
          ex_sigma_s = jnp.zeros_like(ex_sigma_s)
        elif render_mode == 'dynamic_full':
          ex_blendw = jnp.ones_like(ex_blendw)
        elif render_mode == 'blendw':
          ex_rgb_d = jnp.ones_like(ex_rgb_d) # * blendw[...,None]
          ex_rgb_s = jnp.zeros_like(ex_rgb_d)
        elif render_mode == 'deformation_norm':
          # render the amount of deformation in dynamic component in norm
          # need to ensure range [0,1]. TODO: better normalization
          rgb = jnp.clip((warped_points[...,:3] - points), 0, 1)
          ex_rgb_d = ex_rgb_s = jnp.ones_like(rgb) * jnp.sqrt(jnp.sum(rgb ** 2, axis=-1, keepdims=True)) * self.deformation_render_scale
          ex_sigma_s = jnp.zeros_like(ex_sigma_s)
        elif render_mode == 'ray_segmentation':
          # render whether the sum of blendw on a ray is above a threshold or not
          # volume rendering not needed
          threshold = 0.5
          clip_threshold=0.00001
          ex_blendw = jnp.clip(blendw, a_min=clip_threshold)
          blendw_sum = jnp.sum(ex_blendw, -1, keepdims=True) 
          mask = jnp.where(blendw_sum < threshold, 0., 1.) 

          out[f'extra_rgb_{render_mode}'] =  mask * jnp.array([1,0,0])
          continue
        elif render_mode == 'ray_entropy_loss':
          # render the amount of blendw entropy loss applied on each ray
          # volume rendering not needed
          threshold = 0.5
          clip_threshold=0.00001
          ex_blendw = jnp.clip(blendw, a_min=clip_threshold)
          blendw_sum = jnp.sum(ex_blendw, -1, keepdims=True) 
          mask = jnp.where(blendw_sum < threshold, 0., 1.) 
          p = ex_blendw / blendw_sum 
          entropy = mask * -jnp.mean(p * jnp.log(p), -1, keepdims=True)
          # maximum value of -p * jnp.log(p) is 1/e
          entropy *= jnp.e

          out[f'extra_rgb_{render_mode}'] = entropy * jnp.array([1,0,0])
          continue
        elif render_mode == 'shadow_loss_segmentation':
          # render the parts where shadow loss is casted
          # volume rendering not needed
          threshold = 0.2
          mask = jnp.where(threshold < blendw, 1., 0.) * jnp.where(blendw < 1-threshold, 1., 0.) 
          diff = jnp.average(nn.relu(rgb_d - rgb_s), axis=-1)
          mask = jnp.where(diff > 0, 1., 0.) * mask
          mask = jnp.max(mask, axis=-1, keepdims=True)

          out[f'extra_rgb_{render_mode}'] = mask * jnp.array([0,1,0])
          continue
        elif render_mode == 'deblur':
          # the deblurred version of image
          assert self.handle_motion_blur, 'deblur extra rendering can only be used when self.handle_motion_blur is true'
          out[f'extra_rgb_{render_mode}'] = rgb_no_blur
          continue
        elif render_mode == 'blur_pre':
          # the blurry pre image
          assert self.handle_motion_blur, 'blur_pre extra rendering can only be used when self.handle_motion_blur is true'
          out[f'extra_rgb_{render_mode}'] = rgb_adjacent[0]
          continue
        elif render_mode == 'blur_post':
          # the blurry post image
          assert self.handle_motion_blur, 'blur_post extran rendering can only be used when self.handle_motion_blur is true'
          out[f'extra_rgb_{render_mode}'] = rgb_adjacent[1]
          continue
        else:
          raise NotImplementedError(f'Rendering model {render_mode} is not recognized')
        
        extra_render = model_utils.volumetric_rendering_blending(
          ex_rgb_d,
          ex_sigma_d,
          ex_rgb_s,
          ex_sigma_s,
          ex_blendw,
          z_vals,
          directions,
          use_white_background=self.use_white_background,
          sample_at_infinity=use_sample_at_infinity)
        out[f'extra_rgb_{render_mode}'] = extra_render['rgb']
    elif self.blend_mode == 'add':

      # if use_sample_at_infinity:
      #   # dynamic component should not use the last sample located at infinite far away plane
      #   # this allows rays on empty dynamic component to not terminate 
      #   sigma_d = jnp.concatenate([
      #     sigma_d, 
      #     jnp.zeros_like(sigma_d[..., -1:])
      #     ], axis=-1)
      #   rgb_d = jnp.concatenate([
      #     rgb_d, 
      #     jnp.zeros_like(rgb_d[..., -1:, :])
      #     ], axis=-2)
      #   sigma_s = jnp.concatenate([
      #     sigma_s[..., :-1], 
      #     jnp.zeros_like(sigma_s[..., -1:]),
      #     sigma_s[..., -1:]
      #     ], axis=-1)
      #   rgb_s = jnp.concatenate([
      #     rgb_s[..., :-1, :], 
      #     jnp.zeros_like(rgb_s[..., -1:, :]),
      #     rgb_s[..., -1:, :]
      #     ], axis=-2)
      # #   # sigma_d = jnp.concatenate([sigma_d[..., :-1], jnp.zeros_like(sigma_d[..., -1:])], axis=-1)
      # #   sigma_d = jnp.concatenate([sigma_d[..., :-1], jnp.ones_like(sigma_d[..., -1:]) * 0], axis=-1)
      # #   # sigma_d = jax.lax.cond(
      # #   #   level=='fine', 
      # #   #   lambda: jnp.concatenate([sigma_d[..., :-1], jnp.zeros_like(sigma_d[..., -1:])], axis=-1), 
      # #   #   lambda: sigma_d
      # #   #   )

      blendw = sigma_d / jnp.clip(sigma_d + sigma_s, 1e-19)
      out.update(model_utils.volumetric_rendering_addition(
          rgb_d,
          sigma_d,
          rgb_s,
          sigma_s,
          blendw,
          shadow_r,
          z_vals,
          directions,
          use_white_background=self.use_white_background,
          sample_at_infinity=use_sample_at_infinity))
      
      if self.handle_motion_blur:
        rgb_no_blur = out['rgb']
        out['rgb'] = (1 - 2*blur_w) * out['rgb'] + rgb_ray_blur

      extra_renders = list(self.extra_renders)
      if 'mask' in extra_renders:
        # render mask in the last place to re-use previous renderings
        extra_renders.remove('mask')
        extra_renders.append('mask')

      for render_mode in extra_renders:
        ex_rgb_d, ex_sigma_d = rgb_d, sigma_d 
        ex_rgb_s, ex_sigma_s = rgb_s, sigma_s
        ex_blendw = blendw
        ex_shadow_r = shadow_r
        ex_use_white_background = self.use_white_background
        ex_use_green_background = False
        if render_mode == 'static':
          ex_rgb_d = jnp.zeros_like(ex_rgb_d)
          ex_sigma_d = jnp.zeros_like(ex_sigma_d)
          ex_shadow_r = jnp.zeros_like(shadow_r)
        elif render_mode == 'dynamic':
          ex_rgb_s = jnp.zeros_like(ex_rgb_s)
          ex_sigma_s = jnp.zeros_like(ex_sigma_s)
          ex_use_white_background = True
        elif render_mode == 'dynamic_green':
          ex_rgb_s = jnp.zeros_like(ex_rgb_s)
          ex_sigma_s = jnp.zeros_like(ex_sigma_s)
          ex_use_green_background = True
        elif render_mode == 'blendw':
          out[f'extra_rgb_{render_mode}'] =  out['rgb_blendw']
          continue
        elif render_mode == 'mask':
          # render a thresholded blendw as mask
          mask = jnp.where(out['rgb_blendw'] > self.blendw_mask_threshold, 1., 0.)
          if self.use_shadow_model:
            # consider shadow into the mask
            if 'extra_rgb_shadow' not in out:
              raise NotImplementedError('Must render extra_rgb_shadow before mask if shadow model is enabled')
            mask = jnp.where(out['rgb_blendw'] + out['extra_rgb_shadow'] > self.blendw_mask_threshold, 1., 0.)

          out[f'extra_rgb_{render_mode}'] =  mask
          continue
        # elif render_mode == 'deformation_norm':
        #   # Not supported yet!
        #   rgb = jnp.clip((warped_points[...,:3] - points), 0, 1)
        #   ex_rgb_d = ex_rgb_s = jnp.ones_like(rgb) * jnp.sqrt(jnp.sum(rgb ** 2, axis=-1, keepdims=True)) * self.deformation_render_scale
        #   ex_sigma_s = jnp.zeros_like(ex_sigma_s)
        elif render_mode == 'shadow':
          # render the shadow_r
          # only renders the static scene
          ex_rgb_d = jnp.zeros_like(ex_rgb_d)
          ex_sigma_d = jnp.zeros_like(ex_sigma_d)
          ex_rgb_s = jnp.ones_like(ex_rgb_s)
          ex_shadow_r = 1 - shadow_r
        elif render_mode == 'regular_no_shadow':
          ex_shadow_r = jnp.zeros_like(shadow_r)
        elif render_mode == 'ray_segmentation':
          # render whether the sum of blendw on a ray is above a threshold or not
          # volume rendering not needed
          threshold = 0.5
          clip_threshold=0.00001
          ex_blendw = jnp.clip(blendw, a_min=clip_threshold)
          blendw_sum = jnp.sum(ex_blendw, -1, keepdims=True) 
          mask = jnp.where(blendw_sum < threshold, 0., 1.) 

          out[f'extra_rgb_{render_mode}'] =  mask * jnp.array([1,0,0])
          continue
        elif render_mode == 'ray_entropy_loss':
          # render the amount of blendw entropy loss applied on each ray
          # volume rendering not needed
          threshold = 0.5
          clip_threshold=0.00001
          ex_blendw = jnp.clip(blendw, a_min=clip_threshold)
          blendw_sum = jnp.sum(ex_blendw, -1, keepdims=True) 
          mask = jnp.where(blendw_sum < threshold, 0., 1.) 
          p = ex_blendw / blendw_sum 
          entropy = mask * -jnp.mean(p * jnp.log(p), -1, keepdims=True)
          # maximum value of -p * jnp.log(p) is 1/e
          entropy *= jnp.e

          out[f'extra_rgb_{render_mode}'] = entropy * jnp.array([1,0,0])
          continue
        elif render_mode == 'shadow_loss_segmentation':
          # render the parts where shadow loss is casted
          # volume rendering not needed
          threshold = 0.2
          mask = jnp.where(threshold < blendw, 1., 0.) * jnp.where(blendw < 1-threshold, 1., 0.) 
          diff = jnp.average(nn.relu(rgb_d - rgb_s), axis=-1)
          mask = jnp.where(diff > 0, 1., 0.) * mask
          mask = jnp.max(mask, axis=-1, keepdims=True)

          out[f'extra_rgb_{render_mode}'] = mask * jnp.array([0,1,0])
          continue
        else:
          raise NotImplementedError(f'Rendering model {render_mode} is not recognized')
        
        extra_render = model_utils.volumetric_rendering_addition(
          ex_rgb_d,
          ex_sigma_d,
          ex_rgb_s,
          ex_sigma_s,
          ex_blendw,
          ex_shadow_r,
          z_vals,
          directions,
          use_white_background=ex_use_white_background,
          use_green_background=ex_use_green_background,
          sample_at_infinity=use_sample_at_infinity)
        out[f'extra_rgb_{render_mode}'] = extra_render['rgb']

    else:
      raise NotImplementedError(f'Blending mode {self.blend_mode} not recognised')

    # Add a map containing the returned points at the median depth.
    depth_indices = model_utils.compute_depth_index(out['weights'])
    med_points = jnp.take_along_axis(
        # Unsqueeze axes: sample axis, coords.
        warped_points, depth_indices[..., None, None], axis=-2)
    out['med_points'] = med_points

    out['sigma_d'] = sigma_d
    out['rgb_d'] = rgb_d
    out['rgb_s'] = rgb_s
    out['blendw'] = blendw      

    return out

  def __call__(
      self,
      rays_dict: Dict[str, Any],
      extra_params: Dict[str, Any],
      metadata_encoded=False,
      use_warp=True,
      return_points=False,
      return_weights=False,
      return_warp_jacobian=False,
      near=None,
      far=None,
      use_sample_at_infinity=None,
      render_opts=None,
      deterministic=False,
  ):
    """Decompose Nerf Model.

    Args:
      rays_dict: a dictionary containing the ray information. Contains:
        'origins': the ray origins.
        'directions': unit vectors which are the ray directions.
        'viewdirs': (optional) unit vectors which are viewing directions.
        'metadata': a dictionary of metadata indices e.g., for warping.
      extra_params: parameters for the warp e.g., alpha.
      metadata_encoded: if True, assume the metadata is already encoded.
      use_warp: if True use the warp field (if also enabled in the model).
      return_points: if True return the points (and warped points if
        applicable).
      return_weights: if True return the density weights.
      return_warp_jacobian: if True computes and returns the warp Jacobians.
      near: if not None override the default near value.
      far: if not None override the default far value.
      use_sample_at_infinity: override for `self.use_sample_at_infinity`.
      render_opts: an optional dictionary of render options.
      deterministic: whether evaluation should be deterministic.

    Returns:
      ret: list, [(rgb, disp, acc), (rgb_coarse, disp_coarse, acc_coarse)]
    """
    use_warp = self.use_warp and use_warp
    # Extract viewdirs from the ray array
    origins = rays_dict['origins']
    directions = rays_dict['directions']
    metadata = rays_dict['metadata']
    if 'viewdirs' in rays_dict:
      viewdirs = rays_dict['viewdirs']
    else:  # viewdirs are normalized rays_d
      viewdirs = directions

    if near is None:
      near = self.near
    if far is None:
      far = self.far
    if use_sample_at_infinity is None:
      use_sample_at_infinity = self.use_sample_at_infinity

    def find_pixel(u,v):
      for i in range(rays_dict['pixels'].val.shape[0]):
        for j in range(rays_dict['pixels'].val.shape[1]):
          if (rays_dict['pixels'].val[i,j] == [u + 0.5, v + 0.5]).all():
            return ([i,j])

    # Evaluate coarse samples.
    z_vals, points = model_utils.sample_along_rays(
        self.make_rng('coarse'), origins, directions, self.num_coarse_samples,
        near, far, self.use_stratified_sampling,
        self.use_linear_disparity)
    coarse_ret = self.render_samples(
        'coarse',
        points,
        z_vals,
        directions,
        viewdirs,
        metadata,
        extra_params,
        use_warp=use_warp,
        metadata_encoded=metadata_encoded,
        return_warp_jacobian=return_warp_jacobian,
        use_sample_at_infinity=self.use_sample_at_infinity)
    out = {'coarse': coarse_ret}

    # Evaluate fine samples.
    if self.num_fine_samples > 0:
      z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
      z_vals, points = model_utils.sample_pdf(
          self.make_rng('fine'), z_vals_mid, coarse_ret['weights'][..., 1:-1],
          origins, directions, z_vals, self.num_fine_samples,
          self.use_stratified_sampling)
      out['fine'] = self.render_samples(
          'fine',
          points,
          z_vals,
          directions,
          viewdirs,
          metadata,
          extra_params,
          use_warp=use_warp,
          metadata_encoded=metadata_encoded,
          return_warp_jacobian=return_warp_jacobian,
          use_sample_at_infinity=use_sample_at_infinity,
          render_opts=render_opts)


    if not return_weights:
      del out['coarse']['weights']
      del out['fine']['weights']

    if not return_points:
      del out['coarse']['points']
      del out['coarse']['warped_points']
      del out['fine']['points']
      del out['fine']['warped_points']

    return out


def construct_nerf(key, batch_size: int, embeddings_dict: Dict[str, int],
                   near: float, far: float):
  """Neural Randiance Field.

  Args:
    key: jnp.ndarray. Random number generator.
    batch_size: the evaluation batch size used for shape inference.
    embeddings_dict: a dictionary containing the embeddings for each metadata
      type.
    near: the near plane of the scene.
    far: the far plane of the scene.

  Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """
  model = NerfModel(
      embeddings_dict=immutabledict.immutabledict(embeddings_dict),
      near=near,
      far=far)

  init_rays_dict = {
      'origins': jnp.ones((batch_size, 3), jnp.float32),
      'directions': jnp.ones((batch_size, 3), jnp.float32),
      'metadata': {
          'warp': jnp.ones((batch_size, 1), jnp.uint32),
          'camera': jnp.ones((batch_size, 1), jnp.uint32),
          'appearance': jnp.ones((batch_size, 1), jnp.uint32),
          'time': jnp.ones((batch_size, 1), jnp.float32),
      }
  }
  extra_params = {
      'nerf_alpha': 0.0,
      'warp_alpha': 0.0,
      'hyper_alpha': 0.0,
      'hyper_sheet_alpha': 0.0,
  }

  key, key1, key2 = random.split(key, 3)
  params = model.init({
      'params': key,
      'coarse': key1,
      'fine': key2
  }, init_rays_dict, extra_params=extra_params)['params']

  return model, params


def construct_decompose_nerf(key, batch_size: int, embeddings_dict: Dict[str, int],
                   near: float, far: float):
  """Neural Randiance Field.

  Args:
    key: jnp.ndarray. Random number generator.
    batch_size: the evaluation batch size used for shape inference.
    embeddings_dict: a dictionary containing the embeddings for each metadata
      type.
    near: the near plane of the scene.
    far: the far plane of the scene.

  Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """

  model = DecomposeNerfModel(
      embeddings_dict=immutabledict.immutabledict(embeddings_dict),
      near=near,
      far=far)

  init_rays_dict = {
      'origins': jnp.ones((batch_size, 3), jnp.float32),
      'directions': jnp.ones((batch_size, 3), jnp.float32),
      'metadata': {
          'warp': jnp.ones((batch_size, 1), jnp.uint32),
          'camera': jnp.ones((batch_size, 1), jnp.uint32),
          'appearance': jnp.ones((batch_size, 1), jnp.uint32),
          'time': jnp.ones((batch_size, 1), jnp.float32),
      }
  }
  extra_params = {
      'nerf_alpha': 0.0,
      'warp_alpha': 0.0,
      'hyper_alpha': 0.0,
      'hyper_sheet_alpha': 0.0,
      'freeze_blendw': False,
      'freeze_blendw_value': 0.0
  }

  key, key1, key2 = random.split(key, 3)
  params = model.init({
      'params': key,
      'coarse': key1,
      'fine': key2
  }, init_rays_dict, extra_params=extra_params)['params']

  return model, params
