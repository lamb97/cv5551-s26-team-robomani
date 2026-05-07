from typing import Dict, List
import copy
import pathlib

import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class DemoXYZImageDataset(BaseImageDataset):
    def __init__(self,
            dataset_path: str,
            shape_meta: dict,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            exclude_first_n_demos=4,
            demo_prefix='demo_',
            action_source='state_delta',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None):
        super().__init__()

        dataset_dir = pathlib.Path(dataset_path).expanduser()
        assert dataset_dir.is_dir()

        obs_shape_meta = shape_meta['obs']
        rgb_keys = list()
        lowdim_keys = list()
        image_sizes = dict()
        lowdim_shapes = dict()
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                rgb_keys.append(key)
                c, h, w = attr['shape']
                assert c == 3
                image_sizes[key] = (int(w), int(h))
            elif obs_type == 'low_dim':
                lowdim_keys.append(key)
                lowdim_shapes[key] = tuple(attr['shape'])
            else:
                raise ValueError(f'Unsupported observation type: {obs_type}')

        if len(rgb_keys) != 1:
            raise ValueError('DemoXYZImageDataset expects exactly one rgb observation key.')
        if lowdim_shapes != {'eef_xyz': (3,)}:
            raise ValueError(
                'DemoXYZImageDataset expects shape_meta.obs to define only '
                '`eef_xyz` with shape [3] for low-dim input.')
        action_shape = tuple(shape_meta['action']['shape'])
        if action_shape != (3,):
            raise ValueError('DemoXYZImageDataset expects action shape [3].')

        cv2.setNumThreads(1)

        demo_dirs = sorted(
            path for path in dataset_dir.iterdir()
            if path.is_dir() and path.name.startswith(demo_prefix)
        )
        demo_dirs = demo_dirs[int(exclude_first_n_demos):]
        if not demo_dirs:
            raise RuntimeError('No demos left after exclusion.')

        replay_buffer = ReplayBuffer.create_empty_numpy()
        image_paths: List[str] = list()

        for demo_dir in demo_dirs:
            steps_dir = demo_dir / 'steps'
            step_npz_paths = sorted(steps_dir.glob('*.npz'))
            if not step_npz_paths:
                continue

            eef_xyz = list()
            raw_actions = list()
            episode_image_paths = list()
            for npz_path in step_npz_paths:
                png_path = npz_path.with_suffix('.png')
                if not png_path.is_file():
                    raise FileNotFoundError(f'Missing image for {npz_path}')
                with np.load(npz_path, allow_pickle=False) as step_data:
                    eef_xyz.append(step_data['ee_pose_mm_rad'][:3].astype(np.float32))
                    raw_actions.append(step_data['action'][:3].astype(np.float32))
                episode_image_paths.append(str(png_path))

            if not episode_image_paths:
                continue

            eef_xyz = np.stack(eef_xyz, axis=0)
            raw_actions = np.stack(raw_actions, axis=0)
            if len(eef_xyz) < 2:
                continue

            if action_source == 'state_delta':
                # Use forward state differences so action_t describes
                # the motion from obs_t to obs_{t+1}.
                obs_eef_xyz = eef_xyz[:-1]
                actions = np.diff(eef_xyz, axis=0)
                episode_image_paths = episode_image_paths[:-1]
            elif action_source == 'recorded_action':
                obs_eef_xyz = eef_xyz
                actions = raw_actions
            else:
                raise ValueError(f'Unsupported action_source: {action_source}')

            replay_buffer.add_episode({
                'eef_xyz': obs_eef_xyz.astype(np.float32),
                'action': actions.astype(np.float32)
            })
            image_paths.extend(episode_image_paths)

        if replay_buffer.n_episodes == 0:
            raise RuntimeError('No valid demos found in dataset path.')
        if replay_buffer.n_steps != len(image_paths):
            raise RuntimeError('Image path count does not match replay buffer length.')

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        key_first_k = dict()
        if n_obs_steps is not None:
            key_first_k['eef_xyz'] = n_obs_steps

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)

        self.replay_buffer = replay_buffer
        self.image_paths = image_paths
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.image_sizes = image_sizes
        self.sampler = sampler
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
            key_first_k={'eef_xyz': self.n_obs_steps} if self.n_obs_steps is not None else dict()
        )
        val_set.train_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        normalizer['eef_xyz'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['eef_xyz'])
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def _load_image(self, path: str, out_size) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'Failed to read image: {path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, out_size, interpolation=cv2.INTER_AREA)
        return image

    def _sample_image_sequence(self, idx: int) -> np.ndarray:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = \
            self.sampler.indices[idx]
        sample_paths = self.image_paths[buffer_start_idx:buffer_end_idx]
        if not sample_paths:
            raise RuntimeError('Empty image sequence sampled.')

        rgb_key = self.rgb_keys[0]
        out_size = self.image_sizes[rgb_key]
        loaded = np.stack(
            [self._load_image(path, out_size=out_size) for path in sample_paths],
            axis=0
        )

        if (sample_start_idx > 0) or (sample_end_idx < self.sampler.sequence_length):
            padded = np.zeros(
                (self.sampler.sequence_length,) + loaded.shape[1:],
                dtype=loaded.dtype)
            if sample_start_idx > 0:
                padded[:sample_start_idx] = loaded[0]
            if sample_end_idx < self.sampler.sequence_length:
                padded[sample_end_idx:] = loaded[-1]
            padded[sample_start_idx:sample_end_idx] = loaded
            loaded = padded
        return loaded

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        sample = self.sampler.sample_sequence(idx)
        image = self._sample_image_sequence(idx)

        t_slice = slice(self.n_obs_steps)
        obs_dict = {
            'image': np.moveaxis(image[t_slice], -1, 1).astype(np.float32) / 255.0,
            'eef_xyz': sample['eef_xyz'][t_slice].astype(np.float32)
        }
        data = {
            'obs': obs_dict,
            'action': sample['action'].astype(np.float32)
        }
        return dict_apply(data, torch.from_numpy)
