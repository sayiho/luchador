"""Atari Environment based on Arcade Learning Environment"""

from __future__ import absolute_import

import sys
import os
import logging

import numpy as np
from scipy.misc import imresize
from ale_python_interface import ALEInterface

from luchador.util import pprint_dict
from ..base import BaseEnvironment, Outcome

_LG = logging.getLogger(__name__)

__all__ = ['ALEEnvironment']

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROM_DIR = os.path.join(_DIR, 'rom')


class ALEEnvironment(BaseEnvironment):
    """Atari Environment"""

    @staticmethod
    def get_roms():
        """Get the list of ROMs available

        Returns:
          list of srting: Names of available ROMs
        """
        return [rom for rom in os.listdir(_ROM_DIR)
                if rom.endswith('.bin')]

    def __init__(
            self, rom,
            mode='train',
            width=None,
            height=None,
            grayscale=True,
            repeat_action=4,
            buffer_frames=2,
            preprocess_mode='max',
            minimal_action_set=True,
            random_seed=0,
            random_start=None,
            display_screen=False,
            play_sound=False,
            record_screen_path=None,
            record_sound_filename=None,
    ):
        """Initialize ALE Environment with the given parmeters

        Args:
          rom (str): ROM name. Use `get_roms` for the list of available ROMs.

          mode (str): When `train`, a loss of life is considered as terminal
                      condition. When `test`, a loss of life is not considered
                      as terminal condition.

          width (int or None): Output screen width.
                               If None the original width is used.

          height (int or None): Output screen height.
                                If None the original height is used.

          grayscale (bool):
              If True, output screen is gray scale and has no color channel.
              i.e. output shape == (h, w). Otherwise output screen has color
              channel with shape (h, w, 3)

          repeat_action (int):
              When calling `step` method, action is repeated for this numebr
              of times, internally, unless a terminal condition is met.

          minimal_action_set (bool):
              When True, `n_actions` property reports actions only meaningfull
              to the loaded ROM. Otherwise all the 18 actions are dounted.

          random_seed (int): ALE's random seed

          random_start (int or None): When given, at most this number of frames
              are played with action == 0. This technique is often used to
              prevent environment from transitting deterministically, but
              in case of ALE, as reset command does not reset system state
              we may not need to do this. (TODO: Check.)

          buffer_frames (int): The number of latest frame to preprocess.

          preprocess_mode (str): Either `max` or `average`. When obtaining
                                 observation, pixel-wise maximum or average
                                 over buffered frames are taken before resizing

          display_screen (bool): Display sceen when True.

          play_sound (bool): Play sound

          record_screen_path (str): Passed to ALE. Save the original screens
              into the path.
              Note: that this is different from the observation returned by
              `step` method.

          record_screen_filename (str): Passed to ALE. Save sound to a file.

        """
        if mode not in ['test', 'train']:
            raise ValueError('`mode` must be either `test` or `train`')

        if preprocess_mode not in ['max', 'average']:
            raise ValueError(
                '`preprocess_mode` must be either `max` or `average`')

        if repeat_action < 1:
            raise ValueError(
                '`repeat_action` must be integer greater than 0')

        if random_start and random_start < 1:
            raise ValueError(
                '`random_start` must be `None` or integer greater than 0'
            )

        if not rom.endswith('.bin'):
            rom += '.bin'

        rom_path = os.path.join(_ROM_DIR, rom)
        if not os.path.isfile(rom_path):
            raise ValueError('ROM ({}) not found.'.format(rom))

        if display_screen and sys.platform == 'darwin':
            import pygame
            pygame.init()

        # ALE
        self.rom_path = rom_path
        self.random_seed = random_seed
        self.random_start = random_start
        self.display_screen = display_screen
        self.play_sound = play_sound
        self.record_screen_path = record_screen_path
        self.record_sound_filename = record_sound_filename
        self.minimal_action_set = minimal_action_set

        self._init_ale()

        # Buffer
        self.grayscale = grayscale
        self.buffer_frames = buffer_frames
        self.preprocess_mode = preprocess_mode

        self._buffer_index = None
        self._init_buffer()

        # Resize method
        self.width = width
        self.height = height

        self._init_resize()

        # Test/Train mode
        self.mode = mode
        self.life_lost = False

        self.repeat_action = repeat_action
        self.random_start = random_start

    def _init_ale(self):
        ale = ALEInterface()
        ale.setBool('sound', self.play_sound)
        ale.setBool('display_screen', self.display_screen)
        ale.setInt('random_seed', self.random_seed)

        # Frame skip is implemented separately
        ale.setInt('frame_skip', 1)
        ale.setBool('color_averaging', False)
        ale.setFloat('repeat_action_probability', 0.0)
        # Somehow this repeat_action_probability has unexpected effect on game.
        # The larger this value is, the more frames games take to restart.
        # And when 1.0 games completely hang
        # We are setting the default value of 0.0 here, expecting that
        # it has no effect as frame_skip == 1
        # This action repeating is agent's concern
        # so we do not implement an equivalent in our wrapper.

        if self.record_screen_path:
            _LG.info('Recording screens: %s', self.record_screen_path)
            if not os.path.exists(self.record_screen_path):
                os.makedirs(self.record_screen_path)
            ale.setString('record_screen_dir', self.record_screen_path)

        if self.record_sound_filename:
            _LG.info('Recording sound: %s', self.record_sound_filename)
            record_sound_dir = os.path.dirname(self.record_sound_filename)
            if not os.path.exists(record_sound_dir):
                os.makedirs(record_sound_dir)
            ale.setBool('sound', True)
            ale.setString('record_sound_filename', self.record_sound_filename)

        ale.loadROM(self.rom_path)

        self._ale = ale
        self._actions = (
            ale.getMinimalActionSet() if self.minimal_action_set else
            ale.getLegalActionSet()
        )

    def _init_buffer(self):
        orig_width, orig_height = self._ale.getScreenDims()
        channel = 1 if self.grayscale else 3
        n_frames = self.buffer_frames

        buffer_shape = (
            (n_frames, orig_height, orig_width) if self.grayscale else
            (n_frames, orig_height, orig_width, channel)
        )
        self._frame_buffer = np.zeros(buffer_shape, dtype=np.uint8)
        self._buffer_index = 0

        self._get_raw_screen = (
            self._ale.getScreenGrayscale if self.grayscale else
            self._ale.getScreenRGB
        )

        self._get_screen = (
            self._get_max_buffer if self.preprocess_mode == 'max' else
            self._get_average_buffer
        )

    def _get_max_buffer(self):
        return np.max(self._frame_buffer, axis=0)

    def _get_average_buffer(self):
        return np.mean(self._frame_buffer, axis=0)

    def _init_resize(self):
        orig_width, orig_height = self._ale.getScreenDims()
        h = self.height or orig_height
        w = self.width or orig_width
        if h == orig_height and w == orig_width:
            self.resize = None
        else:
            self.resize = (h, w) if self.grayscale else (h, w, 3)

    ###########################################################################
    def _get_dict(self):
        return {self.__class__.__name__: {
            'rom': self.rom_path,
            'mode': self.mode,
            'resize': self.resize,
            'grayscale': self.grayscale,
            'buffer_frames': self.buffer_frames,
            'preprocess_mode': self.preprocess_mode,
            'random_seed': self._ale.getInt('random_seed'),
            'random_start': self.random_start,
            'n_actions': self.n_actions,
            'repeat_action': self.repeat_action,
            'minmal_action_set': self.minimal_action_set,
            'record_screen_path': self._ale.getString('record_screen_path'),
            'display_screen': self._ale.getBool('display_screen'),
            'sound': self._ale.getBool('sound'),
            'record_sound_filename':
            self._ale.getString('record_sound_filename'),
        }}

    def __repr__(self):
        return str(self._get_dict())

    def __str__(self):
        return pprint_dict(self._get_dict())

    ###########################################################################
    @property
    def n_actions(self):
        return len(self._actions)

    ###########################################################################
    def _get_info(self):
        return {
            'lives': self._ale.lives(),
            'total_frame_number': self._ale.getFrameNumber(),
            'episode_frame_number': self._ale.getEpisodeFrameNumber(),
        }

    def reset(self):
        """Reset game

        In test mode, the game is simply initialized. In train mode, if the
        game is in terminal state due to a life loss but not yet game over,
        then only life loss flag is reset so that the next game starts from
        the current state. Otherwise, the game is simply initialized.
        """
        reward = 0
        if (
                self.mode == 'test' or
                not self.life_lost or  # `reset` called in a middle of episode
                self._ale.game_over()  # all lives are lost
        ):
            self._ale.reset_game()
            rand = self.random_start
            repeat = 1 + (np.random.randint(rand) if rand else 0)
            for _ in range(repeat):
                reward += self._step(0)

        self.life_lost = False
        return Outcome(
            reward=reward,
            state=self._get_state(),
            terminal=self._is_terminal(),
            info=self._get_info(),
        )

    ###########################################################################
    # methods for `step` function
    def step(self, action):
        reward = 0
        action = self._actions[action]

        self.life_lost = False
        initial_lives = self._ale.lives()
        for _ in range(self.repeat_action):
            reward += self._step(action)

            if not self._ale.lives() == initial_lives:
                self.life_lost = True

            terminal = self._is_terminal()
            if terminal:
                break

        return Outcome(
            reward=reward,
            state=self._get_state(),
            terminal=terminal,
            info=self._get_info(),
        )

    def _step(self, action):
        reward = self._ale.act(action)
        buffer_ = self._frame_buffer[self._buffer_index]
        self._get_raw_screen(screen_data=buffer_)
        self._buffer_index = (self._buffer_index + 1) % self.buffer_frames
        return reward

    def _get_state(self):
        screen = self._get_screen()
        if self.resize:
            return imresize(screen, self.resize)
        return screen

    def _is_terminal(self):
        if self.mode == 'train':
            return self._ale.game_over() or self.life_lost
        return self._ale.game_over()
