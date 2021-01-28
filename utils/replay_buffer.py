import numpy as np
import random
from collections import namedtuple

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, obs_t, action, reward, obs_tp1, done, sigs_t=[], sigs_tp1=[], target_action=[]):
        data = (obs_t, sigs_t, action, reward, obs_tp1, sigs_tp1, done, target_action)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, signals_t, actions, rewards, obses_tp1, signals_tp1, dones, target_actions = \
            [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, sigs, action, reward, obs_tp1, sigs_tp1, done, target_action = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            if len(target_action):
                signals_t.append(sigs)
                signals_tp1.append(sigs_tp1)
                target_actions.append(np.array(target_action, copy=False))
        if len(target_actions):
            return np.array(obses_t), signals_t, np.array(actions), np.array(rewards), \
                   np.array(obses_tp1), signals_tp1, np.array(dones), np.array(target_actions)
        else:
            return np.array(obses_t), np.array(actions), np.array(rewards), \
                   np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)


class ReplayBuffer2:
    def __init__(self, fields, max_size):
        self.max_size = max_size
        self.fields = fields
        _storage = namedtuple('storage', fields)
        self.storage = _storage(*([] for _ in range(len(fields))))
        self.next_idx = 0
        self.cur_size = 0

    def get_frame_as_dict(self, t):
        assert t < self.cur_size, \
            print("[Error]\tidx %i is larger than current size (%i)" % (t, self.cur_size))
        res = {}
        for field in self.fields:
            data = getattr(self.storage, field)
            if len(data) > t:
                res[field] = data[t]
            else:
                res[field] = None
        return res


def sample(buffer: ReplayBuffer2, bs: int):
    idxes = np.random.choice(buffer.cur_size, bs, replace=False)
    replay_sample = ReplayBuffer2(buffer.fields, bs)

    for idx in idxes:
        add_frame(
            replay_sample,
            buffer.get_frame_as_dict(idx)
        )
    return replay_sample


def add_frame(buffer: ReplayBuffer2, d: dict):  # add one frame at a time
    for k in d.keys():
        v = d[k]
        t = getattr(buffer.storage, k)
        if buffer.cur_size > buffer.next_idx:
            t[buffer.next_idx] = v
        else:
            t.append(v)
    buffer.cur_size = min(buffer.cur_size + 1, buffer.max_size)
    buffer.next_idx = (buffer.next_idx + 1) % buffer.max_size
