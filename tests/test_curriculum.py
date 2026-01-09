from hippotorch.curriculum.callbacks import HippotorchCurriculumCallback


class _MockEnv:
    def __init__(self, length: int = 5, max_length: int = 30):
        self.length = int(length)
        self._max_length = int(max_length)

    def set_corridor_length(self, L: int) -> None:
        self.length = min(int(L), self._max_length)


def test_curriculum_callback_increases_length_on_success_threshold():
    env = _MockEnv(length=5, max_length=15)
    cb = HippotorchCurriculumCallback(
        min_length=5, max_length=15, step=5, window=4, threshold=0.75
    )

    # Four successful episodes → win_rate = 1.0 (>= 0.75), should step to 10
    for _ in range(4):
        cb.on_episode_end(env, reward=1.0)
    assert env.length == 10

    # Mixed successes below threshold → no change
    for r in [1.0, 0.0, 1.0, 0.0]:
        cb.on_episode_end(env, reward=r)
    assert env.length == 10

    # Back to successes to trigger next step to max_length (15)
    for _ in range(4):
        cb.on_episode_end(env, reward=1.0)
    assert env.length == 15

    # Further successes should not exceed max_length
    for _ in range(4):
        cb.on_episode_end(env, reward=1.0)
    assert env.length == 15
