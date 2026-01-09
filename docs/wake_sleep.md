# Wake/Sleep Orchestration

`WakeSleepTrainer` alternates agent updates (wake) with consolidation (sleep) using an attached `Consolidator`.

Minimal loop
```python
from hippotorch import WakeSleepTrainer, TerminalSegmenter

trainer = WakeSleepTrainer(
    agent=agent,
    buffer=buffer,             # HippocampalReplayBuffer with attached Consolidator
    segmenter=TerminalSegmenter(),
    consolidation_schedule="periodic",    # or "interleaved", "adaptive"
    consolidation_interval=1000,
    consolidation_steps=100,
    min_buffer_episodes=10,
    batch_size=256,
)

metrics = trainer.train(env, total_steps=100_000)
```

Schedules
- periodic: sleep every N steps.
- interleaved: short, frequent sleep steps.
- adaptive: triggers sleep when `reward_alignment` degrades.

Tips
- Start with uniform-heavy sampling and ramp semantic via `linear_mixture_schedule`.
- Enable `report_quality=True` in `buffer.consolidate()` to log quality metrics.
- Provide a `query_state_fn` on the trainer to build a semantic query vector matching the encoder input (e.g., `[obs, 0, 0]` for `[obs, action, reward]`).
