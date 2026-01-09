# Dependencies

Core
- torch>=2.1.0

Optional (examples)
- gymnasium (for CartPole and general Gym examples)
- gymnasium-minigrid (MiniGrid tasks; requires pygame/SDL2 on many systems)
- gymnasium-robotics (FetchReach; requires Mujoco tooling)

Dev/Test (optional)
- pytest

Install
```bash
# Core only
pip install -r requirements.txt

# Optional extras (install only what you need)
pip install -r requirements-extras.txt   # may require system deps for minigrid/robotics

# Or install Gym wheels only (skip minigrid/robotics):
make install-extras-wheels

# For a GPU-enabled PyTorch, use the installer from https://pytorch.org/get-started/locally/
```

Notes
- Examples that depend on Gym families are optional; the core library (encoders, memory, consolidator, replay, trainer) only requires PyTorch.
- Some environments require system packages (e.g., Mujoco/robotics) which are outside the scope of this library.

Faster, smaller installs
- CPU-only quickstart (recommended for notebooks):
  - `pip install -r requirements-cpu.txt` (pins torch==2.4.1 on the PyTorch CPU index)
- CUDA 12.1 build:
  - `pip install -r requirements-cu121.txt`

Connection/timeouts
- If network is slow, increase pip timeouts/retries:
  - `pip install --default-timeout 1000 --retries 5 -r requirements.txt`
- You can also bypass PyPI and use the PyTorch CDN (as above) to reduce timeouts.

MiniGrid/pygame tips
- If `pygame` build fails with `sdl2-config: not found`, install SDL2 dev libs (e.g., `sudo apt-get install -y libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev`) or skip `gymnasium-minigrid`.
