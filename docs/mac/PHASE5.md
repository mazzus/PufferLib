# Apple Silicon port: Phase 5 (environment coverage)

Every env in `ocean/` built with the Mac trainer (`./build.sh ENV`, Metal
backend), then smoke-trained for 45 s (`tests/smoke_train_all.sh`): PASS
means it finished at least one train epoch without crashing. SPS is the
dashboard figure early in training, an indication only (see PHASE4.md on
noise).

**70 of 85 envs train on the Mac.**

| env | smoke SPS |
|---|---|
| admiral | 1.6M |
| affine_lock | 169.4K |
| ants | 1.9M |
| backgammon | 1.4M |
| bat | 527.8K |
| battle | 2.8M |
| benchmark | 2.0M |
| blastar | 2.4M |
| boids | 1.8M |
| boss_fight | 1.2M |
| boxoban | 46.0K |
| breakout | 4.7M |
| cartpole | 8.8M |
| chain_mdp | 2.1M |
| chain_reaction | 788.2K |
| checkers | 1.3M |
| chess | 678.9K |
| click | 2.2M |
| clifford | 947.7K |
| connect4 | 567.0K |
| convert | 9.8M |
| convert_circle | 17.1M |
| craftax_classic | 1.3M |
| dino | 455.2K |
| docking | 6.4M |
| double_pendulum | 3.3M |
| drmario | 631.0K |
| drone | 1.4M |
| enduro | 759.1K |
| flappy | 5.0M |
| freeway | 1.9M |
| g2048 | 35.7K |
| go | 107.4K |
| guerrillacheckers | 1.2M |
| hex | 1.1M |
| impulse_wars | 136.8K |
| laser_puzzle | 1.9M |
| lightsout | 1.4M |
| maze | 62.9K |
| mazing_contest | 1.8M |
| memory | 2.2M |
| moba | 285.0K |
| onestateworld | 976.6K |
| onlyfish | 1.6M |
| overcooked | 2.0M |
| pacman | 264.2K |
| pathfinder | 799.3K |
| pong | 1.3M |
| robocode | 62.2K |
| rware | 1.9M |
| scape | 993.0K |
| shared_pool | 599.6K |
| slimevolley | 52.7K |
| snake | 44.7K |
| soccer | 358.0K |
| space_invaders | 2.4M |
| squared | 5.3M |
| squared_continuous | 5.7M |
| target | 2.0M |
| terraform | 93.6K |
| tetris | 154.5K |
| tmaze | 1.8M |
| tower_climb | 696.5K |
| tower_defence | 39.7K |
| trash_pickup | 2.0M |
| tripletriad | 378.3K |
| tron | 674.7K |
| wef | 19.8K |
| whackamole | 6.4M |
| whisker_racer | 1.2M |

## Builds but does not run as configured

| env | reason |
|---|---|
| drive | needs the WOMD map dataset (`ocean/drive/dataset.py`); not a port issue |
| tactical | default config needs a 186 GB rollout arena (obs 121x121x4 per agent-step); too large for any single GPU, including on CUDA |
| template | env template; trains (38 epochs), then exits on a missing `env/perf` log key |

## Not supported on the Mac trainer

| env | reason |
|---|---|
| asteroids, minimal, craftax, nmmo3, osrs_colosseum, osrs_inferno, osrs_pvp, osrs_zulrah, nethack | custom CUDA policy nets (`ocean/*/*.cu`, via `src/ocean.cu`) use `__syncthreads`, warp shuffles and atomics; they need hand-written host and MSL ports (not done) |
| robot_arm | CUDA-only physics |
| matsci | needs LAMMPS (`lammps/library.h`) installed |
| osrs | shared OSRS library, not an env |
| osrs_* (data) | asset download currently returns 404 |
| nethack | also: vendored fast-nle does not compile with Apple clang |

## Fixes made for coverage

- `build.sh`: C extra sources (impulse_wars) compile as C11 objects.
- `PUF_VLA_PTR` (pufferenv.h) for pointer-to-VLA casts in moba.h and nmmo3.h.
- MinGRU row-scan kernels (cooperative) are never chosen with host kernels;
  long-horizon configs (maze, mazing_contest, terraform) aborted before this fix.
- Arena allocation failures report the requested size.
