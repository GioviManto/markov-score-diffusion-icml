#!/usr/bin/env bash
# Submit one registered experiment (or a chained set) to SLURM on the Bocconi HPC.
#
#   hpc/icml_submit.sh e02a                     # capacity, protocol v2
#   hpc/icml_submit.sh e03-screen               # then: e03-confirm --after <jobid>, e03-matched, e03-length
#   hpc/icml_submit.sh e06 | e02b | e08 | e09 | e04 | e05
#   hpc/icml_submit.sh all                      # everything, with the dependencies wired
#
# Env (from hpc/.last_deploy.env, written by icml_deploy.sh) plus optional
#   MSD_HPC_PARTITION  (default: first non-GPU partition sinfo lists)
#   MSD_HPC_CPUS       (default 32)   MSD_HPC_MEM (default 64G)   MSD_HPC_TIME (default 12:00:00)
# Every job: one node, --workers = $SLURM_CPUS_PER_TASK, single-threaded BLAS, outputs under
# $MSD_OUT/<E>/ (outside the source tree), logs under $MSD_HPC_ROOT/logs/. Resumable: a
# resubmitted job skips the cells whose files exist (experiments/common.py: run_cells).
set -euo pipefail
cd "$(dirname "$0")/.."
# MSD_DEPLOY_ENV=hpc/.deploy_<shortsha>.env resubmits into an OLDER deployment (a resumed run
# must land in the output directory its finished cells already live in); default = newest.
DEPLOY_ENV=${MSD_DEPLOY_ENV:-hpc/.last_deploy.env}
[[ -f "$DEPLOY_ENV" ]] || { echo "run hpc/icml_deploy.sh first (no $DEPLOY_ENV)" >&2; exit 2; }
source "$DEPLOY_ENV"
CPUS=${MSD_HPC_CPUS:-32}; MEM=${MSD_HPC_MEM:-64G}; TIME=${MSD_HPC_TIME:-12:00:00}
# Optional: MSD_HPC_WORKERS caps --workers below the cpu count (memory-bound cells; GPU jobs),
# MSD_HPC_GRES adds a --gres line (e.g. gpu:4g.40gb:1), MSD_HPC_VENV picks another venv under
# $MSD_HPC_ROOT (default venv; venv-cuda carries the CUDA torch build for the GPU partitions).
WORKERS_CAP=${MSD_HPC_WORKERS:-}; GRES=${MSD_HPC_GRES:-}; VENV=${MSD_HPC_VENV:-venv}
# MSD_HPC_THREADS sets the BLAS/OMP thread count inside the job. Default 1, which is right
# for the cell-parallel drivers (one thread per worker, no oversubscription). E24 is the
# exception: it measures wall-clock, so it must be told how many threads it is allowed and
# must record that -- a timing at an unstated thread count is not a reproducible number.
THREADS=${MSD_HPC_THREADS:-1}
# MSD_HPC_SERIAL_CPUS: cpus for a driver that declares no --workers (default 4).
# E24 is the case that needs it: it is deliberately serial because it measures
# wall-clock, and it still wants a node with enough cores to be realistic.
SERIAL_CPUS=${MSD_HPC_SERIAL_CPUS:-4}
AFTER=""; WHAT=${1:-}; shift || true
while [[ $# -gt 0 ]]; do case "$1" in --after) AFTER="$2"; shift 2;; *) echo "unknown arg $1" >&2; exit 2;; esac; done
[[ -n "$WHAT" ]] || { echo "usage: $0 <e32|e33-oracles|e33-k-s0..s12|e33-screen|e33-nets-s0|e33-nets-s8|e23|e24c-capacity|e26|e22c|e24c-cpu|e24c-gpu|e25-s0|e25-s4|e25-s8|e25-s12|e24-cpu|e24-gpu|e06c|e07b|e07c|e11b|e12b|e14|e14-seeds|e14b-screen|e14b-confirm|e13x|e02a|e02b|e03-screen|e03-confirm|e03-matched|e03-length|e06|e06b|e08|e09|e04|e05|all> [--after JOBID]" >&2; exit 2; }

partition() {
  if [[ -n "${MSD_HPC_PARTITION:-}" ]]; then echo "$MSD_HPC_PARTITION"; return; fi
  ssh -o BatchMode=yes "$MSD_HPC_HOST" 'sinfo -h -o "%P" | sed "s/\*//" | grep -v -i gpu | head -1'
}
PART=$(partition); [[ -n "$PART" ]] || { echo "no partition found; set MSD_HPC_PARTITION" >&2; exit 2; }
# SLURM does not expand $HOME inside #SBATCH directives: resolve the remote home once and use absolute paths
RHOME=$(ssh -o BatchMode=yes "$MSD_HPC_HOST" 'echo $HOME'); [[ -n "$RHOME" ]] || { echo "cannot resolve remote HOME" >&2; exit 2; }

# name -> "script args" (relative to experiments/); {OUT} is replaced by the remote outputs dir
cmd_for() {
  case "$1" in
    e02a)        echo "e02_capacity.py --only capacity --output-dir {OUT}/E02a";;
    e02b)        echo "e02_capacity.py --only family --output-dir {OUT}/E02b";;
    e03-screen)  echo "e03_baselines.py --only screen --output-dir {OUT}/E03";;
    e03-confirm) echo "e03_baselines.py --only confirm --set winners=\"'auto:{OUT}/E03/screening.csv'\" --output-dir {OUT}/E03";;
    e03-matched) echo "e03_baselines.py --only matched --set winners=\"'auto:{OUT}/E03/screening.csv'\" --output-dir {OUT}/E03";;
    e03-length)  echo "e03_baselines.py --only length --set winners=\"'auto:{OUT}/E03/screening.csv'\" --output-dir {OUT}/E03";;
    e06)         echo "e06_learning_through_noise.py --output-dir {OUT}/E06";;
    e06b)        echo "e06_learning_through_noise.py --set rho=0.5 --set n_chains=2048 --set levels=\"(0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.4)\" --output-dir {OUT}/E06b";;
    e08)         echo "e08_misspecification.py --output-dir {OUT}/E08";;
    e09)         echo "e09_capacity_floor.py --output-dir {OUT}/E09";;
    e11)         echo "e11_grid_validation.py --output-dir {OUT}/E11";;
    e13)         echo "e13_regime_sweep.py --output-dir {OUT}/E13";;
    e12)         echo "e12_sample_complexity.py --output-dir {OUT}/E12";;
    # --- budget and quadrature controls, and the regime robustness of the headline (18 Sep) ---
    e06c)        echo "e06_learning_through_noise.py --set em_checkpoints=\"(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200, 1600, 2200, 3000, 4000, 6000, 8000)\" --output-dir {OUT}/E06c";;
    e07b)        echo "e07_phase_diagram.py --set grid_size=801 --set half_width=12.0 --output-dir {OUT}/E07b";;
    e07c)        echo "e07_phase_diagram.py --set grid_size=1601 --set half_width=16.0 --output-dir {OUT}/E07c";;
    e11b)        echo "e11_grid_validation.py --set lengths=\"(64, 128)\" --output-dir {OUT}/E11b";;
    e12b)        echo "e12_sample_complexity.py --set levels=\"(0.6, 0.7, 0.8, 0.9, 1.0)\" --set seeds=\"(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)\" --set em_checkpoints=\"(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200, 1600, 2200, 3000, 4000, 6000, 8000)\" --output-dir {OUT}/E12b";;
    e14-screen)  echo "e14_regime_baselines.py --only screen --output-dir {OUT}/E14";;
    e14-confirm) echo "e14_regime_baselines.py --only confirm --set winners=\"'auto:{OUT}/E14/screening.csv'\" --output-dir {OUT}/E14";;
    # --- 21 Sep: the protocol's sixteen seeds for the regime comparisons, and the bimodal member ---
    e14-seeds)   echo "e14_regime_baselines.py --only confirm --set seeds=\"(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)\" --set winners=\"'auto:{OUT}/E14/screening.csv'\" --output-dir {OUT}/E14";;
    e14b-screen) echo "e14_regime_baselines.py --only screen --set family=\"'bimodal'\" --set rhos=\"(0.85,)\" --set lengths=\"(32,)\" --output-dir {OUT}/E14b";;
    e14b-confirm) echo "e14_regime_baselines.py --only confirm --set family=\"'bimodal'\" --set rhos=\"(0.85,)\" --set lengths=\"(32,)\" --set seeds=\"(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)\" --set winners=\"'auto:{OUT}/E14b/screening.csv'\" --output-dir {OUT}/E14b";;
    e13x)        echo "e13_regime_sweep.py --set seeds=\"(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)\" --output-dir {OUT}/E13x";;
    e10)         echo "e10_capacity_evidence.py --set em_source=\"'{OUT}/E02a/capacity.csv'\" --output-dir {OUT}/E10";;
    e04)         echo "e04_refinement.py --set em_source=\"'{OUT}/E02a/capacity.csv'\" --output-dir {OUT}/E04";;
    e05)         echo "e05_generation.py --set em_source=\"'{OUT}/E02a/capacity.csv'\" --set net_dir=\"'{OUT}/E03/ckpts'\" --output-dir {OUT}/E05";;
    # --- 25 Sep, after the Jerome call: inference cost and scaling (docs/JEROME_CALL_2026-09-25.md Q7/Q11) ---
    # Cost is the whole measurement here, so both arms must get the same hardware. The CPU job
    # asks for a whole node's threads; the GPU job runs the SAME driver with devices=("cpu","gpu")
    # so the cross-device ratio comes from one process on one machine rather than two runs.
    e24-cpu)     echo "e24_inference_cost.py --set devices=\"('cpu',)\" --output-dir {OUT}/E24";;
    e24-gpu)     echo "e24_inference_cost.py --set devices=\"('cpu', 'gpu')\" --output-dir {OUT}/E24gpu";;
    # ================= confirmatory set of 25 Sep, after the Jerome call ==================
    # docs/HPC_CONFIRMATORY_SET.md defines these. Each widens a laptop result across every
    # factor that could plausibly explain it away, so that a conclusion is a property of the
    # model and not of one configuration.
    #
    # E22c: message shapes, everything the laptop run held fixed now swept -- correlation
    # (3), chain length (2), grid resolution (2, which is what separates model error from
    # quadrature WITHOUT appealing to E11), four innovation families, twelve levels, 512
    # chains. 576 cells, resumable.
    e22c)        echo "e22_message_shapes.py --set 'rhos=(0.5, 0.85, 0.95)' --set 'lengths=(32, 128)' --set 'grids=((801, 12.0), (1601, 16.0))' --set n_chains=512 --set 'sites=(1, 2, 4, 8, 16, 31, 63, 127)' --output-dir {OUT}/E22c";;
    # E24c: the cost numbers on a quiet node with a declared thread count. The laptop run was
    # explicitly EXPLORATORY for exactly this reason.
    e24c-cpu)    echo "e24_inference_cost.py --set \"devices=('cpu',)\" --set n_repeat=15 --set scale_repeat=9 --set 'train_sizes=(32, 128, 512, 2048, 8192)' --output-dir {OUT}/E24c";;
    # E24c-gpu: the SAME driver with both devices in one process, so the cross-device ratio
    # comes from one machine and one run rather than from two jobs compared after the fact.
    # Submit with MSD_JOB_ENV="export E24_REQUIRE_GPU=1" so the job FAILS rather than
    # silently producing a CPU-only cost table (see the 25 Sep incident in E24.json).
    e24c-gpu)    echo "e24_inference_cost.py --set \"devices=('cpu', 'gpu')\" --set n_repeat=15 --set scale_repeat=9 --output-dir {OUT}/E24c_gpu";;
    # E25: the MDN-kernel rung of the information ladder -- Jerome's "small neural network
    # that learns a transition kernel". exp_34 has existed unrun since before the call. It is
    # serial with its own CSV resume, so the seeds are split across four jobs and merged.
    e25-s0)      echo "exp_34_information_ladder.py --set seed0=0  --set seeds=4 --output-dir {OUT}/E25_s0";;
    e25-s4)      echo "exp_34_information_ladder.py --set seed0=4  --set seeds=4 --output-dir {OUT}/E25_s4";;
    e25-s8)      echo "exp_34_information_ladder.py --set seed0=8  --set seeds=4 --output-dir {OUT}/E25_s8";;
    e25-s12)     echo "exp_34_information_ladder.py --set seed0=12 --set seeds=4 --output-dir {OUT}/E25_s12";;
    # E23: how many numbers does a BP message need? A MEASUREMENT on the true messages,
    # deliberately prior to building any Fourier solver -- E24 showed an FFT caps at 1.3x
    # and E22 showed |kappa_6| is never small, so the premise had to be tested first.
    e23)         echo "e23_representation.py --output-dir {OUT}/E23";;
    # E24 capacity: the other half of the "small kernel vs huge score network" claim --
    # does the M-step stay flat in dataset size as the MODEL grows?
    e24c-capacity) echo "e24_inference_cost.py --only capacity --set 'cap_sizes=(128, 512, 2048, 8192)' --set train_repeat=7 --output-dir {OUT}/E24c";;
    # E26: generation without a score (Q8). Reads the fitted mixture kernels from the E02a
    # run of the 15 Sep deployment -- an absolute path, because that output lives under a
    # DIFFERENT deployment tag and {OUT} points at this one.
    e26)         echo "e05_generation.py --set em_source=\"'\$HOME/msd-icml/outputs/ea0435a-20260915T104955Z/E02a/capacity.csv'\" --output-dir {OUT}/E26";;
    # E27: Q8's STRONG form -- generation with no score and no reverse SDE. Same E02a
    # kernels as E26 (absolute path, that output lives under a different deployment tag),
    # so every contrast between the two is paired. `sample` and `steps` fan out over cells
    # and want many cpus; `cost` is a TIMING measurement and is a separate job so it does
    # not share a node with a worker pool -- E24's laptop run was downgraded to exploratory
    # for exactly that reason (the machine was not quiet).
    e27)         echo "e27_direct_sampler.py --only sample,steps --set em_source=\"'\$HOME/msd-icml/outputs/ea0435a-20260915T104955Z/E02a/capacity.csv'\" --output-dir {OUT}/E27";;
    e27-cost)    echo "e27_direct_sampler.py --only cost --output-dir {OUT}/E27";;
    # E28: Q9's second half -- does the ranking hold at every noise level? No training: the
    # E03 confirm checkpoints live under the 15 Sep deployment tag, hence the absolute path.
    e28)         echo "e28_per_level_ranking.py --set ckpt_dir=\"'\$HOME/msd-icml/outputs/ea0435a-20260915T104955Z/E03/ckpts'\" --set em_source=\"'\$HOME/msd-icml/outputs/ea0435a-20260915T104955Z/E02a/capacity.csv'\" --set e03_confirm=\"'\$HOME/msd-icml/outputs/ea0435a-20260915T104955Z/E03/confirm.csv'\" --set winners=\"'\$HOME/msd-icml/outputs/ea0435a-20260915T104955Z/E03/winners.json'\" --output-dir {OUT}/E28";;
    # E29: score nets AT the kernel's parameter count -- the arm that can refute the
    # parameter-efficiency claim. Sharded by architecture: six independent ladders.
    e31-window) echo "e29_matched_parameters.py --set 'archs=(\"window\",)' --set 'sizes=(2048,)' --set net_steps=131072 --set 'net_checkpoints=(8000, 16000, 32000, 65536, 131072)' --output-dir {OUT}/E31_window";;
    e31-mlp)    echo "e29_matched_parameters.py --set 'archs=(\"mlp\",)' --set 'sizes=(2048,)' --set net_steps=131072 --set 'net_checkpoints=(8000, 16000, 32000, 65536, 131072)' --output-dir {OUT}/E31_mlp";;
    e31-bigru)  echo "e29_matched_parameters.py --set 'archs=(\"bigru\",)' --set 'sizes=(2048,)' --set net_steps=131072 --set 'net_checkpoints=(8000, 16000, 32000, 65536, 131072)' --output-dir {OUT}/E31_bigru";;
    e31-tf)     echo "e29_matched_parameters.py --set 'archs=(\"transformer\",)' --set 'sizes=(2048,)' --set net_steps=131072 --set 'net_checkpoints=(8000, 16000, 32000, 65536, 131072)' --output-dir {OUT}/E31_tf";;
    e31-tcn)    echo "e29_matched_parameters.py --set 'archs=(\"tcn\",)' --set 'sizes=(2048,)' --set net_steps=131072 --set 'net_checkpoints=(8000, 16000, 32000, 65536, 131072)' --output-dir {OUT}/E31_tcn";;
    e31-unet)   echo "e29_matched_parameters.py --set 'archs=(\"unet\",)' --set 'sizes=(2048,)' --set net_steps=131072 --set 'net_checkpoints=(8000, 16000, 32000, 65536, 131072)' --output-dir {OUT}/E31_unet";;
    # UNCONTENDED reruns (N-29). Every wall-clock case below MUST be submitted with
    # MSD_HPC_EXCLUSIVE=1. The cited E24 and E27 timings were taken on shared nodes -- e27
    # and e27-cost overlapped on cnode02 for three hours of a 4h19m timing job, both mine --
    # and contention inflates the BP-vs-SDE ratio because 400 sequential network evaluations
    # suffer far more than a handful of BLAS calls. It flatters us, so it must be redone.
    e24x-cpu)    echo "e24_inference_cost.py --set \"devices=('cpu',)\" --set n_repeat=15 --set scale_repeat=9 --set 'train_sizes=(32, 128, 512, 2048, 8192)' --output-dir {OUT}/E24x";;
    e24x-gpu)    echo "e24_inference_cost.py --set \"devices=('cpu', 'gpu')\" --set n_repeat=15 --set scale_repeat=9 --output-dir {OUT}/E24x_gpu";;
    e27x-cost)   echo "e27_direct_sampler.py --only cost --output-dir {OUT}/E27x";;
    # E30 at 3x the EM budget, at the two sizes where at_cap actually bound (mdn 3/16 at
    # M=512 and 14/16 at M=2048). M=32 and M=128 are already budget-free (0/16, 2/16) and
    # are reported as such, so rerunning them would triple the work to change nothing.
    # Longer budget helps the LOSING arm, which is the conservative direction for E30's
    # negative result about the neural kernel.
    e30m-s0)  echo "e30_joint_ladder.py --only ladder --set 'seeds=(0, 1, 2, 3,)' --set 'sizes=(512, 2048)' --set 'em_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200, 1800, 2400, 3600)' --set 'mdn_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200, 1800, 2400, 3600)' --output-dir {OUT}/E30m_s0";;
    e30m-s4)  echo "e30_joint_ladder.py --only ladder --set 'seeds=(4, 5, 6, 7,)' --set 'sizes=(512, 2048)' --set 'em_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200, 1800, 2400, 3600)' --set 'mdn_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200, 1800, 2400, 3600)' --output-dir {OUT}/E30m_s4";;
    e30m-s8)  echo "e30_joint_ladder.py --only ladder --set 'seeds=(8, 9, 10, 11,)' --set 'sizes=(512, 2048)' --set 'em_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200, 1800, 2400, 3600)' --set 'mdn_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200, 1800, 2400, 3600)' --output-dir {OUT}/E30m_s8";;
    e30m-s12) echo "e30_joint_ladder.py --only ladder --set 'seeds=(12, 13, 14, 15,)' --set 'sizes=(512, 2048)' --set 'em_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200, 1800, 2400, 3600)' --set 'mdn_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200, 1800, 2400, 3600)' --output-dir {OUT}/E30m_s12";;
    e30-s0)     echo "e30_joint_ladder.py --set 'seeds=(0, 1, 2, 3,)' --output-dir {OUT}/E30_s0";;
    e30-s4)     echo "e30_joint_ladder.py --set 'seeds=(4, 5, 6, 7,)' --output-dir {OUT}/E30_s4";;
    e30-s8)     echo "e30_joint_ladder.py --set 'seeds=(8, 9, 10, 11,)' --output-dir {OUT}/E30_s8";;
    e30-s12)     echo "e30_joint_ladder.py --set 'seeds=(12, 13, 14, 15,)' --output-dir {OUT}/E30_s12";;
    e29-window)  echo "e29_matched_parameters.py --set 'archs=(\"window\",)'      --output-dir {OUT}/E29_window";;
    e29-mlp)     echo "e29_matched_parameters.py --set 'archs=(\"mlp\",)'         --output-dir {OUT}/E29_mlp";;
    e29-bigru)   echo "e29_matched_parameters.py --set 'archs=(\"bigru\",)'       --output-dir {OUT}/E29_bigru";;
    e29-tf)      echo "e29_matched_parameters.py --set 'archs=(\"transformer\",)' --output-dir {OUT}/E29_tf";;
    e29-tcn)     echo "e29_matched_parameters.py --set 'archs=(\"tcn\",)'         --output-dir {OUT}/E29_tcn";;
    e29-unet)    echo "e29_matched_parameters.py --set 'archs=(\"unet\",)'        --output-dir {OUT}/E29_unet";;
    # E25 at MATCHED EM budget: the current run gives em_bp up to 1200 iterations and the MDN
    # 400. The asymmetry favours the baseline, so results are conservative -- but "em_bp had 3x
    # the compute" is a one-line referee objection and removing it costs one rerun.
    e25m-s0)     echo "exp_34_information_ladder.py --set seed0=0  --set seeds=4 --set 'mdn_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200)' --output-dir {OUT}/E25m_s0";;
    e25m-s4)     echo "exp_34_information_ladder.py --set seed0=4  --set seeds=4 --set 'mdn_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200)' --output-dir {OUT}/E25m_s4";;
    e25m-s8)     echo "exp_34_information_ladder.py --set seed0=8  --set seeds=4 --set 'mdn_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200)' --output-dir {OUT}/E25m_s8";;
    e25m-s12)    echo "exp_34_information_ladder.py --set seed0=12 --set seeds=4 --set 'mdn_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200)' --output-dir {OUT}/E25m_s12";;
    # E27 cost at LARGE batch, where E24 puts BP's crossover with the transformer (~12,600):
    # states the one-sweep speed-up where BP is launch-bound and cheapest, not just at B=128.
    e27-cost-big) echo "e27_direct_sampler.py --only cost --set 'cost_batch=(2048, 8192, 32768)' --set 'steps_ladder=(100, 400)' --output-dir {OUT}/E27big";;
    # ================= 1 Oct: the phi^4 field, the second laboratory (Jerome and Marc) ==================
    # E32 is deterministic and also runs on the laptop; the cluster copy is the registered one for
    # the refinement part at (1601, 8) on the L = 128 cells. E33 is sharded by seed into its own
    # directories (N-27): four kernel jobs on the CPU partition, the network screen and two
    # network jobs on the A100 partition (MSD_HPC_GRES, MSD_HPC_VENV=venv-cuda), chained --after
    # the screen. The oracles job is minutes.
    e32)         echo "e32_field_oracles.py --output-dir {OUT}/E32";;
    e33-oracles) echo "e33_field_learning.py --only oracles --output-dir {OUT}/E33o";;
    e33-k-s0)    echo "e33_field_learning.py --only kernels --set 'seeds=(0, 1, 2, 3)' --output-dir {OUT}/E33k_s0";;
    e33-k-s4)    echo "e33_field_learning.py --only kernels --set 'seeds=(4, 5, 6, 7)' --output-dir {OUT}/E33k_s4";;
    e33-k-s8)    echo "e33_field_learning.py --only kernels --set 'seeds=(8, 9, 10, 11)' --output-dir {OUT}/E33k_s8";;
    e33-k-s12)   echo "e33_field_learning.py --only kernels --set 'seeds=(12, 13, 14, 15)' --output-dir {OUT}/E33k_s12";;
    e33-screen)  echo "e33_field_learning.py --only screen --output-dir {OUT}/E33n";;
    e33-nets-s0) echo "e33_field_learning.py --only nets --set 'seeds=(0, 1, 2, 3, 4, 5, 6, 7)' --set winners=\"'auto:{OUT}/E33n/screening.csv'\" --output-dir {OUT}/E33n_s0";;
    e33-nets-s8) echo "e33_field_learning.py --only nets --set 'seeds=(8, 9, 10, 11, 12, 13, 14, 15)' --set winners=\"'auto:{OUT}/E33n/screening.csv'\" --output-dir {OUT}/E33n_s8";;
    *) echo "unknown experiment $1" >&2; exit 2;;
  esac
}
submit_one() {
  local name=$1 dep=${2:-}
  local cmd; cmd=$(cmd_for "$name"); cmd=${cmd//\{OUT\}/\$HOME/$MSD_OUT}
  local depflag=""; [[ -n "$dep" ]] && depflag="--dependency=afterok:$dep"     # dep may be "id1:id2"
  # --workers only for drivers whose source declares it: the serial oracle ladders (e01, e07)
  # do not, and passing it to them failed E07b at launch on 18 Sep (argparse: unrecognized
  # arguments). Decided here, at submission, from the driver's own source.
  local driver=${cmd%% *} workers_flag='' cpus=$CPUS
  if grep -q -- '"--workers"' "experiments/$driver"; then workers_flag="--workers ${WORKERS_CAP:-\$SLURM_CPUS_PER_TASK}"; else cpus=${MSD_HPC_SERIAL_CPUS:-4}; fi
  local gres_line=''; [[ -n "$GRES" ]] && gres_line="#SBATCH --gres=$GRES"
  # MSD_HPC_EXCLUSIVE=1 demands the whole node. REQUIRED for any wall-clock experiment:
  # on 28-29 Sep e27-cost-big shared cnode07 with ten 16-cpu fitting jobs and measured the
  # 400-step reverse SDE at 431 ms/seq against 200 ms/seq on a quieter run of the SAME
  # cell, which inflated the headline ratio from 1138x to 2392x. Contention hits 400
  # sequential network evaluations far harder than a handful of BLAS calls, so it does not
  # cancel in the ratio -- it flatters us. See NEGATIVE_RESULTS N-29.
  local excl_line=''; [[ -n "${MSD_HPC_EXCLUSIVE:-}" ]] && excl_line='#SBATCH --exclusive'
  local gpu_env_line=''; [[ -n "$GRES" ]] && gpu_env_line="source \$HOME/$MSD_SRC/hpc/icml_gpu_env.sh"
  local jid
  jid=$(ssh -o BatchMode=yes "$MSD_HPC_HOST" "sbatch --parsable $depflag" <<SB
#!/bin/bash
#SBATCH --job-name=msd-$name
#SBATCH --partition=$PART
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=$cpus --mem=$MEM --time=$TIME
#SBATCH --propagate=NONE
$excl_line
$gres_line
#SBATCH --output=$RHOME/$MSD_HPC_ROOT/logs/%x_%j.out --error=$RHOME/$MSD_HPC_ROOT/logs/%x_%j.err
set -euo pipefail
# The login shell carries RLIMIT_CPU = 600 s. Without --propagate=NONE SLURM hands that
# limit to every process in the job, which kills a multi-worker run partway through with
# 'CPU time limit exceeded' (E05, 15 Sep) or leaves the job holding an idle node (E06b).
ulimit -t unlimited 2>/dev/null || true
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS
# A GPU job needs the CUDA 12.4 pin and libnvrtc on the library path, or cupy's device probe
# fails -- selectively, depending on whether torch was imported first, which is how job 669334
# produced a CPU-only cost table from a working H200. Sourced here so it cannot be forgotten.
$gpu_env_line
# MSD_JOB_ENV: extra exports baked into the job body at submission time (e.g.
# "export E24_REQUIRE_GPU=1" to make a GPU job fail closed).
${MSD_JOB_ENV:-}
export PYTHONPATH=\$HOME/$MSD_SRC/src
cd \$HOME/$MSD_SRC/experiments
echo "start=\$(date -Is) host=\$(hostname) job=\$SLURM_JOB_ID cpus=\$SLURM_CPUS_PER_TASK"; cat ../REVISION
\$HOME/$MSD_HPC_ROOT/$VENV/bin/python $cmd $workers_flag
echo "done=\$(date -Is)"
SB
)
  echo "$name -> job $jid (partition $PART, $cpus cpus, dep=${dep:-none})"; LAST_JOB=$jid
}
case "$WHAT" in
  all)
    submit_one e02a "$AFTER"; J_E02A=$LAST_JOB
    submit_one e09 "$AFTER"
    submit_one e03-screen "$AFTER"; J_SCREEN=$LAST_JOB
    submit_one e03-confirm "$J_SCREEN"; J_CONF=$LAST_JOB
    submit_one e03-matched "$J_SCREEN"
    submit_one e03-length "$J_CONF"
    submit_one e06 "$AFTER"; submit_one e02b "$AFTER"; submit_one e08 "$AFTER"
    submit_one e04 "$J_E02A"
    submit_one e05 "$J_E02A:$J_CONF";;
  *) submit_one "$WHAT" "$AFTER";;
esac
