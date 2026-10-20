"""Central configuration. Every physical / simulation constant lives here."""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DATA_DIR = RESULTS_DIR / "data"
FIGURES_DIR = RESULTS_DIR / "figures"

# ------------------------------------------------- boost converter (nominal)
VIN = 12.0            # input voltage [V]
VREF = 24.0           # reference output voltage [V]
L = 1e-3              # inductance [H]
C = 470e-6            # output capacitance [F]
R = 20.0              # load resistance [ohm]
F_SW = 20e3           # switching frequency [Hz]

# ---------------------------------------------------------- simulation
# One control/simulation step = one switching period (50 us).  The averaged
# model is only valid for dynamics well below f_sw; the LC resonance is
# ~230 Hz, so 20 kHz sampling is ~85 samples per resonance period.
T_SW = 1.0 / F_SW
DT = T_SW             # control / integration step [s]

# ------------------------------------------------------- safety limits
DUTY_MIN = 0.05
DUTY_MAX = 0.95

# Initial state at reset: inductor empty, capacitor pre-charged to Vin
# through the diode (what really happens when a boost is connected to a source).
IL0 = 0.0
VOUT0 = VIN

# Open-loop validation (Stage 1)
OPEN_LOOP_DUTIES = (0.3, 0.5, 0.6)
OPEN_LOOP_T_END = 0.2   # [s]

# ---------------------------------------------- conduction-mode handling
# "auto": CCM <-> DCM switching (diode blocks reverse current)  [physical]
# "ccm_only": plain averaged CCM equations (bidirectional, kept only to
#             demonstrate the Stage-1 defect)
CONDUCTION_MODE = "auto"
SUBSTEPS = 5          # sub-steps per control step (mode-switch resolution)

# ------------------------------------------------------------------ PID
# Gains are filled in by experiments/tune_pid.py (see docs in that file).
PID_KP = 0.0049154
PID_KI = 6.6994
PID_KD = 2.7364e-5
PID_BIAS = 0.0            # textbook PID: u = Kp e + Ki int(e) + Kd de/dt (no feed-forward)
PID_DERIV_TAU = 1e-4      # derivative low-pass time constant [s]
PID_AW_GAIN = 500.0       # back-calculation rate 1/Tt [1/s]

# ------------------------------------------------- Stage-2 test scenarios
# Every run starts from Vout = 0, IL = 0 (cold start), then the disturbance
# is applied at T_DIST.  (kind, value, label)
SCENARIO_T_DIST = 0.15
SCENARIO_T_END = 0.30
PID_SCENARIOS = {
    "nominal":      (None,   None, "Nominal: Vin=12 V, R=20 ohm"),
    "vin_12_to_10": ("vin",  10.0, "Vin 12 -> 10 V"),
    "vin_12_to_14": ("vin",  14.0, "Vin 12 -> 14 V"),
    "load_20_to_10": ("load", 10.0, "Load 20 -> 10 ohm"),
    "load_20_to_30": ("load", 30.0, "Load 20 -> 30 ohm"),
}

# ------------------------------------------------------------------ SMC
# Cascade structure: sliding-mode inner CURRENT surface, integral voltage loop
# generating the current reference (see controllers/smc.py for the derivation).
SMC_KV = 0.65641            # current reference per volt of error    [A/V]
SMC_KI = 164.74            # current reference per volt-second      [A/(V s)]
SMC_GAIN = 0.30           # switching gain K (duty units)
SMC_BOUNDARY = 0.8        # boundary layer phi on s = IL - Iref   [A]
SMC_I_MAX = 10.0          # reference-current limit (inrush protection) [A]
SMC_AW_GAIN = 500.0       # back-calculation rate for the integral state [1/s]
SMC_VIN_NOM = VIN         # nominal Vin used by the equivalent control (NOT measured)

# ------------------------------------------------- Gymnasium environment
ENV_CONTROL_STEPS = 4          # converter steps (50 us) per RL decision -> 200 us (5 kHz)
ENV_MAX_STEPS = 400            # RL steps per episode -> 80 ms (~7 settling times of the PID)
# Randomised operating conditions (uniform), as specified for training/testing
ENV_VIN_RANGE = (10.0, 14.0)
ENV_R_RANGE = (10.0, 30.0)
ENV_VREF_RANGE = (20.0, 28.0)
ENV_COMPONENT_TOL = 0.0        # +-fraction on L and C (0 = nominal; used from Stage 8 on)
# Mid-episode disturbance (training randomisation; every field is configurable):
ENV_DISTURBANCE_PROB = 0.5                  # probability that an episode has a disturbance
ENV_DISTURBANCE_WINDOW = (0.25, 0.6)        # time of the step, as a fraction of the episode
ENV_DISTURBANCE_KINDS = ("vin", "load", "both")   # drawn uniformly; "vref" also allowed
ENV_DISTURBANCE_VIN_DELTA = (1.0, 4.0)      # |change in Vin| [V], random sign, clipped to ENV_VIN_RANGE
ENV_DISTURBANCE_LOAD_RATIO = (0.5, 2.0)     # new R / old R, log-uniform, clipped to ENV_R_RANGE
ENV_DISTURBANCE_VREF_DELTA = (1.0, 4.0)     # |change in Vref| [V] when "vref" is in the kinds
# Initial state mix: cold (Vout = 0, IL = 0), precharged (Vout = Vin, IL = 0),
# near-equilibrium (random point close to the regulated operating point)
ENV_INIT_PROBS = {"cold": 0.25, "precharged": 0.25, "equilibrium": 0.50}
# Observation normalisation scales (see environment/boost_env.py)
ENV_E_SCALE = 10.0             # voltage error [V]
ENV_DE_SCALE = 1.0             # error change per RL step [V]
ENV_INT_SCALE = 0.1            # integral of error [V s]
ENV_INT_CLIP = 0.2             # anti-windup clamp of the integral [V s]
ENV_V_SCALE = 24.0             # output voltage [V]
ENV_I_SCALE = 10.0             # inductor current [A]
ENV_OBS_CLIP = 5.0
# Protection limits: exceeding them ends the episode with a penalty
ENV_V_MAX = 60.0               # over-voltage trip [V]
ENV_I_MAX = 30.0               # over-current trip [A]
# Reward weights (see environment/boost_env.py)
ENV_REWARD = dict(
    w_lin=0.5, w_sqrt=0.5,        # tracking: w_lin*x + w_sqrt*sqrt(x), x = |e|/ENV_E_SCALE
    w_over=0.5,                   # extra penalty for output above Vref
    w_duty=0.05, duty_scale=0.05, # pseudo-Huber penalty on duty change (scale in duty units)
    w_de=0.05,                    # penalty on |change in error| per step (damps ringing)
    w_cur=1.0, i_soft=12.0,       # quadratic penalty above a soft current limit [A]
    w_prec=0.5, prec_sigma=(0.5, 0.1),   # smooth precision bonus, Gaussian widths [V]
    alive=1.0,                    # per-step survival bonus (keeps tripping worse than surviving)
    fail_penalty=50.0)            # one-off penalty on a protection trip / invalid state

# ------------------------------------------------------------------ TD3
TD3_MODEL_NAME = "td3_boost"             # saved as models/<name>.zip
TD3_TOTAL_TIMESTEPS = 200_000            # full run; use --timesteps for smoke tests
TD3_SEED = 0
TD3_NET_ARCH = [64, 64]                  # actor and critic hidden layers
TD3_LEARNING_RATE = 1e-3
TD3_BUFFER_SIZE = 200_000
TD3_BATCH_SIZE = 256
TD3_GAMMA = 0.99                         # horizon ~100 RL steps = 20 ms
TD3_TAU = 0.005                          # target-network soft update
TD3_POLICY_DELAY = 2                     # actor/targets updated every 2 critic updates
TD3_TARGET_POLICY_NOISE = 0.2            # target-policy smoothing (normalised action units)
TD3_TARGET_NOISE_CLIP = 0.5
TD3_EXPLORATION_SIGMA = 0.1              # Gaussian exploration noise (normalised units)
TD3_LEARNING_STARTS = 5_000              # random-policy steps before learning
TD3_TRAIN_FREQ = 1
TD3_GRADIENT_STEPS = 1
TD3_LOG_EPISODES = 20                    # print reward statistics every N episodes
TENSORBOARD_DIR = RESULTS_DIR / "tensorboard"

# --------------------------------------------------------- adaptive PID
# Gain ranges as multiples of the validated baseline PID (PID_KP/KI/KD).
# CERTIFIED box = the region verified by experiments/analyze_gain_bounds.py ("tight"):
# the largest box whose full 5^3 grid keeps worst-case PM >= 40 deg and GM >= 4 dB over
# the 27-point operating grid (Vin 10-14 V, R 10-30 ohm, Vref 20-28 V); measured
# 40.3 deg / 4.2 dB against 56.5 deg / 6.0 dB for the baseline itself.
# AdaptivePID raises an error for ranges outside it (override: allow_uncertified=True).
# Rejected boxes: symmetric r = 1.25 / 1.5 / 2 (worst PM 3.8 / -41 / -73 deg) and the
# Stage 6a box Kp 0.25-2, Ki 0.25-1, Kd 0.8-1.5 (only 25 deg / 2.1 dB; the agent drove
# Ki to its floor and settling slowed from 11.5 ms to 37 ms).
ADAPT_CERTIFIED_FACTORS = dict(Kp=(0.7, 1.5), Ki=(0.7, 1.05), Kd=(0.9, 1.2))
# ACTIVE box (configurable, must lie inside the certified box):
ADAPT_RANGE_FACTORS = dict(Kp=(0.7, 1.5), Ki=(0.7, 1.05), Kd=(0.9, 1.2))
ADAPT_KP_RANGE = tuple(PID_KP * f for f in ADAPT_RANGE_FACTORS["Kp"])
ADAPT_KI_RANGE = tuple(PID_KI * f for f in ADAPT_RANGE_FACTORS["Ki"])
ADAPT_KD_RANGE = tuple(PID_KD * f for f in ADAPT_RANGE_FACTORS["Kd"])
# Smoothing of the gain trajectory (applied in log-gain space, once per RL step = 200 us)
ADAPT_SMOOTH_ALPHA = 0.1       # first-order filter toward the requested gain
ADAPT_RATE_LIMIT = 0.05        # max |d ln K| per RL step (5 % per 200 us)
ADAPT_W_GAIN = 0.02            # small reward penalty on distance from the baseline gains
# Baseline-recovery supervisor: when it is engaged the agent's gain requests are ignored
# and the gains are driven back to the baseline PID through the same smoothing/rate limit.
ADAPT_RECOVERY_ENABLED = True
ADAPT_REC_ERR_THR = 2.5        # engage when the filtered |error| stays above this [V] ...
ADAPT_REC_ERR_TAU = 2e-3       # ... (low-pass time constant of |e|) [s]
ADAPT_REC_PERSIST = 8e-3       # ... for this long [s]
ADAPT_REC_IL_WARN = 12.0       # immediate engage: inductor current above this [A]
ADAPT_REC_V_WARN_RATIO = 1.15  # immediate engage: Vout above this multiple of Vref
ADAPT_REC_MIN_HOLD = 5e-3      # stay engaged at least this long [s]
ADAPT_REC_RELEASE_THR = 1.0    # release when filtered |error| is below this [V] ...
ADAPT_REC_RELEASE_HOLD = 5e-3  # ... for this long [s]
ADAPT_MODEL_NAME = "td3_adaptive_pid"

# --------------------------------------------------------- adaptive SMC
# Parameter ranges as multiples of the validated SMC (SMC_KV, SMC_KI, SMC_GAIN = K,
# SMC_BOUNDARY = phi); the SMC law itself is unchanged.
# CERTIFIED box: verified by experiments/analyze_smc_bounds.py ("refine"): all 1296
# corner + mid-point closed-loop simulations (9 extreme plants: Vin 10/14, R 10/30,
# Vref 20/28 + nominal; cold start) passed relative to the baseline SMC, with the joint
# cap K/phi <= 1.2 x baseline (discrete current-loop gain 0.90 at 40 V; instability at 2).
# Wider boxes (Kv,Ki x0.67-1.5 or more, K x0.85-1.2) FAILED plants (slow settling, limit cycles).
ASMC_CERTIFIED_FACTORS = dict(Kv=(0.8, 1.25), Ki=(0.8, 1.25), K=(0.9, 1.1), phi=(0.85, 1.25))
# ACTIVE box (configurable, must lie inside the certified box):
ASMC_RANGE_FACTORS = dict(Kv=(0.8, 1.25), Ki=(0.8, 1.25), K=(0.9, 1.1), phi=(0.9, 1.15))
ASMC_MAX_RATIO_FACTOR = 1.2      # K/phi may not exceed this multiple of the baseline K/phi
ASMC_SMOOTH_ALPHA = 0.1
ASMC_RATE_LIMIT = 0.05           # max |d ln p| per RL step
ASMC_W_PARAM = 0.02              # small reward penalty on distance from the baseline parameters
# Baseline-recovery supervisor (same logic as the adaptive PID, plus a chatter detector)
ASMC_RECOVERY_ENABLED = True
ASMC_REC_ERR_THR = 2.5           # filtered |e| above this [V] ...
ASMC_REC_ERR_TAU = 2e-3          # ... (low-pass tau) [s]
ASMC_REC_PERSIST = 8e-3          # ... for this long [s]
ASMC_REC_IL_WARN = 12.0          # immediate: IL above this [A] (baseline peaks at 10.5 A)
ASMC_REC_V_WARN_RATIO = 1.15     # immediate: Vout above this multiple of Vref
ASMC_REC_CHATTER_THR = 4e-3      # filtered |dD| per 50 us step above this ...
ASMC_REC_CHATTER_TAU = 1e-3
ASMC_REC_CHATTER_PERSIST = 10e-3  # ... for this long (baseline SMC: max 3.95 ms)
ASMC_REC_MIN_HOLD = 5e-3
ASMC_REC_RELEASE_THR = 1.0
ASMC_REC_RELEASE_HOLD = 5e-3
ASMC_MODEL_NAME = "td3_adaptive_smc"

# ------------------------------------------------- final training / evaluation
FINAL_SEEDS = (0, 1, 2)
FINAL_TAG = "final"                       # model files: models/<base>_<tag>_seed<N>.zip
FINAL_MODEL_BASES = {                     # RL controller key -> model-name base
    "direct": "td3_direct",
    "adaptive-pid": "td3_adaptive_pid",
    "adaptive-smc": "td3_adaptive_smc",
}
