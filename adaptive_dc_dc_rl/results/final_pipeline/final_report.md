# Final evaluation report (pipeline)

Seeds for RL controllers: [0, 1, 2] (sample std, n = 3). Scenarios: nominal, vin_12_to_10, vin_12_to_14, load_20_to_10, load_20_to_30. Disturbance at 150 ms, run length 300 ms, 50 us sampling. RL policies are deterministic.

Measured values only: no ranking, scoring or overall winner is computed. `n/a` = undefined (e.g. no disturbance in the nominal scenario, or Vout never re-entered the +-2 % band); `[k/n undefined]` = undefined for k of n seeds.

## rise_time_ms

_10-90 % rise time of the cold start [ms] (dominated by diode pre-charge, not control)_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 1.35 | 1.35 | 1.35 | 1.35 | 1.35 |
| Fixed SMC | 1.35 | 1.35 | 1.35 | 1.35 | 1.35 |
| Direct TD3 | 1.317 ± 0.058 | 1.317 ± 0.058 | 1.317 ± 0.058 | 1.317 ± 0.058 | 1.317 ± 0.058 |
| TD3 adaptive PID (6b) | 1.317 ± 0.029 | 1.317 ± 0.029 | 1.317 ± 0.029 | 1.317 ± 0.029 | 1.317 ± 0.029 |
| TD3 adaptive SMC | 1.35 ± 0 | 1.35 ± 0 | 1.35 ± 0 | 1.35 ± 0 | 1.35 ± 0 |

## settling_time_ms

_cold start: time after which Vout stays within +-2 % of Vref [ms]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 11.5 | 11.5 | 11.5 | 11.5 | 11.5 |
| Fixed SMC | 2.9 | 2.9 | 2.9 | 2.9 | 2.9 |
| Direct TD3 | n/a | n/a | n/a | n/a | n/a |
| TD3 adaptive PID (6b) | 13.02 ± 2.8 | 13.02 ± 2.8 | 13.02 ± 2.8 | 13.02 ± 2.8 | 13.02 ± 2.8 |
| TD3 adaptive SMC | 4.45 ± 2 | 4.45 ± 2 | 4.45 ± 2 | 4.45 ± 2 | 4.45 ± 2 |

## overshoot_pct

_cold-start overshoot above Vref [% of the step]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 0.001456 | 0.001456 | 0.001456 | 0.001456 | 0.001456 |
| Fixed SMC | 2.624 | 2.624 | 2.624 | 2.624 | 2.624 |
| Direct TD3 | 17.04 ± 9.4 | 17.04 ± 9.4 | 17.04 ± 9.4 | 17.04 ± 9.4 | 17.04 ± 9.4 |
| TD3 adaptive PID (6b) | 0.09579 ± 0.17 | 0.09579 ± 0.17 | 0.09579 ± 0.17 | 0.09579 ± 0.17 | 0.09579 ± 0.17 |
| TD3 adaptive SMC | 4.646 ± 0.34 | 4.646 ± 0.34 | 4.646 ± 0.34 | 4.646 ± 0.34 | 4.646 ± 0.34 |

## undershoot_V

_deepest dip below Vref after first reaching 90 % [V]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 5.396 | 5.396 | 5.396 | 5.396 | 5.396 |
| Fixed SMC | 2.065 | 2.065 | 2.065 | 2.065 | 2.065 |
| Direct TD3 | 1.978 ± 0.38 | 1.978 ± 0.38 | 1.978 ± 0.38 | 1.978 ± 0.38 | 1.978 ± 0.38 |
| TD3 adaptive PID (6b) | 5.339 ± 0.51 | 5.339 ± 0.51 | 5.339 ± 0.51 | 5.339 ± 0.51 | 5.339 ± 0.51 |
| TD3 adaptive SMC | 2.04 ± 0.069 | 2.04 ± 0.069 | 2.04 ± 0.069 | 2.04 ± 0.069 | 2.04 ± 0.069 |

## ss_error_V

_mean error over the 10 ms before the disturbance [V]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 3.553e-15 | 3.553e-15 | 3.553e-15 | 3.553e-15 | 3.553e-15 |
| Fixed SMC | 3.553e-15 | 3.553e-15 | 3.553e-15 | 3.553e-15 | 3.553e-15 |
| Direct TD3 | 0.8471 ± 2.7 | 0.8471 ± 2.7 | 0.8471 ± 2.7 | 0.8471 ± 2.7 | 0.8471 ± 2.7 |
| TD3 adaptive PID (6b) | -2.115e-14 ± 4e-14 | -2.115e-14 ± 4e-14 | -2.115e-14 ± 4e-14 | -2.115e-14 ± 4e-14 | -2.115e-14 ± 4e-14 |
| TD3 adaptive SMC | 1.457e-10 ± 2.5e-10 | 1.457e-10 ± 2.5e-10 | 1.457e-10 ± 2.5e-10 | 1.457e-10 ± 2.5e-10 | 1.457e-10 ± 2.5e-10 |

## ss_error_final_V

_mean error over the last 10 ms of the run [V]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 3.553e-15 | 1.522e-12 | 2.842e-14 | -3.553e-15 | -4.974e-14 |
| Fixed SMC | 3.553e-15 | -6.858e-14 | 4.619e-14 | 2.382e-12 | 1.421e-14 |
| Direct TD3 | 0.79 ± 2.7 | 5.897 ± 6.8 | -1.914 ± 5.3 | 4.972 ± 5.9 | 0.9305 ± 2.6 |
| TD3 adaptive PID (6b) | -1.582e-14 ± 3.1e-14 | 1.113e-12 ± 1.3e-12 | -2.842e-14 ± 2.6e-14 | -2.605e-14 ± 6.9e-14 | -4.737e-15 ± 7.7e-14 |
| TD3 adaptive SMC | 3.553e-15 ± 0 | 1.033e-10 ± 1.8e-10 | -8.46e-11 ± 1.5e-10 | 1.882e-08 ± 3.3e-08 | -1.494e-11 ± 2.6e-11 |

## rmse_V

_RMS voltage error over the whole run [V]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 1.375 | 1.42 | 1.406 | 1.386 | 1.377 |
| Fixed SMC | 1.256 | 1.259 | 1.258 | 1.28 | 1.26 |
| Direct TD3 | 4.512 ± 4.3 | 7.038 ± 3.8 | 5.477 ± 3.5 | 6.193 ± 3.6 | 4.52 ± 4.3 |
| TD3 adaptive PID (6b) | 1.382 ± 0.039 | 1.424 ± 0.039 | 1.411 ± 0.04 | 1.392 ± 0.039 | 1.383 ± 0.039 |
| TD3 adaptive SMC | 1.26 ± 0.00022 | 1.262 ± 0.00044 | 1.262 ± 0.00035 | 1.28 ± 0.0042 | 1.263 ± 0.00064 |

## max_dev_after_V

_maximum |Vout - Vref| after the disturbance time [V]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 3.553e-15 | 3.399 | 2.996 | 1.94 | 0.6384 |
| Fixed SMC | 3.553e-15 | 0.6522 | 0.5625 | 2.094 | 0.8031 |
| Direct TD3 | 1.998 ± 1 [1/3 undefined] | 9.329 ± 6.9 [1/3 undefined] | 8.27 ± 6.3 [1/3 undefined] | 8.103 ± 6.8 [1/3 undefined] | 2.273 ± 0.38 [1/3 undefined] |
| TD3 adaptive PID (6b) | 4.145e-14 ± 6.9e-14 | 3.296 ± 0.11 | 2.848 ± 0.17 | 1.894 ± 0.046 | 0.6156 ± 0.023 |
| TD3 adaptive SMC | 6.175e-11 ± 1.1e-10 | 0.5841 ± 0.0046 | 0.4998 ± 0.0025 | 1.891 ± 0.0079 | 0.7045 ± 0.013 |

## recovery_time_ms

_time after the disturbance until Vout stays within +-2 % [ms] (undefined = never recovered, or no disturbance)_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | n/a | 10.6 | 6.5 | 7.65 | 2.1 |
| Fixed SMC | n/a | 4.75 | 3.35 | 9.95 | 5.25 |
| Direct TD3 | n/a | n/a | n/a | n/a | n/a |
| TD3 adaptive PID (6b) | n/a | 8.517 ± 3 | 6.867 ± 0.19 | 7.567 ± 0.6 | 1.983 ± 0.1 |
| TD3 adaptive SMC | n/a | 3.9 ± 0.48 | 2.333 ± 0.24 | 10.42 ± 2.4 | 4.617 ± 0.74 |

## duty_variation

_mean |change of duty| per 50 us step_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 0.0001048 | 0.0001225 | 0.0001245 | 0.0001348 | 0.0001154 |
| Fixed SMC | 0.0001482 | 0.0001691 | 0.0001698 | 0.000179 | 0.0001586 |
| Direct TD3 | 0.006981 ± 0.0062 | 0.005379 ± 0.0038 | 0.008421 ± 0.0082 | 0.006109 ± 0.0041 | 0.006744 ± 0.0059 |
| TD3 adaptive PID (6b) | 0.0001072 ± 3.9e-07 | 0.0001257 ± 4e-07 | 0.0001281 ± 8.5e-08 | 0.0001403 ± 3e-06 | 0.0001186 ± 6.2e-07 |
| TD3 adaptive SMC | 0.0001755 ± 3.1e-06 | 0.0001975 ± 3.2e-06 | 0.0001983 ± 3.2e-06 | 0.0002114 ± 3.5e-06 | 0.0001873 ± 3.3e-06 |

## max_IL_A

_maximum inductor current [A]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 8.885 | 8.885 | 8.885 | 8.885 | 8.885 |
| Fixed SMC | 9.12 | 9.12 | 9.12 | 9.12 | 9.12 |
| Direct TD3 | 18.08 ± 12 | 18.08 ± 12 | 18.25 ± 12 | 18.08 ± 12 | 18.08 ± 12 |
| TD3 adaptive PID (6b) | 8.891 ± 0.017 | 8.891 ± 0.017 | 8.891 ± 0.017 | 8.891 ± 0.017 | 8.891 ± 0.017 |
| TD3 adaptive SMC | 9.335 ± 0.093 | 9.335 ± 0.093 | 9.335 ± 0.093 | 9.335 ± 0.093 | 9.335 ± 0.093 |

## min_IL_A

_minimum inductor current [A]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 0.08487 | 0.08487 | 0.08487 | 0.08487 | 0.08487 |
| Fixed SMC | 0.5995 | 0.5995 | 0.5995 | 0.5995 | 0.5995 |
| Direct TD3 | 0.2503 ± 0.3 | 0.2185 ± 0.33 | 0.2203 ± 0.33 | 0.2262 ± 0.32 | 0.2181 ± 0.33 |
| TD3 adaptive PID (6b) | 0.08681 ± 0.018 | 0.08681 ± 0.018 | 0.08681 ± 0.018 | 0.08681 ± 0.018 | 0.08681 ± 0.018 |
| TD3 adaptive SMC | 0.5995 ± 0 | 0.5995 ± 0 | 0.5995 ± 0 | 0.5995 ± 0 | 0.5995 ± 0 |

## protection_trip

_1 if the run hit a protection limit (Vout > 60 V, IL > 30 A, negative or non-finite state)_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 0 | 0 | 0 | 0 | 0 |
| Fixed SMC | 0 | 0 | 0 | 0 | 0 |
| Direct TD3 | 0.3333 ± 0.58 | 0.3333 ± 0.58 | 0.3333 ± 0.58 | 0.3333 ± 0.58 | 0.3333 ± 0.58 |
| TD3 adaptive PID (6b) | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| TD3 adaptive SMC | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |

## protection trips and supervisor activity

| controller | runs | protection trips | supervisor engagements (sum) |
|---|---|---|---|
| Fixed PID | 5 | 0 | 0 |
| Fixed SMC | 5 | 0 | 0 |
| Direct TD3 | 15 | 5 | 0 |
| TD3 adaptive PID (6b) | 15 | 0 | 15 |
| TD3 adaptive SMC | 15 | 0 | 0 |
