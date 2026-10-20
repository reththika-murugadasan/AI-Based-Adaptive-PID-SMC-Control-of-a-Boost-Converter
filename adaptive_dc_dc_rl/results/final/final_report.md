# Final evaluation report (final)

Seeds for RL controllers: [0, 1, 2] (sample std, n = 3). Scenarios: nominal, vin_12_to_10, vin_12_to_14, load_20_to_10, load_20_to_30. Disturbance at 150 ms, run length 300 ms, 50 us sampling. RL policies are deterministic.

Measured values only: no ranking, scoring or overall winner is computed. `n/a` = undefined (e.g. no disturbance in the nominal scenario, or Vout never re-entered the +-2 % band); `[k/n undefined]` = undefined for k of n seeds.

## rise_time_ms

_10-90 % rise time of the cold start [ms] (dominated by diode pre-charge, not control)_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 1.35 | 1.35 | 1.35 | 1.35 | 1.35 |
| Fixed SMC | 1.35 | 1.35 | 1.35 | 1.35 | 1.35 |
| Direct TD3 | 1.333 ± 0.029 | 1.333 ± 0.029 | 1.333 ± 0.029 | 1.333 ± 0.029 | 1.333 ± 0.029 |
| TD3 adaptive PID (6b) | 1.35 ± 0 | 1.35 ± 0 | 1.35 ± 0 | 1.35 ± 0 | 1.35 ± 0 |
| TD3 adaptive SMC | 1.35 ± 0 | 1.35 ± 0 | 1.35 ± 0 | 1.35 ± 0 | 1.35 ± 0 |

## settling_time_ms

_cold start: time after which Vout stays within +-2 % of Vref [ms]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 11.5 | 11.5 | 11.5 | 11.5 | 11.5 |
| Fixed SMC | 2.9 | 2.9 | 2.9 | 2.9 | 2.9 |
| Direct TD3 | 11.18 ± 14 | 11.18 ± 14 | 11.18 ± 14 | 11.18 ± 14 | 11.18 ± 14 |
| TD3 adaptive PID (6b) | 12.07 ± 2.1 | 12.07 ± 2.1 | 12.07 ± 2.1 | 12.07 ± 2.1 | 12.07 ± 2.1 |
| TD3 adaptive SMC | 5.133 ± 1.9 | 5.133 ± 1.9 | 5.133 ± 1.9 | 5.133 ± 1.9 | 5.133 ± 1.9 |

## overshoot_pct

_cold-start overshoot above Vref [% of the step]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 0.001456 | 0.001456 | 0.001456 | 0.001456 | 0.001456 |
| Fixed SMC | 2.624 | 2.624 | 2.624 | 2.624 | 2.624 |
| Direct TD3 | 3.313 ± 0.07 | 3.313 ± 0.07 | 3.313 ± 0.07 | 3.313 ± 0.07 | 3.313 ± 0.07 |
| TD3 adaptive PID (6b) | 0.004511 ± 0.0057 | 0.004511 ± 0.0057 | 0.004511 ± 0.0057 | 0.004511 ± 0.0057 | 0.004511 ± 0.0057 |
| TD3 adaptive SMC | 3.411 ± 0.38 | 3.411 ± 0.38 | 3.411 ± 0.38 | 3.411 ± 0.38 | 3.411 ± 0.38 |

## undershoot_V

_deepest dip below Vref after first reaching 90 % [V]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 5.396 | 5.396 | 5.396 | 5.396 | 5.396 |
| Fixed SMC | 2.065 | 2.065 | 2.065 | 2.065 | 2.065 |
| Direct TD3 | 2.227 ± 0.21 | 2.227 ± 0.21 | 2.227 ± 0.21 | 2.227 ± 0.21 | 2.227 ± 0.21 |
| TD3 adaptive PID (6b) | 5.127 ± 0.059 | 5.127 ± 0.059 | 5.127 ± 0.059 | 5.127 ± 0.059 | 5.127 ± 0.059 |
| TD3 adaptive SMC | 2 ± 0.016 | 2 ± 0.016 | 2 ± 0.016 | 2 ± 0.016 | 2 ± 0.016 |

## ss_error_V

_mean error over the 10 ms before the disturbance [V]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 3.553e-15 | 3.553e-15 | 3.553e-15 | 3.553e-15 | 3.553e-15 |
| Fixed SMC | 3.553e-15 | 3.553e-15 | 3.553e-15 | 3.553e-15 | 3.553e-15 |
| Direct TD3 | -0.04714 ± 0.0047 | -0.04714 ± 0.0047 | -0.04714 ± 0.0047 | -0.04714 ± 0.0047 | -0.04714 ± 0.0047 |
| TD3 adaptive PID (6b) | 3.553e-15 ± 0 | 3.553e-15 ± 0 | 3.553e-15 ± 0 | 3.553e-15 ± 0 | 3.553e-15 ± 0 |
| TD3 adaptive SMC | 9.474e-15 ± 1e-14 | 9.474e-15 ± 1e-14 | 9.474e-15 ± 1e-14 | 9.474e-15 ± 1e-14 | 9.474e-15 ± 1e-14 |

## ss_error_final_V

_mean error over the last 10 ms of the run [V]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 3.553e-15 | 1.522e-12 | 2.842e-14 | -3.553e-15 | -4.974e-14 |
| Fixed SMC | 3.553e-15 | -6.858e-14 | 4.619e-14 | 2.382e-12 | 1.421e-14 |
| Direct TD3 | -0.008817 ± 0.0066 | 0.03636 ± 0.019 | -0.03429 ± 0.011 | 0.1817 ± 0.072 | -0.0486 ± 0.019 |
| TD3 adaptive PID (6b) | 3.553e-15 ± 0 | -3.685e-13 ± 1.3e-12 | -7.105e-15 ± 3.6e-14 | 3.197e-14 ± 5.8e-14 | -1.459e-14 ± 3e-14 |
| TD3 adaptive SMC | 9.474e-15 ± 1e-14 | 3.123e-05 ± 5e-05 | -2.561e-05 ± 4.6e-05 | 3.314e-13 ± 5.5e-13 | -4.737e-15 ± 3.4e-14 |

## rmse_V

_RMS voltage error over the whole run [V]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 1.375 | 1.42 | 1.406 | 1.386 | 1.377 |
| Fixed SMC | 1.256 | 1.259 | 1.258 | 1.28 | 1.26 |
| Direct TD3 | 1.264 ± 0.011 | 1.267 ± 0.0096 | 1.267 ± 0.0094 | 1.305 ± 0.0065 | 1.268 ± 0.0097 |
| TD3 adaptive PID (6b) | 1.365 ± 0.0071 | 1.407 ± 0.0072 | 1.399 ± 0.0091 | 1.374 ± 0.0071 | 1.366 ± 0.007 |
| TD3 adaptive SMC | 1.259 ± 0.0012 | 1.261 ± 0.0012 | 1.26 ± 0.0011 | 1.279 ± 0.00022 | 1.262 ± 0.0015 |

## max_dev_after_V

_maximum |Vout - Vref| after the disturbance time [V]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 3.553e-15 | 3.399 | 2.996 | 1.94 | 0.6384 |
| Fixed SMC | 3.553e-15 | 0.6522 | 0.5625 | 2.094 | 0.8031 |
| Direct TD3 | 0.04402 ± 0.0037 | 0.4422 ± 0.085 | 0.3935 ± 0.012 | 1.854 ± 0.21 | 0.4339 ± 0.0032 |
| TD3 adaptive PID (6b) | 3.553e-15 ± 0 | 3.288 ± 0.029 | 2.786 ± 0.085 | 1.885 ± 0.033 | 0.6085 ± 0.013 |
| TD3 adaptive SMC | 9.474e-15 ± 1e-14 | 0.64 ± 0.042 | 0.56 ± 0.015 | 1.974 ± 0.062 | 0.7838 ± 0.047 |

## recovery_time_ms

_time after the disturbance until Vout stays within +-2 % [ms] (undefined = never recovered, or no disturbance)_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | n/a | 10.6 | 6.5 | 7.65 | 2.1 |
| Fixed SMC | n/a | 4.75 | 3.35 | 9.95 | 5.25 |
| Direct TD3 | n/a | 4.417 ± 7.6 | 0 ± 0 | 49.12 ± 5.5 [1/3 undefined] | 0 ± 0 |
| TD3 adaptive PID (6b) | n/a | 8.2 ± 1.7 | 9.367 ± 0.28 | 7.45 ± 0.18 | 1.95 ± 0.087 |
| TD3 adaptive SMC | n/a | 4.583 ± 0.19 | 3.067 ± 0.076 | 8.967 ± 0.53 | 4.8 ± 0.44 |

## duty_variation

_mean |change of duty| per 50 us step_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 0.0001048 | 0.0001225 | 0.0001245 | 0.0001348 | 0.0001154 |
| Fixed SMC | 0.0001482 | 0.0001691 | 0.0001698 | 0.000179 | 0.0001586 |
| Direct TD3 | 0.00377 ± 0.0052 | 0.005985 ± 0.0035 | 0.008326 ± 0.0048 | 0.004703 ± 0.0048 | 0.004838 ± 0.0055 |
| TD3 adaptive PID (6b) | 0.0001066 ± 1.9e-06 | 0.0001255 ± 2.7e-06 | 0.0001297 ± 1.3e-06 | 0.0001407 ± 1.4e-06 | 0.0001183 ± 1.9e-06 |
| TD3 adaptive SMC | 0.0001734 ± 1.5e-05 | 0.0003619 ± 0.00013 | 0.0002842 ± 8.2e-05 | 0.0002507 ± 1.2e-05 | 0.0001913 ± 1.6e-05 |

## max_IL_A

_maximum inductor current [A]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 8.885 | 8.885 | 8.885 | 8.885 | 8.885 |
| Fixed SMC | 9.12 | 9.12 | 9.12 | 9.12 | 9.12 |
| Direct TD3 | 8.947 ± 0.15 | 8.947 ± 0.15 | 8.947 ± 0.15 | 8.947 ± 0.15 | 8.947 ± 0.15 |
| TD3 adaptive PID (6b) | 8.88 ± 0.0095 | 8.88 ± 0.0095 | 8.88 ± 0.0095 | 8.88 ± 0.0095 | 8.88 ± 0.0095 |
| TD3 adaptive SMC | 9.251 ± 0.039 | 9.251 ± 0.039 | 9.251 ± 0.039 | 9.251 ± 0.039 | 9.251 ± 0.039 |

## min_IL_A

_minimum inductor current [A]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 0.08487 | 0.08487 | 0.08487 | 0.08487 | 0.08487 |
| Fixed SMC | 0.5995 | 0.5995 | 0.5995 | 0.5995 | 0.5995 |
| Direct TD3 | 0.5486 ± 0.089 | 0.5486 ± 0.089 | 0.5486 ± 0.089 | 0.5486 ± 0.089 | 0.5486 ± 0.089 |
| TD3 adaptive PID (6b) | 0.09338 ± 0.0092 | 0.09338 ± 0.0092 | 0.09338 ± 0.0092 | 0.09338 ± 0.0092 | 0.09338 ± 0.0092 |
| TD3 adaptive SMC | 0.5995 ± 0 | 0.5995 ± 0 | 0.5995 ± 0 | 0.5995 ± 0 | 0.5995 ± 0 |

## protection_trip

_1 if the run hit a protection limit (Vout > 60 V, IL > 30 A, negative or non-finite state)_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 0 | 0 | 0 | 0 | 0 |
| Fixed SMC | 0 | 0 | 0 | 0 | 0 |
| Direct TD3 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| TD3 adaptive PID (6b) | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| TD3 adaptive SMC | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |

## protection trips and supervisor activity

| controller | runs | protection trips | supervisor engagements (sum) |
|---|---|---|---|
| Fixed PID | 5 | 0 | 0 |
| Fixed SMC | 5 | 0 | 0 |
| Direct TD3 | 15 | 0 | 0 |
| TD3 adaptive PID (6b) | 15 | 0 | 15 |
| TD3 adaptive SMC | 15 | 0 | 0 |
