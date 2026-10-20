# Final evaluation report (mid)

Seeds for RL controllers: [0, 1, 2] (sample std, n = 3). Scenarios: nominal, vin_12_to_10, vin_12_to_14, load_20_to_10, load_20_to_30. Disturbance at 150 ms, run length 300 ms, 50 us sampling. RL policies are deterministic.

Measured values only: no ranking, scoring or overall winner is computed. `n/a` = undefined (e.g. no disturbance in the nominal scenario, or Vout never re-entered the +-2 % band); `[k/n undefined]` = undefined for k of n seeds.

## rise_time_ms

_10-90 % rise time of the cold start [ms] (dominated by diode pre-charge, not control)_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 1.35 | 1.35 | 1.35 | 1.35 | 1.35 |
| Fixed SMC | 1.35 | 1.35 | 1.35 | 1.35 | 1.35 |
| Direct TD3 | 1.35 ± 0 | 1.35 ± 0 | 1.35 ± 0 | 1.35 ± 0 | 1.35 ± 0 |
| TD3 adaptive PID (6b) | 1.333 ± 0.029 | 1.333 ± 0.029 | 1.333 ± 0.029 | 1.333 ± 0.029 | 1.333 ± 0.029 |
| TD3 adaptive SMC | 1.35 ± 0 | 1.35 ± 0 | 1.35 ± 0 | 1.35 ± 0 | 1.35 ± 0 |

## settling_time_ms

_cold start: time after which Vout stays within +-2 % of Vref [ms]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 11.5 | 11.5 | 11.5 | 11.5 | 11.5 |
| Fixed SMC | 2.9 | 2.9 | 2.9 | 2.9 | 2.9 |
| Direct TD3 | 3.067 ± 0.72 | 3.067 ± 0.72 | 3.067 ± 0.72 | 3.067 ± 0.72 | 3.067 ± 0.72 |
| TD3 adaptive PID (6b) | 14.03 ± 0.37 | 14.03 ± 0.37 | 14.03 ± 0.37 | 14.03 ± 0.37 | 14.03 ± 0.37 |
| TD3 adaptive SMC | 4.05 ± 1.9 | 4.05 ± 1.9 | 4.05 ± 1.9 | 4.05 ± 1.9 | 4.05 ± 1.9 |

## overshoot_pct

_cold-start overshoot above Vref [% of the step]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 0.001456 | 0.001456 | 0.001456 | 0.001456 | 0.001456 |
| Fixed SMC | 2.624 | 2.624 | 2.624 | 2.624 | 2.624 |
| Direct TD3 | 2.685 ± 0.84 | 2.685 ± 0.84 | 2.685 ± 0.84 | 2.685 ± 0.84 | 2.685 ± 0.84 |
| TD3 adaptive PID (6b) | 5.993e-05 ± 6.1e-05 | 5.993e-05 ± 6.1e-05 | 5.993e-05 ± 6.1e-05 | 5.993e-05 ± 6.1e-05 | 5.993e-05 ± 6.1e-05 |
| TD3 adaptive SMC | 3.212 ± 0.67 | 3.212 ± 0.67 | 3.212 ± 0.67 | 3.212 ± 0.67 | 3.212 ± 0.67 |

## undershoot_V

_deepest dip below Vref after first reaching 90 % [V]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 5.396 | 5.396 | 5.396 | 5.396 | 5.396 |
| Fixed SMC | 2.065 | 2.065 | 2.065 | 2.065 | 2.065 |
| Direct TD3 | 2.291 ± 0.14 | 2.291 ± 0.14 | 2.291 ± 0.14 | 2.291 ± 0.14 | 2.291 ± 0.14 |
| TD3 adaptive PID (6b) | 5.5 ± 0.33 | 5.5 ± 0.33 | 5.5 ± 0.33 | 5.5 ± 0.33 | 5.5 ± 0.33 |
| TD3 adaptive SMC | 2.033 ± 0.035 | 2.033 ± 0.035 | 2.033 ± 0.035 | 2.033 ± 0.035 | 2.033 ± 0.035 |

## ss_error_V

_mean error over the 10 ms before the disturbance [V]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 3.553e-15 | 3.553e-15 | 3.553e-15 | 3.553e-15 | 3.553e-15 |
| Fixed SMC | 3.553e-15 | 3.553e-15 | 3.553e-15 | 3.553e-15 | 3.553e-15 |
| Direct TD3 | -0.03885 ± 0.019 | -0.03885 ± 0.019 | -0.03885 ± 0.019 | -0.03885 ± 0.019 | -0.03885 ± 0.019 |
| TD3 adaptive PID (6b) | 7.306e-16 ± 5.2e-14 | 7.306e-16 ± 5.2e-14 | 7.306e-16 ± 5.2e-14 | 7.306e-16 ± 5.2e-14 | 7.306e-16 ± 5.2e-14 |
| TD3 adaptive SMC | 2.368e-15 ± 2.1e-15 | 2.368e-15 ± 2.1e-15 | 2.368e-15 ± 2.1e-15 | 2.368e-15 ± 2.1e-15 | 2.368e-15 ± 2.1e-15 |

## ss_error_final_V

_mean error over the last 10 ms of the run [V]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 3.553e-15 | 1.522e-12 | 2.842e-14 | -3.553e-15 | -4.974e-14 |
| Fixed SMC | 3.553e-15 | -6.858e-14 | 4.619e-14 | 2.382e-12 | 1.421e-14 |
| Direct TD3 | -0.007127 ± 0.0029 | 0.1444 ± 0.17 | -0.06673 ± 0.013 | 0.6443 ± 0.47 | -0.08191 ± 0.026 |
| TD3 adaptive PID (6b) | -1.279e-15 ± 5.6e-14 | -4.796e-12 ± 8.4e-12 | -3.908e-14 ± 3e-14 | 2.605e-14 ± 5.1e-14 | -2.329e-14 ± 3.4e-14 |
| TD3 adaptive SMC | 2.368e-15 ± 2.1e-15 | 3.349e-12 ± 3.2e-12 | 2.148e-06 ± 3.7e-06 | 1.745e-12 ± 3e-12 | 8.655e-15 ± 3.3e-14 |

## rmse_V

_RMS voltage error over the whole run [V]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 1.375 | 1.42 | 1.406 | 1.386 | 1.377 |
| Fixed SMC | 1.256 | 1.259 | 1.258 | 1.28 | 1.26 |
| Direct TD3 | 1.259 ± 0.0031 | 1.266 ± 0.0068 | 1.267 ± 0.01 | 1.355 ± 0.04 | 1.266 ± 0.0027 |
| TD3 adaptive PID (6b) | 1.391 ± 0.018 | 1.433 ± 0.017 | 1.422 ± 0.016 | 1.401 ± 0.018 | 1.392 ± 0.018 |
| TD3 adaptive SMC | 1.258 ± 0.0012 | 1.261 ± 0.0014 | 1.26 ± 0.0015 | 1.28 ± 0.0016 | 1.263 ± 0.0016 |

## max_dev_after_V

_maximum |Vout - Vref| after the disturbance time [V]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 3.553e-15 | 3.399 | 2.996 | 1.94 | 0.6384 |
| Fixed SMC | 3.553e-15 | 0.6522 | 0.5625 | 2.094 | 0.8031 |
| Direct TD3 | 0.03651 ± 0.017 | 0.4459 ± 0.17 | 0.5399 ± 0.21 | 1.872 ± 0.19 | 0.4632 ± 0.066 |
| TD3 adaptive PID (6b) | 6.395e-14 ± 6.5e-14 | 3.326 ± 0.12 | 2.974 ± 0.043 | 1.931 ± 0.014 | 0.6381 ± 0.0033 |
| TD3 adaptive SMC | 2.368e-15 ± 2.1e-15 | 0.69 ± 0.034 | 0.6272 ± 0.017 | 2.149 ± 0.038 | 0.8831 ± 0.024 |

## recovery_time_ms

_time after the disturbance until Vout stays within +-2 % [ms] (undefined = never recovered, or no disturbance)_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | n/a | 10.6 | 6.5 | 7.65 | 2.1 |
| Fixed SMC | n/a | 4.75 | 3.35 | 9.95 | 5.25 |
| Direct TD3 | n/a | 0.9333 ± 1.6 | 18.17 ± 31 | 116.1 ± 0 [2/3 undefined] | 0.2833 ± 0.49 |
| TD3 adaptive PID (6b) | n/a | 8.5 ± 2.3 | 7.417 ± 1.1 | 7.85 ± 0.2 | 2.083 ± 0.029 |
| TD3 adaptive SMC | n/a | 4.833 ± 0.29 | 4.183 ± 0.64 | 8.833 ± 0.81 | 6.017 ± 0.41 |

## duty_variation

_mean |change of duty| per 50 us step_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 0.0001048 | 0.0001225 | 0.0001245 | 0.0001348 | 0.0001154 |
| Fixed SMC | 0.0001482 | 0.0001691 | 0.0001698 | 0.000179 | 0.0001586 |
| Direct TD3 | 0.0002722 ± 0.0001 | 0.0003338 ± 0.00013 | 0.005758 ± 0.0083 | 0.00581 ± 0.0025 | 0.0008825 ± 0.00099 |
| TD3 adaptive PID (6b) | 0.0001082 ± 9e-07 | 0.0001275 ± 5.5e-07 | 0.0001295 ± 9e-07 | 0.0001397 ± 1.2e-06 | 0.0001189 ± 7.6e-07 |
| TD3 adaptive SMC | 0.0001648 ± 1.5e-05 | 0.0001883 ± 1.3e-05 | 0.0002081 ± 3.1e-05 | 0.0002239 ± 3.5e-05 | 0.0001765 ± 1.3e-05 |

## max_IL_A

_maximum inductor current [A]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 8.885 | 8.885 | 8.885 | 8.885 | 8.885 |
| Fixed SMC | 9.12 | 9.12 | 9.12 | 9.12 | 9.12 |
| Direct TD3 | 8.856 ± 0.045 | 8.856 ± 0.045 | 8.856 ± 0.045 | 8.856 ± 0.045 | 8.856 ± 0.045 |
| TD3 adaptive PID (6b) | 8.885 ± 0.011 | 8.885 ± 0.011 | 8.885 ± 0.011 | 8.885 ± 0.011 | 8.885 ± 0.011 |
| TD3 adaptive SMC | 9.202 ± 0.11 | 9.202 ± 0.11 | 9.202 ± 0.11 | 9.202 ± 0.11 | 9.202 ± 0.11 |

## min_IL_A

_minimum inductor current [A]_

| controller | nominal | vin_12_to_10 | vin_12_to_14 | load_20_to_10 | load_20_to_30 |
|---|---|---|---|---|---|
| Fixed PID | 0.08487 | 0.08487 | 0.08487 | 0.08487 | 0.08487 |
| Fixed SMC | 0.5995 | 0.5995 | 0.5995 | 0.5995 | 0.5995 |
| Direct TD3 | 0.4988 ± 0.17 | 0.4988 ± 0.17 | 0.4988 ± 0.17 | 0.4988 ± 0.17 | 0.4988 ± 0.17 |
| TD3 adaptive PID (6b) | 0.08166 ± 0.016 | 0.08166 ± 0.016 | 0.08166 ± 0.016 | 0.08166 ± 0.016 | 0.08166 ± 0.016 |
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
