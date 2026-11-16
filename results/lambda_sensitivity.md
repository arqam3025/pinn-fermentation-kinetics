# Physics-weight sensitivity (one case: noise 5 %, seed 0)

Produced while debugging the PINN (Nov 2026). True parameters: mu_max 0.15, Ks 0.6, Y 0.55.

| lambda | mu_max | Ks | Y | biomass forecast RMSE (g/L) |
|---|---|---|---|---|
| 0.001 | 0.178 | 9.50 | 0.393 | 2.51 |
| 0.01 | 0.192 | 5.25 | 0.520 | 0.75 |
| 0.1 (used) | 0.190 | 4.72 | 0.548 | 0.46 |
| 1.0 | 0.302 | 1.30 | 0.061 | 10.74 (collapsed to X = 0) |

The first version (physics term on from step 0, no warm-up) collapsed to the trivial X = 0 solution in every case.
