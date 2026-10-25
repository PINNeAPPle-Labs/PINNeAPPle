| study | quantity | mesh | cells | measured | reference | error | source |
|---|---|---|---|---|---|---|---|
| pipe_bend | straight | 16 | 117181 | 0.3196 | 0.3206 | 0.32 % | Hagen-Poiseuille, f = 64 / Re |
| pipe_bend | straight | 24 | 383891 | 0.3205 | 0.3204 | 0.05 % | Hagen-Poiseuille, f = 64 / Re |
| pipe_bend | straight | 32 | 921481 | 0.3210 | 0.3202 | 0.24 % | Hagen-Poiseuille, f = 64 / Re |
| pipe_bend | straight | Richardson (p = 1.03, GCI 0.5 %) | - | 0.3223 | 0.3202 | 0.65 % | Hagen-Poiseuille, f = 64 / Re |
| pipe_bend | bend | 16 | 117181 | 1.5441 | 1.3729 | 12.47 % | White (1929) curved-pipe friction |
| pipe_bend | bend | 24 | 383891 | 1.4712 | 1.3733 | 7.13 % | White (1929) curved-pipe friction |
| pipe_bend | bend | 32 | 921481 | 1.4375 | 1.3734 | 4.67 % | White (1929) curved-pipe friction |
| pipe_bend | bend | Richardson (p = 1.23, GCI 6.9 %) | - | 1.3580 | 1.3734 | 1.13 % | White (1929) curved-pipe friction |
| heated_pipe | Nu | 12 | 21504 | 4.7641 | 4.6919 | 1.54 % | Hausen (1943), Incropera eq. 8.57 |
| heated_pipe | Nu | 16 | 53248 | 4.7110 | 4.6919 | 0.41 % | Hausen (1943), Incropera eq. 8.57 |
| heated_pipe | Nu | 24 | 172032 | 4.6786 | 4.6919 | 0.28 % | Hausen (1943), Incropera eq. 8.57 |
| heated_pipe | Nu | Richardson (p = 2.47, GCI 0.5 %) | - | 4.6597 | 4.6919 | 0.69 % | Hausen (1943), Incropera eq. 8.57 |
| sphere_drag | CD | 3 | 102308 | 1.0833 | 1.0870 | 0.34 % | Clift, Grace & Weber (1978), Table 5.2 |
| sphere_drag | CD | 4 | 124291 | 1.0842 | 1.0870 | 0.26 % | Clift, Grace & Weber (1978), Table 5.2 |
| sphere_drag | CD | 5 | 216328 | 1.0866 | 1.0870 | 0.04 % | Clift, Grace & Weber (1978), Table 5.2 |
| sphere_drag | CD | Richardson: not applied | | | | | observed order -1.41 <= 0: the three meshes are not in the asymptotic range (changes below the discretisation noise) |
| sphere_drag Re 50 | CD | 4 | 124291 | 1.5689 | 1.5743 | 0.34 % | Clift, Grace & Weber (1978), Table 5.2 |
| sphere_drag Re 200 | CD | 4 | 124291 | 0.7775 | 0.7756 | 0.24 % | Clift, Grace & Weber (1978), Table 5.2 |

| surrogate (dp_total_pa from Rc, angle) | held-out nRMSE |
|---|---|
| Polynomial ridge | 0.0146 |
| Gaussian process | 0.0159 |
| MLP (chosen) | 0.0206 |
| Gradient-boosted trees | 0.8896 |
