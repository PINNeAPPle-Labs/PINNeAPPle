A pre-trained surrogate is only useful if you can tell when not to trust it. Most model downloads give you weights and a loss curve, not the range where the answer means something.

We turned the PINNeAPPle model zoo into one static page. No server, no Python. Pick a model, change the inputs, press "Evaluate in your browser": the model's ONNX graph runs on your device, together with its query checks. A small dependency-free ONNX interpreter ships with the library; onnxruntime-web can be used instead. In my local run of the page in headless Chromium, the heat-sink sweep took 28.3 ms.

The page serves 4 models. Each card states a held-out error against an independent reference:
- airfoil (Cl, Cd, Cm): Cd 0.54 % mean abs. error on held-out shapes vs OpenFOAM RANS, and 2.8 % vs XFOIL on NACA 2412;
- forced-convection heat sink: thermal resistance 0.71 % mean error on 900 held-out designs;
- rover wheel on lunar regolith: drawbar pull RMSE 0.22 N on 2000 held-out points.

Trust is part of the output. The ONNX export is refused if it differs from the PyTorch model by more than 1e-5 on the zoo's held-out inputs. Every query is checked for validity range, physical outputs, ensemble agreement and physics checks. In the video I set the airfoil's angle of attack to 16 deg; the card says -2 to 14 deg, so the page marks the answer as an extrapolation instead of quietly giving a number. The same checks run in Python through pp.zoo.query_checks, and a JS test compares the two.

Code, the page and the model cards:
https://github.com/PINNeAPPle-Labs/PINNeAPPle/tree/main/apps/zoo_space
Guide: https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/docs/guides/model_zoo.md

#SciML #SurrogateModels #ONNX #PhysicsInformedML #CFD #ThermalEngineering #OpenSource

---
## Media
1. `media/zoo-onnx-in-browser/zoo_onnx_in_browser.mp4` (49 s, 1080x1350): the static zoo page served locally. The airfoil surrogate is evaluated in the browser with ONNX. Angle of attack 16 deg is flagged as outside the validity range. Then the heat-sink and rover-wheel models.

## Sources
- 4 models: `pinneapple_hub/zoo/manifest.json` / `index.json` written by `pinneapple_hub.zoo.demo.export_space` (models: airfoil_cst_rans_mlp, heatsink_forced_mlp, heatsink_natural_mlp, rover_contact_grc1_viper); also the table in `docs/guides/model_zoo.md`.
- Cd 0.54 %: `pinneapple_hub/zoo/assets/models/airfoil_cst_rans_mlp/model_card.json` -> `validation_metrics["test/Cd_mape_pct"]` = 0.5447 (15 unseen shapes per `docs/guides/model_zoo.md` table).
- 2.8 % vs XFOIL: same card -> `validation_metrics["xfoil_naca2412/Cd_mape_pct"]` = 2.8127.
- Validity -2 to 14 deg: same card -> `validity_range["alpha_deg"]` = [-2.0, 14.0].
- 0.71 %, 900 designs: `pinneapple_hub/zoo/assets/models/heatsink_forced_mlp/model_card.json` -> `validation_metrics["test/R_mean_abs_pct"]` = 0.7115, `["test/n"]` = 900.
- 0.22 N, 2000 points: `pinneapple_hub/zoo/assets/models/rover_contact_grc1_viper/model_card.json` -> `validation_metrics["test/Fx_rmse_N"]` = 0.2201, `["test/n"]` = 2000.
- 1e-5 export parity limit: `docs/guides/model_zoo.md` line 170.
- 28.3 ms: my own run (Playwright, headless Chromium /opt/pw-browsers/chromium, page built with `pinneapple_hub.zoo.publish.build_space` and served with `python -m http.server`). The page printed "Computed in your browser (ONNX, 28.3 ms)" for the heat-sink sweep. The first airfoil evaluation, including the graph load, took 748 ms.
- Built-in interpreter / onnxruntime-web option: `pinneapple_hub/zoo/web/onnx.js` lines 4-5, `pinneapple_hub/zoo/web/zoo.js` line 30.
