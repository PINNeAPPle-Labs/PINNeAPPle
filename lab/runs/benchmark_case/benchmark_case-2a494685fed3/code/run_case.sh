#!/bin/bash
# Runs one OpenRadioss case inside the container. Usage: run_case.sh <case_dir>
set -e
OR=/opt/OpenRadioss
export OPENRADIOSS_PATH=$OR RAD_CFG_PATH=$OR/hm_cfg_files RAD_H3D_PATH=$OR/extlib/h3d/lib/linuxa64
export LD_LIBRARY_PATH=$OR/extlib/ArmFlang_runtime/linuxa64:$OR/extlib/h3d/lib/linuxa64:$OR/extlib/hm_reader/linuxa64:$LD_LIBRARY_PATH
export OMP_STACKSIZE=400m OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
cd "$1"
$OR/exec/starter_linuxa64 -i Bumper_Beam_AP_meshed_0000.rad -np 1 > starter.log 2>&1
$OR/exec/engine_linuxa64 -i Bumper_Beam_AP_meshed_0001.rad > engine.log 2>&1
mkdir -p vtk
for f in Bumper_Beam_AP_meshedA[0-9][0-9][0-9]; do
  $OR/exec/anim_to_vtk_linuxa64 "$f" > "vtk/$f.vtk"
done
rm -f Bumper_Beam_AP_meshedA[0-9][0-9][0-9]
echo DONE > done.flag
