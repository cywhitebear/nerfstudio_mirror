#!/usr/bin/env bash
set -e

# Validate the input arguments
if [ $# -lt 1 ]; then
  echo "Usage: $0 <input_base_path> [<method1> <method2> ...]"
  echo "Available methods:arkit colmap, loftr, lightglue, glomap"
  echo "Default method: arkit"
  exit 1
fi

input_base_path=$1
shift
methods=("${@:-arkit}")

echo "input_base_path: ${input_base_path}"
echo "methods: ${methods[@]}"

remove_and_create_folder() {
  if [ -d "$1" ]; then
    rm -rf "$1"
  fi
  mkdir -p "$1"
}

echo "=== Preprocess ARkit data === "
remove_and_create_folder "${input_base_path}/post"
remove_and_create_folder "${input_base_path}/post/sparse"
remove_and_create_folder "${input_base_path}/post/sparse/online"
remove_and_create_folder "${input_base_path}/post/sparse/online_loop"

echo "1. Undistort image using AVFoundation calibration data"
python arkit_utils/undistort_images/undistort_image_cuda.py --input_base ${input_base_path}

echo "2. Transform ARKit mesh to point3D"
python arkit_utils/mesh_to_points3D/arkitobj2point3D.py --input_base_path ${input_base_path}

echo "3. Transform ARKit pose to COLMAP coordinate"
python arkit_utils/arkit_pose_to_colmap.py --input_database_path ${input_base_path}

echo "4. Optimize pose using selected methods"
if [ "${methods[@]}" -ne "arkit" ]; then
remove_and_create_folder "${input_base_path}/post/sparse/offline"
python arkit_utils/pose_optimization/optimize_pose.py --input_database_path ${input_base_path} --methods "${methods[@]}"
else
  echo "Skipping pose optimization"
fi

echo "5. Prepare dataset for nerfstudio"
python arkit_utils/prepare_nerfstudio_dataset.py --input_path ${input_base_path}

echo "Dataset preparation completed."

echo "6. Start training nerfstudio"
python arkit_utils/run_nerfstudio_dataset.py --input_path ${input_base_path} --method "${methods[@]}"