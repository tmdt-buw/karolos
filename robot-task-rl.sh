#!/bin/bash
# execute in main directory where trainer/trainer.py is valid
# and pass parameters: bash robot-task-rl.sh python_executable commit_hash

if [[ ! -f trainer/trainer.py ]]; then
    echo "trainer/trainer.py does not exist"
    exit 1
fi

latest_master_commit_hash=$(git ls-remote https://git.uni-wuppertal.de/scheiderer/robot-task-rl.git | \
   grep refs/heads/master | cut -f 1)

latest_master_commit_hash="master"

python_executable=${1:-"python3"}
#trainer_path=${2:-"trainer/trainer.py"}
git_commit_hash=${2:-$latest_master_commit_hash}

git checkout ${git_commit_hash}

echo "#SBATCH -p dev" > sbatch_rl
echo "#SBATCH -N 1" >> sbatch_rl
echo "#SBATCH -n 1" >> sbatch_rl
echo "#SBATCH -c 96" >> sbatch_rl
echo "#SBATCH --mem 744000" >> sbatch_rl
echo "#SBATCH -t 0-08:00" >> sbatch_rl
echo "#SBATCH -o logs/slurm.%N.%j.out" >> sbatch_rl
echo "#SBATCH -e logs/slurm.%N.%j.err" >> sbatch_rl
echo ${python_executable} "trainer/trainer.py" >> sbatch_rl

sbatch sbatch_rl
