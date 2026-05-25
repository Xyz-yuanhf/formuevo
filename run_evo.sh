#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1                      # How many nodes required? Usually 1
#SBATCH --cpus-per-task=18             # Number of CPU to request for the job
#SBATCH --ntasks=1                     # Number of time repeated
#SBATCH --mem=96GB                     # How much memory does your job require?
#SBATCH --nodelist=<your-node-list>    # Optional: set to a valid node list for your cluster, or remove this line
#SBATCH --time=04-23:59:59             # How long to run the job for? Jobs exceed this time will be terminated
#SBATCH --gres=gpu:0				   # Do not need GPU
                                       # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                       # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=END,FAIL           # When should you receive an email?
#SBATCH --output=logs/%u.%j.out        # Where should the log files go?
                                       # You must provide an absolute path eg /common/home/module/username/
                                       # If no paths are provided, the output file will be placed in your current working directory

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=<your-partition>      # Replace with a valid partition on your cluster
#SBATCH --account=<your-account>          # Replace with your project or billing account
#SBATCH --qos=<your-qos>                  # Replace with a valid QOS for your cluster
#SBATCH --mail-user=<your-email>          # Replace with your email address
#SBATCH --job-name=FormuEvo               # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.scis.dev/docs/Advanced%20settings/module for more information
module purge
module load Anaconda3/2024.06-1

# Do not remove this line even if you have executed conda init
eval "$(conda shell.bash hook)"

# This command assumes that you've already created the environment previously
# We're using an absolute path here. You may use a relative path, as long as SRUN is execute in the same working directory
conda activate llm

# If you require any packages, install it as usual before the srun job submission.
# Submit your job to the cluster
srun python main.py
