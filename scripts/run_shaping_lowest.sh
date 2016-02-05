#!/bin/sh
# Submit the job to all.q
#$ -clear
#$ -q all.q
# use the current working directory for input and output
#$ -cwd
# export the current $PATH variable for this job
#$ -V
#$ -S /bin/bash
# Set a custom job name
#$ -N alejob
#$ -l virtual_free=1G
# stdout and stderr output
#$ -e $HOME/Logs/alejob.err
#$ -o $HOME/Logs/alejob.log
#separate threads for processes
#$ -pe mt 4
#
#set various paths
RLDIR=$HOME/git/PyALE/src
ALEDIR=$HOME/ALE
LOGDIR=$HOME/Logs

#Experimental configuration
EXP_NAME="sarsa-shaping-lowest"
EXPERIMENT="generic_experiment.py"
EXPERIMENT_OPTIONS="--maxsteps 2000 --numeps 3000 --numtrials 5"
AGENT="agents/ALESarsaShapingAgent.py"
#AGENT_OPTIONS='--eps 0.05 --lambda_ 0.5 --alpha 0.1 --features RAM --actions 0 1 3 4  --trajectory_length 100 --replays 10 --db_size 1000'
AGENT_OPTIONS='--eps 0.05 --lambda_ 0.5 --alpha 0.1 --potential lowest_enemy --actions 0 1 3 4'
ALE_OPTIONS="-game_controller rlglue -repeat_action_probability 0.0 -frame_skip 30"
GAME="space_invaders.bin"

#######Imports#################
#source $HOME/.bash_profile
#module add java
export PYTHONPATH=$RLDIR:$PYTHONPATH
export RLGLUE_PORT=1028
###############################

mkdir -p "$LOGDIR/$EXP_NAME"

cd $RLDIR
#start rlglue
rl_glue &
#run agent
python $AGENT $AGENT_OPTIONS --savepath "$LOGDIR/$EXP_NAME" > $LOGDIR/agent-$EXP_NAME.log 2>&1 &
#run experiment
cd exp
python $EXPERIMENT $EXPERIMENT_OPTIONS  >> $LOGDIR/exp-$EXP_NAME.log 2>&1 &
#run environment (no '&' or job quits!)
cd $ALEDIR
./ale $ALE_OPTIONS roms/$GAME > $LOGDIR/ale-$EXP_NAME.log 2>&1
