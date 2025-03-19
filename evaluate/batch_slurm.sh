
for context_length in 512 1024 2048; do
    for checkpoint in 0 50 100 150 200 250 300; do
        sbatch /home/myang4/TinyZero/evaluate/slurm.sh $context_length $checkpoint
    done
done


# for context_length in 2048; do
#     for checkpoint in 200 250 300 350 400 450; do
#         sbatch /home/myang4/TinyZero/evaluate/slurm.sh $context_length $checkpoint
#     done
# done

# for checkpoint in 0 50 100 150 200 250 300 350 400 450; do
#     sbatch /home/myang4/TinyZero/evaluate/slurm.sh 1024 $checkpoint
# done