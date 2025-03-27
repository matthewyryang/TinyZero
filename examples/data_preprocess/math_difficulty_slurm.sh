for i in 25 26 27 28 29 30 31 33 34; do
# for i in 3 4 6 12 13 14 15 25; do
    sbatch /home/myang4/TinyZero/examples/data_preprocess/math_difficulty.sh $((i * 1024)) $(((i + 1) * 1024))
done

