for i in {0..20}; do
    sbatch /home/myang4/TinyZero/examples/data_preprocess/math_difficulty.sh $((i * 2048)) $(((i + 1) * 2048))
done

