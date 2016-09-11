#!/bin/bash
source $(which virtualenvwrapper.sh)
workon genomic_sel

# Array of species.
#declare -a SPECIES=(arabidopsis wheat pig maize loblolly)

# Array of species.
declare -a SPECIES=(arabidopsis wheat maize) # Start easy.

echo '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
echo "Beginning Optimization. Time is: $(date)"
echo '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'

# Loop over every optimization function for every species and trait.
for file in $( ls optimize_mlp_nn.py ) ; do
    for species in ${SPECIES[@]} ; do
        for trait in $(python $file --species $species --list ) ; do
            echo '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
            echo "INFO: About to train ${file} - ${species} - ${trait}"
            echo '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'

            echo $file | grep 'nn.py' > /dev/null
            if [ $? -eq 0 ] ; then
                # Train neural nets on GPU.
                sleep 10 # Give the GPU time to release memory.
                echo 'Training on GPU - large'
                python $file --species $species --trait $trait \
                    --dryrun --time-stats --gpu --timing-size large \
                    &> timing_logs/${species}_${trait}_large_gpu.log
                sleep 10
                echo 'Training on GPU - small'
                python $file --species $species --trait $trait \
                    --dryrun --time-stats --gpu --timing-size small \
                    &> timing_logs/${species}_${trait}_small_gpu.log
                sleep 10 # Give the GPU time to release memory.
                # Then re-train on CPU to compare times.
                echo 'Training on CPU - large'
                python $file --species $species --trait $trait \
                    --dryrun --time-stats --timing-size large \
                    &> timing_logs/${species}_${trait}_large_cpu.log
                echo 'Training on CPU - small'
                python $file --species $species --trait $trait \
                    --dryrun --time-stats --timing-size small \
                    &> timing_logs/${species}_${trait}_small_cpu.log
            fi

            echo '####################################################'
            echo "Training File Completed. Time is $(date)"
            echo '####################################################'
        done
    done
done

