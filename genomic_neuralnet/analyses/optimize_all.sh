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
# This will populate shelf database files which store the output of the optimizations.
#for file in $( ls optimize*nn.py ) ; do
for file in $( ls | grep 'optimize' | grep 'py' | grep -v 'nn' ) ; do
    for species in ${SPECIES[@]} ; do
        for trait in $(python $file --species $species --list ) ; do
            echo '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'
            echo "INFO: About to train ${file} - ${species} - ${trait}"
            echo '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'

            echo $file | grep 'nn.py' > /dev/null
            if [ $? -eq 0 ] ; then
                # Train neural nets on GPU.
                #sleep 10 # Give the GPU time to release memory.
                #python -u $file --species $species --trait $trait --gpu
                #sleep 10 # Give the GPU time to release memory.
                # Then re-train on CPU to compare times.
                python -u $file --species $species --trait $trait --reuse-celery-cache #--force
            else 
                # Train others in normal mode.
                python -u $file --species $species --trait $trait
            fi

            echo '####################################################'
            echo "Training File Completed. Time is $(date)"
            echo '####################################################'
        done
    done
done

