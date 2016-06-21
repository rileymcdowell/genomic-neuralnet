#!/bin/bash
workon genomic_sel

# Array of species.
declare -a SPECIES=(arabidopsis wheat pig maize loblolly)

# Loop over every optimization function for every species and trait.
# This will populate csv files which store the output of the optimizations.
for file in $( ls optimize* ) ; do
    for species in ${SPECIES[@]} ; do
        for trait in $(python $file --species $species --list ) ; do
            python $f --species $species --trait $trait
        done
    done
done

