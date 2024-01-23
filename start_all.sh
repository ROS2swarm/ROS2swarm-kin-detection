#!/bin/sh
declare -a arenas=("arena" "arena-boxes-pillars" "arena-corners" "arena-corners-pillars" "arena-large" "arena-pillars" "arena-pillars-poles" "arena-poles" "arena-walls" "arena-walls-poles" "two-rooms")
declare -a robo_types=("waffle_pi" "burger")
declare -a robo_count=("3" "4" "5" "6" "7" "8" "9")
declare -a behaviours=("dispersion" "attraction" "drive" "random-walk")
for a in "${arenas[@]}"
do
	for rt in "${robo_types[@]}"
 	do
		for nr in "${robo_count[@]}"
 		do
 			for b in "${behaviours[@]}"
   			do
 				name="${a} ${rt} ${nr} ${b}"
 				name=${name// /_}
 				sbatch -J $name -o ds_cor_log/${name}.txt arena_ds_correction.sh $a $rt $nr $b
 			done
 		done
 	done
done