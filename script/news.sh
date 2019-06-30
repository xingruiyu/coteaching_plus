for seed in 1 2 3 4 5 
do
  for model_type in coteaching_plus 
  do
    python main.py --dataset news --model_type ${model_type} --noise_type symmetric --noise_rate 0.2 --seed ${seed} --result_dir results/trial_${seed}/
    python main.py --dataset news --model_type ${model_type} --noise_type symmetric --noise_rate 0.5 --seed ${seed} --result_dir results/trial_${seed}/
    python main.py --dataset news --model_type ${model_type} --noise_type pairflip --noise_rate 0.45 --seed ${seed} --result_dir results/trial_${seed}/
  done
done
