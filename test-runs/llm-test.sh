# python ./train_llm.py -net facebook/opt-350m  -gpu -b 8 -e 2 -dtr -budget 10719.2
# sleep 30 
# python ./train_llm.py -net gpt2 -gpu -b 8 -e 2 -dtr -budget 7119.35 
# sleep 30 
python ./train_llm.py -net EleutherAI/pythia-160m -gpu -b 16 -e 2 -dtr -budget 11000
sleep 30
python run_cnn.py --experiment_mode dtr --model_name inceptionv3  --input_idx 0 --out_file output.json --epochs 2 --batch_size 256 --memory_budget 6500