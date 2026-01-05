python ./train_llm.py -net facebook/opt-350m  -gpu -b 8 -e 2 -dtr -budget 20480
sleep 30 
python ./train_llm.py -net gpt2 -gpu -b 8 -e 2 -dtr -budget 13312
sleep 30 
python ./train_llm.py -net EleutherAI/pythia-160m -gpu -b 16 -e 2 -dtr -budget 18432
sleep 30
python run_cnn.py --experiment_mode dtr --model_name resnet18  --input_idx 0 --out_file output.json --epochs 2 --batch_size 256 --memory_budget 1459.63
sleep 30 
python run_cnn.py --experiment_mode dtr --model_name resnet50  --input_idx 0 --out_file output.json --epochs 2 --batch_size 256 --memory_budget 6951.73
sleep 30 
python run_cnn.py --experiment_mode dtr --model_name resnet152  --input_idx 0 --out_file output.json --epochs 2 --batch_size 256 --memory_budget 14077.63
sleep 30 
python run_cnn.py --experiment_mode dtr --model_name googlenet  --input_idx 0 --out_file output.json --epochs 2 --batch_size 256 --memory_budget 3440.37
sleep 30
python run_cnn.py --experiment_mode dtr --model_name inceptionv4  --input_idx 0 --out_file output.json --epochs 2 --batch_size 128 --memory_budget 12288
sleep 30
python run_cnn.py --experiment_mode dtr --model_name inceptionv3  --input_idx 0 --out_file output.json --epochs 2 --batch_size 256 --memory_budget 11038.16
