python run_cnn.py --experiment_mode dtr --model_name resnet18  --input_idx 0 --out_file output.csv --epochs 2 --batch_size 256 --memory_budget 800 
sleep 30
python run_cnn.py --experiment_mode dtr --model_name inceptionv4  --input_idx 0 --out_file output.csv --epochs 2 --batch_size 128 --memory_budget 8000 
sleep 30
python run_cnn.py --experiment_mode dtr --model_name googlenet  --input_idx 0 --out_file output.csv --epochs 2 --batch_size 256 --memory_budget 1500 