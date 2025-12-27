python run_cnn.py --experiment_mode dtr --model_name resnet18  --input_idx 0 --out_file output.json --epochs 2 --batch_size 256 --memory_budget 814.51
sleep 30 
python run_cnn.py --experiment_mode dtr --model_name resnet50  --input_idx 0 --out_file output.json --epochs 2 --batch_size 256 --memory_budget 3265.33
sleep 30 
python run_cnn.py --experiment_mode dtr --model_name resnet152  --input_idx 0 --out_file output.json --epochs 2 --batch_size 256 --memory_budget 7626.43
sleep 30 
python run_cnn.py --experiment_mode dtr --model_name googlenet  --input_idx 0 --out_file output.json --epochs 2 --batch_size 256 --memory_budget 1597.17
sleep 30
python run_cnn.py --experiment_mode dtr --model_name inceptionv3  --input_idx 0 --out_file output.json --epochs 2 --batch_size 256 --memory_budget 5303.76
sleep 30
python run_cnn.py --experiment_mode dtr --model_name inceptionv4  --input_idx 0 --out_file output.json --epochs 2 --batch_size 128 --memory_budget 8521.25
