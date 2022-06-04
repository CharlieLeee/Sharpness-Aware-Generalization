python torch_official_train.py --batch_size 4 --seed 4 --epochs 300 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'normalize' &
python torch_official_train.py --batch_size 8 --seed 4 --epochs 300 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'normalize' 
python torch_official_train.py --batch_size 16 --seed 4 --epochs 300 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'normalize' &
python torch_official_train.py --batch_size 32 --seed 4 --epochs 300 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'normalize' 

python torch_official_train.py --batch_size 4 --seed 4 --epochs 300 --baseoptim 'sgd' --secoptim 'none' --model 'resnet18' --norm_type 'normalize' &
python torch_official_train.py --batch_size 8 --seed 4 --epochs 300 --baseoptim 'sgd' --secoptim 'none' --model 'resnet18' --norm_type 'normalize' 
python torch_official_train.py --batch_size 16 --seed 4 --epochs 300 --baseoptim 'sgd' --secoptim 'none' --model 'resnet18' --norm_type 'normalize' &
python torch_official_train.py --batch_size 32 --seed 4 --epochs 300 --baseoptim 'sgd' --secoptim 'none' --model 'resnet18' --norm_type 'normalize' 
