python torch_official_train.py --batch_size 4 --seed 2 --epochs 300 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'normalize' &
python torch_official_train.py --batch_size 8 --seed 2 --epochs 300 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'normalize' 
python torch_official_train.py --batch_size 16 --seed 2 --epochs 300 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'normalize' &
python torch_official_train.py --batch_size 32 --seed 2 --epochs 300 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'normalize' 



python torch_official_train.py --batch_size 4 --seed 2 --epochs 300 --baseoptim 'sgd' --secoptim 'none' --model 'resnet18' --norm_type 'normalize' &
python torch_official_train.py --batch_size 8 --seed 2 --epochs 300 --baseoptim 'sgd' --secoptim 'none' --model 'resnet18' --norm_type 'normalize' 
python torch_official_train.py --batch_size 16 --seed 2 --epochs 300 --baseoptim 'sgd' --secoptim 'none' --model 'resnet18' --norm_type 'normalize' &
python torch_official_train.py --batch_size 32 --seed 2 --epochs 300 --baseoptim 'sgd' --secoptim 'none' --model 'resnet18' --norm_type 'normalize' 



