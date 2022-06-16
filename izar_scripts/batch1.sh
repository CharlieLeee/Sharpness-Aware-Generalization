python torch_official_train.py --batch_size 16 --seed 30 --epochs 100 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'normalize' &
python torch_official_train.py --batch_size 64 --seed 30 --epochs 100 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'normalize' 
python torch_official_train.py --batch_size 128 --seed 30 --epochs 100 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'normalize' &

python torch_official_train.py --batch_size 16 --seed 30 --epochs 100 --baseoptim 'sgd' --secoptim 'none' --model 'resnet18' --norm_type 'normalize' &
python torch_official_train.py --batch_size 64 --seed 30 --epochs 100 --baseoptim 'sgd' --secoptim 'none' --model 'resnet18' --norm_type 'normalize' 
python torch_official_train.py --batch_size 128 --seed 30 --epochs 100 --baseoptim 'sgd' --secoptim 'none' --model 'resnet18' --norm_type 'normalize' &
