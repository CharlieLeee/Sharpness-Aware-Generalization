python torch_official_train.py --batch_size 4 --seed 20 --epochs 100 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'normalize' --lr 0.01 &
python torch_official_train.py --batch_size 8 --seed 20 --epochs 100 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'normalize' --lr 0.01 
python torch_official_train.py --batch_size 16 --seed 20 --epochs 100 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'normalize' --lr 0.01 &
python torch_official_train.py --batch_size 32 --seed 20 --epochs 100 --baseoptim 'sgd' --secoptim 'sam' --model 'resnet18' --norm_type 'normalize' --lr 0.01



python torch_official_train.py --batch_size 4 --seed 20 --epochs 100 --baseoptim 'sgd' --secoptim 'none' --model 'resnet18' --norm_type 'normalize' --lr 0.01 &
python torch_official_train.py --batch_size 8 --seed 20 --epochs 100 --baseoptim 'sgd' --secoptim 'none' --model 'resnet18' --norm_type 'normalize' --lr 0.01 
python torch_official_train.py --batch_size 16 --seed 20 --epochs 100 --baseoptim 'sgd' --secoptim 'none' --model 'resnet18' --norm_type 'normalize' --lr 0.01 &
python torch_official_train.py --batch_size 32 --seed 20 --epochs 100 --baseoptim 'sgd' --secoptim 'none' --model 'resnet18' --norm_type 'normalize' --lr 0.01



