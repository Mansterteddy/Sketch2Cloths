#Training
python pix2pix.py --mode train --output_dir ./ --max_epochs 200 --input_dir ./train --which_direction AtoB

#Testing
python pix2pix.py --mode test --output_dir ./ --input_dir ./val/ --checkpoint train