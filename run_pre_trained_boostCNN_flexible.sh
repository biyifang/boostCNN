#-a	the teacher model
#--data the file where the dataset would be loaded
#--gpu which gpu this job would use
#--epochs the # of iterations in teacher-student process and each iterations in boosting precess
#--model_save the name of the student model which would be created after teacher-student process
#--teacher_model_save the name of the teacher model
#--num_boost_iter the # of iterations in boosting process
#--CNN_one kernel size of the 1st layer in CNN
#--CNN_two kernel size of the 2nd layer in CNN
#--CNN_three kernel size of the 3rd layer in CNN
#--input size the size of the input image which would be fed to CNN
#--image_pf the partial fraction of the image which would be trained in boosting procedure
python pre_trained_boostCNN_flexible.py -a resnet18 --data /home/yyv959/data_bfang --gpu 2 --epochs 100 --model_save 3_128_64test --teacher_model_save resnet18 --num_boost_iter 15 --CNN_one 2 --CNN_two 2 --CNN_three 2 --input_size 150 --image_pf 0.5