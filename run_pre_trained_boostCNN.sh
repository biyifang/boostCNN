#-a	the teacher model
#--data the file where the dataset would be loaded
#--gpu which gpu this job would use
#--epochs the # of iterations in teacher-student process and each iterations in boosting precess
#--model_save the name of the student model which would be created after teacher-student process
#--teacher_model_save the name of the teacher model
#--num_boost_iter the # of iterations in boosting process
python pre_trained_boostCNN.py -a resnet18 --data /home/yyv959/data_bfang --gpu 2 --epochs 100 --model_save 3_128_64test --teacher_model_save resnet18 --num_boost_iter 15
