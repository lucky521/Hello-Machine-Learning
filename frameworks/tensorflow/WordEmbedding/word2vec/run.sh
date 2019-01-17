
# https://github.com/tensorflow/models/tree/master/tutorials/embedding

# if Tensorflow Failed to create Session
export CUDA_VISIBLE_DEVICES=''

python word2vec_optimized.py --train_data=text8  --eval_data=questions-words.txt  --save_path=/tmp/


