
#TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
#TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
#TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

TF_LIB='/usr/local/anaconda3/lib/python3.6/site-packages/tensorflow'
TF_CFLAGS='-I/home/liulu112/liulu-common/abseil-cpp -I/usr/local/anaconda3/lib/python3.6/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0'
TF_LFAGS='-L/usr/local/anaconda3/lib/python3.6/site-packages/tensorflow -ltensorflow_framework'
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -o word2vec_ops.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -L $TF_LIB -ltensorflow_framework

