
import timeit   #查看运行开始到结束所用的时间
import tensorflow as tf
import os
 
def generate_tfrecords(input_filename, output_filename):
    print("\nStart to convert {} to {}\n".format(input_filename, output_filename))
    writer = tf.python_io.TFRecordWriter(output_filename)
 
    for line in open(input_filename, "r"):
        line = line.strip()
        line = line.replace(" ", "")
        data = line.split(",")
        #print(data)
        
        if len(data) != 17:
            print("data len is not 17")
        
        label = int(data[16])
        
        features = [float(i) for i in data[:16]]    #特征不要最后一列数据
        
        if len(features) != 16:
            print("features len is not 16")

        #将数据转化为原生 bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label":
                tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "0":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[0]])),
            "1":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[1]])),
            "2":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[2]])),
            "3":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[3]])),
            "4":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[4]])),
            "5":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[5]])),
            "6":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[6]])),
            "7":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[7]])),
            "8":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[8]])),
            "9":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[9]])),
            "10":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[10]])),
            "11":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[11]])),
            "12":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[12]])),
            "13":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[13]])),
            "14":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[14]])),
            "15":
                tf.train.Feature(float_list=tf.train.FloatList(value=[features[15]])),
        }))
        writer.write(example.SerializeToString())  #序列化为字符串
 
    writer.close()
 

def main():
    generate_tfrecords("./training-data.csv", "./train.tfrecords")
 
 
if __name__ == "__main__":
    main()
