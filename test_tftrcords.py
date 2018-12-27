#coding=utf-8  
#tensorflow tf records 
import tensorflow as tf  
import cv2  
  
# 把train.txt文件格式，每一行：图片路径名   类别标签  
# 将数据打包，转换成tfrecords格式，以便后续高效读取  
def encode_to_tfrecords(lable_file,data_root,new_name='data.tfrecords',resize=None):  
    writer=tf.python_io.TFRecordWriter(data_root+'/'+new_name)  
    num_example=0  
    with open(lable_file,'r') as f:  
        for l in f.readlines():  
            l=l.split()  
            image=cv2.imread(data_root+"/"+l[0])  
            if resize is not None:  
                image=cv2.resize(image,resize)#为了  
            height,width,nchannel=image.shape  
  
            label=int(l[1])  
  
            example=tf.train.Example(features=tf.train.Features(feature={  
                'height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),  
                'width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),  
                'nchannel':tf.train.Feature(int64_list=tf.train.Int64List(value=[nchannel])),  
                'image':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),  
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  
            }))  
            serialized=example.SerializeToString()  
            writer.write(serialized)  
            num_example+=1  
    print lable_file,"样本数据量：",num_example  
    writer.close()  
#读取tfrecords文件  

def decode_from_tfrecords(filename,num_epoch=None):  
    filename_queue=tf.train.string_input_producer([filename],num_epochs=num_epoch)#因为有的训练数据过于庞大，被分成了很多个文件，所以第一个参数就是文件列表名参数  
    reader=tf.TFRecordReader()  
    _,serialized=reader.read(filename_queue)  
    example=tf.parse_single_example(serialized,features={  
        'height':tf.FixedLenFeature([],tf.int64),  
        'width':tf.FixedLenFeature([],tf.int64),  
        'nchannel':tf.FixedLenFeature([],tf.int64),  
        'image':tf.FixedLenFeature([],tf.string),  
        'label':tf.FixedLenFeature([],tf.int64)  
    })  
    label=tf.cast(example['label'], tf.int32)  
    image=tf.decode_raw(example['image'],tf.uint8)  
    image=tf.reshape(image,tf.pack([  
        tf.cast(example['height'], tf.int32),  
        tf.cast(example['width'], tf.int32),  
        tf.cast(example['nchannel'], tf.int32)]))  
    #label=example['label']  
    return image,label  
#根据队列流数据格式，解压出一张图片后，输入一张图片，对其做预处理、及样本随机扩充

def get_batch(image, label, batch_size,crop_size):  
        #数据扩充变换  
    distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])#随机裁剪  
    distorted_image = tf.image.random_flip_up_down(distorted_image)#上下随机翻转  
    #distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)#亮度变化  
    #distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化  
  
    #生成batch  
    #shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大  
    #保证数据打的足够乱  
    images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,  
                                                 num_threads=16,capacity=50000,min_after_dequeue=10000)  
    #images, label_batch=tf.train.batch([distorted_image, label],batch_size=batch_size)  
  
  
  
    # 调试显示  
    #tf.image_summary('images', images)  
    return images, tf.reshape(label_batch, [batch_size]) 

#这个是用于测试阶段，使用的get_batch函数  
def get_test_batch(image, label, batch_size,crop_size):  
        #数据扩充变换  
    distorted_image=tf.image.central_crop(image,39./45.)  
    distorted_image = tf.random_crop(distorted_image, [crop_size, crop_size, 3])#随机裁剪  
    images, label_batch=tf.train.batch([distorted_image, label],batch_size=batch_size)  
    return images, tf.reshape(label_batch, [batch_size])  
#测试上面的压缩、解压代码  

def test():  
    encode_to_tfrecords("data/train.txt","data",(100,100))  
    image,label=decode_from_tfrecords('data/data.tfrecords')  
    batch_image,batch_label=get_batch(image,label,3)#batch 生成测试  
    init=tf.initialize_all_variables()  
    with tf.Session() as session:  
        session.run(init)  
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(coord=coord)  
        for l in range(100000):#每run一次，就会指向下一个样本，一直循环  
            #image_np,label_np=session.run([image,label])#每调用run一次，那么  
            '''''cv2.imshow("temp",image_np) 
            cv2.waitKey()'''  
            #print label_np  
            #print image_np.shape  
  
  
            batch_image_np,batch_label_np=session.run([batch_image,batch_label])  
            print batch_image_np.shape  
            print batch_label_np.shape  
  
  
  
        coord.request_stop()#queue需要关闭，否则报错  
        coord.join(threads)  
#test()  
