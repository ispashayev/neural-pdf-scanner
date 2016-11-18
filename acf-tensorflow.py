'''
Author:		Iskandar Pashayev
Purpose:	Load images for ACF Industries 1956 into tensorflow
'''

'''
We need to partition the data into three sets:
1) Training set - this is a set of data
2) Test set
3) Validation set


Every data point will be a tuple of a vector of numbers representing the values of the pixels in an image and a corresponding label.
Thus, for each set we have a tensor of [# Elements in set, # Elements in vector].
'''

'''
Design decisions:
Assemble the multiple images for each pdf into one giant image

Steps:
1) Figure out where the desired company's region of data is (or if it's even there)
2) Figure out where the income statement is
3) Collect income statement data - ALL AVAILABLE YEARS (maybe by cropping? see tf.image.resize_image_with_crop_or_pad
4) Figure out where the balance sheet is
5) Collect balance sheet data - ALL AVAILABLE YEARS
'''


import tensorflow as tf

filename_queue = tf.train.string_input(producertf.train.match_filenames_once("acf-inds*.jpeg"))
image_reader = tf.WholeFileReader()
_, image_file = image_reader.read(filename_queue)

'''
Decode a JPEG image to a uint8 tensor with shape [height, width, channels]:
contents - the JPEG file, only required parameter
channels - number of color channels for the decoded image - init {0,1,3 | 0 default}
ratio - downscaling ratio - init {z^+ | 1 default}
fancy_upscaling - use a slower but nicer upscaling of the chroma planes - init {T,F | T default}
try_recover_truncated - try to recover an image from truncated input - init {T,F | F default}
acceptable_fraction - min required fraction of lines before a truncated input is accepted - init {R | 1 default}
name - name for the operation
'''
image = tf.image.decode_jpeg(image_file)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    image_tensor = sess.run([image])
    print image_tensor
    coord.join(threads)

