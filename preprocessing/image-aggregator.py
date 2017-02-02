'''
Author: Iskandar Pashayev
Purpose: Combine the broken up jpeg images for each page of the moody's manual
company listing into 1 image. Implemented with multi threading for efficiency.
'''

import os
from PIL import Image
from threading import Thread, Semaphore

def aggregate_imgs(permno, name):
    cwd_path = os.getcwd()
    cwd_contents = os.listdir(cwd_path)
    if permno not in cwd_contents:
        print 'Directory not found for', permno, name
        return
    permno_dir = cwd_path + '/' + permno
    os.chdir(permno_dir)
    image_names = os.listdir(permno_dir)
    image_names.sort()
    images = map(Image.open, image_names)
    widths, heights = zip(*(img.size for img in images))
    aggregate_width, aggregate_height = max(widths), sum(heights)
    aggregated_img = Image.new('RGB', (aggregate_width, aggregate_height))
    height_offset = 0
    for img in images:
        aggregated_img.paste(img, (0, height_offset))
        height_offset += img.size[1]
    aggregated_img.save(name+'.jpeg')
    semaphore.release()


if __name__ == '__main__':
    cwd_path = os.getcwd()
    cwd_contents = os.listdir(cwd_path)
    imgs = [ name for name in cwd_contents if '.jpeg' in name ]
    threads = []
    semaphore = Semaphore(10) # max 10 threads running at a time
    for line in open('permnos.dat','r'):
        permno, name = line.rstrip().split('\t')
        thread = Thread(target = aggregate_imgs, args=(permno,name))
        threads.append(thread)
        semaphore.acquire()
        thread.start()
    for thread in threads:
        thread.join()
