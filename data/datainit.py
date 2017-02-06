#!/usr/bin/python

'''
Author:  Iskandar Pashayev
Purpose: Set up the PDF data so that we can use it with our machine learning algorithms.
	 Since there is a lot of asynchronicity (disk reads, network requests, and timer
	 sleep), we employ a thread pool to optimize our computation time.
'''


import os
import time
import unirest
from multiprocessing import Pool, Lock

# Authentication for pdf2jpg API - hidden from public
import pdf2jpgauthentication as auth

kNumWorkers = 10
kMaxChecks = 150

class DataInitializer(object):
    def __init__(self, path_to_permnos):
        self.permno_path = path_to_permnos
        self.pdf2jpg_endpt = 'https://pdf2jpg-pdf2jpg.p.mashape.com/convert_pdf_to_jpg.php'
        self.num_requests = 0
        self.pool = Pool(kNumWorkers)
        self.lock = Lock()
        self.paths = []
        with open(self.permno_path, 'r') as permnos:
            for line in permnos:
                line = line.rstrip().split('\t')
                permno, name, year = datum[:3]
                self.paths.append(permno + '/' + year)

    def convert_pdfs_to_jpg(self):
        self.pool.map(self._JPGify, self.paths)
        self.pool.map(self._aggregate, self.paths)
                
    def _JPGify(self, path)
                response = unirest.post(self.pdf2jpg_endpt,
                                        headers={
                                            'X-Mashape-Key': auth.mashape_key()
                                        },
                                        params={
                                            'pdf': open(path + '.pdf', mode='r'),
                                            'resolution': 150
                                        })
                response['status'] = 'not done'

                '''
                This loop isn't optimal, but for sake of simplicity, I'm not going to set up
                a distributed database that will push an update when the 'status' key changes
                to done (at least not yet).
                '''
                while response['status'] != 'done':
                    lock.acquire()
                    if self.num_requests >= kMaxChecks:
                        lock.release()
                        raise Exception('Reached maximum number of conversion status checks.')
                    lock.release()
                    time.sleep(30) # sleep for 30 seconds to give server time to run executable
                    response = unirest.get(self.pdf2jpg_endpt +
                                           '?id=' + response['id'] +
                                           '&key=' + response['key'],
                                           headers={
                                               'X-Mashape-Key': auth.mashape_key(),
                                               'Accept':'application/json'
                                           })
                    lock.acquire()
                    self.num_requests += 1
                    lock.release()
                    
                for i in range(1,len(response['pictures'])): # skip front page of manual
                    jpg_url = response['pictures'][i]
                    with open(path + '_' + str(i) + '.jpg', 'wb') as page_i:
                        jpg_data = requests.get(jpg_url).content
                        page_i.write(jpg_data)
    
    def _aggregate(self, path):
        permno, year = path.split('/')
        image_names = [ x for x in os.listdir(permno) if year+'_' in x ]
        image_names.sort()
        images = map(Image.open, image_names)
        widths, heights  = zip(*(img.size for img in images))
        aggregate_width, aggregate_height = max(widths), sum(heights)
        aggregated_img = Image.new('RGB', (aggregate_width, aggregate_height))
        height_offset = 0
        for img in images:
            aggregated_img.paste(img, (0, height_offset))
            height_offset += img.size[1]
        aggregated_img.save(path+'.jpg')
        
        
                

if __name__ == '__main__':
    data_initializer = DataInitializer()
    data_initializer.convert_pdfs_to_jpg()
