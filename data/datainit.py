#!/usr/bin/python

'''
Author:  Iskandar Pashayev
Purpose: Set up the PDF data so that we can use it with our machine learning algorithms.
	 Since there is a lot of asynchronicity (disk reads, network requests, and timer
	 sleep), we employ a thread pool to optimize our computation time.
'''

import os, sys
import time
import requests
import unirest
from PIL import Image
from multiprocessing import Pool, Lock

# Authentication for pdf2jpg API - hidden from public
import pdf2jpgauthentication as auth

kMaxChecks = 150 # no need to make this configurable
kNumWorkers = 10 # no apparent need to make this configurable either

# Link to server executable that will perform the conversion (used by process pool -
# cannot be pickled). Never changed so doesn't need a lock around it.
pdf2jpg_endpt = 'https://pdf2jpg-pdf2jpg.p.mashape.com/convert_pdf_to_jpg.php'

# A global lock that processes must acquire prior to accessing shared data
lock = Lock()

# A global counter for the total number of requests sent to check conversion status of
# request sent to API.
ctr = 0

# Empty list that will be populated with paths to files that need to be converted
file_paths = []

'''
Global function for converting the pages of a particular PDF to JPG images.
PATH is the path to the file to be converted, but excludes the .pdf extension.
'''
def _JPGify(path):

    global ctr
    
    response = unirest.post(pdf2jpg_endpt,
                            headers={
                                'X-Mashape-Key': auth.mashape_key()
                            },
                            params={
                                'pdf': open(path + '.pdf', mode='r'),
                                'resolution': 150
                            })
    
    '''
    This loop isn't optimal, but for sake of simplicity, I'm not going to set up
    a distributed database that will push an update when the 'status' key changes
    to done (at least not yet). Might do that when I start converting the entire
    PDF dataset (30,000+) to JPG though.
    '''
    while True:
        lock.acquire()
        if ctr >= kMaxChecks:
            lock.release()
            raise Exception('Reached maximum number of status checks.')
        lock.release()

        # sleep for 30 seconds to give server time to run conversion executable
        time.sleep(30)
        response = unirest.get(pdf2jpg_endpt +
                               '?id=' + response.body['id'] +
                               '&key=' + response.body['key'],
                               headers={
                                   'X-Mashape-Key': auth.mashape_key(),
                                   'Accept':'application/json'
                               })
        lock.acquire()
        ctr += 1 # this isn't any good atm, need to send back to parent process...
        lock.release()
        if response.body[u'status'] == u'done': break

    # Downloading the JPG images from server
    for i in range(1,len(response.body['pictures'])): # skip front page of manual
        jpg_url = response.body['pictures'][i]
        jpg_data = requests.get(jpg_url).content # note to self: change to unirest eventually
        with open(path + '_' + str(i) + '.jpg', 'wb') as page_i:
            page_i.write(jpg_data)


'''
Combines the individual pages from one PDF (as a JPG a file) into one large JPG.
'''
def _aggregate(path):
    permno, year = path.split('/')
    image_names = [ permno + '/' + x for x in os.listdir(permno) if year+'_' in x ]
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
    

'''
Object used to handle conversion of PDFs into JPG images usable for use in
machine learning algorithms.
'''
class DataInitializer(object):
    def __init__(self, path_to_permnos):
        self.permno_path = path_to_permnos
        self.pool = Pool(kNumWorkers) # Start kNumWorkers worker processes

        # Populate list with paths to files that need to be converted to JPG
        with open(self.permno_path, 'r') as permnos:
            for line in permnos:
                line = line.rstrip().split('\t')
                permno, name, year = line[:3]
                file_paths.append(permno + '/' + year)

    def convert_pdfs_to_jpg(self):
        print 'Beginning conversion of PDFs to JPG files...'
        self.pool.map(_JPGify, file_paths) # blocks until all processes finish
        print 'Done converting.'
        print ctr, 'requests made.'
        print 'Beginning aggregation of JPG image pages per permno year...'
        self.pool.map(_aggregate, file_paths)
        print 'Done aggregation.'

if __name__ == '__main__':
    try:
        if len(sys.argv) == 1: sys.argv.append('permnos.dat')
        data_initializer = DataInitializer(sys.argv[1])
        data_initializer.convert_pdfs_to_jpg()
    except:
        print 'Status update requests made:', ctr
        print 'Unexpected Error:', sys.exc_info()[0]
        raise
