'''
Author:		Iskandar Pashayev
Date created:	10/13/16
Purpose:	Transform the image content of moody's manual pdf scans to jpg

---------------------------------------------------------------------------------
	Change Log
---------------------------------------------------------------------------------
1) 	<Date>
	<Description>
2)
---------------------------------------------------------------------------------
'''
'''
import sys

scan = open('acf-inds-1956.pdf','rb')
scan = scan.readline()

print scan.encode('hex')
'''

# TRY CONTACTING MERGENT TO SEE IF YOU CAN GET THE ORIGINAL IMAGE UPLOADS (PNG IM GUESSING)

import sys

pdf = file("acf-inds-1956.pdf", "rb").read()

startmark = "\xff\xd8"
startfix = 0
endmark = "\xff\xd9"
endfix = 2
i = 0

njpg = 0
while True:

    # Block confirmed good
    istream = pdf.find("stream", i) # i is the beginning of the pdf string
    if istream < 0: # stream keyword not found
        break
    istart = pdf.find(startmark, istream, istream+20) # parsing first 20 bytes of istream?
    print pdf[istream+6:istream+26].encode('hex') # debugging - looking at the hex encoded bytes close to the string keyword
    if istart < 0:
        i = istream+20
        continue
    iend = pdf.find("endstream", istart)
    if iend < 0:
        raise Exception("Didn't find end of stream!")
    iend = pdf.find(endmark, iend-20)
    if iend < 0:
        raise Exception("Didn't find end of JPG!")
    istart += startfix
    iend += endfix
    print "JPG %d from %d to %d" % (njpg, istart, iend)
    jpg = pdf[istart:iend]
    jpgfile = file("jpg%d.jpg" % njpg, "wb")
    jpgfile.write(jpg)
    jpgfile.close()

    njpg += 1
    i = iend
    
