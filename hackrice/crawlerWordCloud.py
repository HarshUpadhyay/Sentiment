﻿import os, urllib2, unirest, PILfrom PIL import Image;from os import path;import numpy as npimport matplotlib.pyplot as pltfrom wordcloud import WordCloud, STOPWORDSd = path.dirname(__file__)# Read the whole text.textPros = open(path.join(d, 'generated data/pros_full.txt')).read()textCons = open(path.join(d, 'generated data/cons_full.txt')).read()# read the mask image# taken from http://www.stencilry.org/stencils/movies/alice%20in%20wonderland/255fk.jpggoogle_mask = np.array(Image.open(path.join(d, "google-logo.jpg")))print STOPWORDSwc = WordCloud(background_color="white", max_words=30000, mask=None,               stopwords=STOPWORDS.add("</p>"))# generate word cloudwc.generate(textPros)# store to filewc.to_file(path.join(d, "google-logo.jpg"))wc.generate(textCons)wc.to_file(path.join(d, "google-logo1.jpg"))# showplt.imshow(wc)plt.axis("off")plt.figure()plt.imshow(google_mask, cmap=plt.cm.gray)plt.axis("off")plt.show()