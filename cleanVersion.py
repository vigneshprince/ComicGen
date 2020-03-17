import bs4
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import time
from PIL import Image, ImageDraw, ImageFont
import textwrap
import re
import nltk
from collections import Iterable
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tag import pos_tag 
from nameparser.parser import HumanName
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import nltk
from google_images_download import google_images_download 
from cartoonizer import cartoonize
import imutils
import shutil
import time
import pickle
import pandas as pd 
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
stop_words = stopwords.words('english')
#import geograpy3
from random import randint
import sys
import numpy as np
from skin import extractSkin
from skin import extractDominantColor
from skin import plotColorBar
from Genage import genage
import markovify
import operator
import os
import cv2
from TitleGenerator import tokenize ,printMatrix,dot,magnitude,similarityMatrix
from TitleGenerator import textRank ,getTitle ,cleanText,averageSentenceLength 
from face_detection import face_detection
from face_points_detection import face_points_detection
from face_swap import warp_image_2d, warp_image_3d, mask_from_points, apply_mask, correct_colours, transformation_from_points
import sys
import nltk
import string
import copy
import math
import random
from newsdatascraper import Scraper
from datetime import date
from datetime import timedelta
import csv
from collections import defaultdict
from bing import bingimgs
from shutil import copyfile

""" open('names.txt', 'w').close()
open('places.txt', 'w').close() """
body=""
dummy=[]
dummy1=[]
def scrap1():
    """
    today=date.today()
    week= today - timedelta(days=7)


    new_scraper = Scraper('f842a58309204957bec6bbe3273a2d32', mode = 'NEWSPAPER')
    articles = new_scraper.fetch_articles_from_specific_dates(query='trump india', date_from = week ,date_to=today, page_size = 5)
    articles.to_csv('test.csv')
    """

    columns = defaultdict(list) 

    with open('test.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for (k,v) in row.items():
                columns[k].append(v)


    str1 = '\n'.join(columns['content'])
    return str1

def scrap():
    my_url = 'https://www.inshorts.com/en/read'
    uClient = uReq(my_url)

    page_html = uClient.read()
    uClient.close()
    page_soup = soup(page_html,"html.parser")
    heading=[]
    data=[]
    containers = page_soup.findAll("div",{"class":"news-card-title news-right-box"})
    contentout = page_soup.findAll("div",{"class":"news-card-content news-right-box"})
    for container in containers:
        heading.append(container.span.text)
    for content1 in contentout:
        content = content1.find("div",{"itemprop":"articleBody"})
        data.append(content.text)
    res = zip(heading , data)
    printnews(res)

def sentiment(text):
    sia=SIA()
    pol_score=sia.polarity_scores(text)
    return pol_score


def printnews(blocks):
    for h , c in blocks:
        print(h + ':' +'\n'*2 + c +'\n'*2)
    nameandlocation(c)

person_list = []
person_names=person_list

def nameandlocation(text):
    java_path = "C:/Program Files/Java/jdk1.8.0_211/bin/java.exe"
    os.environ['JAVAHOME'] = java_path
    st= StanfordNERTagger('stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
                        'stanford-ner/stanford-ner.jar',
                        encoding='utf-8')

    
    sentence = text.split()
    taggedSentence = st.tag(sentence)
    
    test = set() 
    test1= set()
    
    
    for element in range(len(taggedSentence)):
        flag=0
        a = ''
        loc=''
        if element < len(taggedSentence):
            while taggedSentence[element][1] == 'PERSON':
                a += taggedSentence[element][0]+" "
                taggedSentence.pop(element)
            while taggedSentence[element][1] == 'LOCATION':
                loc += taggedSentence[element][0]+" "
                taggedSentence.pop(element)

        if str(a) != '':      
            test.add(a.strip())
        if str(loc) != '':      
            test1.add(loc.strip())

    #places = geograpy3.get_place_context(text = text)
    #print(places.regions)
    
    sett=list(test)
    sett.sort(key=lambda x: len(x.split()), reverse=True)
    lol=[]
    for i in range(len(sett)-1):
        for j in range(i+1,len(sett)):
            if (sett[i].find(sett[j]) != -1): 
                lol.append(sett[j])
    final_names = [x for x in sett if x not in lol]  
    print(final_names)
    with open("names.txt", "a") as myfile:
        for listitem in final_names:
            
            myfile.write('%s\n' % "".join(listitem.split()))
    print(test1)
    with open("places.txt", "a") as myfile:
        myfile.write(test1.pop())
        myfile.write('\n')
    print()


def select_face(im, r=10):
    faces = face_detection(im)

    if len(faces) == 0:
        print('Detect 0 Face !!!')
        exit(-1)

    if len(faces) == 1:
        bbox = faces[0]
    

    points = np.asarray(face_points_detection(im, bbox))
    
    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)
    
    x, y = max(0, left-r), max(0, top-r)
    w, h = min(right+r, im_h)-x, min(bottom+r, im_w)-y

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y+h, x:x+w]

"""
def cartooniser(imagename):
    
    image = cv2.imread(imagename)
    output = cartoonize(image)
    idx = imagename.index(".")
    my_str = imagename[:idx] + "_cartoon" + imagename[idx:]
    cv2.imwrite(my_str, output)
"""
def faceextract(image_path,flag):
    CASCADE="Face_cascade.xml"
    FACE_CASCADE=cv2.CascadeClassifier(CASCADE)
    image=cv2.imread(image_path)
    image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)
    for x,y,w,h in faces:
        sub_img=image[y-10:y+h+10,x-10:x+w+10]
        cv2.imwrite(str("facealone")+str(flag)+".jpg",sub_img)

def skincolor(imagepath):
    image = cv2.imread(imagepath)
    image = imutils.resize(image, width=250)
    skin = extractSkin(image)
    dominantColors = extractDominantColor(skin, hasThresholding=True)
    colours,avgg = plotColorBar(dominantColors)
    return colours,avgg
    
"""
def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)
"""

def bodyselector(imgpath):
    """ faceextract(imgpath,0)
    imgpath="facealone0.jpg" """
    age,gender=genage(imgpath)
    dummy,race=skincolor(imgpath)
    print(age,gender,race)
    if(gender=="F"):
        if(age<50 and race=="sienna"):
            body="black40f.jpg"
        else:
            body="white40f.jpg"

    if(gender=="M"):
        if(age<50 and race=="dimgray"):
            body="black40m.jpg"
        elif(age<60):
            body="white50m.jpg"
        elif(age<70):
            body="white60m.jpg"
    return body
"""
def concater():
    im1 = cv2.imread('facealone.jpg')
    im2 = cv2.imread(body)
    imlist=[im1,im2]
    im_v_resize = vconcat_resize_min([im1, im2])
    cv2.imwrite('cartoonbody.jpg', im_v_resize)
"""

def quotedtext(text):
    text1=text.splitlines()
    final=[]
    for t in text1:
        print(t)
        talk = re.findall(r'\"([^\"]+?)(\"|\-\-\n)',t)
        lst2 = [item[0] for item in talk]
        if(lst2):
            final.append(lst2)
    return final

def resizer(imgpath):
    size=640,480
    img = Image.open(imgpath)
    img = img.resize(size, Image.ANTIALIAS)
    img.save(imgpath, 'PNG')
    return imgpath
    
def titlegen(text,temp):
    article = cleanText(text)

    words = dict(list(enumerate(list(set(tokenize(article))))))
    inverse_words = {y:x for x,y in words.items()}


    word_counts = copy.deepcopy(inverse_words)
    word_counts = dict.fromkeys(word_counts, 0)

 
    sentences = dict(list(enumerate(nltk.sent_tokenize(article))))
    for i in sentences:
        sentences[i] = sentences[i].replace('\n', ' ')

    sentence_words = dict(list(enumerate([tokenize(sentences[s]) for s in sentences])))

    matrix = [[0] * len(words) for i in range(len(sentences))]
    for s in sentence_words:
        for word in sentence_words[s]:
            matrix[s][inverse_words[word]] += 1
            word_counts[word] += 1

    word_sent_count = copy.deepcopy(inverse_words)
    word_sent_count = dict.fromkeys(word_counts, 0)
    for j in range(len(matrix[0])):
        count = 0
        for i in range(len(matrix)):
            if matrix[i][j] > 0:
                count += 1
        word_sent_count[words[j]] = count	
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            tf = 1.0 * matrix[i][j] / len(sentence_words[i])
            idf = math.log(1.0 * len(sentences) / (1 + word_sent_count[words[j]]))
            tf_idf = round(tf * idf, 5)
            matrix[i][j] = tf_idf

    similarity_matrix = similarityMatrix(matrix)
    text_rank = textRank(.85, matrix, 1000)
    avg_sent_length = averageSentenceLength(sentences)
    #print("Average sentence length: " + str(avg_sent_length))
    best = 0
    none_shorter = 1
    while best < len(sentences) and len(sentences[best]) > avg_sent_length:
        best += 1
    if best >= len(sentences):
        best = 0
    else:
        none_shorter = 0
    if none_shorter:
        for i in range(len(text_rank)):
            if text_rank[i] >= best:
                best = i
    else:
        for i in range(len(text_rank)):
            if text_rank[i] >= best and len(sentences[i]) <= avg_sent_length:
                best = i
    message=getTitle(sentences[best])
    print("Title: " + message )
    
    img_path=["place0.jpg","place1.jpg","place2.jpg","place3.jpg"]
    img_path1=["place0.png","place1.png","place2.png","place3.png"]

    values=sentiment(temp[0])
    backgroundhue(img_path[0],values['compound'],"place0.png")
    icon = Image.open(resizer(img_path1[0]))
    x, y = icon.size
    
    im = Image.new("RGB", (x+x+10,y+y+60), "white")
    draw = ImageDraw.Draw(im)
    im.paste(icon, (0,50)) 
    values=sentiment(temp[1])
    backgroundhue(img_path[1],values['compound'],"place1.png")

    icon = Image.open(resizer(img_path1[1]))
    im.paste(icon,(x+10,50))
    values=sentiment(temp[2])
    backgroundhue(img_path[2],values['compound'],"place2.png")

    icon = Image.open(resizer(img_path1[2]))
    im.paste(icon,(0,y+60))
    values=sentiment(temp[3])
    backgroundhue(img_path[3],values['compound'],"place3.png")

    icon = Image.open(resizer(img_path1[3]))
    im.paste(icon,(x+10,y+60))
    del draw
    im.save("test.jpg", "JPEG")

    img = Image.open('test.jpg')
    draw = ImageDraw.Draw(img)
    fnt = ImageFont.truetype("comic.ttf", 16)
    color = 'rgb(0, 0, 0)'
    draw.text((x/2,10), message, font=fnt, fill=color)
    img.save('comicstrip.png', quality=95)
    

def headline(text,img_path,flag):
    
    model = markovify.Text(text, state_size = 1)
    dictt={}
    for i in range(20):
        temp = model.make_sentence()
        if temp is not None:
            l=len(temp.split())
            dictt[temp]=l
    message=min(dictt.items(), key=operator.itemgetter(1))[0]
    print(message)
    img = Image.open(img_path)
    x, y = img.size
    draw = ImageDraw.Draw(img)
    fnt = ImageFont.truetype("comic.ttf", 16)
    img1 = Image.new('RGB', (len(message)*10, 50), color = (73, 109, 137))
    d = ImageDraw.Draw(img1)
    d.text((10,10), message, font=fnt, fill=(255, 255, 0))
    img1.save('pil_text_font.png')
    back_im = img.copy()
    if(flag==0):
        
        back_im.paste(img1, (0, 52))
    elif(flag==1):
        
        back_im.paste(img1, (int(x/2+10), 52))
    elif(flag==2):
        
        back_im.paste(img1, (0, int(y/2+30)))
    else:
        
        back_im.paste(img1, (int(x/2+10), int(y/2+30)))
    back_im.save('comicstrip.png', quality=95)




def balloongen(msg,name,fname):
    para = textwrap.wrap(msg, width=40)
    print (msg)
    MAX_W, MAX_H = 200, 200
    im = Image.open(name)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype('arial.ttf', 20)

    current_h, pad = 100, 10
    for line in para:
        w, h = draw.textsize(line, font=font)
        draw.text((90, current_h), line, font=font,fill="black")
        current_h += h + pad
    
    im.save(fname)

def faceballoon(im1,im2,flag,f):
    image = Image.open(im2)
    image = image.resize((300, 580), Image.ANTIALIAS)
    quality_val = 90
    image.save(im2, 'JPEG', quality=quality_val)
    images_list = [im1,im2]
    imgs = [Image.open(i) for i in images_list]
    min_img_width = min(i.width for i in imgs)

    total_height = 0
    for i, img in enumerate(imgs):
        if img.width > min_img_width:
            imgs[i] = img.resize((min_img_width, int(img.height / img.width * min_img_width)), Image.ANTIALIAS)
        total_height += imgs[i].height

    img_merge = Image.new(imgs[0].mode, (min_img_width+150, total_height))
    y = 0
    
    if(flag==0):
        x=150
        for img in imgs:
            img_merge.paste(img, (x, y))
            x=0
            y += img.height
    else:
        x=0
        for img in imgs:
            img_merge.paste(img, (x, y))
            x+=150
            y += img.height
    out="combinedimg"+str(f)+".png"
    img_merge.save(out)

def cartoongenerator(src,dst,flag):
    out="out"+str(flag)+".jpg"
    
    src_img = cv2.imread(src)
    dst_img = cv2.imread(dst)

    
    src_points, src_shape, src_face = select_face(src_img)
    
    dst_points, dst_shape, dst_face = select_face(dst_img)

    h, w = dst_face.shape[:2]
    
    warped_src_face = warp_image_3d(src_face, src_points[:48], dst_points[:48], (h, w))
   
    mask = mask_from_points((h, w), dst_points)
    mask_src = np.mean(warped_src_face, axis=2) > 0
    mask = np.asarray(mask*mask_src, dtype=np.uint8)

    warped_src_face = apply_mask(warped_src_face, mask)
    dst_face_masked = apply_mask(dst_face, mask)
    warped_src_face = correct_colours(dst_face_masked, warped_src_face, dst_points)

    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)

    x, y, w, h = dst_shape
    dst_img_cp = dst_img.copy()
    dst_img_cp[y:y+h, x:x+w] = output
    output = dst_img_cp
    cv2.imwrite(out, output)

def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')


def shift_hue(arr,hout):
    hsv=rgb_to_hsv(arr)
    hsv[...,0]=hout
    rgb=hsv_to_rgb(hsv)
    return rgb

def backgroundhue(img_name,val,outname):
    img = Image.open(img_name).convert('RGBA')
    arr = np.array(img)
    green_hue = (180-78)/360.0
    red_hue = (180-180)/360.0
    if(val<0):
        new_img = Image.fromarray(shift_hue(arr,red_hue), 'RGBA')
    else:
        new_img = Image.fromarray(shift_hue(arr,green_hue), 'RGBA')
    new_img.save(outname)

def finalcombiner(flag,fname,msg,bl,panel,f):
    basewidth = 200
    name="combinedimg"+str(f)+".png"
    img = Image.open(name)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save('sompic.png')
    file_name = "sompic.png"
    src = cv2.imread(file_name, 1)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    cv2.imwrite("sompic.png", dst)
    background = Image.open(fname)
    foreground = Image.open("sompic.png")
    x,y=background.size
    if(panel==1):
        if(flag==0):
            background.paste(foreground, (0, int(y/2-375)), foreground.convert('RGBA'))
        else:
            
            background.paste(foreground, (int(x/2-250), int(y/2-375)), foreground.convert('RGBA'))
        background.save("comicstrip.png")
    elif(panel==2):
        if(flag==0):
            background.paste(foreground, (int(x/2+10), int(y/2-375)), foreground.convert('RGBA'))
        else:
            
            background.paste(foreground, (int(x-250), int(y/2-375)), foreground.convert('RGBA'))
        background.save("comicstrip.png")
    elif(panel==3):
        if(flag==0):
            background.paste(foreground, (0, y-350), foreground.convert('RGBA'))
        else:
            
            background.paste(foreground, (int(x/2-250), y-350), foreground.convert('RGBA'))
        background.save("comicstrip.png")
    else:
        if(flag==0):
            background.paste(foreground, (int(x/2+10), y-350), foreground.convert('RGBA'))
        else:
            
            background.paste(foreground, (int(x-250), y-350), foreground.convert('RGBA'))
        background.save("comicstrip.png")

    nn="comicstrip.png"
    im = cv2.imread(nn)
    tmp = cv2.imread(bl,0)

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    w, h = tmp.shape[::-1] 
    res = cv2.matchTemplate(im_gray,tmp,cv2.TM_CCOEFF_NORMED) 

    threshold = 0.7
    loc = np.where( res >= threshold)  
    coor=[]
    
    pts=[[0,50,300,525],[301,50,636,525],[652,50,927,525],[950,50,1270,525],[0,585,300,1000],[300,585,633,1000],[650,585,955,1000],[960,585,1276,1000]]

    for pt in zip(*loc[::-1]): 
    
        coor.append(pt)
    print(coor)
    coor1=[]
    for c in coor:
        for c in coor:
            if(c[0]>pts[f][0] and c[0]<pts[f][2] and c[1]>pts[f][1] and c[1]<pts[f][3]):
                coor1.append(c)

    average = [sum(x)/len(x) for x in zip(*coor1)]
    print(average)

    topleftx = average[0]
    toplefty = average[1]
    
    
    #sqbl
    if(len(msg)>100):
        grid = Image.open("comicstrip.png")
        im = Image.open("sqballoon1.png")
        size=250,165
        im_resized = im.resize(size, Image.ANTIALIAS)
        im_resized.save("sqballoon1.png", "PNG")
        im = Image.open("sqballoon1.png")
        grid.paste(im, (int(topleftx-55),int(toplefty-14)),im)
        grid.save("comicstrip.png")
    
        
    para = textwrap.wrap(msg, width=25)
    
    im = Image.open(nn)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype('arial.ttf', 9)
    if(flag==0):
        a=topleftx+20
        b=toplefty+25
    else:
        a=topleftx+20
        b=toplefty+20
    pad =5
    for line in para:
        w, h = draw.textsize(line, font=font)
        draw.text(( a,b), line, font=font,fill="black")
        b += h + pad
    

    im.save(nn)


    
def text_clean(data,column):
  data = data[column].str.lower()#Convert the text to lower
  #We will remove the most repeated words like 'said','us','also'
  data = data.str.replace('said','')
  data = data.str.replace('us','')
  data = data.str.replace('also','')
  data = data.apply(lambda x : re.sub(r'[^a-z]',' ',x))#Remove numbers, special characters
  data = data.apply(lambda x : ' '.join([word for word in nltk.word_tokenize(x) if word not in stop_words]))#Stop word removal
  data = data.apply(lambda x : ' '.join([word for word in x.split() if len(word) > 2 ]))
  return data

def genredetect(msg):
    myList = [item for item in msg.split('\n')]
    newString = ' '.join(myList)

    loaded_model = pickle.load(open("news_model.sav", 'rb'))
    test = pd.DataFrame({'STORY' : newString},index=[0])
    test['clean'] = text_clean(test,'STORY')
    data = pd.read_excel('Data_Train.xlsx')
    data['clean'] = text_clean(data,'STORY')
    X = data['clean']
    tokenizer = Tokenizer(num_words = 25000)
    tokenizer.fit_on_texts(X)
    test_tokens = tokenizer.texts_to_sequences(test['clean'])

    test_tokens = pad_sequences(test_tokens,maxlen= 540)
    predictions = loaded_model.predict_classes(test_tokens)
    index=["Politics","Technology","Entertainment","Business"]
    return index[int(predictions)]

def flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:        
             yield item
#scrap()
"""
faceextract("michelle.jpg",0)
faceextract("barack.jpg",1)
body0=bodyselector("facealone0.jpg",0)
body1=bodyselector("facealone1.jpg",1)"""

text="""

Trump said " I would never be excited about a crowd again" after his visit to India where he addressed a rally of over 1,00,000 people.During their 36-hour visit, President Trump and the first lady attended various events and visited two cities  Ahmedabad and Agra  besides the national capital of India. In over seven decades of its independent history, India's relationship with the United States have always been complicated. The Cold War dynamics and India's nuclear ambitions had dictated the individual leanings of both nations before the turn of the new millennium. But since his visit to India in 2000, economics has become the defining feature of the relationship. India's economic rise post-liberalisation and its unique position to counter-balance China's dominance in Asia places it in a favourable light from the US perspective. However, the complicated nature of the equation has not yet resolved itself. In fact, Donald Trump’s visit to India  despite all the bonhomie on display  came in the backdrop of some hot and cold trade-related negotiations between the two nations that are yet to fructify. Trump has been quite vocal about his stand with respect to Indian markets and has constantly bashed the country as the tariffs king. Modi told the rally that "US has a great leader, and they have a great love for the people of this country."

Ms Trump had accompanied her father US President  on his first official visit to India. Ivanka Trump made the most of her time during her two-day visit to India and thanked the country in a heartfelt post on Insta on Wednesday. Along with her husband Jared Kushner, father US President and First Lady , Ivanka arrived in Ahmedabad on February 24 and departed the next day.

After returning to the US from India, Ivanka shared a picture with Jared Kushner from her Taj Mahal visit, and said in the caption, Ivanka said  "Thank you, India!"   .The picture is a candid shot of Ivanka and Jared holding hands in the backdrop of the Taj Mahal.Jared said "A time well spent"

The White House is warning that more Americans will die of the coronavirus as news emerged that a second man had succumbed to the disease on US soil. Still, top officials are scrambling to show they are on top of the situation as new cases spring up from coast-to-coast.
Democratic presidential candidates, led by former Vice President Joe Biden, are ripping into President Donald Trump for politicizing the situation. The President, his son and his conservative media cheerleaders had earlier claimed that criticism of the administration's efforts represented an orchestrated political campaign to bring him down.Stirred by the virus, the political maelstrom gripping Washington intensified as the first two Americans perished from the coronavirus. Two men have now died from the disease in Washington state's King County. Both had underlying health issues, with one man in his 50s and the other was 70, officials said. 
Trump said "Efforts are being taken"  but Joe Biden says as " No apt methods are being followed".

The second issue with such an argument is that it focuses too much on trade for a very myopic view of the relationship. Trade is just one of the aspects of bilateral ties that prime minister Modi has aptly called "people-driven, people-centric and the most important partnership of the 21st Century."
To make trade negotiations the touchstone of a relationship — that impacts various geopolitical corners of a rapidly changing post-Cold War order and works as a democratic bulwark against the rise of authoritarianism around the world — is a blinkered and self-limiting exercise.
On the contrary, it can be argued that Trump’s maiden visit to India has resulted in substantive deliverables, though a comprehensive trade deal has remained elusive.
Prime Minister Modi and President Trump recognized the increasing importance of the trade and investment dimension of the India-United States relationship, and the need for long-term trade stability that will benefit both the American and Indian economies. They agreed to promptly conclude the ongoing negotiations, which they hope can become phase one of a comprehensive bilateral trade agreement that reflects the true ambition and full potential of the bilateral commercial relations, advancing prosperity, investment, and job creation in both countries.
Trump said "trade friction won’t be allowed to spill into other domains and harm the close strategic partnership."

"""


#print(genredetect(text))

#text1=scrap1()


text = text.replace('“','"').replace('”','"').replace("’","'")
dialogues=list(flatten(quotedtext(text)))
print(dialogues)

temp=[]

initial=0
for i in range(4):
    loc1=text.find(dialogues[i*2+1])
    loc1+=len(dialogues[i*2+1])+1
    txt = text.replace("'s", "  ").replace(",", " ")
    part_text=txt[initial:loc1]
    temp.append(part_text)
    nameandlocation(part_text)
    initial=loc1

""" titlegen(text,temp)

for i in range(4):
    headline(temp[i],"comicstrip.png",i) """

"""
bingimgs("names.txt","person")
bingimgs("places.txt","place")
 """

# image copy for repeated character
""" 
with open('names.txt', 'r') as file:
    li=file.read().replace('\n', ' ')
print(li)
sentence= list(li.split(" ")) 
j=1
itr=1
for k  in range(7):
    for i in range(j,8):
        
        if(sentence[k]==sentence[i]):
            print(k,i)
            src="person"+str(k)+".jpg"
            dst="person"+str(i)+".jpg"
            copyfile(src, dst)
    itr=itr+1
    j=itr """

""" for i in range (8):
    strr="person"+str(i)+".jpg"
    print(strr)
    body=bodyselector(strr)
    cartoongenerator(strr,body,i)
    out="out"+str(i)+".jpg"
    j=i%2
    jstr="balloon"+str(j)+".jpg"
    jstr1="testbloon"+str(j)+".png"
    faceballoon(jstr,out,j,i) """


for i in range(8):
    j=i%2
    jstr1="testbloon"+str(j)+".png"
    jstr2="rtestbloon"+str(j)+".png"
    if(i%2==0):
        val=int(i/2)+1
    else:
        val=math.ceil(i/2)
    
    values=sentiment(temp[val-1])
    if(values['compound']>0):
        print("postive")
        finalcombiner(j,"comicstrip.png",dialogues[i],jstr1,val,i)
    else:
        print("neg")
        finalcombiner(j,"comicstrip.png",dialogues[i],jstr2,val,i)



