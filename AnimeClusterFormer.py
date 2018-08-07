# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 18:08:21 2018

@author: Rahul Nori
"""
#make changes to manage songs,movies and series
#make changes to show next 10 anime names if first 10 anime names do not have correct name

import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob
from sklearn.cluster import AffinityPropagation
import os
import shutil
from Simhash import Simhash
from sklearn.feature_extraction import text
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import multiprocessing
import time
from AnimeChecker import AnimeChecker
from Tkinter import Tk
from tkFileDialog import askopenfilename
import tkFileDialog

def plainString(word,repl):
    global stopwords
    word = re.sub("\[.*?\]",repl,word)
    word = re.sub("[^A-Za-z]",repl,word).lstrip()
    parts = word.split(" ")
    new_word = ""
    for part in parts:
        if len(part)<3:
            continue
        elif part in stopwords:
            continue
        else:
            new_word = new_word + " " + part
    return new_word.lower()

def cleanString(word,repl):
    global stopwords
    word = re.sub("[^A-Za-z]",repl,word).lstrip()
    return word

def getFileNamesFromPath(path,extns):
    #print path,extns
    plainfiles = []
    processedfiles = []
    for extn in extns:
        extnlst = glob.glob(path + "\\" + "*" + extn)
        extnlst = map(lambda x:x.replace(path,""),extnlst)
        plainfiles = plainfiles + extnlst
        extnlst = map(lambda x:plainString(x," "),extnlst)
        processedfiles = processedfiles + extnlst
    #print len(plainfiles),len(processedfiles)
    processedfiles = map(lambda x: plainString(x, " "),processedfiles)
    #print processedfiles[:50]
    return (plainfiles,processedfiles)
    

def distance(value1,value2):
    f = 128
    x = (value1 ^ value2) & ((1 << f) - 1)
    ans = 0
    while x:
        ans += 1
        x &= x - 1
    return ((float)(128-ans))/128

def wait():
    raw_input("Press any key to exit wait")
    
def calculateSimhash(filename):
    return Simhash(filename).value

def calculateSimilarity(filenames):
    #print filenames[0]
    time_start = time.time()
    sim_matrix = []
    filehash = []
    for fi in filenames:
        fihash = Simhash(fi).value
        filehash.append(fihash)
    print "Time taken is ",time.time()-time_start
   
    '''
    #Multiprocessing module
    time_start = time.time()
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=pool_size)
    pool_outputs = pool.map(calculateSimhash, filenames)
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks
    print "Time taken is ",time.time()-time_start
    '''
            
    print "Simhash calculation done."
    #wait()
    
    for fi in range(0,len(filenames)):
        fihash = filehash[fi]
        sim_row = []
        for fo in range(0,len(filenames)):
            fohash = filehash[fo]
            fifodist = distance(fihash,fohash)
            '''
            print filenames[fi],filenames[fo],fifodist
            wait()
            '''
            #print fihash,fohash,fifodist
            sim_row.append(fifodist)
        sim_matrix.append(sim_row)
    print "Simhashing Completed."
    sim_matrix = np.array(sim_matrix)
    return sim_matrix

def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def tfidf(files):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 stop_words='english',
                                 use_idf=True, tokenizer=tokenize_only,ngram_range=(1,1))
    tfidf_matrix = tfidf_vectorizer.fit_transform(files)
    #print tfidf_vectorizer.get_feature_names()
    return tfidf_matrix

def fill_cosine_dict(cluster_elements):
    global cosine_dict
    for element in cluster_elements:
        if element not in cosine_dict:
            cosine_dict[element] = cluster_elements

def fill_simhash_dict(cluster_elements):
    global simhash_dict
    for element in cluster_elements:
        if element not in simhash_dict:
            simhash_dict[element] = cluster_elements
            
def affinityPropagation(dist,filenames,plfilenames,flag):
    ap = AffinityPropagation(affinity='precomputed',max_iter=1000,damping=0.77)
    ap_fit = ap.fit(dist)
    clusters = ap_fit.labels_.tolist()
    #print ap_fit.n_iter_
    num_clusters = len(set(clusters))
    #print "Num of Clusters is ",num_clusters
    #all_clusters = []
    for i in range(num_clusters):
        #print "New Cluster now: ",i
        cluster_el = []
        for idx,cluster in enumerate(clusters):
            if i == cluster:
                #print filenames[idx]
                cluster_el.append(plfilenames[idx])
                if flag == 1:
                    fill_cosine_dict(cluster_el)
                elif flag == 2:
                    fill_simhash_dict(cluster_el)
        #outlierDetection(cluster_el)
        #all_clusters.append(cluster_el)
        #print "-------------------"
        #raw_input("---Continue---")
        
def allSubstrings(string):
    parts = string.split()
    length = len(parts)
    subparts = [parts[i:j+1] for i in xrange(length) for j in xrange(i,length)]
    subparts = map(lambda x:' '.join(x),subparts)
    return subparts


def findLCS(strings):
    strings = map(lambda x:cleanString(x, " "),strings)
    string_db = {}
    for idx,string in enumerate(strings):
        substrings = allSubstrings(string.lower())
        for substring in substrings:
            if substring in string_db:
                string_db[substring] += 1
            else:
                string_db[substring] = 1
    

    sorted_results = [(v[0],v[1]) for v in sorted(string_db.iteritems(), key=lambda(k, v): (-v, -len(k)))]
    #print sorted_results
    return sorted_results[:7]

def isClusterAnime(cluster):
    global animeChecker
    
    isanime = []
    for element in cluster:
        #print element
        words = element[0].split()
        words = filter(lambda x:len(x)>1,words)
        words = ' '.join(words)
        isanime.append(animeChecker.isAnimeTitle(words))#element is a tuple
    #print "Isanime is: ",isanime
    true = isanime.count(True)
    
    true_cluster = []
    for idx,element in enumerate(cluster):
        if isanime[idx] == True:
            true_cluster.append(element)
        
    if true>0:
        names = animeChecker.getAnimeNames(true_cluster)
        if len(names) == 0:
            return True,cluster
        else:
            return True,names
    else:
        return False,cluster
 
def convertClusters(res,cluster):
    clusters = res.split(";")
    clusters = map(lambda x:map(lambda y:int(y),x.split(",")),clusters)
    clusters = map(lambda x:map(lambda y:cluster[y],x),clusters)
    #print clusters
    return clusters

def splitCluster(cluster):
    resp = raw_input("Do you want to split the cluster?(y/n)")
    if resp.lower() == "y":
        res = raw_input("Enter indices of elements you want to split into clusters in the form 0,4;1,3;2 ")
        clusters = convertClusters(res,cluster)
        return clusters
    else:
        return []
    
def printPossibleNames(animenames):
    index = 0
    while True:
        yield animenames[index:index+10]
        index += 10
        
def selectPath(string):
    #print "Choose the destination path you want to make the Folder: {}".format(name)
    root = Tk()
    root.withdraw()
    curdir = os.getcwd()
    filepath = tkFileDialog.askdirectory(parent=root, initialdir=curdir, title=string)
    return filepath
    
def main():
    global stopwords
    global wordCounter
    global simhash_dict
    global cosine_dict
    global animeChecker
    
    #fpath = "C:\\Users\\Rahul Nori\\Downloads\\Video\\"
    #fpath = "C:\\Users\\Rahul Nori\\Downloads\\"
    extensions = [".mp4",".mkv"]
    simhash_dict = {}
    cosine_dict = {}
    wordCounter = Counter()
    srcpath = selectPath("Select folder to process...")
    destpath = selectPath("Select folderpath to save anime...")
    srcpath = srcpath.replace("/","\\") + "\\"
    destpath = destpath.replace("/","\\") + "\\"
    print srcpath,destpath
    animeChecker = AnimeChecker(srcpath,extensions)
    
    mystopwords = ["subbed","episode","horriblesubs","amv","online","watch","subdesu","mpt","tbs","aac","reinforce","oad","bdrip","flac","eng-sub"]
    stopwords = text.ENGLISH_STOP_WORDS.union(mystopwords)
    #fpath = "C:\\Users\\Rahul Nori\\Downloads\\Video\\"
    #fpath = "C:\\Users\\Rahul Nori\\Downloads\\"
    while True:
        extensions = [".mp4",".mkv"]
        plfilenames,prfilenames = getFileNamesFromPath(srcpath,extensions)
        print "Cosine Sim calculation started."
        tfidf_matrix = tfidf(prfilenames)
        cosine_matrix = cosine_similarity(tfidf_matrix)
        affinityPropagation(cosine_matrix,prfilenames,plfilenames,1)
        print "Cosine Sim calculation finished."
        print "Simhash Sim calculation started."
        sim_matrix = calculateSimilarity(prfilenames)
        affinityPropagation(sim_matrix,prfilenames,plfilenames,2)
        print "Simhash Sim finished."
        #wait()
        
        
        print "Starting the clubbing of both results."
        result_dict = {}
        
        for filename in plfilenames:
            cosine_set = set(cosine_dict[filename])
            simhash_set = set(simhash_dict[filename])
            
            result = cosine_set.intersection(simhash_set)
            result_dict[filename] = result
            #print filename, result
            #wait()
        print "Finished the clubbing of both results."
        
        #Form clusters
        print "Cluster forming started."
        cluster_lst = []
        for item in result_dict.values():
            flag = False
            if len(cluster_lst) == 0:
                cluster_lst.append(item)
            else:
                for element in cluster_lst:
                    if element == item:
                        flag = True
                        break
                if flag == True:
                    continue
                else:
                    cluster_lst.append(item)
        
        print "Cluster forming finished."
        
        #To-do : merge clusters
        
        cluster_lst = map(lambda x:list(x),cluster_lst)
        print "The number of clusters is: ",len(cluster_lst)
        
        #for idx,cluster in enumerate(cluster_lst):
        index = 0
        while index<len(cluster_lst):
            print "Cluster number: ",index
            cluster = cluster_lst[index]
            for jdx,element in enumerate(cluster):
                print jdx,element
            print "---------------------------"
            
            new_cluster = splitCluster(cluster)
            if len(new_cluster) == 0:
                pass
            else:
                cluster_lst.pop(index)
                for item in new_cluster:
                    cluster_lst.insert(index,item)
            
                print "Cluster number: ",index
                print "Num of tot_clusters is ",cluster_lst
                cluster = cluster_lst[index]
                for jdx,element in enumerate(cluster):
                    print jdx,element
                print "---------------------------"
            
            
            cluster_names = findLCS(cluster)
            isanime,animenames = isClusterAnime(cluster_names)
            cluster_filename = ""
            if isanime:
                print "Possible anime names are:"
                for idx,aniname in enumerate(animenames[:15]):
                    print idx,": ",aniname
                '''
                while True:
                    yanimenames = printPossibleNames(animenames)
                    for idx,aniname in enumerate(yanimenames):#change this line, handle the 10
                        print idx,": ",aniname
                    resp = raw_input("Print more names?(y/n)")
                    if resp.lower() == "y":
                        pass
                    else:
                        break
                '''
                if raw_input("Choose a name?(y/n) ").lower() == "y":
                    num = int(raw_input("Enter index number: "))
                    cluster_filename = animenames[num]
                    print "The name for the folder you have chosen is ",cluster_filename
                else:
                    resp = raw_input("Was this an anime?(y/n)")
                    if resp.lower() == "y":
                        cluster_filename = raw_input("Enter the foldername you would like for the anime : ")
                        if len(cluster_filename) == 0:
                            print "The foldername is Temp because you did not enter anything"
                            cluster_filename = "Temp"
                        else:
                            print "The foldername you have typed is ",cluster_filename
                    else:
                        #save the filenames somewhere
                        print "Thank you for the info."
                
                #create folder in some fixed path directory in anime folder if it does not exist
                #move these files from here to there
            else:
                resp = raw_input("Does this cluster represent anime files? ")
                if(resp.lower() == "y"):
                    #Save somewhere that this is an anime
                    cluster_filename = raw_input("Enter the foldername: ")
                    if len(cluster_filename) == 0:
                        print "The foldername is Temp because you did not enter anything"
                        cluster_filename = "Temp"
                    else:
                        print "The foldername you have typed is ",cluster_filename
                else:
                    print "Good to know I guessed correctly."
            
            wait()
            index += 1
        wait()

if __name__ == "__main__":
    main()