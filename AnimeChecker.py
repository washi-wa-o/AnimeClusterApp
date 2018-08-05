# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 19:11:47 2018

@author: Rahul Nori
"""

import re
import glob
from sklearn.feature_extraction import text
from collections import Counter

class AnimeChecker(object):
    def __init__(self,path,extensions):
        self.path = path
        self.wordFreq = Counter()
        mystopwords = ["subbed","episode","horriblesubs","amv","online","watch","subdesu","mpt","tbs","aac","reinforce","oad","bdrip","flac","eng-sub"]
        self.stopwords = text.ENGLISH_STOP_WORDS.union(mystopwords)
        self.anime_substrings = set()
        self.english = []
        self.complete_anime_names = []
        self.anime_names = []
        self.reverse_table = {}
    
        with open("english_words.txt") as word_file:
            self.english_words = set(word.strip().lower() for word in word_file)
        
        self.extensions = extensions
        self.files = self.readFilesFromPath(self.path,self.extensions)
        self.populateAnime()
        self.reversePopulate()
        

    def plainString(self,word,repl):
        word = re.sub("[^A-Za-z]",repl,word).lstrip()
        parts = word.split(" ")
        new_word = ""
        for part in parts:
            if len(part)<3:
                continue
            elif part in self.stopwords:
                continue
            else:
                new_word = new_word + " " + part
        return new_word.lower()
    
    def readFilesFromPath(self,path,extns):
        #print path,extns
        plainfiles = []
        processedfiles = []
        for extn in extns:
            extnlst = glob.glob(path + "\\" + "*" + extn)
            extnlst = map(lambda x:x.replace(path,""),extnlst)
            plainfiles = plainfiles + extnlst
            extnlst = map(lambda x:self.plainString(x," "),extnlst)
            processedfiles = processedfiles + extnlst
        #print len(plainfiles),len(processedfiles)
        processedfiles = map(lambda x: self.plainString(x, " "),processedfiles)
        #print processedfiles[:50]
        return processedfiles
        
    def isWordEnglish(self,word):
        if word in self.english_words:
            return True
        return False
    
    def populateAnime(self):
        csv_file = "animetitle.txt"
        anime_rows = []
        with open(csv_file,"r") as csv:
            anime_rows = csv.readlines()
        self.anime_names = [item.split(",",1)[1] for item in anime_rows]
        self.anime_names = map(lambda x:x.strip("\n"),self.anime_names)
        self.complete_anime_names = self.anime_names
        self.anime_names = map(lambda x:self.plainString(x," "),self.anime_names)
        
        anilist = self.anime_names
        newanimenames = []
        for anime in anilist:
            words = anime.split()
            diffword = ""
            for word in words:
                if self.isWordEnglish(word):
                    self.wordFreq[word] += 1
                else:    
                    self.wordFreq[word] += 1
                    diffword += word + " "
            newanimenames.append(diffword)
        
        for name in newanimenames:
            for word in name.split():
                self.anime_substrings.update(self.formSubstrings(word))
                
    def reversePopulate(self):
        for idx,name in enumerate(self.anime_names):
            words = name.split()
            for word in words:
                if word in self.reverse_table:
                    self.reverse_table[word].add(self.complete_anime_names[idx])
                else:
                    self.reverse_table[word] = set()
                    self.reverse_table[word].add(self.complete_anime_names[idx])
            
                
    def formSubstrings(self,content):
        l = len(content)
        substrings = [content[i:j] for i in range(0,l+1) for j in range(i+1,l+1)]
        substrings = filter(lambda x:len(x)>3,substrings)
        return substrings
        
        
    def findConfidenceForWord(self,name):
        #print "confidence for word finding: ",name
        words = name.split()
        substrings = set()
        for word in words:
            substrings.update(self.formSubstrings(word))
        #print "substrings for confidence are: ",substrings
        substr_intersect = substrings.intersection(self.anime_substrings)
        #print "intersection is: ",substr_intersect
        confidence = 0
        confidence += len(substr_intersect)*0.1
        return confidence
        
    def isAnimeTitle(self,filename):
        confidence = 0.0
        englishCount = 0
        nonenglishwords = ""
        words = filename.split()
        
        for word in words:
            #print "word is ",word
            if word in self.wordFreq:
                if self.isWordEnglish(word):
                    englishCount += 1
                    confidence += 0.01
                else:
                    confidence += 1.0
            else:
                if self.isWordEnglish(word):
                    englishCount += 1
                    pass
                else:
                    nonenglishwords +=  word + " "
            
        #print "words and confidence are ",words,nonenglishwords,confidence
        if confidence < 1.0:
            if englishCount > 0 and confidence > 0 and len(nonenglishwords) == 0 :
                engcountindict = confidence/0.01
                if engcountindict>1:
                    confidence = 1.0
                elif len(words)==1 and engcountindict == 1:
                    confidence = 0.501
                else:
                    confidence = 0.0
            elif englishCount > 0 and confidence == 0:
                confidence = 0
            else:
                confidence = self.findConfidenceForWord(nonenglishwords)
                
        if confidence>1.0:
            confidence = 1.0
        #print "Confidence is: ",confidence
        
        if confidence>0.5:
            return True
        else:
            return False

    def getAnimeNames(self,strings):
        #result_set = set()
        result = Counter()
        for idx,string in enumerate(strings):
            words  = string[0].split()#string is a tuple
            #print words
            #temp_set = set()
            temp = Counter()
            for jdx,word in enumerate(words):
                #print "Inside words",word
                if word in self.reverse_table:
                    #print "self reverse ",self.reverse_table[word]
                    for ele in self.reverse_table[word]:
                        #print ele
                        temp[ele] += 1
                    '''
                    if jdx == 0:
                        if len(self.reverse_table[word]) == 1:
                            temp_set.add(self.reverse_table[word])
                        else:
                            temp_set.update(self.reverse_table[word])
                    else:
                        temp_set.intersection(self.reverse_table[word])
                    '''
            #print idx," Temp Set is: ",temp.most_common(10)
            temp_set = temp.most_common(10)
            for element,freq in temp_set:
                result[element] += 1
            sorted_results = [(v[0],v[1]) for v in sorted(result.iteritems(), key=lambda(k, v): (-v, -len(k)))]
        #print "Result is: ",result
        #print "Sorted results are: ",sorted_results
        sorted_results = map(lambda x:x[0],sorted_results)
        sorted_results = filter(lambda x:len(x)>2,sorted_results)
        return list(sorted_results)