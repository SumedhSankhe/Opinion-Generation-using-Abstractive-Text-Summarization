# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:18:28 2019

@author: sumed
"""

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))



def generate_summary(d, sim_thresh = 0.3):
    
    candidates = [[]for i in range(len(d))]
    
    for n,i in enumerate(d):
        candidates[n].append(i)
        for e, j in enumerate(d):
            sim = get_jaccard_sim(i,j)
                
                
    return(candidates)
    
cd = {k: v for k, v in cd.items() if v>50}
x = generate_summary(list(cd.keys()), 0.7)

d = list(cd.keys())
sim_threh = 0.7
