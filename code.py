import copy
import os
import sys
import time
from math import log
from multiprocessing import *

tf = {}
df_global = {}

queue = Queue()
lock = Lock()


def CountWords(words):
    localDic = {}
    for word in words:
        if localDic.get(word) is None:
            localDic[word] = 1
        else:
            localDic[word] += 1
    return localDic


def GetTF_IncDF(localDic, len_of_words):
    for word, n in localDic.items():
        tf = n / float(len_of_words)
        localDic[word] = tf

    return localDic


def TF_POOL(line_tuple):
    #  (1,"STRING")
    index = line_tuple[0]
    words = line_tuple[1].strip().split(" ")
    local_tf = GetTF_IncDF(CountWords(words), len(words))
    return (index, local_tf, len(words))

def TFIDF_POOL(tf_dic):
    global df
    if len(df) == 0:
        with lock:
            df = queue.get()
            
    index = tf_dic[0]
    sum = 0
    for word, tf in tf_dic[1].items():
        sum += tf * df[word]

    with lock:
        with open(sys.path[0] + "//count.txt", "a") as writer:
            writer.write(f"Document {index} has {tf_dic[2]} words.The average of TF - IDFs  {sum / float(len(tf_dic[1]))}\n")


if __name__ == "__main__":
    global df
    df = {}
    with open(sys.path[0] + "//input.txt", "r") as file:
        lines = file.readlines()
    writer =  open(sys.path[0] + "//count.txt", "a")

    indexed_lines = []
    # convert input to a tuple of (index,line) to save the index of the line.
    for index in range(len(lines)):
        indexed_lines.append((index, lines[index]))

    writer = open(sys.path[0] + "//count.txt", "w")
    writer.close()
    full_time = time.time()
    #  n is number of process
    n = cpu_count()
    pool = Pool(processes= n)
    start = time.time()
    tf_list = pool.map(TF_POOL, indexed_lines)
    end = time.time()
    print(f"TF: {end - start}")

    #  Calculate df
    start = time.time()
    with lock:
        for index in range(len(tf_list)):
            for word, k in tf_list[index][1].items():
                if df_global.get(word) is None:
                    df_global[word] = 1
                else:
                    # print("x")
                    df_global[word] += 1
    #
    len_document = len(lines)
    #  idf
    with lock:
        for word, dff in df_global.items():
            df_global[word] = log(len_document / float(dff))
    end = time.time()
    print(f"IDF: {end - start}")

    # adding to the queue to run
    for i in range(n):
        queue.put(df_global)
    start = time.time()
    pool.map(TFIDF_POOL, tf_list)
    end = time.time()
    print(f"TF*IDF and WRITE ON DISK: {end - start}")

    end_full_time = time.time()

    print(F"FULL TIME: {end_full_time - full_time}")