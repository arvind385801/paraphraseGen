#NOTE: runs with python2.7
#SEE: /home/sbailey6/sbailey6/work/redditscraper/scripts/subreddit_csv_maker.py for examples of scraping an entire subreddit

import numpy as np
import urllib, json
import datetime as dt

hub_directory = "/home/sbailey6/Desktop/textGenerator/paraphraseGen/data/AA/script/"
comment_ids_URL = "https://api.pushshift.io/reddit/submission/comment_ids/"
comment_extraction_URL = "https://api.pushshift.io/reddit/search/comment/?ids="
submission_extraction_URL = "https://api.pushshift.io/reddit/search/submission/?ids="

#start by extracting all page ids were original post mentions relapse or struggle
def getPageIds():
    page_ids = set()

    relapse = open(hub_directory + "relapsers.txt", 'r')
    for line in relapse.readlines():
        pg_id = line.split(',')[4]
        page_ids.add(pg_id)
    relapse.close()

    struggle = open(hub_directory + "strugglers.txt", 'r')
    for line in struggle.readlines():
        pg_id = line.split(',')[4]
        page_ids.add(pg_id)
    struggle.close()

    return page_ids

def getImmediateResponses(page_id):
    
    url = urllib.urlopen(comment_ids_URL + page_id)
    response_ids = json.loads(url.read().decode())["data"]
    comment_num = len(response_ids)
    url.close()

    comment_str = ""
    for comment_id in response_ids:
        comment_str += comment_id + ","
    comment_str = comment_str[:-1]

    responses = []

    url = urllib.urlopen(comment_extraction_URL + comment_str)
    reply = json.loads(url.read().decode())["data"]
    for j in range(0, comment_num):
        if(reply[j]["parent_id"][3:] != page_id):
            continue
        response_body = reply[j]["body"]
        responses.append(response_body)
    
    return responses


def main():
    page_ids = getPageIds()
    for page_id in page_ids:
        responses = getImmediateResponses(page_id)
        file = open(page_id + ".txt", 'w')
        file.write("-------original submission--------\n\n")
        url = urllib.urlopen(submission_extraction_URL + page_id)
        reply = json.loads(url.read().decode())["data"][0]
        
        try:
            sub_body = reply["selftext"]
            link = "reddit.com" + reply["permalink"]
        except:
            continue
        
        url.close()
        file.write("link: " + '' .join([i if ord(i) < 128 else ' ' for i in link.replace(u"\u2019", "'").replace(",", "").replace(r"\r\n", ".").replace("\r", ".").replace("\n", ".")]) + "\n\n")
        file.write('' .join([i if ord(i) < 128 else ' ' for i in sub_body.replace(u"\u2019", "'").replace(",", "").replace(r"\r\n", ".").replace("\r", ".").replace("\n", ".")]) + "\n")
        file.write("----------immediate responses--------\n\n")
        for entry in responses:
            file.write('' .join([i if ord(i) < 128 else ' ' for i in entry.replace(u"\u2019", "'").replace(",", "").replace(r"\r\n", ".").replace("\r", ".").replace("\n", ".")]) + "\n")
        file.close()
    
main()
    