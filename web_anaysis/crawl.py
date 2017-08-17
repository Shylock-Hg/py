'''
@brief: a simple web crawl to explore wikipedia page until find target page
@author: TcAth2s
@date:2017.8.7
@email:TcAth2s@icloud.com
'''

import requests as rq
import bs4
import urllib

def continue_crawl(links,target,count = 25):
    if(links[-1] == target):
        print('reach target page')
        return False
    elif(len(links) > count):
        print('explored max count pages')
        return False
    elif(links[-1] in links[:-1]):
        print('reach the repeated page')
        return False
    else:
        return True

def next_link(url):
    f = rq.get(url)
    s = bs4.BeautifulSoup(f.text,'html.parser')
    link_1st = ''
    for a in s.p.find_all('a'):
        link_1st = a.get('href')
        break
    if(link_1st == ''):
        return ''
    else:
        return urllib.parse.urljoin('https://en.wikipedia.org/',link_1st)

links = ['https://en.wikipedia.org/wiki/Dead_Parrot_sketch']
target = 'https://en.wikipedia.org/wiki/Philosophy'

#print(next_link('https://en.wikipedia.org/wiki/Philosophy'))

def crawl(links,target,count = 25):
    links.append(next_link(links[-1]))
    while continue_crawl(links,target,count):
        crawl(links,target,count)
    return links

ls = crawl(links,target)