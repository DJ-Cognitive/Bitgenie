from nltk import text
import pdftitle
import requests
from PyPDF2 import PdfFileReader
import io
import pdfminer
#from pdfminer.high_level import extract_text
#from pdfminer.six import extract_text
from pdfminer.high_level import extract_text
import re
import datetime
#import pdf2image
#from pdf2image import convert_from_path, convert_from_bytes
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd 
from pytrends.request import TrendReq
import fitz
from googlesearch import search
import os
import datefinder
import dateparser
import csv
import time
###########################################################################################################################################
query = "Machine Learning"
###########################################################################################################################################
'''
url = 'https://cdn2.hubspot.net/hubfs/700740/Reports/Newzoo_Preview_Report_Global_Growth_of_Esports_Report_FINAL_2.0.pdf'
r = requests.get(url,stream = True)
r.raw.decode_content = True
inputio = io.BytesIO(r.content)
'''



def get_title(inputio):
    try:
        title = pdftitle.get_title_from_io(inputio)
        
    except:
        title = ''
        pass
    if title == '' or title == None:
        try:
            reader = PdfFileReader(inputio)
            contents = reader.getDocumentInfo()
            cont = dict(contents)
            title = cont['/Title']
        except:
            title = ''
            pass

    return title


def get_text(inputio):
    text1 = extract_text(inputio)
    text = ' '.join(text1.split())
    return text

def get_year(inputio):
    thisYear = datetime.datetime.now().year
    text = get_text(inputio)
    numbers = re.findall('\d+',text)   #find all the numbers in the string.
    numbers = map(int,numbers)
    for number in numbers:
        if (number > 2000) and (number<=thisYear):
            return number

def get_date(text):
  dates = []
  matches = datefinder.find_dates(text,source=True)
  for match in matches:
    if(len(match[1]) > 4):
      date1= match[1]
      date = dateparser.parse(date1)
      return date


def get_poster(url,title):

    filename = title + ".pdf"
    # URL of the image to be downloaded is defined as image_url
    r = requests.get(url) # create HTTP response object

    # send a HTTP request to the server and save
    # the HTTP response in a response object called r
    with open(filename,'wb') as f:

        # Saving received content as a png file in
        # binary format

        # write the contents of the response (r.content)
        # to a new file in binary mode.
        f.write(r.content)


    doc = fitz.open(filename)
    page = doc.loadPage(0) 
    pix = page.getPixmap()
    output = title + ".png"
    pix.writePNG(output)
    doc.close()
    os.remove(filename)
    return



def get_numpages(inputio):
    read_pdf = PdfFileReader(inputio)
    number_of_pages = read_pdf.getNumPages()
    return int(number_of_pages)



def get_summary(text):
    
    
    


    def _create_frequency_table(text_string) -> dict:
        """
        we create a dictionary for the word frequency table.
        For this, we should only use the words that are not part of the stopWords array.
        Removing stop words and making frequency table
        Stemmer - an algorithm to bring words to its root word.
        :rtype: dict
        """
        stopWords = set(stopwords.words("english"))
        words = word_tokenize(text_string)
        ps = PorterStemmer()

        freqTable = dict()
        for word in words:
            word = ps.stem(word)
            if word in stopWords:
                continue
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1

        return freqTable


    def _score_sentences(sentences, freqTable) -> dict:
        """
        score a sentence by its words
        Basic algorithm: adding the frequency of every non-stop word in a sentence divided by total no of words in a sentence.
        :rtype: dict
        """

        sentenceValue = dict()

        for sentence in sentences:
            word_count_in_sentence = (len(word_tokenize(sentence)))
            word_count_in_sentence_except_stop_words = 0
            for wordValue in freqTable:
                if wordValue in sentence.lower():
                    word_count_in_sentence_except_stop_words += 1
                    if sentence[:10] in sentenceValue:
                        sentenceValue[sentence[:10]] += freqTable[wordValue]
                    else:
                        sentenceValue[sentence[:10]] = freqTable[wordValue]

            if sentence[:10] in sentenceValue:
                sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] / word_count_in_sentence_except_stop_words

            '''
            Notice that a potential issue with our score algorithm is that long sentences will have an advantage over short sentences. 
            To solve this, we're dividing every sentence score by the number of words in the sentence.
            
            Note that here sentence[:10] is the first 10 character of any sentence, this is to save memory while saving keys of
            the dictionary.
            '''

        return sentenceValue


    def _find_average_score(sentenceValue) -> int:
        """
        Find the average score from the sentence value dictionary
        :rtype: int
        """
        sumValues = 0
        for entry in sentenceValue:
            sumValues += sentenceValue[entry]

        # Average value of a sentence from original text
        average = (sumValues / len(sentenceValue))

        return average


    def _generate_summary(sentences, sentenceValue, threshold):
        sentence_count = 0
        summary = ''

        for sentence in sentences:
            if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
                summary += " " + sentence
                sentence_count += 1

        return summary


    def run_summarization(text):
        # 1 Create the word frequency table
        freq_table = _create_frequency_table(text)
        

        '''
        We already have a sentence tokenizer, so we just need 
        to run the sent_tokenize() method to create the array of sentences.
        '''

        # 2 Tokenize the sentences
        sentences = sent_tokenize(text)

        

        # 3 Important Algorithm: score the sentences
        sentence_scores = _score_sentences(sentences, freq_table)
        
        # 4 Find the threshold
        threshold = _find_average_score(sentence_scores)
        

        # 5 Important Algorithm: Generate the summary
        summary = _generate_summary(sentences, sentence_scores, 1.3 * threshold)
        return str(summary)
    
    
    return run_summarization(text)


    










'''
print(get_text(inputio))

print(get_title(inputio))
print(get_year(inputio))
#print(get_poster(b))
print(get_numpages(inputio))
print(get_summary(inputio))
'''

def get_links(query):
    QUERY = []
    query1 = query + " market research reports filetype:pdf"
    pytrend = TrendReq()
    pytrend.build_payload(kw_list=[query])
    df = pytrend.related_queries()
    k = df.values()
    m = list(k)
    l = m[0]
    #print(l)
    try:
        top = l['top']
        values_top = top['query']
        top_list = list(values_top)
    except:
        top_list = []
        pass
    try:
        rising = l['rising']
        values_rising = rising['query']
        rising_list = list(values_rising)
    except:
        rising_list= []

    query_list2 = [query] + top_list + rising_list
    if len(query_list2) > 11:
        f = 11
        p = 10
    elif len(query_list2) == 1:
        f= 1
        p = 25 
    else:
        f = len(query_list2)
        p = 10
    #print(f,p)
    for i in query_list2[:f]:
        if f == 1:
            for j in range(25):
                QUERY.append(i)
        else:
            for j in range(10):
                QUERY.append(i)
    print("The number of queries are: "+ str(len(QUERY)))
    LINKS = []   

    for i in range(f):
        query = QUERY[i*10]
        print(query)
        query = query + " market research reports filetype:pdf"
        LINKS.append([x for x in search(query, tld = 'com', num = p, stop = p, pause = 2.0)])
        # for j in search(query, tld="com", num= p, stop=p, pause=2): 
        #     LINKS.append(j)
    print("The number of links obtained are: " + str(len(LINKS)))

    return LINKS



#get_links("Artificial Intelligence")
def main1(query):
    links = get_links(query)
    writer = csv.writer(open('results.csv', 'wt', newline=''))
    writer.writerow(('URL','Title','Year','Pages','Summary'))
    #size = len(links)
    #for m in range(size):
    for i in links[0]:
        try:
            print(i)
            url = i
            r = requests.get(url,stream = True)
            r.raw.decode_content = True
            inputio = io.BytesIO(r.content)
            #print(get_poster(b))
            start = time.time()
            pdf_title = get_title(inputio)
            end1 = time.time()
            print(f"Time taken for fetching title {end1 - start}")
            pdf_date = get_year(inputio)
            pdf_pages = get_numpages(inputio)
            pdf_summary = get_summary(text)
            row = (url,pdf_title,pdf_date,pdf_pages,pdf_summary)
            writer.writerow(row)

        except:
            pass
        
    return

main1(query)




# url = 'https://static.coindesk.com/wp-content/uploads/2021/05/2021-Q1-Crypto-Futures-Options-Market-Research-Report.pdf'


# def main2(url):
#     r = requests.get(url,stream = True)
#     r.raw.decode_content = True
#     inputio = io.BytesIO(r.content)
#     title = get_title(inputio)
#     text = get_text(inputio)
#     print("Title: " + title)
#     print("Year: " + str(get_year(inputio)))
#     print("Date: "+ str(get_date(text)))
#     get_poster(url,title)
#     print("Number of pages: " + str(get_numpages(inputio)))
#     print("Summary: " + get_summary(text))

# main2(url)