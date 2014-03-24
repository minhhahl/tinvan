# coding=UTF-8
import os
import sys
import codecs
import sys
import base64
import json
import nltk
import numpy
import requests
from boilerpipe.extract import Extractor

def summarize(url=None, html=None, n=100, cluster_threshold=5, top_sentences=5):

    # Adapted from "The Automatic Creation of Literature Abstracts" by H.P. Luhn
    #
    # Parameters:
    # * n  - Number of words to consider
    # * cluster_threshold - Distance between words to consider
    # * top_sentences - Number of sentences to return for a "top n" summary
            
    # Begin - nested helper function
    def score_sentences(sentences, important_words):
        scores = []
        sentence_idx = -1
    
        for s in [nltk.tokenize.word_tokenize(s) for s in sentences]:
    
            sentence_idx += 1
            word_idx = []
    
            # For each word in the word list...
            for w in important_words:
                try:
                    # Compute an index for important words in each sentence
    
                    word_idx.append(s.index(w))
                except ValueError, e: # w not in this particular sentence
                    pass
    
            word_idx.sort()
    
            # It is possible that some sentences may not contain any important words
            if len(word_idx)== 0: continue
    
            # Using the word index, compute clusters with a max distance threshold
            # for any two consecutive words
    
            clusters = []
            cluster = [word_idx[0]]
            i = 1
            while i < len(word_idx):
                if word_idx[i] - word_idx[i - 1] < cluster_threshold:
                    cluster.append(word_idx[i])
                else:
                    clusters.append(cluster[:])
                    cluster = [word_idx[i]]
                i += 1
            clusters.append(cluster)
    
            # Score each cluster. The max score for any given cluster is the score 
            # for the sentence.
    
            max_cluster_score = 0
            for c in clusters:
                significant_words_in_cluster = len(c)
                total_words_in_cluster = c[-1] - c[0] + 1
                score = 1.0 * significant_words_in_cluster \
                    * significant_words_in_cluster / total_words_in_cluster
    
                if score > max_cluster_score:
                    max_cluster_score = score
    
            scores.append((sentence_idx, score))
    
        return scores    
    
    # End - nested helper function
    
    extractor = Extractor(extractor='ArticleExtractor', url=url, html=html)

    # It's entirely possible that this "clean page" will be a big mess. YMMV.
    # The good news is that the summarize algorithm inherently accounts for handling
    # a lot of this noise.

    txt = extractor.getText()
    
#    print "-------------------------------------------------"
#    print "                'Text content'"
#    print "-------------------------------------------------"
#    print txt
#    print
    
    sentences = [s for s in nltk.tokenize.sent_tokenize(txt)]
    normalized_sentences = [s.lower() for s in sentences]

    words = [w.lower() for sentence in normalized_sentences for w in
             nltk.tokenize.word_tokenize(sentence)]

    fdist = nltk.FreqDist(words)
    
    stopset = set(nltk.corpus.stopwords.words('english'))
#    print stopset
    
    
    file_dir = os.path.dirname(os.path.abspath(__file__))
    # Add vietnam stopword
    with codecs.open(file_dir + '/stopwords_vn_s1.txt', 'r', 'utf-8') as text_file:
        text = text_file.read()
        text = text.encode('utf-8')
        tokens_vn = nltk.tokenize.word_tokenize(str(text))
        tokens_vn = [w for w in tokens_vn if not w in stopset]
#        print tokens_vn
        
    stopset = stopset.union(set(tokens_vn))
#    stopset = set(tokens_vn)

    top_n_words = [w[0] for w in fdist.items() 
            if w[0] not in stopset][:n]

    scored_sentences = score_sentences(normalized_sentences, top_n_words)

    # Summarization Approach 1:
    # Filter out nonsignificant sentences by using the average score plus a
    # fraction of the std dev as a filter

    avg = numpy.mean([s[1] for s in scored_sentences])
    std = numpy.std([s[1] for s in scored_sentences])
    mean_scored = [(sent_idx, score) for (sent_idx, score) in scored_sentences
                   if score > avg + 0.5 * std]

    # Summarization Approach 2:
    # Another approach would be to return only the top N ranked sentences

    top_n_scored = sorted(scored_sentences, key=lambda s: s[1])[-top_sentences:]
    top_n_scored = sorted(top_n_scored, key=lambda s: s[0])

    # Decorate the post object with summaries

    return dict(top_n_summary=[sentences[idx] for (idx, score) in top_n_scored],
                mean_scored_summary=[sentences[idx] for (idx, score) in mean_scored])

# Main   
if 1:
    # Load the data that PHP sent us
    try:
        data = json.loads(base64.b64decode(sys.argv[1]))
        sample_url = data['url']  

        summary = summarize(url=sample_url)
        
        str_top_n = " ".join(summary['top_n_summary'])
        str_mean = " ".join(summary['mean_scored_summary'])
         
        result = { 'status': 'OK', 'top_n': str_top_n, 'mean': str_mean }
    except:
        result = {'status': 'ERROR'}
            
    # Send it to stdout (to PHP)
    print json.dumps(result)
else:
    # Sample usage

    #sample_url = 'http://radar.oreilly.com/2013/06/phishing-in-facebooks-pond.html'

    #writer_factory = codecs.getwriter("utf-8")
    #sys.stdout = writer_factory(sys.stdout)

    sample_url = 'http://dantri.com.vn/nhip-song-tre/ong-truong-gia-binh-thanh-cong-den-tu-dam-me-va-khac-biet-853163.htm'
    summary = summarize(url=sample_url)

    # Alternatively, you can pass in HTML if you have it. Sometimes this approach may be
    # necessary if you encounter mysterious urllib2.BadStatusLine errors. Here's how
    # that would work:

    # sample_html = requests.get(sample_url).text
    # summary = summarize(html=sample_html)

    print "-------------------------------------------------"
    print "                'Top N Summary'"
    print "-------------------------------------------------"
    print " ".join(summary['top_n_summary'])
    print
    print
    print "-------------------------------------------------"
    print "             'Mean Scored' Summary"
    print "-------------------------------------------------"
    print " ".join(summary['mean_scored_summary'])