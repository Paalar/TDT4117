import random; random.seed(123)
import codecs
import string
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities
from pprint import pprint 

#Returns a list of 'white characters' that are supposed to be removed from the book
def getWhiteCharacters():
    whiteCharacters = []
    for character in string.punctuation:
        whiteCharacters.append(character)
    whiteCharacters = whiteCharacters + ['\n', '\r', '\t']
    return whiteCharacters

whiteCharacters = getWhiteCharacters()
stemmer = PorterStemmer()
wordsToBeRemoved = ['gutenberg']

#Parses the text file into a string list for each paragraph and the paragraphs is put into another list, book
def parseBookToList(file):
    book = [] # The 'book' that will be used for term weighting
    untouchedBook = [] #The book that is used to retrieve the untouched texts
    paragraph = []
    untouchedParagraph = []
    for line in file:
        if len(line.strip()) == 0:
            if len(paragraph) > 0:
                book.append(paragraph)
                untouchedBook.append(untouchedParagraph)
            paragraph = []
            untouchedParagraph = []
        else:
            newParagraph = line.split() #Creates a list from each line
            if (removeParagraphsContaining(newParagraph)): #Checks if the paragraph contains any unwanted words
                untouchedParagraph += newParagraph
                for index in range(len(newParagraph)): #Iterates through every word
                    word = iterateWord(newParagraph[index])
                    newParagraph[index] = word
                paragraph += newParagraph
    return book, untouchedBook

def iterateWord(word):
    word = word.lower() #Lowercase
    word = removeWhiteCharacters(word) #Removing white characters
    word = stemming(word) #Stems the word, reducing it to its core
    return word

def removeWhiteCharacters(word):
    for character in whiteCharacters:
        if character in word:
            word = word.replace(character, '')
    return word

def stemming(word):
    stemmedWord = stemmer.stem(word)
    return stemmedWord

#Removes a paragraph if the word in the list of words is in a paragraph
def removeParagraphsContaining(paragraph):
    counter = 0
    for word in paragraph:
        for badword in wordsToBeRemoved:
            word = word.lower()
            if badword in word:
                counter += 1
    return counter == 0

#Given a list of ids and dictionary, create stop words and return their ids
def getStopIds(file, dictionary):
    stopwords = []
    for line in file:
        stopwords += line.split(',')
    stop_ids = [dictionary.token2id[stopword] for stopword in stopwords if stopword in dictionary.token2id]
    return stop_ids

# Counts number of unique words and their occurences
def getBagOfWords(dictionary, book):
    bagOfWords = []
    for paragraph in book:
        bag = dictionary.doc2bow(paragraph)
        bagOfWords.append(bag)
    return bagOfWords

def printParagraph(paragraph, index, value):
    print('[Paragraph {0}]\n[Value: {1}'.format(index, value))
    lines = 0
    text = ""
    for word in paragraph:
        if ('?' in word or '!' in word or '.' in word):
            lines += 1
        if lines == 5:
            break
        text += ' {0}'.format(word)
    text+='\n'
    print(text)

def iterateSims(similarity, untouchedBook):
    sims = sorted(enumerate(similarity), key=lambda item: -item[1])[:3]
    for index, value in sims:
        printParagraph(untouchedBook[index], index - 1, value)

def __main__():
    #Opens the files used, and parses the book
    bookFile = codecs.open("pg3300.txt", "r", "utf-8")
    stopWordsFile = codecs.open("stopwords.txt", "r", "utf-8")
    book, untouchedBook = parseBookToList(bookFile)
    
    dictionary = corpora.Dictionary(book) #Creates a dictionary from the book
    stop_ids = getStopIds(stopWordsFile, dictionary) #Getting the stop word ids
    dictionary.filter_tokens(stop_ids) #remove the stop words from the dictionary
    dictionary.compactify() 

    corpus = getBagOfWords(dictionary, book) #Bag of words
    tfidf_model = models.TfidfModel(corpus) #Initializes a tf-idf model
    tfidf_corpus = tfidf_model[corpus] #Creates the tfidf corpus
    tfidf_index = similarities.MatrixSimilarity(tfidf_corpus) #Creates the tfidf index

    lsi_model = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100) #Creates the lsi model
    lsi_corpus = lsi_model[tfidf_corpus] #Creates the lsi corpus
    lsi_index = similarities.MatrixSimilarity(lsi_corpus) #creates the lsi index

    while True:
        userQuery = input("User query: ")
        processedQuery = []
        if userQuery == 'break':
            break
        queryList = userQuery.split()
        for word in queryList:
            processedQuery.append(iterateWord(word))
        queryCorpus = dictionary.doc2bow(processedQuery)

        lsi_query = lsi_model[queryCorpus]
        tfidf_query = tfidf_model[queryCorpus]

        print("\nTFIDF-Model query")
        tfidf_sims = tfidf_index[tfidf_query]
        iterateSims(tfidf_sims, untouchedBook)

        print("\nLSI-Model query")
        lsi_sims = lsi_index[lsi_query]
        iterateSims(lsi_sims, untouchedBook)


        #How taxes influence Economics?
        #What is the function of money?

    


if __name__ == '__main__':
    __main__()
