import re
import time
import os, glob
import multiprocessing as mp
import ssl
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class Appearance:
    """
    Appearance of word in document, represented in dictionary
    with document ID aling with frequency
    """

    def __init__(self, docId, frequency):
        self.docId = docId
        self.frequency = frequency

    def __repr__(self):
        return str(self.__dict__)

class Inverted_Index:
    """
    Creating inverted index by dividing all files by number of threads and loading them parallely.
    """

    def __init__(self):
        self.index = dict()
        self.indexedDocuments = None

    def create_index(self, path='neg/', threads_num=4):
        """
        Generate variables for processing documents simultaneously and initiate process.
        """
        pathList = list(glob.glob(os.path.join(path,('*.txt'))))

        processes_args = []
        for i in range(threads_num):
            startIndex = int(i * len(pathList) / threads_num )
            endIndex = int((i + 1) * len(pathList) / threads_num )
            processes_args.append((path, startIndex, endIndex))

        pool = mp.Pool(threads_num)
        self.indexedDocuments = pool.starmap(self.prepare_single_document, processes_args)
        self.merge_dicts()

    def merge_dicts(self):
        """
        Merging products of parallel process
        """

        for i in range(len(self.indexedDocuments)):
            update_dict = {key: [appearance]
                    if key not in self.index
                    else self.index[key] + [appearance]
                           for (key, appearance) in self.indexedDocuments[i].items()}
            self.index.update(update_dict)


    @staticmethod
    def prepare_single_document(path, startIndex, endIndex):
        """
        Document cleanup and generating single inverted index. Removing punctuation, lemmatizing , removing stop words.
        """
        directory = list(glob.glob(os.path.join(path,('*.txt'))))
        documents = directory[startIndex:endIndex]

        indexed_group = dict()
        for document in documents:
            with open(document) as f:
                text = f.readlines()

            ## Remove punctuation

            text = ' '.join(text)
            punctuation = re.sub(r'[^\w\s]', '', text)

            ## Tokenize words

            tokenized = word_tokenize(punctuation)

            ## Lemmatize words

            lemmatizer = WordNetLemmatizer()
            lemmatized = [lemmatizer.lemmatize(w) for w in tokenized]

            ## Remove stop words

            stop_words = set(stopwords.words('english'))
            terms = [w.lower() for w in lemmatized if not w.lower() in stop_words]

            ID = document.replace(str(path), '')

            appearances_dict = dict()
            # Dictionary with each term and the frequency it appears in the text.
            for term in terms:
                term_frequency = appearances_dict[term].frequency if term in appearances_dict  else 0
                appearances_dict[term] = Appearance(ID, term_frequency + 1)

            update_dict = {key: [appearance]
                        if key not in indexed_group
                        else indexed_group[key] + [appearance]
                           for (key, appearance) in appearances_dict.items()}
            indexed_group.update(update_dict)
        return (indexed_group)

    def print_index(self):
        """
        Print Inverted Index
        """
        print(self.index)

    def search_word(self, query):
        """
        Search through dictionary mechanism
        """

        print({term: self.index[term] for term in query.split(' ') if term in self.index})


if __name__ == '__main__':
    index = Inverted_Index()
    start_time = time.time()
    index.create_index()
    print("It took %s seconds to create Inverted Index paralelly" % (time.time() - start_time))
    
    while True:
        inp = input("What are you looking for (enter 'q' to quit): ")
        if inp == 'q':
            break
        else :
            index.search_word(inp)
    