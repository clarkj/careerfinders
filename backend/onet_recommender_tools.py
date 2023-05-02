import json
from collections import defaultdict
from collections import Counter
import json
import math
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import glassdoor_search as gs


# --------------- EXAMPLE STRUCTURE OF ONET DATA --------------- #
'''
45-2041.00 {
    'occupation': 'Graders and Sorters, Agricultural Products', 
    'values': [('Working_Conditions', None), ('Support', None)], 
    'knowledge': [('Administrative', 16), ('Engineering_and_Technology', 23), ('Administration_and_Management', 32), ('English_Language', '51'), ('Biology', '10'), ('Philosophy_and_Theology', '8'), ('Psychology', 10), ('Customer_and_Personal_Service', 28), ('Law_and_Government', 9), ('Computers_and_Electronics', 25), ('Medicine_and_Dentistry', '17'), ('Building_and_Construction', '8'), ('Food_Production', '45'), ('Public_Safety_and_Security', 33), ('History_and_Archeology', '6'), ('Economics_and_Accounting', 7), ('Geography', 6), ('Therapy_and_Counseling', 9), ('Chemistry', '18'), ('Mathematics', 27), ('Communcations_and_Media', 12), ('Physics', '14'), ('Telecommunications', 6), ('Sociology_and_Anthropology', 5), ('Fine_Arts', '5'), ('Sales_and_Marketing', 21), ('Personnel_and_Human_Resources', 24), ('Design', 5), ('Mechanical', 50), ('Education_and_Training', 43), ('Production_and_Processing', 66), ('Foreign_Language', 34), ('Transportation', 24)], 
    'interests': [('Realistic', None)], 
    'cross-skills': [('Programming', 0), ('Persuasion', 22), ('Equipment Maintenance', 0), ('Quality Control Analysis', 22), ('Operation and Control', 13), ('Service Orientation', 22), ('Instructing', 16), ('Coordination', 38), ('Judgment and Decision Making', 31), ('Troubleshooting', 13), ('Installation', 0), ('Management of Financial Resources', 3), ('Equipment Selection', 0), ('Complex Problem Solving', 28), ('Negotiation', 19), ('Operations Monitoring', 16), ('Systems Evaluation', 16), ('Management of Personnel Resources', 10), ('Management of Material Resources', 3), ('Social Perceptiveness', 28), ('Operations Analysis', 0), ('Technology Design', 0), ('Repairing', 0), ('Systems Analysis', 22), ('Time Management', 31)] 
}
inverted_index[term] = [(job1, tf1), (job2, tf2), ...]
'''

default_lemmmatizer = WordNetLemmatizer()

def inverted_index(jobs, default_attribute_item_score=75):
    """ Builds an inverted index ordering ONET jobs for each skill, knowledge, etc...
    
    Arguments
    =========
    
    jobs: dict
        keys: job code
        vals: dictionary e.g.
            {
                'occupation': 'Graders and Sorters, Agricultural Products', 
                'values': [('Working_Conditions', None), ('Support', None)], 
                'knowledge': [('Administrative', 16), ('Engineering_and_Technology', 23), ('Administration_and_Management', 32), ('English_Language', '51'), ('Biology', '10'), ('Philosophy_and_Theology', '8'), ('Psychology', 10), ('Customer_and_Personal_Service', 28), ('Law_and_Government', 9), ('Computers_and_Electronics', 25), ('Medicine_and_Dentistry', '17'), ('Building_and_Construction', '8'), ('Food_Production', '45'), ('Public_Safety_and_Security', 33), ('History_and_Archeology', '6'), ('Economics_and_Accounting', 7), ('Geography', 6), ('Therapy_and_Counseling', 9), ('Chemistry', '18'), ('Mathematics', 27), ('Communcations_and_Media', 12), ('Physics', '14'), ('Telecommunications', 6), ('Sociology_and_Anthropology', 5), ('Fine_Arts', '5'), ('Sales_and_Marketing', 21), ('Personnel_and_Human_Resources', 24), ('Design', 5), ('Mechanical', 50), ('Education_and_Training', 43), ('Production_and_Processing', 66), ('Foreign_Language', 34), ('Transportation', 24)], 
                'interests': [('Realistic', None)], 
                'cross-skills': [('Programming', 0), ('Persuasion', 22), ('Equipment Maintenance', 0), ('Quality Control Analysis', 22), ('Operation and Control', 13), ('Service Orientation', 22), ('Instructing', 16), ('Coordination', 38), ('Judgment and Decision Making', 31), ('Troubleshooting', 13), ('Installation', 0), ('Management of Financial Resources', 3), ('Equipment Selection', 0), ('Complex Problem Solving', 28), ('Negotiation', 19), ('Operations Monitoring', 16), ('Systems Evaluation', 16), ('Management of Personnel Resources', 10), ('Management of Material Resources', 3), ('Social Perceptiveness', 28), ('Operations Analysis', 0), ('Technology Design', 0), ('Repairing', 0), ('Systems Analysis', 22), ('Time Management', 31)] 
            }
    
    Returns
    =======
    
    inverted_index: dict
        For each term (job attribute item like skill, interest, knowledge), the index contains 
        a sorted list of tuples (job_id, importance-score)
        such that tuples with smaller job_ids appear first:
        inverted_index[job_attribute_item] = [(job_id1, importance-score1), (job_id2, importance-score2), ...]
        
    """
    inv_idx = {}

    for job_code, attributes_dict in jobs.items():
        for attr, attribut_items in attributes_dict.items():
                if attr == "occupation": continue
                
                for attr_item, importance in attribut_items:
                    attr_item = attr_item.lower().replace('_', ' ')
                    if attr_item not in inv_idx:
                        inv_idx[attr_item] = []
                    inv_idx[attr_item].append((job_code, importance if importance else default_attribute_item_score))
        
        for attr_item in inv_idx:
            inv_idx[attr_item].sort(key=lambda x: x[1], reverse = True)
    
    return inv_idx

def job_to_idx(jobs):
    job_idx_map = {}
    
    counter = 0
    for code in jobs.keys():
        job_idx_map[code] = counter
        counter += 1

    return job_idx_map

def skill_to_idx(inv_idx): #skill is attribute item
    skil_idx_map = {}
    
    counter = 0
    for s in inv_idx.keys():
        skil_idx_map[s] = counter
        counter += 1

    return skil_idx_map

def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    """ Compute term IDF values from the inverted index.
    Words that are too frequent or too infrequent get pruned.
    
    Arguments
    =========
    
    inv_idx: an inverted index as above
    
    n_docs: int,
        The number of documents.
        
    min_df: int,
        Minimum number of documents a term must occur in.
        Less frequent words get ignored. 
        Documents that appear min_df number of times should be included.
    
    max_df_ratio: float,
        Maximum ratio of documents a term can occur in.
        More frequent words get ignored.
    
    Returns
    =======
    
    idf: dict
        For each term, the dict contains the idf value.
        
    """

    idf = {}
    
    # key: skills, values: list of (job_id, score) tuples 
    for skill, postings in inv_idx.items():
        # dft: document frequency for this skill (term)
        dft = len(postings)
        c = n_docs / (1 + dft)
        idf[skill] = math.log2(c) ## is boutnd to be 0 for almost all of the itmes becasue the skills appear in all the docs!!!
        
    return idf

def compute_doc_norms(inv_idx, idf, n_docs, job_idx_map):
    """ Precompute the euclidean norm of each document.
    
    Arguments
    =========
    
    index: the inverted index as above
    
    idf: dict,
        Precomputed idf values for the terms.
    
    n_docs: int,
        The total number of jobs
    
    Returns
    =======
    
    norms: np.array, size: n_docs
        norms[i] = the norm of job i.
    """
    norms = np.zeros(n_docs)

    for skill, postings in inv_idx.items():

        idf_k = idf.get(skill)
        if idf_k:

            for job_id, score in postings:
                job_idx = job_idx_map.get(job_id)
                norms[job_idx] += (float(score) * idf_k) **2
    
    return np.sqrt(norms)

def dot_scores(query_word_counts, inv_idx, idf):
    """ Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.
    
    Arguments
    =========
    
    query_word_counts: dict,
        A dictionary containing all words that appear in the query;
        Each word is mapped to a count of how many times it appears in the query.
        In other words, query_word_counts[w] = the term frequency of w in the query.
        You may safely assume all words in the dict have been already lowercased.
    
    index: the inverted index as above,
    
    idf: dict,
        Precomputed idf values for the terms.
    
    Returns
    =======
    
    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """
    doc_scores = {}
    
    for term, freq in query_word_counts.items():
        if freq == 0:
            continue
        # get list of career tuples for given skill
        career_tups = inv_idx.get(term)
        # get idf score for given skill
        idf_k = idf.get(term)
        # make sure that this term wasn't pruned from inv_idx and that career_tups isn't empty
        # if idf_k and career_tups:
        #     for job_id, score in career_tups:

        #         # dot = (freq * idf_k) * (score * idf_k) 
        #         doc_scores[job_id] = doc_scores.get(job_id, 0) + dot 
        if not career_tups:
            continue

        for job_id, score in career_tups:
            try:
                doc_scores[job_id] += freq * score
            except:
                doc_scores[job_id] = freq * score
        
    return doc_scores

def index_search(query, index, idf, doc_norms, job_idx_map, score_func=dot_scores, tokenizer=RegexpTokenizer(r'\w+')):
    """ Search the collection of documents for the given query
    
    Arguments
    =========
    
    query: string,
        The query we are looking for.
    
    index: an inverted index as above
    
    idf: idf values precomputed as above
    
    doc_norms: document norms as computed above
    
    score_func: function,
        A function that computes the numerator term of cosine similarity (the dot product) for all documents.
        Takes as input a dictionary of query word counts, the inverted index, and precomputed idf values.
    
    tokenizer: a TreebankWordTokenizer
    
    Returns
    =======
    
    results, list of tuples (score, job_id)
        Sorted list of results such that the first element has
        the highest score, and `job_id` points to the document
        with the highest score.
    
    Note: 
        
    """
    q = tokenizer.tokenize(query)
    q_counts = Counter(q)
    doc_scores = score_func(q_counts, index, idf)

    # ---- This section is for normalizing query (q_norm) ---- #
    q_norm = 0
    # key = term, val = count
    for k, v in q_counts.items():
        idf_k = idf.get(k)
        
        if idf_k: 
            q_norm += (idf_k * v) ** 2

    q_norm = math.sqrt(q_norm)
    # ---- This section is for normalizing query (q_norm) ---- #
    
    results = []
    
    for job_id, v in doc_scores.items():
        job_idx = job_idx_map.get(job_id)

        try:
            numer = int(v)
            denom = q_norm * doc_norms[job_idx]
            sim_score = numer / max(denom, 1)
            results.append((sim_score, job_id))
        except:
            print("failed", job_id, v)
        
    results = sorted(results, key=lambda x: x[0], reverse=True)
    return results

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
    
def top10_results(query, jobs, inv_idx, idf, doc_norms, job_idx_map):

    print("#" * len(query))
    print(query)
    print("#" * len(query))

    results = index_search(query, inv_idx, idf, doc_norms, job_idx_map)
    count = 0
    output = []
    for score, job_id in results:
        if count == 10:
            break
        count += 1
        
        get_job = jobs[job_id]
        occupation = get_job['occupation']
        top = get_job['cross-skills'] + get_job['knowledge']
        top.sort(key=lambda x:int(x[1]), reverse=True)

        review = None
        try:
            review = gs.match_job_title(occupation)
        except Exception as e:
            pass

        result = {
            'score': score,
            'job': occupation,
            'top10': top[:10],
            'review': review,
        }

        result = json.dumps(result, default=np_encoder)
        output.append(result)

    return output

def query_to_skills(query, lemmatizer=default_lemmmatizer):
    query_words = word_tokenize(query)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in set(stopwords.words('english'))]

    return words

















