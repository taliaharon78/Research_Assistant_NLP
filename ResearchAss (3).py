from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from google.colab import userdata
from serpapi import GoogleScholarSearch
from langchain_community.retrievers import ArxivRetriever
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

import json 


def get_llm(temperature = 0.01):
  
  GOOGLE_API_KEY=userdata.get('GOOGLE_KEY')
  genai.configure(api_key=GOOGLE_API_KEY)
  
  llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=temperature, google_api_key=GOOGLE_API_KEY)
  return llm

#  Use LLM to produce search queries based on the user's input
def get_queries(user_input: str) -> str:

  prompt = f"""
  You are a knowledgeable research assistant. Our reaserch question is: {user_input}.
  Come up with 2 - 3 concise search queries for paper lookup on GoogleScholar. 
  Reply in the following format. Do not use dashes or brackets:
  
  <search query 1>
  <search query 2>
  <search query 3>
  
  """
  llm_output = get_llm(0.5).invoke(prompt)
  #print(llm_output.content)
  return llm_output.content



# Search query in Google Scholar 
# Only papers with Arxiv ID are retrieved.
def search_gscholar(query: str, top_k = 10):

  SERP_API_KEY = userdata.get('GUY_SERP_KEY')
  search = GoogleScholarSearch({
      "q": query, 
      "api_key": SERP_API_KEY,
      "hl": "en",
      "num": top_k
    })
  result = search.get_dict()

  paper_ids = []
  if 'organic_results' in result:
    for paper in result['organic_results']:
      #print(paper['position'], paper['title'], paper['link'])  
      if 'arxiv' in paper['link']:
        id = paper['link'].split("/abs/")[1]
        #print(id)
        paper_ids.append(id)
  else:
    print(f"No GScholar results for {query}")
  #return ', '.join(paper_ids)
  return paper_ids

# Search Arxiv for either a querry or a list of IDs. 
def search_arxiv(query_or_ids) -> list:
    arxiv = ArxivRetriever(top_k_results=5)
    all_docs = []

    if isinstance(query_or_ids, str):
        # If input is a string, treat it as a query
        docs = arxiv.invoke(query_or_ids)
        all_docs.extend(docs)
    elif isinstance(query_or_ids, list):
        # If input is a list, treat it as a list of Arxiv IDs
        for query in query_or_ids:
            docs = arxiv.invoke(query)
            all_docs.extend(docs)
    else:
        raise ValueError("Input should be either a string (query) or a list of IDs")

    print("Total Arxiv doc count:", len(all_docs))
    return all_docs

# Check combined list of Gscholar and Arxiv IDs, remove duplicates if any.
def remove_duplicates(ids: list) -> list:
  
  id_set = set()
  unique = []
    
  for id_ in ids:
    if id_ in id_set:
      print(f"Duplicate found! {id_}")       
    else:
      id_set.add(id_)  
      unique.append(id_)
  
  #print(f"Added {len(unique)} unique records")
  return unique

  
#  Use search queries to retrieve matching papers from Arxiv, querrying both
#  Arxiv and GoogleScholar. A combined list of all queries returned, duplicates removed.
def get_papers(queries: str) -> list:

  q_list = queries.split('\n')
  gs_ids = []
  arx_papers = []
  
  # iterate through query list, get papers' Arxiv IDs from Google Scholar results
  # and Arxiv 
  for query in q_list:
    arx_papers = arx_papers + search_arxiv(query)
    gs_ids = gs_ids + search_gscholar(query)
    print(query, gs_ids, len(arx_papers))

  id_list = remove_duplicates(gs_ids)

  # if Google Scholar yielded any Arxiv results, we retrieve those papers and add them
  if len(id_list) > 0:
    arx_papers = arx_papers + search_arxiv(', '.join(id_list))

  return arx_papers 

# Convert Documents list to a str
def doc_to_str(papers: list) -> str:
  
  paperstr = ''
  # Iterate through the search results and output IDs and Titles
  i=1
  for paper in papers:
    paperstr = paperstr + str(i) + ':' + paper.metadata['Entry ID'] + '\n' + paper.metadata['Title'] + '\n'
    paperstr = paperstr + paper.page_content + '\n\n'
    #print(i, ':', paper.metadata['Entry ID'], '\n', paper.metadata['Title'], '\n')
    #print(paper.page_content)
    #print('\n\n')
    i += 1
  
  print(paperstr)
  return paperstr


# basic semantic router for proceed or restart. TBD more options
def llm_rephrase(msg) -> str:
  
  prompt = f"""
  Rephrase the given prompt. Capture the same content but use an arrogant, ironic style in your reply. Yet, try to be brief and clear.
  
  {msg}
  """
  llm_output = get_llm(0.8).invoke(prompt)
  #print(llm_output.content)
  return llm_output.content


# basic semantic router for proceed or restart. TBD more options
def ask_llm(user_input) -> str:
  prompt = f"""
  Analyze the user input as follows: {user_input} 
  If the user wants to continue with report generation, reply CONT.
  Else, reply BREAK
  """
  #print(prompt)
  llm_output = get_llm().invoke(prompt)
  #print(llm_output.content)
  return llm_output.content

def get_entries_from_prompt(user_input, total_count):

  prompt = f"""
  We have a list of {total_count} papers. 
  Search the user input for keywords like "remove," "ignore," or "delete."  Their presence suggests an intention to exclude papers.
  Extract all numbers from the user input. For each extracted number, Check if it's within the valid range (1 to the {total_count}).
  If valid, add it to the exclusion list.
  If no valid exclusion numbers are found, reply with "CONT" to indicate continuing with all papers.
  If there are valid exclusion numbers, reply with a comma-separated list of those numbers.
  Let me illustrate with examples:
  **Example 1:**   
  * User input: "Please exclude paper number 5 and 10"   
  * Keywords found: "exclude"  
  * Extracted numbers: 5, 10
  * Valid numbers: 5, 10 (both within 1-10 range)
  * Reply: **5, 10**
  **Example 2:**',
  * User input: "5, 11"',
  * Keywords found: None (but the presence of numbers suggests exclusion)',
  * Extracted numbers: 5, 11',
  * Valid numbers: 5 (11 is greater than 10)',
  * Reply: **5**',
  **Example 3:**',
  * User input: "ignore 6"',
  * Keywords found: "ignore"',
  * Extracted numbers: 6',
  * Valid numbers: 6',
  * Reply: **6**',
  **Example 4:**',
  * User input: "I am happy with all my papers"',
  * Keywords found: None',
  * Extracted numbers: None',
  * Valid numbers: None',
  * Reply: **CONT** '
  Your output should only be the Reply, no comments or explanations.  

  user input:
  {user_input}
  """

  llm_output = get_llm().invoke(prompt)
  return llm_output.content


# Iterate through the search results and output IDs, Titles and abstract. Get user input.
def screen_papers(papers: list, user_input) -> str:

  entries = get_entries_from_prompt(user_input, str(len(papers)))
  if 'CONT' not in entries:
    response = "Sure. Excluding papers " + entries + '\n'
    entries = entries.split(',')
    for entry in entries:
      if int(entry)-1 < len(papers):
        del papers[int(entry)-1]
        #print("Excluded entry #", int(entry)-1, '\n')
  else: 
    response = "Cool. No papers to exclude.\n"
  prompt = "If you wish to proceed, it will take long time."
  llm_prompt = llm_rephrase(prompt)
  response = response + "May I proceed to generate the report, master?\n" + llm_prompt   
  return response, papers


# Produce the synthesised report from all papers, using langchain summarize-refine chain
def generate_report(papers: list) -> str:

  prompt_template = """In about 500 words, summarize key findings and main theme of the following:
  {text}
  CONCISE SUMMARY:"""
  prompt = PromptTemplate.from_template(prompt_template)

  refine_template = (
      "Your job is to produce a final summary\n"
      "We have provided an existing summary up to a certain point: {existing_answer}\n"
      "We have the opportunity to refine the existing summary"
      "(only if needed) with some more context below.\n"
      "------------\n"
      "{text}\n"
      "------------\n"
      "Given the new context, refine the original summary. "
      "If the context isn't useful, return the original summary."
  )
  refine_prompt = PromptTemplate.from_template(refine_template)
  chain = load_summarize_chain(
      llm=get_llm(),
      chain_type="refine",
      question_prompt=prompt,
      refine_prompt=refine_prompt,
      return_intermediate_steps=True, 
      input_key="input_documents",
      output_key="output_text",
  )

  # Gemini-1.0-pro max token is 16k. Using half 
  text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
      chunk_size=8000, chunk_overlap=0
    )
    
  split_docs = text_splitter.split_documents(papers)

  result = chain.invoke({"input_documents": split_docs}, return_only_outputs=False)

  return result["output_text"]


# Generate reference list from curated papers
def add_references(papers: list) -> str:

  ref_list = "\n\n** References: ** \n"

  for paper in papers:
    ref = '*' + paper.metadata['Title'] + ', ' + paper.metadata['Authors'] + ', ' + paper.metadata['Entry ID'] + '\n'
    ref_list = ref_list + ref

  return ref_list #.splitlines()




