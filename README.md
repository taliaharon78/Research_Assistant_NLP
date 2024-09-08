
# Research Ass
## Research Chatbot with an attitude
#### Advanced NLP, HIT, Aug 24
#### Tali Aharon,    034791236
#### Helit Bauberg,  027466002

This application receives a query or a research topic as user input, refines the query, retrieves relevant scientific papers from Google Scholar and Arxiv, 
extracts key findings, and synthesizes the information into a single concise report. 
The application can run in two ways:
- as a notebook executable by calling __main__, or 
- as a chatbot in Web environment, by launching Gradio web demo.

We have used langchain's API to invoke calls to Gemini-1.0-pro. If you wish to change the LLM, method get_llm should be updated accordingly. Gemini calls are cost-free, but an API key should be generated to access the service.


## Prerequisites

- Google GenAI API key - to enable calls to Gemini. Can be generated here: https://ai.google.dev/gemini-api
- SERP API key - to enable calls to Google Scholar. Free plan supports up to 100 searches per month. Can be generated here: https://serpapi.com/manage-api-key
Note: If Scholar search is unavailable, the bot will return only Arxiv results.  


## Environment Setup

Make sure you run the enviroment setup section in the notebook to install necessary modules, setup Google drive path, and provide access to API keys.
- Google Drive relative path is set to:
  GOOGLE_DRIVE_PATH_AFTER_MYDRIVE = 'Colab Notebooks/NLP_RA' change it if need be.
- API keys are loaded from colab secrets (and not from .env) 
  instructions are here: https://medium.com/@parthdasawant/how-to-use-secrets-in-google-colab-450c38e3ec75


