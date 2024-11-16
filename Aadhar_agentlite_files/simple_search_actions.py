import wikipedia
import duckduckgo_search

class DuckSearch:
    def __init__(self):
        self.ddgs = duckduckgo_search.DDGS()

    def search(self, query):
        return self.ddgs.chat(query)
    
class WikipediaSearch:
    def search(self, query):
        search_results = wikipedia.search(query)
        if not search_results:
            return "No results found."
        article = wikipedia.page(search_results[0])
        return article.summary
