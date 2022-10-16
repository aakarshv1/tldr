from scholarly import scholarly
import cohere
import grab_comments as c


co = cohere.Client('Eo7mYfDMa98LpR22O2g2OcrnBNumnd1JuEEn2RyD')

inp = input("Search: ")
inp2 = input("Number of queries: ")
search_query = scholarly.search_pubs(inp)
#scholarly.pprint((next(search_query)))
c = 0
prompt= 'The following text consists of Passages followed by a short Summary: \n\n Passage: Is Wordle getting tougher to solve? Players seem to be convinced that the game has gotten harder in recent weeks ever since The New York Times bought it from developer Josh Wardle in late January. The Times has come forward and shared that this likely isn’t the case. That said, the NYT did mess with the back end code a bit, removing some offensive and sexual language, as well as some obscure words There is a viral thread claiming that a confirmation bias was at play. One Twitter user went so far as to claim the game has gone to “the dusty section of the dictionary” to find its latest words.\n\nSummary: Wordle has not gotten more difficult to solve.\n--\nPassage: ArtificialIvan, a seven-year-old, London-based payment and expense management software company, has raised $190 million in Series C funding led by ARG Global, with participation from D9 Capital Group and Boulder Capital. Earlier backers also joined the round, including Hilton Group, Roxanne Capital, Paved Roads Ventures, Brook Partners, and Plato Capital.\n\nSummary: ArtificialIvan has raised $190 million in Series C funding. \nPassage: '
while c < int(inp2):
    
    try:
        a = (next(search_query)['bib']['abstract'])
    except StopIteration:
        print("No more queries :(")
    length = len(a)
    if length > 150:
        prompt += a
        prompt += "\n Summary:"
        response = co.generate(
            model='xlarge',
            prompt=prompt,
            max_tokens=50,
            temperature=0.8,
            k=0,
            p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=["--"],
            return_likelihoods='NONE')
        #print(f'Abstract({length}):',a, '\n')
        a += '\nPassage: '
        print('{}\n\n'.format(response.generations[0].text))
        c += 1
