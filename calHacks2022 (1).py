import cohere
import matplotlib.pyplot as plt
import numpy as np

co = cohere.Client('LQLc5N6PhOoUcfrGeEZOVx3fOGIcI1Ttgr0gHp9W')

def classify(input, examples, modelSize='large'):
    """
    Classifies a given input based on examples
    Must provide at a list of Examples of length at least 5
    """
    assert len(examples) >= 5
    if type(input) is str:
        input = [input]
    response = co.classify(
        inputs = input,
        examples = examples,
        model = modelSize,
    )
    return response

def barGraph(courses, values, xLabel, yLabel):
    fig = plt.figure(figsize = (10, 5))
    plt.bar(courses, values, color ='maroon',
        width = 0.4)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()

def articleSummarize(prompt, modelSize='large'):
    """
    Summarizes a passage or a list of strings and returns a response object
    """
    input = 'Passage: Is Wordle getting tougher to solve? Players seem to be convinced that the game has gotten harder in recent weeks ever since The New York Times bought it from developer Josh Wardle in late January. The Times has come forward and shared that this likely isn’t the case. That said, the NYT did mess with the back end code a bit, removing some offensive and sexual language, as well as some obscure words There is a viral thread claiming that a confirmation bias was at play. One Twitter user went so far as to claim the game has gone to “the dusty section of the dictionary” to find its latest words.\n\nTLDR: Wordle has not gotten more difficult to solve.\n--\nPassage: ArtificialIvan, a seven-year-old, London-based payment and expense management software company, has raised $190 million in Series C funding led by ARG Global, with participation from D9 Capital Group and Boulder Capital. Earlier backers also joined the round, including Hilton Group, Roxanne Capital, Paved Roads Ventures, Brook Partners, and Plato Capital.\n\nTLDR: ArtificialIvan has raised $190 million in Series C funding.\n--\n'
    if type(prompt) is list:
        input = 'Passage: '
        for string in prompt:
            input += string
        input += '/nTLDR'
    else:
        input += (f'Passage: {prompt} \n\nTLDR:')
    response = co.generate(
        model = modelSize,
        prompt = input,
        max_tokens=200, 
        temperature=0.8, 
        k=0, 
        p=1, 
        frequency_penalty=0, 
        presence_penalty=0, 
        stop_sequences=["--"], 
        return_likelihoods='NONE')
    return response

#classify demo
examples = [cohere.classify.Example("Elon Musk says Twitter Blue subscribers should be able to pay with dogecoin", "Business news"), 
            cohere.classify.Example("Probability of a US recession in the next 12 months, via WSJ", "Business news"), 
            cohere.classify.Example("European futures slide", "Business news"), 
            cohere.classify.Example("NASDAQ rises 2% to ATH", "Business news"), 
            cohere.classify.Example("FTX Founder is one of the world\'s richest crypto billionaires, with a fortune valued at $20 billion.", "Business news"), 
            cohere.classify.Example("Sweet Potato Macaroni Cheese is #RecipeOfTheDay, and I’m very happy about it!", "Cooking"), 
            cohere.classify.Example("3-Ingredient Slow Cooker recipes", "Cooking"), 
            cohere.classify.Example("This is by far the BEST biscuit recipe I’ve ever tried", "Cooking"), 
            cohere.classify.Example("Baking my first loaf of banana bread...", "Cooking"), 
            cohere.classify.Example("From the queen of Italian cooking, this is one of the most iconic tomato sauce recipes ever", "Cooking"), 
            cohere.classify.Example("I’ve actually read this book and it was extremely insightful. A quick read as well and available as a free audiobook through many libraries.", "Arts & Culture"), 
            cohere.classify.Example("Today’s Daily Cartoon", "Arts & Culture"), 
            cohere.classify.Example("Get a glimpse of the stage adaptation of Hayao Miyazaki’s 2001 Oscar-winning animated feature Spirited Away", "Arts & Culture"), 
            cohere.classify.Example("The #Banksy Exhibit in Cambridge, MA is absolutely terrific.", "Arts & Culture"), 
            cohere.classify.Example("“A Whisper in Time” large abstract paining 48’ x 48’", "Arts & Culture")]
sampleText = 'Any decent trader spends 90% of their efforts exploring how they could be wrong. That should apply to everyone’s decision making.'
print(f'==Classify Demo==\n{classify(sampleText, examples).classifications}\n\n')

#summarize demo
prompt='The National Weather Service announced Tuesday that a freeze warning is in effect for the Bay Area, with freezing temperatures expected in these areas overnight. Temperatures could fall into the mid-20s to low 30s in some areas. In anticipation of the hard freeze, the weather service warns people to take action now'
print(f'\n\n==Summarize Demo==\nSummarization: {articleSummarize(prompt).generations[0].text}\n\n')

#bar graph demo
barGraph(['Postive','Neutral','Negative'],
        [57,23,34],
        'Categories',
        'Counts')