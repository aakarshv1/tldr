import cohere
import grab_comments as c
from random import shuffle

prompt = "The following text is a list of comments under a YouTube video, following by a summary of the concensus of these comments: \n"
l = [i for i in range(len(c.response['items']))]
shuffle(l)
count = 0
for i in l[:48]: 
    prompt += c.response['items'][i]['snippet']['topLevelComment']['snippet']['textDisplay']
    prompt += ". "
    count += 1
prompt += "Summary: "

co = cohere.Client('Eo7mYfDMa98LpR22O2g2OcrnBNumnd1JuEEn2RyD')
response = co.generate(
  model='large',
  prompt=prompt,
  max_tokens=20,
  temperature=0.8,
  k=0,
  p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop_sequences=["--"],
  return_likelihoods='NONE')
print(prompt)
print('{}'.format(response.generations[0].text))