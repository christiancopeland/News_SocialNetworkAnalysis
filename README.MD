### Purpose
- Dynamically generate contextualized Social Network Graph Visualizations from News Articles

### Usage 
- If generating network graph for the first time, provide txt file of article text
    - ```python sna.py --article_text=filename```
- If generating network graph on an article you've run thru the program before, and just want to see the visualization using the network json created previously
    - ```python sna.py --load_file=network_dict.json```