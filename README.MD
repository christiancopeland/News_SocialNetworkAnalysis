### Purpose
- Dynamically generate contextualized Social Network Graph Visualizations from News Articles

### Usage 
- Make sure you have [Ollama](https://ollama.com/) installed, and provide the model name you want to use to create the network data in generate_network_data():
    - ``` python
        data = {
        "model": "qwen2.5-coder:14b",
        "messages": messages,
        "stream": stream,
        "format": json_schema,
        "options": {
            "temperature": 0  # Lower temperature for more consistent outputs
        }
        }
    ```


- ```pip install -r requirements.txt```

- If generating network graph for the first time, provide txt file of article text
    - ```python sna.py --article_text=filename```
- If generating network graph on an article you've run thru the program before, and just want to see the visualization using the network json created previously
    - ```python sna.py --load_file=network_dict.json```

### Example Network Data

``` json
{
    "entities": [
        {
            "name": "Cradock Four",
            "type": ["person","organization","policy","issue","impact","location","event"],
            "subtype": "activist"
        , "description": "Four activists who were killed during the apartheid era in South Africa. Their deaths have never been fully investigated, leading to ongoing interest and allegations of government complicity."
        },
        {
            "name": "Economic Freedom Fighters (EFF)",
            "type": ["person","organization","policy","issue","impact","location","event"],
            "subtype": "politician"
        , "description": "A left-wing opposition party in South Africa that has accused the ANC government of freeing convicted perpetrators and being lenient with apartheid-era violence cases."
        },
        {
            "name": "National Prosecuting Authority (NPA)",
            "type": ["person","organization","policy","issue","impact","location","event"],
            "subtype": "government"
        , "description": "The government body responsible for prosecuting crimes in South Africa. It has been criticized for not reopening investigations into apartheid-era cases."
        },
        {
            "name": "Truth and Reconciliation Commission (TRC)",
            "type": ["person","organization","policy","issue","impact","location","event"],
            "subtype": "government"
        , "description": "A commission established after the end of apartheid in South Africa to investigate human rights violations during the apartheid era and provide a platform for victims to share their stories."
        },
        {
            "name": "Webber Wentzel",
            "type": ["person","organization","policy","issue","impact","location","event"],
            "subtype": "government"
        , "description": "A law firm representing the families of the Cradock Four in their lawsuit against the South African government."
        },
        {
            "name": "Nelson Mandela",
            "type": ["person","organization","policy","issue","impact","location","event"],
            "subtype": "politician"
        , "description": "The first democratically elected president of South Africa, who played a key role in ending apartheid and bringing about reconciliation."
        },
        {
            "name": "Thembisile Makhulu",
            "type": ["person","organization","policy","issue","impact","location","event"],
            "subtype": "politician"
        , "description": "The current president of South Africa, who has been accused by the EFF of being lenient with apartheid-era violence cases."
        },
        {
            "name": "Highgate Hotel Massacre",
            "type": ["person","organization","policy","issue","impact","location","event"],
            "subtype": "activist"
        , "description": "A mass shooting that occurred in 1993 at the Highgate Hotel in East London, South Africa. Five people were killed and five others were injured. No one was ever arrested or investigated until 2023."
        },
        {
            "name": "Nokuthula Simelane",
            "type": ["person","organization","policy","issue","impact","location","event"],
            "subtype": "activist"
        , "description": "A victim of apartheid-era violence who was killed in 1983. Her sister, Thembi Nkadimeng, is a current housing minister in South Africa."
        },
        {
            "name": "Colonel Eugene de Kock",
            "type": ["person","organization","policy","issue","impact","location","event"],
            "subtype": "activist"
        , "description": "A former assassin who was initially sentenced to life in prison for his involvement in apartheid-era violence. He was granted parole in 2015 under the Ramaphosa government."
        },
        {
            "name": "South Africa",
            "type": ["person","organization","policy","issue","impact","location","event"],
            "subtype": "government"
        , "description": "A country located in southern Africa that has a history of apartheid and is currently working towards reconciliation and addressing human rights violations from the past."
        }
    ]
, "relationships": [
        {
            "source": "Cradock Four",
            "target": "Economic Freedom Fighters (EFF)",
            "type": "supports"
        , "description": "The families of the Cradock Four have joined the EFF in accusing the ANC government of freeing convicted perpetrators and being lenient with apartheid-era violence cases."
        },
        {
            "source": "Cradock Four",
            "target": "National Prosecuting Authority (NPA)",
            "type": "opposes"
        , "description": "The families of the Cradock Four have accused the NPA of not reopening investigations into their deaths and other apartheid-era cases."
        },
        {
            "source": "Cradock Four",
            "target": "Truth and Reconciliation Commission (TRC)",
            "type": "supports"
        , "description": "The families of the Cradock Four have supported the TRC's efforts to investigate human rights violations during the apartheid era."
        },
        {
            "source": "Cradock Four",
            "target": "Webber Wentzel",
            "type": "supports"
        , "description": "The families of the Cradock Four have hired Webber Wentzel, a law firm, to represent them in their lawsuit against the South African government."
        },
        {
            "source": "Economic Freedom Fighters (EFF)",
            "target": "National Prosecuting Authority (NPA)",
            "type": "opposes"
        , "description": "The EFF has accused the NPA of not reopening investigations into apartheid-era cases and being lenient with perpetrators."
        },
        {
            "source": "Economic Freedom Fighters (EFF)",
            "target": "Thembisile Makhulu",
            "type": "opposes"
        , "description": "The EFF has accused President Thembisile Makhulu of being lenient with apartheid-era violence cases and freeing convicted perpetrators."
        },
        {
            "source": "Highgate Hotel Massacre",
            "target": "National Prosecuting Authority (NPA)",
            "type": "opposes"
        , "description": "The families of the victims of the Highgate Hotel Massacre have accused the NPA of not reopening investigations into the case."
        },
        {
            "source": "Highgate Hotel Massacre",
            "target": "Truth and Reconciliation Commission (TRC)",
            "type": "supports"
        , "description": "The families of the victims of the Highgate Hotel Massacre have supported the TRC's efforts to investigate human rights violations during the apartheid era."
        },
        {
            "source": "Nokuthula Simelane",
            "target": "National Prosecuting Authority (NPA)",
            "type": "opposes"
        , "description": "The family of Nokuthula Simelane, a victim of apartheid-era violence, has accused the NPA of not reopening investigations into her death."
        },
        {
            "source": "Nokuthula Simelane",
            "target": "Truth and Reconciliation Commission (TRC)",
            "type": "supports"
        , "description": "The family of Nokuthula Simelane, a victim of apartheid-era violence, has supported the TRC's efforts to investigate human rights violations during the apartheid era."
        },
        {
            "source": "Colonel Eugene de Kock",
            "target": "National Prosecuting Authority (NPA)",
            "type": "opposes"
        , "description": "The families of the victims of Colonel Eugene de Kock's involvement in apartheid-era violence have accused the NPA of not reopening investigations into his crimes."
        },
        {
            "source": "Colonel Eugene de Kock",
            "target": "Truth and Reconciliation Commission (TRC)",
            "type": "supports"
        , "description": "The families of the victims of Colonel Eugene de Kock's involvement in apartheid-era violence have supported the TRC's efforts to investigate human rights violations during the apartheid era."
        },
        {
            "source": "South Africa",
            "target": "National Prosecuting Authority (NPA)",
            "type": "supports"
        , "description": "The South African government supports the NPA in its efforts to prosecute crimes and address human rights violations."
        },
        {
            "source": "South Africa",
            "target": "Truth and Reconciliation Commission (TRC)",
            "type": "supports"
        , "description": "The South African government supports the TRC's efforts to investigate human rights violations during the apartheid era and provide a platform for victims to share their stories."
        }
    ]
, "context": [
        {
            "aspect": "history",
            "description": "South Africa has a history of apartheid, which was officially ended in 1994. The country is currently working towards reconciliation and addressing human rights violations from the past."
        },
        {
            "aspect": "politics",
            "description": "The current president of South Africa is Thembisile Makhulu, who has been accused by the EFF of being lenient with apartheid-era violence cases and freeing convicted perpetrators."
        },
        {
            "aspect": "law",
            "description": "The National Prosecuting Authority (NPA) is responsible for prosecuting crimes in South Africa. The families of victims of apartheid-era violence have accused the NPA of not reopening investigations into their cases."
        },
        {
            "aspect": "human rights",
            "description": "The Truth and Reconciliation Commission (TRC) was established to investigate human rights violations during the apartheid era and provide a platform for victims to share their stories. The families of victims have supported the TRC's efforts."
        }
    ]
}
```