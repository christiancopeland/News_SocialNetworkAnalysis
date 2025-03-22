import requests 
import json
import networkx as nx
from netgraph import Graph as NGGraph
from netgraph import InteractiveGraph
import matplotlib.pyplot as plt
import numpy as np
import os 
import textwrap
import cProfile
import pstats
import io
from pstats import SortKey


def profile(fnc):
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval
    return inner

def wrap_text(text, max_line_length=25):
    wrapped = textwrap.fill(text, width=max_line_length)
    return '\n'.join([f"{" "*(i%3)*2}{line}" for i, line in enumerate(wrapped.split('\n'))])


def load_knowledge_graph(filename="network_dict.json"):
    with open(filename, 'r') as f:
        graph_data = json.load(f)

    G = nx.DiGraph()

    # Add nodes from entities
    for entity in graph_data["entities"]:
        G.add_node(entity["name"], node_type=entity["type"])
    
    # Add edges from relationships
    for rel in graph_data["relationships"]:
        G.add_edge(rel["source"], rel["target"], relationship=rel["description"])
    
    return G, graph_data


def message_r1(messages: list, stream: bool = False):
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    
    # Define valid entity types
    VALID_ENTITY_TYPES = [
        "person",
        "organization", 
        "policy",
        "issue",
        "impact",
        "location",
        "event"
    ]

    ENTITY_SUBTYPES = {
    "person": ["politician", "activist", "expert"],
    "organization": ["government", "NGO", "corporation"],
    "policy": ["domestic", "foreign", "economic"],
    "issue": [],
    "impact": [],
    "location": [],
    "event": []
    # ... add more as needed
    }

    subtypes_info = "\n".join([f"- {key}: {', '.join(value)}" for key, value in ENTITY_SUBTYPES.items() if value])
    
    RELATIONSHIP_TYPES = [
    "implements", "opposes", "supports", "criticizes",
    "collaborates_with", "impacts", "responds_to"
                        ]
    
    # Define the exact JSON schema we want the model to follow
    json_schema = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": [VALID_ENTITY_TYPES]  # Restrict to valid types
                        },
                        "subtype": {"type": "string", "enum": [subtype for subtypes in ENTITY_SUBTYPES.values() for subtype in subtypes]},
                        "description": {"type": "string"}
                    },
                    "required": ["name", "type", "subtype", "description"]
                }
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "type": {"type": "string",  "enum": RELATIONSHIP_TYPES},
                        "description": {"type": "string"}
                    },
                    "required": ["source", "target", "type", "description"]
                }
            },
            "context": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "aspect": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["aspect", "description"]
                }
            }
        },
        "required": ["entities", "relationships", "context"]
    }

    data = {
        "model": "qwen2.5-coder:14b",
        "messages": messages,
        "stream": stream,
        "format": json_schema,
        "options": {
            "temperature": 0  # Lower temperature for more consistent outputs
        }
    }

    # Update system message to be more specific about expectations
    messages[0]["content"] = f"""You are a Knowledge Graph creator specializing in news article analysis. Your task is to create a detailed knowledge graph that captures key entities, their relationships, and the broader context of the story.

        Entity types MUST be one of: {', '.join(VALID_ENTITY_TYPES)}
        All entity type values must be lowercase.

        When analyzing the article:
        1. Identify ALL key actors (people, organizations) and their roles
        2. Include specific policies, issues, and their impacts
        3. Capture responses and reactions from different parties
        4. Note any justifications or reasoning given for actions
        5. Include relevant locations and events

        Relationships should:
        - Be binary (one source to one target)
        - Use clear, active verbs in present tense (e.g., "implements", "opposes", "announces", "responds", "justifies")
        - Capture the nature of interactions between entities
        - Include both direct actions and reactions
        - Show cause-and-effect connections

        Context should include:
        - Economic implications
        - Political dynamics
        - International relations
        - Historical context (if mentioned)
        - Potential impacts or consequences

        Follow this JSON structure strictly:
        - entities: array of objects with name (string), type (string), subtype (string) and description (string)
        - relationships: array of objects with source (string), target (string), type (string), and description (string)
        - context: array of objects with aspect (string) and description (string)

        Make sure to capture the full complexity of the story while maintaining clear, direct connections.
        
        Additional instructions:
        1. Assign subtypes to entities where applicable. Valid subtypes are: 
            {subtypes_info}
        2. Use only these relationship types: {RELATIONSHIP_TYPES}
        3. Include temporal information for events and actions in ISO date format (YYYY-MM-DD) where possible.
        4. Assign confidence scores (0-1) to entities and relationships based on how certain the information is.
        5. Group entities or relationships into broader topics where relevant.
        6. Flag potentially controversial or unverified claims.
        """

    response = requests.post(url, headers=headers, json=data, stream=stream)
    
    if stream:
        # For streaming responses, yield each chunk
        full_response = ""
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if not json_response.get("done"):
                    chunk = json_response.get("message", {}).get("content", "")
                    full_response += chunk
                    yield chunk
                else:
                    # Return final statistics if needed
                    yield json_response
        return full_response
    else:
        # For non-streaming, return the full response
        return response


@profile
def create_knowledge_graph(article_text: str = None, load_file: str = None, show_graph: bool = True, ):
    """
    Creates and visualizes a knowledge graph from article text using LLM and NetworkX.
    
    Args:
        article_text (str): The article text to analyze
        show_graph (bool): Whether to display the graph visualization
    
    Returns:
        tuple: (NetworkX graph object, graph data dictionary)
    """

    edge_styles = {
    'imposes': (0.7, '#ff0000', 'dashed'),
    'announces': (2.0, '#00ff00', 'solid'),
    'responds to': (1.2, '#444444', 'dotted')
    }

    if load_file:
        G, data = load_knowledge_graph('network_dict.json')
    elif article_text:

        # Get knowledge graph data from LLM
        messages = [
            {"role": "system", "content": "You are a Knowledge Graph creator..."},  # System message set in message_r1
            {"role": "user", "content": f"Please create a knowledge graph from the provided article: \n{article_text}\n\n"}
        ]
        
        # Get streaming response and accumulate
        print("Generating knowledge graph data...")
        response_full = ""
        for chunk in message_r1(messages, stream=True):
            if isinstance(chunk, str):
                response_full += chunk
                print(chunk, end="", flush=True)
            else:
                print("\nGeneration complete!")

        # content = json.dumps(response_full)
        # json.dump(content, 'network_dict.json')
        with open('network_dict.json', 'w') as f:
            f.write(response_full)

        graph_data = json.loads(response_full)
        
        # Create NetworkX graph
        G = nx.DiGraph()

        # Add nodes from entities to networkx instance
        for entity in graph_data["entities"]:
            G.add_node(entity["name"], node_type=entity["type"])
        
        # Add edges from relationships to networkx instance
        for rel in graph_data["relationships"]:
            G.add_edge(rel["source"], rel["target"], relationship=rel["description"])

    
    if show_graph:

        # Get node+edge labels for netgraph
        # node_labels = {node: str(node) for node in G.nodes()}
        print(G.nodes())
        node_labels = {node: wrap_text(node) for node in G.nodes()}
        print(f"Node Labels: \n {node_labels}")
        # edge_labels = {(u, v): f"{u}-{v}" for u, v in G.edges()}
        edge_labels = {edge: wrap_text(label) for edge, label in nx.get_edge_attributes(G, 'relationship').items()}
        # edge_labels = nx.get_edge_attributes(G, 'relationship')
        # print(f"Edge Labels: \n {edge_labels}")

        # Create visualization
        plt.figure(figsize=(50, 40))
        
        # Create layout
        pos = nx.spring_layout(G, k=5.0/np.sqrt(G.order()), scale=2.0, center=(0.5, 0.5), iterations=50, seed=42)
        # pos = nx.circular_layout(G, scale=1, center=(0.5, 0.5))
        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.kamada_kawai_layout(
        #     G,
        #     weight='relationship_strength')


        # Draw the graph
        nx.draw(G, pos,
                node_color='lightblue',
                node_size=3000,
                arrows=True,
                edge_color='gray',
                arrowsize=10,
                width=1,
                with_labels=False)
        
        # Add node labels
        nx.draw_networkx_labels(G, pos,
                              font_size=10,
                              font_weight='bold',
                              bbox=dict(facecolor='white',
                                      edgecolor='none',
                                      alpha=0.8,
                                      pad=2))
        
        # Add edge labels
        
        nx.draw_networkx_edge_labels(G, pos,
                                   edge_labels,
                                   font_size=8,
                                   label_pos=0.5,
                                   bbox=dict(facecolor='white',
                                           edgecolor='none',
                                           alpha=0.8,
                                           pad=2))
        
        plt.title("Article Knowledge Graph", pad=20, size=16)
        plt.axis('off')
        plt.margins(x=0.2, y=0.2)

        fig, ax = plt.subplots(figsize=(40, 40))
        ax = plt.gca()
        ax.margins(0)
        # plt.tight_layout(pad=0)

        # ax.set_xlim([-1.5, 2.5])  # Extend beyond calculated positions
        # ax.set_ylim([-1.5, 2.5])

        # Non Interactive Netgraph
        # NGGraph(G, node_labels=node_labels, edge_labels=edge_labels, node_layout=pos, edge_layout='bundled', ax=ax)

        # Interactive Netgraph
        plot_instance = InteractiveGraph(G,
                                        node_layout="spring",
                                        
                                        node_labels=node_labels,
                                        edge_labels=edge_labels,
                                        node_label_offset=0.1,
                                        edge_label_offset=0.1)
        
        # print(plt.rcParams)
        # plt.figure(figsize=(20,20))
        plt.show()
    
    return G, graph_data

if __name__ == "__main__":
    # Example usage
    article = """
        Title: Cradock Four: Why apartheid victims are suing South Africa’s government
        Heading: EXPLAINER
        URL: https://www.aljazeera.com/news/2025/1/24/cradock-four-why-apartheid-victims-are-suing-south-africas-government
        Source: https://www.aljazeera.com/

        Content:
        ----------------------------------------
        Families of South Africans murdered by apartheid police – including a group of anti-apartheid activists killed in one of the most gruesome cases of the time in 1985 – are suing the government for damages worth $9m.

        According to a case filed at the High Court in Pretoria on Monday, 25 survivors and victims’ families are suing President Cyril Ramaphosa and his government for what they call a failure to properly investigate apartheid-era offences and deliver justice.



        Recommended Stories


        Among the applicants are families of the “Cradock Four“, who were assassinated 40 years ago. They have accused the government of “gross failure” to prosecute the six apartheid-era security officials allegedly responsible for the murders, and for “suppressing” inquiries into the case.

        The four – Matthew Goniwe, Fort Calata, Sparrow Mkhonto, and Sicelo Mhlauli – were all anti-apartheid activists from the town of Cradock (now Nxuba) in the Eastern Cape province. In 1985, they were abducted and murdered by police, triggering rage among many Black South Africans and marking a turning point for the push for liberation from racist rule.

        However, their alleged killers have all passed away without justice being served.

        Here’s what to know about the Cradock Four and the new case launched against the government 30 years after the end of apartheid:



        What happened in 1985?


        In the Cradock community in the 1980s, the four activists were known for fighting against gruelling conditions for Black South Africans, including poor health infrastructure and high rent. Mathew Goniwe, in particular, was a popular figure and led the Cradock Youth Association (CRADORA). Fort Calata was also a leading member of the group.

        Apartheid police officials surveilled CRADORA constantly and arrested members like Goniwe and Calata several times before the assassinations. Officials had also attempted to split them up: Goniwe, a public school teacher, was transferred to another region to teach, for example, but refused to work there and was fired by the education department.

        On the night of June 27, 1985, the four were travelling together in a vehicle, having just finished rural mobilisation work on the outskirts of the city. Police officials stopped them at a roadblock outside Gqeberha, which was then called Port Elizabeth. The men were abducted and assaulted, and then their bodies were burned and dispersed in different parts of Gqeberha.

        Their deaths caused grief and anger among Black South Africans and marked a crucial intensification of anti-apartheid activism. Thousands of people attended their funeral. The Craddock Four became icons, with T-shirts and posters bearing their names.

        Apartheid government officials denied involvement in the killings. A court inquest into the case in 1987 found that the four had been killed by “unknown persons”.

        However, in 1992, leaked documents revealed that CRADORA leaders Goniwe and Calata had been on the hit list of the Civil Cooperation Bureau, a government death squad. Then-President FW de Klerk called for another inquiry, in which a judge confirmed that the security forces were responsible, although no names were mentioned.



        What did the TRC find and why do families feel betrayed?


        Following the fall of apartheid and the ushering in of democratic rule in 1994, the unity government led by the African National Congress (ANC) party launched a Truth and Reconciliation Commission (TRC) in 1996 to investigate, prosecute, or pardon apartheid-era crimes.

        The Cradock Four case was one of those reviewed. The commission investigated six police officials who were allegedly involved. Namely: officers Eric Alexander Taylor, Gerhardus Johannes Lotz, Nicolaas Janse van Rensburg, Johan van Zyl, Hermanus Barend du Plessis, and Colonel Harold Snyman, who is believed to have ordered the killings. By the time of the hearings, Snyman had passed away.

        Although the court granted pardons for many political criminals at the time, it also ordered the investigations of hundreds of others, including the killers of the Cradock Four, who were denied amnesty. Officials said the men failed to make a “full disclosure” about the circumstances of the killings. The TRC required accused perpetrators to fully disclose events they were involved in to be considered for a pardon.

        At the time, family members of the Cradock Four described their happiness at the decision, believing the South African government would then prosecute the accused men. However, successive governments, from former President Thabo Mbeki (1999-2008) to Ramaphosa, have not concluded the investigations, despite the ANC, which helped usher in democracy under Nelson Mandela, having always been in power. Presently, all six accused officials have passed away, with the last man dying in May 2023.

        Cradock Four families first sued the country’s National Prosecuting Authority (NPA) and the South African police in 2021, asking the court to compel them to finish investigations and determine whether the case would go to trial. However, officials did not reopen another inquiry until January 2024, months after the last accused official died. Proceedings are set to begin in June 2025.

        Critics of the ANC have long alleged that there was a secret agreement between the post-apartheid government and the former white minority government to avoid prosecutions. In 2021, a former NPA official testified to the Supreme Court in a separate case that Mbeki’s administration intervened in the TRC process, and “suppressed” prosecutions in more than 400 cases.

        Mbeki denies those allegations. “We never interfered in the work of the National Prosecuting Authority (NPA),” he said in a statement in March 2024.

        “The executive never prevented the prosecutors from pursuing the cases referred to the NPA by the Truth and Reconciliation Commission. If the investigations … referred to were stopped, they were stopped by the NPA and not at the behest of the Government.”



        What is the new court case about?


        In the new case, families of the Cradock Four joined survivors and families of other victims to sue the government for failing to properly investigate their cases. The suit specifically named President Ramaphosa, the justice and police ministers, the head of the NPA and the national police commissioner.

        The families are seeking “constitutional damages” to the tune of 167 million rand ($9m), for the “egregious violations” of their rights. In the case of the four Cradock activists, relatives said because government officials delayed prosecution, all the accused officers have died, ensuring that no criminal prosecution would be possible, denying the families “justice, truth and closure”.

        The families also asked the courts to force President Ramaphosa to set up an independent commission of inquiry into alleged government interference under the Mbeki administration.

        Odette Geldenhuys, a lawyer at Webber Wentzel, the firm representing the families in the lawsuit, told Al Jazeera the damages, if granted, would serve as an “alternative” form of justice.

        “Over the two decades … not only did victims and families of victims die, but also perpetrators died,” Geldenhuys said. “The criminal law is clear: a dead body cannot be prosecuted. Alternative justice will go some way to deal with the ongoing and inter-generational pain.”

        The funds would be available to all other victims and survivors of apartheid-era political crimes, and would be used for further investigations, memorials and public education, Geldenhuys added.



        Why has the case generated interest in South Africa?


        The Cradock Four were important figures during the apartheid era, but the fact that their deaths were never fully prosecuted, has held the interest of many South Africans, particularly amid allegations of the post-apartheid government’s complicity.

        In a statement on Thursday, the left-wing opposition Economic Freedom Fighters (EFF) party sided with the families and survivors and accused the ANC government of freeing convicted perpetrators, including former assassin, Colonel Eugene de Kock who was initially sentenced to life but was granted parole in 2015, under the Ramaphosa government.

        “The ANC’s handling of apartheid-era violence cases has always been suspiciously lenient,” the EFF’s statement read. “It is unacceptable that over 30 years after the fall of apartheid, these families still do not have answers or closure about the fate of their loved ones.”

        Several other cases that were not fully investigated after the TRC process were also involved in Monday’s suit. Housing Minister Thembi Nkadimeng is, for example, among the applicants in the latest case. Her sister Nokuthula Simelane, killed in 1983, was believed to have been abducted and tortured by apartheid security forces.

        Survivors of the 1993 Highgate Hotel Massacre in the city of East London, when five masked men stormed into the hotel’s bar and shot at people there, are also part of the new case. Five people were killed, but survivors Neville Beling and Karl Weber, who were injured in the shooting, joined Monday’s suit. No one was ever arrested or investigated. In 2023, an official inquiry was opened for the first time, with proceedings beginning this month.

        In total, the case could see the deaths of nearly 30 people newly investigated. However, several perpetrators are likely to have passed away.
            """
    
    # G, data = create_knowledge_graph(load_file="network_dict.json")
    G, data = create_knowledge_graph(article_text=article)
    
    # You can now work with either the NetworkX graph object (G) 
    # or the raw graph data (data) as needed