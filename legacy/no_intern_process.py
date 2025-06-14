import os
import networkx as nx

import pandas as pd
import itertools
from collections import defaultdict
STATIC_PATH = "/home/elinha/NLP/moducon_data/phase_2/raw_file"

china_file_name = "intern_review_china.csv"
japan_file_name = "intern_review_japan.csv"
korea_file_name = "intern_review_korea.csv"
northAmerica_file_name = "intern_review_northAmerica.csv"
color_list = ["ffda11", "BC002D", "0F64CD", "B31942"]
file_name_list = [
    china_file_name,
    japan_file_name,
    korea_file_name,
    northAmerica_file_name,
]

# 기존의 df_dict 생성
df_dict = {
    file_name.split("_")[-1].split(".")[0]: {
        "dataframe": pd.read_csv(os.path.join(STATIC_PATH, file_name)),
        "color": color,
    }
    for color, file_name in zip(color_list, file_name_list)
}

# target_dict 생성 및 데이터 추가
target_dict = {}

# 개별 국가 데이터프레임 저장
for country in ["china", "japan", "korea"]:
    target_dict[country] = {
        "dataframe": df_dict[country]["dataframe"],
        "color": df_dict[country]["color"],
    }

# 동양 전체(orient) 데이터프레임 생성 및 저장
orient_df = pd.concat([
    df_dict[country]["dataframe"] for country in ["china", "japan", "korea"]
])
target_dict["orient"] = {
    "dataframe": orient_df,
    "color": "ff7f00",  # 동양을 나타내는 회색 추가 (필요에 따라 색상 변경 가능)
}

# 북미 데이터프레임 추가 및 열 이름 변경
if "northAmerica" in df_dict:
    north_america_df = df_dict["northAmerica"]["dataframe"].rename(
        columns={"splited_sentence": "SeparatedSentences"}
    )
    target_dict["northAmerica"] = {
        "dataframe": north_america_df,
        "color": df_dict["northAmerica"]["color"],
    }


import re
from collections import Counter

import inflect
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer


from contractions import expand_contractions

# 기존의 전처리 함수들을 여기에 정의합니다 (expand_contractions, convert_number, rm_punct, only_english, pos_tagging, remain_specific_tags, singularize_and_lemmatize, remove_verbs, replace_words)

p = inflect.engine()

include_tags = ["JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS"]


# 아라비아 숫자를 텍스트 숫자로 변경 5 -> five
def replace_number(match):
    number = match.group()
    return p.number_to_words(number)


def convert_number(text):
    converted_text = re.sub(r"\d+", replace_number, text)
    return converted_text


def rm_punct(text):
    cleaned_text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\]。^_`{|}…~]', " ", text)
    return cleaned_text


# remove japanese, korean, chinese words


def only_english(text):
    cleaned_text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return cleaned_text


# POS_tagging


# Text를 단어 토큰과 POS 태그로 변환
def pos_tagging(text):
    word_tokens = word_tokenize(text)
    return pos_tag(word_tokens)


def remain_specific_tags(tagged_sentence, include_tags):
    return [word for word, tag in tagged_sentence if tag in include_tags]


# Lemmatize & remove stopwords


# inflect engine initialize
p = inflect.engine()

# WordNetLemmatizer initialize
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wn.ADJ
    elif treebank_tag.startswith("N"):
        return wn.NOUN
    else:
        return None


def singularize_and_lemmatize(tagged_words):
    new_words = []

    for word, tag in tagged_words:
        if tag.startswith("N"):
            singular_word = p.singular_noun(word)
            if singular_word:
                new_words.append(singular_word)
            else:
                new_words.append(word)
        else:
            wn_tag = get_wordnet_pos(tag)
            if wn_tag is not None:
                lemma_word = lemmatizer.lemmatize(word, wn_tag)
                new_words.append(lemma_word)
            else:
                new_words.append(word)

    return new_words


# stopwords
def remove_verbs(words):
    verbs_to_remove = {
        "i",
        "you",
        "they",
        "them",
        "your",
        "my",
        "he",
        "him",
        "his",
        "her",
        "she",
        "yours",
        "their",
        "hers",
        "movie",
        "movies",
        "prada",
        "ending",
        "cast",
        "fashion",
        "actor",
        "comedy",
        "devil",
        "scene",
        "subtitles",
        "thing",
        "other",
        "character",
        "such",
        "lot",
        "s",
    }
    filtered_words = [word for word in words if word not in verbs_to_remove]
    return filtered_words


def replace_words(sentence):
    replacements = {
        r"\binternben\b": ["intern", "ben"],
        r"\bexperiencesben\b": ["experience", "ben"],
        r"\bbeingben\b": ["being", "ben"],
        r"\bbentwo\b": ["ben", "two"],
        r"\bbens\b": ["ben"],
        r"\bitben\b": ["it", "ben"],
        r"\bheartben\b": ["heart", "ben"],
        r"\botherben\b": ["other", "ben"],
        r"\bbentheir\b": ["ben", "their"],
        r"\bfamilyben\b": ["family", "ben"],
        r"\bfunben\b": ["fun", "ben"],
        r"\binternrobert\b": ["intern", "robert"],
        r"\buprobert\b": ["up", "robert"],
        r"\brealisticrobert\b": ["realistic", "robert"],
        r"\btemperrobert\b": ["temper", "robert"],
        r"\bplacerobert\b": ["place", "robert"],
        r"\btimesrobert\b": ["time", "robert"],
        r"\bniros\b": ["niro"],
        r"\bdeniro\b": ["niro"],
        r"\bniroi\b": ["niro", "I"],
        r"\bmyershe\b": ["she"],
        r"\bmyersworking\b": ["working"],
        r"\banhathaway\b": ["hathaway"],
        r"\bsceneanhathaway\b": ["scene", "hathaway"],
        r"\banhathawayhis\b": ["hathaway", "his"],
        r"\bhathawayde\b": ["hathaway", "de"],
        r"\bhathaways\b": ["hathaway"],
        r"\banhathaways\b": ["hathaway"],
        r"\brottenanhathaway\b": ["rotten", "hathaway"],
        r"\btherejules\b": ["there", "jule"],
        r"\bweakjule\b": ["weak", "jule"],
        r"\bjules\b": ["jule"],
        r"\broutinejule\b": ["routine", "jule"],
        r"\bagejule\b": ["age", "jule"],
        r"\bbenjamin\b": ["ben"],
        r"\bturnben\b": ["turn", "ben"],
        r"\bbenmansay\b": ["ben", "man", "say"],
        r"\bbenis\b": ["ben", "is"],
        r"\bbenmaji\b": ["ben", "maji"],
        r"\bbenbambam\b": ["ben", "bambam"],
        r"\bbengentleman\b": ["ben", "gentleman"],
        r"\bgentorben\b": ["gentle", "ben"],
        r"\brobertdeniro\b": ["robert"],
        r"\bmovierobert\b": ["movie", "robert"],
        r"\brobertdeniel\b": ["robert"],
        r"\brobertdenilo\b": ["robert"],
        r"\broberto\b": ["robert"],
        r"\bpiecerobert\b": ["piece", "robert"],
        r"\broberts\b": ["robert"],
        r"\bbutjule\b": ["but", "jule"],
        r"\brojule\b": ["jule"],
        r"\bannehathaway\b": ["hathaway"],
        r"\bmyloveanne\b": ["my", "love", "hathaway"],
        r"\bunhathaway\b": ["hathaway"],
        r"\banahathaway\b": ["hathaway"],
    }
    for pattern, replacement in replacements.items():
        sentence = re.sub(pattern, " ".join(replacement), sentence)

    return sentence.split()


word_groups = [
    ["ben", "bambam", "whitaker", "whittaker", "van"],
    ["deniro", "niro", "robert"],
    ["anhathaway", "anne", "hathaway", "ann"],
    ["jules", "jule", "jale", "julie"],
]

# 단어 그룹을 딕셔너리로 변환
word_mapping = {}
for group in word_groups:
    for word in group[1:]:
        word_mapping[word] = group[0]


def apply_mapping(sentence):
    return [word_mapping.get(word.lower(), word) for word in sentence]


def preprocess_dataframe(df):
    # 전처리 과정
    sentences = df["SeparatedSentences"].dropna().copy()

    # 여기에 모든 전처리 단계를 적용합니다
    contracted_list = [expand_contractions(sentence.lower()) for sentence in sentences]
    inflected_list = [convert_number(text) for text in contracted_list]
    rm_punctionations = [only_english(rm_punct(text)) for text in inflected_list]
    tag_sentence = [pos_tagging(text) for text in rm_punctionations]
    tag_rm = [remain_specific_tags(text, include_tags) for text in tag_sentence]
    tag_rm_sentence = [pos_tag(sentence) for sentence in tag_rm]
    processed_sentences = [
        singularize_and_lemmatize(sentence) for sentence in tag_rm_sentence
    ]
    processed_sentences_no_stopwords = [
        remove_verbs(word) for word in processed_sentences
    ]
    modified_sentences = [
        replace_words(" ".join(sentence))
        for sentence in processed_sentences_no_stopwords
    ]

    # 빈 리스트와 길이가 1인 단어 제거
    modified_sentences = [
        sentence for sentence in modified_sentences if len(sentence) > 1
    ]

    # 단어 그룹 매핑 적용
    modified_sentences = [apply_mapping(sentence) for sentence in modified_sentences]

    # 빈도수가 평균 이상인 단어만 유지
    all_words = [
        word for sentence in modified_sentences for word in sentence if len(word) > 1
    ]
    word_freq = Counter(all_words)
    df_word_freq = pd.DataFrame(word_freq.items(), columns=["Word", "Frequency"])
    frequency_mean = df_word_freq["Frequency"].mean()
    words_to_keep = set(
        df_word_freq[df_word_freq["Frequency"] > frequency_mean]["Word"]
    )

    modified_sentences_filtered = [
        [word for word in sentence if word in words_to_keep]
        for sentence in modified_sentences
    ]

    return modified_sentences_filtered


# target_dict 업데이트
for key in target_dict.keys():
    if key != "orient":  # 개별 국가 처리
        target_dict[key]["modified_sentences"] = preprocess_dataframe(
            target_dict[key]["dataframe"]
        )
    else:  # 'orient' 처리
        orient_sentences = []
        for country in ["china", "japan", "korea"]:
            orient_sentences.extend(target_dict[country]["modified_sentences"])
        target_dict["orient"]["modified_sentences"] = orient_sentences

# 결과 확인
for key in target_dict.keys():
    print(f"{key}: {len(target_dict[key]['modified_sentences'])} processed sentences")


def count_tokens(sentences):
    all_tokens = [token for sentence in sentences for token in sentence]
    unique_tokens = set(all_tokens)
    return len(all_tokens), len(unique_tokens)


token_counts = {}

for country, data in target_dict.items():
    total_tokens, unique_tokens = count_tokens(data["modified_sentences"])
    token_counts[country] = {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
    }

# 결과 출력
print("토큰 수 통계:")
print("국가\t\t총 토큰 수\t고유 토큰 수")
print("-" * 40)
for country, counts in token_counts.items():
    print(f"{country:<12}\t{counts['total_tokens']:<12}\t{counts['unique_tokens']}")

# 가장 많이 사용된 토큰 (상위 20개)
print("\n각 국가별 가장 많이 사용된 토큰 (상위 20개):")
for country, data in target_dict.items():
    all_tokens = [
        token for sentence in data["modified_sentences"] for token in sentence
    ]
    token_freq = Counter(all_tokens)
    print(f"\n{country}:")
    for token, freq in token_freq.most_common(20):
        print(f"{token}: {freq}")

# 전체 데이터셋에서 가장 많이 사용된 토큰 (상위 20개)
all_tokens = [
    token
    for country, data in target_dict.items()
    for sentence in data["modified_sentences"]
    for token in sentence
]
total_token_freq = Counter(all_tokens)

print("\n전체 데이터셋에서 가장 많이 사용된 토큰 (상위 20개):")
for token, freq in total_token_freq.most_common(20):
    print(f"{token}: {freq}")


target_dict["japan"]["representative_words"] = [
    "ben",
    "senior",
    "elderly",
    "whitaker",
    "elder",
    "older",
    "subordinate",
    "system",
    "able",
    "accurate",
    "knowledge",
    "advice",
    "assistant",
]
target_dict["korea"]["representative_words"] = [
    "whitaker",
    "ben",
    "elder",
    "senior",
    "elderly",
    "older",
    "old",
    "wisdom",
    "ability",
    "passion",
    "grandfather",
    "gentleman",
    "friend",
    "assistant",
    "uncle",
]
target_dict["china"]["representative_words"] = [
    "ben",
    "senior",
    "elderly",
    "whitaker",
    "elderly",
    "elder",
    "intelligent",
    "capable",
    "passion",
    "mentor",
    "experience",
    "wise",
    "assistant",
]
target_dict["northAmerica"]["representative_words"] = [
    "whitaker",
    "ben",
    "elder",
    "eighty",
    "elderly",
    "older",
    "friend",
    "father",
    "respect",
    "guidance",
    "experienced",
    "respect",
    "guidance",
    "friend",
    "father",
]
target_dict["orient"]["representative_words"] = [
    "whitaker",
    "ben",
    "elder",
    "eighty",
    "elderly",
    "older",
    "subordinate",
    "colleague",
    "assistant",
    "trust",
    "sincere",
    "growth",
    "experienced",
]


import random



# Function to generate random colors and ensure they aren't too dark
def random_color():
    while True:
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        # Calculate brightness: 0.299*R + 0.587*G + 0.114*B
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        if brightness > 150:  # Only return the color if it's bright enough
            return color


import community as community_louvain
from collections import defaultdict

target_country = "korea"
#TODO: 반복문으로 
for target_country in ["korea", "japan", "china", "northAmerica"]:

    word_color_mapping = {}
    # Step 1: Extract all unique representative words from all countries
    unique_words = set()
    for country, country_dict in target_dict.items():
        unique_words.update(country_dict["representative_words"])


    for word in unique_words:
        word_color_mapping[word] = random_color()

    target_dict[target_country]['word_color_mapping'] = word_color_mapping


    co_occurrence = defaultdict(int)

    window_size = 20

    modified_sentences = target_dict[target_country]["modified_sentences"]

    # modified_sentences 리스트에서 'Word' 컬럼에 있는 단어들만 남기기
    for sentence in modified_sentences:
        effective_window_size = min(len(sentence), window_size)
        for i in range(len(sentence)):
            window = sentence[i:i+effective_window_size]
            for pair in itertools.combinations(window, 2):
                co_occurrence[pair] += 1

    G = nx.Graph()
    for (word1, word2), weight in co_occurrence.items():
        G.add_edge(word1, word2, weight=weight)

    partition = community_louvain.best_partition(G, weight='weight')
    print(set(partition.values()))
    # print(partition)

    representative_words = target_dict[target_country]['representative_words']
    save_value_nums= []
    for word in representative_words:
        
        cluster_num = partition.get(word, '')
        if cluster_num and cluster_num not in save_value_nums:
            save_value_nums.append(cluster_num)
        elif cluster_num  in save_value_nums:
            print(f'{word}는 이미 {cluster_num}에 있는 단어 입니다.')
        else:
            print(f'{word}는 없는 단어입니다.')
            
    print(f'남은 군집은 {save_value_nums} 입니다.')

    filtered_partition = [key for key, value in partition.items() if value in save_value_nums]

    modified_sentences_filtered = [
        [word for word in sentence if word in filtered_partition]
        for sentence in modified_sentences
    ]


    from collections import Counter
    all_words = [word for sentence in modified_sentences for word in sentence if len(word) > 1]
    word_freq= Counter(all_words)

    df_word_freq = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])

    df_word_freq.head(20)

    frequency_mean = df_word_freq['Frequency'].mean()
    print('frequency_mean', frequency_mean)
    df_word_freq_new = df_word_freq[df_word_freq['Frequency']>frequency_mean]
    df_word_freq_new.sort_values(by='Frequency', ascending=False, inplace=True)


    words_to_keep = set(df_word_freq_new['Word'])


    relevant_words = [
        word for word in list(words_to_keep)
        if word in filtered_partition]
    print(f"평균 이상의 빈도수 중에서 선택된 군집에서 살아남은 단어의 수: {len(relevant_words)}")
    print('target_country', target_country)
    df_dict[target_country]['modified_sentences_filtered'] = modified_sentences_filtered
    df_dict[target_country]['relevant_words'] = relevant_words




for target_country in ["korea", "japan", "china", "northAmerica", "orient"]:

    modified_sentences_filtered = target_dict[target_country]["modified_sentences"]
    relevant_words = target_dict[target_country]["representative_words"]

    H_old_filtered = nx.Graph()
    main_word_connections = {}
    for main_word in relevant_words:
        # Get all edges connected to the main word
        connections = []
        for edge in H_old_filtered.edges(data=True):
            if main_word in edge[:2]:  # if main_word is either source or target
                other_word = edge[0] if edge[1] == main_word else edge[1]
                weight = edge[2]["weight"]
                connections.append((other_word, weight))

        # Sort connections by weight and get top 5
        sorted_connections = sorted(connections, key=lambda x: x[1], reverse=True)[:5]
        main_word_connections[main_word] = sorted_connections

    # # Print results
    # for main_word, connections in main_word_connections.items():
    #     print(f"\nMain word: {main_word}")
    #     print("Supporting words (with weights):")
    #     for word, weight in connections:
    #         print(f"  {word}: {weight}")




    import networkx as nx
    import plotly.graph_objs as go

    default_color = target_dict[target_country]["color"]
    # Prepare the co-occurrence data
    co_occurrence = defaultdict(int)
    window_size = 20

    for sentence in modified_sentences_filtered:
        effective_window_size = min(len(sentence), window_size)
        for i in range(len(sentence)):
            window = sentence[i : i + effective_window_size]
            for pair in itertools.combinations(window, 2):
                co_occurrence[pair] += 1

    # Create the graph
    G = nx.MultiGraph()

    for (word1, word2), weight in co_occurrence.items():
        G.add_edge(word1, word2, weight=weight)

    # 리스트를 평탄화하고 set으로 변환하여 중복 제거
    unique_words = set(word for sublist in modified_sentences_filtered for word in sublist)

    # 결과를 리스트로 변환 (필요 시)
    # relevant_words = list(unique_words)
    relevant_words = target_dict[target_country]["representative_words"]

    # # Filter edges based on relevance to specific words
    # filtered_edges = [(u, v, d) for u, v, d in G.edges(data=True) if u in relevant_words or v in relevant_words]

    # H = nx.Graph()
    # H.add_edges_from(filtered_edges)

    # Focus on top co-occurrences for specific words
    H_old_filtered = nx.Graph()

    for word in relevant_words:
        edges_for_word = [
            (u, v, d) for u, v, d in G.edges(data=True) if (u == word or v == word)
        ]
        sorted_edges = sorted(edges_for_word, key=lambda x: x[2]["weight"], reverse=True)[
            :window_size
        ]
        H_old_filtered.add_edges_from(sorted_edges)

    # Using plotly to visualize
    pos_old_filtered = nx.spring_layout(
        H_old_filtered, seed=42, k=1.2
    )  # Spring layout positions

    # Create edge traces
    edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=0.5, color="#807c6a"), hoverinfo="none", mode="lines"
    )

    for edge in H_old_filtered.edges():
        x0, y0 = pos_old_filtered[edge[0]]
        x1, y1 = pos_old_filtered[edge[1]]
        edge_trace["x"] += (x0, x1, None)
        edge_trace["y"] += (y0, y1, None)


    # Function to generate random colors and ensure they aren't too dark
    def random_color():
        while True:
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            # Calculate brightness: 0.299*R + 0.587*G + 0.114*B
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            if brightness > 150:  # Only return the color if it's bright enough
                return color


    custom_node_colors = {}
    for word in relevant_words:
        random_node_color = word_color_mapping.get(word, random_color())
        custom_node_colors[word] = random_node_color

    # Create node traces
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            showscale=False,  # No color scale as we are using custom colors
            color=[],  # Custom color will be added here
            size=[H_old_filtered.degree(node) * 10 for node in H_old_filtered.nodes()],
        ),
    )


    for node in H_old_filtered.nodes():
        x, y = pos_old_filtered[node]
        node_trace["x"] += (x,)
        node_trace["y"] += (y,)
        node_trace["text"] += (node,)
        # Apply custom color if available, otherwise default to gray
        node_trace["marker"]["color"] += (
            custom_node_colors.get(node, f"#{default_color}"),
        )  # Default to gray

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            # title=f'Co-occurrence Network for china_taiwan (Top {top_num} Edges per Word)',
            titlefont_size=20,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=40),
            annotations=[
                dict(
                    text="Node size based on degree (number of connections)",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1000,  # Set the width of the figure (in pixels)
            height=1000,  # Set the height of the figure (in pixels)
        ),
    )


    # Show the interactive graph
    # fig.show()
    fig.write_html(f"{target_country}_review_keywords_network.html")
