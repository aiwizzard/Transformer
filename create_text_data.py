"""
Cornell movie dataset is in a certain format. In order to train the model we need
to modify the way the data is arranged. This is used to do that. It takes the conversations
and arrange in specific order for the model to be trained. The format is
<question>
<reply>

"""

import os

import config as config

corpus_movie_conv = os.path.join(os.getcwd(), config.data_dir, 'cornell movie-dialogs corpus/movie_conversations.txt')
corpus_movie_lines = os.path.join(os.getcwd(), config.data_dir, 'cornell movie-dialogs corpus/movie_lines.txt')
output_path = os.path.join(os.getcwd(), config.data_dir, 'processed_cornell_data.txt')

with open(corpus_movie_conv, 'r', encoding='iso-8859-1') as c:
    conv = c.readlines()

with open(corpus_movie_lines, 'r', encoding='iso-8859-1') as l:
    lines = l.readlines()

lines_dic = {}
for line in lines:
    objects = line.split(" +++$+++ ")
    lines_dic[objects[0]] = objects[-1]

pairs = []

def create_text_data(output_path: str) -> None:
    with open(output_path, 'w') as f:
        for con in conv:
            ids = eval(con.split(" +++$+++ ")[-1])
            for i in range(len(ids)):                
                if i==len(ids)-1:
                    break
                first = lines_dic[ids[i]]   
                second = lines_dic[ids[i+1]]
                f.write(first)
                f.write(second)
                f.write('\n')

if __name__ == '__main__':
    create_text_data(output_path)