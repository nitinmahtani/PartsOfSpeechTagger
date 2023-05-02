# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys
import numpy as np

def train(training_files):
    states = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD", "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1",
    "NN2", "NP0", "ORD", "PNI", "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0", "UNC", "VBB", "VBD", "VBG", "VBI",
    "VBN", "VBZ", "VDB", "VDD", "VDG", "VDI", "VDN", "VDZ", "VHB", "VHD", "VHG", "VHI", "VHN", "VHZ", "VM0", "VVB", "VVD", "VVG", "VVI",
    "VVN", "VVZ", "XX0", "ZZ0", "AJ0-AV0", "AJ0-VVN", "AJ0-VVD", "AJ0-NN1", "AJ0-VVG", "AVP-PRP", "AVQ-CJS", "CJS-PRP", "CJT-DT0", "CRD-PNI",
    "NN1-NP0", "NN1-VVB", "NN1-VVG", "NN2-VVZ", "VVD-VVN"]

    I = dict.fromkeys(states, 0)
    # tag_count = {}
    cleaned_content = []
    pos_and_pos = {}
    pos_and_word = {}
    for train in training_files:
        
        f = open(train, 'r')
        contents = f.readlines()
        f.close()
        for i in range(len(contents)):
            if contents[i][-1] == '\n':
                contents[i] = contents[i][:-1]
                
            lst = contents[i].split(" ")
            if lst[2] in I:
                I[lst[2]] += 1
            # if lst[2] in tag_count:
            #     tag_count[lst[0]] += 1
            # else:
            #     tag_count[lst[0]] = 1
            cleaned_content.append(lst[0] + lst[2])

            tag_and_word = lst[2] + " and " + lst[0]
            if tag_and_word in pos_and_word:
                pos_and_word[tag_and_word] += 1
            else:
                pos_and_word[tag_and_word] = 1

            if i + 1 < len(contents):

                if contents[i + 1][-1] == '\n':
                    contents[i + 1] = contents[i + 1][:-1]
                lst2 = contents[i + 1].split(" ")
                
                tags = lst[2] + " and " + lst2[2]
                if tags in pos_and_pos:
                    pos_and_pos[tags] += 1
                else:
                    pos_and_pos[tags] = 1
    
    T = {}
    M = {}
    for tags in pos_and_pos:
        lst = tags.split(" ")
        if lst[0] in I:
            T[lst[2] + "|" + lst[0]] = pos_and_pos[tags] / I[lst[0]]
    for t_and_w in pos_and_word:
        lst = t_and_w.split(" ")
        if lst[0] in I:
            M[lst[2] + "|" + lst[0]] = pos_and_word[t_and_w] / I[lst[0]]
    for pos in I:
        I[pos] = I[pos] / len(cleaned_content)

    # print(M)
    # print(len(contents))
    return I, T, M, states

def pre_verb(test):
    f = open(test, 'r')
    contents = f.readlines()
    f.close()
    for i in range(len(contents)):
        if contents[i][-1] == '\n':
            contents[i] = contents[i][:-1]
    return contents

def viterbi(words, pos_tags, I, T, M, prob, prev):

    for t in range(len(words)):
    # Determine values for time step 0
        if t == 0 or words[t - 1] == "." or words[t - 1] == "!" or words[t - 1] == "?":
            for i in range(len(pos_tags)):
                string_key = words[t] + "|" + pos_tags[i]
                
                if string_key in M:
                    prob[t,i] = I[pos_tags[i]] * M[string_key] 
                else:
                    prob[t, i] = I[pos_tags[i]] *  0.00005 # change this?
                    
                prev[t,i] = 29 
        else:
    
            for i in range(len(pos_tags)):
                maxi = -1
                max_index = -1
                for x in range(len(pos_tags)):
                    t_string = pos_tags[i] + "|" + pos_tags[x]
                    m_string = words[t] + "|" + pos_tags[i]

                    if t_string in T and m_string in M:
                        if prob[t-1,x] * T[pos_tags[i] + "|" + pos_tags[x]] * M[words[t] + "|" + pos_tags[i]] > maxi:
                            maxi = prob[t-1,x] * T[pos_tags[i] + "|" + pos_tags[x]] * M[words[t] + "|" + pos_tags[i]]
                            max_index = x

                    elif m_string in M:
                        if prob[t-1,x] * M[words[t] + "|" + pos_tags[i]] * 0.005 > maxi:
                            maxi = prob[t-1,x] * M[words[t] + "|" + pos_tags[i]] * 0.005
                            max_index = x

                    elif t_string in T:
                        if prob[t-1,x] * T[pos_tags[i] + "|" + pos_tags[x]] * 0.00005 > maxi:
                            maxi = prob[t-1,x] * T[pos_tags[i] + "|" + pos_tags[x]] * 0.00005
                            max_index = x

                    # else:
                    #     if prob[t-1,x] * 0.005 *  0.00005 > maxi:
                    #         maxi = prob[t-1,x] * 0.005 *  0.00005
                    #         max_index = x
                            

                prob[t,i] = maxi
                prev[t,i] = max_index

        
           
    return prob, prev

def after_viterbi(words, states, prob, prev, path):
    
    lst = []
    x = np.argmax(prob[-1])
    curr = len(prob) - 1
    while curr != 0:

        lst.insert(0, states[x])
        x = prev[curr][x]
        curr -= 1
    lst.insert(0, states[x])

    final = ""

    for i in range(len(words)):
        final += words[i]
        final += " : "
        final += lst[i]
        final += '\n'
    
    # print(final)
    f = open(path, "w")
    f.write(final)
    f.close()


def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")
    #
    # YOUR IMPLEMENTATION GOES HERE
    #
    
    I, T, M, states = train(training_list)
    test_words = pre_verb(test_file)
    
    print("average val of T is", sum(T.values())/len(T))
    print("average val of M is", sum(M.values())/len(M))

    print("running viterbi")
    prob = np.zeros((len(test_words),len(states)), dtype=float)  
    prev = np.zeros((len(test_words),len(states)), dtype=int) 
    prob, prev = viterbi(test_words, states, I, T, M, prob, prev)
    print("finished viterbi")
    after_viterbi(test_words, states, prob, prev, output_file)

if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    # print("Training files: " + str(training_list))
    # print("Test file: " + test_file)
    # print("Output file: " + output_file)

    # Start the training and tagging operation.
    tag (training_list, test_file, output_file)