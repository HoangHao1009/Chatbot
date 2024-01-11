import json
import csv

class LoadmovieData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.conversations = None
        self.qa_pairs = None

    def loadLinesAndConversations(self):
        lines = {}
        conversations = {}
        with open(self.file_path, 'r', encoding='iso-8859-1') as f:
            for line in f:
                lineJson = json.loads(line)
                # Extract fields for line object
                lineObj = {}
                lineObj["lineID"] = lineJson["id"]
                lineObj["characterID"] = lineJson["speaker"]
                lineObj["text"] = lineJson["text"]
                lines[lineObj['lineID']] = lineObj

                # Extract fields for conversation object
                if lineJson["conversation_id"] not in conversations:
                    convObj = {}
                    convObj["conversationID"] = lineJson["conversation_id"]
                    convObj["movieID"] = lineJson["meta"]["movie_id"]
                    convObj["lines"] = [lineObj]
                else:
                    convObj = conversations[lineJson["conversation_id"]]
                    convObj["lines"].insert(0, lineObj)
                conversations[convObj["conversationID"]] = convObj
        self.lines = lines
        self.conversations = conversations
    
    def extractSentencePairs(self):
        qa_pairs = []
        for conversation in self.conversations.values():
            # Iterate over all the lines of the conversation
            for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
                inputLine = conversation["lines"][i]["text"].strip()
                targetLine = conversation["lines"][i+1]["text"].strip()
                # Filter wrong samples (if one of the lists is empty)
                if inputLine and targetLine:
                    qa_pairs.append([inputLine, targetLine])
        self.qa_pairs = qa_pairs


class Csvwriter:
    def __init__(self, file_path, qa_pairs, delimiter):
        self.file_path = file_path
        self.qa_pairs =qa_pairs
        self.delimiter = delimiter
        self.csv = 'file_path'

    def write(self):
        with open(self.file_path, 'w', encoding = 'utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter = self.delimiter, lineterminator = '\n')
            for pair in self.qa_pairs:
                writer.writerow(pair)

    @staticmethod
    def printLines(file_path, n=10):
        with open(file_path, 'rb') as datafile:
            lines = datafile.readlines()
        for line in lines[:n]:
            print(line)

