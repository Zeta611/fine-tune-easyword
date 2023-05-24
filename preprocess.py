import json

with open("dump.json") as f:
    data = json.load(f)

translations = {entry["english"]: list(entry["translations"].keys()) for entry in data}
for english, translation in translations.items():
    koreans = []
    for words in translation:
        if "," in words:
            koreans += words.split(",")
        else:
            koreans.append(words)
    print(english, koreans)
