import os

assets_dir = "assets"
valid_words = ['H', 'He', 'Hel', 'Hell', 'Hello', 'HelloH', 'HelloHo', 'HelloHow', 'HelloHowc', 'HelloHowca', 'HelloHowcan', 'HelloHowcanI', 'HelloHowcanIa', 'HelloHowcanIas', 'HelloHowcanIass', 'HelloHowcanIassi', 'HelloHowcanIassis', 'HelloHowcanIassist', 'HelloHowcanIassisty', 'HelloHowcanIassistyo', 'HelloHowcanIassistyou', 'HelloHowcanIassistyout', 'HelloHowcanIassistyouto', 'HelloHowcanIassistyoutod', 'HelloHowcanIassistyoutoda', 'HelloHowcanIassistyoutoday']
final_valid_words = []
cop=valid_words
while valid_words:

    word = valid_words[0]  # Always work with the first word in the list
    if os.path.exists(os.path.join(assets_dir, f"{word}.mp4")) and len(word) > 1:
        final_valid_words.append(word)
        # Remove the processed word and all its preceding substrings
        valid_words = [wordy[len(word):] for wordy in valid_words if len(wordy) > len(word)]
    else:
        # If the file does not exist or word length <= 1, just remove the word from consideration
        valid_words.pop(0)
# for w in cop:
#     for f in final_valid_words:
#        cop.remove(f)
#        print("(",f,"---",w,")")
cop_modified = []
for w in cop:
    for f in final_valid_words:
        w = w.replace(f, '')  # Replace occurrences of f in w with an empty string
    cop_modified.append(w.strip())  # Use strip() to remove leading/trailing whitespace



i = 0
print(valid_words)
for index in range(len(cop_modified)):
    if cop_modified[index] == '':
        cop_modified[index] = final_valid_words[i]
        i += 1

k = 0
while k < len(final_valid_words) and cop_modified:
    try:
        j_index = cop_modified.index(final_valid_words[k])
        # Assuming the intention is to remove the word at j_index
        del cop_modified[j_index]
        k += 1
    except ValueError:
        # If the word is not found, move to the next word
        k += 1
        
print(cop_modified)