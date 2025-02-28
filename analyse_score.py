import pandas as pd
from chordgnn.utils.chord_representations import solveChordSegmentation, resolveRomanNumeralCosine, formatRomanNumeral, formatChordLabel, generateRomanText
import copy, os
import numpy as np
import partitura as pt
# Testing with pretrained model
from chordgnn.utils.chord_representations import available_representations
from chordgnn.models.chord import ChordPrediction, PostChordPrediction
import torch
import argparse


parser = argparse.ArgumentParser("Chord Prediction")
parser.add_argument("--use_ckpt", type=str, default="model-2pesui9a-v0/model.ckpt",
                    help="Wandb artifact to use for prediction")
parser.add_argument("--score_path", type=str, default="", help="Path to musicxml input score")
parser.add_argument("--tasks_order", type=str, default="", help="Order of the sequential tasks for multi task MLP separated by comma. For not included tasks the parallel MLP is used.")
parser.add_argument("--post_tasks_order", type=str, default="", help="Order of the sequential tasks for multi task MLP Post-Processing separated by comma. For not included tasks the parallel MLP is used.")

args = parser.parse_args()

def parse_tasks_order(input_str):
    input_str = input_str.replace(" ", "").strip(',')

    if len(input_str) == 0:
        return []

    if not re.fullmatch(r'(\w+|\[\w+(,\w+)*\])(,(\w+|\[\w+(,\w+)*\]))*', input_str):
        raise ValueError("Error: Invalid input format. Ensure elements are valid words or lists of words.")

    parts = re.findall(r'\[([^\[\]]*)\]|(\w+)', input_str)

    result = []
    for part in parts:
        if part[0]:
            result.append(part[0].split(','))
        elif part[1]:
            result.append([part[1]])
    return result

tasks_order = parse_tasks_order(args.tasks_order)
post_tasks_order = parse_tasks_order(args.post_tasks_order)

artifact_dir = os.path.normpath(f"./artifacts/{os.path.basename(args.use_ckpt)}")
if not os.path.exists(artifact_dir):
    import wandb
    api = wandb.Api()
    artifact = api.artifact(args.use_ckpt, type="model")
    artifact_dir = artifact.download()

tasks = {"localkey": 35, "tonkey": 35, "degree1": 22, "degree2": 22, "quality": 16, "inversion": 4,
    "root": 35, "romanNumeral": 76, "hrhythm": 2, "pcset": 94, "bass": 35}
encoder = ChordPrediction(in_feats=83, n_hidden=256, tasks=tasks, n_layers=1, lr=0.0, dropout=0.0,
                        weight_decay=0.0, use_nade=False, use_jk=False, use_rotograd=False, device="cpu", tasks_order=tasks_order).module
model = PostChordPrediction(83, 256, tasks, 1, device="cpu", tasks_order=post_tasks_order ,frozen_model=encoder)
model = model.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"), tasks_order=post_tasks_order , frozen_model=encoder)
encoder = model.frozen_model
model = model.module
score = pt.load_score(args.score_path)
dfdict = {}
with torch.no_grad():
    model.eval()
    encoder.eval()
    prediction = model.predict(encoder.predict(score))

for task in tasks.keys():
    predOnehot = torch.argmax(prediction[task], dim=-1).reshape(-1, 1)
    decoded = available_representations[task].decode(predOnehot)
    dfdict[task] = decoded
dfdict["onset"] = prediction["onset"]
dfdict["s_measure"] = prediction["s_measure"]
df = pd.DataFrame(dfdict)


inputPath = args.score_path
dfout = copy.deepcopy(df)
score = pt.load_score(inputPath)
note_array = score.note_array(include_pitch_spelling=True)
prevkey = ""
bass_part = score.parts[-1]
rn_part = pt.score.Part(id="RNA", part_name="Roman Numerals", quarter_duration=bass_part._quarter_durations[0])
rn_part.add(pt.score.Clef(staff=1, sign="percussion", line=2, octave_change=0), 0)
rn_part.add(pt.score.Staff(number=1, lines=1), 0)

annotations = []
for analysis in dfout.itertuples():
    notes = []
    chord = note_array[(analysis.onset == note_array["onset_beat"]) | (analysis.onset < note_array["onset_beat"]) & (analysis.onset > note_array["onset_beat"] + note_array["duration_beat"])]
    if len(chord) == 0:
        continue
    bass = chord[chord["pitch"] == chord["pitch"].min()]
    thiskey = analysis.localkey
    tonicizedKey = analysis.tonkey
    pcset = analysis.pcset
    numerator = analysis.romanNumeral
    rn2, chordLabel = resolveRomanNumeralCosine(
        analysis.bass,
        pcset,
        thiskey,
        numerator,
        tonicizedKey,
    )
    if thiskey != prevkey:
        rn2fig = f"{thiskey}:{rn2}"
        prevkey = thiskey
    else:
        rn2fig = rn2
    formatted_RN = formatRomanNumeral(rn2fig, thiskey)
    annotations.append((formatted_RN, int(bass_part.inv_beat_map(analysis.onset).item())))



annotations = np.array(annotations, dtype=[("rn", "U10"), ("onset_div", "i4")])

# Infer first chord of piece
rn, onset = annotations[0]
annotations["rn"][0] = rn[:rn.index(":")+1] + "V" if rn.lower().endswith("i64") else rn
key = rn[0]
first_notes = np.unique(note_array[note_array["onset_div"] == onset]["step"])
if len(first_notes) > 1:
    pass
else:
    if abs(pt.utils.music.STEPS[first_notes[0].item().capitalize()] - pt.utils.music.STEPS[key.capitalize()])%7 == 3:
        annotations["rn"][0] = rn[:rn.index(":") + 1] + "V"


end_duration = note_array[note_array["onset_div"] == note_array["onset_div"].max()]["duration_div"].max()
bmask = np.array([True] + [(annotations[i]["rn"] != annotations[i-1]["rn"][annotations[i-1]["rn"].index(":")+1:]) if ":" in annotations[i-1]["rn"] else (annotations[i]["rn"] != annotations[i-1]["rn"]) for i in range(1, len(annotations))])
annotations = annotations[bmask]
durations = np.r_[np.diff(annotations["onset_div"]), end_duration]
for i, (rn, onset) in enumerate(annotations):
    note = pt.score.UnpitchedNote(step="F", octave=5, staff=1)
    word = pt.score.RomanNumeral(rn)
    rn_part.add(note, onset, onset+durations[i].item())
    rn_part.add(word, onset)


for item in bass_part.iter_all(pt.score.TimeSignature):
    rn_part.add(item, item.start.t)
for item in bass_part.measures:
    rn_part.add(item, item.start.t, item.end.t)
pt.score.tie_notes(rn_part)

# # TODO: Repair Short Key changes and check correctness.
# rna_annotations = list(rn_part.iter_all(pt.score.Harmony))
# # find indices of rna_annotations text that contain : character
# key_change = np.array([(i, x.text[:x.text.index(':')]) for i, x in enumerate(rna_annotations) if ":" in x.text], dtype=[("idx", "i4"), ("key", "U10")])
# # find where indices are consecutive
# c = np.where(np.diff(key_change["idx"]) < 2)[0] + 1
# c = c[c != key_change["idx"].argmax()]
# problematic_indices = c[np.where(key_change["key"][c+1] == key_change["key"][c-1])]
# key_change_indices = key_change["key"][problematic_indices]
# for idx in key_change_indices:
#     if "/" in rna_annotations[idx].text:
#         rna_annotations[idx].text = rna_annotations[idx].text[rna_annotations[idx].text.index(":")+1:rna_annotations[idx].text.index("/")]
#     else:
#         rna_annotations[idx].text = rna_annotations[idx].text[rna_annotations[idx].text.index(":")+1:] # + "/ degree difference between previous key and current key"

score.parts.append(rn_part)
pt.save_musicxml(score, f"{os.path.splitext(args.score_path)[0]}-analysis.musicxml")
