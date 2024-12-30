import pandas as pd
import numpy as np
import json as js
import os

source_path = r'dataset\vevo_instrument\origin'
target_path = r'dataset\vevo_instrument\thresholding'

threshold = 0.1

for file in os.listdir(source_path):
    file_path = os.path.join(source_path, file)
    df = pd.read_csv(file_path)
    df = df.drop(df.columns[0], axis=1)

    df = df.apply(lambda row: row.apply(lambda x: int(x >= threshold)), axis=1)

    df.to_csv(os.path.join(target_path, file), index=False)

instrument_list = '''accordion, acousticbassguitar, acousticguitar, bass, beat, bell, bongo, brass, cello, clarinet, classicalguitar, computer, doublebass, drummachine, drums, electricguitar, electricpiano, flute, guitar, harmonica, harp, horn, keyboard, oboe, orchestra, organ, pad, percussion, piano, pipeorgan, rhodes, sampler, saxophone, strings, synthesizer, trombone, trumpet, viola, violin, voice'''
instrument_list = instrument_list.split(', ')

instrument_dict = {instrument: idx for idx, instrument in enumerate(instrument_list)}
instrument_inv_dict = {idx: instrument for idx, instrument in enumerate(instrument_list)}

with open(r"dataset\vevo_meta\instrument.json", "w") as json_file:
    js.dump(instrument_dict, json_file, indent=4)

with open(r"dataset\vevo_meta\instrument_inv.json", "w") as json_file:
    js.dump(instrument_inv_dict, json_file, indent=4)