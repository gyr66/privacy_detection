---
language:
- zh
task_categories:
- token-classification
dataset_info:
  config_name: privacy_detection
  features:
  - name: id
    dtype: string
  - name: tokens
    sequence: string
  - name: ner_tags
    sequence:
      class_label:
        names:
          '0': O
          '1': B-position
          '2': I-position
          '3': B-name
          '4': I-name
          '5': B-movie
          '6': I-movie
          '7': B-organization
          '8': I-organization
          '9': B-company
          '10': I-company
          '11': B-book
          '12': I-book
          '13': B-address
          '14': I-address
          '15': B-scene
          '16': I-scene
          '17': B-mobile
          '18': I-mobile
          '19': B-email
          '20': I-email
          '21': B-game
          '22': I-game
          '23': B-government
          '24': I-government
          '25': B-QQ
          '26': I-QQ
          '27': B-vx
          '28': I-vx
  splits:
  - name: train
    num_bytes: 4899635
    num_examples: 2515
  download_size: 3290405
  dataset_size: 4899635
---
